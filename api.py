"""API de monitoreo transaccional (Proyecto 1 en produccion).

Ejecutar:  uvicorn api:app --host 0.0.0.0 --port 8000
Variables de entorno (opcionales):
  MODEL=b|c|a            modelo a servir (por defecto b, elegido por su costo en Validacion)
  SERVING_DIR=serving    carpeta creada por export_artifacts.py
  BLOCK_THRESHOLD=0.65   puntaje desde el cual se bloquea (el umbral de revision es el optimo de Validacion)
  LOG_DIR=logs           carpeta del registro de uso cuando no hay base de datos
  DATABASE_URL=...       Postgres para el registro de uso (en produccion, Supabase)
"""
import json
import logging
import os
import statistics
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, ConfigDict, Field, field_validator

from fraude.request_features import parse_timestamp
from fraude.request_log import build_request_log, rows_to_csv

SERVING_DIR = Path(os.environ.get('SERVING_DIR', 'serving'))
MODEL_NAME = os.environ.get('MODEL', 'b').lower()
BLOCK_THRESHOLD = float(os.environ.get('BLOCK_THRESHOLD', '0.65'))
LOG_DIR = Path(os.environ.get('LOG_DIR', 'logs'))
REQUEST_LOG = build_request_log(os.environ.get('DATABASE_URL'), LOG_DIR)
WEB_DIR = Path(__file__).resolve().parent / 'web'
CATALOG: List[Dict[str, Any]] = json.loads((WEB_DIR / 'ejemplos.json').read_text(encoding='utf-8'))
EXAMPLES = {example['id']: example['request'] for example in CATALOG}
PAGE = (WEB_DIR / 'index.html').read_text(encoding='utf-8')
logger = logging.getLogger('uvicorn.error')


class Tx(BaseModel):
    model_config = ConfigDict(extra='ignore')
    timestamp: str
    amount: float = Field(ge=0)
    merchant_category: str
    channel: str
    is_international: int = Field(default=0, ge=0, le=1)
    tx_id: Optional[str] = None
    delta_time_hours: Optional[float] = Field(default=None, ge=0)

    @field_validator('timestamp')
    @classmethod
    def _valid_timestamp(cls, v: str) -> str:
        try:
            parse_timestamp(v)
        except ValueError as exc:
            raise ValueError('timestamp debe estar en formato ISO 8601, p. ej. 2026-03-31T14:23:10Z') from exc
        return v


class Profile(BaseModel):
    # Los demas campos del contrato original (roll_*, delta_time_hours...) se calculan en el servidor.
    model_config = ConfigDict(extra='ignore')
    hist_avg_amount: Optional[float] = Field(default=None, ge=0)


class ScoreRequest(BaseModel):
    model_config = ConfigDict(extra='ignore')
    client_id: str
    current_transaction: Tx
    recent_history: List[Tx] = Field(default_factory=list, max_length=200)
    aggregated_profile: Optional[Profile] = None


class ScoreResponse(BaseModel):
    tx_id: Optional[str]
    fraud_risk_score: float
    decision: str
    recommended_action: str
    model: str
    review_threshold: float
    block_threshold: float
    latency_ms: float
    history_used: int
    warnings: List[str]


ACTIONS = {
    'APPROVE': 'APPROVE_WITHOUT_FRICTION',
    'STEP_UP_AUTH': 'REQUEST_OTP_OR_BIOMETRIC_AUTH',
    'BLOCK_TRANSACTION': 'LOCK_CARD_AND_ALERT_ANALYST',
}


def decide(score: float, review_threshold: float, block_threshold: float) -> str:
    if score >= block_threshold:
        return 'BLOCK_TRANSACTION'
    if score >= review_threshold:
        return 'STEP_UP_AUTH'
    return 'APPROVE'


def log_request(row: Dict[str, Any]) -> None:
    # Si el registro falla, la API debe seguir respondiendo.
    try:
        REQUEST_LOG.write(row)
    except Exception:
        logger.exception('No se pudo guardar la peticion en el registro de uso.')


def read_log() -> List[Dict[str, Any]]:
    try:
        return REQUEST_LOG.read()
    except Exception:
        logger.exception('No se pudo leer el registro de uso.')
        raise HTTPException(status_code=503, detail='Registro de uso no disponible.')


@asynccontextmanager
async def lifespan(app: FastAPI):
    if getattr(app.state, 'scorer', None) is None:  # las pruebas inyectan un scorer propio
        from fraude.scoring import Scorer
        app.state.scorer = Scorer(SERVING_DIR, model_name=MODEL_NAME)
    yield


app = FastAPI(title='Monitoreo transaccional: puntaje de riesgo de fraude', version='1.0', lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])


@app.get('/health')
def health() -> Dict[str, Any]:
    scorer = app.state.scorer
    return {'status': 'ok', 'model': scorer.model_name, 'review_threshold': scorer.optimal_threshold,
            'block_threshold': BLOCK_THRESHOLD}


@app.post('/score', response_model=ScoreResponse)
def score(req: ScoreRequest, background: BackgroundTasks) -> ScoreResponse:
    scorer = app.state.scorer
    payload = req.model_dump()
    start = time.perf_counter()
    try:
        value, warnings = scorer.score(payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    latency = (time.perf_counter() - start) * 1000.0
    review = scorer.optimal_threshold
    decision = decide(value, review, BLOCK_THRESHOLD)
    background.add_task(log_request, {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'model': scorer.model_name, 'score': round(value, 6), 'decision': decision,
        'latency_ms': round(latency, 3), 'history_len': len(req.recent_history),
        'warnings': len(warnings)})
    return ScoreResponse(
        tx_id=req.current_transaction.tx_id, fraud_risk_score=round(value, 6), decision=decision,
        recommended_action=ACTIONS[decision], model=scorer.model_name, review_threshold=review,
        block_threshold=BLOCK_THRESHOLD, latency_ms=round(latency, 3),
        history_used=min(len(req.recent_history), 6), warnings=warnings)


@app.get('/stats')
def stats() -> Dict[str, Any]:
    rows = read_log()
    if not rows:
        return {'requests': 0}
    scores = [float(r['score']) for r in rows]
    lat = sorted(float(r['latency_ms']) for r in rows)
    decisions: Dict[str, int] = {}
    for r in rows:
        decisions[r['decision']] = decisions.get(r['decision'], 0) + 1
    n = len(rows)
    return {
        'requests': n,
        'decisions': decisions,
        'share_review_or_block': round(1 - decisions.get('APPROVE', 0) / n, 4),
        'score_mean': round(statistics.fmean(scores), 4),
        'latency_p50_ms': round(lat[n // 2], 3),
        'latency_p95_ms': round(lat[int(0.95 * (n - 1))], 3),
        'latency_max_ms': round(lat[-1], 3),
    }


@app.get('/logs.csv')
def logs_csv() -> Response:
    return Response(rows_to_csv(read_log()), media_type='text/csv',
                    headers={'Content-Disposition': 'attachment; filename="requests.csv"'})


@app.get('/ejemplos')
def ejemplos() -> Dict[str, Any]:
    return EXAMPLES


@app.get('/catalogo')
def catalogo() -> Dict[str, Any]:
    """Ejemplos con contexto y vocabulario para la pagina de demostracion."""
    scorer = app.state.scorer
    return {
        'model': scorer.model_name,
        'review_threshold': scorer.optimal_threshold,
        'block_threshold': BLOCK_THRESHOLD,
        'merchants': [m for m in scorer.pipe['merchant_classes'] if m != 'UNKNOWN'],
        'channels': [c for c in scorer.pipe['channel_classes'] if c != 'UNKNOWN'],
        'examples': CATALOG,
    }


@app.get('/', response_class=HTMLResponse)
def home() -> str:
    return PAGE
