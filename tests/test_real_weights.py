"""Prueba de punta a punta con los pesos reales de serving/.

A diferencia de test_api (puntuador de mentira) y test_model_equivalence (pesos aleatorios),
aqui se cargan los modelos exportados y se envian los ejemplos de la raiz por HTTP:
  ejemplo_normal.json      -> APPROVE
  ejemplo_sospechoso.json  -> BLOCK_TRANSACTION
Ademas se comprueba la misma decision con los tres modelos (A, B y C).
Ejecutar:  python -m tests.test_real_weights
"""
import json
import os
import shutil
import tempfile
from pathlib import Path

tmp = tempfile.mkdtemp()
os.environ['LOG_DIR'] = tmp
os.environ.pop('DATABASE_URL', None)  # el registro va a un CSV temporal, nunca a Supabase

from fastapi.testclient import TestClient  # noqa: E402

import api  # noqa: E402  (despues de fijar LOG_DIR)
from fraude.scoring import Scorer  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
EXPECTED = {
    'ejemplo_normal.json': 'APPROVE',
    'ejemplo_sospechoso.json': 'BLOCK_TRANSACTION',
}


def load(name: str) -> dict:
    return json.loads((ROOT / name).read_text(encoding='utf-8'))


def main() -> None:
    # 1) La API tal como se despliega: modelo por defecto y pesos reales.
    api.app.state.scorer = None
    with TestClient(api.app) as client:
        health = client.get('/health').json()
        assert health['status'] == 'ok', health
        print(f"Modelo servido: {health['model']} | revision >= {health['review_threshold']:.4f}"
              f" | bloqueo >= {health['block_threshold']}")
        for name, decision in EXPECTED.items():
            r = client.post('/score', json=load(name))
            assert r.status_code == 200, r.text
            body = r.json()
            print(f"{name}: puntaje={body['fraud_risk_score']:.6f} decision={body['decision']}"
                  f" latencia={body['latency_ms']} ms")
            assert body['decision'] == decision, f'{name}: se esperaba {decision}, salio {body["decision"]}'
            assert body['recommended_action'] == api.ACTIONS[decision]
            assert body['warnings'] == [], body['warnings']

    # 2) Los tres modelos exportados toman la misma decision con estos ejemplos.
    for model in 'abc':
        scorer = Scorer(api.SERVING_DIR, model_name=model)
        for name, decision in EXPECTED.items():
            value, _ = scorer.score(load(name))
            got = api.decide(value, scorer.optimal_threshold, api.BLOCK_THRESHOLD)
            print(f'Modelo {model.upper()} | {name}: {value:.6f} -> {got}')
            assert got == decision, f'Modelo {model}, {name}: se esperaba {decision}, salio {got}'

    shutil.rmtree(tmp, ignore_errors=True)
    print('OK: con los pesos reales, el ejemplo normal se aprueba y el sospechoso se bloquea.')


if __name__ == '__main__':
    main()
