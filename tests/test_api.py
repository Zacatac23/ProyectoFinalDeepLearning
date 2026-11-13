"""Prueba de la capa HTTP con un puntuador de mentira (sin PyTorch).

Comprueba validaciones, decisiones, advertencias, registro de uso y estadisticas.
El puntaje real del modelo se prueba con export_artifacts.py (equivalencia con el notebook).
Ejecutar:  python -m tests.test_api
"""
import math
import os
import shutil
import tempfile

import numpy as np
from fastapi.testclient import TestClient

tmp = tempfile.mkdtemp()
os.environ['LOG_DIR'] = tmp

import api  # noqa: E402  (despues de fijar LOG_DIR)
from fraude.features import (FeaturePipeline, compute_aggregated_features,  # noqa: E402
                             create_temporal_split, generate_synthetic_transactions)
from fraude.request_features import build_inputs, pipeline_to_dict  # noqa: E402


class StubScorer:
    model_name = 'stub'
    optimal_threshold = 0.21

    def __init__(self, pipe):
        self.pipe = pipe

    def score(self, request):
        inputs, warnings = build_inputs(request, self.pipe)
        amount = float(inputs['agg_features'][0])  # monto escalado (puntaje de mentira, solo para probar la API)
        return 1.0 / (1.0 + math.exp(-amount)), warnings


def main():
    df = compute_aggregated_features(generate_synthetic_transactions(n_clients=200, days=90, seed=42))
    train_df, _, _ = create_temporal_split(df)
    pipe = pipeline_to_dict(FeaturePipeline(seq_len=6).fit(train_df))
    api.app.state.scorer = StubScorer(pipe)

    with TestClient(api.app) as client:
        assert client.get('/health').json()['status'] == 'ok'
        assert client.get('/').status_code == 200
        ex = client.get('/ejemplos').json()

        r = client.post('/score', json=ex['normal'])
        assert r.status_code == 200, r.text
        body = r.json()
        assert body['decision'] in api.ACTIONS and body['history_used'] == 6 and body['warnings'] == []

        r = client.post('/score', json=ex['sondeo_y_vaciado'])
        assert r.status_code == 200
        assert r.json()['fraud_risk_score'] > body['fraud_risk_score']

        # historial corto, comercio desconocido y campos extra del contrato original
        short = {**ex['normal'], 'recent_history': ex['normal']['recent_history'][:2],
                 'aggregated_profile': {'hist_avg_amount': 260.0, 'roll_mean_5': 12.5, 'roll_std_5': 2.5}}
        short['current_transaction'] = {**short['current_transaction'], 'merchant_category': 'CASINO'}
        r = client.post('/score', json=short)
        assert r.status_code == 200 and len(r.json()['warnings']) >= 2, r.text

        # sin historial ni perfil
        r = client.post('/score', json={'client_id': 'X', 'current_transaction': ex['normal']['current_transaction']})
        assert r.status_code == 200 and r.json()['history_used'] == 0

        # validaciones
        bad_ts = {**ex['normal'], 'current_transaction': {**ex['normal']['current_transaction'], 'timestamp': 'ayer'}}
        assert client.post('/score', json=bad_ts).status_code == 422
        neg = {**ex['normal'], 'current_transaction': {**ex['normal']['current_transaction'], 'amount': -5}}
        assert client.post('/score', json=neg).status_code == 422
        future = {**ex['normal'], 'recent_history': [{**ex['normal']['recent_history'][0], 'timestamp': '2027-01-01T00:00:00Z'}]}
        assert client.post('/score', json=future).status_code == 422
        assert client.post('/score', json={'client_id': 'X'}).status_code == 422

        # politica de decision
        assert api.decide(0.10, 0.21, 0.65) == 'APPROVE'
        assert api.decide(0.21, 0.21, 0.65) == 'STEP_UP_AUTH'
        assert api.decide(0.64, 0.21, 0.65) == 'STEP_UP_AUTH'
        assert api.decide(0.65, 0.21, 0.65) == 'BLOCK_TRANSACTION'

        st = client.get('/stats').json()
        assert st['requests'] == 4, st
        print('Estadisticas:', st)
    shutil.rmtree(tmp, ignore_errors=True)
    print('OK: capa HTTP, validaciones, decisiones y registro funcionan.')


if __name__ == '__main__':
    main()
