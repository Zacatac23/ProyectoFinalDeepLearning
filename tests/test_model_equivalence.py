"""Verifica que Scorer (ruta de la API) da el MISMO puntaje que el camino del notebook.

Usa pesos aleatorios (semilla fija): comprueba el cableado, no la calidad del modelo.
Compara, para los tres modelos, el puntaje de la API contra el del notebook
(pipeline.build_sequence_dataset + modelo en lote) sobre historial completo.
Ejecutar:  python -m tests.test_model_equivalence
"""
import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from fraude.features import (FeaturePipeline, compute_aggregated_features,
                             create_temporal_split, generate_synthetic_transactions)
from fraude.request_features import pipeline_to_dict
from fraude.scoring import Scorer, build_model
from fraude.tools import row_to_request


def main(n_samples: int = 150, atol: float = 1e-5) -> None:
    torch.manual_seed(0)
    df = compute_aggregated_features(generate_synthetic_transactions(n_clients=300, days=90, seed=42))
    train_df, _, test_df = create_temporal_split(df)
    pipeline = FeaturePipeline(seq_len=6).fit(train_df)
    pipe = pipeline_to_dict(pipeline)
    ref = pipeline.build_sequence_dataset(df)
    index = {t: i for i, t in enumerate(ref['tx_ids'])}
    ids = np.random.RandomState(1).choice(test_df['tx_id'].values, size=n_samples, replace=False)
    rows = np.array([index[t] for t in ids])

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / 'pipeline.json').write_text(json.dumps(pipe))
        (tmp / 'thresholds.json').write_text(json.dumps({k: {'threshold': 0.2} for k in 'abc'}))
        for name in 'abc':
            model = build_model(name, pipe).eval()
            torch.save(model.state_dict(), tmp / f'model_{name}.pt')
            t = {k: torch.from_numpy(ref[k][rows]) for k in ('seq_num', 'seq_merch', 'seq_chan', 'agg_features')}
            with torch.no_grad():
                if name == 'a':
                    batch = model(t['agg_features'])
                elif name == 'b':
                    batch = model(t['seq_num'], t['seq_merch'], t['seq_chan'])
                else:
                    batch = model(t['seq_num'], t['seq_merch'], t['seq_chan'], t['agg_features'])
            expected = batch.numpy().flatten()

            scorer = Scorer(tmp, model_name=name)
            got = np.array([scorer.score(row_to_request(df, tx))[0] for tx in ids])
            err = float(np.abs(got - expected).max())
            print(f'Modelo {name.upper()}: error maximo API vs notebook = {err:.2e}')
            assert err <= atol, f'El modelo {name} difiere del camino del notebook.'
    print('OK: la API puntua igual que el notebook para los tres modelos.')


if __name__ == '__main__':
    main()
