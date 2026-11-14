"""Verifica que build_inputs() (NumPy) reproduce EXACTAMENTE el pipeline del notebook.

Genera el dataset con la semilla 42, ajusta el FeaturePipeline solo con Train y compara,
para transacciones de Test, las entradas del notebook contra las que construye la API
a partir de una peticion con las 6 transacciones previas.

Referencia: build_sequence_dataset() aplicado al historial COMPLETO de cada cliente. Nota: en el
notebook original val/test se construyen por separado, asi que sus ventanas solo ven
transacciones del mismo periodo (las primeras de cada cliente en Test quedan con relleno).
En produccion hay historial completo, que es lo que reproduce esta prueba.
Ejecutar:  python -m tests.test_features_equivalence
"""
import numpy as np

from fraude.features import (FeaturePipeline, compute_aggregated_features,
                             create_temporal_split, generate_synthetic_transactions)
from fraude.request_features import build_inputs, pipeline_to_dict
from fraude.tools import row_to_request


def main(n_samples: int = 400, atol: float = 1e-4) -> None:
    df_raw = generate_synthetic_transactions(n_clients=1200, days=90, fraud_rate=0.035, seed=42)
    df_feat = compute_aggregated_features(df_raw)
    train_df, val_df, test_df = create_temporal_split(df_feat, train_days=60, val_days=15)
    pipeline = FeaturePipeline(seq_len=6).fit(train_df)
    pipe = pipeline_to_dict(pipeline)

    ref = pipeline.build_sequence_dataset(df_feat)  # historial completo por cliente
    index = {tx: i for i, tx in enumerate(ref['tx_ids'])}
    rng = np.random.RandomState(0)
    sample_ids = rng.choice(test_df['tx_id'].values, size=n_samples, replace=False)

    worst = {'seq_num': 0.0, 'agg_features': 0.0}
    bad_ids = 0
    for tx_id in sample_ids:
        got, _ = build_inputs(row_to_request(df_feat, tx_id), pipe)
        i = index[tx_id]
        ok_ids = (np.array_equal(got['seq_merch'], ref['seq_merch'][i])
                  and np.array_equal(got['seq_chan'], ref['seq_chan'][i]))
        bad_ids += 0 if ok_ids else 1
        for key in worst:
            worst[key] = max(worst[key], float(np.abs(got[key] - ref[key][i]).max()))

    print(f'Transacciones comparadas: {n_samples}')
    print(f'Error maximo seq_num: {worst["seq_num"]:.2e} | agg_features: {worst["agg_features"]:.2e}')
    print(f'Ids de comercio/canal distintos: {bad_ids}')
    assert bad_ids == 0, 'Los ids categoricos no coinciden con el notebook.'
    assert all(v <= atol for v in worst.values()), 'Las variables numericas difieren del notebook.'
    print('OK: la API construye las mismas entradas que el notebook.')


if __name__ == '__main__':
    main()
