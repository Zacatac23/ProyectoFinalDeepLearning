"""Utilidades con pandas para construir peticiones reales a partir del dataset sintetico."""
from typing import Any, Dict

import pandas as pd


def row_to_request(df_feat: pd.DataFrame, tx_id: str, history_len: int = 6) -> Dict[str, Any]:
    """Arma la peticion de la API para una transaccion del dataset (con sus previas)."""
    row = df_feat[df_feat['tx_id'] == tx_id].iloc[0]
    client = df_feat[df_feat['client_id'] == row['client_id']].sort_values('timestamp')
    previous = client[client['timestamp'] < row['timestamp']].tail(history_len)

    def as_tx(r) -> Dict[str, Any]:
        return {
            'timestamp': r['timestamp'].isoformat(),
            'amount': float(r['amount']),
            'merchant_category': r['merchant_category'],
            'channel': r['channel'],
            'is_international': int(r['is_international']),
        }

    return {
        'client_id': row['client_id'],
        'current_transaction': {**as_tx(row), 'tx_id': tx_id},
        'recent_history': [as_tx(r) for _, r in previous.iterrows()],
        'aggregated_profile': {'hist_avg_amount': float(row['hist_avg_amount'])},
    }
