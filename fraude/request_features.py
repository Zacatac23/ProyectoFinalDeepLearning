"""Convierte una peticion de la API en las entradas exactas del modelo.

Reproduce, solo con NumPy (sin pandas ni scikit-learn), lo que hacen en el notebook
`compute_aggregated_features` y `FeaturePipeline.build_sequence_dataset`.
La equivalencia se verifica en tests/test_features_equivalence.py.
"""
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

SEQ_NUM_COLS = ['amount', 'delta_time_hours', 'hour_of_day', 'is_international']
NO_PREVIOUS_DELTA = 999.0  # igual que el fillna del notebook


def pipeline_to_dict(pipeline) -> Dict[str, Any]:
    """Serializa un FeaturePipeline ya ajustado (solo con Train) a un dict JSON."""
    return {
        'seq_len': int(pipeline.seq_len),
        'agg_feature_cols': list(pipeline.agg_feature_cols),
        'seq_num_cols': list(SEQ_NUM_COLS),
        'agg_mean': pipeline.scaler_agg.mean_.tolist(),
        'agg_scale': pipeline.scaler_agg.scale_.tolist(),
        'seq_mean': pipeline.scaler_num_seq.mean_.tolist(),
        'seq_scale': pipeline.scaler_num_seq.scale_.tolist(),
        'merchant_classes': [str(c) for c in pipeline.le_merchant.classes_],
        'channel_classes': [str(c) for c in pipeline.le_channel.classes_],
    }


def parse_timestamp(value: str) -> datetime:
    """Acepta ISO 8601 (con o sin 'Z'); devuelve datetime ingenuo en UTC."""
    text = value.strip()
    if text.endswith('Z'):
        text = text[:-1] + '+00:00'
    ts = datetime.fromisoformat(text)
    if ts.tzinfo is not None:
        ts = ts.astimezone(timezone.utc).replace(tzinfo=None)
    return ts


def _class_id(classes: List[str], value: str) -> int:
    return classes.index(value) if value in classes else classes.index('UNKNOWN')


def build_inputs(request: Dict[str, Any], pipe: Dict[str, Any]) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """Devuelve (entradas, advertencias).

    request: {
      'current_transaction': {timestamp, amount, merchant_category, channel, is_international},
      'recent_history': [ {... mismos campos, delta_time_hours opcional ...} ],
      'aggregated_profile': {'hist_avg_amount': opcional}
    }
    """
    warnings: List[str] = []
    seq_len = pipe['seq_len']
    cur = request['current_transaction']
    cur_ts = parse_timestamp(cur['timestamp'])

    history = sorted(request.get('recent_history') or [], key=lambda t: parse_timestamp(t['timestamp']))
    history = history[-seq_len:]  # solo se necesitan las ultimas K transacciones previas
    hist_ts = [parse_timestamp(t['timestamp']) for t in history]
    if any(t > cur_ts for t in hist_ts):
        raise ValueError('recent_history contiene transacciones posteriores a la transaccion actual.')

    # delta_time_hours de cada transaccion respecto a su predecesora (999.0 si no se conoce)
    history_deltas: List[float] = []
    for i, tx in enumerate(history):
        if i > 0:
            delta = (hist_ts[i] - hist_ts[i - 1]).total_seconds() / 3600.0
        elif tx.get('delta_time_hours') is not None:
            delta = float(tx['delta_time_hours'])
        else:
            delta = NO_PREVIOUS_DELTA
        history_deltas.append(delta)
    cur_delta = (cur_ts - hist_ts[-1]).total_seconds() / 3600.0 if history else NO_PREVIOUS_DELTA

    if len(history) < seq_len:
        warnings.append(
            f'Historial de {len(history)} transacciones (se recomiendan {seq_len}): '
            'la ventana se rellena como en el entrenamiento y la primera transaccion se trata como inicio de cliente.')

    # ---- variables agregadas (12, en el orden de pipe['agg_feature_cols']) ----
    amount = float(cur['amount'])
    is_intl = float(cur.get('is_international', 0))
    prev5 = history[-5:]
    prev_amounts = np.array([float(t['amount']) for t in prev5], dtype=np.float64)
    if len(prev_amounts) == 0:
        roll_mean, roll_max, roll_std, roll_intl = amount, amount, 0.0, 0.0
    else:
        roll_mean = float(prev_amounts.mean())
        roll_max = float(prev_amounts.max())
        roll_std = float(prev_amounts.std(ddof=1)) if len(prev_amounts) > 1 else 0.0
        roll_intl = float(sum(float(t.get('is_international', 0)) for t in prev5))

    profile = request.get('aggregated_profile') or {}
    hist_avg: Optional[float] = profile.get('hist_avg_amount')
    if hist_avg is None:
        if history:
            hist_avg = float(np.mean([float(t['amount']) for t in history]))
            warnings.append('hist_avg_amount no enviado: se estimo con el historial recibido.')
        else:
            hist_avg = 0.0  # igual que el notebook para la primera transaccion de un cliente
            warnings.append('Sin historial ni hist_avg_amount: se usa 0.0 como en el entrenamiento.')
    hist_avg = float(hist_avg)

    agg_values = {
        'amount': amount,
        'delta_time_hours': cur_delta,
        'hour_of_day': float(cur_ts.hour),
        'day_of_week': float(cur_ts.weekday()),
        'hist_avg_amount': hist_avg,
        'amount_to_hist_avg_ratio': amount / (hist_avg + 1e-5),
        'roll_mean_5': roll_mean,
        'roll_max_5': roll_max,
        'roll_std_5': roll_std,
        'amount_to_roll_mean_ratio': amount / (roll_mean + 1e-5),
        'roll_intl_count_5': roll_intl,
        'is_international': is_intl,
    }
    agg = np.array([agg_values[c] for c in pipe['agg_feature_cols']], dtype=np.float64)
    agg = (agg - np.array(pipe['agg_mean'])) / np.array(pipe['agg_scale'])

    # ---- secuencia: ultimas K-1 previas + la actual, con relleno al inicio ----
    window = [
        {'amount': float(t['amount']), 'delta_time_hours': d, 'hour_of_day': float(ts.hour),
         'is_international': float(t.get('is_international', 0)),
         'merchant_category': t['merchant_category'], 'channel': t['channel']}
        for t, d, ts in zip(history[-(seq_len - 1):], history_deltas[-(seq_len - 1):], hist_ts[-(seq_len - 1):])
    ]
    window.append({'amount': amount, 'delta_time_hours': cur_delta, 'hour_of_day': float(cur_ts.hour),
                   'is_international': is_intl, 'merchant_category': cur['merchant_category'],
                   'channel': cur['channel']})

    seq_mean, seq_scale = np.array(pipe['seq_mean']), np.array(pipe['seq_scale'])
    num = np.array([[w[c] for c in pipe['seq_num_cols']] for w in window], dtype=np.float64)
    num = (num - seq_mean) / seq_scale
    merch = np.array([_class_id(pipe['merchant_classes'], w['merchant_category']) for w in window], dtype=np.int64)
    chan = np.array([_class_id(pipe['channel_classes'], w['channel']) for w in window], dtype=np.int64)

    pad = seq_len - len(window)
    if pad > 0:
        num = np.vstack([np.zeros((pad, num.shape[1])), num])
        merch = np.concatenate([np.zeros(pad, dtype=np.int64), merch])
        chan = np.concatenate([np.zeros(pad, dtype=np.int64), chan])

    for value, label in ((cur['merchant_category'], 'merchant_category'), (cur['channel'], 'channel')):
        classes = pipe['merchant_classes'] if label == 'merchant_category' else pipe['channel_classes']
        if value not in classes:
            warnings.append(f'{label} desconocido ("{value}"): se trata como UNKNOWN.')

    return {
        'seq_num': num.astype(np.float32),
        'seq_merch': merch,
        'seq_chan': chan,
        'agg_features': agg.astype(np.float32),
    }, warnings
