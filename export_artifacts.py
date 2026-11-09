"""Prepara el paquete de servicio a partir de los pesos entrenados en el notebook.

Pasos:
  1. Regenera los datos sinteticos (semilla 42) y ajusta el FeaturePipeline SOLO con Train.
  2. Carga los pesos de artefactos/*.pt y verifica que reproducen las metricas reportadas.
  3. Calibra el umbral por costo en Validacion (igual que el notebook).
  4. Guarda en serving/: pipeline.json, model_[abc].pt, thresholds.json, reproduced_metrics.json.
  5. Mide la latencia de extremo a extremo.

Uso:
    python export_artifacts.py --weights-dir artefactos --out serving
"""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from fraude.features import (FeaturePipeline, compute_aggregated_features,
                             create_temporal_split, generate_synthetic_transactions)
from fraude.models import HybridDualBranchNet, SequentialFraudLSTM, TabularBaselineMLP
from fraude.request_features import pipeline_to_dict
from fraude.scoring import Scorer, build_model
from fraude.tools import row_to_request
from fraude.training import (DEVICE, TransactionSequenceDataset, compute_metrics,
                             find_optimal_threshold, predict_dataset)

WEIGHT_FILES = {'a': 'modelo_a_baseline.pt', 'b': 'modelo_b_lstm.pt', 'c': 'modelo_c_hybrid.pt'}
MODEL_TYPE = {'a': 'tab', 'b': 'seq', 'c': 'hybrid'}

METRICS_FILE = 'metricas_completas.json'
METRICS_KEYS = {'a': 'model_a_baseline', 'b': 'model_b_sequential', 'c': 'model_c_hybrid'}


def load_expected(weights_dir: Path) -> dict:
    """Metricas de prueba guardadas junto con los pesos."""
    saved = json.loads((weights_dir / METRICS_FILE).read_text(encoding='utf-8'))['test_metrics']
    return {name: saved[key] for name, key in METRICS_KEYS.items()}


def load_state_dict(path: Path) -> dict:
    """Acepta un state_dict, un dict con 'state_dict' o un modelo completo guardado con torch.save."""
    obj = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(obj, torch.nn.Module):
        return obj.state_dict()
    if isinstance(obj, dict):
        for key in ('state_dict', 'model_state_dict', 'model'):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
        return obj
    raise TypeError(f'Formato de pesos no reconocido en {path}: {type(obj)}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--weights-dir', type=Path, default=Path('artefactos'))
    parser.add_argument('--out', type=Path, default=Path('serving'))
    parser.add_argument('--tol-prauc', type=float, default=0.02, help='Tolerancia de AUC-PR frente a las metricas guardadas.')
    parser.add_argument('--bench', type=int, default=200, help='Peticiones para medir latencia.')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print('1) Regenerando datos (semilla 42) y ajustando el pipeline solo con Train...')
    df_raw = generate_synthetic_transactions(n_clients=1200, days=90, fraud_rate=0.035, seed=42)
    df_feat = compute_aggregated_features(df_raw)
    train_df, val_df, test_df = create_temporal_split(df_feat, train_days=60, val_days=15)
    pipeline = FeaturePipeline(seq_len=6).fit(train_df)
    pipe = pipeline_to_dict(pipeline)
    (args.out / 'pipeline.json').write_text(json.dumps(pipe, indent=2), encoding='utf-8')
    print(f'   Train {len(train_df):,} | Val {len(val_df):,} | Test {len(test_df):,} | '
          f'comercios {len(pipe["merchant_classes"])} | canales {len(pipe["channel_classes"])}')

    val_loader = DataLoader(TransactionSequenceDataset(pipeline.build_sequence_dataset(val_df)),
                            batch_size=128, shuffle=False)
    test_loader = DataLoader(TransactionSequenceDataset(pipeline.build_sequence_dataset(test_df)),
                             batch_size=128, shuffle=False)

    print('2) Cargando pesos y verificando contra las metricas guardadas...')
    expected = load_expected(args.weights_dir)
    thresholds, report, all_ok = {}, {}, True
    for name, filename in WEIGHT_FILES.items():
        path = args.weights_dir / filename
        if not path.exists():
            raise SystemExit(f'No se encontro {path}. Revisa --weights-dir.')
        model = build_model(name, pipe)
        try:
            model.load_state_dict(load_state_dict(path))
        except RuntimeError as exc:
            raise SystemExit(
                f'Los pesos de {path.name} no encajan con la arquitectura del notebook:\n{exc}\n'
                'Probablemente fueron guardados con otra configuracion.')
        model.to(DEVICE).eval()

        val_preds, val_y, _ = predict_dataset(model, val_loader, model_type=MODEL_TYPE[name])
        thr, val_cost, _ = find_optimal_threshold(val_y, val_preds)
        test_preds, test_y, _ = predict_dataset(model, test_loader, model_type=MODEL_TYPE[name])
        metrics = compute_metrics(test_y, test_preds, threshold=thr)
        thresholds[name] = {'threshold': thr, 'val_cost_quetzales': float(val_cost)}
        exp = expected[name]
        diffs = {k: metrics[k] - exp[k] for k in ('pr_auc', 'roc_auc', 'f1', 'recall', 'precision', 'cost_quetzales')}
        ok = abs(diffs['pr_auc']) <= args.tol_prauc
        all_ok &= ok
        report[name] = {'test_metrics': metrics, 'expected_saved': exp, 'difference': diffs, 'within_tolerance': ok}
        print(f'   Modelo {name.upper()}: umbral {thr:.2f} (guardado {exp["threshold"]:.2f}) | '
              f'AUC-PR {metrics["pr_auc"]:.3f} (guardado {exp["pr_auc"]:.3f}) | '
              f'recall {metrics["recall"]:.3f} | costo Q{metrics["cost_quetzales"]:,.0f} '
              f'(guardado Q{exp["cost_quetzales"]:,.0f}) -> {"OK" if ok else "FUERA DE TOLERANCIA"}')
        torch.save({k: v.cpu() for k, v in model.state_dict().items()}, args.out / f'model_{name}.pt')

    (args.out / 'thresholds.json').write_text(json.dumps(thresholds, indent=2), encoding='utf-8')

    print('3) Midiendo latencia de extremo a extremo (variables + modelo, 1 hilo CPU)...')
    ids = test_df['tx_id'].values[:: max(1, len(test_df) // args.bench)][: args.bench]
    requests = [row_to_request(df_feat, tx) for tx in ids]
    latency = {}
    for name in ('b', 'c'):
        scorer = Scorer(args.out, model_name=name)
        scorer.benchmark(requests[:10])  # calentamiento
        latency[name] = scorer.benchmark(requests)
        print(f'   Modelo {name.upper()}: p50 {latency[name]["p50_ms"]:.2f} ms | '
              f'p95 {latency[name]["p95_ms"]:.2f} ms | max {latency[name]["max_ms"]:.2f} ms ({latency[name]["n"]} peticiones)')

    out = {'models': report, 'latency': latency, 'all_within_tolerance': all_ok,
           'torch_version': torch.__version__}
    (args.out / 'reproduced_metrics.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(f'\nPaquete de servicio guardado en {args.out}/')
    if not all_ok:
        print('AVISO: algun modelo no reproduce sus metricas guardadas. Revisen que los .pt y '
              f'{METRICS_FILE} sean de la misma corrida.')


if __name__ == '__main__':
    main()
