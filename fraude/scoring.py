"""Puntuacion en linea: carga el pipeline y los pesos exportados y evalua una peticion."""
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

from fraude.models import HybridDualBranchNet, SequentialFraudLSTM, TabularBaselineMLP
from fraude.request_features import build_inputs

MODEL_FILES = {'a': 'model_a.pt', 'b': 'model_b.pt', 'c': 'model_c.pt'}


def build_model(name: str, pipe: Dict[str, Any]) -> torch.nn.Module:
    n_merch, n_chan = len(pipe['merchant_classes']), len(pipe['channel_classes'])
    n_agg = len(pipe['agg_feature_cols'])
    if name == 'a':
        return TabularBaselineMLP(input_dim=n_agg)
    if name == 'b':
        return SequentialFraudLSTM(num_merchants=n_merch, num_channels=n_chan, num_features_dim=4)
    if name == 'c':
        return HybridDualBranchNet(num_merchants=n_merch, num_channels=n_chan,
                                   num_features_dim=4, agg_features_dim=n_agg)
    raise ValueError(f'Modelo desconocido: {name}')


class Scorer:
    def __init__(self, serving_dir: Path, model_name: str = 'c'):
        serving_dir = Path(serving_dir)
        self.model_name = model_name.lower()
        self.pipe = json.loads((serving_dir / 'pipeline.json').read_text(encoding='utf-8'))
        self.thresholds = json.loads((serving_dir / 'thresholds.json').read_text(encoding='utf-8'))
        self.model = build_model(self.model_name, self.pipe)
        state = torch.load(serving_dir / MODEL_FILES[self.model_name], map_location='cpu')
        self.model.load_state_dict(state)
        self.model.eval()
        torch.set_num_threads(1)

    @property
    def optimal_threshold(self) -> float:
        return float(self.thresholds[self.model_name]['threshold'])

    def score(self, request: Dict[str, Any]) -> Tuple[float, List[str]]:
        inputs, warnings = build_inputs(request, self.pipe)
        t = {k: torch.from_numpy(v).unsqueeze(0) for k, v in inputs.items()}
        with torch.no_grad():
            if self.model_name == 'a':
                out = self.model(t['agg_features'])
            elif self.model_name == 'b':
                out = self.model(t['seq_num'], t['seq_merch'], t['seq_chan'])
            else:
                out = self.model(t['seq_num'], t['seq_merch'], t['seq_chan'], t['agg_features'])
        return float(out.item()), warnings

    def benchmark(self, requests: List[Dict[str, Any]]) -> Dict[str, float]:
        """Latencia de extremo a extremo (variables + modelo) por peticion, en ms."""
        times = []
        for req in requests:
            start = time.perf_counter()
            self.score(req)
            times.append((time.perf_counter() - start) * 1000.0)
        times.sort()
        return {
            'n': len(times),
            'p50_ms': times[len(times) // 2],
            'p95_ms': times[int(0.95 * (len(times) - 1))],
            'max_ms': times[-1],
        }
