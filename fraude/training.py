"""Dataset, entrenamiento y metricas. Codigo extraido SIN CAMBIOS del notebook (celdas 3 y 5)."""
from typing import Dict, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    precision_recall_curve, auc, roc_auc_score, roc_curve,
    f1_score, precision_score, recall_score, confusion_matrix
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TransactionSequenceDataset(Dataset):
    def __init__(self, data_dict: Dict[str, np.ndarray]):
        self.seq_num = torch.tensor(data_dict['seq_num'], dtype=torch.float32)
        self.seq_merch = torch.tensor(data_dict['seq_merch'], dtype=torch.long)
        self.seq_chan = torch.tensor(data_dict['seq_chan'], dtype=torch.long)
        self.agg_features = torch.tensor(data_dict['agg_features'], dtype=torch.float32)
        self.labels = torch.tensor(data_dict['labels'], dtype=torch.float32).unsqueeze(1)
        self.fraud_types = data_dict['fraud_types']
        self.tx_ids = data_dict['tx_ids']
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        return {
            'seq_num': self.seq_num[idx],
            'seq_merch': self.seq_merch[idx],
            'seq_chan': self.seq_chan[idx],
            'agg_features': self.agg_features[idx],
            'label': self.labels[idx],
            'fraud_type': self.fraud_types[idx],
            'tx_id': self.tx_ids[idx]
        }


# ==========================================
# 5. FUNCIONES DE ENTRENAMIENTO Y EVALUACIÓN
# ==========================================

COST_FN = 4200.0 # Quetzales por Fraude no detectado (Falso Negativo)
COST_FP = 180.0  # Quetzales por Falso bloqueo a cliente legítimo (Falso Positivo)

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_type: str = 'seq',
    epochs: int = 25,
    lr: float = 0.002,
    pos_weight_val: float = 10.0
) -> nn.Module:
    model.to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    best_val_prauc = -1.0
    best_weights = None
    
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            y_true = batch['label'].to(DEVICE)
            weights = torch.where(y_true == 1.0, torch.tensor(pos_weight_val, device=DEVICE), torch.tensor(1.0, device=DEVICE))
            loss_fn = nn.BCELoss(weight=weights)
            
            if model_type == 'tab':
                x = batch['agg_features'].to(DEVICE)
                preds = model(x)
            elif model_type == 'seq':
                num = batch['seq_num'].to(DEVICE)
                merch = batch['seq_merch'].to(DEVICE)
                chan = batch['seq_chan'].to(DEVICE)
                preds = model(num, merch, chan)
            elif model_type == 'hybrid':
                num = batch['seq_num'].to(DEVICE)
                merch = batch['seq_merch'].to(DEVICE)
                chan = batch['seq_chan'].to(DEVICE)
                agg = batch['agg_features'].to(DEVICE)
                preds = model(num, merch, chan, agg)
                
            loss = loss_fn(preds, y_true)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            
        val_preds, val_y, _ = predict_dataset(model, val_loader, model_type)
        val_p, val_r, _ = precision_recall_curve(val_y, val_preds)
        val_prauc = auc(val_r, val_p)
        scheduler.step(val_prauc)
        
        if val_prauc > best_val_prauc:
            best_val_prauc = val_prauc
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
    model.load_state_dict(best_weights)
    return model

def predict_dataset(
    model: nn.Module,
    loader: DataLoader,
    model_type: str = 'seq',
    shuffle_order: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_preds = []
    all_labels = []
    all_fraud_types = []
    
    with torch.no_grad():
        for batch in loader:
            y_true = batch['label'].cpu().numpy().flatten()
            ft = batch['fraud_type']
            
            if model_type == 'tab':
                x = batch['agg_features'].to(DEVICE)
                preds = model(x)
            elif model_type == 'seq':
                num = batch['seq_num'].to(DEVICE)
                merch = batch['seq_merch'].to(DEVICE)
                chan = batch['seq_chan'].to(DEVICE)
                
                if shuffle_order:
                    batch_size, seq_len, _ = num.shape
                    perm = torch.randperm(seq_len)
                    num = num[:, perm, :]
                    merch = merch[:, perm]
                    chan = chan[:, perm]
                    
                preds = model(num, merch, chan)
            elif model_type == 'hybrid':
                num = batch['seq_num'].to(DEVICE)
                merch = batch['seq_merch'].to(DEVICE)
                chan = batch['seq_chan'].to(DEVICE)
                agg = batch['agg_features'].to(DEVICE)
                
                if shuffle_order:
                    batch_size, seq_len, _ = num.shape
                    perm = torch.randperm(seq_len)
                    num = num[:, perm, :]
                    merch = merch[:, perm]
                    chan = chan[:, perm]
                    
                preds = model(num, merch, chan, agg)
                
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(y_true)
            all_fraud_types.extend(ft)
            
    return np.array(all_preds), np.array(all_labels), np.array(all_fraud_types)

def compute_metrics(y_true: np.ndarray, y_scores: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    p, r, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = float(auc(r, p))
    roc_auc = float(roc_auc_score(y_true, y_scores))
    
    y_pred = (y_scores >= threshold).astype(int)
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    cost = float(fn * COST_FN + fp * COST_FP)
    
    return {
        'pr_auc': pr_auc,
        'roc_auc': roc_auc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'threshold': threshold,
        'cost_quetzales': cost,
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'tn': int(tn)
    }

def find_optimal_threshold(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, float, List[Dict]]:
    thresholds = np.linspace(0.01, 0.99, 100)
    best_thresh = 0.5
    min_cost = float('inf')
    cost_curve = []
    
    for th in thresholds:
        y_pred = (y_scores >= th).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        cost = fn * COST_FN + fp * COST_FP
        cost_curve.append({'threshold': float(th), 'cost': float(cost), 'fn': int(fn), 'fp': int(fp), 'tp': int(tp)})
        
        if cost < min_cost:
            min_cost = cost
            best_thresh = float(th)
            
    return best_thresh, min_cost, cost_curve
