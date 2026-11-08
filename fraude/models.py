"""Arquitecturas de los tres modelos. Codigo extraido SIN CAMBIOS del notebook (celda 4)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 4. ARQUITECTURAS DE DEEP LEARNING
# ==========================================

class TabularBaselineMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x_agg):
        return torch.sigmoid(self.net(x_agg))


class SequentialFraudLSTM(nn.Module):
    def __init__(
        self,
        num_merchants: int,
        num_channels: int,
        num_features_dim: int = 4,
        embed_dim_merch: int = 16,
        embed_dim_chan: int = 8,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        self.merch_embed = nn.Embedding(num_merchants, embed_dim_merch)
        self.chan_embed = nn.Embedding(num_channels, embed_dim_chan)
        
        in_dim = embed_dim_merch + embed_dim_chan + num_features_dim
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        
    def forward(self, seq_num, seq_merch, seq_chan):
        emb_m = self.merch_embed(seq_merch)
        emb_c = self.chan_embed(seq_chan)
        x_seq = torch.cat([seq_num, emb_m, emb_c], dim=-1)
        
        _, (hn, _) = self.lstm(x_seq)
        return torch.sigmoid(self.head(hn[-1]))


class HybridDualBranchNet(nn.Module):
    def __init__(
        self,
        num_merchants: int,
        num_channels: int,
        num_features_dim: int = 4,
        agg_features_dim: int = 12,
        embed_dim_merch: int = 16,
        embed_dim_chan: int = 8,
        lstm_hidden_dim: int = 64,
        mlp_hidden_dim: int = 64,
        fusion_dim: int = 64,
        dropout: float = 0.2
    ):
        super().__init__()
        self.merch_embed = nn.Embedding(num_merchants, embed_dim_merch)
        self.chan_embed = nn.Embedding(num_channels, embed_dim_chan)
        in_dim = embed_dim_merch + embed_dim_chan + num_features_dim
        
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )
        self.seq_proj = nn.Linear(lstm_hidden_dim, 32)
        
        self.mlp_branch = nn.Sequential(
            nn.Linear(agg_features_dim, mlp_hidden_dim),
            nn.BatchNorm1d(mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(32 + 32, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, 1)
        )
        
    def forward(self, seq_num, seq_merch, seq_chan, agg_features):
        emb_m = self.merch_embed(seq_merch)
        emb_c = self.chan_embed(seq_chan)
        x_seq = torch.cat([seq_num, emb_m, emb_c], dim=-1)
        
        _, (hn, _) = self.lstm(x_seq)
        seq_latent = F.relu(self.seq_proj(hn[-1]))
        agg_latent = self.mlp_branch(agg_features)
        
        fused = torch.cat([seq_latent, agg_latent], dim=-1)
        return torch.sigmoid(self.fusion(fused))
