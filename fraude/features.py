"""Datos sinteticos, ingenieria de variables y FeaturePipeline.

Codigo extraido SIN CAMBIOS del notebook proyecto1_monitoreo_transaccional.ipynb
(celdas 2 y 3). No depende de PyTorch.
"""
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ==========================================
# 2. GENERADOR SINTÉTICO Y SPLIT TEMPORAL
# ==========================================

def generate_synthetic_transactions(
    n_clients: int = 1200,
    days: int = 90,
    fraud_rate: float = 0.035,
    seed: int = 42
) -> pd.DataFrame:
    np.random.seed(seed)
    start_date = datetime(2026, 1, 1, 0, 0, 0)
    
    merchants = {
        'SUPERMARKET': {'base_amount': (100, 450), 'channels': ['POS', 'ONLINE']},
        'GAS_STATION': {'base_amount': (150, 350), 'channels': ['POS']},
        'RESTAURANT': {'base_amount': (80, 500), 'channels': ['POS', 'ONLINE']},
        'ELECTRONICS': {'base_amount': (800, 4000), 'channels': ['POS', 'ONLINE']},
        'LUXURY_JEWELRY': {'base_amount': (2500, 9000), 'channels': ['POS', 'ONLINE']},
        'DIGITAL_SERVICE': {'base_amount': (20, 150), 'channels': ['ONLINE']},
        'ATM_CASH': {'base_amount': (200, 2000), 'channels': ['ATM']},
        'TRANSFER': {'base_amount': (100, 3000), 'channels': ['TRANSFER']}
    }
    merchant_types = list(merchants.keys())
    
    client_profiles = []
    for client_id in range(1, n_clients + 1):
        freq = np.random.uniform(0.5, 3.5)
        avg_spending = np.random.lognormal(mean=5.2, sigma=0.5)
        pref_merchant = np.random.choice(merchant_types, p=[0.35, 0.2, 0.2, 0.05, 0.02, 0.1, 0.05, 0.03])
        is_edge_case_client = (client_id % 40 == 0)
        client_profiles.append({
            'client_id': f"CLI_{client_id:04d}",
            'freq': freq,
            'avg_spending': avg_spending,
            'pref_merchant': pref_merchant,
            'is_edge_case': is_edge_case_client
        })
    
    transactions = []
    tx_counter = 1
    
    for prof in client_profiles:
        cid = prof['client_id']
        current_time = start_date + timedelta(hours=float(np.random.uniform(0, 48)))
        end_time = start_date + timedelta(days=days)
        
        while current_time < end_time:
            hours_to_next = np.random.exponential(scale=24.0 / prof['freq'])
            current_time += timedelta(hours=float(hours_to_next))
            if current_time >= end_time:
                break
                
            is_fraud = np.random.rand() < fraud_rate
            fraud_type = 'none'
            
            if is_fraud:
                fraud_mechanism = np.random.choice(
                    ['probing_drainage', 'channel_velocity_shift', 'amount_spike'],
                    p=[0.45, 0.35, 0.20]
                )
                
                if fraud_mechanism == 'probing_drainage':
                    for probe_i in range(3):
                        p_time = current_time + timedelta(minutes=float((probe_i + 1) * np.random.uniform(2, 6)))
                        transactions.append({
                            'tx_id': f"TX_{tx_counter:07d}",
                            'client_id': cid,
                            'timestamp': p_time,
                            'amount': round(float(np.random.uniform(5.0, 25.0)), 2),
                            'merchant_category': 'DIGITAL_SERVICE',
                            'channel': 'ONLINE',
                            'is_international': 1,
                            'is_fraud': 1,
                            'fraud_type': 'probing_drainage'
                        })
                        tx_counter += 1
                    
                    drain_time = current_time + timedelta(minutes=float(4 * np.random.uniform(7, 12)))
                    transactions.append({
                        'tx_id': f"TX_{tx_counter:07d}",
                        'client_id': cid,
                        'timestamp': drain_time,
                        'amount': round(float(np.random.uniform(3500.0, 8500.0)), 2),
                        'merchant_category': 'LUXURY_JEWELRY' if np.random.rand() > 0.5 else 'ELECTRONICS',
                        'channel': 'ONLINE',
                        'is_international': 1,
                        'is_fraud': 1,
                        'fraud_type': 'probing_drainage'
                    })
                    tx_counter += 1
                    current_time = drain_time + timedelta(hours=float(np.random.uniform(12, 36)))
                    continue
                    
                elif fraud_mechanism == 'channel_velocity_shift':
                    n_burst = np.random.randint(2, 4)
                    for burst_i in range(n_burst):
                        b_time = current_time + timedelta(minutes=float((burst_i + 1) * np.random.uniform(1, 4)))
                        transactions.append({
                            'tx_id': f"TX_{tx_counter:07d}",
                            'client_id': cid,
                            'timestamp': b_time,
                            'amount': round(float(np.random.uniform(800.0, 2200.0)), 2),
                            'merchant_category': np.random.choice(['ELECTRONICS', 'DIGITAL_SERVICE', 'TRANSFER']),
                            'channel': 'ONLINE',
                            'is_international': 1,
                            'is_fraud': 1,
                            'fraud_type': 'channel_velocity_shift'
                        })
                        tx_counter += 1
                    current_time = b_time + timedelta(hours=float(np.random.uniform(8, 24)))
                    continue
                    
                elif fraud_mechanism == 'amount_spike':
                    transactions.append({
                        'tx_id': f"TX_{tx_counter:07d}",
                        'client_id': cid,
                        'timestamp': current_time,
                        'amount': round(float(np.random.uniform(4000.0, 9500.0)), 2),
                        'merchant_category': 'ATM_CASH' if np.random.rand() > 0.5 else 'TRANSFER',
                        'channel': 'ATM' if np.random.rand() > 0.5 else 'TRANSFER',
                        'is_international': 0,
                        'is_fraud': 1,
                        'fraud_type': 'amount_spike'
                    })
                    tx_counter += 1
                    continue
            
            # Legítima
            m_cat = prof['pref_merchant'] if np.random.rand() > 0.25 else np.random.choice(merchant_types)
            allowed_channels = merchants[m_cat]['channels']
            channel = np.random.choice(allowed_channels)
            
            if prof['is_edge_case'] and np.random.rand() < 0.08:
                amount = float(np.random.uniform(2500, 6000))
                is_intl = 1
                f_type = 'legit_travel_outlier'
            else:
                low_b, high_b = merchants[m_cat]['base_amount']
                amount = float(np.random.uniform(low_b, high_b)) * np.random.uniform(0.8, 1.3)
                is_intl = 1 if (np.random.rand() < 0.02) else 0
                f_type = 'none'
                
            transactions.append({
                'tx_id': f"TX_{tx_counter:07d}",
                'client_id': cid,
                'timestamp': current_time,
                'amount': round(amount, 2),
                'merchant_category': m_cat,
                'channel': channel,
                'is_international': is_intl,
                'is_fraud': 0,
                'fraud_type': f_type
            })
            tx_counter += 1

    df = pd.DataFrame(transactions)
    df = df.sort_values(by='timestamp').reset_index(drop=True)
    return df

def create_temporal_split(df: pd.DataFrame, train_days: int = 60, val_days: int = 15):
    min_time = df['timestamp'].min()
    val_cutoff = min_time + timedelta(days=train_days)
    test_cutoff = val_cutoff + timedelta(days=val_days)
    
    train_df = df[df['timestamp'] < val_cutoff].copy().reset_index(drop=True)
    val_df = df[(df['timestamp'] >= val_cutoff) & (df['timestamp'] < test_cutoff)].copy().reset_index(drop=True)
    test_df = df[df['timestamp'] >= test_cutoff].copy().reset_index(drop=True)
    
    return train_df, val_df, test_df


# ==========================================
# 3. INGENIERÍA DE CARACTERÍSTICAS Y PIPELINE
# ==========================================

def compute_aggregated_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(by=['client_id', 'timestamp']).reset_index(drop=True)
    
    df['prev_timestamp'] = df.groupby('client_id')['timestamp'].shift(1)
    df['delta_time_hours'] = (df['timestamp'] - df['prev_timestamp']).dt.total_seconds() / 3600.0
    df['delta_time_hours'] = df['delta_time_hours'].fillna(999.0)
    
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    df['cumsum_amount'] = df.groupby('client_id')['amount'].cumsum() - df['amount']
    df['cumcount_tx'] = df.groupby('client_id').cumcount()
    df['hist_avg_amount'] = df['cumsum_amount'] / df['cumcount_tx'].replace(0, 1)
    df['hist_avg_amount'] = df['hist_avg_amount'].fillna(df['amount'])
    
    df['amount_to_hist_avg_ratio'] = df['amount'] / (df['hist_avg_amount'] + 1e-5)
    
    grouped = df.groupby('client_id')
    df['roll_mean_5'] = grouped['amount'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean()).fillna(df['amount'])
    df['roll_max_5'] = grouped['amount'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).max()).fillna(df['amount'])
    df['roll_std_5'] = grouped['amount'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).std()).fillna(0.0)
    
    df['amount_to_roll_mean_ratio'] = df['amount'] / (df['roll_mean_5'] + 1e-5)
    df['roll_intl_count_5'] = grouped['is_international'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).sum()).fillna(0.0)
    
    df = df.drop(columns=['prev_timestamp', 'cumsum_amount', 'cumcount_tx'])
    df = df.sort_values(by='timestamp').reset_index(drop=True)
    return df

class FeaturePipeline:
    def __init__(self, seq_len: int = 6):
        self.seq_len = seq_len
        self.scaler_agg = StandardScaler()
        self.scaler_num_seq = StandardScaler()
        self.le_merchant = LabelEncoder()
        self.le_channel = LabelEncoder()
        
        self.agg_feature_cols = [
            'amount', 'delta_time_hours', 'hour_of_day', 'day_of_week',
            'hist_avg_amount', 'amount_to_hist_avg_ratio',
            'roll_mean_5', 'roll_max_5', 'roll_std_5',
            'amount_to_roll_mean_ratio', 'roll_intl_count_5',
            'is_international'
        ]
        self.is_fitted = False
        
    def fit(self, train_df: pd.DataFrame):
        merchants = list(train_df['merchant_category'].unique()) + ['UNKNOWN']
        channels = list(train_df['channel'].unique()) + ['UNKNOWN']
        
        self.le_merchant.fit(merchants)
        self.le_channel.fit(channels)
        
        self.scaler_agg.fit(train_df[self.agg_feature_cols])
        seq_num_cols = ['amount', 'delta_time_hours', 'hour_of_day', 'is_international']
        self.scaler_num_seq.fit(train_df[seq_num_cols])
        
        self.is_fitted = True
        return self

    def build_sequence_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        if not self.is_fitted:
            raise ValueError("Pipeline no ha sido ajustado. Llama a .fit() primero.")
            
        merchant_ids = self.le_merchant.transform(
            df['merchant_category'].apply(lambda x: x if x in self.le_merchant.classes_ else 'UNKNOWN')
        )
        channel_ids = self.le_channel.transform(
            df['channel'].apply(lambda x: x if x in self.le_channel.classes_ else 'UNKNOWN')
        )
        
        seq_num_cols = ['amount', 'delta_time_hours', 'hour_of_day', 'is_international']
        scaled_num = self.scaler_num_seq.transform(df[seq_num_cols])
        
        df_processed = df.copy()
        df_processed['merchant_id'] = merchant_ids
        df_processed['channel_id'] = channel_ids
        for idx, col in enumerate(seq_num_cols):
            df_processed[f'scaled_{col}'] = scaled_num[:, idx]
            
        client_groups = df_processed.groupby('client_id')
        
        seq_num_list = []
        seq_cat_merchant_list = []
        seq_cat_channel_list = []
        labels_list = []
        fraud_types_list = []
        tx_ids_list = []
        agg_features_list = []
        
        X_agg = self.scaler_agg.transform(df[self.agg_feature_cols])
        df_processed['orig_idx'] = np.arange(len(df_processed))
        
        for client_id, group in client_groups:
            group = group.sort_values(by='timestamp')
            num_arr = group[[f'scaled_{col}' for col in seq_num_cols]].values
            merch_arr = group['merchant_id'].values
            chan_arr = group['channel_id'].values
            y_arr = group['is_fraud'].values
            ft_arr = group['fraud_type'].values
            tx_arr = group['tx_id'].values
            orig_idx_arr = group['orig_idx'].values
            
            n_tx = len(group)
            for i in range(n_tx):
                start_i = max(0, i - self.seq_len + 1)
                window_num = num_arr[start_i : i + 1]
                window_merch = merch_arr[start_i : i + 1]
                window_chan = chan_arr[start_i : i + 1]
                
                curr_len = len(window_num)
                pad_len = self.seq_len - curr_len
                
                if pad_len > 0:
                    pad_num = np.zeros((pad_len, num_arr.shape[1]), dtype=np.float32)
                    pad_merch = np.zeros(pad_len, dtype=np.int64)
                    pad_chan = np.zeros(pad_len, dtype=np.int64)
                    
                    full_window_num = np.vstack([pad_num, window_num])
                    full_window_merch = np.concatenate([pad_merch, window_merch])
                    full_window_chan = np.concatenate([pad_chan, window_chan])
                else:
                    full_window_num = window_num
                    full_window_merch = window_merch
                    full_window_chan = window_chan
                    
                seq_num_list.append(full_window_num)
                seq_cat_merchant_list.append(full_window_merch)
                seq_cat_channel_list.append(full_window_chan)
                labels_list.append(y_arr[i])
                fraud_types_list.append(ft_arr[i])
                tx_ids_list.append(tx_arr[i])
                agg_features_list.append(X_agg[orig_idx_arr[i]])
                
        return {
            'seq_num': np.array(seq_num_list, dtype=np.float32),
            'seq_merch': np.array(seq_cat_merchant_list, dtype=np.int64),
            'seq_chan': np.array(seq_cat_channel_list, dtype=np.int64),
            'agg_features': np.array(agg_features_list, dtype=np.float32),
            'labels': np.array(labels_list, dtype=np.float32),
            'fraud_types': np.array(fraud_types_list),
            'tx_ids': np.array(tx_ids_list)
        }

