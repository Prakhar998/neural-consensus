#!/usr/bin/env python3
"""
REAL DATA Baseline Comparison - FIXED VERSION

Uses actual states from your CSV files:
- node_metrics.csv: HEALTHY, WARNING, DEGRADED, DEGRADED_SEVERE, FAILED
- backblaze_sample.csv: failure column (0/1)

Binary classification: HEALTHY vs ANY_ISSUE
"""

import numpy as np
import pandas as pd
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
import time

# Import model architecture
from train_predictive import (
    PredictiveModel, NUM_FEATURES, WINDOW_SIZE, NUM_PREDICTION_CLASSES
)


# =============================================================================
# LOAD REAL DATA
# =============================================================================

def load_node_metrics(filepath='datasets/node_metrics.csv'):
    """Load real node metrics."""
    print(f"\n📂 Loading {filepath}...")
    df = pd.read_csv(filepath)
    print(f"   Records: {len(df):,}")
    print(f"   Machines: {df['machine_id'].nunique()}")
    print(f"   States: {dict(df['state'].value_counts())}")
    return df


def load_backblaze_data(filepath='datasets/backblaze_sample.csv'):
    """Load Backblaze data."""
    print(f"\n📂 Loading {filepath}...")
    df = pd.read_csv(filepath)
    print(f"   Records: {len(df):,}")
    print(f"   Drives: {df['serial_number'].nunique()}")
    print(f"   Failures: {df['failure'].sum()}")
    return df


# =============================================================================
# FEATURE EXTRACTION - USING ACTUAL CSV COLUMNS
# =============================================================================

def extract_from_node_metrics(df, window_size=50):
    """
    Extract features using ACTUAL columns from node_metrics.csv.
    
    Uses real states: HEALTHY, WARNING, DEGRADED, DEGRADED_SEVERE, FAILED
    Binary label: 0 = HEALTHY, 1 = any other state
    """
    print(f"\n🔧 Extracting from node_metrics (window={window_size})...")
    
    all_features = []
    all_labels = []
    
    machines = df['machine_id'].unique()
    
    for machine_id in machines:
        machine_data = df[df['machine_id'] == machine_id].sort_values('timestamp')
        
        if len(machine_data) < window_size:
            continue
        
        # Sliding windows with 50% overlap
        for start in range(0, len(machine_data) - window_size + 1, window_size // 2):
            window = machine_data.iloc[start:start + window_size]
            
            # Build feature matrix [window_size, NUM_FEATURES]
            features = np.zeros((window_size, NUM_FEATURES), dtype=np.float32)
            
            for i, (_, row) in enumerate(window.iterrows()):
                # Latency features (0-7) - DIRECTLY from CSV
                lat = row['heartbeat_latency_ms']
                features[i, 0] = min(1.0, lat / 200.0)
                features[i, 1] = 0  # jitter - computed from window
                features[i, 2] = 0  # trend - computed from window
                
                # Message features (8-13)
                features[i, 8] = min(1.0, row['messages_received'] / 50.0)
                features[i, 9] = min(1.0, row['messages_sent'] / 50.0)
                total_msg = row['messages_sent'] + row['messages_received'] + 1
                features[i, 10] = row['messages_dropped'] / total_msg
                
                # Heartbeat features (14-18)
                features[i, 14] = row['response_rate']
                features[i, 15] = min(1.0, row['missed_heartbeats'] / 5.0)
                features[i, 16] = 1.0 if row['missed_heartbeats'] > 2 else 0.0
                
                # Response time proxy
                features[i, 19] = min(1.0, lat * 0.8 / 200.0)
                features[i, 20] = min(1.0, lat * 2.0 / 500.0)
                
                # Resource usage
                features[i, 23] = row['cpu_usage']
                features[i, 24] = row['memory_usage']
            
            # Window-level statistics
            latencies = window['heartbeat_latency_ms'].values
            features[:, 3] = np.mean(latencies) / 200.0
            features[:, 4] = np.std(latencies) / 100.0
            features[:, 5] = np.min(latencies) / 200.0
            features[:, 6] = np.max(latencies) / 200.0
            features[:, 7] = (np.max(latencies) - np.min(latencies)) / 200.0
            
            # Jitter (std of differences)
            if len(latencies) > 1:
                jitter = np.std(np.diff(latencies))
                features[:, 1] = min(1.0, jitter / 50.0)
                trend = (latencies[-1] - latencies[0]) / len(latencies)
                features[:, 2] = np.tanh(trend / 10.0)
            
            all_features.append(features)
            
            # BINARY LABEL from actual state column
            final_state = window.iloc[-1]['state']
            if final_state == 'HEALTHY':
                label = 0  # Healthy
            else:
                label = 1  # Any issue (WARNING, DEGRADED, FAILED, etc.)
            
            all_labels.append(label)
    
    X = np.array(all_features)
    y = np.array(all_labels)
    
    print(f"   Sequences: {len(X)}")
    print(f"   Healthy: {np.sum(y == 0)}, Issues: {np.sum(y == 1)}")
    
    return X, y


def extract_from_backblaze(df, window_size=50):
    """
    Extract features from Backblaze SMART data.
    
    Maps SMART attributes to node metrics format.
    Binary label: 0 = healthy drive, 1 = failing/failed drive
    """
    print(f"\n🔧 Extracting from Backblaze (window={window_size})...")
    
    all_features = []
    all_labels = []
    
    drives = df['serial_number'].unique()
    
    for drive_id in drives:
        drive_data = df[df['serial_number'] == drive_id].sort_values('date')
        
        if len(drive_data) < window_size:
            continue
        
        for start in range(0, len(drive_data) - window_size + 1, window_size // 2):
            window = drive_data.iloc[start:start + window_size]
            
            features = np.zeros((window_size, NUM_FEATURES), dtype=np.float32)
            health_scores = []
            
            for i, (_, row) in enumerate(window.iterrows()):
                # SMART attributes
                s5 = (row.get('smart_5_raw', 0) or 0)    # Reallocated sectors
                s187 = (row.get('smart_187_raw', 0) or 0)  # Uncorrectable errors
                s188 = (row.get('smart_188_raw', 0) or 0)  # Command timeout
                s197 = (row.get('smart_197_raw', 0) or 0)  # Pending sectors
                s198 = (row.get('smart_198_raw', 0) or 0)  # Offline uncorrectable
                s194 = (row.get('smart_194_raw', 30) or 30)  # Temperature
                s9 = (row.get('smart_9_raw', 0) or 0)    # Power-on hours
                
                # Health score (0=good, 1=bad)
                health = min(1.0, (
                    (s5 / 100) * 0.25 +
                    (s187 / 50) * 0.20 +
                    (s188 / 20) * 0.15 +
                    (s197 / 50) * 0.20 +
                    (s198 / 50) * 0.20
                ))
                health_scores.append(health)
                
                # Map to latency-like metrics
                lat = 20 + health * 180  # 20-200ms based on health
                features[i, 0] = min(1.0, lat / 200.0)
                features[i, 14] = max(0.2, 1.0 - health * 0.7)  # response rate
                features[i, 15] = min(1.0, health)  # missed heartbeats proxy
                features[i, 16] = 1.0 if health > 0.4 else 0.0
                features[i, 10] = health * 0.3  # message drops
                features[i, 23] = min(1.0, s194 / 60.0)  # temp
                features[i, 24] = min(1.0, s9 / 50000.0)  # hours
            
            # Window stats
            hs = np.array(health_scores)
            features[:, 3] = np.mean(hs)
            features[:, 4] = np.std(hs)
            features[:, 5] = np.min(hs)
            features[:, 6] = np.max(hs)
            if len(hs) > 1:
                features[:, 2] = np.tanh((hs[-1] - hs[0]) * 5)
            
            all_features.append(features)
            
            # Binary label
            has_failure = window['failure'].max()
            dtf = window.iloc[-1].get('days_to_failure', -1)
            
            if has_failure == 1 or (dtf != -1 and dtf < 14):
                label = 1  # Issue
            elif hs[-1] > 0.3:
                label = 1  # Degraded
            else:
                label = 0  # Healthy
            
            all_labels.append(label)
    
    X = np.array(all_features)
    y = np.array(all_labels)
    
    print(f"   Sequences: {len(X)}")
    print(f"   Healthy: {np.sum(y == 0)}, Issues: {np.sum(y == 1)}")
    
    return X, y


# =============================================================================
# MODELS
# =============================================================================

class ThresholdDetector:
    """Threshold-based detector."""
    
    def fit(self, X, y):
        return self
    
    def predict(self, X):
        preds = np.zeros(len(X), dtype=int)
        for i, sample in enumerate(X):
            lat = np.mean(sample[:, 0]) * 200
            resp = np.mean(sample[:, 14])
            missed = np.max(sample[:, 15]) * 5
            
            if missed > 2 or resp < 0.7 or lat > 80:
                preds[i] = 1
            else:
                preds[i] = 0
        return preds


class PhiAccrualDetector:
    """Phi Accrual detector."""
    
    def __init__(self):
        self.mean = 0.1
        self.std = 0.05
    
    def fit(self, X, y):
        healthy = X[y == 0]
        if len(healthy) > 0:
            lats = healthy[:, -1, 0]
            self.mean = np.mean(lats)
            self.std = max(np.std(lats), 0.02)
        return self
    
    def predict(self, X):
        preds = np.zeros(len(X), dtype=int)
        for i, sample in enumerate(X):
            lat = np.mean(sample[:, 0])
            phi = max(0, ((lat - self.mean) / self.std) ** 2 / 2)
            preds[i] = 1 if phi > 5 else 0
        return preds


class RandomForestBaseline:
    """Random Forest."""
    
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100, max_depth=15, random_state=42,
            n_jobs=-1, class_weight='balanced'
        )
    
    def _feat(self, X):
        feats = []
        for sample in X:
            f = []
            for j in range(min(sample.shape[1], 25)):
                seq = sample[:, j]
                f.extend([seq[-1], seq.mean(), seq.std(), seq.min(), seq.max()])
            feats.append(f)
        return np.array(feats)
    
    def fit(self, X, y):
        self.model.fit(self._feat(X), y)
        return self
    
    def predict(self, X):
        return self.model.predict(self._feat(X))


class VanillaLSTM(nn.Module):
    """Simple LSTM."""
    
    def __init__(self, inp, hid=64):
        super().__init__()
        self.lstm = nn.LSTM(inp, hid, 2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hid, 2)  # Binary
    
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def train_lstm(X_train, y_train, epochs=50):
    """Train LSTM."""
    model = VanillaLSTM(NUM_FEATURES, 64)
    
    weights = torch.tensor([1.0, len(y_train) / (np.sum(y_train) + 1)], dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.long)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=32, shuffle=True)
    
    model.train()
    for ep in range(epochs):
        loss_sum = 0
        for bx, by in loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
        
        if (ep + 1) % 10 == 0:
            print(f"      LSTM Epoch {ep+1}/{epochs}: loss={loss_sum/len(loader):.4f}")
    
    return model


class BinaryProactiveGuard(nn.Module):
    """ProactiveGuard for binary classification."""
    
    def __init__(self, input_size=32, hidden_size=128, seq_len=50):
        super().__init__()
        
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # ResNet-style conv
        self.conv = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, 7, padding=3),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Conv1d(hidden_size, hidden_size, 3, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # LSTM
        self.lstm = nn.LSTM(hidden_size, hidden_size // 2, 2, 
                           batch_first=True, bidirectional=True, dropout=0.2)
        
        # Attention
        self.attention = nn.MultiheadAttention(hidden_size, 4, dropout=0.1, batch_first=True)
        self.attn_norm = nn.LayerNorm(hidden_size)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2)  # Binary output
        )
    
    def forward(self, x):
        x = self.input_proj(x)
        
        # CNN
        x_cnn = x.permute(0, 2, 1)
        x_cnn = self.conv(x_cnn).squeeze(-1)
        
        # LSTM
        x_lstm, _ = self.lstm(x)
        x_lstm = x_lstm[:, -1, :]
        
        # Attention
        x_attn, _ = self.attention(x, x, x)
        x_attn = self.attn_norm(x + x_attn)
        x_attn = x_attn.mean(dim=1)
        
        # Combine
        combined = torch.cat([x_cnn, x_lstm, x_attn], dim=1)
        return self.classifier(combined)


def train_proactiveguard(X_train, y_train, epochs=60):
    """Train ProactiveGuard."""
    model = BinaryProactiveGuard(NUM_FEATURES, 128, WINDOW_SIZE)
    
    # Class weights for imbalance
    n_pos = np.sum(y_train)
    n_neg = len(y_train) - n_pos
    weights = torch.tensor([1.0, n_neg / (n_pos + 1)], dtype=torch.float32)
    
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.long)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=64, shuffle=True)
    
    model.train()
    for ep in range(epochs):
        loss_sum, correct, total = 0, 0, 0
        
        for bx, by in loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            loss_sum += loss.item()
            correct += (out.argmax(1) == by).sum().item()
            total += len(by)
        
        scheduler.step()
        
        if (ep + 1) % 15 == 0:
            print(f"      PG Epoch {ep+1}/{epochs}: loss={loss_sum/len(loader):.4f}, acc={correct/total:.4f}")
    
    return model


# =============================================================================
# METRICS
# =============================================================================

def calc_metrics(y_true, y_pred):
    """Calculate metrics for binary classification."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    acc = np.mean(y_true == y_pred)
    
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    
    fpr = fp / (fp + tn) * 100 if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) * 100 if (fn + tp) > 0 else 0
    
    return {
        'accuracy': round(acc, 4),
        'precision': round(prec, 4),
        'recall': round(rec, 4),
        'f1': round(f1, 4),
        'fpr': round(fpr, 2),
        'fnr': round(fnr, 2),
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)
    }


# =============================================================================
# MAIN
# =============================================================================

def run_comparison():
    """Run comparison on real data."""
    print("="*70)
    print("REAL DATA BASELINE COMPARISON")
    print("Binary Classification: Healthy vs Any Issue")
    print("="*70)
    
    # Load data
    node_df = load_node_metrics('datasets/node_metrics.csv')
    bb_df = load_backblaze_data('datasets/backblaze_sample.csv')
    
    # Extract features
    X_node, y_node = extract_from_node_metrics(node_df)
    X_bb, y_bb = extract_from_backblaze(bb_df)
    
    # Combine
    print(f"\n📊 Combining datasets...")
    X_all = np.concatenate([X_node, X_bb])
    y_all = np.concatenate([y_node, y_bb])
    print(f"   Total: {len(X_all)} sequences")
    print(f"   Healthy: {np.sum(y_all == 0)}, Issues: {np.sum(y_all == 1)}")
    
    # Split (no stratify needed for binary with enough samples)
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42
    )
    print(f"   Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"   Test healthy: {np.sum(y_test == 0)}, issues: {np.sum(y_test == 1)}")
    
    results = {}
    
    # 1. ProactiveGuard
    print("\n" + "="*70)
    print("[1/5] PROACTIVEGUARD (ResNet + LSTM + Attention)")
    print("="*70)
    
    pg = train_proactiveguard(X_train, y_train, epochs=60)
    pg.eval()
    
    start = time.time()
    with torch.no_grad():
        y_pred = pg(torch.tensor(X_test, dtype=torch.float32)).argmax(1).numpy()
    pg_time = (time.time() - start) * 1000 / len(X_test)
    
    results['ProactiveGuard'] = calc_metrics(y_test, y_pred)
    results['ProactiveGuard']['time_ms'] = round(pg_time, 3)
    print(f"   Accuracy: {results['ProactiveGuard']['accuracy']}")
    print(f"   F1: {results['ProactiveGuard']['f1']}, Precision: {results['ProactiveGuard']['precision']}, Recall: {results['ProactiveGuard']['recall']}")
    print(f"   FPR: {results['ProactiveGuard']['fpr']}%, FNR: {results['ProactiveGuard']['fnr']}%")
    
    # 2. Threshold
    print("\n" + "="*70)
    print("[2/5] THRESHOLD-BASED")
    print("="*70)
    
    t = ThresholdDetector().fit(X_train, y_train)
    start = time.time()
    y_pred = t.predict(X_test)
    t_time = (time.time() - start) * 1000 / len(X_test)
    
    results['Threshold'] = calc_metrics(y_test, y_pred)
    results['Threshold']['time_ms'] = round(t_time, 3)
    print(f"   Accuracy: {results['Threshold']['accuracy']}, F1: {results['Threshold']['f1']}")
    print(f"   FPR: {results['Threshold']['fpr']}%, FNR: {results['Threshold']['fnr']}%")
    
    # 3. Phi Accrual
    print("\n" + "="*70)
    print("[3/5] PHI ACCRUAL")
    print("="*70)
    
    p = PhiAccrualDetector().fit(X_train, y_train)
    start = time.time()
    y_pred = p.predict(X_test)
    p_time = (time.time() - start) * 1000 / len(X_test)
    
    results['Phi Accrual'] = calc_metrics(y_test, y_pred)
    results['Phi Accrual']['time_ms'] = round(p_time, 3)
    print(f"   Accuracy: {results['Phi Accrual']['accuracy']}, F1: {results['Phi Accrual']['f1']}")
    print(f"   FPR: {results['Phi Accrual']['fpr']}%, FNR: {results['Phi Accrual']['fnr']}%")
    
    # 4. Random Forest
    print("\n" + "="*70)
    print("[4/5] RANDOM FOREST")
    print("="*70)
    
    rf = RandomForestBaseline().fit(X_train, y_train)
    start = time.time()
    y_pred = rf.predict(X_test)
    rf_time = (time.time() - start) * 1000 / len(X_test)
    
    results['Random Forest'] = calc_metrics(y_test, y_pred)
    results['Random Forest']['time_ms'] = round(rf_time, 3)
    print(f"   Accuracy: {results['Random Forest']['accuracy']}, F1: {results['Random Forest']['f1']}")
    print(f"   FPR: {results['Random Forest']['fpr']}%, FNR: {results['Random Forest']['fnr']}%")
    
    # 5. Vanilla LSTM
    print("\n" + "="*70)
    print("[5/5] VANILLA LSTM")
    print("="*70)
    
    lstm = train_lstm(X_train, y_train, epochs=50)
    lstm.eval()
    
    start = time.time()
    with torch.no_grad():
        y_pred = lstm(torch.tensor(X_test, dtype=torch.float32)).argmax(1).numpy()
    lstm_time = (time.time() - start) * 1000 / len(X_test)
    
    results['Vanilla LSTM'] = calc_metrics(y_test, y_pred)
    results['Vanilla LSTM']['time_ms'] = round(lstm_time, 3)
    print(f"   Accuracy: {results['Vanilla LSTM']['accuracy']}, F1: {results['Vanilla LSTM']['f1']}")
    print(f"   FPR: {results['Vanilla LSTM']['fpr']}%, FNR: {results['Vanilla LSTM']['fnr']}%")
    
    # ===================
    # SUMMARY
    # ===================
    print("\n" + "="*70)
    print("FINAL RESULTS ON REAL DATA")
    print("="*70)
    
    print(f"\n{'Method':<18} {'Acc':<8} {'F1':<8} {'Prec':<8} {'Recall':<8} {'FPR%':<8} {'FNR%':<8}")
    print("-"*66)
    
    for name, r in sorted(results.items(), key=lambda x: x[1]['f1'], reverse=True):
        print(f"{name:<18} {r['accuracy']:<8} {r['f1']:<8} {r['precision']:<8} "
              f"{r['recall']:<8} {r['fpr']:<8} {r['fnr']:<8}")
    
    # Save
    Path('results/baselines').mkdir(parents=True, exist_ok=True)
    with open('results/baselines/real_data_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Saved to results/baselines/real_data_results.json")
    
    # LaTeX
    print("\n" + "="*70)
    print("LATEX TABLE FOR PAPER")
    print("="*70)
    
    print("""
\\begin{table}[t]
\\centering
\\caption{Comparison of Failure Detection Methods on Real-World Data}
\\label{tab:baseline}
\\begin{tabular}{lcccccc}
\\toprule
\\textbf{Method} & \\textbf{Accuracy} & \\textbf{F1} & \\textbf{Precision} & \\textbf{Recall} & \\textbf{FPR} & \\textbf{FNR} \\\\
\\midrule""")
    
    for name, r in sorted(results.items(), key=lambda x: x[1]['f1'], reverse=True):
        print(f"{name} & {r['accuracy']} & {r['f1']} & {r['precision']} & {r['recall']} & {r['fpr']}\\% & {r['fnr']}\\% \\\\")
    
    print("""\\bottomrule
\\end{tabular}
\\vspace{1mm}

\\footnotesize{Data sources: Google Cluster Traces (72,000 node observations) and Backblaze Hard Drive Data (10,000 SMART records). Binary classification task: healthy vs. any failure/degradation.}
\\end{table}
""")
    
    return results


if __name__ == "__main__":
    run_comparison()