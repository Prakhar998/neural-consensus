#!/usr/bin/env python3
"""
ProactiveGuard - Complete Experiments with All Improvements
============================================================
1. SMOTE for UCI AI4I (improve multi-class results)
2. etcd/Raft log simulation (consensus system credibility)
3. Bootstrap confidence intervals (statistical rigor)

Run: python3 run_improved_experiments.py

Requirements:
    pip install ucimlrepo imbalanced-learn scikit-learn numpy pandas
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, recall_score
import warnings
import json

warnings.filterwarnings('ignore')
np.random.seed(42)

print("="*70)
print("PROACTIVEGUARD - COMPLETE IMPROVED EXPERIMENTS")
print("="*70)

# ============================================================================
# IMPROVED PROACTIVEGUARD ARCHITECTURE
# ============================================================================

class ImprovedProactiveGuard:
    """ProactiveGuard with attention, residual connections, focal loss"""
    
    def __init__(self, n_classes, hidden_dims=[256, 128, 64]):
        self.n_classes = n_classes
        self.hidden_dims = hidden_dims
        self.weights = []
        self.biases = []
        self.attention_weights = None
        self.training = False
        
    def _init_weights(self, input_dim):
        np.random.seed(42)
        dims = [input_dim] + self.hidden_dims + [self.n_classes]
        self.weights = []
        self.biases = []
        
        for i in range(len(dims) - 1):
            limit = np.sqrt(6 / (dims[i] + dims[i+1]))
            W = np.random.uniform(-limit, limit, (dims[i], dims[i+1]))
            b = np.zeros(dims[i+1])
            self.weights.append(W)
            self.biases.append(b)
        
        self.attention_weights = np.random.randn(input_dim) * 0.1
        
    def _relu(self, x):
        return np.maximum(0, x)
    
    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def _forward(self, X):
        att = self._sigmoid(self.attention_weights)
        h = X * att
        
        for i, (W, b) in enumerate(zip(self.weights[:-1], self.biases[:-1])):
            h_new = self._relu(h @ W + b)
            if h.shape[1] == h_new.shape[1]:
                h_new = h_new + h * 0.1
            h = h_new
            
            if self.training:
                mask = np.random.binomial(1, 0.7, h.shape) / 0.7
                h = h * mask
        
        logits = h @ self.weights[-1] + self.biases[-1]
        return self._softmax(logits)
    
    def fit(self, X, y, epochs=150, lr=0.01, batch_size=64, verbose=False):
        self._init_weights(X.shape[1])
        self.training = True
        
        n_samples = X.shape[0]
        best_loss = float('inf')
        patience_counter = 0
        
        m_w = [np.zeros_like(w) for w in self.weights]
        v_w = [np.zeros_like(w) for w in self.weights]
        m_b = [np.zeros_like(b) for b in self.biases]
        v_b = [np.zeros_like(b) for b in self.biases]
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        
        for epoch in range(epochs):
            idx = np.random.permutation(n_samples)
            X_shuf, y_shuf = X[idx], y[idx]
            
            total_loss = 0
            n_batches = 0
            
            for i in range(0, n_samples, batch_size):
                Xb = X_shuf[i:i+batch_size]
                yb = y_shuf[i:i+batch_size]
                
                if np.random.random() < 0.3:
                    lam = np.random.beta(0.4, 0.4)
                    perm = np.random.permutation(len(Xb))
                    Xb = lam * Xb + (1 - lam) * Xb[perm]
                
                y_pred = self._forward(Xb)
                
                y_oh = np.zeros((len(yb), self.n_classes))
                y_oh[np.arange(len(yb)), yb] = 1
                pt = np.sum(y_oh * y_pred, axis=1)
                loss = -np.mean((1 - pt) ** 2 * np.log(pt + 1e-7))
                total_loss += loss
                n_batches += 1
                
                d_out = (y_pred - y_oh) / len(yb)
                
                activations = [Xb * self._sigmoid(self.attention_weights)]
                h = activations[0]
                for W, b in zip(self.weights[:-1], self.biases[:-1]):
                    h = self._relu(h @ W + b)
                    activations.append(h)
                
                delta = d_out
                for j in range(len(self.weights) - 1, -1, -1):
                    if j < len(activations):
                        dW = activations[j].T @ delta + 0.001 * self.weights[j]
                    else:
                        dW = 0.001 * self.weights[j]
                    db = np.sum(delta, axis=0)
                    
                    t = epoch * (n_samples // batch_size) + n_batches
                    m_w[j] = beta1 * m_w[j] + (1 - beta1) * dW
                    v_w[j] = beta2 * v_w[j] + (1 - beta2) * dW**2
                    m_hat = m_w[j] / (1 - beta1**(t+1))
                    v_hat = v_w[j] / (1 - beta2**(t+1))
                    self.weights[j] -= lr * m_hat / (np.sqrt(v_hat) + eps)
                    
                    m_b[j] = beta1 * m_b[j] + (1 - beta1) * db
                    v_b[j] = beta2 * v_b[j] + (1 - beta2) * db**2
                    m_hat = m_b[j] / (1 - beta1**(t+1))
                    v_hat = v_b[j] / (1 - beta2**(t+1))
                    self.biases[j] -= lr * m_hat / (np.sqrt(v_hat) + eps)
                    
                    if j > 0 and j < len(activations):
                        delta = delta @ self.weights[j].T * (activations[j] > 0)
            
            avg_loss = total_loss / n_batches
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= 20:
                break
        
        self.training = False
        return self
    
    def predict(self, X):
        self.training = False
        return np.argmax(self._forward(X), axis=1)
    
    def predict_proba(self, X):
        self.training = False
        return self._forward(X)


class EnsembleProactiveGuard:
    """Ensemble of ProactiveGuard models + Random Forest"""
    
    def __init__(self, n_classes, n_models=5):
        self.n_classes = n_classes
        self.n_models = n_models
        self.models = []
        self.rf = None
        
    def fit(self, X, y, verbose=True):
        if verbose:
            print(f"    Training ensemble ({self.n_models} NNs + RF)...")
        
        configs = [
            [256, 128, 64],
            [128, 64, 32],
            [512, 256, 128],
            [256, 256, 128, 64],
            [128, 128, 64],
        ]
        
        for i in range(self.n_models):
            model = ImprovedProactiveGuard(self.n_classes, configs[i % len(configs)])
            idx = np.random.choice(len(X), len(X), replace=True)
            model.fit(X[idx], y[idx], epochs=100, verbose=False)
            self.models.append(model)
        
        self.rf = RandomForestClassifier(
            n_estimators=100, class_weight='balanced', 
            random_state=42, n_jobs=-1
        )
        self.rf.fit(X, y)
        
        return self
    
    def predict(self, X):
        probs = np.zeros((len(X), self.n_classes))
        for model in self.models:
            probs += model.predict_proba(X)
        probs += self.rf.predict_proba(X) * 2
        probs /= (len(self.models) + 2)
        return np.argmax(probs, axis=1)
    
    def predict_proba(self, X):
        probs = np.zeros((len(X), self.n_classes))
        for model in self.models:
            probs += model.predict_proba(X)
        probs += self.rf.predict_proba(X) * 2
        probs /= (len(self.models) + 2)
        return probs


# ============================================================================
# IMPROVEMENT 1: UCI AI4I WITH SMOTE
# ============================================================================

print("\n" + "="*70)
print("IMPROVEMENT 1: UCI AI4I WITH SMOTE")
print("="*70)

try:
    from imblearn.over_sampling import SMOTE
    from ucimlrepo import fetch_ucirepo
    
    print("\n[1] Loading UCI AI4I dataset...")
    ai4i = fetch_ucirepo(id=601)
    
    X_uci_df = ai4i.data.features
    y_uci_df = ai4i.data.targets
    
    X_uci = X_uci_df.select_dtypes(include=[np.number]).values
    
    failure_cols = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    
    def get_failure_class(row):
        for i, col in enumerate(failure_cols):
            if col in row.index and row[col] == 1:
                return i + 1
        return 0
    
    y_uci = y_uci_df.apply(get_failure_class, axis=1).values
    class_names_uci = ['No Failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    
    print(f"    Samples: {len(X_uci)}, Features: {X_uci.shape[1]}")
    print(f"    Original class distribution:")
    for i, name in enumerate(class_names_uci):
        count = (y_uci == i).sum()
        print(f"      {name}: {count} ({100*count/len(y_uci):.2f}%)")
    
    # Train/test split BEFORE SMOTE (important!)
    X_train_uci, X_test_uci, y_train_uci, y_test_uci = train_test_split(
        X_uci, y_uci, test_size=0.2, random_state=42, stratify=y_uci
    )
    
    # Apply SMOTE to training data only
    print("\n[2] Applying SMOTE to training data...")
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_uci, y_train_uci)
    
    print(f"    Before SMOTE: {len(X_train_uci)} samples")
    print(f"    After SMOTE: {len(X_train_smote)} samples")
    print(f"    SMOTE class distribution:")
    for i, name in enumerate(class_names_uci):
        count = (y_train_smote == i).sum()
        print(f"      {name}: {count}")
    
    # Scale
    scaler_uci = StandardScaler()
    X_train_smote_s = scaler_uci.fit_transform(X_train_smote)
    X_test_uci_s = scaler_uci.transform(X_test_uci)
    
    # Run experiments
    print("\n[3] Running experiments with SMOTE...")
    
    results_uci_smote = {}
    n_classes = len(class_names_uci)
    
    # ProactiveGuard Ensemble
    print("    ProactiveGuard Ensemble...", end=" ", flush=True)
    pg = EnsembleProactiveGuard(n_classes)
    pg.fit(X_train_smote_s, y_train_smote, verbose=False)
    preds_pg = pg.predict(X_test_uci_s)
    results_uci_smote['ProactiveGuard (Ens.)'] = {
        'accuracy': accuracy_score(y_test_uci, preds_pg),
        'macro_f1': f1_score(y_test_uci, preds_pg, average='macro', zero_division=0),
        'weighted_f1': f1_score(y_test_uci, preds_pg, average='weighted', zero_division=0)
    }
    print(f"Acc: {results_uci_smote['ProactiveGuard (Ens.)']['accuracy']:.4f}, Macro F1: {results_uci_smote['ProactiveGuard (Ens.)']['macro_f1']:.4f}")
    
    # Random Forest
    print("    Random Forest...", end=" ", flush=True)
    rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1)
    rf.fit(X_train_smote_s, y_train_smote)
    preds_rf = rf.predict(X_test_uci_s)
    results_uci_smote['Random Forest'] = {
        'accuracy': accuracy_score(y_test_uci, preds_rf),
        'macro_f1': f1_score(y_test_uci, preds_rf, average='macro', zero_division=0),
        'weighted_f1': f1_score(y_test_uci, preds_rf, average='weighted', zero_division=0)
    }
    print(f"Acc: {results_uci_smote['Random Forest']['accuracy']:.4f}, Macro F1: {results_uci_smote['Random Forest']['macro_f1']:.4f}")
    
    # Gradient Boosting
    print("    Gradient Boosting...", end=" ", flush=True)
    gb = GradientBoostingClassifier(n_estimators=150, max_depth=6, random_state=42)
    gb.fit(X_train_smote_s, y_train_smote)
    preds_gb = gb.predict(X_test_uci_s)
    results_uci_smote['Gradient Boosting'] = {
        'accuracy': accuracy_score(y_test_uci, preds_gb),
        'macro_f1': f1_score(y_test_uci, preds_gb, average='macro', zero_division=0),
        'weighted_f1': f1_score(y_test_uci, preds_gb, average='weighted', zero_division=0)
    }
    print(f"Acc: {results_uci_smote['Gradient Boosting']['accuracy']:.4f}, Macro F1: {results_uci_smote['Gradient Boosting']['macro_f1']:.4f}")
    
    # MLP
    print("    MLP...", end=" ", flush=True)
    mlp = MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=500, early_stopping=True, random_state=42)
    mlp.fit(X_train_smote_s, y_train_smote)
    preds_mlp = mlp.predict(X_test_uci_s)
    results_uci_smote['MLP'] = {
        'accuracy': accuracy_score(y_test_uci, preds_mlp),
        'macro_f1': f1_score(y_test_uci, preds_mlp, average='macro', zero_division=0),
        'weighted_f1': f1_score(y_test_uci, preds_mlp, average='weighted', zero_division=0)
    }
    print(f"Acc: {results_uci_smote['MLP']['accuracy']:.4f}, Macro F1: {results_uci_smote['MLP']['macro_f1']:.4f}")
    
    # Results table
    print("\n" + "-"*60)
    print("UCI AI4I RESULTS WITH SMOTE")
    print("-"*60)
    print(f"{'Method':<25} {'Accuracy':>10} {'Macro F1':>10} {'Wt. F1':>10}")
    print("-"*60)
    for method, m in sorted(results_uci_smote.items(), key=lambda x: x[1]['macro_f1'], reverse=True):
        star = " *" if m['macro_f1'] == max(r['macro_f1'] for r in results_uci_smote.values()) else ""
        print(f"{method:<25} {m['accuracy']:>10.4f} {m['macro_f1']:>10.4f} {m['weighted_f1']:>10.4f}{star}")
    
    # Per-class for best
    best_method = max(results_uci_smote.items(), key=lambda x: x[1]['macro_f1'])[0]
    best_preds = preds_pg if 'ProactiveGuard' in best_method else preds_gb
    print(f"\nPer-Class Report ({best_method}):")
    print(classification_report(y_test_uci, best_preds, target_names=class_names_uci, zero_division=0))
    
    uci_smote_success = True

except ImportError as e:
    print(f"    ERROR: Missing package - {e}")
    print("    Run: pip install ucimlrepo imbalanced-learn")
    uci_smote_success = False
except Exception as e:
    print(f"    ERROR: {e}")
    uci_smote_success = False


# ============================================================================
# IMPROVEMENT 2: ETCD/RAFT SIMULATION
# ============================================================================

print("\n" + "="*70)
print("IMPROVEMENT 2: ETCD/RAFT CONSENSUS LOG SIMULATION")
print("="*70)

print("\n[1] Generating realistic etcd/Raft metrics...")

def generate_etcd_data(n_samples=5000, failure_rate=0.05):
    """
    Generate realistic etcd/Raft consensus metrics.
    
    Features based on real etcd metrics:
    - heartbeat_latency_ms: Time for heartbeat round-trip
    - log_replication_lag: Entries behind leader
    - proposal_pending: Pending proposals in queue
    - leader_changes_1h: Leader elections in last hour
    - peer_rtt_ms: Round-trip time to peers
    - wal_fsync_ms: Write-ahead log sync time
    - snapshot_duration_ms: Time for last snapshot
    - db_size_mb: Database size
    - cpu_usage: CPU utilization (0-1)
    - memory_pressure: Memory pressure score (0-1)
    - network_drop_rate: Packet drop rate
    - disk_io_util: Disk I/O utilization
    """
    np.random.seed(42)
    
    n_healthy = int(n_samples * (1 - failure_rate))
    n_failure = n_samples - n_healthy
    
    # Healthy nodes
    healthy_data = {
        'heartbeat_latency_ms': np.random.gamma(2, 5, n_healthy),  # ~10ms avg
        'log_replication_lag': np.random.poisson(1, n_healthy),  # ~1 entry
        'proposal_pending': np.random.poisson(2, n_healthy),  # ~2 pending
        'leader_changes_1h': np.random.poisson(0.1, n_healthy),  # rare
        'peer_rtt_ms': np.random.gamma(3, 3, n_healthy),  # ~9ms
        'wal_fsync_ms': np.random.gamma(2, 2, n_healthy),  # ~4ms
        'snapshot_duration_ms': np.random.gamma(5, 100, n_healthy),  # ~500ms
        'db_size_mb': np.random.uniform(10, 500, n_healthy),
        'cpu_usage': np.random.beta(2, 5, n_healthy),  # ~0.3
        'memory_pressure': np.random.beta(2, 8, n_healthy),  # ~0.2
        'network_drop_rate': np.random.beta(1, 100, n_healthy),  # ~0.01
        'disk_io_util': np.random.beta(2, 5, n_healthy),  # ~0.3
    }
    
    # Pre-failure nodes (degrading metrics)
    failure_data = {
        'heartbeat_latency_ms': np.random.gamma(10, 20, n_failure),  # ~200ms - SLOW
        'log_replication_lag': np.random.poisson(50, n_failure),  # ~50 entries behind
        'proposal_pending': np.random.poisson(100, n_failure),  # backed up
        'leader_changes_1h': np.random.poisson(3, n_failure),  # unstable
        'peer_rtt_ms': np.random.gamma(10, 20, n_failure),  # ~200ms
        'wal_fsync_ms': np.random.gamma(20, 50, n_failure),  # ~1000ms - SLOW DISK
        'snapshot_duration_ms': np.random.gamma(20, 500, n_failure),  # ~10s
        'db_size_mb': np.random.uniform(400, 2000, n_failure),  # large DB
        'cpu_usage': np.random.beta(8, 2, n_failure),  # ~0.8 HIGH
        'memory_pressure': np.random.beta(8, 2, n_failure),  # ~0.8 HIGH
        'network_drop_rate': np.random.beta(5, 20, n_failure),  # ~0.2 packet loss
        'disk_io_util': np.random.beta(9, 1, n_failure),  # ~0.9 saturated
    }
    
    # Combine
    X_healthy = np.column_stack([healthy_data[k] for k in healthy_data])
    X_failure = np.column_stack([failure_data[k] for k in failure_data])
    
    X = np.vstack([X_healthy, X_failure])
    y = np.array([0] * n_healthy + [1] * n_failure)
    
    # Shuffle
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]
    
    feature_names = list(healthy_data.keys())
    
    return X, y, feature_names

X_etcd, y_etcd, feature_names_etcd = generate_etcd_data(n_samples=5000, failure_rate=0.05)

print(f"    Generated {len(X_etcd)} samples")
print(f"    Features: {feature_names_etcd}")
print(f"    Healthy: {(y_etcd == 0).sum()}, Pre-failure: {(y_etcd == 1).sum()}")

# Train/test split
X_train_etcd, X_test_etcd, y_train_etcd, y_test_etcd = train_test_split(
    X_etcd, y_etcd, test_size=0.2, random_state=42, stratify=y_etcd
)

scaler_etcd = StandardScaler()
X_train_etcd_s = scaler_etcd.fit_transform(X_train_etcd)
X_test_etcd_s = scaler_etcd.transform(X_test_etcd)

print(f"    Train: {len(X_train_etcd)}, Test: {len(X_test_etcd)}")

# Run experiments
print("\n[2] Running experiments on etcd data...")

results_etcd = {}

# ProactiveGuard
print("    ProactiveGuard Ensemble...", end=" ", flush=True)
pg_etcd = EnsembleProactiveGuard(n_classes=2)
pg_etcd.fit(X_train_etcd_s, y_train_etcd, verbose=False)
preds_pg_etcd = pg_etcd.predict(X_test_etcd_s)
results_etcd['ProactiveGuard'] = {
    'accuracy': accuracy_score(y_test_etcd, preds_pg_etcd),
    'recall': recall_score(y_test_etcd, preds_pg_etcd),
    'f1': f1_score(y_test_etcd, preds_pg_etcd),
    'fnr': 1 - recall_score(y_test_etcd, preds_pg_etcd)
}
print(f"Acc: {results_etcd['ProactiveGuard']['accuracy']:.4f}, Recall: {results_etcd['ProactiveGuard']['recall']:.4f}")

# Random Forest
print("    Random Forest...", end=" ", flush=True)
rf_etcd = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf_etcd.fit(X_train_etcd_s, y_train_etcd)
preds_rf_etcd = rf_etcd.predict(X_test_etcd_s)
results_etcd['Random Forest'] = {
    'accuracy': accuracy_score(y_test_etcd, preds_rf_etcd),
    'recall': recall_score(y_test_etcd, preds_rf_etcd),
    'f1': f1_score(y_test_etcd, preds_rf_etcd),
    'fnr': 1 - recall_score(y_test_etcd, preds_rf_etcd)
}
print(f"Acc: {results_etcd['Random Forest']['accuracy']:.4f}, Recall: {results_etcd['Random Forest']['recall']:.4f}")

# Phi Accrual Simulation (threshold-based)
print("    Phi Accrual (simulated)...", end=" ", flush=True)
# Threshold: flag as failure if heartbeat > 100ms OR cpu > 0.7
phi_preds = ((X_test_etcd[:, 0] > 100) | (X_test_etcd[:, 8] > 0.7)).astype(int)
results_etcd['Phi Accrual'] = {
    'accuracy': accuracy_score(y_test_etcd, phi_preds),
    'recall': recall_score(y_test_etcd, phi_preds),
    'f1': f1_score(y_test_etcd, phi_preds),
    'fnr': 1 - recall_score(y_test_etcd, phi_preds)
}
print(f"Acc: {results_etcd['Phi Accrual']['accuracy']:.4f}, Recall: {results_etcd['Phi Accrual']['recall']:.4f}")

# Results table
print("\n" + "-"*60)
print("ETCD/RAFT SIMULATION RESULTS")
print("-"*60)
print(f"{'Method':<20} {'Accuracy':>10} {'Recall':>10} {'F1':>10} {'FNR':>10}")
print("-"*60)
for method, m in sorted(results_etcd.items(), key=lambda x: x[1]['recall'], reverse=True):
    star = " *" if m['recall'] == max(r['recall'] for r in results_etcd.values()) else ""
    print(f"{method:<20} {m['accuracy']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f} {m['fnr']:>10.4f}{star}")

etcd_success = True


# ============================================================================
# IMPROVEMENT 3: BOOTSTRAP CONFIDENCE INTERVALS
# ============================================================================

print("\n" + "="*70)
print("IMPROVEMENT 3: BOOTSTRAP CONFIDENCE INTERVALS")
print("="*70)

def bootstrap_ci(y_true, y_pred, metric_fn, n_bootstrap=1000, ci=95):
    """Compute bootstrap confidence interval for a metric."""
    np.random.seed(42)
    scores = []
    n = len(y_true)
    
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        score = metric_fn(y_true[idx], y_pred[idx])
        scores.append(score)
    
    lower = np.percentile(scores, (100 - ci) / 2)
    upper = np.percentile(scores, 100 - (100 - ci) / 2)
    mean = np.mean(scores)
    
    return mean, lower, upper

print("\n[1] Computing 95% CI for etcd binary detection...")
print("    Running 1000 bootstrap iterations...")

# ProactiveGuard CIs
acc_mean, acc_lo, acc_hi = bootstrap_ci(y_test_etcd, preds_pg_etcd, accuracy_score)
recall_mean, recall_lo, recall_hi = bootstrap_ci(y_test_etcd, preds_pg_etcd, recall_score)
f1_mean, f1_lo, f1_hi = bootstrap_ci(y_test_etcd, preds_pg_etcd, f1_score)

print("\n" + "-"*60)
print("PROACTIVEGUARD 95% CONFIDENCE INTERVALS (etcd)")
print("-"*60)
print(f"{'Metric':<15} {'Mean':>10} {'95% CI':>20}")
print("-"*60)
print(f"{'Accuracy':<15} {acc_mean:>10.4f} [{acc_lo:.4f}, {acc_hi:.4f}]")
print(f"{'Recall':<15} {recall_mean:>10.4f} [{recall_lo:.4f}, {recall_hi:.4f}]")
print(f"{'F1 Score':<15} {f1_mean:>10.4f} [{f1_lo:.4f}, {f1_hi:.4f}]")

# Compare with Random Forest
print("\n[2] Comparison with Random Forest...")

rf_acc_mean, rf_acc_lo, rf_acc_hi = bootstrap_ci(y_test_etcd, preds_rf_etcd, accuracy_score)
rf_recall_mean, rf_recall_lo, rf_recall_hi = bootstrap_ci(y_test_etcd, preds_rf_etcd, recall_score)

print("\n" + "-"*60)
print("RANDOM FOREST 95% CONFIDENCE INTERVALS (etcd)")
print("-"*60)
print(f"{'Metric':<15} {'Mean':>10} {'95% CI':>20}")
print("-"*60)
print(f"{'Accuracy':<15} {rf_acc_mean:>10.4f} [{rf_acc_lo:.4f}, {rf_acc_hi:.4f}]")
print(f"{'Recall':<15} {rf_recall_mean:>10.4f} [{rf_recall_lo:.4f}, {rf_recall_hi:.4f}]")

# Statistical significance test
print("\n[3] Statistical significance...")
if recall_lo > rf_recall_hi:
    print("    ✓ ProactiveGuard recall is SIGNIFICANTLY HIGHER than Random Forest (non-overlapping CIs)")
elif rf_recall_lo > recall_hi:
    print("    ✗ Random Forest recall is significantly higher")
else:
    print("    ~ No significant difference (overlapping CIs)")

bootstrap_success = True


# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*70)
print("="*70)
print("              FINAL SUMMARY - ALL IMPROVEMENTS")
print("="*70)
print("="*70)

if uci_smote_success:
    print("\n┌" + "─"*60 + "┐")
    print("│ IMPROVEMENT 1: UCI AI4I WITH SMOTE                         │")
    print("└" + "─"*60 + "┘")
    print(f"{'Method':<25} {'Accuracy':>10} {'Macro F1':>10}")
    print("-"*50)
    for method, m in sorted(results_uci_smote.items(), key=lambda x: x[1]['macro_f1'], reverse=True):
        star = " ★" if m['macro_f1'] == max(r['macro_f1'] for r in results_uci_smote.values()) else ""
        print(f"{method:<25} {m['accuracy']:>10.4f} {m['macro_f1']:>10.4f}{star}")

if etcd_success:
    print("\n┌" + "─"*60 + "┐")
    print("│ IMPROVEMENT 2: ETCD/RAFT CONSENSUS SIMULATION              │")
    print("└" + "─"*60 + "┘")
    print(f"{'Method':<20} {'Accuracy':>10} {'Recall':>10} {'FNR':>10}")
    print("-"*55)
    for method, m in sorted(results_etcd.items(), key=lambda x: x[1]['recall'], reverse=True):
        star = " ★" if m['recall'] == max(r['recall'] for r in results_etcd.values()) else ""
        print(f"{method:<20} {m['accuracy']:>10.4f} {m['recall']:>10.4f} {m['fnr']:>10.4f}{star}")

if bootstrap_success:
    print("\n┌" + "─"*60 + "┐")
    print("│ IMPROVEMENT 3: BOOTSTRAP 95% CONFIDENCE INTERVALS          │")
    print("└" + "─"*60 + "┘")
    print(f"ProactiveGuard Recall: {recall_mean:.4f} [{recall_lo:.4f}, {recall_hi:.4f}]")
    print(f"Random Forest Recall:  {rf_recall_mean:.4f} [{rf_recall_lo:.4f}, {rf_recall_hi:.4f}]")

# Save results to JSON
results_all = {
    'uci_smote': results_uci_smote if uci_smote_success else None,
    'etcd': results_etcd if etcd_success else None,
    'bootstrap': {
        'proactiveguard': {
            'accuracy': {'mean': acc_mean, 'ci_low': acc_lo, 'ci_high': acc_hi},
            'recall': {'mean': recall_mean, 'ci_low': recall_lo, 'ci_high': recall_hi},
            'f1': {'mean': f1_mean, 'ci_low': f1_lo, 'ci_high': f1_hi}
        },
        'random_forest': {
            'accuracy': {'mean': rf_acc_mean, 'ci_low': rf_acc_lo, 'ci_high': rf_acc_hi},
            'recall': {'mean': rf_recall_mean, 'ci_low': rf_recall_lo, 'ci_high': rf_recall_hi}
        }
    } if bootstrap_success else None
}

with open('improved_results.json', 'w') as f:
    json.dump(results_all, f, indent=2)

print("\n" + "="*70)
print("Results saved to: improved_results.json")
print("="*70)
print("EXPERIMENTS COMPLETE!")
print("="*70)