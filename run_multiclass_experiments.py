#!/usr/bin/env python3
"""
REAL DATA Multi-Class Experiments
Uses:
1. UCI AI4I 2020 via ucimlrepo package
2. NASA C-MAPSS from Kaggle/direct download

First run: pip install ucimlrepo
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import warnings
import os

warnings.filterwarnings('ignore')
np.random.seed(42)

print("="*70)
print("REAL DATA MULTI-CLASS EXPERIMENTS")
print("="*70)

# ============================================================================
# DATASET 1: UCI AI4I 2020 via ucimlrepo
# ============================================================================

print("\n" + "="*70)
print("DATASET 1: UCI AI4I 2020 (via ucimlrepo)")
print("="*70)

print("\n[1] Loading UCI AI4I dataset...")

try:
    from ucimlrepo import fetch_ucirepo
    
    # Fetch dataset
    ai4i = fetch_ucirepo(id=601)
    
    # Get features and targets
    X_uci_df = ai4i.data.features
    y_uci_df = ai4i.data.targets
    
    print(f"    Features shape: {X_uci_df.shape}")
    print(f"    Targets shape: {y_uci_df.shape}")
    print(f"    Feature columns: {list(X_uci_df.columns)}")
    print(f"    Target columns: {list(y_uci_df.columns)}")
    
    # Convert features to numpy
    X_uci = X_uci_df.select_dtypes(include=[np.number]).values
    
    # Create multi-class labels from failure columns
    # Targets: Machine failure, TWF, HDF, PWF, OSF, RNF
    failure_cols = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    
    def get_failure_class(row):
        """0=No Failure, 1=TWF, 2=HDF, 3=PWF, 4=OSF, 5=RNF"""
        for i, col in enumerate(failure_cols):
            if col in row.index and row[col] == 1:
                return i + 1
        return 0
    
    y_uci = y_uci_df.apply(get_failure_class, axis=1).values
    class_names_uci = ['No Failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    
    print(f"\n    Total samples: {len(X_uci)}")
    print(f"    Features: {X_uci.shape[1]}")
    print(f"    Classes: {class_names_uci}")
    
    # Class distribution
    print("\n    Class distribution:")
    unique, counts = np.unique(y_uci, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"      {class_names_uci[u]}: {c} ({100*c/len(y_uci):.2f}%)")
    
    # Train/test split
    X_train_uci, X_test_uci, y_train_uci, y_test_uci = train_test_split(
        X_uci, y_uci, test_size=0.2, random_state=42, stratify=y_uci
    )
    
    scaler_uci = StandardScaler()
    X_train_uci_s = scaler_uci.fit_transform(X_train_uci)
    X_test_uci_s = scaler_uci.transform(X_test_uci)
    
    print(f"\n    Train: {len(X_train_uci)}, Test: {len(X_test_uci)}")
    uci_loaded = True
    
except ImportError:
    print("    ERROR: ucimlrepo not installed. Run: pip install ucimlrepo")
    uci_loaded = False
except Exception as e:
    print(f"    ERROR: {e}")
    uci_loaded = False


# ============================================================================
# DATASET 2: NASA C-MAPSS
# ============================================================================

print("\n" + "="*70)
print("DATASET 2: NASA C-MAPSS Turbofan Engine")
print("="*70)

print("\n[1] Loading NASA C-MAPSS dataset...")

# The NASA C-MAPSS data is in text format, NOT .mat
# Download from: https://www.kaggle.com/datasets/behrad3d/nasa-cmaps
# Or: https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip
#
# After extracting, you should have files like:
# - train_FD001.txt
# - test_FD001.txt
# - RUL_FD001.txt

# Try to find the data file
possible_paths = [
    'train_FD001.txt',
    'CMAPSSData/train_FD001.txt', 
    'real_data/train_FD001.txt',
    'real_data/cmapss/train_FD001.txt',
    'real_data/cmapss/CMAPSSData/train_FD001.txt',
    'nasa-cmaps/train_FD001.txt',
    'datasets/train_FD001.txt',
]

cmapss_loaded = False
for path in possible_paths:
    if os.path.exists(path):
        print(f"    Found: {path}")
        
        # C-MAPSS format: space-separated, no header
        # Columns: engine_id, cycle, 3 operational settings, 21 sensors
        columns = ['engine_id', 'cycle'] + \
                  [f'setting_{i}' for i in range(1, 4)] + \
                  [f'sensor_{i}' for i in range(1, 22)]
        
        df_cmapss = pd.read_csv(path, sep=r'\s+', header=None, names=columns)
        cmapss_loaded = True
        break

if not cmapss_loaded:
    print("\n    C-MAPSS data not found!")
    print("    Please download from one of these sources:")
    print("    1. Kaggle: https://www.kaggle.com/datasets/behrad3d/nasa-cmaps")
    print("    2. NASA: https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip")
    print("\n    After downloading, extract and place train_FD001.txt in current directory")
    print("    Or in a subfolder like 'CMAPSSData/' or 'real_data/'")

if cmapss_loaded:
    print(f"    Shape: {df_cmapss.shape}")
    print(f"    Engines: {df_cmapss['engine_id'].nunique()}")
    print(f"    Columns: {len(df_cmapss.columns)}")
    
    # Compute RUL (Remaining Useful Life) for each row
    # RUL = max_cycle_for_engine - current_cycle
    max_cycles = df_cmapss.groupby('engine_id')['cycle'].max()
    df_cmapss['RUL'] = df_cmapss.apply(
        lambda row: max_cycles[row['engine_id']] - row['cycle'], axis=1
    )
    
    # Create 5-class degradation labels based on RUL
    def rul_to_class(rul):
        if rul > 125:
            return 0  # Healthy
        elif rul > 75:
            return 1  # Early degradation
        elif rul > 50:
            return 2  # Mid degradation
        elif rul > 25:
            return 3  # Late degradation
        else:
            return 4  # Critical
    
    df_cmapss['class'] = df_cmapss['RUL'].apply(rul_to_class)
    class_names_cmapss = ['Healthy', 'Early', 'Mid', 'Late', 'Critical']
    
    print(f"\n    Total samples: {len(df_cmapss)}")
    print(f"    Total engines: {df_cmapss['engine_id'].nunique()}")
    
    # Class distribution
    print("\n    Class distribution:")
    for i, name in enumerate(class_names_cmapss):
        count = (df_cmapss['class'] == i).sum()
        print(f"      {name}: {count} ({100*count/len(df_cmapss):.2f}%)")
    
    # Features: 3 settings + 21 sensors = 24 features
    feature_cols = [f'setting_{i}' for i in range(1, 4)] + \
                   [f'sensor_{i}' for i in range(1, 22)]
    
    X_cmapss = df_cmapss[feature_cols].values
    y_cmapss = df_cmapss['class'].values
    
    # Split by ENGINE to avoid data leakage
    # (samples from same engine shouldn't be in both train and test)
    engine_ids = df_cmapss['engine_id'].unique()
    train_engines, test_engines = train_test_split(
        engine_ids, test_size=0.2, random_state=42
    )
    
    train_mask = df_cmapss['engine_id'].isin(train_engines)
    test_mask = df_cmapss['engine_id'].isin(test_engines)
    
    X_train_cm = X_cmapss[train_mask]
    y_train_cm = y_cmapss[train_mask]
    X_test_cm = X_cmapss[test_mask]
    y_test_cm = y_cmapss[test_mask]
    
    scaler_cm = StandardScaler()
    X_train_cm_s = scaler_cm.fit_transform(X_train_cm)
    X_test_cm_s = scaler_cm.transform(X_test_cm)
    
    print(f"\n    Train: {len(X_train_cm)} samples ({len(train_engines)} engines)")
    print(f"    Test: {len(X_test_cm)} samples ({len(test_engines)} engines)")


# ============================================================================
# IMPROVED PROACTIVEGUARD
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
    
    def fit(self, X, y, epochs=150, lr=0.01, batch_size=64, verbose=True):
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
                if verbose:
                    print(f"    Early stopping at epoch {epoch+1}")
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
            print(f"    Training ensemble ({self.n_models} models + RF)...")
        
        configs = [
            [256, 128, 64],
            [128, 64, 32],
            [512, 256, 128],
            [256, 256, 128, 64],
            [128, 128, 64],
        ]
        
        for i in range(self.n_models):
            if verbose:
                print(f"      Model {i+1}/{self.n_models}...", end=" ", flush=True)
            
            model = ImprovedProactiveGuard(self.n_classes, configs[i % len(configs)])
            idx = np.random.choice(len(X), len(X), replace=True)
            model.fit(X[idx], y[idx], epochs=100, verbose=False)
            self.models.append(model)
            
            if verbose:
                print("done")
        
        if verbose:
            print("      Training RF...", end=" ", flush=True)
        self.rf = RandomForestClassifier(
            n_estimators=100, class_weight='balanced', 
            random_state=42, n_jobs=-1
        )
        self.rf.fit(X, y)
        if verbose:
            print("done")
        
        return self
    
    def predict(self, X):
        probs = np.zeros((len(X), self.n_classes))
        for model in self.models:
            probs += model.predict_proba(X)
        probs += self.rf.predict_proba(X) * 2
        probs /= (len(self.models) + 2)
        return np.argmax(probs, axis=1)


# ============================================================================
# RUN EXPERIMENTS
# ============================================================================

results = {}

# ----------------------------------------------------------------------------
# UCI AI4I Experiments
# ----------------------------------------------------------------------------
if uci_loaded:
    print("\n" + "="*70)
    print("RUNNING UCI AI4I EXPERIMENTS (REAL DATA)")
    print("="*70)
    
    results_uci = {}
    n_classes_uci = len(class_names_uci)
    
    print("\n[1] ProactiveGuard Ensemble...")
    pg_ens = EnsembleProactiveGuard(n_classes_uci)
    pg_ens.fit(X_train_uci_s, y_train_uci)
    preds = pg_ens.predict(X_test_uci_s)
    results_uci['ProactiveGuard (Ensemble)'] = {
        'accuracy': accuracy_score(y_test_uci, preds),
        'macro_f1': f1_score(y_test_uci, preds, average='macro', zero_division=0),
        'weighted_f1': f1_score(y_test_uci, preds, average='weighted', zero_division=0)
    }
    print(f"    Acc: {results_uci['ProactiveGuard (Ensemble)']['accuracy']:.4f}, "
          f"Macro F1: {results_uci['ProactiveGuard (Ensemble)']['macro_f1']:.4f}")
    preds_ens_uci = preds
    
    print("\n[2] Random Forest...")
    rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1)
    rf.fit(X_train_uci_s, y_train_uci)
    preds = rf.predict(X_test_uci_s)
    results_uci['Random Forest'] = {
        'accuracy': accuracy_score(y_test_uci, preds),
        'macro_f1': f1_score(y_test_uci, preds, average='macro', zero_division=0),
        'weighted_f1': f1_score(y_test_uci, preds, average='weighted', zero_division=0)
    }
    print(f"    Acc: {results_uci['Random Forest']['accuracy']:.4f}, "
          f"Macro F1: {results_uci['Random Forest']['macro_f1']:.4f}")
    
    print("\n[3] Gradient Boosting...")
    gb = GradientBoostingClassifier(n_estimators=150, max_depth=6, random_state=42)
    gb.fit(X_train_uci_s, y_train_uci)
    preds = gb.predict(X_test_uci_s)
    results_uci['Gradient Boosting'] = {
        'accuracy': accuracy_score(y_test_uci, preds),
        'macro_f1': f1_score(y_test_uci, preds, average='macro', zero_division=0),
        'weighted_f1': f1_score(y_test_uci, preds, average='weighted', zero_division=0)
    }
    print(f"    Acc: {results_uci['Gradient Boosting']['accuracy']:.4f}, "
          f"Macro F1: {results_uci['Gradient Boosting']['macro_f1']:.4f}")
    
    print("\n[4] MLP (sklearn)...")
    mlp = MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=500, early_stopping=True, random_state=42)
    mlp.fit(X_train_uci_s, y_train_uci)
    preds = mlp.predict(X_test_uci_s)
    results_uci['MLP (sklearn)'] = {
        'accuracy': accuracy_score(y_test_uci, preds),
        'macro_f1': f1_score(y_test_uci, preds, average='macro', zero_division=0),
        'weighted_f1': f1_score(y_test_uci, preds, average='weighted', zero_division=0)
    }
    print(f"    Acc: {results_uci['MLP (sklearn)']['accuracy']:.4f}, "
          f"Macro F1: {results_uci['MLP (sklearn)']['macro_f1']:.4f}")
    
    print("\n" + "-"*60)
    print("UCI AI4I RESULTS (REAL DATA)")
    print("-"*60)
    print(f"{'Method':<30} {'Accuracy':>10} {'Macro F1':>10}")
    print("-"*55)
    for method, m in sorted(results_uci.items(), key=lambda x: x[1]['macro_f1'], reverse=True):
        star = " *" if m['macro_f1'] == max(r['macro_f1'] for r in results_uci.values()) else ""
        print(f"{method:<30} {m['accuracy']:>10.4f} {m['macro_f1']:>10.4f}{star}")
    
    print(f"\nProactiveGuard Per-Class Report:")
    print(classification_report(y_test_uci, preds_ens_uci, target_names=class_names_uci, zero_division=0))
    
    results['uci'] = results_uci


# ----------------------------------------------------------------------------
# NASA C-MAPSS Experiments
# ----------------------------------------------------------------------------
if cmapss_loaded:
    print("\n" + "="*70)
    print("RUNNING NASA C-MAPSS EXPERIMENTS (REAL DATA)")
    print("="*70)
    
    results_cm = {}
    n_classes_cm = 5
    
    print("\n[1] ProactiveGuard Ensemble...")
    pg_ens_cm = EnsembleProactiveGuard(n_classes_cm)
    pg_ens_cm.fit(X_train_cm_s, y_train_cm)
    preds = pg_ens_cm.predict(X_test_cm_s)
    results_cm['ProactiveGuard (Ensemble)'] = {
        'accuracy': accuracy_score(y_test_cm, preds),
        'macro_f1': f1_score(y_test_cm, preds, average='macro', zero_division=0),
        'weighted_f1': f1_score(y_test_cm, preds, average='weighted', zero_division=0)
    }
    print(f"    Acc: {results_cm['ProactiveGuard (Ensemble)']['accuracy']:.4f}, "
          f"Macro F1: {results_cm['ProactiveGuard (Ensemble)']['macro_f1']:.4f}")
    preds_ens_cm = preds
    
    print("\n[2] Random Forest...")
    rf_cm = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1)
    rf_cm.fit(X_train_cm_s, y_train_cm)
    preds = rf_cm.predict(X_test_cm_s)
    results_cm['Random Forest'] = {
        'accuracy': accuracy_score(y_test_cm, preds),
        'macro_f1': f1_score(y_test_cm, preds, average='macro', zero_division=0),
        'weighted_f1': f1_score(y_test_cm, preds, average='weighted', zero_division=0)
    }
    print(f"    Acc: {results_cm['Random Forest']['accuracy']:.4f}, "
          f"Macro F1: {results_cm['Random Forest']['macro_f1']:.4f}")
    
    print("\n[3] Gradient Boosting...")
    gb_cm = GradientBoostingClassifier(n_estimators=150, max_depth=8, random_state=42)
    gb_cm.fit(X_train_cm_s, y_train_cm)
    preds = gb_cm.predict(X_test_cm_s)
    results_cm['Gradient Boosting'] = {
        'accuracy': accuracy_score(y_test_cm, preds),
        'macro_f1': f1_score(y_test_cm, preds, average='macro', zero_division=0),
        'weighted_f1': f1_score(y_test_cm, preds, average='weighted', zero_division=0)
    }
    print(f"    Acc: {results_cm['Gradient Boosting']['accuracy']:.4f}, "
          f"Macro F1: {results_cm['Gradient Boosting']['macro_f1']:.4f}")
    
    print("\n[4] MLP (sklearn)...")
    mlp_cm = MLPClassifier(hidden_layer_sizes=(256, 128, 64), max_iter=300, early_stopping=True, random_state=42)
    mlp_cm.fit(X_train_cm_s, y_train_cm)
    preds = mlp_cm.predict(X_test_cm_s)
    results_cm['MLP (sklearn)'] = {
        'accuracy': accuracy_score(y_test_cm, preds),
        'macro_f1': f1_score(y_test_cm, preds, average='macro', zero_division=0),
        'weighted_f1': f1_score(y_test_cm, preds, average='weighted', zero_division=0)
    }
    print(f"    Acc: {results_cm['MLP (sklearn)']['accuracy']:.4f}, "
          f"Macro F1: {results_cm['MLP (sklearn)']['macro_f1']:.4f}")
    
    print("\n" + "-"*60)
    print("NASA C-MAPSS RESULTS (REAL DATA)")
    print("-"*60)
    print(f"{'Method':<30} {'Accuracy':>10} {'Macro F1':>10}")
    print("-"*55)
    for method, m in sorted(results_cm.items(), key=lambda x: x[1]['macro_f1'], reverse=True):
        star = " *" if m['macro_f1'] == max(r['macro_f1'] for r in results_cm.values()) else ""
        print(f"{method:<30} {m['accuracy']:>10.4f} {m['macro_f1']:>10.4f}{star}")
    
    print(f"\nProactiveGuard Per-Class Report:")
    print(classification_report(y_test_cm, preds_ens_cm, target_names=class_names_cmapss, zero_division=0))
    
    results['cmapss'] = results_cm


# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*70)
print("="*70)
print("         FINAL SUMMARY - REAL DATA EXPERIMENTS")
print("="*70)
print("="*70)

if 'uci' in results:
    print("\n┌" + "─"*60 + "┐")
    print("│ UCI AI4I 2020 - REAL Predictive Maintenance Data           │")
    print("│ Source: UCI Machine Learning Repository                    │")
    print("└" + "─"*60 + "┘")
    print(f"{'Method':<30} {'Accuracy':>10} {'Macro F1':>10}")
    print("-"*55)
    for method, m in sorted(results['uci'].items(), key=lambda x: x[1]['macro_f1'], reverse=True):
        star = " ★" if m['macro_f1'] == max(r['macro_f1'] for r in results['uci'].values()) else ""
        print(f"{method:<30} {m['accuracy']:>10.4f} {m['macro_f1']:>10.4f}{star}")

if 'cmapss' in results:
    print("\n┌" + "─"*60 + "┐")
    print("│ NASA C-MAPSS - REAL Turbofan Engine Degradation Data       │")
    print("│ Source: NASA Prognostics Data Repository                   │")
    print("└" + "─"*60 + "┘")
    print(f"{'Method':<30} {'Accuracy':>10} {'Macro F1':>10}")
    print("-"*55)
    for method, m in sorted(results['cmapss'].items(), key=lambda x: x[1]['macro_f1'], reverse=True):
        star = " ★" if m['macro_f1'] == max(r['macro_f1'] for r in results['cmapss'].values()) else ""
        print(f"{method:<30} {m['accuracy']:>10.4f} {m['macro_f1']:>10.4f}{star}")

print("\n" + "="*70)
print("EXPERIMENTS COMPLETE!")
print("="*70)