#!/usr/bin/env python3
"""
Quick Test Script for Baseline Experiments

This generates synthetic data and runs all baselines to verify the code works.
After confirming it works, modify load_your_data() to use your real data.

Usage:
    python test_baselines_quick.py
"""

import numpy as np
import json
from pathlib import Path

# Check available libraries
try:
    import torch
    TORCH = True
except ImportError:
    TORCH = False
    print("PyTorch not installed - LSTM baseline will be skipped")

try:
    from sklearn.ensemble import RandomForestClassifier
    SKLEARN = True
except ImportError:
    SKLEARN = False
    print("scikit-learn not installed - RF baseline will be skipped")


# =============================================================================
# CUSTOM METRICS (no sklearn needed)
# =============================================================================

def accuracy_score(y_true, y_pred):
    """Calculate accuracy without sklearn."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(y_true == y_pred)


def f1_score_manual(y_true, y_pred, average='weighted'):
    """Calculate F1 score without sklearn."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    classes = np.unique(np.concatenate([y_true, y_pred]))
    
    f1_scores = []
    weights = []
    
    for c in classes:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)
        weights.append(np.sum(y_true == c))
    
    if average == 'weighted':
        total_weight = sum(weights)
        if total_weight > 0:
            return sum(f * w for f, w in zip(f1_scores, weights)) / total_weight
        return 0.0
    elif average == 'macro':
        return np.mean(f1_scores)
    else:
        return f1_scores


def confusion_matrix_manual(y_true, y_pred):
    """Calculate confusion matrix without sklearn."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(classes)
    
    cm = np.zeros((n_classes, n_classes), dtype=int)
    
    for i, c_true in enumerate(classes):
        for j, c_pred in enumerate(classes):
            cm[i, j] = np.sum((y_true == c_true) & (y_pred == c_pred))
    
    return cm


# =============================================================================
# DATA GENERATION
# =============================================================================

def generate_test_data(n_train=5000, n_test=1000, seq_len=20, n_features=32, n_classes=9):
    """Generate synthetic data that mimics failure detection patterns."""
    np.random.seed(42)
    
    # Class probabilities (more healthy samples)
    p = [0.50, 0.05, 0.15, 0.05, 0.10, 0.05, 0.04, 0.03, 0.03]
    
    def make_data(n):
        X = np.random.randn(n, seq_len, n_features).astype(np.float32)
        y = np.random.choice(n_classes, n, p=p)
        
        # Add signal: different patterns for different classes
        for i in range(n):
            if y[i] == 0:  # Healthy
                X[i, :, 0] = np.random.normal(50, 5, seq_len)  # Low stable latency
                X[i, :, 1] = np.random.normal(10, 2, seq_len)  # Low jitter
            elif y[i] == 2:  # Crash
                X[i, :, 0] = np.random.normal(800, 100, seq_len)  # Very high latency
                X[i, :, 2] = np.random.randint(5, 15, seq_len)  # Many missed HBs
            elif y[i] == 4:  # Slow
                X[i, :, 0] = np.random.normal(300, 50, seq_len)  # Medium-high latency
                X[i, :, 1] = np.random.normal(80, 15, seq_len)  # High jitter
            elif y[i] == 3:  # Byzantine
                # Random inconsistent patterns
                X[i, :, 0] = np.random.uniform(20, 500, seq_len)
                X[i, :, 3] = np.random.uniform(-1, 1, seq_len)  # Inconsistent
        return X, y
    
    X_train, y_train = make_data(n_train)
    X_test, y_test = make_data(n_test)
    
    return X_train, X_test, y_train, y_test


# =============================================================================
# BASELINE IMPLEMENTATIONS
# =============================================================================

class ThresholdDetector:
    """Simple threshold-based detection."""
    def predict(self, X):
        if len(X.shape) == 3:
            X = X[:, -1, :]  # Last timestep
        preds = np.zeros(len(X), dtype=int)
        for i, x in enumerate(X):
            if x[0] > 500:  # High latency
                preds[i] = 2  # Crash
            elif x[1] > 50:  # High jitter
                preds[i] = 4  # Slow
            elif len(x) > 2 and x[2] > 3:  # Missed heartbeats
                preds[i] = 2  # Crash
        return preds


class PhiAccrualDetector:
    """Phi Accrual failure detector."""
    def __init__(self, threshold=8.0):
        self.threshold = threshold
        self.mean = 50.0
        self.std = 10.0
        
    def fit(self, X, y):
        healthy = X[y == 0]
        if len(healthy.shape) == 3:
            healthy = healthy[:, -1, :]
        self.mean = np.mean(healthy[:, 0])
        self.std = max(np.std(healthy[:, 0]), 1.0)
        return self
        
    def predict(self, X):
        if len(X.shape) == 3:
            X = X[:, -1, :]
        preds = np.zeros(len(X), dtype=int)
        for i, x in enumerate(X):
            z = (x[0] - self.mean) / self.std
            phi = z * z / 2 if z > 0 else 0
            if phi > self.threshold:
                preds[i] = 2  # Failure
        return preds


class RandomForestBaseline:
    """Random Forest baseline."""
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        
    def _extract_features(self, X):
        if len(X.shape) == 2:
            return X
        n, seq, feat = X.shape
        features = []
        for i in range(n):
            f = []
            for j in range(feat):
                s = X[i, :, j]
                f.extend([s[-1], np.mean(s), np.std(s), np.min(s), np.max(s)])
            features.append(f)
        return np.array(features)
        
    def fit(self, X, y):
        self.model.fit(self._extract_features(X), y)
        return self
        
    def predict(self, X):
        return self.model.predict(self._extract_features(X))


class VanillaLSTM:
    """Simple LSTM without attention."""
    def __init__(self, input_size, num_classes, hidden=64, layers=2, epochs=30):
        self.epochs = epochs
        self.device = torch.device('cpu')
        
        class LSTMModel(torch.nn.Module):
            def __init__(self_inner):
                super().__init__()
                self_inner.lstm = torch.nn.LSTM(input_size, hidden, layers, batch_first=True)
                self_inner.fc = torch.nn.Linear(hidden, num_classes)
            def forward(self_inner, x):
                out, (h, c) = self_inner.lstm(x)
                return self_inner.fc(h[-1])
                
        self.model = LSTMModel()
        self.opt = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        
    def fit(self, X, y):
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
        
        self.model.train()
        for ep in range(self.epochs):
            total_loss = 0
            for bx, by in loader:
                self.opt.zero_grad()
                out = self.model(bx)
                loss = self.loss_fn(out, by)
                loss.backward()
                self.opt.step()
                total_loss += loss.item()
            if (ep + 1) % 10 == 0:
                print(f"    LSTM Epoch {ep+1}/{self.epochs}: loss={total_loss/len(loader):.4f}")
        return self
        
    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            out = self.model(torch.tensor(X, dtype=torch.float32))
            return out.argmax(dim=1).numpy()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*60)
    print("BASELINE EXPERIMENTS - Quick Test")
    print("="*60)
    
    # Generate data
    print("\nGenerating synthetic test data...")
    X_train, X_test, y_train, y_test = generate_test_data()
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Print class distribution more cleanly
    unique, counts = np.unique(y_train, return_counts=True)
    print(f"Classes: {dict(zip(unique.tolist(), counts.tolist()))}")
    
    results = {}
    
    # 1. Threshold
    print("\n[1/4] Threshold-based Detector...")
    t = ThresholdDetector()
    y_pred = t.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score_manual(y_test, y_pred, average='weighted')
    results['threshold'] = {'accuracy': float(acc), 'f1': float(f1)}
    print(f"  Accuracy: {acc:.4f}, F1: {f1:.4f}")
    
    # 2. Phi Accrual
    print("\n[2/4] Phi Accrual Detector...")
    p = PhiAccrualDetector()
    p.fit(X_train, y_train)
    y_pred = p.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score_manual(y_test, y_pred, average='weighted')
    results['phi_accrual'] = {'accuracy': float(acc), 'f1': float(f1)}
    print(f"  Accuracy: {acc:.4f}, F1: {f1:.4f}")
    
    # 3. Random Forest
    if SKLEARN:
        print("\n[3/4] Random Forest...")
        rf = RandomForestBaseline()
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score_manual(y_test, y_pred, average='weighted')
        results['random_forest'] = {'accuracy': float(acc), 'f1': float(f1)}
        print(f"  Accuracy: {acc:.4f}, F1: {f1:.4f}")
    else:
        print("\n[3/4] Random Forest - SKIPPED (install: pip3 install scikit-learn)")
    
    # 4. Vanilla LSTM
    if TORCH:
        print("\n[4/4] Vanilla LSTM...")
        lstm = VanillaLSTM(X_train.shape[-1], len(np.unique(y_train)), epochs=20)
        lstm.fit(X_train, y_train)
        y_pred = lstm.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score_manual(y_test, y_pred, average='weighted')
        results['vanilla_lstm'] = {'accuracy': float(acc), 'f1': float(f1)}
        print(f"  Accuracy: {acc:.4f}, F1: {f1:.4f}")
    else:
        print("\n[4/4] Vanilla LSTM - SKIPPED (install: pip3 install torch)")
    
    # Summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"\n{'Method':<20} {'Accuracy':<12} {'F1 Score':<12}")
    print("-"*44)
    for name, r in sorted(results.items(), key=lambda x: x[1]['f1'], reverse=True):
        print(f"{name:<20} {r['accuracy']:.4f}       {r['f1']:.4f}")
    
    # Save
    Path("results").mkdir(exist_ok=True)
    with open("results/baseline_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to results/baseline_test_results.json")
    
    print("\n" + "="*60)
    print("TEST COMPLETE!")
    print("="*60)
    print("\nThese are results on SYNTHETIC data.")
    print("\nTo get ALL baselines, install missing packages:")
    print("  pip3 install scikit-learn torch")
    print("\nThen run again.")


if __name__ == "__main__":
    main()