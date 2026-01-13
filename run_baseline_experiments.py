#!/usr/bin/env python3
"""
FINAL Baseline Experiments - Testing on Edge Cases

The previous tests showed both ProactiveGuard and Vanilla LSTM at 100%.
This version tests on CHALLENGING scenarios where architecture matters:
1. Noisy data
2. Subtle degradation patterns  
3. Mixed failure modes
4. Adversarial-like patterns

Usage:
    python3 run_final_baselines.py
"""

import numpy as np
import json
import time
from pathlib import Path

from train_predictive import (
    PredictiveModel, extract_features, SyntheticObservation,
    generate_predictive_scenario, generate_predictive_training_data,
    NUM_FEATURES, WINDOW_SIZE, NUM_PREDICTION_CLASSES, PREDICTION_HORIZONS,
    HORIZON_NAMES, create_degraded_observation
)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    SKLEARN = True
except ImportError:
    SKLEARN = False


def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive metrics."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    accuracy = np.mean(y_true == y_pred)
    
    # Binary: healthy (0) vs any failure (>0)
    binary_true = (y_true > 0).astype(int)
    binary_pred = (y_pred > 0).astype(int)
    binary_acc = np.mean(binary_true == binary_pred)
    
    tp = np.sum((binary_pred == 1) & (binary_true == 1))
    tn = np.sum((binary_pred == 0) & (binary_true == 0))
    fp = np.sum((binary_pred == 1) & (binary_true == 0))
    fn = np.sum((binary_pred == 0) & (binary_true == 1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    return {
        'accuracy': float(accuracy),
        'binary_accuracy': float(binary_acc),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'fpr': float(fpr) * 100,  # As percentage
        'fnr': float(fnr) * 100,  # As percentage
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)
    }


def generate_challenging_test_data(num_samples=400, seed=99999):
    """
    Generate CHALLENGING test data that differentiates architectures:
    1. Noisy healthy data (borderline cases)
    2. Subtle early degradation (hard to detect)
    3. Intermittent failures (on-off patterns)
    4. Mixed/ambiguous patterns
    """
    print(f"\n📊 Generating CHALLENGING test data...")
    np.random.seed(seed)
    
    all_features = []
    all_labels = []
    
    samples_per_category = num_samples // 5
    
    # =========================================
    # 1. NOISY HEALTHY (should NOT trigger alerts)
    # =========================================
    print(f"   [1/5] Noisy healthy samples ({samples_per_category})...")
    for i in range(samples_per_category):
        observations = []
        
        # High noise level - looks suspicious but is healthy
        noise = np.random.uniform(1.5, 3.0)
        
        for t in range(WINDOW_SIZE + 5):
            obs = SyntheticObservation()
            obs.timestamp_ms = t * 100
            obs.node_id = f"noisy_healthy_{i}"
            
            # Noisy but still healthy range
            obs.heartbeat_latency_ms = max(5, np.random.normal(25, 8 * noise))
            obs.latency_jitter_ms = max(1, np.random.normal(8, 4 * noise))
            obs.latency_trend = np.random.normal(0, 3 * noise)
            obs.response_rate = min(1.0, max(0.8, np.random.normal(0.92, 0.05 * noise)))
            obs.missed_heartbeats = 1 if np.random.random() < 0.15 else 0
            obs.response_time_avg_ms = max(5, np.random.normal(18, 5 * noise))
            obs.response_time_max_ms = max(20, np.random.normal(45, 12 * noise))
            obs.messages_sent = max(0, int(np.random.normal(10, 3)))
            obs.messages_received = max(0, int(np.random.normal(9, 3)))
            obs.messages_dropped = 1 if np.random.random() < 0.1 else 0
            obs.out_of_order_count = 1 if np.random.random() < 0.05 else 0
            
            observations.append(obs)
        
        features = extract_features(observations[-WINDOW_SIZE:])
        all_features.append(features)
        all_labels.append(0)  # HEALTHY despite noise
    
    # =========================================
    # 2. SUBTLE EARLY DEGRADATION (should detect)
    # =========================================
    print(f"   [2/5] Subtle degradation samples ({samples_per_category})...")
    for i in range(samples_per_category):
        observations = []
        
        # Very subtle degradation - easy to miss
        degradation_rate = 0.002 + np.random.uniform(0, 0.003)
        
        for t in range(WINDOW_SIZE + 5):
            obs = SyntheticObservation()
            obs.timestamp_ms = t * 100
            obs.node_id = f"subtle_degrade_{i}"
            
            # Slowly increasing latency
            base_latency = 20 + t * degradation_rate * 50
            obs.heartbeat_latency_ms = max(5, np.random.normal(base_latency, 3))
            obs.latency_jitter_ms = max(1, np.random.normal(5 + t * 0.05, 2))
            obs.latency_trend = degradation_rate * 100  # Subtle positive trend
            obs.response_rate = min(1.0, max(0.85, 0.98 - t * 0.001))
            obs.missed_heartbeats = 0
            obs.response_time_avg_ms = max(5, np.random.normal(15 + t * 0.1, 3))
            obs.response_time_max_ms = max(20, np.random.normal(30 + t * 0.2, 5))
            obs.messages_sent = max(0, int(np.random.normal(10, 2)))
            obs.messages_received = max(0, int(np.random.normal(10, 2)))
            obs.messages_dropped = 0
            obs.out_of_order_count = 0
            
            observations.append(obs)
        
        features = extract_features(observations[-WINDOW_SIZE:])
        all_features.append(features)
        all_labels.append(PREDICTION_HORIZONS.get('degraded_30s', 1))  # Early warning
    
    # =========================================
    # 3. INTERMITTENT FAILURES (tricky pattern)
    # =========================================
    print(f"   [3/5] Intermittent failure samples ({samples_per_category})...")
    for i in range(samples_per_category):
        observations = []
        
        # On-off failure pattern
        failure_freq = np.random.uniform(0.2, 0.4)
        
        for t in range(WINDOW_SIZE + 5):
            obs = SyntheticObservation()
            obs.timestamp_ms = t * 100
            obs.node_id = f"intermittent_{i}"
            
            # Intermittent spikes
            is_spike = np.random.random() < failure_freq
            
            if is_spike:
                obs.heartbeat_latency_ms = np.random.uniform(80, 200)
                obs.latency_jitter_ms = np.random.uniform(20, 50)
                obs.response_rate = np.random.uniform(0.4, 0.7)
                obs.missed_heartbeats = np.random.randint(1, 3)
            else:
                obs.heartbeat_latency_ms = max(5, np.random.normal(22, 5))
                obs.latency_jitter_ms = max(1, np.random.normal(6, 2))
                obs.response_rate = min(1.0, np.random.normal(0.96, 0.02))
                obs.missed_heartbeats = 0
            
            obs.latency_trend = 5 if is_spike else 0
            obs.response_time_avg_ms = obs.heartbeat_latency_ms * 0.7
            obs.response_time_max_ms = obs.heartbeat_latency_ms * 1.5
            obs.messages_sent = max(0, int(np.random.normal(10, 2)))
            obs.messages_received = max(0, int(np.random.normal(8 if is_spike else 10, 2)))
            obs.messages_dropped = 2 if is_spike else 0
            obs.out_of_order_count = 1 if is_spike else 0
            
            observations.append(obs)
        
        features = extract_features(observations[-WINDOW_SIZE:])
        all_features.append(features)
        all_labels.append(PREDICTION_HORIZONS.get('degraded_10s', 3))  # Degraded
    
    # =========================================
    # 4. CLEAR FAILURES (baseline comparison)
    # =========================================
    print(f"   [4/5] Clear failure samples ({samples_per_category})...")
    failure_types = ['crash', 'slow', 'byzantine', 'partition']
    per_type = samples_per_category // len(failure_types)
    
    for ft in failure_types:
        for i in range(per_type):
            scenario_seed = seed + hash(ft) % 50000 + i * 1000
            
            features, labels, _ = generate_predictive_scenario(
                failure_type=ft,
                failure_time_step=80 + i * 5,
                total_steps=120 + i * 5,
                seed=scenario_seed
            )
            
            if len(features) > 0:
                # Take near-failure sample
                idx = min(len(features) - 1, len(features) - 5)
                all_features.append(features[idx])
                all_labels.append(labels[idx])
    
    # =========================================
    # 5. AMBIGUOUS/MIXED PATTERNS
    # =========================================
    print(f"   [5/5] Ambiguous pattern samples ({samples_per_category})...")
    for i in range(samples_per_category):
        observations = []
        
        # Mix of healthy and degraded signals
        pattern_type = i % 3
        
        for t in range(WINDOW_SIZE + 5):
            obs = SyntheticObservation()
            obs.timestamp_ms = t * 100
            obs.node_id = f"ambiguous_{i}"
            
            if pattern_type == 0:
                # High latency but good response rate (slow network, not failure)
                obs.heartbeat_latency_ms = np.random.uniform(60, 100)
                obs.latency_jitter_ms = np.random.uniform(3, 8)
                obs.response_rate = np.random.uniform(0.95, 1.0)
                obs.missed_heartbeats = 0
                label = 0  # Actually healthy (just slow network)
                
            elif pattern_type == 1:
                # Low latency but dropping messages (early partition)
                obs.heartbeat_latency_ms = np.random.uniform(15, 25)
                obs.latency_jitter_ms = np.random.uniform(2, 5)
                obs.response_rate = np.random.uniform(0.7, 0.85)
                obs.missed_heartbeats = 1 if np.random.random() < 0.3 else 0
                obs.messages_dropped = np.random.randint(1, 4)
                label = PREDICTION_HORIZONS.get('degraded_20s', 2)  # Early warning
                
            else:
                # Erratic jitter (early byzantine)
                obs.heartbeat_latency_ms = np.random.uniform(20, 80)
                obs.latency_jitter_ms = np.random.uniform(15, 40)
                obs.response_rate = np.random.uniform(0.8, 0.95)
                obs.missed_heartbeats = 0
                obs.out_of_order_count = np.random.randint(1, 5)
                label = PREDICTION_HORIZONS.get('degraded_10s', 3)
            
            obs.latency_trend = np.random.uniform(-2, 5)
            obs.response_time_avg_ms = obs.heartbeat_latency_ms * 0.7
            obs.response_time_max_ms = obs.heartbeat_latency_ms * 2
            obs.messages_sent = max(0, int(np.random.normal(10, 2)))
            obs.messages_received = max(0, int(np.random.normal(9, 2)))
            
            observations.append(obs)
        
        features = extract_features(observations[-WINDOW_SIZE:])
        all_features.append(features)
        all_labels.append(label)
    
    X = np.array(all_features)
    y = np.array(all_labels)
    
    # Shuffle
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]
    
    print(f"\n   Total: {len(X)} samples")
    print(f"   Healthy: {np.sum(y == 0)}, Failure/Degraded: {np.sum(y > 0)}")
    
    return X, y


# =============================================================================
# BASELINES
# =============================================================================

class ThresholdDetector:
    """Multi-threshold detector."""
    
    def predict(self, X):
        preds = np.zeros(len(X), dtype=int)
        
        for i, sample in enumerate(X):
            if len(sample.shape) > 1:
                # Analyze full window
                latencies = sample[:, 0] * 200  # Denormalize
                jitters = sample[:, 1] * 100
                response_rates = sample[:, 14]
                missed_hbs = sample[:, 15] * 5
                trends = sample[:, 2] * 20
                
                avg_latency = np.mean(latencies)
                max_latency = np.max(latencies)
                avg_jitter = np.mean(jitters)
                min_response = np.min(response_rates)
                max_missed = np.max(missed_hbs)
                avg_trend = np.mean(trends)
            else:
                avg_latency = sample[0] * 200
                max_latency = avg_latency
                avg_jitter = sample[1] * 100
                min_response = sample[14] if len(sample) > 14 else 1.0
                max_missed = sample[15] * 5 if len(sample) > 15 else 0
                avg_trend = sample[2] * 20 if len(sample) > 2 else 0
            
            # Decision logic
            if max_missed > 3 or min_response < 0.3:
                preds[i] = 8  # Imminent
            elif max_latency > 150 or avg_latency > 100:
                preds[i] = 5  # Failed
            elif avg_jitter > 30 or avg_trend > 15:
                preds[i] = 3  # Degraded
            elif avg_latency > 50 or min_response < 0.8:
                preds[i] = 2  # Early warning
            else:
                preds[i] = 0  # Healthy
        
        return preds


class PhiAccrualDetector:
    """Phi Accrual with window analysis."""
    
    def __init__(self):
        self.mean = 20.0
        self.std = 5.0
    
    def fit(self, X, y):
        healthy = X[y == 0]
        if len(healthy) > 0:
            if len(healthy.shape) == 3:
                latencies = healthy[:, :, 0].flatten() * 200
            else:
                latencies = healthy[:, 0] * 200
            self.mean = np.mean(latencies)
            self.std = max(np.std(latencies), 3.0)
        return self
    
    def predict(self, X):
        preds = np.zeros(len(X), dtype=int)
        
        for i, sample in enumerate(X):
            if len(sample.shape) > 1:
                latencies = sample[:, 0] * 200
                current = latencies[-1]
                window_mean = np.mean(latencies[-10:])
                trend = latencies[-1] - latencies[0]
            else:
                current = sample[0] * 200
                window_mean = current
                trend = 0
            
            # Phi calculation
            z = (current - self.mean) / self.std
            phi = max(0, z * z / 2)
            
            # Trend bonus
            if trend > 20:
                phi += 3
            
            if phi > 20:
                preds[i] = 8
            elif phi > 12:
                preds[i] = 5
            elif phi > 6:
                preds[i] = 3
            elif phi > 3:
                preds[i] = 2
            else:
                preds[i] = 0
        
        return preds


class RandomForestBaseline:
    """Random Forest with comprehensive features."""
    
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
    
    def _extract_features(self, X):
        if len(X.shape) == 2:
            return X
        
        n_samples = X.shape[0]
        features = []
        
        for i in range(n_samples):
            f = []
            sample = X[i]
            
            # For each important feature channel
            for j in [0, 1, 2, 3, 14, 15]:  # latency, jitter, trend, response, missed
                if j < sample.shape[1]:
                    seq = sample[:, j]
                    f.extend([
                        seq[-1], seq[-5:].mean(), seq.mean(), seq.std(),
                        seq.min(), seq.max(), seq[-1] - seq[0],
                        np.percentile(seq, 90), np.percentile(seq, 10)
                    ])
            features.append(f)
        
        return np.array(features)
    
    def fit(self, X, y):
        self.model.fit(self._extract_features(X), y)
        return self
    
    def predict(self, X):
        return self.model.predict(self._extract_features(X))


class VanillaLSTM(nn.Module):
    """Vanilla LSTM - no attention, no ResNet."""
    
    def __init__(self, input_size, hidden=64, layers=2, num_classes=9):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, layers, batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class VanillaLSTMTrainer:
    """Train vanilla LSTM."""
    
    def __init__(self, input_size, num_classes=9, epochs=60):
        self.model = VanillaLSTM(input_size, hidden=64, layers=2, num_classes=num_classes)
        self.epochs = epochs
    
    def fit(self, X, y):
        # Class weights
        counts = np.bincount(y, minlength=NUM_PREDICTION_CLASSES)
        weights = len(y) / (counts + 1) / NUM_PREDICTION_CLASSES
        class_weights = torch.tensor(weights, dtype=torch.float32)
        
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001)
        
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=32, shuffle=True)
        
        self.model.train()
        for ep in range(self.epochs):
            total_loss, correct, total = 0, 0, 0
            for bx, by in loader:
                optimizer.zero_grad()
                out = self.model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                correct += (out.argmax(1) == by).sum().item()
                total += len(by)
            
            if (ep + 1) % 15 == 0:
                print(f"      Epoch {ep+1}/{self.epochs}: loss={total_loss/len(loader):.4f}, acc={correct/total:.4f}")
        
        return self
    
    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            out = self.model(torch.tensor(X, dtype=torch.float32))
            return out.argmax(1).numpy()


def load_proactiveguard():
    """Load ProactiveGuard."""
    try:
        ckpt = torch.load('models/predictive_model.pt', weights_only=False, map_location='cpu')
        model = PredictiveModel(
            input_size=NUM_FEATURES, hidden_size=128, latent_size=64,
            seq_len=WINDOW_SIZE, num_classes=NUM_PREDICTION_CLASSES
        )
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        return model
    except Exception as e:
        print(f"   Error: {e}")
        return None


def run_experiments():
    """Run all experiments on challenging data."""
    print("="*70)
    print("FINAL BASELINE EXPERIMENTS - CHALLENGING TEST DATA")
    print("="*70)
    
    # Get training data
    print("\n📊 Loading training data...")
    X_train, y_train, _ = generate_predictive_training_data()
    print(f"   Training: {len(X_train)} samples")
    
    # Generate challenging test data
    X_test, y_test = generate_challenging_test_data(num_samples=400, seed=99999)
    
    results = {}
    
    # 1. ProactiveGuard
    print("\n" + "="*70)
    print("[1/5] PROACTIVEGUARD (ResNet + LSTM + Attention)")
    print("="*70)
    
    model = load_proactiveguard()
    if model:
        with torch.no_grad():
            out = model(torch.tensor(X_test, dtype=torch.float32))
            y_pred = out['class_logits'].argmax(1).numpy()
        
        m = calculate_metrics(y_test, y_pred)
        results['ProactiveGuard'] = m
        print(f"   Accuracy: {m['accuracy']:.4f}, F1: {m['f1']:.4f}")
        print(f"   FPR: {m['fpr']:.2f}%, FNR: {m['fnr']:.2f}%")
    
    # 2. Threshold
    print("\n" + "="*70)
    print("[2/5] THRESHOLD-BASED")
    print("="*70)
    
    t = ThresholdDetector()
    y_pred = t.predict(X_test)
    m = calculate_metrics(y_test, y_pred)
    results['Threshold'] = m
    print(f"   Accuracy: {m['accuracy']:.4f}, F1: {m['f1']:.4f}")
    print(f"   FPR: {m['fpr']:.2f}%, FNR: {m['fnr']:.2f}%")
    
    # 3. Phi Accrual
    print("\n" + "="*70)
    print("[3/5] PHI ACCRUAL")
    print("="*70)
    
    p = PhiAccrualDetector()
    p.fit(X_train, y_train)
    y_pred = p.predict(X_test)
    m = calculate_metrics(y_test, y_pred)
    results['Phi Accrual'] = m
    print(f"   Accuracy: {m['accuracy']:.4f}, F1: {m['f1']:.4f}")
    print(f"   FPR: {m['fpr']:.2f}%, FNR: {m['fnr']:.2f}%")
    
    # 4. Random Forest
    if SKLEARN:
        print("\n" + "="*70)
        print("[4/5] RANDOM FOREST")
        print("="*70)
        
        rf = RandomForestBaseline()
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        m = calculate_metrics(y_test, y_pred)
        results['Random Forest'] = m
        print(f"   Accuracy: {m['accuracy']:.4f}, F1: {m['f1']:.4f}")
        print(f"   FPR: {m['fpr']:.2f}%, FNR: {m['fnr']:.2f}%")
    
    # 5. Vanilla LSTM
    print("\n" + "="*70)
    print("[5/5] VANILLA LSTM (no attention)")
    print("="*70)
    
    lstm = VanillaLSTMTrainer(NUM_FEATURES, NUM_PREDICTION_CLASSES, epochs=60)
    lstm.fit(X_train, y_train)
    y_pred = lstm.predict(X_test)
    m = calculate_metrics(y_test, y_pred)
    results['Vanilla LSTM'] = m
    print(f"   Accuracy: {m['accuracy']:.4f}, F1: {m['f1']:.4f}")
    print(f"   FPR: {m['fpr']:.2f}%, FNR: {m['fnr']:.2f}%")
    
    # ===================
    # FINAL SUMMARY
    # ===================
    print("\n" + "="*70)
    print("FINAL RESULTS ON CHALLENGING DATA")
    print("="*70)
    
    print(f"\n{'Method':<20} {'Accuracy':<10} {'F1':<10} {'Precision':<10} {'Recall':<10} {'FPR':<10} {'FNR':<10}")
    print("-"*80)
    
    for name, r in sorted(results.items(), key=lambda x: x[1]['f1'], reverse=True):
        print(f"{name:<20} {r['accuracy']:.4f}     {r['f1']:.4f}     {r['precision']:.4f}     "
              f"{r['recall']:.4f}     {r['fpr']:.2f}%     {r['fnr']:.2f}%")
    
    # Save
    Path('results/baselines').mkdir(parents=True, exist_ok=True)
    with open('results/baselines/final_comparison.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Saved to results/baselines/final_comparison.json")
    
    # LaTeX
    print("\n" + "="*70)
    print("LATEX TABLE FOR PAPER")
    print("="*70)
    print("""
\\begin{table}[h]
\\centering
\\caption{Comparison with Baseline Methods on Challenging Test Data}
\\label{tab:baseline_comparison}
\\begin{tabular}{lcccccc}
\\hline
\\textbf{Method} & \\textbf{Accuracy} & \\textbf{F1} & \\textbf{Precision} & \\textbf{Recall} & \\textbf{FPR (\\%)} & \\textbf{FNR (\\%)} \\\\
\\hline""")
    
    for name, r in sorted(results.items(), key=lambda x: x[1]['f1'], reverse=True):
        print(f"{name} & {r['accuracy']:.3f} & {r['f1']:.3f} & {r['precision']:.3f} & "
              f"{r['recall']:.3f} & {r['fpr']:.1f} & {r['fnr']:.1f} \\\\")
    
    print("""\\hline
\\end{tabular}
\\end{table}
""")
    
    return results


if __name__ == "__main__":
    run_experiments()