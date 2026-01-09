#!/usr/bin/env python3
"""
Test predictive model on generated cluster test data.
FIXED: Aligned feature generation with training data format.
"""

import pandas as pd
import numpy as np
import torch
from pathlib import Path
from loguru import logger

logger.remove()
logger.add(lambda msg: print(msg, end=""), format="{time:HH:mm:ss} | {level:<7} | {message}\n")

from train_predictive import (
    PredictiveModel, extract_features, SyntheticObservation,
    NUM_FEATURES, WINDOW_SIZE, NUM_PREDICTION_CLASSES, PREDICTION_HORIZONS,
    create_degraded_observation, generate_predictive_scenario
)


def generate_test_data_aligned(num_healthy=100, num_failure=100):
    """
    Generate test data using the SAME method as training data.
    This ensures feature distributions match.
    """
    logger.info("Generating aligned test data...")
    
    all_features = []
    all_labels = []
    
    # Generate healthy sequences (same as training)
    logger.info(f"  Generating {num_healthy} healthy sequences...")
    for i in range(num_healthy):
        seed = 9999 + i * 100  # Different seeds than training
        
        observations = []
        for t in range(WINDOW_SIZE + 10):
            obs = SyntheticObservation()
            obs.timestamp_ms = t * 100
            obs.node_id = f"test_node_{i}"
            
            # Healthy metrics (matching training distribution)
            obs.heartbeat_latency_ms = max(5, np.random.normal(20, 5))
            obs.latency_jitter_ms = max(1, np.random.normal(5, 2))
            obs.latency_trend = np.random.normal(0, 1)
            obs.response_rate = min(1.0, max(0.9, np.random.normal(0.98, 0.02)))
            obs.missed_heartbeats = 0
            obs.response_time_avg_ms = max(5, np.random.normal(15, 3))
            obs.response_time_max_ms = max(10, np.random.normal(30, 5))
            obs.messages_sent = max(0, int(np.random.normal(10, 2)))
            obs.messages_received = max(0, int(np.random.normal(10, 2)))
            obs.messages_dropped = 0
            obs.out_of_order_count = 0
            
            observations.append(obs)
        
        features = extract_features(observations[-WINDOW_SIZE:])
        all_features.append(features)
        all_labels.append(PREDICTION_HORIZONS['healthy'])
    
    # Generate failure sequences using SAME method as training
    failure_types = ['crash', 'slow', 'byzantine', 'partition']
    failures_per_type = num_failure // len(failure_types)
    
    logger.info(f"  Generating {num_failure} failure sequences...")
    for failure_type in failure_types:
        for i in range(failures_per_type):
            seed = 8888 + hash(failure_type) % 1000 + i * 100
            
            # Use the SAME generation function as training
            features, labels, ttf = generate_predictive_scenario(
                failure_type=failure_type,
                failure_time_step=100 + i * 10,
                total_steps=150 + i * 10,
                seed=seed
            )
            
            if len(features) > 0:
                # Take a random sample from the sequence
                idx = np.random.randint(0, len(features))
                all_features.append(features[idx])
                all_labels.append(labels[idx])
    
    X = np.array(all_features)
    y = np.array(all_labels)
    
    # Shuffle
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]
    
    logger.info(f"  Total: {len(X)} sequences")
    
    # Label distribution
    unique, counts = np.unique(y, return_counts=True)
    for u, c in zip(unique, counts):
        name = [k for k, v in PREDICTION_HORIZONS.items() if v == u][0]
        logger.info(f"    {name}: {c}")
    
    return X, y


def test_model():
    """Test predictive model on aligned test data."""
    logger.info("="*60)
    logger.info("TESTING PREDICTIVE MODEL")
    logger.info("="*60)
    
    # Load model
    logger.info("\n📦 Loading model...")
    try:
        checkpoint = torch.load('models/predictive_model.pt', weights_only=False, map_location='cpu')
        
        model = PredictiveModel(
            input_size=NUM_FEATURES,
            hidden_size=128,
            latent_size=64,
            seq_len=WINDOW_SIZE,
            num_classes=NUM_PREDICTION_CLASSES
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        logger.info("✅ Model loaded")
        logger.info(f"   Training accuracy was: {checkpoint.get('best_val_acc', 'N/A')}")
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        return
    
    # Generate aligned test data
    logger.info("\n📊 Generating test data (aligned with training distribution)...")
    X_test, y_test = generate_test_data_aligned(num_healthy=100, num_failure=100)
    
    # Run predictions
    logger.info("\n🔮 Running predictions...")
    X_tensor = torch.tensor(X_test, dtype=torch.float32)
    
    with torch.no_grad():
        outputs = model(X_tensor)
        predictions = torch.argmax(outputs['class_logits'], dim=1).numpy()
        confidence = outputs['confidence'].squeeze(-1).numpy()
        ttf_pred = outputs['ttf'].squeeze(-1).numpy() * 30
    
    # Calculate metrics
    logger.info("\n" + "="*60)
    logger.info("RESULTS")
    logger.info("="*60)
    
    # Overall accuracy
    accuracy = (predictions == y_test).mean()
    logger.info(f"\n📊 Overall Accuracy: {accuracy:.4f}")
    
    # Binary accuracy
    binary_true = (y_test > 0).astype(int)
    binary_pred = (predictions > 0).astype(int)
    binary_acc = (binary_pred == binary_true).mean()
    logger.info(f"📊 Binary Accuracy (healthy vs warning): {binary_acc:.4f}")
    
    # Confusion matrix
    tp = ((binary_pred == 1) & (binary_true == 1)).sum()
    tn = ((binary_pred == 0) & (binary_true == 0)).sum()
    fp = ((binary_pred == 1) & (binary_true == 0)).sum()
    fn = ((binary_pred == 0) & (binary_true == 1)).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    logger.info(f"\n📊 Failure Detection Metrics:")
    logger.info(f"   Precision: {precision:.4f}")
    logger.info(f"   Recall: {recall:.4f}")
    logger.info(f"   F1 Score: {f1:.4f}")
    
    logger.info(f"\n📊 Confusion Matrix:")
    logger.info(f"   TP (correct warnings):  {tp}")
    logger.info(f"   TN (correct healthy):   {tn}")
    logger.info(f"   FP (false alarms):      {fp}")
    logger.info(f"   FN (missed failures):   {fn}")
    
    # Per-class accuracy
    logger.info(f"\n📊 Per-Class Performance:")
    logger.info(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Count':<8}")
    logger.info("-" * 64)
    
    for name, cls_id in PREDICTION_HORIZONS.items():
        mask = y_test == cls_id
        if mask.sum() == 0:
            continue
        
        cls_tp = ((predictions == cls_id) & (y_test == cls_id)).sum()
        cls_fp = ((predictions == cls_id) & (y_test != cls_id)).sum()
        cls_fn = ((predictions != cls_id) & (y_test == cls_id)).sum()
        
        cls_precision = cls_tp / (cls_tp + cls_fp) if (cls_tp + cls_fp) > 0 else 0
        cls_recall = cls_tp / (cls_tp + cls_fn) if (cls_tp + cls_fn) > 0 else 0
        cls_f1 = 2 * cls_precision * cls_recall / (cls_precision + cls_recall) if (cls_precision + cls_recall) > 0 else 0
        
        logger.info(f"{name:<20} {cls_precision:<12.4f} {cls_recall:<12.4f} {cls_f1:<12.4f} {mask.sum():<8}")
    
    logger.info(f"\n📊 Confidence: {confidence.mean():.4f} (avg)")
    logger.info(f"📊 TTF Prediction: {ttf_pred.mean():.2f}s (avg)")
    
    # Save results
    results = {
        'dataset': 'Aligned Test Data',
        'num_sequences': len(X_test),
        'accuracy': float(accuracy),
        'binary_accuracy': float(binary_acc),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'confusion': {'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)},
        'avg_confidence': float(confidence.mean()),
        'avg_ttf_prediction': float(ttf_pred.mean())
    }
    
    import json
    Path('results/paper1').mkdir(parents=True, exist_ok=True)
    with open('results/paper1/test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n✅ Results saved to results/paper1/test_results.json")
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("SUMMARY FOR PAPER")
    logger.info("="*60)
    logger.info(f"✅ Overall Accuracy: {accuracy:.1%}")
    logger.info(f"✅ Binary Detection Accuracy: {binary_acc:.1%}")
    logger.info(f"✅ Failure Detection Precision: {precision:.4f}")
    logger.info(f"✅ Failure Detection Recall: {recall:.4f}")
    logger.info(f"✅ Failure Detection F1: {f1:.4f}")
    
    if fp + tn > 0:
        logger.info(f"✅ False Positive Rate: {fp/(fp+tn)*100:.2f}%")
    if fn + tp > 0:
        logger.info(f"✅ False Negative Rate: {fn/(fn+tp)*100:.2f}%")
    
    return results


if __name__ == "__main__":
    test_model()