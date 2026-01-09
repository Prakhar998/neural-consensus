#!/usr/bin/env python3
"""
Retrain predictive model with better class balance.
Focus on reducing false positives (healthy misclassified as failures).
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split, WeightedRandomSampler
from pathlib import Path
from loguru import logger
from collections import defaultdict

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss} | {level:<7} | {message}")

from train_predictive import (
    PredictiveModel, extract_features, SyntheticObservation,
    generate_predictive_training_data, generate_predictive_scenario,
    NUM_FEATURES, WINDOW_SIZE, NUM_PREDICTION_CLASSES, PREDICTION_HORIZONS,
    HORIZON_NAMES, PredictiveLoss
)


def generate_balanced_training_data():
    """Generate training data with MORE healthy samples."""
    logger.info("="*60)
    logger.info("GENERATING BALANCED TRAINING DATA")
    logger.info("="*60)
    
    all_features = []
    all_labels = []
    all_ttf = []
    
    # MORE healthy data (double the amount)
    logger.info("\nGenerating HEALTHY scenarios (extra)...")
    for scenario in range(20):  # More healthy scenarios
        seed = 42 + scenario * 100
        
        observations = []
        scenario_features = []
        
        for i in range(150):  # Longer sequences
            obs = SyntheticObservation()
            obs.timestamp_ms = i * 100
            obs.node_id = f"healthy_node_{scenario}"
            
            # Healthy metrics
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
            
            if len(observations) >= WINDOW_SIZE:
                features = extract_features(observations[-WINDOW_SIZE:])
                scenario_features.append(features)
        
        all_features.append(np.array(scenario_features))
        all_labels.append(np.zeros(len(scenario_features), dtype=int))
        all_ttf.append(np.ones(len(scenario_features)))
    
    logger.info(f"  Generated {sum(len(f) for f in all_features)} healthy samples")
    
    # Failure scenarios (same as before)
    failure_types = ['crash', 'slow', 'byzantine', 'partition']
    
    for failure_type in failure_types:
        logger.info(f"\nGenerating {failure_type} prediction scenarios...")
        
        for scenario in range(5):
            seed = 42 + hash(failure_type) % 1000 + scenario * 100
            failure_time = 100 + scenario * 20
            
            features, labels, ttf = generate_predictive_scenario(
                failure_type=failure_type,
                failure_time_step=failure_time,
                total_steps=failure_time + 50,
                seed=seed
            )
            
            if len(features) > 0:
                all_features.append(features)
                all_labels.append(labels)
                all_ttf.append(ttf)
    
    # Combine
    X = np.concatenate(all_features)
    y = np.concatenate(all_labels)
    ttf = np.concatenate(all_ttf)
    
    # Shuffle
    idx = np.random.permutation(len(X))
    X, y, ttf = X[idx], y[idx], ttf[idx]
    
    # Print distribution
    logger.info(f"\nTotal samples: {len(X)}")
    unique, counts = np.unique(y, return_counts=True)
    for u, c in zip(unique, counts):
        name = HORIZON_NAMES.get(u, f"class_{u}")
        pct = c / len(y) * 100
        logger.info(f"  {name}: {c} ({pct:.1f}%)")
    
    return X, y, ttf


def train_balanced_model(X, y, ttf, epochs=100, batch_size=64, lr=0.001):
    """Train with focal loss and better class weighting."""
    logger.info("="*60)
    logger.info("TRAINING BALANCED MODEL")
    logger.info("="*60)
    
    model = PredictiveModel(
        input_size=NUM_FEATURES,
        hidden_size=128,
        latent_size=64,
        seq_len=WINDOW_SIZE,
        num_classes=NUM_PREDICTION_CLASSES
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")
    
    # Prepare data
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    ttf_tensor = torch.tensor(ttf, dtype=torch.float32)
    binary_tensor = (y_tensor > 0).long()
    
    # Calculate class weights - INCREASE weight for healthy class
    class_counts = np.bincount(y, minlength=NUM_PREDICTION_CLASSES)
    
    # Custom weights: healthy class gets MORE weight to reduce FP
    class_weights = torch.zeros(NUM_PREDICTION_CLASSES)
    total_samples = len(y)
    
    for i in range(NUM_PREDICTION_CLASSES):
        if class_counts[i] > 0:
            # Standard inverse frequency weight
            weight = total_samples / (NUM_PREDICTION_CLASSES * class_counts[i])
            
            # BOOST healthy class weight by 2x
            if i == 0:  # healthy
                weight *= 2.0
            
            class_weights[i] = weight
        else:
            class_weights[i] = 1.0
    
    logger.info(f"Class weights: {class_weights.numpy().round(2)}")
    
    # Use weighted sampler to balance batches
    sample_weights = class_weights[y_tensor].numpy()
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    dataset = TensorDataset(X_tensor, y_tensor, ttf_tensor, binary_tensor)
    
    # Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, len(dataset)))
    
    train_set = torch.utils.data.Subset(dataset, train_indices)
    val_set = torch.utils.data.Subset(dataset, val_indices)
    
    # Create sampler for training only
    train_weights = sample_weights[train_indices]
    train_sampler = WeightedRandomSampler(
        weights=train_weights,
        num_samples=len(train_weights),
        replacement=True
    )
    
    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    
    logger.info(f"Train: {train_size}, Val: {val_size}")
    
    # Focal loss to focus on hard examples
    class FocalLoss(nn.Module):
        def __init__(self, weight=None, gamma=2.0):
            super().__init__()
            self.weight = weight
            self.gamma = gamma
        
        def forward(self, inputs, targets):
            ce_loss = nn.functional.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
            pt = torch.exp(-ce_loss)
            focal_loss = ((1 - pt) ** self.gamma) * ce_loss
            return focal_loss.mean()
    
    criterion_class = FocalLoss(weight=class_weights, gamma=2.0)
    criterion_binary = nn.CrossEntropyLoss()
    criterion_ttf = nn.MSELoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
    
    best_val_acc = 0
    best_healthy_recall = 0
    patience = 25
    no_improve = 0
    
    Path('models').mkdir(parents=True, exist_ok=True)
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_x, batch_y, batch_ttf, batch_binary in train_loader:
            optimizer.zero_grad()
            
            outputs = model(batch_x)
            
            loss_class = criterion_class(outputs['class_logits'], batch_y)
            loss_binary = criterion_binary(outputs['binary_logits'], batch_binary)
            loss_ttf = criterion_ttf(outputs['ttf'].squeeze(-1), batch_ttf)
            
            # Total loss with emphasis on classification
            loss = loss_class + 0.3 * loss_binary + 0.3 * loss_ttf
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            _, pred = outputs['class_logits'].max(1)
            train_total += batch_y.size(0)
            train_correct += pred.eq(batch_y).sum().item()
        
        scheduler.step()
        
        # Validation with healthy recall tracking
        model.eval()
        val_correct = 0
        val_total = 0
        healthy_correct = 0
        healthy_total = 0
        
        with torch.no_grad():
            for batch_x, batch_y, batch_ttf, batch_binary in val_loader:
                outputs = model(batch_x)
                _, pred = outputs['class_logits'].max(1)
                
                val_total += batch_y.size(0)
                val_correct += pred.eq(batch_y).sum().item()
                
                # Track healthy class specifically
                healthy_mask = batch_y == 0
                if healthy_mask.sum() > 0:
                    healthy_total += healthy_mask.sum().item()
                    healthy_correct += (pred[healthy_mask] == 0).sum().item()
        
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        healthy_recall = healthy_correct / healthy_total if healthy_total > 0 else 0
        
        if (epoch + 1) % 5 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs} - "
                       f"train_acc: {train_acc:.4f}, val_acc: {val_acc:.4f}, "
                       f"healthy_recall: {healthy_recall:.4f}")
        
        # Save based on BOTH overall accuracy AND healthy recall
        combined_score = val_acc * 0.5 + healthy_recall * 0.5
        
        if combined_score > (best_val_acc * 0.5 + best_healthy_recall * 0.5):
            best_val_acc = val_acc
            best_healthy_recall = healthy_recall
            no_improve = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'num_classes': NUM_PREDICTION_CLASSES,
                'prediction_horizons': PREDICTION_HORIZONS,
                'best_val_acc': best_val_acc,
                'best_healthy_recall': best_healthy_recall
            }, 'models/predictive_model.pt')
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    checkpoint = torch.load('models/predictive_model.pt', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    logger.info(f"\nBest validation accuracy: {best_val_acc:.4f}")
    logger.info(f"Best healthy recall: {best_healthy_recall:.4f}")
    logger.info("Model saved to models/predictive_model.pt")
    
    return model


def quick_evaluate(model, num_test=200):
    """Quick evaluation on test data."""
    logger.info("\n" + "="*60)
    logger.info("QUICK EVALUATION")
    logger.info("="*60)
    
    # Generate balanced test data
    test_features = []
    test_labels = []
    
    # 50% healthy
    for i in range(num_test // 2):
        observations = []
        for t in range(WINDOW_SIZE + 10):
            obs = SyntheticObservation()
            obs.timestamp_ms = t * 100
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
        test_features.append(features)
        test_labels.append(0)  # healthy
    
    # 50% failures
    failure_types = ['crash', 'slow', 'byzantine', 'partition']
    per_type = (num_test // 2) // len(failure_types)
    
    for ft in failure_types:
        for i in range(per_type):
            seed = 7777 + hash(ft) + i * 100
            features, labels, _ = generate_predictive_scenario(
                failure_type=ft,
                failure_time_step=100,
                total_steps=150,
                seed=seed
            )
            if len(features) > 0:
                idx = len(features) - 10  # Near failure
                test_features.append(features[idx])
                test_labels.append(labels[idx])
    
    X_test = np.array(test_features)
    y_test = np.array(test_labels)
    
    # Predict
    model.eval()
    X_tensor = torch.tensor(X_test, dtype=torch.float32)
    
    with torch.no_grad():
        outputs = model(X_tensor)
        predictions = torch.argmax(outputs['class_logits'], dim=1).numpy()
    
    # Metrics
    accuracy = (predictions == y_test).mean()
    
    binary_true = (y_test > 0).astype(int)
    binary_pred = (predictions > 0).astype(int)
    binary_acc = (binary_pred == binary_true).mean()
    
    # Healthy recall
    healthy_mask = y_test == 0
    healthy_recall = (predictions[healthy_mask] == 0).mean() if healthy_mask.sum() > 0 else 0
    
    # Failure recall
    failure_mask = y_test > 0
    failure_recall = (predictions[failure_mask] > 0).mean() if failure_mask.sum() > 0 else 0
    
    tp = ((binary_pred == 1) & (binary_true == 1)).sum()
    tn = ((binary_pred == 0) & (binary_true == 0)).sum()
    fp = ((binary_pred == 1) & (binary_true == 0)).sum()
    fn = ((binary_pred == 0) & (binary_true == 1)).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    logger.info(f"\n📊 Overall Accuracy: {accuracy:.4f}")
    logger.info(f"📊 Binary Accuracy: {binary_acc:.4f}")
    logger.info(f"📊 Healthy Recall: {healthy_recall:.4f}")
    logger.info(f"📊 Failure Recall: {failure_recall:.4f}")
    logger.info(f"📊 Precision: {precision:.4f}")
    logger.info(f"📊 F1 Score: {f1:.4f}")
    
    logger.info(f"\n📊 Confusion Matrix:")
    logger.info(f"   TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    
    fp_rate = fp / (fp + tn) * 100 if (fp + tn) > 0 else 0
    fn_rate = fn / (fn + tp) * 100 if (fn + tp) > 0 else 0
    logger.info(f"   False Positive Rate: {fp_rate:.1f}%")
    logger.info(f"   False Negative Rate: {fn_rate:.1f}%")


def main():
    logger.info("="*60)
    logger.info("RETRAINING WITH BALANCED DATA")
    logger.info("="*60)
    
    # Generate balanced data
    X, y, ttf = generate_balanced_training_data()
    
    # Train
    model = train_balanced_model(X, y, ttf, epochs=100, batch_size=64)
    
    # Evaluate
    quick_evaluate(model, num_test=200)
    
    logger.info("\n" + "="*60)
    logger.info("DONE! Run test_on_google_cluster.py again to see improved results")
    logger.info("="*60)


if __name__ == "__main__":
    main()