#!/usr/bin/env python3
"""
Paper 1: Predictive Detection Experiments

Experiments:
1. Prediction Accuracy vs Horizon
2. False Prediction Rate
3. Downtime Comparison (Reactive vs Predictive)
4. Graceful Handoff Success Rate
5. Overhead Analysis
"""

import sys
import os
import json
import numpy as np
import torch
from pathlib import Path
from loguru import logger
from collections import defaultdict

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss} | {level:<7} | {message}")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_predictive import (
    PredictiveModel, generate_predictive_scenario,
    PREDICTION_HORIZONS, NUM_PREDICTION_CLASSES, WINDOW_SIZE, NUM_FEATURES
)
from train_advanced_v2 import create_cluster, extract_features


def load_predictive_model(path='models/predictive_model.pt'):
    """Load trained predictive model."""
    checkpoint = torch.load(path, weights_only=False, map_location='cpu')
    
    model = PredictiveModel(
        input_size=NUM_FEATURES,
        hidden_size=128,
        latent_size=64,
        seq_len=WINDOW_SIZE,
        num_classes=checkpoint.get('num_classes', NUM_PREDICTION_CLASSES)
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


# ============== EXPERIMENT 1: Accuracy vs Horizon ==============

def exp1_accuracy_vs_horizon(model, num_trials=20):
    """
    Measure prediction accuracy at different time horizons.
    
    Key question: How far ahead can we reliably predict?
    """
    logger.info("="*60)
    logger.info("EXPERIMENT 1: Prediction Accuracy vs Horizon")
    logger.info("="*60)
    
    horizons_seconds = [5, 10, 15, 20, 25, 30]
    failure_types = ['crash', 'slow', 'byzantine', 'partition']
    
    results = {h: {'correct': 0, 'total': 0, 'by_type': defaultdict(lambda: {'correct': 0, 'total': 0})} 
               for h in horizons_seconds}
    
    for failure_type in failure_types:
        logger.info(f"\nTesting {failure_type}...")
        
        for trial in range(num_trials):
            seed = 42 + hash(failure_type) + trial * 100
            failure_time = 150  # Fixed failure time
            
            features, labels, ttf = generate_predictive_scenario(
                failure_type=failure_type,
                failure_time_step=failure_time,
                total_steps=failure_time + 30,
                seed=seed
            )
            
            if len(features) == 0:
                continue
            
            for horizon in horizons_seconds:
                steps_before = int(horizon * 10)
                target_step = failure_time - steps_before - WINDOW_SIZE
                
                if target_step < 0 or target_step >= len(features):
                    continue
                
                X = torch.tensor(features[target_step:target_step+1], dtype=torch.float32)
                
                with torch.no_grad():
                    pred = model.predict(X).item()
                
                true_label = labels[target_step]
                
                # Correct if both predict warning/failure or both predict healthy
                is_correct = (pred > 0) == (true_label > 0)
                
                results[horizon]['total'] += 1
                results[horizon]['by_type'][failure_type]['total'] += 1
                
                if is_correct:
                    results[horizon]['correct'] += 1
                    results[horizon]['by_type'][failure_type]['correct'] += 1
    
    # Calculate accuracies
    logger.info("\n" + "-"*60)
    logger.info("RESULTS")
    logger.info("-"*60)
    
    summary = {}
    logger.info(f"\n{'Horizon':<12} {'Accuracy':<12} {'Samples':<10}")
    logger.info("-" * 34)
    
    for horizon in horizons_seconds:
        if results[horizon]['total'] > 0:
            acc = results[horizon]['correct'] / results[horizon]['total']
            summary[horizon] = {
                'accuracy': float(acc),
                'total': results[horizon]['total'],
                'by_type': {}
            }
            
            for ft, data in results[horizon]['by_type'].items():
                if data['total'] > 0:
                    ft_acc = data['correct'] / data['total']
                    summary[horizon]['by_type'][ft] = float(ft_acc)
            
            logger.info(f"T-{horizon}s{'':<7} {acc:.4f}{'':<6} {results[horizon]['total']}")
    
    return summary


# ============== EXPERIMENT 2: False Prediction Rate ==============

def exp2_false_prediction_rate(model, duration_steps=500, num_trials=10):
    """
    Measure false positive rate on healthy systems.
    
    Key question: How often do we incorrectly predict failure?
    """
    logger.info("\n" + "="*60)
    logger.info("EXPERIMENT 2: False Prediction Rate")
    logger.info("="*60)
    
    results = {'false_warnings': 0, 'total_predictions': 0, 'by_trial': []}
    
    for trial in range(num_trials):
        seed = 42 + trial * 100
        clock, network, nodes, injector = create_cluster(seed=seed)
        
        for node in nodes:
            node.start()
        clock.run_for(2000)
        
        target = nodes[1]
        observations = []
        trial_false = 0
        trial_total = 0
        
        for _ in range(duration_steps):
            clock.run_for(100)
            for node in nodes:
                node.tick()
            
            obs = target.generate_observation("detector")
            observations.append(obs)
            
            if len(observations) >= WINDOW_SIZE:
                features = extract_features(observations[-WINDOW_SIZE:])
                X = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
                
                with torch.no_grad():
                    pred = model.predict(X).item()
                
                trial_total += 1
                if pred > 0:  # Predicted warning/failure on healthy system
                    trial_false += 1
        
        fp_rate = trial_false / trial_total if trial_total > 0 else 0
        results['by_trial'].append(fp_rate)
        results['false_warnings'] += trial_false
        results['total_predictions'] += trial_total
        
        logger.info(f"Trial {trial+1}: FP rate = {fp_rate:.4f}")
    
    overall_fp = results['false_warnings'] / results['total_predictions']
    results['overall_fp_rate'] = float(overall_fp)
    results['mean_fp_rate'] = float(np.mean(results['by_trial']))
    results['std_fp_rate'] = float(np.std(results['by_trial']))
    
    logger.info(f"\nOverall FP rate: {overall_fp:.4f}")
    logger.info(f"Mean ± Std: {results['mean_fp_rate']:.4f} ± {results['std_fp_rate']:.4f}")
    
    return results


# ============== EXPERIMENT 3: Downtime Comparison ==============

def exp3_downtime_comparison(model, num_trials=20):
    """
    Compare downtime: reactive vs predictive detection.
    
    Reactive: Detect after failure, then recover
    Predictive: Detect before failure, graceful handoff
    """
    logger.info("\n" + "="*60)
    logger.info("EXPERIMENT 3: Downtime Comparison")
    logger.info("="*60)
    
    failure_types = ['crash', 'slow', 'byzantine', 'partition']
    
    results = {
        'reactive': {'by_type': defaultdict(list)},
        'predictive': {'by_type': defaultdict(list)}
    }
    
    # Simulation parameters
    detection_timeout = 500  # Reactive detection timeout (ms)
    recovery_time = 1000     # Time to elect new leader (ms)
    handoff_time = 200       # Graceful handoff time (ms)
    
    for failure_type in failure_types:
        logger.info(f"\nTesting {failure_type}...")
        
        for trial in range(num_trials):
            seed = 42 + hash(failure_type) + trial * 100
            failure_time_step = 100  # Failure at step 100
            
            features, labels, ttf = generate_predictive_scenario(
                failure_type=failure_type,
                failure_time_step=failure_time_step,
                total_steps=failure_time_step + 50,
                seed=seed
            )
            
            if len(features) == 0:
                continue
            
            # REACTIVE: Downtime = detection_timeout + recovery_time
            reactive_downtime = detection_timeout + recovery_time
            results['reactive']['by_type'][failure_type].append(reactive_downtime)
            
            # PREDICTIVE: Find earliest warning
            predictive_downtime = reactive_downtime  # Default to reactive
            
            for step in range(len(features)):
                X = torch.tensor(features[step:step+1], dtype=torch.float32)
                
                with torch.no_grad():
                    pred = model.predict(X).item()
                
                if pred > 0:  # Warning detected
                    # Calculate time before failure
                    steps_to_failure = failure_time_step - WINDOW_SIZE - step
                    time_to_failure_ms = steps_to_failure * 100
                    
                    if time_to_failure_ms > handoff_time:
                        # Graceful handoff possible
                        predictive_downtime = handoff_time
                    else:
                        # Not enough time, but still faster
                        predictive_downtime = time_to_failure_ms + recovery_time
                    
                    break
            
            results['predictive']['by_type'][failure_type].append(predictive_downtime)
    
    # Calculate statistics
    logger.info("\n" + "-"*60)
    logger.info("RESULTS (Downtime in ms)")
    logger.info("-"*60)
    
    summary = {'reactive': {}, 'predictive': {}, 'improvement': {}}
    
    logger.info(f"\n{'Failure Type':<15} {'Reactive':<15} {'Predictive':<15} {'Reduction':<12}")
    logger.info("-" * 57)
    
    for failure_type in failure_types:
        reactive_mean = np.mean(results['reactive']['by_type'][failure_type])
        predictive_mean = np.mean(results['predictive']['by_type'][failure_type])
        reduction = (reactive_mean - predictive_mean) / reactive_mean * 100
        
        summary['reactive'][failure_type] = float(reactive_mean)
        summary['predictive'][failure_type] = float(predictive_mean)
        summary['improvement'][failure_type] = float(reduction)
        
        logger.info(f"{failure_type:<15} {reactive_mean:<15.0f} {predictive_mean:<15.0f} {reduction:<12.1f}%")
    
    # Overall
    all_reactive = [v for vals in results['reactive']['by_type'].values() for v in vals]
    all_predictive = [v for vals in results['predictive']['by_type'].values() for v in vals]
    
    overall_reactive = np.mean(all_reactive)
    overall_predictive = np.mean(all_predictive)
    overall_reduction = (overall_reactive - overall_predictive) / overall_reactive * 100
    
    summary['overall'] = {
        'reactive_mean': float(overall_reactive),
        'predictive_mean': float(overall_predictive),
        'reduction_percent': float(overall_reduction)
    }
    
    logger.info("-" * 57)
    logger.info(f"{'OVERALL':<15} {overall_reactive:<15.0f} {overall_predictive:<15.0f} {overall_reduction:<12.1f}%")
    
    return summary


# ============== EXPERIMENT 4: Time-to-Failure Prediction ==============

def exp4_ttf_prediction(model, num_trials=20):
    """
    Evaluate time-to-failure regression accuracy.
    
    Key question: How accurate is our TTF prediction?
    """
    logger.info("\n" + "="*60)
    logger.info("EXPERIMENT 4: Time-to-Failure Prediction")
    logger.info("="*60)
    
    failure_types = ['crash', 'slow', 'byzantine', 'partition']
    
    results = {
        'all_errors': [],
        'by_type': defaultdict(list),
        'by_horizon': defaultdict(list)
    }
    
    for failure_type in failure_types:
        logger.info(f"\nTesting {failure_type}...")
        
        for trial in range(num_trials):
            seed = 42 + hash(failure_type) + trial * 100
            failure_time_step = 150
            
            features, labels, ttf = generate_predictive_scenario(
                failure_type=failure_type,
                failure_time_step=failure_time_step,
                total_steps=failure_time_step + 30,
                seed=seed
            )
            
            if len(features) == 0:
                continue
            
            # Test at different horizons
            for horizon_s in [5, 10, 20, 30]:
                steps_before = int(horizon_s * 10)
                target_step = failure_time_step - steps_before - WINDOW_SIZE
                
                if target_step < 0 or target_step >= len(features):
                    continue
                
                X = torch.tensor(features[target_step:target_step+1], dtype=torch.float32)
                
                with torch.no_grad():
                    outputs = model(X)
                    ttf_pred = outputs['ttf'].item() * 30  # Denormalize
                
                ttf_true = ttf[target_step] * 30  # Denormalize
                error = abs(ttf_pred - ttf_true)
                
                results['all_errors'].append(error)
                results['by_type'][failure_type].append(error)
                results['by_horizon'][horizon_s].append(error)
    
    # Calculate statistics
    logger.info("\n" + "-"*60)
    logger.info("RESULTS (MAE in seconds)")
    logger.info("-"*60)
    
    summary = {
        'overall_mae': float(np.mean(results['all_errors'])),
        'overall_std': float(np.std(results['all_errors'])),
        'by_type': {},
        'by_horizon': {}
    }
    
    logger.info(f"\nBy Failure Type:")
    for ft, errors in results['by_type'].items():
        mae = np.mean(errors)
        summary['by_type'][ft] = float(mae)
        logger.info(f"  {ft}: MAE = {mae:.2f}s")
    
    logger.info(f"\nBy Horizon:")
    for horizon, errors in sorted(results['by_horizon'].items()):
        mae = np.mean(errors)
        summary['by_horizon'][str(horizon)] = float(mae)
        logger.info(f"  T-{horizon}s: MAE = {mae:.2f}s")
    
    logger.info(f"\nOverall: MAE = {summary['overall_mae']:.2f}s ± {summary['overall_std']:.2f}s")
    
    return summary


# ============== EXPERIMENT 5: Overhead Analysis ==============

def exp5_overhead_analysis(model, num_iterations=1000):
    """
    Measure computational overhead of predictive detection.
    
    Key question: How much extra computation does prediction add?
    """
    logger.info("\n" + "="*60)
    logger.info("EXPERIMENT 5: Overhead Analysis")
    logger.info("="*60)
    
    import time
    
    # Generate sample input
    sample_features = np.random.randn(WINDOW_SIZE, NUM_FEATURES).astype(np.float32)
    X = torch.tensor(sample_features).unsqueeze(0)
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(X)
    
    # Measure inference time
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(X)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms
    
    results = {
        'mean_inference_ms': float(np.mean(times)),
        'std_inference_ms': float(np.std(times)),
        'p50_inference_ms': float(np.percentile(times, 50)),
        'p95_inference_ms': float(np.percentile(times, 95)),
        'p99_inference_ms': float(np.percentile(times, 99)),
        'iterations': num_iterations
    }
    
    logger.info(f"\nInference Time (ms):")
    logger.info(f"  Mean: {results['mean_inference_ms']:.3f}")
    logger.info(f"  Std:  {results['std_inference_ms']:.3f}")
    logger.info(f"  P50:  {results['p50_inference_ms']:.3f}")
    logger.info(f"  P95:  {results['p95_inference_ms']:.3f}")
    logger.info(f"  P99:  {results['p99_inference_ms']:.3f}")
    
    # Compare with heartbeat interval
    heartbeat_interval = 150  # ms
    overhead_percent = results['mean_inference_ms'] / heartbeat_interval * 100
    results['overhead_percent'] = float(overhead_percent)
    
    logger.info(f"\nOverhead relative to heartbeat interval ({heartbeat_interval}ms):")
    logger.info(f"  {overhead_percent:.2f}%")
    
    return results


# ============== MAIN ==============

def run_all_experiments(output_dir='results/paper1'):
    """Run all Paper 1 experiments."""
    logger.info("="*60)
    logger.info("PAPER 1: PREDICTIVE FAILURE DETECTION")
    logger.info("Complete Experiment Suite")
    logger.info("="*60)
    
    # Load model
    model = load_predictive_model()
    logger.info("Model loaded successfully")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    # Experiment 1
    all_results['exp1_accuracy_vs_horizon'] = exp1_accuracy_vs_horizon(model, num_trials=10)
    
    # Experiment 2
    all_results['exp2_false_prediction_rate'] = exp2_false_prediction_rate(model, duration_steps=300, num_trials=5)
    
    # Experiment 3
    all_results['exp3_downtime_comparison'] = exp3_downtime_comparison(model, num_trials=10)
    
    # Experiment 4
    all_results['exp4_ttf_prediction'] = exp4_ttf_prediction(model, num_trials=10)
    
    # Experiment 5
    all_results['exp5_overhead'] = exp5_overhead_analysis(model, num_iterations=500)
    
    # Save all results
    with open(f"{output_dir}/all_experiments.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    logger.info("\n" + "="*60)
    logger.info("ALL EXPERIMENTS COMPLETE!")
    logger.info(f"Results saved to {output_dir}/")
    logger.info("="*60)
    
    return all_results


if __name__ == "__main__":
    run_all_experiments()