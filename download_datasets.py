#!/usr/bin/env python3
"""
Generate realistic test datasets for predictive failure detection.

Since real cluster datasets (Google, LANL) are large and hard to download,
we generate synthetic but realistic test data based on published statistics
from those datasets.

Reference statistics from:
- Google Cluster Data 2011 (Reiss et al.)
- LANL Failure Data (Schroeder & Gibson, DSN 2006)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import json

np.random.seed(42)

def generate_machine_events(num_machines=100, duration_hours=24*7, failure_rate=0.05):
    """
    Generate realistic machine events based on Google Cluster statistics.
    
    Event types:
    - 0: ADD (machine added)
    - 1: REMOVE (machine failed/removed)
    - 2: UPDATE (machine updated)
    """
    print("📊 Generating machine events...")
    
    events = []
    start_time = 0
    
    # Add all machines at start
    for machine_id in range(num_machines):
        events.append({
            'timestamp': start_time + np.random.randint(0, 3600),
            'machine_id': machine_id,
            'event_type': 0,  # ADD
            'cpu_capacity': np.random.uniform(0.5, 1.0),
            'memory_capacity': np.random.uniform(0.5, 1.0)
        })
    
    # Generate failures over time
    total_seconds = duration_hours * 3600
    num_failures = int(num_machines * failure_rate * (duration_hours / 24))
    
    for _ in range(num_failures):
        machine_id = np.random.randint(0, num_machines)
        failure_time = np.random.randint(3600, total_seconds)
        
        # REMOVE event (failure)
        events.append({
            'timestamp': failure_time,
            'machine_id': machine_id,
            'event_type': 1,  # REMOVE
            'cpu_capacity': 0,
            'memory_capacity': 0
        })
        
        # Maybe re-add after repair (80% chance)
        if np.random.random() < 0.8:
            repair_time = failure_time + np.random.randint(300, 7200)  # 5min to 2hr
            if repair_time < total_seconds:
                events.append({
                    'timestamp': repair_time,
                    'machine_id': machine_id,
                    'event_type': 0,  # ADD
                    'cpu_capacity': np.random.uniform(0.5, 1.0),
                    'memory_capacity': np.random.uniform(0.5, 1.0)
                })
    
    df = pd.DataFrame(events)
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"   Generated {len(df)} machine events")
    print(f"   Failures: {(df['event_type'] == 1).sum()}")
    
    return df


def generate_task_events(num_machines=100, num_tasks=10000, duration_hours=24*7):
    """
    Generate realistic task events based on Google Cluster statistics.
    
    Event types:
    - 0: SUBMIT
    - 1: SCHEDULE  
    - 2: EVICT
    - 3: FAIL
    - 4: FINISH
    - 5: KILL
    - 6: LOST
    """
    print("📊 Generating task events...")
    
    events = []
    total_seconds = duration_hours * 3600
    
    for task_id in range(num_tasks):
        machine_id = np.random.randint(0, num_machines)
        submit_time = np.random.randint(0, total_seconds - 3600)
        
        # SUBMIT
        events.append({
            'timestamp': submit_time,
            'job_id': task_id // 10,
            'task_index': task_id % 10,
            'machine_id': machine_id,
            'event_type': 0,  # SUBMIT
            'cpu_request': np.random.uniform(0.01, 0.5),
            'memory_request': np.random.uniform(0.01, 0.5)
        })
        
        # SCHEDULE (shortly after submit)
        schedule_time = submit_time + np.random.randint(1, 60)
        events.append({
            'timestamp': schedule_time,
            'job_id': task_id // 10,
            'task_index': task_id % 10,
            'machine_id': machine_id,
            'event_type': 1,  # SCHEDULE
            'cpu_request': np.random.uniform(0.01, 0.5),
            'memory_request': np.random.uniform(0.01, 0.5)
        })
        
        # Task outcome (based on Google statistics)
        outcome_roll = np.random.random()
        task_duration = np.random.exponential(300)  # Mean 5 minutes
        end_time = schedule_time + int(task_duration)
        
        if outcome_roll < 0.85:  # 85% FINISH successfully
            event_type = 4  # FINISH
        elif outcome_roll < 0.92:  # 7% FAIL
            event_type = 3  # FAIL
        elif outcome_roll < 0.97:  # 5% KILL
            event_type = 5  # KILL
        else:  # 3% LOST (machine failure)
            event_type = 6  # LOST
        
        if end_time < total_seconds:
            events.append({
                'timestamp': end_time,
                'job_id': task_id // 10,
                'task_index': task_id % 10,
                'machine_id': machine_id,
                'event_type': event_type,
                'cpu_request': np.random.uniform(0.01, 0.5),
                'memory_request': np.random.uniform(0.01, 0.5)
            })
    
    df = pd.DataFrame(events)
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"   Generated {len(df)} task events")
    event_counts = df['event_type'].value_counts().sort_index()
    event_names = {0: 'SUBMIT', 1: 'SCHEDULE', 2: 'EVICT', 3: 'FAIL', 4: 'FINISH', 5: 'KILL', 6: 'LOST'}
    for et, count in event_counts.items():
        print(f"   {event_names.get(et, et)}: {count}")
    
    return df


def generate_node_metrics(num_machines=100, duration_hours=24, sample_interval_sec=60):
    """
    Generate node health metrics time series.
    
    This simulates what your consensus nodes would report:
    - Heartbeat latency
    - Response rate
    - CPU/Memory usage
    - Message counts
    """
    print("📊 Generating node metrics...")
    
    records = []
    total_seconds = duration_hours * 3600
    num_samples = total_seconds // sample_interval_sec
    
    # Some machines will have failures
    failing_machines = np.random.choice(num_machines, size=int(num_machines * 0.1), replace=False)
    failure_times = {m: np.random.randint(total_seconds // 2, total_seconds) for m in failing_machines}
    
    for machine_id in range(num_machines):
        is_failing = machine_id in failing_machines
        fail_time = failure_times.get(machine_id, total_seconds + 1)
        
        for sample_idx in range(num_samples):
            timestamp = sample_idx * sample_interval_sec
            
            # Calculate time to failure
            time_to_failure = fail_time - timestamp
            
            # Generate metrics based on health state
            if time_to_failure < 0:
                # Already failed
                latency = 0
                response_rate = 0
                missed_heartbeats = 10
                cpu_usage = 0
                memory_usage = 0
                messages_dropped = 10
                state = 'FAILED'
            elif time_to_failure < 60:
                # Imminent failure (< 1 minute)
                intensity = 1 - (time_to_failure / 60)
                latency = 20 * (1 + intensity * 8)
                response_rate = max(0.1, 1.0 - intensity * 0.8)
                missed_heartbeats = int(intensity * 5)
                cpu_usage = min(1.0, 0.5 + intensity * 0.5)
                memory_usage = min(1.0, 0.5 + intensity * 0.4)
                messages_dropped = int(intensity * 8)
                state = 'DEGRADED_SEVERE'
            elif time_to_failure < 300:
                # Degrading (< 5 minutes)
                intensity = 1 - (time_to_failure / 300)
                latency = 20 * (1 + intensity * 4)
                response_rate = max(0.5, 1.0 - intensity * 0.4)
                missed_heartbeats = int(intensity * 2)
                cpu_usage = 0.5 + intensity * 0.3
                memory_usage = 0.5 + intensity * 0.2
                messages_dropped = int(intensity * 3)
                state = 'DEGRADED'
            elif time_to_failure < 1800:
                # Early warning (< 30 minutes)
                intensity = 1 - (time_to_failure / 1800)
                latency = 20 * (1 + intensity * 1.5)
                response_rate = max(0.8, 1.0 - intensity * 0.15)
                missed_heartbeats = 0
                cpu_usage = 0.4 + intensity * 0.15
                memory_usage = 0.4 + intensity * 0.1
                messages_dropped = int(intensity * 1)
                state = 'WARNING'
            else:
                # Healthy
                latency = max(5, np.random.normal(20, 5))
                response_rate = min(1.0, max(0.9, np.random.normal(0.98, 0.02)))
                missed_heartbeats = 0
                cpu_usage = np.random.uniform(0.2, 0.6)
                memory_usage = np.random.uniform(0.3, 0.6)
                messages_dropped = 0
                state = 'HEALTHY'
            
            records.append({
                'timestamp': timestamp,
                'machine_id': machine_id,
                'heartbeat_latency_ms': latency,
                'response_rate': response_rate,
                'missed_heartbeats': missed_heartbeats,
                'cpu_usage': cpu_usage,
                'memory_usage': memory_usage,
                'messages_sent': max(0, int(np.random.normal(10, 2))),
                'messages_received': max(0, int(np.random.normal(10, 2))) if response_rate > 0.5 else 0,
                'messages_dropped': messages_dropped,
                'state': state,
                'time_to_failure': max(0, time_to_failure) if is_failing else -1
            })
    
    df = pd.DataFrame(records)
    
    print(f"   Generated {len(df)} metric records")
    print(f"   Machines: {num_machines}")
    print(f"   Failing machines: {len(failing_machines)}")
    state_counts = df['state'].value_counts()
    for state, count in state_counts.items():
        print(f"   {state}: {count}")
    
    return df


def main():
    """Generate all test datasets."""
    print("="*60)
    print("GENERATING TEST DATASETS")
    print("="*60)
    
    datasets_dir = Path('datasets')
    datasets_dir.mkdir(exist_ok=True)
    
    # Generate machine events
    machine_df = generate_machine_events(num_machines=100, duration_hours=24*7)
    machine_df.to_csv(datasets_dir / 'machine_events.csv', index=False)
    
    # Generate task events
    task_df = generate_task_events(num_machines=100, num_tasks=10000)
    task_df.to_csv(datasets_dir / 'task_events.csv', index=False)
    
    # Generate node metrics (most important for your model)
    metrics_df = generate_node_metrics(num_machines=50, duration_hours=24)
    metrics_df.to_csv(datasets_dir / 'node_metrics.csv', index=False)
    
    # Save metadata
    metadata = {
        'generated_at': datetime.now().isoformat(),
        'description': 'Synthetic test data based on Google Cluster and LANL statistics',
        'files': {
            'machine_events.csv': 'Machine add/remove/failure events',
            'task_events.csv': 'Task lifecycle events',
            'node_metrics.csv': 'Node health metrics time series (best for testing)'
        },
        'statistics': {
            'num_machines': 100,
            'num_tasks': 10000,
            'failure_rate': 0.05,
            'duration_hours': 24 * 7
        }
    }
    
    with open(datasets_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "="*60)
    print("📁 Generated files:")
    print("="*60)
    
    for f in sorted(datasets_dir.iterdir()):
        if f.is_file():
            size = f.stat().st_size
            if size > 1024*1024:
                size_str = f"{size/1024/1024:.1f} MB"
            else:
                size_str = f"{size/1024:.1f} KB"
            print(f"  {f.name}: {size_str}")
    
    print("\n✅ Test datasets ready!")
    print("   Run: python3 test_on_google_cluster.py")


if __name__ == "__main__":
    main()