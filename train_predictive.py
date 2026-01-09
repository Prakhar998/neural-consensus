#!/usr/bin/env python3
"""
Paper 1: Predictive Failure Detection Training
ProactiveGuard: Deep Learning for Predictive Failure Detection in Distributed Consensus

SELF-CONTAINED VERSION - No external dependencies on other training scripts
"""

import sys
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from pathlib import Path
from loguru import logger
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss} | {level:<7} | {message}")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ============== CONFIGURATION ==============

NUM_FEATURES = 32
WINDOW_SIZE = 50

# Prediction horizons
PREDICTION_HORIZONS = {
    'healthy': 0,
    'degraded_30s': 1,
    'degraded_20s': 2,
    'degraded_10s': 3,
    'degraded_5s': 4,
    'failed_crash': 5,
    'failed_slow': 6,
    'failed_byzantine': 7,
    'failed_partition': 8,
}

HORIZON_NAMES = {v: k for k, v in PREDICTION_HORIZONS.items()}
NUM_PREDICTION_CLASSES = len(PREDICTION_HORIZONS)


# ============== SIMULATION IMPORTS ==============

try:
    from simulation import SimulationClock, ClockMode, Network, NetworkConfig, FailureInjector
    from simulation.node import NodeObservation
    from protocols.raft import RaftNode, RaftConfig
    SIMULATION_AVAILABLE = True
except ImportError:
    logger.warning("Simulation modules not found. Using synthetic data generation.")
    SIMULATION_AVAILABLE = False


# ============== SYNTHETIC OBSERVATION (Fallback) ==============

@dataclass
class SyntheticObservation:
    """Synthetic observation when simulation not available."""
    timestamp_ms: float = 0
    node_id: str = "node_0"
    observer_id: str = "observer"
    heartbeat_latency_ms: float = 20.0
    latency_jitter_ms: float = 5.0
    latency_trend: float = 0.0
    messages_sent: int = 10
    messages_received: int = 10
    messages_dropped: int = 0
    out_of_order_count: int = 0
    heartbeat_interval_actual_ms: float = 150.0
    heartbeat_interval_expected_ms: float = 150.0
    missed_heartbeats: int = 0
    response_rate: float = 1.0
    response_time_avg_ms: float = 15.0
    response_time_max_ms: float = 30.0
    term: int = 1
    log_length: int = 100
    commit_index: int = 99
    is_leader: bool = False
    vote_participation: float = 1.0
    label: str = "healthy"


def generate_synthetic_observation(base_params: Optional[Dict] = None) -> SyntheticObservation:
    """Generate a synthetic healthy observation."""
    obs = SyntheticObservation()
    
    if base_params:
        for key, value in base_params.items():
            if hasattr(obs, key):
                setattr(obs, key, value)
    
    # Add some random variation
    obs.heartbeat_latency_ms = max(1, np.random.normal(20, 5))
    obs.latency_jitter_ms = max(0, np.random.normal(5, 2))
    obs.response_time_avg_ms = max(1, np.random.normal(15, 3))
    obs.messages_sent = max(0, int(np.random.normal(10, 2)))
    obs.messages_received = max(0, int(np.random.normal(10, 2)))
    
    return obs


# ============== CLUSTER CREATION ==============

def create_cluster(num_nodes=5, seed=42):
    """Create a Raft cluster for simulation."""
    if not SIMULATION_AVAILABLE:
        return None, None, None, None
    
    clock = SimulationClock(mode=ClockMode.SIMULATED)
    network = Network(clock, NetworkConfig(
        min_latency_ms=5,
        max_latency_ms=50,
        message_loss_probability=0.01
    ), seed=seed)
    
    node_ids = [f"node_{i}" for i in range(num_nodes)]
    nodes = []
    
    for i, node_id in enumerate(node_ids):
        peer_ids = [n for n in node_ids if n != node_id]
        node = RaftNode(
            node_id=node_id,
            clock=clock,
            peer_ids=peer_ids,
            config=RaftConfig(heartbeat_interval_ms=150),
            seed=seed + i
        )
        network.register_node(node)
        nodes.append(node)
    
    injector = FailureInjector(network, clock, seed=seed)
    return clock, network, nodes, injector


# ============== OBSERVATION MODIFICATION ==============

def create_modified_observation(base_obs, modification_type: str, intensity: float = 1.0):
    """
    Create a modified observation to simulate different failure patterns.
    Works with both real NodeObservation and SyntheticObservation.
    """
    # Create a copy with modified values
    if SIMULATION_AVAILABLE and hasattr(base_obs, '__class__') and base_obs.__class__.__name__ == 'NodeObservation':
        obs = NodeObservation(
            timestamp_ms=base_obs.timestamp_ms,
            node_id=base_obs.node_id,
            observer_id=base_obs.observer_id,
            heartbeat_latency_ms=base_obs.heartbeat_latency_ms,
            latency_jitter_ms=base_obs.latency_jitter_ms,
            latency_trend=base_obs.latency_trend,
            messages_sent=base_obs.messages_sent,
            messages_received=base_obs.messages_received,
            messages_dropped=base_obs.messages_dropped,
            out_of_order_count=base_obs.out_of_order_count,
            heartbeat_interval_actual_ms=base_obs.heartbeat_interval_actual_ms,
            heartbeat_interval_expected_ms=base_obs.heartbeat_interval_expected_ms,
            missed_heartbeats=base_obs.missed_heartbeats,
            response_rate=base_obs.response_rate,
            response_time_avg_ms=base_obs.response_time_avg_ms,
            response_time_max_ms=base_obs.response_time_max_ms,
            term=base_obs.term,
            log_length=base_obs.log_length,
            commit_index=base_obs.commit_index,
            is_leader=base_obs.is_leader,
            vote_participation=base_obs.vote_participation,
            label=modification_type
        )
    else:
        obs = SyntheticObservation(
            timestamp_ms=base_obs.timestamp_ms,
            node_id=base_obs.node_id,
            observer_id=base_obs.observer_id,
            heartbeat_latency_ms=base_obs.heartbeat_latency_ms,
            latency_jitter_ms=base_obs.latency_jitter_ms,
            latency_trend=base_obs.latency_trend,
            messages_sent=base_obs.messages_sent,
            messages_received=base_obs.messages_received,
            messages_dropped=base_obs.messages_dropped,
            out_of_order_count=base_obs.out_of_order_count,
            heartbeat_interval_actual_ms=base_obs.heartbeat_interval_actual_ms,
            heartbeat_interval_expected_ms=base_obs.heartbeat_interval_expected_ms,
            missed_heartbeats=base_obs.missed_heartbeats,
            response_rate=base_obs.response_rate,
            response_time_avg_ms=base_obs.response_time_avg_ms,
            response_time_max_ms=base_obs.response_time_max_ms,
            term=base_obs.term,
            log_length=base_obs.log_length,
            commit_index=base_obs.commit_index,
            is_leader=base_obs.is_leader,
            vote_participation=base_obs.vote_participation,
            label=modification_type
        )
    
    if modification_type == 'slow':
        obs.heartbeat_latency_ms = base_obs.heartbeat_latency_ms * (3.0 + intensity * 5.0)
        obs.latency_jitter_ms = base_obs.latency_jitter_ms * (2.0 + intensity * 3.0)
        obs.response_time_avg_ms = base_obs.response_time_avg_ms * (3.0 + intensity * 4.0)
        obs.response_time_max_ms = base_obs.response_time_max_ms * (4.0 + intensity * 5.0)
        obs.response_rate = max(0.3, base_obs.response_rate - intensity * 0.4)
        obs.latency_trend = 5.0 + intensity * 10.0
        obs.missed_heartbeats = min(2, base_obs.missed_heartbeats + int(intensity))
        
    elif modification_type == 'crash':
        obs.heartbeat_latency_ms = 0
        obs.response_rate = 0
        obs.response_time_avg_ms = 0
        obs.messages_received = 0
        obs.missed_heartbeats = min(10, 3 + int(intensity * 5))
        obs.latency_jitter_ms = 0
        
    elif modification_type == 'byzantine':
        import random
        if random.random() > 0.5:
            obs.heartbeat_latency_ms = base_obs.heartbeat_latency_ms * 0.3
        else:
            obs.heartbeat_latency_ms = base_obs.heartbeat_latency_ms * 4.0
        obs.latency_jitter_ms = base_obs.latency_jitter_ms * (5.0 + intensity * 3.0)
        obs.out_of_order_count = base_obs.out_of_order_count + int(intensity * 5)
        obs.messages_dropped = base_obs.messages_dropped + int(intensity * 3)
        
    elif modification_type == 'partition':
        obs.messages_received = 0
        obs.messages_dropped = base_obs.messages_sent
        obs.response_rate = 0
        obs.missed_heartbeats = min(10, 5 + int(intensity * 3))
        obs.heartbeat_latency_ms = 0
    
    return obs


# ============== FEATURE EXTRACTION ==============

def extract_features(observations, window_size=50) -> np.ndarray:
    """Extract features from observations."""
    features = np.zeros((window_size, NUM_FEATURES), dtype=np.float32)
    
    n_obs = min(len(observations), window_size)
    start_idx = window_size - n_obs
    
    latencies = []
    response_rates = []
    
    for i, obs in enumerate(observations[-window_size:]):
        idx = start_idx + i
        
        latencies.append(obs.heartbeat_latency_ms)
        response_rates.append(obs.response_rate)
        
        # Latency features (0-7)
        features[idx, 0] = min(1.0, obs.heartbeat_latency_ms / 200.0)
        features[idx, 1] = min(1.0, obs.latency_jitter_ms / 100.0)
        features[idx, 2] = np.tanh(obs.latency_trend / 20.0)
        
        # Message features (8-13)
        total = obs.messages_sent + obs.messages_received + 1
        features[idx, 8] = min(1.0, obs.messages_received / 50.0)
        features[idx, 9] = min(1.0, obs.messages_sent / 50.0)
        features[idx, 10] = obs.messages_dropped / (total + 1)
        features[idx, 11] = obs.out_of_order_count / (total + 1)
        
        # Heartbeat features (14-18)
        features[idx, 14] = obs.response_rate
        features[idx, 15] = min(1.0, obs.missed_heartbeats / 5.0)
        features[idx, 16] = 1.0 if obs.missed_heartbeats > 2 else 0.0
        
        # Response features (19-22)
        features[idx, 19] = min(1.0, obs.response_time_avg_ms / 200.0)
        features[idx, 20] = min(1.0, obs.response_time_max_ms / 500.0)
        
        # Raft features (23-27)
        features[idx, 23] = 1.0 if obs.is_leader else 0.0
        features[idx, 24] = min(1.0, obs.term / 10.0)
        features[idx, 25] = min(1.0, obs.log_length / 100.0)
        features[idx, 26] = min(1.0, obs.commit_index / 100.0)
        
        # Slow-specific indicators (28-31)
        is_responding = obs.response_rate > 0.3
        is_high_latency = obs.heartbeat_latency_ms > 50
        features[idx, 28] = 1.0 if (is_responding and is_high_latency) else 0.0
        
        if obs.response_time_avg_ms > 0:
            features[idx, 29] = min(1.0, obs.heartbeat_latency_ms / (obs.response_time_avg_ms + 1))
        
        features[idx, 30] = 1.0 if obs.latency_trend > 5 else 0.0
    
    # Window-level stats
    if n_obs > 1:
        features[:, 3] = np.mean(latencies) / 200.0
        features[:, 4] = np.std(latencies) / 100.0 if len(latencies) > 1 else 0
        features[:, 5] = np.min(latencies) / 200.0
        features[:, 6] = np.max(latencies) / 200.0
        features[:, 7] = (np.max(latencies) - np.min(latencies)) / 200.0
        
        avg_latency = np.mean(latencies)
        avg_response = np.mean(response_rates)
        features[:, 31] = 1.0 if (avg_latency > 80 and avg_response > 0.3) else 0.0
        features[:, 12] = 1.0 if avg_response < 0.1 else 0.0
    
    return features


# ============== MODEL ARCHITECTURE ==============

class ResBlock(nn.Module):
    """Residual block for 1D convolution."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + residual)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block."""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _ = x.size()
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1)
        return x * w


class PredictiveModel(nn.Module):
    """
    Predictive failure detection model.
    
    Features:
    - ResNet-style convolutional blocks
    - Bidirectional LSTM for temporal patterns
    - Multi-head attention
    - Multiple output heads (classification, TTF regression, binary)
    """
    
    def __init__(self, input_size=32, hidden_size=128, latent_size=64, 
                 seq_len=50, num_classes=9):
        super().__init__()
        
        self.input_size = input_size
        self.seq_len = seq_len
        self.num_classes = num_classes
        
        # Feature extraction
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.conv_layers = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, 7, padding=3),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            ResBlock(hidden_size),
            SEBlock(hidden_size),
            ResBlock(hidden_size),
            SEBlock(hidden_size),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.lstm = nn.LSTM(
            hidden_size, hidden_size // 2,
            num_layers=2, batch_first=True,
            bidirectional=True, dropout=0.2
        )
        
        self.attention = nn.MultiheadAttention(
            hidden_size, 4, dropout=0.1, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(hidden_size)
        
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, latent_size),
            nn.Tanh()
        )
        
        # Classification head (prediction horizon)
        self.classifier = nn.Sequential(
            nn.Linear(latent_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
        # Regression head (time-to-failure)
        self.ttf_regressor = nn.Sequential(
            nn.Linear(latent_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Binary head (healthy vs pre-failure/failed)
        self.binary_classifier = nn.Sequential(
            nn.Linear(latent_size, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
        
        # Confidence estimation
        self.confidence_head = nn.Sequential(
            nn.Linear(latent_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Feature extraction
        x = self.input_proj(x)
        
        # CNN branch
        x_cnn = x.permute(0, 2, 1)
        x_cnn = self.conv_layers(x_cnn).squeeze(-1)
        
        # LSTM branch
        x_lstm, _ = self.lstm(x)
        x_lstm = x_lstm[:, -1, :]
        
        # Attention branch
        x_attn, _ = self.attention(x, x, x)
        x_attn = self.attn_norm(x + x_attn)
        x_attn = x_attn.mean(dim=1)
        
        # Fusion
        combined = torch.cat([x_cnn, x_lstm, x_attn], dim=1)
        latent = self.fusion(combined)
        
        # Outputs
        class_logits = self.classifier(latent)
        ttf_pred = self.ttf_regressor(latent)
        binary_logits = self.binary_classifier(latent)
        confidence = self.confidence_head(latent)
        
        return {
            'class_logits': class_logits,
            'ttf': ttf_pred,
            'binary_logits': binary_logits,
            'confidence': confidence,
            'latent': latent
        }
    
    def predict(self, x):
        outputs = self.forward(x)
        return torch.argmax(outputs['class_logits'], dim=1)
    
    def predict_with_ttf(self, x):
        outputs = self.forward(x)
        class_pred = torch.argmax(outputs['class_logits'], dim=1)
        ttf_pred = outputs['ttf'].squeeze(-1) * 30.0
        confidence = outputs['confidence'].squeeze(-1)
        return class_pred, ttf_pred, confidence


# ============== DEGRADATION PATTERNS ==============

def create_degraded_observation(base_obs, time_to_failure_steps, failure_type):
    """
    Create observation showing gradual degradation before failure.
    """
    # Calculate degradation intensity
    if time_to_failure_steps > 300:
        intensity = 0.0
    elif time_to_failure_steps > 200:
        intensity = 0.2
    elif time_to_failure_steps > 100:
        intensity = 0.4
    elif time_to_failure_steps > 50:
        intensity = 0.6
    elif time_to_failure_steps > 0:
        intensity = 0.8
    else:
        intensity = 1.0
    
    if intensity == 0.0:
        return base_obs, 'healthy'
    
    if intensity == 1.0:
        return create_modified_observation(base_obs, failure_type, 1.0), f'failed_{failure_type}'
    
    # Create degraded observation
    if SIMULATION_AVAILABLE:
        try:
            obs = NodeObservation(
                timestamp_ms=base_obs.timestamp_ms,
                node_id=base_obs.node_id,
                observer_id=base_obs.observer_id,
                heartbeat_latency_ms=base_obs.heartbeat_latency_ms,
                latency_jitter_ms=base_obs.latency_jitter_ms,
                latency_trend=base_obs.latency_trend,
                messages_sent=base_obs.messages_sent,
                messages_received=base_obs.messages_received,
                messages_dropped=base_obs.messages_dropped,
                out_of_order_count=base_obs.out_of_order_count,
                heartbeat_interval_actual_ms=base_obs.heartbeat_interval_actual_ms,
                heartbeat_interval_expected_ms=base_obs.heartbeat_interval_expected_ms,
                missed_heartbeats=base_obs.missed_heartbeats,
                response_rate=base_obs.response_rate,
                response_time_avg_ms=base_obs.response_time_avg_ms,
                response_time_max_ms=base_obs.response_time_max_ms,
                term=base_obs.term,
                log_length=base_obs.log_length,
                commit_index=base_obs.commit_index,
                is_leader=base_obs.is_leader,
                vote_participation=base_obs.vote_participation,
                label='degraded'
            )
        except:
            obs = SyntheticObservation(
                timestamp_ms=base_obs.timestamp_ms,
                node_id=base_obs.node_id,
                observer_id=getattr(base_obs, 'observer_id', 'observer'),
                heartbeat_latency_ms=base_obs.heartbeat_latency_ms,
                latency_jitter_ms=base_obs.latency_jitter_ms,
                latency_trend=getattr(base_obs, 'latency_trend', 0),
                messages_sent=base_obs.messages_sent,
                messages_received=base_obs.messages_received,
                messages_dropped=base_obs.messages_dropped,
                out_of_order_count=getattr(base_obs, 'out_of_order_count', 0),
                heartbeat_interval_actual_ms=getattr(base_obs, 'heartbeat_interval_actual_ms', 150),
                heartbeat_interval_expected_ms=getattr(base_obs, 'heartbeat_interval_expected_ms', 150),
                missed_heartbeats=base_obs.missed_heartbeats,
                response_rate=base_obs.response_rate,
                response_time_avg_ms=getattr(base_obs, 'response_time_avg_ms', 15),
                response_time_max_ms=getattr(base_obs, 'response_time_max_ms', 30),
                term=getattr(base_obs, 'term', 1),
                log_length=getattr(base_obs, 'log_length', 100),
                commit_index=getattr(base_obs, 'commit_index', 99),
                is_leader=getattr(base_obs, 'is_leader', False),
                vote_participation=getattr(base_obs, 'vote_participation', 1.0),
                label='degraded'
            )
    else:
        obs = SyntheticObservation()
        obs.timestamp_ms = base_obs.timestamp_ms
        obs.node_id = base_obs.node_id
        obs.heartbeat_latency_ms = base_obs.heartbeat_latency_ms
        obs.latency_jitter_ms = base_obs.latency_jitter_ms
        obs.messages_sent = base_obs.messages_sent
        obs.messages_received = base_obs.messages_received
        obs.messages_dropped = base_obs.messages_dropped
        obs.missed_heartbeats = base_obs.missed_heartbeats
        obs.response_rate = base_obs.response_rate
        obs.response_time_avg_ms = base_obs.response_time_avg_ms
        obs.label = 'degraded'
    
    # Apply degradation based on failure type
    if failure_type == 'crash':
        obs.heartbeat_latency_ms = base_obs.heartbeat_latency_ms * (1 + intensity * 3)
        obs.response_time_avg_ms = base_obs.response_time_avg_ms * (1 + intensity * 4)
        obs.response_rate = max(0.5, base_obs.response_rate - intensity * 0.4)
        obs.missed_heartbeats = int(intensity * 3)
        obs.latency_trend = intensity * 15
        
    elif failure_type == 'slow':
        obs.heartbeat_latency_ms = base_obs.heartbeat_latency_ms * (1 + intensity * 5)
        obs.latency_jitter_ms = base_obs.latency_jitter_ms * (1 + intensity * 4)
        obs.response_time_avg_ms = base_obs.response_time_avg_ms * (1 + intensity * 5)
        obs.response_rate = max(0.6, base_obs.response_rate - intensity * 0.2)
        obs.latency_trend = intensity * 20
        
    elif failure_type == 'byzantine':
        import random
        obs.latency_jitter_ms = base_obs.latency_jitter_ms * (1 + intensity * 6)
        obs.out_of_order_count = int(base_obs.out_of_order_count + intensity * 5)
        obs.messages_dropped = int(base_obs.messages_dropped + intensity * 3)
        if random.random() < intensity * 0.3:
            obs.heartbeat_latency_ms = base_obs.heartbeat_latency_ms * 0.2
        else:
            obs.heartbeat_latency_ms = base_obs.heartbeat_latency_ms * (1 + intensity * 2)
            
    elif failure_type == 'partition':
        obs.messages_dropped = int(base_obs.messages_dropped + intensity * 10)
        obs.response_rate = max(0.3, base_obs.response_rate - intensity * 0.5)
        obs.missed_heartbeats = int(intensity * 4)
        obs.latency_jitter_ms = base_obs.latency_jitter_ms * (1 + intensity * 3)
    
    # Determine label
    if time_to_failure_steps > 200:
        label = 'degraded_30s'
    elif time_to_failure_steps > 100:
        label = 'degraded_20s'
    elif time_to_failure_steps > 50:
        label = 'degraded_10s'
    else:
        label = 'degraded_5s'
    
    return obs, label


# ============== DATA GENERATION ==============

def generate_predictive_scenario(failure_type, failure_time_step=150, total_steps=250, seed=42):
    """Generate a complete scenario with pre-failure degradation."""
    np.random.seed(seed)
    
    if SIMULATION_AVAILABLE:
        clock, network, nodes, injector = create_cluster(seed=seed)
        
        if clock is not None:
            for node in nodes:
                node.start()
            clock.run_for(2000)
            target = nodes[1]
    
    observations = []
    features_list = []
    labels_list = []
    ttf_list = []
    
    timestamp = 0
    
    for step in range(total_steps):
        timestamp += 100
        
        # Generate base observation
        if SIMULATION_AVAILABLE and 'clock' in dir() and clock is not None:
            clock.run_for(100)
            for node in nodes:
                if node.is_alive:
                    node.tick()
            base_obs = target.generate_observation("collector")
        else:
            base_obs = generate_synthetic_observation()
            base_obs.timestamp_ms = timestamp
        
        # Calculate time to failure
        time_to_failure = failure_time_step - step
        
        # Generate observation with appropriate degradation
        obs, label_name = create_degraded_observation(base_obs, time_to_failure, failure_type)
        observations.append(obs)
        
        # Extract features
        if len(observations) >= WINDOW_SIZE:
            features = extract_features(observations[-WINDOW_SIZE:])
            features_list.append(features)
            
            label = PREDICTION_HORIZONS.get(label_name, PREDICTION_HORIZONS['healthy'])
            labels_list.append(label)
            
            ttf_normalized = max(0, time_to_failure) / 300.0
            ttf_list.append(ttf_normalized)
    
    if len(features_list) == 0:
        return np.array([]), np.array([]), np.array([])
    
    return np.array(features_list), np.array(labels_list), np.array(ttf_list)


def generate_predictive_training_data(samples_per_scenario=100, scenarios_per_type=5):
    """Generate balanced training data for predictive detection."""
    logger.info("="*60)
    logger.info("GENERATING PREDICTIVE TRAINING DATA")
    logger.info("="*60)
    
    all_features = []
    all_labels = []
    all_ttf = []
    
    failure_types = ['crash', 'slow', 'byzantine', 'partition']
    
    # Generate failure scenarios
    for failure_type in failure_types:
        logger.info(f"\nGenerating {failure_type} prediction scenarios...")
        
        for scenario in range(scenarios_per_type):
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
        
        logger.info(f"  Generated {scenarios_per_type} scenarios")
    
    # Generate healthy data
    logger.info("\nGenerating healthy scenarios...")
    for scenario in range(scenarios_per_type * 2):
        seed = 42 + scenario * 100
        
        if SIMULATION_AVAILABLE:
            clock, network, nodes, injector = create_cluster(seed=seed)
            if clock is not None:
                for node in nodes:
                    node.start()
                clock.run_for(2000)
                target = nodes[1]
        
        observations = []
        scenario_features = []
        
        for i in range(samples_per_scenario):
            if SIMULATION_AVAILABLE and 'clock' in dir() and clock is not None:
                clock.run_for(100)
                for node in nodes:
                    node.tick()
                obs = target.generate_observation("collector")
            else:
                obs = generate_synthetic_observation()
                obs.timestamp_ms = i * 100
            
            observations.append(obs)
            
            if len(observations) >= WINDOW_SIZE:
                features = extract_features(observations[-WINDOW_SIZE:])
                scenario_features.append(features)
        
        if len(scenario_features) > 0:
            all_features.append(np.array(scenario_features))
            all_labels.append(np.zeros(len(scenario_features), dtype=int))
            all_ttf.append(np.ones(len(scenario_features)))
    
    # Combine all data
    if len(all_features) == 0:
        logger.error("No data generated!")
        return np.array([]), np.array([]), np.array([])
    
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
        logger.info(f"  {name}: {c}")
    
    return X, y, ttf


# ============== TRAINING ==============

class PredictiveLoss(nn.Module):
    """Combined loss for predictive model."""
    
    def __init__(self, class_weights=None):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.mse_loss = nn.MSELoss()
        self.binary_ce = nn.CrossEntropyLoss()
    
    def forward(self, outputs, targets, ttf_targets, binary_targets):
        loss_class = self.ce_loss(outputs['class_logits'], targets)
        loss_ttf = self.mse_loss(outputs['ttf'].squeeze(-1), ttf_targets)
        loss_binary = self.binary_ce(outputs['binary_logits'], binary_targets)
        
        total_loss = loss_class + 0.5 * loss_ttf + 0.3 * loss_binary
        
        return total_loss, {
            'class': loss_class.item(),
            'ttf': loss_ttf.item(),
            'binary': loss_binary.item()
        }


def train_predictive_model(X, y, ttf, epochs=100, batch_size=64, lr=0.001):
    """Train the predictive failure detection model."""
    logger.info("="*60)
    logger.info("TRAINING PREDICTIVE MODEL")
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
    
    dataset = TensorDataset(X_tensor, y_tensor, ttf_tensor, binary_tensor)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    
    logger.info(f"Train: {train_size}, Val: {val_size}")
    
    # Class weights
    class_counts = np.bincount(y, minlength=NUM_PREDICTION_CLASSES)
    class_weights = torch.zeros(NUM_PREDICTION_CLASSES)
    for i in range(NUM_PREDICTION_CLASSES):
        if class_counts[i] > 0:
            class_weights[i] = len(y) / (NUM_PREDICTION_CLASSES * class_counts[i])
        else:
            class_weights[i] = 1.0
    
    # Loss and optimizer
    criterion = PredictiveLoss(class_weights=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
    
    best_val_acc = 0
    patience = 20
    no_improve = 0
    history = defaultdict(list)
    
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
            loss, _ = criterion(outputs, batch_y, batch_ttf, batch_binary)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            _, pred = outputs['class_logits'].max(1)
            train_total += batch_y.size(0)
            train_correct += pred.eq(batch_y).sum().item()
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_ttf_error = 0
        
        with torch.no_grad():
            for batch_x, batch_y, batch_ttf, batch_binary in val_loader:
                outputs = model(batch_x)
                
                _, pred = outputs['class_logits'].max(1)
                val_total += batch_y.size(0)
                val_correct += pred.eq(batch_y).sum().item()
                
                ttf_pred = outputs['ttf'].squeeze(-1)
                val_ttf_error += torch.abs(ttf_pred - batch_ttf).sum().item()
        
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        val_ttf_mae = val_ttf_error / val_total * 30
        
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['val_ttf_mae'].append(val_ttf_mae)
        
        if (epoch + 1) % 5 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs} - "
                       f"train_acc: {train_acc:.4f}, val_acc: {val_acc:.4f}, "
                       f"TTF MAE: {val_ttf_mae:.2f}s")
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'num_classes': NUM_PREDICTION_CLASSES,
                'prediction_horizons': PREDICTION_HORIZONS,
                'best_val_acc': best_val_acc
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
    logger.info("Model saved to models/predictive_model.pt")
    
    return model, dict(history)


# ============== EVALUATION ==============

def evaluate_predictive_model(model, X, y, ttf):
    """Comprehensive evaluation of predictive model."""
    logger.info("\n" + "="*60)
    logger.info("PREDICTIVE MODEL EVALUATION")
    logger.info("="*60)
    
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32)
    
    with torch.no_grad():
        outputs = model(X_tensor)
        predictions = torch.argmax(outputs['class_logits'], dim=1).numpy()
        ttf_predictions = outputs['ttf'].squeeze(-1).numpy() * 30
    
    results = {}
    
    # Overall accuracy
    accuracy = (predictions == y).mean()
    results['overall_accuracy'] = float(accuracy)
    logger.info(f"\nOverall accuracy: {accuracy:.4f}")
    
    # Binary accuracy
    binary_true = (y > 0).astype(int)
    binary_pred = (predictions > 0).astype(int)
    binary_acc = (binary_pred == binary_true).mean()
    results['binary_accuracy'] = float(binary_acc)
    logger.info(f"Binary accuracy (healthy vs warning): {binary_acc:.4f}")
    
    # TTF error
    ttf_true = ttf * 30
    ttf_mae = np.abs(ttf_predictions - ttf_true).mean()
    results['ttf_mae_seconds'] = float(ttf_mae)
    logger.info(f"TTF Mean Absolute Error: {ttf_mae:.2f} seconds")
    
    # Per-horizon accuracy
    logger.info(f"\n{'Horizon':<15} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Count':<8}")
    logger.info("-" * 53)
    
    results['per_horizon'] = {}
    
    for name, cls_id in PREDICTION_HORIZONS.items():
        mask = y == cls_id
        if mask.sum() == 0:
            continue
        
        tp = ((predictions == cls_id) & (y == cls_id)).sum()
        fp = ((predictions == cls_id) & (y != cls_id)).sum()
        fn = ((predictions != cls_id) & (y == cls_id)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        logger.info(f"{name:<15} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f} {mask.sum():<8}")
        
        results['per_horizon'][name] = {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'count': int(mask.sum())
        }
    
    # Early warning analysis
    logger.info("\n" + "-"*53)
    logger.info("EARLY WARNING ANALYSIS")
    logger.info("-"*53)
    
    pre_failure_classes = [1, 2, 3, 4]
    
    true_pre_failure = np.isin(y, pre_failure_classes)
    pred_pre_failure = np.isin(predictions, pre_failure_classes)
    
    early_warning_precision = (pred_pre_failure & true_pre_failure).sum() / (pred_pre_failure.sum() + 1e-10)
    early_warning_recall = (pred_pre_failure & true_pre_failure).sum() / (true_pre_failure.sum() + 1e-10)
    
    results['early_warning'] = {
        'precision': float(early_warning_precision),
        'recall': float(early_warning_recall)
    }
    
    logger.info(f"Early warning precision: {early_warning_precision:.4f}")
    logger.info(f"Early warning recall: {early_warning_recall:.4f}")
    
    return results


def plot_results(history, eval_results, output_dir='results/paper1'):
    """Generate plots for Paper 1."""
    try:
        import matplotlib.pyplot as plt
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        Path(f"{output_dir}/figures").mkdir(parents=True, exist_ok=True)
        
        # Training curves
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        axes[0].plot(history['train_acc'], label='Train')
        axes[0].plot(history['val_acc'], label='Validation')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Training Progress')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(history['val_ttf_mae'])
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE (seconds)')
        axes[1].set_title('Time-to-Failure Prediction Error')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/figures/training_curves.png", dpi=150)
        plt.close()
        
        # Per-horizon F1 scores
        if 'per_horizon' in eval_results:
            names = list(eval_results['per_horizon'].keys())
            f1_scores = [eval_results['per_horizon'][n]['f1'] for n in names]
            
            plt.figure(figsize=(10, 5))
            colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(names)))
            plt.barh(range(len(names)), f1_scores, color=colors)
            plt.yticks(range(len(names)), names)
            plt.xlabel('F1 Score')
            plt.title('F1 Score by Prediction Class')
            plt.xlim(0, 1.1)
            for i, f1 in enumerate(f1_scores):
                plt.text(f1 + 0.02, i, f'{f1:.2f}', va='center')
            plt.grid(True, alpha=0.3, axis='x')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/figures/per_class_f1.png", dpi=150)
            plt.close()
        
        logger.info(f"\nPlots saved to {output_dir}/figures/")
        
    except ImportError:
        logger.warning("matplotlib not available, skipping plots")


# ============== MAIN ==============

def main():
    parser = argparse.ArgumentParser(description='Train Predictive Failure Detector')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--samples', type=int, default=100, help='Samples per scenario')
    parser.add_argument('--scenarios', type=int, default=5, help='Scenarios per failure type')
    parser.add_argument('--output', type=str, default='results/paper1', help='Output directory')
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("PAPER 1: PREDICTIVE FAILURE DETECTION")
    logger.info("ProactiveGuard Training Pipeline")
    logger.info("="*60)
    
    if not SIMULATION_AVAILABLE:
        logger.warning("Using synthetic data (simulation modules not found)")
    
    # Generate training data
    X, y, ttf = generate_predictive_training_data(
        samples_per_scenario=args.samples,
        scenarios_per_type=args.scenarios
    )
    
    if len(X) == 0:
        logger.error("No training data generated!")
        return
    
    # Train model
    model, history = train_predictive_model(
        X, y, ttf,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
    
    # Evaluate
    eval_results = evaluate_predictive_model(model, X, y, ttf)
    
    # Save results
    Path(args.output).mkdir(parents=True, exist_ok=True)
    
    import json
    with open(f"{args.output}/evaluation_results.json", 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    # Plot results
    plot_results(history, eval_results, args.output)
    
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETE!")
    logger.info(f"Model saved to: models/predictive_model.pt")
    logger.info(f"Results saved to: {args.output}/")
    logger.info("="*60)


if __name__ == "__main__":
    main()