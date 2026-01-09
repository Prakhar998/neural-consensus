"""
Feature Extraction Module
Converts raw node observations into features for neural network.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import deque

from simulation.node import NodeObservation


# Feature indices for reference
FEATURE_NAMES = [
    'latency_mean',
    'latency_std',
    'latency_trend',
    'jitter',
    'message_rate',
    'drop_rate',
    'heartbeat_regularity',
    'missed_heartbeats',
    'response_rate',
    'response_time_norm',
    'term_freshness',
    'log_progress',
    'commit_progress',
    'is_leader',
    'time_since_last_msg',
    'health_score'
]

NUM_FEATURES = len(FEATURE_NAMES)


@dataclass
class ObservationWindow:
    """Sliding window of observations for a single node."""
    node_id: str
    window_size: int = 20
    observations: deque = field(default_factory=lambda: deque(maxlen=20))
    
    def __post_init__(self):
        self.observations = deque(maxlen=self.window_size)
    
    def add(self, obs: NodeObservation):
        self.observations.append(obs)
    
    def is_ready(self) -> bool:
        return len(self.observations) >= self.window_size
    
    def get_observations(self) -> List[NodeObservation]:
        return list(self.observations)
    
    def clear(self):
        self.observations.clear()
    
    def __len__(self) -> int:
        return len(self.observations)


class FeatureExtractor:
    """
    Extracts normalized features from node observations.
    
    Converts raw observations into a fixed-size feature vector
    suitable for neural network input.
    """
    
    def __init__(
        self,
        window_size: int = 20,
        expected_heartbeat_ms: float = 150.0,
        latency_scale: float = 100.0,
        normalize: bool = True
    ):
        self.window_size = window_size
        self.expected_heartbeat_ms = expected_heartbeat_ms
        self.latency_scale = latency_scale
        self.normalize = normalize
        
        # Running statistics for normalization
        self._feature_means: Optional[np.ndarray] = None
        self._feature_stds: Optional[np.ndarray] = None
        self._n_samples = 0
    
    def extract_single(self, obs: NodeObservation) -> np.ndarray:
        """Extract features from a single observation."""
        features = np.zeros(NUM_FEATURES, dtype=np.float32)
        
        # Latency features
        features[0] = obs.heartbeat_latency_ms / self.latency_scale
        features[1] = obs.latency_jitter_ms / self.latency_scale
        features[2] = np.tanh(obs.latency_trend / 10.0)  # Normalized trend
        features[3] = obs.latency_jitter_ms / (obs.heartbeat_latency_ms + 1e-6)
        
        # Message features
        total_msgs = obs.messages_sent + obs.messages_received + 1
        features[4] = min(1.0, total_msgs / 100.0)  # Message rate
        features[5] = obs.messages_dropped / (total_msgs + 1e-6)  # Drop rate
        
        # Heartbeat features
        expected = obs.heartbeat_interval_expected_ms
        actual = obs.heartbeat_interval_actual_ms
        features[6] = 1.0 - min(1.0, abs(actual - expected) / expected) if expected > 0 else 0.0
        features[7] = min(1.0, obs.missed_heartbeats / 5.0)
        
        # Response features
        features[8] = obs.response_rate
        features[9] = min(1.0, obs.response_time_avg_ms / self.latency_scale)
        
        # Raft-specific features
        features[10] = 0.0  # Term freshness (computed in window)
        features[11] = 0.0  # Log progress (computed in window)
        features[12] = 0.0  # Commit progress (computed in window)
        features[13] = 1.0 if obs.is_leader else 0.0
        
        # Timing
        features[14] = 0.0  # Time since last message (computed in window)
        
        # Composite health score
        features[15] = self._compute_health_score(obs)
        
        return features
    
    def extract_window(self, observations: List[NodeObservation]) -> np.ndarray:
        """
        Extract features from a window of observations.
        
        Returns:
            Array of shape (window_size, num_features)
        """
        if len(observations) < self.window_size:
            # Pad with zeros if not enough observations
            padding = self.window_size - len(observations)
            features = np.zeros((self.window_size, NUM_FEATURES), dtype=np.float32)
            start_idx = padding
        else:
            observations = observations[-self.window_size:]
            features = np.zeros((self.window_size, NUM_FEATURES), dtype=np.float32)
            start_idx = 0
        
        # Extract features for each observation
        for i, obs in enumerate(observations):
            features[start_idx + i] = self.extract_single(obs)
        
        # Compute window-level features
        if len(observations) >= 2:
            # Term freshness (is term increasing?)
            terms = [o.term for o in observations]
            term_changes = sum(1 for i in range(1, len(terms)) if terms[i] > terms[i-1])
            term_freshness = term_changes / len(observations)
            
            # Log progress
            log_lengths = [o.log_length for o in observations]
            log_growth = (log_lengths[-1] - log_lengths[0]) / (len(observations) + 1)
            
            # Commit progress
            commits = [o.commit_index for o in observations]
            commit_growth = (commits[-1] - commits[0]) / (len(observations) + 1)
            
            # Time since last message
            timestamps = [o.timestamp_ms for o in observations]
            time_gaps = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
            avg_gap = np.mean(time_gaps) if time_gaps else 0
            
            # Update window-level features for all rows
            for i in range(start_idx, self.window_size):
                features[i, 10] = term_freshness
                features[i, 11] = np.tanh(log_growth)
                features[i, 12] = np.tanh(commit_growth)
                features[i, 14] = min(1.0, avg_gap / 1000.0)
        
        if self.normalize and self._feature_means is not None:
            features = (features - self._feature_means) / (self._feature_stds + 1e-8)
        
        return features
    
    def _compute_health_score(self, obs: NodeObservation) -> float:
        """Compute a composite health score from 0 (unhealthy) to 1 (healthy)."""
        scores = []
        
        # Low latency is good
        latency_score = max(0, 1.0 - obs.heartbeat_latency_ms / (self.latency_scale * 2))
        scores.append(latency_score)
        
        # Low jitter is good
        jitter_score = max(0, 1.0 - obs.latency_jitter_ms / self.latency_scale)
        scores.append(jitter_score)
        
        # High response rate is good
        scores.append(obs.response_rate)
        
        # No missed heartbeats is good
        hb_score = max(0, 1.0 - obs.missed_heartbeats / 5.0)
        scores.append(hb_score)
        
        # No drops is good
        total = obs.messages_sent + obs.messages_received + 1
        drop_score = 1.0 - obs.messages_dropped / total
        scores.append(drop_score)
        
        return np.mean(scores)
    
    def fit(self, windows: List[np.ndarray]):
        """Compute normalization statistics from training data."""
        all_features = np.concatenate(windows, axis=0)
        self._feature_means = np.mean(all_features, axis=0)
        self._feature_stds = np.std(all_features, axis=0)
        self._n_samples = len(all_features)
    
    def update_stats(self, features: np.ndarray):
        """Online update of normalization statistics."""
        if self._feature_means is None:
            self._feature_means = np.mean(features, axis=0)
            self._feature_stds = np.std(features, axis=0)
            self._n_samples = len(features)
        else:
            # Welford's online algorithm
            n = len(features)
            new_n = self._n_samples + n
            
            new_mean = (self._feature_means * self._n_samples + np.sum(features, axis=0)) / new_n
            
            # Update std (approximate)
            new_var = ((self._feature_stds ** 2) * self._n_samples + np.var(features, axis=0) * n) / new_n
            self._feature_stds = np.sqrt(new_var)
            
            self._feature_means = new_mean
            self._n_samples = new_n
    
    def get_normalization_params(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get normalization parameters."""
        return self._feature_means, self._feature_stds
    
    def set_normalization_params(self, means: np.ndarray, stds: np.ndarray):
        """Set normalization parameters (for transfer learning)."""
        self._feature_means = means
        self._feature_stds = stds


class EnvironmentNormalizer:
    """
    Normalizes features to be environment-agnostic.
    Used for transfer learning across different deployments.
    """
    
    def __init__(self):
        self.baseline_means: Optional[np.ndarray] = None
        self.baseline_stds: Optional[np.ndarray] = None
        self.fitted = False
    
    def fit(self, healthy_observations: List[np.ndarray]):
        """
        Fit normalizer on healthy baseline data.
        
        Args:
            healthy_observations: List of feature arrays from healthy operation
        """
        all_features = np.concatenate(healthy_observations, axis=0)
        self.baseline_means = np.mean(all_features, axis=0)
        self.baseline_stds = np.std(all_features, axis=0)
        self.baseline_stds = np.where(self.baseline_stds < 1e-6, 1.0, self.baseline_stds)
        self.fitted = True
    
    def transform(self, features: np.ndarray) -> np.ndarray:
        """
        Transform features to z-scores relative to healthy baseline.
        
        After transformation, 0 = normal, positive = above normal, negative = below normal.
        """
        if not self.fitted:
            return features
        
        return (features - self.baseline_means) / self.baseline_stds
    
    def fit_transform(self, healthy_observations: List[np.ndarray]) -> List[np.ndarray]:
        """Fit and transform in one step."""
        self.fit(healthy_observations)
        return [self.transform(obs) for obs in healthy_observations]