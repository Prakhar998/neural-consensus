# neural/advanced_features.py
"""
Advanced Feature Extraction with more discriminative features.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import deque
from scipy import stats
from simulation.node import NodeObservation


# Extended feature set - 32 features
ADVANCED_FEATURE_NAMES = [
    # Latency features (8)
    'latency_mean',
    'latency_std',
    'latency_min',
    'latency_max',
    'latency_range',
    'latency_skew',
    'latency_kurtosis',
    'latency_trend',
    
    # Jitter features (4)
    'jitter_mean',
    'jitter_std',
    'jitter_max',
    'jitter_trend',
    
    # Message features (6)
    'msg_rate_in',
    'msg_rate_out',
    'msg_drop_rate',
    'msg_out_of_order_rate',
    'msg_balance',  # ratio of in/out
    'msg_total',
    
    # Heartbeat features (5)
    'heartbeat_regularity',
    'heartbeat_missed_count',
    'heartbeat_missed_rate',
    'heartbeat_interval_std',
    'time_since_heartbeat',
    
    # Response features (4)
    'response_rate',
    'response_time_mean',
    'response_time_std',
    'response_time_max',
    
    # Raft-specific features (5)
    'term_freshness',
    'log_growth_rate',
    'commit_growth_rate',
    'is_leader',
    'election_activity',
]

NUM_ADVANCED_FEATURES = len(ADVANCED_FEATURE_NAMES)


@dataclass
class AdvancedObservationWindow:
    """Enhanced sliding window with statistics."""
    node_id: str
    window_size: int = 50
    observations: deque = field(default_factory=lambda: deque(maxlen=50))
    
    # Cached statistics
    _latency_history: deque = field(default_factory=lambda: deque(maxlen=100))
    _jitter_history: deque = field(default_factory=lambda: deque(maxlen=100))
    _heartbeat_times: deque = field(default_factory=lambda: deque(maxlen=50))
    
    def __post_init__(self):
        self.observations = deque(maxlen=self.window_size)
        self._latency_history = deque(maxlen=100)
        self._jitter_history = deque(maxlen=100)
        self._heartbeat_times = deque(maxlen=50)
    
    def add(self, obs: NodeObservation):
        self.observations.append(obs)
        self._latency_history.append(obs.heartbeat_latency_ms)
        self._jitter_history.append(obs.latency_jitter_ms)
        self._heartbeat_times.append(obs.timestamp_ms)
    
    def is_ready(self) -> bool:
        return len(self.observations) >= self.window_size
    
    def get_observations(self) -> List[NodeObservation]:
        return list(self.observations)
    
    def get_latency_stats(self) -> Dict[str, float]:
        if len(self._latency_history) < 2:
            return {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'skew': 0, 'kurtosis': 0}
        
        data = np.array(self._latency_history)
        return {
            'mean': np.mean(data),
            'std': np.std(data),
            'min': np.min(data),
            'max': np.max(data),
            'skew': stats.skew(data) if len(data) > 2 else 0,
            'kurtosis': stats.kurtosis(data) if len(data) > 3 else 0
        }
    
    def clear(self):
        self.observations.clear()
        self._latency_history.clear()
        self._jitter_history.clear()
        self._heartbeat_times.clear()


class AdvancedFeatureExtractor:
    """
    Advanced feature extraction with:
    - Statistical features (mean, std, skew, kurtosis)
    - Trend detection
    - Rate calculations
    - Normalized features
    """
    
    def __init__(
        self,
        window_size: int = 50,
        expected_heartbeat_ms: float = 150.0,
        normalize: bool = True
    ):
        self.window_size = window_size
        self.expected_heartbeat_ms = expected_heartbeat_ms
        self.normalize = normalize
        
        # Normalization parameters
        self._feature_means: Optional[np.ndarray] = None
        self._feature_stds: Optional[np.ndarray] = None
        self._n_samples = 0
    
    def extract_single(self, obs: NodeObservation, history: Optional[List[NodeObservation]] = None) -> np.ndarray:
        """Extract features from a single observation with optional history context."""
        features = np.zeros(NUM_ADVANCED_FEATURES, dtype=np.float32)
        
        # Basic latency features
        features[0] = obs.heartbeat_latency_ms / 100.0  # Normalized
        features[1] = obs.latency_jitter_ms / 50.0
        
        if history and len(history) > 1:
            latencies = [o.heartbeat_latency_ms for o in history[-20:]]
            features[2] = min(latencies) / 100.0
            features[3] = max(latencies) / 100.0
            features[4] = (max(latencies) - min(latencies)) / 100.0
            
            if len(latencies) > 3:
                features[5] = np.clip(stats.skew(latencies), -3, 3) / 3.0
                features[6] = np.clip(stats.kurtosis(latencies), -3, 10) / 10.0
            
            # Trend: linear regression slope
            if len(latencies) > 5:
                x = np.arange(len(latencies))
                slope, _, _, _, _ = stats.linregress(x, latencies)
                features[7] = np.tanh(slope / 10.0)
        
        # Jitter features
        features[8] = obs.latency_jitter_ms / 50.0
        if history and len(history) > 1:
            jitters = [o.latency_jitter_ms for o in history[-20:]]
            features[9] = np.std(jitters) / 30.0
            features[10] = max(jitters) / 100.0
            if len(jitters) > 5:
                x = np.arange(len(jitters))
                slope, _, _, _, _ = stats.linregress(x, jitters)
                features[11] = np.tanh(slope / 5.0)
        
        # Message features
        total_msg = obs.messages_sent + obs.messages_received + 1
        features[12] = min(1.0, obs.messages_received / 50.0)
        features[13] = min(1.0, obs.messages_sent / 50.0)
        features[14] = obs.messages_dropped / (total_msg + 1)
        features[15] = obs.out_of_order_count / (total_msg + 1)
        features[16] = obs.messages_received / (obs.messages_sent + 1)
        features[17] = min(1.0, total_msg / 100.0)
        
        # Heartbeat features
        expected = obs.heartbeat_interval_expected_ms
        actual = obs.heartbeat_interval_actual_ms
        if expected > 0:
            features[18] = 1.0 - min(1.0, abs(actual - expected) / expected)
        features[19] = min(1.0, obs.missed_heartbeats / 5.0)
        features[20] = obs.missed_heartbeats / 10.0
        
        if history and len(history) > 1:
            intervals = []
            for i in range(1, min(10, len(history))):
                dt = history[-i].timestamp_ms - history[-i-1].timestamp_ms
                if dt > 0:
                    intervals.append(dt)
            if intervals:
                features[21] = np.std(intervals) / 100.0
        
        features[22] = 0.0  # Time since heartbeat (computed in window)
        
        # Response features
        features[23] = obs.response_rate
        features[24] = min(1.0, obs.response_time_avg_ms / 100.0)
        features[25] = 0.0  # Response time std (computed in window)
        features[26] = min(1.0, obs.response_time_max_ms / 200.0)
        
        # Raft features
        features[27] = 0.0  # Term freshness (computed in window)
        features[28] = 0.0  # Log growth rate (computed in window)
        features[29] = 0.0  # Commit growth rate (computed in window)
        features[30] = 1.0 if obs.is_leader else 0.0
        features[31] = 0.0  # Election activity (computed in window)
        
        return features
    
    def extract_window(self, observations: List[NodeObservation]) -> np.ndarray:
        """
        Extract features from a window of observations.
        
        Returns:
            Array of shape (window_size, num_features)
        """
        actual_len = len(observations)
        
        if actual_len < self.window_size:
            # Pad with zeros at the beginning
            padding = self.window_size - actual_len
            features = np.zeros((self.window_size, NUM_ADVANCED_FEATURES), dtype=np.float32)
            start_idx = padding
        else:
            observations = observations[-self.window_size:]
            features = np.zeros((self.window_size, NUM_ADVANCED_FEATURES), dtype=np.float32)
            start_idx = 0
        
        # Extract per-observation features
        for i, obs in enumerate(observations):
            history = observations[:i+1] if i > 0 else None
            features[start_idx + i] = self.extract_single(obs, history)
        
        # Compute window-level features
        if len(observations) >= 2:
            # Term freshness
            terms = [o.term for o in observations]
            term_changes = sum(1 for i in range(1, len(terms)) if terms[i] > terms[i-1])
            term_freshness = term_changes / len(observations)
            
            # Log growth
            log_lengths = [o.log_length for o in observations]
            log_growth = (log_lengths[-1] - log_lengths[0]) / (len(observations) + 1)
            
            # Commit growth
            commits = [o.commit_index for o in observations]
            commit_growth = (commits[-1] - commits[0]) / (len(observations) + 1)
            
            # Time since last heartbeat (approximate)
            timestamps = [o.timestamp_ms for o in observations]
            time_gaps = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
            avg_gap = np.mean(time_gaps) if time_gaps else 0
            
            # Election activity (frequency of missed heartbeats leading to elections)
            missed = [o.missed_heartbeats for o in observations]
            election_activity = sum(1 for m in missed if m > 2) / len(missed)
            
            # Response time std across window
            response_times = [o.response_time_avg_ms for o in observations]
            response_std = np.std(response_times) if len(response_times) > 1 else 0
            
            # Update window-level features
            for i in range(start_idx, self.window_size):
                features[i, 22] = min(1.0, avg_gap / 500.0)
                features[i, 25] = min(1.0, response_std / 50.0)
                features[i, 27] = term_freshness
                features[i, 28] = np.tanh(log_growth)
                features[i, 29] = np.tanh(commit_growth)
                features[i, 31] = election_activity
        
        # Normalize if enabled
        if self.normalize and self._feature_means is not None:
            features = (features - self._feature_means) / (self._feature_stds + 1e-8)
        
        return features
    
    def fit(self, windows: List[np.ndarray]):
        """Compute normalization statistics from training data."""
        all_features = np.concatenate(windows, axis=0)
        self._feature_means = np.mean(all_features, axis=0)
        self._feature_stds = np.std(all_features, axis=0)
        self._feature_stds = np.where(self._feature_stds < 1e-6, 1.0, self._feature_stds)
        self._n_samples = len(all_features)
    
    def get_normalization_params(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._feature_means is None:
            return np.zeros(NUM_ADVANCED_FEATURES), np.ones(NUM_ADVANCED_FEATURES)
        return self._feature_means, self._feature_stds
    
    def set_normalization_params(self, means: np.ndarray, stds: np.ndarray):
        self._feature_means = means
        self._feature_stds = stds