"""
Data Collection Module
Collects observations from simulation for training neural detector.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path
import json
from loguru import logger

from simulation.node import NodeObservation
from simulation.clock import SimulationClock
from neural.features import FeatureExtractor, ObservationWindow, NUM_FEATURES


@dataclass
class ObservationBuffer:
    """Buffer for collecting observations per node."""
    node_id: str
    observations: List[NodeObservation] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)
    timestamps: List[int] = field(default_factory=list)
    
    def add(self, obs: NodeObservation, label: str):
        self.observations.append(obs)
        self.labels.append(label)
        self.timestamps.append(obs.timestamp_ms)
    
    def __len__(self) -> int:
        return len(self.observations)
    
    def clear(self):
        self.observations.clear()
        self.labels.clear()
        self.timestamps.clear()


class DataCollector:
    """
    Collects training data from simulation.
    
    Generates labeled windows of observations for neural network training.
    """
    
    def __init__(
        self,
        clock: SimulationClock,
        window_size: int = 20,
        collection_interval_ms: int = 100,
        feature_extractor: Optional[FeatureExtractor] = None
    ):
        self.clock = clock
        self.window_size = window_size
        self.collection_interval_ms = collection_interval_ms
        self.feature_extractor = feature_extractor or FeatureExtractor(window_size=window_size)
        
        # Buffers per node
        self.buffers: Dict[str, ObservationBuffer] = defaultdict(
            lambda: ObservationBuffer(node_id="")
        )
        
        # Collected windows
        self.windows: List[np.ndarray] = []
        self.window_labels: List[int] = []
        self.window_metadata: List[Dict[str, Any]] = []
        
        # Label mapping
        self.label_to_idx = {
            'healthy': 0,
            'pre_failure': 1,
            'crashed': 2,
            'byzantine': 3,
            'partitioned': 4,
            'slow': 5
        }
        
        # Statistics
        self.observations_collected = 0
        self.windows_generated = 0
        
        # Scheduling
        self._collection_cancel = None
        self._is_collecting = False
    
    def start_collection(self):
        """Start automatic collection."""
        if self._is_collecting:
            return
        
        self._is_collecting = True
        logger.info("Started data collection")
    
    def stop_collection(self):
        """Stop automatic collection."""
        self._is_collecting = False
        if self._collection_cancel:
            self._collection_cancel()
            self._collection_cancel = None
        logger.info(f"Stopped data collection. Collected {self.windows_generated} windows")
    
    def collect_observation(self, obs: NodeObservation, label: str):
        """
        Collect a single observation.
        
        Args:
            obs: NodeObservation from simulation
            label: Label for this observation
        """
        if not self._is_collecting:
            return
        
        node_id = obs.node_id
        
        # Initialize buffer if needed
        if node_id not in self.buffers:
            self.buffers[node_id] = ObservationBuffer(node_id=node_id)
        
        self.buffers[node_id].add(obs, label)
        self.observations_collected += 1
        
        # Generate window if we have enough observations
        if len(self.buffers[node_id]) >= self.window_size:
            self._generate_window(node_id)
    
    def _generate_window(self, node_id: str):
        """Generate a training window from buffer."""
        buffer = self.buffers[node_id]
        
        if len(buffer) < self.window_size:
            return
        
        # Get last window_size observations
        observations = buffer.observations[-self.window_size:]
        labels = buffer.labels[-self.window_size:]
        
        # Extract features
        features = self.feature_extractor.extract_window(observations)
        
        # Determine window label (use most recent non-healthy, or healthy)
        window_label = 'healthy'
        for label in reversed(labels):
            if label != 'healthy':
                window_label = label
                break
        
        label_idx = self.label_to_idx.get(window_label, 0)
        
        # Store
        self.windows.append(features)
        self.window_labels.append(label_idx)
        self.window_metadata.append({
            'node_id': node_id,
            'timestamp': observations[-1].timestamp_ms,
            'label': window_label
        })
        
        self.windows_generated += 1
        
        # Keep buffer manageable (sliding window)
        if len(buffer) > self.window_size * 2:
            buffer.observations = buffer.observations[-self.window_size:]
            buffer.labels = buffer.labels[-self.window_size:]
            buffer.timestamps = buffer.timestamps[-self.window_size:]
    
    def collect_from_nodes(self, nodes: List, observer_id: str = "collector"):
        """
        Collect observations from a list of nodes.
        
        Args:
            nodes: List of BaseNode or RaftNode
            observer_id: ID of the observer
        """
        for node in nodes:
            obs = node.generate_observation(observer_id)
            label = obs.label or 'healthy'
            self.collect_observation(obs, label)
    
    def get_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get collected data as numpy arrays.
        
        Returns:
            features: Array of shape (n_samples, window_size, n_features)
            labels: Array of shape (n_samples,)
        """
        if not self.windows:
            return np.array([]), np.array([])
        
        features = np.stack(self.windows, axis=0)
        labels = np.array(self.window_labels)
        
        return features, labels
    
    def get_balanced_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get class-balanced dataset (undersample majority class).
        
        Returns:
            features, labels with balanced classes
        """
        features, labels = self.get_dataset()
        
        if len(features) == 0:
            return features, labels
        
        # Find minimum class count
        unique, counts = np.unique(labels, return_counts=True)
        min_count = min(counts)
        
        # Sample equally from each class
        balanced_indices = []
        for class_idx in unique:
            class_indices = np.where(labels == class_idx)[0]
            sampled = np.random.choice(class_indices, min_count, replace=False)
            balanced_indices.extend(sampled)
        
        np.random.shuffle(balanced_indices)
        
        return features[balanced_indices], labels[balanced_indices]
    
    def save_dataset(self, path: str):
        """Save collected dataset to file."""
        features, labels = self.get_dataset()
        
        np.savez(
            path,
            features=features,
            labels=labels,
            metadata=self.window_metadata,
            label_mapping=self.label_to_idx
        )
        
        logger.info(f"Saved dataset to {path}: {len(features)} samples")
    
    def load_dataset(self, path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load dataset from file."""
        data = np.load(path, allow_pickle=True)
        features = data['features']
        labels = data['labels']
        
        logger.info(f"Loaded dataset from {path}: {len(features)} samples")
        return features, labels
    
    def clear(self):
        """Clear all collected data."""
        self.buffers.clear()
        self.windows.clear()
        self.window_labels.clear()
        self.window_metadata.clear()
        self.observations_collected = 0
        self.windows_generated = 0
    
    def stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        _, labels = self.get_dataset()
        
        label_counts = {}
        if len(labels) > 0:
            unique, counts = np.unique(labels, return_counts=True)
            idx_to_label = {v: k for k, v in self.label_to_idx.items()}
            label_counts = {idx_to_label[int(u)]: int(c) for u, c in zip(unique, counts)}
        
        return {
            'observations_collected': self.observations_collected,
            'windows_generated': self.windows_generated,
            'nodes_tracked': len(self.buffers),
            'label_distribution': label_counts
        }