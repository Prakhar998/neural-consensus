"""
Neural Failure Detector Module
Integrates neural network with Raft for failure detection.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
from loguru import logger

from .features import FeatureExtractor, ObservationWindow, NUM_FEATURES
from .encoder import LSTMAutoencoder
from .classifier import FailureClassifier, CombinedModel, FailureClass, FAILURE_NAMES


@dataclass
class DetectionResult:
    """Result of failure detection for a single node."""
    node_id: str
    timestamp_ms: int
    
    # Classification
    predicted_class: FailureClass
    class_name: str
    confidence: float
    class_probabilities: Dict[str, float]
    
    # Anomaly detection
    anomaly_score: float
    is_anomaly: bool
    
    # Decision
    is_failure: bool
    failure_type: Optional[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'node_id': self.node_id,
            'timestamp_ms': self.timestamp_ms,
            'predicted_class': self.predicted_class.value,
            'class_name': self.class_name,
            'confidence': self.confidence,
            'anomaly_score': self.anomaly_score,
            'is_anomaly': self.is_anomaly,
            'is_failure': self.is_failure,
            'failure_type': self.failure_type
        }


class NeuralFailureDetector:
    """
    Neural network-based failure detector.
    
    Replaces timeout-based detection with learned detection.
    Can predict failures before they happen and classify failure types.
    """
    
    def __init__(
        self,
        model: Optional[CombinedModel] = None,
        window_size: int = 20,
        feature_dim: int = NUM_FEATURES,
        anomaly_threshold: float = 0.5,
        confidence_threshold: float = 0.7,
        device: str = 'cpu'
    ):
        self.window_size = window_size
        self.feature_dim = feature_dim
        self.anomaly_threshold = anomaly_threshold
        self.confidence_threshold = confidence_threshold
        self.device = torch.device(device)
        
        # Model
        if model is not None:
            self.model = model.to(self.device)
        else:
            self.model = CombinedModel(
                input_size=feature_dim,
                hidden_size=64,
                latent_size=32,
                num_layers=2,
                seq_len=window_size,
                num_classes=len(FailureClass)
            ).to(self.device)
        
        self.model.eval()
        
        # Feature extraction
        self.feature_extractor = FeatureExtractor(window_size=window_size)
        
        # Observation windows per node
        self.windows: Dict[str, ObservationWindow] = defaultdict(
            lambda: ObservationWindow(node_id="", window_size=window_size)
        )
        
        # Detection history
        self.detection_history: Dict[str, List[DetectionResult]] = defaultdict(list)
        
        # Adaptive threshold (learned from healthy baseline)
        self._baseline_anomaly_scores: List[float] = []
        self._adaptive_threshold: Optional[float] = None
        
        # Statistics
        self.detections_made = 0
        self.failures_detected = 0
    
    def add_observation(self, observation) -> Optional[DetectionResult]:
        """
        Add new observation and optionally run detection.
        
        Args:
            observation: NodeObservation from simulation
            
        Returns:
            DetectionResult if window is full, None otherwise
        """
        node_id = observation.node_id
        
        # Initialize window if needed
        if node_id not in self.windows:
            self.windows[node_id] = ObservationWindow(node_id=node_id, window_size=self.window_size)
        
        self.windows[node_id].add(observation)
        
        # Run detection if window is ready
        if self.windows[node_id].is_ready():
            return self.detect(node_id, observation.timestamp_ms)
        
        return None
    
    def detect(self, node_id: str, timestamp_ms: int) -> DetectionResult:
        """
        Run failure detection for a specific node.
        
        Args:
            node_id: Node to check
            timestamp_ms: Current timestamp
            
        Returns:
            DetectionResult with classification and anomaly info
        """
        window = self.windows.get(node_id)
        if window is None or not window.is_ready():
            # Return healthy by default if not enough data
            return DetectionResult(
                node_id=node_id,
                timestamp_ms=timestamp_ms,
                predicted_class=FailureClass.HEALTHY,
                class_name="healthy",
                confidence=0.0,
                class_probabilities={},
                anomaly_score=0.0,
                is_anomaly=False,
                is_failure=False,
                failure_type=None
            )
        
        # Extract features
        observations = window.get_observations()
        features = self.feature_extractor.extract_window(observations)
        
        # Convert to tensor
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            result = self.model.predict_failure(x)
        
        # Extract results
        class_id = result['class_id'].item()
        predicted_class = FailureClass(class_id)
        confidence = result['confidence'].item()
        anomaly_score = result['anomaly_score'].item()
        probs = result['probabilities'].squeeze().cpu().numpy()
        
        # Build probability dict
        class_probs = {FAILURE_NAMES[FailureClass(i)]: float(probs[i]) for i in range(len(probs))}
        
        # Determine if anomaly
        threshold = self._adaptive_threshold or self.anomaly_threshold
        is_anomaly = anomaly_score > threshold
        
        # Determine if failure
        is_failure = (
            predicted_class != FailureClass.HEALTHY and 
            confidence >= self.confidence_threshold
        ) or is_anomaly
        
        failure_type = FAILURE_NAMES[predicted_class] if is_failure else None
        
        # Create result
        detection = DetectionResult(
            node_id=node_id,
            timestamp_ms=timestamp_ms,
            predicted_class=predicted_class,
            class_name=FAILURE_NAMES[predicted_class],
            confidence=confidence,
            class_probabilities=class_probs,
            anomaly_score=anomaly_score,
            is_anomaly=is_anomaly,
            is_failure=is_failure,
            failure_type=failure_type
        )
        
        # Update statistics
        self.detections_made += 1
        if is_failure:
            self.failures_detected += 1
        
        # Store in history
        self.detection_history[node_id].append(detection)
        
        return detection
    
    def detect_all(self, timestamp_ms: int) -> Dict[str, DetectionResult]:
        """Run detection for all nodes with ready windows."""
        results = {}
        for node_id, window in self.windows.items():
            if window.is_ready():
                results[node_id] = self.detect(node_id, timestamp_ms)
        return results
    
    def calibrate(self, healthy_observations: List[List]) -> float:
        """
        Calibrate anomaly threshold using healthy baseline data.
        
        Args:
            healthy_observations: List of observation windows from healthy operation
            
        Returns:
            Calibrated threshold
        """
        anomaly_scores = []
        
        for obs_window in healthy_observations:
            features = self.feature_extractor.extract_window(obs_window)
            x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                score = self.model.get_anomaly_score(x).item()
                anomaly_scores.append(score)
        
        # Set threshold at mean + 2*std (95th percentile assuming normal)
        mean_score = np.mean(anomaly_scores)
        std_score = np.std(anomaly_scores)
        self._adaptive_threshold = mean_score + 2 * std_score
        
        self._baseline_anomaly_scores = anomaly_scores
        
        logger.info(f"Calibrated anomaly threshold: {self._adaptive_threshold:.4f} "
                   f"(mean={mean_score:.4f}, std={std_score:.4f})")
        
        return self._adaptive_threshold
    
    def get_node_embedding(self, node_id: str) -> Optional[np.ndarray]:
        """Get current latent embedding for a node."""
        window = self.windows.get(node_id)
        if window is None or not window.is_ready():
            return None
        
        observations = window.get_observations()
        features = self.feature_extractor.extract_window(observations)
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            latent = self.model.encode(x)
        
        return latent.squeeze().cpu().numpy()
    
    def get_all_embeddings(self) -> Dict[str, np.ndarray]:
        """Get embeddings for all nodes with ready windows."""
        embeddings = {}
        for node_id in self.windows:
            emb = self.get_node_embedding(node_id)
            if emb is not None:
                embeddings[node_id] = emb
        return embeddings
    
    def clear_history(self, node_id: Optional[str] = None):
        """Clear detection history."""
        if node_id:
            self.detection_history[node_id].clear()
        else:
            self.detection_history.clear()
    
    def clear_windows(self, node_id: Optional[str] = None):
        """Clear observation windows."""
        if node_id:
            if node_id in self.windows:
                self.windows[node_id].clear()
        else:
            for window in self.windows.values():
                window.clear()
    
    def load_model(self, path: str):
        """Load model from file."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'feature_means' in checkpoint:
            self.feature_extractor.set_normalization_params(
                checkpoint['feature_means'],
                checkpoint['feature_stds']
            )
        
        if 'anomaly_threshold' in checkpoint:
            self._adaptive_threshold = checkpoint['anomaly_threshold']
        
        self.model.eval()
        logger.info(f"Loaded model from {path}")
    
    def save_model(self, path: str):
        """Save model to file."""
        means, stds = self.feature_extractor.get_normalization_params()
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'feature_means': means,
            'feature_stds': stds,
            'anomaly_threshold': self._adaptive_threshold,
            'window_size': self.window_size,
            'feature_dim': self.feature_dim
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Saved model to {path}")
    
    def stats(self) -> Dict[str, Any]:
        """Get detector statistics."""
        return {
            'detections_made': self.detections_made,
            'failures_detected': self.failures_detected,
            'nodes_tracked': len(self.windows),
            'anomaly_threshold': self._adaptive_threshold or self.anomaly_threshold,
            'confidence_threshold': self.confidence_threshold
        }


class HybridFailureDetector:
    """
    Combines neural detection with traditional timeout detection.
    
    Uses neural network as primary detector, falls back to timeout
    if neural network is uncertain.
    """
    
    def __init__(
        self,
        neural_detector: NeuralFailureDetector,
        timeout_ms: int = 500,
        neural_weight: float = 0.7
    ):
        self.neural = neural_detector
        self.timeout_ms = timeout_ms
        self.neural_weight = neural_weight
        
        # Last heartbeat times
        self.last_heartbeat: Dict[str, int] = {}
    
    def record_heartbeat(self, node_id: str, timestamp_ms: int):
        """Record heartbeat from a node."""
        self.last_heartbeat[node_id] = timestamp_ms
    
    def add_observation(self, observation) -> Optional[DetectionResult]:
        """Add observation and record heartbeat."""
        self.record_heartbeat(observation.node_id, observation.timestamp_ms)
        return self.neural.add_observation(observation)
    
    def is_failed(self, node_id: str, current_time_ms: int) -> Tuple[bool, str]:
        """
        Check if node has failed using hybrid approach.
        
        Returns:
            (is_failed, reason)
        """
        # Neural detection
        neural_result = self.neural.detect(node_id, current_time_ms)
        neural_failed = neural_result.is_failure
        neural_confidence = neural_result.confidence
        
        # Timeout detection
        last_hb = self.last_heartbeat.get(node_id, 0)
        time_since_hb = current_time_ms - last_hb
        timeout_failed = time_since_hb > self.timeout_ms
        
        # Combine decisions
        if neural_confidence >= 0.8:
            # High confidence neural prediction
            return neural_failed, f"neural:{neural_result.failure_type}"
        elif timeout_failed and neural_failed:
            # Both agree on failure
            return True, f"hybrid:{neural_result.failure_type}"
        elif timeout_failed and not neural_failed:
            # Timeout says failed, neural says healthy
            # Trust neural if confident, else timeout
            if neural_confidence >= 0.6:
                return False, "neural:healthy"
            else:
                return True, "timeout"
        elif not timeout_failed and neural_failed:
            # Neural predicts failure before timeout
            # This is the value-add of neural detection!
            if neural_confidence >= 0.7:
                return True, f"neural_early:{neural_result.failure_type}"
            else:
                return False, "uncertain"
        else:
            # Both say healthy
            return False, "healthy"