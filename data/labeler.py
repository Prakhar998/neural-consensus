"""
Auto-Labeling Module
Automatically labels observations based on node state and injected failures.
"""

from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum
from loguru import logger

from simulation.node import BaseNode, NodeState, FailureType, NodeObservation
from simulation.failures import FailureInjector


class LabelStrategy(Enum):
    """Labeling strategies."""
    GROUND_TRUTH = "ground_truth"      # Use actual node state
    PRE_FAILURE_WINDOW = "pre_failure" # Label observations before failure as pre_failure
    SYMPTOM_BASED = "symptom_based"    # Label based on observed symptoms


@dataclass
class LabelConfig:
    """Configuration for auto-labeling."""
    strategy: LabelStrategy = LabelStrategy.PRE_FAILURE_WINDOW
    pre_failure_window_ms: int = 5000  # Label as pre_failure this long before crash
    symptom_thresholds: Dict[str, float] = None
    
    def __post_init__(self):
        if self.symptom_thresholds is None:
            self.symptom_thresholds = {
                'latency_ratio': 2.0,      # 2x normal latency
                'jitter_ratio': 3.0,       # 3x normal jitter
                'drop_rate': 0.1,          # 10% drop rate
                'missed_heartbeats': 3     # 3 missed heartbeats
            }


class AutoLabeler:
    """
    Automatically labels observations for training.
    
    Uses knowledge of injected failures and node state
    to generate ground truth labels.
    """
    
    def __init__(
        self,
        config: Optional[LabelConfig] = None,
        failure_injector: Optional[FailureInjector] = None
    ):
        self.config = config or LabelConfig()
        self.injector = failure_injector
        
        # Track scheduled failures for pre_failure labeling
        self.scheduled_failures: Dict[str, int] = {}  # node_id -> failure_time_ms
        
        # Baseline stats for symptom-based labeling
        self.baseline_stats: Dict[str, Dict[str, float]] = {}
        
        # Label mapping
        self.state_to_label = {
            NodeState.ALIVE: 'healthy',
            NodeState.CRASHED: 'crashed',
            NodeState.SLOW: 'slow',
            NodeState.BYZANTINE: 'byzantine',
            NodeState.PARTITIONED: 'partitioned'
        }
        
        self.failure_to_label = {
            FailureType.CRASH: 'crashed',
            FailureType.CRASH_RECOVERY: 'crashed',
            FailureType.SLOW: 'slow',
            FailureType.BYZANTINE_LYING: 'byzantine',
            FailureType.BYZANTINE_SELECTIVE: 'byzantine',
            FailureType.BYZANTINE_EQUIVOCATING: 'byzantine',
            FailureType.OMISSION: 'partitioned'
        }
    
    def schedule_failure(self, node_id: str, failure_time_ms: int, failure_type: FailureType):
        """Register a scheduled failure for pre_failure labeling."""
        self.scheduled_failures[node_id] = failure_time_ms
        logger.debug(f"Scheduled failure for {node_id} at t={failure_time_ms}ms")
    
    def clear_scheduled(self, node_id: Optional[str] = None):
        """Clear scheduled failures."""
        if node_id:
            self.scheduled_failures.pop(node_id, None)
        else:
            self.scheduled_failures.clear()
    
    def set_baseline(self, node_id: str, stats: Dict[str, float]):
        """Set baseline statistics for symptom-based labeling."""
        self.baseline_stats[node_id] = stats
    
    def compute_baseline(self, observations: List[NodeObservation]):
        """Compute baseline from healthy observations."""
        if not observations:
            return
        
        node_id = observations[0].node_id
        
        latencies = [o.heartbeat_latency_ms for o in observations]
        jitters = [o.latency_jitter_ms for o in observations]
        
        self.baseline_stats[node_id] = {
            'mean_latency': sum(latencies) / len(latencies),
            'mean_jitter': sum(jitters) / len(jitters),
            'max_latency': max(latencies),
            'max_jitter': max(jitters)
        }
    
    def label(self, node: BaseNode, current_time_ms: int) -> str:
        """
        Generate label for current node state.
        
        Args:
            node: The node to label
            current_time_ms: Current simulation time
            
        Returns:
            Label string
        """
        if self.config.strategy == LabelStrategy.GROUND_TRUTH:
            return self._label_ground_truth(node)
        
        elif self.config.strategy == LabelStrategy.PRE_FAILURE_WINDOW:
            return self._label_with_pre_failure(node, current_time_ms)
        
        elif self.config.strategy == LabelStrategy.SYMPTOM_BASED:
            obs = node.generate_observation("labeler")
            return self._label_by_symptoms(obs)
        
        return 'healthy'
    
    def _label_ground_truth(self, node: BaseNode) -> str:
        """Label based on actual node state."""
        return self.state_to_label.get(node.state, 'healthy')
    
    def _label_with_pre_failure(self, node: BaseNode, current_time_ms: int) -> str:
        """Label with pre_failure detection."""
        node_id = node.node_id
        
        # Check if node is already failed
        if node.state != NodeState.ALIVE:
            return self.state_to_label.get(node.state, 'healthy')
        
        # Check for scheduled failure
        if node_id in self.scheduled_failures:
            failure_time = self.scheduled_failures[node_id]
            time_to_failure = failure_time - current_time_ms
            
            if 0 < time_to_failure <= self.config.pre_failure_window_ms:
                return 'pre_failure'
        
        # Check injector if available
        if self.injector:
            active = self.injector.get_active_failures()
            if node_id in active:
                return self.failure_to_label.get(active[node_id], 'healthy')
        
        return 'healthy'
    
    def _label_by_symptoms(self, obs: NodeObservation) -> str:
        """Label based on observed symptoms."""
        node_id = obs.node_id
        baseline = self.baseline_stats.get(node_id)
        thresholds = self.config.symptom_thresholds
        
        if baseline is None:
            # No baseline, use absolute thresholds
            if obs.missed_heartbeats >= thresholds['missed_heartbeats']:
                return 'crashed'
            if obs.heartbeat_latency_ms > 500:
                return 'slow'
            return obs.label or 'healthy'
        
        # Compare to baseline
        latency_ratio = obs.heartbeat_latency_ms / (baseline['mean_latency'] + 1e-6)
        jitter_ratio = obs.latency_jitter_ms / (baseline['mean_jitter'] + 1e-6)
        
        # Detect slow
        if latency_ratio > thresholds['latency_ratio']:
            return 'slow'
        
        # Detect pre_failure (high jitter often precedes crash)
        if jitter_ratio > thresholds['jitter_ratio']:
            return 'pre_failure'
        
        # Detect crash
        if obs.missed_heartbeats >= thresholds['missed_heartbeats']:
            return 'crashed'
        
        # Detect partition (messages dropped)
        total_msgs = obs.messages_sent + obs.messages_received + 1
        drop_rate = obs.messages_dropped / total_msgs
        if drop_rate > thresholds['drop_rate']:
            return 'partitioned'
        
        return 'healthy'
    
    def label_observation(self, obs: NodeObservation, current_time_ms: int) -> str:
        """
        Label a single observation.
        
        Convenience method when you don't have the node object.
        """
        # Use existing label if present
        if obs.label and obs.label != 'healthy':
            return obs.label
        
        # Check scheduled failures
        if obs.node_id in self.scheduled_failures:
            failure_time = self.scheduled_failures[obs.node_id]
            time_to_failure = failure_time - current_time_ms
            
            if 0 < time_to_failure <= self.config.pre_failure_window_ms:
                return 'pre_failure'
            elif time_to_failure <= 0:
                return 'crashed'
        
        # Fall back to symptom-based
        if self.config.strategy == LabelStrategy.SYMPTOM_BASED:
            return self._label_by_symptoms(obs)
        
        return obs.label or 'healthy'


class ScenarioLabeler:
    """
    Labels entire scenarios for batch data generation.
    """
    
    def __init__(self, auto_labeler: AutoLabeler):
        self.labeler = auto_labeler
    
    def label_scenario(
        self,
        observations: List[NodeObservation],
        failure_times: Dict[str, int],
        failure_types: Dict[str, str]
    ) -> List[str]:
        """
        Label a complete scenario.
        
        Args:
            observations: List of observations in time order
            failure_times: Dict of node_id -> failure timestamp
            failure_types: Dict of node_id -> failure type
            
        Returns:
            List of labels for each observation
        """
        labels = []
        pre_failure_window = self.labeler.config.pre_failure_window_ms
        
        for obs in observations:
            node_id = obs.node_id
            timestamp = obs.timestamp_ms
            
            if node_id in failure_times:
                failure_time = failure_times[node_id]
                failure_type = failure_types.get(node_id, 'crashed')
                
                if timestamp >= failure_time:
                    labels.append(failure_type)
                elif failure_time - timestamp <= pre_failure_window:
                    labels.append('pre_failure')
                else:
                    labels.append('healthy')
            else:
                labels.append('healthy')
        
        return labels