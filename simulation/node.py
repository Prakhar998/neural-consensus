"""
Base Node Module
Provides the base class for all nodes in the distributed system.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, TYPE_CHECKING
from enum import Enum
import random
from loguru import logger

from .clock import SimulationClock

if TYPE_CHECKING:
    from .network import Network, Message


class NodeState(Enum):
    ALIVE = "alive"
    CRASHED = "crashed"
    SLOW = "slow"
    BYZANTINE = "byzantine"
    PARTITIONED = "partitioned"


class FailureType(Enum):
    CRASH = "crash"
    CRASH_RECOVERY = "crash_recovery"
    SLOW = "slow"
    BYZANTINE_LYING = "byzantine_lying"
    BYZANTINE_SELECTIVE = "byzantine_selective"
    BYZANTINE_EQUIVOCATING = "byzantine_equivocating"
    OMISSION = "omission"


@dataclass
class NodeObservation:
    timestamp_ms: int
    node_id: str
    observer_id: str
    heartbeat_latency_ms: float = 0.0
    latency_jitter_ms: float = 0.0
    latency_trend: float = 0.0
    messages_sent: int = 0
    messages_received: int = 0
    messages_dropped: int = 0
    out_of_order_count: int = 0
    heartbeat_interval_actual_ms: float = 0.0
    heartbeat_interval_expected_ms: float = 150.0
    missed_heartbeats: int = 0
    response_rate: float = 1.0
    response_time_avg_ms: float = 0.0
    response_time_max_ms: float = 0.0
    term: int = 0
    log_length: int = 0
    commit_index: int = 0
    is_leader: bool = False
    vote_participation: float = 1.0
    label: Optional[str] = None
    time_to_failure_ms: Optional[int] = None


@dataclass
class NodeMetrics:
    messages_sent: int = 0
    messages_received: int = 0
    messages_dropped: int = 0
    heartbeats_sent: int = 0
    heartbeats_received: int = 0
    heartbeats_missed: int = 0
    elections_started: int = 0
    elections_won: int = 0
    votes_granted: int = 0
    last_heartbeat_sent_ms: int = 0
    last_heartbeat_received_ms: int = 0
    last_message_sent_ms: int = 0
    last_message_received_ms: int = 0
    recent_latencies: List[float] = field(default_factory=list)
    max_latency_window: int = 100


class BaseNode(ABC):
    def __init__(self, node_id: str, clock: SimulationClock, seed: Optional[int] = None):
        self.node_id = node_id
        self.clock = clock
        self.random = random.Random(seed)
        self.network: Optional[Network] = None
        self._state = NodeState.ALIVE
        self._is_alive = True
        self._failure_type: Optional[FailureType] = None
        self._failure_config: Dict[str, Any] = {}
        self._scheduled_recovery_time: Optional[int] = None
        self._slowdown_factor: float = 1.0
        self._byzantine_targets: List[str] = []
        self._omission_probability: float = 0.0
        self.metrics = NodeMetrics()
        self._observations: List[NodeObservation] = []
    
    @property
    def is_alive(self) -> bool:
        return self._is_alive and self._state not in [NodeState.CRASHED]
    
    @property
    def state(self) -> NodeState:
        return self._state
    
    @abstractmethod
    def receive_message(self, message: Message):
        pass
    
    @abstractmethod
    def tick(self):
        pass
    
    def inject_failure(self, failure_type: FailureType, duration_ms: Optional[int] = None, **config):
        logger.info(f"Injecting {failure_type.value} failure into node {self.node_id}")
        self._failure_type = failure_type
        self._failure_config = config
        
        if failure_type == FailureType.CRASH:
            self._state = NodeState.CRASHED
            self._is_alive = False
        elif failure_type == FailureType.CRASH_RECOVERY:
            self._state = NodeState.CRASHED
            self._is_alive = False
        elif failure_type == FailureType.SLOW:
            self._state = NodeState.SLOW
            self._slowdown_factor = config.get('slowdown_factor', 5.0)
        elif failure_type in [FailureType.BYZANTINE_LYING, FailureType.BYZANTINE_SELECTIVE, FailureType.BYZANTINE_EQUIVOCATING]:
            self._state = NodeState.BYZANTINE
            self._byzantine_targets = config.get('targets', [])
        elif failure_type == FailureType.OMISSION:
            self._omission_probability = config.get('probability', 0.5)
        
        if duration_ms is not None:
            self._scheduled_recovery_time = self.clock.now() + duration_ms
            self.clock.schedule(duration_ms, self._recover)
    
    def _recover(self):
        logger.info(f"Node {self.node_id} recovering from {self._failure_type}")
        self._state = NodeState.ALIVE
        self._is_alive = True
        self._failure_type = None
        self._failure_config = {}
        self._slowdown_factor = 1.0
        self._byzantine_targets = []
        self._omission_probability = 0.0
        self._scheduled_recovery_time = None
        self.on_recovery()
    
    def on_recovery(self):
        pass
    
    def crash(self, permanent: bool = True):
        self.inject_failure(FailureType.CRASH if permanent else FailureType.CRASH_RECOVERY)
    
    def slow_down(self, factor: float = 5.0, duration_ms: Optional[int] = None):
        self.inject_failure(FailureType.SLOW, duration_ms, slowdown_factor=factor)
    
    def go_byzantine(self, behavior: str = "lying", targets: Optional[List[str]] = None, duration_ms: Optional[int] = None):
        failure_map = {"lying": FailureType.BYZANTINE_LYING, "selective": FailureType.BYZANTINE_SELECTIVE, "equivocating": FailureType.BYZANTINE_EQUIVOCATING}
        self.inject_failure(failure_map.get(behavior, FailureType.BYZANTINE_LYING), duration_ms, targets=targets or [])
    
    def _should_drop_message(self, message: Message) -> bool:
        if not self.is_alive:
            return True
        if self._failure_type == FailureType.OMISSION and self.random.random() < self._omission_probability:
            self.metrics.messages_dropped += 1
            return True
        if self._failure_type == FailureType.BYZANTINE_SELECTIVE and message.sender_id not in self._byzantine_targets:
            return True
        return False
    
    def _get_processing_delay(self) -> int:
        return int(1 * self._slowdown_factor)
    
    def generate_observation(self, observer_id: str) -> NodeObservation:
        recent_latencies = self.metrics.recent_latencies[-20:]
        avg_latency = sum(recent_latencies) / len(recent_latencies) if recent_latencies else 0
        
        jitter = 0
        trend = 0
        if len(recent_latencies) >= 2:
            jitter = sum(abs(recent_latencies[i] - recent_latencies[i-1]) for i in range(1, len(recent_latencies))) / (len(recent_latencies) - 1)
            first_half = recent_latencies[:len(recent_latencies)//2]
            second_half = recent_latencies[len(recent_latencies)//2:]
            if first_half and second_half:
                trend = sum(second_half)/len(second_half) - sum(first_half)/len(first_half)
        
        obs = NodeObservation(
            timestamp_ms=self.clock.now(),
            node_id=self.node_id,
            observer_id=observer_id,
            heartbeat_latency_ms=avg_latency,
            latency_jitter_ms=jitter,
            latency_trend=trend,
            messages_sent=self.metrics.messages_sent,
            messages_received=self.metrics.messages_received,
            messages_dropped=self.metrics.messages_dropped,
            missed_heartbeats=self.metrics.heartbeats_missed,
        )
        
        if self._state == NodeState.CRASHED:
            obs.label = "crashed"
        elif self._state == NodeState.BYZANTINE:
            obs.label = "byzantine"
        elif self._state == NodeState.SLOW:
            obs.label = "slow"
        elif self._scheduled_recovery_time:
            obs.label = "pre_failure"
            obs.time_to_failure_ms = max(0, self._scheduled_recovery_time - self.clock.now())
        else:
            obs.label = "healthy"
        
        return obs
    
    def record_latency(self, latency_ms: float):
        self.metrics.recent_latencies.append(latency_ms)
        if len(self.metrics.recent_latencies) > self.metrics.max_latency_window:
            self.metrics.recent_latencies.pop(0)
    
    def send_message(self, message: Message) -> bool:
        if not self.is_alive or self.network is None:
            return False
        delay = self._get_processing_delay()
        if delay > 1:
            self.clock.schedule(delay, self._do_send, message)
            return True
        return self._do_send(message)
    
    def _do_send(self, message: Message) -> bool:
        message.sender_id = self.node_id
        self.metrics.messages_sent += 1
        self.metrics.last_message_sent_ms = self.clock.now()
        return self.network.send(message)
    
    def get_peer_ids(self) -> List[str]:
        if self.network is None:
            return []
        return [n for n in self.network.get_node_ids() if n != self.node_id]
    
    def reset_metrics(self):
        self.metrics = NodeMetrics()


class SimpleNode(BaseNode):
    def __init__(self, node_id: str, clock: SimulationClock):
        super().__init__(node_id, clock)
        self.received_messages: List[Message] = []
    
    def receive_message(self, message: Message):
        if self._should_drop_message(message):
            return
        self.metrics.messages_received += 1
        self.metrics.last_message_received_ms = self.clock.now()
        self.received_messages.append(message)
        if message.sent_time_ms > 0:
            self.record_latency(message.received_time_ms - message.sent_time_ms)
    
    def tick(self):
        pass