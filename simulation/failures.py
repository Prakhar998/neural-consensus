"""
Failure Injection Module
Provides strategies for injecting various types of failures.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Any, TYPE_CHECKING
from enum import Enum
import random
from loguru import logger

from .clock import SimulationClock
from .node import BaseNode, FailureType, NodeState

if TYPE_CHECKING:
    from .network import Network


@dataclass
class FailurePattern:
    name: str
    description: str
    start_time_ms: int = 0
    failure_type: FailureType = FailureType.CRASH
    duration_ms: Optional[int] = None
    target_nodes: List[str] = field(default_factory=list)
    num_targets: int = 1
    config: Dict[str, Any] = field(default_factory=dict)
    symptom_start_ms: int = 5000
    symptom_config: Dict[str, Any] = field(default_factory=dict)


class FailureInjector:
    def __init__(self, network: Network, clock: SimulationClock, seed: Optional[int] = None):
        self.network = network
        self.clock = clock
        self.random = random.Random(seed)
        self._active_failures: Dict[str, FailureType] = {}
        self._scheduled_failures: List[FailurePattern] = []
        self._failure_callbacks: List[Callable[[str, FailureType], None]] = []
        self._recovery_callbacks: List[Callable[[str], None]] = []
        self.failures_injected = 0
        self.recoveries = 0
    
    def inject_crash(self, node_id: str, recovery_time_ms: Optional[int] = None) -> bool:
        node = self.network.get_node(node_id)
        if node is None:
            return False
        
        node.inject_failure(
            FailureType.CRASH if recovery_time_ms is None else FailureType.CRASH_RECOVERY,
            duration_ms=recovery_time_ms
        )
        
        self._active_failures[node_id] = FailureType.CRASH
        self.failures_injected += 1
        self._notify_failure(node_id, FailureType.CRASH)
        
        if recovery_time_ms:
            self.clock.schedule(recovery_time_ms, self._handle_recovery, node_id)
        
        logger.info(f"Crashed node {node_id}" + (f" (recovery in {recovery_time_ms}ms)" if recovery_time_ms else " (permanent)"))
        return True
    
    def inject_slowdown(self, node_id: str, slowdown_factor: float = 5.0, duration_ms: Optional[int] = None) -> bool:
        node = self.network.get_node(node_id)
        if node is None:
            return False
        
        node.inject_failure(FailureType.SLOW, duration_ms=duration_ms, slowdown_factor=slowdown_factor)
        self._active_failures[node_id] = FailureType.SLOW
        self.failures_injected += 1
        
        if duration_ms:
            self.clock.schedule(duration_ms, self._handle_recovery, node_id)
        
        return True
    
    def inject_byzantine(self, node_id: str, behavior: str = "lying", targets: Optional[List[str]] = None, duration_ms: Optional[int] = None) -> bool:
        node = self.network.get_node(node_id)
        if node is None:
            return False
        
        failure_type = {
            "lying": FailureType.BYZANTINE_LYING,
            "selective": FailureType.BYZANTINE_SELECTIVE,
            "equivocating": FailureType.BYZANTINE_EQUIVOCATING
        }.get(behavior, FailureType.BYZANTINE_LYING)
        
        node.inject_failure(failure_type, duration_ms=duration_ms, targets=targets or [])
        self._active_failures[node_id] = failure_type
        self.failures_injected += 1
        
        if duration_ms:
            self.clock.schedule(duration_ms, self._handle_recovery, node_id)
        
        return True
    
    def inject_partition(self, group_a: List[str], group_b: List[str], duration_ms: Optional[int] = None):
        self.network.create_partition(group_a, group_b)
        if duration_ms:
            self.clock.schedule(duration_ms, lambda: self.network.heal_partition(group_a, group_b))
    
    def inject_omission(self, node_id: str, probability: float = 0.5, duration_ms: Optional[int] = None) -> bool:
        node = self.network.get_node(node_id)
        if node is None:
            return False
        
        node.inject_failure(FailureType.OMISSION, duration_ms=duration_ms, probability=probability)
        self._active_failures[node_id] = FailureType.OMISSION
        self.failures_injected += 1
        
        if duration_ms:
            self.clock.schedule(duration_ms, self._handle_recovery, node_id)
        
        return True
    
    def inject_random_failure(self, failure_type: Optional[FailureType] = None, exclude_nodes: Optional[List[str]] = None, duration_ms: Optional[int] = None) -> Optional[str]:
        healthy_nodes = [
            node_id for node_id in self.network.get_node_ids()
            if node_id not in self._active_failures and node_id not in (exclude_nodes or [])
        ]
        
        if not healthy_nodes:
            return None
        
        target = self.random.choice(healthy_nodes)
        failure = failure_type or self.random.choice([FailureType.CRASH, FailureType.SLOW, FailureType.BYZANTINE_LYING])
        
        if failure == FailureType.CRASH:
            self.inject_crash(target, duration_ms)
        elif failure == FailureType.SLOW:
            self.inject_slowdown(target, duration_ms=duration_ms)
        else:
            self.inject_byzantine(target, duration_ms=duration_ms)
        
        return target
    
    def recover_node(self, node_id: str) -> bool:
        node = self.network.get_node(node_id)
        if node is None or node_id not in self._active_failures:
            return False
        
        node._recover()
        self._handle_recovery(node_id)
        return True
    
    def recover_all(self):
        for node_id in list(self._active_failures.keys()):
            self.recover_node(node_id)
        self.network.heal_partition()
    
    def _handle_recovery(self, node_id: str):
        if node_id in self._active_failures:
            del self._active_failures[node_id]
            self.recoveries += 1
            self._notify_recovery(node_id)
            logger.info(f"Node {node_id} recovered")
    
    def on_failure(self, callback: Callable[[str, FailureType], None]):
        self._failure_callbacks.append(callback)
    
    def on_recovery(self, callback: Callable[[str], None]):
        self._recovery_callbacks.append(callback)
    
    def _notify_failure(self, node_id: str, failure_type: FailureType):
        for cb in self._failure_callbacks:
            try:
                cb(node_id, failure_type)
            except Exception as e:
                logger.error(f"Failure callback error: {e}")
    
    def _notify_recovery(self, node_id: str):
        for cb in self._recovery_callbacks:
            try:
                cb(node_id)
            except Exception as e:
                logger.error(f"Recovery callback error: {e}")
    
    def get_active_failures(self) -> Dict[str, FailureType]:
        return dict(self._active_failures)
    
    def get_healthy_nodes(self) -> List[str]:
        return [node_id for node_id in self.network.get_node_ids() if node_id not in self._active_failures]
    
    def stats(self) -> Dict[str, Any]:
        return {
            "failures_injected": self.failures_injected,
            "recoveries": self.recoveries,
            "active_failures": len(self._active_failures),
            "healthy_nodes": len(self.get_healthy_nodes())
        }


TRAINING_PATTERNS = {
    "gradual_slowdown": FailurePattern(
        name="gradual_slowdown",
        description="Node gradually slows down before crashing",
        failure_type=FailureType.SLOW,
        symptom_start_ms=10000,
        config={"slowdown_factor": 10.0}
    ),
    "sudden_crash": FailurePattern(
        name="sudden_crash",
        description="Node crashes with no warning",
        failure_type=FailureType.CRASH,
        symptom_start_ms=0
    ),
    "byzantine_equivocation": FailurePattern(
        name="byzantine_equivocation",
        description="Node starts sending conflicting messages",
        failure_type=FailureType.BYZANTINE_EQUIVOCATING,
        symptom_start_ms=0
    ),
}