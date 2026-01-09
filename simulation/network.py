"""
Network Simulation Module
Simulates a network with configurable delays, message loss, and partitions.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Callable, Any, TYPE_CHECKING
from enum import Enum
import random
import uuid
from loguru import logger

from .clock import SimulationClock

if TYPE_CHECKING:
    from .node import BaseNode


class MessageType(Enum):
    VOTE_REQUEST = "vote_request"
    VOTE_RESPONSE = "vote_response"
    APPEND_ENTRIES = "append_entries"
    APPEND_ENTRIES_RESPONSE = "append_entries_response"
    HEARTBEAT = "heartbeat"
    HEARTBEAT_RESPONSE = "heartbeat_response"
    CLIENT_REQUEST = "client_request"
    CLIENT_RESPONSE = "client_response"
    CUSTOM = "custom"


@dataclass
class Message:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    type: MessageType = MessageType.CUSTOM
    sender_id: str = ""
    receiver_id: str = ""
    term: int = 0
    payload: Dict[str, Any] = field(default_factory=dict)
    sent_time_ms: int = 0
    received_time_ms: int = 0
    
    def __repr__(self) -> str:
        return f"Message({self.type.value}: {self.sender_id}→{self.receiver_id}, term={self.term})"


@dataclass 
class NetworkStats:
    messages_sent: int = 0
    messages_delivered: int = 0
    messages_dropped: int = 0
    messages_delayed: int = 0
    total_latency_ms: int = 0
    
    @property
    def delivery_rate(self) -> float:
        return self.messages_delivered / self.messages_sent if self.messages_sent else 1.0
    
    @property
    def average_latency_ms(self) -> float:
        return self.total_latency_ms / self.messages_delivered if self.messages_delivered else 0.0


class NetworkCondition(Enum):
    PERFECT = "perfect"
    NORMAL = "normal"
    DEGRADED = "degraded"
    UNRELIABLE = "unreliable"
    PARTITIONED = "partitioned"


@dataclass
class NetworkConfig:
    min_latency_ms: int = 1
    max_latency_ms: int = 10
    latency_std_ms: int = 2
    message_loss_probability: float = 0.0
    message_duplicate_probability: float = 0.0
    out_of_order_probability: float = 0.0
    max_out_of_order_delay_ms: int = 50
    
    @classmethod
    def from_condition(cls, condition: NetworkCondition) -> NetworkConfig:
        configs = {
            NetworkCondition.PERFECT: cls(min_latency_ms=0, max_latency_ms=1, message_loss_probability=0.0),
            NetworkCondition.NORMAL: cls(min_latency_ms=5, max_latency_ms=50, latency_std_ms=10, message_loss_probability=0.01),
            NetworkCondition.DEGRADED: cls(min_latency_ms=20, max_latency_ms=200, latency_std_ms=50, message_loss_probability=0.05, out_of_order_probability=0.1),
            NetworkCondition.UNRELIABLE: cls(min_latency_ms=50, max_latency_ms=500, latency_std_ms=100, message_loss_probability=0.15, out_of_order_probability=0.2, message_duplicate_probability=0.05),
        }
        return configs.get(condition, cls())


class Network:
    def __init__(self, clock: SimulationClock, config: Optional[NetworkConfig] = None, seed: Optional[int] = None):
        self.clock = clock
        self.config = config or NetworkConfig()
        self.random = random.Random(seed)
        self._nodes: Dict[str, BaseNode] = {}
        self._partitions: Set[frozenset] = set()
        self._link_configs: Dict[frozenset, NetworkConfig] = {}
        self._in_flight_messages: List[Message] = []
        self.stats = NetworkStats()
        self._message_handlers: List[Callable[[Message], None]] = []
    
    def register_node(self, node: BaseNode):
        if node.node_id in self._nodes:
            logger.warning(f"Node {node.node_id} already registered, replacing")
        self._nodes[node.node_id] = node
        node.network = self
    
    def unregister_node(self, node_id: str):
        if node_id in self._nodes:
            del self._nodes[node_id]
    
    def get_node(self, node_id: str) -> Optional[BaseNode]:
        return self._nodes.get(node_id)
    
    def get_all_nodes(self) -> List[BaseNode]:
        return list(self._nodes.values())
    
    def get_node_ids(self) -> List[str]:
        return list(self._nodes.keys())
    
    def send(self, message: Message) -> bool:
        message.sent_time_ms = self.clock.now()
        self.stats.messages_sent += 1
        
        if message.sender_id not in self._nodes or message.receiver_id not in self._nodes:
            return False
        
        if self._is_partitioned(message.sender_id, message.receiver_id):
            self.stats.messages_dropped += 1
            return True
        
        config = self._get_link_config(message.sender_id, message.receiver_id)
        
        if self.random.random() < config.message_loss_probability:
            self.stats.messages_dropped += 1
            return True
        
        delay_ms = self._calculate_latency(config)
        
        if self.random.random() < config.out_of_order_probability:
            delay_ms += self.random.randint(0, config.max_out_of_order_delay_ms)
        
        self._in_flight_messages.append(message)
        self.clock.schedule(delay_ms, self._deliver_message, message)
        
        if self.random.random() < config.message_duplicate_probability:
            dup_delay = delay_ms + self.random.randint(1, 50)
            self.clock.schedule(dup_delay, self._deliver_message, message)
        
        return True
    
    def _deliver_message(self, message: Message):
        if message in self._in_flight_messages:
            self._in_flight_messages.remove(message)
        
        if self._is_partitioned(message.sender_id, message.receiver_id):
            self.stats.messages_dropped += 1
            return
        
        receiver = self._nodes.get(message.receiver_id)
        if receiver is None or not receiver.is_alive:
            self.stats.messages_dropped += 1
            return
        
        message.received_time_ms = self.clock.now()
        latency = message.received_time_ms - message.sent_time_ms
        
        self.stats.messages_delivered += 1
        self.stats.total_latency_ms += latency
        
        for handler in self._message_handlers:
            try:
                handler(message)
            except Exception as e:
                logger.error(f"Message handler error: {e}")
        
        try:
            receiver.receive_message(message)
        except Exception as e:
            logger.error(f"Error delivering message to {message.receiver_id}: {e}")
    
    def _calculate_latency(self, config: NetworkConfig) -> int:
        mean = (config.min_latency_ms + config.max_latency_ms) / 2
        latency = self.random.gauss(mean, config.latency_std_ms)
        return int(max(config.min_latency_ms, min(config.max_latency_ms, latency)))
    
    def _get_link_config(self, node_a: str, node_b: str) -> NetworkConfig:
        link = frozenset([node_a, node_b])
        return self._link_configs.get(link, self.config)
    
    def set_link_config(self, node_a: str, node_b: str, config: NetworkConfig):
        self._link_configs[frozenset([node_a, node_b])] = config
    
    def create_partition(self, group_a: List[str], group_b: List[str]):
        for node_a in group_a:
            for node_b in group_b:
                self._partitions.add(frozenset([node_a, node_b]))
        logger.info(f"Created partition: {group_a} <-> {group_b}")
    
    def heal_partition(self, group_a: Optional[List[str]] = None, group_b: Optional[List[str]] = None):
        if group_a is None and group_b is None:
            self._partitions.clear()
        else:
            for node_a in (group_a or []):
                for node_b in (group_b or []):
                    self._partitions.discard(frozenset([node_a, node_b]))
    
    def isolate_node(self, node_id: str):
        other_nodes = [n for n in self._nodes.keys() if n != node_id]
        self.create_partition([node_id], other_nodes)
    
    def reconnect_node(self, node_id: str):
        to_remove = [p for p in self._partitions if node_id in p]
        for p in to_remove:
            self._partitions.discard(p)
    
    def _is_partitioned(self, node_a: str, node_b: str) -> bool:
        return frozenset([node_a, node_b]) in self._partitions
    
    def add_message_handler(self, handler: Callable[[Message], None]):
        self._message_handlers.append(handler)
    
    def remove_message_handler(self, handler: Callable[[Message], None]):
        if handler in self._message_handlers:
            self._message_handlers.remove(handler)
    
    def set_condition(self, condition: NetworkCondition):
        self.config = NetworkConfig.from_condition(condition)
    
    def reset_stats(self):
        self.stats = NetworkStats()
    
    def in_flight_count(self) -> int:
        return len(self._in_flight_messages)


class MessageCollector:
    def __init__(self, network: Network):
        self.network = network
        self.messages: List[Message] = []
        self._active = False
    
    def start(self):
        if not self._active:
            self.network.add_message_handler(self._handle_message)
            self._active = True
    
    def stop(self):
        if self._active:
            self.network.remove_message_handler(self._handle_message)
            self._active = False
    
    def _handle_message(self, message: Message):
        self.messages.append(message)
    
    def clear(self):
        self.messages.clear()
    
    def get_messages(self, sender_id: Optional[str] = None, receiver_id: Optional[str] = None, message_type: Optional[MessageType] = None) -> List[Message]:
        result = self.messages
        if sender_id:
            result = [m for m in result if m.sender_id == sender_id]
        if receiver_id:
            result = [m for m in result if m.receiver_id == receiver_id]
        if message_type:
            result = [m for m in result if m.type == message_type]
        return result