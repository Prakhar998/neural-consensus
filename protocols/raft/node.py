"""
Raft Node Implementation
Based on "In Search of an Understandable Consensus Algorithm" (Ongaro & Ousterhout, 2014)
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Callable, TYPE_CHECKING
import random
from loguru import logger

from simulation.clock import SimulationClock
from simulation.node import BaseNode
from simulation.network import Message, MessageType

from .state import RaftState, RaftRole
from .messages import (
    LogEntry, RequestVoteRequest, RequestVoteResponse,
    AppendEntriesRequest, AppendEntriesResponse,
    ClientRequest, ClientResponse, RaftMessageType
)

if TYPE_CHECKING:
    from simulation.network import Network


class RaftConfig:
    def __init__(
        self,
        heartbeat_interval_ms: int = 150,
        election_timeout_min_ms: int = 300,
        election_timeout_max_ms: int = 500,
        max_entries_per_append: int = 100
    ):
        self.heartbeat_interval_ms = heartbeat_interval_ms
        self.election_timeout_min_ms = election_timeout_min_ms
        self.election_timeout_max_ms = election_timeout_max_ms
        self.max_entries_per_append = max_entries_per_append


class RaftNode(BaseNode):
    def __init__(
        self,
        node_id: str,
        clock: SimulationClock,
        peer_ids: Optional[List[str]] = None,
        config: Optional[RaftConfig] = None,
        persist_path: Optional[str] = None,
        seed: Optional[int] = None
    ):
        super().__init__(node_id, clock, seed)
        self.config = config or RaftConfig()
        self.peer_ids = peer_ids or []
        self.cluster_size = len(self.peer_ids) + 1
        self.raft_state = RaftState(node_id, persist_path)
        self._election_timer_cancel: Optional[Callable] = None
        self._heartbeat_timer_cancel: Optional[Callable] = None
        self._last_heartbeat_time: int = 0
        self._state_machine: Dict[str, Any] = {}
        self._pending_requests: Dict[str, Callable[[ClientResponse], None]] = {}
        self._on_leader_change: List[Callable[[Optional[str]], None]] = []
        self._on_commit: List[Callable[[LogEntry], None]] = []
        self.elections_started = 0
        self.elections_won = 0
        self.entries_committed = 0
    
    def start(self):
        if not self.is_alive:
            return
        logger.info(f"Starting Raft node {self.node_id}")
        self._reset_election_timer()
    
    def stop(self):
        logger.info(f"Stopping Raft node {self.node_id}")
        self._cancel_election_timer()
        self._cancel_heartbeat_timer()
    
    def on_recovery(self):
        super().on_recovery()
        self.raft_state.role = RaftRole.FOLLOWER
        self.start()
    
    def receive_message(self, message: Message):
        if self._should_drop_message(message):
            return
        
        self.metrics.messages_received += 1
        self.metrics.last_message_received_ms = self.clock.now()
        
        if message.sent_time_ms > 0:
            self.record_latency(message.received_time_ms - message.sent_time_ms)
        
        try:
            payload = message.payload
            msg_type = payload.get('type')
            
            if msg_type == RaftMessageType.REQUEST_VOTE.value:
                self._handle_request_vote(RequestVoteRequest.from_dict(payload), message.sender_id)
            elif msg_type == RaftMessageType.REQUEST_VOTE_RESPONSE.value:
                self._handle_request_vote_response(RequestVoteResponse.from_dict(payload))
            elif msg_type == RaftMessageType.APPEND_ENTRIES.value:
                self._handle_append_entries(AppendEntriesRequest.from_dict(payload), message.sender_id)
            elif msg_type == RaftMessageType.APPEND_ENTRIES_RESPONSE.value:
                self._handle_append_entries_response(AppendEntriesResponse.from_dict(payload))
            elif msg_type == RaftMessageType.CLIENT_REQUEST.value:
                self._handle_client_request(ClientRequest.from_dict(payload), message.sender_id)
        except Exception as e:
            logger.error(f"Error handling message: {e}")
    
    def tick(self):
        if not self.is_alive:
            return
        self._apply_committed_entries()
    
    def _reset_election_timer(self):
        self._cancel_election_timer()
        timeout = self.random.randint(self.config.election_timeout_min_ms, self.config.election_timeout_max_ms)
        self._election_timer_cancel = self.clock.schedule(timeout, self._on_election_timeout).cancel
    
    def _cancel_election_timer(self):
        if self._election_timer_cancel:
            self._election_timer_cancel()
            self._election_timer_cancel = None
    
    def _on_election_timeout(self):
        if not self.is_alive or self.raft_state.role == RaftRole.LEADER:
            return
        logger.info(f"Node {self.node_id} election timeout, starting election")
        self._start_election()
    
    def _start_election(self):
        self.elections_started += 1
        self.raft_state.become_candidate()
        self._reset_election_timer()
        
        request = RequestVoteRequest(
            term=self.raft_state.current_term,
            candidate_id=self.node_id,
            last_log_index=self.raft_state.last_log_index,
            last_log_term=self.raft_state.last_log_term
        )
        
        for peer_id in self.peer_ids:
            self._send_to_peer(peer_id, request.to_dict())
        
        self._check_election_result()
    
    def _handle_request_vote(self, request: RequestVoteRequest, sender_id: str):
        if self.raft_state.maybe_step_down(request.term):
            self._reset_election_timer()
        
        vote_granted = self.raft_state.can_vote_for(
            request.candidate_id, request.term, request.last_log_term, request.last_log_index
        )
        
        if vote_granted:
            self.raft_state.voted_for = request.candidate_id
            self._reset_election_timer()
        
        response = RequestVoteResponse(term=self.raft_state.current_term, vote_granted=vote_granted, voter_id=self.node_id)
        self._send_to_peer(sender_id, response.to_dict())
    
    def _handle_request_vote_response(self, response: RequestVoteResponse):
        if self.raft_state.maybe_step_down(response.term):
            self._reset_election_timer()
            return
        
        if self.raft_state.role != RaftRole.CANDIDATE or response.term != self.raft_state.current_term:
            return
        
        if response.vote_granted:
            self.raft_state.record_vote(response.voter_id)
            self._check_election_result()
    
    def _check_election_result(self):
        if self.raft_state.role == RaftRole.CANDIDATE and self.raft_state.has_majority_votes(self.cluster_size):
            self._become_leader()
    
    def _become_leader(self):
        self.elections_won += 1
        self.raft_state.become_leader(self.peer_ids)
        self._cancel_election_timer()
        self._start_heartbeat_timer()
        
        for cb in self._on_leader_change:
            cb(self.node_id)
        
        self._send_heartbeats()
    
    def _start_heartbeat_timer(self):
        self._cancel_heartbeat_timer()
        
        def send_and_reschedule():
            if self.raft_state.role == RaftRole.LEADER and self.is_alive:
                self._send_heartbeats()
                self._heartbeat_timer_cancel = self.clock.schedule(self.config.heartbeat_interval_ms, send_and_reschedule).cancel
        
        send_and_reschedule()
    
    def _cancel_heartbeat_timer(self):
        if self._heartbeat_timer_cancel:
            self._heartbeat_timer_cancel()
            self._heartbeat_timer_cancel = None
    
    def _send_heartbeats(self):
        if self.raft_state.role != RaftRole.LEADER:
            return
        for peer_id in self.peer_ids:
            self._send_append_entries(peer_id)
    
    def _send_append_entries(self, peer_id: str):
        next_idx = self.raft_state.leader_state.next_index.get(peer_id, 1)
        prev_log_index = next_idx - 1
        prev_log_term = self.raft_state.get_log_term(prev_log_index)
        
        entries = self.raft_state.get_entries_from(next_idx)[:self.config.max_entries_per_append]
        
        request = AppendEntriesRequest(
            term=self.raft_state.current_term,
            leader_id=self.node_id,
            prev_log_index=prev_log_index,
            prev_log_term=prev_log_term,
            entries=entries,
            leader_commit=self.raft_state.commit_index
        )
        self._send_to_peer(peer_id, request.to_dict())
    
    def _handle_append_entries(self, request: AppendEntriesRequest, sender_id: str):
        if request.term < self.raft_state.current_term:
            response = AppendEntriesResponse(term=self.raft_state.current_term, success=False, follower_id=self.node_id, match_index=0)
            self._send_to_peer(sender_id, response.to_dict())
            return
        
        self._reset_election_timer()
        
        if request.term > self.raft_state.current_term or self.raft_state.role != RaftRole.FOLLOWER:
            self.raft_state.become_follower(request.term, request.leader_id)
        
        self.raft_state.current_leader = request.leader_id
        self._last_heartbeat_time = self.clock.now()
        
        if request.prev_log_index > 0:
            prev_entry = self.raft_state.get_log_entry(request.prev_log_index)
            if prev_entry is None or prev_entry.term != request.prev_log_term:
                response = AppendEntriesResponse(term=self.raft_state.current_term, success=False, follower_id=self.node_id, match_index=self.raft_state.last_log_index)
                self._send_to_peer(sender_id, response.to_dict())
                return
        
        if request.entries:
            start_index = request.prev_log_index + 1
            for i, entry in enumerate(request.entries):
                log_index = start_index + i
                existing = self.raft_state.get_log_entry(log_index)
                if existing is None:
                    self.raft_state.append_entry(entry)
                elif existing.term != entry.term:
                    self.raft_state.delete_entries_from(log_index)
                    self.raft_state.append_entry(entry)
        
        self.raft_state.maybe_update_commit_index(request.leader_commit)
        
        response = AppendEntriesResponse(term=self.raft_state.current_term, success=True, follower_id=self.node_id, match_index=self.raft_state.last_log_index)
        self._send_to_peer(sender_id, response.to_dict())
    
    def _handle_append_entries_response(self, response: AppendEntriesResponse):
        if self.raft_state.maybe_step_down(response.term):
            self._reset_election_timer()
            return
        
        if self.raft_state.role != RaftRole.LEADER:
            return
        
        follower_id = response.follower_id
        
        if response.success:
            self.raft_state.leader_state.match_index[follower_id] = response.match_index
            self.raft_state.leader_state.next_index[follower_id] = response.match_index + 1
            
            new_commit = self.raft_state.calculate_commit_index(self.cluster_size)
            if new_commit > self.raft_state.commit_index:
                self.raft_state.commit_index = new_commit
        else:
            current_next = self.raft_state.leader_state.next_index.get(follower_id, 1)
            self.raft_state.leader_state.next_index[follower_id] = max(1, current_next - 1)
            self._send_append_entries(follower_id)
    
    def _handle_client_request(self, request: ClientRequest, sender_id: str):
        if self.raft_state.role != RaftRole.LEADER:
            response = ClientResponse(success=False, error="Not leader", leader_hint=self.raft_state.current_leader, request_id=request.request_id)
            self._send_client_response(sender_id, response)
            return
        
        entry = LogEntry(
            term=self.raft_state.current_term,
            index=self.raft_state.last_log_index + 1,
            command=request.command,
            client_id=request.client_id,
            request_id=request.request_id
        )
        self.raft_state.append_entry(entry)
        self._pending_requests[request.request_id] = lambda resp: self._send_client_response(sender_id, resp)
        self._send_heartbeats()
    
    def submit_command(self, command: Any, callback: Optional[Callable[[ClientResponse], None]] = None) -> bool:
        if self.raft_state.role != RaftRole.LEADER:
            if callback:
                callback(ClientResponse(success=False, error="Not leader", leader_hint=self.raft_state.current_leader))
            return False
        
        request_id = f"{self.node_id}_{self.clock.now()}"
        entry = LogEntry(term=self.raft_state.current_term, index=self.raft_state.last_log_index + 1, command=command, client_id=self.node_id, request_id=request_id)
        self.raft_state.append_entry(entry)
        
        if callback:
            self._pending_requests[request_id] = callback
        
        self._send_heartbeats()
        return True
    
    def _send_client_response(self, client_id: str, response: ClientResponse):
        msg = Message(type=MessageType.CLIENT_RESPONSE, sender_id=self.node_id, receiver_id=client_id, payload=response.to_dict())
        self.send_message(msg)
    
    def _apply_committed_entries(self):
        while self.raft_state.last_applied < self.raft_state.commit_index:
            self.raft_state.last_applied += 1
            entry = self.raft_state.get_log_entry(self.raft_state.last_applied)
            
            if entry:
                result = self._apply_to_state_machine(entry.command)
                self.entries_committed += 1
                
                for cb in self._on_commit:
                    cb(entry)
                
                if entry.request_id in self._pending_requests:
                    callback = self._pending_requests.pop(entry.request_id)
                    callback(ClientResponse(success=True, result=result, request_id=entry.request_id))
    
    def _apply_to_state_machine(self, command: Any) -> Any:
        if not isinstance(command, dict):
            return None
        
        op = command.get('op')
        key = command.get('key')
        value = command.get('value')
        
        if op == 'set':
            self._state_machine[key] = value
            return value
        elif op == 'get':
            return self._state_machine.get(key)
        elif op == 'delete':
            return self._state_machine.pop(key, None)
        return None
    
    def get_state_machine(self) -> Dict[str, Any]:
        return dict(self._state_machine)
    
    def _send_to_peer(self, peer_id: str, payload: Dict[str, Any]):
        msg = Message(type=MessageType.CUSTOM, sender_id=self.node_id, receiver_id=peer_id, term=self.raft_state.current_term, payload=payload)
        self.send_message(msg)
    
    def on_leader_change(self, callback: Callable[[Optional[str]], None]):
        self._on_leader_change.append(callback)
    
    def on_commit(self, callback: Callable[[LogEntry], None]):
        self._on_commit.append(callback)
    
    @property
    def is_leader(self) -> bool:
        return self.raft_state.role == RaftRole.LEADER
    
    @property
    def is_follower(self) -> bool:
        return self.raft_state.role == RaftRole.FOLLOWER
    
    @property
    def is_candidate(self) -> bool:
        return self.raft_state.role == RaftRole.CANDIDATE
    
    def status(self) -> Dict[str, Any]:
        return {
            **self.raft_state.summary(),
            'is_alive': self.is_alive,
            'state_machine_size': len(self._state_machine),
            'elections_started': self.elections_started,
            'elections_won': self.elections_won,
            'entries_committed': self.entries_committed
        }