"""
Raft State Management
Based on the Raft paper Figure 2.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import json
import os
from loguru import logger

from .messages import LogEntry


class RaftRole(Enum):
    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    LEADER = "leader"


@dataclass
class PersistentState:
    current_term: int = 0
    voted_for: Optional[str] = None
    log: List[LogEntry] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {'current_term': self.current_term, 'voted_for': self.voted_for, 'log': [entry.to_dict() for entry in self.log]}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PersistentState':
        return cls(current_term=data.get('current_term', 0), voted_for=data.get('voted_for'), log=[LogEntry.from_dict(e) for e in data.get('log', [])])
    
    def save(self, filepath: str):
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f)
    
    @classmethod
    def load(cls, filepath: str) -> 'PersistentState':
        if not os.path.exists(filepath):
            return cls()
        with open(filepath, 'r') as f:
            return cls.from_dict(json.load(f))


@dataclass
class VolatileState:
    commit_index: int = 0
    last_applied: int = 0


@dataclass
class LeaderState:
    next_index: Dict[str, int] = field(default_factory=dict)
    match_index: Dict[str, int] = field(default_factory=dict)
    
    def initialize(self, peer_ids: List[str], last_log_index: int):
        for peer_id in peer_ids:
            self.next_index[peer_id] = last_log_index + 1
            self.match_index[peer_id] = 0
    
    def reset(self):
        self.next_index.clear()
        self.match_index.clear()


class RaftState:
    def __init__(self, node_id: str, persist_path: Optional[str] = None):
        self.node_id = node_id
        self.persist_path = persist_path
        self._role = RaftRole.FOLLOWER
        self.persistent = PersistentState()
        self.volatile = VolatileState()
        self.leader_state = LeaderState()
        self.current_leader: Optional[str] = None
        self.votes_received: set = set()
        
        if persist_path and os.path.exists(persist_path):
            self.persistent = PersistentState.load(persist_path)
    
    @property
    def role(self) -> RaftRole:
        return self._role
    
    @role.setter
    def role(self, new_role: RaftRole):
        if new_role != self._role:
            logger.info(f"Node {self.node_id}: {self._role.value} -> {new_role.value}")
            self._role = new_role
    
    @property
    def current_term(self) -> int:
        return self.persistent.current_term
    
    @current_term.setter
    def current_term(self, term: int):
        if term != self.persistent.current_term:
            self.persistent.current_term = term
            self.persistent.voted_for = None
            self._persist()
    
    @property
    def voted_for(self) -> Optional[str]:
        return self.persistent.voted_for
    
    @voted_for.setter
    def voted_for(self, candidate_id: Optional[str]):
        if candidate_id != self.persistent.voted_for:
            self.persistent.voted_for = candidate_id
            self._persist()
    
    @property
    def log(self) -> List[LogEntry]:
        return self.persistent.log
    
    @property
    def commit_index(self) -> int:
        return self.volatile.commit_index
    
    @commit_index.setter
    def commit_index(self, index: int):
        self.volatile.commit_index = index
    
    @property
    def last_applied(self) -> int:
        return self.volatile.last_applied
    
    @last_applied.setter
    def last_applied(self, index: int):
        self.volatile.last_applied = index
    
    @property
    def last_log_index(self) -> int:
        return self.log[-1].index if self.log else 0
    
    @property
    def last_log_term(self) -> int:
        return self.log[-1].term if self.log else 0
    
    def get_log_entry(self, index: int) -> Optional[LogEntry]:
        if index <= 0 or index > len(self.log):
            return None
        return self.log[index - 1]
    
    def get_log_term(self, index: int) -> int:
        entry = self.get_log_entry(index)
        return entry.term if entry else 0
    
    def append_entry(self, entry: LogEntry):
        self.persistent.log.append(entry)
        self._persist()
    
    def append_entries(self, entries: List[LogEntry], start_index: int):
        if start_index <= len(self.log):
            self.persistent.log = self.persistent.log[:start_index - 1]
        self.persistent.log.extend(entries)
        self._persist()
    
    def delete_entries_from(self, index: int):
        if index <= len(self.log):
            self.persistent.log = self.persistent.log[:index - 1]
            self._persist()
    
    def get_entries_from(self, start_index: int) -> List[LogEntry]:
        if start_index <= 0:
            start_index = 1
        if start_index > len(self.log):
            return []
        return self.log[start_index - 1:]
    
    def become_follower(self, term: int, leader_id: Optional[str] = None):
        self.role = RaftRole.FOLLOWER
        self.current_term = term
        self.current_leader = leader_id
        self.votes_received.clear()
        self.leader_state.reset()
    
    def become_candidate(self):
        self.role = RaftRole.CANDIDATE
        self.current_term = self.current_term + 1
        self.voted_for = self.node_id
        self.votes_received = {self.node_id}
        self.current_leader = None
        self.leader_state.reset()
    
    def become_leader(self, peer_ids: List[str]):
        self.role = RaftRole.LEADER
        self.current_leader = self.node_id
        self.leader_state.initialize(peer_ids, self.last_log_index)
        self.votes_received.clear()
        logger.info(f"Node {self.node_id} became leader for term {self.current_term}")
    
    def record_vote(self, voter_id: str) -> int:
        self.votes_received.add(voter_id)
        return len(self.votes_received)
    
    def has_majority_votes(self, cluster_size: int) -> bool:
        return len(self.votes_received) > cluster_size // 2
    
    def maybe_step_down(self, term: int) -> bool:
        if term > self.current_term:
            self.become_follower(term)
            return True
        return False
    
    def _persist(self):
        if self.persist_path:
            self.persistent.save(self.persist_path)
    
    def is_log_up_to_date(self, last_log_term: int, last_log_index: int) -> bool:
        my_last_term = self.last_log_term
        my_last_index = self.last_log_index
        if last_log_term != my_last_term:
            return last_log_term > my_last_term
        return last_log_index >= my_last_index
    
    def can_vote_for(self, candidate_id: str, term: int, last_log_term: int, last_log_index: int) -> bool:
        if term < self.current_term:
            return False
        if self.voted_for is not None and self.voted_for != candidate_id:
            if term == self.current_term:
                return False
        return self.is_log_up_to_date(last_log_term, last_log_index)
    
    def maybe_update_commit_index(self, leader_commit: int):
        if leader_commit > self.commit_index:
            self.commit_index = min(leader_commit, self.last_log_index)
    
    def calculate_commit_index(self, cluster_size: int) -> int:
        if self.role != RaftRole.LEADER:
            return self.commit_index
        
        match_indices = list(self.leader_state.match_index.values())
        match_indices.append(self.last_log_index)
        match_indices.sort(reverse=True)
        
        majority = cluster_size // 2 + 1
        if len(match_indices) >= majority:
            potential_commit = match_indices[majority - 1]
            if potential_commit > 0:
                entry = self.get_log_entry(potential_commit)
                if entry and entry.term == self.current_term:
                    return max(self.commit_index, potential_commit)
        
        return self.commit_index
    
    def summary(self) -> Dict[str, Any]:
        return {
            'node_id': self.node_id,
            'role': self.role.value,
            'term': self.current_term,
            'voted_for': self.voted_for,
            'log_length': len(self.log),
            'last_log_index': self.last_log_index,
            'last_log_term': self.last_log_term,
            'commit_index': self.commit_index,
            'last_applied': self.last_applied,
            'current_leader': self.current_leader,
            'votes_received': len(self.votes_received)
        }