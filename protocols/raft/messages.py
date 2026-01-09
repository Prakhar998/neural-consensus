"""
Raft Protocol Messages
Based on the Raft paper Figure 2.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class RaftMessageType(Enum):
    REQUEST_VOTE = "request_vote"
    REQUEST_VOTE_RESPONSE = "request_vote_response"
    APPEND_ENTRIES = "append_entries"
    APPEND_ENTRIES_RESPONSE = "append_entries_response"
    CLIENT_REQUEST = "client_request"
    CLIENT_RESPONSE = "client_response"


@dataclass
class LogEntry:
    term: int
    index: int
    command: Any
    client_id: Optional[str] = None
    request_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {'term': self.term, 'index': self.index, 'command': self.command, 'client_id': self.client_id, 'request_id': self.request_id}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LogEntry':
        return cls(term=data['term'], index=data['index'], command=data['command'], client_id=data.get('client_id'), request_id=data.get('request_id'))


@dataclass
class RequestVoteRequest:
    term: int
    candidate_id: str
    last_log_index: int
    last_log_term: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {'type': RaftMessageType.REQUEST_VOTE.value, 'term': self.term, 'candidate_id': self.candidate_id, 'last_log_index': self.last_log_index, 'last_log_term': self.last_log_term}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RequestVoteRequest':
        return cls(term=data['term'], candidate_id=data['candidate_id'], last_log_index=data['last_log_index'], last_log_term=data['last_log_term'])


@dataclass
class RequestVoteResponse:
    term: int
    vote_granted: bool
    voter_id: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {'type': RaftMessageType.REQUEST_VOTE_RESPONSE.value, 'term': self.term, 'vote_granted': self.vote_granted, 'voter_id': self.voter_id}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RequestVoteResponse':
        return cls(term=data['term'], vote_granted=data['vote_granted'], voter_id=data['voter_id'])


@dataclass
class AppendEntriesRequest:
    term: int
    leader_id: str
    prev_log_index: int
    prev_log_term: int
    entries: List[LogEntry] = field(default_factory=list)
    leader_commit: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {'type': RaftMessageType.APPEND_ENTRIES.value, 'term': self.term, 'leader_id': self.leader_id, 'prev_log_index': self.prev_log_index, 'prev_log_term': self.prev_log_term, 'entries': [e.to_dict() for e in self.entries], 'leader_commit': self.leader_commit}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AppendEntriesRequest':
        return cls(term=data['term'], leader_id=data['leader_id'], prev_log_index=data['prev_log_index'], prev_log_term=data['prev_log_term'], entries=[LogEntry.from_dict(e) for e in data.get('entries', [])], leader_commit=data.get('leader_commit', 0))
    
    @property
    def is_heartbeat(self) -> bool:
        return len(self.entries) == 0


@dataclass
class AppendEntriesResponse:
    term: int
    success: bool
    follower_id: str
    match_index: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {'type': RaftMessageType.APPEND_ENTRIES_RESPONSE.value, 'term': self.term, 'success': self.success, 'follower_id': self.follower_id, 'match_index': self.match_index}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AppendEntriesResponse':
        return cls(term=data['term'], success=data['success'], follower_id=data['follower_id'], match_index=data.get('match_index', 0))


@dataclass
class ClientRequest:
    client_id: str
    request_id: str
    command: Any
    
    def to_dict(self) -> Dict[str, Any]:
        return {'type': RaftMessageType.CLIENT_REQUEST.value, 'client_id': self.client_id, 'request_id': self.request_id, 'command': self.command}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ClientRequest':
        return cls(client_id=data['client_id'], request_id=data['request_id'], command=data['command'])


@dataclass
class ClientResponse:
    success: bool
    result: Any = None
    error: Optional[str] = None
    leader_hint: Optional[str] = None
    request_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {'type': RaftMessageType.CLIENT_RESPONSE.value, 'success': self.success, 'result': self.result, 'error': self.error, 'leader_hint': self.leader_hint, 'request_id': self.request_id}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ClientResponse':
        return cls(success=data['success'], result=data.get('result'), error=data.get('error'), leader_hint=data.get('leader_hint'), request_id=data.get('request_id'))


def parse_raft_message(data: Dict[str, Any]):
    parsers = {
        RaftMessageType.REQUEST_VOTE.value: RequestVoteRequest.from_dict,
        RaftMessageType.REQUEST_VOTE_RESPONSE.value: RequestVoteResponse.from_dict,
        RaftMessageType.APPEND_ENTRIES.value: AppendEntriesRequest.from_dict,
        RaftMessageType.APPEND_ENTRIES_RESPONSE.value: AppendEntriesResponse.from_dict,
        RaftMessageType.CLIENT_REQUEST.value: ClientRequest.from_dict,
        RaftMessageType.CLIENT_RESPONSE.value: ClientResponse.from_dict,
    }
    parser = parsers.get(data.get('type'))
    if parser is None:
        raise ValueError(f"Unknown message type: {data.get('type')}")
    return parser(data)