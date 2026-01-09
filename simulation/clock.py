"""
Simulation Clock Module
Provides logical time management for the distributed system simulation.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, List, Optional
from enum import Enum
import heapq
import time
import threading
from loguru import logger


class ClockMode(Enum):
    SIMULATED = "simulated"
    REALTIME = "realtime"


@dataclass(order=True)
class ScheduledEvent:
    time: int
    sequence: int
    callback: Callable = field(compare=False)
    args: tuple = field(default_factory=tuple, compare=False)
    kwargs: dict = field(default_factory=dict, compare=False)
    cancelled: bool = field(default=False, compare=False)
    
    def cancel(self):
        self.cancelled = True


class SimulationClock:
    def __init__(
        self,
        mode: ClockMode = ClockMode.SIMULATED,
        start_time_ms: int = 0,
        tick_interval_ms: int = 1
    ):
        self.mode = mode
        self._current_time_ms = start_time_ms
        self.tick_interval_ms = tick_interval_ms
        self._event_queue: List[ScheduledEvent] = []
        self._event_sequence = 0
        self._start_wall_time = time.monotonic()
        self._start_sim_time = start_time_ms
        self._lock = threading.Lock()
        self.events_processed = 0
        self.events_cancelled = 0
    
    @property
    def current_time_ms(self) -> int:
        if self.mode == ClockMode.REALTIME:
            elapsed = (time.monotonic() - self._start_wall_time) * 1000
            return self._start_sim_time + int(elapsed)
        return self._current_time_ms
    
    @property
    def current_time_s(self) -> float:
        return self.current_time_ms / 1000.0
    
    def now(self) -> int:
        return self.current_time_ms
    
    def schedule(self, delay_ms: int, callback: Callable, *args, **kwargs) -> ScheduledEvent:
        with self._lock:
            scheduled_time = self.current_time_ms + delay_ms
            event = ScheduledEvent(
                time=scheduled_time,
                sequence=self._event_sequence,
                callback=callback,
                args=args,
                kwargs=kwargs
            )
            self._event_sequence += 1
            heapq.heappush(self._event_queue, event)
            return event
    
    def schedule_at(self, time_ms: int, callback: Callable, *args, **kwargs) -> ScheduledEvent:
        delay = max(0, time_ms - self.current_time_ms)
        return self.schedule(delay, callback, *args, **kwargs)
    
    def schedule_recurring(self, interval_ms: int, callback: Callable, *args, **kwargs) -> Callable[[], None]:
        cancelled = [False]
        
        def recurring_wrapper():
            if not cancelled[0]:
                callback(*args, **kwargs)
                self.schedule(interval_ms, recurring_wrapper)
        
        def cancel():
            cancelled[0] = True
        
        self.schedule(interval_ms, recurring_wrapper)
        return cancel
    
    def tick(self, advance_ms: Optional[int] = None) -> int:
        if self.mode == ClockMode.REALTIME:
            return self._process_due_events()
        
        advance = advance_ms if advance_ms is not None else self.tick_interval_ms
        target_time = self._current_time_ms + advance
        events_processed = 0
        
        with self._lock:
            while self._event_queue and self._event_queue[0].time <= target_time:
                event = heapq.heappop(self._event_queue)
                
                if event.cancelled:
                    self.events_cancelled += 1
                    continue
                
                self._current_time_ms = event.time
                self._lock.release()
                try:
                    event.callback(*event.args, **event.kwargs)
                except Exception as e:
                    logger.error(f"Error in scheduled event: {e}")
                finally:
                    self._lock.acquire()
                
                events_processed += 1
                self.events_processed += 1
            
            self._current_time_ms = target_time
        
        return events_processed
    
    def _process_due_events(self) -> int:
        current = self.current_time_ms
        events_processed = 0
        
        with self._lock:
            while self._event_queue and self._event_queue[0].time <= current:
                event = heapq.heappop(self._event_queue)
                if event.cancelled:
                    self.events_cancelled += 1
                    continue
                
                self._lock.release()
                try:
                    event.callback(*event.args, **event.kwargs)
                except Exception as e:
                    logger.error(f"Error in scheduled event: {e}")
                finally:
                    self._lock.acquire()
                
                events_processed += 1
                self.events_processed += 1
        
        return events_processed
    
    def run_until(self, target_time_ms: int) -> int:
        total_events = 0
        while self._current_time_ms < target_time_ms:
            advance = min(self.tick_interval_ms, target_time_ms - self._current_time_ms)
            total_events += self.tick(advance)
        return total_events
    
    def run_for(self, duration_ms: int) -> int:
        return self.run_until(self.current_time_ms + duration_ms)
    
    def run_until_no_events(self, max_time_ms: Optional[int] = None) -> int:
        total_events = 0
        max_time = max_time_ms or (self._current_time_ms + 3600000)
        
        while self._event_queue and self._current_time_ms < max_time:
            next_event_time = self._event_queue[0].time
            if next_event_time > max_time:
                break
            total_events += self.tick(next_event_time - self._current_time_ms)
        
        return total_events
    
    def pending_events(self) -> int:
        with self._lock:
            return len([e for e in self._event_queue if not e.cancelled])
    
    def clear_events(self):
        with self._lock:
            self._event_queue.clear()
    
    def reset(self, start_time_ms: int = 0):
        with self._lock:
            self._current_time_ms = start_time_ms
            self._event_queue.clear()
            self._event_sequence = 0
            self._start_wall_time = time.monotonic()
            self._start_sim_time = start_time_ms
            self.events_processed = 0
            self.events_cancelled = 0
    
    def stats(self) -> dict:
        return {
            "current_time_ms": self.current_time_ms,
            "mode": self.mode.value,
            "pending_events": self.pending_events(),
            "events_processed": self.events_processed,
            "events_cancelled": self.events_cancelled
        }


def ms_to_s(ms: int) -> float:
    return ms / 1000.0

def s_to_ms(s: float) -> int:
    return int(s * 1000)

def format_time(ms: int) -> str:
    if ms < 1000:
        return f"{ms}ms"
    elif ms < 60000:
        return f"{ms/1000:.2f}s"
    elif ms < 3600000:
        return f"{ms/60000:.2f}m"
    else:
        return f"{ms/3600000:.2f}h"