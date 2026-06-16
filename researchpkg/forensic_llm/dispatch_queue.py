"""Priority dispatch queue for hypothesis investigation tasks."""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from researchpkg.forensic_llm.models import (
    DispatchQueueItem,
)
from researchpkg.forensic_llm.plan_utils import (
    dispatch_queue_sort_key,
)


@dataclass
class DispatchQueueManager:
    """Thread-safe dispatch queue (core_spec §4.4.1)."""

    items: List[DispatchQueueItem]
    max_parallel: int = 1
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _running: Dict[str, threading.Thread] = field(default_factory=dict, repr=False)

    def pending(self) -> List[DispatchQueueItem]:
        with self._lock:
            pending = [i for i in self.items if i.status == "pending"]
            return sorted(pending, key=dispatch_queue_sort_key)

    def next_pending(self) -> Optional[DispatchQueueItem]:
        pending = self.pending()
        return pending[0] if pending else None

    def running_count(self) -> int:
        with self._lock:
            return sum(1 for i in self.items if i.status == "running")

    def mark_running(self, task_id: str) -> None:
        with self._lock:
            for item in self.items:
                if item.task_id == task_id:
                    item.status = "running"
                    return

    def mark_terminal(self, task_id: str, status: str) -> None:
        with self._lock:
            for item in self.items:
                if item.task_id == task_id:
                    item.status = status
                    return

    def set_priority(self, task_id: str, new_priority: int) -> bool:
        with self._lock:
            for item in self.items:
                if item.task_id == task_id and item.status == "pending":
                    item.dispatch_priority = new_priority
                    return True
        return False

    def contains_task(self, task_id: str) -> bool:
        with self._lock:
            return any(i.task_id == task_id for i in self.items)

    def inject(self, item: DispatchQueueItem) -> bool:
        """Append a pending hypothesis task if task_id is not already queued."""
        with self._lock:
            if any(i.task_id == item.task_id for i in self.items):
                return False
            item.status = "pending"
            max_seq = max((i.dispatch_sequence for i in self.items), default=-1)
            if item.dispatch_sequence == 0:
                item.dispatch_sequence = max_seq + 1
            self.items.append(item)
            return True

    def min_pending_priority(self) -> int:
        with self._lock:
            pending = [i.dispatch_priority for i in self.items if i.status == "pending"]
            return min(pending) if pending else 1

    def all_terminal(self) -> bool:
        with self._lock:
            return all(
                i.status in ("completed", "failed", "skipped") for i in self.items
            )

    def snapshot(self) -> List[dict]:
        with self._lock:
            return [
                i.model_dump() for i in sorted(self.items, key=dispatch_queue_sort_key)
            ]

    def run_serial_or_parallel(
        self,
        spawn_fn: Callable[[DispatchQueueItem], None],
        join_fn: Callable[[str], None],
        should_stop: Callable[[], bool],
    ) -> None:
        """
        Dispatch tasks until queue is done or should_stop() is true.

        spawn_fn starts a worker for the item (may run in-thread or thread).
        join_fn waits for task_id completion when at capacity.
        """
        while not should_stop():
            with self._lock:
                if self.all_terminal():
                    break
                n_running = sum(1 for i in self.items if i.status == "running")
                if n_running >= self.max_parallel:
                    running_ids = [
                        i.task_id for i in self.items if i.status == "running"
                    ]
                else:
                    running_ids = []
                    nxt = None
                    for item in sorted(self.items, key=dispatch_queue_sort_key):
                        if item.status == "pending":
                            nxt = item
                            break
                    if nxt is None:
                        if n_running == 0:
                            break
                    else:
                        nxt.status = "running"
                        spawn_fn(nxt)

            if running_ids and self.max_parallel > 0:
                for tid in running_ids:
                    join_fn(tid)
            elif not running_ids:
                # spawned in-thread synchronously
                with self._lock:
                    if self.all_terminal():
                        break

        # drain remaining
        with self._lock:
            running_ids = [i.task_id for i in self.items if i.status == "running"]
        for tid in running_ids:
            join_fn(tid)
