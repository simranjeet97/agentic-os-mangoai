"""
core/task_queue.py — Priority queue for pending tasks with dependency tracking.

Features:
  - asyncio-safe priority queue (min-heap: lower number = higher priority)
  - Parallel execution of independent tasks
  - Dependency DAG: task B waits until all its dependencies are DONE
  - Full task lifecycle: PENDING → RUNNING → DONE | FAILED | CANCELLED
"""

from __future__ import annotations

import asyncio
import heapq
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Coroutine, Optional

from pydantic import BaseModel, Field

from core.logging_config import get_logger

logger = get_logger(__name__)


# ── Task state ────────────────────────────────────────────────────────────────


class QueuedTaskStatus(str, Enum):
    PENDING = "pending"
    WAITING = "waiting"       # blocked by dependencies
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


class QueuedTask(BaseModel):
    """A single unit of work inside the TaskQueue."""

    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    priority: int = Field(default=3, ge=1, le=5)   # 1=highest, 5=lowest
    dependencies: list[str] = Field(default_factory=list)  # task_ids
    status: QueuedTaskStatus = QueuedTaskStatus.PENDING
    payload: dict[str, Any] = Field(default_factory=dict)
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # ── Heap ordering (priority queue) ────────────────────────────────────────

    def __lt__(self, other: "QueuedTask") -> bool:
        return (self.priority, self.created_at) < (other.priority, other.created_at)


# ── Heap entry wrapper (avoids comparing QueuedTask objects directly) ─────────


class _HeapItem:
    """Wrapper stored in heapq to keep heap invariants clean."""

    __slots__ = ("priority", "seq", "task")

    _counter: int = 0

    def __init__(self, task: QueuedTask) -> None:
        _HeapItem._counter += 1
        self.priority = task.priority
        self.seq = _HeapItem._counter
        self.task = task

    def __lt__(self, other: "_HeapItem") -> bool:
        return (self.priority, self.seq) < (other.priority, other.seq)


# ── TaskQueue ─────────────────────────────────────────────────────────────────


class TaskQueue:
    """
    Async priority queue with dependency-aware scheduling.

    Usage:
        q = TaskQueue()

        # Enqueue tasks
        t1 = await q.enqueue("search web", priority=2, payload={...})
        t2 = await q.enqueue("write file",  priority=3, dependencies=[t1.task_id])

        # Run all ready tasks (caller supplies the executor coroutine)
        await q.run_all(executor_fn)
    """

    def __init__(self, max_parallel: int = 4) -> None:
        self._heap: list[_HeapItem] = []
        self._tasks: dict[str, QueuedTask] = {}         # task_id → QueuedTask
        self._lock = asyncio.Lock()
        self._done_event: dict[str, asyncio.Event] = {} # task_id → event
        self.max_parallel = max_parallel

    # ── Public API ────────────────────────────────────────────────────────────

    async def enqueue(
        self,
        name: str,
        priority: int = 3,
        dependencies: Optional[list[str]] = None,
        payload: Optional[dict[str, Any]] = None,
    ) -> QueuedTask:
        """Add a task to the queue. Returns the created QueuedTask."""
        async with self._lock:
            task = QueuedTask(
                name=name,
                priority=priority,
                dependencies=dependencies or [],
                payload=payload or {},
                status=QueuedTaskStatus.WAITING if dependencies else QueuedTaskStatus.PENDING,
            )
            self._tasks[task.task_id] = task
            self._done_event[task.task_id] = asyncio.Event()
            if not task.dependencies:
                heapq.heappush(self._heap, _HeapItem(task))
            logger.info(
                "Task enqueued",
                task_id=task.task_id,
                name=name,
                priority=priority,
                deps=dependencies,
            )
            return task

    async def cancel(self, task_id: str) -> bool:
        """Cancel a pending or waiting task. Returns True if cancelled."""
        async with self._lock:
            task = self._tasks.get(task_id)
            if task and task.status in (QueuedTaskStatus.PENDING, QueuedTaskStatus.WAITING):
                task.status = QueuedTaskStatus.CANCELLED
                self._done_event[task_id].set()
                logger.info("Task cancelled", task_id=task_id)
                return True
        return False

    def get_task(self, task_id: str) -> Optional[QueuedTask]:
        return self._tasks.get(task_id)

    def all_tasks(self) -> list[QueuedTask]:
        return list(self._tasks.values())

    def pending_count(self) -> int:
        return sum(
            1 for t in self._tasks.values()
            if t.status in (QueuedTaskStatus.PENDING, QueuedTaskStatus.WAITING, QueuedTaskStatus.RUNNING)
        )

    async def wait_for(self, task_id: str, timeout: Optional[float] = None) -> QueuedTask:
        """Block until the given task is complete (done/failed/cancelled)."""
        event = self._done_event.get(task_id)
        if event is None:
            raise KeyError(f"Unknown task: {task_id}")
        await asyncio.wait_for(event.wait(), timeout=timeout)
        return self._tasks[task_id]

    # ── Execution engine ──────────────────────────────────────────────────────

    async def run_all(
        self,
        executor_fn: Callable[[QueuedTask], Coroutine[Any, Any, Any]],
    ) -> list[QueuedTask]:
        """
        Drain the queue, executing tasks as their dependencies resolve.
        Runs up to `max_parallel` tasks concurrently.
        Returns all tasks once the queue is empty.
        """
        semaphore = asyncio.Semaphore(self.max_parallel)

        async def _run_one(task: QueuedTask) -> None:
            # Wait for all dependencies to finish
            for dep_id in task.dependencies:
                dep = self._tasks.get(dep_id)
                if dep is None:
                    continue
                await asyncio.wait_for(
                    self._done_event[dep_id].wait(), timeout=300
                )
                if dep.status == QueuedTaskStatus.FAILED:
                    async with self._lock:
                        task.status = QueuedTaskStatus.FAILED
                        task.error = f"Dependency {dep_id} failed"
                        task.completed_at = datetime.utcnow()
                        self._done_event[task.task_id].set()
                    logger.warning("Task blocked by failed dependency", task_id=task.task_id, dep_id=dep_id)
                    return

            async with semaphore:
                async with self._lock:
                    if task.status == QueuedTaskStatus.CANCELLED:
                        return
                    task.status = QueuedTaskStatus.RUNNING
                    task.started_at = datetime.utcnow()

                logger.info("Running task", task_id=task.task_id, name=task.name)
                try:
                    result = await executor_fn(task)
                    async with self._lock:
                        task.result = result
                        task.status = QueuedTaskStatus.DONE
                        task.completed_at = datetime.utcnow()
                except Exception as exc:
                    logger.error("Task failed", task_id=task.task_id, error=str(exc))
                    async with self._lock:
                        task.status = QueuedTaskStatus.FAILED
                        task.error = str(exc)
                        task.completed_at = datetime.utcnow()
                finally:
                    self._done_event[task.task_id].set()
                    # Promote waiting tasks whose dependencies are now all done
                    await self._promote_waiting_tasks()

        # Gather all tasks that are already pending + launch coroutines for waiting tasks
        coroutines = [_run_one(t) for t in self._tasks.values()
                      if t.status in (QueuedTaskStatus.PENDING, QueuedTaskStatus.WAITING)]
        if coroutines:
            await asyncio.gather(*coroutines, return_exceptions=True)

        return self.all_tasks()

    async def _promote_waiting_tasks(self) -> None:
        """After a task completes, add newly-unblocked tasks to the heap."""
        async with self._lock:
            for task in self._tasks.values():
                if task.status != QueuedTaskStatus.WAITING:
                    continue
                deps_done = all(
                    self._tasks.get(dep_id, QueuedTask(name="", task_id=dep_id)).status
                    in (QueuedTaskStatus.DONE, QueuedTaskStatus.CANCELLED)
                    for dep_id in task.dependencies
                )
                if deps_done:
                    task.status = QueuedTaskStatus.PENDING
                    heapq.heappush(self._heap, _HeapItem(task))

    # ── Iteration helper ──────────────────────────────────────────────────────

    async def _pop_ready(self) -> Optional[QueuedTask]:
        """Pop the highest-priority PENDING task, or None if none ready."""
        async with self._lock:
            while self._heap:
                item = heapq.heappop(self._heap)
                if item.task.status == QueuedTaskStatus.PENDING:
                    return item.task
            return None
