from __future__ import annotations

import asyncio
import logging
from typing import Optional

from rmms_ai_server.models.protocol import (
    ProgressSSEEvent, PartialResultEvent, FinalResultEvent,
    ProgressStatus, FinalStatus, StepResultURL, StepError,
)

logger = logging.getLogger(__name__)


class SSEManager:
    def __init__(self):
        self._subscribers: dict[str, list[asyncio.Queue]] = {}

    def subscribe(self, task_id: str) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        if task_id not in self._subscribers:
            self._subscribers[task_id] = []
        self._subscribers[task_id].append(queue)
        return queue

    def unsubscribe(self, task_id: str, queue: asyncio.Queue) -> None:
        if task_id in self._subscribers:
            try:
                self._subscribers[task_id].remove(queue)
            except ValueError:
                pass
            if not self._subscribers[task_id]:
                del self._subscribers[task_id]

    def _publish(self, task_id: str, event: dict) -> None:
        queues = self._subscribers.get(task_id, [])
        dead_queues = []
        for q in queues:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                dead_queues.append(q)
        for q in dead_queues:
            self.unsubscribe(task_id, q)

    def send_progress_running(self, task_id: str, step_index: int, step_type: str,
                               percent: float, message: str = "") -> None:
        event = ProgressSSEEvent(
            task_id=task_id, step_index=step_index, step_type=step_type,
            status=ProgressStatus.RUNNING, percent=percent, message=message,
        )
        self._publish(task_id, event.model_dump())

    def send_progress_completed(self, task_id: str, step_index: int, step_type: str,
                                 urls: list[StepResultURL]) -> None:
        event = ProgressSSEEvent(
            task_id=task_id, step_index=step_index, step_type=step_type,
            status=ProgressStatus.COMPLETED, urls=urls,
        )
        self._publish(task_id, event.model_dump())

    def send_progress_failed(self, task_id: str, step_index: int, step_type: str,
                              error: StepError) -> None:
        event = ProgressSSEEvent(
            task_id=task_id, step_index=step_index, step_type=step_type,
            status=ProgressStatus.FAILED, error=error,
        )
        self._publish(task_id, event.model_dump())

    def send_partial_result(self, task_id: str, step_index: int, step_type: str,
                             data: dict) -> None:
        event = PartialResultEvent(
            task_id=task_id, step_index=step_index, step_type=step_type, data=data,
        )
        self._publish(task_id, event.model_dump())

    def send_final_result(self, task_id: str, status: FinalStatus,
                           urls: list[StepResultURL] = None,
                           errors: list[StepError] = None,
                           message: str = "") -> None:
        event = FinalResultEvent(
            task_id=task_id, status=status,
            urls=urls or [], errors=errors or [], message=message,
        )
        self._publish(task_id, event.model_dump())


sse_manager = SSEManager()
