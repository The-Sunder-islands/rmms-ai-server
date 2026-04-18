from __future__ import annotations

import asyncio
import logging
import shutil
import time
import uuid
from pathlib import Path
from typing import Optional

from rmms_ai_server.config import settings
from rmms_ai_server.models.protocol import (
    TaskInfo, TaskStatus, PipelineStep, StepResultURL, StepError,
)
from rmms_ai_server.models.errors import ServerError, ErrorCode, QuotaError
from rmms_ai_server.core.pipeline_runner import pipeline_runner

logger = logging.getLogger(__name__)


class TaskManager:
    def __init__(self):
        self._tasks: dict[str, TaskInfo] = {}
        self._lock = asyncio.Lock()
        self._active_count = 0
        self._start_time = time.time()
        self._cancel_events: dict[str, asyncio.Event] = {}

    @property
    def start_time(self) -> float:
        return self._start_time

    @property
    def active_count(self) -> int:
        return self._active_count

    @property
    def queued_count(self) -> int:
        return sum(1 for t in self._tasks.values() if t.status == TaskStatus.QUEUED)

    def is_cancelled(self, task_id: str) -> bool:
        ev = self._cancel_events.get(task_id)
        return ev is not None and ev.is_set()

    async def create_task(
        self,
        pipeline: list[PipelineStep],
        input_path: str,
        device_preference: Optional[str] = None,
        priority: Optional[int] = None,
    ) -> TaskInfo:
        async with self._lock:
            if self._active_count >= settings.max_concurrent_tasks:
                raise QuotaError(
                    ErrorCode.SERVER_OVERLOADED,
                    f"Too many concurrent tasks (max {settings.max_concurrent_tasks})",
                )

            task_id = uuid.uuid4().hex[:16]
            task = TaskInfo(
                task_id=task_id,
                status=TaskStatus.QUEUED,
                pipeline=pipeline,
                created_at=time.time(),
            )
            self._tasks[task_id] = task
            self._cancel_events[task_id] = asyncio.Event()
            self._active_count += 1

        output_dir = str(settings.resolved_output_dir / task_id)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        asyncio.create_task(
            self._run_task(task_id, input_path, output_dir, device_preference)
        )

        return task

    async def _run_task(
        self,
        task_id: str,
        input_path: str,
        output_dir: str,
        device_preference: Optional[str],
    ) -> None:
        task = self._tasks.get(task_id)
        if task is None:
            return

        try:
            await pipeline_runner.run_pipeline(
                task_id=task_id,
                pipeline=task.pipeline,
                input_path=input_path,
                output_dir=output_dir,
                device_preference=device_preference,
            )
        except Exception as e:
            logger.exception(f"Pipeline runner error for task {task_id}")
            task = self._tasks.get(task_id)
            if task is not None:
                task.status = TaskStatus.FAILED
                task.error = str(e)
                task.finished_at = time.time()
        finally:
            task = self._tasks.get(task_id)
            if task is not None:
                self._active_count = max(0, self._active_count - 1)
            self._cancel_events.pop(task_id, None)

    def get_task(self, task_id: str) -> Optional[TaskInfo]:
        return self._tasks.get(task_id)

    def list_tasks(self) -> list[TaskInfo]:
        return list(self._tasks.values())

    async def delete_task(self, task_id: str) -> bool:
        async with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return False

            was_active = task.status in (TaskStatus.QUEUED, TaskStatus.RUNNING)

            task.status = TaskStatus.CANCELLED
            task.finished_at = time.time()

            cancel_ev = self._cancel_events.get(task_id)
            if cancel_ev is not None:
                cancel_ev.set()

            if was_active:
                self._active_count = max(0, self._active_count - 1)

            del self._tasks[task_id]

        upload_dir = str(settings.resolved_upload_dir / task_id)
        output_dir = str(settings.resolved_output_dir / task_id)

        for d in (upload_dir, output_dir):
            p = Path(d)
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)

        return True

    async def cleanup_expired(self) -> int:
        now = time.time()
        to_delete = []

        async with self._lock:
            for task_id, task in self._tasks.items():
                if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
                    if task.finished_at and (now - task.finished_at) > settings.task_ttl_seconds:
                        to_delete.append(task_id)

        for task_id in to_delete:
            await self.delete_task(task_id)

        if to_delete:
            logger.info(f"Cleaned up {len(to_delete)} expired tasks")

        return len(to_delete)


task_manager = TaskManager()
