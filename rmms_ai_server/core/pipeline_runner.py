from __future__ import annotations

import asyncio
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Optional

from rmms_ai_server.config import settings
from rmms_ai_server.models.protocol import (
    PipelineStep, StepResultURL, StepError, FinalStatus, TaskStatus,
)
from rmms_ai_server.models.errors import RMMSAIError, PipelineError, ErrorCode
from rmms_ai_server.core.sse_manager import sse_manager
from rmms_ai_server.core.cache_manager import cache_manager
from rmms_ai_server.engine.split_engine import run_split
from rmms_ai_server.engine.midi_engine import run_midi
from rmms_ai_server.engine.detect_engine import run_detect
from rmms_ai_server.engine.generate_engine import run_generate

logger = logging.getLogger(__name__)

CAPABILITY_RUNNERS = {
    "split": run_split,
    "midi": run_midi,
    "detect": run_detect,
    "generate": run_generate,
}


class PipelineRunner:
    def __init__(self):
        self._executor: Optional[ThreadPoolExecutor] = None

    def _get_executor(self) -> ThreadPoolExecutor:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=settings.max_concurrent_tasks,
                thread_name_prefix="rmms-worker",
            )
        return self._executor

    async def run_pipeline(
        self,
        task_id: str,
        pipeline: list[PipelineStep],
        input_path: str,
        output_dir: str,
        device_preference: Optional[str] = None,
    ) -> None:
        from rmms_ai_server.core.task_manager import task_manager

        task = task_manager.get_task(task_id)
        if task is None:
            return

        task.status = TaskStatus.RUNNING
        task.started_at = time.time()

        all_urls: list[StepResultURL] = []
        all_errors: list[StepError] = []
        step_outputs: dict[int, str] = {}

        for i, step in enumerate(pipeline):
            task.current_step = i

            sse_manager.send_progress_running(
                task_id, i, step.capability, 0.0, f"Starting {step.capability}..."
            )

            step_input_path = input_path
            if step.input and step.input.from_step in step_outputs:
                step_input_path = step_outputs[step.input.from_step]
                if step.input.stem:
                    stem_path = os.path.join(step_outputs[step.input.from_step], f"{step.input.stem}.wav")
                    if os.path.isfile(stem_path):
                        step_input_path = stem_path

            step_output_dir = os.path.join(output_dir, f"step_{i}_{step.capability}")
            os.makedirs(step_output_dir, exist_ok=True)

            try:
                result = await self._run_step(
                    task_id, i, step, step_input_path, step_output_dir, device_preference
                )

                if result and "output_dir" in result:
                    step_outputs[i] = result["output_dir"]

                step_urls = self._collect_urls(task_id, step_output_dir, i, step.capability)
                all_urls.extend(step_urls)

                sse_manager.send_progress_completed(
                    task_id, i, step.capability, step_urls
                )

            except RMMSAIError as e:
                error = StepError(step_index=i, step_type=step.capability, error=e.message)
                all_errors.append(error)
                sse_manager.send_progress_failed(task_id, i, step.capability, error)

                for j in range(i + 1, len(pipeline)):
                    dep = pipeline[j].input
                    if dep and dep.from_step == i:
                        skip_error = StepError(
                            step_index=j, step_type=pipeline[j].capability,
                            error=f"Skipped: dependency step {i} failed"
                        )
                        all_errors.append(skip_error)

                break

            except Exception as e:
                error = StepError(step_index=i, step_type=step.capability, error=str(e))
                all_errors.append(error)
                sse_manager.send_progress_failed(task_id, i, step.capability, error)
                break

        if all_errors:
            final_status = FinalStatus.ERROR if len(all_errors) == len(pipeline) else FinalStatus.PARTIAL_ERROR
        else:
            final_status = FinalStatus.DONE

        task.status = TaskStatus.COMPLETED if final_status == FinalStatus.DONE else TaskStatus.FAILED
        task.finished_at = time.time()
        task.result_urls = all_urls
        task.step_errors = all_errors

        sse_manager.send_final_result(
            task_id, final_status, urls=all_urls, errors=all_errors
        )

    async def _run_step(
        self,
        task_id: str,
        step_index: int,
        step: PipelineStep,
        input_path: str,
        output_dir: str,
        device_preference: Optional[str],
    ) -> dict:
        runner = CAPABILITY_RUNNERS.get(step.capability)
        if runner is None:
            raise PipelineError(ErrorCode.PIPELINE_INVALID, f"Unknown capability: {step.capability}")

        params = dict(step.params)
        if device_preference and "device_type" not in params:
            params["device_type"] = device_preference

        def _progress_callback(percent: float, message: str):
            sse_manager.send_progress_running(
                task_id, step_index, step.capability, percent, message
            )

        params["progress_callback"] = _progress_callback

        frozen_params = dict(params)

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self._get_executor(),
            lambda p=frozen_params: runner(input_path=input_path, output_dir=output_dir, **p)
        )

        return result

    def _collect_urls(self, task_id: str, output_dir: str, step_index: int,
                       step_type: str) -> list[StepResultURL]:
        urls = []
        if not os.path.isdir(output_dir):
            return [StepResultURL(step_index=step_index, step_type=step_type, urls=[])]

        file_urls = []
        for fname in sorted(os.listdir(output_dir)):
            fpath = os.path.join(output_dir, fname)
            if os.path.isfile(fpath):
                file_urls.append(f"/api/v1/files/{task_id}/{step_type}/{fname}")

        return [StepResultURL(step_index=step_index, step_type=step_type, urls=file_urls)]


pipeline_runner = PipelineRunner()
