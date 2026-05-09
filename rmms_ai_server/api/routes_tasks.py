from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional

import time

from fastapi import APIRouter, File, Form, UploadFile, Query, Request
from fastapi.responses import JSONResponse

from rmms_ai_server.config import settings
from rmms_ai_server.models.protocol import TaskStatus, PipelineStep
from rmms_ai_server.models.errors import InputError, ErrorCode, RMMSAIError
from rmms_ai_server.models.requests import TaskSubmitJSON, TaskSubmitResponse, TaskStatusResponse
from rmms_ai_server.core.task_manager import task_manager
from rmms_ai_server.core.pipeline_defs import resolve_pipeline
from rmms_ai_server.core.cache_manager import cache_manager

router = APIRouter()

ALLOWED_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.wma', '.aiff', '.opus'}


def _allowed_file(filename: str) -> bool:
    if not filename:
        return False
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXTENSIONS


@router.post("/tasks", response_model=TaskSubmitResponse)
async def submit_task(
    request: Request,
    file: Optional[UploadFile] = File(None),
    pipeline: Optional[str] = Form(None),
    preset: Optional[str] = Form(None),
    device_preference: Optional[str] = Form(None),
    priority: Optional[int] = Form(None),
    force_refresh: bool = Form(False),
    audio_url: Optional[str] = Form(None),
):
    if file is None and audio_url is None:
        raise InputError(ErrorCode.INPUT_MISSING, "No audio file or URL provided")

    input_path = None
    file_hash = None

    if file is not None:
        if not file.filename:
            raise InputError(ErrorCode.INPUT_MISSING, "Empty filename")

        if not _allowed_file(file.filename):
            raise InputError(
                ErrorCode.INPUT_FORMAT_UNSUPPORTED,
                f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            )

        content = await file.read()
        if len(content) > settings.max_upload_bytes:
            raise InputError(
                ErrorCode.INPUT_TOO_LARGE,
                f"File too large ({len(content) // (1024*1024)}MB). Maximum: {settings.max_upload_mb}MB"
            )

        task_id_placeholder = os.urandom(8).hex()
        upload_dir = str(settings.resolved_upload_dir / task_id_placeholder)
        Path(upload_dir).mkdir(parents=True, exist_ok=True)

        input_path = os.path.join(upload_dir, file.filename)
        with open(input_path, "wb") as f:
            f.write(content)

        file_hash = cache_manager.compute_file_hash(input_path)

    elif audio_url:
        raise InputError(ErrorCode.INPUT_INVALID_FORMAT, "URL input not yet implemented")

    import json
    pipeline_steps = None
    if pipeline:
        try:
            pipeline_steps = json.loads(pipeline)
        except json.JSONDecodeError:
            raise InputError(ErrorCode.INPUT_INVALID_PARAMS, "Invalid pipeline JSON")

    resolved = resolve_pipeline(pipeline_steps, preset)

    if not force_refresh and file_hash:
        params_key = {"preset": preset, "pipeline": pipeline_steps, "device": device_preference}
        cached = cache_manager.get(file_hash, params_key)
        if cached:
            return TaskSubmitResponse(
                task_id=cached["task_id"],
                status="done",
                message="Cached result",
                cached=True,
                created_at=cached.get("created_at", ""),
            )

    task = await task_manager.create_task(
        pipeline=resolved,
        input_path=input_path,
        device_preference=device_preference,
        priority=priority,
    )

    if file_hash:
        params_key = {"preset": preset, "pipeline": pipeline_steps, "device": device_preference}
        cache_manager.put(file_hash, params_key, {"task_id": task.task_id})

    return JSONResponse(
        status_code=201,
        content=TaskSubmitResponse(
            task_id=task.task_id,
            status="queued",
            message="Task submitted successfully",
            pipeline=resolved,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(task.created_at)),
        ).model_dump(by_alias=True),
    )


@router.get("/tasks")
async def list_tasks(
    status: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    all_tasks = task_manager.list_tasks()
    if status:
        all_tasks = [t for t in all_tasks if t.status.value == status]
    total = len(all_tasks)
    paged = all_tasks[offset:offset + limit]
    return {
        "tasks": [
            {
                "task_id": t.task_id,
                "status": t.status.value,
                "current_step": t.current_step,
                "percent": t.percent,
            }
            for t in paged
        ],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    task = task_manager.get_task(task_id)
    if task is None:
        raise InputError(ErrorCode.TASK_NOT_FOUND, f"Task '{task_id}' not found")

    def _ts(ts: float | None) -> str | None:
        if ts is None:
            return None
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))

    pipeline_steps = task.pipeline
    step_type = pipeline_steps[task.current_step].capability if pipeline_steps and task.current_step < len(pipeline_steps) else ""

    return {
        "task_id": task.task_id,
        "status": task.status.value,
        "current_step": task.current_step,
        "step_type": step_type,
        "percent": task.percent,
        "error": task.error,
        "completed_steps": [si.step_index for si in task.result_urls],
        "failed_steps": [se.step_index for se in task.step_errors],
        "completed_urls": [u.model_dump() for u in task.result_urls],
        "errors": [e.model_dump() for e in task.step_errors],
        "created_at": _ts(task.created_at),
        "started_at": _ts(task.started_at),
        "finished_at": _ts(task.finished_at),
    }


@router.delete("/tasks/{task_id}")
async def delete_task(task_id: str):
    deleted = await task_manager.delete_task(task_id)
    if not deleted:
        raise InputError(ErrorCode.TASK_NOT_FOUND, f"Task '{task_id}' not found")
    return {"status": "cancelled", "task_id": task_id, "message": "Task cancelled. Input and output files deleted."}
