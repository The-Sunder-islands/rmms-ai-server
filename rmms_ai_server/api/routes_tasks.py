from __future__ import annotations

import os
import json
import time
import urllib.request
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, UploadFile, Query, Request, Body
from fastapi.responses import JSONResponse

from rmms_ai_server.config import settings
from rmms_ai_server.models.protocol import TaskStatus, PipelineStep
from rmms_ai_server.models.errors import InputError, ErrorCode, RMMSAIError, QuotaError
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


def _save_upload(content: bytes, filename: str) -> tuple[str, str]:
    task_id_placeholder = os.urandom(8).hex()
    upload_dir = str(settings.resolved_upload_dir / task_id_placeholder)
    Path(upload_dir).mkdir(parents=True, exist_ok=True)
    input_path = os.path.join(upload_dir, filename)
    with open(input_path, "wb") as f:
        f.write(content)
    return input_path, upload_dir


def _download_file(url: str) -> bytes:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "RMMS-AI-Server/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            if resp.status != 200:
                raise InputError(ErrorCode.INPUT_FORMAT_UNSUPPORTED, f"Failed to download file: HTTP {resp.status}")
            size = resp.headers.get("Content-Length")
            if size and int(size) > settings.max_upload_bytes:
                raise InputError(ErrorCode.INPUT_TOO_LARGE,
                                 f"Remote file too large ({int(size) // (1024*1024)}MB). Maximum: {settings.max_upload_mb}MB")
            return resp.read()
    except InputError:
        raise
    except Exception as e:
        raise InputError(ErrorCode.INPUT_FORMAT_UNSUPPORTED, f"Failed to download file: {e}")


@router.post("/tasks")
async def submit_task(request: Request):
    content_type = request.headers.get("content-type", "")

    if "application/json" in content_type:
        return await _submit_json(request)
    elif "multipart/form-data" in content_type:
        return await _submit_multipart(request)
    else:
        raise InputError(ErrorCode.INPUT_FORMAT_UNSUPPORTED,
                         f"Unsupported Content-Type: {content_type}. Use multipart/form-data or application/json")


async def _submit_json(request: Request):
    body = await request.json()
    try:
        data = TaskSubmitJSON(**body)
    except Exception as e:
        raise InputError(ErrorCode.PIPELINE_INVALID, f"Invalid request body: {e}")

    if data.pipeline:
        pipeline_raw = data.pipeline
        if isinstance(pipeline_raw, dict) and "steps" in pipeline_raw:
            resolved = resolve_pipeline(pipeline_raw["steps"], None)
        elif isinstance(pipeline_raw, list):
            resolved = resolve_pipeline(pipeline_raw, None)
        else:
            raise InputError(ErrorCode.PIPELINE_INVALID, "Invalid pipeline format, expected {steps: [...]}")
    else:
        resolved = resolve_pipeline(None, None)

    if not data.input_url:
        if not resolved or resolved[0].input is not None:
            raise InputError(ErrorCode.INPUT_MISSING, "No input_url provided and first step has no input")
    input_path = None
    if data.input_url:
        content = _download_file(data.input_url)
        filename = os.path.basename(data.input_url.split("?")[0].split("/")[-1]) or "input.wav"
        if not _allowed_file(filename):
            raise InputError(ErrorCode.INPUT_FORMAT_UNSUPPORTED,
                             f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}")
        if len(content) > settings.max_upload_bytes:
            raise InputError(ErrorCode.INPUT_TOO_LARGE,
                             f"File too large ({len(content) // (1024*1024)}MB). Maximum: {settings.max_upload_mb}MB")
        input_path, _ = _save_upload(content, filename)

    file_hash = cache_manager.compute_file_hash(input_path) if input_path else None

    if not data.force_refresh and file_hash:
        params_key = {"pipeline": [s.model_dump(by_alias=True) for s in resolved],
                      "device": data.device_preference,
                      "output_format": data.output_format,
                      "output_package": data.output_package}
        cached = cache_manager.get(file_hash, params_key)
        if cached:
            return TaskSubmitResponse(
                task_id=cached["task_id"], status="done", message="Cached result",
                cached=True, created_at=cached.get("created_at", ""),
            )

    try:
        task = await task_manager.create_task(
            pipeline=resolved, input_path=input_path,
            device_preference=data.device_preference,
            priority=data.priority,
            output_format=data.output_format or "wav",
        )
    except QuotaError:
        raise
    except Exception as e:
        raise InputError(ErrorCode.SERVER_ERROR, str(e))

    if file_hash:
        params_key = {"pipeline": [s.model_dump(by_alias=True) for s in resolved],
                      "device": data.device_preference,
                      "output_format": data.output_format,
                      "output_package": data.output_package}
        cache_manager.put(file_hash, params_key, {"task_id": task.task_id})

    return JSONResponse(
        status_code=201,
        content=TaskSubmitResponse(
            task_id=task.task_id, status="queued", message="Task submitted successfully",
            pipeline=resolved,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(task.created_at)),
        ).model_dump(by_alias=True),
    )


async def _submit_multipart(request: Request):
    form = await request.form()
    file = form.get("file")
    pipeline_str = form.get("pipeline")
    preset = form.get("preset")
    device_preference = form.get("device_preference")
    priority = form.get("priority")
    force_refresh = form.get("force_refresh", "false") == "true"
    output_format = form.get("output_format")
    output_package = form.get("output_package")

    if isinstance(priority, str):
        try:
            priority = int(priority)
        except (ValueError, TypeError):
            priority = None

    if file is None:
        raise InputError(ErrorCode.INPUT_MISSING, "No audio file provided")

    filename = getattr(file, "filename", "")
    if not _allowed_file(filename):
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

    input_path, _ = _save_upload(content, filename)
    file_hash = cache_manager.compute_file_hash(input_path)

    pipeline_steps = None
    if pipeline_str:
        try:
            pipeline_steps = json.loads(pipeline_str)
        except json.JSONDecodeError:
            raise InputError(ErrorCode.INPUT_INVALID_PARAMS, "Invalid pipeline JSON")

    resolved = resolve_pipeline(pipeline_steps, preset)

    if not force_refresh and file_hash:
        params_key = {"preset": preset, "pipeline": pipeline_steps, "device": device_preference,
                      "output_format": output_format, "output_package": output_package}
        cached = cache_manager.get(file_hash, params_key)
        if cached:
            return TaskSubmitResponse(
                task_id=cached["task_id"], status="done", message="Cached result",
                cached=True, created_at=cached.get("created_at", ""),
            )

    try:
        task = await task_manager.create_task(
            pipeline=resolved, input_path=input_path,
            device_preference=device_preference, priority=priority,
            output_format=output_format or "wav",
        )
    except QuotaError:
        raise
    except Exception as e:
        raise InputError(ErrorCode.SERVER_ERROR, str(e))

    if file_hash:
        params_key = {"preset": preset, "pipeline": pipeline_steps, "device": device_preference,
                      "output_format": output_format, "output_package": output_package}
        cache_manager.put(file_hash, params_key, {"task_id": task.task_id})

    return JSONResponse(
        status_code=201,
        content=TaskSubmitResponse(
            task_id=task.task_id, status="queued", message="Task submitted successfully",
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
