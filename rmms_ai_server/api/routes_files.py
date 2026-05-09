from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

from rmms_ai_server.config import settings
from rmms_ai_server.models.errors import InputError, ErrorCode

router = APIRouter()


@router.get("/files/{task_id}/{filename}")
async def download_file(task_id: str, filename: str, step_type: str = None):
    task_dir = settings.resolved_output_dir / task_id
    if not task_dir.is_dir():
        raise InputError(ErrorCode.TASK_NOT_FOUND, f"Task '{task_id}' not found")

    file_path = None

    task_dir_resolved = task_dir.resolve()
    for child in sorted(task_dir_resolved.iterdir()):
        if child.is_dir() and child.name.startswith("step_"):
            candidate = child / filename
            candidate_resolved = candidate.resolve()
            if candidate_resolved.is_file() and str(candidate_resolved).startswith(str(task_dir_resolved)):
                file_path = candidate_resolved
                break
    if file_path is None:
        candidate = task_dir_resolved / filename
        candidate_resolved = candidate.resolve()
        if candidate_resolved.is_file() and str(candidate_resolved).startswith(str(task_dir_resolved)):
            file_path = candidate_resolved
        else:
            raise InputError(ErrorCode.INPUT_MISSING, f"File '{filename}' not found for task '{task_id}'")

    media_types = {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".flac": "audio/flac",
        ".mid": "audio/midi",
        ".midi": "audio/midi",
        ".json": "application/json",
        ".zip": "application/zip",
    }
    ext = os.path.splitext(filename)[1].lower()
    media_type = media_types.get(ext, "application/octet-stream")

    return FileResponse(
        path=str(file_path),
        media_type=media_type,
        filename=filename,
    )
