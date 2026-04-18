from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

from rmms_ai_server.config import settings
from rmms_ai_server.models.errors import InputError, ErrorCode

router = APIRouter()


@router.get("/files/{task_id}/{step_type}/{filename}")
async def download_file(task_id: str, step_type: str, filename: str):
    task_dir = settings.resolved_output_dir / task_id
    if not task_dir.is_dir():
        raise InputError(ErrorCode.INPUT_MISSING, f"Task '{task_id}' not found")

    for child in sorted(task_dir.iterdir()):
        if child.is_dir() and child.name.endswith(f"_{step_type}"):
            candidate = child / filename
            if candidate.is_file():
                file_path = candidate
                break
    else:
        candidate = task_dir / filename
        if candidate.is_file():
            file_path = candidate
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
