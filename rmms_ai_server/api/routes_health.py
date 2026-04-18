from __future__ import annotations

import time

from fastapi import APIRouter

from rmms_ai_server import __version__, PROTOCOL_VERSION
from rmms_ai_server.config import settings
from rmms_ai_server.core.task_manager import task_manager
from rmms_ai_server.models.protocol import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="ok",
        version=f"{__version__} (protocol {PROTOCOL_VERSION})",
        uptime_seconds=round(time.time() - task_manager.start_time, 1),
        model_loaded=True,
        active_tasks=task_manager.active_count,
        queued_tasks=task_manager.queued_count,
    )
