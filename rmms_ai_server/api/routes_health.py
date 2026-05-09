from __future__ import annotations

import time

from fastapi import APIRouter

from rmms_ai_server import __version__
from rmms_ai_server.core.task_manager import task_manager
from rmms_ai_server.models.protocol import HealthResponse

router = APIRouter()


def _get_loaded_model() -> str:
    try:
        from rmms_ai_server.engine.split_engine import get_loaded_models
        models = get_loaded_models()
        if models:
            return models[0]
    except Exception:
        pass
    return "none"


@router.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="ok",
        version=__version__,
        uptime_seconds=round(time.time() - task_manager.start_time, 1),
        model_loaded=_get_loaded_model(),
        active_tasks=task_manager.active_count,
        queued_tasks=task_manager.queued_count,
    )
