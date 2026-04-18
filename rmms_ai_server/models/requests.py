from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field

from .protocol import PipelineStep


class TaskSubmitJSON(BaseModel):
    pipeline: list[PipelineStep]
    device_preference: Optional[str] = None
    priority: Optional[int] = Field(None, ge=1, le=10)
    force_refresh: bool = False
    callback_url: Optional[str] = None


class TaskSubmitResponse(BaseModel):
    task_id: str
    status: str = "queued"
    message: str = ""


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    current_step: int = 0
    percent: float = 0.0
    error: Optional[str] = None
    result_urls: list[dict[str, Any]] = []
    step_errors: list[dict[str, Any]] = []
