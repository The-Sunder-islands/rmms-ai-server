from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field

from .protocol import PipelineStep, StepResultURL, StepError


class TaskSubmitJSON(BaseModel):
    input_url: Optional[str] = None
    pipeline: Optional[dict[str, Any]] = None
    device_preference: Optional[str] = None
    device_index: Optional[int] = None
    priority: Optional[int] = Field(None, ge=1, le=10)
    output_format: Optional[str] = None
    output_package: Optional[str] = None
    force_refresh: bool = False
    callback_url: Optional[str] = None


class TaskSubmitResponse(BaseModel):
    task_id: str
    status: str = "queued"
    message: str = ""
    pipeline: Optional[list[PipelineStep]] = Field(None, alias="pipeline")
    cached: bool = False
    created_at: str = ""

    model_config = {"populate_by_name": True}


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    current_step: int = 0
    step_type: str = ""
    percent: float = 0.0
    error: Optional[str] = None
    completed_steps: list[int] = []
    failed_steps: list[int] = []
    completed_urls: list[StepResultURL] = []
    errors: list[StepError] = []
    result_urls: list[dict[str, Any]] = []
    step_errors: list[dict[str, Any]] = []
    created_at: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
