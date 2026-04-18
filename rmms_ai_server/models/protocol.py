from __future__ import annotations

from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


class CapabilityStatus(str, Enum):
    AVAILABLE = "available"
    NOT_IMPLEMENTED = "not_implemented"
    DEGRADED = "degraded"


class TrackType(str, Enum):
    AUDIO = "audio"
    MIDI = "midi"
    HYBRID = "hybrid"


class TaskStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class FinalStatus(str, Enum):
    DONE = "done"
    PARTIAL_ERROR = "partial_error"
    ERROR = "error"
    CANCELLED = "cancelled"


class ProgressStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ParamType(str, Enum):
    INTEGER = "integer"
    FLOAT = "float"
    STRING = "string"
    BOOLEAN = "boolean"
    ENUM = "enum"
    ARRAY = "array"


class ParamDef(BaseModel):
    key: str
    type: ParamType
    label: str = ""
    description: str = ""
    default: Any = None
    required: bool = True
    choices: Optional[list[str]] = None
    min_val: Optional[float] = Field(None, alias="min")
    max_val: Optional[float] = Field(None, alias="max")
    step: Optional[float] = None

    model_config = {"populate_by_name": True}


class DeviceUnit(BaseModel):
    id: str
    name: str
    type: str
    memory_total_mb: Optional[int] = None
    memory_used_mb: Optional[int] = None
    utilization_pct: Optional[float] = None


class DeviceInfo(BaseModel):
    type: str
    name: str
    available: bool
    units: list[DeviceUnit] = []


class SchedulerInfo(BaseModel):
    type: str = "fifo"
    max_concurrent: int = 4


class OutputFormat(BaseModel):
    format: str
    packaging: str = "separate"


class Capability(BaseModel):
    id: str
    name: str
    description: str = ""
    status: CapabilityStatus = CapabilityStatus.AVAILABLE
    input_types: list[str] = []
    output_types: list[TrackType] = []
    param_defs: list[ParamDef] = []
    models: list[str] = []
    default_model: str = ""


class CapabilitiesResponse(BaseModel):
    protocol_version: str
    capabilities: list[Capability]
    devices: list[DeviceInfo]
    scheduler: SchedulerInfo
    output_formats: list[OutputFormat]


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = ""
    uptime_seconds: float = 0.0
    model_loaded: bool = False
    active_tasks: int = 0
    queued_tasks: int = 0


class StepInput(BaseModel):
    from_step: int
    stem: str = ""


class PipelineStep(BaseModel):
    capability: str
    params: dict[str, Any] = {}
    input: Optional[StepInput] = None


class StepResultURL(BaseModel):
    step_index: int
    step_type: str
    urls: list[str] = []


class StepError(BaseModel):
    step_index: int
    step_type: str
    error: str = ""


class ProgressRunning(BaseModel):
    status: ProgressStatus = ProgressStatus.RUNNING
    percent: float = 0.0
    message: str = ""


class ProgressCompleted(BaseModel):
    status: ProgressStatus = ProgressStatus.COMPLETED
    urls: list[StepResultURL] = []


class ProgressFailed(BaseModel):
    status: ProgressStatus = ProgressStatus.FAILED
    error: StepError


ProgressEvent = ProgressRunning | ProgressCompleted | ProgressFailed


class PartialResultEvent(BaseModel):
    type: str = "partial_result"
    task_id: str = ""
    step_index: int = 0
    step_type: str = ""
    data: dict[str, Any] = {}


class ProgressSSEEvent(BaseModel):
    type: str = "progress"
    task_id: str = ""
    step_index: int = 0
    step_type: str = ""
    status: ProgressStatus = ProgressStatus.RUNNING
    percent: float = 0.0
    message: str = ""
    urls: list[StepResultURL] = []
    error: Optional[StepError] = None


class FinalResultEvent(BaseModel):
    type: str = "final_result"
    task_id: str = ""
    status: FinalStatus = FinalStatus.DONE
    urls: list[StepResultURL] = []
    errors: list[StepError] = []
    message: str = ""


class TaskInfo(BaseModel):
    task_id: str
    status: TaskStatus = TaskStatus.QUEUED
    pipeline: list[PipelineStep] = []
    created_at: float = 0.0
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    current_step: int = 0
    percent: float = 0.0
    error: Optional[str] = None
    result_urls: list[StepResultURL] = []
    step_errors: list[StepError] = []


class TaskListResponse(BaseModel):
    tasks: list[TaskInfo]
    total: int
