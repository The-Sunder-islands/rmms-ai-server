from __future__ import annotations

from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


class CapabilityStatus(str, Enum):
    IMPLEMENTED = "implemented"
    NOT_IMPLEMENTED = "not_implemented"


class TrackType(str, Enum):
    AUDIO = "audio"
    MIDI = "midi"
    HYBRID = "hybrid"


class TaskStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    DONE = "done"
    PARTIAL_ERROR = "partial_error"
    ERROR = "error"
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
    INT = "int"
    FLOAT = "float"
    STRING = "string"
    BOOL = "bool"
    ENUM = "enum"
    MULTI_ENUM = "multi_enum"


class ParamDef(BaseModel):
    key: str
    type: ParamType
    label: str = ""
    description: str = ""
    default: Any = None
    required: bool = True
    choices: Optional[list[str]] = None
    options: Optional[list[dict[str, str]]] = None
    min_val: Optional[float] = Field(None, alias="min")
    max_val: Optional[float] = Field(None, alias="max")
    step: Optional[float] = None
    decimals: Optional[int] = None
    group: str = ""

    model_config = {"populate_by_name": True}


class DeviceUnit(BaseModel):
    device_index: int = 0
    name: str
    memory_total_mb: Optional[int] = None
    memory_used_mb: Optional[int] = None


class DeviceInfo(BaseModel):
    device_type: str = ""
    available: bool = False
    count: int = 0
    install_hint: Optional[str] = None
    units: list[DeviceUnit] = []


class SchedulerInfo(BaseModel):
    max_concurrent_tasks: int = 4
    max_queue_size: int = 20


class Capability(BaseModel):
    id: str
    label: str = ""
    description: str = ""
    status: CapabilityStatus = CapabilityStatus.IMPLEMENTED
    param_defs: list[ParamDef] = []
    models: list[str] = []
    default_model: Optional[str] = None


class CapabilitiesResponse(BaseModel):
    protocol_version: str
    server_version: str = ""
    capabilities: list[Capability]
    devices: list[DeviceInfo]
    scheduler: SchedulerInfo
    output_formats: list[str] = []
    output_packages: list[str] = []
    max_upload_bytes: int = 0


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = ""
    uptime_seconds: float = 0.0
    model_loaded: str = ""
    active_tasks: int = 0
    queued_tasks: int = 0


class StepInput(BaseModel):
    from_step: int
    stem: str = ""


class PipelineStep(BaseModel):
    capability: str = Field(alias="type")
    model: Optional[str] = None
    model_params: dict[str, Any] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)
    input: Optional[StepInput] = None

    model_config = {"populate_by_name": True}


class StepResultURL(BaseModel):
    step_index: int
    step_type: str
    urls: list[str] = []


class StepError(BaseModel):
    step_index: int
    step_type: str = ""
    error: dict[str, Any] = Field(default_factory=dict)


class ProgressSSEEvent(BaseModel):
    type: str = "progress"
    task_id: str = ""
    step_index: int = 0
    step_type: str = ""
    status: ProgressStatus = ProgressStatus.RUNNING
    percent: float = 0.0
    message: str = ""
    urls: list[str] = []
    error: Optional[dict[str, Any]] = None


class FinalResultEvent(BaseModel):
    type: str = "final_result"
    task_id: str = ""
    status: FinalStatus = FinalStatus.DONE
    urls: list[StepResultURL] = []
    errors: list[StepError] = []
    completed_steps: list[int] = []
    failed_steps: list[int] = []
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
    output_format: str = "wav"


class TaskListResponse(BaseModel):
    tasks: list[TaskInfo]
    total: int
