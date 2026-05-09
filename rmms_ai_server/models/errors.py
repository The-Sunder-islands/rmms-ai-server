from __future__ import annotations

from enum import Enum
from typing import Any, Optional


class ErrorCode(str, Enum):
    INPUT_FORMAT_UNSUPPORTED = "INPUT_FORMAT_UNSUPPORTED"
    INPUT_TOO_LARGE = "INPUT_TOO_LARGE"
    INPUT_CORRUPT = "INPUT_CORRUPT"
    INPUT_SAMPLERATE_UNSUPPORTED = "INPUT_SAMPLERATE_UNSUPPORTED"
    INPUT_CHANNEL_UNSUPPORTED = "INPUT_CHANNEL_UNSUPPORTED"
    INPUT_MISSING = "INPUT_MISSING"
    INPUT_INVALID_PARAMS = "INPUT_INVALID_PARAMS"
    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
    MODEL_LOAD_FAILED = "MODEL_LOAD_FAILED"
    MODEL_INFER_FAILED = "MODEL_INFER_FAILED"
    MODEL_INCOMPATIBLE = "MODEL_INCOMPATIBLE"
    DEVICE_UNAVAILABLE = "DEVICE_UNAVAILABLE"
    DEVICE_OOM = "DEVICE_OOM"
    DEVICE_TIMEOUT = "DEVICE_TIMEOUT"
    DEVICE_ERROR = "DEVICE_ERROR"
    SERVER_BUSY = "SERVER_BUSY"
    SERVER_ERROR = "SERVER_ERROR"
    SERVER_CONFIG_ERROR = "SERVER_CONFIG_ERROR"
    SERVER_INTERNAL = "SERVER_INTERNAL"
    QUOTA_EXCEEDED = "QUOTA_EXCEEDED"
    QUOTA_RATE_LIMIT = "QUOTA_RATE_LIMIT"
    PIPELINE_INVALID = "PIPELINE_INVALID"
    PIPELINE_CIRCULAR_DEPENDENCY = "PIPELINE_CIRCULAR_DEPENDENCY"
    PIPELINE_MISSING_INPUT = "PIPELINE_MISSING_INPUT"
    CAPABILITY_NOT_IMPLEMENTED = "CAPABILITY_NOT_IMPLEMENTED"
    TASK_NOT_FOUND = "TASK_NOT_FOUND"
    AUTH_REQUIRED = "AUTH_REQUIRED"
    AUTH_INVALID = "AUTH_INVALID"


class RMMSAIError(Exception):
    def __init__(
        self,
        code: ErrorCode,
        message: str,
        details: Optional[dict[str, Any]] = None,
        http_status: int = 500,
    ):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}
        self.http_status = http_status

    def to_dict(self) -> dict:
        return {
            "code": self.code.value,
            "message": self.message,
            "details": self.details,
        }


class InputError(RMMSAIError):
    def __init__(self, code: ErrorCode, message: str, **kwargs):
        super().__init__(code, message, http_status=400, **kwargs)


class ModelError(RMMSAIError):
    def __init__(self, code: ErrorCode, message: str, **kwargs):
        super().__init__(code, message, http_status=500, **kwargs)


class DeviceError(RMMSAIError):
    def __init__(self, code: ErrorCode, message: str, **kwargs):
        super().__init__(code, message, http_status=503, **kwargs)


class ServerError(RMMSAIError):
    def __init__(self, code: ErrorCode, message: str, **kwargs):
        super().__init__(code, message, http_status=500, **kwargs)


class AuthError(RMMSAIError):
    def __init__(self, code: ErrorCode, message: str, **kwargs):
        status = 403 if code == ErrorCode.AUTH_INVALID else 401
        super().__init__(code, message, http_status=status, **kwargs)


class QuotaError(RMMSAIError):
    def __init__(self, code: ErrorCode, message: str, **kwargs):
        super().__init__(code, message, http_status=429, **kwargs)


class PipelineError(RMMSAIError):
    def __init__(self, code: ErrorCode, message: str, **kwargs):
        super().__init__(code, message, http_status=400, **kwargs)


class CapabilityError(RMMSAIError):
    def __init__(self, code: ErrorCode, message: str, **kwargs):
        super().__init__(code, message, http_status=501, **kwargs)
