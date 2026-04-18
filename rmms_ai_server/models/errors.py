from __future__ import annotations

from enum import Enum
from typing import Any, Optional


class ErrorCode(str, Enum):
    INPUT_INVALID_FORMAT = "INPUT_INVALID_FORMAT"
    INPUT_FILE_TOO_LARGE = "INPUT_FILE_TOO_LARGE"
    INPUT_UNSUPPORTED_TYPE = "INPUT_UNSUPPORTED_TYPE"
    INPUT_MISSING = "INPUT_MISSING"
    INPUT_INVALID_PARAMS = "INPUT_INVALID_PARAMS"
    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
    MODEL_LOAD_FAILED = "MODEL_LOAD_FAILED"
    MODEL_INFER_FAILED = "MODEL_INFER_FAILED"
    DEVICE_NOT_AVAILABLE = "DEVICE_NOT_AVAILABLE"
    DEVICE_BUSY = "DEVICE_BUSY"
    DEVICE_ERROR = "DEVICE_ERROR"
    SERVER_INTERNAL = "SERVER_INTERNAL"
    SERVER_OVERLOADED = "SERVER_OVERLOADED"
    SERVER_TIMEOUT = "SERVER_TIMEOUT"
    QUOTA_EXCEEDED = "QUOTA_EXCEEDED"
    PIPELINE_STEP_FAILED = "PIPELINE_STEP_FAILED"
    PIPELINE_INVALID = "PIPELINE_INVALID"
    CAPABILITY_NOT_AVAILABLE = "CAPABILITY_NOT_AVAILABLE"
    TASK_NOT_FOUND = "TASK_NOT_FOUND"
    AUTH_INVALID_KEY = "AUTH_INVALID_KEY"
    AUTH_MISSING_KEY = "AUTH_MISSING_KEY"


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
        super().__init__(code, message, http_status=401, **kwargs)


class QuotaError(RMMSAIError):
    def __init__(self, code: ErrorCode, message: str, **kwargs):
        super().__init__(code, message, http_status=429, **kwargs)


class PipelineError(RMMSAIError):
    def __init__(self, code: ErrorCode, message: str, **kwargs):
        super().__init__(code, message, http_status=400, **kwargs)


class CapabilityError(RMMSAIError):
    def __init__(self, code: ErrorCode, message: str, **kwargs):
        super().__init__(code, message, http_status=501, **kwargs)
