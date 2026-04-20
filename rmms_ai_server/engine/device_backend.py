from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from rmms_ai_server.models.protocol import DeviceInfo, DeviceUnit

logger = logging.getLogger(__name__)

_backends: dict[str, type[DeviceBackend]] = {}
_backends_lock = threading.Lock()
_instances: dict[str, DeviceBackend] = {}


@dataclass(frozen=True)
class SectionConfig:
    threshold: float
    section_duration: float
    max_sections: int


class DeviceBackend(ABC):
    @property
    @abstractmethod
    def device_type(self) -> str: ...

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def is_available(self) -> bool: ...

    @abstractmethod
    def device_count(self) -> int: ...

    @abstractmethod
    def get_device_info(self) -> DeviceInfo: ...

    @abstractmethod
    def acquire_device(self, preferred: Optional[int] = None) -> Optional[int]: ...

    @abstractmethod
    def release_device(self, device_id: int) -> None: ...

    @abstractmethod
    def get_torch_device_str(self, device_id: int) -> str: ...

    @abstractmethod
    def get_section_config(self) -> SectionConfig: ...


def register_backend(backend_cls: type[DeviceBackend]) -> type[DeviceBackend]:
    with _backends_lock:
        _backends[backend_cls.device_type.fget(None)] = backend_cls
    return backend_cls


def get_backend(device_type: str) -> Optional[DeviceBackend]:
    with _backends_lock:
        if device_type in _instances:
            return _instances[device_type]
        cls = _backends.get(device_type)
        if cls is None:
            return None
        instance = cls()
        _instances[device_type] = instance
        return instance


def get_all_backends() -> list[DeviceBackend]:
    result = []
    with _backends_lock:
        for dtype, cls in _backends.items():
            if dtype not in _instances:
                _instances[dtype] = cls()
            result.append(_instances[dtype])
    return result


_AUTO_PRIORITY = ["cuda", "dml", "npu", "xpu", "mps", "cpu"]


def resolve_device_type(device_type: str) -> str:
    if device_type != "auto":
        return device_type

    for dtype in _AUTO_PRIORITY:
        backend = get_backend(dtype)
        if backend is not None and backend.is_available():
            logger.info(f"Auto-detected device: {dtype}")
            return dtype

    return "cpu"


def _auto_discover():
    for module_name in [
        "rmms_ai_server.engine.cuda_backend",
        "rmms_ai_server.engine.dml_backend",
        "rmms_ai_server.engine.npu_backend",
        "rmms_ai_server.engine.xpu_backend",
        "rmms_ai_server.engine.mps_backend",
        "rmms_ai_server.engine.cpu_backend",
    ]:
        try:
            __import__(module_name)
        except Exception:
            pass


_auto_discover()
