from __future__ import annotations

import logging
import threading
from typing import Optional

from rmms_ai_server.models.protocol import DeviceInfo, DeviceUnit
from .device_backend import DeviceBackend, SectionConfig, register_backend

logger = logging.getLogger(__name__)

_DML_NEW_API = None


def _detect_api():
    global _DML_NEW_API
    if _DML_NEW_API is not None:
        return _DML_NEW_API
    try:
        import torch
        if hasattr(torch, "dml") and hasattr(torch.dml, "is_available"):
            _DML_NEW_API = True
            return True
    except Exception:
        pass
    try:
        import torch_directml
        _DML_NEW_API = False
        return False
    except Exception:
        pass
    _DML_NEW_API = False
    return False


@register_backend
class DMLBackend(DeviceBackend):
    _lock = threading.Lock()
    _available: dict[int, bool] = {}

    @property
    def device_type(self) -> str:
        return "dml"

    @property
    def name(self) -> str:
        return "DirectML"

    def is_available(self) -> bool:
        try:
            if _detect_api():
                import torch
                return torch.dml.is_available()
            else:
                import torch_directml
                return torch_directml.device_count() > 0
        except Exception:
            return False

    def device_count(self) -> int:
        try:
            if _detect_api():
                import torch
                return torch.dml.device_count()
            else:
                import torch_directml
                return torch_directml.device_count()
        except Exception:
            return 0

    def get_device_info(self) -> DeviceInfo:
        count = self.device_count()
        units = []
        for i in range(count):
            try:
                if _detect_api():
                    import torch
                    dev_name = torch.dml.device_name(i)
                else:
                    dev_name = f"DirectML Device {i}"
                units.append(DeviceUnit(id=str(i), name=dev_name, type="dml"))
            except Exception:
                units.append(DeviceUnit(id=str(i), name=f"DML:{i}", type="dml"))
        return DeviceInfo(
            type="dml", name="DirectML",
            available=count > 0, units=units,
        )

    def acquire_device(self, preferred: Optional[int] = None) -> Optional[int]:
        count = self.device_count()
        if count == 0:
            return None
        with self._lock:
            for i in range(count):
                if i not in self._available:
                    self._available[i] = True
            if preferred is not None:
                if 0 <= preferred < count and self._available.get(preferred, True):
                    self._available[preferred] = False
                    return preferred
                return None
            for i in range(count):
                if self._available.get(i, True):
                    self._available[i] = False
                    return i
        return None

    def release_device(self, device_id: int) -> None:
        with self._lock:
            self._available[device_id] = True

    def get_torch_device_str(self, device_id: int) -> str:
        if _detect_api():
            return f"dml:{device_id}"
        return f"privateuseone:{device_id}"

    def get_section_config(self) -> SectionConfig:
        return SectionConfig(threshold=150.0, section_duration=150.0, max_sections=0)
