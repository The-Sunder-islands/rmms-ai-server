from __future__ import annotations

import logging
import threading
from typing import Optional

from rmms_ai_server.models.protocol import DeviceInfo, DeviceUnit
from .device_backend import DeviceBackend, register_backend

logger = logging.getLogger(__name__)


@register_backend
class XPUBackend(DeviceBackend):
    _lock = threading.Lock()
    _available: dict[int, bool] = {}

    @property
    def device_type(self) -> str:
        return "xpu"

    @property
    def name(self) -> str:
        return "Intel XPU"

    def is_available(self) -> bool:
        try:
            import torch
            return hasattr(torch, "xpu") and torch.xpu.is_available()
        except Exception:
            return False

    def device_count(self) -> int:
        try:
            import torch
            if hasattr(torch, "xpu"):
                return torch.xpu.device_count()
        except Exception:
            pass
        return 0

    def get_device_info(self) -> DeviceInfo:
        count = self.device_count()
        units = []
        for i in range(count):
            try:
                import torch
                name = torch.xpu.get_device_name(i)
                units.append(DeviceUnit(id=str(i), name=name, type="xpu"))
            except Exception:
                units.append(DeviceUnit(id=str(i), name=f"XPU:{i}", type="xpu"))
        return DeviceInfo(
            type="xpu", name="Intel XPU",
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
        return f"xpu:{device_id}"
