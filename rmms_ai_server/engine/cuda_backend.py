from __future__ import annotations

import logging
import threading
from typing import Optional

from rmms_ai_server.models.protocol import DeviceInfo, DeviceUnit
from .device_backend import DeviceBackend, register_backend

logger = logging.getLogger(__name__)


@register_backend
class CUDABackend(DeviceBackend):
    _lock = threading.Lock()
    _available: dict[int, bool] = {}

    @property
    def device_type(self) -> str:
        return "cuda"

    @property
    def name(self) -> str:
        return "CUDA GPU"

    def is_available(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    def device_count(self) -> int:
        try:
            import torch
            return torch.cuda.device_count()
        except Exception:
            return 0

    def get_device_info(self) -> DeviceInfo:
        count = self.device_count()
        units = []
        for i in range(count):
            try:
                import torch
                name = torch.cuda.get_device_name(i)
                props = torch.cuda.get_device_properties(i)
                total_mb = props.total_memory // (1024 * 1024)
                used_mb = torch.cuda.memory_allocated(i) // (1024 * 1024)
                units.append(DeviceUnit(
                    id=str(i), name=name, type="cuda",
                    memory_total_mb=total_mb, memory_used_mb=used_mb,
                ))
            except Exception:
                units.append(DeviceUnit(id=str(i), name=f"CUDA:{i}", type="cuda"))
        return DeviceInfo(
            type="cuda", name="CUDA GPU",
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
        return f"cuda:{device_id}"
