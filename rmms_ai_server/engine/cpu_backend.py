from __future__ import annotations

import logging
import os
from typing import Optional

from rmms_ai_server.models.protocol import DeviceInfo, DeviceUnit
from .device_backend import DeviceBackend, register_backend

logger = logging.getLogger(__name__)


@register_backend
class CPUBackend(DeviceBackend):
    @property
    def device_type(self) -> str:
        return "cpu"

    @property
    def name(self) -> str:
        return "CPU"

    def is_available(self) -> bool:
        return True

    def device_count(self) -> int:
        return 1

    def get_device_info(self) -> DeviceInfo:
        cores = os.cpu_count() or 1
        return DeviceInfo(
            type="cpu", name="CPU", available=True,
            units=[DeviceUnit(id="0", name=f"CPU ({cores} cores)", type="cpu")],
        )

    def acquire_device(self, preferred: Optional[int] = None) -> Optional[int]:
        return 0

    def release_device(self, device_id: int) -> None:
        pass

    def get_torch_device_str(self, device_id: int) -> str:
        return "cpu"
