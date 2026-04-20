from __future__ import annotations

import logging
from typing import Optional

from rmms_ai_server.models.protocol import DeviceInfo, DeviceUnit
from .device_backend import DeviceBackend, SectionConfig, register_backend

logger = logging.getLogger(__name__)


@register_backend
class MPSBackend(DeviceBackend):
    @property
    def device_type(self) -> str:
        return "mps"

    @property
    def name(self) -> str:
        return "Apple Metal Performance Shaders"

    def is_available(self) -> bool:
        try:
            import torch
            return torch.backends.mps.is_available()
        except Exception:
            return False

    def device_count(self) -> int:
        return 1 if self.is_available() else 0

    def get_device_info(self) -> DeviceInfo:
        return DeviceInfo(
            type="mps", name="Apple MPS",
            available=self.is_available(),
            units=[DeviceUnit(id="0", name="Apple GPU", type="mps")] if self.is_available() else [],
        )

    def acquire_device(self, preferred: Optional[int] = None) -> Optional[int]:
        return 0 if self.is_available() else None

    def release_device(self, device_id: int) -> None:
        pass

    def get_torch_device_str(self, device_id: int) -> str:
        return "mps"

    def get_section_config(self) -> SectionConfig:
        return SectionConfig(threshold=300.0, section_duration=300.0, max_sections=0)
