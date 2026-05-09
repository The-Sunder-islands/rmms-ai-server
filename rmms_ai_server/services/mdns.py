from __future__ import annotations

import logging
import socket
from typing import Optional

from rmms_ai_server import PROTOCOL_VERSION

logger = logging.getLogger(__name__)


class MDNSService:
    def __init__(self, port: int = 8170, name: str = "RMMS AI Server"):
        self._port = port
        self._name = name
        self._zeroconf = None
        self._service_info = None

    @staticmethod
    def _get_available_devices() -> str:
        try:
            from rmms_ai_server.engine.device_backend import get_all_backends
            devices = [b.device_type for b in get_all_backends() if b.is_available()]
            return ",".join(devices) if devices else "cpu"
        except Exception:
            return "cpu"

    def start(self) -> None:
        try:
            from zeroconf import Zeroconf, ServiceInfo
        except ImportError:
            logger.warning("zeroconf not installed, mDNS disabled")
            return

        self._zeroconf = Zeroconf()

        try:
            hostname = socket.gethostname()
            local_ip = socket.gethostbyname(hostname)
        except Exception:
            local_ip = "0.0.0.0"

        self._service_info = ServiceInfo(
            type_="_rmms-ai._tcp.local.",
            name=f"{self._name}._rmms-ai._tcp.local.",
            addresses=[socket.inet_aton(local_ip)],
            port=self._port,
            properties={
                "protocol_version": PROTOCOL_VERSION.encode() if isinstance(PROTOCOL_VERSION, str) else str(PROTOCOL_VERSION).encode(),
                "devices": self._get_available_devices().encode(),
                "path": b"/api/v1",
            },
        )

        self._zeroconf.register_service(self._service_info)
        logger.info(f"mDNS registered: {self._name}._rmms-ai._tcp.local.:{self._port}")

    def stop(self) -> None:
        if self._zeroconf and self._service_info:
            try:
                self._zeroconf.unregister_service(self._service_info)
            except Exception:
                pass
            try:
                self._zeroconf.close()
            except Exception:
                pass
            logger.info("mDNS service unregistered")
