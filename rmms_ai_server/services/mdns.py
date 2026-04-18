from __future__ import annotations

import logging
import socket
from typing import Optional

logger = logging.getLogger(__name__)


class MDNSService:
    def __init__(self, port: int = 8170, name: str = "RMMS AI Server"):
        self._port = port
        self._name = name
        self._zeroconf = None
        self._service_info = None

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
                "protocol": b"1.0.0-alpha",
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
