from __future__ import annotations

import argparse
import logging
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="rmms-ai-server",
        description="RMMS AI Server - Reference implementation of RMMS AI Protocol",
    )
    parser.add_argument("--host", default=None, help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=None, help="Bind port (default: 8170)")
    parser.add_argument("--workers", type=int, default=1, help="Uvicorn workers (default: 1)")
    parser.add_argument("--log-level", default=None,
                        choices=["debug", "info", "warning", "error"],
                        help="Log level (default: info)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (dev only)")
    parser.add_argument("--no-mdns", action="store_true", help="Disable mDNS broadcasting")
    parser.add_argument("--no-auth", action="store_true", help="Disable API key authentication")

    args = parser.parse_args()

    from rmms_ai_server.config import settings

    host = args.host or settings.host
    port = args.port or settings.port
    log_level = (args.log_level or settings.log_level).lower()

    if args.no_mdns:
        settings.mdns_enabled = False
    if args.no_auth:
        settings.api_key = ""

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    import uvicorn
    uvicorn.run(
        "rmms_ai_server.app:create_app",
        host=host,
        port=port,
        workers=args.workers if not args.reload else 1,
        factory=True,
        reload=args.reload,
        log_level=log_level,
    )


if __name__ == "__main__":
    main()
