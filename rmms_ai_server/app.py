from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from rmms_ai_server import __version__, PROTOCOL_VERSION
from rmms_ai_server.config import settings
from starlette.middleware.cors import CORSMiddleware
from rmms_ai_server.models.errors import RMMSAIError
from rmms_ai_server.core.auth import verify_api_key
from rmms_ai_server.api.routes_health import router as health_router
from rmms_ai_server.api.routes_capabilities import router as capabilities_router
from rmms_ai_server.api.routes_tasks import router as tasks_router
from rmms_ai_server.api.routes_events import router as events_router
from rmms_ai_server.api.routes_files import router as files_router
from rmms_ai_server.api.routes_test_page import router as test_page_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"RMMS AI Server v{__version__} (protocol {PROTOCOL_VERSION}) starting...")
    logger.info(f"Upload dir: {settings.resolved_upload_dir}")
    logger.info(f"Output dir: {settings.resolved_output_dir}")
    logger.info(f"Model cache dir: {settings.resolved_model_cache_dir}")
    logger.info(f"Auth: {'enabled' if settings.auth_enabled else 'disabled'}")
    logger.info(f"Max concurrent tasks: {settings.max_concurrent_tasks}")

    settings.resolved_upload_dir.mkdir(parents=True, exist_ok=True)
    settings.resolved_output_dir.mkdir(parents=True, exist_ok=True)
    settings.resolved_model_cache_dir.mkdir(parents=True, exist_ok=True)

    mdns_service = None
    if settings.mdns_enabled:
        try:
            from rmms_ai_server.services.mdns import MDNSService
            mdns_service = MDNSService(
                port=settings.port,
                name=settings.mdns_name,
            )
            mdns_service.start()
            logger.info(f"mDNS broadcasting as {settings.mdns_name}._rmms-ai._tcp")
        except Exception as e:
            logger.warning(f"mDNS start failed: {e}")

    yield

    if mdns_service:
        try:
            mdns_service.stop()
        except Exception:
            pass

    logger.info("RMMS AI Server shutting down")


def create_app() -> FastAPI:
    app = FastAPI(
        title="RMMS AI Server",
        description="Reference implementation of RMMS AI Protocol",
        version=__version__,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_router, prefix="/api/v1", tags=["Health"])
    app.include_router(capabilities_router, prefix="/api/v1", tags=["Capabilities"])
    app.include_router(tasks_router, prefix="/api/v1", tags=["Tasks"])
    app.include_router(events_router, prefix="/api/v1", tags=["Events"])
    app.include_router(files_router, prefix="/api/v1", tags=["Files"])
    app.include_router(test_page_router, tags=["Test Page"])

    @app.exception_handler(RMMSAIError)
    async def rmms_error_handler(request: Request, exc: RMMSAIError):
        return JSONResponse(
            status_code=exc.http_status,
            content=exc.to_dict(),
        )

    @app.middleware("http")
    async def auth_middleware(request: Request, call_next):
        if request.url.path.startswith("/api/v1"):
            try:
                await verify_api_key(request)
            except RMMSAIError as e:
                return JSONResponse(status_code=e.http_status, content=e.to_dict())
        response = await call_next(request)
        return response

    return app
