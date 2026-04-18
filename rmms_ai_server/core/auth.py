from __future__ import annotations

from fastapi import Request

from rmms_ai_server.config import settings
from rmms_ai_server.models.errors import AuthError, ErrorCode


async def verify_api_key(request: Request) -> None:
    if not settings.auth_enabled:
        return

    api_key = request.headers.get("X-API-Key", "")
    if not api_key:
        raise AuthError(ErrorCode.AUTH_MISSING_KEY, "API key required. Set X-API-Key header.")

    if api_key != settings.api_key:
        raise AuthError(ErrorCode.AUTH_INVALID_KEY, "Invalid API key.")
