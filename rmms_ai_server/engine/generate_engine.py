from __future__ import annotations

from rmms_ai_server.models.errors import CapabilityError, ErrorCode


def run_generate(**kwargs):
    raise CapabilityError(
        ErrorCode.CAPABILITY_NOT_IMPLEMENTED,
        "generate capability is not yet implemented",
    )
