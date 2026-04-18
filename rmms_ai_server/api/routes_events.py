from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from rmms_ai_server.core.task_manager import task_manager
from rmms_ai_server.core.sse_manager import sse_manager
from rmms_ai_server.models.errors import InputError, ErrorCode

router = APIRouter()


@router.get("/tasks/{task_id}/events")
async def task_events(task_id: str, request: Request):
    task = task_manager.get_task(task_id)
    if task is None:
        raise InputError(ErrorCode.TASK_NOT_FOUND, f"Task '{task_id}' not found")

    async def event_generator():
        queue = sse_manager.subscribe(task_id)
        try:
            while True:
                if await request.is_disconnected():
                    break

                try:
                    event_data = await asyncio.wait_for(queue.get(), timeout=30.0)
                except asyncio.TimeoutError:
                    yield f": keepalive\n\n"
                    continue

                event_type = event_data.get("type", "message")
                data = json.dumps(event_data, ensure_ascii=False)
                yield f"event: {event_type}\ndata: {data}\n\n"

                if event_type == "final_result":
                    break
        finally:
            sse_manager.unsubscribe(task_id, queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
