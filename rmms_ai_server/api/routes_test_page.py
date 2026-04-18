from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()

_TEST_HTML_PATH = Path(__file__).parent.parent / "static" / "test.html"


@router.get("/", response_class=HTMLResponse)
async def test_page():
    if _TEST_HTML_PATH.is_file():
        return _TEST_HTML_PATH.read_text(encoding="utf-8")
    return HTMLResponse("<h1>RMMS AI Server</h1><p>Test page not found.</p>", status_code=404)
