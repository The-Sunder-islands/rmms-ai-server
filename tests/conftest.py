from __future__ import annotations

import io
import wave
import struct
import math
import pytest
from typing import Generator
from fastapi.testclient import TestClient
from pathlib import Path

from rmms_ai_server.app import create_app
from rmms_ai_server.config import settings
from rmms_ai_server.core.task_manager import task_manager
from rmms_ai_server.core.cache_manager import cache_manager
from rmms_ai_server.core.sse_manager import sse_manager


def generate_sine_wav(
    duration: float = 1.0,
    sample_rate: int = 44100,
    frequency: float = 440.0,
    channels: int = 1,
) -> bytes:
    num_samples = int(duration * sample_rate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        for i in range(num_samples):
            value = int(32767 * 0.3 * math.sin(2 * math.pi * frequency * i / sample_rate))
            wf.writeframes(struct.pack("<h", value))
    return buf.getvalue()


@pytest.fixture(autouse=True)
def reset_singletons():
    cache_manager.clear()
    task_manager._tasks.clear()
    task_manager._active_count = 0
    task_manager._cancel_events.clear()
    sse_manager._subscribers.clear()
    yield


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    settings.mdns_enabled = False
    app = create_app()
    with TestClient(app) as c:
        yield c


@pytest.fixture
def sample_wav() -> bytes:
    return generate_sine_wav(duration=0.5, frequency=440.0)


@pytest.fixture
def sample_wav_stereo() -> bytes:
    return generate_sine_wav(duration=0.5, frequency=440.0, channels=2)


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    d = tmp_path / "rmms_test"
    d.mkdir()
    return d
