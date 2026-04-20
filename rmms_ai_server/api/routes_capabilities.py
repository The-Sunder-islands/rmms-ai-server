from __future__ import annotations

from fastapi import APIRouter

from rmms_ai_server import PROTOCOL_VERSION
from rmms_ai_server.config import settings
from rmms_ai_server.engine.device_backend import get_all_backends
from rmms_ai_server.models.protocol import (
    CapabilitiesResponse, Capability, CapabilityStatus, TrackType,
    ParamDef, ParamType, DeviceInfo, SchedulerInfo, OutputFormat,
)

router = APIRouter()


def _build_capabilities() -> list[Capability]:
    return [
        Capability(
            id="split",
            name="Stem Separation",
            description="Separate audio into individual stems using Demucs AI models",
            status=CapabilityStatus.AVAILABLE,
            input_types=["audio"],
            output_types=[TrackType.AUDIO],
            param_defs=[
                ParamDef(key="stem_count", type=ParamType.ENUM, label="Stem Count",
                         default="6", choices=["4", "6"]),
                ParamDef(key="model", type=ParamType.ENUM, label="Model",
                         default="htdemucs_6s", choices=["htdemucs", "htdemucs_6s", "mdx_extra_q"]),
                ParamDef(key="shifts", type=ParamType.INTEGER, label="Shifts",
                         default=1, min_val=1, max_val=20),
                ParamDef(key="overlap", type=ParamType.FLOAT, label="Overlap",
                         default=0.17, min_val=0.0, max_val=0.99, step=0.01),
                ParamDef(key="device_type", type=ParamType.ENUM, label="Device",
                         default="auto", choices=["auto", "cuda", "dml", "npu", "xpu", "mps", "cpu"]),
            ],
            models=["htdemucs", "htdemucs_6s", "mdx_extra_q"],
            default_model="htdemucs_6s",
        ),
        Capability(
            id="midi",
            name="MIDI Transcription",
            description="Transcribe audio to MIDI using basic-pitch",
            status=CapabilityStatus.AVAILABLE,
            input_types=["audio"],
            output_types=[TrackType.MIDI],
            param_defs=[
                ParamDef(key="onset_threshold", type=ParamType.FLOAT, label="Onset Threshold",
                         default=0.6, min_val=0.0, max_val=1.0, step=0.05),
                ParamDef(key="frame_threshold", type=ParamType.FLOAT, label="Frame Threshold",
                         default=0.3, min_val=0.0, max_val=1.0, step=0.05),
            ],
            models=["basic-pitch"],
            default_model="basic-pitch",
        ),
        Capability(
            id="detect",
            name="Note Detection",
            description="Detect note events from audio using AutoSong pipeline",
            status=CapabilityStatus.AVAILABLE,
            input_types=["audio"],
            output_types=[TrackType.MIDI],
            param_defs=[
                ParamDef(key="instrument_id", type=ParamType.INTEGER, label="Instrument ID",
                         default=0, min_val=0, max_val=36),
                ParamDef(key="scale_type", type=ParamType.ENUM, label="Scale",
                         default="0", choices=["0", "1", "2", "3", "4", "5"]),
                ParamDef(key="scale_root", type=ParamType.INTEGER, label="Scale Root",
                         default=0, min_val=0, max_val=11),
                ParamDef(key="bpm", type=ParamType.FLOAT, label="BPM (0=auto)",
                         default=0, min_val=0, max_val=300),
            ],
            models=["autosong"],
            default_model="autosong",
        ),
        Capability(
            id="generate",
            name="AI Composition",
            description="AI-assisted music composition and MIDI generation",
            status=CapabilityStatus.NOT_IMPLEMENTED,
            input_types=["params"],
            output_types=[TrackType.MIDI],
            param_defs=[],
            models=[],
            default_model="",
        ),
    ]


def _build_devices() -> list[DeviceInfo]:
    devices = []
    for backend in get_all_backends():
        if backend.is_available():
            devices.append(backend.get_device_info())
    return devices


@router.get("/capabilities", response_model=CapabilitiesResponse)
async def get_capabilities():
    return CapabilitiesResponse(
        protocol_version=PROTOCOL_VERSION,
        capabilities=_build_capabilities(),
        devices=_build_devices(),
        scheduler=SchedulerInfo(
            type="fifo",
            max_concurrent=settings.max_concurrent_tasks,
        ),
        output_formats=[
            OutputFormat(format="wav", packaging="separate"),
            OutputFormat(format="flac", packaging="separate"),
            OutputFormat(format="wav", packaging="zip"),
        ],
    )
