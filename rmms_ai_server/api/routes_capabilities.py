from __future__ import annotations

from fastapi import APIRouter

from rmms_ai_server import PROTOCOL_VERSION, __version__
from rmms_ai_server.config import settings
from rmms_ai_server.engine.device_backend import get_all_backends
from rmms_ai_server.models.protocol import (
    CapabilitiesResponse, Capability, CapabilityStatus,
    ParamDef, ParamType, DeviceInfo, SchedulerInfo,
)

router = APIRouter()


def _build_capabilities() -> list[Capability]:
    return [
        Capability(
            id="split",
            label="Stem Separation",
            description="Separate audio into individual stems using Demucs AI models",
            status=CapabilityStatus.IMPLEMENTED,
            param_defs=[
                ParamDef(key="model", type=ParamType.ENUM, label="Model",
                         description="Separation model to use",
                         default="htdemucs_6s",
                         options=[
                             {"value": "htdemucs", "label": "HTDemucs (4 stems)"},
                             {"value": "htdemucs_6s", "label": "HTDemucs 6-stem"},
                         ],
                         group="basic"),
                ParamDef(key="shifts", type=ParamType.INT, label="Shifts",
                         description="Number of random shifts for better quality",
                         default=1, min_val=1, max_val=20, step=1,
                         group="advanced"),
                ParamDef(key="overlap", type=ParamType.FLOAT, label="Overlap",
                         description="Overlap between chunks",
                         default=0.17, min_val=0.0, max_val=0.99, step=0.01,
                         decimals=2, group="advanced"),
                ParamDef(key="device_type", type=ParamType.ENUM, label="Device",
                         description="Compute device for inference",
                         default="auto",
                         choices=["auto", "cuda", "dml", "npu", "xpu", "mps", "cpu"],
                         group="advanced"),
            ],
            models=["htdemucs", "htdemucs_6s", "mdx_extra_q"],
            default_model="htdemucs_6s",
        ),
        Capability(
            id="midi",
            label="MIDI Transcription",
            description="Transcribe audio to MIDI using basic-pitch",
            status=CapabilityStatus.IMPLEMENTED,
            param_defs=[
                ParamDef(key="onset_threshold", type=ParamType.FLOAT, label="Onset Threshold",
                         default=0.6, min_val=0.0, max_val=1.0, step=0.05,
                         decimals=2, group="basic"),
                ParamDef(key="frame_threshold", type=ParamType.FLOAT, label="Frame Threshold",
                         default=0.3, min_val=0.0, max_val=1.0, step=0.05,
                         decimals=2, group="basic"),
            ],
            models=["basic-pitch"],
            default_model="basic-pitch",
        ),
        Capability(
            id="detect",
            label="Note Detection",
            description="Detect note events from audio using AutoSong pipeline",
            status=CapabilityStatus.IMPLEMENTED,
            param_defs=[
                ParamDef(key="instrument_id", type=ParamType.INT, label="Instrument ID",
                         description="Instrument type for note detection",
                         default=0, min_val=0, max_val=36, group="basic"),
                ParamDef(key="scale_type", type=ParamType.ENUM, label="Scale",
                         description="Musical scale type",
                         default="0", choices=["0", "1", "2", "3", "4", "5"],
                         group="basic"),
                ParamDef(key="scale_root", type=ParamType.INT, label="Scale Root",
                         description="Root note of the scale (0=C, 1=C#, ...)",
                         default=0, min_val=0, max_val=11, group="basic"),
                ParamDef(key="bpm", type=ParamType.FLOAT, label="BPM (0=auto)",
                         description="Beats per minute, 0 for auto-detect",
                         default=0.0, min_val=0.0, max_val=300.0,
                         group="basic"),
            ],
            models=["autosong"],
            default_model="autosong",
        ),
        Capability(
            id="generate",
            label="AI Composition",
            description="AI-assisted music composition and MIDI generation",
            status=CapabilityStatus.NOT_IMPLEMENTED,
            param_defs=[],
            models=[],
            default_model=None,
        ),
    ]


def _build_devices() -> list[DeviceInfo]:
    devices = []
    for backend in get_all_backends():
        info = backend.get_device_info()
        if info.available:
            devices.append(info)
        else:
            devices.append(info)
    return devices


@router.get("/capabilities", response_model=CapabilitiesResponse)
async def get_capabilities():
    return CapabilitiesResponse(
        protocol_version=PROTOCOL_VERSION,
        server_version=__version__,
        capabilities=_build_capabilities(),
        devices=_build_devices(),
        scheduler=SchedulerInfo(
            max_concurrent_tasks=settings.max_concurrent_tasks,
            max_queue_size=settings.max_queue_size,
        ),
        output_formats=["wav", "flac", "mp3"],
        output_packages=["separate", "zip"],
        max_upload_bytes=settings.max_upload_bytes,
    )
