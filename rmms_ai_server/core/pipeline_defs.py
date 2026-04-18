from __future__ import annotations

from rmms_ai_server.models.protocol import PipelineStep


PRESET_PIPELINES: dict[str, list[dict]] = {
    "split": [
        {"capability": "split"},
    ],
    "split+midi": [
        {"capability": "split"},
        {"capability": "midi", "input": {"from_step": 0, "stem": ""}},
    ],
    "split+detect": [
        {"capability": "split"},
        {"capability": "detect", "input": {"from_step": 0, "stem": ""}},
    ],
    "full": [
        {"capability": "split"},
        {"capability": "midi", "input": {"from_step": 0, "stem": ""}},
        {"capability": "detect", "input": {"from_step": 0, "stem": ""}},
    ],
}


def resolve_pipeline(pipeline_data: list[dict] | None, preset_name: str | None = None) -> list[PipelineStep]:
    if pipeline_data:
        return [PipelineStep(**step) for step in pipeline_data]

    if preset_name and preset_name in PRESET_PIPELINES:
        return [PipelineStep(**step) for step in PRESET_PIPELINES[preset_name]]

    return [PipelineStep(capability="split")]
