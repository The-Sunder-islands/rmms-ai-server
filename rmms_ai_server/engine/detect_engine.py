from __future__ import annotations

import json
import logging
import os
from typing import Callable, Optional

from rmms_ai_server.models.errors import ModelError, ErrorCode
from rmms_ai_server.analysis.autosong import autosong_from_file, AutoSongConfig, result_to_dict

logger = logging.getLogger(__name__)


def run_detect(
    input_path: str,
    output_dir: str,
    instrument_id: int = 0,
    instrument_sub: int = 0,
    scale_type: int = 0,
    scale_root: int = 0,
    bpm: float = 0.0,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> dict:
    input_path = os.path.abspath(input_path)
    if not os.path.isfile(input_path):
        raise ModelError(ErrorCode.INPUT_MISSING, f"Input file not found: {input_path}")

    os.makedirs(output_dir, exist_ok=True)

    if progress_callback:
        progress_callback(0.0, "Computing spectrogram...")

    config = AutoSongConfig(
        instrument_id=instrument_id,
        instrument_sub=instrument_sub,
        scale_type=scale_type,
        scale_root=scale_root,
        bpm=bpm,
    )

    try:
        if progress_callback:
            progress_callback(10.0, "Running AutoSong pipeline...")

        result = autosong_from_file(input_path, config)

        if progress_callback:
            progress_callback(80.0, "Writing note events...")

        result_dict = result_to_dict(result)

        base_name = os.path.splitext(os.path.basename(input_path))[0]
        notes_path = os.path.join(output_dir, f"{base_name}_notes.json")
        with open(notes_path, "w", encoding="utf-8") as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)

        if progress_callback:
            progress_callback(95.0, "Note detection complete")

        logger.info(f"Detect complete: {len(result.notes)} notes, BPM={result.config.bpm:.1f}")

        return {
            "output_dir": output_dir,
            "notes_path": notes_path,
            "note_events": result_dict,
            "total_notes": len(result.notes),
            "bpm": result.config.bpm,
        }

    except Exception as e:
        raise ModelError(ErrorCode.MODEL_INFER_FAILED, f"AutoSong pipeline failed: {e}")
