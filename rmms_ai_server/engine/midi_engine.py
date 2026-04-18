from __future__ import annotations

import logging
import os
from typing import Callable, Optional

from rmms_ai_server.models.errors import ModelError, ErrorCode

logger = logging.getLogger(__name__)


def run_midi(
    input_path: str,
    output_dir: str,
    onset_threshold: float = 0.6,
    frame_threshold: float = 0.3,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> dict:
    try:
        from basic_pitch.inference import predict
        from basic_pitch import ICASSP_2022_MODEL_PATH
    except ImportError:
        raise ModelError(
            ErrorCode.CAPABILITY_NOT_AVAILABLE,
            "basic-pitch is not installed. Install with: pip install rmms-ai-server[midi]"
        )

    input_path = os.path.abspath(input_path)
    if not os.path.isfile(input_path):
        raise ModelError(ErrorCode.INPUT_MISSING, f"Input file not found: {input_path}")

    os.makedirs(output_dir, exist_ok=True)

    if progress_callback:
        progress_callback(0.0, "Loading basic-pitch model...")

    try:
        if progress_callback:
            progress_callback(10.0, "Running MIDI transcription...")

        model_output, midi_data, note_events = predict(
            input_path,
            onset_threshold=onset_threshold,
            frame_threshold=frame_threshold,
        )

        base_name = os.path.splitext(os.path.basename(input_path))[0]
        midi_path = os.path.join(output_dir, f"{base_name}.mid")
        midi_data.write(midi_path)

        if progress_callback:
            progress_callback(90.0, "MIDI transcription complete")

        note_count = len(note_events) if hasattr(note_events, '__len__') else 0
        logger.info(f"MIDI transcription complete: {midi_path}, {note_count} notes")

        return {
            "output_dir": output_dir,
            "midi_path": midi_path,
            "note_count": note_count,
        }

    except Exception as e:
        raise ModelError(ErrorCode.MODEL_INFER_FAILED, f"basic-pitch inference failed: {e}")
