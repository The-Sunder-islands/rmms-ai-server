import numpy as np
import soundfile as sf
from typing import List, Optional, Dict
from dataclasses import dataclass

from .stft import (
    compute_spectrogram, frequency_to_midi, midi_to_frequency,
    compress_amplitude, expand_amplitude, N_FFT, HOP_LENGTH
)
from .peak_detection import (
    track_harmonics_over_time, compute_onset_strength, detect_onsets,
    HarmonicGroup
)
from .note_detection import (
    PipelineState, NoteEvent,
    pipeline_step1_init, pipeline_step2_scale,
    pipeline_step3_note_detection, pipeline_step4_instrument_classify,
    pipeline_step5_note_refinement, pipeline_step6_structure_analysis,
    get_instrument_params, calculate_scale_notes, fnv1a_hash,
    NOTE_NAMES, SCALE_NAMES,
    INSTRUMENT_TEMPO_MULTIPLIERS, INSTRUMENT_AMPLITUDE_SENSITIVITY,
    INSTRUMENT_NOTE_AMPLITUDE, INSTRUMENT_FREQ_BANDS,
    STRONG_AMPLITUDE_THRESHOLD, STRONG_POSITION_THRESHOLD,
    MEDIUM_AMPLITUDE_THRESHOLD, MEDIUM_SECONDARY_THRESHOLD,
    WEAK_POSITION_THRESHOLD, WEAK_AMPLITUDE_THRESHOLD,
    NOTE_REFINEMENT_LOW, NOTE_REFINEMENT_HIGH,
    FNV_OFFSET, FNV_PRIME,
)

NOTE_COUNT_THRESHOLD_4 = 1.0
NOTE_COUNT_THRESHOLD_3 = 0.75
NOTE_COUNT_THRESHOLD_2 = 0.5
INNER_THRESHOLD = 0.33
GLOBAL_DATA_STRIDE = 437112
NOTE_STRIDE = 328
NOTE_ENTRY_STRIDE = 172


@dataclass
class AutoSongConfig:
    instrument_id: int = 0
    instrument_sub: int = 0
    scale_type: int = 0
    scale_root: int = 0
    custom_scale: Optional[List[int]] = None
    bpm: float = 120.0
    n_fft: int = N_FFT
    hop_length: int = HOP_LENGTH
    sample_rate: int = 44100


@dataclass
class AutoSongResult:
    notes: List[NoteEvent]
    state: PipelineState
    onsets: List[float]
    spectrogram: np.ndarray
    config: AutoSongConfig


def _estimate_bpm(onset_times: List[float], sample_rate: int = 44100) -> float:
    if len(onset_times) < 2:
        return 120.0

    intervals = np.diff(onset_times)
    intervals = intervals[intervals > 0.1]

    if len(intervals) == 0:
        return 120.0

    hist, bin_edges = np.histogram(intervals, bins=50, range=(0.2, 2.0))
    best_idx = np.argmax(hist)
    best_interval = (bin_edges[best_idx] + bin_edges[best_idx + 1]) / 2.0

    bpm = 60.0 / best_interval
    while bpm > 200:
        bpm /= 2.0
    while bpm < 60:
        bpm *= 2.0

    return bpm


def _apply_note_count_threshold(state: PipelineState) -> PipelineState:
    if not state.notes:
        return state

    pitch_groups: Dict[int, List[NoteEvent]] = {}
    for note in state.notes:
        key = round(note.midi_number)
        if key not in pitch_groups:
            pitch_groups[key] = []
        pitch_groups[key].append(note)

    n_pitches = len(pitch_groups)
    if n_pitches >= 4:
        threshold = NOTE_COUNT_THRESHOLD_4
    elif n_pitches == 3:
        threshold = NOTE_COUNT_THRESHOLD_3
    elif n_pitches == 2:
        threshold = NOTE_COUNT_THRESHOLD_2
    else:
        threshold = 0.0

    if threshold > 0:
        state.notes = [n for n in state.notes if n.amplitude >= threshold or n.confidence >= INNER_THRESHOLD]

    return state


def autosong(audio: np.ndarray, config: Optional[AutoSongConfig] = None) -> AutoSongResult:
    if config is None:
        config = AutoSongConfig()

    if audio.ndim == 1:
        audio = audio.reshape(1, -1)

    sample_rate = config.sample_rate
    n_fft = config.n_fft
    hop_length = config.hop_length

    spectrogram = compute_spectrogram(audio, n_fft=n_fft, hop_length=hop_length)

    harmonic_groups = track_harmonics_over_time(spectrogram, sample_rate, n_fft, hop_length)

    onset_strength = compute_onset_strength(spectrogram)
    onset_times = detect_onsets(onset_strength, hop_length, sample_rate)

    if config.bpm <= 0:
        config.bpm = _estimate_bpm(onset_times, sample_rate)

    state = pipeline_step1_init()
    state.scale_type = config.scale_type
    state.scale_root = config.scale_root
    state.custom_scale = config.custom_scale

    state = pipeline_step2_scale(state)

    state = pipeline_step3_note_detection(state, harmonic_groups, sample_rate, hop_length)

    state = pipeline_step4_instrument_classify(
        state, config.instrument_id, config.instrument_sub, config.bpm
    )

    state = pipeline_step5_note_refinement(state)

    state = _apply_note_count_threshold(state)

    state = pipeline_step6_structure_analysis(state)

    return AutoSongResult(
        notes=state.notes,
        state=state,
        onsets=onset_times,
        spectrogram=spectrogram,
        config=config,
    )


def autosong_from_file(file_path: str, config: Optional[AutoSongConfig] = None) -> AutoSongResult:
    audio, sr = sf.read(file_path, dtype='int16')

    if audio.ndim == 1:
        audio = audio.reshape(-1, 1)
    if audio.shape[1] > 1:
        audio = audio[:, 0].reshape(-1, 1)
    audio = audio.T

    if config is None:
        config = AutoSongConfig()
    config.sample_rate = sr

    return autosong(audio, config)


def get_note_name(midi_number: float) -> str:
    note_idx = int(round(midi_number)) % 12
    octave = int(round(midi_number)) // 12 - 1
    return f"{NOTE_NAMES[note_idx]}{octave}"


def result_to_dict(result: AutoSongResult) -> Dict:
    return {
        "notes": [
            {
                "midi_number": n.midi_number,
                "note_name": get_note_name(n.midi_number),
                "frequency": midi_to_frequency(n.midi_number),
                "start_time": round(n.start_time, 4),
                "duration": round(n.duration, 4),
                "amplitude": round(n.amplitude, 4),
                "instrument_id": n.instrument_id,
                "instrument_sub": n.instrument_sub,
                "confidence": round(n.confidence, 4),
                "label": n.label,
            }
            for n in result.notes
        ],
        "onsets": [round(t, 4) for t in result.onsets],
        "bpm": result.config.bpm,
        "scale": SCALE_NAMES.get(result.config.scale_type, "unknown"),
        "scale_root": NOTE_NAMES[result.config.scale_root],
        "total_notes": len(result.notes),
    }
