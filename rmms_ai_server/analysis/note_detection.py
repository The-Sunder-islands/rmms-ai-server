import numpy as np
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, field
from .stft import frequency_to_midi, midi_to_frequency, N_FFT, HOP_LENGTH
from .peak_detection import HarmonicGroup, track_harmonics_over_time

STRONG_AMPLITUDE_THRESHOLD = 0.85
STRONG_POSITION_THRESHOLD = 0.6
MEDIUM_AMPLITUDE_THRESHOLD = 0.6
MEDIUM_SECONDARY_THRESHOLD = 0.6
WEAK_POSITION_THRESHOLD = 0.2
WEAK_AMPLITUDE_THRESHOLD = 0.4

NOTE_REFINEMENT_LOW = 0.5
NOTE_REFINEMENT_HIGH = 0.8

FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3

INSTRUMENT_TEMPO_MULTIPLIERS = {
    1: 0.5, 2: 0.5, 5: 0.5, 13: 0.5, 21: 0.5, 24: 0.5,
    8: 0.25, 9: 0.25, 11: 0.25,
    16: 2.0, 18: 2.0, 19: 2.0, 20: 2.0,
}

INSTRUMENT_AMPLITUDE_SENSITIVITY = {
    9: 0.0195,
    2: 0.0098, 8: 0.0098,
    21: 0.4,
    7: 0.35, 14: 0.35,
}

INSTRUMENT_NOTE_AMPLITUDE = {
    16: 0.4, 19: 0.4,
    5: 0.8, 36: 0.8,
}

DEFAULT_TEMPO_MULTIPLIER = 1.0
DEFAULT_AMPLITUDE_SENSITIVITY_A = 0.15
DEFAULT_AMPLITUDE_SENSITIVITY_B = 0.25
DEFAULT_NOTE_AMPLITUDE = 1.0

INSTRUMENT_FREQ_BANDS = {
    1: (0.3, 0.8, 0.3), 2: (0.3, 0.8, 0.3), 3: (0.3, 0.8, 0.3),
    4: (0.3, 0.8, 0.3), 5: (0.3, 0.8, 0.3), 6: (0.3, 0.8, 0.3),
    7: (0.3, 0.8, 0.3), 8: (0.3, 0.8, 0.3), 9: (0.3, 0.8, 0.3),
    10: (0.3, 0.8, 0.3), 11: (0.3, 0.8, 0.3), 13: (0.3, 0.8, 0.3),
    14: (0.3, 0.8, 0.3), 16: (0.3, 0.8, 0.3), 18: (0.3, 0.8, 0.3),
    19: (0.3, 0.8, 0.3), 20: (0.3, 0.8, 0.3), 21: (0.3, 0.8, 0.3),
    24: (0.3, 0.8, 0.3),
}

SCALE_INTERVALS = {
    0: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    1: [2, 2, 1, 2, 2, 2, 1],
    2: [2, 1, 2, 2, 1, 2, 2],
    3: [2, 1, 2, 1, 2, 2, 1, 2],
    4: [2, 2, 1, 2, 1, 2, 2, 1],
    5: [2, 1, 1, 2, 2, 1, 2, 2],
}

SCALE_NAMES = {
    0: "chromatic", 1: "major", 2: "minor",
    3: "harmonic_minor", 4: "melodic_minor", 5: "dorian",
}

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


@dataclass
class NoteEvent:
    midi_number: float
    start_time: float
    duration: float
    amplitude: float
    instrument_id: int = 0
    instrument_sub: int = 0
    confidence: float = 0.0
    label: str = ""


@dataclass
class PipelineState:
    flags: int = 0x101
    enabled: int = 1
    freq_band_low: float = 0.3
    freq_band_mid: float = 0.8
    freq_band_high: float = 0.3
    amplitude_default: float = 1.0
    amplitude_secondary: float = 1.0
    default_count: int = 4
    scale_type: int = 0
    scale_root: int = 0
    custom_scale: Optional[List[int]] = None
    scale_notes: List[int] = field(default_factory=lambda: [1] * 12)
    scale_note_indices: List[int] = field(default_factory=lambda: list(range(12)))
    notes: List[NoteEvent] = field(default_factory=list)
    amplitude_sensitivity: float = 0.15
    note_amplitude: float = 1.0
    tempo_multiplier: float = 1.0


def fnv1a_hash(data: bytes) -> int:
    h = FNV_OFFSET & 0xFFFFFFFFFFFFFFFF
    for byte in data:
        h ^= byte
        h = (h * FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return h


def calculate_scale_notes(scale_type: int, root: int = 0,
                          custom_scale: Optional[List[int]] = None) -> Tuple[List[int], List[int]]:
    notes = [0] * 12
    indices = [0] * 12

    if custom_scale is not None and len(custom_scale) > 0:
        intervals = custom_scale
    elif scale_type in SCALE_INTERVALS:
        intervals = SCALE_INTERVALS[scale_type]
    else:
        intervals = SCALE_INTERVALS[0]

    note_idx = root % 12
    notes[note_idx] = 1
    seq = 0
    indices[note_idx] = seq

    for interval in intervals:
        note_idx = (note_idx + interval) % 12
        if notes[note_idx]:
            break
        notes[note_idx] = 1
        seq += 1
        indices[note_idx] = seq

    return notes, indices


def pipeline_step1_init() -> PipelineState:
    return PipelineState()


def pipeline_step2_scale(state: PipelineState) -> PipelineState:
    notes, indices = calculate_scale_notes(state.scale_type, state.scale_root, state.custom_scale)
    state.scale_notes = notes
    state.scale_note_indices = indices
    return state


def _evaluate_amplitude_curve(position_ratio: float, curve_type: str = "default") -> float:
    if curve_type == "attack":
        return min(1.0, position_ratio * 3.0)
    elif curve_type == "sustain":
        if position_ratio < 0.1:
            return position_ratio / 0.1
        elif position_ratio > 0.9:
            return (1.0 - position_ratio) / 0.1
        else:
            return 1.0
    else:
        return np.sin(np.pi * position_ratio)


def pipeline_step3_note_detection(state: PipelineState,
                                  harmonic_groups_over_time: List[List[HarmonicGroup]],
                                  sample_rate: int,
                                  hop_length: int = HOP_LENGTH) -> PipelineState:
    n_frames = len(harmonic_groups_over_time)
    if n_frames == 0:
        return state

    amplitude_values = np.zeros(n_frames)
    secondary_values = np.zeros(n_frames)

    for t, groups in enumerate(harmonic_groups_over_time):
        if not groups:
            continue
        total_energy = sum(g.total_energy for g in groups)
        max_energy = max(g.total_energy for g in groups)
        amplitude_values[t] = min(1.0, max_energy / (total_energy + 1e-10))
        secondary_values[t] = min(1.0, total_energy / (len(groups) + 1e-10))

    max_amplitude_idx = int(np.argmax(amplitude_values))

    for t, groups in enumerate(harmonic_groups_over_time):
        if not groups:
            continue

        position_ratio = t / max(n_frames - 1, 1)
        amplitude = amplitude_values[t]
        secondary = secondary_values[t]

        amplitude = max(0.0, min(1.0, amplitude))
        secondary = max(0.0, min(1.0, secondary))

        label = ""
        if amplitude > STRONG_AMPLITUDE_THRESHOLD and position_ratio > STRONG_POSITION_THRESHOLD:
            label = "strong"
        elif amplitude > MEDIUM_AMPLITUDE_THRESHOLD and secondary > MEDIUM_SECONDARY_THRESHOLD:
            label = "medium"
        elif position_ratio < WEAK_POSITION_THRESHOLD and amplitude < WEAK_AMPLITUDE_THRESHOLD:
            label = "weak"

        for group in groups:
            note = NoteEvent(
                midi_number=group.midi_number,
                start_time=t * hop_length / sample_rate,
                duration=hop_length / sample_rate,
                amplitude=amplitude,
                confidence=secondary,
                label=label,
            )
            state.notes.append(note)

    return state


def get_instrument_params(instrument_id: int, instrument_sub: int = 0) -> Dict:
    tempo_mult = INSTRUMENT_TEMPO_MULTIPLIERS.get(instrument_id, DEFAULT_TEMPO_MULTIPLIER)

    if instrument_id in INSTRUMENT_AMPLITUDE_SENSITIVITY:
        amp_sens = INSTRUMENT_AMPLITUDE_SENSITIVITY[instrument_id]
    elif (instrument_sub - 8) & 0xFFFFFFFD == 0:
        amp_sens = DEFAULT_AMPLITUDE_SENSITIVITY_B
    else:
        amp_sens = DEFAULT_AMPLITUDE_SENSITIVITY_A

    if instrument_id in INSTRUMENT_NOTE_AMPLITUDE:
        note_amp = INSTRUMENT_NOTE_AMPLITUDE[instrument_id]
    else:
        note_amp = DEFAULT_NOTE_AMPLITUDE

    freq_bands = INSTRUMENT_FREQ_BANDS.get(instrument_id, (0.3, 0.8, 0.3))

    return {
        "tempo_multiplier": tempo_mult,
        "amplitude_sensitivity": amp_sens,
        "note_amplitude": note_amp,
        "freq_bands": freq_bands,
    }


def pipeline_step4_instrument_classify(state: PipelineState,
                                       instrument_id: int,
                                       instrument_sub: int = 0,
                                       bpm: float = 120.0) -> PipelineState:
    params = get_instrument_params(instrument_id, instrument_sub)
    state.tempo_multiplier = params["tempo_multiplier"]
    state.amplitude_sensitivity = params["amplitude_sensitivity"]
    state.note_amplitude = params["note_amplitude"]
    state.freq_band_low, state.freq_band_mid, state.freq_band_high = params["freq_bands"]

    beat_duration = 60.0 / bpm
    step_duration = beat_duration * state.tempo_multiplier

    filtered_notes = []
    for note in state.notes:
        if note.amplitude < state.amplitude_sensitivity:
            continue
        note.amplitude *= state.note_amplitude
        note.instrument_id = instrument_id
        note.instrument_sub = instrument_sub
        filtered_notes.append(note)

    state.notes = filtered_notes
    return state


def pipeline_step5_note_refinement(state: PipelineState) -> PipelineState:
    if not state.notes:
        return state

    notes_sorted = sorted(state.notes, key=lambda n: (n.midi_number, n.start_time))

    merged = []
    current = None

    for note in notes_sorted:
        if current is None:
            current = NoteEvent(
                midi_number=note.midi_number,
                start_time=note.start_time,
                duration=note.duration,
                amplitude=note.amplitude,
                instrument_id=note.instrument_id,
                instrument_sub=note.instrument_sub,
                confidence=note.confidence,
                label=note.label,
            )
            continue

        gap = note.start_time - (current.start_time + current.duration)
        same_pitch = abs(note.midi_number - current.midi_number) < 0.5

        if same_pitch and gap < 0.05:
            current.duration = (note.start_time + note.duration) - current.start_time
            current.amplitude = max(current.amplitude, note.amplitude)
            current.confidence = max(current.confidence, note.confidence)
        else:
            if current.amplitude >= NOTE_REFINEMENT_LOW:
                if current.amplitude >= NOTE_REFINEMENT_HIGH:
                    current.label = "strong"
                else:
                    current.label = "medium"
            else:
                current.label = "weak"
            merged.append(current)
            current = NoteEvent(
                midi_number=note.midi_number,
                start_time=note.start_time,
                duration=note.duration,
                amplitude=note.amplitude,
                instrument_id=note.instrument_id,
                instrument_sub=note.instrument_sub,
                confidence=note.confidence,
                label=note.label,
            )

    if current is not None:
        if current.amplitude >= NOTE_REFINEMENT_LOW:
            if current.amplitude >= NOTE_REFINEMENT_HIGH:
                current.label = "strong"
            else:
                current.label = "medium"
        else:
            current.label = "weak"
        merged.append(current)

    state.notes = merged
    return state


def pipeline_step6_structure_analysis(state: PipelineState) -> PipelineState:
    if not state.notes:
        return state

    notes = state.notes
    total_duration = max(n.start_time + n.duration for n in notes)

    if total_duration <= 0:
        return state

    n_segments = max(1, int(total_duration / 4.0))
    segment_duration = total_duration / n_segments

    segment_energy = np.zeros(n_segments)
    segment_note_count = np.zeros(n_segments)

    for note in notes:
        seg_start = int(note.start_time / segment_duration)
        seg_end = int((note.start_time + note.duration) / segment_duration)
        seg_start = max(0, min(seg_start, n_segments - 1))
        seg_end = max(0, min(seg_end, n_segments - 1))
        for s in range(seg_start, seg_end + 1):
            segment_energy[s] += note.amplitude
            segment_note_count[s] += 1

    max_energy = np.max(segment_energy) if np.max(segment_energy) > 0 else 1.0
    normalized_energy = segment_energy / max_energy

    for i, note in enumerate(notes):
        seg = int(note.start_time / segment_duration)
        seg = max(0, min(seg, n_segments - 1))

        if normalized_energy[seg] > 0.8:
            note.label = "climax"
        elif normalized_energy[seg] < 0.2:
            note.label = "drop_zone"
        elif seg == 0 and segment_note_count[seg] <= 2:
            note.label = "lonely_start"
        elif i > 0 and abs(note.midi_number - notes[i - 1].midi_number) < 2:
            note.label = "motif"

    state.notes = notes
    return state
