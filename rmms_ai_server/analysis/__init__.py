from .stft import (
    compute_spectrogram, compress_amplitude, expand_amplitude,
    frequency_to_midi, midi_to_frequency, N_FFT, HOP_LENGTH,
)
from .peak_detection import (
    detect_peaks, group_harmonics, track_harmonics_over_time,
    compute_onset_strength, detect_onsets,
    SpectralPeak, HarmonicGroup,
)
from .note_detection import (
    PipelineState, NoteEvent, get_instrument_params, calculate_scale_notes,
    pipeline_step1_init, pipeline_step2_scale,
    pipeline_step3_note_detection, pipeline_step4_instrument_classify,
    pipeline_step5_note_refinement, pipeline_step6_structure_analysis,
    NOTE_NAMES, SCALE_NAMES,
)
from .autosong import autosong, autosong_from_file, AutoSongConfig, AutoSongResult, result_to_dict
