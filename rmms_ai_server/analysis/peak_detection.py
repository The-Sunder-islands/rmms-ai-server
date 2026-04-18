import numpy as np
from typing import List, Optional
from .stft import compute_spectrogram, frequency_to_midi, midi_to_frequency, N_FFT, HOP_LENGTH

PEAK_THRESHOLD = 0.15
HARMONIC_TOLERANCE_SEMITONES = 1.5
MAX_HARMONICS = 16
MIN_FUNDAMENTAL_FREQ = 27.5
MAX_FUNDAMENTAL_FREQ = 4186.0


class SpectralPeak:
    __slots__ = ['bin_idx', 'frequency', 'magnitude', 'time_frame']

    def __init__(self, bin_idx: int, frequency: float, magnitude: float, time_frame: int):
        self.bin_idx = bin_idx
        self.frequency = frequency
        self.magnitude = magnitude
        self.time_frame = time_frame


class HarmonicGroup:
    __slots__ = ['fundamental_freq', 'midi_number', 'peaks', 'total_energy', 'harmonic_energies']

    def __init__(self, fundamental_freq: float):
        self.fundamental_freq = fundamental_freq
        self.midi_number = frequency_to_midi(fundamental_freq)
        self.peaks: List[SpectralPeak] = []
        self.total_energy = 0.0
        self.harmonic_energies: List[float] = []


def detect_peaks(magnitude_frame: np.ndarray, sample_rate: int,
                 n_fft: int = N_FFT, threshold: float = PEAK_THRESHOLD) -> List[SpectralPeak]:
    peaks = []
    n_bins = len(magnitude_frame)
    max_mag = np.max(magnitude_frame)
    if max_mag < 1e-10:
        return peaks

    normalized = magnitude_frame / max_mag

    for i in range(2, n_bins - 2):
        if normalized[i] < threshold:
            continue
        if (normalized[i] >= normalized[i - 1] and
                normalized[i] >= normalized[i + 1] and
                normalized[i] >= normalized[i - 2] and
                normalized[i] >= normalized[i + 2]):
            freq = i * sample_rate / n_fft
            peaks.append(SpectralPeak(
                bin_idx=i,
                frequency=freq,
                magnitude=float(normalized[i]),
                time_frame=0
            ))

    return peaks


def _find_harmonic_peak(peaks: List[SpectralPeak], target_freq: float,
                        tolerance_semitones: float = HARMONIC_TOLERANCE_SEMITONES) -> Optional[SpectralPeak]:
    best_peak = None
    best_distance = tolerance_semitones

    for peak in peaks:
        if peak.frequency <= 0 or target_freq <= 0:
            continue
        semitone_distance = abs(12.0 * np.log2(peak.frequency / target_freq))
        if semitone_distance < best_distance:
            best_distance = semitone_distance
            best_peak = peak

    return best_peak


def group_harmonics(peaks: List[SpectralPeak], sample_rate: int,
                    n_fft: int = N_FFT) -> List[HarmonicGroup]:
    peaks_sorted = sorted(peaks, key=lambda p: p.magnitude, reverse=True)
    used_peaks = set()
    groups = []

    for peak in peaks_sorted:
        if id(peak) in used_peaks:
            continue
        if peak.frequency < MIN_FUNDAMENTAL_FREQ or peak.frequency > MAX_FUNDAMENTAL_FREQ:
            continue

        group = HarmonicGroup(peak.frequency)
        group.peaks.append(peak)
        used_peaks.add(id(peak))
        group.harmonic_energies.append(peak.magnitude)

        for h in range(2, MAX_HARMONICS + 1):
            harmonic_freq = peak.frequency * h
            if harmonic_freq > sample_rate / 2:
                break

            harmonic_peak = _find_harmonic_peak(peaks, harmonic_freq)
            if harmonic_peak is not None and id(harmonic_peak) not in used_peaks:
                group.peaks.append(harmonic_peak)
                used_peaks.add(id(harmonic_peak))
                group.harmonic_energies.append(harmonic_peak.magnitude)
            else:
                group.harmonic_energies.append(0.0)

        group.total_energy = sum(p.magnitude for p in group.peaks)
        groups.append(group)

    return groups


def track_harmonics_over_time(spectrogram: np.ndarray, sample_rate: int,
                              n_fft: int = N_FFT,
                              hop_length: int = HOP_LENGTH) -> List[List[HarmonicGroup]]:
    n_frames = spectrogram.shape[2] if spectrogram.ndim == 3 else spectrogram.shape[1]
    all_groups = []

    for t in range(n_frames):
        if spectrogram.ndim == 3:
            frame = spectrogram[0, :, t]
        else:
            frame = spectrogram[:, t]

        peaks = detect_peaks(frame, sample_rate, n_fft)
        for p in peaks:
            p.time_frame = t

        groups = group_harmonics(peaks, sample_rate, n_fft)
        all_groups.append(groups)

    return all_groups


def compute_onset_strength(spectrogram: np.ndarray) -> np.ndarray:
    if spectrogram.ndim == 3:
        mag = spectrogram[0]
    else:
        mag = spectrogram

    flux = np.zeros(mag.shape[1])
    for t in range(1, mag.shape[1]):
        diff = mag[:, t] - mag[:, t - 1]
        flux[t] = np.sum(np.maximum(diff, 0))

    return flux


def detect_onsets(onset_strength: np.ndarray, hop_length: int = HOP_LENGTH,
                  sample_rate: int = 44100, threshold: float = 0.3) -> List[float]:
    times = []
    mean_strength = np.mean(onset_strength)
    if mean_strength < 1e-10:
        return times

    normalized = onset_strength / (mean_strength + 1e-10)
    min_gap_frames = int(0.05 * sample_rate / hop_length)

    last_frame = -min_gap_frames
    for i in range(1, len(normalized) - 1):
        if (normalized[i] > threshold and
                normalized[i] >= normalized[i - 1] and
                normalized[i] >= normalized[i + 1] and
                i - last_frame >= min_gap_frames):
            times.append(i * hop_length / sample_rate)
            last_frame = i

    return times
