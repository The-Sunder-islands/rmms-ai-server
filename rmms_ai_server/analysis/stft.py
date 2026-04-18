import numpy as np

try:
    import pyfftw
    _USE_PYFFTW = True
except ImportError:
    _USE_PYFFTW = False

try:
    import torch
    _USE_TORCH = torch.cuda.is_available()
except ImportError:
    _USE_TORCH = False

N_FFT = 2048
HOP_LENGTH = 512
COMPRESS_EXPONENT = 1.660964
EXPAND_EXPONENT = 0.60206
MAX_INT16 = 32767.0


def _fft_backend(data: np.ndarray, n: int, axis: int = -1) -> np.ndarray:
    if _USE_PYFFTW:
        return pyfftw.interfaces.numpy_fft.rfft(data, n=n, axis=axis)
    if _USE_TORCH:
        t = torch.from_numpy(data).cuda()
        result = torch.fft.rfft(t, n=n, dim=axis)
        return result.cpu().numpy()
    return np.fft.rfft(data, n=n, axis=axis)


def _ifft_backend(data: np.ndarray, n: int, axis: int = -1) -> np.ndarray:
    if _USE_PYFFTW:
        return pyfftw.interfaces.numpy_fft.irfft(data, n=n, axis=axis)
    if _USE_TORCH:
        t = torch.from_numpy(data).cuda()
        result = torch.fft.irfft(t, n=n, dim=axis)
        return result.cpu().numpy()
    return np.fft.irfft(data, n=n, axis=axis)


def compress_amplitude(samples: np.ndarray) -> np.ndarray:
    normalized = np.abs(samples).astype(np.float64) / MAX_INT16
    compressed = np.power(normalized, COMPRESS_EXPONENT)
    signs = np.sign(samples)
    return compressed * signs


def expand_amplitude(compressed: np.ndarray) -> np.ndarray:
    normalized = np.abs(compressed).astype(np.float64)
    expanded = np.power(normalized, EXPAND_EXPONENT)
    expanded = np.clip(expanded, 0.0, 1.0)
    signs = np.sign(compressed)
    result = expanded * signs * MAX_INT16
    return result.astype(np.int16)


def stft(audio: np.ndarray, n_fft: int = N_FFT, hop_length: int = HOP_LENGTH) -> np.ndarray:
    if audio.ndim == 1:
        audio = audio.reshape(1, -1)

    n_frames = 1 + (audio.shape[1] - n_fft) // hop_length
    window = np.hanning(n_fft).astype(np.float64)

    frames = np.zeros((audio.shape[0], n_fft, n_frames), dtype=np.float64)
    for t in range(n_frames):
        start = t * hop_length
        segment = audio[:, start:start + n_fft]
        frames[:, :, t] = segment * window[np.newaxis, :]

    spectrum = _fft_backend(frames, n=n_fft, axis=1)
    return spectrum


def istft(spectrum: np.ndarray, n_fft: int = N_FFT, hop_length: int = HOP_LENGTH,
          length: int = None) -> np.ndarray:
    audio_frames = _ifft_backend(spectrum, n=n_fft, axis=1)
    window = np.hanning(n_fft).astype(np.float64)
    audio_frames = audio_frames * window[np.newaxis, :, np.newaxis]

    n_frames = audio_frames.shape[2]
    output_len = n_fft + hop_length * (n_frames - 1)
    output = np.zeros((audio_frames.shape[0], output_len), dtype=np.float64)
    window_sum = np.zeros(output_len, dtype=np.float64)

    for t in range(n_frames):
        start = t * hop_length
        output[:, start:start + n_fft] += audio_frames[:, :, t]
        window_sum[start:start + n_fft] += window ** 2

    window_sum = np.maximum(window_sum, 1e-10)
    output /= window_sum[np.newaxis, :]

    if length is not None:
        output = output[:, :length]

    return output


def compute_spectrogram(audio: np.ndarray, n_fft: int = N_FFT,
                        hop_length: int = HOP_LENGTH) -> np.ndarray:
    compressed = compress_amplitude(audio)
    spectrum = stft(compressed, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(spectrum)
    return magnitude


def frequency_to_bin(freq: float, sample_rate: int, n_fft: int = N_FFT) -> int:
    return int(round(freq * n_fft / sample_rate))


def bin_to_frequency(bin_idx: int, sample_rate: int, n_fft: int = N_FFT) -> float:
    return bin_idx * sample_rate / n_fft


def frequency_to_midi(freq: float) -> float:
    if freq <= 0:
        return 0.0
    return 69.0 + 12.0 * np.log2(freq / 440.0)


def midi_to_frequency(midi: float) -> float:
    return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))
