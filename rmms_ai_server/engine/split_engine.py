from __future__ import annotations

import logging
import os
import shutil
import threading
from typing import Any, Callable, Optional

import numpy as np
import soundfile as sf
import torch

from rmms_ai_server.config import settings
from rmms_ai_server.models.errors import DeviceError, ModelError, ErrorCode
from .device_backend import get_backend, resolve_device_type

logger = logging.getLogger(__name__)

STEM_4 = ["vocals", "drums", "bass", "other"]
STEM_6 = ["vocals", "drums", "bass", "other", "guitar", "piano"]
MODEL_MAP = {4: "htdemucs", 6: "htdemucs_6s"}
OVERLAP = 0.17

_model_cache: dict[str, Any] = {}
_model_cache_lock = threading.Lock()


def _get_model(model_name: str, repo: Optional[str] = None):
    with _model_cache_lock:
        if model_name in _model_cache:
            return _model_cache[model_name]

    try:
        from demucs.pretrained import get_model
        from demucs.apply import BagOfModels
        model = get_model(model_name, repo=repo)
        with _model_cache_lock:
            _model_cache[model_name] = model
        return model
    except Exception as e:
        raise ModelError(ErrorCode.MODEL_LOAD_FAILED, f"Failed to load model '{model_name}': {e}")


def _saturating_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    result = a.astype(np.int32) + b.astype(np.int32)
    result = np.clip(result, -32768, 32767)
    return result.astype(np.int16)


def _calc_section_count(duration_sec: float, device_type: str) -> int:
    if device_type == "npu":
        return 1 if duration_sec <= 600 else max(1, int(duration_sec / 600))
    elif device_type == "cuda":
        return 1 if duration_sec <= 300 else max(1, int(duration_sec / 300))
    else:
        if duration_sec <= 300:
            return 1
        elif duration_sec <= 600:
            return 2
        else:
            return max(1, min(8, int(duration_sec / 300)))


def _create_backing_track(output_dir: str, stem_names: list[str]):
    backing_stems_4 = ["drums", "other", "bass"]
    backing_stems_6 = ["vocals", "other", "drums", "bass", "guitar", "piano"]

    _merge_stems_to_file(output_dir, backing_stems_4, "backing.wav")

    if len(stem_names) == 6:
        _merge_stems_to_file(output_dir, backing_stems_6, "backing_full.wav")


def _merge_stems_to_file(output_dir: str, stem_names: list[str], out_name: str):
    arrays = []
    sr = None
    for stem in stem_names:
        path = os.path.join(output_dir, f"{stem}.wav")
        if not os.path.isfile(path):
            return
        data, s = sf.read(path, dtype='int16')
        if sr is None:
            sr = s
        arrays.append(data)

    if len(arrays) < 2:
        return

    result = arrays[0].copy()
    for arr in arrays[1:]:
        min_len = min(len(result), len(arr))
        result = _saturating_add(result[:min_len], arr[:min_len])

    out_path = os.path.join(output_dir, out_name)
    sf.write(out_path, result, sr)
    logger.info(f"Created backing track: {out_name}")


def _read_audio(path: str, samplerate: Optional[int] = None, channels: Optional[int] = None) -> tuple[torch.Tensor, int]:
    from demucs.audio import AudioFile
    af = AudioFile(path)
    wav = af.read(samplerate=samplerate, channels=channels)
    sr = af.samplerate()
    if wav.dim() == 2:
        wav = wav.unsqueeze(0)
    return wav, sr


def _apply_demucs(
    model, mix: torch.Tensor, device: torch.device,
    shifts: int = 1, overlap: float = OVERLAP,
    progress: bool = False, callback: Optional[Callable[[float, str], None]] = None,
) -> torch.Tensor:
    from demucs.apply import apply_model
    ref = mix.mean(0)
    mix_norm = mix - ref.mean()
    result = apply_model(
        model, mix_norm, shifts=shifts, overlap=overlap,
        device=device, progress=False,
    )
    result = result * ref.std() + ref.mean()
    return result


def _write_stem_wav(path: str, audio: torch.Tensor, samplerate: int):
    audio_np = audio.cpu().numpy()
    if audio_np.ndim == 2:
        audio_np = audio_np.T
    elif audio_np.ndim == 3:
        audio_np = audio_np.squeeze(0).T
    sf.write(path, audio_np, samplerate)


def run_split(
    input_path: str,
    output_dir: str,
    stem_count: int = 6,
    shifts: int = 1,
    device_type: str = "cpu",
    model: Optional[str] = None,
    repo: Optional[str] = None,
    overlap: float = OVERLAP,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> dict:
    input_path = os.path.abspath(input_path)
    if not os.path.isfile(input_path):
        raise ModelError(ErrorCode.INPUT_MISSING, f"Input file not found: {input_path}")

    if stem_count not in (4, 6):
        raise ModelError(ErrorCode.INPUT_INVALID_PARAMS, f"stem_count must be 4 or 6, got {stem_count}")

    if model is None:
        model = MODEL_MAP[stem_count]

    os.makedirs(output_dir, exist_ok=True)

    device_type = resolve_device_type(device_type)

    backend = get_backend(device_type)
    if backend is None or not backend.is_available():
        raise DeviceError(ErrorCode.DEVICE_NOT_AVAILABLE, f"Device type '{device_type}' not available")

    device_id = backend.acquire_device()
    if device_id is None:
        raise DeviceError(ErrorCode.DEVICE_BUSY, f"No available {device_type} device")

    try:
        info = sf.info(input_path)
        duration = info.duration
        n_sections = _calc_section_count(duration, device_type)
        stem_names = STEM_6 if stem_count == 6 else STEM_4
        device_str = backend.get_torch_device_str(device_id)
        torch_device = torch.device(device_str)

        logger.info(f"Split: {input_path}, model={model}, stems={stem_count}, "
                     f"sections={n_sections}, device={device_str}")

        if progress_callback:
            progress_callback(0.0, "Loading model...")

        demucs_model = _get_model(model, repo)

        if n_sections <= 1:
            result = _split_single(
                demucs_model, input_path, output_dir, model, torch_device,
                shifts, overlap, progress_callback
            )
        else:
            result = _split_sectioned(
                demucs_model, input_path, output_dir, model, torch_device,
                shifts, overlap, n_sections, stem_names, progress_callback
            )

        _create_backing_track(output_dir, stem_names)

        if progress_callback:
            progress_callback(100.0, "Separation complete")

        return result

    finally:
        backend.release_device(device_id)


def _split_single(
    model, input_path: str, output_dir: str, model_name: str, device: torch.device,
    shifts: int, overlap: float,
    progress_callback: Optional[Callable[[float, str], None]],
) -> dict:
    if progress_callback:
        progress_callback(5.0, "Reading audio...")

    wav, sr = _read_audio(input_path)

    if progress_callback:
        progress_callback(10.0, "Separating audio...")

    try:
        separated = _apply_demucs(model, wav, device, shifts=shifts, overlap=overlap)
    except Exception as e:
        raise ModelError(ErrorCode.MODEL_INFER_FAILED, f"Demucs inference failed: {e}")

    if separated.dim() == 4:
        separated = separated.squeeze(0)

    sources = model.sources
    total_stems = separated.shape[0]
    for i in range(total_stems):
        stem_name = sources[i] if i < len(sources) else f"stem_{i}"
        out_path = os.path.join(output_dir, f"{stem_name}.wav")
        _write_stem_wav(out_path, separated[i], sr)

        if progress_callback:
            pct = 10.0 + 85.0 * ((i + 1) / total_stems)
            progress_callback(pct, f"Writing stem: {stem_name}")

    return {"output_dir": output_dir, "model": model_name}


def _split_sectioned(
    model, input_path: str, output_dir: str, model_name: str, device: torch.device,
    shifts: int, overlap: float, n_sections: int, stem_names: list[str],
    progress_callback: Optional[Callable[[float, str], None]],
) -> dict:
    data, sr = sf.read(input_path, dtype='int16')
    total_samples = len(data)
    section_len = total_samples // n_sections

    sections_dir = os.path.join(output_dir, "__sections__")
    os.makedirs(sections_dir, exist_ok=True)

    try:
        section_paths = []
        for i in range(n_sections):
            start = i * section_len
            end = (i + 1) * section_len if i < n_sections - 1 else total_samples
            chunk = data[start:end]
            sec_path = os.path.join(sections_dir, f"section_{i}.wav")
            sf.write(sec_path, chunk, sr)
            section_paths.append(sec_path)

        logger.info(f"Split into {n_sections} sections")

        for i, sec_path in enumerate(section_paths):
            if progress_callback:
                pct = 5.0 + 85.0 * (i / n_sections)
                progress_callback(pct, f"Processing section {i + 1}/{n_sections}...")

            try:
                wav, sec_sr = _read_audio(sec_path)
                separated = _apply_demucs(model, wav, device, shifts=shifts, overlap=overlap)
            except Exception as e:
                raise ModelError(ErrorCode.MODEL_INFER_FAILED,
                                 f"Demucs section {i + 1} failed: {e}")

            if separated.dim() == 4:
                separated = separated.squeeze(0)

            sources = model.sources
            for j in range(separated.shape[0]):
                stem_name = sources[j] if j < len(sources) else f"stem_{j}"
                out_path = os.path.join(sections_dir, f"{stem_name}_{i}.wav")
                _write_stem_wav(out_path, separated[j], sec_sr)

        for stem in stem_names:
            merged = _merge_section_stems(sections_dir, stem, n_sections, sr)
            if merged is not None:
                out_path = os.path.join(output_dir, f"{stem}.wav")
                sf.write(out_path, merged, sr)
                logger.info(f"Merged stem: {stem}.wav")

    finally:
        shutil.rmtree(sections_dir, ignore_errors=True)

    return {"output_dir": output_dir, "model": model_name}


def _merge_section_stems(sections_dir: str, stem: str, n_sections: int, sr: int) -> Optional[np.ndarray]:
    first_path = os.path.join(sections_dir, f"{stem}_0.wav")
    if not os.path.isfile(first_path):
        return None
    first_data, _ = sf.read(first_path, dtype='int16')
    os.remove(first_path)

    for i in range(1, n_sections):
        chunk_path = os.path.join(sections_dir, f"{stem}_{i}.wav")
        if not os.path.isfile(chunk_path):
            continue
        chunk_data, _ = sf.read(chunk_path, dtype='int16')
        os.remove(chunk_path)
        first_data = np.concatenate([first_data, chunk_data], axis=0)

    return first_data
