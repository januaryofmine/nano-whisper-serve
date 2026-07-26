"""Mel front-end — wav -> log-mel spectrogram, matching openai/whisper bit-close.

Spec (openai/whisper audio.py:13-157), reproduced exactly:
  N_FFT=400, HOP_LENGTH=160, n_mels=80, hann_window, stft[..., :-1] (drop last frame),
  power = |stft|**2, mel_filters @ mag, clamp(1e-10).log10(),
  maximum(x, x.max()-8.0)  # GLOBAL max floor, 8 decades,
  (x + 4.0) / 4.0          # affine norm,
  pad_or_trim to N_SAMPLES=480000 (30s, right pad) -> N_FRAMES=3000.
The mel filterbank is loaded from engine/assets/mel_filters.npz (key "mel_80").
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 80
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000
N_FRAMES = N_SAMPLES // HOP_LENGTH      # 3000

_ASSETS = Path(__file__).parent / "assets"


def load_audio(path: str | Path) -> np.ndarray:
    """Load an audio file as mono float32 @ 16 kHz in [-1, 1).

    Fixtures are already 16 kHz mono, so no resample is needed here — reading with soundfile
    yields the identical float array the reference oracle saw (ref_whisper.read_wav_mono16k),
    which is what keeps the mel gate honest. soundfile already returns float in [-1, 1) for PCM
    (i.e. the int16/32768.0 scaling is applied internally).
    """
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:  # stereo -> mono
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        raise ValueError(f"{path}: sample rate {sr} != {SAMPLE_RATE} — re-create the fixture at 16 kHz")
    return np.ascontiguousarray(audio, dtype=np.float32)


def pad_or_trim(audio: np.ndarray, length: int = N_SAMPLES) -> np.ndarray:
    """Right-pad with zeros or trim to `length` samples (axis=-1)."""
    if audio.shape[-1] > length:
        return audio[..., :length]
    if audio.shape[-1] < length:
        pad = [(0, 0)] * (audio.ndim - 1) + [(0, length - audio.shape[-1])]
        return np.pad(audio, pad)
    return audio


@lru_cache(maxsize=None)
def mel_filters(n_mels: int = N_MELS) -> torch.Tensor:
    """Precomputed mel filterbank [n_mels, N_FFT//2 + 1] from assets (key 'mel_{n_mels}')."""
    with np.load(_ASSETS / "mel_filters.npz") as f:
        return torch.from_numpy(f[f"mel_{n_mels}"]).float()


def log_mel_spectrogram(audio: np.ndarray | torch.Tensor, n_mels: int = N_MELS) -> torch.Tensor:
    """audio [n_samples] -> log-mel [n_mels, n_frames]. Exact whisper spec (see module docstring)."""
    if not isinstance(audio, torch.Tensor):
        audio = torch.from_numpy(np.asarray(audio, dtype=np.float32))
    audio = audio.float()

    window = torch.hann_window(N_FFT, device=audio.device)
    stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2                  # drop last frame; power spectrum

    mel_spec = mel_filters(n_mels).to(audio.device) @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)  # GLOBAL max floor
    log_spec = (log_spec + 4.0) / 4.0                          # affine normalize
    return log_spec
