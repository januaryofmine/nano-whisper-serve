"""Mel front-end — wav -> log-mel spectrogram, matching openai/whisper bit-close.

SKELETON (MM 2.0). Bodies land in MM 2.1. Spec facts (openai/whisper audio.py:13-157):
  N_FFT=400, HOP_LENGTH=160, n_mels=80, hann_window, stft[..., :-1] (drop last frame),
  power = |stft|**2, mel_filters @ mag, clamp(1e-10).log10(),
  maximum(x, x.max()-8.0)  # GLOBAL max floor, 8 decades,
  (x + 4.0) / 4.0          # affine norm,
  pad_or_trim to N_SAMPLES=480000 (30s, right pad) -> N_FRAMES=3000.
The mel filterbank is loaded from engine/assets/mel_filters.npz (key "mel_80").
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

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

    MM 2.1: read via soundfile; if already 16 kHz mono (our fixtures are), no resample.
    Match openai/whisper's int16 path (divide by 32768.0) when reading PCM.
    """
    raise NotImplementedError("MM 2.1")


def pad_or_trim(audio: np.ndarray, length: int = N_SAMPLES) -> np.ndarray:
    """Right-pad with zeros or trim to `length` samples (axis=-1)."""
    raise NotImplementedError("MM 2.1")


def mel_filters(n_mels: int = N_MELS) -> np.ndarray:
    """Load the precomputed mel filterbank [n_mels, N_FFT//2+1] from assets."""
    raise NotImplementedError("MM 2.1")


def log_mel_spectrogram(audio: np.ndarray, n_mels: int = N_MELS):
    """audio [n_samples] -> log-mel [n_mels, n_frames]. See module docstring for exact spec."""
    raise NotImplementedError("MM 2.1")
