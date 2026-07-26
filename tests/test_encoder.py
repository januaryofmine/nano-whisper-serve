"""MM 2.2 correctness gate — hand-written encoder matches openai/whisper's.

Tests the whole encoder path (weight load + conv stem + sinusoidal pos + 12 self-attn blocks +
ln_post) against the oracle's encoder output in tests/ref_whisper.npz. fp32 CPU. Gate: max abs diff < 1e-3.

Regenerate the reference if missing:  .venv/bin/python -m engine.ref_whisper
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from engine import audio, model

ROOT = Path(__file__).resolve().parent.parent
REF_NPZ = ROOT / "tests" / "ref_whisper.npz"
WEIGHTS = ROOT / "models" / "whisper-small"
FIXTURES = ROOT / "tests" / "fixtures" / "audio"
CLIPS = ("vi", "en")
TOL = 1e-3


@pytest.fixture(scope="module")
def whisper_model():
    if not (WEIGHTS / "small.pt").exists():
        pytest.skip("whisper-small weights missing")
    return model.load_whisper_small(str(WEIGHTS))


def test_encoder_loads_strict():
    """Name-map proof: encoder weights load with strict=True (no missing/unexpected keys)."""
    if not (WEIGHTS / "small.pt").exists():
        pytest.skip("whisper-small weights missing")
    model.load_whisper_small(str(WEIGHTS))  # raises if the name map is wrong


@pytest.mark.parametrize("stem", CLIPS)
def test_encoder_matches_reference(stem, whisper_model):
    if not REF_NPZ.exists():
        pytest.skip("ref_whisper.npz missing — run `python -m engine.ref_whisper`")
    ref = np.load(REF_NPZ)[f"{stem}__encoder"]  # [1500, 768]

    a = audio.pad_or_trim(audio.load_audio(FIXTURES / f"{stem}.wav"))
    mel = audio.log_mel_spectrogram(a)  # [80, 3000]
    with torch.no_grad():
        enc = whisper_model.encoder(mel.unsqueeze(0)).squeeze(0)  # [1500, 768]
    enc = enc.numpy()

    assert enc.shape == ref.shape == (1500, 768), f"{stem}: shape {enc.shape} vs {ref.shape}"
    max_abs = float(np.abs(enc - ref).max())
    assert max_abs < TOL, f"{stem}: max|Δenc|={max_abs:.3e} exceeds {TOL:.0e}"
