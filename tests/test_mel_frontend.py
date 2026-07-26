"""MM 2.1 correctness gate — our log-mel matches openai/whisper bit-close.

The oracle's mel (tests/ref_whisper.npz) was computed from the SAME fixture wav via soundfile;
feeding our engine the identical float array isolates the STFT/filterbank from any resampler
mismatch (fixtures are 16 kHz mono). Gate: max abs diff < 1e-4.

Regenerate the reference if missing:  .venv/bin/python -m engine.ref_whisper
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from engine import audio

ROOT = Path(__file__).resolve().parent.parent
REF_NPZ = ROOT / "tests" / "ref_whisper.npz"
FIXTURES = ROOT / "tests" / "fixtures" / "audio"
CLIPS = ("vi", "en")
TOL = 1e-4


@pytest.mark.parametrize("stem", CLIPS)
def test_mel_matches_reference(stem):
    if not REF_NPZ.exists():
        pytest.skip("ref_whisper.npz missing — run `python -m engine.ref_whisper`")
    ref = np.load(REF_NPZ)[f"{stem}__mel"]  # [80, 3000]

    wav = FIXTURES / f"{stem}.wav"
    a = audio.pad_or_trim(audio.load_audio(wav))
    mel = np.asarray(audio.log_mel_spectrogram(a))

    assert mel.shape == ref.shape == (80, 3000), f"{stem}: shape {mel.shape} vs {ref.shape}"
    max_abs = float(np.abs(mel - ref).max())
    assert max_abs < TOL, f"{stem}: max|Δmel|={max_abs:.3e} exceeds {TOL:.0e}"
