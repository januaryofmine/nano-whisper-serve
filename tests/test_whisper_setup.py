"""MM 2.0 acceptance gate — the setup is ready (NOT a correctness gate; the engine is not built yet).

Verifies the reference answer key + fixtures + shipped assets exist and are well-formed, so that
MM 2.1–2.4 have something to gate against. Regenerate with:
    .venv/bin/python tests/fixtures/make_fixtures.py     # fixtures
    .venv/bin/python -m engine.ref_whisper               # ref_whisper.{npz,json}
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures" / "audio"
REF_NPZ = ROOT / "tests" / "ref_whisper.npz"
REF_JSON = ROOT / "tests" / "ref_whisper.json"
MEL_FILTERS = ROOT / "engine" / "assets" / "mel_filters.npz"
CLIPS = ("vi", "en")


def test_mel_filters_shipped():
    assert MEL_FILTERS.exists(), "engine/assets/mel_filters.npz missing (MM 2.0)"
    d = np.load(MEL_FILTERS)
    assert d["mel_80"].shape == (80, 201)


def test_fixtures_exist_16k_mono():
    sf = pytest.importorskip("soundfile")
    for stem in CLIPS:
        wav = FIXTURES / f"{stem}.wav"
        assert wav.exists(), f"fixture {wav} missing (run make_fixtures.py)"
        audio, sr = sf.read(str(wav))
        assert sr == 16000, f"{stem}.wav sr={sr}, expected 16000"
        assert audio.ndim == 1, f"{stem}.wav is not mono"
        assert len(audio) / sr <= 30.0, f"{stem}.wav longer than one 30s segment"


def test_ref_json_answer_key():
    assert REF_JSON.exists(), "tests/ref_whisper.json missing (run engine.ref_whisper)"
    m = json.loads(REF_JSON.read_text(encoding="utf-8"))
    assert m["meta"]["model"] == "small"
    assert m["meta"]["whisper_version"] and m["meta"]["torch_version"]
    for stem in CLIPS:
        clip = m["clips"][stem]
        assert clip["language"] == stem
        assert len(clip["sot_sequence"]) == 4  # [sot, lang, transcribe, no_timestamps]
        assert len(clip["generated_tokens"]) > 0
        assert isinstance(clip["text"], str) and clip["text"].strip()


def test_ref_npz_tensors():
    if not REF_NPZ.exists():
        pytest.skip("ref_whisper.npz not generated (regenerable via engine.ref_whisper)")
    d = np.load(REF_NPZ)
    for stem in CLIPS:
        assert d[f"{stem}__mel"].shape == (80, 3000), f"{stem} mel shape wrong"
        assert d[f"{stem}__encoder"].shape == (1500, 768), f"{stem} encoder shape wrong"
