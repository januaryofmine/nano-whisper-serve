"""MM 2.3.C correctness gate — the engine transcribes, token-for-token vs openai/whisper.

The v0.2 deliverable: mel -> encoder -> greedy decode with SOT/suppress produces the SAME token
stream as the oracle (ref_whisper.json generated_tokens) on both fixtures. Cache OFF (naive). fp32 CPU.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine import audio, decode, model, tokenizer as tk

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "models" / "whisper-small"
REF_JSON = ROOT / "tests" / "ref_whisper.json"
FIXTURES = ROOT / "tests" / "fixtures" / "audio"
CLIPS = ("vi", "en")


def _need():
    if not (WEIGHTS / "small.pt").exists():
        pytest.skip("whisper-small weights missing")
    if not REF_JSON.exists():
        pytest.skip("ref_whisper.json missing — run `python -m engine.ref_whisper`")


@pytest.fixture(scope="module")
def whisper_model():
    _need()
    return model.load_whisper_small(str(WEIGHTS))


@pytest.mark.parametrize("stem", CLIPS)
def test_transcript_matches_reference(stem, whisper_model):
    _need()
    m = json.loads(REF_JSON.read_text(encoding="utf-8"))
    clip = m["clips"][stem]

    a = audio.load_audio(FIXTURES / f"{stem}.wav")
    tok = tk.get_tokenizer(clip["language"])
    result = decode.transcribe(a, whisper_model, tok)

    assert result["tokens"] == clip["generated_tokens"], (
        f"{stem}: token-for-token mismatch\n  got : {result['tokens']}\n  want: {clip['generated_tokens']}"
    )
    assert result["text"].strip() == clip["text"].strip()
