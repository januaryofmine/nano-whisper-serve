"""MM 2.3.B gate — hand-built (tiktoken-only) tokenizer matches whisper's special-token layout.

Gate: SOT sequence == the oracle's (ref_whisper.json), the special ids are exact, non_speech_tokens
matches whisper's suppress set, and decode(generated_tokens) round-trips to the stored text.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine import tokenizer as tk

ROOT = Path(__file__).resolve().parent.parent
REF_JSON = ROOT / "tests" / "ref_whisper.json"
CLIPS = ("vi", "en")


def test_special_ids():
    t = tk.get_tokenizer("en")
    assert (t.sot, t.eot, t.transcribe, t.no_timestamps, t.timestamp_begin) == (50258, 50257, 50359, 50363, 50364)
    assert tk.get_tokenizer("en").language_token == 50259
    assert tk.get_tokenizer("vi").language_token == 50278


@pytest.mark.parametrize("stem", CLIPS)
def test_sot_sequence_matches_reference(stem):
    if not REF_JSON.exists():
        pytest.skip("ref_whisper.json missing")
    m = json.loads(REF_JSON.read_text(encoding="utf-8"))
    t = tk.get_tokenizer(stem)
    assert t.sot_sequence == tuple(m["clips"][stem]["sot_sequence"])


def test_non_speech_tokens_match_whisper():
    whisper = pytest.importorskip("whisper")
    mine = tk.get_tokenizer("en").non_speech_tokens
    ref = whisper.tokenizer.get_tokenizer(True, num_languages=99, language="en", task="transcribe").non_speech_tokens
    assert tuple(mine) == tuple(ref)


@pytest.mark.parametrize("stem", CLIPS)
def test_decode_roundtrip(stem):
    if not REF_JSON.exists():
        pytest.skip("ref_whisper.json missing")
    m = json.loads(REF_JSON.read_text(encoding="utf-8"))
    t = tk.get_tokenizer(stem)
    assert t.decode(m["clips"][stem]["generated_tokens"]).strip() == m["clips"][stem]["text"].strip()
