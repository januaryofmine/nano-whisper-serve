"""MM 2.4 correctness gate — the two KV caches are a pure speedup (output unchanged).

Cached decode must produce the SAME tokens as the naive path AND the oracle, on both fixtures.
Plus cache mechanics: cross-attn cache is static (computed once), self-attn cache grows by one per step.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from engine import audio, cache as kv, decode, model, tokenizer as tk

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "models" / "whisper-small"
REF_JSON = ROOT / "tests" / "ref_whisper.json"
FIXTURES = ROOT / "tests" / "fixtures" / "audio"
CLIPS = ("vi", "en")


def _need():
    if not (WEIGHTS / "small.pt").exists():
        pytest.skip("whisper-small weights missing")
    if not REF_JSON.exists():
        pytest.skip("ref_whisper.json missing")


@pytest.fixture(scope="module")
def whisper_model():
    _need()
    return model.load_whisper_small(str(WEIGHTS))


@pytest.mark.parametrize("stem", CLIPS)
def test_cached_matches_naive_and_reference(stem, whisper_model):
    _need()
    ref = json.loads(REF_JSON.read_text(encoding="utf-8"))["clips"][stem]["generated_tokens"]
    a = audio.load_audio(FIXTURES / f"{stem}.wav")
    tok = tk.get_tokenizer(stem)
    naive = decode.transcribe(a, whisper_model, tok, use_cache=False)["tokens"]
    cached = decode.transcribe(a, whisper_model, tok, use_cache=True)["tokens"]
    assert cached == naive == ref, f"{stem}: cached/naive/ref diverge"


def test_cross_cache_static_self_grows(whisper_model):
    """Cross-attn KV is computed once (same tensor object across steps); self-attn KV grows +1/step."""
    _need()
    a = audio.load_audio(FIXTURES / "en.wav")
    tok = tk.get_tokenizer("en")
    with torch.no_grad():
        mel = audio.log_mel_spectrogram(audio.pad_or_trim(a))
        xa = whisper_model.encoder(mel.unsqueeze(0))
        c = kv.KVCache()
        b0 = whisper_model.decoder.blocks[0]
        self_id, cross_id = id(b0.attn), id(b0.cross_attn)

        sot = list(tok.sot_sequence)
        whisper_model.decoder(torch.tensor([sot]), xa, kv_cache=c)  # prefill (4 tokens)
        cross_after_prefill = c.get_or_compute_cross(cross_id, lambda: None)
        assert c.self_len(self_id) == len(sot)                      # self cache = 4

        whisper_model.decoder(torch.tensor([[1234]]), xa, kv_cache=c)  # one decode step
        assert c.self_len(self_id) == len(sot) + 1                  # self grew to 5
        # cross-attn cache is the SAME object (recomputed would be a new tensor)
        assert c.get_or_compute_cross(cross_id, lambda: None)[0] is cross_after_prefill[0]
