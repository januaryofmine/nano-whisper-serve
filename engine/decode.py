"""Greedy decode with SOT/suppress startup — the transcript producer.

SKELETON (MM 2.0). Bodies -> MM 2.3 (naive, caches off) then MM 2.4 (caches on).

SOT sequence:  [sot, lang, transcribe, no_timestamps]   where lang = sot + 1 + language_index.
SuppressBlank: at the FIRST GENERATED token (index = sample_begin = len(SOT)), NOT abs step 0 —
    force encode(" ") + [eot] to -inf.
SuppressTokens (every step): -inf on non_speech_tokens + {transcribe, translate, sot, sot_prev,
    sot_lm, no_speech} + every id >= timestamp_begin (timestamps are a CLAUDE.md §4 non-goal).
Greedy: logits[:, -1].argmax(). Never emit/stop-on-eot inside the SOT prefix; stop on eot or
    when length > n_text_ctx (448), counted AFTER the prefix.

Ref: local/WhisperLiveKit/step-07 §5 (decoding.py:417-433, 581-607, 674-704, GreedyDecoder);
     local/CTranslate2 step-03 (decoding_utils.cc:172 suppress-begin, decoding.cc:21/359/922).
"""
from __future__ import annotations

import numpy as np


def transcribe(audio: np.ndarray, model, tokenizer, language: str, use_cache: bool = True) -> dict:
    """audio [n_samples] -> {"tokens": [...ids], "text": str}. use_cache=False = naive (MM 2.3)."""
    # TODO MM 2.3: mel = pad_or_trim(log_mel_spectrogram(audio)); xa = model.encoder(mel)  # ONCE
    # TODO MM 2.3: tokens = SOT sequence; loop: decoder(...) -> suppress -> argmax -> append; stop
    # TODO MM 2.4: thread a KVCache through the decoder so cross-attn is projected once, self grows
    raise NotImplementedError("MM 2.3 / 2.4")
