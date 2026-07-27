"""Greedy decode with SOT/suppress startup — the transcript producer (the v0.2 deliverable).

MM 2.3.C: naive (recompute the full token sequence each step, cache OFF — output is identical to the
cached version, which is MM 2.4's pure speedup). Matches openai/whisper's greedy without_timestamps path.

SOT sequence:  [sot, lang, transcribe, no_timestamps]   (from the tokenizer).
SuppressBlank: only at the FIRST generated step (len(tokens)==sample_begin) → -inf on encode(" ")+[eot].
SuppressTokens (every step): -inf on non_speech_tokens ∪ {transcribe, translate, sot, sot_prev, sot_lm, no_speech}.
  (NOT >= timestamp_begin — whisper relies on the no_timestamps prompt, so matching it means NOT suppressing them.)
Greedy: argmax; stop on eot; sample_len = n_text_ctx//2. generated = tokens after the SOT prefix, up to eot.

Ref: local/WhisperLiveKit/step-07 §5 (decoding.py GreedyDecoder + logit filters).
"""
from __future__ import annotations

import numpy as np
import torch

from . import audio as _audio
from .cache import KVCache


def _suppress_ids(tok) -> list[int]:
    ids = set(tok.non_speech_tokens)
    ids.update([tok.transcribe, tok.translate, tok.sot, tok.sot_prev, tok.sot_lm, tok.no_speech])
    return sorted(ids)


@torch.no_grad()
def transcribe(audio: np.ndarray, model, tokenizer, use_cache: bool = True) -> dict:
    """audio [n_samples] float32 @16k -> {"tokens": [content ids], "text": str}.

    use_cache=True: prefill the SOT prompt once, then feed only the new token each step; cross-attn KV is
    computed once (static), self-attn KV grows +1/step. use_cache=False: naive recompute of the full prefix
    each step. Both produce identical tokens (the cache is a pure speedup); this is the MM 2.4 gate.
    """
    mel = _audio.log_mel_spectrogram(_audio.pad_or_trim(audio))     # [80, 3000]
    xa = model.encoder(mel.unsqueeze(0))                            # [1, 1500, 768]  (encoder runs ONCE)

    tokens = list(tokenizer.sot_sequence)
    sample_begin = len(tokens)
    n_ctx = model.dims.n_text_ctx
    suppress = _suppress_ids(tokenizer)
    blank = tokenizer.encode(" ") + [tokenizer.eot]

    cache = KVCache() if use_cache else None
    cur = tokens                                                    # first step feeds the whole SOT prefix

    for _ in range(n_ctx // 2):                                     # whisper sample_len
        logits = model.decoder(torch.tensor([cur]), xa, kv_cache=cache)[0, -1].clone()
        if len(tokens) == sample_begin:                            # SuppressBlank: first generated token only
            logits[blank] = float("-inf")
        logits[suppress] = float("-inf")                           # SuppressTokens: every step
        nxt = int(logits.argmax())
        if nxt == tokenizer.eot:
            break
        tokens.append(nxt)
        if len(tokens) > n_ctx:                                    # 448 budget (prompt + generated)
            break
        cur = [nxt] if use_cache else tokens                       # cached: only the new token; naive: full prefix

    generated = tokens[sample_begin:]                              # drop the SOT prefix; eot already excluded
    return {"tokens": generated, "text": tokenizer.decode(generated)}
