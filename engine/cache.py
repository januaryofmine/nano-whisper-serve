"""The two KV caches — the M2 headline over M1's single self-attn cache.

SKELETON (MM 2.0). Bodies -> MM 2.4 (MM 2.3 runs with caches OFF, naive recompute).

An explicit dict threaded into MultiHeadAttention.forward(..., kv_cache=dict), keyed per
module id (NOT openai/whisper's install_kv_cache_hooks — the vendored copy dropped those).

  SELF-attention cache  (grows +1 row / decode step):
    torch.cat([cached_k, k], dim=1)   # seq axis = 1 for [batch, seq, state] layout
    positional offset = cached key length  (index positional_embedding[offset:offset+T])

  CROSS-attention cache ("memory", static — "prefill in disguise"):
    project K,V from the encoder output `xa` ONCE on pass 1 (guard: if id not in cache),
    reuse unchanged every later step. NEVER recompute.

Ref: local/WhisperLiveKit/step-07 §4 (model.py:100-146, 306-311);
     local/CTranslate2 step-05 (attention.cc:371 static memory, 536-557 self growth).
"""
from __future__ import annotations


class KVCache:
    """Dict-based two-cache holder. MM 2.4."""

    def __init__(self):
        self._store: dict = {}

    def grow_self(self, key_id: str, k, v):
        """Append new self-attn k,v on dim=1; return the full accumulated (k, v)."""
        raise NotImplementedError("MM 2.4")

    def get_or_compute_cross(self, key_id: str, project_fn, xa):
        """Return cached cross-attn (k, v); compute once from xa on first call, then static."""
        raise NotImplementedError("MM 2.4")

    def self_len(self, key_id: str) -> int:
        """Cached self-attn key length = the positional offset for the next token (0 if empty)."""
        raise NotImplementedError("MM 2.4")
