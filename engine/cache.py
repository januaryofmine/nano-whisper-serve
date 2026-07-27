"""The two KV caches — the M2 headline over M1's single self-attn cache.

An explicit dict keyed by `id(attention_module)`. Self-attn and cross-attn are different nn.Module
objects, so they get distinct entries automatically (no name-mangling needed — this is the vendored
openai/whisper's dict pattern, NOT install_kv_cache_hooks).

  SELF-attention  (grows +1 row / decode step):  torch.cat([cached, new], dim=1)  [batch, seq, state].
    positional offset = cached key length (indexed by TextDecoder before the step).
  CROSS-attention ("memory", static — "prefill in disguise"):  project K,V from encoder xa ONCE on the
    first pass, reused unchanged every later step. NEVER recomputed.

Ref: local/WhisperLiveKit/step-07 §4 (model.py:100-146, 306-311); local/CTranslate2 step-05
(attention.cc:371 static memory, 536-557 self growth).
"""
from __future__ import annotations

import torch


class KVCache:
    def __init__(self):
        self._self: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}   # module_id -> (k, v) accumulated
        self._cross: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}  # module_id -> (k, v) static

    def self_len(self, module_id: int) -> int:
        """Cached self-attn key length = the positional offset for the next token (0 if empty)."""
        kv = self._self.get(module_id)
        return 0 if kv is None else kv[0].shape[1]

    def grow_self(self, module_id: int, k_new: torch.Tensor, v_new: torch.Tensor):
        """Append new self-attn k,v on the seq axis (dim=1); return the full accumulated (k, v)."""
        prev = self._self.get(module_id)
        if prev is None:
            k, v = k_new, v_new
        else:
            k = torch.cat([prev[0], k_new], dim=1)
            v = torch.cat([prev[1], v_new], dim=1)
        self._self[module_id] = (k, v)
        return k, v

    def get_or_compute_cross(self, module_id: int, compute_fn):
        """Return cached cross-attn (k, v); compute once from xa on first call, then static."""
        kv = self._cross.get(module_id)
        if kv is None:
            kv = compute_fn()
            self._cross[module_id] = kv
        return kv
