"""Whisper-small architecture in plain PyTorch (encoder-decoder).

SKELETON (MM 2.0). Encoder bodies -> MM 2.2, decoder -> MM 2.3, cache wiring -> MM 2.4.

Whisper-small dims (from openai/whisper config; verify against the checkpoint):
  n_mels=80, n_audio_ctx=1500, n_audio_state=768, n_audio_head=12, n_audio_layer=12,
  n_text_ctx=448, n_text_state=768, n_text_head=12, n_text_layer=12, n_vocab=51865.
  mlp_hidden = 4*768 = 3072 ; head_dim = 768/12 = 64.

Numerics gotchas (both openai/whisper + CTranslate2):
  - LayerNorm (has bias), eps=1e-5  (NOT RMSNorm 1e-6 like Qwen).
  - fp32 softmax + fp32 LayerNorm even under fp16 weights.
  - attention scale = (head_dim)**-0.25 applied to BOTH q and k.
  - K-projection has NO bias (query/value/out do).
  - Tied output: logits = x @ token_embedding.weight.T (.float()).  No separate LM head.
  - Encoder pos = SINUSOIDAL (fixed) ; decoder pos = LEARNED param.
  - Encoder: NO attention mask (audio padded to fixed 30s / 1500 frames).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class WhisperDims:
    n_mels: int = 80
    n_audio_ctx: int = 1500
    n_audio_state: int = 768
    n_audio_head: int = 12
    n_audio_layer: int = 12
    n_text_ctx: int = 448
    n_text_state: int = 768
    n_text_head: int = 12
    n_text_layer: int = 12
    n_vocab: int = 51865


class MultiHeadAttention(nn.Module):
    """Self- or cross-attention. kv_cache is an explicit dict (NOT hooks)."""

    def forward(self, x, xa=None, mask=None, kv_cache=None):
        # TODO MM 2.3/2.4: q=query(x); self: k,v=key/value(x) [+grow self cache dim=1];
        #   cross: k,v from xa [compute once, static]; scale=(d_head)**-0.25 on q&k;
        #   qk(+mask); softmax fp32; out_proj. See cache.py for the two-cache split.
        raise NotImplementedError("MM 2.3/2.4")


class ResidualAttentionBlock(nn.Module):
    """Pre-norm: self-attn -> (cross-attn if decoder) -> MLP."""

    def forward(self, x, xa=None, mask=None, kv_cache=None):
        raise NotImplementedError("MM 2.2/2.3")


class AudioEncoder(nn.Module):
    """conv1(k3s1)->GELU->conv2(k3s2, 3000->1500)->GELU->+sinusoids->N blocks->ln_post. Runs ONCE."""

    def forward(self, mel):  # [1,80,3000] -> [1,1500,768]
        raise NotImplementedError("MM 2.2")


class TextDecoder(nn.Module):
    """token_emb + learned pos; N blocks (self+cross+mlp); tied output projection."""

    def forward(self, tokens, xa, kv_cache=None):  # -> logits [1, seq, n_vocab]
        # TODO MM 2.3: offset = cached self-attn key length; index positional_embedding[offset:offset+T]
        raise NotImplementedError("MM 2.3")


class Whisper(nn.Module):
    def __init__(self, dims: WhisperDims):
        super().__init__()
        self.dims = dims
        # TODO MM 2.2/2.3: self.encoder = AudioEncoder(...); self.decoder = TextDecoder(...)


def load_whisper_small(model_dir: str) -> Whisper:
    """Load the whisper-small checkpoint into hand-written modules (name-map, tied emb, k-proj no bias)."""
    raise NotImplementedError("MM 2.2")
