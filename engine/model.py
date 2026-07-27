"""Whisper-small architecture in plain PyTorch (encoder-decoder).

Encoder (MM 2.2) implemented; decoder bodies + cache wiring land in MM 2.3 / 2.4.
Submodule names match the openai/whisper checkpoint so `load_state_dict(strict=True)` is the
name-map proof. Numerics faithful to openai/whisper 20250625: LayerNorm eps 1e-5 in fp32,
attention via F.scaled_dot_product_attention (same kernel the oracle uses → matches bit-close;
SDPA's internal 1/sqrt(head_dim) scale == the d**-0.25-on-both-q,k form), K-projection has no
bias, encoder has NO mask (is_causal=False).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class WhisperDims:
    n_mels: int = 80
    n_vocab: int = 51865
    n_audio_ctx: int = 1500
    n_audio_state: int = 768
    n_audio_head: int = 12
    n_audio_layer: int = 12
    n_text_ctx: int = 448
    n_text_state: int = 768
    n_text_head: int = 12
    n_text_layer: int = 12


class LayerNorm(nn.LayerNorm):
    """Whisper computes LayerNorm in fp32 then casts back (stability under fp16)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return super().forward(x.float()).type(x.dtype)


def sinusoids(length: int, channels: int, max_timescale: int = 10000) -> torch.Tensor:
    """Fixed sinusoidal position table [length, channels] (encoder only)."""
    assert channels % 2 == 0
    log_inc = math.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_inc * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, None] * inv_timescales[None, :]
    return torch.cat([scaled_time.sin(), scaled_time.cos()], dim=1)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = nn.Linear(n_state, n_state)
        self.key = nn.Linear(n_state, n_state, bias=False)  # K-projection: NO bias
        self.value = nn.Linear(n_state, n_state)
        self.out = nn.Linear(n_state, n_state)

    def forward(self, x, xa=None, mask=None, kv_cache=None):
        q = self.query(x)
        if kv_cache is None:
            # self-attn (xa is None) or cross-attn (xa is the encoder output), recomputed each call
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            raise NotImplementedError("MM 2.4: kv_cache path (self grows, cross static)")
        return self.out(self._qkv_attention(q, k, v, mask))

    def _qkv_attention(self, q, k, v, mask=None):
        # Mirror openai/whisper 20250625: F.scaled_dot_product_attention (SDPA applies the
        # 1/sqrt(head_dim) scale internally = the (d**-0.25 on both q,k) form). A `mask` present with
        # n_ctx > 1 means the causal decoder-prefill case -> is_causal=True; encoder (mask=None) and
        # single-token decode (n_ctx==1) -> is_causal=False. Matching SDPA = matching the oracle's kernel.
        n_batch, n_ctx, _ = q.shape
        q = q.view(n_batch, q.shape[1], self.n_head, -1).permute(0, 2, 1, 3)
        k = k.view(n_batch, k.shape[1], self.n_head, -1).permute(0, 2, 1, 3)
        v = v.view(n_batch, v.shape[1], self.n_head, -1).permute(0, 2, 1, 3)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=(mask is not None and n_ctx > 1))
        return a.permute(0, 2, 1, 3).flatten(2)             # [b, tq, n_state]


class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
        super().__init__()
        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = LayerNorm(n_state)
        self.cross_attn = MultiHeadAttention(n_state, n_head) if cross_attention else None
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None
        self.mlp = nn.Sequential(nn.Linear(n_state, 4 * n_state), nn.GELU(), nn.Linear(4 * n_state, n_state))
        self.mlp_ln = LayerNorm(n_state)

    def forward(self, x, xa=None, mask=None, kv_cache=None):
        x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)
        if self.cross_attn is not None:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache)
        x = x + self.mlp(self.mlp_ln(x))
        return x


class AudioEncoder(nn.Module):
    """conv1(k3s1)->GELU->conv2(k3s2, halves time)->GELU->+sinusoids->N self-attn blocks->ln_post."""

    def __init__(self, n_mels, n_ctx, n_state, n_head, n_layer):
        super().__init__()
        self.conv1 = nn.Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))
        self.blocks = nn.ModuleList(ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer))
        self.ln_post = LayerNorm(n_state)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:  # [b, n_mels, 3000] -> [b, 1500, n_state]
        x = F.gelu(self.conv1(mel))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)                                  # channel-major -> seq-major
        assert x.shape[1:] == self.positional_embedding.shape, "encoder input shape != pos-emb"
        x = (x + self.positional_embedding).to(x.dtype)
        for block in self.blocks:
            x = block(x)                                        # self-attn only, NO mask
        return self.ln_post(x)


class TextDecoder(nn.Module):
    """token emb + learned pos + N blocks (self-attn causal + cross-attn to encoder) + ln + tied output."""

    def __init__(self, n_vocab, n_ctx, n_state, n_head, n_layer):
        super().__init__()
        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))  # LEARNED (vs sinusoidal enc)
        self.blocks = nn.ModuleList(
            ResidualAttentionBlock(n_state, n_head, cross_attention=True) for _ in range(n_layer)
        )
        self.ln = LayerNorm(n_state)
        mask = torch.empty(n_ctx, n_ctx).fill_(float("-inf")).triu_(1)  # strict-upper -inf (causal)
        self.register_buffer("mask", mask, persistent=False)            # not in state_dict

    def forward(self, tokens: torch.Tensor, xa: torch.Tensor, kv_cache=None) -> torch.Tensor:
        # offset = number of already-cached self-attn positions (0 when cache OFF / prefill). MM 2.4 fills it.
        offset = 0 if kv_cache is None else kv_cache.self_len(id(self.blocks[0].attn))
        x = self.token_embedding(tokens) + self.positional_embedding[offset:offset + tokens.shape[-1]]
        x = x.to(xa.dtype)
        for block in self.blocks:
            x = block(x, xa, mask=self.mask, kv_cache=kv_cache)  # self-attn causal; cross-attn to xa
        x = self.ln(x)
        return (x @ self.token_embedding.weight.T).float()      # tied output; logits in fp32


class Whisper(nn.Module):
    def __init__(self, dims: WhisperDims):
        super().__init__()
        self.dims = dims
        self.encoder = AudioEncoder(dims.n_mels, dims.n_audio_ctx, dims.n_audio_state,
                                    dims.n_audio_head, dims.n_audio_layer)
        self.decoder = TextDecoder(dims.n_vocab, dims.n_text_ctx, dims.n_text_state,
                                   dims.n_text_head, dims.n_text_layer)

    @property
    def is_multilingual(self) -> bool:
        return self.dims.n_vocab >= 51865

    @property
    def num_languages(self) -> int:
        return self.dims.n_vocab - 51765 - int(self.is_multilingual)


def load_whisper_small(model_dir: str) -> Whisper:
    """Load the whisper-small checkpoint into hand-written modules (encoder + decoder)."""
    ckpt = torch.load(Path(model_dir) / "small.pt", map_location="cpu")
    m = Whisper(WhisperDims(**ckpt["dims"]))
    m.load_state_dict(ckpt["model_state_dict"], strict=True)  # strict=True == full name-map proof
    m.eval()
    return m
