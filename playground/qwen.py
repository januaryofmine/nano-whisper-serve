"""
qwen.py — MM 1.1: hand-written Qwen3-0.6B text engine (naive, NO KV cache).

Plain PyTorch. Each greedy step re-runs the WHOLE sequence through the 28 decoder
layers (O(n^2) work) — that's the point: this is the floor. The KV cache (MM 1.2)
removes the prefix recompute. Runs fp32 on CPU so it matches the HF reference
(playground/ref_qwen.json) token-for-token.

API:
    weights, cfg = load("models/qwen3-0.6b")
    ids = generate(weights, cfg, prompt_ids, max_new_tokens=32)   # -> list[int] (generated only)
"""
import json
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file


# ---------------------------------------------------------------------------
# load
# ---------------------------------------------------------------------------
def load(model_dir):
    """Read safetensors weights (as fp32) + the arch config we need."""
    model_dir = Path(model_dir)
    weights = load_file(str(model_dir / "model.safetensors"))
    weights = {k: v.float() for k, v in weights.items()}  # fp32 everywhere → matches HF fp32
    c = json.loads((model_dir / "config.json").read_text())
    cfg = dict(
        n_layers=c["num_hidden_layers"],          # 28
        n_heads=c["num_attention_heads"],          # 16
        n_kv_heads=c["num_key_value_heads"],       # 8
        head_dim=c["head_dim"],                    # 128 (NOT hidden/heads=64)
        eps=c["rms_norm_eps"],                     # 1e-6
        theta=c["rope_theta"],                     # 1e6
        eos=c["eos_token_id"],                     # 151645
    )
    return weights, cfg


# ---------------------------------------------------------------------------
# building blocks (all fp32)
# ---------------------------------------------------------------------------
def rmsnorm(x, w, eps):
    # x: [..., d].  RMS over the last dim, then scale by w. No mean-subtraction, no bias.
    var = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(var + eps) * w


def rope_cos_sin(seqlen, head_dim, theta):
    # rotate-half (GPT-NeoX) RoPE tables. inv_freq over half the head_dim.
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))  # [hd/2]
    pos = torch.arange(seqlen, dtype=torch.float32)                                             # [T]
    freqs = torch.outer(pos, inv_freq)                                                          # [T, hd/2]
    emb = torch.cat([freqs, freqs], dim=-1)                                                     # [T, hd]
    return emb.cos(), emb.sin()


def rotate_half(x):
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x, cos, sin):
    # x: [T, heads, hd];  cos/sin: [T, hd] -> broadcast over heads
    cos = cos[:, None, :]
    sin = sin[:, None, :]
    return x * cos + rotate_half(x) * sin


# ---------------------------------------------------------------------------
# forward (single sequence, causal, NO cache)
# ---------------------------------------------------------------------------
@torch.no_grad()
def forward(weights, cfg, ids):
    w = weights
    T = len(ids)
    H, HKV, HD = cfg["n_heads"], cfg["n_kv_heads"], cfg["head_dim"]
    eps = cfg["eps"]
    x = w["model.embed_tokens.weight"][torch.as_tensor(ids)]          # [T, 1024]

    cos, sin = rope_cos_sin(T, HD, cfg["theta"])                      # [T, 128] each
    # causal mask: query i may attend key j<=i.  upper triangle (j>i) = -inf
    mask = torch.full((T, T), float("-inf")).triu(1)                 # [T, T]
    scale = HD ** -0.5

    for i in range(cfg["n_layers"]):
        p = f"model.layers.{i}."
        # ---- self-attention ----
        h = rmsnorm(x, w[p + "input_layernorm.weight"], eps)         # [T, 1024]
        q = (h @ w[p + "self_attn.q_proj.weight"].T).view(T, H, HD)  # [T, 16, 128]
        k = (h @ w[p + "self_attn.k_proj.weight"].T).view(T, HKV, HD)  # [T, 8, 128]
        v = (h @ w[p + "self_attn.v_proj.weight"].T).view(T, HKV, HD)  # [T, 8, 128]
        # per-head QK-RMSNorm BEFORE RoPE (Qwen3-specific)
        q = rmsnorm(q, w[p + "self_attn.q_norm.weight"], eps)
        k = rmsnorm(k, w[p + "self_attn.k_norm.weight"], eps)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        # GQA: each of the 8 kv heads serves H/HKV=2 query heads -> repeat_interleave
        rep = H // HKV
        k = k.repeat_interleave(rep, dim=1)                          # [T, 16, 128]
        v = v.repeat_interleave(rep, dim=1)
        # [T, heads, hd] -> [heads, T, hd]
        q, k, v = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
        scores = (q @ k.transpose(-1, -2)) * scale + mask            # [16, T, T]
        attn = torch.softmax(scores, dim=-1)
        o = (attn @ v).transpose(0, 1).reshape(T, H * HD)            # [T, 2048]
        x = x + o @ w[p + "self_attn.o_proj.weight"].T               # [T, 1024]
        # ---- MLP (SwiGLU) ----
        h = rmsnorm(x, w[p + "post_attention_layernorm.weight"], eps)
        gate = h @ w[p + "mlp.gate_proj.weight"].T                   # [T, 3072]
        up = h @ w[p + "mlp.up_proj.weight"].T                       # [T, 3072]
        x = x + (torch.nn.functional.silu(gate) * up) @ w[p + "mlp.down_proj.weight"].T

    x = rmsnorm(x, w["model.norm.weight"], eps)                      # [T, 1024]
    logits = x @ w["lm_head.weight"].T                               # [T, vocab]
    return logits


@torch.no_grad()
def generate(weights, cfg, prompt_ids, max_new_tokens=32):
    """Greedy decode. NO cache -> re-runs the full sequence every step."""
    ids = list(prompt_ids)
    out = []
    for _ in range(max_new_tokens):
        logits = forward(weights, cfg, ids)      # [T, vocab]
        nxt = int(logits[-1].argmax())           # greedy = argmax over the last position
        ids.append(nxt)
        out.append(nxt)
        if nxt == cfg["eos"]:
            break
    return out


# ---------------------------------------------------------------------------
# CLI: compare against the reference answer key + print baseline tok/s
# ---------------------------------------------------------------------------
def main():
    root = Path(__file__).resolve().parent.parent
    weights, cfg = load(root / "models" / "qwen3-0.6b")
    ref = json.loads((root / "playground" / "ref_qwen.json").read_text())
    torch.manual_seed(0)
    total_toks, total_time, all_match = 0, 0.0, True
    for rec in ref["records"]:
        want = rec["generated_ids"]
        t0 = time.time()
        got = generate(weights, cfg, rec["prompt_ids"], max_new_tokens=len(want))
        dt = time.time() - t0
        n = min(len(want), len(got))
        match = got[:n] == want[:n]
        all_match &= match
        total_toks += len(got)
        total_time += dt
        mark = "✅" if match else "❌"
        print(f"{mark} {rec['prompt']!r}: {len(got)} toks in {dt:.1f}s"
              + ("" if match else f"  first-mismatch@{next((i for i in range(n) if got[i]!=want[i]), n)}"))
    print(f"\n{'ALL MATCH ✅' if all_match else 'MISMATCH ❌'} | baseline {total_toks/total_time:.2f} tok/s (fp32 CPU, no cache)")


if __name__ == "__main__":
    main()
