"""
ref_hf.py — the "answer key" for Milestone 1.

This is NOT the engine. It runs HuggingFace `transformers.generate(do_sample=False)`
(the reference oracle) on a few fixed prompts and dumps the exact token ids + the model's
architecture facts to JSON. Your hand-written engine (playground/qwen.py) must reproduce
these token ids — that's the correctness gate before any tokens/sec number counts.

Usage:
    python playground/ref_hf.py                         # auto device/dtype, default model
    python playground/ref_hf.py --dtype float32         # most reproducible (recommended for the gate)
    python playground/ref_hf.py --max-new-tokens 40 --out playground/ref_qwen.json

Run your engine in the SAME dtype/device as this reference when comparing token-for-token.
fp32 gives the longest exact match; bf16/fp16 may diverge in late tokens (floating point,
not a bug — see 1-mini-nano-vllm.md gotchas).
"""

import argparse
import json
import platform

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Fixed prompts — deterministic under greedy. Keep these stable so the answer key is stable.
PROMPTS = [
    "The capital of France is",
    "Once upon a time,",
    "Vietnam is a country in",
]


def resolve_device(arg: str) -> str:
    if arg != "auto":
        return arg
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"  # skip mps on purpose: CPU fp32 is the most reproducible reference


def resolve_dtype(arg: str, device: str) -> torch.dtype:
    if arg != "auto":
        return getattr(torch, arg)
    # auto: fp32 on CPU (stable), bf16 on GPU (matches how the engine will run)
    return torch.float32 if device == "cpu" else torch.bfloat16


def arch_facts(cfg) -> dict:
    """Pull the numbers your hand-written model must match (confirms the config.json facts)."""
    head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // cfg.num_attention_heads)
    # transformers 5.x may nest rope_theta under rope_parameters/rope_scaling → look there too
    rope_theta = getattr(cfg, "rope_theta", None)
    if rope_theta is None:
        d = cfg.to_dict()
        rope_theta = (d.get("rope_theta")
                      or (d.get("rope_parameters") or {}).get("rope_theta")
                      or (d.get("rope_scaling") or {}).get("rope_theta"))
    return {
        "model_type": getattr(cfg, "model_type", None),
        "num_hidden_layers": cfg.num_hidden_layers,
        "hidden_size": cfg.hidden_size,
        "num_attention_heads": cfg.num_attention_heads,
        "num_key_value_heads": getattr(cfg, "num_key_value_heads", cfg.num_attention_heads),
        "head_dim": head_dim,
        "intermediate_size": cfg.intermediate_size,
        "vocab_size": cfg.vocab_size,
        "rope_theta": rope_theta,
        "rms_norm_eps": getattr(cfg, "rms_norm_eps", None),
        "tie_word_embeddings": getattr(cfg, "tie_word_embeddings", None),
        "hidden_act": getattr(cfg, "hidden_act", None),
        "max_position_embeddings": getattr(cfg, "max_position_embeddings", None),
        "torch_dtype": str(getattr(cfg, "torch_dtype", None)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto", help="auto | float32 | bfloat16 | float16")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--out", default="playground/ref_qwen.json")
    args = ap.parse_args()

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    print(f"[ref] model={args.model} device={device} dtype={dtype}")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()

    facts = arch_facts(model.config)
    print("[ref] architecture facts:")
    for k, v in facts.items():
        print(f"        {k}: {v}")

    records = []
    for prompt in PROMPTS:
        enc = tok(prompt, return_tensors="pt").to(device)  # raw tokenization, no chat template
        prompt_ids = enc["input_ids"][0].tolist()
        with torch.no_grad():
            out = model.generate(
                **enc,
                do_sample=False,          # GREEDY — deterministic, so the ids are the answer key
                num_beams=1,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
            )
        full_ids = out[0].tolist()
        gen_ids = full_ids[len(prompt_ids):]
        records.append({
            "prompt": prompt,
            "prompt_ids": prompt_ids,
            "generated_ids": gen_ids,
            "generated_text": tok.decode(gen_ids),
        })
        print(f"[ref] {prompt!r} -> +{len(gen_ids)} toks: {tok.decode(gen_ids)!r}")

    payload = {
        "model": args.model,
        "device": device,
        "dtype": str(dtype),
        "max_new_tokens": args.max_new_tokens,
        "env": {
            "torch": torch.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "arch": facts,
        "records": records,
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[ref] wrote answer key → {args.out}")


if __name__ == "__main__":
    main()
