"""
bench_qwen.py — MM 1.3: static-batching throughput sweep -> the first B_crit curve.

Measures tokens/sec of the cached engine as a function of batch size (uniform: the same
prompt decoded N ways in lockstep). Follows the standard inference-bench discipline
(gpt-fast / nano-vllm): one untimed warmup, device-synchronize around the timed region,
fixed output length, tok/s = batch * max_new_tokens / elapsed. Records GPU metadata so the
JSON is reproducible and environment-agnostic (CLAUDE.md §7).

Usage (CPU smoke):   python benchmarks/bench_qwen.py --device cpu --batches 1,2,4 --max-new-tokens 16
Usage (GPU curve):   python benchmarks/bench_qwen.py --device cuda --batches 1,2,4,8,16,32,64,128,256
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "playground"))


def env_meta(device: str) -> dict:
    m = {"torch": torch.__version__, "python": platform.python_version(),
         "platform": platform.platform(), "device": device}
    if device == "cuda" and torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        m.update(gpu=p.name, vram_gb=round(p.total_memory / 1e9, 1), cuda=torch.version.cuda)
    return m


def main() -> None:
    import qwen  # playground/qwen.py

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda")
    ap.add_argument("--batches", default="1,2,4,8,16,32", help="comma-separated batch sizes")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--model", default=str(ROOT / "models" / "qwen3-0.6b"))
    ap.add_argument("--out", default=str(ROOT / "benchmarks" / "qwen_tokens_per_sec_vs_batch"))
    a = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    torch.set_default_device(device)               # new tensors (cache/mask/rope) land on device
    batches = [int(b) for b in a.batches.split(",")]

    weights, cfg = qwen.load(a.model)
    weights = {k: v.to(device) for k, v in weights.items()}
    # tokenize the prompt once via the reference oracle's ids if available, else a tiny fallback
    ref = ROOT / "playground" / "ref_qwen.json"
    prompt_ids = json.loads(ref.read_text())["records"][0]["prompt_ids"] if ref.exists() else [785]

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    results = []
    for B in batches:
        qwen.generate_batched(weights, cfg, prompt_ids, B, max_new_tokens=4)   # warmup (untimed)
        sync()
        t0 = time.time()
        qwen.generate_batched(weights, cfg, prompt_ids, B, max_new_tokens=a.max_new_tokens)
        sync()
        dt = time.time() - t0
        tps = B * a.max_new_tokens / dt
        results.append({"batch": B, "tok_per_s": round(tps, 2), "elapsed_s": round(dt, 3),
                        "latency_ms_per_step": round(1000 * dt / a.max_new_tokens, 2)})
        print(f"  batch={B:>4}  {tps:8.2f} tok/s   ({dt:.2f}s, {1000*dt/a.max_new_tokens:.1f} ms/step)")

    payload = {"env": env_meta(device), "max_new_tokens": a.max_new_tokens,
               "prompt_len": len(prompt_ids), "results": results}
    Path(a.out + ".json").write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {a.out}.json")

    # plot (import late so the sweep works even if matplotlib is missing)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = [r["batch"] for r in results]
        ys = [r["tok_per_s"] for r in results]
        plt.figure(figsize=(6, 4))
        plt.plot(xs, ys, "o-")
        plt.xscale("log", base=2)
        plt.xlabel("batch size (sequences)")
        plt.ylabel("tokens / sec")
        gpu = payload["env"].get("gpu", device)
        plt.title(f"Qwen3-0.6B static-batching throughput ({gpu})")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(a.out + ".png", dpi=120)
        print(f"wrote {a.out}.png")
    except Exception as e:  # noqa: BLE001 — plotting is optional glue
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
