<div align="center">

# nano-whisper-serve

A Whisper inference engine, hand-written in plain PyTorch.

![Python 3.11+](https://img.shields.io/badge/python-3.11+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![tests](https://img.shields.io/badge/tests-pytest-0A9EDC)

</div>

## Key features

* **Speech-to-text transcription**<br/>Turns a 16 kHz audio clip into text. Multilingual, built on Whisper-small's 99 languages, with Vietnamese and English verified end-to-end.
* **Runs on CPU or GPU**<br/>Plain PyTorch (FP32 or BF16): transcribes on a laptop CPU with no special setup, and moves to a single GPU for speed with no code change.
* **Reference-grade accuracy**<br/>Produces the same transcript as `openai/whisper` (small), token-for-token under greedy decoding so output quality matches the reference model exactly.
* **Fast incremental decoding**<br/>Greedy decode is accelerated by two KV caches (self- and cross-attention), running ~2.2× faster than naive recompute while returning identical tokens.
* **Simple integration**<br/>Depends only on PyTorch and `tiktoken` at inference time, loads the standard ~460 MB Whisper-small checkpoint with no conversion step, and transcribes a clip in a few lines of Python.

## Status

| Milestone | What | State |
|---|---|---|
| Qwen text engine (`playground/`) | hand-written Qwen3-0.6B forward + greedy + KV cache + static batching | ✅ |
| Whisper core (`engine/`) | mel → encoder (once) → decode loop with two caches (static cross-attn + growing self-attn) | ✅ |
| Serving engine (`serving/`) | continuous batching over N concurrent audio streams; RTF + throughput benchmarks | working |
| Streaming shell (`demo/`) | WebSocket, partial transcripts, LocalAgreement token locking | working |

## Installation and usage

Clone the repo and install the dependencies into a virtualenv:

```bash
git clone https://github.com/januaryofmine/nano-whisper-serve
cd nano-whisper-serve
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

Download the Whisper-small weights (public, ~460 MB) into `models/`:

```bash
.venv/bin/python -c "import whisper; whisper.load_model('small', download_root='models/whisper-small')"
```

Transcribe an audio file in a few lines of Python:

```python
import librosa
from engine import decode, model, tokenizer

m = model.load_whisper_small("models/whisper-small")                 # load once
audio, _ = librosa.load("audio.mp3", sr=16000, mono=True)            # any format -> 16 kHz mono
result = decode.transcribe(audio, m, tokenizer.get_tokenizer("vi"))  # language: "vi" or "en"
print(result["text"])
```

`transcribe` returns `{"text", "tokens"}` and processes the first 30-second segment.

Run the correctness gate (differential test against `openai/whisper`):

```bash
.venv/bin/python tests/fixtures/make_fixtures.py   # fetch the VN + EN audio fixtures
.venv/bin/python -m engine.ref_whisper             # build the reference answer key
.venv/bin/python -m pytest tests/ -q
```

The Qwen text-engine sandbox lives in [`playground/`](playground/) and the throughput sweep in [`benchmarks/`](benchmarks/).

## Qwen3-0.6B, plain PyTorch

**Correctness first.** The hand-written engine reproduces HuggingFace
`generate(do_sample=False)` **token-for-token** (fp32, greedy is deterministic) on every
prompt, a differential test against a reference oracle (`playground/ref_qwen.json`).

**Optimization story**:

| Step | Change | Number | Roofline explanation |
|---|---|---|---|
| v0.0 | naive forward + greedy, **no cache** | floor (~5.6 tok/s, CPU fp32) | recomputes the whole prefix every step → O(n²) work; deep memory-bound |
| v0.1 | **+ KV cache** (contiguous, not paged) | ~2.7× (CPU A/B, same output) | each decode step computes K/V for only the new token → single-token decode is memory-bound, arithmetic intensity ≈ 1; the cache removes the redundant prefix recompute |
| v0.1 | **+ static batching** | **27 → 2095 tok/s, ~76×** (P100) | batching amortizes one weight read across the batch → intensity rises with batch size, moving from memory-bound toward compute-bound at `B_crit` |

### The B_crit curve (throughput vs batch size, Tesla P100, fp32)

![tokens/sec vs batch size](benchmarks/qwen_tokens_per_sec_vs_batch.png)

| batch | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 |
|---|--|--|--|--|--|--|--|--|--|
| tok/s | 27 | 56 | 111 | 217 | 431 | 850 | 1496 | 1949 | 2095 |
| ms/step | 37 | 36 | 36 | 37 | 37 | 38 | 43 | 66 | 122 |

Throughput scales **~linearly up to ~batch 64** while per-step latency stays flat (~37 ms) —
the **memory-bound** regime, where adding a sequence is nearly free because the weight read is
amortized. Past ~batch 64–128 the curve **bends** and per-step latency climbs (43 → 66 → 122 ms):
the matmuls have become **compute-bound**. That knee is `B_crit`
(≈ peak_FLOP / peak_bandwidth for the GPU). Raw numbers + GPU metadata:
[`benchmarks/qwen_tokens_per_sec_vs_batch.json`](benchmarks/qwen_tokens_per_sec_vs_batch.json).

## Whisper-small, plain PyTorch

The engine transcribes audio end-to-end, log-mel → encoder (run once) → greedy decode with two KV
caches — all hand-written. It reproduces `openai/whisper` (small, greedy, `without_timestamps`)
**token-for-token** on both a Vietnamese and an English clip.

**Correctness is gated in layers** against a reference oracle (`engine/ref_whisper.py` dumps
`openai/whisper`'s own mel, encoder output, and token stream as the answer key). Each layer must match
before the next is trusted:

| Layer | Check | Result |
|---|---|---|
| mel front-end | `max\|Δ\|` vs `whisper.log_mel_spectrogram` | **0.0** (bit-exact) |
| encoder output | `max\|Δ\|` vs `whisper` encoder | **0.0** (bit-exact) |
| transcript | token-for-token vs `whisper` greedy | **exact** (VN + EN, + 8 diverse/edge clips) |

**Optimization story**:

| Step | Change | Number | Roofline explanation |
|---|---|---|---|
| v0.2 | naive greedy decode, **no cache** | ~14 tok/s (CPU) | recomputes the whole token prefix every step **and** re-projects cross-attention K/V from all 1500 encoder frames × 12 layers each step |
| v0.2 | **+ two KV caches** | **~2.2×** (CPU, identical output) | self-attention K/V grow one row per step; cross-attention K/V are computed **once** from the static encoder output and reused — decode stays memory-bound (arithmetic intensity ≈ 1), the caches just remove the redundant recompute |

### self-attention cache vs cross-attention cache

Whisper is encoder–decoder, so the decoder carries **two** attention caches that behave differently:

- **Self-attention cache**: identical to a decoder-only LLM (Milestone 1): it holds the K/V of the tokens
  generated so far and **grows by one entry per step** (`torch.cat` on the sequence axis). The new token's
  positional offset is the current cached length.
- **Cross-attention cache**: the decoder attends to the encoder output, which is **fixed for the whole
  30-second segment**. So its K/V are projected **once** on the first decode pass and reused unchanged for
  every later token, "prefill in disguise". The naive path re-reads 1500 encoder frames through 12 layers
  every step for nothing.

That second cache is the only structural addition over the text decoder,  the greedy loop, the
growing self-attention cache, and the attention/LayerNorm blocks all port directly.
