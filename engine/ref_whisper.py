"""Reference oracle for Milestone 2 — the layered answer key.

Runs openai/whisper (small) on the fixtures and dumps, per clip:
  - log-mel spectrogram + encoder output           -> tests/ref_whisper.npz  (MM 2.1 / 2.2 gates)
  - greedy token ids (generated) + SOT sequence + text  -> tests/ref_whisper.json  (MM 2.3 / 2.4 gate)

This is the *only* place openai/whisper is used — it is the reference oracle, NOT part of the
hand-written engine path. CPU + fp32 = deterministic answer key. The same wav is later read by the
engine, so feeding the identical float array to both isolates the engine's mel/encoder/decode code
from any I/O or resampler mismatch (WLK step-04 refinement).

Usage:
    .venv/bin/python -m engine.ref_whisper            # reads tests/fixtures/audio/{vi,en}.wav
    .venv/bin/python -m engine.ref_whisper --model small --download-root models/whisper-small
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import whisper

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures" / "audio"
OUT_NPZ = ROOT / "tests" / "ref_whisper.npz"
OUT_JSON = ROOT / "tests" / "ref_whisper.json"

# fixture filename stem -> language code
CLIPS = {"vi": "vi", "en": "en"}


def read_wav_mono16k(path: Path) -> np.ndarray:
    """Read a wav as float32 mono. Fixtures are already 16 kHz mono (asserted)."""
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:  # stereo -> mono
        audio = audio.mean(axis=1)
    if sr != whisper.audio.SAMPLE_RATE:
        raise ValueError(f"{path.name}: sample rate {sr} != 16000 — re-create fixture at 16 kHz")
    return np.ascontiguousarray(audio, dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="small")
    ap.add_argument("--download-root", default=str(ROOT / "models" / "whisper-small"))
    args = ap.parse_args()

    device = "cpu"  # fp32 CPU = deterministic reference
    model = whisper.load_model(args.model, device=device, download_root=args.download_root)
    model.eval()

    n_mels = model.dims.n_mels
    npz: dict[str, np.ndarray] = {}
    manifest: dict = {
        "meta": {
            "whisper_version": whisper.__version__,
            "torch_version": torch.__version__,
            "model": args.model,
            "device": device,
            "dtype": "float32",
            "dims": {k: int(v) for k, v in vars(model.dims).items()},
            "token_semantics": (
                "generated_tokens = content tokens only (openai/whisper DecodingResult.tokens): "
                "excludes the SOT prefix and the trailing eot, and contains no timestamp tokens. "
                "MM 2.3 gate: the engine's generated content tokens (up to but not including the "
                "eot it stops on) must equal this list, token-for-token."
            ),
        },
        "clips": {},
    }

    for stem, lang in CLIPS.items():
        wav = FIXTURES / f"{stem}.wav"
        if not wav.exists():
            raise FileNotFoundError(f"missing fixture {wav} — create it in MM 2.0 fixtures step")
        audio = read_wav_mono16k(wav)

        with torch.no_grad():
            mel = whisper.log_mel_spectrogram(
                whisper.pad_or_trim(torch.from_numpy(audio)), n_mels=n_mels
            )  # [n_mels, 3000]
            enc = model.encoder(mel.unsqueeze(0).to(device))  # [1, 1500, 768]
            options = whisper.DecodingOptions(
                task="transcribe", language=lang, without_timestamps=True,
                temperature=0.0, beam_size=None, fp16=False,
            )
            result = whisper.decode(model, mel.to(device), options)  # greedy

        tok = whisper.tokenizer.get_tokenizer(
            model.is_multilingual, num_languages=model.num_languages,
            language=lang, task="transcribe",
        )
        sot_seq = list(tok.sot_sequence_including_notimestamps)  # [sot, lang, transcribe, no_ts]

        npz[f"{stem}__mel"] = mel.cpu().numpy().astype(np.float32)
        npz[f"{stem}__encoder"] = enc.squeeze(0).cpu().numpy().astype(np.float32)
        manifest["clips"][stem] = {
            "language": lang,
            "duration_s": round(len(audio) / whisper.audio.SAMPLE_RATE, 3),
            "n_samples": int(len(audio)),
            "sot_sequence": [int(t) for t in sot_seq],
            "generated_tokens": [int(t) for t in result.tokens],  # what the engine must match
            "text": result.text,
        }
        print(f"[{stem}/{lang}] mel{tuple(mel.shape)} enc{tuple(enc.shape[1:])} "
              f"{len(result.tokens)} tok | {result.text!r}")

    OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_NPZ, **npz)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)  # ensure_ascii=False so VN survives
    print(f"\nwrote {OUT_NPZ.relative_to(ROOT)} ({OUT_NPZ.stat().st_size // 1024} KB) "
          f"+ {OUT_JSON.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
