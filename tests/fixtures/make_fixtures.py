"""Create the M2 correctness fixtures: 1 Vietnamese + 1 English clip, 16 kHz mono wav.

Uses the HF datasets-server "rows" API to fetch exactly ONE clip per language and downloads
just that single audio file (no `datasets` shard pulls — FLEURS's 691 MB parquet row-groups made
streaming/rows unusable). Content/transcript don't matter: the oracle is openai/whisper's own
output, so any clean speech clip in the right language works.

  EN: hf-internal-testing/librispeech_asr_dummy [clean/validation]  (flac, ~public)
  VN: doof-ferb/fpt_fosd [default/train]                            (mp3, FPT Open Speech Dataset)

Saved as 16 kHz mono so the mel gate feeds the identical float array to engine and oracle.

Usage:  .venv/bin/python tests/fixtures/make_fixtures.py
"""
from __future__ import annotations

import json
import tempfile
import urllib.request
from pathlib import Path

import librosa
import soundfile as sf

OUT = Path(__file__).resolve().parent / "audio"
TARGET_SR = 16000
MAX_S = 15.0  # keep fixtures short (≤ one 30s Whisper segment, small on disk)

# (stem, dataset, config, split, human label)
WANT = [
    ("en", "hf-internal-testing/librispeech_asr_dummy", "clean", "validation", "English"),
    ("vi", "doof-ferb/fpt_fosd", "default", "train", "Vietnamese"),
]


def pick_richest_src(dataset: str, config: str, split: str, scan: int = 12) -> str:
    """Fetch `scan` rows; return the audio src of the one with the longest transcript
    (proxy for a longer, more meaningful clip → a stronger token-for-token fixture)."""
    url = (f"https://datasets-server.huggingface.co/rows?dataset={dataset}"
           f"&config={config}&split={split}&offset=0&length={scan}")
    with urllib.request.urlopen(url, timeout=60) as r:
        data = json.load(r)
    best_src, best_len = None, -1
    for item in data["rows"]:
        row = item["row"]
        audio = row["audio"]
        src = audio[0]["src"] if isinstance(audio, list) else audio
        text = next((row[k] for k in ("transcription", "sentence", "text") if row.get(k)), "")
        if len(text) > best_len:
            best_src, best_len = src, len(text)
    if best_src is None:
        raise RuntimeError(f"{dataset}: no audio rows returned")
    return best_src


def make(stem: str, dataset: str, config: str, split: str, label: str) -> None:
    src = pick_richest_src(dataset, config, split)
    ext = ".mp3" if ".mp3" in src else (".flac" if ".flac" in src else ".wav")
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        urllib.request.urlretrieve(src, tmp.name)
        # librosa loads flac/mp3 (mp3 via ffmpeg), resamples to 16k, downmixes to mono in one call
        audio, _ = librosa.load(tmp.name, sr=TARGET_SR, mono=True)
    Path(tmp.name).unlink(missing_ok=True)

    if len(audio) / TARGET_SR > MAX_S:
        audio = audio[: int(MAX_S * TARGET_SR)]
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{stem}.wav"
    sf.write(str(path), audio, TARGET_SR, subtype="PCM_16")
    print(f"[{stem}] {label} ({dataset}): {len(audio)/TARGET_SR:.1f}s -> {path.name}")


if __name__ == "__main__":
    for spec in WANT:
        make(*spec)
    print("done")
