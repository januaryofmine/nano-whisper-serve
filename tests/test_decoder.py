"""MM 2.3.A correctness gate — hand-written TextDecoder matches openai/whisper's.

Compares decoder logits (teacher-forced SOT prefix + encoder output) against whisper's own decoder
module, bit-close (same SDPA kernel). Isolates the decoder from the greedy loop/tokenizer (MM 2.3.B/C).
Cache OFF. fp32 CPU.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from engine import audio, model

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "models" / "whisper-small"
REF_JSON = ROOT / "tests" / "ref_whisper.json"
FIXTURES = ROOT / "tests" / "fixtures" / "audio"
CLIPS = ("vi", "en")
TOL = 1e-3


def _need():
    if not (WEIGHTS / "small.pt").exists():
        pytest.skip("whisper-small weights missing")
    if not REF_JSON.exists():
        pytest.skip("ref_whisper.json missing — run `python -m engine.ref_whisper`")


@pytest.fixture(scope="module")
def models():
    _need()
    import whisper
    mine = model.load_whisper_small(str(WEIGHTS))
    ora = whisper.load_model("small", device="cpu", download_root=str(WEIGHTS)).eval()
    return mine, ora


def test_full_model_loads_strict():
    _need()
    model.load_whisper_small(str(WEIGHTS))  # raises if encoder+decoder name-map is wrong


@pytest.mark.parametrize("stem", CLIPS)
def test_decoder_logits_match_reference(stem, models):
    mine, ora = models
    m = json.loads(REF_JSON.read_text(encoding="utf-8"))
    sot = torch.tensor([m["clips"][stem]["sot_sequence"]])  # [1, 4] teacher-forced prefix

    a = audio.pad_or_trim(audio.load_audio(FIXTURES / f"{stem}.wav"))
    with torch.no_grad():
        xa = mine.encoder(audio.log_mel_spectrogram(a).unsqueeze(0))  # [1,1500,768]
        mine_logits = mine.decoder(sot, xa)                            # [1,4,51865]
        ora_logits = ora.decoder(sot, xa)

    assert mine_logits.shape == ora_logits.shape
    max_abs = float((mine_logits - ora_logits).abs().max())
    assert max_abs < TOL, f"{stem}: max|Δlogits|={max_abs:.3e} exceeds {TOL:.0e}"
    # next-token argmax (what greedy will use) must agree at every prefix position
    assert torch.equal(mine_logits.argmax(-1), ora_logits.argmax(-1)), f"{stem}: argmax disagrees"


# --- independent property tests (do NOT rely on the oracle; guard against a copied bug) ---

def test_decoder_is_causal():
    """Perturbing a later token must not change logits at earlier positions."""
    if not (WEIGHTS / "small.pt").exists():
        pytest.skip("whisper-small weights missing")
    m = model.load_whisper_small(str(WEIGHTS))
    xa = torch.randn(1, 1500, 768)
    toks = torch.tensor([[50258, 50259, 50359, 50363]])
    perturbed = toks.clone()
    perturbed[0, -1] = 1234
    with torch.no_grad():
        base, pert = m.decoder(toks, xa), m.decoder(perturbed, xa)
    assert torch.equal(base[:, :-1], pert[:, :-1]), "causality violated: earlier positions changed"
    assert (base[:, -1] - pert[:, -1]).abs().max() > 0, "changed position should differ"


def test_decoder_uses_cross_attention():
    """Logits must depend on the encoder output xa (cross-attention is actually wired)."""
    if not (WEIGHTS / "small.pt").exists():
        pytest.skip("whisper-small weights missing")
    m = model.load_whisper_small(str(WEIGHTS))
    toks = torch.tensor([[50258, 50259, 50359, 50363]])
    with torch.no_grad():
        a = m.decoder(toks, torch.randn(1, 1500, 768))
        b = m.decoder(toks, torch.zeros(1, 1500, 768))
    assert (a - b).abs().max() > 0, "logits independent of xa — cross-attention not wired"
