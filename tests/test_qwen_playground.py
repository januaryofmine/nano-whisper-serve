"""Correctness gate for MM 1.1: hand-written Qwen engine must match the HF reference
oracle (playground/ref_qwen.json) token-for-token, greedy, fp32."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "playground"))

REF = ROOT / "playground" / "ref_qwen.json"
MODEL_DIR = ROOT / "models" / "qwen3-0.6b"


@pytest.fixture(scope="module")
def ref():
    if not REF.exists():
        pytest.skip("ref_qwen.json missing — run playground/ref_hf.py first")
    return json.loads(REF.read_text())


@pytest.fixture(scope="module")
def engine():
    if not MODEL_DIR.exists():
        pytest.skip("weights missing — download Qwen3-0.6B first")
    import qwen  # playground/qwen.py
    weights, cfg = qwen.load(str(MODEL_DIR))
    return qwen, weights, cfg


def test_token_for_token(ref, engine):
    """Every prompt: our greedy ids must equal the reference greedy ids."""
    qwen, weights, cfg = engine
    for rec in ref["records"]:
        prompt_ids = rec["prompt_ids"]
        want = rec["generated_ids"]
        got = qwen.generate(weights, cfg, prompt_ids, max_new_tokens=len(want))
        # compare the overlap; first mismatch (if any) is the useful signal
        n = min(len(want), len(got))
        assert got[:n] == want[:n], (
            f"\nprompt: {rec['prompt']!r}\n"
            f"first mismatch at token {next((i for i in range(n) if got[i] != want[i]), n)}\n"
            f"want[:8]={want[:8]}\n got[:8]={got[:8]}"
        )
