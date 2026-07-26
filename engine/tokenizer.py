"""Whisper BPE tokenizer (tiktoken) — thin wrapper.

SKELETON (MM 2.0). Body -> MM 2.3.

CLAUDE.md §7 permits tiktoken as a library. Pragmatic option (decide at MM 2.3): reuse
openai/whisper's own `whisper.tokenizer.get_tokenizer(multilingual=True, language=..., task="transcribe")`
— it is just a tiktoken.Encoding wrapper; the *learning* is the SOT/suppress logic in decode.py,
not re-typing the 51865-entry BPE table.

Key facts (local/WhisperLiveKit/step-07 §6):
  language token id = sot + 1 + language_index (fixed offset).
  decode() drops any token >= timestamp_begin.
  special accessors: sot, eot, transcribe, translate, sot_prev, sot_lm, no_speech,
    no_timestamps, timestamp_begin ; non_speech_tokens = the suppress set. "vi" is supported.
"""
from __future__ import annotations


def get_tokenizer(language: str, task: str = "transcribe"):
    """Return a Whisper multilingual tokenizer for `language`. MM 2.3."""
    raise NotImplementedError("MM 2.3")
