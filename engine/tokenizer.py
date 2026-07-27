"""Whisper BPE tokenizer — tiktoken only (engine stays independent of the openai/whisper package).

Ports whisper's special-token layout + LANGUAGES + non_speech suppress set into our own module; the
BPE ranks come from the shipped assets/multilingual.tiktoken (tiktoken is the sanctioned tokenizer
library — CLAUDE.md §7). Verified against the oracle: special ids, SOT sequence, non_speech_tokens.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
from functools import cached_property, lru_cache
from pathlib import Path

import tiktoken

_ASSETS = Path(__file__).parent / "assets"

# Whisper language codes, in the exact order that fixes the language-token ids (id = sot+1+index).
LANGUAGES = (
    "en zh de es ru ko fr ja pt tr pl ca nl ar sv it id hi fi vi he uk el ms cs ro da hu ta no th ur "
    "hr bg lt la mi ml cy sk te fa lv bn sr az sl kn et mk br eu is hy ne mn bs kk sq sw gl mr pa si "
    "km sn yo so af oc ka be tg sd gu am yi lo uz fo ht ps tk nn mt sa lb my bo tl mg as tt haw ln ha "
    "ba jw su yue"
).split()

_PAT = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


@lru_cache(maxsize=None)
def _build_encoding(num_languages: int = 99) -> tiktoken.Encoding:
    """Reproduce whisper.tokenizer.get_encoding('multilingual') using the shipped vocab asset."""
    ranks = {
        base64.b64decode(tok): int(rank)
        for tok, rank in (line.split() for line in open(_ASSETS / "multilingual.tiktoken") if line)
    }
    n = len(ranks)
    specials = [
        "<|endoftext|>",
        "<|startoftranscript|>",
        *[f"<|{lang}|>" for lang in LANGUAGES[:num_languages]],
        "<|translate|>",
        "<|transcribe|>",
        "<|startoflm|>",
        "<|startofprev|>",
        "<|nospeech|>",
        "<|notimestamps|>",
        *[f"<|{i * 0.02:.2f}|>" for i in range(1501)],
    ]
    special_tokens = {tok: n + i for i, tok in enumerate(specials)}
    return tiktoken.Encoding(
        name="multilingual", explicit_n_vocab=n + len(specials), pat_str=_PAT,
        mergeable_ranks=ranks, special_tokens=special_tokens,
    )


@dataclass(frozen=True)
class Tokenizer:
    encoding: tiktoken.Encoding
    language: str
    task: str = "transcribe"
    num_languages: int = 99

    def _sid(self, s: str) -> int:
        return self.encoding.encode_single_token(s)

    @property
    def eot(self) -> int: return self._sid("<|endoftext|>")
    @property
    def sot(self) -> int: return self._sid("<|startoftranscript|>")
    @property
    def transcribe(self) -> int: return self._sid("<|transcribe|>")
    @property
    def translate(self) -> int: return self._sid("<|translate|>")
    @property
    def sot_lm(self) -> int: return self._sid("<|startoflm|>")
    @property
    def sot_prev(self) -> int: return self._sid("<|startofprev|>")
    @property
    def no_speech(self) -> int: return self._sid("<|nospeech|>")
    @property
    def no_timestamps(self) -> int: return self._sid("<|notimestamps|>")
    @property
    def timestamp_begin(self) -> int: return self._sid("<|0.00|>")

    @property
    def language_token(self) -> int:
        return self.sot + 1 + LANGUAGES.index(self.language)

    @property
    def sot_sequence(self) -> tuple[int, ...]:
        """[sot, language, task, no_timestamps] — the greedy-decode startup prefix."""
        task_id = self.transcribe if self.task == "transcribe" else self.translate
        return (self.sot, self.language_token, task_id, self.no_timestamps)

    @cached_property
    def non_speech_tokens(self) -> tuple[int, ...]:
        """Ids to suppress so the model won't emit non-speech annotations (♪, [NAME], etc.)."""
        symbols = list('"#()*+/:;<=>@[\\]^_`{|}~「」『』')
        symbols += "<< >> <<< >>> -- --- -( -[ (' (\" (( )) ((( ))) [[ ]] {{ }} ♪♪ ♪♪♪".split()
        miscellaneous = set("♩♪♫♬♭♮♯")
        # allow "-"/"'" between words but not at the start of a word
        result = {self.encoding.encode(" -")[0], self.encoding.encode(" '")[0]}
        for symbol in symbols + list(miscellaneous):
            for tokens in (self.encoding.encode(symbol), self.encoding.encode(" " + symbol)):
                if len(tokens) == 1 or symbol in miscellaneous:
                    result.add(tokens[0])
        return tuple(sorted(result))

    def encode(self, text: str) -> list[int]:
        return self.encoding.encode(text)

    def decode(self, token_ids: list[int]) -> str:
        """Text output: drop timestamp tokens (>= timestamp_begin), like whisper."""
        ids = [t for t in token_ids if t < self.timestamp_begin]
        return self.encoding.decode(ids)


def get_tokenizer(language: str, task: str = "transcribe", num_languages: int = 99) -> Tokenizer:
    return Tokenizer(_build_encoding(num_languages), language=language, task=task, num_languages=num_languages)
