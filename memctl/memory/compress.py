"""Compressors (one item -> a shorter text) and consolidators (many items -> one text).

The engine calls these for COMPACT and CONSOLIDATE. Which one runs is an
experiment setting (`memory.compressor`, `memory.consolidator`), so the same
controller can be measured with a lossless and a lossy rewriter.
"""

from __future__ import annotations

import re
from typing import Protocol

from memctl.memory.items import count_tokens, truncate_tokens

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"[\w-]+")


def sentences(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCE_END.split(text) if part.strip()]


def salience(sentence: str) -> float:
    """How specific a sentence looks: identifiers, numbers and names score, plain words do not."""
    words = _WORD.findall(sentence)
    specific = 0
    for position, word in enumerate(words):
        has_digit = any(ch.isdigit() for ch in word)
        capitalised = word[:1].isupper() and position > 0
        if has_digit or capitalised:
            specific += 1
    return float(specific)


class Compressor(Protocol):
    name: str
    calls: int

    def compress(self, text: str, max_tokens: int) -> str: ...


class TruncateCompressor:
    """Keeps the first `max_tokens` tokens. Cheap and lossy."""

    name = "truncate"

    def __init__(self) -> None:
        self.calls = 0

    def compress(self, text: str, max_tokens: int) -> str:
        self.calls += 1
        return truncate_tokens(text, max_tokens)


class ExtractiveCompressor:
    """Keeps the most specific sentences that fit, in their original order."""

    name = "extractive"

    def __init__(self) -> None:
        self.calls = 0

    def compress(self, text: str, max_tokens: int) -> str:
        self.calls += 1
        parts = sentences(text)
        ranked = sorted(range(len(parts)), key=lambda n: (-salience(parts[n]), n))
        chosen, used = [], 0
        for number in ranked:
            size = count_tokens(parts[number])
            if used + size <= max_tokens:
                chosen.append(number)
                used += size
        if not chosen:  # even the best sentence is too long
            return truncate_tokens(parts[ranked[0]], max_tokens) if parts else ""
        return " ".join(parts[number] for number in sorted(chosen))


class LLMCompressor:
    """Asks a language model for a shorter version. The model is any `memctl.llm.LLM`."""

    name = "llm"

    def __init__(self, llm) -> None:
        self.llm = llm
        self.calls = 0

    def compress(self, text: str, max_tokens: int) -> str:
        self.calls += 1
        prompt = (
            f"Rewrite the text below in at most {max_tokens} words. Keep every name, number, "
            f"identifier and stated fact. Reply with the rewritten text only.\n\n{text}"
        )
        return truncate_tokens(self.llm.generate(prompt, max_new_tokens=2 * max_tokens).strip(), max_tokens)


class Consolidator(Protocol):
    name: str
    calls: int

    def consolidate(self, texts: list[str], max_tokens: int | None) -> str: ...


class DedupConsolidator:
    """Joins the texts and drops repeated sentences. If `max_tokens` is given and the
    result is longer, the compressor shortens it."""

    name = "dedup"

    def __init__(self, compressor: Compressor | None = None) -> None:
        self.compressor = compressor or ExtractiveCompressor()
        self.calls = 0

    def consolidate(self, texts: list[str], max_tokens: int | None) -> str:
        self.calls += 1
        seen, kept = set(), []
        for text in texts:
            for sentence in sentences(text):
                key = " ".join(sentence.lower().split())
                if key not in seen:
                    seen.add(key)
                    kept.append(sentence)
        merged = " ".join(kept)
        if max_tokens is not None and count_tokens(merged) > max_tokens:
            merged = self.compressor.compress(merged, max_tokens)
        return merged


class LLMConsolidator:
    name = "llm"

    def __init__(self, llm) -> None:
        self.llm = llm
        self.calls = 0

    def consolidate(self, texts: list[str], max_tokens: int | None) -> str:
        self.calls += 1
        limit = max_tokens or sum(count_tokens(text) for text in texts)
        notes = "\n".join(f"- {text}" for text in texts)
        prompt = (
            f"Merge the notes below into one note of at most {limit} words. State each fact once. "
            f"Keep every name, number and identifier. Reply with the merged note only.\n\n{notes}"
        )
        return truncate_tokens(self.llm.generate(prompt, max_new_tokens=2 * limit).strip(), limit)


def build_compressor(name: str = "extractive", llm=None) -> Compressor:
    if name == "truncate":
        return TruncateCompressor()
    if name == "extractive":
        return ExtractiveCompressor()
    if name == "llm":
        if llm is None:
            raise ValueError("the llm compressor needs a model (memory.compression_model)")
        return LLMCompressor(llm)
    raise KeyError(f"unknown compressor '{name}' (known: truncate, extractive, llm)")


def build_consolidator(name: str = "dedup", compressor: Compressor | None = None, llm=None) -> Consolidator:
    if name == "dedup":
        return DedupConsolidator(compressor)
    if name == "llm":
        if llm is None:
            raise ValueError("the llm consolidator needs a model (memory.compression_model)")
        return LLMConsolidator(llm)
    raise KeyError(f"unknown consolidator '{name}' (known: dedup, llm)")
