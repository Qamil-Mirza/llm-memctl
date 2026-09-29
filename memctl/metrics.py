"""Answer scoring. Phase 1 has token F1; BLEU-1 and the LLM judge are added in Phase 2."""

from __future__ import annotations

import re
import string
from collections import Counter


def normalize_answer(text: str) -> list[str]:
    """Lowercase, remove punctuation and articles, split into words (SQuAD / LoCoMo style)."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the|and)\b", " ", text)
    return text.split()


def token_f1(prediction: str, gold: str) -> float:
    """Harmonic mean of word precision and recall between the prediction and the gold answer."""
    predicted, expected = normalize_answer(prediction), normalize_answer(gold)
    if not predicted or not expected:
        return float(predicted == expected)
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(predicted), overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None
