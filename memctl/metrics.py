"""Answer scoring: token F1 and BLEU-1 (as in the LoCoMo, A-Mem and Memory-R1 papers)."""

from __future__ import annotations

import math
import re
import string
from collections import Counter

REFUSALS = ("not mentioned", "no information")


def normalize_answer(text: str) -> list[str]:
    """Lowercase, remove punctuation and articles, split into words (SQuAD / LoCoMo style)."""
    text = str(text).lower().replace(",", "")
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


def bleu1(prediction: str, gold: str) -> float:
    """Share of predicted words that appear in the gold answer, with a penalty for short answers."""
    predicted, expected = normalize_answer(prediction), normalize_answer(gold)
    if not predicted or not expected:
        return float(predicted == expected)
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    precision = overlap / len(predicted)
    brevity = 1.0 if len(predicted) >= len(expected) else math.exp(1 - len(expected) / len(predicted))
    return brevity * precision


def is_refusal(prediction: str) -> bool:
    """True if the answer says the information is not in the conversation."""
    return any(phrase in prediction.lower() for phrase in REFUSALS)


def score_answer(prediction: str, gold: str, category: str) -> dict[str, float]:
    """F1 and BLEU-1 for one answer.

    Adversarial questions have no answer in the conversation. As in the official
    LoCoMo scoring, the answer is right (1.0) only if the model says so.
    """
    if category == "adversarial":
        right = float(is_refusal(prediction))
        return {"f1": right, "bleu1": right}
    return {"f1": round(token_f1(prediction, gold), 4), "bleu1": round(bleu1(prediction, gold), 4)}


def mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None
