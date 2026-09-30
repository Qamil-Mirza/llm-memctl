"""Local answer scoring for QA benchmarks: token F1 and BLEU-1 (as in the LoCoMo paper).

These need no model. They are stricter and noisier than an LLM judge: a right
answer phrased differently scores low. Use them to compare controllers under
the same task model, not as absolute accuracy.
"""

from __future__ import annotations

import math
import re
import string
from collections import Counter

REFUSALS = ("not mentioned", "no information", "unknown", "cannot be determined", "don't know", "do not know")
F1_CORRECT_THRESHOLD = 0.5


def normalize_answer(text: str) -> list[str]:
    """Lowercase, remove punctuation and articles, split into words (SQuAD / LoCoMo style)."""
    text = str(text).lower().replace(",", "")
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the|and)\b", " ", text)
    return text.split()


def token_f1(prediction: str, gold: str) -> float:
    predicted, expected = normalize_answer(prediction), normalize_answer(gold)
    if not predicted or not expected:
        return float(predicted == expected)
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(predicted), overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def bleu1(prediction: str, gold: str) -> float:
    predicted, expected = normalize_answer(prediction), normalize_answer(gold)
    if not predicted or not expected:
        return float(predicted == expected)
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    brevity = 1.0 if len(predicted) >= len(expected) else math.exp(1 - len(expected) / len(predicted))
    return brevity * overlap / len(predicted)


def is_refusal(prediction: str) -> bool:
    return any(phrase in prediction.lower() for phrase in REFUSALS)


def contains_answer(prediction: str, gold: str) -> bool:
    """Every word of the gold answer appears in the prediction."""
    predicted, expected = Counter(normalize_answer(prediction)), Counter(normalize_answer(gold))
    return bool(expected) and not (expected - predicted)


def score_answer(prediction: str, gold: str, unanswerable: bool = False) -> dict:
    """F1, BLEU-1 and a right/wrong decision. For an unanswerable question the only right answer is a refusal."""
    prediction = prediction or ""
    if unanswerable:
        right = is_refusal(prediction)
        return {"f1": float(right), "bleu1": float(right), "correct": right, "decided_by": "refusal"}
    f1 = token_f1(prediction, gold)
    correct = f1 >= F1_CORRECT_THRESHOLD or contains_answer(prediction, gold)
    return {"f1": round(f1, 4), "bleu1": round(bleu1(prediction, gold), 4), "correct": correct,
            "decided_by": f"f1>={F1_CORRECT_THRESHOLD} or contains gold"}
