"""The three answer shares of a reader cell (answered-correct, answered-wrong, unknown), kept disjoint.

An answer is "unknown" when it starts with "unknown" (the frozen 7b5fc30 prompt's abstention word). The judge grades
every answer, so an "unknown" answer can in principle be judged correct (for example when the gold answer itself
says the information is not given). Such an answer would be counted twice if correct and unknown were tallied
separately: it is counted here as unknown only, and flagged, so a report can say that the headline accuracy (every
answer judged correct) and the answered-correct share differ.
"""

from __future__ import annotations


def is_unknown(answer: str | None) -> bool:
    return (answer or "").strip().lower().startswith("unknown")


def answer_shares(rows: list[dict]) -> dict:
    """rows: dicts with `correct` (bool) and `unknown` (bool) or `answer` (text). Returns the three disjoint shares
    (summing to 1), the headline accuracy, and the ids of unknown answers judged correct (`flagged`)."""
    if not rows:
        raise ValueError("no rows")
    counts = {"answered_correct": 0, "answered_wrong": 0, "unknown": 0}
    flagged = []
    correct = 0
    for row in rows:
        unknown = bool(row["unknown"]) if "unknown" in row else is_unknown(row.get("answer"))
        correct += bool(row["correct"])
        if unknown:
            counts["unknown"] += 1
            if row["correct"]:
                flagged.append(row.get("question_id"))
        elif row["correct"]:
            counts["answered_correct"] += 1
        else:
            counts["answered_wrong"] += 1
    n = len(rows)
    shares = {key: value / n for key, value in counts.items()}
    return {**shares, "accuracy": correct / n, "n": n, "flagged": flagged,
            "p_correct_given_answered": (counts["answered_correct"] / (n - counts["unknown"]))
            if n > counts["unknown"] else None}
