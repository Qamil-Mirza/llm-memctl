"""Failure attribution: every wrong answer gets exactly one label.

The labels separate MEMORY failures (the evidence was not in front of the
model) from REASONING failures (the evidence was there and the model still got
it wrong). This split is the core of the thesis.
"""

from __future__ import annotations

EVIDENCE_DROPPED = "evidence_dropped"
ARCHIVED_NOT_RECALLED = "evidence_archived_not_recalled"
IN_STORE_NOT_RETRIEVED = "evidence_in_store_not_retrieved"
PRESENT_MODEL_WRONG = "evidence_present_model_wrong"
NO_EVIDENCE_LABEL = "no_evidence_label"

LABELS = [EVIDENCE_DROPPED, ARCHIVED_NOT_RECALLED, IN_STORE_NOT_RETRIEVED, PRESENT_MODEL_WRONG, NO_EVIDENCE_LABEL]
MEMORY_FAILURES = [EVIDENCE_DROPPED, ARCHIVED_NOT_RECALLED, IN_STORE_NOT_RETRIEVED]

EXPLANATIONS = {
    EVIDENCE_DROPPED: "A needed item had been deleted.",
    ARCHIVED_NOT_RECALLED: "A needed item was in the archive and the agent did not recall it.",
    IN_STORE_NOT_RETRIEVED: "A needed item was in the store but the search did not return it.",
    PRESENT_MODEL_WRONG: "All needed items were in the prompt. The model reasoned wrongly.",
    NO_EVIDENCE_LABEL: "The benchmark gives no evidence for this question, so the cause is unknown.",
}


def failure_label(evidence: list[dict]) -> str:
    """The label for one wrong answer, given where each evidence item was.

    If several evidence items were missing for different reasons, the most
    severe reason wins: dropped, then archived, then not retrieved.
    """
    if not evidence:
        return NO_EVIDENCE_LABEL
    missing = [e for e in evidence if not e["in_prompt"]]
    if any(e["place"] == "DROPPED" for e in missing):
        return EVIDENCE_DROPPED
    if any(e["place"] == "ARCHIVE" for e in missing):
        return ARCHIVED_NOT_RECALLED
    if any(e["place"] == "STORE" for e in missing):
        return IN_STORE_NOT_RETRIEVED
    return PRESENT_MODEL_WRONG
