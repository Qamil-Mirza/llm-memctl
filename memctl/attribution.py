"""Failure attribution: every failed query gets exactly one label.

The point is to keep three things apart: what the controller did to memory,
what retrieval did, and what the task model did with the evidence it had.
"""

from __future__ import annotations

from memctl.hindsight.evidence import RequirementStatus
from memctl.memory.state import MemoryState

EVICTED = "evicted"
COMPRESSION_LOST_DETAIL = "compression_lost_detail"
CONSOLIDATION_INCORRECT = "consolidation_incorrect"
ARCHIVED_NOT_RETRIEVED = "archived_not_retrieved"
RETRIEVED_BUT_IGNORED = "retrieved_but_ignored"
TASK_MODEL_REASONING = "task_model_reasoning"
INVALID_ACTION = "invalid_action"
UNKNOWN = "unknown"

# Most severe first: when a query misses several things, the first matching label wins.
LABELS = [
    EVICTED, COMPRESSION_LOST_DETAIL, CONSOLIDATION_INCORRECT, ARCHIVED_NOT_RETRIEVED,
    RETRIEVED_BUT_IGNORED, TASK_MODEL_REASONING, INVALID_ACTION, UNKNOWN,
]
MEMORY_LABELS = [EVICTED, COMPRESSION_LOST_DETAIL, CONSOLIDATION_INCORRECT, INVALID_ACTION]
RETRIEVAL_LABELS = [ARCHIVED_NOT_RETRIEVED]
MODEL_LABELS = [RETRIEVED_BUT_IGNORED, TASK_MODEL_REASONING]

EXPLANATIONS = {
    EVICTED: "Needed information had been deleted.",
    COMPRESSION_LOST_DETAIL: "Compaction kept the item but lost the needed detail.",
    CONSOLIDATION_INCORRECT: "Consolidation produced an item without the needed information.",
    ARCHIVED_NOT_RETRIEVED: "Needed information was in the archive and was not retrieved.",
    RETRIEVED_BUT_IGNORED: "Needed information had been retrieved into context and the task model still failed.",
    TASK_MODEL_REASONING: "All needed information was in context. The task model failed.",
    INVALID_ACTION: "The controller's action was rejected and the fallback then deleted needed information.",
    UNKNOWN: "No evidence label is available for this query.",
}


def failure_label(
    statuses: list[RequirementStatus] | None, state: MemoryState, retrieved_now: list[str] | None = None
) -> tuple[str, str | None]:
    """(label, cause) for one failed query, from where its evidence was when the agent read memory.

    `retrieved_now` are the items retrieved at this step. `cause` says who deleted the
    information when the label is `evicted`: `controller`, or `harness` for the forced fallback.
    """
    if not statuses:
        return UNKNOWN, None
    missing = [status for status in statuses if not status.satisfied]
    if not missing:
        retrieved = set(retrieved_now or ())
        used_retrieval = any(item_id in retrieved for status in statuses for item_id in status.active)
        return (RETRIEVED_BUT_IGNORED if used_retrieval else TASK_MODEL_REASONING), None
    for kind in (EVICTED, COMPRESSION_LOST_DETAIL, CONSOLIDATION_INCORRECT):
        for status in missing:
            if status.loss == kind:
                if kind == EVICTED and status.loss_cause == "harness" and _controller_failed_at(state, status.loss_step, status.loss_item):
                    return INVALID_ACTION, "harness"
                return kind, status.loss_cause
    if any(status.archived for status in missing):
        return ARCHIVED_NOT_RETRIEVED, None
    return UNKNOWN, None


def _controller_failed_at(state: MemoryState, step: int | None, item_id: str | None) -> bool:
    """Was the forced eviction at `step` down to the controller's invalid actions?

    Yes if a rejected action at that step targeted the lost item, or if every action the
    controller emitted at that step was rejected (it did nothing valid, so the fallback ran).
    """
    rows = []
    for row in reversed(state.action_history):
        if row["step"] < (step or 0):
            break
        if row["step"] == step and row["source"] == "controller":
            rows.append(row)
    if not rows:
        return False
    rejected = [row for row in rows if row["status"] == "rejected"]
    if not rejected:
        return False
    if any(item_id in row["target_ids"] for row in rejected):
        return True
    return len(rejected) == len(rows)
