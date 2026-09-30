"""The hindsight oracle. For analysis only: it sees which items later queries need.

Two methods:

- `approx` (default): evict what is never needed again, then, if still over
  budget, remove the item whose next need is furthest away. When archiving and
  retrieval are allowed it archives instead of deleting and retrieves each item
  at the step it is needed. **Approximate when item sizes differ**: furthest-
  next-use is optimal only for equal sizes, so treat it as a strong reference,
  not a bound.
- `exact`: follow the plan from hindsight/exact.py (delete-only action set). If
  no exact plan can be computed the controller falls back to `approx` and says
  so in `decision_info()["oracle_method"]`.

The oracle never compacts or consolidates, so when those operations are allowed
a controller that uses them may legitimately beat it.
"""

from __future__ import annotations

import time

from memctl.controllers.base import MemoryController
from memctl.hindsight.collect import NEVER, Hindsight
from memctl.hindsight.exact import ExactPlan, OracleUnavailable, solve_delete_only
from memctl.memory.actions import MemoryAction, Operation
from memctl.memory.state import MemoryView
from memctl.task import TaskState


class OracleController(MemoryController):
    name = "oracle"
    uses_hindsight = True

    def __init__(self, config: dict, seed: int = 0) -> None:
        super().__init__(config, seed)
        self.method = config.get("method", "approx")
        self.eager = bool(config.get("eager", True))
        self.hindsight: Hindsight | None = None
        self.plan: ExactPlan | None = None
        self.method_used = self.method
        self.fallback_reason = ""
        self.solve_seconds = 0.0  # time spent computing the exact plan; not part of decision latency

    @property
    def display_name(self) -> str:
        return self.config.get("label") or ("oracle_exact" if self.method == "exact" else "oracle_approx")

    def receive_hindsight(self, hindsight: Hindsight) -> None:
        self.hindsight, self.plan = hindsight, None
        self.method_used, self.fallback_reason = self.method, ""
        self.solve_seconds = 0.0
        if self.method == "exact":
            started = time.perf_counter()
            try:
                self.plan = solve_delete_only(
                    hindsight, self.episode.budget, float(self.config.get("time_limit_s", 60.0)),
                    int(self.config.get("max_variables", 50_000)),
                )
            except OracleUnavailable as error:
                self.method_used, self.fallback_reason = "approx", str(error)
            self.solve_seconds = time.perf_counter() - started

    def decide(self, memory: MemoryView, task: TaskState) -> list[MemoryAction]:
        if self.hindsight is None:
            raise RuntimeError("the oracle was not given hindsight")
        return self._follow_plan(memory) if self.plan is not None else self._approximate(memory)

    def _follow_plan(self, memory: MemoryView) -> list[MemoryAction]:
        gone = [
            item.id for item in memory.active
            if not item.pinned and self.plan.keep_until.get(item.id, 0) < memory.step
        ]
        return [MemoryAction(Operation.EVICT, tuple(gone), confidence=1.0)] if gone else []

    def _approximate(self, memory: MemoryView) -> list[MemoryAction]:
        step, hindsight = memory.step, self.hindsight
        can_archive = self.allows(Operation.MOVE_TO_ARCHIVE) and self.allows(Operation.RETRIEVE_FROM_ARCHIVE)
        can_evict = self.allows(Operation.EVICT)
        actions: list[MemoryAction] = []
        projected = memory.active_tokens

        due = [item for item in memory.archived if hindsight.next_need(item.id, step) == step]
        if due and can_archive:
            actions.append(
                MemoryAction(
                    Operation.RETRIEVE_FROM_ARCHIVE, tuple(item.id for item in due),
                    parameters={"method": "oracle"}, confidence=1.0,
                )
            )
            projected += sum(item.token_count for item in due)

        movable = [item for item in memory.active if not item.pinned]
        never = [item for item in movable if hindsight.next_need(item.id, step) == NEVER]
        needed = [item for item in movable if hindsight.next_need(item.id, step) != NEVER]
        drop_operation = Operation.EVICT if can_evict else Operation.MOVE_TO_ARCHIVE
        dropped: list[str] = []
        if self.eager:
            dropped = [item.id for item in never]
            projected -= sum(item.token_count for item in never)
        else:
            for item in never:  # oldest first, only as far as the budget requires
                if projected <= memory.budget:
                    break
                dropped.append(item.id)
                projected -= item.token_count
        if dropped:
            actions.append(MemoryAction(drop_operation, tuple(dropped), confidence=1.0))

        if projected > memory.budget:
            # Furthest next use goes first. Items needed at this very step go last.
            furthest_first = sorted(needed, key=lambda item: -hindsight.next_need(item.id, step))
            removed = []
            for item in furthest_first:
                if projected <= memory.budget:
                    break
                removed.append(item.id)
                projected -= item.token_count
            if removed:
                operation = Operation.MOVE_TO_ARCHIVE if can_archive else Operation.EVICT
                actions.append(MemoryAction(operation, tuple(removed), confidence=1.0))
        return actions

    def decision_info(self) -> dict:
        info = {"oracle_method": self.method_used, "oracle_solve_s": self.solve_seconds}
        if self.fallback_reason:
            info["oracle_fallback_reason"] = self.fallback_reason
        if self.plan is not None:
            info["oracle_covered"] = self.plan.covered
        return info
