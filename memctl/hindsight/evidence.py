"""Where each piece of needed evidence is right now, traced through rewrites.

A requirement names the observations that state a fact and a `needle` string.
Its *lineage* is those observations plus everything derived from them by
compaction or consolidation. A lineage member is a *carrier* if it is one of the
original observations or its content still contains the needle. This string
trace is what lets a failure be blamed on eviction, on a lossy rewrite, or on
retrieval, instead of on "memory" in general.

The tracker also watches future requirements (known from hindsight) and reports
the moment one becomes unrecoverable, which is the per-decision regret signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import islice

from memctl.memory.actions import ActionResult, ActionStatus, Operation
from memctl.memory.items import Fidelity, MemoryItem, Tier
from memctl.memory.state import MemoryState
from memctl.task import Dependency, EvidenceRequirement

REMOVALS = (
    Operation.EVICT, Operation.MOVE_TO_ARCHIVE, Operation.COMPACT,
    Operation.COMPACT_AND_ARCHIVE, Operation.CONSOLIDATE,
)


@dataclass
class RequirementStatus:
    requirement: EvidenceRequirement
    active: list[str] = field(default_factory=list)  # carriers the agent can read
    archived: list[str] = field(default_factory=list)  # carriers a retrieval could bring back
    loss: str | None = None  # why no carrier is reachable: evicted | compression_lost_detail | consolidation_incorrect
    loss_cause: str | None = None  # for evicted: controller | harness | intervention
    loss_step: int | None = None
    loss_item: str | None = None  # the carrier whose removal counts as the loss
    not_seen_yet: bool = False  # none of the source items has arrived

    @property
    def satisfied(self) -> bool:
        return bool(self.active)

    @property
    def recoverable(self) -> bool:
        return bool(self.active or self.archived)

    def log_row(self) -> dict:
        return {
            "source_ids": list(self.requirement.item_ids),
            "needle": self.requirement.needle,
            "active_carriers": self.active,
            "archived_carriers": self.archived,
            "loss": self.loss,
            "loss_cause": self.loss_cause,
            "loss_step": self.loss_step,
            "loss_item": self.loss_item,
        }


class EvidenceTracker:
    def __init__(self, state: MemoryState, future: list[Dependency] | None = None) -> None:
        """`future` is every dependency of the episode, from hindsight; None turns regret tracking off."""
        self.state = state
        self._children: dict[str, list[str]] = {}
        self._indexed = 0
        # Regret tracking: requirements that can still be met, and which items they hang on.
        self._alive: dict[tuple[str, int], tuple[Dependency, EvidenceRequirement]] = {}
        self._by_item: dict[str, set[tuple[str, int]]] = {}
        self._last_need: dict[str, int] = {}
        for dependency in future or []:
            for index, requirement in enumerate(dependency.requirements):
                key = (dependency.query_id, index)
                self._alive[key] = (dependency, requirement)
                for item_id in requirement.item_ids:
                    self._by_item.setdefault(item_id, set()).add(key)
                    self._last_need[item_id] = max(self._last_need.get(item_id, 0), dependency.step)

    # ---- lineage -------------------------------------------------------------

    def _index_new_items(self) -> None:
        items = self.state.items
        if self._indexed == len(items):
            return
        for item in islice(items.values(), self._indexed, None):
            for parent in item.derived_from_ids:
                self._children.setdefault(parent, []).append(item.id)
                for key in self._by_item.get(parent, ()):
                    self._by_item.setdefault(item.id, set()).add(key)
                if parent in self._last_need:
                    self._last_need[item.id] = max(self._last_need.get(item.id, 0), self._last_need[parent])
        self._indexed = len(items)

    def lineage(self, requirement: EvidenceRequirement) -> list[MemoryItem]:
        """The source items that have arrived, and everything derived from them."""
        self._index_new_items()
        found, queue, seen = [], list(requirement.item_ids), set()
        while queue:
            item_id = queue.pop()
            if item_id in seen or item_id not in self.state.items:
                continue
            seen.add(item_id)
            found.append(self.state.items[item_id])
            queue.extend(self._children.get(item_id, ()))
        return found

    def status(self, requirement: EvidenceRequirement) -> RequirementStatus:
        result = RequirementStatus(requirement)
        lineage = self.lineage(requirement)
        if not lineage:
            result.not_seen_yet = True
            return result
        sources = set(requirement.item_ids)
        losses = []
        for item in lineage:
            carries = item.id in sources or requirement.needle in item.content
            if carries and item.tier is Tier.ACTIVE:
                result.active.append(item.id)
            elif carries and item.tier is Tier.ARCHIVE:
                result.archived.append(item.id)
            elif carries and item.superseded_by is None:  # a carrier that was deleted outright
                event = self.state.tier_events[item.id][-1]
                losses.append(("evicted", event["source"], event["step"], item.id))
            elif not carries and item.fidelity is not Fidelity.FULL:  # a rewrite that dropped the needle
                kind = "consolidation_incorrect" if item.fidelity is Fidelity.CONSOLIDATED else "compression_lost_detail"
                losses.append((kind, None, item.created_at, item.id))
        if not result.recoverable and losses:
            order = {"evicted": 0, "compression_lost_detail": 1, "consolidation_incorrect": 2}
            result.loss, result.loss_cause, result.loss_step, result.loss_item = min(
                losses, key=lambda loss: (order[loss[0]], -loss[2])
            )
        return result

    def dependency_status(self, dependency: Dependency) -> list[RequirementStatus]:
        return [self.status(requirement) for requirement in dependency.requirements]

    # ---- regret: requirements that just became unrecoverable ------------------

    def after_actions(self, step: int, results: list[ActionResult]) -> list[dict]:
        """One row per future requirement that the actions of this step destroyed."""
        if not self._alive:
            return []
        self._index_new_items()
        touched: dict[tuple[str, int], ActionResult] = {}
        for result in results:
            if result.status is not ActionStatus.APPLIED or result.action.operation not in REMOVALS:
                continue
            for item_id in result.action.target_ids:
                for key in self._by_item.get(item_id, ()):
                    if key in self._alive:
                        touched.setdefault(key, result)
        rows = []
        for key, result in touched.items():
            dependency, requirement = self._alive[key]
            if dependency.step < step:  # the query has already been asked
                del self._alive[key]
                continue
            status = self.status(requirement)
            still_to_come = any(item_id not in self.state.items for item_id in requirement.item_ids)
            if status.recoverable or still_to_come:
                continue
            del self._alive[key]
            rows.append(
                {
                    "step": step,
                    "query_id": dependency.query_id,
                    "query_step": dependency.step,
                    "steps_until_needed": dependency.step - step,
                    "operation": result.action.operation.value,
                    "source": result.source.value,
                    "target_ids": [i for i in result.action.target_ids if key in self._by_item.get(i, ())],
                    "loss": status.loss,
                }
            )
        return rows

    def useless_active_tokens(self, step: int, exclude: str | None = None) -> int:
        """Tokens in ACTIVE held by items that no query at or after `step` needs
        (`exclude` is the observation just added, which is being answered, not kept)."""
        self._index_new_items()
        return sum(
            item.token_count for item in self.state.active()
            if item.id != exclude and self._last_need.get(item.id, 0) < step
        )
