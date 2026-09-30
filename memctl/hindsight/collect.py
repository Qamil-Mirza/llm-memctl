"""Hindsight: what an episode turned out to need, known only after it has been seen once.

It is built from a *reference pass*: the same episode (same seed) run with an
unlimited budget and no controller. That gives every item's size and arrival
step and every query's evidence. It is exact when the observation stream does
not depend on memory decisions, and only approximate otherwise (`exact=False`).

Hindsight is privileged. It goes to the oracle, to regret and attribution
analysis, and optionally to reward shaping or imitation labels. It is never part
of what an ordinary controller sees.
"""

from __future__ import annotations

import zlib
from bisect import bisect_left
from dataclasses import dataclass, field

from memctl.task import Dependency, EvidenceRequirement

NEVER = 10**9


def content_hash(content: str) -> int:
    return zlib.crc32(content.encode())


@dataclass(frozen=True)
class ItemRecord:
    id: str
    step: int
    tokens: int
    source_type: str
    content_hash: int = 0


@dataclass
class Hindsight:
    items: dict[str, ItemRecord]
    dependencies: list[Dependency]
    total_tokens: int  # tokens of the whole uncompressed history: the 100% budget
    horizon: int
    exact: bool = True
    reference_success: float = 1.0
    needs: dict[str, list[int]] = field(default_factory=dict)  # designated carrier -> query steps
    all_needs: dict[str, list[int]] = field(default_factory=dict)  # any alternative -> query steps

    def __post_init__(self) -> None:
        needs: dict[str, set[int]] = {}
        all_needs: dict[str, set[int]] = {}
        for dependency in self.dependencies:
            for requirement in dependency.requirements:
                needs.setdefault(self.designated(requirement), set()).add(dependency.step)
                for item_id in requirement.item_ids:
                    all_needs.setdefault(item_id, set()).add(dependency.step)
        self.needs = {item_id: sorted(steps) for item_id, steps in needs.items()}
        self.all_needs = {item_id: sorted(steps) for item_id, steps in all_needs.items()}

    def designated(self, requirement: EvidenceRequirement) -> str:
        """When several items state the same thing, the one a frugal policy would keep:
        the smallest, and the earliest among equals."""
        known = [self.items[i] for i in requirement.item_ids if i in self.items]
        if not known:
            return requirement.item_ids[0]
        return min(known, key=lambda record: (record.tokens, record.step)).id

    def next_need(self, item_id: str, step: int) -> int:
        """The first step at or after `step` at which the item is needed, or NEVER."""
        steps = self.needs.get(item_id)
        if not steps:
            return NEVER
        position = bisect_left(steps, step)
        return steps[position] if position < len(steps) else NEVER

    def needed_at_or_after(self, item_id: str, step: int) -> bool:
        return self.next_need(item_id, step) != NEVER

    def ever_needed(self, item_id: str) -> bool:
        return item_id in self.all_needs

    def matches(self, item_id: str, content: str) -> bool:
        """Is this the observation the reference pass saw under this id? False means the
        episode has diverged and hindsight no longer describes it."""
        record = self.items.get(item_id)
        return record is not None and record.content_hash == content_hash(content)


def hindsight_from_reference(state, dependencies: list[Dependency], exact: bool, success: float) -> Hindsight:
    """Build hindsight from the final memory state of a reference pass."""
    items = {
        item.id: ItemRecord(item.id, item.created_at, item.token_count, item.source_type.value, content_hash(item.content))
        for item in state.items.values()
    }
    return Hindsight(
        items=items,
        dependencies=list(dependencies),
        total_tokens=sum(record.tokens for record in items.values()),
        horizon=state.step,
        exact=exact,
        reference_success=success,
    )
