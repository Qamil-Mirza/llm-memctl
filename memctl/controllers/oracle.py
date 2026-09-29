"""The oracle: a hindsight plan built from the benchmark's evidence labels.

This is the yardstick, not a real method. It is the only controller that sees
evidence labels (`uses_evidence_labels = True`); the runner hands it an
`OraclePlan` and hands other controllers nothing.

Two methods:
- `belady`: evict the item whose next need is furthest away (never-needed items
  first). Exactly optimal only when items have equal size and evicted items can
  come back.
- `ilp`: follow the exact plan from oracle_ilp.py, which handles different item
  sizes and items that cannot come back into context.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from memctl.benchmarks.types import Conversation
from memctl.controllers.base import Controller, Placement, evict_until_fits
from memctl.controllers.oracle_ilp import solve_keep_plan
from memctl.features import ItemFeatures
from memctl.items import Item, Place
from memctl.memory import MemoryState

NEVER = 10**9


@dataclass
class OraclePlan:
    """What hindsight says about every item."""

    needs: dict[str, list[int]]  # item id -> steps at which a question needs it
    keep_until: dict[str, int] = field(default_factory=dict)  # exact plan: last step to stay in CONTEXT
    allow_drop: bool = True
    needed_items_go_to: Place = Place.STORE

    def next_need(self, item_id: str, step: int) -> int:
        """The next step at or after `step` that needs the item, or NEVER."""
        return min((s for s in self.needs.get(item_id, []) if s >= step), default=NEVER)

    def needs_left(self, item_id: str, step: int) -> int:
        return sum(1 for s in self.needs.get(item_id, []) if s >= step)

    def place_for(self, item_id: str, step: int) -> Place:
        """The oracle's choice for one item at one step."""
        if self.next_need(item_id, step) == NEVER:
            return Place.DROPPED if self.allow_drop else Place.STORE
        if self.keep_until.get(item_id, 0) >= step:
            return Place.CONTEXT
        return self.needed_items_go_to


def build_plan(conversation: Conversation, budget: int, allow_drop: bool, needed_items_go_to: str = "STORE") -> OraclePlan:
    keep_until, _ = solve_keep_plan(conversation, budget)
    return OraclePlan(conversation.need_times(), keep_until, allow_drop, Place(needed_items_go_to))


class OracleController(Controller):
    name = "oracle"
    uses_evidence_labels = True

    def __init__(self, config: dict, seed: int = 0) -> None:
        super().__init__(config, seed)
        self.method = config.get("method", "ilp")
        self.plan: OraclePlan | None = None

    @property
    def display_name(self) -> str:
        return "oracle" if self.method == "ilp" else f"oracle-{self.method}"

    def receive_plan(self, plan: OraclePlan) -> None:
        self.plan = plan

    def decide(self, state: MemoryState, features: dict[str, ItemFeatures], budget: int) -> list[Placement]:
        plan, step = self.plan, state.step

        def eviction_rank(item: Item) -> tuple:
            """Smaller rank = evicted earlier."""
            next_need = plan.next_need(item.id, step)
            if next_need == NEVER:
                return (0, item.arrival_step)
            if self.method == "ilp":
                planned_to_stay = plan.keep_until.get(item.id, 0) >= step
                return (2 if planned_to_stay else 1, -next_need, item.arrival_step)
            value_per_token = plan.needs_left(item.id, step) / item.tokens
            return (1, -next_need, value_per_token)

        def destination(item: Item) -> Place:
            if plan.next_need(item.id, step) == NEVER:
                return Place.DROPPED if state.allow_drop else Place.STORE
            return plan.needed_items_go_to

        def reason(item: Item) -> str:
            next_need = plan.next_need(item.id, step)
            if next_need == NEVER:
                return "hindsight: no later question needs this item"
            return f"hindsight: next needed at step {next_need}, but other items are needed sooner or more"

        order = sorted(state.candidates(), key=eviction_rank)
        return evict_until_fits(state, order, destination, reason, confidence=lambda item: 1.0)
