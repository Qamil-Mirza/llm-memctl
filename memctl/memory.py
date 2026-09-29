"""MemoryState: where every item is. It enforces the budget and handles recall.

The budget rule, in one sentence: the tokens of all CONTEXT items plus the
index lines of all ARCHIVE items may never exceed `budget`. Placements are
checked *before* anything moves, so the state is never over budget.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from memctl.embed import Embedder, cosine
from memctl.items import Item, Place, count_tokens

if TYPE_CHECKING:
    from memctl.controllers.base import Placement


class BudgetError(Exception):
    """A set of placements would push the prompt over the token budget."""


class PlacementError(Exception):
    """A placement is not allowed (unknown item, illegal move, drop disabled)."""


class RecallError(Exception):
    """recall(id) was called for an item that is not in ARCHIVE."""


class MemoryState:
    """Tracks the place of every item for one conversation."""

    def __init__(
        self,
        budget: int,
        embedder: Embedder,
        allow_drop: bool = True,
        recall_cost: int = 50,
        store_top_k: int = 5,
        charge_archive_index: bool = True,
        reference_threshold: float = 0.3,
    ) -> None:
        self.budget = budget
        self.embedder = embedder
        self.allow_drop = allow_drop
        self.recall_cost = recall_cost
        self.store_top_k = store_top_k
        self.charge_archive_index = charge_archive_index
        self.reference_threshold = reference_threshold

        self.items: dict[str, Item] = {}
        self.place: dict[str, Place] = {}
        self.vectors: dict[str, object] = {}
        self.incoming: Item | None = None  # an item waiting for the controller to place it
        self.step = 0
        self.trigger = "overflow"  # why the controller is running: overflow | arrival | session_end
        self.last_used: dict[str, int] = {}
        self.use_count: dict[str, int] = {}
        self.recalls = 0
        self.recall_tokens = 0
        self.store_searches = 0

    # ---- reading the state -------------------------------------------------

    def in_place(self, place: Place) -> list[Item]:
        """All items currently in `place`, oldest first."""
        found = [self.items[i] for i, p in self.place.items() if p == place]
        return sorted(found, key=lambda item: item.arrival_step)

    def candidates(self) -> list[Item]:
        """Items the controller is asked about: everything in CONTEXT plus the incoming item."""
        items = self.in_place(Place.CONTEXT)
        return items + [self.incoming] if self.incoming else items

    def index_tokens(self, item: Item) -> int:
        """Tokens that the item's archive index line costs in the prompt."""
        return count_tokens(item.index_line()) if self.charge_archive_index else 0

    def used_tokens(self, places: dict[str, Place] | None = None) -> int:
        """Budgeted tokens for a placement map (default: the current one)."""
        places = self.place if places is None else places
        total = 0
        for item_id, place in places.items():
            if place == Place.CONTEXT:
                total += self.items[item_id].tokens
            elif place == Place.ARCHIVE:
                total += self.index_tokens(self.items[item_id])
        return total

    def archive_index(self) -> list[str]:
        return [item.index_line() for item in self.in_place(Place.ARCHIVE)]

    # ---- changing the state ------------------------------------------------

    def offer(self, item: Item) -> bool:
        """Add a new item. Returns True if it went straight into CONTEXT.

        If it does not fit, it waits in `self.incoming` and the controller
        must place it (and make room) with `apply`.
        """
        if item.id in self.items:
            raise PlacementError(f"item {item.id} was already added")
        if self.incoming is not None:
            raise PlacementError(f"item {self.incoming.id} is still waiting to be placed")
        self.items[item.id] = item
        self.vectors[item.id] = self.embedder.embed(item.text)
        self.last_used[item.id] = self.step
        self.use_count[item.id] = 0
        if self.used_tokens() + item.tokens <= self.budget:
            self.place[item.id] = Place.CONTEXT
            return True
        self.incoming = item
        return False

    def planned_places(self, placements: list[Placement]) -> dict[str, Place]:
        """The placement map that would result from `placements`. Checks every move is legal."""
        planned = dict(self.place)
        incoming_id = self.incoming.id if self.incoming else None
        for p in placements:
            if p.item_id not in self.items:
                raise PlacementError(f"unknown item {p.item_id}")
            current = self.place.get(p.item_id)
            if current == Place.DROPPED:
                raise PlacementError(f"item {p.item_id} was dropped and cannot be moved")
            if p.place == Place.DROPPED and not self.allow_drop:
                raise PlacementError("DROPPED is switched off (allow_drop=false)")
            if p.place == Place.CONTEXT and current != Place.CONTEXT and p.item_id != incoming_id:
                raise PlacementError(f"item {p.item_id} can only return to the prompt via search or recall")
            planned[p.item_id] = Place(p.place)
        if incoming_id is not None and incoming_id not in planned:
            raise PlacementError(f"the incoming item {incoming_id} was not placed")
        return planned

    def apply(self, placements: list[Placement]) -> None:
        """Move items. All-or-nothing: if the result breaks a rule, nothing changes."""
        planned = self.planned_places(placements)
        needed = self.used_tokens(planned)
        if needed > self.budget:
            raise BudgetError(f"placements need {needed} tokens but the budget is {self.budget}")
        self.place = planned
        self.incoming = None

    # ---- using memory --------------------------------------------------------

    def touch(self, item_id: str) -> None:
        """Record that an item was used at the current step."""
        self.last_used[item_id] = self.step
        self.use_count[item_id] += 1

    def similarity(self, item_id: str, query_vector: object) -> float:
        return cosine(self.vectors[item_id], query_vector)

    def mark_referenced(self, text: str) -> list[str]:
        """Touch the CONTEXT items that new text refers to (similarity above a threshold)."""
        query = self.embedder.embed(text)
        hit = [
            item.id
            for item in self.in_place(Place.CONTEXT)
            if self.similarity(item.id, query) >= self.reference_threshold
        ]
        for item_id in hit:
            self.touch(item_id)
        return hit

    def search_store(self, query_text: str) -> list[Item]:
        """Return the top-k STORE items most similar to the query."""
        self.store_searches += 1
        query = self.embedder.embed(query_text)
        stored = self.in_place(Place.STORE)
        ranked = sorted(stored, key=lambda item: self.similarity(item.id, query), reverse=True)
        top = ranked[: self.store_top_k]
        for item in top:
            self.touch(item.id)
        return top

    def recall(self, item_id: str) -> Item:
        """Bring back one archived item for this turn. Each recall has a token cost."""
        if self.place.get(item_id) != Place.ARCHIVE:
            where = self.place.get(item_id)
            raise RecallError(f"cannot recall {item_id}: it is {where.value if where else 'unknown'}")
        item = self.items[item_id]
        self.recalls += 1
        self.recall_tokens += self.recall_cost + item.tokens
        self.touch(item_id)
        return item
