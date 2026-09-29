"""The Controller interface, the Placement it returns, and the registry.

To add a controller: write one file with a Controller subclass, then add one
line to `registry()` below.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

from memctl.features import ItemFeatures
from memctl.items import Item, Place
from memctl.memory import BudgetError, MemoryState


@dataclass(frozen=True)
class Placement:
    """One decision: put this item in this place, and why."""

    item_id: str
    place: Place
    reason: str
    confidence: float | None = None


class Controller(ABC):
    """Decides where items live. Subclasses implement `decide`."""

    name = "base"
    ignores_budget = False  # True only for full_context, which is given an unlimited budget
    runs_on_every_arrival = False  # True for controllers that file items even when there is room

    def __init__(self, config: dict, seed: int = 0) -> None:
        self.config = config
        self.seed = seed

    @property
    def display_name(self) -> str:
        """Name used in every output. Fake-Jev runs override this to include FAKE."""
        return self.name

    @abstractmethod
    def decide(
        self, state: MemoryState, features: dict[str, ItemFeatures], budget: int
    ) -> list[Placement]:
        """Return placements for `state.candidates()`.

        Items that are not mentioned stay where they are, except the incoming
        item (`state.incoming`), which must always be placed. After the
        placements are applied the state must fit in `budget`.
        """


def evict_until_fits(
    state: MemoryState,
    eviction_order: list[Item],
    destination: Callable[[Item], Place],
    reason: Callable[[Item], str],
    confidence: Callable[[Item], float | None] = lambda item: None,
    start_from: dict[str, Place] | None = None,
) -> list[Placement]:
    """Evict items in the given order until the incoming item fits in the budget.

    Shared by most controllers: they only differ in the order and destination.
    If archiving does not free enough room (index lines also cost tokens),
    archived evictions fall back to STORE. `start_from` is an optional placement
    map to start from, for controllers that have already made some choices.
    """
    planned = dict(start_from if start_from is not None else state.place)
    if state.incoming:
        planned.setdefault(state.incoming.id, Place.CONTEXT)
    chosen: dict[str, Placement] = {}
    for item in eviction_order:
        if state.used_tokens(planned) <= state.budget:
            break
        planned[item.id] = destination(item)
        chosen[item.id] = Placement(item.id, planned[item.id], reason(item), confidence(item))
    for item_id, placement in list(chosen.items()):
        if state.used_tokens(planned) <= state.budget:
            break
        if placement.place == Place.ARCHIVE:
            planned[item_id] = Place.STORE
            note = placement.reason + " (archive index is full, so STORE instead)"
            chosen[item_id] = Placement(item_id, Place.STORE, note, placement.confidence)
    if state.used_tokens(planned) > state.budget:
        raise BudgetError("evicting every candidate still does not fit the budget")
    placements = list(chosen.values())
    if state.incoming and planned[state.incoming.id] == Place.CONTEXT:
        placements.append(Placement(state.incoming.id, Place.CONTEXT, "new item fits in context"))
    return placements


def registry() -> dict[str, type[Controller]]:
    """Every controller by its config name. Add new controllers here (one line each)."""
    from memctl.controllers import jev, rules

    return {
        "full_context": rules.FullContext,
        "keep_newest": rules.KeepNewest,
        "lru": rules.LeastRecentlyUsed,
        "random": rules.RandomEviction,
        "file_everything": rules.FileEverything,
        "jev": jev.JevController,
    }


def build_controller(config: dict, seed: int = 0) -> Controller:
    """Create the controller named in `config["name"]`."""
    name = config.get("name")
    known = registry()
    if name not in known:
        raise KeyError(f"unknown controller '{name}'. Known controllers: {sorted(known)}")
    return known[name](config, seed)
