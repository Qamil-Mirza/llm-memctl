"""Rule-based controllers: simple baselines that need no model."""

from __future__ import annotations

import random

from memctl.controllers.base import Controller, Placement, evict_until_fits
from memctl.features import ItemFeatures
from memctl.items import Place
from memctl.memory import MemoryState


class FullContext(Controller):
    """Never evicts. The runner gives it an unlimited budget (upper reference)."""

    name = "full_context"
    ignores_budget = True

    def decide(self, state: MemoryState, features: dict[str, ItemFeatures], budget: int) -> list[Placement]:
        if state.incoming is None:
            return []
        return [Placement(state.incoming.id, Place.CONTEXT, "full_context keeps everything")]


class KeepNewest(Controller):
    """FIFO: the oldest items are evicted first and go to STORE."""

    name = "keep_newest"

    def decide(self, state: MemoryState, features: dict[str, ItemFeatures], budget: int) -> list[Placement]:
        oldest_first = sorted(state.candidates(), key=lambda item: item.arrival_step)
        return evict_until_fits(
            state,
            oldest_first,
            destination=lambda item: Place.STORE,
            reason=lambda item: f"oldest item in context (age {features[item.id].age_steps} steps)",
        )


class LeastRecentlyUsed(Controller):
    """The item that has gone unused for longest is evicted first and goes to STORE."""

    name = "lru"

    def decide(self, state: MemoryState, features: dict[str, ItemFeatures], budget: int) -> list[Placement]:
        stalest_first = sorted(
            state.candidates(),
            key=lambda item: (-features[item.id].steps_since_last_use, item.arrival_step),
        )
        return evict_until_fits(
            state,
            stalest_first,
            destination=lambda item: Place.STORE,
            reason=lambda item: f"not used for {features[item.id].steps_since_last_use} steps",
        )


class RandomEviction(Controller):
    """Evicts items in a random (seeded) order. Destination is picked from `destinations`."""

    name = "random"

    def __init__(self, config: dict, seed: int = 0) -> None:
        super().__init__(config, seed)
        self.rng = random.Random(seed)
        self.destinations = [Place(p) for p in config.get("destinations", ["STORE"])]

    def decide(self, state: MemoryState, features: dict[str, ItemFeatures], budget: int) -> list[Placement]:
        order = state.candidates()
        self.rng.shuffle(order)
        picks = {item.id: self.rng.choice(self.destinations) for item in order}
        return evict_until_fits(
            state,
            order,
            destination=lambda item: picks[item.id],
            reason=lambda item: f"picked at random (seed {self.seed})",
        )


class FileEverything(Controller):
    """RAG-style: only the last few items stay in context, the rest go to STORE."""

    name = "file_everything"
    runs_on_every_arrival = True

    def __init__(self, config: dict, seed: int = 0) -> None:
        super().__init__(config, seed)
        self.keep_last = int(config.get("keep_last", 4))

    def decide(self, state: MemoryState, features: dict[str, ItemFeatures], budget: int) -> list[Placement]:
        oldest_first = sorted(state.candidates(), key=lambda item: item.arrival_step)
        number_old = max(0, len(oldest_first) - self.keep_last)
        planned = dict(state.place)
        if state.incoming:
            planned[state.incoming.id] = Place.CONTEXT
        placements = []
        for position, item in enumerate(oldest_first):
            is_old = position < number_old
            if not is_old and state.used_tokens(planned) <= budget:
                break  # the remaining recent items fit
            planned[item.id] = Place.STORE
            why = f"not among the last {self.keep_last} items" if is_old else "recent items exceed the budget"
            placements.append(Placement(item.id, Place.STORE, f"filed: {why}"))
        if state.incoming and planned[state.incoming.id] == Place.CONTEXT:
            placements.append(Placement(state.incoming.id, Place.CONTEXT, "one of the newest items"))
        return placements
