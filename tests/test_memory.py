"""MemoryState rules: the budget is never exceeded and dropped items are gone for good."""

import random

import pytest

from memctl.controllers.base import Placement, build_controller, registry
from memctl.controllers.jev import FakeJevClient, JevController
from memctl.controllers.oracle import OraclePlan
from memctl.embed import HashingEmbedder
from memctl.features import compute_features
from memctl.items import Place
from memctl.memory import BudgetError, MemoryState, PlacementError, RecallError
from tests.conftest import item


def test_items_go_to_context_while_there_is_room(state):
    assert state.offer(item(1)) is True
    assert state.place["i1"] == Place.CONTEXT
    assert state.used_tokens() == item(1).tokens


def test_item_that_does_not_fit_waits_for_the_controller(state):
    for n in range(1, 5):
        fits = state.offer(item(n))
        if not fits:
            break
    assert state.incoming is not None
    assert state.incoming.id not in state.place
    assert state.used_tokens() <= state.budget


def test_apply_rejects_placements_that_break_the_budget_and_changes_nothing(state):
    while state.offer(item(len(state.items) + 1)):
        pass
    before = dict(state.place)
    with pytest.raises(BudgetError):
        state.apply([Placement(state.incoming.id, Place.CONTEXT, "no room was made")])
    assert state.place == before
    assert state.incoming is not None


def test_incoming_item_must_be_placed(state):
    while state.offer(item(len(state.items) + 1)):
        pass
    with pytest.raises(PlacementError):
        state.apply([])


def test_dropped_items_cannot_be_recalled_or_moved(state):
    state.offer(item(1))
    state.apply([Placement("i1", Place.DROPPED, "test")])
    with pytest.raises(RecallError):
        state.recall("i1")
    with pytest.raises(PlacementError):
        state.apply([Placement("i1", Place.STORE, "bring it back")])
    assert state.search_store(item(1).text) == []


def test_drop_can_be_switched_off():
    state = MemoryState(budget=60, embedder=HashingEmbedder(), allow_drop=False)
    state.offer(item(1))
    with pytest.raises(PlacementError):
        state.apply([Placement("i1", Place.DROPPED, "test")])
    assert state.place["i1"] == Place.CONTEXT


def test_recall_works_only_for_archived_items_and_costs_tokens(state):
    state.offer(item(1))
    state.offer(item(2))
    state.apply([Placement("i1", Place.ARCHIVE, "test"), Placement("i2", Place.STORE, "test")])
    assert state.recall("i1").id == "i1"
    assert state.recalls == 1
    assert state.recall_tokens == state.recall_cost + item(1).tokens
    with pytest.raises(RecallError):
        state.recall("i2")


def test_archive_index_lines_count_toward_the_budget(state):
    state.offer(item(1))
    state.apply([Placement("i1", Place.ARCHIVE, "test")])
    assert 0 < state.used_tokens() < item(1).tokens


@pytest.mark.parametrize("words", [1, 2, 6, 7, 8, 30])
def test_an_index_line_never_costs_more_than_the_item(state, words):
    from memctl.items import count_tokens

    assert count_tokens(item(1, words).index_line()) <= item(1, words).tokens
    if words >= 7:
        assert count_tokens(item(1, words).index_line()) < item(1, words).tokens


def test_store_search_returns_the_most_similar_items(state):
    state.store_top_k = 1
    state.offer(item(1))
    state.offer(item(2))
    state.apply([Placement("i1", Place.STORE, "t"), Placement("i2", Place.STORE, "t")])
    assert [found.id for found in state.search_store(item(2).text)] == ["i2"]


def all_controllers():
    controllers = [build_controller({"name": name}, seed=3) for name in registry() if name != "jev"]
    controllers.append(build_controller({"name": "random", "destinations": ["STORE", "ARCHIVE", "DROPPED"]}, 5))
    controllers.append(JevController({"name": "jev"}, client=FakeJevClient()))
    for controller in controllers:
        if controller.uses_evidence_labels:  # the oracle needs a plan: every 7th item is needed at the end
            controller.receive_plan(OraclePlan(needs={f"i{n}": [200] for n in range(1, 151, 7)}))
    return controllers


@pytest.mark.parametrize("controller", all_controllers(), ids=lambda c: c.display_name)
@pytest.mark.parametrize("budget", [25, 60, 200])
def test_budget_is_never_exceeded_by_any_controller(controller, budget):
    """Feed 150 items of random sizes (some bigger than the whole budget) and check every step."""
    rng = random.Random(budget)
    state = MemoryState(budget=10**9 if controller.ignores_budget else budget, embedder=HashingEmbedder())
    for n in range(1, 151):
        state.step = n
        new = item(n, words=rng.randint(1, 40))
        fits = state.offer(new)
        assert state.used_tokens() <= state.budget
        if not fits or controller.runs_on_every_arrival or n % 25 == 0:
            state.trigger = "overflow" if not fits else "session_end"
            features = compute_features(state, state.candidates(), new.text)
            state.apply(controller.decide(state, features, state.budget))
        assert state.used_tokens() <= state.budget
        assert state.incoming is None
        assert set(state.place) == set(state.items)  # every item is in exactly one place
