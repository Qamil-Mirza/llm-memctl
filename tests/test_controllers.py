"""The registry and the behaviour of each rule controller."""

import pytest

from memctl.controllers.base import Controller, build_controller, registry
from memctl.controllers.jev import FakeJevClient, JevController, JevUnavailableError, choice_confidence
from memctl.embed import HashingEmbedder
from memctl.features import compute_features
from memctl.items import Place
from memctl.memory import MemoryState
from tests.conftest import item


def fill(state: MemoryState, controller: Controller, count: int) -> None:
    """Add `count` equal-sized items, calling the controller whenever it is needed."""
    for n in range(1, count + 1):
        state.step = n
        fits = state.offer(item(n))
        if not fits or controller.runs_on_every_arrival:
            features = compute_features(state, state.candidates(), item(n).text)
            state.apply(controller.decide(state, features, state.budget))


def ids(state: MemoryState, place: Place) -> list[str]:
    return [i.id for i in state.in_place(place)]


def test_registry_builds_every_controller_by_name():
    for name, controller_class in registry().items():
        controller = build_controller({"name": name, "allow_fake_jev": True}, seed=1)
        assert isinstance(controller, controller_class)
        assert controller.name == name


def test_registry_rejects_unknown_names():
    with pytest.raises(KeyError, match="unknown controller 'nope'"):
        build_controller({"name": "nope"})


def test_keep_newest_evicts_the_oldest_to_store(state):
    fill(state, build_controller({"name": "keep_newest"}), 6)
    assert ids(state, Place.STORE) == ["i1", "i2", "i3"]
    assert ids(state, Place.CONTEXT) == ["i4", "i5", "i6"]


def test_lru_keeps_an_old_item_that_was_used_recently(state):
    controller = build_controller({"name": "lru"})
    fill(state, controller, 3)
    state.step = 4
    state.touch("i1")  # the oldest item is used again
    state.offer(item(4))
    features = compute_features(state, state.candidates(), "unrelated")
    state.apply(controller.decide(state, features, state.budget))
    assert ids(state, Place.STORE) == ["i2"]
    assert "i1" in ids(state, Place.CONTEXT)


def test_random_is_repeatable_for_a_seed():
    def final_places(seed: int) -> dict:
        state = MemoryState(budget=60, embedder=HashingEmbedder())
        fill(state, build_controller({"name": "random"}, seed=seed), 30)
        return dict(state.place)

    assert final_places(7) == final_places(7)
    assert final_places(7) != final_places(8)


def test_file_everything_keeps_only_the_last_few(state):
    state.budget = 1000
    fill(state, build_controller({"name": "file_everything", "keep_last": 2}), 10)
    assert ids(state, Place.CONTEXT) == ["i9", "i10"]
    assert len(ids(state, Place.STORE)) == 8


def test_full_context_never_evicts():
    state = MemoryState(budget=10**9, embedder=HashingEmbedder())
    fill(state, build_controller({"name": "full_context"}), 50)
    assert len(ids(state, Place.CONTEXT)) == 50


def test_jev_fails_loudly_without_a_key_or_the_fake_flag(monkeypatch):
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    with pytest.raises(JevUnavailableError, match="--allow-fake-jev"):
        build_controller({"name": "jev"})


def test_fake_jev_is_labelled_fake_everywhere(state):
    controller = JevController({"name": "jev"}, client=FakeJevClient())
    fill(state, controller, 8)
    assert controller.display_name == "jev-FAKE"
    assert controller.call_log and all(call["fake"] for call in controller.call_log)
    assert all(call["response"]["model"] == "FAKE-jev" for call in controller.call_log)


def test_jev_request_is_a_typed_choice_among_the_places(state):
    controller = JevController({"name": "jev"}, client=FakeJevClient())
    fill(state, controller, 8)
    request = controller.call_log[0]["request"]
    question = next(iter(request["questions"].values()))
    assert question["type"] == "choice"
    assert set(question["criteria"]) == {"CONTEXT", "STORE", "ARCHIVE", "DROPPED"}
    assert "`memory_items.item_0`" in question["instructions"]
    assert "key" not in str(controller.call_log).lower().replace("api_key_env", "")


def test_jev_does_not_offer_dropped_when_drop_is_off():
    state = MemoryState(budget=60, embedder=HashingEmbedder(), allow_drop=False)
    controller = JevController({"name": "jev"}, client=FakeJevClient())
    fill(state, controller, 8)
    question = next(iter(controller.call_log[0]["request"]["questions"].values()))
    assert "DROPPED" not in question["criteria"]
    assert ids(state, Place.DROPPED) == []


def test_choice_confidence_matches_the_documented_formula():
    assert choice_confidence({"a": 1.0, "b": 0.0, "c": 0.0}) == 1.0
    assert choice_confidence({"a": 0.5, "b": 0.5}) == 0.0
    assert choice_confidence({"billing": 0.88, "technical": 0.12, "sales": 0.0}) == pytest.approx(0.82)
