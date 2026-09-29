"""The oracle: optimal on hand-made examples, exact plan checked against brute force."""

import itertools
import random

import pytest

from memctl.benchmarks.types import Conversation, Event, Question
from memctl.controllers.base import build_controller
from memctl.controllers.oracle import NEVER, build_plan
from memctl.controllers.oracle_ilp import solve_keep_plan
from memctl.embed import HashingEmbedder
from memctl.features import compute_features
from memctl.items import Place, make_item
from memctl.memory import MemoryState


def conversation(sizes: dict[str, int], questions: list[tuple[str, ...]], ask_after: dict[int, int] | None = None) -> Conversation:
    """Items arrive in order; `questions` are asked at the end unless `ask_after` places them earlier.

    `sizes` maps item id to a number of words. `ask_after[q] = n` asks question q after n items.
    """
    ask_after = ask_after or {}
    events: list[Event] = []

    def add_questions(after_items: int | None) -> None:
        for number, evidence in enumerate(questions):
            if ask_after.get(number) == after_items:
                events.append(Event("question", question=Question(f"q{number}", "?", "gold", "single-hop", evidence)))

    for count, (item_id, words) in enumerate(sizes.items(), start=1):
        text = " ".join(f"{item_id}w{k}" for k in range(words))
        events.append(Event("item", item=make_item(item_id, text, len(events) + 1, source={"speaker": "A"})))
        add_questions(count)
    add_questions(None)
    return Conversation("tiny", events)


def covered_questions(conv: Conversation, controller_config: dict, budget: int) -> int:
    """Run a controller (no model needed) and count questions with ALL evidence in CONTEXT when asked."""
    controller = build_controller(controller_config)
    state = MemoryState(budget=budget, embedder=HashingEmbedder())
    if controller.uses_evidence_labels:
        controller.receive_plan(build_plan(conv, budget, allow_drop=True))
    covered = 0
    for step, event in enumerate(conv.events, start=1):
        state.step = step
        if event.kind == "item" and not state.offer(event.item):
            features = compute_features(state, state.candidates(), event.item.text)
            state.apply(controller.decide(state, features, budget))
        if event.kind == "question":
            covered += all(state.place.get(i) == Place.CONTEXT for i in event.question.evidence_ids)
    return covered


def tokens(words: int) -> int:
    return make_item("x", " ".join(["w"] * words), 1, source={"speaker": "A"}).tokens


def test_oracle_is_optimal_on_a_tiny_hand_made_example():
    """Room for 2 of 4 equal items. `a` and `d` are needed; `b` and `c` never are."""
    conv = conversation({"a": 5, "b": 5, "c": 5, "d": 5}, questions=[("a",), ("d",)])
    budget = 2 * tokens(5)
    _, best_possible = solve_keep_plan(conv, budget)
    assert best_possible == 2
    assert covered_questions(conv, {"name": "oracle", "method": "belady"}, budget) == 2
    assert covered_questions(conv, {"name": "oracle", "method": "ilp"}, budget) == 2
    assert covered_questions(conv, {"name": "keep_newest"}, budget) == 1  # it evicted `a`


def test_oracle_drops_unneeded_items_and_never_drops_needed_ones():
    conv = conversation({"a": 5, "b": 5, "c": 5, "d": 5, "e": 5}, questions=[("a",), ("b",), ("e",)])
    budget = 2 * tokens(5)
    controller = build_controller({"name": "oracle"})
    controller.receive_plan(build_plan(conv, budget, allow_drop=True))
    state = MemoryState(budget=budget, embedder=HashingEmbedder())
    for step, event in enumerate(conv.events, start=1):
        state.step = step
        if event.kind == "item" and not state.offer(event.item):
            state.apply(controller.decide(state, compute_features(state, state.candidates(), "q"), budget))
    assert {i for i, p in state.place.items() if p == Place.DROPPED} <= {"c", "d"}
    assert all(state.place[i] in (Place.CONTEXT, Place.STORE) for i in ("a", "b", "e"))


def test_exact_plan_beats_belady_when_item_sizes_differ():
    """One big item answers 1 question; two small items answer 1 each; only big OR both small fit."""
    conv = conversation({"big": 22, "s1": 8, "s2": 8, "pad": 8}, questions=[("big",), ("s1",), ("s2",)])
    budget = tokens(22)
    assert 2 * tokens(8) <= budget < tokens(22) + tokens(8)
    assert covered_questions(conv, {"name": "oracle", "method": "ilp"}, budget) == 2
    assert covered_questions(conv, {"name": "oracle", "method": "belady"}, budget) == 1  # it kept `big`
    _, best_possible = solve_keep_plan(conv, budget)
    assert best_possible == 2


def test_belady_prefers_the_item_needed_sooner():
    """`a` is needed right away and `b` only at the end; with room for one, keep `a` first."""
    conv = conversation({"a": 5, "b": 5, "c": 5}, questions=[("a",), ("b",)], ask_after={0: 3})
    plan = build_plan(conv, tokens(5), allow_drop=True)
    assert plan.next_need("a", 1) < plan.next_need("b", 1)
    assert plan.next_need("c", 1) == NEVER
    assert plan.place_for("c", 1) == Place.DROPPED


def brute_force_best(conv: Conversation, budget: int) -> int:
    """Try every set of evidence items that fits (questions are all at the end)."""
    items = {i.id: i for i in conv.items()}
    evidence = sorted({i for q in conv.questions() for i in q.evidence_ids})
    best = 0
    for size in range(len(evidence) + 1):
        for kept in itertools.combinations(evidence, size):
            if sum(items[i].tokens for i in kept) <= budget:
                best = max(best, sum(all(i in kept for i in q.evidence_ids) for q in conv.questions()))
    return best


@pytest.mark.parametrize("seed", range(8))
def test_exact_plan_matches_brute_force_on_random_small_conversations(seed):
    rng = random.Random(seed)
    sizes = {f"i{n}": rng.randint(2, 25) for n in range(9)}
    questions = [tuple(rng.sample(sorted(sizes), rng.choice([1, 1, 2]))) for _ in range(6)]
    conv = conversation(sizes, questions)
    budget = conv.total_tokens() // 3
    keep_until, covered = solve_keep_plan(conv, budget)
    assert covered == brute_force_best(conv, budget)
    assert sum(i.tokens for i in conv.items() if i.id in keep_until) <= budget
    assert covered_questions(conv, {"name": "oracle", "method": "ilp"}, budget) >= covered
    assert covered >= covered_questions(conv, {"name": "keep_newest"}, budget)


def test_other_controllers_are_never_given_the_plan():
    for name in ["full_context", "keep_newest", "lru", "random", "file_everything"]:
        controller = build_controller({"name": name})
        assert controller.uses_evidence_labels is False
        assert not hasattr(controller, "receive_plan") and not hasattr(controller, "plan")


@pytest.mark.parametrize("method", ["ilp", "belady"])
def test_oracle_follows_the_same_move_rules_as_every_controller(method):
    """The oracle acts only through MemoryState.apply, so a filed item never returns to CONTEXT,
    a dropped item never moves, and the budget holds. Its exact plan obeys the same rule."""
    rng = random.Random(4)
    sizes = {f"i{n}": rng.randint(2, 25) for n in range(40)}
    questions = [tuple(rng.sample(sorted(sizes), rng.choice([1, 2]))) for _ in range(12)]
    conv = conversation(sizes, questions, ask_after={0: 10, 1: 20, 2: 30})
    budget = conv.total_tokens() // 4
    controller = build_controller({"name": "oracle", "method": method})
    controller.receive_plan(build_plan(conv, budget, allow_drop=True))
    state = MemoryState(budget=budget, embedder=HashingEmbedder())
    left_context: set[str] = set()
    for step, event in enumerate(conv.events, start=1):
        state.step = step
        if event.kind == "item" and not state.offer(event.item):
            state.apply(controller.decide(state, compute_features(state, state.candidates(), "q"), budget))
        in_context = {i.id for i in state.in_place(Place.CONTEXT)}
        assert not (in_context & left_context), "an item came back into context"
        left_context |= {i for i in state.place if i not in in_context}
        assert state.used_tokens() <= budget

    # The exact plan keeps each item for one unbroken stretch from its arrival: it never plans a return.
    plan = controller.plan
    for item_id, last_step in plan.keep_until.items():
        assert last_step >= next(i.arrival_step for i in conv.items() if i.id == item_id)


def test_no_controller_may_move_a_filed_item_back_into_context(state=None):
    from memctl.controllers.base import Placement
    from memctl.memory import PlacementError

    state = MemoryState(budget=100, embedder=HashingEmbedder())
    state.offer(make_item("a", "some words here", 1, source={"speaker": "A"}))
    state.apply([Placement("a", Place.STORE, "filed")])
    with pytest.raises(PlacementError):
        state.apply([Placement("a", Place.CONTEXT, "the oracle is not exempt")])
