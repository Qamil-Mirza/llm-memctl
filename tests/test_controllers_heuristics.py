"""Heuristic controllers: which items each removes under pressure, and how."""

import pytest

from memctl.controllers import build_controller
from memctl.embed import HashingEmbedder
from memctl.memory.actions import ActionStatus, Operation
from memctl.memory.engine import MemoryEngine
from memctl.memory.items import SourceType, Tier
from tests.helpers import DELETE_ONLY, decide, episode_info, make_state, task_for

FOUR = ["a1 a2 a3", "b1 b2 b3", "c1 c2 c3", "d1 d2 d3"]  # 3 tokens each
FACT = "As of step 1, the access code of vault-317 is K93Q."
FILLER = "The system ran a routine check and nothing changed."


def removed(actions, operation=Operation.EVICT):
    return [i for action in actions if action.operation is operation for i in action.target_ids]


def test_controllers_do_nothing_while_memory_fits():
    for name in ("fifo", "lru", "lfu", "random", "age_decay", "salience", "no_controller"):
        assert decide(build_controller({"name": name}), make_state(FOUR, budget=12), DELETE_ONLY) == []


def test_fifo_evicts_the_oldest_until_the_state_fits():
    actions = decide(build_controller({"name": "fifo"}), make_state(FOUR, budget=7), DELETE_ONLY)
    assert removed(actions) == ["o0", "o1"]


def test_lru_evicts_the_item_unused_for_longest():
    state = make_state(FOUR, budget=9)
    state.record_access(["o0"])  # at step 4
    assert removed(decide(build_controller({"name": "lru"}), state, DELETE_ONLY)) == ["o1"]


def test_lfu_evicts_the_least_used_item():
    state = make_state(FOUR, budget=9)
    state.record_access(["o0", "o1", "o3"])
    assert removed(decide(build_controller({"name": "lfu"}), state, DELETE_ONLY)) == ["o2"]


def test_age_decay_prefers_recent_and_reused_items():
    state = make_state(FOUR, budget=6)
    for _ in range(5):
        state.record_access(["o0"])
    assert removed(decide(build_controller({"name": "age_decay", "tau": 2.0}), state, DELETE_ONLY)) == ["o1", "o2"]


def test_random_is_reproducible_and_depends_on_the_seed():
    def run(seed):
        return removed(decide(build_controller({"name": "random"}, seed), make_state(FOUR * 5, budget=20), DELETE_ONLY))

    assert run(1) == run(1)
    assert run(1) != run(2)


def test_no_controller_never_acts_and_the_fallback_makes_it_fifo():
    state = make_state(FOUR, budget=7)
    assert decide(build_controller({"name": "no_controller"}), state, DELETE_ONLY) == []
    MemoryEngine(DELETE_ONLY).enforce_budget(state)
    assert [item.id for item in state.active()] == ["o2", "o3"]


def test_similarity_keeps_what_resembles_the_current_observation():
    embedder = HashingEmbedder()
    contents = ["vault codes and keys", "weather was mild today", "lunch menu for friday", "which vault codes changed"]
    state = make_state(contents, budget=13, embedder=embedder)
    actions = decide(build_controller({"name": "similarity"}), state, DELETE_ONLY, embedder=embedder)
    assert removed(actions) and "o0" not in removed(actions) and "o3" not in removed(actions)


def test_salience_removes_unspecific_tool_output_before_a_user_fact():
    state = make_state([FACT, FILLER, FILLER], budget=25, sources=[SourceType.USER, SourceType.TOOL_OUTPUT, SourceType.USER])
    assert removed(decide(build_controller({"name": "salience"}), state, DELETE_ONLY)) == ["o1"]


def test_removal_can_be_archiving_instead_of_deleting():
    state = make_state(FOUR, budget=7)
    actions = decide(build_controller({"name": "fifo", "removal": ["MOVE_TO_ARCHIVE"]}), state)
    assert removed(actions, Operation.MOVE_TO_ARCHIVE) == ["o0", "o1"] and removed(actions) == []


def test_a_removal_operation_the_experiment_does_not_allow_is_a_configuration_error():
    controller = build_controller({"name": "fifo", "removal": ["MOVE_TO_ARCHIVE"]})
    with pytest.raises(ValueError, match="MOVE_TO_ARCHIVE"):
        controller.reset(episode_info(make_state(FOUR), DELETE_ONLY))


def test_retrieval_brings_back_the_matching_archived_item_when_a_query_arrives():
    contents = [FACT, "As of step 2, the serial of node-12 is B77Z.", FILLER]
    state = make_state(contents, budget=30)
    engine = MemoryEngine()
    controller = build_controller({"name": "lru", "removal": ["MOVE_TO_ARCHIVE"], "retrieve": {"top_k": 1}})
    controller.reset(episode_info(state))
    for item_id in ("o0", "o1"):
        state.move(item_id, Tier.ARCHIVE, "MOVE_TO_ARCHIVE", "controller")
    state.step = 4
    state.ingest("q", "Question: what is the access code of vault-317?", SourceType.USER)
    actions = controller.decide(state.view(), task_for(state, "q", query=True))
    assert removed(actions, Operation.RETRIEVE_FROM_ARCHIVE) == ["o0"]
    retrieve = next(a for a in actions if a.operation is Operation.RETRIEVE_FROM_ARCHIVE)
    assert retrieve.parameters["method"] == "lexical" and "vault-317" in retrieve.parameters["query"]
    results = engine.apply(state, actions)
    assert all(r.status is ActionStatus.APPLIED for r in results)
    assert state.get("o0").tier is Tier.ACTIVE and state.active_tokens <= state.budget


def test_retrieval_makes_room_without_removing_what_it_just_retrieved_or_the_query():
    state = make_state([FACT, FILLER, FILLER], budget=32)
    controller = build_controller({"name": "fifo", "removal": ["MOVE_TO_ARCHIVE"], "retrieve": {"top_k": 1}})
    controller.reset(episode_info(state))
    state.move("o0", Tier.ARCHIVE, "MOVE_TO_ARCHIVE", "controller")
    state.step = 4
    state.ingest("q", "Question: what is the access code of vault-317?", SourceType.USER)
    actions = controller.decide(state.view(), task_for(state, "q", query=True))
    MemoryEngine().apply(state, actions)
    active = [item.id for item in state.active()]
    assert "o0" in active and "q" in active and state.active_tokens <= state.budget


def test_no_retrieval_happens_for_observations_that_need_no_response():
    state = make_state([FACT, FILLER], budget=100)
    controller = build_controller({"name": "fifo", "removal": ["MOVE_TO_ARCHIVE"], "retrieve": {"top_k": 1}})
    controller.reset(episode_info(state))
    state.move("o0", Tier.ARCHIVE, "MOVE_TO_ARCHIVE", "controller")
    assert controller.decide(state.view(), task_for(state, "o1", query=False)) == []


def test_compaction_is_tried_on_long_items_before_anything_is_evicted():
    long_item = " ".join([FILLER] * 4 + [FACT] + [FILLER] * 4)
    state = make_state([long_item, FACT.replace("317", "318")], budget=65)
    controller = build_controller({"name": "fifo", "removal": ["COMPACT", "EVICT"], "min_compact_tokens": 30})
    actions = decide(controller, state)
    assert removed(actions, Operation.COMPACT) == ["o0"] and removed(actions) == []
    MemoryEngine().apply(state, actions)
    assert state.active_tokens <= 65 and any("K93Q" in item.content for item in state.active())


def test_archive_everything_keeps_only_the_newest_items_in_active():
    state = make_state(FOUR, budget=100)
    actions = decide(build_controller({"name": "archive_everything", "keep_last": 1}), state)
    assert removed(actions, Operation.MOVE_TO_ARCHIVE) == ["o0", "o1", "o2"]


def test_consolidation_merges_items_that_restate_the_same_fact():
    state = make_state([FACT, FILLER, f"{FILLER} {FACT} {FILLER}"], budget=100)
    controller = build_controller({"name": "fifo", "consolidate": {}})
    actions = decide(controller, state)
    assert [a.target_ids for a in actions if a.operation is Operation.CONSOLIDATE] == [("o0", "o2")]


def test_every_decision_reports_loggable_scores():
    state = make_state(FOUR, budget=7)
    controller = build_controller({"name": "fifo"})
    decide(controller, state, DELETE_ONLY)
    info = controller.decision_info()
    assert set(info["scores"]) == {"o0", "o1", "o2", "o3"}


def test_consolidation_can_also_shorten_the_merged_item():
    state = make_state([FACT, FILLER, f"{FILLER} {FACT} {FILLER}"], budget=100)
    actions = decide(build_controller({"name": "fifo", "consolidate": {"ratio": 0.4}}), state)
    (merge,) = [a for a in actions if a.operation is Operation.CONSOLIDATE]
    assert merge.parameters["max_tokens"] == 20  # 40% of 15 + 35 tokens
    MemoryEngine().apply(state, actions)
    merged = state.active()[-1]
    assert "K93Q" in merged.content and merged.token_count <= 20


def test_a_token_target_makes_a_rule_evict_below_the_budget():
    actions = decide(build_controller({"name": "fifo", "target_tokens": 6}), make_state(FOUR, budget=12), DELETE_ONLY)
    assert removed(actions) == ["o0", "o1"]


def test_a_fitted_retrieval_takes_only_what_fits_in_the_room():
    state = make_state([FACT, "As of step 2, the access code of vault-318 is X11Y.", FILLER], budget=25)
    controller = build_controller(
        {"name": "fifo", "removal": ["MOVE_TO_ARCHIVE"], "retrieve": {"top_k": 2, "fit": True}})
    controller.reset(episode_info(state))
    for item_id in ("o0", "o1"):
        state.move(item_id, Tier.ARCHIVE, "MOVE_TO_ARCHIVE", "controller")
    state.step = 4
    state.ingest("q", "access code vault-317?", SourceType.USER)
    actions = controller.decide(state.view(), task_for(state, "q", query=True))
    assert removed(actions, Operation.RETRIEVE_FROM_ARCHIVE) == ["o0"]  # the second hit would not fit beside it
    MemoryEngine().apply(state, actions)
    assert state.active_tokens <= state.budget


def test_retrieval_controls_can_rerank_by_length_and_keep_one_speaker():
    state = make_state([])
    for number, (speaker, text) in enumerate([
        ("assistant", "the vault code K93Q is in the long assistant answer about vault codes " * 3),
        ("user", "my vault code is K93Q"),
        ("user", "vault code K93Q noted for the vault"),
    ]):
        state.ingest(f"t{number}", text, SourceType.USER, metadata={"speaker": speaker})
        state.move(f"t{number}", Tier.ARCHIVE, "MOVE_TO_ARCHIVE", "controller")
    state.step = 9
    state.ingest("q", "what is my vault code?", SourceType.USER)
    for retrieve, expected in (({"top_k": 1, "candidates": 3, "rerank": "shortest"}, ["t1"]),
                               ({"top_k": 2, "candidates": 3, "speaker": "user"}, None)):
        controller = build_controller({"name": "archive_everything", "keep_last": 0, "retrieve": retrieve})
        controller.reset(episode_info(state))
        got = removed(controller.decide(state.view(), task_for(state, "q", query=True)), Operation.RETRIEVE_FROM_ARCHIVE)
        if expected:
            assert got == expected
        else:
            assert set(got) <= {"t1", "t2"} and got
