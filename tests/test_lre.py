"""The LRE baseline (§27): its label rule, its features, the JSON model against sklearn, and both controller modes."""

import math

import pytest

pytest.importorskip("sklearn")

from memctl.controllers import build_controller
from memctl.lre import LREModel, answer_overlap_labels, answer_recall, normalize, traj_features
from memctl.memory.actions import Operation
from memctl.memory.items import SourceType
from tests.helpers import decide, episode_info, make_state, task_for

ARCHIVE_OPS = (Operation.KEEP, Operation.MOVE_TO_ARCHIVE, Operation.RETRIEVE_FROM_ARCHIVE, Operation.NO_OP)


def test_normalize_drops_articles_and_punctuation_and_keeps_other_stop_words():
    assert normalize("The cat's in a Box, isn't it?") == "cats in box isnt it"


def test_label_is_answer_token_recall_at_least_0_4():
    assert answer_recall("blue Honda Civic", "I drive a Honda") == pytest.approx(1 / 3)
    assert answer_overlap_labels(["I drive a Honda", "my Honda is blue", "nothing"], ["blue Honda Civic"]) == [0, 1, 0]
    assert answer_overlap_labels(["anything"], [""]) == [0]  # an empty answer labels nothing
    assert answer_overlap_labels(["it was 42"], [42]) == [1]  # numeric answers are compared as text


def test_trajectory_features_are_lre_s_six():
    f = traj_features("Where is My Car? 12", 1, 4)
    assert f == [0.25, 1 / 3, math.log1p(5), 2.0, 1.0, 3.0]


def _toy_model(tmp_path):
    texts = ["my dog Rex is brown", "the weather is nice", "Rex likes the park", "nothing to report",
             "I bought a red car", "cars are fast", "hello there", "dog food brand Acme"] * 5
    labels = [1, 0, 1, 0, 1, 0, 0, 1] * 5
    trajs = [traj_features(t, i % 8, 8) for i, t in enumerate(texts)]
    model = LREModel.fit(texts, trajs, labels, {"fold": "toy"})
    path = tmp_path / "lre.json"
    model.save(path)
    return model, path, texts, trajs


def test_json_model_scores_equal_sklearn(tmp_path):
    import numpy as np
    from scipy.sparse import csr_matrix, hstack

    model, path, texts, trajs = _toy_model(tmp_path)
    vectorizer, scaler, clf = model._sklearn
    x = hstack([vectorizer.transform(texts), csr_matrix(scaler.transform(np.asarray(trajs)))]).tocsr()
    expected = clf.decision_function(x)
    loaded = LREModel.load(path)
    got = [loaded.logit(t, i % 8, 8) for i, t in enumerate(texts)]
    assert np.allclose(got, expected, atol=1e-9)
    assert loaded.n_parameters == len(loaded.vocabulary) + 7


def test_unknown_mode_and_missing_model_are_refused(tmp_path):
    _, path, _, _ = _toy_model(tmp_path)
    with pytest.raises(ValueError):
        build_controller({"name": "lre"})
    with pytest.raises(ValueError):
        build_controller({"name": "lre", "model": str(path), "mode": "bogus"})


def test_slot_archives_on_arrival_and_retrieves_the_k_best_of_the_bm25_shortlist(tmp_path):
    model, path, _, _ = _toy_model(tmp_path)
    contents = ["Rex the dog is brown", "the dog weather report", "dog park nothing", "dog food brand Acme",
                "hello dog"]
    state = make_state(contents, budget=1000)
    controller = build_controller({"name": "lre", "model": str(path), "mode": "slot", "select_k": 2,
                                   "retrieve_candidates": 4})
    controller.reset(episode_info(state, ARCHIVE_OPS))
    # Not a question: everything but the current input is archived; nothing is retrieved.
    actions = controller.decide(state.view(), task_for(state))
    assert [a.operation for a in actions] == [Operation.MOVE_TO_ARCHIVE]
    assert set(actions[0].target_ids) == {"o0", "o1", "o2", "o3"}
    for action in actions:
        state_apply(state, action)
    ask(state, "o5", "Where is the dog?")
    # The question: one retrieval of the 2 best by LRE's logit among the 4 BM25 hits.
    actions = controller.decide(state.view(), task_for(state, query=True))
    retrieve = [a for a in actions if a.operation is Operation.RETRIEVE_FROM_ARCHIVE]
    info = controller.decision_info()
    assert len(retrieve) == 1 and info["candidates"] == 4 and len(info["selected_ids"]) == 2
    ranked = sorted(info["logits"], key=lambda i: -info["logits"][i])
    assert list(retrieve[0].target_ids) == ranked[:2]
    # The scorer is query-blind: the question is never scored, and n counts the 5 turns only.
    assert "o5" not in info["logits"] and len(controller.scores.order) == 5


def test_slot_shows_a_short_shortlist_whole(tmp_path):
    _, path, _, _ = _toy_model(tmp_path)
    state = make_state(["Rex dog", "weather"], budget=1000)
    controller = build_controller({"name": "lre", "model": str(path), "mode": "slot", "select_k": 8})
    controller.reset(episode_info(state, ARCHIVE_OPS))
    state_apply(state, controller.decide(state.view(), task_for(state))[0])
    ask(state, "o2", "Where is Rex?")
    actions = controller.decide(state.view(), task_for(state, query=True))
    assert [list(a.target_ids) for a in actions if a.operation is Operation.RETRIEVE_FROM_ARCHIVE] == [["o0"]]


def test_native_archives_the_lowest_scoring_items_not_the_oldest(tmp_path):
    model, path, _, _ = _toy_model(tmp_path)
    contents = ["my dog Rex is brown", "the weather is nice", "nothing to report", "Rex likes the park", "hello there"]
    state = make_state(contents, budget=1000)
    config = {"name": "lre", "model": str(path), "mode": "native", "removal": ["MOVE_TO_ARCHIVE"],
              "target_tokens": state.active_tokens - 1}
    actions = decide(build_controller(config), state, ARCHIVE_OPS)
    moved = [i for a in actions if a.operation is Operation.MOVE_TO_ARCHIVE for i in a.target_ids]
    logits = {f"o{i}": model.logit(t, i, 5) for i, t in enumerate(contents)}
    assert moved == [min(logits, key=logits.get)]
    fifo = decide(build_controller({**config, "name": "fifo", "model": None, "mode": None}), state, ARCHIVE_OPS)
    assert [i for a in fifo for i in a.target_ids] == ["o0"]  # same config, recency: the oldest goes


def test_native_retrieves_as_fifo_floor_and_never_removes_the_question(tmp_path):
    from memctl.memory.actions import MemoryAction

    _, path, _, _ = _toy_model(tmp_path)
    contents = ["Rex dog brown", "weather nice", "Rex park", "hello there", "Where is Rex the dog?"]
    retrieve = {"top_k": 5, "fit": True, "method": "lexical"}
    sizes = make_state(contents).active()
    config = {"removal": ["MOVE_TO_ARCHIVE"], "retrieve": retrieve,
              "target_tokens": sizes[0].token_count + sizes[2].token_count + sizes[4].token_count}
    chosen = {}
    for name in ("fifo", "lre"):
        state = make_state(contents, budget=1000)
        state_apply(state, MemoryAction(Operation.MOVE_TO_ARCHIVE, ("o0", "o2")))
        extra = {"model": str(path), "mode": "native"} if name == "lre" else {}
        controller = build_controller({"name": name, **config, **extra})
        controller.reset(episode_info(state, ARCHIVE_OPS))
        actions = controller.decide(state.view(), task_for(state, query=True))
        chosen[name] = [list(a.target_ids) for a in actions if a.operation is Operation.RETRIEVE_FROM_ARCHIVE]
        moved = [i for a in actions if a.operation is Operation.MOVE_TO_ARCHIVE for i in a.target_ids]
        assert set(moved) == {"o1", "o3"}  # the question stays; retrieved items are protected
    assert chosen["lre"] == chosen["fifo"] == [["o0", "o2"]]  # the same floor: BM25 top hits that fit


def state_apply(state, action):
    from memctl.memory.engine import MemoryEngine

    MemoryEngine(ARCHIVE_OPS).apply(state, [action])


def ask(state, item_id, question):
    state.step += 1
    state.ingest(item_id, question, SourceType.USER)
