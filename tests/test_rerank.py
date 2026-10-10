"""§32: the rerankers in the head's slot, and the checks the review asked for (question-dependent features, the
budget fallback for top-k picks, designated-carrier labels of fallback sessions, disjoint answer shares)."""

import numpy as np
import pytest
import torch

from memctl.analysis.shares import answer_shares, is_unknown
from memctl.config import resolve
from memctl.controllers import build_controller
from memctl.controllers.rl import Decision
from memctl.embed import HashingEmbedder
from memctl.envs.longmemeval import parse_instance
from memctl.features import ITEM_FEATURES
from memctl.harness.runner import Experiment
from memctl.hindsight.collect import Hindsight, ItemRecord
from memctl.memory.actions import Operation
from memctl.memory.engine import MemoryEngine
from memctl.memory.items import SourceType
from memctl.rerank import CrossEncoderScorer, LogisticModel, rows, rrf_scores, top_k
from memctl.rl.expert import make_expert
from memctl.task import Dependency, EvidenceRequirement
from tests.helpers import episode_info, make_state, task_for

ARCHIVE_OPS = (Operation.KEEP, Operation.MOVE_TO_ARCHIVE, Operation.RETRIEVE_FROM_ARCHIVE, Operation.NO_OP)
SLOT = {"retrieval_method": "lexical", "floor_head": True, "keep_none": True}
CONTENTS = ["my dog Rex likes the dog park", "the dog and the cat", "the cat house is red cat",
            "weather report for Monday", "Rex the dog is brown", "cat food brand Acme"]


def _apply(state, actions):
    MemoryEngine(ARCHIVE_OPS).apply(state, list(actions))


def _ask(contents, question, controller_config, budget=1000, record=False):
    """Archive every turn (keep_none), then ask; returns (state, controller, actions at the question)."""
    state = make_state(contents, budget=budget, embedder=HashingEmbedder())
    controller = build_controller(controller_config)
    controller.record = record
    controller.reset(episode_info(state, ARCHIVE_OPS, HashingEmbedder()))
    _apply(state, controller.decide(state.view(), task_for(state)))
    state.step += 1
    state.ingest("q", question, SourceType.USER)
    return state, controller, controller.decide(state.view(), task_for(state, query=True))


def test_retrieval_score_and_observation_similarity_depend_on_the_question():
    """The head's features are query-aware: change the question and the same turn's features change."""
    features = {}
    for question in ("where is the dog park", "where is the cat house cat"):
        torch.manual_seed(0)
        _, controller, _ = _ask(CONTENTS, question, {"name": "rl", "retrieve_floor": 2, "retrieve_candidates": 8,
                                                      **SLOT}, record=True)
        decision = controller.recorded[-1]
        features[question] = {item_id: decision.items[j] for j, item_id in enumerate(decision.item_ids)}
    dog, cat = features.values()
    shared = sorted(set(dog) & set(cat))
    assert shared == ["o1"]  # "the dog and the cat" is a candidate for both questions
    score, similarity = ITEM_FEATURES.index("retrieval_score"), ITEM_FEATURES.index("observation_similarity")
    assert dog["o1"][score] != pytest.approx(cat["o1"][score])
    assert dog["o1"][similarity] != pytest.approx(cat["o1"][similarity])
    # The question-free features of that turn do not change.
    fixed = [ITEM_FEATURES.index(name) for name in ("log_tokens", "specific_tokens", "specific_density", "archived")]
    assert np.allclose(dog["o1"][fixed], cat["o1"][fixed])


def test_a_reranker_sees_the_heads_pool_and_features_and_retrieves_its_k_best(tmp_path):
    import json

    question = "where is the dog park cat"
    torch.manual_seed(0)
    _, head, _ = _ask(CONTENTS, question, {"name": "rl", "retrieve_floor": 2, "retrieve_candidates": 4, **SLOT},
                      record=True)
    pool = head.recorded[-1]
    assert len(pool.item_ids) == 4
    model = LogisticModel(np.zeros(30), np.ones(30), -np.eye(30)[ITEM_FEATURES.index("log_tokens")], 0.0)
    path = tmp_path / "lr.json"
    path.write_text(json.dumps(model.to_json()))
    for scorer in ({"kind": "bm25"}, {"kind": "pointwise", "model": str(path)}):
        config = {"name": "rerank", "retrieve_floor": 2, "retrieve_candidates": 4, **SLOT, "scorer": scorer}
        state, rerank, actions = _ask(CONTENTS, question, config, record=True)
        decision = rerank.recorded[-1]
        # The same 4 BM25 candidates, in the same order, with the head's own feature rows.
        assert decision.item_ids == pool.item_ids and np.array_equal(decision.items, pool.items)
        info = rerank.decision_info()
        retrieved = [i for a in actions if a.operation is Operation.RETRIEVE_FROM_ARCHIVE for i in a.target_ids]
        assert len(retrieved) == 2 and retrieved == info["selected_ids"]
        assert info["candidates"] == 4 and info["select_ms"] >= 0 and info["search_ms"] >= 0
        if scorer["kind"] == "bm25":  # the BM25 order itself
            assert retrieved == pool.item_ids[:2]
        else:  # the toy model scores minus log_tokens: the two shortest candidates (ties in BM25 order)
            sizes = [state.get(i).token_count for i in pool.item_ids]
            shortest = sorted(range(4), key=lambda j: (sizes[j], j))[:2]
            assert retrieved == [pool.item_ids[j] for j in shortest]
    # keep_none: at the question the turn before it is archived (after the search, so it is not a candidate).
    moved = [i for a in actions if a.operation is Operation.MOVE_TO_ARCHIVE for i in a.target_ids]
    assert moved == ["o5"] and "o5" not in pool.item_ids


def test_top_k_ties_keep_the_bm25_order_and_rrf_fuses_within_the_pool():
    assert top_k([1.0, 2.0, 2.0, 0.5], 3) == [1, 2, 0]
    # BM25 order 0,1,2; dense order 2,1,0. RRF: 1/61 + 1/63 for the ends against 2/62 for the middle, so the ends
    # lead, tied, and the tie keeps the BM25 order.
    fused = rrf_scores([0.1, 0.5, 0.9])
    assert fused[0] == pytest.approx(1 / 61 + 1 / 63) == pytest.approx(fused[2]) and fused[1] == pytest.approx(2 / 62)
    assert top_k(fused, 3) == [0, 2, 1]
    assert top_k(rrf_scores([0.9, 0.5, 0.1]), 3) == [0, 1, 2]  # both orders agree: unchanged


def test_the_json_logistic_model_equals_sklearn(tmp_path):
    from sklearn.linear_model import LogisticRegression

    from memctl.rerank_train import fit_lr

    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, 30)).astype(np.float32)
    x[:, 5] = 0.0  # a constant column (as source_tool_output always is) must not divide by zero
    y = (x[:, 0] + 0.5 * x[:, 3] + rng.normal(size=300) > 0).astype(int)
    model = fit_lr(x, y, 1.0, None)
    mean, scale = x.mean(0), np.where(x.std(0) > 1e-12, x.std(0), 1.0)
    reference = LogisticRegression(C=1.0, max_iter=5000).fit((x - mean) / scale, y)
    assert np.allclose(model.decision(x), reference.decision_function((x - mean) / scale), atol=1e-8)
    loaded = LogisticModel.from_json(model.to_json())
    assert np.allclose(loaded.decision(x), model.decision(x)) and loaded.parameters == 31
    assert rows(x[:4, :24], x[0, 24:]).shape == (4, 30)


def test_a_cache_only_cross_encoder_refuses_a_miss(tmp_path):
    with pytest.raises(FileNotFoundError):
        CrossEncoderScorer("any", str(tmp_path / "none.jsonl"), cache_only=True)
    cache = tmp_path / "ce.jsonl"
    scorer = CrossEncoderScorer("toy", str(cache))
    scorer._models["toy"] = type("Toy", (), {"predict": lambda self, pairs, **_: [len(t) for _, t in pairs]})()
    assert list(scorer.score("q", ["ab", "abcd"])) == [2.0, 4.0]
    replay = CrossEncoderScorer("toy", str(cache), cache_only=True)
    assert list(replay.score("q", ["abcd", "ab"])) == [4.0, 2.0]
    with pytest.raises(KeyError):
        replay.score("another question", ["ab"])


def test_floor_head_picks_that_do_not_fit_are_returned_by_enforce_budget_and_counted():
    """A top-k pick is not budget-checked by the controller (floor_head has no room check): what does not fit is
    returned to the archive by the harness's enforce_budget (never deleted), and counted as a forced removal."""
    long_turns = [f"dog park report number {n} " + "filler words " * 12 for n in range(6)]
    for name, extra in (("rl", {}), ("rerank", {"scorer": {"kind": "bm25"}})):
        torch.manual_seed(0)
        state, controller, actions = _ask(long_turns, "where is the dog park",
                                          {"name": name, "retrieve_floor": 4, "retrieve_candidates": 6, **SLOT,
                                           **extra}, budget=80)
        retrieved = [i for a in actions if a.operation is Operation.RETRIEVE_FROM_ARCHIVE for i in a.target_ids]
        assert len(retrieved) == 4
        assert sum(state.get(i).token_count for i in retrieved) + state.get("q").token_count > state.budget
        engine = MemoryEngine(ARCHIVE_OPS)
        engine.apply(state, actions)
        forced = engine.enforce_budget(state)
        assert state.active_tokens <= state.budget and forced
        assert all(r.action.operation is Operation.MOVE_TO_ARCHIVE for r in forced)
        assert {i for r in forced for i in r.action.target_ids} <= set(retrieved)
        assert state.get("q") in state.active()  # the question is never the one removed
    # In a whole episode, every forced removal is counted. At 5% they are all returned picks; at 1% some turns
    # are larger than the whole budget and are deleted on arrival (EVICT) -- a harness property every arm shares
    # (it happens before any selection; 6 of 500 `fixed8` episodes in §27 had one 7.7k-token turn evicted).
    for fraction, operations in ((0.05, {"MOVE_TO_ARCHIVE"}), (0.01, {"MOVE_TO_ARCHIVE", "EVICT"})):
        experiment = Experiment(resolve({"env": {"name": "synthetic_recall", "horizon": 120},
                                         "controller": {"name": "rerank", "retrieve_floor": 8,
                                                        "retrieve_candidates": 16, **SLOT, "scorer": {"kind": "bm25"}},
                                         "memory": {"budget": {"fraction": fraction},
                                                    "allowed_operations": [o.value for o in ARCHIVE_OPS]}}))
        episode = experiment.run_episode(seed=0, detail=False).episode
        harness = episode["action_counts"]["harness"]
        assert episode["forced_evictions"] == sum(harness.values()) > 0
        assert set(harness) == operations


def test_designated_carrier_labels_one_turn_of_a_fallback_session():
    """An answer session with no marked turn is one requirement whose carriers are all its turns. The expert's
    label is the designated carrier only (the shortest turn, the earliest among equals): the other turns of that
    session are labelled 0 although the evaluation accepts any of them; and when the designated turn is not in the
    shortlist, no candidate of that requirement is labelled at all."""
    instance = {
        "question_id": "toy", "question_type": "single-session-user", "question": "What colour is my car?",
        "answer": "red", "question_date": "2023/05/30", "answer_session_ids": ["s1"],
        "haystack_session_ids": ["s0", "s1"], "haystack_dates": ["2023/05/01", "2023/05/02"],
        "haystack_sessions": [[{"role": "user", "content": "hello there"}],
                              [{"role": "user", "content": "I bought a car last week, it is red and shiny"},
                               {"role": "assistant", "content": "Nice car!"},
                               {"role": "user", "content": "Red car"}]],
    }
    episode, fallback = parse_instance(instance)
    assert episode.questions[0].evidence_ids == ()  # no has_answer turn
    assert fallback == (("s1:0", "s1:1", "s1:2"),)
    requirement = EvidenceRequirement(fallback[0], "")
    items = {t.id: ItemRecord(t.id, n + 1, len(t.content.split()), "user") for n, t in enumerate(episode.turns)}
    hindsight = Hindsight(items, [Dependency("q", 10, (requirement,))], total_tokens=20, horizon=10)
    assert hindsight.designated(requirement) == "s1:1"  # "Nice car!" (2 tokens) is the shortest turn
    expert = make_expert(hindsight, kind="regret")

    def labels(shortlist):
        decision = Decision(step=10, items=np.zeros((len(shortlist), 24), np.float32),
                            global_features=np.zeros(6, np.float32), item_ids=list(shortlist),
                            roots=[(i,) for i in shortlist], n_active=0, savings=np.zeros((0, 4), np.int64),
                            mask=np.zeros((0, 4), bool), excess=0, retrieve_tokens=[1] * len(shortlist))
        return expert(decision)[1]

    assert labels(["s0:0", "s1:0", "s1:1", "s1:2"]) == [0, 0, 1, 0]
    assert labels(["s0:0", "s1:0", "s1:2"]) == [0, 0, 0]  # a carrier is in the pool, but no label


def test_the_three_answer_shares_are_disjoint_and_an_unknown_judged_correct_is_flagged():
    rows_ = [{"question_id": "a", "answer": "Paris", "correct": True},
             {"question_id": "b", "answer": "Lyon", "correct": False},
             {"question_id": "c", "answer": "Unknown.", "correct": False},
             {"question_id": "d", "answer": "unknown - not mentioned", "correct": True}]
    shares = answer_shares(rows_)
    assert shares["answered_correct"] + shares["answered_wrong"] + shares["unknown"] == pytest.approx(1.0)
    assert (shares["answered_correct"], shares["answered_wrong"], shares["unknown"]) == (0.25, 0.25, 0.5)
    assert shares["flagged"] == ["d"] and shares["accuracy"] == 0.5
    assert shares["p_correct_given_answered"] == 0.5
    assert is_unknown("  UNKNOWN") and not is_unknown("I don't know") and not is_unknown(None)
    clean = answer_shares(rows_[:3])
    assert clean["flagged"] == [] and clean["accuracy"] == pytest.approx(clean["answered_correct"])


def test_the_rrf_arms_dense_similarities_are_cached_and_replayed(tmp_path, monkeypatch):
    from memctl.rerank import DenseSimilarity

    cache = tmp_path / "dense.jsonl"
    dense = DenseSimilarity("toy-dense", str(cache))
    monkeypatch.setattr(dense, "_compute", lambda query, texts: [float(len(t)) for t in texts])
    assert list(dense.score("q", ["a", "abc"])) == [1.0, 3.0]
    replay = DenseSimilarity("toy-dense", str(cache), cache_only=True)
    assert list(replay.score("q", ["abc", "a"])) == [3.0, 1.0]  # no model needed: every pair is cached
    with pytest.raises(KeyError):
        replay.score("q", ["new turn"])
