"""The LoCoMo and LongMemEval adapters, and local answer scoring."""

import json
from pathlib import Path

import pytest

from memctl.config import resolve
from memctl.envs import build_env
from memctl.harness.runner import run_one
from memctl.judge import Judge
from memctl.llm import ScriptedLLM
from memctl.metrics import score_answer, token_f1

LOCOMO = Path("data/locomo/locomo10.json")
LONGMEMEVAL = Path("data/longmemeval/longmemeval_oracle.json")
needs_locomo = pytest.mark.skipif(not LOCOMO.exists(), reason="LoCoMo data file is not present")
needs_longmemeval = pytest.mark.skipif(not LONGMEMEVAL.exists(), reason="LongMemEval data file is not present")
ARCHIVE_OPS = ["KEEP", "EVICT", "MOVE_TO_ARCHIVE", "RETRIEVE_FROM_ARCHIVE", "NO_OP"]


def test_local_scoring():
    assert token_f1("7 May 2023", "7 May 2023") == 1.0
    assert token_f1("the beach", "a mountain") == 0.0
    assert score_answer("It was on 7 May 2023, I think", "7 May 2023")["correct"]
    assert not score_answer("sometime in June", "7 May 2023")["correct"]
    assert score_answer("unknown", "anything", unanswerable=True)["correct"]
    assert not score_answer("Paris", "anything", unanswerable=True)["correct"]


def test_the_judge_reads_yes_or_no():
    judge = Judge({}, llm=ScriptedLLM(lambda prompt: "Yes." if "Answer to grade: 7 May" in prompt else "no"))
    assert judge.is_correct("When?", "7 May 2023", "7 May") is True
    assert judge.is_correct("When?", "7 May 2023", "June") is False
    assert judge.calls == 2


FIXTURE = [
    {
        "question_id": "q1", "question_type": "single-session-user", "question": "What colour is my bike?",
        "answer": "teal", "question_date": "2023/05/02 (Tue) 10:00",
        "haystack_dates": ["2023/05/01 (Mon) 09:00", "2023/04/01 (Sat) 09:00"],
        "haystack_session_ids": ["s_late", "s_early"],
        "haystack_sessions": [
            [{"role": "user", "content": "My bike is teal.", "has_answer": True},
             {"role": "assistant", "content": "Nice colour."}],
            [{"role": "user", "content": "I like hiking."}, {"role": "assistant", "content": "Good for you."}],
        ],
        "answer_session_ids": ["s_late"],
    },
    {
        "question_id": "q2_abs", "question_type": "multi-session", "question": "What is my cat called?",
        "answer": "not mentioned", "question_date": "2023/05/02 (Tue) 10:00",
        "haystack_dates": ["2023/04/01 (Sat) 09:00"], "haystack_session_ids": ["s1"],
        "haystack_sessions": [[{"role": "user", "content": "I have a dog."}]], "answer_session_ids": ["s1"],
    },
]


def test_longmemeval_orders_sessions_by_date_and_marks_evidence(tmp_path):
    path = tmp_path / "lme.json"
    path.write_text(json.dumps(FIXTURE))
    env = build_env({"name": "longmemeval", "path": str(path)})
    observation = env.reset(0)
    seen = []
    while observation is not None:
        seen.append(observation)
        observation = env.step("teal" if observation.requires_response else None).observation
    assert [o.id for o in seen] == ["s_early:0", "s_early:1", "s_late:0", "s_late:1", "q0000"]
    assert seen[0].metadata["date"].startswith("2023/04/01") and seen[-1].requires_response
    (dependency,) = env.get_ground_truth_dependencies()
    assert dependency.requirements[0].item_ids == ("s_late:0",) and dependency.step == 5
    assert env.task_success() == 1.0

    env.reset(1)  # unanswerable, and no turn is marked: falls back to the answer session
    while env.get_observation() is not None:
        env.step("unknown" if env.get_observation().requires_response else None)
    (dependency,) = env.get_ground_truth_dependencies()
    assert dependency.requirements[0].item_ids == ("s1:0",) and env.task_success() == 1.0


def test_a_missing_benchmark_file_says_where_to_get_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="huggingface"):
        build_env({"name": "longmemeval", "path": str(tmp_path / "nope.json")})
    with pytest.raises(FileNotFoundError, match="download"):
        build_env({"name": "locomo", "path": str(tmp_path / "nope.json")})


def qa_config(env, controller, fraction, operations=None):
    memory = {"budget": {"fraction": fraction}}
    if operations:
        memory["allowed_operations"] = operations
    return resolve(
        {"env": env, "agent": {"name": "llm", "model": {"backend": "stub"}}, "controller": controller, "memory": memory}
    )


@needs_locomo
def test_locomo_runs_end_to_end_and_reports_where_the_evidence_was():
    env = {"name": "locomo"}
    result = run_one(qa_config(env, {"name": "fifo"}, 0.25), seed=0)
    episode = result.episode
    assert episode["queries"] == 199 and episode["steps"] == 419 + 199
    assert 0 < episode["needed_hit_rate"] < 1  # a quarter of the history cannot hold all the evidence
    assert 0 < episode["evidence_complete_rate"] <= episode["needed_hit_rate"] + 0.2
    assert set(episode["failures"]) <= {"evicted", "task_model_reasoning", "unknown"}
    assert episode["env_stats"]["scored_by"] == "local f1" and episode["agent_model"] == "stub"
    limited = run_one(qa_config({**env, "max_questions": 5}, {"name": "full_context"}, 1.0), seed=0).episode
    assert limited["queries"] == 5 and limited["needed_hit_rate"] == 1.0 and "evicted" not in limited["failures"]


@needs_locomo
def test_locomo_seeds_walk_through_the_ten_conversations():
    env = build_env({"name": "locomo", "max_questions": 1})
    first, second = env.reset(0).content, env.reset(1).content
    assert first != second and env.reset(10).content == first


@needs_longmemeval
def test_longmemeval_real_file_runs_with_archive_and_retrieval():
    env = {"name": "longmemeval"}
    controller = {"name": "fifo", "removal": ["MOVE_TO_ARCHIVE"], "retrieve": {"top_k": 5}}
    episode = run_one(qa_config(env, controller, 0.25, ARCHIVE_OPS), seed=0).episode
    assert episode["queries"] == 1 and episode["retrieved_items"] > 0
    assert episode["env_stats"]["episode"] and episode["needed_hit_rate"] is not None


@needs_longmemeval
def test_a_longmemeval_subset_keeps_training_off_the_evaluation_questions():
    whole = build_env({"name": "longmemeval"})
    train = build_env({"name": "longmemeval", "subset": [100, 500]})
    evaluated = {whole.load_episode(seed).id for seed in range(100)}
    trained = {train.load_episode(seed).id for seed in range(100_000, 100_400)}
    assert len(trained) == 400 and not evaluated & trained
    assert train.load_episode(0).id == whole.load_episode(100).id


def test_stratified_folds_cover_every_question_once_and_balance_question_types():
    from collections import Counter

    from memctl.envs.longmemeval import fold_indices

    types = ["a"] * 50 + ["b"] * 30 + ["c"] * 20
    instances = [{"question_type": t, "question_id": f"q{i}" + ("_abs" if i % 10 == 0 else "")} for i, t in enumerate(types)]
    abstentions = {i for i in range(100) if i % 10 == 0}
    tests = [fold_indices(instances, 5, fold) for fold in range(5)]
    assert sorted(i for fold in tests for i in fold) == list(range(100))
    for fold, test in enumerate(tests):
        assert len(test) == 20 and Counter(types[i] for i in test) == {"a": 10, "b": 6, "c": 4}
        assert sorted(test + fold_indices(instances, 5, fold, "train")) == list(range(100))
        assert 1 <= len(abstentions & set(test)) <= 3
        half_a, half_b = fold_indices(instances, 5, fold, "train_a"), fold_indices(instances, 5, fold, "train_b")
        assert sorted(half_a + half_b) == fold_indices(instances, 5, fold, "train") and len(half_a) == 40
        assert Counter(types[i] for i in half_a) == {"a": 20, "b": 12, "c": 8}
    with pytest.raises(ValueError):
        fold_indices(instances, 5, 5)


@needs_longmemeval
def test_a_longmemeval_fold_spans_all_question_types():
    test = build_env({"name": "longmemeval", "folds": {"k": 5, "fold": 0}})
    train = build_env({"name": "longmemeval", "folds": {"k": 5, "fold": 0, "part": "train"}})
    tested = {test.load_episode(seed).id for seed in range(100)}
    trained = {train.load_episode(seed).id for seed in range(400)}
    assert len(tested) == 100 and len(trained) == 400 and not tested & trained
    categories = {test.load_episode(seed).questions[0].category for seed in range(100)}
    assert len(categories) == 6


@needs_longmemeval
def test_answer_sessions_without_marked_turns_still_count_as_evidence():
    from memctl.envs.longmemeval import _load

    env = build_env({"name": "longmemeval"})
    instances = _load(env.path)

    def marked_sessions(instance):
        return {sid for sid, turns in zip(instance["haystack_session_ids"], instance["haystack_sessions"])
                if any(turn.get("has_answer") for turn in turns)}

    partial = next(n for n, inst in enumerate(instances)
                   if marked_sessions(inst) and marked_sessions(inst) < set(inst["answer_session_ids"]))
    env.reset(partial)
    while not env.is_done():
        env.step("unknown" if env.get_observation() and env.get_observation().requires_response else None)
    [dependency] = env.get_ground_truth_dependencies()
    unmarked = set(instances[partial]["answer_session_ids"]) - marked_sessions(instances[partial])
    whole_sessions = [r for r in dependency.requirements if len(r.item_ids) > 1]
    assert len(whole_sessions) == len(unmarked) and any(len(r.item_ids) == 1 for r in dependency.requirements)


def test_the_longmemeval_judge_style_picks_the_official_prompt_for_each_question_type():
    judge = Judge({"style": "longmemeval", "backend": "stub"})
    temporal = judge.prompt("How many days?", "18", "19 days", "temporal-reasoning")
    assert "off-by-one" in temporal and temporal.endswith("Answer yes or no only.")
    assert "Rubric: G" in judge.prompt("Q", "G", "R", "single-session-preference")
    assert "updated answer" in judge.prompt("Q", "G", "R", "knowledge-update")
    assert judge.prompt("Q", "G", "R", "multi-session") == judge.prompt("Q", "G", "R", "single-session-user")
    assert "Reply with exactly one word" in Judge({"backend": "stub"}).prompt("Q", "G", "R", "temporal-reasoning")
    with pytest.raises(ValueError):
        Judge({"style": "lenient", "backend": "stub"})


@needs_longmemeval
def test_a_composed_longmemeval_episode_asks_each_question_after_its_own_history():
    env = build_env({"name": "longmemeval", "folds": {"k": 5, "fold": 0, "part": "train"}, "compose": 3})
    first = env.reset(7)
    order, answered = [first], []
    while not env.is_done():
        observation = env.get_observation()
        if observation.requires_response:
            answered.append((observation.id, len(order)))
            dependencies = {d.query_id: d for d in env.get_ground_truth_dependencies()}
            assert dependencies[observation.id].step == len(order)
            assert all(i.startswith(f"i") for r in dependencies[observation.id].requirements for i in r.item_ids)
        env.step("unknown" if observation.requires_response else None)
        if env.get_observation() is not None:
            order.append(env.get_observation())
    assert len(answered) == 3 and len({o.id for o in order}) == len(order)
    turns = [o for o in order if not o.requires_response]
    dates = [t.metadata["date"] for t in turns]
    assert dates == sorted(dates) and {t.id.split("/")[0] for t in turns} == {"i0", "i1", "i2"}
    for query_id, position in answered:
        before = order[position - 2]
        assert before.id == env._questions[query_id].after or before.requires_response


@needs_longmemeval
def test_composed_groupings_differ_across_seeds():
    env = build_env({"name": "longmemeval", "folds": {"k": 5, "fold": 0, "part": "train"}, "compose": 4})
    groups = {env.load_episode(seed).id for seed in range(20)}
    members = [set(g.split("+")) for g in groups]
    assert len(groups) == 20 and all(len(m) == 4 for m in members)
    assert len(set.union(*members)) > 30  # not 5 fixed quadruples cycling


LONGMEMEVAL_S = Path("data/longmemeval/longmemeval_s_cleaned.json")


@pytest.mark.skipif(not LONGMEMEVAL_S.exists(), reason="LongMemEval-S is not present")
def test_sharded_loading_plays_exactly_the_full_file_episodes(monkeypatch):
    """Every fold, 10 seeds, single and composed: the same episode, turns, questions and dependencies."""
    import memctl.envs.longmemeval as lme
    from memctl.splits import fold_indices

    path = str(LONGMEMEVAL_S)
    full = lme._load(path)
    for fold in range(5):
        assert fold_indices(full, 5, fold) == fold_indices(lme._index(path), 5, fold)

    def play(env, seed):
        first = env.reset(seed)
        turns, dependencies = [(first.id, first.content)], []
        while not env.is_done():
            observation = env.get_observation()
            if observation.requires_response:
                dependencies.append(env.get_ground_truth_dependencies())
            env.step("unknown" if observation.requires_response else None)
            if env.get_observation() is not None:
                turns.append((env.get_observation().id, env.get_observation().content))
        episode = env._episode
        return episode.id, turns, [(q.id, q.gold, q.category) for q in episode.questions], dependencies

    for compose in (1, 4):
        for fold in range(5):
            config = {"name": "longmemeval", "path": path, "folds": {"k": 5, "fold": fold}, "compose": compose}
            sharded = build_env(config)
            with monkeypatch.context() as patch:  # the old loader: the whole file in this process
                patch.setattr(lme, "_index", lambda p: full)
                patch.setattr(lme, "_instance", lambda p, n: full[n])
                whole = build_env(config)
                expected = [play(whole, seed) for seed in range(10)]
            assert [play(sharded, seed) for seed in range(10)] == expected
