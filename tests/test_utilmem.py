"""The UtilMem adapter (§30) on a small synthetic fixture: no GPU, no network, stub models."""

import json

import pytest

from memctl.envs import build_env
from memctl.envs.utilmem import (
    JUDGE_PROMPT, UtilMemEnv, is_refusal, normalized_robustness, parse_question, parse_score, question_pool,
    split_bundles,
)
from memctl.harness.runner import run_one


def _session(topic, turns=2, long=False):
    out = []
    for n in range(turns):
        out.append({"role": "user", "content": f"Question {n} about {topic}." + (" filler" * 1500 if long else "")})
        out.append({"role": "assistant", "content": f"Answer {n} about {topic}."})
    return out


def _bundle(number):
    """Two domains; studychat has evidence sessions gt-a and gt-b, finance has gt-c; three distractors."""
    sessions = [
        ("noise_studychat_x_1", "2026/08/01 (Sat) 09:00", "noise", "studychat", _session("decision trees")),
        ("gt-a", "2026/08/02 (Sun) 09:00", "gt", "studychat", _session("binary tree insert", long=True)),
        ("gt-c", "2026/08/03 (Mon) 09:00", "gt", "finance", _session("credit card debt")),
        ("noise_finance_y_2", "2026/08/04 (Tue) 09:00", "noise", "finance", _session("monopoly cash")),
        ("gt-b", "2026/08/05 (Wed) 09:00", "gt", "studychat", _session("RNN text generation")),
        ("noise_fitness_z_3", "2026/08/06 (Thu) 09:00", "noise", "fitness", _session("knee rehab")),
    ]
    return {
        "sample_id": f"multi_{number:04d}_abc",
        "question_type": "multi-domain-multi-session-user",
        "question_pool": {"studychat": ["Summarize my binary tree and RNN work.", "List my AI assignments."],
                          "finance": ["Plan my credit card debt payoff."]},
        "answer_session_ids": {"studychat": ["gt-a", "gt-b"], "finance": ["gt-c"]},
        "bundle_meta": {},
        "haystack_dates": [s[1] for s in sessions],
        "haystack_session_ids": [s[0] for s in sessions],
        "haystack_sessions": [s[4] for s in sessions],
        "haystack_session_meta": [{"session_id": s[0], "timestamp": s[1], "role": s[2], "domain": s[3]} for s in sessions],
    }


@pytest.fixture
def path(tmp_path):
    file = tmp_path / "multi_domain_eval_strong.json"
    file.write_text(json.dumps([_bundle(n) for n in range(10)]))
    return str(file)


def test_split_is_fixed_and_disjoint():
    dev, evaluation = split_bundles(100)
    assert dev == [0, 3, 6, 9, 10, 17, 20, 26, 31, 32, 37, 48, 50, 59, 66, 69, 78, 79, 82, 83]  # §30, committed
    assert len(evaluation) == 80 and not set(dev) & set(evaluation)
    assert split_bundles(10) == split_bundles(10) and len(split_bundles(10)[0]) == 2


def test_question_pool_parts_sample_and_order(path):
    dev, evaluation = split_bundles(10)
    every = question_pool(path, "all")
    assert len(every) == 30 and [q["domain"] for q in every[:3]] == ["studychat", "studychat", "finance"]
    assert {q["bundle"] for q in question_pool(path, "dev")} == set(dev)
    assert {q["bundle"] for q in question_pool(path, "eval")} == set(evaluation)
    sample = question_pool(path, "eval", {"seed": 30, "per_domain": 3})
    assert [q["domain"] for q in sample].count("studychat") == 3 and [q["domain"] for q in sample].count("finance") == 3
    assert sample == question_pool(path, "eval", {"seed": 30, "per_domain": 3})
    assert sample == sorted(sample, key=lambda q: (q["bundle"], q["domain"] != "studychat", q["number"]))


def test_neutral_ids_hide_the_noise_label_and_evidence_is_grouped_by_session(path):
    episode, groups = parse_question(_bundle(0), "studychat", 0)
    assert len(episode.turns) == 24 and not any("noise" in t.id or "noise" in str(t.metadata) for t in episode.turns)
    assert [t.id for t in episode.turns[:2]] == ["s000:0", "s000:1"]
    assert groups == (tuple(f"s001:{n}" for n in range(4)), tuple(f"s004:{n}" for n in range(4)))
    assert episode.questions[0].id == "multi_0000_abc/studychat/q0" and episode.questions[0].after is None
    oracle, oracle_groups = parse_question(_bundle(0), "studychat", 0, oracle=True)
    assert {t.metadata["session"] for t in oracle.turns} == {"s001", "s004"} and oracle_groups == groups
    cut, _ = parse_question(_bundle(0), "studychat", 0, max_turn_tokens=50)
    assert max(len(t.content.split()) for t in cut.turns) <= 50 < max(len(t.content.split()) for t in episode.turns)


def test_episode_walk_logs_every_answer_and_one_requirement_per_session(path):
    env = build_env({"name": "utilmem", "path": path, "part": "all"})
    assert isinstance(env, UtilMemEnv)
    observation = env.reset(2)  # the third question: finance
    while observation is not None:
        result = env.step("unknown" if observation.requires_response else None)
        if observation.requires_response:
            assert result.info["scored"] and result.info["correct"] is False and result.info["category"] == "finance"
        observation = result.observation
    (dependency,) = env.get_ground_truth_dependencies()
    assert [r.item_ids for r in dependency.requirements] == [tuple(f"s002:{n}" for n in range(4))]
    assert env.episode_stats()["answers"] == [{"question_id": "multi_0000_abc/finance/q0", "domain": "finance",
                                               "refusal": True}]


def test_shards_cover_the_pool_once(path):
    pool = question_pool(path, "eval")
    parts = [build_env({"name": "utilmem", "path": path, "shard": {"k": 3, "index": i}}).pool for i in range(3)]
    assert sorted(json.dumps(q) for p in parts for q in p) == sorted(json.dumps(q) for q in pool)


def test_run_one_fifo_and_oracle(path):
    config = {
        "episodes": 1,
        "env": {"name": "utilmem", "path": path, "part": "all", "max_turn_tokens": 200},
        "agent": {"name": "llm", "model": {"backend": "stub"}},
        "controller": {"name": "fifo", "removal": ["MOVE_TO_ARCHIVE"],
                       "retrieve": {"top_k": 2, "method": "lexical", "fit": True}, "target_tokens": 40},
        "memory": {"budget": {"fraction": 0.3}, "count_labels": True,
                   "allowed_operations": ["KEEP", "MOVE_TO_ARCHIVE", "RETRIEVE_FROM_ARCHIVE", "NO_OP"]},
        "hindsight": {"enabled": False},
    }
    result = run_one(config)
    assert result.episode["queries"] == 1 and result.episode["evidence_complete_rate"] is not None
    assert len(result.failures) == 1 and len(result.failures[0]["evidence"]) == 2
    oracle = run_one({**config, "env": {**config["env"], "oracle": True}, "controller": {"name": "full_context"}})
    assert oracle.episode["evidence_complete_rate"] == 1.0


def test_judge_helpers():
    assert "{context}" in JUDGE_PROMPT and "{answer_noisy}" in JUDGE_PROMPT
    JUDGE_PROMPT.format(context="c", question="q", answer_oracle="a", answer_noisy="b")
    assert parse_score('{"analysis": "x", "sub_dimensions": {"a": "mild"}, "score": 7, "failure_mode": "omission"}') == 7
    assert parse_score('```json\n{"final_score": "9"}\n```') == 9
    assert parse_score('broken {"score": 4, ') == 4
    assert parse_score('{"score": 11}') is None and parse_score("no score") is None
    assert is_refusal("Unknown.") and is_refusal("I don't know which plan you mean.") and not is_refusal("Plan: pay A first.")
    assert normalized_robustness([10, 10]) == 1.0 and normalized_robustness([1, 1]) == 0.0
