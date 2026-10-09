"""Experiment 28: the query rewriter (memctl/query_rewrite.py) and its hook in the RL controller."""

from datetime import date
from types import SimpleNamespace

import pytest
import torch

from memctl.config import resolve
from memctl.harness.runner import Experiment
from memctl.query_rewrite import QueryRewriter, parse_rewrite, pool, rule_range, split_question, widen
from memctl.retrieval import LexicalRetriever

ASKED = date(2023, 5, 30)  # a Tuesday


@pytest.mark.parametrize("question, expected", [
    ("What did I buy two weeks ago?", (date(2023, 5, 12), date(2023, 5, 20))),
    ("What did I do last Saturday?", (date(2023, 5, 25), date(2023, 5, 29))),
    ("Where did I go in March?", (date(2023, 2, 27), date(2023, 4, 2))),
    ("What did I do on the 5th of April?", (date(2023, 4, 3), date(2023, 4, 7))),
    ("In the past three months, how many books did I read?", (date(2023, 2, 27), ASKED)),
    ("Where did I go in June?", (date(2022, 5, 30), date(2022, 7, 2))),  # the latest June before the question
])
def test_rule_range_reads_time_phrases_counted_back_from_the_question_date(question, expected):
    assert rule_range(question, ASKED, 1.0) == expected


def test_rule_range_ignores_questions_without_a_time_and_words_that_only_look_like_one():
    assert rule_range("How many days ago did I start yoga?", ASKED) is None
    assert rule_range("May I ask what my sister's name is?", ASKED) is None
    assert rule_range("What did I buy two weeks ago?", None) is None
    assert rule_range("What did I eat yesterday?", ASKED)[1] == ASKED  # never after the question date


def test_slack_widens_the_window():
    narrow, wide = rule_range("two weeks ago", ASKED, 0.5), rule_range("two weeks ago", ASKED, 4.0)
    assert wide[0] < narrow[0] and narrow[1] < wide[1]


def test_split_question_and_parse_rewrite():
    assert split_question("(asked on 2023/05/30 (Tue) 23:40) What did I buy?") == (ASKED, "What did I buy?")
    assert split_question("Question: (asked on 2023/05/30 (Tue) 23:40) What?") == (ASKED, "What?")
    assert parse_rewrite("QUERY: yoga class\nRANGE: 2023/05/10 - 2023/05/01") == \
        ("yoga class", (date(2023, 5, 1), date(2023, 5, 10)), True)
    assert parse_rewrite("QUERY: yoga\nRANGE: none") == ("yoga", None, True)
    assert parse_rewrite("I think you went to yoga.") == ("", None, False)  # malformed: the raw question
    assert widen((date(2023, 5, 20), date(2023, 5, 29)), 3, ASKED) == (date(2023, 5, 17), ASKED)


def _items():
    texts = [("yoga class at the studio", "2023/05/01 (Mon) 10:00"), ("yoga mat shopping", "2023/05/20 (Sat) 10:00"),
             ("dinner recipes", "2023/05/21 (Sun) 10:00"), ("yoga retreat plans", "2023/03/02 (Thu) 10:00")]
    return [SimpleNamespace(id=f"t{n}", content=c, metadata={"date": d}, created_at=n) for n, (c, d) in enumerate(texts)]


def test_pool_drop_keeps_only_in_range_and_demote_puts_them_first():
    items, retriever = _items(), LexicalRetriever()
    span = (date(2023, 5, 15), date(2023, 5, 25))
    raw = [i.id for i, _ in pool(retriever, "yoga", items, 3, None)]
    assert raw == [i.id for i, _ in retriever.search("yoga", items, 3)]
    assert [i.id for i, _ in pool(retriever, "yoga", items, 3, span, "drop")] == ["t1"]
    demoted = [i.id for i, _ in pool(retriever, "yoga", items, 3, span, "demote")]
    assert demoted[0] == "t1" and set(demoted) == set(raw)
    with pytest.raises(ValueError):
        pool(retriever, "yoga", items, 3, span, "shuffle")


def test_llm_rewriter_replays_from_the_cache_and_a_miss_is_an_error(tmp_path):
    prompt = tmp_path / "v1.txt"
    prompt.write_text("Asked {asked}. Q: {question}")
    seen = []

    def model(text):
        seen.append(text)
        return "QUERY: mat\nRANGE: 2023/05/19 - 2023/05/21"

    config = {"mode": "llm", "action": "drop", "pad_days": 0, "prompt_file": str(prompt),
              "model": {"backend": "scripted", "function": model, "cache_dir": str(tmp_path / "cache")}}
    content = "(asked on 2023/05/30 (Tue) 23:40) What yoga thing did I buy?"
    first = [i.id for i, _ in QueryRewriter(config).search(LexicalRetriever(), content, _items(), 3)]
    assert first == ["t1"] and seen == ["Asked 2023/05/30 (Tuesday). Q: What yoga thing did I buy?"]
    replay = QueryRewriter({**config, "model": {**config["model"], "cache_only": True}})
    assert [i.id for i, _ in replay.search(LexicalRetriever(), content, _items(), 3)] == first and len(seen) == 1
    with pytest.raises(RuntimeError):
        replay.search(LexicalRetriever(), content.replace("yoga", "dance"), _items(), 3)
    with pytest.raises(ValueError):
        QueryRewriter({"mode": "rule", "action": "shuffle"})


def test_the_rl_controller_runs_with_a_rule_rewriter():
    torch.manual_seed(0)
    settings = resolve({"env": {"name": "synthetic_recall", "horizon": 60},
                        "controller": {"name": "rl", "keep_none": True, "floor_head": True, "retrieve_floor": 3,
                                       "retrieve_candidates": 8, "query_rewrite": {"mode": "rule", "action": "drop"}},
                        "memory": {"budget": {"fraction": 0.25}, "allowed_operations": [
                            "KEEP", "MOVE_TO_ARCHIVE", "RETRIEVE_FROM_ARCHIVE", "NO_OP"]}})
    experiment = Experiment(settings)
    episode = experiment.run_episode(seed=0, detail=False).episode
    assert episode["invalid_actions"] == 0
    assert experiment.controller.rewriter is not None and experiment.controller.rewriter.action == "drop"


def test_inspect_reads_locomo_dev_questions_only_and_refuses_longmemeval():
    from pathlib import Path

    from memctl import rewrite_gate

    with pytest.raises(PermissionError):
        rewrite_gate._check_dev_only([0])
    with pytest.raises(PermissionError):
        rewrite_gate._check_dev_only(["dev:conv-26_q0", "e47becba"])  # a LongMemEval question id
    rewrite_gate._check_dev_only(["dev:conv-26_q0"])
    with pytest.raises(PermissionError):
        rewrite_gate.inspect(1, ids=[3])  # refused before any output is read
    assert rewrite_gate._lme_date("1:56 pm on 8 May, 2023") == "2023/05/08 (Mon) 13:56"
    if Path(rewrite_gate.LOCOMO_PATH).exists():
        dev = rewrite_gate.dev_questions()
        assert len(dev) > 200 and all(q.id.startswith("dev:") for q in dev)
        asked, _ = split_question(dev[0].content)
        assert asked is not None and all(i.metadata["date"][:4].isdigit() for i in dev[0].items)
