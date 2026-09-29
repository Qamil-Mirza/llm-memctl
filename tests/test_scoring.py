"""Scores, failure attribution, the LoCoMo loader and the report."""

import json
from pathlib import Path

import pytest
import yaml

from memctl.attribution import failure_label
from memctl.judge import Judge
from memctl.metrics import bleu1, score_answer, token_f1
from memctl.report import closed_gap
from memctl.run import run


def evidence(place: str, in_prompt: bool) -> dict:
    return {"item_id": "x", "place": place, "in_prompt": in_prompt}


def test_f1_and_bleu1():
    assert token_f1("7 May 2023", "7 May 2023") == 1.0
    assert token_f1("the 7 May", "7 May 2023") == pytest.approx(0.8)
    assert token_f1("blue", "red") == 0.0
    assert bleu1("7 May 2023", "7 May 2023") == 1.0
    assert bleu1("May", "7 May 2023") == pytest.approx(0.1353, abs=1e-3)  # right word, but too short
    assert bleu1("it was on 7 May 2023 I think", "7 May 2023") == pytest.approx(3 / 8)


def test_adversarial_questions_are_right_only_when_the_model_declines():
    assert score_answer("Not mentioned in the conversation.", "Not mentioned in the conversation", "adversarial")["f1"] == 1.0
    assert score_answer("She felt proud", "Not mentioned in the conversation", "adversarial")["f1"] == 0.0


@pytest.mark.parametrize(
    "items, expected",
    [
        ([], "no_evidence_label"),
        ([evidence("CONTEXT", True)], "evidence_present_model_wrong"),
        ([evidence("STORE", True), evidence("ARCHIVE", True)], "evidence_present_model_wrong"),
        ([evidence("STORE", False)], "evidence_in_store_not_retrieved"),
        ([evidence("ARCHIVE", False), evidence("STORE", False)], "evidence_archived_not_recalled"),
        ([evidence("DROPPED", False), evidence("ARCHIVE", False)], "evidence_dropped"),
        ([evidence("CONTEXT", True), evidence("DROPPED", False)], "evidence_dropped"),
    ],
)
def test_every_wrong_answer_gets_exactly_one_label(items, expected):
    assert failure_label(items) == expected


def test_closed_gap():
    assert closed_gap(0.5, baseline=0.4, oracle=0.8) == 0.25
    assert closed_gap(0.8, baseline=0.4, oracle=0.8) == 1.0
    assert closed_gap(0.5, baseline=0.4, oracle=0.4) is None


def test_judge_reads_the_verdict():
    class Fixed:
        name = "fixed"

        def __init__(self, reply):
            self.reply = reply

        def generate(self, prompt, max_new_tokens, shared_prefix=""):
            return self.reply

    assert Judge(Fixed("CORRECT")).is_correct("q", "gold", "answer", "single-hop") is True
    assert Judge(Fixed("WRONG")).is_correct("q", "gold", "answer", "single-hop") is False
    assert Judge(Fixed("CORRECT")).is_correct("q", "gold", "She was proud", "adversarial") is False


@pytest.mark.skipif(not Path("data/locomo/locomo10.json").exists(), reason="LoCoMo not downloaded")
def test_locomo_loader():
    from memctl.benchmarks import load_benchmark

    conversations = load_benchmark({"name": "locomo", "limit": 2, "questions_per_conversation": 20}, seed=0)
    assert [c.id for c in conversations] == ["conv-26", "conv-30"]
    first = conversations[0]
    assert len(first.items()) == 419 and len(first.questions()) == 20
    assert len({q.category for q in first.questions()}) == 5
    item_ids = {i.id for i in first.items()}
    assert all(set(q.evidence_ids) <= item_ids for q in first.questions())
    kinds = [e.kind for e in first.events]
    assert kinds.index("question") > max(i for i, k in enumerate(kinds) if k == "item")  # asked at the end


def test_report_is_written_with_labels_for_every_wrong_answer(tmp_path):
    config = yaml.safe_load(open("configs/smoke_jev.yaml"))
    config["output_dir"], config["model"]["cache_dir"] = str(tmp_path / "runs"), str(tmp_path / "cache")
    for name in ["keep_newest", "oracle", "jev"]:
        config["controller"] = {"name": name}
        folder = run(config, allow_fake_jev=True)
    answers = [json.loads(line) for line in (folder / "answers.jsonl").open()]
    assert all((a["failure_label"] is None) == a["correct"] for a in answers)
    decisions = [json.loads(line) for line in (folder / "decisions.jsonl").open()]
    assert all(d["oracle_place"] in ("CONTEXT", "STORE", "DROPPED") for d in decisions if d["kind"] != "keep")
    report = (folder / "report.md").read_text()
    assert "FAKE JEV" in report and "memory failures" in report and "Closed gap" in report
    assert "| oracle |" in report and "| keep_newest |" in report
    assert (folder / "failure_attribution.png").exists() and (folder / "items_by_place.png").exists()
    assert "average number of tokens in the prompt" in report and (folder / "accuracy_vs_cost.png").exists()
