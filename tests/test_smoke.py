"""End-to-end smoke test: 2 conversations, fake Jev, stub model. Must finish in under 2 minutes."""

import json
import time

import pytest
import yaml

from memctl.llm import CachedLLM, StubLLM
from memctl.benchmarks import load_benchmark
from memctl.run import run


def config_for(controller: str, tmp_path) -> dict:
    config = yaml.safe_load(open("configs/smoke_jev.yaml"))
    config["controller"] = {"name": controller}
    config["output_dir"] = str(tmp_path / "runs")
    config["model"]["cache_dir"] = str(tmp_path / "cache")
    return config


def read_jsonl(path) -> list[dict]:
    return [json.loads(line) for line in open(path)]


def test_smoke_run_with_fake_jev(tmp_path):
    started = time.time()
    folder = run(config_for("jev", tmp_path), allow_fake_jev=True)
    assert time.time() - started < 120

    for name in ["config.yaml", "git_commit.txt", "env.txt", "decisions.jsonl", "answers.jsonl", "metrics.json"]:
        assert (folder / name).exists(), name
    assert "FAKE" in folder.name
    metrics = json.loads((folder / "metrics.json").read_text())
    assert metrics["controller"] == "jev-FAKE" and metrics["fake_jev"] is True
    assert metrics["conversations"] == 2 and metrics["questions"] > 0

    decisions = read_jsonl(folder / "decisions.jsonl")
    assert decisions and all(d["controller"] == "jev-FAKE" for d in decisions)
    moves = [d for d in decisions if d["kind"] != "keep"]
    keeps = [d for d in decisions if d["kind"] == "keep"]
    assert all("FAKE" in d["reason"] for d in moves)
    assert {"step", "item_id", "features", "place", "reason", "confidence"} <= set(moves[0])
    assert len(keeps) == metrics["cost"]["controller_calls"]  # kept items are logged once per call
    assert all(set(d["oracle_would_move"]) <= set(d["kept_item_ids"]) for d in keeps)
    new_items_kept = sum(1 for d in decisions if d["kind"] == "keep_new_item")
    graded_keeps = sum(len(d["kept_item_ids"]) for d in keeps) + new_items_kept
    assert metrics["oracle_agreement"]["keeps"]["decisions"] == graded_keeps
    assert folder.name.split("_")[-1] not in ("seed0", "")  # the folder name ends with the git hash

    answers = read_jsonl(folder / "answers.jsonl")
    assert all(row["evidence"] for row in answers)
    assert len(read_jsonl(folder / "jev_calls.jsonl")) == metrics["cost"]["jev_calls"]


def test_features_and_decisions_never_contain_evidence_labels(tmp_path):
    """Controllers must not see which items are evidence."""
    folder = run(config_for("keep_newest", tmp_path), allow_fake_jev=False)
    text = (folder / "decisions.jsonl").read_text()
    assert "evidence" not in text and "gold" not in text


def test_full_context_has_all_evidence_and_beats_a_tight_budget(tmp_path):
    full = json.loads((run(config_for("full_context", tmp_path)) / "metrics.json").read_text())
    tight = json.loads((run(config_for("file_everything", tmp_path)) / "metrics.json").read_text())
    assert full["evidence_availability"]["in_context"] == 1.0
    assert full["answer_quality"]["f1"] >= tight["answer_quality"]["f1"]


def test_generation_cache_makes_reruns_free(tmp_path):
    llm = CachedLLM(StubLLM(), str(tmp_path / "cache"), {"name": "stub"})
    prompt = "## Memory in context\n(Ann) the sky is teal\n## Archive index\n## Question\nwhat colour is the sky"
    first = llm.generate(prompt, 16)
    second = llm.generate(prompt, 16)
    assert first == second and (llm.misses, llm.hits) == (1, 1)


def test_synthetic_benchmark_is_deterministic_and_well_formed():
    first, again = load_benchmark({"name": "synthetic"}, seed=0), load_benchmark({"name": "synthetic"}, seed=0)
    assert [c.events for c in first] == [c.events for c in again]
    for conversation in first:
        item_ids = {i.id for i in conversation.items()}
        for question in conversation.questions():
            assert set(question.evidence_ids) <= item_ids
