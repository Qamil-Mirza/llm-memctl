"""Tests for memctl.compare: the cross-budget tables and the paired test."""

from __future__ import annotations

import json
import math

import pytest
import yaml

from memctl.compare import (
    binomial_two_sided,
    budget_label,
    budget_order,
    controller_row,
    load_runs,
    paired,
    verdict,
    write_comparison,
)


def answer(question_id: str, correct: bool, category: str = "single-hop", f1: float = 0.5) -> dict:
    """One row of answers.jsonl, with the fields compare.py reads."""
    return {
        "question_id": question_id,
        "category": category,
        "correct": correct,
        "scores": {"f1": f1, "bleu1": f1, "judge": float(correct)},
        "evidence": [{"item_id": "D1:1", "place": "CONTEXT", "in_prompt": correct}],
        "failure_label": None if correct else ("evidence_present_model_wrong" if correct is False else None),
        "prompt_tokens": 100,
    }


def wrong_because(question_id: str, label: str, category: str = "single-hop") -> dict:
    row = answer(question_id, False, category)
    row["failure_label"] = label
    return row


# ------------------------------------------------------------------ the exact binomial test


def test_binomial_two_sided_is_symmetric_and_exact():
    # 10 discordant pairs split 10-0 is 2 * (1/2)^10
    assert binomial_two_sided(10, 10) == pytest.approx(2 / 1024)
    assert binomial_two_sided(0, 10) == pytest.approx(2 / 1024)  # same, the other way round
    assert binomial_two_sided(5, 10) == pytest.approx(1.0)  # an even split cannot be significant


def test_binomial_two_sided_matches_a_hand_computed_case():
    # 9 of 10: two-sided p = 2 * (C(10,9) + C(10,10)) / 2^10 = 2 * 11 / 1024
    assert binomial_two_sided(9, 10) == pytest.approx(2 * 11 / 1024)


def test_binomial_two_sided_no_discordant_pairs_is_not_significant():
    assert binomial_two_sided(0, 0) == 1.0


def test_binomial_two_sided_never_exceeds_one():
    for trials in range(0, 40):
        for successes in range(trials + 1):
            assert 0.0 <= binomial_two_sided(successes, trials) <= 1.0


def test_binomial_two_sided_agrees_with_scipy_when_it_is_installed():
    scipy_stats = pytest.importorskip("scipy.stats")
    for successes, trials in [(9, 10), (13, 27), (20, 34), (1, 5), (50, 100)]:
        expected = scipy_stats.binomtest(successes, trials, 0.5).pvalue
        assert binomial_two_sided(successes, trials) == pytest.approx(expected, rel=1e-9)


# ------------------------------------------------------------------ the 2x2 table


def make_run(controller: str, answers: list[dict], budget: str = "25%") -> dict:
    return {"controller": controller, "budget": budget, "answers": answers,
            "metrics": {"controller": controller, "model": "test"}, "folder": None}


def test_paired_counts_the_four_outcomes():
    baseline = make_run("keep_newest", [answer("q1", True), answer("q2", True),
                                        answer("q3", False), answer("q4", False)])
    oracle = make_run("oracle", [answer("q1", True), answer("q2", False),
                                 answer("q3", True), answer("q4", False)])
    result = paired(baseline, oracle, lambda row: True)
    assert result["both_right"] == 1  # q1
    assert result["only_baseline_right"] == 1  # q2
    assert result["only_oracle_right"] == 1  # q3
    assert result["both_wrong"] == 1  # q4
    assert result["discordant"] == 2
    assert result["questions"] == 4


def test_paired_pairs_by_question_id_not_by_position():
    baseline = make_run("keep_newest", [answer("q1", True), answer("q2", False)])
    oracle = make_run("oracle", [answer("q2", True), answer("q1", True)])  # reversed order
    result = paired(baseline, oracle, lambda row: True)
    assert result["both_right"] == 1 and result["only_oracle_right"] == 1
    assert result["only_baseline_right"] == 0


def test_paired_ignores_questions_present_in_only_one_run():
    baseline = make_run("keep_newest", [answer("q1", True), answer("only_mine", True)])
    oracle = make_run("oracle", [answer("q1", False), answer("only_theirs", True)])
    result = paired(baseline, oracle, lambda row: True)
    assert result["questions"] == 1
    assert result["skipped"] == 2
    assert result["only_baseline_right"] == 1


def test_paired_respects_the_subset_filter():
    baseline = make_run("keep_newest", [answer("q1", True), answer("adv", True, "adversarial")])
    oracle = make_run("oracle", [answer("q1", True), answer("adv", False, "adversarial")])
    everything = paired(baseline, oracle, lambda row: True)
    no_adversarial = paired(baseline, oracle, lambda row: row["category"] != "adversarial")
    assert everything["questions"] == 2 and everything["only_baseline_right"] == 1
    assert no_adversarial["questions"] == 1 and no_adversarial["only_baseline_right"] == 0


def test_paired_returns_none_when_nothing_is_shared():
    baseline = make_run("keep_newest", [answer("a", True)])
    oracle = make_run("oracle", [answer("b", True)])
    assert paired(baseline, oracle, lambda row: True) is None


def test_paired_accuracies_match_the_marginal_counts():
    baseline = make_run("keep_newest", [answer(f"q{i}", i % 3 == 0) for i in range(30)])
    oracle = make_run("oracle", [answer(f"q{i}", i % 2 == 0) for i in range(30)])
    result = paired(baseline, oracle, lambda row: True)
    assert result["baseline_accuracy"] == sum(1 for i in range(30) if i % 3 == 0)
    assert result["oracle_accuracy"] == sum(1 for i in range(30) if i % 2 == 0)


# ------------------------------------------------------------------ the plain-English verdict


def test_verdict_calls_a_small_number_of_disagreements_not_solid():
    # The Mac's B = 25% result: the oracle won 13 and this is too few to conclude from.
    result = {"p_value": 0.5, "discordant": 9, "oracle_accuracy": 60, "baseline_accuracy": 55}
    assert "Not solid" in verdict(result)


def test_verdict_calls_a_large_p_value_not_solid():
    result = {"p_value": 0.34, "discordant": 34, "oracle_accuracy": 60, "baseline_accuracy": 55}
    assert "Not solid" in verdict(result)


def test_verdict_calls_a_small_p_value_solid_and_names_the_winner():
    result = {"p_value": 0.001, "discordant": 40, "oracle_accuracy": 70, "baseline_accuracy": 50}
    text = verdict(result)
    assert "Solid" in text and "Not solid" not in text and "oracle" in text


def test_verdict_names_the_baseline_when_the_baseline_wins():
    result = {"p_value": 0.001, "discordant": 40, "oracle_accuracy": 50, "baseline_accuracy": 70}
    assert "keep_newest" in verdict(result)


# ------------------------------------------------------------------ table rows


def test_controller_row_reports_the_memory_and_reasoning_split():
    run = make_run("keep_newest", [
        answer("q1", True),
        wrong_because("q2", "evidence_dropped"),
        wrong_because("q3", "evidence_in_store_not_retrieved"),
        wrong_because("q4", "evidence_present_model_wrong"),
    ])
    name, accuracy, f1, bleu1, tokens, memory, reasoning, questions = controller_row(run, lambda r: True)
    assert name == "keep_newest"
    assert accuracy == 0.25 and questions == 4
    assert memory == 2 and reasoning == 1
    assert tokens == 100.0


def test_controller_row_with_no_matching_questions_is_all_n_a():
    run = make_run("keep_newest", [answer("adv", True, "adversarial")])
    row = controller_row(run, lambda r: r["category"] != "adversarial")
    assert row[0] == "keep_newest" and row[-1] == 0 and row[1] is None


# ------------------------------------------------------------------ labels and ordering


def test_budget_label_reads_fractions_and_fixed_token_counts():
    assert budget_label({"fraction": 0.25}) == "25%"
    assert budget_label({"fraction": 0.1}) == "10%"
    assert budget_label({"tokens": 300}) == "300 tokens"


def test_budget_order_sorts_by_size_and_puts_unlimited_last():
    assert sorted(["50%", "10%", "25%"], key=budget_order) == ["10%", "25%", "50%"]
    assert budget_order("unlimited") == math.inf


# ------------------------------------------------------------------ end to end over a folder


def write_run_folder(parent, controller: str, fraction: float, answers: list[dict]) -> None:
    folder = parent / f"2026-01-01_{controller}_locomo_B{round(fraction * 100)}pct_seed0_abc1234"
    folder.mkdir(parents=True)
    (folder / "config.yaml").write_text(yaml.safe_dump(
        {"benchmark": {"name": "locomo"}, "budget": {"fraction": fraction},
         "model": {"name": "Qwen/Qwen3.5-4B"}, "seed": 0}))
    (folder / "metrics.json").write_text(json.dumps(
        {"controller": controller, "model": "Qwen/Qwen3.5-4B", "questions": len(answers)}))
    (folder / "answers.jsonl").write_text("".join(json.dumps(a) + "\n" for a in answers))


def test_write_comparison_covers_every_budget_and_both_readings(tmp_path):
    for fraction in (0.10, 0.25):
        write_run_folder(tmp_path, "keep_newest", fraction,
                         [answer("q1", True), answer("q2", False), answer("adv", True, "adversarial")])
        write_run_folder(tmp_path, "oracle", fraction,
                         [answer("q1", True), answer("q2", True), answer("adv", False, "adversarial")])
    text = write_comparison(tmp_path).read_text()
    assert "## Scores, all questions" in text
    assert "## Scores, without adversarial questions" in text
    assert "### Budget 10%" in text and "### Budget 25%" in text
    assert "McNemar" in text
    assert text.index("### Budget 10%") < text.index("### Budget 25%")  # sorted by size


def test_write_comparison_says_so_when_a_controller_is_missing_at_a_budget(tmp_path):
    write_run_folder(tmp_path, "keep_newest", 0.25, [answer("q1", True)])
    text = write_comparison(tmp_path).read_text()
    assert "Not available" in text and "`oracle`" in text


def test_write_comparison_keeps_only_the_newest_run_per_controller_and_budget(tmp_path):
    write_run_folder(tmp_path, "keep_newest", 0.25, [answer("q1", True)])
    runs = load_runs(tmp_path)
    assert len(runs) == 1
    # a second folder for the same controller and budget replaces the first
    folder = tmp_path / "2026-01-02_keep_newest_locomo_B25pct_seed0_def5678"
    folder.mkdir()
    (folder / "config.yaml").write_text(yaml.safe_dump(
        {"benchmark": {"name": "locomo"}, "budget": {"fraction": 0.25},
         "model": {"name": "Qwen/Qwen3.5-4B"}, "seed": 0}))
    (folder / "metrics.json").write_text(json.dumps({"controller": "keep_newest", "model": "Qwen/Qwen3.5-4B"}))
    (folder / "answers.jsonl").write_text(json.dumps(answer("q1", False)) + "\n")
    runs = load_runs(tmp_path)
    assert len(runs) == 1
    assert runs[0]["answers"][0]["correct"] is False  # the newer one won


def test_write_comparison_warns_when_4_bit_and_16_bit_runs_are_mixed(tmp_path):
    write_run_folder(tmp_path, "keep_newest", 0.25, [answer("q1", True)])
    folder = tmp_path / "2026-01-02_oracle_locomo_B25pct_seed0_def5678"
    folder.mkdir()
    (folder / "config.yaml").write_text(yaml.safe_dump(
        {"benchmark": {"name": "locomo"}, "budget": {"fraction": 0.25},
         "model": {"name": "Qwen/Qwen3.5-4B", "load_in_4bit": True}, "seed": 0}))
    (folder / "metrics.json").write_text(json.dumps({"controller": "oracle", "model": "Qwen/Qwen3.5-4B"}))
    (folder / "answers.jsonl").write_text(json.dumps(answer("q1", True)) + "\n")
    assert "not comparable" in write_comparison(tmp_path).read_text()


def test_write_comparison_refuses_an_empty_folder(tmp_path):
    with pytest.raises(SystemExit):
        write_comparison(tmp_path)


# ------------------------------------------------------------------ placement against retrieval


def with_evidence(question_id: str, correct: bool, places: list[tuple[str, bool]],
                  category: str = "single-hop") -> dict:
    """An answer row whose evidence sits in the given places.

    `places` is a list of (place, in_prompt). For a STORE item, in_prompt means it was
    returned by the search; for an ARCHIVE item, that the agent recalled it.
    """
    row = answer(question_id, correct, category)
    row["evidence"] = [
        {
            "item_id": f"D1:{i}",
            "place": place,
            "retrieved_from_store": place == "STORE" and in_prompt,
            "recalled_from_archive": place == "ARCHIVE" and in_prompt,
            "in_prompt": place == "CONTEXT" or in_prompt,
        }
        for i, (place, in_prompt) in enumerate(places)
    ]
    return row


def run_with(controller: str, answers: list[dict], store_at_end: int = 100, budget: str = "25%") -> dict:
    return {"controller": controller, "budget": budget, "answers": answers, "folder": None,
            "metrics": {"controller": controller, "model": "test",
                        "final_places": {"STORE": store_at_end}, "cost": {"recalls": 0}}}


def test_evidence_split_separates_placement_from_retrieval():
    from memctl.compare import evidence_split
    run = run_with("keep_newest", [
        with_evidence("q1", True, [("CONTEXT", True)]),               # placement
        with_evidence("q2", True, [("STORE", True)]),                 # retrieval rescued it
        with_evidence("q3", False, [("STORE", False)]),               # lost
        with_evidence("q4", True, [("CONTEXT", True), ("STORE", True)]),  # partly rescued
    ])
    split = evidence_split(run, lambda r: True)
    assert split["questions"] == 4
    assert split["all_in_context"] == 0.25          # only q1
    assert split["rescued_by_retrieval"] == 0.50    # q2 and q4
    assert split["all_in_prompt"] == 0.75           # q1, q2, q4
    assert split["store_recall"] == pytest.approx(2 / 3)   # 2 of 3 stored items found
    assert split["items_in_store_at_end"] == 100


def test_evidence_split_shares_always_add_up():
    from memctl.compare import evidence_split
    run = run_with("lru", [
        with_evidence("q1", True, [("CONTEXT", True)]),
        with_evidence("q2", True, [("STORE", True)]),
        with_evidence("q3", False, [("STORE", False)]),
    ])
    split = evidence_split(run, lambda r: True)
    assert split["all_in_context"] + split["rescued_by_retrieval"] == pytest.approx(split["all_in_prompt"])


def test_evidence_split_ignores_questions_with_no_evidence_label():
    from memctl.compare import evidence_split
    row = answer("no_label", False)
    row["evidence"] = []
    run = run_with("keep_newest", [with_evidence("q1", True, [("CONTEXT", True)]), row])
    assert evidence_split(run, lambda r: True)["questions"] == 1


def test_evidence_split_counts_archive_recalls_as_rescued():
    from memctl.compare import evidence_split
    run = run_with("archiver", [with_evidence("q1", True, [("ARCHIVE", True)])])
    split = evidence_split(run, lambda r: True)
    assert split["all_in_context"] == 0.0 and split["rescued_by_retrieval"] == 1.0
    assert split["evidence_in_archive"] == 1 and split["store_recall"] is None


def test_decomposition_attributes_the_gap_to_placement_and_retrieval():
    from memctl.compare import decomposition
    # keep_newest: nothing in context, store finds half. oracle: everything in context.
    baseline = run_with("keep_newest", [
        with_evidence("q1", False, [("STORE", True)]),
        with_evidence("q2", False, [("STORE", False)]),
    ])
    oracle = run_with("oracle", [
        with_evidence("q1", True, [("CONTEXT", True)]),
        with_evidence("q2", True, [("CONTEXT", True)]),
    ], store_at_end=3)
    text = decomposition(baseline, oracle, lambda r: True)
    assert "1.000" in text          # placement difference: 0 -> 1
    assert "-0.500" in text         # retrieval difference: 0.5 -> 0
    assert "Items left in the store: 100 against 3" in text


def test_decomposition_warns_when_the_store_explains_more_than_placement():
    from memctl.compare import decomposition
    baseline = run_with("keep_newest", [with_evidence("q1", False, [("STORE", False)]),
                                        with_evidence("q2", False, [("STORE", False)])])
    # the oracle gains entirely through retrieval, not placement
    oracle = run_with("oracle", [with_evidence("q1", True, [("STORE", True)]),
                                 with_evidence("q2", True, [("STORE", True)])], store_at_end=2)
    assert "comes from its store than from its placement" in decomposition(baseline, oracle, lambda r: True)


def test_decomposition_is_silent_when_placement_explains_more():
    from memctl.compare import decomposition
    baseline = run_with("keep_newest", [with_evidence("q1", False, [("STORE", False)])])
    oracle = run_with("oracle", [with_evidence("q1", True, [("CONTEXT", True)])])
    assert "comes from its store" not in decomposition(baseline, oracle, lambda r: True)


# ------------------------------------------------------------------ the archive note


def test_archive_note_appears_when_nothing_was_archived():
    from memctl.compare import archive_note
    runs = [run_with("keep_newest", [with_evidence("q1", True, [("STORE", True)])])]
    assert "were not exercised" in "\n".join(archive_note(runs))


def test_archive_note_is_absent_when_an_item_was_archived():
    from memctl.compare import archive_note
    runs = [run_with("archiver", [with_evidence("q1", True, [("ARCHIVE", True)])])]
    assert archive_note(runs) == []


def test_archive_note_is_absent_when_a_recall_was_made():
    from memctl.compare import archive_note
    run = run_with("keep_newest", [with_evidence("q1", True, [("STORE", True)])])
    run["metrics"]["cost"]["recalls"] = 3
    assert archive_note([run]) == []
