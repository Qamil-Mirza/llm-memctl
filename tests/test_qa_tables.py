"""Per-category QA tables rebuilt from run logs."""

import json

import yaml

from memctl.analysis.qa_tables import cell_rows, paired_difference, summarise, table


def _cell(root, name, data_path, wrong, seeds=(0, 1, 2, 3)):
    cell = root / name
    cell.mkdir(parents=True)
    (cell / "config.yaml").write_text(yaml.safe_dump({"env": {"name": "longmemeval", "path": str(data_path)}}))
    with open(cell / "episodes.jsonl", "w") as handle:
        for seed in seeds:
            handle.write(json.dumps({"seed": seed, "queries": 1, "tokens_processed": 100 + seed, "task_model_calls": 1}) + "\n")
    with open(cell / "failures.jsonl", "w") as handle:
        for seed in wrong:
            handle.write(json.dumps({"seed": seed, "query_id": "q0000"}) + "\n")
    with open(cell / "steps.jsonl", "w") as handle:
        handle.write(json.dumps({"seed": 0, "observation_id": "q0000", "scored": True, "correct": 0 not in wrong,
                                 "agent_action": "blue", "active_tokens_at_read": 7}) + "\n")
    return cell


def test_tables_separate_abstention_and_pair_by_question(tmp_path):
    data = [
        {"question_id": "a", "question_type": "temporal-reasoning", "answer": "blue"},
        {"question_id": "b", "question_type": "temporal-reasoning", "answer": "red"},
        {"question_id": "c", "question_type": "knowledge-update", "answer": "green"},
        {"question_id": "d_abs", "question_type": "knowledge-update", "answer": "none"},
    ]
    path = tmp_path / "lme.json"
    path.write_text(json.dumps(data))
    sweep = tmp_path / "sweep"
    good = _cell(sweep, "good__fraction0.1", path, wrong=[2])
    _cell(sweep, "bad__fraction0.1", path, wrong=[0, 1, 2])

    rows = cell_rows(good)
    assert [r["category"] for r in rows] == ["temporal-reasoning", "temporal-reasoning", "knowledge-update", "abstention"]
    assert rows[0]["f1"] == 1.0 and rows[0]["tokens"] == 100 and rows[1]["f1"] is None and rows[1]["tokens"] == 101
    summary = summarise(rows)
    assert summary["n"] == 3 and abs(summary["accuracy"] - 2 / 3) < 1e-9 and summary["refusal_n"] == 1
    assert summary["by_category"] == {"knowledge-update": 0.0, "temporal-reasoning": 1.0}

    difference = paired_difference(rows[:3], cell_rows(sweep / "bad__fraction0.1")[:3])
    assert abs(difference["mean"] - 2 / 3) < 1e-9 and difference["ci"][0] >= 0
    text = table([sweep], per_category=True, baseline="bad")
    assert "good" in text and "+0.667" in text


def test_the_evidence_gate_pairs_with_fifo_and_names_paid_cells(tmp_path):
    from memctl.analysis.gate import gate, main  # noqa: F401
    from memctl.analysis.gate import split_label

    for fold in range(2):
        sweep = tmp_path / f"exp13_gate_f{fold}"
        for label, value in (("fifo_top5__fraction0.02", 0.5), ("compose_floor5__fraction0.02", 0.9),
                             ("learned_floor5__fraction0.02", 0.5), ("fifo_top5_t2000__fraction0.01", 0.4),
                             ("compose_floor5_t2000__fraction0.01", 0.8)):
            cell = sweep / label
            cell.mkdir(parents=True)
            with open(cell / "episodes.jsonl", "w") as handle:
                for seed in range(20):
                    handle.write(json.dumps({"seed": seed, "needed_hit_rate": value + (0.05 if seed % 2 else 0)}) + "\n")
    rows = {(r["controller"], r["target"]): r for r in gate(sorted(tmp_path.iterdir()))}
    assert rows[("compose_floor5", "fill")]["go"] and not rows[("learned_floor5", "fill")]["go"]
    assert rows[("compose_floor5", "fill")]["n"] == 40 and abs(rows[("compose_floor5", "fill")]["difference"] - 0.4) < 1e-9
    assert split_label("compose_floor5_t2000__fraction0.01") == ("compose_floor5", "t2000", "0.01")
