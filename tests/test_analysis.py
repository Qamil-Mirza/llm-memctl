"""Analysis: statistics, the report and the figures, on a tiny sweep."""

import json

import pytest

from memctl.analysis.load import load_runs
from memctl.analysis.plots import plot_all, plot_training, styles
from memctl.analysis.report import build_report
from memctl.analysis.stats import minimum_detectable_effect, oracle_gap_closed, paired_bootstrap, paired_differences
from memctl.doctor import check, render
from memctl.sweep import run_sweep


def test_paired_statistics():
    a, b = {0: 0.9, 1: 0.8, 2: 0.7, 3: 1.0}, {0: 0.5, 1: 0.6, 2: 0.4, 3: 0.9, 9: 0.0}
    differences = paired_differences(a, b)
    assert differences == pytest.approx([0.4, 0.2, 0.3, 0.1])  # seed 9 has no partner
    result = paired_bootstrap(differences)
    assert result["n"] == 4 and result["ci_low"] > 0 and result["mean"] == pytest.approx(0.25)
    assert paired_bootstrap([0.0, 0.0, 0.0])["p"] == 1.0
    assert 0 < minimum_detectable_effect(differences) < 1
    assert minimum_detectable_effect([0.1] * 50) == 0.0
    assert oracle_gap_closed(0.75, 0.5, 1.0) == 0.5 and oracle_gap_closed(0.5, 0.5, 0.5) is None


def test_a_controller_keeps_its_colour_and_nothing_gets_a_ninth_hue():
    first = styles(["fifo", "lru", "oracle_exact"])
    second = styles(["salience", "lru", "fifo", "random"])
    assert first["lru"] == second["lru"] and first["fifo"] == second["fifo"]
    many = styles([f"controller_{n}" for n in range(11)])
    colours = [style["color"] for style in many.values() if not style.get("other")]
    assert len(colours) == 8 and len(set(colours)) == 8
    assert sum(1 for style in many.values() if style.get("other")) == 3
    assert first["oracle_exact"].get("reference")


@pytest.fixture(scope="module")
def sweep(tmp_path_factory):
    root = tmp_path_factory.mktemp("sweep")
    spec = {
        "name": "tiny",
        "base": {
            "episodes": 6, "env": {"name": "synthetic_recall", "horizon": 120},
            "memory": {"allowed_operations": ["KEEP", "EVICT", "MOVE_TO_ARCHIVE", "RETRIEVE_FROM_ARCHIVE", "NO_OP"]},
            "logging": {"detail_episodes": 2},
        },
        "grid": {
            "controller": [
                "no_controller", "fifo", "salience",
                {"name": "lru", "label": "lru_archive", "removal": ["MOVE_TO_ARCHIVE"], "retrieve": {"top_k": 2}},
                {"name": "oracle", "method": "exact"},
            ],
            "memory.budget.fraction": [0.1, 0.5],
        },
    }
    return run_sweep(spec, workers=2, output_dir=str(root))


def test_the_report_compares_controllers_at_equal_budget(sweep):
    text, data = build_report(sweep)
    (condition,) = data["conditions"]
    assert condition["budgets"] == ["10%", "50%"]
    cell = condition["cells"]["10%"]
    assert cell["fifo"]["vs_baseline"]["baseline"] == "no_controller" and cell["fifo"]["vs_baseline"]["mean"] == 0.0
    assert cell["salience"]["oracle"] == "oracle_exact" and 0 < cell["salience"]["oracle_gap_closed"] <= 1
    assert cell["salience"]["regret_vs_oracle"] >= 0 and cell["salience"]["vs_baseline"]["n"] == 6
    for heading in ("Task success by memory budget", "Fraction of the oracle gap closed", "Failure attribution",
                    "Items acted on per episode", "Retrieval", "Needed memories destroyed"):
        assert heading in text, heading
    assert len(load_runs(sweep)) == 10


def test_every_figure_is_written(sweep):
    written = {path.name for path in plot_all(sweep)}
    assert {"success_vs_budget.png", "success_vs_latency.png", "oracle_gap_closed.png", "memory_occupancy.png",
            "actions_over_time.png", "failure_attribution.png", "regret_distribution.png", "retrieval_hit_rate.png"} <= written
    assert json.loads((sweep / "plots" / "index.json").read_text())


def test_the_learning_curve_is_drawn_from_a_training_log(tmp_path):
    rows = [
        {"phase": 0, "algorithm": "bc", "iteration": 0, "episodes_total": 10, "train_success": 0.5},
        {"phase": 1, "algorithm": "ppo", "iteration": 0, "episodes_total": 20, "train_success": 0.6, "success@0.1": 0.55},
    ]
    (tmp_path / "train_log.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert plot_training(tmp_path).exists()


def test_the_doctor_reports_a_status_for_every_gated_component(monkeypatch):
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    report = check(base_url="http://127.0.0.1:9/v1")  # nothing listens there
    statuses = {row["component"]: row["status"] for row in report["components"]}
    assert statuses["JEV controller (real model)"] == "BLOCKED"
    assert statuses["exact oracle (integer program)"] == "READY"
    assert statuses["task model / prompted controller via an OpenAI-compatible endpoint"] == "BLOCKED"
    assert all(row["to_unblock"] for row in report["components"] if row["status"] == "BLOCKED")
    assert "JEV_API_KEY" in render(report)
