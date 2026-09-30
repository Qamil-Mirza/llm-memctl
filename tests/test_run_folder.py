"""Run folders: what is written, that a run can be resumed, and that sweeps are paired."""

import json

import pytest
import yaml

from memctl.config import ConfigError, resolve
from memctl.harness import runner
from memctl.runlog import read_jsonl
from memctl.sweep import cells, run_sweep

BASE = {
    "name": "toy",
    "episodes": 4,
    "env": {"name": "synthetic_recall", "horizon": 80},
    "controller": {"name": "lru"},
    "memory": {"budget": {"fraction": 0.2}},
    "logging": {"detail_episodes": 2},
}


def test_a_run_writes_every_artifact(tmp_path):
    folder = runner.run_experiment(resolve(BASE), tmp_path / "run")
    for name in ("config.yaml", "metadata.json", "summary.json", "episodes.jsonl", "steps.jsonl",
                 "memory_actions.jsonl", "rewards.jsonl", "items.jsonl", "failures.jsonl"):
        assert (folder / name).exists(), name
    assert (folder / "checkpoints").is_dir() and (folder / "plots").is_dir()

    saved = yaml.safe_load((folder / "config.yaml").read_text())
    assert saved["memory"]["allowed_operations"] == ["KEEP", "EVICT", "NO_OP"] and saved["schema_version"] == 1
    metadata = json.loads((folder / "metadata.json").read_text())
    assert metadata["status"] == "completed" and "commit" in metadata["git"]
    assert metadata["cpu"]["logical_cores"] and "numpy" in metadata["packages"]
    assert metadata["models"]["task_model"] == "scripted_reader"

    episodes = read_jsonl(folder / "episodes.jsonl")
    assert [e["episode_id"] for e in episodes] == ["ep00000", "ep00001", "ep00002", "ep00003"]
    assert [e["seed"] for e in episodes] == [0, 1, 2, 3]
    summary = json.loads((folder / "summary.json").read_text())
    assert summary["episodes"] == 4 and 0 <= summary["task_success"] <= 1
    assert {row["episode_id"] for row in read_jsonl(folder / "steps.jsonl")} == {"ep00000", "ep00001"}
    step = read_jsonl(folder / "steps.jsonl")[0]
    assert step["experiment_id"] == folder.name and step["seed"] == 0


def test_deleted_items_stay_in_the_logs(tmp_path):
    folder = runner.run_experiment(resolve(BASE), tmp_path / "run")
    items = read_jsonl(folder / "items.jsonl")
    deleted = [item for item in items if item["tier"] == "DELETED"]
    assert deleted and all(item["content"] for item in deleted)


def test_an_interrupted_run_resumes_without_repeating_or_duplicating(tmp_path, monkeypatch):
    original = runner.Experiment.run_episode
    calls = []

    def crash_on_third(self, index=0, **kwargs):
        calls.append(index)
        if index == 2 and len(calls) == 3:
            raise KeyboardInterrupt
        return original(self, index, **kwargs)

    monkeypatch.setattr(runner.Experiment, "run_episode", crash_on_third)
    with pytest.raises(KeyboardInterrupt):
        runner.run_experiment(resolve(BASE), tmp_path / "run")
    folder = tmp_path / "run"
    assert len(read_jsonl(folder / "episodes.jsonl")) == 2
    assert json.loads((folder / "metadata.json").read_text())["status"] == "failed"

    runner.run_experiment(resolve(BASE), folder)
    assert calls == [0, 1, 2, 2, 3]
    resumed = read_jsonl(folder / "episodes.jsonl")
    assert [e["episode_id"] for e in resumed] == ["ep00000", "ep00001", "ep00002", "ep00003"]

    monkeypatch.setattr(runner.Experiment, "run_episode", original)
    fresh = runner.run_experiment(resolve(BASE), tmp_path / "fresh")
    assert [e["task_success"] for e in resumed] == [e["task_success"] for e in read_jsonl(fresh / "episodes.jsonl")]
    assert json.loads((folder / "metadata.json").read_text())["resumed"] is True


def test_a_folder_cannot_be_reused_for_a_different_config(tmp_path):
    runner.run_experiment(resolve(BASE), tmp_path / "run")
    other = resolve({**BASE, "controller": {"name": "fifo"}})
    with pytest.raises(ValueError, match="different config"):
        runner.run_experiment(other, tmp_path / "run")


def test_bad_configs_are_refused():
    with pytest.raises(ConfigError):
        resolve({"memory": {"allowed_operations": ["TELEPORT"]}})
    with pytest.raises(ConfigError):
        resolve({"memory": {"budget": {"fraction": 1.5}}})
    with pytest.raises(ConfigError):
        resolve({"schema_version": 99})
    assert resolve({"memory": {"budget": {"tokens": 500}}})["memory"]["budget"] == {"tokens": 500}


def test_sweep_cells_share_seeds_and_get_stable_names(tmp_path):
    sweep = {
        "name": "toy_sweep",
        "base": {**BASE, "episodes": 2},
        "grid": {"controller": ["fifo", {"name": "oracle", "method": "exact"}], "memory.budget.fraction": [0.1, 1.0]},
    }
    labels = [label for label, _ in cells(sweep)]
    assert labels == ["fifo__fraction0.1", "fifo__fraction1", "oracle-exact__fraction0.1", "oracle-exact__fraction1"]
    root = run_sweep(sweep, workers=2, output_dir=str(tmp_path))
    seeds = {label: [e["seed"] for e in read_jsonl(root / label / "episodes.jsonl")] for label in labels}
    assert all(value == [0, 1] for value in seeds.values())
    histories = {label: [e["history_tokens"] for e in read_jsonl(root / label / "episodes.jsonl")] for label in labels}
    assert histories["fifo__fraction0.1"] == histories["oracle-exact__fraction0.1"]
