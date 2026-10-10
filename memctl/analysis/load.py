"""Read run folders back. Analysis depends only on the files a run writes."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml


from memctl.runlog import read_jsonl, trace_exists  # noqa: E402,F401  (gz-aware; old per-step logs are compressed)


@dataclass
class Run:
    folder: Path
    config: dict
    summary: dict
    episodes: list[dict]
    metadata: dict = field(default_factory=dict)

    @property
    def controller(self) -> str:
        return self.summary.get("controller") or self.config["controller"].get("label") or self.config["controller"]["name"]

    @property
    def fraction(self) -> float | None:
        return self.config["memory"]["budget"].get("fraction")

    def by_seed(self, metric: str = "task_success") -> dict[int, float]:
        return {episode["seed"]: episode[metric] for episode in self.episodes if episode.get(metric) is not None}

    def rows(self, name: str) -> list[dict]:
        """Rows of one of the run's jsonl files, e.g. rows("failures")."""
        return read_jsonl(self.folder / f"{name}.jsonl")


def load_runs(root: str | Path) -> list[Run]:
    """Every finished or partly finished run under `root`."""
    runs = []
    for episodes_file in sorted(Path(root).rglob("episodes.jsonl")):
        folder = episodes_file.parent
        episodes = read_jsonl(episodes_file)
        if not episodes:
            continue
        config = yaml.safe_load((folder / "config.yaml").read_text())
        summary_file, metadata_file = folder / "summary.json", folder / "metadata.json"
        summary = json.loads(summary_file.read_text()) if summary_file.exists() else {}
        metadata = json.loads(metadata_file.read_text()) if metadata_file.exists() else {}
        runs.append(Run(folder, config, summary, episodes, metadata))
    return runs


def flatten(config: dict, prefix: str = "") -> dict:
    flat = {}
    for key, value in config.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict) and value.get("label"):
            flat[name] = value["label"]  # a labelled component reads as its label, not as its parts
        elif isinstance(value, dict):
            flat.update(flatten(value, name + "."))
        else:
            flat[name] = json.dumps(value) if isinstance(value, list) else value
    return flat


def condition_keys(runs: list[Run]) -> list[str]:
    """Config keys that vary across the runs, other than the controller and the budget."""
    flats = [flatten(run.config) for run in runs]
    keys = sorted({key for flat in flats for key in flat})
    ignore = ("controller.", "memory.budget.", "logging.", "shadow_controllers")
    varying = []
    for key in keys:
        if key == "controller" or key.startswith(ignore):
            continue
        if len({json.dumps(flat.get(key), sort_keys=True, default=str) for flat in flats}) > 1:
            varying.append(key)
    return varying


def condition_of(run: Run, keys: list[str]) -> tuple:
    flat = flatten(run.config)
    return tuple((key, flat.get(key)) for key in keys)


def condition_label(condition: tuple) -> str:
    if not condition:
        return "all runs"
    return ", ".join(f"{key.split('.')[-1]}={value}" for key, value in condition)


def condition_sort_key(condition: tuple) -> tuple:
    """Orders conditions by their values, numbers as numbers (horizon 200 before 1000)."""
    key = []
    for _, value in condition:
        number = isinstance(value, (int, float)) and not isinstance(value, bool)
        key.append((0, float(value), "") if number else (1, 0.0, str(value)))
    return tuple(key)
