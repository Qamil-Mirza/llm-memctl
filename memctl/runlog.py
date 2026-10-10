"""The run folder: structured artifacts for one experiment.

    runs/<experiment_id>/
        config.yaml            the resolved config
        metadata.json          git commit, machine, packages, model ids, status
        summary.json           aggregate metrics (written at the end)
        episodes.jsonl         one row per finished episode  <- the resume marker
        steps.jsonl            one row per timestep (detail episodes only)
        memory_actions.jsonl   one row per action, controller or forced (detail episodes only)
        rewards.jsonl          one row per timestep, every reward term (detail episodes only)
        retrieval_events.jsonl one row per retrieval (detail episodes only)
        items.jsonl            every item ever created, deleted ones included (detail episodes only)
        failures.jsonl         one row per failed query, with its attribution
        regret.jsonl           one row per evidence requirement a decision destroyed
        checkpoints/  plots/

An episode's rows are written only when the episode has finished, and its
`episodes.jsonl` row is written last. Resuming therefore means: skip the
episode ids already in `episodes.jsonl`.
"""

from __future__ import annotations

import gzip
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from memctl.sysinfo import collect_metadata

DETAIL_FILES = {
    "steps": "steps.jsonl",
    "actions": "memory_actions.jsonl",
    "rewards": "rewards.jsonl",
    "retrievals": "retrieval_events.jsonl",
    "items": "items.jsonl",
    "failures": "failures.jsonl",
    "regrets": "regret.jsonl",
}


def trace_path(path: Path) -> Path:
    """`path`, or its gzipped copy `path.gz` when only that exists (old per-step logs are kept compressed)."""
    path = Path(path)
    gz = path.with_name(path.name + ".gz")
    return gz if not path.exists() and gz.exists() else path


def trace_exists(path: Path) -> bool:
    return trace_path(path).exists()


def open_trace(path: Path):
    """Open a run file for reading as text, transparently from `path.gz` when the plain file was compressed."""
    path = trace_path(path)
    return gzip.open(path, "rt") if path.suffix == ".gz" else open(path)


def read_jsonl(path: Path) -> list[dict]:
    if not trace_exists(path):
        return []
    with open_trace(path) as file:
        return [json.loads(line) for line in file if line.strip()]


def _default(value):
    """JSON for the few non-JSON values that reach the logs (enums, numpy numbers, tuples)."""
    if hasattr(value, "value"):
        return value.value
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def append_jsonl(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("a") as file:
        for row in rows:
            file.write(json.dumps(row, default=_default) + "\n")


class RunLogger:
    def __init__(self, folder: str | Path, config: dict, models: dict | None = None) -> None:
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        (self.folder / "checkpoints").mkdir(exist_ok=True)
        (self.folder / "plots").mkdir(exist_ok=True)
        config_path = self.folder / "config.yaml"
        self.resumed = (self.folder / "episodes.jsonl").exists()
        if self.resumed and config_path.exists():
            saved = yaml.safe_load(config_path.read_text())
            if _without_logging(saved) != _without_logging(config):
                raise ValueError(
                    f"{self.folder} already holds results from a different config. "
                    "Use a new experiment id, or delete the folder to start again."
                )
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        previous = {}
        if self.resumed and (self.folder / "metadata.json").exists():
            previous = json.loads((self.folder / "metadata.json").read_text())
        self.metadata = {
            **collect_metadata(), "models": models or {}, "status": "running", "resumed": self.resumed,
            "experiment_id": self.folder.name,
        }
        # A resumed folder holds episodes from every session that wrote to it: keep each one's git state.
        history = previous.get("git_history") or ([previous["git"]] if previous.get("git") else [])
        self.metadata["git_history"] = history + [self.metadata.get("git")]
        self._write_metadata()
        if self.resumed:
            self._drop_unfinished_rows()

    def _write_metadata(self) -> None:
        self.metadata["peak_rss_mb"] = peak_rss_mb()  # so a job queue can budget memory (2026-10-07 OOM incident)
        (self.folder / "metadata.json").write_text(json.dumps(self.metadata, indent=2, default=_default))

    def completed(self) -> set[str]:
        return {row["episode_id"] for row in read_jsonl(self.folder / "episodes.jsonl")}

    def _drop_unfinished_rows(self) -> None:
        """A crash may have left rows of an episode that never got its episodes.jsonl row."""
        finished = self.completed()
        for name in DETAIL_FILES.values():
            path = self.folder / name
            rows = read_jsonl(path)
            kept = [row for row in rows if row.get("episode_id") in finished]
            if len(kept) != len(rows):
                path.write_text("".join(json.dumps(row, default=_default) + "\n" for row in kept))

    def write_episode(self, result) -> None:
        for attribute, name in DETAIL_FILES.items():
            append_jsonl(self.folder / name, getattr(result, attribute))
        append_jsonl(self.folder / "episodes.jsonl", [result.episode])

    def finalize(self, summary: dict, status: str = "completed") -> None:
        (self.folder / "summary.json").write_text(json.dumps(summary, indent=2, default=_default))
        self.metadata["status"] = status
        self.metadata["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self._write_metadata()


# Keys that say where a model is served, not what the experiment is. A cell resumed on a new pod (a new URL) is
# the same experiment; before Experiment 20 such a resume was refused (EXPERIMENTS.md §20).
ENDPOINT_KEYS = ("base_url", "timeout_s")


def _without_endpoints(value):
    if isinstance(value, dict):
        return {k: _without_endpoints(v) for k, v in value.items() if k not in ENDPOINT_KEYS}
    if isinstance(value, list):
        return [_without_endpoints(v) for v in value]
    return value


def _without_logging(config: dict) -> dict:
    """The config's identity for resuming: without logging, the episode count and serving endpoints."""
    return _without_endpoints({key: value for key, value in (config or {}).items() if key not in ("logging", "episodes")})


def peak_rss_mb() -> float:
    """This process's peak resident memory so far, in MB (Linux reports ru_maxrss in KB)."""
    import resource

    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
