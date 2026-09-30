"""Run a grid of experiments:  python -m memctl.sweep --config configs/sweeps/x.yaml

A sweep file has a `base` config and a `grid` of dotted keys, each with a list
of values. Every combination becomes one run in `runs/<name>/<cell>/`:

    name: exp1_delete_only
    base: {episodes: 100, env: {name: synthetic_recall}}
    grid:
      controller: [fifo, lru, {name: oracle, method: exact}]
      memory.budget.fraction: [0.1, 0.25, 0.5, 1.0]
      env.horizon: [200, 500, 1000]

Every cell uses the same seeds, so cells can be compared episode by episode.
Cell folders have stable names, so running the sweep again resumes it.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import multiprocessing
import os
import time
from pathlib import Path

import yaml

from memctl.config import resolve
from memctl.harness.runner import run_experiment
from memctl.util import set_dotted


def _value_label(key: str, value) -> str:
    if isinstance(value, dict):
        if value.get("label"):
            return str(value["label"])
        extras = "-".join(str(v) for k, v in value.items() if k != "name" and not isinstance(v, (dict, list)))
        return f"{value.get('name', key)}{'-' + extras if extras else ''}"
    short = key.split(".")[-1]
    if key == "controller":
        return str(value)
    if isinstance(value, float):
        return f"{short}{value:g}"
    return f"{short}{value}"


def cells(sweep: dict) -> list[tuple[str, dict]]:
    """(cell label, resolved config) for every combination in the grid."""
    grid = sweep.get("grid", {})
    keys = list(grid)
    found = []
    for values in itertools.product(*(grid[key] for key in keys)):
        config = copy.deepcopy(sweep.get("base", {}))
        labels = []
        for key, value in zip(keys, values):
            if key == "controller" and isinstance(value, str):
                value = {"name": value}
            set_dotted(config, key, copy.deepcopy(value))
            labels.append(_value_label(key, value).replace("/", "-").replace(" ", ""))  # a label is a folder name
        config["name"] = sweep["name"]
        found.append(("__".join(labels), resolve(config)))
    labels = [label for label, _ in found]
    if len(set(labels)) != len(labels):
        raise ValueError("two grid cells have the same label; give the controllers a `label`")
    return found


def _run_cell(job: tuple[str, dict, str]) -> tuple[str, float, str]:
    label, config, folder = job
    started = time.time()
    try:
        run_experiment(config, folder)
        return label, time.time() - started, "ok"
    except Exception as error:  # one failed cell must not stop the grid
        return label, time.time() - started, f"FAILED: {type(error).__name__}: {error}"


def run_sweep(sweep: dict, workers: int | None = None, output_dir: str = "runs") -> Path:
    root = Path(output_dir) / sweep["name"]
    root.mkdir(parents=True, exist_ok=True)
    (root / "sweep.yaml").write_text(yaml.safe_dump(sweep, sort_keys=False))
    jobs = [(label, config, str(root / label)) for label, config in cells(sweep)]
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    print(f"{len(jobs)} cells, {workers} workers -> {root}", flush=True)
    failures = []
    with multiprocessing.get_context("spawn").Pool(workers) as pool:
        for number, (label, seconds, status) in enumerate(pool.imap_unordered(_run_cell, jobs), start=1):
            print(f"[{number}/{len(jobs)}] {label}  {seconds:.0f}s  {status}", flush=True)
            if status != "ok":
                failures.append((label, status))
    if failures:
        print(f"{len(failures)} cells failed:")
        for label, status in failures:
            print(f"  {label}: {status}")
    return root


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a grid of experiments.")
    parser.add_argument("--config", required=True, help="path to a sweep YAML file")
    parser.add_argument("--workers", type=int, help="parallel processes (default: cores - 2)")
    parser.add_argument("--episodes", type=int, help="override base.episodes")
    parser.add_argument("--output-dir", default="runs")
    args = parser.parse_args()
    sweep = yaml.safe_load(Path(args.config).read_text())
    if args.episodes:
        sweep.setdefault("base", {})["episodes"] = args.episodes
    print(f"sweep folder: {run_sweep(sweep, args.workers, args.output_dir)}")


if __name__ == "__main__":
    main()
