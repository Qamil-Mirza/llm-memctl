"""Run one experiment:  python -m memctl.run --config configs/x.yaml

Resume an interrupted one:  python -m memctl.run --resume runs/<experiment_id>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from memctl.config import load_config, resolve
from memctl.harness.runner import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one memory-controller experiment.")
    parser.add_argument("--config", help="path to a YAML config")
    parser.add_argument("--resume", help="a run folder to continue; its saved config.yaml is used")
    parser.add_argument("--episodes", type=int, help="override the number of episodes")
    parser.add_argument("--controller", help="override the controller name")
    parser.add_argument("--budget-fraction", type=float, help="override the budget, e.g. 0.25")
    parser.add_argument("--seed", type=int, help="override the base seed")
    args = parser.parse_args()
    if bool(args.config) == bool(args.resume):
        parser.error("give exactly one of --config and --resume")

    folder = None
    if args.resume:
        folder = Path(args.resume)
        config = resolve(yaml.safe_load((folder / "config.yaml").read_text()))
    else:
        config = load_config(args.config)
    if args.episodes:
        config["episodes"] = args.episodes
    if args.controller:
        config["controller"] = {"name": args.controller}
    if args.budget_fraction:
        config["memory"]["budget"] = {"fraction": args.budget_fraction}
    if args.seed is not None:
        config["seed"] = args.seed
    folder = run_experiment(config, folder, progress=True)
    print(f"run folder: {folder}")


if __name__ == "__main__":
    main()
