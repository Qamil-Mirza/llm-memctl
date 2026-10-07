"""Calibration of a needed model on held-out questions (Experiment 14).

For a fold's test part: plays the configured policy with no reader, samples active items at every decision, and
compares the needed model's P(needed again) with hindsight. Reports AUC and a reliability table binned around
the keep thresholds of the token-price rule, P* = price x tokens / need_value.

    python -m memctl.rl.calibration --config configs/rl/lme_n1/needed_f0.yaml --model runs/lme_needed_f0/needed.pt \\
        --episodes 100 --out runs/exp14_calibration_f0.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from memctl.config import resolve
from memctl.harness.runner import Experiment
from memctl.rl.needed import collect, reliability
from memctl.rl.policy import NeededModel
from memctl.sysinfo import collect_metadata

BINS = (0, 0.005, 0.014, 0.045, 0.14, 1.0001)


def auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    """Probability a random positive outranks a random negative (ties count half)."""
    positives, negatives = scores[labels == 1], scores[labels == 0]
    if not len(positives) or not len(negatives):
        return None
    order = np.argsort(np.concatenate([positives, negatives]), kind="mergesort")
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    values = np.concatenate([positives, negatives])
    for value in np.unique(values):  # average ranks over ties
        tied = values == value
        ranks[tied] = ranks[tied].mean()
    return float((ranks[: len(positives)].sum() - len(positives) * (len(positives) + 1) / 2) / (len(positives) * len(negatives)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    config["env"]["folds"]["part"] = "test"
    settings = config.get("needed", {})
    experiment = Experiment(resolve(config))
    rng = random.Random(0)
    x, y = collect(experiment, range(args.episodes), settings.get("budget_fractions", [0.02, 0.05]), rng, "active",
                   int(settings.get("per_decision", 2)))
    model = NeededModel.load(args.model)
    with torch.no_grad():
        p = torch.sigmoid(model(torch.from_numpy(x))).numpy()
    report = {"rows": len(y), "base_rate": float(y.mean()), "auc": auc(p, y), "brier": float(((p - y) ** 2).mean()),
              "reliability": reliability(p, y, BINS), "metadata": collect_metadata(), "config": args.config, "model": args.model}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("rows", "base_rate", "auc", "brier", "reliability")}, indent=2))


if __name__ == "__main__":
    main()
