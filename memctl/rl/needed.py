"""A calibrated estimate of P(an item will be needed again | what the controller sees).

    python -m memctl.rl.needed --config configs/rl/d5/needed_s0.yaml

Under a priced archive the right operation for an item being removed is a Bayes
decision: archive it iff P(needed again) x (cost of losing it) > price. The
imitation policy decides *which* item to remove; this model supplies the
probability, and the RL controller applies the rule when it is configured with
`needed_model` and `archive_price` (memctl/controllers/rl.py).

Training data come from the configured policy's own episodes (on-policy, so the
estimate is P(needed | the policy chose to remove it, features)): every item the
policy removes is a row of features, labelled 1 if hindsight says it is needed
again. Hindsight labels only; it never acts. Held-out episodes report the Brier
score and a reliability table.

It also estimates `need_value`, what archiving (rather than deleting) a needed
item is worth in queries, on further training seeds: the policy is run once
archiving everything it removes and once deleting everything, and the queries
gained are divided by the needed items it removed. The controller archives iff
P(needed) x need_value > price.

Config: an ordinary experiment config with `controller: {name: rl, checkpoint: ...}`
plus

    needed:
      episodes: 200                # training episodes (seeds from seed_offset); 50 more are held out
      seed_offset: 200000
      budget_fractions: [0.02, 0.05, 0.1, 0.25]
      hidden: 32
      epochs: 200
      value_episodes: 50           # episodes for need_value, after the held-out ones
      output: runs/needed_s0       # writes needed.pt and needed.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn

from memctl.config import resolve
from memctl.harness.runner import Experiment
from memctl.hindsight.collect import NEVER
from memctl.rl.policy import REMOVAL_OPERATIONS, NeededModel
from memctl.sysinfo import collect_metadata

N_REMOVALS = len(REMOVAL_OPERATIONS)


def rows_of(decision) -> np.ndarray:
    """Model input for each ACTIVE candidate of a decision: item features plus global features."""
    n = decision.n_active
    return np.concatenate([decision.items[:n], np.repeat(decision.global_features[None, :], n, axis=0)], axis=1)


def collect(experiment: Experiment, seeds: range, fractions: list[float], rng: random.Random,
            rows: str = "removed", per_decision: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Features and hindsight labels (needed again or not). `rows`: "removed" uses the items the policy removes
    (P(needed | the policy chose to remove it)); "active" samples up to `per_decision` active items at every
    decision (P(needed | in memory)), what a rule over all kept items needs."""
    controller = experiment.controller
    controller.record, controller.greedy = True, True
    features, labels = [], []
    for seed in seeds:
        experiment.config["memory"]["budget"] = {"fraction": rng.choice(fractions)}
        hindsight = experiment.hindsight(seed)
        experiment.run_episode(seed=seed, detail=False)
        experiment._hindsight.pop(seed, None)
        for decision in controller.recorded:
            if rows == "active":
                chosen = rng.sample(range(decision.n_active), min(per_decision, decision.n_active))
            elif decision.picks:
                chosen = [pick // N_REMOVALS for pick in decision.picks]
            else:
                continue
            table = rows_of(decision)
            for i in chosen:
                need = min(hindsight.next_need(root, decision.step) for root in decision.roots[i])
                features.append(table[i])
                labels.append(float(need != NEVER))
    return np.array(features, dtype=np.float32), np.array(labels, dtype=np.float32)


def estimate_need_value(experiment: Experiment, model: NeededModel, seeds: range, fractions: list[float],
                        rng: random.Random) -> dict:
    """Queries gained per needed item removed, archiving everything against deleting everything."""
    controller = experiment.controller
    saved = (controller.needed_model, controller.archive_price, controller.record)
    gained = needed_removed = 0.0
    try:
        for seed in seeds:
            experiment.config["memory"]["budget"] = {"fraction": rng.choice(fractions)}
            hindsight = experiment.hindsight(seed)
            controller.needed_model, controller.archive_price, controller.record = model, -1.0, True  # archive all
            kept = experiment.run_episode(seed=seed, detail=False).episode
            for decision in controller.recorded:
                for pick in decision.picks:
                    i = pick // N_REMOVALS
                    needed_removed += min(hindsight.next_need(r, decision.step) for r in decision.roots[i]) != NEVER
            controller.archive_price = float("inf")  # delete all
            dropped = experiment.run_episode(seed=seed, detail=False).episode
            experiment._hindsight.pop(seed, None)
            gained += kept["correct"] - dropped["correct"]
    finally:
        controller.needed_model, controller.archive_price, controller.record = saved
    return {"value": gained / needed_removed if needed_removed else 1.0, "queries_gained": gained,
            "needed_items_removed": needed_removed, "episodes": len(seeds)}


def reliability(probabilities: np.ndarray, labels: np.ndarray, bins=(0, 0.01, 0.03, 0.1, 0.3, 1.0001)) -> list[dict]:
    table = []
    for low, high in zip(bins[:-1], bins[1:]):
        inside = (probabilities >= low) & (probabilities < high)
        if inside.any():
            table.append({"bin": [low, min(high, 1.0)], "count": int(inside.sum()),
                          "predicted": float(probabilities[inside].mean()), "observed": float(labels[inside].mean())})
    return table


def fit(config: dict) -> Path:
    config = resolve(config)
    settings = {"episodes": 200, "held_out": 50, "seed_offset": 200_000, "budget_fractions": [0.02, 0.05, 0.1, 0.25],
                "hidden": 32, "epochs": 200, "lr": 0.01, "value_episodes": 50, **config.get("needed", {})}
    folder = Path(settings.get("output") or Path(config["logging"]["output_dir"]) / config["name"])
    folder.mkdir(parents=True, exist_ok=True)
    rng = random.Random(int(config["seed"]))
    torch.manual_seed(int(config["seed"]))
    experiment = Experiment(config)
    offset = int(settings["seed_offset"]) + 1000 * int(config["seed"])
    rows, per = settings.get("rows", "removed"), int(settings.get("per_decision", 8))
    train_x, train_y = collect(experiment, range(offset, offset + int(settings["episodes"])), settings["budget_fractions"],
                               rng, rows, per)
    held = range(offset + int(settings["episodes"]), offset + int(settings["episodes"]) + int(settings["held_out"]))
    test_x, test_y = collect(experiment, held, settings["budget_fractions"], rng, rows, per)

    model = NeededModel(train_x.shape[1], int(settings["hidden"]))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(settings["lr"]), weight_decay=1e-4)
    x, y = torch.from_numpy(train_x), torch.from_numpy(train_y)
    for _ in range(int(settings["epochs"])):  # full batch: a few thousand rows
        loss = nn.functional.binary_cross_entropy_with_logits(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(torch.from_numpy(test_x))).numpy()
    if int(settings["value_episodes"]) > 0:
        need_value = estimate_need_value(
            experiment, model, range(held.stop, held.stop + int(settings["value_episodes"])), settings["budget_fractions"], rng,
        )
    else:  # fixed in advance (needed.need_value), e.g. from a measured accuracy difference
        need_value = {"value": float(settings.get("need_value", 1.0)), "fixed": True}
    report = {
        "need_value": need_value,
        **collect_metadata(), "policy": config["controller"].get("checkpoint"),
        "train_rows": len(train_y), "train_base_rate": float(train_y.mean()),
        "test_rows": len(test_y), "test_base_rate": float(test_y.mean()),
        "test_brier": float(((probabilities - test_y) ** 2).mean()),
        "test_brier_base_rate": float(((train_y.mean() - test_y) ** 2).mean()),
        "reliability": reliability(probabilities, test_y), "settings": settings,
    }
    torch.save({"state_dict": model.state_dict(), "config": model.config}, folder / "needed.pt")
    (folder / "needed.json").write_text(json.dumps(report, indent=2))
    return folder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    folder = fit(yaml.safe_load(Path(args.config).read_text()))
    report = json.loads((folder / "needed.json").read_text())
    print(json.dumps({k: report[k] for k in ("need_value", "train_rows", "test_base_rate", "test_brier", "test_brier_base_rate", "reliability")}, indent=2))


if __name__ == "__main__":
    main()
