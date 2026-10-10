"""Train the Experiment 19 k-chooser on out-of-sample head outputs (EXPERIMENTS.md §19, pre-registered).

Reads the steps of `k_mode: oracle` runs, where a head ranks lists it was not trained on (head A on half B,
head B on half A). Each question gives the chooser's features and the all-found labels at k = 5, 8 and 16.
Fits one KChooser by binary cross-entropy, full batch, with fixed settings, and saves it. With `--kind mass`
(§19b) it fits a MassChooser instead (two parameters per k on the softmax mass on the top k) and reports the
expected calibration error per k on the training rows, which §19b gates at 0.10.

    python -m memctl.rl.train_k_chooser runs/exp19_xfit_f0/head_a_on_b runs/exp19_xfit_f0/head_b_on_a \\
        --out runs/exp19_chooser_f0/k_chooser.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from memctl.rl.policy import KChooser, MassChooser
from memctl.runlog import open_trace

EPOCHS, LR, WEIGHT_DECAY = 400, 0.01, 1e-4


def rows(cells: list[Path], field: str = "k_features") -> tuple[torch.Tensor, torch.Tensor]:
    features, labels = [], []
    for cell in cells:
        for line in open_trace(cell / "steps.jsonl"):
            info = json.loads(line).get("controller_info") or {}
            if "k_labels" in info:
                features.append(info[field])
                labels.append([float(x) for x in info["k_labels"]])
    return torch.tensor(features), torch.tensor(labels)


def ece(predicted: list[float], observed: list[float], bins: int = 5) -> float:
    """Expected calibration error: |mean predicted - mean observed| per equal-width bin, weighted by count."""
    total = 0.0
    for b in range(bins):
        members = [(p, y) for p, y in zip(predicted, observed) if b / bins <= p < (b + 1) / bins or (b == bins - 1 and p == 1.0)]
        if members:
            total += len(members) * abs(sum(p for p, _ in members) - sum(y for _, y in members)) / len(members)
    return total / len(predicted)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cells", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--kind", choices=["mlp", "mass"], default="mlp")
    args = parser.parse_args()
    torch.manual_seed(0)
    x, y = rows(args.cells, "k_mass" if args.kind == "mass" else "k_features")
    if args.kind == "mass":
        model = MassChooser()
    else:
        model = KChooser()
        model.mean.copy_(x.mean(0))
        model.std.copy_(x.std(0).clamp_min(1e-6))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    for _ in range(EPOCHS):
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
    model.eval()
    model.save(args.out)
    report = {"n": len(x), "loss": float(loss.detach()), "label_rates": y.mean(0).tolist(),
              "predicted_rates": torch.sigmoid(model(x)).mean(0).tolist(), "cells": [str(c) for c in args.cells]}
    with torch.no_grad():
        report["ece"] = [ece(p, t) for p, t in zip(torch.sigmoid(model(x)).T.tolist(), y.T.tolist())]
    if args.kind == "mass":
        report["a"], report["b"] = model.a.tolist(), model.b.tolist()
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
