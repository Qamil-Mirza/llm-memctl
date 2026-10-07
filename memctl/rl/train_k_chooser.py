"""Train the Experiment 19 k-chooser on out-of-sample head outputs (EXPERIMENTS.md §19, pre-registered).

Reads the steps of `k_mode: oracle` runs, where a head ranks lists it was not trained on (head A on half B,
head B on half A). Each question gives the chooser's features and the all-found labels at k = 5, 8 and 16.
Fits one KChooser by binary cross-entropy, full batch, with fixed settings, and saves it.

    python -m memctl.rl.train_k_chooser runs/exp19_xfit_f0/head_a_on_b runs/exp19_xfit_f0/head_b_on_a \\
        --out runs/exp19_chooser_f0/k_chooser.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from memctl.rl.policy import KChooser

EPOCHS, LR, WEIGHT_DECAY = 400, 0.01, 1e-4


def rows(cells: list[Path]) -> tuple[torch.Tensor, torch.Tensor]:
    features, labels = [], []
    for cell in cells:
        for line in open(cell / "steps.jsonl"):
            info = json.loads(line).get("controller_info") or {}
            if "k_labels" in info:
                features.append(info["k_features"])
                labels.append([float(x) for x in info["k_labels"]])
    return torch.tensor(features), torch.tensor(labels)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cells", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(0)
    x, y = rows(args.cells)
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
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
