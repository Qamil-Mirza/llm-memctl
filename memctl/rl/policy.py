"""The learned scorer: a small network over per-item features.

It gives every candidate item one logit per operation (the "how to remove it"
columns, plus one "retrieve it" column for archived candidates) and gives the
state one value estimate. It is deliberately small: a few thousand parameters,
no raw history.

Architectures:
- `mlp`       each item is scored from its own features and the global features
- `deepsets`  as `mlp`, plus a summary (mean and max) of all candidates, so an item
              is scored relative to what else is in memory
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn

from memctl.memory.actions import Operation

# Column order of the policy's output. Fixed, so one checkpoint works under any action set:
# operations an experiment does not allow are masked out, not removed.
REMOVAL_OPERATIONS = [
    Operation.EVICT, Operation.MOVE_TO_ARCHIVE, Operation.COMPACT, Operation.COMPACT_AND_ARCHIVE
]
RETRIEVE_COLUMN = len(REMOVAL_OPERATIONS)
N_COLUMNS = RETRIEVE_COLUMN + 1


class ItemPolicy(nn.Module):
    def __init__(self, item_dim: int, global_dim: int, hidden: int = 64, architecture: str = "deepsets",
                 residual: float | None = None, residual_index: int | None = None) -> None:
        super().__init__()
        if architecture not in ("mlp", "deepsets"):
            raise ValueError(f"unknown architecture '{architecture}' (known: mlp, deepsets)")
        self.config = {"item_dim": item_dim, "global_dim": global_dim, "hidden": hidden, "architecture": architecture}
        # Optional residual on the retrieve column: logit + alpha x the item's retrieval score (feature
        # `residual_index`), alpha learned from `residual`. A large start keeps the search's own order until
        # training moves away from it (Experiment 15).
        self.residual = None
        if residual is not None:
            self.config.update(residual=float(residual), residual_index=int(residual_index))
            self.residual = nn.Parameter(torch.tensor(float(residual)))
            self.residual_index = int(residual_index)
        self.architecture = architecture
        self.encode = nn.Sequential(nn.Linear(item_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        context = global_dim + (2 * hidden if architecture == "deepsets" else 0)
        self.score = nn.Sequential(nn.Linear(hidden + context, hidden), nn.ReLU(), nn.Linear(hidden, N_COLUMNS))
        self.value = nn.Sequential(nn.Linear(2 * hidden + global_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, items: torch.Tensor, global_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """items [N, item_dim], global_features [global_dim] -> (logits [N, N_COLUMNS], value [])."""
        encoded = self.encode(items)
        pooled = torch.cat([encoded.mean(dim=0), encoded.max(dim=0).values])
        state = torch.cat([pooled, global_features])
        context = state if self.architecture == "deepsets" else global_features
        logits = self.score(torch.cat([encoded, context.expand(encoded.shape[0], -1)], dim=1))
        if self.residual is not None:
            bonus = torch.zeros_like(logits)
            bonus[:, RETRIEVE_COLUMN] = self.residual * items[:, self.residual_index]
            logits = logits + bonus
        return logits, self.value(state).squeeze(-1)


class NeededModel(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 32) -> None:
        super().__init__()
        self.config = {"input_dim": input_dim, "hidden": hidden}
        self.net = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        """Logits, one per row."""
        return self.net(rows).squeeze(-1)

    @staticmethod
    def load(path: str | Path) -> "NeededModel":
        checkpoint = torch.load(Path(path), weights_only=True)
        model = NeededModel(**checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model


K_FEATURE_DIM = 16 + 3 + 2  # sorted top-16 logits, gaps at ranks 5/6, 8/9 and 16/17, softmax entropy, log question tokens


def k_features(retrieve_logits: torch.Tensor, question_tokens: int) -> list[float]:
    """What the k-chooser sees of one ranking (Experiment 19): the head's view of the shortlist, not the items."""
    ranked = torch.sort(retrieve_logits, descending=True).values.tolist()
    top = (ranked + [ranked[-1]] * 16)[:16] if ranked else [0.0] * 16
    gaps = [ranked[i - 1] - ranked[i] if len(ranked) > i else 0.0 for i in (5, 8, 16)]
    entropy = float(torch.distributions.Categorical(logits=retrieve_logits).entropy()) if ranked else 0.0
    return top + gaps + [entropy, math.log1p(question_tokens)]


class KChooser(nn.Module):
    """P(every gold item is in the head's top k), one output per k (Experiment 19), from `k_features`."""

    def __init__(self, ks: tuple[int, ...] = (5, 8, 16), hidden: int = 32) -> None:
        super().__init__()
        self.config = {"ks": list(ks), "hidden": hidden}
        self.register_buffer("mean", torch.zeros(K_FEATURE_DIM))
        self.register_buffer("std", torch.ones(K_FEATURE_DIM))
        self.net = nn.Sequential(nn.Linear(K_FEATURE_DIM, hidden), nn.ReLU(), nn.Linear(hidden, len(ks)))

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        """Logits, one column per k."""
        return self.net((rows - self.mean) / self.std)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"config": self.config, "state_dict": self.state_dict()}, Path(path))

    @staticmethod
    def load(path: str | Path) -> "KChooser":
        checkpoint = torch.load(Path(path), weights_only=True)
        model = KChooser(tuple(checkpoint["config"]["ks"]), checkpoint["config"]["hidden"])
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model
