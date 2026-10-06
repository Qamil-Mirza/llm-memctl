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
    def __init__(self, item_dim: int, global_dim: int, hidden: int = 64, architecture: str = "deepsets") -> None:
        super().__init__()
        if architecture not in ("mlp", "deepsets"):
            raise ValueError(f"unknown architecture '{architecture}' (known: mlp, deepsets)")
        self.config = {"item_dim": item_dim, "global_dim": global_dim, "hidden": hidden, "architecture": architecture}
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
