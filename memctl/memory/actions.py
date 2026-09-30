"""The action schema: what a controller may ask the memory engine to do."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from memctl.memory.items import Tier


class Operation(str, Enum):
    KEEP = "KEEP"
    EVICT = "EVICT"
    MOVE_TO_ARCHIVE = "MOVE_TO_ARCHIVE"
    RETRIEVE_FROM_ARCHIVE = "RETRIEVE_FROM_ARCHIVE"
    COMPACT = "COMPACT"
    COMPACT_AND_ARCHIVE = "COMPACT_AND_ARCHIVE"
    CONSOLIDATE = "CONSOLIDATE"
    NO_OP = "NO_OP"
    # Defined so that controllers and logs can name them; no handler yet (see engine.py).
    PROMOTE = "PROMOTE"
    DEMOTE = "DEMOTE"
    PIN = "PIN"
    UNPIN = "UNPIN"
    UPDATE = "UPDATE"
    SUPERSEDE = "SUPERSEDE"


class ActionStatus(str, Enum):
    APPLIED = "applied"
    REJECTED = "rejected"


class ActionSource(str, Enum):
    CONTROLLER = "controller"  # the controller chose it
    HARNESS = "harness"  # the budget fallback forced it
    INTERVENTION = "intervention"  # an analysis forced it (counterfactual ablation)


@dataclass(frozen=True)
class MemoryAction:
    operation: Operation
    target_ids: tuple[str, ...] = ()
    destination: Tier | None = None
    parameters: dict = field(default_factory=dict)
    confidence: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "operation", Operation(self.operation))
        object.__setattr__(self, "target_ids", tuple(self.target_ids))

    def log_row(self) -> dict:
        return {
            "operation": self.operation.value,
            "target_ids": list(self.target_ids),
            "destination": self.destination.value if self.destination else None,
            "parameters": self.parameters,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class ActionResult:
    """What happened to one action."""

    action: MemoryAction
    status: ActionStatus
    source: ActionSource
    step: int
    reason: str = ""
    created_ids: tuple[str, ...] = ()
    tokens_freed: int = 0  # reduction in active tokens (negative when a retrieval adds tokens)

    @property
    def applied(self) -> bool:
        return self.status is ActionStatus.APPLIED

    def log_row(self) -> dict:
        return {
            **self.action.log_row(),
            "status": self.status.value,
            "source": self.source.value,
            "step": self.step,
            "reason": self.reason,
            "created_ids": list(self.created_ids),
            "tokens_freed": self.tokens_freed,
        }
