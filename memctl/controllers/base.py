"""The MemoryController interface. Every controller, learned or not, is one of these.

A controller manages memory and nothing else: it never produces the agent's
task actions and never sees ground truth. Whatever it needs as input (raw text,
embeddings, features, an API payload) it builds itself from the `MemoryView` and
`TaskState` it is given, so its preprocessing stays inside its own adapter.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from memctl.memory.actions import ActionResult, MemoryAction, Operation
from memctl.memory.state import MemoryView
from memctl.task import AgentStep, Observation, TaskState


@dataclass(frozen=True)
class EpisodeInfo:
    episode_id: str
    seed: int
    budget: int
    archive_budget: int | None
    allowed_operations: frozenset[Operation]
    goal: str = ""
    horizon: int | None = None
    embedder: object | None = None  # the embedder that produced item embeddings, if any


@dataclass(frozen=True)
class StepEvent:
    """A new observation has entered memory."""

    step: int
    observation: Observation
    item_id: str


@dataclass(frozen=True)
class Feedback:
    """What followed the controller's last decision."""

    step: int
    results: tuple[ActionResult, ...]  # its own actions, applied or rejected
    forced: tuple[ActionResult, ...]  # evictions the harness had to force
    reward: float
    reward_terms: dict = field(default_factory=dict)
    agent_step: AgentStep | None = None
    scored: bool = False
    correct: bool | None = None
    done: bool = False


class MemoryController(ABC):
    name = "base"
    model_id = "none"  # recorded with every run
    uses_hindsight = False  # True only for oracles; the runner then calls receive_hindsight
    ignores_budget = False  # True only for the unlimited-context reference

    def __init__(self, config: dict, seed: int = 0) -> None:
        self.config = config
        self.seed = seed
        self.episode: EpisodeInfo | None = None

    @property
    def display_name(self) -> str:
        """The name used in every output."""
        return self.config.get("label") or self.name

    def reset(self, episode: EpisodeInfo) -> None:
        """Start of an episode."""
        self.episode = episode

    def observe(self, event: StepEvent) -> None:
        """A new item arrived. For controllers that keep their own running state."""

    @abstractmethod
    def decide(self, memory: MemoryView, task: TaskState) -> list[MemoryAction]:
        """The memory actions to take now. An empty list means do nothing.

        Called once per step, after the new observation has been added to ACTIVE
        and before the agent reads memory. If ACTIVE is still over budget after
        these actions, the harness evicts oldest-first and records it against
        the controller.
        """

    def update(self, feedback: Feedback) -> None:
        """The outcome of the last decision. Learning controllers use it."""

    def save(self, path: str | Path) -> None:
        """Write whatever is needed to restore the controller. Default: nothing to save."""

    def load(self, path: str | Path) -> None:
        """Restore what `save` wrote."""

    def decision_info(self) -> dict:
        """Loggable detail about the last decision: scores, probabilities, features,
        latency, token counts, model calls, cost. Never hidden reasoning."""
        return {}

    def receive_hindsight(self, hindsight) -> None:
        raise NotImplementedError(f"{self.name} does not use hindsight")

    def allows(self, operation: Operation) -> bool:
        return self.episode is not None and operation in self.episode.allowed_operations
