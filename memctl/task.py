"""Types shared by environments, agents, controllers and the harness.

Nothing here carries ground truth except `Dependency`, which environments give
to the evaluator and the oracle. Controllers and agents never receive one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from memctl.memory.items import SourceType


@dataclass(frozen=True)
class Observation:
    """One thing the environment shows the agent. It becomes a memory item with the same id."""

    id: str
    content: str
    source_type: SourceType = SourceType.OBSERVATION
    requires_response: bool = False  # True when the agent must act on it (a question, an instruction)
    metadata: dict = field(default_factory=dict)  # visible to agent and controller; never labels
    pinned: bool = False
    parent_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvidenceRequirement:
    """One piece of information a query needs.

    `item_ids` are the observations that state it; any one of them is enough.
    `needle` is the text an item must still contain to count as carrying it,
    which is how a compacted or consolidated item is judged.
    """

    item_ids: tuple[str, ...]
    needle: str


@dataclass(frozen=True)
class Dependency:
    """Ground truth: which evidence the query observed at `step` needs."""

    query_id: str
    step: int
    requirements: tuple[EvidenceRequirement, ...]
    gold: str | None = None
    category: str = ""


@dataclass
class StepResult:
    observation: Observation | None  # the next observation, or None when the episode is over
    reward: float
    done: bool
    info: dict = field(default_factory=dict)  # evaluation only, e.g. {"correct": True, "gold": "K93Q"}


@dataclass(frozen=True)
class TaskState:
    """What a controller or agent may know about the task right now."""

    step: int
    goal: str
    observation: Observation
    horizon: int | None = None
    goal_embedding: object | None = None


@dataclass
class AgentStep:
    action: str | None
    used_item_ids: tuple[str, ...] = ()  # the memory items the task model relied on
    info: dict = field(default_factory=dict)  # prompt_tokens, output_tokens, model_calls, latency_s
