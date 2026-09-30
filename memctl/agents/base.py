"""The Agent interface: the frozen task model. It reads memory and never changes it."""

from __future__ import annotations

from abc import ABC, abstractmethod

from memctl.memory.state import MemoryView
from memctl.task import AgentStep, Observation, TaskState


class Agent(ABC):
    name = "base"
    model_id = "none"  # recorded with every run
    cheap = False  # True when a call costs nothing, so the reference pass may use the real agent

    def __init__(self, config: dict) -> None:
        self.config = config

    def reset(self, seed: int) -> None:
        """Called at the start of every episode."""

    @abstractmethod
    def act(self, memory: MemoryView, observation: Observation, task: TaskState) -> AgentStep:
        """Respond to `observation` using only `memory.active`.

        Called only for observations with `requires_response`. `used_item_ids`
        in the result feeds the access counts that LRU-style policies read.
        """


class NullAgent(Agent):
    """Never answers. Used for reference passes when the real task model is expensive."""

    name = "null"
    cheap = True

    def act(self, memory: MemoryView, observation: Observation, task: TaskState) -> AgentStep:
        return AgentStep(None)
