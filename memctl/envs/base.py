"""The TaskEnvironment interface. The harness knows environments only through it."""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod

from memctl.task import Dependency, Observation, StepResult


class TaskEnvironment(ABC):
    name = "base"
    goal = ""  # the task instruction, shown to the agent outside the memory budget
    # True when what the environment shows next depends on what the agent did. Hindsight
    # from a reference pass is then only approximate (see hindsight/collect.py).
    stream_depends_on_agent = False

    def __init__(self, config: dict) -> None:
        self.config = config

    @abstractmethod
    def reset(self, seed: int) -> Observation:
        """Start a new episode and return the first observation."""

    @abstractmethod
    def step(self, agent_action: str | None) -> StepResult:
        """Take the agent's action for the current observation and move to the next one."""

    @abstractmethod
    def get_observation(self) -> Observation | None: ...

    @abstractmethod
    def is_done(self) -> bool: ...

    @abstractmethod
    def get_reward(self) -> float:
        """The reward given by the most recent step."""

    @abstractmethod
    def task_success(self) -> float:
        """Downstream task success for the episode so far, between 0 and 1."""

    def get_ground_truth_dependencies(self) -> list[Dependency]:
        """Evidence needed by each query issued so far. Evaluation and oracle only."""
        return []

    def episode_stats(self) -> dict:
        """Environment-specific counts for the episode log."""
        return {}

    def snapshot(self):
        return copy.deepcopy(self.__dict__)

    def restore(self, snapshot) -> None:
        self.__dict__.update(copy.deepcopy(snapshot))
