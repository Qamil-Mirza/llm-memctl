"""Scripted task models for the synthetic environments.

A scripted agent succeeds exactly when the text it needs is in ACTIVE memory,
so task success measures memory management and nothing else. `noise` makes it
fail at a known rate, which is how the failure attribution is checked.
"""

from __future__ import annotations

import random

from memctl.agents.base import Agent
from memctl.envs.grammar import UNKNOWN, find_facts, parse_query
from memctl.memory.state import MemoryView
from memctl.task import AgentStep, Observation, TaskState


def latest_statement(memory: MemoryView, attribute: str, entity: str) -> tuple[str, str] | None:
    """(value, id of the item stating it) for the most recent statement in ACTIVE, or None."""
    best = None
    for item in memory.active:
        if entity not in item.content:
            continue
        for stamp, found_attribute, found_entity, value in find_facts(item.content):
            if found_attribute == attribute and found_entity == entity and (best is None or stamp > best[0]):
                best = (stamp, value, item.id)
    return (best[1], best[2]) if best else None


class ScriptedReader(Agent):
    """Answers the synthetic recall queries by reading ACTIVE memory."""

    name = "scripted_reader"
    model_id = "scripted_reader"
    cheap = True

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.noise = float(config.get("noise", 0.0))
        self.rng = random.Random(0)

    def reset(self, seed: int) -> None:
        self.rng = random.Random(f"reader-{seed}")

    def act(self, memory: MemoryView, observation: Observation, task: TaskState) -> AgentStep:
        info = {"prompt_tokens": memory.active_tokens, "output_tokens": 1, "model_calls": 1}
        parsed = parse_query(observation.content)
        if parsed is None:
            return AgentStep(None, (), info)
        attribute, via, entity = parsed
        used = []
        if via:  # two hops: first find who the `via` of the entity is
            link = latest_statement(memory, via, entity)
            if link is None:
                return AgentStep(UNKNOWN, (), info)
            entity = link[0]
            used.append(link[1])
        found = latest_statement(memory, attribute, entity)
        if found is None:
            return AgentStep(UNKNOWN, tuple(used), info)
        used.append(found[1])
        if self.rng.random() < self.noise:  # a reasoning failure: the evidence was there
            return AgentStep(UNKNOWN, (), {**info, "injected_noise": True})
        return AgentStep(found[0], tuple(used), info)


class ScriptedToolAgent(Agent):
    """Runs the workflow environment's instructions, reading tokens from ACTIVE memory.

    If the token a stage needs is not in memory it restarts the job, which is
    what makes a memory failure change the rest of the episode.
    """

    name = "scripted_tool_agent"
    model_id = "scripted_tool_agent"
    cheap = True

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.noise = float(config.get("noise", 0.0))
        self.rng = random.Random(0)

    def reset(self, seed: int) -> None:
        self.rng = random.Random(f"tool-agent-{seed}")

    def act(self, memory: MemoryView, observation: Observation, task: TaskState) -> AgentStep:
        from memctl.envs.workflow import INSTRUCTION

        info = {"prompt_tokens": memory.active_tokens, "output_tokens": 4, "model_calls": 1}
        match = INSTRUCTION.search(observation.content)
        if match is None:
            return AgentStep(None, (), info)
        stage, job, previous = match.groups()
        if previous is None:
            return AgentStep(f"run {stage} {job}", (), info)
        found = latest_statement(memory, f"{previous} token", job)
        if found is None:
            return AgentStep(f"restart {job}", (), info)
        if self.rng.random() < self.noise:  # a reasoning failure: the token was there
            return AgentStep(f"restart {job}", (), {**info, "injected_noise": True})
        return AgentStep(f"run {stage} {job} {found[0]}", (found[1],), info)
