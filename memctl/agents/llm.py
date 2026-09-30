"""A language model as the frozen task model.

The prompt is: the task instruction, every ACTIVE memory item, then the current
observation. The model sees nothing else: no archive, no deleted items.
"""

from __future__ import annotations

import time

from memctl.agents.base import Agent
from memctl.llm import build_llm
from memctl.memory.items import count_tokens
from memctl.memory.state import MemoryView
from memctl.task import AgentStep, Observation, TaskState

DEFAULT_INSTRUCTIONS = (
    "You are an assistant with a limited memory. Use only the memory below. "
    "Answer with a short phrase. If the memory does not contain the answer, reply: unknown"
)


def memory_line(item) -> str:
    who = item.metadata.get("speaker") or item.source_type.value
    when = item.metadata.get("date")
    label = f"{who}, {when}" if when else who
    return f"[{item.id}] ({label}) {item.content}"


class LLMAgent(Agent):
    name = "llm"

    def __init__(self, config: dict, llm=None) -> None:
        super().__init__(config)
        self.llm = llm or build_llm(config.get("model", {"backend": "stub"}))
        self.model_id = self.llm.name
        self.max_new_tokens = int(config.get("max_new_tokens", 48))
        self.instructions = config.get("instructions")

    def build_prompt(self, memory: MemoryView, observation: Observation, task: TaskState) -> str:
        lines = [memory_line(item) for item in memory.active if item.id != observation.id]
        instructions = self.instructions or task.goal or DEFAULT_INSTRUCTIONS
        return "\n".join(
            [instructions, "", "## Memory", *(lines or ["(empty)"]), "", "## Current input", observation.content, "",
             "Answer:"]
        )

    def act(self, memory: MemoryView, observation: Observation, task: TaskState) -> AgentStep:
        prompt = self.build_prompt(memory, observation, task)
        started = time.perf_counter()
        answer = self.llm.generate(prompt, self.max_new_tokens).strip()
        info = {
            "prompt_tokens": count_tokens(prompt),
            "output_tokens": count_tokens(answer),
            "model_calls": 1,
            "latency_s": time.perf_counter() - started,
        }
        return AgentStep(answer, self._used(memory, observation, answer), info)

    @staticmethod
    def _used(memory: MemoryView, observation: Observation, answer: str) -> tuple[str, ...]:
        """An approximation of which items the model relied on: those that contain its answer.

        A language model does not say what it read, so access counts for LRU-style
        policies are only approximate with this agent (design decision E15).
        """
        needle = answer.strip().strip(".").lower()
        if len(needle) < 2 or needle == "unknown":
            return ()
        return tuple(item.id for item in memory.active if item.id != observation.id and needle in item.content.lower())
