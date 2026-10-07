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

# Added after the task instructions when `reasoning: true`. A 7B reader asked for a short phrase cannot
# count across sessions or do date arithmetic, and falls back to "unknown" or "yesterday" even with
# the evidence in view (Experiment 11e); a short note before the answer is LongMemEval's Chain-of-Note.
REASONING_INSTRUCTIONS = (
    "Each memory line shows who said it and when. First write a brief note: list the memory lines that "
    "bear on the question, then do any counting, adding or date arithmetic. Give dates as absolute dates "
    "(for example 7 May 2023), never relative ones such as 'yesterday' or 'last year'. Reply unknown only "
    "if the memory has nothing at all on the topic. End with one line of the form 'Answer: <short phrase>'."
)


def final_answer(output: str) -> str:
    """The text after the last 'Answer:' marker, else the last non-empty line."""
    marker = output.lower().rfind("answer:")
    if marker >= 0:
        rest = output[marker + len("answer:"):].strip()
        return rest.splitlines()[0].strip() if rest else ""
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return lines[-1] if lines else ""


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
        # reasoning: a brief note, then "Answer: ..." (parsed out); memory is shown in the order it
        # arrived, so items retrieved from the archive sit at their place in the conversation.
        self.reasoning = bool(config.get("reasoning", False))

    def build_prompt(self, memory: MemoryView, observation: Observation, task: TaskState) -> str:
        items = [item for item in memory.active if item.id != observation.id]
        if self.reasoning:
            items.sort(key=lambda item: item.created_at)
        lines = [memory_line(item) for item in items]
        instructions = self.instructions or task.goal or DEFAULT_INSTRUCTIONS
        if self.reasoning:
            instructions = f"{instructions}\n{REASONING_INSTRUCTIONS}"
        return "\n".join(
            [instructions, "", "## Memory", *(lines or ["(empty)"]), "", "## Current input", observation.content, "",
             "Note:" if self.reasoning else "Answer:"]
        )

    def act(self, memory: MemoryView, observation: Observation, task: TaskState) -> AgentStep:
        prompt = self.build_prompt(memory, observation, task)
        started = time.perf_counter()
        output = self.llm.generate(prompt, self.max_new_tokens).strip()
        answer = (final_answer(output) or "unknown") if self.reasoning else output
        info = {
            "prompt_tokens": count_tokens(prompt),
            "output_tokens": count_tokens(output),
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
