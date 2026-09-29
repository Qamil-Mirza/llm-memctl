"""The frozen agent: builds the prompt from MemoryState and handles recall(id).

The agent only reads memory. It never changes where items live. The model
itself and the disk cache are in llm.py.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from memctl.items import Item, Place, count_tokens
from memctl.llm import LLM
from memctl.memory import MemoryState, RecallError

RECALL_PATTERN = re.compile(r"recall\(\s*[\[<]?([\w:.\-]+)[\]>]?\s*\)")
NOT_MENTIONED = "Not mentioned in the conversation"
INSTRUCTIONS = (
    "Below is your memory of a long conversation between two people, followed by a question.\n"
    "Rules:\n"
    "- Answer with a short phrase that states the fact asked for. Do not copy a whole message.\n"
    "- Each message starts with (speaker, time it was sent). For questions about when something "
    "happened, work out the date from the time the message was sent.\n"
    "- The archive index lists items stored elsewhere. Only their first words are shown. To read "
    "one in full, reply with exactly recall(<id>) and nothing else.\n"
    f"- If the memory does not contain the answer, reply: {NOT_MENTIONED}"
)


@dataclass
class AnswerResult:
    text: str
    prompt_tokens: int  # tokens of the largest prompt sent for this question
    retrieved_ids: list[str] = field(default_factory=list)
    recalled_ids: list[str] = field(default_factory=list)
    failed_recalls: list[str] = field(default_factory=list)
    final_prompt: str = ""


def section(title: str, lines: list[str]) -> str:
    return f"## {title}\n" + ("\n".join(lines) if lines else "(empty)")


def build_prompt(
    state: MemoryState, question: str, retrieved: list[Item], recalled: list[Item]
) -> tuple[str, str]:
    """The prompt for one question, as (shared part, question-specific part).

    The shared part depends only on the memory, so it is the same for every
    question asked at the same moment. The model backend can reuse it.
    """
    shared = "\n\n".join(
        [
            INSTRUCTIONS,
            section("Memory in context", [i.prompt_line() for i in state.in_place(Place.CONTEXT)]),
            section("Archive index", state.archive_index()),
        ]
    )
    specific = "\n\n".join(
        [
            "",
            section("Retrieved from store (this turn only)", [i.prompt_line() for i in retrieved]),
            section("Recalled from archive (this turn only)", [i.prompt_line() for i in recalled]),
            section("Question", [question]),
            "Answer:",
        ]
    )
    return shared, specific


class Agent:
    """Answers one question at a time."""

    def __init__(self, llm: LLM, max_new_tokens: int = 64, max_recall_rounds: int = 2) -> None:
        self.llm = llm
        self.max_new_tokens = max_new_tokens
        self.max_recall_rounds = max_recall_rounds

    def answer(self, state: MemoryState, question: str) -> AnswerResult:
        retrieved = state.search_store(question)
        result = AnswerResult(text="", prompt_tokens=0, retrieved_ids=[i.id for i in retrieved])
        recalled: list[Item] = []
        for round_number in range(self.max_recall_rounds + 1):
            shared, specific = build_prompt(state, question, retrieved, recalled)
            result.final_prompt = shared + specific
            result.prompt_tokens = max(result.prompt_tokens, count_tokens(result.final_prompt))
            output = self.llm.generate(result.final_prompt, self.max_new_tokens, shared_prefix=shared).strip()
            request = RECALL_PATTERN.search(output)
            repeated = request is not None and request.group(1) in result.recalled_ids
            if request is None or repeated or round_number == self.max_recall_rounds:
                result.text = output
                return result
            try:
                recalled.append(state.recall(request.group(1)))
                result.recalled_ids.append(request.group(1))
            except RecallError:
                result.failed_recalls.append(request.group(1))
                result.text = output
                return result
        return result
