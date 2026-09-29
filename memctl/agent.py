"""The frozen agent: builds the prompt from MemoryState, handles recall(id), caches outputs.

Backends: `stub` (no model, for tests). Hugging Face and vLLM backends are
added in Phase 2 behind the same `LLM` interface.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from memctl.embed import content_words
from memctl.items import Item, Place, count_tokens
from memctl.memory import MemoryState, RecallError

RECALL_PATTERN = re.compile(r"recall\(\s*\[?([\w:.\-]+)\]?\s*\)")
INSTRUCTIONS = (
    "You answer questions about a long conversation using the memory below.\n"
    "Answer in a few words. If you need an archived item, reply only with recall(<id>)."
)


class LLM(Protocol):
    name: str

    def generate(self, prompt: str, max_new_tokens: int) -> str: ...


class StubLLM:
    """A model-free stand-in: answers with the memory line that best matches the question.

    It asks for recall(id) when an archive index line matches better than anything it can read.
    """

    name = "stub"

    def generate(self, prompt: str, max_new_tokens: int) -> str:
        memory, _, rest = prompt.partition("## Archive index")
        index, _, question = rest.partition("## Question")
        wanted = set(content_words(question))

        def best(section: str) -> tuple[int, str]:
            lines = [line for line in section.splitlines() if line.startswith("[")]
            scored = [(len(wanted & set(content_words(line))), line) for line in lines]
            return max(scored, default=(0, ""))

        memory_score, memory_line = best(memory)
        index_score, index_line = best(index.partition("## Recalled")[0])
        if index_score > memory_score:
            return f"recall({index_line[1:index_line.index(']')]})"
        if memory_score == 0:
            return "unknown"
        return memory_line.split(") ", 1)[-1]


class CachedLLM:
    """Wraps a backend and saves every generation to disk, keyed by prompt + settings."""

    def __init__(self, backend: LLM, cache_dir: str, settings: dict) -> None:
        self.backend = backend
        self.name = backend.name
        self.settings = settings
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def generate(self, prompt: str, max_new_tokens: int) -> str:
        key_material = {"prompt": prompt, "max_new_tokens": max_new_tokens, **self.settings}
        key = hashlib.sha256(json.dumps(key_material, sort_keys=True).encode()).hexdigest()
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            self.hits += 1
            return json.loads(path.read_text())["output"]
        self.misses += 1
        output = self.backend.generate(prompt, max_new_tokens)
        path.write_text(json.dumps({**key_material, "output": output}, indent=1))
        return output


def build_llm(model_config: dict) -> LLM:
    """Create the answering model named in the config, wrapped in the disk cache."""
    backend_name = model_config.get("backend", "stub")
    if backend_name != "stub":
        raise KeyError(f"unknown model backend '{backend_name}' (available now: stub)")
    settings = {k: model_config.get(k) for k in ("backend", "name", "temperature", "seed")}
    return CachedLLM(StubLLM(), model_config.get("cache_dir", "cache/generations"), settings)


@dataclass
class AnswerResult:
    text: str
    prompt_tokens: int  # tokens of the largest prompt sent for this question
    retrieved_ids: list[str] = field(default_factory=list)
    recalled_ids: list[str] = field(default_factory=list)
    failed_recalls: list[str] = field(default_factory=list)
    final_prompt: str = ""


def build_prompt(state: MemoryState, question: str, retrieved: list[Item], recalled: list[Item]) -> str:
    """The full prompt for one question. Sections appear in a fixed order."""

    def section(title: str, lines: list[str]) -> str:
        return f"## {title}\n" + ("\n".join(lines) if lines else "(empty)")

    parts = [
        INSTRUCTIONS,
        section("Memory in context", [i.prompt_line() for i in state.in_place(Place.CONTEXT)]),
        section("Retrieved from store (this turn only)", [i.prompt_line() for i in retrieved]),
        section("Recalled from archive (this turn only)", [i.prompt_line() for i in recalled]),
        section("Archive index", state.archive_index()),
        section("Question", [question]),
    ]
    return "\n\n".join(parts) + "\n\nAnswer:"


class Agent:
    """Answers one question at a time. It never changes where items live."""

    def __init__(self, llm: LLM, max_new_tokens: int = 64, max_recall_rounds: int = 2) -> None:
        self.llm = llm
        self.max_new_tokens = max_new_tokens
        self.max_recall_rounds = max_recall_rounds

    def answer(self, state: MemoryState, question: str) -> AnswerResult:
        retrieved = state.search_store(question)
        result = AnswerResult(text="", prompt_tokens=0, retrieved_ids=[i.id for i in retrieved])
        recalled: list[Item] = []
        for round_number in range(self.max_recall_rounds + 1):
            prompt = build_prompt(state, question, retrieved, recalled)
            result.final_prompt = prompt
            result.prompt_tokens = max(result.prompt_tokens, count_tokens(prompt))
            output = self.llm.generate(prompt, self.max_new_tokens).strip()
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
