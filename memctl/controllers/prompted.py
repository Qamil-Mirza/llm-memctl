"""A language model, prompted, as the memory controller.

The model is shown the items in context, the budget and the operations this
experiment allows, and must reply with a JSON list of actions. Anything that
parses as an action is passed to the engine untouched, so the model's mistakes
(an operation that is not allowed, an id that does not exist) are rejected and
counted there like any other controller's. A reply that is not JSON yields no
action, and the forced fallback then shows up in the metrics.

    controller:
      name: prompted_llm
      model: {backend: openai, name: llama3.1:8b, base_url: http://localhost:11434/v1}
      call: pressure        # pressure (default): only when over budget or a query could use the archive
"""

from __future__ import annotations

import json
import re
import time

from memctl.controllers.base import MemoryController
from memctl.llm import build_llm
from memctl.memory.actions import MemoryAction, Operation
from memctl.memory.items import count_tokens, truncate_tokens
from memctl.memory.state import MemoryView
from memctl.retrieval import LexicalRetriever
from memctl.task import TaskState

OPERATION_HELP = {
    Operation.KEEP: "leave the items where they are",
    Operation.EVICT: "delete the items for good; they can never be read again",
    Operation.MOVE_TO_ARCHIVE: "move the items out of context; they can be retrieved later",
    Operation.RETRIEVE_FROM_ARCHIVE: "bring archived items back into context (this uses budget)",
    Operation.COMPACT: "replace each item with a shorter version of itself; detail may be lost",
    Operation.COMPACT_AND_ARCHIVE: "keep a shorter version in context and archive the full item",
    Operation.CONSOLIDATE: "merge two or more items into a single item",
    Operation.NO_OP: "do nothing",
}
_JSON_LIST = re.compile(r"\[.*\]", re.DOTALL)


class PromptedLLMController(MemoryController):
    name = "prompted_llm"

    def __init__(self, config: dict, seed: int = 0, llm=None) -> None:
        super().__init__(config, seed)
        self.llm = llm or build_llm(config.get("model", {"backend": "stub"}))
        self.model_id = self.llm.name
        self.call = config.get("call", "pressure")
        self.excerpt_tokens = int(config.get("excerpt_tokens", 40))
        self.max_new_tokens = int(config.get("max_new_tokens", 256))
        self.archive_candidates = int(config.get("archive_candidates", 8))
        self._retriever = LexicalRetriever()
        self._info: dict = {}

    def decide(self, memory: MemoryView, task: TaskState) -> list[MemoryAction]:
        self._info = {}
        can_retrieve = self.allows(Operation.RETRIEVE_FROM_ARCHIVE) and task.observation.requires_response
        shortlist = (
            [item for item, _ in self._retriever.search(task.observation.content, memory.archived, self.archive_candidates)]
            if can_retrieve else []
        )
        if self.call == "pressure" and not memory.over_budget and not shortlist:
            return []
        prompt = self.build_prompt(memory, task, shortlist)
        started = time.perf_counter()
        reply = self.llm.generate(prompt, self.max_new_tokens)
        actions, problems = self.parse(reply)
        self._info = {
            "model_calls": 1,
            "input_tokens": count_tokens(prompt),
            "output_tokens": count_tokens(reply),
            "latency_s": time.perf_counter() - started,
            "reply": reply,
            "parse_ok": problems == 0 and (bool(actions) or reply.strip() in ("[]", "")),
            "unparsed_entries": problems,
        }
        return actions

    def build_prompt(self, memory: MemoryView, task: TaskState, shortlist: list) -> str:
        operations = [op for op in OPERATION_HELP if self.allows(op)]
        lines = [
            "You manage the memory of an AI assistant. The assistant can read only the items in context.",
            f"Context budget: {memory.budget} tokens. In context now: {memory.active_tokens} tokens.",
        ]
        if memory.over_budget:
            lines.append(
                f"The context is over budget: free at least {memory.active_tokens - memory.budget} tokens. "
                "Remove what is least likely to be needed later. If you free too little, the oldest items are deleted."
            )
        lines += ["", "Operations you may use:"]
        lines += [f"- {op.value}: {OPERATION_HELP[op]}" for op in operations]
        lines += ["", "Items in context (id | tokens | age in steps | source | times used | text):"]
        for item in memory.active:
            text = truncate_tokens(item.content, self.excerpt_tokens)
            lines.append(
                f"{item.id} | {item.token_count} | {memory.step - item.created_at} | {item.source_type.value} | "
                f"{item.access_count} | {text}"
            )
        if shortlist:
            lines += ["", "Archived items that may be relevant (id | tokens | text):"]
            lines += [f"{i.id} | {i.token_count} | {truncate_tokens(i.content, self.excerpt_tokens)}" for i in shortlist]
        lines += [
            "",
            f'The assistant\'s current input: "{task.observation.content}"',
            "",
            "Reply with a JSON list of actions and nothing else. Example:",
            '[{"operation": "' + operations[0].value + '", "target_ids": ["' + (memory.active[0].id if memory.active else "id") + '"]}]',
            "",
            "## Actions",
        ]
        return "\n".join(lines)

    @staticmethod
    def parse(reply: str) -> tuple[list[MemoryAction], int]:
        """(actions, number of entries that could not be read as an action)."""
        match = _JSON_LIST.search(reply)
        if not match:
            return [], 1 if reply.strip() else 0
        try:
            entries = json.loads(match.group(0))
        except json.JSONDecodeError:
            return [], 1
        actions, problems = [], 0
        for entry in entries if isinstance(entries, list) else []:
            try:
                operation = Operation(str(entry["operation"]).upper())
                targets = tuple(str(i) for i in entry.get("target_ids", []))
                parameters = entry.get("parameters") if isinstance(entry.get("parameters"), dict) else {}
                actions.append(MemoryAction(operation, targets, parameters=parameters))
            except (KeyError, ValueError, TypeError, AttributeError):
                problems += 1
        return actions, problems

    def decision_info(self) -> dict:
        return self._info
