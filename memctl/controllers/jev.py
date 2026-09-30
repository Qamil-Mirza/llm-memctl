"""The JEV controller: a lightweight decision model behind an adapter.

Everything specific to JEV lives in this file. The rest of the framework sees a
MemoryController that returns MemoryActions; it does not know that JEV reads a
JSON state and answers typed Choice questions, and it would not change if JEV
read raw tokens, embeddings or features instead.

How a decision is framed (TypeSafe's Jev model, docs.typesafe.ai):
- one request per batch of candidate items; `state` holds the items and the budget
- one Choice question per item; the options are the operations this experiment allows
- the answer gives a choice, a probability per option and a confidence

JEV answers each question on its own, so its choices may leave memory over
budget. With `repair: true` (default) the adapter then removes the items JEV
was least sure about keeping, each by its next most probable operation, and
marks those actions `repaired`. With `repair: false` the harness fallback does
it and the forced evictions count against JEV.

Status: the real client needs `JEV_API_KEY`. Without it the controller refuses
to start unless `allow_fake: true`, in which case a hand-written stand-in runs
and every output is labelled FAKE. The HTTP client was written from the
documented request and response shapes and has never been run against the live
API: treat its response parsing as unverified until the first real call.
"""

from __future__ import annotations

import json
import math
import os
import time
import urllib.request
from dataclasses import dataclass
from typing import Protocol

from memctl.controllers.base import EpisodeInfo, MemoryController
from memctl.embed import cosine
from memctl.memory.actions import MemoryAction, Operation
from memctl.memory.items import Fidelity, MemoryItem, count_tokens
from memctl.memory.state import MemoryView
from memctl.retrieval import LexicalRetriever
from memctl.task import TaskState

API_KEY_ENV = "JEV_API_KEY"
ENDPOINT = "https://api.typesafe.ai/v1/systemone"
USD_PER_MILLION_INPUT_TOKENS = 0.042  # docs.typesafe.ai/models, read 2026-09-28; output tokens are free

OPTION_HELP = {
    Operation.KEEP: "Keep it in context. For items the assistant is likely to need soon or often.",
    Operation.EVICT: "Delete it for good. For filler and anything with no lasting information.",
    Operation.MOVE_TO_ARCHIVE: "Move it out of context into the archive. It stays retrievable. "
    "For items that may be needed later but not soon.",
    Operation.COMPACT: "Replace it with a shorter version. For long items with a little useful content.",
    Operation.COMPACT_AND_ARCHIVE: "Keep a short version in context and archive the full item. "
    "For long items whose detail may matter later.",
    Operation.RETRIEVE_FROM_ARCHIVE: "Bring it back into context now. For archived items the current input needs.",
}
ACTIVE_OPTIONS = (
    Operation.KEEP, Operation.EVICT, Operation.MOVE_TO_ARCHIVE, Operation.COMPACT, Operation.COMPACT_AND_ARCHIVE
)


class JevUnavailableError(RuntimeError):
    """The real JEV API cannot be used and the fake client was not allowed."""


@dataclass(frozen=True)
class JevAnswer:
    choice: str
    probabilities: dict[str, float]
    confidence: float


@dataclass(frozen=True)
class JevResponse:
    answers: dict[str, JevAnswer]
    model: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    cost_usd: float
    fake: bool


class JevClient(Protocol):
    is_fake: bool

    def ask(self, state: dict, questions: dict[str, dict]) -> JevResponse: ...


def choice_confidence(probabilities: dict[str, float]) -> float:
    """Confidence of a Choice as documented by TypeSafe: (n * peak - 1) / (n - 1)."""
    n = len(probabilities)
    if n < 2:
        return 1.0
    return max(0.0, min(1.0, (n * max(probabilities.values()) - 1) / (n - 1)))


class TypeSafeJevClient:
    """The real client: one HTTPS request per call. The key is sent in a header and never logged."""

    is_fake = False

    def __init__(self, api_key: str, model: str = "jev-latest", endpoint: str = ENDPOINT, timeout_s: float = 60.0) -> None:
        self._api_key = api_key
        self.model = model
        self.endpoint = endpoint
        self.timeout_s = timeout_s

    def build_request(self, state: dict, questions: dict[str, dict]) -> dict:
        return {"model": self.model, "state": state, "questions": questions}

    def ask(self, state: dict, questions: dict[str, dict]) -> JevResponse:
        body = json.dumps(self.build_request(state, questions)).encode()
        request = urllib.request.Request(
            self.endpoint, data=body,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self._api_key}"},
        )
        started = time.perf_counter()
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            payload = json.load(response)
        return self.parse_response(payload, time.perf_counter() - started, fallback_tokens=count_tokens(body.decode()))

    def parse_response(self, payload: dict, latency_s: float, fallback_tokens: int = 0) -> JevResponse:
        raw_answers = payload.get("answers") or payload.get("results") or {}
        answers = {}
        for question_id, raw in raw_answers.items():
            probabilities = {str(k): float(v) for k, v in (raw.get("probabilities") or {}).items()}
            confidence = raw.get("confidence")
            answers[question_id] = JevAnswer(
                str(raw["choice"]), probabilities,
                float(confidence) if confidence is not None else choice_confidence(probabilities),
            )
        usage = payload.get("usage") or {}
        input_tokens = int(usage.get("input_tokens", fallback_tokens))
        return JevResponse(
            answers, str(payload.get("model", self.model)), input_tokens, int(usage.get("output_tokens", 0)),
            latency_s, input_tokens * USD_PER_MILLION_INPUT_TOKENS / 1e6, fake=False,
        )


class FakeJevClient:
    """A stand-in for tests. A hand-written heuristic, NOT the JEV model: it says nothing about JEV."""

    is_fake = True

    def ask(self, state: dict, questions: dict[str, dict]) -> JevResponse:
        started = time.perf_counter()
        answers = {}
        for question_id, question in questions.items():
            item = state["memory_items"][question_id]
            weights = self._weights(item, list(question["criteria"]))
            total = sum(weights.values())
            probabilities = {option: round(weight / total, 4) for option, weight in weights.items()}
            best = max(probabilities, key=probabilities.get)
            answers[question_id] = JevAnswer(best, probabilities, round(choice_confidence(probabilities), 4))
        tokens = count_tokens(json.dumps(state)) + count_tokens(json.dumps(questions))
        return JevResponse(answers, "FAKE-jev", tokens, 0, time.perf_counter() - started, 0.0, fake=True)

    @staticmethod
    def _weights(item: dict, options: list[str]) -> dict[str, float]:
        """Relevant or fresh items stay, long ones are compacted or archived, short unused ones go."""
        relevance = item["similarity_to_current_input"]
        freshness = 1.0 / (1.0 + item["steps_since_last_use"])
        long = item["tokens"] >= 30
        weights = {
            "KEEP": 0.2 + 2.0 * relevance + 1.5 * freshness,
            "EVICT": 1.2 if not long and item["times_used"] == 0 else 0.2,
            "MOVE_TO_ARCHIVE": 1.0,
            "COMPACT": 1.4 if long and item["fidelity"] == "FULL" else 0.05,
            "COMPACT_AND_ARCHIVE": 1.2 if long and item["fidelity"] == "FULL" else 0.05,
            "RETRIEVE_FROM_ARCHIVE": 0.1 + 3.0 * relevance,
        }
        return {option: weights[option] for option in options}


def make_jev_client(config: dict) -> JevClient:
    """The real client if a key is present; the fake one only when explicitly allowed."""
    key = os.environ.get(API_KEY_ENV)
    if key:
        return TypeSafeJevClient(key, config.get("model", "jev-latest"), config.get("endpoint", ENDPOINT))
    if config.get("allow_fake"):
        return FakeJevClient()
    raise JevUnavailableError(
        f"No JEV API key found in ${API_KEY_ENV}. Set it to run the JEV controller, or set "
        "controller.allow_fake: true to run a hand-written stand-in (all results are then labelled FAKE)."
    )


def credentials_available() -> bool:
    return bool(os.environ.get(API_KEY_ENV))


class JEVController(MemoryController):
    name = "jev"

    def __init__(self, config: dict, seed: int = 0, client: JevClient | None = None) -> None:
        super().__init__(config, seed)
        self.client = client or make_jev_client(config)
        self.model_id = "FAKE-jev" if self.client.is_fake else config.get("model", "jev-latest")
        self.call = config.get("call", "pressure")
        self.repair = bool(config.get("repair", True))
        self.batch = int(config.get("max_items_per_request", 40))
        self.retrieve_candidates = int(config.get("retrieve_candidates", 8))
        self.min_compact_tokens = int(config.get("min_compact_tokens", 30))
        self.compact_ratio = float(config.get("compact_ratio", 0.5))
        self._retriever = LexicalRetriever()
        self._info: dict = {}
        self.call_log: list[dict] = []  # one entry per request; never contains the API key

    @property
    def display_name(self) -> str:
        return self.config.get("label") or ("jev-FAKE" if self.client.is_fake else "jev")

    def reset(self, episode: EpisodeInfo) -> None:
        super().reset(episode)
        self.call_log = []

    # ---- preprocessing: MemoryView -> what JEV reads ---------------------------

    def _describe(self, item: MemoryItem, memory: MemoryView, task: TaskState) -> dict:
        current = memory.get(task.observation.id)
        return {
            "text": item.content,
            "source": item.source_type.value,
            "tokens": item.token_count,
            "age_steps": memory.step - item.created_at,
            "times_used": item.access_count,
            "steps_since_last_use": memory.step - item.last_accessed_at,
            "fidelity": item.fidelity.value,
            "was_retrieved_before": item.retrieval_count > 0,
            "similarity_to_current_input": round(
                max(0.0, cosine(item.embedding, current.embedding if current else None)), 4
            ),
            "is_current_input": item.id == task.observation.id,
        }

    def _options(self, item: MemoryItem, archived: bool) -> list[Operation]:
        if archived:
            return [Operation.RETRIEVE_FROM_ARCHIVE, Operation.KEEP]
        options = [op for op in ACTIVE_OPTIONS if op is Operation.KEEP or self.allows(op)]
        if item.fidelity is not Fidelity.FULL or item.token_count < self.min_compact_tokens:
            options = [op for op in options if op not in (Operation.COMPACT, Operation.COMPACT_AND_ARCHIVE)]
        return options

    def _question(self, key: str, options: list[Operation], archived: bool) -> dict:
        if archived:
            instructions = (
                f"The memory item `memory_items.{key}` is in the archive. Should it be brought back into "
                "context so the assistant can use it for the current input?"
            )
            criteria = {
                Operation.RETRIEVE_FROM_ARCHIVE.value: OPTION_HELP[Operation.RETRIEVE_FROM_ARCHIVE],
                Operation.KEEP.value: "Leave it in the archive. It is not needed for the current input.",
            }
        else:
            instructions = (
                f"What should be done with the memory item `memory_items.{key}` so that the assistant can "
                "still do its later work within the context budget?"
            )
            criteria = {op.value: OPTION_HELP[op] for op in options}
        return {"type": "choice", "instructions": instructions, "criteria": criteria}

    # ---- the decision -------------------------------------------------------------

    def decide(self, memory: MemoryView, task: TaskState) -> list[MemoryAction]:
        self._info = {}
        shortlist: list[MemoryItem] = []
        if self.allows(Operation.RETRIEVE_FROM_ARCHIVE) and task.observation.requires_response and memory.archived:
            hits = self._retriever.search(task.observation.content, memory.archived, self.retrieve_candidates)
            shortlist = [item for item, _ in hits]
        judge_active = memory.over_budget or self.call == "every_step"
        active = [item for item in memory.active if not item.pinned] if judge_active else []
        if not active and not shortlist:
            return []

        candidates = [(item, False) for item in active] + [(item, True) for item in shortlist]
        answers: dict[str, JevAnswer] = {}
        totals = {"model_calls": 0, "input_tokens": 0, "output_tokens": 0, "latency_s": 0.0, "cost_usd": 0.0}
        for start in range(0, len(candidates), self.batch):
            batch = candidates[start : start + self.batch]
            keys = {item.id: f"item_{n}" for n, (item, _) in enumerate(batch)}
            state = {
                "situation": "An AI assistant works on a long task. Its context has limited space, so memory "
                "items must be kept, shortened, archived or deleted. Later it will need some of them.",
                "context_budget_tokens": memory.budget,
                "tokens_in_context_now": memory.active_tokens,
                "tokens_to_free": max(0, memory.active_tokens - memory.budget),
                "current_input": task.observation.content,
                "memory_items": {keys[item.id]: self._describe(item, memory, task) for item, _ in batch},
            }
            questions = {
                keys[item.id]: self._question(keys[item.id], self._options(item, archived), archived)
                for item, archived in batch
            }
            response = self.client.ask(state, questions)
            for item, _ in batch:
                answers[item.id] = response.answers[keys[item.id]]
            totals["model_calls"] += 1
            totals["input_tokens"] += response.input_tokens
            totals["output_tokens"] += response.output_tokens
            totals["latency_s"] += response.latency_s
            totals["cost_usd"] += response.cost_usd
            self.call_log.append(
                {"step": memory.step, "fake": response.fake, "model": response.model, "items": len(batch),
                 "input_tokens": response.input_tokens, "latency_s": response.latency_s, "cost_usd": response.cost_usd}
            )
            self.model_id = response.model

        actions, repaired = self._actions(memory, active, shortlist, answers, task.observation.content)
        self._info = {
            **totals,
            "fake": self.client.is_fake,
            "probabilities": {item_id: answer.probabilities for item_id, answer in answers.items()},
            "confidence": {item_id: answer.confidence for item_id, answer in answers.items()},
            "choices": {item_id: answer.choice for item_id, answer in answers.items()},
            "repaired_items": repaired,
        }
        return actions

    def _saving(self, item: MemoryItem, operation: Operation) -> int:
        if operation in (Operation.COMPACT, Operation.COMPACT_AND_ARCHIVE):
            return item.token_count - math.ceil(self.compact_ratio * item.token_count)
        return item.token_count if operation in (Operation.EVICT, Operation.MOVE_TO_ARCHIVE) else 0

    def _actions(self, memory, active, shortlist, answers, query: str) -> tuple[list[MemoryAction], int]:
        chosen: dict[str, Operation] = {item.id: Operation(answers[item.id].choice) for item in active}
        retrieve = [item for item in shortlist if answers[item.id].choice == Operation.RETRIEVE_FROM_ARCHIVE.value]
        projected = memory.active_tokens + sum(item.token_count for item in retrieve)
        projected -= sum(self._saving(item, chosen[item.id]) for item in active)

        repaired: set[str] = set()
        if self.repair and projected > memory.budget:
            kept = [item for item in active if chosen[item.id] is Operation.KEEP]
            kept.sort(key=lambda item: answers[item.id].probabilities.get("KEEP", 0.0))
            for item in kept:
                if projected <= memory.budget:
                    break
                others = {k: v for k, v in answers[item.id].probabilities.items() if k != "KEEP"}
                if not others:
                    continue
                chosen[item.id] = Operation(max(others, key=others.get))
                projected -= self._saving(item, chosen[item.id])
                repaired.add(item.id)

        actions = []
        if retrieve:
            actions.append(
                MemoryAction(
                    Operation.RETRIEVE_FROM_ARCHIVE, tuple(item.id for item in retrieve),
                    parameters={"method": "jev", "query": query},
                    confidence=min(answers[item.id].confidence for item in retrieve),
                )
            )
        for item in active:
            operation = chosen[item.id]
            if operation is Operation.KEEP:
                continue
            parameters = {"repaired": True} if item.id in repaired else {}
            if operation in (Operation.COMPACT, Operation.COMPACT_AND_ARCHIVE):
                parameters["ratio"] = self.compact_ratio
            actions.append(MemoryAction(operation, (item.id,), parameters=parameters, confidence=answers[item.id].confidence))
        return actions, len(repaired)

    def decision_info(self) -> dict:
        return self._info
