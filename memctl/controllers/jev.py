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

Where the requests go (EXPERIMENTS.md §26). The endpoint is, in order: config
`endpoint` (a full URL), config `base_url`, `$JEV_BASE_URL` (both a server root;
`/v1/systemone` is appended), else TypeSafe's API. The model name is config
`model`, else `$JEV_MODEL`, else `jev-latest`. A server other than TypeSafe's
(for example a self-hosted OpenJev, which speaks the same wire API) never gets
the TypeSafe key: it gets `$OPENJEV_API_KEY` if set, else no Authorization
header, and its `cost_usd` is 0 (its cost is the GPU time it runs on). With
nothing set, the requests and prices are exactly as before.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from memctl.controllers.base import EpisodeInfo, MemoryController
from memctl.embed import cosine
from memctl.memory.actions import MemoryAction, Operation
from memctl.memory.items import Fidelity, MemoryItem, count_tokens
from memctl.memory.state import MemoryView
from memctl.retrieval import LexicalRetriever
from memctl.task import TaskState

API_KEY_ENV = "JEV_API_KEY"
BASE_URL_ENV = "JEV_BASE_URL"
MODEL_ENV = "JEV_MODEL"
SELF_HOSTED_KEY_ENV = "OPENJEV_API_KEY"  # the key of a self-hosted server; the TypeSafe key never goes there
SYSTEMONE_PATH = "/v1/systemone"
DEFAULT_MODEL = "jev-latest"
ENDPOINT = "https://api.typesafe.ai/v1/systemone"
RETRY_STATUSES = (429, 503, 529)  # rate limit, backend unavailable, overloaded: all carry retry-after
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
    server_model_s: float | None = None  # the server's own model time, from a `Server-Timing: model;dur=` header
    cached: bool = False  # replayed from the client's cache; the timings are those of the original call


class JevClient(Protocol):
    is_fake: bool

    def ask(self, state: dict, questions: dict[str, dict]) -> JevResponse: ...


def choice_confidence(probabilities: dict[str, float]) -> float:
    """Confidence of a Choice as documented by TypeSafe: (n * peak - 1) / (n - 1)."""
    n = len(probabilities)
    if n < 2:
        return 1.0
    return max(0.0, min(1.0, (n * max(probabilities.values()) - 1) / (n - 1)))


def resolve_endpoint(config: dict) -> str:
    """Config `endpoint`, then config `base_url`, then $JEV_BASE_URL, then TypeSafe's API."""
    if config.get("endpoint"):
        return str(config["endpoint"])
    base = config.get("base_url") or os.environ.get(BASE_URL_ENV)
    return str(base).rstrip("/") + SYSTEMONE_PATH if base else ENDPOINT


def resolve_model(config: dict) -> str:
    return str(config.get("model") or os.environ.get(MODEL_ENV) or DEFAULT_MODEL)


def server_model_seconds(header: str | None) -> float | None:
    """`model;dur=41.2, server;dur=2.8, total;dur=44.0` -> 0.0412 (OpenJev's Server-Timing header)."""
    for part in (header or "").split(","):
        name, _, rest = part.strip().partition(";")
        if name == "model" and rest.startswith("dur="):
            try:
                return float(rest[4:]) / 1000.0
            except ValueError:
                return None
    return None


class TypeSafeJevClient:
    """The real client: one HTTPS request per call. The key is sent in a header and never logged.

    It also serves any server with the same wire API (a self-hosted OpenJev): `api_key` may then be None (no
    Authorization header) and `usd_per_million` 0. `retries` re-sends a request that got 429, 503 or 529, after
    the server's retry-after; the default 0 keeps the original behaviour (the error is raised).

    `cache_dir` (off by default) stores each answer under a hash of the request body (model, state, questions; not
    the endpoint), with the latency and server time of the call that made it. A replay returns the stored answer
    and its original timings, marked `cached`. With `cache_only` a request that is not stored is an error, so a
    replay can never call a server (EXPERIMENTS.md §26: OpenJev's pod first, the reader's pod after).
    """

    is_fake = False

    def __init__(
        self, api_key: str | None, model: str = DEFAULT_MODEL, endpoint: str = ENDPOINT, timeout_s: float = 60.0,
        usd_per_million: float = USD_PER_MILLION_INPUT_TOKENS, retries: int = 0, cache_dir: str | None = None,
        cache_only: bool = False,
    ) -> None:
        self._api_key = api_key
        self.model = model
        self.endpoint = endpoint
        self.timeout_s = timeout_s
        self.usd_per_million = usd_per_million
        self.retries = retries
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.cache_only = cache_only
        if cache_only and not cache_dir:
            raise ValueError("cache_only needs a cache_dir")

    def build_request(self, state: dict, questions: dict[str, dict]) -> dict:
        return {"model": self.model, "state": state, "questions": questions}

    def ask(self, state: dict, questions: dict[str, dict]) -> JevResponse:
        body = json.dumps(self.build_request(state, questions)).encode()
        cache_file = None
        if self.cache_dir:
            cache_file = self.cache_dir / f"{hashlib.sha256(body).hexdigest()}.json"
            if cache_file.exists():
                entry = json.loads(cache_file.read_text())
                response = self.parse_response(
                    entry["payload"], entry["latency_s"], count_tokens(body.decode()), entry.get("server_model_s")
                )
                return dataclasses.replace(response, cached=True)
            if self.cache_only:
                raise JevUnavailableError(f"cache_only: no stored answer for this request in {self.cache_dir}")
        headers = {"Content-Type": "application/json", "User-Agent": "memctl"}  # proxies refuse the urllib default (403)
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        started = time.perf_counter()
        for attempt in range(self.retries + 1):
            request = urllib.request.Request(self.endpoint, data=body, headers=headers)
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                    payload = json.load(response)
                    timing = response.headers.get("server-timing")
                break
            except urllib.error.HTTPError as error:
                if error.code not in RETRY_STATUSES or attempt == self.retries:
                    raise
                time.sleep(min(30.0, float(error.headers.get("retry-after") or 2**attempt)))
        latency_s, server_s = time.perf_counter() - started, server_model_seconds(timing)
        if cache_file is not None:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            partial = cache_file.with_suffix(f".{os.getpid()}.tmp")
            partial.write_text(json.dumps({"payload": payload, "latency_s": latency_s, "server_model_s": server_s}))
            partial.replace(cache_file)  # atomic, so parallel workers never read half an entry
        return self.parse_response(
            payload, latency_s, fallback_tokens=count_tokens(body.decode()), server_model_s=server_s
        )

    def parse_response(
        self, payload: dict, latency_s: float, fallback_tokens: int = 0, server_model_s: float | None = None
    ) -> JevResponse:
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
            latency_s, input_tokens * self.usd_per_million / 1e6, fake=False, server_model_s=server_model_s,
        )


class FakeJevClient:
    """A stand-in for tests. A hand-written heuristic, NOT the JEV model: it says nothing about JEV."""

    is_fake = True

    def ask(self, state: dict, questions: dict[str, dict]) -> JevResponse:
        started = time.perf_counter()
        answers = {}
        for question_id, question in questions.items():
            items = state["memory_items"]
            if question_id in items:  # a question about one item: its operations
                weights = self._weights(items[question_id], list(question["criteria"]))
            else:  # a selection question whose options are the items: the more similar, the likelier
                weights = {key: 0.05 + items[key]["similarity_to_current_input"] for key in question["criteria"]}
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
    """The real client if a key is present; the fake one only when explicitly allowed.

    A server other than TypeSafe's (see the module docstring) needs no TypeSafe key and is never sent one.
    """
    endpoint, model = resolve_endpoint(config), resolve_model(config)
    options = {
        "timeout_s": float(config.get("timeout_s", 60.0)), "retries": int(config.get("retries", 0)),
        "cache_dir": config.get("cache_dir"), "cache_only": bool(config.get("cache_only", False)),
    }
    if endpoint != ENDPOINT:
        return TypeSafeJevClient(os.environ.get(SELF_HOSTED_KEY_ENV), model, endpoint, usd_per_million=0.0, **options)
    key = os.environ.get(API_KEY_ENV)
    if key:
        return TypeSafeJevClient(key, model, endpoint, **options)
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
        self.model_id = "FAKE-jev" if self.client.is_fake else resolve_model(config)
        self.call = config.get("call", "pressure")
        self.repair = bool(config.get("repair", True))
        self.batch = int(config.get("max_items_per_request", 40))
        self.retrieve_candidates = int(config.get("retrieve_candidates", 8))
        self.min_compact_tokens = int(config.get("min_compact_tokens", 30))
        self.compact_ratio = float(config.get("compact_ratio", 0.5))
        # mode "manage" (default): the JEV decisions above. Mode "select" (EXPERIMENTS.md §26, arm B): everything
        # but the current input is archived on arrival, as the §19 head's keep_none; at a question one request
        # asks a single Choice over the `retrieve_candidates` BM25 hits and the `select_k` most probable are shown.
        self.mode = config.get("mode", "manage")
        if self.mode not in ("manage", "select"):
            raise ValueError(f"unknown jev mode '{self.mode}' (known: manage, select)")
        self.select_k = int(config.get("select_k", 8))
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

    def _new_totals(self) -> dict:
        return {"model_calls": 0, "input_tokens": 0, "output_tokens": 0, "latency_s": 0.0, "cost_usd": 0.0,
                "server_model_s": 0.0, "cached_calls": 0}

    def _ask(self, state: dict, questions: dict, memory: MemoryView, n_items: int, totals: dict) -> JevResponse:
        response = self.client.ask(state, questions)
        totals["model_calls"] += 1
        totals["input_tokens"] += response.input_tokens
        totals["output_tokens"] += response.output_tokens
        totals["latency_s"] += response.latency_s
        totals["cost_usd"] += response.cost_usd
        totals["server_model_s"] += response.server_model_s or 0.0
        totals["cached_calls"] += int(response.cached)
        self.call_log.append(
            {"step": memory.step, "fake": response.fake, "model": response.model, "items": n_items,
             "input_tokens": response.input_tokens, "latency_s": response.latency_s, "cost_usd": response.cost_usd,
             "server_model_s": response.server_model_s, "cached": response.cached}
        )
        self.model_id = response.model
        return response

    def decide(self, memory: MemoryView, task: TaskState) -> list[MemoryAction]:
        self._info = {}
        if self.mode == "select":
            return self._select(memory, task)
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
        totals = self._new_totals()
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
            response = self._ask(state, questions, memory, len(batch), totals)
            for item, _ in batch:
                answers[item.id] = response.answers[keys[item.id]]

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

    def _select(self, memory: MemoryView, task: TaskState) -> list[MemoryAction]:
        """Arm B of §26: archive on arrival; at a question, one Choice over the BM25 shortlist ranks it."""
        cleared = [item.id for item in memory.active if not item.pinned and item.id != task.observation.id]
        actions = [MemoryAction(Operation.MOVE_TO_ARCHIVE, tuple(cleared), parameters={"method": "keep_none"})] \
            if cleared and self.allows(Operation.MOVE_TO_ARCHIVE) else []
        if not (self.allows(Operation.RETRIEVE_FROM_ARCHIVE) and task.observation.requires_response and memory.archived):
            return actions
        query = task.observation.content
        shortlist = [item for item, _ in self._retriever.search(query, memory.archived, self.retrieve_candidates)]
        totals, probabilities = self._new_totals(), {}
        chosen = shortlist  # a shortlist no longer than select_k is shown whole, as the head does; no call
        if len(shortlist) > self.select_k:
            keys = {item.id: f"item_{n}" for n, item in enumerate(shortlist)}
            state = {
                "situation": "An AI assistant must answer the current input. Its context has room for only "
                f"{self.select_k} of these archived memory items. Pick the one it most needs.",
                "current_input": query,
                "memory_items": {keys[item.id]: self._describe(item, memory, task) for item in shortlist},
            }
            question = {
                "type": "choice",
                "instructions": "Which memory item does the assistant most need to answer the current input?",
                "criteria": {keys[item.id]: f"memory_items.{keys[item.id]}" for item in shortlist},
            }
            response = self._ask(state, {"selection": question}, memory, len(shortlist), totals)
            by_key = response.answers["selection"].probabilities
            probabilities = {item.id: by_key.get(keys[item.id], 0.0) for item in shortlist}
            # The k most probable; ties keep the BM25 order (sorted is stable).
            chosen = sorted(shortlist, key=lambda item: -probabilities[item.id])[: self.select_k]
        actions.append(MemoryAction(
            Operation.RETRIEVE_FROM_ARCHIVE, tuple(item.id for item in chosen),
            parameters={"method": "jev_select", "query": query},
        ))
        self._info = {**totals, "fake": self.client.is_fake, "candidates": len(shortlist),
                      "selected_ids": [item.id for item in chosen], "probabilities": probabilities}
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
