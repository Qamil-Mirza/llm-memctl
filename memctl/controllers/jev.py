"""The Jev controller: asks TypeSafe's Jev model where each item should live.

Jev answers typed questions about a `state`. We send one request per decision:
the state holds the memory items and their features, and there is one Choice
question per item with the places as options. See docs/jev.md.

Without an API key the controller can only run with `FakeJevClient`, and only
when `--allow-fake-jev` is passed. Fake results are labelled FAKE everywhere.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import asdict, dataclass
from typing import Protocol

from memctl.controllers.base import Controller, Placement, evict_until_fits
from memctl.features import ItemFeatures
from memctl.items import Place
from memctl.memory import MemoryState

API_KEY_ENV = "JEV_API_KEY"
PRICE_PER_MILLION_INPUT_TOKENS_USD = 0.042  # docs.typesafe.ai/models (output tokens are free)

PLACE_DESCRIPTIONS = {
    "CONTEXT": "Keep it in the prompt. For items the assistant is likely to need soon or often.",
    "STORE": "File it in searchable memory. It returns only when a later question is similar to it.",
    "ARCHIVE": "Put it in cold storage. Only a one-line label stays visible, and reading it again "
    "costs extra. For long items that are rarely needed but must not be lost.",
    "DROPPED": "Delete it for good. For greetings, filler and anything with no lasting information.",
}


class JevUnavailableError(RuntimeError):
    """Raised when the real Jev API cannot be used and fake Jev was not allowed."""


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
    """Anything that can answer Choice questions about a state."""

    is_fake: bool

    def ask(self, state: dict, questions: dict[str, dict]) -> JevResponse: ...


def choice_confidence(probabilities: dict[str, float]) -> float:
    """Confidence of a Choice as documented by TypeSafe: (n * peak - 1) / (n - 1)."""
    n = len(probabilities)
    if n < 2:
        return 1.0
    return max(0.0, min(1.0, (n * max(probabilities.values()) - 1) / (n - 1)))


class FakeJevClient:
    """A stand-in for Jev used in tests. It is a hand-written heuristic, NOT the real model."""

    is_fake = True

    def ask(self, state: dict, questions: dict[str, dict]) -> JevResponse:
        started = time.perf_counter()
        answers = {}
        for question_id, question in questions.items():
            key = re.search(r"`memory_items\.(\w+)`", question["instructions"]).group(1)
            weights = self._weights(state["memory_items"][key], list(question["criteria"]))
            total = sum(weights.values())
            probabilities = {option: round(w / total, 4) for option, w in weights.items()}
            best = max(probabilities, key=probabilities.get)
            answers[question_id] = JevAnswer(best, probabilities, round(choice_confidence(probabilities), 4))
        return JevResponse(answers, "FAKE-jev", 0, 0, time.perf_counter() - started, 0.0, fake=True)

    def _weights(self, item: dict, options: list[str]) -> dict[str, float]:
        """Relevant or fresh items stay, long ones are archived, short unused ones are dropped."""
        relevance = item["similarity_to_current_message"]
        freshness = 1.0 / (1.0 + item["steps_since_last_use"])
        words = len(item["text"].split())
        unused = item["times_referenced"] == 0
        weights = {
            "CONTEXT": 0.2 + 2.0 * relevance + 1.5 * freshness,
            "STORE": 1.0,
            "ARCHIVE": 1.6 if words >= 15 else 0.3,
            "DROPPED": 1.3 if words <= 7 and unused else 0.1,
        }
        return {option: weights[option] for option in options}


def make_jev_client(allow_fake: bool) -> JevClient:
    """Pick the Jev client. Fails loudly unless a key exists or fake Jev is allowed."""
    if os.environ.get(API_KEY_ENV):
        raise JevUnavailableError(
            f"{API_KEY_ENV} is set, but the real Jev client is added in Phase 3 (see docs/jev.md)."
        )
    if allow_fake:
        return FakeJevClient()
    raise JevUnavailableError(
        f"No Jev API key found in ${API_KEY_ENV}. Set it, or pass --allow-fake-jev to run with the "
        "fake client (results will be labelled FAKE)."
    )


class JevController(Controller):
    """Asks Jev for a place for every candidate item, then makes sure the result fits."""

    name = "jev"

    def __init__(self, config: dict, seed: int = 0, client: JevClient | None = None) -> None:
        super().__init__(config, seed)
        self.client = client or make_jev_client(bool(config.get("allow_fake_jev", False)))
        self.call_log: list[dict] = []  # one entry per request; never contains the API key

    @property
    def display_name(self) -> str:
        return "jev-FAKE" if self.client.is_fake else "jev"

    def decide(self, state: MemoryState, features: dict[str, ItemFeatures], budget: int) -> list[Placement]:
        candidates = state.candidates()
        if not candidates:
            return []
        keys = {item.id: f"item_{n}" for n, item in enumerate(candidates)}
        options = [p.value for p in Place if p != Place.DROPPED or state.allow_drop]
        request_state = self._build_state(state, features, keys)
        questions = {keys[item.id]: self._question(keys[item.id], options) for item in candidates}
        response = self.client.ask(request_state, questions)
        self._log(state, request_state, questions, response)

        answers = {item.id: response.answers[keys[item.id]] for item in candidates}
        tag = "FAKE Jev" if response.fake else "Jev"
        placements = {}
        for item in candidates:
            answer = answers[item.id]
            moved = answer.choice != Place.CONTEXT.value
            if moved or item is state.incoming:
                placements[item.id] = Placement(
                    item.id, Place(answer.choice), f"{tag} chose {answer.choice}", answer.confidence
                )
        return self._make_it_fit(state, placements, answers, tag)

    def _make_it_fit(self, state, placements, answers, tag) -> list[Placement]:
        """If Jev kept too much in CONTEXT, evict the items it was least sure should stay."""
        planned = state.planned_places(list(placements.values()))
        if state.used_tokens(planned) <= state.budget:
            return list(placements.values())

        def next_best_place(item) -> Place:
            others = {k: v for k, v in answers[item.id].probabilities.items() if k != "CONTEXT"}
            return Place(max(others, key=others.get))

        kept = [item for item in state.candidates() if planned.get(item.id) == Place.CONTEXT]
        weakest_first = sorted(kept, key=lambda item: answers[item.id].probabilities["CONTEXT"])
        extra = evict_until_fits(
            state,
            weakest_first,
            destination=next_best_place,
            reason=lambda item: f"{tag} chose CONTEXT but it did not fit; moved to its next-best place",
            confidence=lambda item: answers[item.id].confidence,
            start_from=planned,
        )
        for placement in extra:
            if placement.place != Place.CONTEXT:  # keep Jev's own entry for items that stay
                placements[placement.item_id] = placement
        return list(placements.values())

    def _build_state(self, state: MemoryState, features: dict[str, ItemFeatures], keys: dict) -> dict:
        memory_items = {}
        for item in state.candidates():
            f = features[item.id]
            memory_items[keys[item.id]] = {
                "text": item.text,
                "speaker": item.speaker(),
                "type": item.kind,
                "tokens": f.tokens,
                "age_steps": f.age_steps,
                "times_referenced": f.times_referenced,
                "steps_since_last_use": f.steps_since_last_use,
                "similarity_to_current_message": f.similarity_to_query,
                "is_newest_message": f.is_incoming,
            }
        return {
            "situation": "An AI assistant is having a long conversation. Its prompt has limited "
            "space, so each memory item must be kept in one place. Later it will be asked "
            "questions about facts from this conversation.",
            "reason_for_deciding_now": state.trigger,
            "prompt_budget_tokens": state.budget,
            "memory_items": memory_items,
        }

    def _question(self, key: str, options: list[str]) -> dict:
        return {
            "type": "choice",
            "instructions": f"Where should the memory item `memory_items.{key}` be kept so that the "
            "assistant can still answer later questions about this conversation?",
            "criteria": {option: PLACE_DESCRIPTIONS[option] for option in options},
        }

    def _log(self, state: MemoryState, request_state: dict, questions: dict, response: JevResponse) -> None:
        self.call_log.append(
            {
                "step": state.step,
                "fake": response.fake,
                "request": {"state": request_state, "questions": questions},
                "response": asdict(response),
                "latency_s": round(response.latency_s, 4),
                "cost_usd": response.cost_usd,
            }
        )

