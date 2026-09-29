"""Runs one conversation: feed events to the memory, call the controller, ask questions."""

from __future__ import annotations

from dataclasses import dataclass, field

from memctl.agent import Agent
from memctl.benchmarks.types import Conversation, Question
from memctl.controllers.base import Controller, Placement
from memctl.features import compute_features
from memctl.items import Place
from memctl.memory import MemoryState
from memctl.metrics import token_f1


@dataclass
class ConversationResult:
    decisions: list[dict] = field(default_factory=list)  # rows of decisions.jsonl
    answers: list[dict] = field(default_factory=list)  # rows of answers.jsonl
    context_tokens: list[int] = field(default_factory=list)  # budgeted tokens after each event
    controller_calls: int = 0


def run_controller(
    state: MemoryState, controller: Controller, trigger: str, query: str, conversation_id: str
) -> list[dict]:
    """Ask the controller for placements, apply them, and return the decision rows."""
    state.trigger = trigger
    candidates = state.candidates()
    features = compute_features(state, candidates, query)
    placements: list[Placement] = controller.decide(state, features, state.budget)
    state.apply(placements)  # raises if the budget or a placement rule is broken
    return [
        {
            "conversation_id": conversation_id,
            "step": state.step,
            "trigger": trigger,
            "controller": controller.display_name,
            "item_id": p.item_id,
            "features": features[p.item_id].as_dict(),
            "place": p.place.value,
            "reason": p.reason,
            "confidence": p.confidence,
            "candidates_considered": len(candidates),
            "oracle_place": None,  # filled in Phase 2
            "agrees_with_oracle": None,
        }
        for p in placements
    ]


def evidence_report(state: MemoryState, question: Question, retrieved: list[str], recalled: list[str]) -> list[dict]:
    """Where each needed evidence item was when the question was asked."""
    report = []
    for item_id in question.evidence_ids:
        place = state.place.get(item_id)
        available = place == Place.CONTEXT or item_id in retrieved or item_id in recalled
        report.append(
            {
                "item_id": item_id,
                "place": place.value if place else "NOT_SEEN_YET",
                "retrieved_from_store": item_id in retrieved,
                "recalled_from_archive": item_id in recalled,
                "in_prompt": available,
            }
        )
    return report


def run_conversation(
    conversation: Conversation, controller: Controller, agent: Agent, state: MemoryState
) -> ConversationResult:
    result = ConversationResult()
    last_text = ""
    for step, event in enumerate(conversation.events, start=1):
        state.step = step
        if event.kind == "item":
            last_text = event.item.text
            state.mark_referenced(event.item.text)
            fits = state.offer(event.item)
            if not fits or controller.runs_on_every_arrival:
                trigger = "arrival" if fits else "overflow"
                result.decisions += run_controller(state, controller, trigger, last_text, conversation.id)
                result.controller_calls += 1
        elif event.kind == "session_end":
            result.decisions += run_controller(state, controller, "session_end", last_text, conversation.id)
            result.controller_calls += 1
        elif event.kind == "question":
            result.answers.append(ask(state, agent, event.question, conversation.id))
        result.context_tokens.append(state.used_tokens())
    return result


def ask(state: MemoryState, agent: Agent, question: Question, conversation_id: str) -> dict:
    """Ask one question and build its answers.jsonl row."""
    state.mark_referenced(question.text)
    answer = agent.answer(state, question.text)
    return {
        "conversation_id": conversation_id,
        "question_id": question.id,
        "step": state.step,
        "category": question.category,
        "question": question.text,
        "gold_answer": question.gold_answer,
        "model_answer": answer.text,
        "scores": {"f1": round(token_f1(answer.text, question.gold_answer), 4)},
        "evidence": evidence_report(state, question, answer.retrieved_ids, answer.recalled_ids),
        "retrieved_ids": answer.retrieved_ids,
        "recalled_ids": answer.recalled_ids,
        "failed_recalls": answer.failed_recalls,
        "prompt_tokens": answer.prompt_tokens,
    }
