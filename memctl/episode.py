"""Runs one conversation: feed events to the memory, call the controller, ask questions."""

from __future__ import annotations

from dataclasses import dataclass, field

from memctl.agent import Agent
from memctl.attribution import failure_label
from memctl.benchmarks.types import Conversation, Question
from memctl.controllers.base import Controller, Placement
from memctl.controllers.oracle import OraclePlan
from memctl.features import compute_features
from memctl.items import Place
from memctl.judge import Judge
from memctl.memory import MemoryState
from memctl.metrics import score_answer

F1_CORRECT_THRESHOLD = 0.5  # used to decide right/wrong only when no judge is configured


@dataclass
class ConversationResult:
    decisions: list[dict] = field(default_factory=list)  # rows of decisions.jsonl
    answers: list[dict] = field(default_factory=list)  # rows of answers.jsonl
    context_tokens: list[int] = field(default_factory=list)  # budgeted tokens after each event
    controller_calls: int = 0


def run_controller(
    state: MemoryState,
    controller: Controller,
    trigger: str,
    query: str,
    conversation_id: str,
    plan: OraclePlan | None = None,
) -> list[dict]:
    """Ask the controller for placements, apply them, and return the decision rows.

    `plan` is the oracle's hindsight plan. It is used only AFTER the controller
    has decided, to note what the oracle would have done with the same item.
    """
    state.trigger = trigger
    candidates = state.candidates()
    features = compute_features(state, candidates, query)
    placements: list[Placement] = controller.decide(state, features, state.budget)
    state.apply(placements)  # raises if the budget or a placement rule is broken
    rows = []
    for p in placements:
        oracle_place = plan.place_for(p.item_id, state.step).value if plan else None
        rows.append(
            {
                "kind": "move" if p.place != Place.CONTEXT else "keep_new_item",
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
                "oracle_place": oracle_place,
                "agrees_with_oracle": (p.place.value == oracle_place) if plan else None,
            }
        )
    rows.append(keep_row(state, controller, trigger, conversation_id, plan, {p.item_id for p in placements}))
    return rows


def keep_row(state, controller, trigger, conversation_id, plan, already_logged: set[str]) -> dict:
    """One row per controller call listing every item it left in CONTEXT.

    Keeping an item is a decision too, so it is graded against the oracle like a move.
    """
    kept = [i.id for i in state.in_place(Place.CONTEXT) if i.id not in already_logged]
    oracle = {i: plan.place_for(i, state.step).value for i in kept} if plan else {}
    return {
        "kind": "keep",
        "conversation_id": conversation_id,
        "step": state.step,
        "trigger": trigger,
        "controller": controller.display_name,
        "kept_item_ids": kept,
        "place": "CONTEXT",
        "oracle_would_move": {i: place for i, place in oracle.items() if place != "CONTEXT"},
        "agreed": sum(1 for place in oracle.values() if place == "CONTEXT") if plan else None,
    }


def evidence_report(state: MemoryState, question: Question, retrieved: list[str], recalled: list[str]) -> list[dict]:
    """Where each needed evidence item was when the question was asked."""
    report = []
    for item_id in question.evidence_ids:
        place = state.place.get(item_id)
        available = place == Place.CONTEXT or item_id in retrieved or item_id in recalled
        report.append(
            {
                "item_id": item_id,
                "text": state.items[item_id].prompt_line() if item_id in state.items else "",
                "place": place.value if place else "NOT_SEEN_YET",
                "retrieved_from_store": item_id in retrieved,
                "recalled_from_archive": item_id in recalled,
                "in_prompt": available,
            }
        )
    return report


def run_conversation(
    conversation: Conversation,
    controller: Controller,
    agent: Agent,
    state: MemoryState,
    plan: OraclePlan | None = None,
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
                result.decisions += run_controller(state, controller, trigger, last_text, conversation.id, plan)
                result.controller_calls += 1
        elif event.kind == "session_end":
            result.decisions += run_controller(state, controller, "session_end", last_text, conversation.id, plan)
            result.controller_calls += 1
        elif event.kind == "question":
            result.answers.append(ask(state, agent, event.question, conversation.id))
        result.context_tokens.append(state.used_tokens())
    return result


def ask(state: MemoryState, agent: Agent, question: Question, conversation_id: str) -> dict:
    """Ask one question and record the answer, its scores and where the evidence was."""
    answer = agent.answer(state, question.text)
    return {
        "conversation_id": conversation_id,
        "question_id": question.id,
        "step": state.step,
        "category": question.category,
        "question": question.text,
        "gold_answer": question.gold_answer,
        "model_answer": answer.text,
        "scores": score_answer(answer.text, question.gold_answer, question.category),
        "evidence": evidence_report(state, question, answer.retrieved_ids, answer.recalled_ids),
        "retrieved_ids": answer.retrieved_ids,
        "recalled_ids": answer.recalled_ids,
        "failed_recalls": answer.failed_recalls,
        "prompt_tokens": answer.prompt_tokens,
    }


def grade(answers: list[dict], judge: Judge | None) -> None:
    """Decide right or wrong for every answer, and label each wrong one with its cause.

    This runs after all questions are answered, so the answering model can be
    unloaded before the judge model is loaded.
    """
    for row in answers:
        if judge:
            row["correct"] = judge.is_correct(row["question"], row["gold_answer"], row["model_answer"], row["category"])
            row["scores"]["judge"] = float(row["correct"])
            row["correct_decided_by"] = f"judge ({judge.name})"
        else:
            row["correct"] = row["scores"]["f1"] >= F1_CORRECT_THRESHOLD
            row["correct_decided_by"] = f"f1 >= {F1_CORRECT_THRESHOLD}"
        row["failure_label"] = None if row["correct"] else failure_label(row["evidence"])
