"""The shared shape of every benchmark: a conversation is a list of events.

Evidence labels live on `Question.evidence_ids`. Only the evaluator and the
oracle read them. Controllers and features never see Question objects.
"""

from __future__ import annotations

from dataclasses import dataclass

from memctl.items import Item


@dataclass(frozen=True)
class Question:
    id: str
    text: str
    gold_answer: str
    category: str  # single-hop | multi-hop | temporal | open-domain | adversarial
    evidence_ids: tuple[str, ...]  # ids of the items needed to answer (may be empty)


@dataclass(frozen=True)
class Event:
    """One thing that happens: a new item arrives, a session ends, or a question is asked."""

    kind: str  # item | session_end | question
    item: Item | None = None
    question: Question | None = None


@dataclass
class Conversation:
    id: str
    events: list[Event]

    def items(self) -> list[Item]:
        return [e.item for e in self.events if e.kind == "item"]

    def questions(self) -> list[Question]:
        return [e.question for e in self.events if e.kind == "question"]

    def total_tokens(self) -> int:
        """Length of the full history in tokens (the 100% budget)."""
        return sum(item.tokens for item in self.items())

    def need_times(self) -> dict[str, list[int]]:
        """For each evidence item, the steps at which a question needs it (evaluation only)."""
        needs: dict[str, list[int]] = {}
        for step, event in enumerate(self.events, start=1):
            if event.kind == "question":
                for item_id in event.question.evidence_ids:
                    needs.setdefault(item_id, []).append(step)
        return needs
