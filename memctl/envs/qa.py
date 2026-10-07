"""Shared machinery for question-answering benchmarks (LoCoMo, LongMemEval).

An episode is a history of turns followed by questions. These are memory
evaluations: every question comes after the history has ended, so they test
retention, archival, retrieval and evidence preservation under a budget. They
are not long-horizon control tasks, because nothing the agent does changes what
it sees next.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from memctl.envs.base import TaskEnvironment
from memctl.memory.items import SourceType
from memctl.metrics import score_answer
from memctl.task import Dependency, EvidenceRequirement, Observation, StepResult


@dataclass
class QAItem:
    id: str
    question: str
    gold: str
    category: str
    evidence_ids: tuple[str, ...]
    unanswerable: bool = False
    after: str | None = None  # ask right after this turn; None: after the whole history


@dataclass
class QAEpisode:
    id: str
    turns: list[Observation]
    questions: list[QAItem]
    turn_text: dict[str, str] = field(default_factory=dict)


class QAEnvironment(TaskEnvironment):
    """Shows the turns of one episode in order, then asks its questions."""

    goal = (
        "Below is your memory of a long conversation, followed by a question about it. Answer with a short "
        "phrase that states the fact asked for. If the memory does not contain the answer, reply: unknown"
    )

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.max_questions = config.get("max_questions")
        self.judge = None
        if config.get("judge"):
            from memctl.judge import Judge

            self.judge = Judge(config["judge"])
        self._episode: QAEpisode | None = None
        self._observations: list[Observation] = []
        self._questions: dict[str, QAItem] = {}
        self._index = 0
        self._last_reward = 0.0
        self._scores: list[dict] = []

    def load_episode(self, seed: int) -> QAEpisode:
        raise NotImplementedError

    def reset(self, seed: int) -> Observation:
        episode = self.load_episode(seed)
        questions = episode.questions[: self.max_questions] if self.max_questions else episode.questions
        self._episode = episode
        self._questions = {f"q{number:04d}": question for number, question in enumerate(questions)}
        asked_after: dict[str | None, list[Observation]] = {}
        for query_id, question in self._questions.items():
            asked_after.setdefault(question.after, []).append(
                Observation(query_id, f"Question: {question.question}", SourceType.USER, requires_response=True))
        self._observations = []
        for turn in episode.turns:
            self._observations.append(turn)
            self._observations.extend(asked_after.pop(turn.id, []))
        self._observations.extend(asked_after.pop(None, []))
        self._observations.extend(o for rest in asked_after.values() for o in rest)  # an `after` not in the history
        self._step_of = {o.id: n + 1 for n, o in enumerate(self._observations)}
        self.horizon = len(self._observations)
        self._index, self._last_reward, self._scores = 0, 0.0, []
        return self._observations[0]

    def step(self, agent_action: str | None) -> StepResult:
        current = self._observations[self._index]
        info: dict = {"scored": False, "correct": None}
        self._last_reward = 0.0
        if current.requires_response:
            question = self._questions[current.id]
            scores = score_answer(agent_action or "", question.gold, question.unanswerable)
            if self.judge is not None and not question.unanswerable:
                scores["correct"] = self.judge.is_correct(question.question, question.gold, agent_action or "", question.category)
                scores["decided_by"] = f"judge ({self.judge.name})"
            self._scores.append({**scores, "category": question.category})
            self._last_reward = float(scores["correct"])
            info = {"scored": True, "correct": bool(scores["correct"]), "gold": question.gold, "query_id": current.id,
                    "f1": scores["f1"], "category": question.category}
        self._index += 1
        return StepResult(self.get_observation(), self._last_reward, self.is_done(), info)

    def get_observation(self) -> Observation | None:
        return None if self.is_done() else self._observations[self._index]

    def is_done(self) -> bool:
        return self._index >= len(self._observations)

    def get_reward(self) -> float:
        return self._last_reward

    def task_success(self) -> float:
        return sum(1 for s in self._scores if s["correct"]) / len(self._scores) if self._scores else 0.0

    def get_ground_truth_dependencies(self) -> list[Dependency]:
        seen = {observation.id for observation in self._observations[: self._index + 1]}
        dependencies = []
        for query_id, question in self._questions.items():
            if query_id not in seen:
                continue
            requirements = tuple(
                EvidenceRequirement((evidence_id,), self._episode.turn_text.get(evidence_id, ""))
                for evidence_id in question.evidence_ids
            )
            dependencies.append(Dependency(query_id, self._step_of[query_id], requirements, question.gold, question.category))
        return dependencies

    def episode_stats(self) -> dict:
        categories: dict[str, list[float]] = {}
        for score in self._scores:
            categories.setdefault(score["category"], []).append(float(score["correct"]))
        return {
            "episode": self._episode.id if self._episode else None,
            "mean_f1": sum(s["f1"] for s in self._scores) / len(self._scores) if self._scores else None,
            "accuracy_by_category": {name: sum(v) / len(v) for name, v in categories.items()},
            "scored_by": "judge" if self.judge is not None else "local f1",
        }
