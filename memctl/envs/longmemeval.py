"""LongMemEval (Wu et al., 2025): 500 questions, each with its own chat history.

One episode is one question: seed s plays instance s mod N. The history is the
instance's sessions in date order, turn by turn, followed by the single
question. Turns marked `has_answer` are the evidence; when an instance marks
none, any turn of its answer sessions counts.

Files (download from the Hugging Face dataset `xiaowu0162/longmemeval-cleaned`):
- longmemeval_oracle.json      only the evidence sessions (15 MB); little memory pressure
- longmemeval_s_cleaned.json   about 115k tokens of history per question (277 MB)

The official metric is an LLM judge. Without `env.judge` the answer is scored
locally by token F1, which under-credits correct answers phrased differently.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from memctl.envs.qa import QAEnvironment, QAEpisode, QAItem
from memctl.memory.items import SourceType
from memctl.task import Dependency, EvidenceRequirement, Observation

SOURCE = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned"


@lru_cache(maxsize=1)
def _load(path: str) -> list[dict]:
    return json.loads(Path(path).read_text())


def parse_instance(instance: dict, max_turn_tokens: int | None = None) -> tuple[QAEpisode, tuple[str, ...]]:
    """(episode, ids of the fallback evidence) for one LongMemEval instance."""
    order = sorted(range(len(instance["haystack_sessions"])), key=lambda n: instance["haystack_dates"][n])
    turns, text_of, marked, in_answer_sessions = [], {}, [], []
    answer_sessions = set(instance.get("answer_session_ids", []))
    for position in order:
        session_id = instance["haystack_session_ids"][position]
        date = instance["haystack_dates"][position]
        for number, turn in enumerate(instance["haystack_sessions"][position]):
            item_id = f"{session_id}:{number}"
            if item_id in text_of:  # a session id can repeat in a haystack
                item_id = f"{session_id}#{position}:{number}"
            text = turn["content"]
            if max_turn_tokens:
                from memctl.memory.items import truncate_tokens

                text = truncate_tokens(text, max_turn_tokens)
            source = SourceType.USER if turn["role"] == "user" else SourceType.OBSERVATION
            metadata = {"speaker": turn["role"], "session": session_id, "date": date}
            turns.append(Observation(item_id, text, source, metadata=metadata))
            text_of[item_id] = text
            if turn.get("has_answer"):
                marked.append(item_id)
            if session_id in answer_sessions:
                in_answer_sessions.append(item_id)
    unanswerable = str(instance["question_id"]).endswith("_abs")
    question = QAItem(
        str(instance["question_id"]),
        f"(asked on {instance.get('question_date', '')}) {instance['question']}",
        str(instance["answer"]), instance["question_type"], tuple(marked), unanswerable,
    )
    return QAEpisode(str(instance["question_id"]), turns, [question], text_of), tuple(in_answer_sessions)


class LongMemEvalEnv(QAEnvironment):
    name = "longmemeval"

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.path = config.get("path", "data/longmemeval/longmemeval_oracle.json")
        self.max_turn_tokens = config.get("max_turn_tokens")
        # Optional [start, end): seed s plays question start + s mod (end - start), so a training
        # run can be kept off the questions an evaluation uses (seeds 0-99 play questions 0-99).
        self.subset = tuple(config["subset"]) if config.get("subset") else None
        self._fallback: tuple[str, ...] = ()
        if not Path(self.path).exists():
            raise FileNotFoundError(f"LongMemEval not found at {self.path}. Download a split from {SOURCE}")

    def load_episode(self, seed: int) -> QAEpisode:
        instances = _load(self.path)
        start, end = self.subset or (0, len(instances))
        episode, self._fallback = parse_instance(instances[start + seed % (end - start)], self.max_turn_tokens)
        return episode

    def get_ground_truth_dependencies(self) -> list[Dependency]:
        dependencies = super().get_ground_truth_dependencies()
        for number, dependency in enumerate(dependencies):
            if not dependency.requirements and self._fallback:
                # No turn is marked: any turn of the answer sessions is accepted as evidence.
                requirement = EvidenceRequirement(self._fallback, "")
                dependencies[number] = Dependency(
                    dependency.query_id, dependency.step, (requirement,), dependency.gold, dependency.category
                )
        return dependencies
