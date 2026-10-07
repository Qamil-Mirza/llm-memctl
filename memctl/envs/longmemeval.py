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

import fcntl
import hashlib
import json
import os
import random
from functools import lru_cache
from pathlib import Path

from memctl.envs.qa import QAEnvironment, QAEpisode, QAItem
from memctl.splits import fold_indices  # noqa: F401  (re-exported)
from memctl.memory.items import SourceType
from memctl.task import Dependency, EvidenceRequirement, Observation

SOURCE = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned"


@lru_cache(maxsize=1)
def _load(path: str) -> list[dict]:
    """The whole file. About 2.3 GB of Python objects for the _s file: single-process tools only; an
    environment reads the index and one shard per instance instead (_index, _instance)."""
    return json.loads(Path(path).read_text())


@lru_cache(maxsize=4)
def _sha256(source: Path) -> str:
    digest = hashlib.sha256()
    with open(source, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=4)
def _shard_dir(path: str) -> Path:
    """`<file>.shards/`: index.json (type and id of every question, plus the source's size and mtime) and
    one NNNN.json per instance. Built once from the source, under a lock; rebuilt when the source's sha256
    differs from the one recorded, so shards of another file are never loaded."""
    source = Path(path)
    folder = source.with_name(source.name + ".shards")
    stamp = {"sha256": _sha256(source)}
    index = folder / "index.json"
    if index.exists() and json.loads(index.read_text()).get("source") == stamp:
        return folder
    folder.mkdir(exist_ok=True)
    with open(folder / ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)  # one builder; the others wait, then find it built
        if index.exists() and json.loads(index.read_text()).get("source") == stamp:
            return folder
        instances = json.loads(source.read_text())
        for number, instance in enumerate(instances):
            (folder / f"{number:04d}.json").write_text(json.dumps(instance))
        entries = [{"question_type": i["question_type"], "question_id": i["question_id"]} for i in instances]
        tmp = folder / "index.json.tmp"
        tmp.write_text(json.dumps({"source": stamp, "questions": entries}))
        os.replace(tmp, index)
    return folder


@lru_cache(maxsize=4)
def _index(path: str) -> list[dict]:
    """Question type and id of every instance, in file order (what the splits need)."""
    return json.loads((_shard_dir(path) / "index.json").read_text())["questions"]


@lru_cache(maxsize=8)
def _instance(path: str, number: int) -> dict:
    return json.loads((_shard_dir(path) / f"{number:04d}.json").read_text())


def parse_instance(instance: dict, max_turn_tokens: int | None = None,
                   prefix: str = "") -> tuple[QAEpisode, tuple[tuple[str, ...], ...]]:
    """(episode, fallback evidence) for one LongMemEval instance.

    The fallback has one entry per answer session with no turn marked `has_answer`: that session's
    turn ids, any one of which counts. 41 of the 500 instances mark turns in only some of their
    answer sessions, and 21 mark none. `prefix` is put before every turn id, so instances can share an
    episode (filler sessions repeat across instances)."""
    order = sorted(range(len(instance["haystack_sessions"])), key=lambda n: instance["haystack_dates"][n])
    turns, text_of, marked = [], {}, []
    session_turns: dict[str, list[str]] = {}
    marked_sessions: set[str] = set()
    answer_sessions = set(instance.get("answer_session_ids", []))
    for position in order:
        session_id = instance["haystack_session_ids"][position]
        date = instance["haystack_dates"][position]
        for number, turn in enumerate(instance["haystack_sessions"][position]):
            item_id = f"{prefix}{session_id}:{number}"
            if item_id in text_of:  # a session id can repeat in a haystack
                item_id = f"{prefix}{session_id}#{position}:{number}"
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
                marked_sessions.add(session_id)
            if session_id in answer_sessions:
                session_turns.setdefault(session_id, []).append(item_id)
    unanswerable = str(instance["question_id"]).endswith("_abs")
    question = QAItem(
        str(instance["question_id"]),
        f"(asked on {instance.get('question_date', '')}) {instance['question']}",
        str(instance["answer"]), instance["question_type"], tuple(marked), unanswerable,
        after=turns[-1].id if turns else None,
    )
    fallback = tuple(tuple(ids) for session, ids in session_turns.items() if session not in marked_sessions)
    return QAEpisode(str(instance["question_id"]), turns, [question], text_of), fallback


class LongMemEvalEnv(QAEnvironment):
    name = "longmemeval"

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.path = config.get("path", "data/longmemeval/longmemeval_oracle.json")
        self.max_turn_tokens = config.get("max_turn_tokens")
        # Optional [start, end): seed s plays question start + s mod (end - start), so a training
        # run can be kept off the questions an evaluation uses (seeds 0-99 play questions 0-99).
        self.subset = tuple(config["subset"]) if config.get("subset") else None
        # Optional stratified fold, {k: 5, fold: 0, part: test|train}: seed s plays the s-th question of
        # the fold (mod its size). Use it instead of `subset` for evaluations over all question types.
        self.folds = dict(config["folds"]) if config.get("folds") else None
        if self.folds and self.subset:
            raise ValueError("set env.subset or env.folds, not both")
        # Optional compose: m > 1 makes one training episode out of m instances: their sessions merged in
        # date order, each question asked right after its own instance's last turn. One question per
        # episode starves imitation of labels (Experiment 11, peer review T4); m questions give m.
        self.compose = int(config.get("compose", 1))
        self._fallback: dict[str, tuple[tuple[str, ...], ...]] = {}
        self._chosen: list[int] | None = None
        if not Path(self.path).exists():
            raise FileNotFoundError(f"LongMemEval not found at {self.path}. Download a split from {SOURCE}")

    def load_episode(self, seed: int) -> QAEpisode:
        instances = _index(self.path)
        if self.folds:
            if self._chosen is None:
                self._chosen = fold_indices(instances, int(self.folds.get("k", 5)), int(self.folds["fold"]),
                                            self.folds.get("part", "test"))
            pool = self._chosen
        else:
            start, end = self.subset or (0, len(instances))
            pool = list(range(start, end))
        if self.compose <= 1:
            episode, fallback = parse_instance(_instance(self.path, pool[seed % len(pool)]), self.max_turn_tokens)
            self._fallback = {episode.questions[0].id: fallback}
            return episode
        # A seed-derived sample, so groupings differ across seeds and DAgger iterations (not fixed quadruples).
        chosen = random.Random(seed).sample(pool, self.compose)
        parts = [parse_instance(_instance(self.path, index), self.max_turn_tokens, f"i{j}/")
                 for j, index in enumerate(chosen)]
        turns = sorted(((t.metadata.get("date", ""), j, n, t) for j, (e, _) in enumerate(parts) for n, t in enumerate(e.turns)),
                       key=lambda row: row[:3])
        self._fallback = {e.questions[0].id: fallback for e, fallback in parts}
        text_of = {k: v for e, _ in parts for k, v in e.turn_text.items()}
        return QAEpisode("+".join(e.id for e, _ in parts), [row[3] for row in turns],
                         [q for e, _ in parts for q in e.questions], text_of)

    def get_ground_truth_dependencies(self) -> list[Dependency]:
        dependencies = super().get_ground_truth_dependencies()
        for number, dependency in enumerate(dependencies):
            fallback = self._fallback.get(self._questions[dependency.query_id].id, ())
            if fallback:
                # An answer session with no marked turn: any one of its turns is accepted as evidence.
                extra = tuple(EvidenceRequirement(ids, "") for ids in fallback)
                dependencies[number] = Dependency(
                    dependency.query_id, dependency.step, dependency.requirements + extra, dependency.gold,
                    dependency.category,
                )
        return dependencies
