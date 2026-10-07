"""Heuristic controllers: fixed rules, no learning.

Almost all of them are one idea: give every ACTIVE item a keep-priority, and
when memory is over budget remove the lowest-priority items until it fits. They
differ only in the priority. What "remove" means (delete, archive, compact) and
whether the controller searches the archive when a query arrives are config
options shared by all of them, which is how delete-only, archive-only,
compression-only and retrieval variants of the same rule are built.

    controller:
      name: lru
      removal: [COMPACT, MOVE_TO_ARCHIVE]   # tried in this order; default [EVICT]
      retrieve: {top_k: 3, method: lexical} # omit for no retrieval; method: lexical | lexical_label |
                                            # dense | fusion (model: ...); fit: true caps it to the room
      target_tokens: 3000                   # optional: evict down to this, below the budget
      consolidate: {ratio: 0.4}             # omit for no consolidation; ratio also shortens the merged item
"""

from __future__ import annotations

import math
import random
from abc import abstractmethod

from memctl.controllers.base import EpisodeInfo, MemoryController
from memctl.embed import cosine
from memctl.memory.actions import MemoryAction, Operation
from memctl.memory.compress import salience, sentences
from memctl.memory.items import Fidelity, MemoryItem, SourceType
from memctl.memory.state import MemoryView
from memctl.retrieval import build_retriever
from memctl.task import TaskState

REMOVAL_OPERATIONS = (
    Operation.EVICT, Operation.MOVE_TO_ARCHIVE, Operation.COMPACT, Operation.COMPACT_AND_ARCHIVE
)


def origin(item: MemoryItem) -> int:
    """When the item's underlying content first entered memory."""
    return item.metadata.get("origin_created_at", item.created_at)


class NoController(MemoryController):
    """Never acts. With the harness fallback this is plain FIFO truncation: the no-controller arm."""

    name = "no_controller"

    def decide(self, memory: MemoryView, task: TaskState) -> list[MemoryAction]:
        return []


class FullContext(NoController):
    """Never acts and is given an unlimited budget: the infinite-context reference."""

    name = "full_context"
    ignores_budget = True


class PriorityController(MemoryController):
    """Removes the lowest-priority items when over budget. Subclasses define the priority."""

    def __init__(self, config: dict, seed: int = 0) -> None:
        super().__init__(config, seed)
        self.removal = [Operation(op) for op in config.get("removal", ["EVICT"])]
        unknown = [op.value for op in self.removal if op not in REMOVAL_OPERATIONS]
        if unknown:
            raise ValueError(f"{unknown} cannot be used as removal operations")
        self.retrieve_config = config.get("retrieve")
        self.consolidate_config = config.get("consolidate")
        self.min_compact_tokens = int(config.get("min_compact_tokens", 30))
        self.compact_ratio = float(config.get("compact_ratio", 0.5))
        self.retriever = None
        # Target fill: remove down to this many active tokens even when the budget allows more, as the
        # RL controller's target_tokens does (a matched control for it). None fills up to the budget.
        self.target_tokens = int(config["target_tokens"]) if config.get("target_tokens") else None
        self._info: dict = {}

    def reset(self, episode: EpisodeInfo) -> None:
        super().reset(episode)
        needed = list(self.removal)
        if self.retrieve_config is not None:
            needed.append(Operation.RETRIEVE_FROM_ARCHIVE)
            self.retriever = build_retriever(self.retrieve_config.get("method", "lexical"), episode.embedder,
                                             self.retrieve_config.get("model"))
        if self.consolidate_config is not None:
            needed.append(Operation.CONSOLIDATE)
        missing = [op.value for op in needed if op not in episode.allowed_operations]
        if missing:
            raise ValueError(
                f"controller '{self.display_name}' is configured to use {missing}, which this "
                "experiment does not allow (memory.allowed_operations)"
            )

    @abstractmethod
    def priority(self, item: MemoryItem, memory: MemoryView, task: TaskState) -> float:
        """Higher means keep. Ties are broken oldest-first."""

    # ---- the shared decision procedure -------------------------------------

    def decide(self, memory: MemoryView, task: TaskState) -> list[MemoryAction]:
        actions: list[MemoryAction] = []
        protected: set[str] = set()
        projected = memory.active_tokens

        retrieval = self._retrieval(memory, task)
        if retrieval is not None:
            actions.append(retrieval)
            protected.update(retrieval.target_ids)
            projected += sum(memory.get(item_id).token_count for item_id in retrieval.target_ids)

        merge = self._consolidation(memory, task)
        if merge is not None:
            actions.append(merge)
            protected.update(merge.target_ids)  # they are replaced; savings are not counted on

        actions.extend(self.proactive(memory, task, protected))
        for action in actions:
            if action.operation in (Operation.MOVE_TO_ARCHIVE, Operation.EVICT):
                projected -= sum(memory.get(item_id).token_count for item_id in action.target_ids)
                protected.update(action.target_ids)

        self._info = {}
        limit = min(memory.budget, self.target_tokens) if self.target_tokens else memory.budget
        if projected > limit:  # priorities are only needed, and only logged, under pressure
            scores = {item.id: self.priority(item, memory, task) for item in memory.active}
            self._info = {"scores": {item_id: round(score, 4) for item_id, score in scores.items()}}
            candidates = sorted(
                (item for item in memory.active if not item.pinned and item.id not in protected),
                key=lambda item: (scores[item.id], origin(item), item.created_at),
            )
            actions.extend(self._relieve(candidates, projected - limit))
        return actions

    def proactive(self, memory: MemoryView, task: TaskState, protected: set[str]) -> list[MemoryAction]:
        """Actions taken whether or not memory is over budget. Default: none."""
        return []

    def _relieve(self, candidates: list[MemoryItem], excess: int) -> list[MemoryAction]:
        """Free at least `excess` tokens, lowest priority first, one removal operation at a time."""
        actions, touched = [], set()
        for operation in self.removal:
            for item in candidates:
                if excess <= 0:
                    return actions
                if item.id in touched:
                    continue
                if operation in (Operation.COMPACT, Operation.COMPACT_AND_ARCHIVE):
                    if item.fidelity is not Fidelity.FULL or item.token_count < self.min_compact_tokens:
                        continue
                    saved = item.token_count - math.ceil(self.compact_ratio * item.token_count)
                    actions.append(MemoryAction(operation, (item.id,), parameters={"ratio": self.compact_ratio}))
                else:
                    saved = item.token_count
                    actions.append(MemoryAction(operation, (item.id,)))
                touched.add(item.id)
                excess -= saved
        return actions

    def _retrieval(self, memory: MemoryView, task: TaskState) -> MemoryAction | None:
        """Search the archive when the agent is about to answer something."""
        if self.retriever is None or not task.observation.requires_response or not memory.archived:
            return None
        config = self.retrieve_config
        query = task.observation.content
        top_k = int(config.get("top_k", 3))
        # Controls for a learned re-ranker (Experiment 15): search `candidates` deep, optionally keep only one
        # speaker's turns, then take `top_k` by search order or, with rerank: shortest, by length.
        pool = self.retriever.search(query, memory.archived, int(config.get("candidates", top_k)))
        if config.get("speaker"):
            pool = [(item, score) for item, score in pool if item.metadata.get("speaker") == config["speaker"]]
        if config.get("rerank") == "shortest":
            pool = sorted(pool, key=lambda pair: (pair[0].token_count, -pair[1]))
        hits = pool[:top_k]
        hits = [(item, score) for item, score in hits if score >= float(config.get("min_score", 0.0))]
        if config.get("fit"):
            # As the RL controller's retrieval floor: top results in rank order while they fit in the
            # limit beside pinned items, so the harness never has to put one back.
            limit = min(memory.budget, self.target_tokens) if self.target_tokens else memory.budget
            room = limit - sum(item.token_count for item in memory.active if item.pinned)
            kept, used = [], 0
            for item, score in hits:
                if used + item.token_count > room:
                    break
                kept.append((item, score))
                used += item.token_count
            hits = kept
        if not hits:
            return None
        return MemoryAction(
            Operation.RETRIEVE_FROM_ARCHIVE,
            tuple(item.id for item, _ in hits),
            parameters={
                "query": query,
                "method": self.retriever.name,
                "scores": {item.id: round(score, 4) for item, score in hits},
            },
        )

    def _consolidation(self, memory: MemoryView, task: TaskState) -> MemoryAction | None:
        """Merge the newest item with older items that repeat one of its specific sentences."""
        if self.consolidate_config is None:
            return None
        newest = memory.get(task.observation.id)
        if newest is None or newest.pinned:
            return None
        specific = [s for s in sentences(newest.content) if salience(s) > 0]
        if not specific:
            return None
        limit = int(self.consolidate_config.get("max_group", 4))
        group = [
            item for item in memory.active
            if item.id != newest.id and not item.pinned and any(s in item.content for s in specific)
        ][: limit - 1]
        if not group:
            return None
        parameters = {}
        if self.consolidate_config.get("archive_sources"):
            parameters["archive_sources"] = True
        if self.consolidate_config.get("ratio"):  # also shorten the merged item to this share of its sources
            total = newest.token_count + sum(item.token_count for item in group)
            parameters["max_tokens"] = max(1, math.ceil(float(self.consolidate_config["ratio"]) * total))
        return MemoryAction(Operation.CONSOLIDATE, tuple(i.id for i in group) + (newest.id,), parameters=parameters)

    def decision_info(self) -> dict:
        return self._info


class Fifo(PriorityController):
    """Keep the newest."""

    name = "fifo"

    def priority(self, item, memory, task):
        return float(origin(item))


class Lru(PriorityController):
    """Keep what the task model used most recently."""

    name = "lru"

    def priority(self, item, memory, task):
        return float(item.last_accessed_at)


class Lfu(PriorityController):
    """Keep what the task model used most often."""

    name = "lfu"

    def priority(self, item, memory, task):
        return float(item.access_count)


class RandomEviction(PriorityController):
    name = "random"

    def __init__(self, config: dict, seed: int = 0) -> None:
        super().__init__(config, seed)
        self.rng = random.Random(seed)

    def reset(self, episode: EpisodeInfo) -> None:
        super().reset(episode)
        self.rng = random.Random(f"{self.seed}-{episode.seed}")

    def priority(self, item, memory, task):
        return self.rng.random()


class AgeDecay(PriorityController):
    """Importance decays with age and is reinforced by use: (1 + uses) * exp(-age / tau)."""

    name = "age_decay"

    def priority(self, item, memory, task):
        tau = float(self.config.get("tau", 50.0))
        return (1 + item.access_count) * math.exp(-(memory.step - origin(item)) / tau)


class Similarity(PriorityController):
    """Keep what resembles the current observation and the goal (embedding cosine)."""

    name = "similarity"

    def priority(self, item, memory, task):
        current = memory.get(task.observation.id)
        now = cosine(item.embedding, current.embedding) if current is not None else 0.0
        return max(now, cosine(item.embedding, task.goal_embedding))


class Salience(PriorityController):
    """A fixed importance score: who said it, and how specific it is per token."""

    name = "salience"
    SOURCE_WEIGHTS = {
        SourceType.USER: 1.0,
        SourceType.CONSOLIDATED_MEMORY: 0.9,
        SourceType.GENERATED_SUMMARY: 0.8,
        SourceType.OBSERVATION: 0.6,
        SourceType.ACTION: 0.5,
        SourceType.RETRIEVED_DOCUMENT: 0.4,
        SourceType.TOOL_OUTPUT: 0.3,
    }

    def __init__(self, config: dict, seed: int = 0) -> None:
        super().__init__(config, seed)
        overrides = {SourceType(k): v for k, v in config.get("source_weights", {}).items()}
        self.weights = {**self.SOURCE_WEIGHTS, **overrides}
        self._cache: dict[str, float] = {}

    def reset(self, episode: EpisodeInfo) -> None:
        super().reset(episode)
        self._cache = {}

    def priority(self, item, memory, task):
        if item.id not in self._cache:  # an item's content never changes, so neither does its score
            specific = sum(salience(sentence) for sentence in sentences(item.content))
            density = specific / max(1, item.token_count)
            self._cache[item.id] = self.weights[item.source_type] * (0.2 + min(1.0, specific / 3.0)) + density
        return self._cache[item.id]


class ArchiveEverything(Fifo):
    """RAG-style: only the last few items stay in ACTIVE, the rest are archived at once.
    Pair it with `retrieve` to search the archive when a query arrives."""

    name = "archive_everything"

    def __init__(self, config: dict, seed: int = 0) -> None:
        super().__init__({"removal": ["MOVE_TO_ARCHIVE"], **config}, seed)
        self.keep_last = int(config.get("keep_last", 4))

    def proactive(self, memory, task, protected):
        movable = [item for item in memory.active if not item.pinned and item.id not in protected]
        old = movable[: max(0, len(movable) - self.keep_last)]
        return [MemoryAction(Operation.MOVE_TO_ARCHIVE, tuple(item.id for item in old))] if old else []
