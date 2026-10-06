"""Numeric features of memory items and of the memory state, for learned controllers.

Computed only from what a controller may see (a MemoryView and a TaskState).
Any controller may use them; none has to. `FEATURE_VERSION` is stored in every
checkpoint so a policy is never fed features it was not trained on.
"""

from __future__ import annotations

import math

import numpy as np

from memctl.memory.compress import salience, sentences
from memctl.memory.items import Fidelity, MemoryItem, SourceType, Tier
from memctl.memory.state import MemoryView
from memctl.task import TaskState

# Version 2 adds `bridge_score` (the follow-the-clue search, memctl/retrieval.py). Version 1
# checkpoints still load and are fed version 1 features.
FEATURE_VERSION = 2
SOURCES = list(SourceType)
FIDELITIES = list(Fidelity)
ITEM_FEATURES = (
    ["age", "tokens_over_budget", "log_tokens"]
    + [f"source_{source.value}" for source in SOURCES]
    + ["archived"]
    + [f"fidelity_{fidelity.value}" for fidelity in FIDELITIES]
    + ["log_access_count", "time_since_access", "goal_similarity", "observation_similarity",
       "max_similarity_to_other", "retrieved_before", "is_current_observation", "specific_tokens",
       "specific_density", "retrieval_score"]
)
GLOBAL_FEATURES = [
    "active_over_budget", "excess_over_budget", "step_over_horizon", "active_count", "archive_over_budget",
    "requires_response",
]
ITEM_DIM, GLOBAL_DIM = len(ITEM_FEATURES), len(GLOBAL_FEATURES)
VERSION_FEATURES = {1: ITEM_FEATURES, 2: ITEM_FEATURES + ["bridge_score"]}


def origin(item: MemoryItem) -> int:
    return item.metadata.get("origin_created_at", item.created_at)


class Featurizer:
    """Builds feature matrices. Keeps a per-episode cache of what never changes for an item."""

    def __init__(self, use_embeddings: bool = False, embedding_dim: int = 0, version: int = 1) -> None:
        if version not in VERSION_FEATURES:
            raise ValueError(f"unknown feature version {version}; known: {sorted(VERSION_FEATURES)}")
        self.use_embeddings = use_embeddings
        self.embedding_dim = embedding_dim if use_embeddings else 0
        self.version = version
        self.base_dim = len(VERSION_FEATURES[version])
        self._static: dict[str, np.ndarray] = {}

    @property
    def item_dim(self) -> int:
        return self.base_dim + self.embedding_dim

    def reset(self) -> None:
        self._static = {}

    def _static_part(self, item: MemoryItem) -> np.ndarray:
        if item.id not in self._static:
            specific = sum(salience(sentence) for sentence in sentences(item.content))
            self._static[item.id] = np.array(
                [min(specific, 5.0) / 5.0, specific / max(1, item.token_count)], dtype=np.float32
            )
        return self._static[item.id]

    def items(
        self, items: list[MemoryItem], memory: MemoryView, task: TaskState, retrieval_scores: dict[str, float] | None = None,
        bridge_scores: dict[str, float] | None = None,
    ) -> np.ndarray:
        """One row per item, in the order given."""
        scores = retrieval_scores or {}
        bridge = bridge_scores or {}
        horizon = math.log1p(task.horizon or 1000)
        current = memory.get(task.observation.id)
        current_vector = current.embedding if current is not None else None
        rows = np.zeros((len(items), self.item_dim), dtype=np.float32)
        vectors = [item.embedding for item in items]
        have_vectors = bool(items) and all(vector is not None for vector in vectors)
        if have_vectors:
            matrix = np.stack(vectors)
            similarity = matrix @ matrix.T
            np.fill_diagonal(similarity, 0.0)
            active = np.array([item.tier is Tier.ACTIVE for item in items])
            redundancy = (similarity * active[None, :]).max(axis=1) if active.any() else np.zeros(len(items))
            to_current = matrix @ current_vector if current_vector is not None else np.zeros(len(items))
            to_goal = matrix @ task.goal_embedding if task.goal_embedding is not None else np.zeros(len(items))
        for n, item in enumerate(items):
            row = rows[n]
            row[0] = math.log1p(memory.step - origin(item)) / horizon
            row[1] = min(2.0, item.token_count / memory.budget)
            row[2] = math.log1p(item.token_count) / 6.0
            row[3 + SOURCES.index(item.source_type)] = 1.0
            base = 3 + len(SOURCES)
            row[base] = float(item.tier is Tier.ARCHIVE)
            row[base + 1 + FIDELITIES.index(item.fidelity)] = 1.0
            base += 1 + len(FIDELITIES)
            row[base] = math.log1p(item.access_count) / 3.0
            row[base + 1] = math.log1p(memory.step - item.last_accessed_at) / horizon
            if have_vectors:
                row[base + 2] = to_goal[n]
                row[base + 3] = to_current[n]
                row[base + 4] = redundancy[n]
            row[base + 5] = float(item.retrieval_count > 0)
            row[base + 6] = float(item.id == task.observation.id)
            row[base + 7 : base + 9] = self._static_part(item)
            row[base + 9] = scores.get(item.id, 0.0)
            if self.version >= 2:
                row[base + 10] = bridge.get(item.id, 0.0)
            if self.embedding_dim and item.embedding is not None:
                row[self.base_dim:] = item.embedding[: self.embedding_dim]
        return rows

    @staticmethod
    def globals(memory: MemoryView, task: TaskState) -> np.ndarray:
        return np.array(
            [
                min(3.0, memory.active_tokens / memory.budget),
                min(2.0, max(0, memory.active_tokens - memory.budget) / memory.budget),
                memory.step / (task.horizon or 1000),
                len(memory.active) / 100.0,
                min(5.0, memory.archive_tokens / memory.budget),
                float(task.observation.requires_response),
            ],
            dtype=np.float32,
        )
