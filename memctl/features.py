"""Per-item features that every controller can use.

Features are computed only from what the agent has seen so far. Benchmark
evidence labels are never part of them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from memctl.items import Item
from memctl.memory import MemoryState

KINDS = ("dialogue_turn", "tool_output", "observation")


@dataclass(frozen=True)
class ItemFeatures:
    item_id: str
    age_steps: int  # steps since the item arrived
    tokens: int  # size in tokens
    kind: str  # dialogue_turn | tool_output | observation
    times_referenced: int  # how often it was used (referenced, retrieved or recalled)
    steps_since_last_use: int
    similarity_to_query: float  # cosine similarity to the current query, 0..1
    is_incoming: bool  # True for the new item that triggered the decision

    def as_dict(self) -> dict:
        return asdict(self)

    def as_vector(self) -> list[float]:
        """Numbers only, in a fixed order, for learned policies."""
        one_hot = [1.0 if self.kind == kind else 0.0 for kind in KINDS]
        return [
            float(self.age_steps),
            float(self.tokens),
            float(self.times_referenced),
            float(self.steps_since_last_use),
            self.similarity_to_query,
            1.0 if self.is_incoming else 0.0,
            *one_hot,
        ]


def compute_features(state: MemoryState, items: list[Item], query_text: str) -> dict[str, ItemFeatures]:
    """Features for each item, given the current query (usually the newest text)."""
    query = state.embedder.embed(query_text)
    incoming_id = state.incoming.id if state.incoming else None
    features = {}
    for item in items:
        features[item.id] = ItemFeatures(
            item_id=item.id,
            age_steps=state.step - item.arrival_step,
            tokens=item.tokens,
            kind=item.kind,
            times_referenced=state.use_count[item.id],
            steps_since_last_use=state.step - state.last_used[item.id],
            similarity_to_query=round(max(0.0, state.similarity(item.id, query)), 4),
            is_incoming=item.id == incoming_id,
        )
    return features
