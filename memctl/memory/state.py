"""MemoryState: every item, which tier it is in, and the token accounting.

Only the engine (engine.py) and the harness change the state. Controllers and
agents get a `MemoryView`, which holds immutable items and no deleted ones.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from memctl.memory.items import Fidelity, MemoryItem, SourceType, Tier, count_tokens, label_prefix

HISTORY_IN_VIEW = 64  # how many recent action and retrieval records a view carries


@dataclass(frozen=True)
class MemoryView:
    """A read-only picture of memory at one moment."""

    step: int
    budget: int
    archive_budget: int | None
    active: tuple[MemoryItem, ...]  # oldest first
    archived: tuple[MemoryItem, ...]  # oldest first
    active_tokens: int
    archive_tokens: int
    recent_actions: tuple[dict, ...] = ()
    recent_retrievals: tuple[dict, ...] = ()

    @property
    def free_tokens(self) -> int:
        return self.budget - self.active_tokens

    @property
    def over_budget(self) -> bool:
        return self.active_tokens > self.budget

    def get(self, item_id: str) -> MemoryItem | None:
        for item in self.active + self.archived:
            if item.id == item_id:
                return item
        return None


class MemoryState:
    def __init__(self, budget: int, archive_budget: int | None = None, embedder=None, count_labels: bool = False) -> None:
        self.budget = budget
        self.count_labels = count_labels  # item tokens include their reader label (memory.count_labels)
        self.archive_budget = archive_budget
        self.embedder = embedder
        self.step = 0
        self.items: dict[str, MemoryItem] = {}  # every item ever created, deleted ones included
        self.task_metadata: dict = {}
        self.retrieval_history: list[dict] = []
        self.action_history: list[dict] = []
        self.tier_events: dict[str, list[dict]] = {}  # item id -> how and when it changed tier
        self.active_tokens = 0
        self.archive_tokens = 0
        self._by_tier: dict[Tier, dict[str, None]] = {tier: {} for tier in Tier}
        self._derived_counter = 0

    # ---- reading -----------------------------------------------------------

    def get(self, item_id: str) -> MemoryItem | None:
        return self.items.get(item_id)

    def in_tier(self, tier: Tier) -> list[MemoryItem]:
        """Items in `tier`, oldest first (ties broken by insertion order)."""
        found = [self.items[item_id] for item_id in self._by_tier[tier]]
        return sorted(found, key=lambda item: item.created_at)

    def active(self) -> list[MemoryItem]:
        return self.in_tier(Tier.ACTIVE)

    def archived(self) -> list[MemoryItem]:
        return self.in_tier(Tier.ARCHIVE)

    def view(self) -> MemoryView:
        return MemoryView(
            step=self.step,
            budget=self.budget,
            archive_budget=self.archive_budget,
            active=tuple(self.active()),
            archived=tuple(self.archived()),
            active_tokens=self.active_tokens,
            archive_tokens=self.archive_tokens,
            recent_actions=tuple(self.action_history[-HISTORY_IN_VIEW:]),
            recent_retrievals=tuple(self.retrieval_history[-HISTORY_IN_VIEW:]),
        )

    def summary(self) -> dict:
        """A small description of the state for the per-step log."""
        return {
            "step": self.step,
            "budget": self.budget,
            "active_tokens": self.active_tokens,
            "archive_tokens": self.archive_tokens,
            "active_count": len(self._by_tier[Tier.ACTIVE]),
            "archive_count": len(self._by_tier[Tier.ARCHIVE]),
            "deleted_count": len(self._by_tier[Tier.DELETED]),
            "active_ids": [item.id for item in self.active()],
        }

    # ---- changing (engine and harness only) -------------------------------

    def new_id(self, prefix: str) -> str:
        self._derived_counter += 1
        return f"{prefix}{self._derived_counter:05d}"

    def ingest(
        self,
        item_id: str,
        content: str,
        source_type: SourceType,
        *,
        metadata: dict | None = None,
        parent_ids: tuple[str, ...] = (),
        derived_from_ids: tuple[str, ...] = (),
        fidelity: Fidelity = Fidelity.FULL,
        pinned: bool = False,
    ) -> MemoryItem:
        """Add a new item to ACTIVE. The state may be over budget afterwards."""
        if item_id in self.items:
            raise ValueError(f"item {item_id} already exists")
        item = MemoryItem(
            id=item_id,
            content=content,
            token_count=count_tokens(label_prefix(dict(metadata or {}), source_type) + content
                                     if self.count_labels else content),
            created_at=self.step,
            source_type=SourceType(source_type),
            fidelity=fidelity,
            last_accessed_at=self.step,
            parent_ids=tuple(parent_ids),
            derived_from_ids=tuple(derived_from_ids),
            pinned=pinned,
            embedding=self.embedder.embed(content) if self.embedder else None,
            metadata=dict(metadata or {}),
        )
        self.items[item_id] = item
        self._by_tier[Tier.ACTIVE][item_id] = None
        self.active_tokens += item.token_count
        return item

    def update(self, item_id: str, **changes) -> MemoryItem:
        """Replace fields of an item that do not affect tier or tokens."""
        item = replace(self.items[item_id], **changes)
        self.items[item_id] = item
        return item

    def move(self, item_id: str, tier: Tier, operation: str, source: str) -> MemoryItem:
        """Change an item's tier and keep the token totals and the event trail right."""
        item = self.items[item_id]
        if item.tier is tier:
            return item
        del self._by_tier[item.tier][item_id]
        self._add_tokens(item.tier, -item.token_count)
        item = replace(item, tier=tier)
        self.items[item_id] = item
        self._by_tier[tier][item_id] = None
        self._add_tokens(tier, item.token_count)
        self.tier_events.setdefault(item_id, []).append(
            {"step": self.step, "operation": operation, "source": source, "tier": tier.value}
        )
        return item

    def _add_tokens(self, tier: Tier, amount: int) -> None:
        if tier is Tier.ACTIVE:
            self.active_tokens += amount
        elif tier is Tier.ARCHIVE:
            self.archive_tokens += amount

    def record_access(self, item_ids) -> None:
        """The task model used these items at the current step."""
        for item_id in item_ids:
            item = self.items.get(item_id)
            if item is not None and item.tier is not Tier.DELETED:
                self.update(item_id, access_count=item.access_count + 1, last_accessed_at=self.step)
