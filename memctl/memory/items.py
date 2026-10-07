"""The unit of memory (MemoryItem), where it can live (Tier) and token counting."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class Tier(str, Enum):
    ACTIVE = "ACTIVE"  # in the agent's context, limited by the token budget
    ARCHIVE = "ARCHIVE"  # kept, but the agent cannot read it until a controller retrieves it
    DELETED = "DELETED"  # gone for the agent and the controller; kept only in the logs


class Fidelity(str, Enum):
    FULL = "FULL"
    COMPACT = "COMPACT"
    CONSOLIDATED = "CONSOLIDATED"


class SourceType(str, Enum):
    USER = "user"
    OBSERVATION = "observation"
    TOOL_OUTPUT = "tool_output"
    ACTION = "action"
    RETRIEVED_DOCUMENT = "retrieved_document"
    GENERATED_SUMMARY = "generated_summary"
    CONSOLIDATED_MEMORY = "consolidated_memory"


_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]")


def label_prefix(metadata: dict, source_type) -> str:
    """The "speaker, date: " a compact-labelled reader puts before an item (agents/llm.py compact_line)."""
    who = metadata.get("speaker") or getattr(source_type, "value", str(source_type))
    when = metadata.get("date")
    return f"{who}, {when}: " if when else f"{who}: "


def count_tokens(text: str) -> int:
    """Count tokens as words plus punctuation marks.

    Model-free, so a budget means the same thing for every task model.
    """
    return len(_TOKEN_PATTERN.findall(text))


def truncate_tokens(text: str, max_tokens: int) -> str:
    """The longest prefix of `text` that has at most `max_tokens` tokens."""
    end = 0
    for number, match in enumerate(_TOKEN_PATTERN.finditer(text)):
        if number >= max_tokens:
            break
        end = match.end()
    return text[:end]


@dataclass(frozen=True)
class MemoryItem:
    """One piece of memory. The content of an id never changes: compaction and
    consolidation create new items and link them with `derived_from_ids`."""

    id: str
    content: str
    token_count: int
    created_at: int  # step at which the item entered memory
    source_type: SourceType = SourceType.OBSERVATION
    tier: Tier = Tier.ACTIVE
    fidelity: Fidelity = Fidelity.FULL
    last_accessed_at: int = 0
    access_count: int = 0  # times the task model used it
    reference_count: int = 0  # times another item was derived from it
    retrieval_count: int = 0  # times it came back from the archive
    parent_ids: tuple[str, ...] = ()  # structure given by the environment (e.g. the action that caused it)
    derived_from_ids: tuple[str, ...] = ()  # items whose content was rewritten into this one
    pinned: bool = False
    superseded_by: str | None = None
    embedding: object | None = field(default=None, compare=False, repr=False)
    metadata: dict = field(default_factory=dict, compare=False)  # never ground-truth labels

    def log_row(self) -> dict:
        """Everything about the item except its embedding, as plain JSON types."""
        return {
            "id": self.id,
            "content": self.content,
            "token_count": self.token_count,
            "created_at": self.created_at,
            "source_type": self.source_type.value,
            "tier": self.tier.value,
            "fidelity": self.fidelity.value,
            "last_accessed_at": self.last_accessed_at,
            "access_count": self.access_count,
            "reference_count": self.reference_count,
            "retrieval_count": self.retrieval_count,
            "parent_ids": list(self.parent_ids),
            "derived_from_ids": list(self.derived_from_ids),
            "pinned": self.pinned,
            "superseded_by": self.superseded_by,
            "metadata": self.metadata,
        }
