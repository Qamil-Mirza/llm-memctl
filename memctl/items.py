"""The unit of memory (Item), the places it can live (Place), and token counting."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class Place(str, Enum):
    """The four places an item can be. An item is in exactly one at a time."""

    CONTEXT = "CONTEXT"  # in the prompt, limited by the token budget B
    STORE = "STORE"  # searched automatically each turn (embedding top-k)
    ARCHIVE = "ARCHIVE"  # cold storage; only an index line is shown; needs recall(id)
    DROPPED = "DROPPED"  # deleted for good


_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]")


def count_tokens(text: str) -> int:
    """Count tokens as words plus punctuation marks.

    This is a simple, model-free count so budgets mean the same thing for
    every model. See docs/decisions.md for why we do not use a model tokenizer.
    """
    return len(_TOKEN_PATTERN.findall(text))


@dataclass(frozen=True)
class Item:
    """One unit of history: a dialogue turn, a tool output or an observation."""

    id: str
    text: str
    tokens: int
    arrival_step: int
    kind: str = "dialogue_turn"  # dialogue_turn | tool_output | observation
    source: dict = field(default_factory=dict)  # e.g. speaker, session, date. Never labels.

    def speaker(self) -> str:
        return str(self.source.get("speaker", self.kind))

    def prompt_line(self) -> str:
        """How the full item looks inside the prompt."""
        date = self.source.get("date")
        who = f"{self.speaker()}, {date}" if date else self.speaker()
        return f"[{self.id}] ({who}) {self.text}"

    def index_line(self, max_words: int = 6) -> str:
        """The one-line archive index entry: id plus a short label.

        It always costs fewer tokens than the item itself, so archiving always frees room.
        """
        words = self.text.split()
        label = " ".join(words[:max_words]) + (" …" if len(words) > max_words else "")
        return f"[{self.id}] {self.speaker()}: {label}"


def make_item(
    id: str, text: str, arrival_step: int, kind: str = "dialogue_turn", source: dict | None = None
) -> Item:
    """Build an Item and count its tokens (as it will appear in the prompt)."""
    item = Item(id=id, text=text, tokens=0, arrival_step=arrival_step, kind=kind, source=source or {})
    return Item(id, text, count_tokens(item.prompt_line()), arrival_step, kind, item.source)
