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
        """How the full item looks inside the prompt. No id: small models tend to answer with it."""
        date = self.source.get("date")
        who = f"{self.speaker()}, {date}" if date else self.speaker()
        return f"({who}) {self.text}"

    def search_text(self) -> str:
        """The text that is embedded for search. Including the speaker doubles recall on LoCoMo."""
        return f"{self.speaker()}: {self.text}"

    def index_line(self, max_words: int = 4) -> str:
        """The one-line archive index entry: id plus the first few words.

        It never costs more tokens than the item itself, and much less for long items.
        """
        words = self.text.split()
        label = " ".join(words[:max_words]) + (" …" if len(words) > max_words + 1 else "")
        return f"[{self.id}] {label}"


def make_item(
    id: str, text: str, arrival_step: int, kind: str = "dialogue_turn", source: dict | None = None
) -> Item:
    """Build an Item and count its tokens (as it will appear in the prompt)."""
    item = Item(id=id, text=text, tokens=0, arrival_step=arrival_step, kind=kind, source=source or {})
    return Item(id, text, count_tokens(item.prompt_line()), arrival_step, kind, item.source)
