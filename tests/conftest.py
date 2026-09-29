"""Shared helpers for the tests."""

import pytest

from memctl.embed import HashingEmbedder
from memctl.items import Item, make_item
from memctl.memory import MemoryState


def item(n: int, words: int = 10, step: int | None = None) -> Item:
    """A test item whose text has `words` words."""
    text = " ".join(f"word{n}x{k}" for k in range(words))
    return make_item(f"i{n}", text, arrival_step=step if step is not None else n, source={"speaker": "Ann"})


@pytest.fixture
def state() -> MemoryState:
    return MemoryState(budget=60, embedder=HashingEmbedder())
