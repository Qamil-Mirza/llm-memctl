"""Archive search, as a component a controller may use.

The harness never searches. A controller that wants retrieval calls a Retriever
and then emits RETRIEVE_FROM_ARCHIVE with the ids it chose, so every retrieval
is a logged controller decision.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Protocol, Sequence

from memctl.embed import Embedder, content_words, cosine
from memctl.memory.items import MemoryItem


class Retriever(Protocol):
    name: str

    def search(self, query: str, items: Sequence[MemoryItem], k: int) -> list[tuple[MemoryItem, float]]:
        """The `k` best items for the query, best first, with their scores."""
        ...


class LexicalRetriever:
    """BM25 over content words, with document frequencies taken from the candidates."""

    name = "lexical"

    def __init__(self, k1: float = 1.2, b: float = 0.75) -> None:
        self.k1, self.b = k1, b

    def search(self, query: str, items: Sequence[MemoryItem], k: int) -> list[tuple[MemoryItem, float]]:
        if not items or k <= 0:
            return []
        documents = [Counter(content_words(item.content)) for item in items]
        lengths = [sum(document.values()) for document in documents]
        average = sum(lengths) / len(lengths) or 1.0
        frequency = Counter(word for document in documents for word in document)
        scored = []
        for item, document, length in zip(items, documents, lengths):
            score = 0.0
            for word in set(content_words(query)):
                count = document.get(word, 0)
                if count:
                    idf = math.log(1 + (len(items) - frequency[word] + 0.5) / (frequency[word] + 0.5))
                    score += idf * count * (self.k1 + 1) / (count + self.k1 * (1 - self.b + self.b * length / average))
            if score > 0:
                scored.append((item, score))
        scored.sort(key=lambda pair: (-pair[1], pair[0].created_at))
        return scored[:k]


class EmbeddingRetriever:
    """Cosine similarity between the query and each item's stored embedding."""

    name = "embedding"

    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder

    def search(self, query: str, items: Sequence[MemoryItem], k: int) -> list[tuple[MemoryItem, float]]:
        if not items or k <= 0:
            return []
        vector = self.embedder.embed(query)
        scored = []
        for item in items:
            embedding = item.embedding if item.embedding is not None else self.embedder.embed(item.content)
            scored.append((item, cosine(vector, embedding)))
        scored.sort(key=lambda pair: (-pair[1], pair[0].created_at))
        return [pair for pair in scored[:k] if pair[1] > 0]


def build_retriever(method: str = "lexical", embedder: Embedder | None = None) -> Retriever:
    if method == "lexical":
        return LexicalRetriever()
    if method == "embedding":
        if embedder is None:
            raise ValueError("embedding retrieval needs memory.embedder to be set")
        return EmbeddingRetriever(embedder)
    raise KeyError(f"unknown retrieval method '{method}' (known: lexical, embedding)")
