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


def bridge_search(
    retriever: Retriever, query: str, active: Sequence[MemoryItem], archived: Sequence[MemoryItem],
    k: int, seeds: int = 2, exclude: Sequence[str] = (), clue_words: int = 3,
) -> list[tuple[MemoryItem, float]]:
    """Follow-the-clue search for multi-hop questions: the archive items reached through a bridge.

    A two-hop question ("the priority of the courier of sensor-171") names the first fact's
    subject but not the second fact's ("clerk-172"), so a search with the question's words cannot
    find the second fact. Seeds are the best matches to the question, in active memory and the
    archive, that contain the question's rarest word (what it names). Each of the first `seeds`
    gives a second query: the question's words the seed lacks, plus the seed's `clue_words`
    rarest words the question lacks (the bridge entity, not the filler around it). The `k` best
    archive items for those queries are returned, best first, each with its best score. Nothing
    here knows the task's sentence forms.
    """
    pool = [item for item in list(active) + list(archived) if item.id not in set(exclude)]
    if not pool or not archived or k <= 0:
        return []
    question = set(content_words(query))
    # The first hop is about what the question names, which is its rarest word in memory (the entity,
    # not "question" or "what"): seeds must contain it. Without such a word, any match may seed.
    words_of = {item.id: set(content_words(item.content)) for item in pool}
    frequency = Counter(word for item in pool for word in words_of[item.id])
    present = [word for word in question if frequency[word]]
    anchor = min(present, key=lambda word: (frequency[word], word)) if present else None
    best: dict[str, tuple[MemoryItem, float]] = {}
    used = 0
    for seed, _ in retriever.search(query, pool, len(pool)):
        if anchor is not None and anchor not in words_of[seed.id]:
            continue
        words = words_of[seed.id]
        # The seed's clue is its rarest new words (the bridge entity), not the filler around it.
        new = sorted(words - question, key=lambda word: (frequency[word], word))[:clue_words]
        if not new:
            continue
        second = " ".join(sorted(question - words) + new)
        for item, score in retriever.search(second, [a for a in archived if a.id != seed.id], k):
            if score > best.get(item.id, (None, 0.0))[1]:
                best[item.id] = (item, score)
        used += 1
        if used == seeds:
            break
    ranked = sorted(best.values(), key=lambda pair: (-pair[1], pair[0].created_at))
    return ranked[:k]


def build_retriever(method: str = "lexical", embedder: Embedder | None = None) -> Retriever:
    if method == "lexical":
        return LexicalRetriever()
    if method == "embedding":
        if embedder is None:
            raise ValueError("embedding retrieval needs memory.embedder to be set")
        return EmbeddingRetriever(embedder)
    raise KeyError(f"unknown retrieval method '{method}' (known: lexical, embedding)")
