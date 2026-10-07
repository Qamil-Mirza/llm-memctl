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


def labelled_text(item: MemoryItem) -> str:
    """The item as a search sees it with labels: "speaker (date): content" (Experiment 12)."""
    who, when = item.metadata.get("speaker", ""), item.metadata.get("date", "")
    return f"{who} ({when}): {item.content}" if who or when else item.content


class LexicalRetriever:
    """BM25 over content words, with document frequencies taken from the candidates. With `labels`,
    each item is indexed with its speaker and date (labelled_text)."""

    def __init__(self, k1: float = 1.2, b: float = 0.75, labels: bool = False) -> None:
        self.k1, self.b, self.labels = k1, b, labels
        self.name = "lexical_label" if labels else "lexical"

    def search(self, query: str, items: Sequence[MemoryItem], k: int) -> list[tuple[MemoryItem, float]]:
        if not items or k <= 0:
            return []
        text = labelled_text if self.labels else (lambda item: item.content)
        documents = [Counter(content_words(text(item))) for item in items]
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


_DENSE_MODELS: dict[str, object] = {}
_DENSE_VECTORS: dict[tuple[str, str], object] = {}


class DenseRetriever:
    """Cosine similarity of sentence embeddings of labelled text, from a sentence-transformers model.
    Models and vectors are cached per process, so every cell of a sweep embeds a turn once."""

    def __init__(self, model: str = "BAAI/bge-small-en-v1.5", device: str = "cpu") -> None:
        self.model_name, self.device = model, device
        self.name = f"dense:{model}"

    def _model(self):
        if self.model_name not in _DENSE_MODELS:
            from sentence_transformers import SentenceTransformer  # optional dependency

            _DENSE_MODELS[self.model_name] = SentenceTransformer(self.model_name, device=self.device)
        return _DENSE_MODELS[self.model_name]

    def vectors(self, texts: list[str]):
        import numpy as np

        missing = list(dict.fromkeys(t for t in texts if (self.model_name, t) not in _DENSE_VECTORS))
        if missing:
            encoded = self._model().encode(missing, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
            for text, vector in zip(missing, encoded):
                _DENSE_VECTORS[(self.model_name, text)] = np.asarray(vector, dtype=np.float32)
        return np.stack([_DENSE_VECTORS[(self.model_name, t)] for t in texts])

    def search(self, query: str, items: Sequence[MemoryItem], k: int) -> list[tuple[MemoryItem, float]]:
        if not items or k <= 0:
            return []
        scores = self.vectors([labelled_text(item) for item in items]) @ self.vectors([query])[0]
        scored = sorted(zip(items, (float(x) for x in scores)), key=lambda pair: (-pair[1], pair[0].created_at))
        return [pair for pair in scored[:k] if pair[1] > 0]


class FusionRetriever:
    """Reciprocal rank fusion (constant 60) of labelled BM25 and dense search, each ranked to `depth`.
    The score is the fused RRF score, not a BM25 score: a policy's retrieval-score feature (score over
    the best score in the shortlist) changes meaning with it."""

    def __init__(self, model: str = "BAAI/bge-small-en-v1.5", depth: int = 50, constant: int = 60) -> None:
        self.lexical, self.dense = LexicalRetriever(labels=True), DenseRetriever(model)
        self.depth, self.constant = depth, constant
        self.name = f"fusion:{model}"

    def search(self, query: str, items: Sequence[MemoryItem], k: int) -> list[tuple[MemoryItem, float]]:
        if not items or k <= 0:
            return []
        depth = max(self.depth, k)
        fused: dict[str, float] = {}
        by_id = {}
        for ranking in (self.lexical.search(query, items, depth), self.dense.search(query, items, depth)):
            for rank, (item, _) in enumerate(ranking, 1):
                by_id[item.id] = item
                fused[item.id] = fused.get(item.id, 0.0) + 1.0 / (self.constant + rank)
        scored = sorted(((by_id[i], score) for i, score in fused.items()), key=lambda pair: (-pair[1], pair[0].created_at))
        return scored[:k]


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


RETRIEVAL_METHODS = ("lexical", "lexical_label", "embedding", "dense", "fusion")


def build_retriever(method: str = "lexical", embedder: Embedder | None = None, model: str | None = None) -> Retriever:
    """`model` names the sentence-transformers model of `dense` and `fusion` (default bge-small)."""
    if method == "lexical":
        return LexicalRetriever()
    if method == "lexical_label":
        return LexicalRetriever(labels=True)
    if method == "embedding":
        if embedder is None:
            raise ValueError("embedding retrieval needs memory.embedder to be set")
        return EmbeddingRetriever(embedder)
    if method == "dense":
        return DenseRetriever(model or "BAAI/bge-small-en-v1.5")
    if method == "fusion":
        return FusionRetriever(model or "BAAI/bge-small-en-v1.5")
    raise KeyError(f"unknown retrieval method '{method}' (known: {', '.join(RETRIEVAL_METHODS)})")
