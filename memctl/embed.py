"""Text embedders for similarity features and embedding search.

`HashingEmbedder` needs no model and is deterministic; it measures word overlap.
A sentence-transformers model can be named in the config for real semantics.
"""

from __future__ import annotations

import re
import zlib
from typing import Protocol

import numpy as np

_WORD = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an and are as at be but by did do does for from had has have he her his i in is it its "
    "me my of on or our she so that the their them they this to was we were what when where "
    "which who will with you your".split()
)


def content_words(text: str) -> list[str]:
    """Lowercased words, without stopwords and single letters."""
    return [w for w in _WORD.findall(text.lower()) if len(w) > 1 and w not in _STOPWORDS]


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, text: str) -> np.ndarray:
        """A unit-length vector for the text."""
        ...


class HashingEmbedder:
    """Bag of words: each word is hashed into one of `dim` buckets."""

    def __init__(self, dim: int = 256) -> None:
        self.name = f"hashing-{dim}"
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        for word in content_words(text):
            vector[zlib.crc32(word.encode()) % self.dim] += 1.0
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 0 else vector


class SentenceTransformerEmbedder:
    """Dense embeddings from a sentence-transformers model (downloaded on first use)."""

    def __init__(self, model_name: str, device: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer  # optional dependency

        self.name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        dimension = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        self.dim = int(dimension())  # the method was renamed between sentence-transformers versions
        self._seen: dict[str, np.ndarray] = {}

    def embed(self, text: str) -> np.ndarray:
        if text not in self._seen:
            vector = self.model.encode(text, normalize_embeddings=True, show_progress_bar=False)
            self._seen[text] = np.asarray(vector, dtype=np.float32)
        return self._seen[text]


def build_embedder(config: dict | str | None) -> Embedder | None:
    """None or "none" for no embeddings, "hashing", or a sentence-transformers model name."""
    if config is None:
        return None
    if isinstance(config, str):
        config = {"name": config}
    name = config.get("name", "hashing")
    if name in ("none", None):
        return None
    if name == "hashing":
        return HashingEmbedder(int(config.get("dim", 256)))
    return SentenceTransformerEmbedder(name, config.get("device"))


def cosine(a, b) -> float:
    """Cosine similarity of two unit vectors; 0 if either is missing."""
    if a is None or b is None:
        return 0.0
    return float(np.dot(a, b))
