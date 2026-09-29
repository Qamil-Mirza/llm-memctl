"""Text embedders used for STORE search and the similarity feature.

The default `HashingEmbedder` needs no model download and is deterministic,
which keeps tests and smoke runs fast. A sentence-transformers model can be
swapped in from the config for real experiments.
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
    """Lowercased words, without stopwords and single letters (such as the "s" in "Ann's")."""
    return [w for w in _WORD.findall(text.lower()) if len(w) > 1 and w not in _STOPWORDS]


class Embedder(Protocol):
    def embed(self, text: str) -> np.ndarray:
        """Return a unit-length vector for the text."""
        ...


class HashingEmbedder:
    """Bag-of-words embedding: each word is hashed into one of `dim` buckets."""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        for word in content_words(text):
            vector[zlib.crc32(word.encode()) % self.dim] += 1.0
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 0 else vector


class SentenceTransformerEmbedder:
    """Dense embeddings from a sentence-transformers model (downloaded on first use)."""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer  # optional dependency

        self.model = SentenceTransformer(model_name)

    def embed(self, text: str) -> np.ndarray:
        return np.asarray(self.model.encode(text, normalize_embeddings=True), dtype=np.float32)


def build_embedder(name: str) -> Embedder:
    """`hashing` for the built-in embedder, otherwise a sentence-transformers model name."""
    if name == "hashing":
        return HashingEmbedder()
    return SentenceTransformerEmbedder(name)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two unit vectors."""
    return float(np.dot(a, b))
