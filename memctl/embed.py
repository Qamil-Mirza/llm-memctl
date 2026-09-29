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


DEFAULT_EMBEDDER = "BAAI/bge-small-en-v1.5"

# "An earlier item was referenced" means: its similarity to the new message is in the top 5%
# of all message pairs. Similarity scales differ a lot between embedders, so each has its own
# threshold. Measured on 3 LoCoMo conversations (106,360 pairs), without using any labels.
REFERENCE_THRESHOLDS = {"hashing": 0.29, "BAAI/bge-small-en-v1.5": 0.77}


class Embedder(Protocol):
    reference_threshold: float

    def embed(self, text: str) -> np.ndarray:
        """Return a unit-length vector for the text."""
        ...


class HashingEmbedder:
    """Bag-of-words embedding: each word is hashed into one of `dim` buckets."""

    def __init__(self, dim: int = 512, reference_threshold: float = REFERENCE_THRESHOLDS["hashing"]) -> None:
        self.dim = dim
        self.reference_threshold = reference_threshold

    def embed(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        for word in content_words(text):
            vector[zlib.crc32(word.encode()) % self.dim] += 1.0
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 0 else vector


class SentenceTransformerEmbedder:
    """Dense embeddings from a sentence-transformers model (downloaded on first use)."""

    def __init__(self, model_name: str, reference_threshold: float) -> None:
        from sentence_transformers import SentenceTransformer  # optional dependency

        self.model = SentenceTransformer(model_name)
        self.reference_threshold = reference_threshold
        self.seen: dict[str, np.ndarray] = {}  # texts repeat a lot, so remember their vectors

    def embed(self, text: str) -> np.ndarray:
        if text not in self.seen:
            vector = self.model.encode(text, normalize_embeddings=True, show_progress_bar=False)
            self.seen[text] = np.asarray(vector, dtype=np.float32)
        return self.seen[text]


def build_embedder(name: str = DEFAULT_EMBEDDER, reference_threshold: float | None = None) -> Embedder:
    """`hashing` for the built-in test embedder, otherwise a sentence-transformers model name.

    The dense default is for real runs. `hashing` only measures word overlap and is for tests.
    """
    if reference_threshold is None:
        if name not in REFERENCE_THRESHOLDS:
            raise KeyError(
                f"no reference threshold is known for embedder '{name}'. Set memory.reference_threshold "
                "in the config (see REFERENCE_THRESHOLDS in memctl/embed.py for how it is measured)."
            )
        reference_threshold = REFERENCE_THRESHOLDS[name]
    if name == "hashing":
        return HashingEmbedder(reference_threshold=reference_threshold)
    return SentenceTransformerEmbedder(name, reference_threshold)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two unit vectors."""
    return float(np.dot(a, b))
