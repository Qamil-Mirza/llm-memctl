"""Standard rerankers of the §19 head's candidate pool (EXPERIMENTS.md §32).

The §19 head, as deployed (`fixed8`), is a top-k reranker: memory keeps nothing but the current turn
(`keep_none`), and at a question the head picks 8 of the 32 BM25 candidates by its logit. Every scorer here ranks
the same 32 candidates and the controller (memctl/controllers/rerank.py) shows the 8 best. Higher scores are better;
ties keep the BM25 order. No scorer sees gold evidence.

Kinds (the `scorer` block of the `rerank` controller):
- `bm25`: the BM25 order itself (no learning).
- `rrf`: reciprocal rank fusion (constant 60) of the BM25 rank and the bge-small dense rank, both counted *within*
  the 32-candidate pool, so the pool is the head's. (FusionRetriever fuses ranks over the whole archive with a
  labelled BM25; that would change the pool, so it is not used here. Within the pool, the dense order is the same
  as the whole-archive dense order; only the RRF rank numbers differ.)
- `pointwise`: a classifier on the head's own inputs, 24 item features and 6 global features
  (memctl/features.py), one row per candidate. `model` is a JSON logistic regression (`lr`) or a pickled
  scikit-learn classifier (`gbdt`), written by memctl/rerank_train.py.
- `cross_encoder`: a sentence-transformers CrossEncoder on (question, turn text). Scores are cached on disk by
  (model, question, turn text), so a later pass (the reader stage) replays them; `cache_only: true` makes a miss an
  error.
- `head`: the §19 head's retrieve logit (the controller's own policy; used for latency measurement only).
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from memctl.features import GLOBAL_FEATURES, ITEM_FEATURES
from memctl.memory.items import MemoryItem
from memctl.retrieval import DenseRetriever, labelled_text

KINDS = ("bm25", "rrf", "pointwise", "cross_encoder", "head")
RRF_CONSTANT = 60
POINTWISE_COLUMNS = list(ITEM_FEATURES) + list(GLOBAL_FEATURES)


def rows(item_features: np.ndarray, global_features: np.ndarray) -> np.ndarray:
    """The pointwise rows: each candidate's 24 item features followed by the 6 global features."""
    item_features = np.asarray(item_features, dtype=np.float32)
    if item_features.shape[1] != len(ITEM_FEATURES):
        raise ValueError(f"expected {len(ITEM_FEATURES)} item features, got {item_features.shape[1]}")
    repeated = np.repeat(np.asarray(global_features, dtype=np.float32)[None, :], len(item_features), axis=0)
    return np.concatenate([item_features, repeated], axis=1)


def top_k(scores: Sequence[float], k: int) -> list[int]:
    """Indices of the k best scores, best first; ties keep the original (BM25) order."""
    order = sorted(range(len(scores)), key=lambda j: (-float(scores[j]), j))
    return order[: max(0, k)]


def rrf_scores(dense: Sequence[float], constant: int = RRF_CONSTANT) -> np.ndarray:
    """RRF of the pool's BM25 rank (its order) and its dense rank (ties broken by BM25 order)."""
    n = len(dense)
    dense_order = sorted(range(n), key=lambda j: (-float(dense[j]), j))
    dense_rank = np.empty(n, dtype=np.int64)
    dense_rank[dense_order] = np.arange(1, n + 1)
    return np.array([1.0 / (constant + j + 1) + 1.0 / (constant + dense_rank[j]) for j in range(n)])


class LogisticModel:
    """Standardised logistic regression stored as JSON (no sklearn needed to score)."""

    def __init__(self, mean, scale, coef, intercept, meta: dict | None = None) -> None:
        self.mean = np.asarray(mean, dtype=np.float64)
        self.scale = np.asarray(scale, dtype=np.float64)
        self.coef = np.asarray(coef, dtype=np.float64)
        self.intercept = float(intercept)
        self.meta = meta or {}

    @property
    def parameters(self) -> int:
        return len(self.coef) + 1

    def decision(self, x: np.ndarray) -> np.ndarray:
        return ((np.asarray(x, dtype=np.float64) - self.mean) / self.scale) @ self.coef + self.intercept

    def to_json(self) -> dict:
        return {"kind": "lr", "columns": POINTWISE_COLUMNS, "mean": self.mean.tolist(), "scale": self.scale.tolist(),
                "coef": self.coef.tolist(), "intercept": self.intercept, "meta": self.meta}

    @classmethod
    def from_json(cls, data: dict) -> "LogisticModel":
        if data.get("columns") != POINTWISE_COLUMNS:
            raise ValueError("model columns differ from memctl.features: retrain it")
        return cls(data["mean"], data["scale"], data["coef"], data["intercept"], data.get("meta"))


def load_pointwise(path: str | Path):
    path = Path(path)
    if path.suffix == ".json":
        return LogisticModel.from_json(json.loads(path.read_text()))
    with open(path, "rb") as handle:
        return pickle.load(handle)


def pointwise_scores(model, x: np.ndarray) -> np.ndarray:
    if isinstance(model, LogisticModel):
        return model.decision(x)
    return np.asarray(model.decision_function(x) if hasattr(model, "decision_function") else
                      model.predict_proba(x)[:, 1], dtype=np.float64)


def gbdt_size(model) -> int:
    """Nodes in a fitted HistGradientBoosting model (its parameter count, for the size column)."""
    predictors = getattr(model, "_predictors", None) or []
    return sum(len(tree.nodes) for stage in predictors for tree in stage)


def _key(*parts: str) -> str:
    digest = hashlib.sha1()
    for part in parts:
        digest.update(part.encode())
        digest.update(b"\0")
    return digest.hexdigest()


class CrossEncoderScorer:
    """A CrossEncoder with an append-only JSONL score cache (one line per new pair; safe across processes)."""

    _models: dict[str, object] = {}

    def __init__(self, model: str, cache: str | None = None, cache_only: bool = False, max_length: int = 512,
                 batch_size: int = 32) -> None:
        self.model_name, self.cache_path, self.cache_only = model, cache, cache_only
        self.max_length, self.batch_size = max_length, batch_size
        self.scores: dict[str, float] = {}
        if cache and Path(cache).exists():
            for line in open(cache):
                if line.strip():
                    row = json.loads(line)
                    self.scores[row["k"]] = row["s"]
        elif cache_only:
            raise FileNotFoundError(f"cache_only {type(self).__name__}, but no cache at {cache}")

    def _model(self):
        if self.model_name not in self._models:
            from sentence_transformers import CrossEncoder  # optional dependency

            self._models[self.model_name] = CrossEncoder(self.model_name, device="cpu", max_length=self.max_length)
        return self._models[self.model_name]

    def _compute(self, query: str, texts: list[str]):
        return self._model().predict([(query, text) for text in texts], batch_size=self.batch_size,
                                     show_progress_bar=False)

    def score(self, query: str, texts: Sequence[str]) -> np.ndarray:
        keys = [_key(self.model_name, query, text) for text in texts]
        missing = [j for j, key in enumerate(keys) if key not in self.scores]
        if missing:
            if self.cache_only:
                raise KeyError(f"{type(self).__name__} cache miss ({len(missing)} pairs) with cache_only: true")
            values = self._compute(query, [texts[j] for j in missing])
            lines = []
            for j, value in zip(missing, values):
                self.scores[keys[j]] = float(value)
                lines.append(json.dumps({"k": keys[j], "s": float(value)}) + "\n")
            if self.cache_path:
                Path(self.cache_path).parent.mkdir(parents=True, exist_ok=True)
                with open(self.cache_path, "a") as handle:  # one write per batch; O_APPEND keeps lines whole
                    handle.write("".join(lines))
                    handle.flush()
                    os.fsync(handle.fileno())
        return np.array([self.scores[key] for key in keys], dtype=np.float64)


class DenseSimilarity(CrossEncoderScorer):
    """The bge-small cosine of (question, labelled turn text), with the same disk cache (key prefix "dense:")."""

    def __init__(self, model: str, cache: str | None = None, cache_only: bool = False) -> None:
        super().__init__(f"dense:{model}", cache, cache_only)
        self.dense = DenseRetriever(model)

    def _compute(self, query: str, texts: list[str]):
        return self.dense.vectors(texts) @ self.dense.vectors([query])[0]


class Scorer:
    """Scores a candidate pool; built from the controller's `scorer` block."""

    def __init__(self, config: dict) -> None:
        self.config = dict(config)
        self.kind = self.config.get("kind")
        if self.kind not in KINDS:
            raise ValueError(f"unknown scorer kind {self.kind!r}; known: {KINDS}")
        self.uses_features = self.kind in ("pointwise", "head")
        self.model = None
        self.dense = None
        self.cross = None
        if self.kind == "pointwise":
            if not self.config.get("model"):
                raise ValueError("a pointwise scorer needs `model`")
            self.model = load_pointwise(self.config["model"])
        elif self.kind == "rrf":
            self.dense = DenseSimilarity(self.config.get("model", "BAAI/bge-small-en-v1.5"), self.config.get("cache"),
                                         bool(self.config.get("cache_only", False)))
        elif self.kind == "cross_encoder":
            self.cross = CrossEncoderScorer(self.config.get("model", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
                                            self.config.get("cache"), bool(self.config.get("cache_only", False)),
                                            int(self.config.get("max_length", 512)))

    @property
    def parameters(self) -> int:
        if self.kind == "pointwise":
            return self.model.parameters if isinstance(self.model, LogisticModel) else gbdt_size(self.model)
        return 0

    def score(self, query: str, items: Sequence[MemoryItem], item_features: np.ndarray | None = None,
              global_features: np.ndarray | None = None, head_logits: np.ndarray | None = None) -> np.ndarray:
        n = len(items)
        if n == 0:
            return np.zeros(0)
        if self.kind == "bm25":
            return -np.arange(n, dtype=np.float64)
        if self.kind == "rrf":
            return rrf_scores(self.dense.score(query, [labelled_text(item) for item in items]))
        if self.kind == "pointwise":
            return pointwise_scores(self.model, rows(item_features, global_features))
        if self.kind == "cross_encoder":
            return self.cross.score(query, [item.content for item in items])
        if head_logits is None:
            raise ValueError("the head scorer needs the policy's logits")
        return np.asarray(head_logits, dtype=np.float64)


class Stopwatch:
    """Accumulates wall time of the calls it wraps (for the selection-latency column)."""

    def __init__(self) -> None:
        self.seconds = 0.0

    def wrap(self, function):
        def timed(*args, **kwargs):
            started = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                self.seconds += time.perf_counter() - started

        return timed
