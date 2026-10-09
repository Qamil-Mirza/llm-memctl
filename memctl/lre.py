"""LRE, "Learning What Not to Forget" (arXiv 2606.20954): its conversational-QA scorer, re-implemented (EXPERIMENTS.md §27).

Source read: github.com/NusRAT-LiA/LRE at commit 1aa51d672d772b27a272cab1393cd9968c13d158 (pyproject says MIT; the
repository has no LICENSE file). What is copied, from `experiments/conversational_qa/`:

- Labels (`proxy_labels.answer_overlap_labels`, the `lre_self_sup` condition): a unit is positive when its tokens
  cover at least 0.4 of the gold answer's tokens (recall, not F1). Tokens are the answer's and the unit's text after
  `dataset.normalize`: lower case, the articles a/an/the removed, punctuation deleted, split on white space. Stop
  words other than the articles are kept.
- Features (`lre_qa.lre_scores` with fuse=True): TF-IDF of the unit's text (sklearn's TfidfVectorizer with English
  stop words and at most 10,000 terms, fitted on the training units) beside six trajectory features
  (`dataset._traj_features`, `lre_qa.TRAJ_KEYS`), standardised on the training units.
- Model: sklearn LogisticRegression (L2, C = 1, class_weight "balanced", max_iter 1000, random_state 0).

Nothing here sees the question: the scorer is query-blind, as in the paper.

Inference needs no sklearn: the model is stored as JSON (vocabulary, idf, weights, scaler) and TF-IDF is recomputed
with sklearn's own analyser rules (lower case, token pattern (?u)\\b\\w\\w+\\b, its English stop-word list, raw
counts times idf, L2-normalised). A test checks that the JSON model scores equal sklearn's on the same rows.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

SOURCE = "github.com/NusRAT-LiA/LRE@1aa51d672d772b27a272cab1393cd9968c13d158"
FORMAT = "memctl-lre-1"
TRAJ_KEYS = ("f_pos", "f_recency", "f_log_len", "f_num_digit", "f_has_q", "f_caps_words")
THRESHOLD = 0.4  # their SELFSUP_THRESHOLD, metric "recall"
MAX_FEATURES = 10000
TOKEN_PATTERN = r"(?u)\b\w\w+\b"  # sklearn's default

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = re.compile(r"[^\w\s]")


# ---- labels (their rule) ------------------------------------------------------------------

def normalize(text: str) -> str:
    """Their `dataset.normalize`."""
    text = _ARTICLES.sub(" ", text.lower())
    text = _PUNCT.sub("", text)
    return " ".join(text.split())


def content_tokens(text: str) -> set[str]:
    return set(normalize(text).split())


def answer_recall(answer: str, text: str) -> float:
    """Share of the answer's tokens found in the text (their `_overlap` with metric "recall")."""
    answer_tokens = content_tokens(answer)
    if not answer_tokens:
        return 0.0
    return len(answer_tokens & content_tokens(text)) / len(answer_tokens)


def answer_overlap_labels(texts: list[str], answers: list[str], threshold: float = THRESHOLD) -> list[int]:
    """Their `answer_overlap_labels`: 1 when some non-empty answer has recall >= threshold in the unit."""
    answers = [str(a) for a in answers if a is not None and str(a).strip() and content_tokens(str(a))]
    return [int(max((answer_recall(a, t) for a in answers), default=0.0) >= threshold) for t in texts]


# ---- features ---------------------------------------------------------------------------

def traj_features(text: str, idx: int, n: int) -> list[float]:
    """Their `_traj_features`, in TRAJ_KEYS order. `idx` is the unit's place (0-based) among `n` units."""
    tokens = text.split()
    return [
        idx / max(1, n),
        1.0 / (1 + (n - 1 - idx)),
        math.log1p(len(tokens)),
        float(sum(c.isdigit() for c in text)),
        float("?" in text),
        float(sum(1 for w in tokens if w[:1].isupper())),
    ]


# ---- the model --------------------------------------------------------------------------

class LREModel:
    """The fitted scorer. logit(unit) = intercept + w_text . tfidf(text) + w_traj . standardised(traj)."""

    def __init__(self, data: dict) -> None:
        if data.get("format") != FORMAT:
            raise ValueError(f"not an LRE model of format {FORMAT}")
        self.data = data
        self.vocabulary = {word: column for column, word in enumerate(data["vocabulary"])}
        self.idf = data["idf"]
        self.coef_text = data["coef_text"]
        self.stop_words = frozenset(data["stop_words"])
        self.token_pattern = re.compile(data.get("token_pattern", TOKEN_PATTERN))
        self.mean, self.scale, self.coef_traj = data["traj_mean"], data["traj_scale"], data["coef_traj"]
        self.intercept = float(data["intercept"])

    # -- training (sklearn only here) --
    @classmethod
    def fit(cls, texts: list[str], trajs: list[list[float]], labels: list[int], meta: dict | None = None) -> "LREModel":
        import numpy as np
        from scipy.sparse import csr_matrix, hstack
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        y = np.asarray(labels, dtype=int)
        if len(set(y.tolist())) < 2:
            raise ValueError("LRE needs both labels in the training rows")
        vectorizer = TfidfVectorizer(stop_words="english", min_df=1, max_features=MAX_FEATURES)
        x_text = vectorizer.fit_transform(texts)
        traj = np.asarray(trajs, dtype=float)
        scaler = StandardScaler().fit(traj)
        x = hstack([x_text, csr_matrix(scaler.transform(traj))]).tocsr()
        model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=0).fit(x, y)
        n_text = x_text.shape[1]
        coef = model.coef_[0]
        vocabulary = [None] * n_text
        for word, column in vectorizer.vocabulary_.items():
            vocabulary[column] = word
        data = {
            "format": FORMAT,
            "source": SOURCE,
            "vocabulary": vocabulary,
            "idf": vectorizer.idf_.tolist(),
            "coef_text": coef[:n_text].tolist(),
            "stop_words": sorted(vectorizer.get_stop_words()),
            "token_pattern": vectorizer.token_pattern,
            "traj_keys": list(TRAJ_KEYS),
            "traj_mean": scaler.mean_.tolist(),
            "traj_scale": scaler.scale_.tolist(),
            "coef_traj": coef[n_text:].tolist(),
            "intercept": float(model.intercept_[0]),
            "train": {**(meta or {}), "rows": int(len(y)), "positives": int(y.sum()),
                      "n_iter": int(model.n_iter_[0])},
        }
        made = cls(data)
        made._sklearn = (vectorizer, scaler, model)  # for the equality test only
        return made

    # -- storage --
    def save(self, path: str | Path) -> int:
        text = json.dumps(self.data, separators=(",", ":"))
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(text)
        return len(text.encode())

    @classmethod
    def load(cls, path: str | Path) -> "LREModel":
        return cls(json.loads(Path(path).read_text()))

    # -- scoring --
    def text_logit(self, text: str) -> float:
        """w_text . tfidf(text): constant for a text, so a controller computes it once per item."""
        counts = Counter(w for w in self.token_pattern.findall(text.lower()) if w not in self.stop_words)
        weights = {self.vocabulary[w]: c * self.idf[self.vocabulary[w]] for w, c in counts.items() if w in self.vocabulary}
        norm = math.sqrt(sum(v * v for v in weights.values()))
        if norm == 0:
            return 0.0
        return sum(self.coef_text[column] * v for column, v in weights.items()) / norm

    def traj_logit(self, features: list[float]) -> float:
        return sum(c * (f - m) / (s if s else 1.0) for c, f, m, s in zip(self.coef_traj, features, self.mean, self.scale))

    def logit(self, text: str, idx: int, n: int) -> float:
        return self.intercept + self.text_logit(text) + self.traj_logit(traj_features(text, idx, n))

    def size_bytes(self) -> int:
        return len(json.dumps(self.data, separators=(",", ":")).encode())

    @property
    def n_parameters(self) -> int:
        return len(self.coef_text) + len(self.coef_traj) + 1
