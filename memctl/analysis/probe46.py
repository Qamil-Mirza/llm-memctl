"""§46 measurement A: positions, labels, controls and the per-layer contrastive probe.

Spec: research/proposals/2026-10-30-exp46-instrument-check.md §2-§3 (rev 0, e70d03f). Positions are token indices into
the full chat text (`jacobian.wrap_chat`, as §45), found from character offsets:
- P_last: the last prompt token (the generation prompt's final token, as §45 `last_prompt_index`).
- RC P_B: the first token overlapping the supervisor's name in the "{A}'s supervisor is {B}" sentence.
- RC P_q: the first token overlapping A's name inside the question (its last occurrence in the prompt).
- TU P_cur / P_old: the first token overlapping the first occurrence of the current / superseded city.
Residual = output of decoder layer l (0-35), pre final norm = HF hidden_states[l + 1].

Probe: StandardScaler + multinomial L2 logistic regression, fitted on dev. Dev CV (CV_SCHEME): folds grouped by the
(format x hardness) cell, so every label is in every training fold (each dev label occurs in one dev base case only,
so base-grouped folds would hold out the label itself). C is chosen by that CV. Controls on the same folds: c1
Hewitt-Liang control task, a random pool label per base case (seed 46): its dev-CV accuracy measures how far base-case
identity alone is decodable, and selectivity = probe dev-CV - c1 dev-CV; c2 presence, the first distractor name in
the prompt (H1/H2 cells only), dev CV only (its held labels never occur in dev); c3 chance = 1 / pool items present.
Readout criterion (§3 with rev 1 ruling 3 and rev 2 ruling 4): dev-CV conditions read before the held read
(selectivity >= 0.30; c2 dev-CV accuracy below the Wilson lower bound of the probe's dev-CV accuracy), then held
accuracy >= 0.70 with Wilson lower bound >= 0.55; held accuracy - 1/12 and chance reported beside.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path
from typing import Sequence

import numpy as np

from memctl.analysis.maps import wilson

METHOD_VERSION = "exp46_probe_v0"
SEED = 46
N_LAYERS = 36
C_GRID = (0.001, 0.01, 0.1, 1.0)
CV_FOLDS = 5                      # used only when no groups are given
CV_SCHEME = "format_hardness"     # dev-CV groups: the (format, hardness) cell of each prompt
ACC_MIN, LOWER_MIN, SELECTIVITY_MIN = 0.70, 0.55, 0.30
VERDICT_LAYER_MAX = 29            # §5: "first decodable layer at P_last <= 29"
TASKS = ("RC", "TU")
# (task, target, position): the probe reads; the first is the verdict probe (§5)
READS = (("RC", "supervisor", "P_last"), ("RC", "supervisor", "P_q"),
         ("TU", "current", "P_last"), ("TU", "superseded", "P_last"))
VERDICT_READ = ("RC", "supervisor", "P_last")
DESIGN_MARKER = "design_committed.json"


# ---------------------------------------------------------------------------------------------- positions

def _token_at(offsets: Sequence[tuple[int, int]], char_start: int, char_end: int) -> int:
    for i, (s, e) in enumerate(offsets):
        if e > char_start and s < char_end and e > s:
            return i
    raise ValueError(f"no token overlaps chars {char_start}-{char_end}")


def char_spans(case: dict) -> dict[str, tuple[int, int]]:
    """Prompt-relative character spans of the §46 sites for one case."""
    p = case["prompt"]
    out: dict[str, tuple[int, int]] = {}
    if case["task"] == "RC":
        a, b = case["entities"][0], case["entities"][1]
        m = re.search(re.escape(f"{a}'s supervisor is ") + r"(" + re.escape(b) + r")\b", p)
        if m is None:
            raise ValueError(f"{case['case_id']}: supervisor sentence not found")
        out["P_B"] = m.span(1)
        q = p.rfind(case["question"])
        if q < 0:
            raise ValueError(f"{case['case_id']}: question not found")
        m = re.search(r"\b" + re.escape(a) + r"\b", case["question"])
        out["P_q"] = (q + m.start(), q + m.end())
    elif case["task"] == "TU":
        cur, old = case["gold"], case["competitors"][0]
        for name, city in (("P_cur", cur), ("P_old", old)):
            m = re.search(r"\b" + re.escape(city) + r"\b", p)
            if m is None:
                raise ValueError(f"{case['case_id']}: {city} not found")
            out[name] = m.span()
    else:
        raise ValueError(f"task {case['task']} is not in §46")
    return out


def positions(tokenizer, chat: str, case: dict) -> dict[str, int]:
    """Token indices (into the chat's token ids, no special tokens added) of P_last and the task's sites."""
    enc = tokenizer(chat, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    base = chat.index(case["prompt"])
    out = {"P_last": len(enc["input_ids"]) - 1}
    for name, (s, e) in char_spans(case).items():
        out[name] = _token_at(offsets, base + s, base + e)
    return out


# ---------------------------------------------------------------------------------------------- labels

def first_distractor(case: dict) -> str | None:
    """The name of the first distractor fact as it appears in the prompt (H1/H2); None on H0."""
    names = {d["name"] for d in case["distractors"]}
    if not names:
        return None
    best = None
    for n in names:
        m = re.search(r"\b" + re.escape(n) + r"\b", case["prompt"])
        if m and (best is None or m.start() < best[0]):
            best = (m.start(), n)
    return best[1] if best else None


def names_present(case: dict, pool: Sequence[str]) -> int:
    return sum(1 for n in pool if re.search(r"\b" + re.escape(n) + r"\b", case["prompt"]))


def control_labels(base_indices: Sequence[int], pool: Sequence[str], seed: int = SEED) -> dict[int, str]:
    """c1: one random pool label per base case (Hewitt-Liang control task), fixed by the seed."""
    rng = random.Random(seed)
    return {b: rng.choice(list(pool)) for b in sorted(set(base_indices))}


def target_label(case: dict, target: str) -> str:
    if target == "supervisor":
        return case["entities"][1]            # B (B' in v1); in v2 B is the gold
    if target == "current":
        return case["gold"]
    if target == "superseded":
        return case["competitors"][0]
    raise ValueError(target)


# ---------------------------------------------------------------------------------------------- probe

def cv_group(case: dict) -> str:
    return f"{case['format']}|{case['hardness']}"


def _make(c: float):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000))


def _splits(y: np.ndarray, groups, folds: int, seed: int):
    """Grouped folds (one group held out at a time) when groups are given, else stratified k-fold; None if fewer
    than two folds are possible."""
    from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold

    if groups is not None:
        g = np.asarray(groups)
        if len(set(g.tolist())) < 2:
            return None
        return list(LeaveOneGroupOut().split(np.zeros(len(y)), y, g))
    counts = np.unique(y, return_counts=True)[1]
    k = int(min(folds, counts.min())) if len(counts) > 1 else 0
    if k < 2:
        return None
    return list(StratifiedKFold(n_splits=k, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))


def _oof_predict(X: np.ndarray, y: np.ndarray, c: float, splits) -> np.ndarray:
    pred = np.empty(len(y), dtype=object)
    for tr, te in splits:
        if len(set(y[tr].tolist())) < 2:
            pred[te] = y[tr][0] if len(tr) else None
            continue
        pred[te] = _make(c).fit(X[tr], y[tr]).predict(X[te])
    return pred


class ConstantModel:
    """The fit on a single-class training set: predicts that class."""

    def __init__(self, label):
        self.label = label

    def predict(self, X):
        return np.array([self.label] * len(X), dtype=object)


def fit_probe(X: np.ndarray, y: Sequence[str], c_grid: Sequence[float] = C_GRID, folds: int = CV_FOLDS,
              seed: int = SEED, groups=None) -> dict:
    """Dev fit: C by CV (grouped if `groups`), refit on all dev rows. Returns {"model", "C", "cv_scores"}."""
    y = np.asarray(y)
    splits = _splits(y, groups, folds, seed) if len(c_grid) > 1 else None
    scores = {}
    if splits:
        for c in c_grid:
            scores[c] = float(np.mean(_oof_predict(X, y, c, splits) == y))
        best = max(c_grid, key=lambda c: (scores[c], -c))
    else:
        best = c_grid[len(c_grid) // 2]

    if len(set(y.tolist())) < 2:
        return {"model": ConstantModel(y[0] if len(y) else None), "C": best, "cv_acc": None, "cv_scores": {},
                "degenerate": True}
    model = _make(best).fit(X, y)
    return {"model": model, "C": best, "cv_acc": scores.get(best), "cv_scores": {str(c): s for c, s in scores.items()}}


def cv_accuracy(X: np.ndarray, y: Sequence[str], c: float, folds: int = CV_FOLDS, seed: int = SEED,
                groups=None) -> dict:
    """Pooled out-of-fold accuracy at a fixed C, with a Wilson interval."""
    y = np.asarray(y)
    splits = _splits(y, groups, folds, seed) if len(y) else None
    if not splits:
        return {"n": int(len(y)), "correct": None, "acc": None, "lower": None, "upper": None, "folds": 0}
    ok = int(np.sum(_oof_predict(X, y, c, splits) == y))
    p, lo, hi = wilson(ok, len(y))
    return {"n": int(len(y)), "correct": ok, "acc": p, "lower": lo, "upper": hi, "folds": len(splits)}


def accuracy(model, X: np.ndarray, y: Sequence[str]) -> dict:
    """Accuracy with a Wilson interval; labels the probe never saw count as wrong."""
    y = np.asarray(y)
    if len(y) == 0:
        return {"n": 0, "correct": 0, "acc": None, "lower": None, "upper": None}
    pred = model.predict(X)
    ok = int(np.sum(pred == y))
    p, lo, hi = wilson(ok, len(y))
    return {"n": int(len(y)), "correct": ok, "acc": p, "lower": lo, "upper": hi}


def dev_conditions(probe_cv: dict, c1_cv: dict, presence_cv: dict | None) -> bool:
    """The dev-CV clauses, read before any held read: selectivity (probe - c1) >= 0.30 and, where a presence control
    exists (RC), its accuracy below the probe's dev-CV Wilson lower bound."""
    if probe_cv["acc"] is None or c1_cv["acc"] is None:
        return False
    if probe_cv["acc"] - c1_cv["acc"] < SELECTIVITY_MIN:
        return False
    if presence_cv is not None:
        if presence_cv["acc"] is None or probe_cv is None or probe_cv["lower"] is None:
            return False
        if presence_cv["acc"] >= probe_cv["lower"]:
            return False
    return True


def decodable(held: dict, dev_ok: bool) -> bool:
    """§3 readout criterion: the dev-CV clauses hold and held accuracy >= 0.70 with Wilson lower bound >= 0.55."""
    if not dev_ok or held["acc"] is None:
        return False
    return held["acc"] >= ACC_MIN and held["lower"] >= LOWER_MIN


def first_decodable(per_layer: dict[int, bool]) -> int | None:
    for layer in sorted(per_layer):
        if per_layer[layer]:
            return layer
    return None


# ---------------------------------------------------------------------------------------------- held guard

def design_stamp(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def require_design(out_dir: Path, config: dict) -> dict:
    """The held read refuses unless design_committed.json exists, names a commit and matches this design."""
    marker = Path(out_dir) / DESIGN_MARKER
    if not marker.exists():
        raise PermissionError(f"{marker} is missing: commit the dev design before any held read")
    payload = json.loads(marker.read_text())
    if not payload.get("commit"):
        raise PermissionError(f"{marker} names no commit")
    if payload.get("design_sha256") != design_stamp(config):
        raise PermissionError(f"{marker} design_sha256 does not match this probe design")
    return payload
