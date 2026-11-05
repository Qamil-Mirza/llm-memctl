"""§51 inference (proposal rev 1a §7 + §13.1): the paired base-cluster bootstrap stratified by task (B = 10,000,
seed 5102), one-sided p-values over the valid draws, the degenerate rule, TOST, intersection-union, Holm, the M
selection ranking, the M3 Commit interaction and the pooled paired AUROC difference. Pure functions (no model).

A statistic is a function of a resample: `stat(idx_by_task) -> float | None`, where idx_by_task maps each task to an
integer array of resampled base positions (with repeats) into that task's base list. None (an empty denominator,
an undefined AUROC) marks a failed replicate (§7.1): dropped and counted; more than 1% of the attempted B -> no state.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

B_DEFAULT, SEED = 10_000, 5102
FAIL_MAX = 0.01


def boot_draws(n_by_task: dict[str, int], B: int = B_DEFAULT, seed: int = SEED) -> list[dict[str, np.ndarray]]:
    """B resamples; tasks in sorted order from one generator (so a fixed seed fixes every draw)."""
    rng = np.random.default_rng(seed)
    tasks = sorted(n_by_task)
    mats = {t: rng.integers(0, n_by_task[t], size=(B, n_by_task[t])) if n_by_task[t] else np.zeros((B, 0), int)
            for t in tasks}
    return [{t: mats[t][b] for t in tasks} for b in range(B)]


def bootstrap(stat: Callable, n_by_task: dict[str, int], B: int = B_DEFAULT, seed: int = SEED) -> dict:
    """The point estimate (identity resample) and the valid replicate values."""
    ident = {t: np.arange(n) for t, n in n_by_task.items()}
    est = stat(ident)
    vals = []
    for d in boot_draws(n_by_task, B, seed):
        v = stat(d)
        if v is not None and np.isfinite(v):
            vals.append(float(v))
    n_fail = B - len(vals)
    return {"est": None if est is None else float(est), "draws": np.asarray(vals), "B": B, "B_valid": len(vals),
            "n_fail": n_fail, "fail_ok": n_fail <= FAIL_MAX * B}


def p_greater(res: dict, theta0: float = 0.0) -> float | None:
    """H0: θ ≤ θ0 vs θ > θ0. p = (1 + #{θ*_b ≤ θ0}) / (B_valid + 1); degenerate (all draws = θ̂): 1/(B_valid+1) if θ̂
    > θ0 else 1. None when the statistic has no state (estimate undefined or > 1% failed)."""
    if res["est"] is None or not res["fail_ok"] or res["B_valid"] == 0:
        return None
    d, bv = res["draws"], res["B_valid"]
    if np.all(d == res["est"]):
        return 1.0 / (bv + 1) if res["est"] > theta0 else 1.0
    return float((1 + np.sum(d <= theta0)) / (bv + 1))


def p_less(res: dict, theta0: float = 0.0) -> float | None:
    """Mirror: H0: θ ≥ θ0 vs θ < θ0."""
    if res["est"] is None or not res["fail_ok"] or res["B_valid"] == 0:
        return None
    d, bv = res["draws"], res["B_valid"]
    if np.all(d == res["est"]):
        return 1.0 / (bv + 1) if res["est"] < theta0 else 1.0
    return float((1 + np.sum(d >= theta0)) / (bv + 1))


def bounds(res: dict) -> dict:
    """One-sided 97.5% percentile bounds (reported beside the p-value; the p-value decides)."""
    if res["B_valid"] == 0:
        return {"lo": None, "hi": None}
    lo, hi = np.percentile(res["draws"], [2.5, 97.5])
    return {"lo": float(lo), "hi": float(hi)}


def p_sided(res: dict, direction: int, theta0: float = 0.0) -> float | None:
    """direction +1: alternative θ > θ0; −1: θ < θ0."""
    return p_greater(res, theta0) if direction > 0 else p_less(res, theta0)


def tost(res: dict, margin: float) -> float | None:
    """Equivalence |θ| < margin: the larger of the two one-sided p-values. margin 0 (an SD of 0) -> unestimable:
    p = 1 (§13.4a)."""
    if margin <= 0:
        return 1.0
    a, b = p_greater(res, -margin), p_less(res, margin)
    return None if a is None or b is None else max(a, b)


def iut(ps: Sequence[float | None]) -> float | None:
    """Intersection-union composite: the maximum p (None if any elementary test has no state)."""
    return None if any(p is None for p in ps) else max(ps)


def holm(ps: dict[str, float | None], alpha: float = 0.05) -> dict[str, dict]:
    """Holm step-down over the named hypotheses; a None p is never rejected and counts in m."""
    m = len(ps)
    order = sorted(ps, key=lambda k: (1.0 if ps[k] is None else ps[k], k))
    out, stop = {}, False
    for i, k in enumerate(order):
        p = ps[k]
        thr = alpha / (m - i)
        rej = (not stop) and p is not None and p <= thr
        if not rej:
            stop = True
        adj = None if p is None else min(1.0, max(min(1.0, (m - j) * (ps[order[j]] if ps[order[j]] is not None else 1.0))
                                                    for j in range(i + 1)))
        out[k] = {"p": p, "threshold": thr, "reject": rej, "p_holm": adj}
    return out


# ---------------------------------------------------------------------------------------------- data layout

class Bases:
    """Per-task base lists and per-base row values for base-weighted statistics.

    `values[arm][base_key]` = list of per-row values (already restricted to a population); a base absent or with an
    empty list has no rows in that population. Base-weighted mean: each base with rows has weight 1, equal within."""

    def __init__(self, base_task: dict[str, str]):
        self.tasks = sorted(set(base_task.values()))
        self.by_task = {t: sorted(b for b, x in base_task.items() if x == t) for t in self.tasks}

    def n_by_task(self) -> dict[str, int]:
        return {t: len(v) for t, v in self.by_task.items()}

    def resampled(self, idx: dict[str, np.ndarray]) -> list[str]:
        return [self.by_task[t][i] for t in self.tasks for i in idx.get(t, [])]


def bw_mean(per_base: dict[str, float], keys: Sequence[str]) -> float | None:
    v = [per_base[k] for k in keys if k in per_base]
    return float(np.mean(v)) if v else None


def diff_stat(bases: Bases, a: dict[str, float], b: dict[str, float]) -> Callable:
    """Paired base-weighted mean difference a − b over bases present in both."""
    def stat(idx):
        keys = [k for k in bases.resampled(idx) if k in a and k in b]
        if not keys:
            return None
        return float(np.mean([a[k] - b[k] for k in keys]))
    return stat


def interaction_stat(bases: Bases, fa_x: dict, fa_y: dict, una_x: dict, una_y: dict) -> Callable:
    """§13.4b: [(x − y) on FA] − [(x − y) on UNA], base-weighted per population over the same resample."""
    def stat(idx):
        keys = bases.resampled(idx)
        k1 = [k for k in keys if k in fa_x and k in fa_y]
        k2 = [k for k in keys if k in una_x and k in una_y]
        if not k1 or not k2:
            return None
        return float(np.mean([fa_x[k] - fa_y[k] for k in k1]) - np.mean([una_x[k] - una_y[k] for k in k2]))
    return stat


def auroc(scores: Sequence[float], labels: Sequence[int]) -> float | None:
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    allv = np.concatenate([pos, neg])
    ranks = _rankdata(allv)
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), float)
    xs = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return r


def auroc_diff_stat(bases: Bases, rows_by_base: dict[str, list[tuple[float, float, int]]]) -> Callable:
    """§13.4c: pooled AUROC(full) − AUROC(baseline) over the resampled bases' rows; rows = (full, baseline, label).
    A replicate with one class overall fails."""
    def stat(idx):
        rs = [r for k in bases.resampled(idx) for r in rows_by_base.get(k, [])]
        if not rs:
            return None
        f, b, y = zip(*rs)
        a1, a0 = auroc(f, y), auroc(b, y)
        return None if a1 is None or a0 is None else a1 - a0
    return stat


def m_rank(effect: float, se: float, sign: int) -> float:
    """§13.1: oriented effect / max(SE, 0.01)."""
    return sign * effect / max(se, 0.01)


def m_order(cands: Sequence[dict]) -> list[dict]:
    """Eligible M candidates (oriented effect ≥ 0.05, bound beyond 0, support ≥ 20) by rank; ties: fewer components,
    fewer changed parameters, submission index. Each dict: effect, se, sign, n_components, n_params, index,
    bound_ok, support."""
    el = [c for c in cands if c["sign"] * c["effect"] >= 0.05 and c["bound_ok"] and c["support"] >= 20]
    return sorted(el, key=lambda c: (-m_rank(c["effect"], c["se"], c["sign"]), c["n_components"], c["n_params"],
                                     c["index"]))
