"""§50 why false abstention: the answer-bearing rule, elicited scores, margins, cell estimands with the base bootstrap,
profile labels, the S validity gate, the stage-2 projector and arm vectors, Holm. Pure functions (no model).

Spec: research/proposals/2026-11-02-exp50-why-false-abstention.md rev 1 (+ rev 1a); D-70. Fixed choices of this build
(the proposal leaves them open; each is stamped into the design file):
- Answer-bearing note: SOME occurrence of the gold (word boundary) has none of the negators within the 6 words before
  it; for TU the gold is also the last pool city mentioned in the note.
- S gate AUROC: answered ANS rows (outcome gold, valid S) against UNA-rel rows (all hardness) and, separately, answered
  ANS-H1 against UNA-swap (H1); 95% CI by base bootstrap within the task. Yes rates: answered ANS >= 0.70, and UNA-rel
  and UNA-swap each <= 0.30. Polarity agreement over every dev-check row with both scores valid.
- R_S cell weights: the false-abstention count per task x hardness cell (equal weights if none).
- Random arms: r̂ ~ N(0, I) normalised, seeded by sha256("50-{pair}-{arm}-{seed}").
"""

from __future__ import annotations

import hashlib
import re
from typing import Sequence

import numpy as np

from memctl.analysis.abst49 import auroc  # noqa: F401  (re-exported)
from memctl.analysis.maps import spearman  # noqa: F401

METHOD_VERSION = "exp50_v0"
NEGATORS = ("not", "no longer", "never", "moved from")
NEG_WINDOW = 6
YES = ("yes", " yes", "Yes", " Yes")
NO = ("no", " no", "No", " No")
MASS_MIN = 0.5
GATE_AUROC_LO, GATE_YES_ANS, GATE_YES_UNA, GATE_POLARITY = 0.75, 0.70, 0.30, 0.70
FEAS_N, FEAS_BASES = 30, 15
DELTA_M = 0.5
DELTA_S_FRAC = 0.25
R_S_MIN = 0.20
OVERLAP_MIN = 0.50
BOOT_N, BOOT_SEED, DROP_MAX = 10_000, 0, 0.05
L_RELEASE_MIN, L_PAIRS_MIN, L_BASES_MIN = 0.50, 20, 10
NOOP_MIN = 0.95
USA_AUROC_LO, UM_SPEARMAN = 0.60, 0.30
COS_MAX = 0.8
OPEN_PAIRS, OPEN_BASES = 20, 10
RAND_SEEDS = 5
LAYERS = tuple(range(1, 36))


# ---------------------------------------------------------------------------------------------- note rule

def answer_bearing(note: str, gold: str | None, task: str, cities: Sequence[str]) -> bool:
    if not gold:
        return False
    occ = [m for m in re.finditer(r"\b" + re.escape(gold) + r"\b", note)]
    if not occ:
        return False
    ok = False
    for m in occ:
        before = " ".join(re.findall(r"[A-Za-z']+", note[:m.start()])[-NEG_WINDOW:]).lower()
        if not any(re.search(r"\b" + re.escape(n) + r"\b", before) for n in NEGATORS):
            ok = True
            break
    if not ok:
        return False
    if task == "TU":
        hits = sorted((mm.start(), c) for c in cities for mm in re.finditer(r"\b" + re.escape(c) + r"\b", note))
        return bool(hits) and hits[-1][1] == gold
    return True


# ---------------------------------------------------------------------------------------------- scores

def softmax(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    e = np.exp(z - z.max())
    return e / e.sum()


def yes_no(logits: np.ndarray, yes_ids: Sequence[int], no_ids: Sequence[int]) -> dict:
    p = softmax(logits)
    py, pn = float(p[list(yes_ids)].sum()), float(p[list(no_ids)].sum())
    mass = py + pn
    s = py / mass if mass > 0 else None
    return {"p_yes": py, "p_no": pn, "mass": mass, "s": s, "valid": mass >= MASS_MIN,
            "yes": None if s is None else bool(s >= 0.5)}


def expected_digit(logits: np.ndarray, digit_ids: Sequence[Sequence[int]]) -> dict:
    """digit_ids[d] = token ids for digit d (forms summed). Expected value over the renormalised digit mass."""
    p = softmax(logits)
    pd = np.array([p[list(ids)].sum() for ids in digit_ids])
    mass = float(pd.sum())
    return {"kv": float((pd * np.arange(len(pd))).sum() / mass) if mass > 0 else None, "mass": mass}


def margins(logits: np.ndarray, gold: str, city_ids: dict[str, int], unknown_id: int) -> dict:
    z = np.asarray(logits, dtype=np.float64)
    zg = z[city_ids[gold]]
    rival = max(z[i] for c, i in city_ids.items() if c != gold)
    return {"m_city": float(zg - rival), "m_commit": float(zg - z[unknown_id])}


# ---------------------------------------------------------------------------------------------- estimands

def _cell_key(r: dict, coarse: bool) -> tuple:
    return (r["task"], r["hardness"]) if coarse else (r["task"], r["hardness"], r["gold"])


def cell_delta(rows: Sequence[dict], field: str, coarse: bool | None = None) -> dict:
    """Δ = Σ_c w_c (mean_fa − mean_ans) / Σ w_c over cells with both classes, w_c = #fa in c. rows: dicts with
    task, hardness, gold, cls ('fa' / 'ans') and `field`. coarse None: task x hardness x city with the fallback to
    task x hardness when < 50% of false abstentions are in overlapping cells."""
    rs = [r for r in rows if r.get(field) is not None]
    n_fa = sum(r["cls"] == "fa" for r in rs)
    for c in ([False, True] if coarse is None else [coarse]):
        cells: dict = {}
        for r in rs:
            cells.setdefault(_cell_key(r, c), {"fa": [], "ans": []})[r["cls"]].append(r[field])
        both = {k: v for k, v in cells.items() if v["fa"] and v["ans"]}
        covered = sum(len(v["fa"]) for v in both.values())
        if coarse is None and not c and (n_fa == 0 or covered / n_fa < OVERLAP_MIN):
            continue
        if not both:
            return {"delta": None, "coarse": c, "n_cells": 0, "covered_fa": 0, "n_fa": n_fa,
                    "cells_no_overlap": len(cells)}
        w = np.array([len(v["fa"]) for v in both.values()], float)
        d = np.array([np.mean(v["fa"]) - np.mean(v["ans"]) for v in both.values()])
        dof = sum(len(v["ans"]) - 1 for v in both.values())
        ss = sum(((np.asarray(v["ans"]) - np.mean(v["ans"])) ** 2).sum() for v in both.values())
        sd = float(np.sqrt(ss / dof)) if dof > 0 and ss > 0 else None
        delta = float((w * d).sum() / w.sum())
        return {"delta": delta, "coarse": c, "n_cells": len(both), "covered_fa": covered, "n_fa": n_fa,
                "cells_no_overlap": len(cells) - len(both), "sd_ans": sd,
                "delta_std": delta / sd if sd else None}
    return {"delta": None, "coarse": None, "n_cells": 0, "covered_fa": 0, "n_fa": n_fa, "cells_no_overlap": None}


def reference_s(ans_rows: Sequence[dict], una_rows: Sequence[dict], fa_rows: Sequence[dict]) -> float | None:
    """R_S = Σ_h w_h (s̄_answered,h − s̄_UNA-rel,h) / Σ w_h over the task's hardness cells, w_h = #fa at h."""
    out, ws = 0.0, 0.0
    hs = sorted({r["hardness"] for r in ans_rows} & {r["hardness"] for r in una_rows})
    if not hs:
        return None
    w = {h: sum(r["hardness"] == h for r in fa_rows) for h in hs}
    if sum(w.values()) == 0:
        w = {h: 1 for h in hs}
    for h in hs:
        a = [r["s"] for r in ans_rows if r["hardness"] == h]
        u = [r["s"] for r in una_rows if r["hardness"] == h]
        if a and u and w[h]:
            out += w[h] * (np.mean(a) - np.mean(u))
            ws += w[h]
    return out / ws if ws else None


def boot_bases(bases: Sequence[str], stat, n: int = BOOT_N, seed: int = BOOT_SEED) -> dict:
    """stat(list of base keys, with repeats) -> float or None. Resample bases with replacement; None replicates are
    dropped and counted; unstable if more than 5% drop. 95% and 90% percentile intervals."""
    ub = sorted(set(bases))
    rng = np.random.default_rng(seed)
    vals, dropped = [], 0
    for _ in range(n):
        v = stat([ub[i] for i in rng.integers(0, len(ub), len(ub))])
        if v is None or not np.isfinite(v):
            dropped += 1
        else:
            vals.append(v)
    if not vals:
        return {"lo95": None, "hi95": None, "lo90": None, "hi90": None, "dropped": dropped, "unstable": True}
    lo95, hi95, lo90, hi90 = np.percentile(vals, [2.5, 97.5, 5, 95])
    return {"lo95": float(lo95), "hi95": float(hi95), "lo90": float(lo90), "hi90": float(hi90), "dropped": dropped,
            "unstable": dropped > DROP_MAX * n, "n_boot": n}


def expand(rows_by_base: dict[str, list[dict]], sample: Sequence[str]) -> list[dict]:
    """Rows of the resampled bases; a base drawn k times contributes k copies (all its rows together)."""
    out = []
    for b in sample:
        out.extend(rows_by_base.get(b, []))
    return out


def state(point: float | None, ci: dict, delta: float | None, valid: bool = True) -> str:
    """§4.5 five states for one measure (delta = the equivalence margin; None: no 'similar' state)."""
    if not valid:
        return "invalid / underpowered"
    if point is None or ci.get("lo95") is None:
        return "inconclusive"
    if delta is not None and ci["hi95"] < 0 and point <= -delta:
        return "lower"
    if delta is None and ci["hi95"] < 0:
        return "lower"
    if delta is not None and -delta < ci["lo90"] and ci["hi90"] < delta:
        return "similar"
    if delta is not None and ci["lo95"] > 0 and point >= delta:
        return "higher"
    if delta is None and ci["lo95"] > 0:
        return "higher"
    return "inconclusive"


def profile(s_state: str, m_state: str) -> str:
    if s_state == "lower" and m_state == "similar":
        return "B-compatible"
    if s_state == "similar" and m_state == "lower":
        return "C-compatible"
    if s_state == "lower" and m_state == "lower":
        return "Joint"
    if s_state == "similar" and m_state == "similar":
        return "Neither separates"
    return f"Other (S {s_state}, M {m_state})"


def feasible(fa_rows: Sequence[dict], ans_rows: Sequence[dict]) -> bool:
    return (len(fa_rows) >= FEAS_N and len({r["base_key"] for r in fa_rows}) >= FEAS_BASES
            and len(ans_rows) >= FEAS_N and len({r["base_key"] for r in ans_rows}) >= FEAS_BASES)


# ---------------------------------------------------------------------------------------------- stage 2

def projector(U: np.ndarray) -> np.ndarray:
    """Π = U (UᵀU)⁺ Uᵀ for U [d, k]."""
    U = np.asarray(U, dtype=np.float64)
    return U @ np.linalg.pinv(U.T @ U) @ U.T


def unit(v: np.ndarray) -> np.ndarray | None:
    n = np.linalg.norm(v)
    return None if not np.isfinite(n) or n == 0 else np.asarray(v, np.float64) / n


def arm_vectors(h_r: np.ndarray, h_d: np.ndarray, u_s: np.ndarray | None, u_m: np.ndarray | None) -> dict:
    """The patched states of §5's table (W, P-S, P-M, P-span, P-perp) and each arm's displacement."""
    h_r, h_d = np.asarray(h_r, np.float64), np.asarray(h_d, np.float64)
    delta = h_d - h_r
    out = {"W": delta}
    if u_s is not None:
        out["P-S"] = u_s * (u_s @ delta)
    if u_m is not None:
        out["P-M"] = u_m * (u_m @ delta)
    cols = [u for u in (u_s, u_m) if u is not None]
    if cols:
        P = projector(np.stack(cols, axis=1))
        out["P-span"] = P @ delta
        out["P-perp"] = delta - P @ delta
    return {k: {"state": (h_r + v).astype(np.float32), "disp": v, "norm": float(np.linalg.norm(v))}
            for k, v in out.items()}


def rand_seed(pair: str, arm: str, k: int) -> int:
    return int(hashlib.sha256(f"50-{pair}-{arm}-{k}".encode()).hexdigest()[:8], 16)


def rand_states(h_r: np.ndarray, disp: np.ndarray, pair: str, arm: str, k_seeds: int = RAND_SEEDS,
                delta: np.ndarray | None = None) -> list[np.ndarray]:
    """h_r + ‖patch_X‖ · r̂ · sign(r̂ᵀΔ) for k seeded unit vectors: the norm is the arm's displacement ‖disp‖, the sign is
    taken against the whole Δ = h_donor − h_recipient (spec §5; build review fix 3). `delta` defaults to `disp` only
    for callers without a donor."""
    out = []
    norm = float(np.linalg.norm(disp))
    ref = np.asarray(disp if delta is None else delta, np.float64)
    for k in range(k_seeds):
        r = np.random.default_rng(rand_seed(pair, arm, k)).standard_normal(len(h_r))
        r /= np.linalg.norm(r)
        sgn = 1.0 if float(r @ ref) >= 0 else -1.0
        out.append((np.asarray(h_r, np.float64) + norm * sgn * r).astype(np.float32))
    return out


def holm(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, prev = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, max(prev, (m - i) * p))
        out[k] = adj
        prev = adj
    return out


def boot_p(vals: Sequence[float]) -> float:
    """Two-sided bootstrap p for a mean against 0 from its replicate values."""
    v = np.asarray(vals, float)
    if len(v) == 0:
        return 1.0
    return float(min(1.0, 2 * min((v <= 0).mean(), (v >= 0).mean())))
