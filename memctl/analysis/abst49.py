"""§49 legible abstentions: outcomes, the P_ans prefix, probes, repair vectors, gates, utility and the base-clustered
bootstrap. Pure functions (no model); the passes are in scripts/exp49_run.py, the report in scripts/exp49_report.py.

Spec: research/proposals/2026-11-01-exp49-legible-abstentions.md rev 0a (D-63). Readout: patch46 `first_city` on the
answer text (F2: the first line after the last "Answer:"; F1: the whole 16-token generation, the §45 F1 budget), no
judge. Classes: gold / abstain / competitor / wrong (another pool city) / degenerate (no pool city and no "unknown"
on the answer text, or no "Answer:" line in F2). UNA rows have no gold: any city is an unsupported answer ("wrong").

Rev 0b (B1-B7) overrides rev 0a. Fixed choices of this build (the proposal leaves them open; each is stamped into
the design file):
- P_pre = the last chat-template token (§45 / §46 P_last: the assistant header's newline after the prompt's "Note:").
- P_ans = the last token of the model's own generated prefix through "Answer:"; undefined (trial excluded from M2 / M3,
  counted) without the marker, when the prefix's text does not end in "Answer:" (a merged token), or when the
  generation hit 160 tokens without completing the Answer line (B3).
- Dev parts (B4): dev index k = base_index // 2, part k mod 3: 0 fit, 1 select, 2 gate-tune. The 12-way probes are fit
  on the fit part (their StandardScaler too) and used unchanged on every later part and on held; the sufficiency probe
  is refit on fit + select at its layer (B4).
- Centroids (B3) come from the fit + select parts (the gate-tune part, where α is tuned, contributes none): μ_c at
  (site, ℓ*) over gold-outcome trials with gold c, F2 if >= 5, else pooled F1 + F2 if >= 5 (at P_ans F1 has no
  site, so the fallback adds nothing there), else unestimable -> that R1 trial is "no release". μ_abst over ANS-F2
  false abstentions with a P_ans, >= 20, else R1 is not estimable as a whole.
- Utility: +1 gold, 0 abstain, −λ for wrong, competitor or degenerate (a degenerate output is not an abstention).
- The F1 replication arm adds α_LF·(μ_ĉ − μ_e) at ℓ*_F1(P_pre), step 0, gated on a pool-city output e with ĉ ≠ e;
  its random control is a matched-norm seeded vector at the same site.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Sequence

import numpy as np

from memctl.analysis import patch46 as T
from memctl.analysis.maps import ANSWER_MARKER, answer_line_done, wilson  # noqa: F401  (re-exported)

METHOD_VERSION = "exp49_v0"
SEED = 49
PROBE_C = 0.5
ALPHAS = (0.5, 1.0, 2.0)
LAMBDAS = (1.0, 2.0)
MAX_NEW = {"F1": 16, "F2": 160}
MAX_NEW_RELEASE = 16
R2_LAYER, R2_ALPHA = 29, 2.0          # B5: R2 (one-position), added at P_ans only
NOOP_PER_SPLIT = 20
R2_FILE = Path("runs/exp38_dev/qwen3b/directions.npz")
CLASSES = ("gold", "abstain", "competitor", "wrong", "degenerate")
UNSUPPORTED = ("wrong", "competitor", "degenerate")
ARMS = ("R1", "R2", "R3", "R4", "Dgold")
BOOT_N, BOOT_SEED = 10_000, 0
HELD_FALSE_ABST_MIN = 30
DESIGN_MARKER = "design_committed.json"


# ---------------------------------------------------------------------------------------------- outcomes

def answer_text(text: str, fmt: str) -> str | None:
    """F2: the first line after the last "Answer:" (None without the marker); F1: the whole text."""
    if fmt == "F1":
        return text
    low = text.lower()
    m = low.rfind(ANSWER_MARKER)
    if m < 0:
        return None
    return text[m + len(ANSWER_MARKER):].split("\n")[0]


def classify(answer: str | None, gold: str | None, competitors: Sequence[str], cities: Sequence[str]) -> dict:
    if answer is None:
        return {"outcome": "degenerate", "item": None}
    hit = T.first_city(answer, cities)
    if hit is None:
        return {"outcome": "degenerate", "item": None}
    item = hit[1]
    if item == "unknown":
        return {"outcome": "abstain", "item": item}
    if gold is not None and item == gold:
        return {"outcome": "gold", "item": item}
    if item in competitors:
        return {"outcome": "competitor", "item": item}
    return {"outcome": "wrong", "item": item}


def utility(outcome: str, lam: float) -> float:
    if outcome == "gold":
        return 1.0
    if outcome == "abstain":
        return 0.0
    return -lam


# ---------------------------------------------------------------------------------------------- P_ans prefix

def ans_prefix(gen_ids: Sequence[int], decode) -> dict:
    """The index j such that gen_ids[:j] is the generated prefix through "Answer:". `decode(ids) -> str`.
    Returns {"j", "ok", "why"}; ok False when there is no marker, nothing follows it, or the prefix's text does not
    end in "Answer:" (a merged token)."""
    text = decode(list(gen_ids))
    m = text.lower().rfind(ANSWER_MARKER)
    if m < 0:
        return {"j": None, "ok": False, "why": "no Answer: marker"}
    end = m + len(ANSWER_MARKER)
    j = None
    for k in range(len(gen_ids) + 1):
        if len(decode(list(gen_ids[:k]))) >= end:
            j = k
            break
    if j is None:
        return {"j": None, "ok": False, "why": "marker not reached"}
    head = decode(list(gen_ids[:j]))
    if not head.rstrip().lower().endswith(ANSWER_MARKER):
        return {"j": j, "ok": False, "why": f"prefix ends {head[-12:]!r}"}
    return {"j": j, "ok": True, "why": None}


def note_flags(note: str, gold: str | None, cities: Sequence[str]) -> dict:
    """note_gold: the gold city's string is in the generated note; note_substitute: another pool city is."""
    present = [c for c in cities if re.search(r"\b" + re.escape(c) + r"\b", note)]
    return {"note_gold": gold is not None and gold in present, "note_substitute": any(c != gold for c in present),
            "note_cities": present}


# ---------------------------------------------------------------------------------------------- probes

def probe_model(c: float = PROBE_C):
    """§46's pipeline (StandardScaler, exact row-span projection, lbfgs multinomial) at a fixed C."""
    from memctl.analysis.probe46 import _make
    return _make(c)


PARTS = ("fit", "select", "tune")


def dev_part(base_index: int) -> str:
    """B4: dev index k = base_index // 2; part k mod 3."""
    return PARTS[(base_index // 2) % 3]


def auroc(scores, labels) -> float | None:
    s, y = np.asarray(scores, float), np.asarray(labels)
    pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return None
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def choose_layer(X: np.ndarray, y: Sequence, fit: np.ndarray, sel: np.ndarray, metric: str = "acc") -> dict:
    """ℓ* = the layer with the best select-part score of a probe fit on the fit part (C 0.5); ties to the lower
    layer. X [n, L, d]; metric 'acc' (12-way) or 'auroc' (binary, class 1 = positive)."""
    y = np.asarray(y)
    fit, sel = np.asarray(fit, bool), np.asarray(sel, bool)
    scores = {}
    for layer in range(X.shape[1]):
        m = probe_model().fit(X[fit, layer], y[fit])
        if not sel.any():
            scores[layer] = None
        elif metric == "acc":
            scores[layer] = float(np.mean(m.predict(X[sel, layer]) == y[sel]))
        else:
            p = m.predict_proba(X[sel, layer])[:, list(m.classes_).index(1)]
            scores[layer] = auroc(p, y[sel])
    best = max(scores, key=lambda l: (scores[l] if scores[l] is not None else -1, -l))
    return {"layer": int(best), "score_by_layer": scores, "metric": metric, "n_fit": int(fit.sum()),
            "n_select": int(sel.sum())}


def shuffled_labels(y: Sequence, base_keys: Sequence[str], seed: int = SEED) -> np.ndarray:
    """Label-shuffled control: the per-base labels permuted across bases (every row of a base keeps one label)."""
    first: dict = {}
    for k, lab in zip(base_keys, y):
        if first.setdefault(k, lab) != lab:
            raise ValueError(f"base {k} has more than one label")
    ks = sorted(first)
    perm = np.random.default_rng(seed).permutation(len(ks))
    new = {ks[i]: first[ks[p]] for i, p in enumerate(perm)}
    return np.array([new[k] for k in base_keys], dtype=object)


# ---------------------------------------------------------------------------------------------- vectors

MU_MIN, MU_ABST_MIN = 5, 20


def class_means(X: np.ndarray, labels: Sequence[str]) -> dict[str, np.ndarray]:
    labels = np.asarray(labels)
    return {c: X[labels == c].astype(np.float64).mean(0) for c in sorted(set(labels.tolist()))}


def centroids(X: np.ndarray, labels: Sequence[str], fmts: Sequence[str], classes: Sequence[str],
              min_n: int = MU_MIN) -> dict:
    """B3: per class, the F2 mean if >= min_n F2 trials, else the pooled F1 + F2 mean if >= min_n, else
    unestimable. X [n, d] gold-outcome trials. Returns {"mu": {c: vec}, "source": {c: 'F2' | 'pooled' | None},
    "n": {c: (n_F2, n_all)}}."""
    labels, fmts = np.asarray(labels), np.asarray(fmts)
    mu, src, n = {}, {}, {}
    for c in classes:
        f2 = (labels == c) & (fmts == "F2")
        al = labels == c
        n[c] = (int(f2.sum()), int(al.sum()))
        if f2.sum() >= min_n:
            mu[c], src[c] = X[f2].astype(np.float64).mean(0), "F2"
        elif al.sum() >= min_n:
            mu[c], src[c] = X[al].astype(np.float64).mean(0), "pooled"
        else:
            src[c] = None
    return {"mu": mu, "source": src, "n": n}


def array_sha(arrays: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(arrays):
        h.update(str(k).encode())
        h.update(np.ascontiguousarray(arrays[k], dtype=np.float64).tobytes())
    return h.hexdigest()


def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("zero direction")
    return v / n


def r1_vector(h: np.ndarray, mu: dict[str, np.ndarray], mu_abst: np.ndarray, target: str, alpha: float) -> np.ndarray:
    """α·‖h‖·unit(μ_target − μ_abst) (float32)."""
    if target not in mu:
        raise KeyError(f"no dev class mean for {target!r}")
    return (alpha * np.linalg.norm(h.astype(np.float64)) * unit(mu[target] - mu_abst)).astype(np.float32)


def trial_seed(case_id: str, arm: str, seed: int = SEED) -> int:
    return int(hashlib.sha256(f"{seed}-{case_id}-{arm}".encode()).hexdigest()[:8], 16)


def random_like(norm: float, d: int, case_id: str, arm: str) -> np.ndarray:
    g = np.random.default_rng(trial_seed(case_id, arm)).standard_normal(d)
    return (g / np.linalg.norm(g) * norm).astype(np.float32)


def r2_vector(path: Path = R2_FILE) -> dict:
    """The §38 / §40 vector unchanged: v_29 of the §38 laptop file byte-exact, steer = 2·v_29 (exp40's checks). Rev 0b
    B5: added at P_ans only (one position, as §39's Steer-0), labelled "R2 (one-position)"."""
    from memctl.analysis import exp40
    raw = Path(path).read_bytes()
    v = np.load(path)["L29"]
    return exp40.check_direction(v, hashlib.sha256(raw).hexdigest())


def lf_vector(mu: dict[str, np.ndarray], target: str, output: str, alpha: float = 1.0) -> np.ndarray:
    return (alpha * (mu[target] - mu[output])).astype(np.float32)


# ---------------------------------------------------------------------------------------------- gates and utility

def policy_outcomes(base_outcomes: Sequence[str], arm_outcomes: Sequence[str], fire: Sequence[bool]) -> list[str]:
    return [a if f else b for b, a, f in zip(base_outcomes, arm_outcomes, fire)]


def mean_utility(outcomes: Sequence[str], lam: float) -> float | None:
    return float(np.mean([utility(o, lam) for o in outcomes])) if len(outcomes) else None


def choose_tau(scores: Sequence[float], base_outcomes: Sequence[str], arm_outcomes: Sequence[str],
               lam: float = 1.0) -> dict:
    """τ maximising U(fire iff score >= τ) over the candidate set {each observed score} ∪ {+inf (never)}; ties go to
    the larger τ (fewer fires)."""
    s = np.asarray(scores, dtype=float)
    cands = sorted(set(s.tolist())) + [float("inf")]
    best = None
    for tau in cands:
        u = mean_utility(policy_outcomes(base_outcomes, arm_outcomes, s >= tau), lam)
        if best is None or u > best[1] or (u == best[1] and tau > best[0]):
            best = (tau, u)
    return {"tau": best[0], "U": best[1], "n": len(s)}


def choose_alpha(false_outcomes: dict[float, Sequence[str]], genuine_outcomes: dict[float, Sequence[str]]) -> dict:
    """B4: α maximising (gold rate on false abstentions) − (unsupported rate on genuine abstentions) on the gate-tune
    H1 rows; ties to the smaller α."""
    def rate(o, cls):
        return float(np.mean([x in cls for x in o])) if len(o) else 0.0
    obj = {a: rate(false_outcomes[a], ("gold",)) - rate(genuine_outcomes[a], UNSUPPORTED) for a in false_outcomes}
    best = max(sorted(obj), key=lambda a: (obj[a], -a))
    return {"alpha": best, "objective_by_alpha": {str(a): v for a, v in obj.items()}}


def choose_alpha_lf(wrong_outcomes: dict[float, Sequence[str]], gold_outcomes: dict[float, Sequence[str]]) -> dict:
    """B4: α_LF maximising (gold rate on gate-tune F1-wrong) − (1 − kept rate on gate-tune F1-gold); ties smaller."""
    def g(o):
        return float(np.mean([x == "gold" for x in o])) if len(o) else 0.0
    obj = {a: g(wrong_outcomes[a]) - (1.0 - g(gold_outcomes[a]) if len(gold_outcomes[a]) else 0.0)
           for a in wrong_outcomes}
    best = max(sorted(obj), key=lambda a: (obj[a], -a))
    return {"alpha": best, "objective_by_alpha": {str(a): v for a, v in obj.items()}}


# ---------------------------------------------------------------------------------------------- bootstrap

def cluster_boot(keys: Sequence[str], x: Sequence[float], y: Sequence[float] | None = None, n: int = BOOT_N,
                 seed: int = BOOT_SEED) -> dict:
    """Base-clustered bootstrap of mean(x) (or of the paired mean(x) − mean(y), x and y per trial on the same trials):
    resample bases with replacement, the statistic is the trial-weighted mean over the resampled bases. 95%
    percentile interval."""
    keys = list(keys)
    x = np.asarray(x, dtype=float)
    d = x if y is None else x - np.asarray(y, dtype=float)
    if len(d) == 0:
        return {"n": 0, "n_bases": 0, "est": None, "lo": None, "hi": None}
    uk = sorted(set(keys))
    idx = {k: i for i, k in enumerate(uk)}
    g = np.array([idx[k] for k in keys])
    sums = np.bincount(g, weights=d, minlength=len(uk))
    cnts = np.bincount(g, minlength=len(uk)).astype(float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(uk), size=(n, len(uk)))
    stat = sums[draws].sum(1) / cnts[draws].sum(1)
    lo, hi = np.percentile(stat, [2.5, 97.5])
    return {"n": int(len(d)), "n_bases": len(uk), "est": float(d.mean()), "lo": float(lo), "hi": float(hi)}


def boot_diff_unpaired(keys_a, a, keys_b, b, n: int = BOOT_N, seed: int = BOOT_SEED) -> dict:
    """Difference of two means on different populations, each resampled by its own bases (Q-6)."""
    ra = _boot_stats(keys_a, a, n, seed)
    rb = _boot_stats(keys_b, b, n, seed + 1)
    if ra is None or rb is None:
        return {"est": None, "lo": None, "hi": None}
    d = ra - rb
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"est": float(np.mean(a) - np.mean(b)), "lo": float(lo), "hi": float(hi)}


def _boot_stats(keys, x, n, seed):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return None
    uk = sorted(set(keys))
    idx = {k: i for i, k in enumerate(uk)}
    g = np.array([idx[k] for k in keys])
    sums = np.bincount(g, weights=x, minlength=len(uk))
    cnts = np.bincount(g, minlength=len(uk)).astype(float)
    draws = np.random.default_rng(seed).integers(0, len(uk), size=(n, len(uk)))
    return sums[draws].sum(1) / cnts[draws].sum(1)


# ---------------------------------------------------------------------------------------------- design guard

def design_sha(design: dict) -> str:
    return hashlib.sha256(json.dumps(design, sort_keys=True).encode()).hexdigest()


def require_design(root: Path) -> dict:
    """Held stages refuse unless design.json exists and design_committed.json names a commit and its sha."""
    d, m = Path(root) / "design.json", Path(root) / DESIGN_MARKER
    if not d.exists() or not m.exists():
        raise PermissionError(f"{m} is missing: commit the dev design before any held stage")
    design, marker = json.loads(d.read_text()), json.loads(m.read_text())
    if not marker.get("commit"):
        raise PermissionError(f"{m} names no commit")
    if marker.get("design_sha256") != design_sha(design):
        raise PermissionError(f"{m}: design_sha256 does not match design.json")
    return design
