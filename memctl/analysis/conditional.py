"""§41: conditional steering. Does a signal at ':' of the unsteered pass say when steering will help?

Pre-registration: research/proposals/2026-10-23-exp41-conditional-steering.md (rev 0 and rev 1, 877f86d). Pure logic
(numpy, fractions); the dev run is scripts/exp41_dev.py, the fit and the frozen rule scripts/exp41_report.py.

- Signals (all from the UNSTEERED pass at ':', step 0; no generation needed beyond it):
  s_margin = max logit over the refusal-start ids - max logit over every other id (rev 1 iii);
  s_proj   = <h_29(':'), v_29 / ||v_29||> (the layer-29 decoder output, pre final norm, as §38 captured it);
  s_evid   = s_margin(intact) - s_margin(cut).
  The refusal-start ids are the first generated ids of every refusal-class row of the §38 dev A and B rows
  (`refusal_start_ids`), computed from the file, printed and frozen into the rule.
- Labels (rev 1 ii, option a): a text byte-equal to the cached §35 A final answer of its row takes the cached A judge
  verdict, else byte-equal to the cached B final answer takes the B verdict (A is checked first), else the D1 string
  rule (PROVISIONAL). The label source is recorded per row. Option (b), the 7B judge for the new texts, is a hook
  (`label(..., judge=)`, `judge_timing_gate`): five timed calls must project the whole set under 45 min.
- Utility per row (rev 1 iv): class 'refusal' or 'degenerate' -> 0; an answer-class phrase matching
  SOFT_REFUSAL_PATTERN (case-insensitive re.search) -> 0, whatever its label; else +1 if correct, -lambda if wrong.
- Weights (rev 1 vi): R rows weight 1; C rows weight w_C = (n_R / 3) / n_C so that the C total is a third of the R
  total (C : R = 1 : 3, §40's caps 40 : 120). On the dev rows (28, 39): w_C = (28/3)/39 = 28/117. Exact fractions.
- Rule: fire iff signal > tau (sign '+') or signal < tau (sign '-'); Cond = the Steer output where the rule fires,
  else the A output. Fit over 3 signals x 2 signs x tau in the observed values of that signal plus -inf and +inf.
  Order (`fit`): (1) highest weighted utility at lambda 1; (2) highest unweighted utility at lambda 1; (3) signal
  order s_margin, s_proj, s_evid; (4) sign '+' before '-'; (5) among the taus still tied for that (signal, sign), the
  lower median of the sorted tied taus (a threshold inside the plateau rather than on its edge).
- Gate (rev 0 §4 / rev 1 v): weighted U(Cond) >= weighted U(Steer) + 3 at lambda 1, the 3 in weighted units (one unit
  = one R row); the unweighted difference is printed beside.
"""
from __future__ import annotations

import math
import re
import statistics
from fractions import Fraction

import numpy as np

from memctl.analysis import steering as st

SIGNALS = ("s_margin", "s_proj", "s_evid")
SIGNS = ("+", "-")
LAMBDA = 1
LAMBDA_BESIDE = 2
GATE_UNITS = 3
CR_RATIO = Fraction(1, 3)              # C total weight : R total weight
JUDGE_LIMIT_MIN = 45.0
JUDGE_TIMING_N = 5
LABEL_SOURCES = ("cache:A", "cache:B", "judge7b", "D1")
TIE_BREAK = ("weighted U(lambda 1) desc; unweighted U(lambda 1) desc; signal order s_margin, s_proj, s_evid; "
             "sign + before -; lower median of the remaining tied taus")
GATE_RULE = (f"weighted U(Cond) - weighted U(Steer) >= {GATE_UNITS} at lambda {LAMBDA}; units are weighted "
             "(an R row = 1, a C row = w_C)")
D1_BIAS = ("D1 (the gold string or every gold token in the phrase) marks about half the correct paraphrases wrong "
           "(rev 1 ii): new texts are mostly Steer outputs, so the bias tilts the fit toward steering being worse "
           "than it is (wrong answers cost -lambda); it is stated, not corrected")


# --- signals --------------------------------------------------------------------------------------------------

def refusal_start_ids(rows) -> list[int]:
    """First generated ids of every refusal-class row among the §38 dev A and B rows (any pop, setting 'none')."""
    ids = sorted({int(r["gen_ids"][0]) for r in rows
                  if r["kind"] in ("A", "B") and r["setting"]["key"] == "none" and r["cls"] == "refusal"
                  and r["gen_ids"]})
    if not ids:
        raise ValueError("no refusal-class row with a generated id: the refusal-start set would be empty")
    return ids


def margin(logits, ids) -> float:
    """max over `ids` minus max over every other id (float64)."""
    x = np.asarray(logits, dtype=np.float64)
    mask = np.zeros(x.shape[0], bool)
    mask[list(ids)] = True
    if mask.all() or not mask.any():
        raise ValueError("the refusal-start set must be a proper, non-empty subset of the vocabulary")
    return float(x[mask].max() - x[~mask].max())


def step0_summary(logits, ids, top: int = 20) -> dict:
    """What a row keeps of the step-0 logits: the margin and its two terms, the argmax, the top ids."""
    x = np.asarray(logits, dtype=np.float64)
    mask = np.zeros(x.shape[0], bool)
    mask[list(ids)] = True
    other = np.where(~mask, x, -np.inf)
    order = np.argsort(-x, kind="stable")[:top]
    return {"refusal_ids": [int(i) for i in ids], "refusal_max": float(x[mask].max()),
            "other_max": float(other.max()), "other_argmax": int(other.argmax()), "argmax": int(x.argmax()),
            "margin": margin(x, ids), "top": [[int(i), float(x[i])] for i in order], "vocab": int(x.shape[0])}


def projection(h, v) -> float:
    """<h, v / ||v||> in float64."""
    h64, v64 = np.asarray(h, np.float64), np.asarray(v, np.float64)
    return float(h64 @ (v64 / np.linalg.norm(v64)))


def signals_of(margin_intact: float, margin_cut: float, h29, v29) -> dict:
    return {"s_margin": float(margin_intact), "s_proj": projection(h29, v29),
            "s_evid": float(margin_intact) - float(margin_cut)}


# --- labels and utility ---------------------------------------------------------------------------------------

def soft_refusal(phrase: str, cls: str) -> bool:
    return cls == "answer" and re.search(st.SOFT_REFUSAL_PATTERN, phrase or "", re.IGNORECASE) is not None


def label(final_answer: str, phrase: str, cls: str, gold: str, cached_a: str | None, verdict_a: bool | None,
          cached_b: str | None, verdict_b: bool | None, judge=None) -> tuple[bool, str]:
    """(correct, source). Byte-equal to the cached A final answer with an A verdict -> that verdict; else to the
    cached B one -> the B verdict; else `judge(final_answer)` when given (option b); else D1 (provisional)."""
    if cached_a is not None and verdict_a is not None and final_answer == cached_a:
        return bool(verdict_a), "cache:A"
    if cached_b is not None and verdict_b is not None and final_answer == cached_b:
        return bool(verdict_b), "cache:B"
    if judge is not None:
        return bool(judge(final_answer)), "judge7b"
    return st.provisional_gold(phrase, gold, cls), "D1"


def outcome(cls: str, phrase: str, correct: bool) -> str:
    """'correct', 'wrong', 'soft' (answer-class hedge), 'refusal' or 'degenerate'."""
    if cls in ("refusal", "degenerate"):
        return cls
    if soft_refusal(phrase, cls):
        return "soft"
    return "correct" if correct else "wrong"


def utility(out: str, lam=LAMBDA):
    return 1 if out == "correct" else -lam if out == "wrong" else 0


def weights(n_r: int, n_c: int, ratio: Fraction = CR_RATIO) -> dict:
    """R weight 1; C weight (ratio * n_R) / n_C, exact."""
    if n_c == 0:
        return {"R": Fraction(1), "C": Fraction(0)}
    return {"R": Fraction(1), "C": Fraction(ratio) * n_r / n_c}


def total(outcomes: dict, pops: dict, w: dict | None = None, lam=LAMBDA):
    """sum over rows of weight(pop) * utility; `outcomes[q]` an outcome string, `pops[q]` 'R' or 'C'."""
    return sum((w[pops[q]] if w else 1) * utility(o, lam) for q, o in outcomes.items())


# --- rules ----------------------------------------------------------------------------------------------------

def fires(value: float, sign: str, tau: float) -> bool:
    if sign == "+":
        return value > tau
    if sign == "-":
        return value < tau
    raise ValueError(f"sign must be '+' or '-', not {sign!r}")


def tau_value(tau) -> float:
    """A tau as stored in JSON ('inf', '-inf' or a number) -> float."""
    return float(tau)


def apply_rule(rule: dict, signals: dict) -> bool:
    """§40 calls this: True iff the frozen rule fires (steer) on a row's signals. Pure; `signals` holds at least the
    rule's signal (a finite number), `rule` = {'signal', 'sign', 'tau'} as rule.json stores it."""
    name = rule["signal"]
    if name not in SIGNALS:
        raise ValueError(f"unknown signal {name!r}")
    value = signals[name]
    if value is None or not math.isfinite(float(value)):
        raise ValueError(f"signal {name} is {value!r}; the rule needs a finite value")
    return fires(float(value), rule["sign"], tau_value(rule["tau"]))


def cond_outcomes(sig: dict, sign: str, tau: float, a_out: dict, steer_out: dict) -> dict:
    return {q: steer_out[q] if fires(sig[q], sign, tau) else a_out[q] for q in a_out}


def tau_grid(values) -> list[float]:
    return [-math.inf] + sorted(set(float(v) for v in values)) + [math.inf]


def lower_median(xs: list[float]) -> float:
    s = sorted(xs)
    return s[(len(s) - 1) // 2]


def fit(signals: dict, a_out: dict, steer_out: dict, pops: dict, w: dict, lam=LAMBDA) -> dict:
    """`signals[name][q]`. Every (signal, sign, tau) scored; the TIE_BREAK order chooses one."""
    cands = []
    for si, name in enumerate(SIGNALS):
        sig = signals[name]
        for gi, sign in enumerate(SIGNS):
            for tau in tau_grid(sig.values()):
                c = cond_outcomes(sig, sign, tau, a_out, steer_out)
                cands.append({"signal": name, "sign": sign, "tau": tau, "si": si, "gi": gi,
                              "weighted": total(c, pops, w, lam), "unweighted": total(c, pops, None, lam),
                              "fires": sum(fires(sig[q], sign, tau) for q in sig)})
    best_w = max(c["weighted"] for c in cands)
    tied = [c for c in cands if c["weighted"] == best_w]
    best_u = max(c["unweighted"] for c in tied)
    tied = [c for c in tied if c["unweighted"] == best_u]
    si = min(c["si"] for c in tied)
    tied = [c for c in tied if c["si"] == si]
    gi = min(c["gi"] for c in tied)
    tied = [c for c in tied if c["gi"] == gi]
    tau = lower_median([c["tau"] for c in tied])
    chosen = next(c for c in tied if c["tau"] == tau)
    return {"chosen": {k: chosen[k] for k in ("signal", "sign", "tau", "weighted", "unweighted", "fires")},
            "n_tied_at_top": sum(1 for c in cands if c["weighted"] == best_w),
            "n_tied_final": len(tied), "tied_taus": [c["tau"] for c in tied],
            "candidates": [{k: c[k] for k in ("signal", "sign", "tau", "weighted", "unweighted", "fires")}
                           for c in cands]}


def best_per_signal(fitted: dict) -> dict:
    """The best (sign, tau) per signal under the same order (descriptive)."""
    out = {}
    for name in SIGNALS:
        cs = [c for c in fitted["candidates"] if c["signal"] == name]
        top = max((c["weighted"], c["unweighted"]) for c in cs)
        cs = [c for c in cs if (c["weighted"], c["unweighted"]) == top]
        sign = "+" if any(c["sign"] == "+" for c in cs) else "-"
        cs = [c for c in cs if c["sign"] == sign]
        tau = lower_median([c["tau"] for c in cs])
        out[name] = next(c for c in cs if c["tau"] == tau)
    return out


def gate(u_cond, u_steer, units=GATE_UNITS) -> dict:
    diff = u_cond - u_steer
    return {"u_cond": u_cond, "u_steer": u_steer, "diff": diff, "need": units, "pass": diff >= units}


# --- descriptive readouts -------------------------------------------------------------------------------------

def coverage_risk(outcomes: dict) -> dict:
    """coverage = answered (correct or wrong) / rows; risk = wrong / answered."""
    n = len(outcomes)
    k_c = sum(o == "correct" for o in outcomes.values())
    k_w = sum(o == "wrong" for o in outcomes.values())
    ans = k_c + k_w
    return {"n": n, "correct": k_c, "wrong": k_w, "answered": ans, "coverage": ans / n if n else None,
            "risk": k_w / ans if ans else None}


def risk_coverage_curve(sig: dict, sign: str, a_out: dict, steer_out: dict, pops: dict, w: dict) -> list[dict]:
    out = []
    for tau in tau_grid(sig.values()):
        c = cond_outcomes(sig, sign, tau, a_out, steer_out)
        out.append({"tau": tau, "fires": sum(fires(sig[q], sign, tau) for q in sig), **coverage_risk(c),
                    "weighted_l1": total(c, pops, w, 1), "weighted_l2": total(c, pops, w, 2),
                    "unweighted_l1": total(c, pops, None, 1)})
    return out


def auroc(scores: list[float], labels: list[bool]) -> float | None:
    """P(score of a positive > score of a negative), ties 1/2 (Mann-Whitney); None without both classes."""
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return None
    k = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return k / (len(pos) * len(neg))


# --- option (b): the 7B judge on the laptop (a hook; this build never loads the 7B) ----------------------------

def judge_timing_gate(call_seconds: list[float], n_calls: int, limit_min: float = JUDGE_LIMIT_MIN) -> dict:
    """Five timed judge calls -> projection for `n_calls`; option (b) runs only if it projects under `limit_min`."""
    if len(call_seconds) < JUDGE_TIMING_N:
        raise ValueError(f"the gate needs {JUDGE_TIMING_N} timed calls, got {len(call_seconds)}")
    per = statistics.mean(call_seconds)
    proj = per * n_calls / 60.0
    return {"timed_calls": len(call_seconds), "per_call_s": per, "n_calls": n_calls, "projected_min": proj,
            "limit_min": limit_min, "pass": proj < limit_min}


def fmt(x) -> str:
    """A utility or weight for print: an exact fraction and its decimal."""
    if isinstance(x, Fraction):
        return f"{float(x):.3f}" if x.denominator == 1 else f"{x} = {float(x):.3f}"
    return f"{x:.3f}" if isinstance(x, float) else str(x)
