"""§47 the §44 bridge: span interchange on natural rows, readouts, controls and the B0-B4 reads (pure functions).

Spec: research/proposals/2026-10-31-exp47-bridge.md rev 0 (be2b9e5) + rev 1 (a9b70ff). The passes are in
scripts/exp47_patch.py.

- S1 = exactly the differing token positions of the clean and swap chats (all occurrences, never the range between).
- Readout (i): Δ = logit(g_id) - logit(s_id) at P1 of the §44 forced sequence (chat + forced `Answer:` + whitespace +
  shared prefix); T(l) = (Δ_patched(l) - Δ_clean) / (Δ_swap - Δ_clean); "transferred at l" = T(l) >= 0.5.
- Readout (ii): greedy generation from the patched chat, stopping at the end of the Answer line (max 160): the Answer
  line if present, else the whole generation; refusal (the §23 rule) -> "unknown"; D1 against G / S -> "G" / "S";
  both, neither or degenerate -> "other".
- k1: per site position, h_clean + r with ||r|| = ||h_swap - h_clean|| (seeded by row, layer, position).
- k3: the input embeddings at the sites overwritten by the swap tokens' (the swap prompt itself): must reproduce the
  swap run's forced top-1 and generated outcome on >= 0.95 of entered rows.
- L*_class (§46 §4 verbatim): the largest layer such that every l <= L* has Transfer >= 0.70, Wilson lower bound
  >= 0.55 and k1 (share with T >= 0.5) <= 0.10. Fewer than 8 entered rows: descriptive (no L*); 8-11: thin.
"""

from __future__ import annotations

import hashlib
import math
from typing import Sequence

import numpy as np

METHOD_VERSION = "exp47_bridge_v0"
SEED = 47
N_LAYERS = 36
T_THRESHOLD = 0.5
TRANSFER_MIN, LOWER_MIN, K1_MAX = 0.70, 0.55, 0.10
MID_STACK_L = 10
K3_MIN = 0.95
MIN_ROWS, THIN_ROWS = 8, 12                 # < 8 descriptive; 8-11 thin
DELTA_TOL = 0.05                            # |Δ - Δ_§44| on the unpatched clean and swap rows
SANITY_TOL = 0.05
SANITY_MAX_FAIL = 0.05                      # exit above 5 % of rows failing a sanity
MAX_NEW = 160
S2_LAYERS = tuple(range(0, 36, 3))
OUTCOMES = ("G", "S", "unknown", "other")


def wilson(k: int, n: int, z: float = 1.959963984540054):
    if n <= 0:
        return None, None, None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


# ---------------------------------------------------------------------------------------------- sites and controls

def diff_positions(clean: Sequence[int], swap: Sequence[int]) -> list[int]:
    if len(clean) != len(swap):
        raise ValueError("clean and swap differ in length: not position-aligned")
    return [i for i, (a, b) in enumerate(zip(clean, swap)) if a != b]


def k1_seed(question_id: str, layer: int, position: int, site: str = "S1", seed: int = SEED) -> int:
    return int(hashlib.sha256(f"{seed}-{question_id}-{site}-{layer}-{position}".encode()).hexdigest()[:8], 16)


def k1_vectors(h_clean: np.ndarray, h_swap: np.ndarray, question_id: str, layer: int, positions: Sequence[int],
               site: str = "S1") -> np.ndarray:
    """[n_sites, d]: per site position, h_clean + a random direction of norm ||h_swap - h_clean||."""
    out = np.empty_like(h_clean, dtype=np.float32)
    for i, pos in enumerate(positions):
        rng = np.random.default_rng(k1_seed(question_id, layer, pos, site))
        r = rng.standard_normal(h_clean.shape[-1]).astype(np.float32)
        r *= np.linalg.norm(h_swap[i] - h_clean[i]) / max(np.linalg.norm(r), 1e-12)
        out[i] = h_clean[i] + r
    return out


# ---------------------------------------------------------------------------------------------- readouts

def delta(logits, g_id: int, s_id: int) -> float:
    x = np.asarray(logits, dtype=np.float64)
    return float(x[g_id] - x[s_id])


def transfer(d_patched: float, d_clean: float, d_swap: float) -> float | None:
    den = d_swap - d_clean
    return None if abs(den) < 1e-9 else (d_patched - d_clean) / den


def answer_phrase(text: str) -> str:
    """The Answer line if present (after the last marker, first line), else the whole generation's first line."""
    low = (text or "").lower()
    m = low.rfind("answer:")
    if m >= 0:
        return text[m + len("answer:"):].split("\n")[0].strip()
    return (text or "").strip().split("\n")[0].strip()


def generated_outcome(text: str, gold: str, substitute: str, token_ids=None) -> str:
    from memctl.analysis import answer_selection as sel
    from memctl.analysis import steering as st
    phrase = answer_phrase(text)
    cls = st.classify(phrase, token_ids)
    if cls == "refusal":
        return "unknown"
    if cls != "answer":
        return "other"
    g, s = sel.d1(phrase, gold) is True, sel.d1(phrase, substitute) is True
    if g and not s:
        return "G"
    if s and not g:
        return "S"
    return "other"


# ---------------------------------------------------------------------------------------------- statistics

def share_profile(flags: dict[int, list[bool]]) -> dict[int, dict]:
    out = {}
    for layer in sorted(flags):
        f = flags[layer]
        k = sum(bool(x) for x in f)
        p, lo, hi = wilson(k, len(f))
        out[layer] = {"n": len(f), "k": k, "share": p, "lower": lo, "upper": hi}
    return out


def l_star(transfer_prof: dict[int, dict], k1_prof: dict[int, dict]) -> int | None:
    best = None
    for layer in sorted(transfer_prof):
        t, k = transfer_prof[layer], k1_prof.get(layer)
        ok = (t["share"] is not None and t["share"] >= TRANSFER_MIN and t["lower"] >= LOWER_MIN
              and k is not None and k["share"] is not None and k["share"] <= K1_MAX)
        if not ok:
            break
        best = layer
    return best


def class_status(n: int) -> str:
    return "descriptive" if n < MIN_ROWS else ("thin" if n < THIN_ROWS else "read")


def row_l_star(t: Sequence[float | None]) -> int | None:
    """Per row: the last layer of the initial run with T >= 0.5 (None if layer 0 already fails)."""
    best = None
    for layer, v in enumerate(t):
        if v is None or v < T_THRESHOLD:
            break
        best = layer
    return best


def b1_reading(l_success: int | None, status: str) -> str:
    if status == "descriptive":
        return "descriptive (fewer than 8 entered success rows)"
    if l_success is not None and l_success >= MID_STACK_L:
        return "yes: mid-stack carriage holds on natural rows (L* >= 10)"
    return "no: the controlled-task result does not transfer (L* < 10); B2 has no reference depth; the bridge closes"


def b2_reading(l_success: int | None, l_failure: int | None, status_fail: str, b1_yes: bool) -> str:
    if status_fail == "descriptive":
        return "descriptive (fewer than 8 entered failure rows)"
    if not b1_yes:
        return "no reference depth (B1 not met); reported only"
    if l_failure is not None and l_failure >= MID_STACK_L:
        return ("the binding is carried and used on failures as on successes: the failure is downstream of evidence "
                "use (answer production), on these rows")
    return "the binding is not carried on failures while it is on successes: evidence-use failure, on these rows"
