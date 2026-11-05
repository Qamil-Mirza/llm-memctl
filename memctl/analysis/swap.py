"""§44 rev 1 amendment §B-§C: the swap conditions (clean / swap / cut at the forced P1), evidence following, the per-row
failure class, bidirectional swap patching with three negative controls, and condition U (no "unknown" option).

Pre-registration: research/proposals/2026-10-29-exp44-patching.md, "Rev 1 amendment" (D-45) §B-§C, rev 2 rulings
(5146f5b). The swap subset and each row's substitute come from scripts/exp44_swap_gate.py (runs/exp44_swapgate/).

Per swap row, three forced-P1 runs (§34 `readout_for`: chat text + "Answer:" + the gold's teacher-forced whitespace
tokens; P1 = the last token; the substitute has the same forced whitespace, the gate checked it):
  clean = the intact prompt (gold G in its one evidence line); swap = the substitute S in its place, every other byte
  identical; cut = §34c / rev 0 (evidence lines removed). Per run, at the final layer: Delta = logit(G1) - logit(S1)
  (G1 / S1 = the first content token of G / S, `tracked_token_ids`), log p(G1), log p(S1), their ranks, log p(unknown)
  (log-sum-exp over the family's refusal ids, each id also kept) and the top-1 id.
Evidence following: F = Delta_clean - Delta_swap; follows(m) iff F >= m and Delta_clean > 0 > Delta_swap; m = 1 nat
  (margins 0.5 / 1 / 2 reported). Class (m): answer_production if follows(m); evidence_use if |F| < m; else
  indeterminate. Computed on every row; read on the in-view failures (§23 refusal or wrong), successes = reference.
Patching (rows that follow at 1 nat only): forward = the clean run with its P1 residual at layer l's output overwritten
  by the swap run's, T_fwd(l) = (Delta_p - Delta_clean) / (Delta_swap - Delta_clean); reverse = the swap run with the
  clean residual, T_rev(l) = (Delta_p - Delta_swap) / (Delta_clean - Delta_swap). d_T = (first l with T >= 0.5) / L.
  Sanity, asserted on every patched row: |T(L-1) - 1| < 0.01 (construction), |T(0)| < 0.05 (empirical bound).
Controls (every patched row, forward direction, into the clean run):
  mismatched donor (rev 6, difference patch): the clean P1 residual at layer l set to
    h_clean(l) + (h_swap_donor(l) - h_clean_donor(l)), the donor row's (`choose_donor`) own swap-minus-clean
    difference at the scale of the real patch; reported as the Delta shift toward S, shift(l) = Delta_clean -
    Delta_p(l) (positive = toward S), expected ~ 0;
  self-overwrite: the clean residual into the clean run at layers 0, L//2, L-1: T = 0 (asserted, |shift| < 1e-4);
  random (rev 6, difference-norm perturbation): h_clean(l) + eps, eps the seeded Gaussian direction [44, row, layer]
    scaled to ||h_swap(l) - h_clean(l)||, added to (not replacing) the clean residual.
  Rev 6 (31553f6) drops the rev 1 full-state overwrites (donor h_swap_donor(l); random of norm ||h_swap(l)||): they
  measured generic disruption at 10-100x the patch scale. The pass count per row is unchanged.
Condition U: the §35 frozen-note arm B input (cached note + newline + B_INSTRUCTION ("... Do not reply unknown.") +
  newline + "Answer:"), greedy, 64 new tokens, on in-view refusal rows; gold = D1 on the phrase (provisional, as §38's
  laptop gold). Also under the §38 Steer setting (add alpha * v_L at every step: Qwen 3B L29 alpha 2, Llama 3B L17
  alpha 2, vnorm) and, for S2, the §38 push itself on arm A (note + "Answer:") with A unsteered beside.
Rev 3 (7acce03, uniform): the shared prefix = the longest common token prefix of G's and S's content tokens (after
  the leading whitespace tokens). It is teacher-forced after P1's forced whitespace in every run of the row (clean,
  swap, cut, patched, donor); G1 / S1 above are then the first divergent tokens, and Delta, F, T, ranks, log p(unknown)
  and top-1 are read, and the residual patched, at the position that predicts them (the last token). An empty prefix
  is exactly the rev 2 behaviour. U is unaffected.
torch is imported lazily (inside the engine calls).
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from dataclasses import dataclass
from statistics import median

import numpy as np

from memctl.analysis import patching as pt
from memctl.analysis.jacobian import ablate_evidence
from memctl.analysis.jlens_readout import ranks_of, readout_for, tracked_token_ids

# --- pre-registered numbers (rev 1 amendment §B-§C) ----------------------------------------------------------------
FOLLOW_MARGIN = 1.0
MARGINS = (0.5, 1.0, 2.0)
T_THRESHOLD = 0.5
T_FINAL_TOL = 0.01
T_ZERO_TOL = 0.05
SELF_TOL = 1e-4
MIN_GAP = 1e-6
RANDOM_SEED = 44
CLASSES = ("answer_production", "evidence_use", "indeterminate")
FAILURE_LABELS = ("refusal", "wrong")
S0_RATE, S0_LB = 0.80, 0.70
S1_IND_MAX, S1_MARGIN_SHIFT, S1_HALF_WIDTH = 0.20, 0.15, 0.15
S1_AP_MAJORITY, S1_EU_MAJORITY = 0.50, 0.25
S2_MIN_PER_CLASS = 8
S3_DONOR_BOUND = 0.5
S3_SPECIFICITY_RATIO = 0.25          # rev 6 ruling 3, the pre-declared fallback
U1_HIGH, U1_HIGH_LB, U1_LOW, U1_LOW_UB = 0.50, 0.40, 0.25, 0.35
BOOT_SAMPLES = pt.BOOT_SAMPLES
U_MAX_NEW = 64
# the §38 Steer setting per family: (layer, alpha, unit, directions file)
STEER = {"qwen3b": (29, 2.0, "vnorm", "runs/exp38_dev/qwen3b/directions.npz"),
         "llama3b": (17, 2.0, "vnorm", "runs/exp43_llama_dev/llama3b/directions.npz")}
U_KINDS = ("U", "U_steer", "A", "A_steer")          # A / A_steer only on swap-subset refusal rows (S2)

METHOD = {"spec": "research/proposals/2026-10-29-exp44-patching.md rev 1 amendment §B-§C (D-45), rev 2 (5146f5b)",
          "runs": "clean / swap / cut, §34 forced readout_for, prefix cache to P1-1, P1 run with capture on every layer",
          "delta": "logit(G first content token) - logit(S first content token), final logits, float64",
          "prefix": "rev 3: the shared G/S token prefix teacher-forced after the whitespace; read and patch at the "
                    "position predicting the first divergent token",
          "follows": "F = Delta_clean - Delta_swap >= 1 and Delta_clean > 0 > Delta_swap",
          "patch": "steering.overwrite_spec(l, h_donor(l)) at P1, one layer per pass, forward and reverse",
          "controls": "rev 6 (31553f6): mismatched donor (previous cyclic swap row with other first ids) as a "
                      "difference patch h_clean(l) + (h_swap_donor(l) - h_clean_donor(l)); self at 0/L//2/L-1; "
                      "random = h_clean(l) + Gaussian direction [44, row, layer] of norm ||h_swap(l) - h_clean(l)||",
          "controls_rev": 6,
          "unknown": "log-sum-exp of log p over the family's refusal ids", "residual": "decoder layer output"}


GATE_REV8 = {"gate_rev": 8, "gate": "rev 8 (83fe60a, D-47): runs/exp44_swapgate/rev8, no 'other', bare numbers "
                                  "form-matched, leading determiner matched; passes and controls unchanged"}


def method_sha256(gate_rev: int = 3) -> str:
    """The S / U row method stamp. gate_rev 8 (§44 rev 8) adds GATE_REV8 so its rows never share a stamp with the
    rev 3 / rev 6 rows; any other value is the unchanged rev 6 stamp."""
    m = dict(METHOD, **GATE_REV8) if gate_rev >= 8 else METHOD
    return hashlib.sha256(json.dumps(m, sort_keys=True).encode()).hexdigest()


class SanityError(RuntimeError):
    """T(L-1) != 1, |T(0)| >= 0.05 or a self-overwrite moved Delta, on a patched row."""


# --- arithmetic -------------------------------------------------------------------------------------------------

def readout(logits, g: int, s: int, refusal_ids) -> dict:
    """The per-run measures from one P1 logit vector."""
    x = np.asarray(logits, dtype=np.float64)
    lp = pt.log_softmax(x)
    rk = ranks_of(np.asarray(logits, dtype=np.float32)[None, :], [g, s])[0]
    lp_ref = [float(lp[i]) for i in refusal_ids]
    return {"delta": float(x[g] - x[s]), "logit_g": float(x[g]), "logit_s": float(x[s]), "lp_g": float(lp[g]),
            "lp_s": float(lp[s]), "rank_g": int(rk[0]), "rank_s": int(rk[1]),
            "lp_unknown": float(np.logaddexp.reduce(lp_ref)) if lp_ref else None, "lp_refusal_ids": lp_ref,
            "top1": int(np.argmax(x))}


def follows(d_clean: float, d_swap: float, margin: float = FOLLOW_MARGIN) -> bool:
    return (d_clean - d_swap) >= margin and d_clean > 0 > d_swap


def failure_class(d_clean: float, d_swap: float, margin: float = FOLLOW_MARGIN) -> str:
    """answer_production if the row follows the evidence; evidence_use if |F| < margin; else indeterminate."""
    if follows(d_clean, d_swap, margin):
        return "answer_production"
    if abs(d_clean - d_swap) < margin:
        return "evidence_use"
    return "indeterminate"


def transfer(d_patched, d_from: float, d_to: float) -> list[float] | None:
    """T(l) = (Delta_p(l) - d_from) / (d_to - d_from): forward from = clean, to = swap; reverse from = swap, to = clean.
    None when the two runs' Delta coincide."""
    gap = d_to - d_from
    if abs(gap) < MIN_GAP:
        return None
    return [(float(d) - d_from) / gap for d in d_patched]


def transfer_depth(t: list[float] | None, n_layers: int, threshold: float = T_THRESHOLD) -> float | None:
    return pt.causal_depth(t, n_layers, threshold)


def transfer_sanity(t: list[float] | None) -> dict:
    if t is None:
        return {"t_final": None, "t_zero": None, "final_ok": None, "zero_ok": None, "pass": None}
    final_ok, zero_ok = abs(t[-1] - 1.0) < T_FINAL_TOL, abs(t[0]) < T_ZERO_TOL
    return {"t_final": t[-1], "t_zero": t[0], "final_ok": final_ok, "zero_ok": zero_ok, "pass": final_ok and zero_ok}


def shift_toward_s(d_patched, d_clean: float) -> list[float]:
    """Delta shift toward S of a control patched into the clean run (positive = Delta fell, i.e. toward S)."""
    return [d_clean - float(d) for d in d_patched]


def random_overwrite(norm: float, hidden: int, row_index: int, layer: int, seed: int = RANDOM_SEED) -> np.ndarray:
    """A seeded Gaussian direction per (row, layer) scaled to `norm`."""
    g = np.random.default_rng([seed, row_index, layer]).standard_normal(hidden)
    return (g / np.linalg.norm(g) * norm).astype(np.float32)


def difference_patch(h_clean: np.ndarray, donor_clean: np.ndarray, donor_swap: np.ndarray) -> np.ndarray:
    """Rev 6 mismatched donor at one layer (or all, row-wise): h_clean + (h_swap_donor - h_clean_donor)."""
    return (np.asarray(h_clean, np.float64) + (np.asarray(donor_swap, np.float64)
                                               - np.asarray(donor_clean, np.float64))).astype(np.float32)


def difference_perturbation(h_clean: np.ndarray, h_swap: np.ndarray, row_index: int, layer: int,
                            seed: int = RANDOM_SEED) -> tuple[np.ndarray, float]:
    """Rev 6 random control at one layer: (h_clean + eps, ||eps||), eps the seeded [44, row, layer] direction scaled to
    the row's own difference norm ||h_swap - h_clean|| at that layer."""
    norm = float(np.linalg.norm(np.asarray(h_swap, np.float64) - np.asarray(h_clean, np.float64)))
    eps = random_overwrite(norm, len(h_clean), row_index, layer, seed)
    return (np.asarray(h_clean, np.float32) + eps).astype(np.float32), norm


def choose_donor(order: list[dict], i: int) -> int | None:
    """The mismatched donor of row i of `order` (rows with gold_first_id / substitute_first_id, (fold, id) order): the
    nearest earlier row, cyclically, whose gold and substitute first ids are both outside row i's {G1, S1}."""
    def ids(e):        # rev 3: the read (first divergent) ids; equal to the first ids for an empty prefix
        return e.get("gold_read_id", e["gold_first_id"]), e.get("substitute_read_id", e["substitute_first_id"])
    own = set(ids(order[i]))
    n = len(order)
    for k in range(1, n):
        j = (i - k) % n
        if not set(ids(order[j])) & own:
            return j
    return None


def self_layers(n_layers: int) -> tuple[int, ...]:
    return tuple(sorted({0, n_layers // 2, n_layers - 1}))


# --- inputs -----------------------------------------------------------------------------------------------------

def content_ids(tokenizer, text: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """(leading whitespace-only tokens, content tokens) of ids(" " + text), the §34 split."""
    from memctl.analysis.jacobian import _ids, _margin_length
    answer = _ids(tokenizer, " " + text)
    k = _margin_length(tokenizer, answer)
    return tuple(answer[:k - 1]), tuple(answer[k - 1:])


def shared_prefix(g_content, s_content) -> tuple[tuple[int, ...], int | None, int | None]:
    """Rev 3: the longest common token prefix of the gold's and the substitute's content tokens, and the first
    divergent token of each (None when one sequence is a prefix of the other: no divergent token)."""
    n = 0
    while n < len(g_content) and n < len(s_content) and g_content[n] == s_content[n]:
        n += 1
    g = g_content[n] if n < len(g_content) else None
    s = s_content[n] if n < len(s_content) else None
    return tuple(g_content[:n]), g, s


@dataclass(frozen=True)
class SwapInputs:
    question_id: str
    fold: int
    clean_prompt: str
    swap_prompt: str
    cut_prompt: str
    clean_ids: tuple[int, ...]
    swap_ids: tuple[int, ...]
    cut_ids: tuple[int, ...]
    g_id: int
    s_id: int
    refusal_ids: tuple[int, ...]
    gold_token_text: str
    sub_token_text: str
    n_leading_ws: int
    prefix_ids: tuple[int, ...] = ()       # rev 3: the shared prefix, teacher-forced after P1's forced whitespace
    g_first_id: int | None = None          # the gold's first content token (the §34 tracked id)


def sha_ids(ids) -> str:
    return pt.sha256_ids(ids)


def build_swap_inputs(tokenizer, record, swap_prompt: str, old_line: str, new_line: str, substitute: str,
                      refusal_ids, chat_text_fn=None) -> SwapInputs:
    """Clean / swap / cut forced readouts, aligned by role (patching.check_alignment on clean vs swap and clean vs
    cut); the swap must keep the gold's tracked ids and forced whitespace, and S must share G's leading whitespace
    tokens. Rev 3: the shared prefix of G and S (content tokens) is appended to all three sequences after the forced
    whitespace; the read / patch position is then the last token (it predicts the first divergent token), g_id / s_id
    are the divergent tokens, and the alignment check covers the forced prefix (it ends every sequence, identical).
    No divergent token (one is a prefix of the other) is an AlignmentError. An empty prefix is the rev 2 behaviour."""
    clean = readout_for(tokenizer, record.prompt, record.memory_lines, record.evidence_lines, record.gold,
                        chat_text_fn=chat_text_fn)
    memory = tuple(new_line if l == old_line else l for l in record.memory_lines)
    evidence = tuple(new_line if l == old_line else l for l in record.evidence_lines)
    swap = readout_for(tokenizer, swap_prompt, memory, evidence, record.gold, chat_text_fn=chat_text_fn)
    if swap.tracked != clean.tracked or swap.n_leading_ws != clean.n_leading_ws:
        raise pt.AlignmentError(f"{record.question_id}: tracked ids or forced whitespace differ between clean and swap")
    cut_prompt = ablate_evidence(record.prompt, record.memory_lines, record.evidence_lines)
    keep = set(record.evidence_lines)
    cut = readout_for(tokenizer, cut_prompt, tuple(l for l in record.memory_lines if l not in keep), (), record.gold,
                      chat_text_fn=chat_text_fn)
    if cut.tracked != clean.tracked or cut.n_leading_ws != clean.n_leading_ws:
        raise pt.AlignmentError(f"{record.question_id}: tracked ids or forced whitespace differ between clean and cut")
    g_lead, g_content = content_ids(tokenizer, record.gold)
    s_lead, s_content = content_ids(tokenizer, substitute)
    if s_lead != g_lead:
        raise pt.AlignmentError(f"{record.question_id}: the substitute's forced whitespace differs from the gold's")
    prefix, g_div, s_div = shared_prefix(g_content, s_content)
    if g_div is None or s_div is None:
        raise pt.AlignmentError(f"{record.question_id}: no divergent token after the shared prefix "
                                f"({tokenizer.decode(list(prefix))!r})")
    if g_content[0] != clean.tracked["gold"]:
        raise pt.AlignmentError(f"{record.question_id}: the gold's first content token differs from the tracked id")
    ids = {k: tuple(r.input_ids) + prefix for k, r in (("clean", clean), ("swap", swap), ("cut", cut))}
    for k in ids:
        if ids[k][len(ids[k]) - len(prefix) - 1] != ids["clean"][len(ids["clean"]) - len(prefix) - 1]:
            raise pt.AlignmentError(f"{record.question_id}: the forced position before the prefix differs ({k})")
    for k, other, n_ev in (("swap", swap, 0), ("cut", cut, len(record.evidence_lines))):
        pt.check_alignment(pt.RowInputs(record.question_id, record.fold, record.prompt, "", ids["clean"], ids[k],
                                        len(ids["clean"]) - 1, len(ids[k]) - 1, clean.positions["Q"][0],
                                        other.positions["Q"][0], g_div, clean.tracked["unknown_sp"],
                                        clean.gold_token_text, clean.n_leading_ws, n_ev))
    if prefix and not all(v[-len(prefix):] == prefix for v in ids.values()):
        raise pt.AlignmentError(f"{record.question_id}: the forced prefix does not end every sequence")
    return SwapInputs(record.question_id, record.fold, record.prompt, swap_prompt, cut_prompt, ids["clean"],
                      ids["swap"], ids["cut"], int(g_div), int(s_div), tuple(refusal_ids),
                      tokenizer.decode([g_div]), tokenizer.decode([s_div]), clean.n_leading_ws, prefix,
                      int(clean.tracked["gold"]))


# --- the passes -------------------------------------------------------------------------------------------------

def _capture_run(engine, ids, g, s, refusal_ids):
    prep = pt._prepare(engine, ids)
    logits, captured = pt._p1_logits(engine, prep, [], capture=range(engine.n_layers))
    if sorted(captured) != list(range(engine.n_layers)):
        raise RuntimeError(f"captured layers {sorted(captured)} != 0..{engine.n_layers - 1}")
    resid = np.stack([captured[l] for l in range(engine.n_layers)]).astype(np.float32)
    return prep, readout(logits, g, s, refusal_ids), resid


def base_passes(engine, x: SwapInputs) -> tuple[dict, dict]:
    """(row, state): clean / swap / cut readouts, F and the classes; `state` keeps the two prepared caches and the
    (L, hidden) P1 residuals of clean and swap for the patch passes."""
    started = time.perf_counter()
    clean_prep, clean, h_clean = _capture_run(engine, x.clean_ids, x.g_id, x.s_id, x.refusal_ids)
    swap_prep, swap, h_swap = _capture_run(engine, x.swap_ids, x.g_id, x.s_id, x.refusal_ids)
    cut_prep = pt._prepare(engine, x.cut_ids)
    cut = readout(pt._p1_logits(engine, cut_prep, [])[0], x.g_id, x.s_id, x.refusal_ids)
    del cut_prep
    F = clean["delta"] - swap["delta"]
    row = {"question_id": x.question_id, "fold": x.fold, "n_layers": engine.n_layers, "g_id": x.g_id, "s_id": x.s_id,
           "prefix_ids": list(x.prefix_ids), "g_first_id": x.g_first_id,
           "gold_token_text": x.gold_token_text, "sub_token_text": x.sub_token_text, "n_leading_ws": x.n_leading_ws,
           "refusal_ids": list(x.refusal_ids), "n_tokens": {"clean": len(x.clean_ids), "swap": len(x.swap_ids),
                                                            "cut": len(x.cut_ids)},
           "clean": clean, "swap": swap, "cut": cut, "F": F,
           "follows": {f"{m:g}": follows(clean["delta"], swap["delta"], m) for m in MARGINS},
           "cls": {f"{m:g}": failure_class(clean["delta"], swap["delta"], m) for m in MARGINS},
           "resid_norm": {"clean": [float(np.linalg.norm(v)) for v in h_clean],
                          "swap": [float(np.linalg.norm(v)) for v in h_swap]},
           "patched": False, "timing": {"base_s": time.perf_counter() - started}}
    return row, {"clean_prep": clean_prep, "swap_prep": swap_prep, "h_clean": h_clean, "h_swap": h_swap}


def _sweep(engine, prep, vectors: dict[int, np.ndarray], x: SwapInputs) -> dict[int, dict]:
    from memctl.analysis.steering import overwrite_spec
    return {l: readout(pt._p1_logits(engine, prep, [overwrite_spec(l, v)])[0], x.g_id, x.s_id, x.refusal_ids)
            for l, v in vectors.items()}


def patch_passes(engine, x: SwapInputs, row: dict, state: dict,
                 donor_resid: tuple[np.ndarray, np.ndarray] | None, row_index: int) -> dict:
    """Forward and reverse sweeps over every layer, then the three controls; updates and returns `row`.
    `donor_resid` = (h_clean_donor, h_swap_donor), the donor row's (L, hidden) P1 residuals, or None (no donor)."""
    started = time.perf_counter()
    L = engine.n_layers
    h_clean, h_swap = state["h_clean"], state["h_swap"]
    dc, ds = row["clean"]["delta"], row["swap"]["delta"]
    fwd = _sweep(engine, state["clean_prep"], {l: h_swap[l] for l in range(L)}, x)
    rev = _sweep(engine, state["swap_prep"], {l: h_clean[l] for l in range(L)}, x)
    d_fwd = [fwd[l]["delta"] for l in range(L)]
    d_rev = [rev[l]["delta"] for l in range(L)]
    t_fwd, t_rev = transfer(d_fwd, dc, ds), transfer(d_rev, ds, dc)
    out = {"fwd": {k: [fwd[l][k] for l in range(L)] for k in ("delta", "lp_g", "lp_s", "lp_unknown", "top1")},
           "rev": {k: [rev[l][k] for l in range(L)] for k in ("delta", "lp_g", "lp_s", "lp_unknown", "top1")},
           "T_fwd": t_fwd, "T_rev": t_rev, "d_T_fwd": transfer_depth(t_fwd, L), "d_T_rev": transfer_depth(t_rev, L),
           "sanity_fwd": transfer_sanity(t_fwd), "sanity_rev": transfer_sanity(t_rev)}
    selfr = _sweep(engine, state["clean_prep"], {l: h_clean[l] for l in self_layers(L)}, x)
    self_shift = {str(l): dc - selfr[l]["delta"] for l in self_layers(L)}
    out["control_self"] = {"layers": list(self_layers(L)), "shift": self_shift,
                           "T": {str(l): (selfr[l]["delta"] - dc) / (ds - dc) if abs(ds - dc) > MIN_GAP else None
                                 for l in self_layers(L)},
                           "pass": all(abs(v) < SELF_TOL for v in self_shift.values())}
    rnd_vecs, rnd_norms = {}, []
    for l in range(L):
        rnd_vecs[l], n = difference_perturbation(h_clean[l], h_swap[l], row_index, l)
        rnd_norms.append(n)
    rnd = _sweep(engine, state["clean_prep"], rnd_vecs, x)
    d_rnd = [rnd[l]["delta"] for l in range(L)]
    out["control_random"] = {"kind": "difference_norm_perturbation", "delta": d_rnd,
                             "shift": shift_toward_s(d_rnd, dc), "T": transfer(d_rnd, dc, ds),
                             "top1": [rnd[l]["top1"] for l in range(L)], "eps_norm": rnd_norms}
    if donor_resid is not None:
        donor_clean, donor_swap = donor_resid
        mm = _sweep(engine, state["clean_prep"],
                    {l: difference_patch(h_clean[l], donor_clean[l], donor_swap[l]) for l in range(L)}, x)
        d_mm = [mm[l]["delta"] for l in range(L)]
        out["control_mismatch"] = {"kind": "difference_patch", "delta": d_mm, "shift": shift_toward_s(d_mm, dc),
                                   "T": transfer(d_mm, dc, ds), "top1": [mm[l]["top1"] for l in range(L)],
                                   "donor_diff_norm": [float(np.linalg.norm(donor_swap[l].astype(np.float64)
                                                                            - donor_clean[l])) for l in range(L)]}
    else:
        out["control_mismatch"] = None
    row.update(out)
    row["patched"] = True
    row["timing"]["patch_s"] = time.perf_counter() - started
    row["timing"]["n_patch_passes"] = 4 * L + len(self_layers(L)) - (0 if donor_resid is not None else L)
    return row


def check_patch_sanity(row: dict) -> None:
    if not row.get("patched"):
        return
    for key in ("sanity_fwd", "sanity_rev"):
        s = row[key]
        if s["pass"] is not True:
            raise SanityError(f"{row['question_id']}: {key} failed: T(L-1)={s['t_final']}, T(0)={s['t_zero']}")
    if not row["control_self"]["pass"]:
        raise SanityError(f"{row['question_id']}: self-overwrite moved Delta: {row['control_self']['shift']}")


# --- condition U ------------------------------------------------------------------------------------------------

def u_continuation(note: str, kind: str) -> str:
    """U / U_steer: the §35 arm B continuation (note + newline + B_INSTRUCTION + newline + 'Answer:'); A / A_steer:
    the arm A continuation (note + 'Answer:'), the §38 push's input."""
    from memctl.analysis import answer_selection as sel
    return sel.frozen_continuation("B" if kind.startswith("U") else "A", note)


def u_wording(model: str) -> dict:
    from memctl.analysis import answer_selection as sel
    from memctl.analysis import steering as st
    layer, alpha, unit, path = STEER[model]
    return {"b_instruction": sel.B_INSTRUCTION, "answer_marker": sel.ANSWER_MARKER, "max_new_tokens": U_MAX_NEW,
            "sampling": st.SAMPLING, "steer": {"layer": layer, "alpha": alpha, "unit": unit, "directions": path,
                                               "steps": "every step"},
            "classifier": st.CLASSIFIER_VERSION, "gold": "steering.provisional_gold (D1 on the phrase, answers only)"}


def u_wording_sha256(model: str) -> str:
    return hashlib.sha256(json.dumps(u_wording(model), sort_keys=True).encode()).hexdigest()


def steer_vector(model: str, directions_json: dict | None = None) -> np.ndarray:
    """alpha * v_L from the §38 / §43 directions file; the raw vector's sha is checked against directions.json."""
    from memctl.analysis import steering as st
    from pathlib import Path
    layer, alpha, unit, path = STEER[model]
    v = np.load(path)[f"L{layer}"]
    meta = directions_json or json.loads(Path(path).with_suffix(".json").read_text())
    want = meta["layers"][str(layer)]["vector_sha256"]
    if st.vector_sha(v) != want:
        raise ValueError(f"{model}: v_{layer} sha {st.vector_sha(v)} != directions.json {want}")
    return st.scaled_vector(v, alpha, unit)


def u_generate(engine, record, kinds, steer_vec: np.ndarray, layer: int, max_new: int = U_MAX_NEW) -> dict:
    """One greedy generation per kind on the cached note's continuation; rows carry the phrase, class and D1 gold."""
    from memctl.analysis import answer_selection as sel
    from memctl.analysis import steering as st
    note, _ = sel.split_note(record.output)
    if note is None:
        return {"no_marker": True}
    out = {"no_marker": False}
    for kind in kinds:
        item = st._make(kind, record, record.prompt, u_continuation(note, kind))
        prepared = engine.prepare(item)
        specs = [st.add_spec(layer, steer_vec)] if kind.endswith("_steer") else []
        t0 = time.perf_counter()
        gen = engine.run(prepared, specs, max_new)
        phrase = st.phrase_of(gen.generated)
        cls = st.classify(phrase, gen.gen_ids)
        out[kind] = {"input_sha256": item.input_sha256, "last_token": engine.last_token_text(prepared),
                     "n_input_tokens": len(prepared.ids), "generated": gen.generated, "gen_ids": gen.gen_ids,
                     "phrase": phrase, "cls": cls, "gold": st.provisional_gold(phrase, record.gold, cls),
                     "vector_sha256": st.vector_sha(steer_vec) if specs else None,
                     "seconds": time.perf_counter() - t0 + prepared.prefill_s}
    return out


# --- claims (§C) ------------------------------------------------------------------------------------------------

def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values)


def proportion(flags: list[bool], samples: int = BOOT_SAMPLES, seed: int = 0) -> dict:
    out = pt.bootstrap([1.0 if f else 0.0 for f in flags], stat=_mean, samples=samples, seed=seed)
    out["k"] = sum(bool(f) for f in flags)
    out["half_width"] = None if out["value"] is None else (out["ci_high"] - out["ci_low"]) / 2
    return out


def diff_of_proportions(a: list[bool], b: list[bool], samples: int = BOOT_SAMPLES, seed: int = 0) -> dict:
    """p(a) - p(b) with a percentile bootstrap resampling each group separately."""
    if not a or not b:
        return {"n_a": len(a), "n_b": len(b), "value": None, "ci_low": None, "ci_high": None}
    rng = random.Random(seed)
    fa, fb = [float(x) for x in a], [float(x) for x in b]
    draws = sorted(_mean(rng.choices(fa, k=len(fa))) - _mean(rng.choices(fb, k=len(fb))) for _ in range(samples))
    low, high = pt._interval(draws)
    return {"n_a": len(a), "n_b": len(b), "value": _mean(fa) - _mean(fb), "ci_low": low, "ci_high": high}


def s0(success_rows: list[dict], samples: int = BOOT_SAMPLES) -> dict:
    """Following rate (1 nat) among the small model's successes: valid iff rate >= 0.80 and CI lower >= 0.70."""
    st_ = proportion([r["follows"]["1"] for r in success_rows], samples)
    ok = st_["value"] is not None and st_["value"] >= S0_RATE and st_["ci_low"] >= S0_LB
    st_["by_margin"] = {f"{m:g}": _share([r["follows"][f"{m:g}"] for r in success_rows]) for m in MARGINS}
    return {**st_, "pass": ok, "reading": None if st_["value"] is None else ("valid" if ok else "swap not valid")}


def _share(flags) -> float | None:
    flags = list(flags)
    return sum(map(bool, flags)) / len(flags) if flags else None


def s1(failure_rows: list[dict], samples: int = BOOT_SAMPLES) -> dict:
    """Shares of the three classes among in-view failures; (a) indeterminate <= 0.20, (b) the answer-production share
    moves < 0.15 between the 0.5- and 2-nat margins, (c) CI half-width <= 0.15; then the share is read."""
    n = len(failure_rows)
    shares = {f"{m:g}": {c: _share([r["cls"][f"{m:g}"] == c for r in failure_rows]) for c in CLASSES} for m in MARGINS}
    ap = proportion([r["cls"]["1"] == "answer_production" for r in failure_rows], samples)
    if n == 0:
        return {"n": 0, "shares": shares, "ap": ap, "a": None, "b": None, "c": None, "established": None,
                "reading": None}
    a = shares["1"]["indeterminate"] <= S1_IND_MAX
    move = abs(shares["0.5"]["answer_production"] - shares["2"]["answer_production"])
    b = move < S1_MARGIN_SHIFT
    c = ap["half_width"] <= S1_HALF_WIDTH
    est = a and b and c
    if not est:
        reading = "not reliably distinguishable on these rows"
    elif ap["value"] >= S1_AP_MAJORITY:
        reading = "answer-production failures are the majority"
    elif ap["value"] <= S1_EU_MAJORITY:
        reading = "evidence-use failures are the majority"
    else:
        reading = "mixed"
    return {"n": n, "shares": shares, "ap": ap, "indeterminate": shares["1"]["indeterminate"], "ap_margin_move": move,
            "a": a, "b": b, "c": c, "established": est, "reading": reading}


def s2(refusal_rows: list[dict], samples: int = BOOT_SAMPLES) -> dict:
    """Among refusal rows (each with cls and steer_gold): p(Steer gold | answer-production) - p(Steer gold |
    evidence-use), CI lower > 0; read only with >= 8 rows per class."""
    ap = [bool(r["steer_gold"]) for r in refusal_rows if r["cls"]["1"] == "answer_production"]
    eu = [bool(r["steer_gold"]) for r in refusal_rows if r["cls"]["1"] == "evidence_use"]
    d = diff_of_proportions(ap, eu, samples)
    readable = len(ap) >= S2_MIN_PER_CLASS and len(eu) >= S2_MIN_PER_CLASS
    shown = None if not readable or d["value"] is None else d["ci_low"] > 0
    return {**d, "p_ap": _share(ap), "p_eu": _share(eu), "readable": readable, "shown": shown,
            "reading": ("descriptive (fewer than 8 refusal rows in a class)" if not readable
                        else "push helps answer-production refusals more" if shown else "not shown")}


def s3(patched_rows: list[dict], samples: int = BOOT_SAMPLES) -> dict:
    """Median forward transfer depth with CI, the reverse depth beside, and the S3 validity check (rev 6 ruling 3).

    shift(l) = Delta_clean - Delta_patched(l) (toward S positive), the same quantity for the donor control, the random
    control and the forward patch. Mean over the patched (following) rows, per layer.
    Primary: the donor's mean shift within +-0.5 nat at every layer. Pre-declared fallback (used only if the primary
    fails): at every layer l >= l0 = ceil(median d_T_fwd * L), donor mean shift(l) <= 0.25 * forward mean shift(l)
    (signed, as written). S3 is read iff either holds; the absolute check is reported in both cases."""
    fwd = [r["d_T_fwd"] for r in patched_rows if r.get("d_T_fwd") is not None]
    rev = [r["d_T_rev"] for r in patched_rows if r.get("d_T_rev") is not None]
    mm = [r["control_mismatch"]["shift"] for r in patched_rows if r.get("control_mismatch")]
    per_layer = [float(np.mean(col)) for col in zip(*mm)] if mm else []
    within = bool(per_layer) and all(abs(v) <= S3_DONOR_BOUND for v in per_layer)
    rnd = [r["control_random"]["shift"] for r in patched_rows if r.get("control_random")]
    fwd_shift = [shift_toward_s(r["fwd"]["delta"], r["clean"]["delta"]) for r in patched_rows
                 if r.get("control_mismatch") and r.get("fwd")]
    fwd_per_layer = [float(np.mean(col)) for col in zip(*fwd_shift)] if fwd_shift else []
    fb = s3_fallback(per_layer, fwd_per_layer, median(fwd) if fwd else None)
    fallback_used = (not within) and fb["pass"] is True
    read = within or fallback_used
    return {"d_T_fwd": pt.bootstrap(fwd, median, samples), "d_T_rev": pt.bootstrap(rev, median, samples),
            "asymmetry_median": float(median([a - b for a, b in zip(fwd, rev)])) if fwd and len(fwd) == len(rev)
            else None,
            "shift_definition": "shift(l) = Delta_clean - Delta_patched(l), positive = toward S; mean over the "
                                "patched (following) rows; same quantity for donor, random and forward patch",
            "mismatch_mean_shift_per_layer": per_layer,
            "mismatch_max_abs_mean_shift": max((abs(v) for v in per_layer), default=None),
            "mismatch_within_bound": within if mm else None,
            "forward_mean_shift_per_layer": fwd_per_layer,
            "fallback": fb, "fallback_used": fallback_used,
            "validity": ("primary (|donor mean shift| <= 0.5 at every layer)" if within
                         else "fallback (specificity ratio <= 0.25 from the median d_T onward)" if fallback_used
                         else "not read (primary and fallback both fail)"),
            "random_mean_shift_per_layer": [float(np.mean(col)) for col in zip(*rnd)] if rnd else [],
            "read": read, "n_rows": len(patched_rows), "n_with_donor": len(mm)}


def s3_fallback(donor_per_layer: list[float], fwd_per_layer: list[float], median_d_t: float | None,
                ratio: float = S3_SPECIFICITY_RATIO) -> dict:
    """Rev 6 ruling 3 fallback: at every layer from l0 = ceil(median d_T * L) on, donor <= ratio * forward."""
    L = len(donor_per_layer)
    if not L or len(fwd_per_layer) != L or median_d_t is None:
        return {"pass": None, "l0": None, "ratio": ratio, "failing_layers": None}
    l0 = min(L - 1, math.ceil(median_d_t * L - 1e-9))
    failing = [l for l in range(l0, L) if not donor_per_layer[l] <= ratio * fwd_per_layer[l]]
    return {"pass": not failing, "l0": l0, "ratio": ratio, "failing_layers": failing,
            "donor_over_forward": [donor_per_layer[l] / fwd_per_layer[l] if fwd_per_layer[l] else None
                                   for l in range(l0, L)]}


def u1(u_rows: list[dict], samples: int = BOOT_SAMPLES) -> dict:
    """Share of in-view refusals that become gold under U, with the Steer-under-U minus U difference (paired)."""
    rows = [r for r in u_rows if not r.get("no_marker")]
    share = proportion([r["U"]["gold"] for r in rows], samples)
    if share["value"] is None:
        reading = None
    elif share["value"] >= U1_HIGH and share["ci_low"] > U1_HIGH_LB:
        reading = "the refusal disposition is mostly a prompt artefact"
    elif share["value"] <= U1_LOW and share["ci_high"] < U1_LOW_UB:
        reading = "not the format"
    else:
        reading = "partial"
    paired = [float(r["U_steer"]["gold"]) - float(r["U"]["gold"]) for r in rows if "U_steer" in r]
    return {**share, "n_no_marker": len(u_rows) - len(rows), "reading": reading,
            "steer_minus_u": pt.bootstrap(paired, _mean, samples),
            "u_steer_share": _share([r["U_steer"]["gold"] for r in rows if "U_steer" in r]),
            "u_refusal_share": _share([r["U"]["cls"] == "refusal" for r in rows])}
