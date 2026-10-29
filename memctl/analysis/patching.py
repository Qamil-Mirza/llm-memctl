"""§44: single-position activation patching, clean -> evidence-cut, at the forced answer position P1.

Pre-registration: research/proposals/2026-10-29-exp44-patching.md (rev 0, 6e21476); D-44.

Per row (a §34 population PromptRecord), with the §34 forced readout (`jlens_readout.readout_for`: ids(chat text) +
ids("Answer:") + the gold's teacher-forced whitespace-only tokens; P1 = the last token, which predicts the gold's
first content token):

- clean: the intact prompt. The KV cache covers every token before P1; the P1 token is run once with a capture hook
  on every decoder layer: h_clean(l) = the output of decoder layer l at P1 (pre final norm), l = 0..L-1, and
  m_clean = log p(gold first token) from the final logits.
- corrupt: the §34c cut prompt (`ablate_evidence`: evidence lines removed by exact span, the rest identical, forced
  `Answer:` + whitespace as §34, no note). KV cache up to P1-1, then the P1 token: m_corrupt and the gold's rank.
- patched(l): the corrupt cache, the P1 token run with `steering.overwrite_spec(l, h_clean(l))` (the P1 residual at
  layer l's output replaced by the clean one; layers > l run on the corrupt cache). m_patched(l) from the final logits.

R(l) = (m_patched(l) - m_corrupt) / (m_clean - m_corrupt), unclipped (`clip_r` for plots only). Causal emergence depth
d_c = (first l with R(l) >= 0.5) / L. Sanity (spec §2, asserted on every row that the run treats as asserted, see
`sanity_asserted`): |R(L-1) - 1| < 0.01 and |R(0)| < 0.05.

Alignment by role: the cut prompt is shorter (the evidence is gone), so P1 is matched as the answer position, not by
index: both sequences end at their own P1 (= len - 1), the P1 token, the tail from the question line's last token to
P1 and the tracked gold id are identical (`check_alignment`, raises AlignmentError).

Evidence-dependent filter (spec §3): clean final gold rank <= 10 and corrupt final gold rank > 10. The single
pre-declared dev relaxation: if fewer than 40 of the 126 dev rows survive for a model, corrupt rank > 3 (recorded in
rev 1 before any confirmation read). `PATCH_ELIGIBLE` (clean <= 10 and corrupt > 3) is the widest of the two: rows
outside it can never enter a claim, so by default the run records only their clean and corrupt passes.
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

from memctl.analysis.jacobian import ablate_evidence
from memctl.analysis.jlens_readout import ranks_of, readout_for

# --- pre-registered numbers (spec §2-§4) ------------------------------------------------------------------------
R_THRESHOLD = 0.5
SANITY_FINAL_TOL = 0.01
SANITY_ZERO_TOL = 0.05
CLIP = (-0.25, 1.25)
K_CLEAN = 10                 # clean final gold rank <= 10
K_CORRUPT = 10               # corrupt final gold rank > 10
K_CORRUPT_RELAXED = 3        # the one pre-declared dev relaxation: corrupt final rank > 3
RELAX_BELOW = 40             # ... applied if fewer than 40 of the 126 dev rows survive
DEV_FOLDS = (0, 1)
CONFIRM_FOLDS = (2, 3, 4)
N_DEV = 126
C1_JLENS_UPPER = 0.65        # C1: CI upper < 0.65 -> J-lens right
C1_LOGIT_LOWER = 0.70        # C1: CI lower > 0.70 -> logit lens right
C2_DEPTH = 0.464             # C2: mean R at layer round(0.464 * 28) = 13 on the 7B
C2_HIGH, C2_LOW = 0.5, 0.25
C3_EARLY_UPPER = 0.75        # C3: CI upper < 0.75 -> lenses under-read the 3B too
C3_AGREE = (0.85, 0.92)      # C3: CI contains 0.85-0.92 -> agrees with both lenses
BOOT_SAMPLES = 10000
MIN_GAP = 1e-6               # |m_clean - m_corrupt| below this: R undefined (None)

METHOD = {"spec": "research/proposals/2026-10-29-exp44-patching.md rev 0 (6e21476)",
          "clean": "§34 forced readout_for, prefix cache to P1-1, P1 run with capture on every layer",
          "corrupt": "ablate_evidence + readout_for(cut, non-evidence memory, (), gold): §34c ablated, forced",
          "patch": "steering.overwrite_spec(l, h_clean(l)) at P1 on the corrupt cache, one layer per pass",
          "metric": "log_softmax(final logits)[gold first content token], float32",
          "residual": "decoder layer output, pre final norm"}


def method_sha256() -> str:
    return hashlib.sha256(json.dumps(METHOD, sort_keys=True).encode()).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def sha256_ids(ids) -> str:
    return hashlib.sha256(json.dumps(list(map(int, ids))).encode()).hexdigest()


class AlignmentError(ValueError):
    """The clean and cut sequences do not end at the same answer position by role."""


class SanityError(RuntimeError):
    """R(L-1) != 1 or R(0) != 0 beyond the spec's tolerance on an asserted row."""


# --- arithmetic -------------------------------------------------------------------------------------------------

def recovery(m_patched, m_clean: float, m_corrupt: float) -> list[float] | None:
    """R(l) for every layer, unclipped; None when the clean and corrupt log p coincide (undefined)."""
    gap = m_clean - m_corrupt
    if abs(gap) < MIN_GAP:
        return None
    return [(float(m) - m_corrupt) / gap for m in m_patched]


def clip_r(r: float) -> float:
    return min(max(r, CLIP[0]), CLIP[1])


def causal_depth(r: list[float] | None, n_layers: int, threshold: float = R_THRESHOLD) -> float | None:
    """(first l with R(l) >= threshold) / L; None if no layer reaches it (or R is undefined)."""
    if r is None:
        return None
    for layer, value in enumerate(r):
        if value >= threshold:
            return layer / n_layers
    return None


def sanity(r: list[float] | None) -> dict:
    """The spec's two bounds; `pass` is None when R is undefined."""
    if r is None:
        return {"r_final": None, "r_zero": None, "final_ok": None, "zero_ok": None, "pass": None}
    final_ok = abs(r[-1] - 1.0) < SANITY_FINAL_TOL
    zero_ok = abs(r[0]) < SANITY_ZERO_TOL
    return {"r_final": r[-1], "r_zero": r[0], "final_ok": final_ok, "zero_ok": zero_ok, "pass": final_ok and zero_ok}


def evidence_dependent(rank_clean: int, rank_corrupt: int, k_corrupt: int = K_CORRUPT) -> bool:
    return rank_clean <= K_CLEAN and rank_corrupt > k_corrupt


def patch_eligible(rank_clean: int, rank_corrupt: int) -> bool:
    """In the widest pre-declared filter (the relaxed one): the only rows any claim can read."""
    return evidence_dependent(rank_clean, rank_corrupt, K_CORRUPT_RELAXED)


def choose_filter(dev_rows: list[dict]) -> dict:
    """Spec §3 on the dev rows of one model: the strict filter unless fewer than 40 of them survive, then the
    relaxed one. `dev_rows`: per-row outputs (rank_clean, rank_corrupt) of folds 0-1."""
    strict = [r for r in dev_rows if evidence_dependent(r["rank_clean"], r["rank_corrupt"], K_CORRUPT)]
    relaxed = [r for r in dev_rows if evidence_dependent(r["rank_clean"], r["rank_corrupt"], K_CORRUPT_RELAXED)]
    relax = len(strict) < RELAX_BELOW
    return {"n_dev": len(dev_rows), "n_strict": len(strict), "n_relaxed": len(relaxed), "relaxed": relax,
            "k_corrupt": K_CORRUPT_RELAXED if relax else K_CORRUPT}


def log_softmax(logits: np.ndarray) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float64)
    x = x - x.max()
    return (x - math.log(np.exp(x).sum())).astype(np.float64)


# --- prompts and alignment --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class RowInputs:
    question_id: str
    fold: int
    clean_prompt: str
    cut_prompt: str
    clean_ids: tuple[int, ...]
    cut_ids: tuple[int, ...]
    clean_p1: int
    cut_p1: int
    clean_q: int
    cut_q: int
    gold_id: int
    unknown_id: int
    gold_token_text: str
    n_leading_ws: int
    n_evidence_lines: int


def build_inputs(tokenizer, record, chat_text_fn=None) -> RowInputs:
    """Clean = §34 forced readout of the intact prompt; cut = §34c (ablate_evidence, forced). Aligned by role."""
    clean = readout_for(tokenizer, record.prompt, record.memory_lines, record.evidence_lines, record.gold,
                        chat_text_fn=chat_text_fn)
    cut_prompt = ablate_evidence(record.prompt, record.memory_lines, record.evidence_lines)
    evidence = set(record.evidence_lines)
    memory = tuple(l for l in record.memory_lines if l not in evidence)
    cut = readout_for(tokenizer, cut_prompt, memory, (), record.gold, chat_text_fn=chat_text_fn)
    inputs = RowInputs(record.question_id, record.fold, record.prompt, cut_prompt, tuple(clean.input_ids),
                       tuple(cut.input_ids), clean.positions["P1"][0], cut.positions["P1"][0],
                       clean.positions["Q"][0], cut.positions["Q"][0], clean.tracked["gold"],
                       clean.tracked["unknown_sp"], clean.gold_token_text, clean.n_leading_ws, len(record.evidence_lines))
    if cut.tracked != clean.tracked or cut.n_leading_ws != clean.n_leading_ws:
        raise AlignmentError(f"{record.question_id}: tracked ids or forced whitespace differ between clean and cut")
    check_alignment(inputs)
    return inputs


def check_alignment(x: RowInputs) -> None:
    """P1 is the answer position in both sequences: the last token, the same token, the same distance from the
    question line's last token, an identical tail from there to P1; and the cut is strictly shorter."""
    where = x.question_id
    if x.clean_p1 != len(x.clean_ids) - 1 or x.cut_p1 != len(x.cut_ids) - 1:
        raise AlignmentError(f"{where}: P1 is not the last token of its sequence")
    if x.clean_ids[x.clean_p1] != x.cut_ids[x.cut_p1]:
        raise AlignmentError(f"{where}: the P1 tokens differ ({x.clean_ids[x.clean_p1]} vs {x.cut_ids[x.cut_p1]})")
    tail_clean, tail_cut = x.clean_p1 - x.clean_q, x.cut_p1 - x.cut_q
    if tail_clean != tail_cut or x.clean_ids[x.clean_q:] != x.cut_ids[x.cut_q:]:
        raise AlignmentError(f"{where}: the question-to-P1 tails differ ({tail_clean} vs {tail_cut} tokens)")
    if x.n_evidence_lines and not len(x.cut_ids) < len(x.clean_ids):
        raise AlignmentError(f"{where}: the cut sequence ({len(x.cut_ids)}) is not shorter than the clean ({len(x.clean_ids)})")


# --- the passes -------------------------------------------------------------------------------------------------

def _prepare(engine, ids):
    from memctl.analysis.steering import Prepared

    t0 = time.perf_counter()
    cache = engine._prefill(list(ids[:-1]))
    return Prepared(list(ids), int(ids[-1]), cache, len(ids) - 1, time.perf_counter() - t0)


def _p1_logits(engine, prepared, specs, capture=()) -> tuple[np.ndarray, dict]:
    gen = engine.run(prepared, specs, max_new_tokens=1, capture=tuple(capture), return_logits=True)
    return np.asarray(gen.logits0, dtype=np.float32), gen.captured


def _readout(logits: np.ndarray, gold: int, unknown: int) -> dict:
    lp = log_softmax(logits)
    rk = ranks_of(logits[None, :], [gold, unknown])[0]
    return {"lp_gold": float(lp[gold]), "lp_unknown": float(lp[unknown]), "rank_gold": int(rk[0]),
            "rank_unknown": int(rk[1]), "top1": int(np.argmax(logits))}


def patch_row(engine, x: RowInputs, patch: bool | None = None) -> dict:
    """The three passes for one row. `patch` None: patch only `patch_eligible` rows; True / False forces it.
    Returns the measured row (no stamps); R and d_c when patched."""
    from memctl.analysis.steering import overwrite_spec

    L = engine.n_layers
    started = time.perf_counter()
    clean_prep = _prepare(engine, x.clean_ids)
    clean_logits, captured = _p1_logits(engine, clean_prep, [], capture=range(L))
    if sorted(captured) != list(range(L)):
        raise RuntimeError(f"{x.question_id}: captured layers {sorted(captured)} != 0..{L - 1}")
    del clean_prep
    clean = _readout(clean_logits, x.gold_id, x.unknown_id)
    cut_prep = _prepare(engine, x.cut_ids)
    corrupt = _readout(_p1_logits(engine, cut_prep, [])[0], x.gold_id, x.unknown_id)
    t_base = time.perf_counter() - started
    do_patch = patch_eligible(clean["rank_gold"], corrupt["rank_gold"]) if patch is None else bool(patch)
    row = {"question_id": x.question_id, "fold": x.fold, "n_layers": L,
           "n_tokens_clean": len(x.clean_ids), "n_tokens_cut": len(x.cut_ids), "p1_clean": x.clean_p1,
           "p1_cut": x.cut_p1, "p1_token": int(x.clean_ids[-1]), "gold_id": x.gold_id, "unknown_id": x.unknown_id,
           "gold_token_text": x.gold_token_text, "n_leading_ws": x.n_leading_ws, "n_evidence_lines": x.n_evidence_lines,
           "lp_clean": clean["lp_gold"], "lp_corrupt": corrupt["lp_gold"],
           "rank_clean": clean["rank_gold"], "rank_corrupt": corrupt["rank_gold"],
           "lp_unknown_clean": clean["lp_unknown"], "lp_unknown_corrupt": corrupt["lp_unknown"],
           "top1_clean": clean["top1"], "top1_corrupt": corrupt["top1"],
           "eligible": patch_eligible(clean["rank_gold"], corrupt["rank_gold"]),
           "evidence_dependent": evidence_dependent(clean["rank_gold"], corrupt["rank_gold"]),
           "patched": do_patch, "resid_norm_clean": [float(np.linalg.norm(captured[l])) for l in range(L)]}
    if do_patch:
        per = [_readout(_p1_logits(engine, cut_prep, [overwrite_spec(l, captured[l])])[0], x.gold_id, x.unknown_id)
               for l in range(L)]
        r = recovery([p["lp_gold"] for p in per], clean["lp_gold"], corrupt["lp_gold"])
        row.update({"lp_patched": [p["lp_gold"] for p in per], "rank_patched": [p["rank_gold"] for p in per],
                    "lp_unknown_patched": [p["lp_unknown"] for p in per], "top1_patched": [p["top1"] for p in per],
                    "R": r, "d_c": causal_depth(r, L), "sanity": sanity(r)})
    row["timing"] = {"base_s": t_base, "total_s": time.perf_counter() - started,
                     "n_patch_passes": L if do_patch else 0}
    return row


def sanity_asserted(row: dict) -> bool:
    """Spec §2 asserts the bounds on every row; they are asserted on the patched rows inside the widest filter
    (where the gap is a real cut effect); a row patched outside it (--patch-all) has its bounds recorded only."""
    return bool(row.get("patched")) and bool(row.get("eligible"))


def check_sanity(row: dict) -> None:
    if not sanity_asserted(row):
        return
    s = row["sanity"]
    if s["pass"] is not True:
        raise SanityError(f"{row['question_id']}: sanity failed: R(L-1)={s['r_final']}, R(0)={s['r_zero']}")


# --- statistics -------------------------------------------------------------------------------------------------

def _interval(draws: list[float]) -> tuple[float, float]:
    draws = sorted(draws)
    return draws[int(0.025 * len(draws))], draws[min(len(draws) - 1, int(0.975 * len(draws)))]


def bootstrap(values: list[float], stat=median, samples: int = BOOT_SAMPLES, seed: int = 0) -> dict:
    """The statistic with a 95% percentile interval over `samples` resamples of the rows."""
    if not values:
        return {"n": 0, "value": None, "ci_low": None, "ci_high": None}
    rng = random.Random(seed)
    draws = [stat(rng.choices(values, k=len(values))) for _ in range(samples)]
    low, high = _interval(draws)
    return {"n": len(values), "value": float(stat(values)), "ci_low": float(low), "ci_high": float(high)}


def mean(values) -> float:
    values = list(values)
    return sum(values) / len(values)


def c1_reading(stat: dict) -> str | None:
    if stat["value"] is None:
        return None
    if stat["ci_high"] < C1_JLENS_UPPER:
        return "J-lens right"
    if stat["ci_low"] > C1_LOGIT_LOWER:
        return "logit lens right"
    return "both partly right"


def c2_reading(stat: dict) -> str | None:
    if stat["value"] is None:
        return None
    if stat["value"] > C2_HIGH and stat["ci_low"] > C2_HIGH:
        return "carries most of the answer"
    if stat["value"] < C2_LOW and stat["ci_high"] < C2_LOW:
        return "carries little"
    return "neither (between)"


def c3_reading(stat: dict) -> str | None:
    if stat["value"] is None:
        return None
    if stat["ci_high"] < C3_EARLY_UPPER:
        return "lenses under-read the 3B (late picture a readout artefact)"
    if stat["ci_low"] <= C3_AGREE[0] and stat["ci_high"] >= C3_AGREE[1]:
        return "agrees with both lenses"
    return "neither pre-registered reading"


def c3_overlaps_band(stat: dict) -> bool | None:
    """Shown beside the literal C3 reading (the CI contains the whole 0.85-0.92 band): whether it overlaps it."""
    if stat["value"] is None:
        return None
    return stat["ci_low"] <= C3_AGREE[1] and stat["ci_high"] >= C3_AGREE[0]


def c2_layer(n_layers: int = 28) -> int:
    return int(round(C2_DEPTH * n_layers))
