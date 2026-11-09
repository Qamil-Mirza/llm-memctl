"""Stage 0 scoring: the authors' answer extraction and equivalence (vendored), the four categories, the doubt coder.

Gold = the answer parsed from original_answer exactly as the authors' score_response does it:
  gsm8k   : "#### x" (extract_gsm8k_gt_answer); prediction = sanitize_answer(extract_answer(text)); string equality
  math500 : original_answer is already a bare answer (not boxed) -> strip_answer_string; prediction =
            strip_answer_string(extract_answer(text)); math_equal
"stale" compares a U answer with the same prefix's N answer by the same rule (string equality after sanitize for
gsm8k, math_equal for math500). Each sympy-backed call runs under the authors' 5 s timeout (timeout -> not equal).
"""
from __future__ import annotations

import multiprocessing as mp
import re
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FTimeout

from selfdoubt.vendor.interrupt_lrm import math_utils as MU
from selfdoubt.vendor.interrupt_lrm.math_parsing_util import extract_answer, math_equal, strip_answer_string

TIMEOUT_S = 5
_POOL = None


def _pool():
    global _POOL
    if _POOL is None:
        _POOL = ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context("spawn"))
    return _POOL


def _timed(fn, *args, default=False):
    fut = _pool().submit(fn, *args)
    try:
        return fut.result(timeout=TIMEOUT_S)
    except FTimeout:
        fut.cancel()
        return default


def gold_answer(source: str, original_answer: str) -> str:
    if source == "gsm8k":
        return MU.extract_gsm8k_gt_answer(original_answer)
    if source == "math500":
        return strip_answer_string(original_answer)
    raise ValueError(source)


def _pred(source: str, text: str) -> str:
    p = extract_answer(text)
    return MU.sanitize_answer(p) if source == "gsm8k" else strip_answer_string(p)


def pred_answer(source: str, text: str) -> str:
    return _timed(_pred, source, text, default="")


def is_correct(source: str, original_answer: str, text: str) -> bool:
    """The authors' score_response (gold from original_answer), with their 5 s timeout -> 0."""
    return bool(_timed(MU.score_response, source, original_answer, text, default=False))


def _same(source: str, a: str, b: str) -> bool:
    if not a or not b or a == "[invalid]" or b == "[invalid]":
        return False
    if source == "gsm8k":
        return a == b
    return bool(math_equal(a, b))


def same_answer(source: str, a: str, b: str) -> bool:
    return bool(_timed(_same, source, a, b, default=False))


def valid_pred(p: str) -> bool:
    return bool(p) and p != "[invalid]"


def has_final_answer(text: str) -> bool:
    return "</think>" in text and "boxed" in text.split("</think>", 1)[1]


def is_truncated(rec: dict) -> bool:
    """Hit max_tokens without a final answer (no \\boxed after </think>)."""
    return bool(rec["hit_max"]) and not has_final_answer(rec["text"])


def categorize(*, n_valid: bool, n_is_gold: bool, truncated: bool, correct: bool, stale: bool) -> str:
    """Category of one U0/UG continuation at a cut. Drop rules are per item-cut and come first."""
    if not n_valid:
        return "dropped_ref_invalid"
    if n_is_gold:
        return "dropped_ref_is_gold"
    if truncated:
        return "truncated"
    if correct:
        return "new"
    if stale:
        return "stale"
    return "other"


# ---------------------------------------------------------------- doubt wording (placeholder coder)

# TODO(Stage 0, dev): replace/augment with the LLM judge (proposal: keyword rules + judge, checked against 50 dev
# hand labels). These rules are a placeholder only; R0.3 computed from them is labelled PLACEHOLDER in the report.
DOUBT_PATTERNS = [r"\bwait\b", r"\bhmm+\b", r"\bdouble[- ]check", r"\bre-?check", r"\bmisread",
                  r"\bis (that|this) (right|correct)\b", r"\bconfus", r"\bconflict", r"\bcontradict",
                  r"\bnot sure\b", r"\bdoubt", r"\bmistake\b", r"\btypo\b", r"\bbut the original\b",
                  r"\bdid i\b", r"\bam i\b"]
_DOUBT_RE = re.compile("|".join(DOUBT_PATTERNS), re.I)


def mask_injected(text: str, injected: str, update: str) -> str:
    """Remove the injected wrapper text (and any echo of it, line by line, and the update itself) before coding."""
    frags = [injected, update] + [ln.strip() for ln in injected.splitlines() if ln.strip()]
    for f in sorted(set(frags), key=len, reverse=True):
        if f:
            text = text.replace(f, " ")
    return text


def doubt_flag(text: str, injected: str, update: str) -> bool:
    thinking = text.split("</think>", 1)[0]
    return bool(_DOUBT_RE.search(mask_injected(thinking, injected, update)))
