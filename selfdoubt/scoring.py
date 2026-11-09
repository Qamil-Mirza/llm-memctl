r"""Stage 0 scoring: answer extraction (vendored) with the frozen Stage 0 normaliser, the four categories, the doubt coder.

gsm8k   : the authors' rule, unchanged: gold "#### x" (extract_gsm8k_gt_answer); prediction =
          sanitize_answer(extract_answer(text)); string equality.
math500 : the Stage 0 normaliser (Advisor, 2026-11-09), replacing the authors' strip_answer_string(extract_answer(.)),
          which strips twice and so deletes pmatrix row separators (a correct (-2, -14, -7) became "-2-14-7"):
          - gold, prediction and the N reference all take ONE path (canon): gold is canonicalised as if the model had
            written \boxed{gold};
          - vectors / tuples (pmatrix, bmatrix, array, and "(a, b, ...)" incl. \left( \right)) are parsed from the raw
            boxed text into a list of entries, each stripped once, and compared elementwise with math_equal; equal
            length (shape) is required, and a scalar never equals a vector. Intervals with mixed brackets ("(0,1]",
            "[0,1)") and "[a, b]" stay scalars (vendored math_equal handles them);
          - scalars: extract_answer(text) (one strip), then strip_answer_string's prefix keys (x=, x\\in, ...; the only
            other thing the released second strip changed on dev golds and stub outputs), then math_equal; a stripped
            scalar that is a parenthesised tuple becomes a vector.
          is_correct_authors() keeps the authors' score_response for comparison.
"stale" compares a U answer with the same prefix's N answer by the same rule (string equality after sanitize for
gsm8k, the normaliser for math500). Each sympy-backed call runs under the authors' 5 s timeout (timeout -> not equal).
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


def boxed_raw(text: str) -> str | None:
    """Content of the last \\boxed{...} before any normalisation (extract_answer's own brace matching), or None."""
    if "boxed" not in text:
        return None
    ans = text.split("boxed")[-1]
    if not ans:
        return ""
    if ans[0] != "{":
        return ans.split("$")[0].strip()
    depth, out = 1, []
    for c in ans[1:]:
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(c)
    return "".join(out)


def _split_top(s: str, sep: str) -> list[str]:
    """Split on sep at bracket depth 0 ({}, (), [])."""
    parts, depth, cur = [], 0, []
    for c in s:
        if c in "{([":
            depth += 1
        elif c in "})]":
            depth -= 1
        if c == sep and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(c)
    parts.append("".join(cur))
    return parts


_MATRIX_RE = re.compile(r"\\begin\{(pmatrix|bmatrix|matrix|array)\}(\{[^}]*\})?(.*)\\end\{\1\}", re.S)


def vector_entries(raw: str):
    """list[str] for a vector/tuple, list[list[str]] for a matrix with >= 2 rows and >= 2 columns, else None.
    Entries are stripped once (strip_answer_string)."""
    if raw is None:
        return None
    s = re.sub(r"\n\s*", "", str(raw)).strip().rstrip(".").strip()
    for t in ("\\left", "\\right", "\\!", "\\,", "\\;", "$"):
        s = s.replace(t, "")
    s = s.strip()
    m = _MATRIX_RE.fullmatch(s)
    if m:
        rows = [r for r in re.split(r"\\\\", m.group(3)) if r.strip()]
        grid = [[strip_answer_string(x) for x in r.split("&")] for r in rows]
        if not grid:
            return None
        if len(grid) > 1 and all(len(r) > 1 for r in grid):
            return grid
        return [x for r in grid for x in r]
    if len(s) >= 2 and s[0] == "(" and s[-1] == ")":
        depth = 0
        for k, c in enumerate(s):                     # the opening "(" must close at the last character
            depth += c in "{(["
            depth -= c in "})]"
            if depth == 0 and k < len(s) - 1:
                return None
        parts = _split_top(s[1:-1], ",")
        if len(parts) >= 2:
            return [strip_answer_string(x) for x in parts]
    return None


_PREFIX_KEYS = ["x=", "y=", "z=", "x\\in", "y\\in", "z\\in", "x\\to", "y\\to", "z\\to"]   # strip_answer_string's own list


def _drop_prefix_keys(p):
    """What the released second strip did usefully: drop these keys, which the first strip misses while spaces remain
    ('x \\in [-2,7]'). Applied on the scalar path to gold, prediction and N alike; never touches row separators."""
    if isinstance(p, str):
        for k in _PREFIX_KEYS:
            p = p.replace(k, "")
    return p


def _tuple_from_scalar(p):
    """A stripped scalar that is a parenthesised tuple (e.g. 'x \\in (2, \\infty)' after the strip removed 'x\\in')
    becomes a vector too, so gold and prediction meet on one side. No row separators here, so no second strip."""
    if isinstance(p, str) and len(p) >= 2 and p[0] == "(" and p[-1] == ")":
        depth = 0
        for k, c in enumerate(p):
            depth += c in "{(["
            depth -= c in "})]"
            if depth == 0 and k < len(p) - 1:
                return p
        parts = _split_top(p[1:-1], ",")
        if len(parts) >= 2 and all(parts):
            return parts
    return p


def canon_pred(source: str, text: str):
    p = extract_answer(text)
    if source == "gsm8k":
        return MU.sanitize_answer(p)
    v = vector_entries(boxed_raw(text))
    return v if v is not None else _tuple_from_scalar(_drop_prefix_keys(p))


def canon_gold(source: str, original_answer: str):
    if source == "gsm8k":
        return MU.extract_gsm8k_gt_answer(original_answer)
    v = vector_entries(original_answer)
    return v if v is not None else _tuple_from_scalar(_drop_prefix_keys(
        extract_answer("\\boxed{" + original_answer + "}")))


def gold_answer(source: str, original_answer: str):
    if source not in ("gsm8k", "math500"):
        raise ValueError(source)
    return canon_gold(source, original_answer)


def pred_answer(source: str, text: str):
    return _timed(canon_pred, source, text, default="")


def _equal(a, b) -> bool:
    if isinstance(a, list) or isinstance(b, list):
        if not (isinstance(a, list) and isinstance(b, list)) or len(a) != len(b):
            return False
        return all(_equal(x, y) for x, y in zip(a, b))
    if not a or not b or a == "[invalid]" or b == "[invalid]":
        return False
    return bool(math_equal(a, b))


def _correct(source: str, original_answer: str, text: str) -> bool:
    if source == "gsm8k":
        return bool(MU.score_response(source, original_answer, text))
    return _equal(canon_pred(source, text), canon_gold(source, original_answer))


def is_correct(source: str, original_answer: str, text: str) -> bool:
    """gsm8k: the authors' score_response; math500: the Stage 0 normaliser. 5 s timeout -> False."""
    return bool(_timed(_correct, source, original_answer, text, default=False))


def is_correct_authors(source: str, original_answer: str, text: str) -> bool:
    """The authors' score_response as released (math500 misses matrix/tuple forms), for comparison only."""
    return bool(_timed(MU.score_response, source, original_answer, text, default=False))


def _same(source: str, a, b) -> bool:
    if source == "gsm8k":
        return bool(a) and bool(b) and a != "[invalid]" and b != "[invalid]" and a == b
    return _equal(a, b)


def same_answer(source: str, a, b) -> bool:
    return bool(_timed(_same, source, a, b, default=False))


def valid_pred(p) -> bool:
    if isinstance(p, list):
        return bool(p) and all(valid_pred(x) for x in p)
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
    # TODO(coder build): Qwen3-1.7B sometimes never closes </think> (PROPOSAL "Found while building" item 5); then the
    # reasoning is the text before the first \boxed, not the whole output.
    thinking = text.split("</think>", 1)[0]
    return bool(_DOUBT_RE.search(mask_injected(thinking, injected, update)))
