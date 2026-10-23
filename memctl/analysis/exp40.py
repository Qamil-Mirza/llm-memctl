"""§40: the fixed layer-29 intervention on LoCoMo, a set it has never seen. Pure logic (numpy only); the pod runner is
scripts/exp40_pod_run.py, the report scripts/exp40_report.py.

Pre-registration: research/proposals/2026-10-23-exp40-fixed-setting-validation.md (rev 0; rev 1 at 877f86d), the
Advisor's binding ruling on R/C selection (2026-10-23, after rev 1), D-34; §41 rev 1 for the conditional rule.

- The setting is fixed: layer 29, alpha 2 (unit ||v_29||), v_29 byte-exact from runs/exp38_dev/qwen3b/directions.npz.
  `check_setting` refuses any other (layer, alpha); there is no sweep.
- Settings: Steer-all (every step), Steer-0 (step 0 only, descriptive), Random (a seeded Gaussian direction, rng
  [40, 29], rescaled to ||2 v_29||, one vector for every row), none (A, B, A-cut).
- Selection (the ruling): on the 600 rows, after the N pass, the A pass and the judge on every marked A answer:
  C = the first 40 rows in shuffle order whose A answer is judged correct (verdict yes and class 'answer');
  R = the first 120 marked rows in shuffle order whose A answer is a refusal (the §23 rule on the final answer, as §38).
  Both use only the A outputs and the judge, before any steered arm.
- Smoke (D-22; rev 2, 61f9f6b): batched against single-sequence notes on 5 rows: the same first generated token on
  >= 4 of 5 and the same marker outcome (batched = single) on >= 4 of 5, else abort; the absolute marker rate is
  printed, not gated (the R < 40 rule is the only population guard); every judge reply a yes/no verdict, else abort.
- Judge agreement (the ruling): the 7B re-scores the 60 frozen cached §35 rows; agreement < 0.90 stops the run before
  any arm (exit EXIT_AGREEMENT; the pod script then stops the pod).
"""
from __future__ import annotations

import numpy as np

from memctl.analysis import steering as st

LAYER, ALPHA, UNIT = 29, 2.0, "vnorm"
RANDOM_SEED = 40
MAX_NEW = st.MAX_NEW_TOKENS_POD             # 64, as the §38 pod confirmation it validates (build review 456dd11)
NOTE_MAX_NEW = 256
NOTE_BATCH = 32
R_CAP, C_CAP = 120, 40
R_MIN_FOR_CLAIMS = 40
SMOKE_N, SMOKE_MIN = 5, 4
AGREEMENT_N = 60
AGREEMENT_BAR = st.JUDGE_AGREEMENT_BAR    # 0.90
EXIT_SETTING, EXIT_INPUT, EXIT_SMOKE, EXIT_AGREEMENT = 2, 3, 5, 7
# byte-exact inputs (the §38 dev direction; the §39/§41 lineage)
SOURCE_NPZ_SHA256 = "be1e3a09bdb5157d24126ebe16b14c035d429ebd51409e9b615b347c5b29f921"
V29_SHA256 = "a7f967ecdf2f59a9e1f64d99e366d33011f27daa8b09b18f5ff33d1e1ba20c6c"
STEER_VECTOR_SHA256 = "e0000366813f3c94cf4875c5ba948fc7f4bac220099cc8848edb79196045d71c"   # 2 * v_29
# §41's frozen refusal-start id set (73c3371 review: [9788] = ' unknown'), used for the logged margin when no rule
# file is shipped; a shipped rule's own `refusal_start_ids` take precedence.
DEFAULT_REFUSAL_IDS = (9788,)

STEER_ALL = st.Setting("steer", LAYER, ALPHA, UNIT)
STEER_0 = st.Setting("steer", LAYER, ALPHA, UNIT, steps=(0,))
RANDOM = st.Setting("random", LAYER, ALPHA, UNIT)
NONE = st.Setting("none")
# the order arms run in (primary claim first); (name, pop, kind, settings)
ARM_ORDER = (
    ("R:A:steer_all+random+steer0", "R", "A", ("steer_all", "random", "steer0")),
    ("R:Ncut notes", "R", "ncut", ()),
    ("R:cut:none+steer_all", "R", "cut", ("none", "steer_all")),
    ("C:A:steer_all", "C", "A", ("steer_all",)),
    ("R:B:none", "R", "B", ("none",)),
    ("adv:B:none", "adv", "B", ("none",)),
    ("adv:A:steer_all", "adv", "A", ("steer_all",)),
    ("C:Ncut notes", "C", "ncut", ()),
    ("C:cut:none", "C", "cut", ("none",)),
    ("adv:Ncut notes", "adv", "ncut", ()),
    ("adv:cut:none", "adv", "cut", ("none",)),
)
PRIMARY_ARMS = 4      # the first four entries carry Claim 1, Control X and Control C
WORDING = {"setting": "layer 29, alpha 2 (unit ||v_29||), v_29 of runs/exp38_dev/qwen3b/directions.npz byte-exact",
           "random": "rng [40, 29] Gaussian, norm ||2 v_29||, one vector", "max_new_tokens": MAX_NEW,
           "note": "the §23 fixed8 template on LoCoMo, 256 new tokens, greedy, batched 32 left-padded",
           "A": "note (split_note of the stripped N output, byte-exact) + 'Answer:'",
           "B": "note + newline + B_INSTRUCTION + newline + 'Answer:'",
           "cut": "ablate_evidence(prompt) + its own N-cut note (no marker -> no note) + 'Answer:'",
           "R": "first 120 marked A refusals (§23 rule) in shuffle order", "C": "first 40 marked A answers judged correct",
           "classifier": st.CLASSIFIER_VERSION}


def setting_of(name: str, layer: int = LAYER) -> st.Setting:
    """'steer_all' / 'steer0' / 'random' / 'none' at the engine's layer 29 (the stub maps it)."""
    if name == "none":
        return NONE
    base = {"steer_all": STEER_ALL, "steer0": STEER_0, "random": RANDOM}[name]
    return st.Setting(base.mode, layer, base.alpha, base.unit, base.steps)


def check_setting(layer, alpha) -> None:
    """The run accepts exactly (29, 2); anything else is refused (no sweep exists)."""
    if int(layer) != LAYER or float(alpha) != ALPHA or float(layer) != float(int(layer)):
        raise ValueError(f"REFUSED: §40 runs only layer {LAYER}, alpha {ALPHA:g}; got layer {layer}, alpha {alpha}")


def check_direction(v: np.ndarray, npz_sha: str | None = None) -> dict:
    """v_29 byte-exact; the steer vector 2 * v_29 byte-exact; the seed-40 random vector norm-matched."""
    if npz_sha is not None and npz_sha != SOURCE_NPZ_SHA256:
        raise ValueError(f"REFUSED: directions.npz sha {npz_sha}, expected {SOURCE_NPZ_SHA256}")
    if v.dtype != np.float32 or st.vector_sha(v) != V29_SHA256:
        raise ValueError(f"REFUSED: v_29 sha {st.vector_sha(v)} ({v.dtype}), expected {V29_SHA256}")
    steer = st.scaled_vector(v, ALPHA, UNIT)
    if not np.array_equal(steer, (2.0 * v.astype(np.float64)).astype(np.float32)) or \
            st.vector_sha(steer) != STEER_VECTOR_SHA256:
        raise ValueError(f"REFUSED: steer vector sha {st.vector_sha(steer)}, expected {STEER_VECTOR_SHA256}")
    rnd = st.random_vector(steer, LAYER, seed=RANDOM_SEED)
    return {"v": v, "steer": steer, "random": rnd, "v_sha256": V29_SHA256, "steer_sha256": STEER_VECTOR_SHA256,
            "random_sha256": st.vector_sha(rnd), "v_norm": float(np.linalg.norm(v.astype(np.float64))),
            "steer_norm": float(np.linalg.norm(steer.astype(np.float64))),
            "random_norm": float(np.linalg.norm(rnd.astype(np.float64)))}


def cosine(a, b) -> float:
    a64, b64 = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(a64 @ b64 / (np.linalg.norm(a64) * np.linalg.norm(b64)))


# --- selection ----------------------------------------------------------------------------------------------------

def select(order: list[str], a_rows: dict[str, dict], verdict: dict[str, bool | None], r_cap: int = R_CAP,
           c_cap: int = C_CAP) -> dict:
    """`order`: the 600 in shuffle order; `a_rows[q]`: the A-pass row of a MARKED row (unmarked rows are absent);
    `verdict[q]`: the judge's verdict on that row's A answer (None: no verdict). R = first r_cap refusals; C = first
    c_cap rows with verdict yes and class 'answer'. Disjoint by construction (a refusal is never class 'answer')."""
    R = [q for q in order if q in a_rows and a_rows[q]["refusal"]][:r_cap]
    C = [q for q in order if q in a_rows and not a_rows[q]["refusal"] and a_rows[q]["cls"] == "answer"
         and verdict.get(q) is True][:c_cap]
    return {"R": R, "C": C}


def pass_counts(order: list[str], marked: set[str], a_rows: dict[str, dict], verdict: dict[str, bool | None]) -> dict:
    """The ruling's counts on the 600: marker rate, refusal rate, judged-correct rate (over the 600 and over the
    marked rows)."""
    n = len(order)
    m = [q for q in order if q in marked]
    ref = [q for q in m if a_rows.get(q, {}).get("refusal")]
    cor = [q for q in m if verdict.get(q) is True and a_rows.get(q, {}).get("cls") == "answer"]
    nov = [q for q in m if q in a_rows and verdict.get(q) is None]
    return {"n": n, "marked": len(m), "marker_rate": len(m) / n if n else None,
            "refusals": len(ref), "refusal_rate": len(ref) / n if n else None,
            "refusal_rate_of_marked": len(ref) / len(m) if m else None,
            "judged_correct": len(cor), "judged_correct_rate": len(cor) / n if n else None,
            "judged_correct_rate_of_marked": len(cor) / len(m) if m else None, "no_verdict": len(nov)}


# --- the smoke ----------------------------------------------------------------------------------------------------

def smoke_notes_check(batched: list[dict], single: list[dict], need: int = SMOKE_MIN) -> dict:
    """Rows {'first_id', 'marker', 'output'}; batching drift only (rev 2): the same first id on >= need and the same
    marker outcome (batched = single) on >= need. The absolute marker rate is returned and printed, never gated."""
    same = sum(b["first_id"] is not None and b["first_id"] == s["first_id"] for b, s in zip(batched, single))
    marker = sum(bool(b["marker"]) for b in batched)
    agree = sum(bool(b["marker"]) == bool(s["marker"]) for b, s in zip(batched, single))
    exact = sum(b["output"] == s["output"] for b, s in zip(batched, single))
    problems = []
    if same < need:
        problems.append(f"batched vs single-sequence notes: same first token on {same}/{len(batched)} < {need}")
    if agree < need:
        problems.append(f"batched vs single-sequence notes: same marker outcome on {agree}/{len(batched)} < {need}")
    return {"same_first": same, "marker_agree": agree, "marker_batched": marker,
            "marker_single": sum(bool(s["marker"]) for s in single), "exact_text": exact, "n": len(batched),
            "need": need, "problems": problems, "pass": not problems}


def agreement(replies: list[str], cached: list[bool], bar: float = AGREEMENT_BAR) -> dict:
    """A reply that is not a yes/no verdict counts as a disagreement."""
    verdicts = [st.parse_verdict(r) for r in replies]
    agree = sum(v is not None and v == c for v, c in zip(verdicts, cached))
    n = len(cached)
    rate = agree / n if n else None
    return {"n": n, "agree": agree, "rate": rate, "bar": bar, "unparseable": sum(v is None for v in verdicts),
            "pass": rate is not None and rate >= bar}
