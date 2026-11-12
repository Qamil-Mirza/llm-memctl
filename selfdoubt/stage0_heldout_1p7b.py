#!/usr/bin/env python
"""Stage 0 held-out readings for Qwen3-1.7B under the Advisor's rule of 2026-11-09 (after the Stage 0 pod).

    .venv/bin/python selfdoubt/stage0_heldout_1p7b.py --heldout-confirmed

1. Readings: Qwen3-1.7B, GSM8K held-out 250 only ("1.7B, GSM8K only"). R0.1, R0.2 (0.9 primary, 0.5 secondary),
   R0.4 from stage0_report.readings (frozen scorer, item bootstrap, 95%, three-way, 30-failure floor). R0.3 is left
   out (doubt coder not built). Pooled 1.7B readings wait for MATH completion.
2. Descriptive counts on the same rows: categories per arm x cut, N == gold drops, N != FULL rows, max-token rows.
3. MATH-500: no scoring and no text read. An item is COMPLETE only if all 11 continuations finished; incomplete
   items are listed with their missing (arm, cut) rows. A re-run cost estimate uses the measured MATH CONT rate.

Prints counts and readings only (no predictions or text). Writes runs/selfdoubt/full/heldout_1p7b_readings.json,
a tables-only copy at selfdoubt/results/stage0_heldout_1p7b_gsm8k.json (committed), and per-row categories at
runs/selfdoubt/full/Qwen3-1.7B/scored_gsm8k_heldout.jsonl (local only).

Exploratory notes (Advisor, 2026-11-09; not readings): guidance moves stale -> new while "other" barely moves; from
X0 to 0.9, U0 stale rises and other falls with total failures flat (X0's reference is FULL, not N: not like-for-like).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from selfdoubt import common as C  # noqa: E402
from selfdoubt import stage0_report as R  # noqa: E402

MODEL_DIR = "Qwen3-1.7B"
ARMS_AT_CUT = ("N", "U0", "UG")
ARMS_AT_X0 = ("U0", "UG")
EXPECTED = [(a, c) for c in C.CUTS for a in ARMS_AT_CUT] + [(a, C.X0) for a in ARMS_AT_X0]
USD_PER_HOUR = 0.59
MAX_TOKENS = 16384


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--heldout-confirmed", action="store_true")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out-root", default=str(C.RUNS))
    a = ap.parse_args(argv)
    if not a.heldout_confirmed:
        raise SystemExit("held-out readings need --heldout-confirmed")
    assert len(EXPECTED) == 11
    out_root = Path(a.out_root)
    mode_dir = out_root / "full"
    items_all = C.load_items()
    split, sha = C.ensure_split(items_all, out_root / "split.json")
    items = {int(r["id"]): r for r in items_all}
    recs = list(C.Store(mode_dir / MODEL_DIR / "gen.jsonl").recs.values())
    ho = {src: set(split["heldout"][src]) for src in C.SOURCES}
    assert all(r["item_id"] in ho[r["source"]] for r in recs), "non-held-out record in the full run"

    # ---- GSM8K readings -------------------------------------------------------------------------------------
    g = [r for r in recs if r["source"] == "gsm8k"]
    g_full = {r["item_id"] for r in g if r["arm"] == "FULL"}
    have = Counter(r["item_id"] for r in g if r["arm"] != "FULL")
    assert len(g_full) == 250 and all(have[i] == 11 for i in g_full), "GSM8K not complete"
    rows = R.score_model(g, items)
    tab = R.table(rows)
    rd = R.readings(rows, a.n_boot)
    rd.pop("R0.3", None)
    drops = {str(c): sum(1 for r in rows if r["arm"] == "U0" and r["cut"] == c and r["ref_is_gold"]) for c in C.CUTS}
    n_vs_full = {str(c): sum(1 for r in rows if r["arm"] == "N" and r["cut"] == c and r["n_differs_from_full"])
                 for c in C.CUTS}
    # same-row readings restricted to N == FULL rows are not taken; N != FULL rows are listed by count only.
    hit = [{"item_id": r["item_id"], "arm": r["arm"], "cut": r["cut"], "n_gen_tokens": r["n_gen_tokens"]}
           for r in g if r["hit_max"]]
    hit_cat = []
    for h in hit:
        m = [x for x in rows if x["item_id"] == h["item_id"] and x["arm"] == h["arm"] and x["cut"] == h["cut"]]
        hit_cat.append(dict(h, category=m[0]["category"] if m else "FULL"))

    # ---- MATH completeness (keys only) ----------------------------------------------------------------------
    m = [r for r in recs if r["source"] == "math500"]
    m_full = sorted({r["item_id"] for r in m if r["arm"] == "FULL"})
    got = {(r["item_id"], r["arm"], r["cut"]) for r in m if r["arm"] != "FULL"}
    missing = {i: [(arm, c) for arm, c in EXPECTED if (i, arm, c) not in got] for i in m_full}
    incomplete = {i: v for i, v in missing.items() if v}
    miss_by = Counter((arm, c) for v in incomplete.values() for arm, c in v)
    n_missing = sum(len(v) for v in incomplete.values())
    phases = [json.loads(x) for x in (mode_dir / MODEL_DIR / "phases.jsonl").read_text().splitlines() if x]
    mc = [p for p in phases if p["phase"] == "CONT" and p["n_requests"] == 1100][-1]
    rate = mc["gen_tok_per_s"]                         # finished + cut-row tokens over wall: the measured 570
    lo_tok = mc["gen_tokens_partial"]                  # tokens the cut rows had already produced: a floor
    hi_tok = n_missing * MAX_TOKENS                    # every missing row runs to the cap: a ceiling
    setup_s = 5 * 60                                   # pod setup + 1.7B load, measured ~3.5 min on this pod
    cost = lambda tok: round((tok / rate + setup_s) / 3600 * USD_PER_HOUR, 3)  # noqa: E731

    out = {
        "rule": "Advisor 2026-11-09: GSM8K-only 1.7B readings; MATH item counts only if all 11 rows finished",
        "split_sha256": sha, "provenance": C.provenance(), "n_boot": a.n_boot,
        "gsm8k": {"label": "1.7B, GSM8K only", "n_items": len(g_full), "readings": rd, "table": tab,
                  "u0_ref_is_gold_drops_by_cut": drops, "n_differs_from_full_by_cut": n_vs_full,
                  "hit_max_rows": hit_cat},
        "math500": {"n_items_run": len(m_full), "n_complete": len(m_full) - len(incomplete),
                    "n_incomplete": len(incomplete), "n_missing_rows": n_missing,
                    "missing_by_arm_cut": {f"{arm}@{c}": n for (arm, c), n in sorted(miss_by.items())},
                    "incomplete": {str(i): [f"{arm}@{c}" for arm, c in v] for i, v in sorted(incomplete.items())},
                    "rerun_estimate": {"rate_tok_per_s": round(rate, 1), "tokens_floor": lo_tok,
                                       "tokens_ceiling": hi_tok, "setup_s": setup_s,
                                       "usd_floor": cost(lo_tok), "usd_ceiling": cost(hi_tok),
                                       "hours_ceiling": round((hi_tok / rate + setup_s) / 3600, 2)}},
    }
    path = mode_dir / "heldout_1p7b_readings.json"
    path.write_text(json.dumps(out, indent=1, default=str) + "\n")
    # Per-row categories for Stage 2 (stale/new row ids). Held-out, under runs/ (gitignored): local only.
    # "doubt" here is the PLACEHOLDER keyword coder, not a validated label.
    scored = mode_dir / MODEL_DIR / "scored_gsm8k_heldout.jsonl"
    scored.write_text("".join(json.dumps(r, default=str) + "\n" for r in rows))
    # Tables-only copy for the record (runs/ is gitignored).
    rec = Path(__file__).resolve().parent / "results" / "stage0_heldout_1p7b_gsm8k.json"
    rec.parent.mkdir(exist_ok=True)
    rec.write_text(json.dumps(out, indent=1, default=str) + "\n")

    print(f"== Qwen3-1.7B, GSM8K only: {len(g_full)} held-out items (split sha {sha[:12]})")
    for k, v in tab.items():
        print(f"   {k:8s} {v}")
    print(f"   U0 drops (N == gold) by cut: {drops}")
    print(f"   N != FULL by cut: {n_vs_full}")
    print(f"   hit-max rows: {hit_cat}")
    for k, v in rd.items():
        print(f"   {k}: {v}")
    mm = out["math500"]
    print(f"== MATH-500 (no scoring): run {mm['n_items_run']}, complete {mm['n_complete']}, incomplete "
          f"{mm['n_incomplete']} ({mm['n_missing_rows']} rows missing)")
    print(f"   missing by arm@cut: {mm['missing_by_arm_cut']}")
    print(f"   rerun: {mm['rerun_estimate']}")
    print(f"report: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
