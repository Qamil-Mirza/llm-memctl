#!/usr/bin/env python
"""Stage 0 held-out readings per source and pooled, with the timeout ledger (Advisor GOs, 2026-11-09).

    .venv/bin/python selfdoubt/stage0_heldout_readings.py --model 1.7B --heldout-confirmed   # MATH-500 + pooled
    .venv/bin/python selfdoubt/stage0_heldout_readings.py --model 8B --heldout-confirmed     # GSM8K, MATH-500 100, pooled

Frozen scorer rules (1baf905) with the _timed fix (a timed-out worker is killed; rule unchanged: 5 s -> not equal).
Per source: R0.1, R0.2 (0.9 vs X0 primary, 0.5 secondary), R0.4 from stage0_report.readings (item bootstrap, 95%,
three-way, 30-failure floor per reading; R0.3 left out), the descriptive table per arm x cut, U0 drops by cut
(N == gold; N invalid, which includes N rows at max tokens with no answer), N != FULL by cut, every max-token record by
arm x cut with its category, R0.1's truncated-as-failure sensitivity (inside readings), and the timeout ledger
(selfdoubt/ledger.py: rows resting on a 5 s timeout by source x arm x cut, rescored at 60 s as a SENSITIVITY; the
readings are recomputed on the rescored rows only if a category moves).
Pooled: the same readings with the item bootstrap STRATIFIED by source (stage0_report.bootstrap swapped while
stage0_report.readings runs).
Prints counts and readings only (no text). Per-row categories go next to each store (local, runs/ is gitignored);
a tables-only copy goes to selfdoubt/results/ (committed).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from selfdoubt import common as C  # noqa: E402
from selfdoubt import ledger as LG  # noqa: E402
from selfdoubt import stage0_report as R  # noqa: E402

EXPECTED = [(a, c) for c in C.CUTS for a in ("N", "U0", "UG")] + [(a, C.X0) for a in ("U0", "UG")]
CONFIG = {
    # source -> (store relative to C.RUNS, expected items, write per-row file?)
    "1.7B": {"model": "Qwen/Qwen3-1.7B", "results": "stage0_heldout_1p7b_math500.json",
             "label": "1.7B", "sources": {
                 # GSM8K rows are already reported (stage0_heldout_1p7b.py); rescored here for the ledger and the pool,
                 # and its per-row file is not rewritten.
                 "gsm8k": ("full/Qwen3-1.7B/gen.jsonl", 250, False),
                 "math500": ("s0c_pod1/heldout/full/Qwen3-1.7B/gen.jsonl", 250, True)}},
    "8B": {"model": "Qwen/Qwen3-8B", "results": "stage0_heldout_8b.json", "label": "8B", "sources": {
        "gsm8k": ("s0c_pod2/heldout/full/Qwen3-8B/gen.jsonl", 250, True),
        "math500": ("s0c_pod2/heldout/full/Qwen3-8B/gen.jsonl", 100, True)}},
}


def stratified_bootstrap(source_of: dict):
    """A drop-in for stage0_report.bootstrap that resamples items within each source stratum."""
    def boot(groups: dict, stat, level: float, n_boot: int, seed: int = R.BOOT_SEED):
        ids = sorted(groups)
        point = stat([r for i in ids for r in groups[i]])
        if not ids or point is None:
            return point, None, None
        strata = {}
        for i in ids:
            strata.setdefault(source_of[i], []).append(i)
        rng = np.random.default_rng(seed)
        vals = []
        for _ in range(n_boot):
            rs = []
            for s in sorted(strata):
                v = strata[s]
                for k in rng.integers(0, len(v), len(v)):
                    rs += groups[v[k]]
            x = stat(rs)
            if x is not None:
                vals.append(x)
        if not vals:
            return point, None, None
        a = (1 - level) / 2
        return point, float(np.quantile(vals, a)), float(np.quantile(vals, 1 - a))
    return boot


def readings(rows, n_boot, source_of=None):
    if source_of is None:
        rd = R.readings(rows, n_boot)
    else:
        orig = R.bootstrap
        R.bootstrap = stratified_bootstrap(source_of)
        try:
            rd = R.readings(rows, n_boot)
        finally:
            R.bootstrap = orig
    rd.pop("R0.3", None)
    return rd


def describe(rows: list[dict]) -> dict:
    u0 = lambda c, cat: sum(1 for r in rows if r["arm"] == "U0" and r["cut"] == c and r["category"] == cat)  # noqa
    return {"table": R.table(rows),
            "u0_drops_by_cut": {str(c): {"N_is_gold": u0(c, "dropped_ref_is_gold"),
                                         "N_invalid": u0(c, "dropped_ref_invalid")} for c in C.CUTS},
            "n_differs_from_full_by_cut": {str(c): sum(1 for r in rows if r["arm"] == "N" and r["cut"] == c
                                                       and r["n_differs_from_full"]) for c in C.CUTS}}


def hit_max(recs, rows) -> dict:
    cat = {(r["item_id"], r["arm"], r["cut"]): r["category"] for r in rows}
    hit = [r for r in recs if r["hit_max"]]
    cell = lambda r: f"{r['arm']}@{r['cut'] if r['cut'] is not None else '-'}"  # noqa
    return {"n_records": len(hit), "by_arm_cut": dict(sorted(Counter(cell(r) for r in hit).items())),
            "by_arm_cut_category": dict(sorted(Counter(
                f"{cell(r)}|{cat.get((r['item_id'], r['arm'], r['cut']), 'FULL')}" for r in hit).items()))}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(CONFIG), required=True)
    ap.add_argument("--heldout-confirmed", action="store_true")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out-root", default=str(C.RUNS))
    a = ap.parse_args(argv)
    if not a.heldout_confirmed:
        raise SystemExit("held-out readings need --heldout-confirmed")
    cfg = CONFIG[a.model]
    root = Path(a.out_root)
    items_all = C.load_items()
    split, sha = C.ensure_split(items_all, root / "split.json")
    items = {int(r["id"]): r for r in items_all}

    out = {"rule": f"Advisor GO 2026-11-09: {cfg['label']} held-out Stage 0 readings per source and pooled (item "
                   "bootstrap stratified by source); scorer rules frozen at 1baf905, _timed fix applied; timeout "
                   f"ledger with a {LG.SENS_TIMEOUT_S} s sensitivity rescore",
           "model": cfg["model"], "split_sha256": sha, "provenance": C.provenance(), "n_boot": a.n_boot,
           "stores": {s: v[0] for s, v in cfg["sources"].items()}}
    all_rows, all_sens, any_move = [], [], False
    for src, (rel, n_exp, write_rows) in cfg["sources"].items():
        recs = [r for r in C.Store(root / rel).recs.values() if r["source"] == src]
        assert recs and all(r["model"] == cfg["model"] for r in recs), f"{src}: wrong model"
        ho = set(split["heldout"][src])
        assert all(r["item_id"] in ho for r in recs), f"non-held-out {src} record"
        full = {r["item_id"] for r in recs if r["arm"] == "FULL"}
        got = Counter(r["item_id"] for r in recs if r["arm"] != "FULL")
        keys = {(r["item_id"], r["arm"], r["cut"]) for r in recs if r["arm"] != "FULL"}
        assert len(full) == n_exp and all(got[i] == 11 for i in full) and len(got) == n_exp, f"{src} not complete"
        assert all((i, arm, c) in keys for i in full for arm, c in EXPECTED), f"{src} missing (arm, cut) rows"
        rows, sens, led = LG.score_with_ledger(recs, items)
        moved = bool(led["category_moves_sensitivity"])
        any_move |= moved
        sec = {"label": f"{cfg['label']}, {src} only", "n_items": n_exp, "readings": readings(rows, a.n_boot),
               "readings_sensitivity_60s": readings(sens, a.n_boot) if moved else "no category moves: identical",
               **describe(rows), "hit_max": hit_max(recs, rows), "timeout_ledger": led}
        out[src] = sec
        all_rows += rows
        all_sens += sens
        if write_rows:
            scored = root / rel.replace("gen.jsonl", f"scored_{src}_heldout.jsonl")
            scored.write_text("".join(json.dumps(r, default=str) + "\n" for r in rows))
            print(f"rows: {scored}")
    source_of = {r["item_id"]: r["source"] for r in all_rows}
    out["pooled"] = {"label": f"{cfg['label']}, pooled GSM8K + MATH-500 (bootstrap stratified by source)",
                     "n_items": len(source_of), "readings": readings(all_rows, a.n_boot, source_of),
                     "readings_sensitivity_60s": (readings(all_sens, a.n_boot, source_of) if any_move
                                                  else "no category moves: identical"),
                     "table": R.table(all_rows)}
    rec = Path(__file__).resolve().parent / "results" / cfg["results"]
    rec.write_text(json.dumps(out, indent=1, default=str) + "\n")
    for k in list(cfg["sources"]) + ["pooled"]:
        d = out[k]
        print(f"== {d['label']}: {d['n_items']} items (split sha {sha[:12]})")
        for t, v in d["table"].items():
            print(f"   {t:8s} {v}")
        for t in ("u0_drops_by_cut", "n_differs_from_full_by_cut", "hit_max", "timeout_ledger"):
            if t in d:
                print(f"   {t}: {d[t]}")
        for t, v in d["readings"].items():
            print(f"   {t}: {v}")
        if isinstance(d["readings_sensitivity_60s"], dict):
            for t, v in d["readings_sensitivity_60s"].items():
                print(f"   [60 s] {t}: {v}")
        else:
            print(f"   [60 s] {d['readings_sensitivity_60s']}")
    print(f"tables: {rec}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
