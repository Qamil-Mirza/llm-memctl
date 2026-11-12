#!/usr/bin/env python
"""Stage 0 scoring and readings (selfdoubt/PROPOSAL.md, "Scoring" and "Readings, fixed now").

    .venv/bin/python selfdoubt/stage0_report.py --mode smoke            # or stub; dev items only
    .venv/bin/python selfdoubt/stage0_report.py --mode full --split-name heldout --heldout-confirmed

Categories per U0/UG continuation at a cut (X in 0.3/0.5/0.9): new (equals the original gold), stale (equals that
prefix's N answer and not gold), truncated (hit max_tokens with no final answer), other. Item-cuts whose N answer
equals the gold (or whose N is truncated / has no answer) are dropped. At X = 0 there is no prefix and no N; the FULL
answer is the stale reference for the descriptive table only.

Readings (per model; three-way SHOWN / NOT SHOWN / UNDETERMINED from item-level bootstrap intervals; no reading with
fewer than 30 U0 failures in the pool):
  R0.1  stale / (stale + other) over U0 rows pooled over cuts            > 0.5      (95%)
  R0.2  U0 failure rate at 0.9 minus at X = 0, same items (0.9 pool)     > 0        (95%); 0.5 secondary
  R0.3  doubt rate in U0 failures minus successes, pooled over cuts      in +-0.10  (90%, equivalence) PLACEHOLDER
  R0.4  stale rate UG minus U0 (stale / non-truncated pool rows), paired < 0        (95%)
Failure = stale or other (truncated rows are reported separately and excluded from rates; a sensitivity line counts
them as failures).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from selfdoubt import common as C  # noqa: E402
from selfdoubt import scoring as S  # noqa: E402

MIN_FAILURES = 30
BOOT_SEED = 56


def verdict(lo, hi, bar, direction):
    if lo is None or hi is None:
        return "UNDETERMINED"
    if direction == "gt":
        return "SHOWN" if lo > bar else ("NOT SHOWN" if hi < bar else "UNDETERMINED")
    if direction == "lt":
        return "SHOWN" if hi < bar else ("NOT SHOWN" if lo > bar else "UNDETERMINED")
    if direction == "within":
        return "SHOWN" if (lo >= -bar and hi <= bar) else ("NOT SHOWN" if (hi < -bar or lo > bar) else "UNDETERMINED")
    raise ValueError(direction)


def bootstrap(groups: dict, stat, level: float, n_boot: int, seed: int = BOOT_SEED):
    """Percentile interval over items: resample item ids with replacement, pool their rows, recompute stat."""
    ids = sorted(groups)
    point = stat([r for i in ids for r in groups[i]])
    if not ids or point is None:
        return point, None, None
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(ids), len(ids))
        v = stat([r for k in pick for r in groups[ids[k]]])
        if v is not None:
            vals.append(v)
    if not vals:
        return point, None, None
    a = (1 - level) / 2
    return point, float(np.quantile(vals, a)), float(np.quantile(vals, 1 - a))


def _rate(num, den):
    return None if den == 0 else num / den


def score_model(recs: list[dict], items: dict[int, dict]) -> list[dict]:
    """One scored row per non-FULL record."""
    full = {r["item_id"]: r for r in recs if r["arm"] == "FULL"}
    nrec = {(r["item_id"], r["cut"]): r for r in recs if r["arm"] == "N"}
    cache = {}

    def info(r):
        k = r["key"]
        if k not in cache:
            it = items[r["item_id"]]
            p = S.pred_answer(r["source"], r["text"])
            tr = S.is_truncated(r)
            cache[k] = {"pred": p, "truncated": tr, "valid": (not tr) and S.valid_pred(p),
                        "correct": S.is_correct(r["source"], it["original_answer"], r["text"])}
        return cache[k]

    rows = []
    for r in recs:
        if r["arm"] == "FULL":
            continue
        it = items[r["item_id"]]
        me = info(r)
        fr = full.get(r["item_id"])
        fi = info(fr) if fr else None
        row = {"item_id": r["item_id"], "source": r["source"], "arm": r["arm"], "cut": r["cut"], **me,
               "gold": S.gold_answer(r["source"], it["original_answer"])}
        if r["arm"] == "N":
            row["n_differs_from_full"] = bool(fi and fi["valid"] and me["valid"]
                                              and not S.same_answer(r["source"], me["pred"], fi["pred"]))
            row["category"] = "N"
        else:
            ref = nrec.get((r["item_id"], r["cut"])) if r["cut"] != C.X0 else fr
            ri = info(ref) if ref else None
            n_valid = bool(ri and ri["valid"])
            row["ref"] = "N" if r["cut"] != C.X0 else "FULL"
            row["ref_pred"] = ri["pred"] if ri else None
            row["ref_is_gold"] = bool(ri and ri["correct"])
            stale = n_valid and me["valid"] and (not me["correct"]) and S.same_answer(r["source"], me["pred"],
                                                                                       ri["pred"])
            row["stale"] = bool(stale)
            row["category"] = S.categorize(n_valid=n_valid, n_is_gold=row["ref_is_gold"], truncated=me["truncated"],
                                           correct=me["correct"], stale=row["stale"])
            row["doubt"] = S.doubt_flag(r["text"], r.get("injected", ""), it["update"])
        rows.append(row)
    return rows


def readings(rows: list[dict], n_boot: int) -> dict:
    by = defaultdict(list)
    for r in rows:
        by[(r["arm"], r["cut"])].append(r)
    cuts = list(C.CUTS)
    pool = lambda r: r["category"] in ("new", "stale", "other")          # noqa: E731
    fail = lambda r: r["category"] in ("stale", "other")                  # noqa: E731
    u0 = [r for r in rows if r["arm"] == "U0" and r["cut"] in cuts]
    n_fail = sum(fail(r) for r in u0)
    out = {"n_u0_failures_pooled": n_fail, "min_failures": MIN_FAILURES}

    def groups(rs):
        g = defaultdict(list)
        for r in rs:
            g[r["item_id"]].append(r)
        return g

    def pack(name, point, lo, hi, level, bar, direction, extra=None):
        d = {"point": point, "ci": [lo, hi], "level": level, "bar": bar, "direction": direction,
             "verdict": verdict(lo, hi, bar, direction)}
        if n_fail < MIN_FAILURES:
            d["verdict"] = f"NO READING (U0 failures {n_fail} < {MIN_FAILURES})"
        d.update(extra or {})
        out[name] = d

    # R0.1
    def s01(rs):
        f = [r for r in rs if fail(r)]
        return _rate(sum(r["category"] == "stale" for r in f), len(f))
    pack("R0.1", *bootstrap(groups(u0), s01, 0.95, n_boot), 0.95, 0.5, "gt")

    # R0.2: same items: those in the U0 pool at cut c (N valid, N != gold, U0 not truncated) and with a
    # non-truncated X = 0 U0 row.
    def r02(c):
        at_c = {r["item_id"]: r for r in by[("U0", c)] if pool(r)}
        at_0 = {r["item_id"]: r for r in by[("U0", C.X0)] if not r["truncated"]}
        ids = sorted(set(at_c) & set(at_0))
        g = {i: [dict(at_c[i], _t="c"), dict(at_0[i], _t="0")] for i in ids}

        def s(rs):
            a = [r for r in rs if r["_t"] == "c"]
            b = [r for r in rs if r["_t"] == "0"]
            if not a or not b:
                return None
            return sum(fail(r) for r in a) / len(a) - sum(not r["correct"] for r in b) / len(b)
        return g, s
    g, s = r02(0.9)
    pack("R0.2", *bootstrap(g, s, 0.95, n_boot), 0.95, 0.0, "gt", {"n_items": len(g)})
    g, s = r02(0.5)
    pack("R0.2_secondary_cut0.5", *bootstrap(g, s, 0.95, n_boot), 0.95, 0.0, "gt", {"n_items": len(g)})

    # R0.3 (placeholder coder)
    def s03(rs):
        f = [r for r in rs if fail(r)]
        ok = [r for r in rs if r["category"] == "new"]
        if not f or not ok:
            return None
        return sum(r["doubt"] for r in f) / len(f) - sum(r["doubt"] for r in ok) / len(ok)
    pack("R0.3", *bootstrap(groups([r for r in u0 if pool(r)]), s03, 0.90, n_boot), 0.90, 0.10, "within",
         {"coder": "PLACEHOLDER keyword rules (LLM judge TODO); not a reading until the coder is validated on dev"})

    # R0.4: paired item-cuts where both U0 and UG are in the pool.
    pairs = defaultdict(list)
    for c in cuts:
        a = {r["item_id"]: r for r in by[("U0", c)] if pool(r)}
        b = {r["item_id"]: r for r in by[("UG", c)] if pool(r)}
        for i in set(a) & set(b):
            pairs[i] += [a[i], b[i]]

    def s04(rs):
        a = [r for r in rs if r["arm"] == "U0"]
        b = [r for r in rs if r["arm"] == "UG"]
        if not a:
            return None
        return sum(r["category"] == "stale" for r in b) / len(b) - sum(r["category"] == "stale" for r in a) / len(a)
    pack("R0.4", *bootstrap(pairs, s04, 0.95, n_boot), 0.95, 0.0, "lt",
         {"n_item_cuts": sum(len(v) for v in pairs.values()) // 2})

    # sensitivity: truncated counted as failure for R0.1's denominator
    def s01t(rs):
        f = [r for r in rs if r["category"] in ("stale", "other", "truncated")]
        return _rate(sum(r["category"] == "stale" for r in f), len(f))
    p, lo, hi = bootstrap(groups(u0), s01t, 0.95, n_boot)
    out["R0.1_sensitivity_truncated_as_failure"] = {"point": p, "ci": [lo, hi]}
    return out


def table(rows: list[dict]) -> dict:
    t = {}
    for arm in ("U0", "UG"):
        for c in list(C.CUTS) + [C.X0]:
            rs = [r for r in rows if r["arm"] == arm and r["cut"] == c]
            t[f"{arm}@{c}"] = dict(Counter(r["category"] for r in rs), n=len(rs))
    for c in C.CUTS:
        rs = [r for r in rows if r["arm"] == "N" and r["cut"] == c]
        t[f"N@{c}"] = {"n": len(rs), "truncated": sum(r["truncated"] for r in rs),
                       "no_answer": sum(not r["valid"] for r in rs), "is_gold": sum(r["correct"] for r in rs),
                       "differs_from_full": sum(r["n_differs_from_full"] for r in rs)}
    return t


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "stub", "full"], required=True)
    ap.add_argument("--split-name", choices=["dev", "heldout"], default="dev")
    ap.add_argument("--heldout-confirmed", action="store_true",
                    help="readings on held-out; only after dev choices (normaliser, coder, cap) are frozen")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out-root", default=str(C.RUNS))
    a = ap.parse_args(argv)
    if a.split_name == "heldout" and not a.heldout_confirmed:
        raise SystemExit("held-out readings need --heldout-confirmed")
    out_root = Path(a.out_root)
    items_all = C.load_items()
    split, sha = C.ensure_split(items_all, out_root / "split.json")
    keep = {i for src in C.SOURCES for i in split[a.split_name][src]}
    items = {int(r["id"]): r for r in items_all}
    mode_dir = out_root / a.mode
    report = {"mode": a.mode, "split_name": a.split_name, "split_sha256": sha, "n_boot": a.n_boot,
              "provenance": C.provenance(), "models": {}}
    for md in sorted(p for p in mode_dir.iterdir() if p.is_dir() and (p / "gen.jsonl").exists()):
        recs = [r for r in C.Store(md / "gen.jsonl").recs.values() if r["item_id"] in keep]
        full = [r for r in recs if r["arm"] == "FULL"]
        rows = score_model(recs, items)
        fi = [{"item_id": r["item_id"], "truncated": S.is_truncated(r), "hit_max": r["hit_max"],
               "pred": S.pred_answer(r["source"], r["text"]), "n_gen_tokens": r["n_gen_tokens"],
               "trace_tokens": r.get("total_reasoning_length")} for r in full]
        report["models"][md.name] = {
            "n_items": len(full), "full": fi, "table": table(rows), "readings": readings(rows, a.n_boot),
            "rows": rows}
        (md / "scored.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (mode_dir / "report.json").write_text(json.dumps(report, indent=1, default=str) + "\n")
    for m, d in report["models"].items():
        print(f"== {m}: {d['n_items']} items ({a.split_name}, split sha {sha[:12]})")
        for f in d["full"]:
            print(f"   FULL item {f['item_id']}: pred={f['pred']!r} truncated={f['truncated']} "
                  f"gen={f['n_gen_tokens']} trace_tokens={f['trace_tokens']}")
        for k, v in d["table"].items():
            print(f"   {k:10s} {v}")
        for k, v in d["readings"].items():
            print(f"   {k}: {v}")
    print(f"report: {mode_dir / 'report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
