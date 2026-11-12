#!/usr/bin/env python
"""Stage 1 replication across models and sources (PROPOSAL 362b16c, frozen by the Advisor before any row exists).

Held-out, S1 only, plain payload, cut 0.9, arms N / V0 / VG / I1 / I2b. Readings, unchanged from c69f28e:
  R1.B1-rep = acc(I2b) - acc(VG)                          (paired by item in the S1 store)
  R1.B2-rep = acc(V0 | S1) - acc(UG | Stage 0, cut 0.9)   (same items, joined by item id to that model/source's store)
three-way on a 90% interval, band +-0.10 (SHOWN inside, NOT SHOWN wholly outside, else UNDETERMINED).
PRIMARY = pooled per model over GSM8K + MATH-500, item bootstrap STRATIFIED by source (2,000 draws, seed 56);
SECONDARY = each cell (stage1_report._paired, the same bootstrap the 1.7B GSM8K reading used).
Descriptive only: acceptance per arm and the wrapper lift acc(I2b) - acc(I1). Stop rule per cell: payload sanity
(I1 or I2b acceptance > 0), else the cell is a payload failure and is not read (nor pooled).
Conventions as frozen: items with N = gold or N invalid dropped; truncated rows excluded pairwise.

    .venv/bin/python selfdoubt/stage1_rep.py --heldout-confirmed --layout two     # or --layout one
    .venv/bin/python selfdoubt/stage1_rep.py --dry-check                           # 1.7B GSM8K cell only ($0)
Prints counts and rates only (no text). Writes selfdoubt/results/stage1_rep.json (tables only).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from selfdoubt import common as C  # noqa: E402
from selfdoubt import stage1 as S1  # noqa: E402
from selfdoubt import stage1_report as SR  # noqa: E402

WT = Path(__file__).resolve().parents[1]
RUNS = WT / "runs/selfdoubt"
CUT = 0.9
ARMS = ("N", "V0", "VG", "I1", "I2b")
N_BOOT, SEED, LEVEL = 2000, 56, 0.90
MODELS = {"1.7B": "Qwen/Qwen3-1.7B", "8B": "Qwen/Qwen3-8B"}
# Stage 0 held-out stores (UG at cut 0.9 for R1.B2; the same files are bundled as the pods' FULL stores)
STAGE0 = {("1.7B", "gsm8k"): "full/Qwen3-1.7B/gen.jsonl",
          ("1.7B", "math500"): "s0c_pod1/heldout/full/Qwen3-1.7B/gen.jsonl",
          ("8B", "gsm8k"): "s0c_pod2/heldout/full/Qwen3-8B/gen.jsonl",
          ("8B", "math500"): "s0c_pod2/heldout/full/Qwen3-8B/gen.jsonl"}
N_ITEMS = {("1.7B", "gsm8k"): 250, ("1.7B", "math500"): 250, ("8B", "gsm8k"): 250, ("8B", "math500"): 100}
# S1 stores: 1.7B GSM8K is the existing held-out S1 store (8244173); the rest come from the replication pod(s)
LAYOUTS = {
    "one": {("1.7B", "math500"): "s1rep_pod/heldout/stage1/Qwen3-1.7B/gen.jsonl",
            ("8B", "gsm8k"): "s1rep_pod/heldout/stage1/Qwen3-8B/gen.jsonl",
            ("8B", "math500"): "s1rep_pod/heldout/stage1/Qwen3-8B/gen.jsonl"},
    "two": {("1.7B", "math500"): "s1rep_a/heldout/stage1/Qwen3-1.7B/gen.jsonl",
            ("8B", "gsm8k"): "s1rep_b/heldout/stage1/Qwen3-8B/gen.jsonl",
            ("8B", "math500"): "s1rep_b/heldout/stage1/Qwen3-8B/gen.jsonl"},
}
S1_GSM8K_17 = "s1ho_pod/heldout/stage1/Qwen3-1.7B/gen.jsonl"


def cell(store: Path, stage0_store: Path, model: str, source: str, split_name: str = "heldout") -> dict:
    """Per-item accepted flags (kept rows only), the R1 differences and the descriptive table for one cell."""
    from selfdoubt import stage0_report as R
    keep = SR.split_ids(split_name)
    items = {int(r["id"]): r for r in C.load_items()}
    recs = [r for r in C.Store(store).recs.values()
            if r["item_id"] in keep and r["source"] == source and r["model"] == model]
    got = {r.get("sys", "s1") for r in recs}
    if got != {"s1"}:
        raise SystemExit(f"{store} [{source}]: expected sys s1 only, found {sorted(got)}")
    cuts = {r["cut"] for r in recs}
    if cuts != {CUT}:
        raise SystemExit(f"{store} [{source}]: expected cut {CUT}, found {sorted(cuts)}")
    rows = [r for r in S1.score(recs, items) if not r["arm"].startswith("I") or (r.get("payload") or "plain") == "plain"]
    ok = lambda r: not r["category"].startswith("dropped") and r["category"] != "truncated"  # noqa: E731
    by = {}
    for r in rows:
        if ok(r):
            by.setdefault(r["item_id"], {})[r["arm"]] = r["accepted"]
    s0 = [r for r in C.Store(stage0_store).recs.values()
          if r["item_id"] in keep and r["item_id"] in by and r["source"] == source and r["model"] == model]
    ug = {r["item_id"]: r["category"] == "new" for r in R.score_model(s0, items)
          if r["arm"] == "UG" and r["cut"] == CUT and ok(r)}
    b1 = {i: d["I2b"] - d["VG"] for i, d in by.items() if "I2b" in d and "VG" in d}
    b2 = {i: by[i]["V0"] - ug[i] for i in by if "V0" in by[i] and i in ug}
    lift = {i: d["I2b"] - d["I1"] for i, d in by.items() if "I2b" in d and "I1" in d}
    not_dropped = {r["item_id"] for r in rows if not r["category"].startswith("dropped")}
    acc = {}
    for a in ARMS[1:]:                 # acceptance per arm: not dropped, not truncated (the S1.readings convention)
        v = [by[i][a] for i in by if a in by[i]]
        acc[a] = (sum(v) / len(v)) if v else None
    cats = {}
    for r in rows:
        cats.setdefault(r["arm"], {}).setdefault(r["category"], 0)
        cats[r["arm"]][r["category"]] += 1
    arms_present = sorted({r["arm"] for r in recs})
    return {"model": model, "source": source, "store": str(store), "stage0_store": str(stage0_store),
            "n_items_in_store": len({r["item_id"] for r in recs}), "arms_in_store": arms_present,
            "n_items_not_dropped": len(not_dropped), "acceptance": acc, "categories": cats,
            "diffs": {"R1.B1": b1, "R1.B2": b2, "lift": lift},
            "r1b2_pairing": {"join": "item id", "s1_v0_items_kept": sum(1 for i in by if "V0" in by[i]),
                             "stage0_ug_items_kept": len(ug), "pairs": len(b2)},
            "paired_means": {"R1.B1": {"acc_I2b": _m([by[i]["I2b"] for i in b1]), "acc_VG": _m([by[i]["VG"] for i in b1])},
                             "R1.B2": {"acc_V0_S1": _m([by[i]["V0"] for i in b2]),
                                       "acc_UG_stage0": _m([float(ug[i]) for i in b2])}}}


def _m(xs):
    return float(np.mean(xs)) if xs else None


def sanity(c: dict) -> dict:
    a = c["acceptance"]
    ok = any((a.get(k) or 0) > 0 for k in ("I1", "I2b"))
    return {"pass": bool(ok), "rule": "I1 or I2b acceptance > 0 (PROPOSAL 362b16c), else payload failure: not read",
            "I1": a.get("I1"), "I2b": a.get("I2b")}


def stratified(diffs_by_source: dict, n_boot: int = N_BOOT, level: float = LEVEL, seed: int = SEED) -> dict:
    """Mean over all pooled items (one value per item); percentile interval from an item bootstrap that resamples
    within each source stratum (stratum sizes kept), sources in sorted order (stage0_heldout_readings convention)."""
    strata = {s: [d[i] for i in sorted(d)] for s, d in sorted(diffs_by_source.items()) if d}
    allv = [v for s in strata for v in strata[s]]
    if not allv:
        return {"n_items": 0, "n_by_source": {}, "point": None, "ci": [None, None], "level": level}
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        tot, n = 0.0, 0
        for s in sorted(strata):
            v = strata[s]
            k = rng.integers(0, len(v), len(v))
            tot += float(np.sum(np.asarray(v)[k])); n += len(v)
        bs.append(tot / n)
    a = (1 - level) / 2
    return {"n_items": len(allv), "n_by_source": {s: len(v) for s, v in strata.items()}, "point": float(np.mean(allv)),
            "ci": [float(np.quantile(bs, a)), float(np.quantile(bs, 1 - a))], "level": level}


def _read(p: dict, what: str) -> dict:
    from selfdoubt.stage0_report import verdict
    p["stat"] = what
    p["verdict"] = verdict(*p["ci"], SR.EQ_BAND, "within") if p["n_items"] else "UNDETERMINED"
    return p


STAT = {"R1.B1": "acc(I2b) - acc(VG), S1", "R1.B2": "acc(V0 | S1) - acc(UG | Stage 0, cut 0.9), same items"}


def readings(cells: dict, n_boot: int = N_BOOT) -> dict:
    """cells: {(model_label, source): cell(...)} -> per-cell (secondary) and pooled-per-model (primary) readings."""
    out = {"rules": "PROPOSAL 362b16c (frozen); conventions c69f28e", "cut": CUT, "payload": "plain", "setting": "s1",
           "arms": list(ARMS), "band": SR.EQ_BAND, "level": LEVEL, "n_boot": n_boot, "seed": SEED,
           "per_cell": {}, "pooled": {}}
    readable = {}
    for (ml, src), c in sorted(cells.items()):
        sn = sanity(c)
        pc = {k: c[k] for k in ("model", "source", "n_items_in_store", "arms_in_store", "n_items_not_dropped",
                                "acceptance", "categories", "r1b2_pairing", "paired_means")}
        pc["payload_sanity"] = sn
        lift = SR._paired(c["diffs"]["lift"], n_boot, 0.95)
        pc["wrapper_lift_descriptive"] = {"stat": "acc(I2b) - acc(I1), S1 (descriptive only)", **lift}
        if not sn["pass"]:
            pc["note"] = "payload failure: cell not read and not pooled"
        else:
            readable[(ml, src)] = c
            for name in ("R1.B1", "R1.B2"):
                pc[name] = _read(SR._paired(c["diffs"][name], n_boot, LEVEL), STAT[name])
        pc["r1b2_pairing"]["min_pairs"] = SR.MIN_B2_PAIRS
        pc["r1b2_pairing"]["below_min"] = pc["r1b2_pairing"]["pairs"] < SR.MIN_B2_PAIRS
        out["per_cell"][f"{ml}|{src}"] = pc
    for ml in sorted({k[0] for k in cells}):
        srcs = sorted(s for (m, s) in cells if m == ml)
        pool = {s: readable[(ml, s)] for s in srcs if (ml, s) in readable}
        po = {"model": MODELS[ml], "sources_planned": srcs, "sources_read": sorted(pool),
              "bootstrap": "item bootstrap stratified by source"}
        if set(pool) != set(srcs):
            po["note"] = "a cell failed payload sanity: the pool is not the frozen pool"
        for name in ("R1.B1", "R1.B2"):
            po[name] = _read(stratified({s: c["diffs"][name] for s, c in pool.items()}, n_boot), STAT[name])
        po["wrapper_lift_descriptive"] = stratified({s: c["diffs"]["lift"] for s, c in pool.items()}, n_boot, 0.95)
        out["pooled"][ml] = po
    return out


def build(layout: str) -> dict:
    paths = {("1.7B", "gsm8k"): S1_GSM8K_17, **LAYOUTS[layout]}
    cells = {}
    for (ml, src), rel in sorted(paths.items()):
        cells[(ml, src)] = cell(RUNS / rel, RUNS / STAGE0[(ml, src)], MODELS[ml], src)
        n = cells[(ml, src)]["n_items_in_store"]
        if n != N_ITEMS[(ml, src)]:
            raise SystemExit(f"{ml} {src}: {n} items in the S1 store, expected {N_ITEMS[(ml, src)]}: incomplete cell")
    return cells


def main(argv=None):
    from selfdoubt import scoring as SC
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", choices=sorted(LAYOUTS), default="two")
    ap.add_argument("--heldout-confirmed", action="store_true")
    ap.add_argument("--dry-check", action="store_true", help="1.7B GSM8K cell only (existing store; $0)")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--out", default=str(WT / "selfdoubt/results/stage1_rep.json"))
    a = ap.parse_args(argv)
    SC.TIMEOUTS.clear()
    if a.dry_check:
        c = {("1.7B", "gsm8k"): cell(RUNS / S1_GSM8K_17, RUNS / STAGE0[("1.7B", "gsm8k")], MODELS["1.7B"], "gsm8k")}
        out = readings(c, a.n_boot)
        out["timeouts"] = len(SC.TIMEOUTS)
        print(json.dumps(out, indent=1))
        return 0
    if not a.heldout_confirmed:
        raise SystemExit("held-out replication readings need --heldout-confirmed")
    out = readings(build(a.layout), a.n_boot)
    out["timeout_ledger"] = {"n_calls_timed_out": len(SC.TIMEOUTS),
                             "calls_by_fn": {f: sum(1 for t in SC.TIMEOUTS if t[0] == f) for f in {t[0] for t in SC.TIMEOUTS}}}
    if SC.TIMEOUTS:                    # sensitivity: rescore at 60 s; readings reported beside, primary unchanged
        SC.TIMEOUTS.clear(); old = SC.TIMEOUT_S; SC.TIMEOUT_S = 60
        try:
            out["readings_sensitivity_60s"] = readings(build(a.layout), a.n_boot)
            out["timeout_ledger"]["n_calls_timed_out_sensitivity"] = len(SC.TIMEOUTS)
        finally:
            SC.TIMEOUT_S = old
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out["pooled"][k] for k in out["pooled"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
