#!/usr/bin/env python
"""Stage 1 dev choices and readings (selfdoubt/stage1.py).

    # dev: which cut? Delta_V proxy from the Stage 0 dev store (UG vs U0 acceptance per cut, item bootstrap)
    .venv/bin/python selfdoubt/stage1_report.py cut --stage0-store runs/selfdoubt/s0c_pod1/dev/full/Qwen3-1.7B/gen.jsonl
    # readings on a Stage 1 store (dev; held-out needs --heldout-confirmed, after the dev choices are frozen)
    .venv/bin/python selfdoubt/stage1_report.py read --store runs/selfdoubt/s1_dev/dev/stage1/Qwen3-1.7B/gen.jsonl
    # v2 (PROPOSAL c69f28e): plain-payload readings under S1t and S0 and the fixed primary-setting rule
    .venv/bin/python selfdoubt/stage1_report.py v2 --store-s1t <out>/dev/stage1_s1t/Qwen3-1.7B/gen.jsonl \
        --store-s0 <out>/dev/stage1_s0/Qwen3-1.7B/gen.jsonl
    # R1.B1 / R1.B2 for the S1 setting (plain; three-way equivalence within +-0.10 on a 90% interval)
    .venv/bin/python selfdoubt/stage1_report.py r1b --store <S1 store> --stage0-store <Stage 0 store, same split>
Prints counts and rates only (no text).
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


def split_ids(name: str) -> set[int]:
    split, _ = C.ensure_split(C.load_items())
    return {i for src in C.SOURCES for i in split[name][src]}


def cut_choice(store: Path, n_boot: int = 2000) -> dict:
    """Per cut: acceptance (= gold) of UG and U0 on the Stage 0 dev store, and UG - U0 with an item-bootstrap 95%
    interval (pool: item-cuts the Stage 0 scorer keeps; truncated rows excluded pairwise)."""
    from selfdoubt import stage0_report as R
    items_all = C.load_items()
    dev = split_ids("dev")
    recs = [r for r in C.Store(store).recs.values()]
    if any(r["item_id"] not in dev for r in recs):
        raise SystemExit("cut choice reads dev only; this store holds non-dev items")
    rows = R.score_model(recs, {int(r["id"]): r for r in items_all})
    out = {}
    for x in C.CUTS:
        by = {}
        for r in rows:
            if r["cut"] == x and r["arm"] in ("U0", "UG") and not r["category"].startswith("dropped"):
                by.setdefault(r["item_id"], {})[r["arm"]] = r
        pairs = {i: (d["UG"]["category"] == "new") - (d["U0"]["category"] == "new") for i, d in by.items()
                 if "UG" in d and "U0" in d and "truncated" not in (d["UG"]["category"], d["U0"]["category"])}
        ids = sorted(pairs)
        if not ids:
            continue
        rng = np.random.default_rng(56)
        bs = [np.mean([pairs[i] for i in rng.choice(ids, len(ids))]) for _ in range(n_boot)]
        acc = lambda a: float(np.mean([by[i][a]["category"] == "new" for i in ids]))  # noqa: E731
        out[str(x)] = {"n_items": len(ids), "acc_U0": acc("U0"), "acc_UG": acc("UG"),
                       "dV_proxy": float(np.mean([pairs[i] for i in ids])),
                       "ci95": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]}
    return out


def read(store: Path, split_name: str, n_boot: int) -> dict:
    """Readings with "plain" (PRIMARY) and "doc" (SECONDARY, descriptive), the payload sanity check and category counts."""
    keep = split_ids(split_name)
    items = {int(r["id"]): r for r in C.load_items()}
    recs = [r for r in C.Store(store).recs.values() if r["item_id"] in keep]
    rows = S1.score(recs, items)
    present = sorted({r.get("payload") or "plain" for r in rows if r["arm"].startswith("I")})
    out = {"split_name": split_name, "note": S1.OFF_POLICY_NOTE, "primary": "plain", "by_payload": {}}
    for pl in present:
        out["by_payload"][pl] = S1.readings(rows, n_boot, payload=pl)
    out["payload_sanity"] = S1.payload_sanity({pl: d["acceptance"] for pl, d in out["by_payload"].items()})
    cats = {}
    for r in rows:
        k = r["arm"] + (f"|{r.get('payload')}" if r["arm"].startswith("I") else "")
        cats.setdefault(k, {}).setdefault(r["category"], 0)
        cats[k][r["category"]] += 1
    out["categories"] = cats
    (store.parent / f"stage1_readings_{split_name}.json").write_text(json.dumps(out, indent=1) + "\n")
    (store.parent / f"scored_{split_name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return out


def _paired(diffs: dict, n_boot: int, level: float, seed: int = 56) -> dict:
    """Mean of per-item differences with an item-bootstrap percentile interval (one value per item)."""
    ids = sorted(diffs)
    if not ids:
        return {"n_items": 0, "point": None, "ci": [None, None], "level": level}
    rng = np.random.default_rng(seed)
    bs = [np.mean([diffs[i] for i in rng.choice(ids, len(ids))]) for _ in range(n_boot)]
    a = (1 - level) / 2
    return {"n_items": len(ids), "point": float(np.mean([diffs[i] for i in ids])),
            "ci": [float(np.quantile(bs, a)), float(np.quantile(bs, 1 - a))], "level": level}


EQ_BAND = 0.10     # R1.B1 / R1.B2: SHOWN if the 90% interval is inside +-0.10 (inclusive), NOT SHOWN if wholly outside
MIN_B2_PAIRS = 200  # Advisor 2026-11-09: fewer R1.B2 pairs than this is reported as such (the reading is unchanged)


def r1b(store: Path, stage0_store: Path, split_name: str, n_boot: int) -> dict:
    """PROPOSAL c69f28e, S1 setting, plain payload; three-way on a 90% interval, band +-0.10.
    R1.B1 = acc(I2b) - acc(VG), paired by item in the S1 store (rows the Stage 1 scorer keeps; truncated excluded).
    R1.B2 = acc(V0 under S1) - acc(UG under the authors' prompt, Stage 0 rows at the same cut), on the same items
    (each side kept by its own scorer: not dropped, not truncated)."""
    from selfdoubt import stage0_report as R
    keep = split_ids(split_name)
    items = {int(r["id"]): r for r in C.load_items()}
    recs = [r for r in C.Store(store).recs.values() if r["item_id"] in keep]
    systems = {r.get("sys", "s1") for r in recs}
    if systems != {"s1"}:
        raise SystemExit(f"R1.B readings are for the S1 store only; this store holds {sorted(systems)}")
    cuts = {r["cut"] for r in recs}
    assert len(cuts) == 1, cuts
    cut = cuts.pop()
    ok = lambda r: not r["category"].startswith("dropped") and r["category"] != "truncated"  # noqa: E731
    by = {}
    for r in S1.score(recs, items):
        if ok(r) and (not r["arm"].startswith("I") or (r.get("payload") or "plain") == "plain"):
            by.setdefault(r["item_id"], {})[r["arm"]] = r["accepted"]
    b1 = {i: d["I2b"] - d["VG"] for i, d in by.items() if "I2b" in d and "VG" in d}
    s0 = [r for r in C.Store(stage0_store).recs.values() if r["item_id"] in keep and r["item_id"] in by]
    ug = {r["item_id"]: r["category"] == "new" for r in R.score_model(s0, items)
          if r["arm"] == "UG" and r["cut"] == cut and ok(r)}
    b2 = {i: by[i]["V0"] - ug[i] for i in by if "V0" in by[i] and i in ug}
    v0 = {i for i in by if "V0" in by[i]}
    out = {"split_name": split_name, "cut": cut, "payload": "plain", "band": EQ_BAND,
           "r1b2_pairing": {"join": "item id", "s1_v0_items_kept": len(v0), "stage0_ug_items_kept": len(ug),
                            "pairs": len(b2), "min_pairs": MIN_B2_PAIRS, "below_min": len(b2) < MIN_B2_PAIRS,
                            "note": (f"only {len(b2)} R1.B2 pairs (< {MIN_B2_PAIRS}); reported, not shrunk silently"
                                     if len(b2) < MIN_B2_PAIRS else "ok")}}
    for name, d, what in (("R1.B1", b1, "acc(I2b) - acc(VG), S1"),
                          ("R1.B2", b2, "acc(V0 under S1) - acc(UG, Stage 0, authors' prompt), same items")):
        p = _paired(d, n_boot, 0.90)
        p["stat"] = what
        p["verdict"] = R.verdict(*p["ci"], EQ_BAND, "within") if p["n_items"] else "UNDETERMINED"
        out[name] = p
    return out


def v2(store_s1t: Path, store_s0: Path, split_name: str, n_boot: int) -> dict:
    """Stage 1 v2 (PROPOSAL c69f28e): plain-payload readings under S1t and S0, then the fixed primary-setting rule:
    S1t if its gate passes, else S0 if its gate passes, else the 1.7B selectivity reading closes UNDETERMINED."""
    keep = split_ids(split_name)
    items = {int(r["id"]): r for r in C.load_items()}
    out = {"split_name": split_name, "payload": "plain", "by_sys": {}}
    for sy, path in (("s1t", store_s1t), ("s0", store_s0)):
        recs = [r for r in C.Store(path).recs.values() if r["item_id"] in keep]
        got = {r.get("sys", "s1") for r in recs}
        if got != {sy}:
            raise SystemExit(f"{path}: expected sys {sy}, found {sorted(got)}")
        rows = S1.score(recs, items)
        rd = S1.readings(rows, n_boot, payload="plain")
        cats = {}
        for r in rows:
            if not r["arm"].startswith("I") or (r.get("payload") or "plain") == "plain":
                cats.setdefault(r["arm"], {}).setdefault(r["category"], 0)
                cats[r["arm"]][r["category"]] += 1
        rd["categories"] = cats
        rd["payload_sanity"] = S1.payload_sanity({"plain": rd["acceptance"]})
        out["by_sys"][sy] = rd
    g = {sy: out["by_sys"][sy]["gate"]["pass"] for sy in ("s1t", "s0")}
    prim = "s1t" if g["s1t"] else ("s0" if g["s0"] else None)
    out["primary"] = {"setting": prim, "gates": g,
                      "rule": "S1t if its dev gate passes, else S0 if its gate passes, else closed UNDETERMINED",
                      "decision": (f"primary setting {prim}" if prim else
                                   "neither gate passes: the 1.7B selectivity reading closes as UNDETERMINED")}
    return out


def heldout(store_s1t: Path, store_s1: Path, stage0_store: Path, split_name: str, n_boot: int) -> dict:
    """The Stage 1 held-out readings, all frozen at c69f28e, nothing new:
    - the selectivity reading (gate, then dI2b - 0.5 dV) on the PRIMARY setting S1t (v2 dev rule: S1t's gate passed);
    - R1.B1 and R1.B2 on the S1 setting (r1b), with the R1.B2 item-id join reported.
    Plain payload throughout. Run on dev stores it is the dry check (the numbers the rules were formed on)."""
    items = {int(r["id"]): r for r in C.load_items()}
    keep = split_ids(split_name)
    recs = [r for r in C.Store(store_s1t).recs.values() if r["item_id"] in keep]
    got = {r.get("sys", "s1") for r in recs}
    if got != {"s1t"}:
        raise SystemExit(f"{store_s1t}: expected sys s1t, found {sorted(got)}")
    rd = S1.readings(S1.score(recs, items), n_boot, payload="plain")
    return {"split_name": split_name, "payload": "plain", "rules": "PROPOSAL c69f28e (frozen); primary setting s1t",
            "primary_s1t": {k: rd[k] for k in ("n_items", "acceptance", "deltas", "gate", "selectivity", "U_lambda1", "U_lambda2") if k in rd},
            "payload_sanity_s1t": S1.payload_sanity({"plain": rd["acceptance"]}),
            "s1_r1b": r1b(store_s1, stage0_store, split_name, n_boot)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cut")
    c.add_argument("--stage0-store", required=True)
    c.add_argument("--n-boot", type=int, default=2000)
    r = sub.add_parser("read")
    r.add_argument("--store", required=True)
    r.add_argument("--split-name", choices=["dev", "heldout"], default="dev")
    r.add_argument("--heldout-confirmed", action="store_true")
    r.add_argument("--n-boot", type=int, default=2000)
    v = sub.add_parser("v2")
    v.add_argument("--store-s1t", required=True)
    v.add_argument("--store-s0", required=True)
    v.add_argument("--split-name", choices=["dev", "heldout"], default="dev")
    v.add_argument("--heldout-confirmed", action="store_true")
    v.add_argument("--n-boot", type=int, default=2000)
    b = sub.add_parser("r1b")
    b.add_argument("--store", required=True, help="S1 Stage 1 store")
    b.add_argument("--stage0-store", required=True, help="Stage 0 store of the same split (UG rows at the same cut)")
    b.add_argument("--split-name", choices=["dev", "heldout"], default="dev")
    b.add_argument("--heldout-confirmed", action="store_true")
    b.add_argument("--n-boot", type=int, default=2000)
    h = sub.add_parser("heldout", help="the frozen Stage 1 held-out readings (S1t selectivity + S1 R1.B1/B2)")
    h.add_argument("--store-s1t", required=True)
    h.add_argument("--store-s1", required=True)
    h.add_argument("--stage0-store", required=True, help="Stage 0 store of the same split (UG rows at the same cut)")
    h.add_argument("--split-name", choices=["dev", "heldout"], default="heldout")
    h.add_argument("--heldout-confirmed", action="store_true")
    h.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args(argv)
    if a.cmd == "cut":
        print(json.dumps(cut_choice(Path(a.stage0_store), a.n_boot), indent=1))
        return 0
    if a.split_name == "heldout" and not a.heldout_confirmed:
        raise SystemExit("held-out readings need --heldout-confirmed (after the dev choices are frozen)")
    if a.cmd == "v2":
        out = v2(Path(a.store_s1t), Path(a.store_s0), a.split_name, a.n_boot)
        (Path(a.store_s1t).parent.parent.parent / f"stage1_v2_{a.split_name}.json").write_text(
            json.dumps(out, indent=1) + "\n")
    elif a.cmd == "heldout":
        out = heldout(Path(a.store_s1t), Path(a.store_s1), Path(a.stage0_store), a.split_name, a.n_boot)
        (Path(a.store_s1t).parent.parent.parent / f"stage1_final_{a.split_name}.json").write_text(
            json.dumps(out, indent=1) + "\n")
    elif a.cmd == "r1b":
        out = r1b(Path(a.store), Path(a.stage0_store), a.split_name, a.n_boot)
    else:
        out = read(Path(a.store), a.split_name, a.n_boot)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
