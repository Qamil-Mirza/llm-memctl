#!/usr/bin/env python
"""Stage 1 dev choices and readings (selfdoubt/stage1.py).

    # dev: which cut? Delta_V proxy from the Stage 0 dev store (UG vs U0 acceptance per cut, item bootstrap)
    .venv/bin/python selfdoubt/stage1_report.py cut --stage0-store runs/selfdoubt/s0c_pod1/dev/full/Qwen3-1.7B/gen.jsonl
    # readings on a Stage 1 store (dev; held-out needs --heldout-confirmed, after the dev choices are frozen)
    .venv/bin/python selfdoubt/stage1_report.py read --store runs/selfdoubt/s1_dev/dev/stage1/Qwen3-1.7B/gen.jsonl
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
    a = ap.parse_args(argv)
    if a.cmd == "cut":
        print(json.dumps(cut_choice(Path(a.stage0_store), a.n_boot), indent=1))
        return 0
    if a.split_name == "heldout" and not a.heldout_confirmed:
        raise SystemExit("held-out readings need --heldout-confirmed (after the dev choices are frozen)")
    print(json.dumps(read(Path(a.store), a.split_name, a.n_boot), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
