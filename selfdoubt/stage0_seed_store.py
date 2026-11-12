#!/usr/bin/env python
"""Seed a pod's store for the Stage 0 completion run (pod 1, Advisor rule of 2026-11-09).

    python selfdoubt/stage0_seed_store.py --src runs/selfdoubt/full/Qwen3-1.7B/gen.jsonl \
        --dst runs/selfdoubt/s0c_pod1/heldout/full/Qwen3-1.7B/gen.jsonl --source math500 --expect-items 19 --expect-drop 54

Keeps one source's records. For every item whose continuations are incomplete (fewer than the 11 expected (arm, cut)
rows), it drops ALL of that item's continuation rows and keeps its FULL record. The pod's resume then regenerates all
11 rows from the same FULL prefix with the same per-row seeds (common.request_seed). Complete items are kept as they
are and are skipped on the pod. Only the key fields (source, item_id, arm, cut) are parsed; kept lines are copied
byte for byte. No text is read or printed.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from selfdoubt import common as C  # noqa: E402

EXPECTED = {(a, c) for c in C.CUTS for a in ("N", "U0", "UG")} | {(a, C.X0) for a in ("U0", "UG")}


def seed(src: Path, dst: Path, source: str) -> dict:
    lines, rows = [], defaultdict(set)
    full = set()
    for raw in src.read_text().splitlines():
        if not raw.strip():
            continue
        r = json.loads(raw)
        if r["source"] != source:
            continue
        lines.append((r["item_id"], r["arm"], raw))
        if r["arm"] == "FULL":
            full.add(r["item_id"])
        else:
            rows[r["item_id"]].add((r["arm"], r["cut"]))
    assert all(rows[i] <= EXPECTED for i in rows), "unexpected (arm, cut) in store"
    assert set(rows) <= full, "continuations without a FULL record"
    incomplete = sorted(i for i in full if rows[i] != EXPECTED)
    keep = [raw for i, arm, raw in lines if arm == "FULL" or i not in incomplete]
    dropped = sum(1 for i, arm, _ in lines if arm != "FULL" and i in incomplete)
    if dst.exists():
        raise SystemExit(f"{dst} exists; refusing to overwrite")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("".join(x + "\n" for x in keep))
    return {"source": source, "n_full": len(full), "n_complete": len(full) - len(incomplete),
            "incomplete": incomplete, "n_dropped_rows": dropped, "n_kept_lines": len(keep)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--source", required=True, choices=list(C.SOURCES))
    ap.add_argument("--expect-items", type=int, default=None, help="abort unless this many items are incomplete")
    ap.add_argument("--expect-drop", type=int, default=None, help="abort unless this many finished rows are dropped")
    a = ap.parse_args(argv)
    out = seed(Path(a.src), Path(a.dst), a.source)
    if a.expect_items is not None and len(out["incomplete"]) != a.expect_items:
        Path(a.dst).unlink()
        raise SystemExit(f"expected {a.expect_items} incomplete items, found {len(out['incomplete'])}")
    if a.expect_drop is not None and out["n_dropped_rows"] != a.expect_drop:
        Path(a.dst).unlink()
        raise SystemExit(f"expected {a.expect_drop} dropped rows, found {out['n_dropped_rows']}")
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
