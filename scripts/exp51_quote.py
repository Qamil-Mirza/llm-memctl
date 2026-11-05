#!/usr/bin/env python
"""§51 GPU quote for S + H + T-city on one A40 pod (strict fp32), after the B1 / B4 shortcut (c) and the T-format
drop, from the inventory's row counts x per-row GPU seconds. The per-row seconds are ESTIMATES until the timed
measurement pod (scripts/exp51_pod_measure.sh) replaces them (--measured <measure.json>).

    .venv/bin/python scripts/exp51_quote.py [--measured runs/exp51/gpu_measure/pod/measure/measure.json] [--rate 0.59]

Batched route (the user's choice, 2026-11-02): generation passes run at a fixed batch layout; component hooks (M2),
the hybrid estimand and teacher-forced reads run at batch 1. Every arm on a split uses the same backend and layout.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# A40 strict fp32 estimates (s per row): memory-bound decode ~25 ms / token at batch 1 (12.4 GB weights at ~0.7 TB/s
# with overhead), ~600-token prefill ~0.3 s; at batch 32-64 the decode step is shared and prefill dominates.
EST = {"full_b1": 2.5, "full_batched": 0.35, "resume_batched": 0.25, "prefix_batched": 0.15, "prefix_b1": 0.6,
       "read_b1": 0.4}
FIXED_S = {"pod start + image": 300, "install": 240, "download 3B": 180, "D-22 smoke (dummy)": 180,
           "copy back + checks": 300}
FACTOR = 1.30
RULE = ("Fixed before any GPU measurement (Advisor re-check, 2026-11-02): one backend and one fixed batch layout per "
        "split for every confirmatory arm, B0 included. GPU batched vs CPU batch-1 on the 64 measurement E rows: graded-"
        "class agreement >= 62/64 -> the batched route is usable (token mismatches disclosed); below -> not used for "
        "confirmatory work.")


def rows(bases: int) -> dict:
    return {"all": 13 * bases, "ANS": 6 * bases, "UNA": 4 * bases, "CF": 3 * bases, "AU": 10 * bases}


def workload() -> list[tuple[str, str, int, str]]:
    S, H, T = rows(96), rows(144), rows(72)
    w = [("replay check: 200 E rows x (B0 + one edit), batched layout", "full_batched", 400, "§10"),
         ("replay check: the same at batch 1 (CPU parity reference)", "full_b1", 400, "§10"),
         ("S: B0", "full_batched", S["all"], ""),
         ("S: B1 whole, B2, B3, B5 x 4, R candidates 3 x 4 (19 full arms)", "full_batched", 19 * S["all"], ""),
         ("S: B1 online x 6 + B4 x 4 as B0 prefix + continuation (c)", "resume_batched", 10 * S["all"], ""),
         ("S: M candidates 12 + revert + B6 x 5 + f-int x 4 + B1 grid x 16 (ANS + UNA)", "prefix_batched",
          38 * S["AU"], ""),
         ("S: transplant + clamp (batch 1)", "prefix_b1", 2 * S["AU"], "hooks"),
         ("S: reads (transplant / clamp values, f, predictor)", "read_b1", 4 * S["AU"] + S["all"] + S["ANS"], ""),
         ("H: B0, B1 whole, B2, B3, B5, R candidate (6 full arms)", "full_batched", 6 * H["all"], ""),
         ("H: B1 + B4 online via (c)", "resume_batched", 2 * H["all"], ""),
         ("H: B0, M candidate, revert, B6 x 5, f-int, matched B1 (prefix)", "prefix_batched", 10 * H["AU"], ""),
         ("H: transplant + clamp (batch 1)", "prefix_b1", 2 * H["AU"], "hooks"),
         ("H: reads", "read_b1", 4 * H["AU"] + H["all"] + H["ANS"], ""),
         ("H: hybrid answer-site, secondary (batch 1)", "prefix_b1", 2 * H["AU"], "secondary"),
         ("T-city: B0 + 5 full arms", "full_batched", 6 * T["all"], "secondary"),
         ("T-city: B1 + B4 via (c)", "resume_batched", 2 * T["all"], "secondary")]
    return w


def quote(unit: dict, rate: float) -> dict:
    items, total = [], 0.0
    for what, kind, n, note in workload():
        s = n * unit[kind]
        total += s
        items.append({"what": what, "unit": kind, "rows": n, "s_per_row": unit[kind], "h": round(s / 3600, 2),
                      "note": note})
    fixed = sum(FIXED_S.values())
    run_h = total * FACTOR / 3600
    all_h = run_h + fixed / 3600
    sec = sum(i["h"] for i in items if i["note"] == "secondary") * FACTOR
    return {"items": items, "fixed_s": FIXED_S, "factor": FACTOR, "run_h": round(run_h, 2), "total_h": round(all_h, 2),
            "rate_usd_h": rate, "usd": round(all_h * rate, 2), "usd_without_secondary": round((all_h - sec) * rate, 2),
            "hard_stop_usd": round(all_h * rate * 1.15, 2), "unit_s": unit}


def measured_units(path: Path) -> dict:
    m = json.loads(Path(path).read_text())["steps"]
    u = dict(EST)
    big = max((k for k in m if k.startswith("total_b") and k != "total_b1" and "s_per_row" in m[k]),
              key=lambda k: int(k.split("_b")[1]), default=None)
    if "total_b1" in m:
        u["full_b1"] = m["total_b1"]["s_per_row"]
    if big:
        u["full_batched"] = m[big]["s_per_row"]
    for k, v in m.items():
        if k.startswith("prefix_b") and "s_per_row" in v:
            u["prefix_batched"] = v["s_per_row"]
        if k.startswith("resume_b") and "s_per_row" in v:
            u["resume_batched"] = v["s_per_row"]
    if "read_b1" in m:
        u["read_b1"] = m["read_b1"]["s_per_row"]
        u["prefix_b1"] = max(u["prefix_b1"], m["read_b1"]["s_per_row"] * 1.5)
    return u


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--measured", default=None)
    p.add_argument("--rates", default="0.59,0.35", help="A40 secure, community ($/h, live read 2026-11-02)")
    a = p.parse_args(argv)
    unit = measured_units(Path(a.measured)) if a.measured else dict(EST)
    rates = [float(x) for x in a.rates.split(",")]
    q = quote(unit, rates[0])
    q["basis"] = "measured" if a.measured else "ESTIMATED (no pod timing yet)"
    q["rule_fixed_before_measurement"] = RULE
    q["by_rate"] = {}
    for r, label in zip(rates, ("secure", "community", "other")):
        qr = quote(unit, r)
        b1 = quote({**unit, "full_batched": unit["full_b1"], "resume_batched": unit["full_b1"] * 0.4,
                    "prefix_batched": unit["prefix_b1"]}, r)
        q["by_rate"][label] = {"rate_usd_h": r, "usd": qr["usd"], "hard_stop_usd": qr["hard_stop_usd"],
                               "usd_without_secondary": qr["usd_without_secondary"],
                               "batch1_everything_usd": b1["usd"]}
    out = ROOT / "runs/exp51/run/gpu_quote.json"
    out.write_text(json.dumps(q, indent=1))
    print(json.dumps({"basis": q["basis"], "total_h": q["total_h"], "by_rate": q["by_rate"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
