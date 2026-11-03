#!/usr/bin/env python
"""§51 workload inventory (§10): every split x row x arm x dose x seed x estimand x capture, each costed per row on
the laptop CPU from measured timing (runs/exp51/run/timing.json and the B0 records' own generation times). No cost in
dollars: a GPU quote needs pod timing first (§10). Secondary work (T-city, T-format, the hybrid estimand, natural-chat
rows) is listed last and flagged as the first to drop.

    .venv/bin/python scripts/exp51_run.py --stage inventory      # -> runs/exp51/run/inventory.json + .md
"""

from __future__ import annotations

import glob
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import exp51_common as CM  # noqa: E402

PASS_MAX = {"paired": 64, "prefix": 256, "read": 512}


def unit_costs(run: Path) -> dict:
    """Seconds per row: total-effect generation by set (B0 records), controlled prefix and teacher-forced read
    (timing units), model load."""
    t = CM.read_json(run / "timing.json", {})
    recs = [json.loads(Path(f).read_text()) for f in glob.glob(str(run / "b0/E/*.json"))]
    by = defaultdict(list)
    for r in recs:
        s = r["case_id"].split("_")[1]
        by["ANS" if s.startswith("ANS") else s].append(r["gen_s"])
    total = {k: float(np.mean(v)) for k, v in by.items()}
    allv = [x for v in by.values() for x in v]
    units = t.get("units", [])
    prefix = float(np.mean([u["edit_prefix_s"] for u in units if "edit_prefix_s" in u])) if units else 9.0
    read = float(np.mean([u["read_s"] for u in units if "read_s" in u])) if units else 7.0
    return {"total_by_set": total, "total_mean": float(np.mean(allv)) if allv else 25.0, "n_b0_records": len(recs),
            "prefix": prefix, "read": read, "load_s": t.get("load_s", 10.0), "apply_s": t.get("apply_s", 0.6)}


def split_counts(sizes: dict, sealed_ok: bool = False) -> dict:
    """Rows per split by set from the generator's sizes (no sealed outputs are read)."""
    out = {}
    for split, n in sizes.items():
        b = 3 * n
        out[split] = {"bases": b, "ANS": 6 * b, "UNA": 4 * b, "CF": 3 * b, "all": 13 * b}
    return out


def inventory(run: Path) -> dict:
    counts = json.loads((ROOT / "runs/exp51/cases51_counts.json").read_text())
    sizes = counts["sizes"]
    rc = split_counts(sizes)
    u = unit_costs(run)
    T, P, R = u["total_mean"], u["prefix"], u["read"]
    items = []

    def add(stage, what, n_rows, sec_per_row, note="", secondary=False):
        items.append({"stage": stage, "what": what, "rows": int(n_rows), "sec_per_row": round(sec_per_row, 2),
                      "cpu_h": round(n_rows * sec_per_row / 3600, 2), "note": note, "secondary": secondary})
    E, S, H = rc["E"], rc["S"], rc["H"]
    # stage 1 (pre-F0 analyst work)
    add("1", "B0 on E (total effect; P_ans residual captured)", E["all"], T)
    add("1", "B0 on S", S["all"], T)
    add("1", "no-op gate (24 E rows: zero edit total + prefix, B0 prefix replay)", 24, T + 2 * P)
    # stage 3 editor run (worst cases)
    add("3", "editor: 30 paired passes, worst case (64 pairs, both arms edited)", 30 * 128, T,
        "upper bound; an unedited arm is served from B0 (64 sequences per pass)")
    add("3", "editor: 30 prefix passes, worst case (256 rows)", 30 * 256, P, "alternative: upper bound")
    add("3", "editor: 30 read passes, worst case (512 texts)", 30 * 512, R, "alternative: upper bound")
    add("3", "editor: a mix of 10 paired (64 edited sequences) passes", 10 * 64, T, "alternative: mix")
    add("3", "editor: a mix of 10 prefix passes (256 rows)", 10 * 256, P, "alternative: mix")
    add("3", "editor: a mix of 10 read passes (512 texts)", 10 * 512, R, "alternative: mix")
    add("3", "harness replay: 3 submissions x all E (total effect)", 3 * E["all"], T)
    add("3", "harness replay: 3 submissions x E ANS (controlled prefix, M entry)", 3 * E["ANS"], P)
    # stage 4 S
    r_arms = [("B1 online x 6 doses", 6), ("B1 whole-generation at the selected dose", 1), ("B2", 1), ("B3", 1),
              ("B4 x 4 alphas", 4), ("B5 x 4 doses", 4), ("R candidates: 3 submissions x 4 doses", 12)]
    for name, k in r_arms:
        add("4", f"S total effect: {name}", k * S["all"], T)
    add("4", "S, if B1 / B4 online are run as B0 prefix + triggered continuation (exact in exact arithmetic): "
        "the 11 B1 / B4 settings", 11 * S["all"], P, "alternative: shortcut (saves the note regeneration)")
    m_rows = S["ANS"] + S["UNA"]
    add("4", "S prefix: M candidates 3 x 4 doses (ANS + UNA)", 12 * m_rows, P)
    add("4", "S prefix: revert of the primary component (selected M)", m_rows, P)
    add("4", "S prefix: B6 x 5 seeds (selected M)", 5 * m_rows, P)
    add("4", "S prefix: transplant + clamp (2 reads + 1 generation each)", 2 * m_rows, P + 2 * R)
    add("4", "S prefix: f-intervention x 4 doses + B1 signed grid x 16 (FA + UNA)", 20 * m_rows, P)
    add("4", "S reads: f under up to 4 manipulations (all rows)", S["all"], R)
    add("4", "S reads: M_city + unknown log-odds (ANS)", S["ANS"], R)
    # stage 5 H (and the replay)
    add("5", "replay check: 200 E rows x (B0 + one edit), total effect", 400, T, "pod (CPU<->pod parity)")
    for name, k in [("B0", 1), ("B1 frozen dose (+ whole-generation variant)", 2), ("B2", 1), ("B3", 1), ("B4", 1),
                    ("B5", 1), ("R candidate", 1)]:
        add("5", f"H total effect: {name}", k * H["all"], T)
    hm = H["ANS"] + H["UNA"]
    add("5", "H prefix: B0 + M candidate + revert + B6 x 5 + f-int + matched B1", 10 * hm, P)
    add("5", "H prefix: transplant + clamp", 2 * hm, P + 2 * R)
    add("5", "H reads: f (manipulations) + predictor features", H["all"] + H["ANS"], R)
    for tsplit in ("Tcity", "Tformat"):
        add("5", f"{tsplit} total effect: B0 + B1-B5 + R candidate (7 arms)", 7 * rc[tsplit]["all"], T,
            "separate transfer report; first to drop", secondary=True)
    add("5", "H hybrid answer-site (secondary): M candidate + B0", 2 * hm, P, "secondary", secondary=True)
    add("5", "LongMemEval 165 + LoCoMo §40 rows (descriptive): B0 + R candidate", 2 * 300, T,
        "previously read; lowest priority", secondary=True)
    by_stage = Counter()
    for it in items:
        if it["note"].startswith("alternative"):
            continue
        by_stage[it["stage"] + (" (secondary)" if it["secondary"] else "")] += it["cpu_h"]
    passes_30_worst = 30 * 128 * T / 3600
    r = min(1.0, 48.0 / passes_30_worst)
    out = {"units_s": u, "rows": rc, "items": items, "totals": {"cpu_h_by_stage": dict(sorted(by_stage.items())),
           "editor_30_passes_worst_h": round(passes_30_worst, 1), "editor_reduction_r": round(r, 3),
           "editor_passes_in_48h_worst": int(48 * 3600 // (128 * T))}}
    CM.write_json(run / "inventory.json", out)
    lines = ["| Stage | Work | Rows | s/row | CPU h | Note |", "|---|---|---|---|---|---|"]
    for it in items:
        lines.append(f"| {it['stage']}{' (sec.)' if it['secondary'] else ''} | {it['what']} | {it['rows']} | "
                     f"{it['sec_per_row']} | {it['cpu_h']} | {it['note']} |")
    (run / "inventory.md").write_text("\n".join(lines) + "\n\nTotals: " + json.dumps(out["totals"]) + "\n")
    return out
