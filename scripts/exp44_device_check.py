#!/usr/bin/env python
"""§44 rev 8: device consistency of the 7B confirmation rows, laptop (CPU bf16) vs pod (CUDA bf16).

    .venv/bin/python scripts/exp44_device_check.py                       # runs/exp44_patch/qwen7b vs qwen7b_gpu
    .venv/bin/python scripts/exp44_device_check.py --laptop A --gpu B --out runs/exp44_patch/device_check_qwen7b

Spec: research/proposals/2026-10-29-exp44-patching.md "Rev 8" (GPU pod): the rows both devices have done are reported
as a device-consistency check before the GPU rows are read as the confirmation. For every row on both sides (same
question id; folds 2-4 only, the GPU run has no others): whether d_c is equal (both None counts as equal), the d_c
difference in layers, max |R(l) difference| per row and overall, eligibility flips under the rev 5 / rev 7 filter
(patching.patch_eligible: the relaxed rank filter with the 1-nat gap, re-derived from each side's recorded lp / rank,
never the stored field), gap-sign flips (sign of lp_clean - lp_corrupt) and 1-nat-gap flips, R(0)-flag flips, and the
largest |lp| differences of the clean and cut passes. No claim is computed and no d_c value is printed (equality and
differences only), so the check can be read before the confirmation read.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memctl.analysis import patching as pt  # noqa: E402

CONFIRM_FOLDS = (2, 3, 4)


def load(d: Path) -> dict[str, dict]:
    out = {}
    for p in sorted(d.glob("*.json")):
        if p.name.startswith("_"):
            continue
        r = json.loads(p.read_text())
        if r.get("fold") in CONFIRM_FOLDS:
            out[r["question_id"]] = r
    return out


def _sign(x: float) -> int:
    return (x > 0) - (x < 0)


def compare(a: dict, b: dict) -> dict:
    """One row, a = laptop, b = GPU."""
    el_a = pt.patch_eligible(a["rank_clean"], a["rank_corrupt"], a["lp_clean"], a["lp_corrupt"])
    el_b = pt.patch_eligible(b["rank_clean"], b["rank_corrupt"], b["lp_clean"], b["lp_corrupt"])
    g_a, g_b = a["lp_clean"] - a["lp_corrupt"], b["lp_clean"] - b["lp_corrupt"]
    row = {"question_id": a["question_id"], "fold": a["fold"],
           "eligible": [el_a, el_b], "eligibility_flip": el_a != el_b,
           "gap_sign_flip": _sign(g_a) != _sign(g_b), "gap_1nat_flip": pt.gap_ok(a["lp_clean"], a["lp_corrupt"])
           != pt.gap_ok(b["lp_clean"], b["lp_corrupt"]),
           "rank_clean_equal": a["rank_clean"] == b["rank_clean"], "rank_corrupt_equal": a["rank_corrupt"] == b["rank_corrupt"],
           "abs_lp_clean_diff": abs(a["lp_clean"] - b["lp_clean"]),
           "abs_lp_corrupt_diff": abs(a["lp_corrupt"] - b["lp_corrupt"]),
           "abs_gap_diff": abs(g_a - g_b), "patched": [bool(a.get("patched")), bool(b.get("patched"))]}
    if a.get("patched") and b.get("patched"):
        ra, rb = a.get("R"), b.get("R")
        if ra is None or rb is None:
            row["max_abs_R_diff"] = None if ra is None and rb is None else math.inf
        else:
            row["max_abs_R_diff"] = max(abs(x - y) for x, y in zip(ra, rb))
        da, db = a.get("d_c"), b.get("d_c")
        row["d_c_equal"] = da == db
        row["d_c_diff_layers"] = None if da is None or db is None else round((db - da) * a["n_layers"])
        row["d_c_one_none"] = (da is None) != (db is None)
        fa = pt.derive(a).get("r0_flag"), pt.derive(b).get("r0_flag")
        row["r0_flag_flip"] = fa[0] != fa[1]
    return row


def summarise(rows: list[dict], n_laptop: int, n_gpu: int) -> dict:
    both = [r for r in rows if "d_c_equal" in r]
    rmax = [r["max_abs_R_diff"] for r in both if r["max_abs_R_diff"] is not None]
    return {"n_laptop_rows": n_laptop, "n_gpu_rows": n_gpu, "n_common": len(rows), "n_both_patched": len(both),
            "n_patched_on_one_side_only": sum(r["patched"][0] != r["patched"][1] for r in rows),
            "d_c_equal": sum(r["d_c_equal"] for r in both), "d_c_unequal": [
                {"question_id": r["question_id"], "diff_layers": r["d_c_diff_layers"], "one_none": r["d_c_one_none"]}
                for r in both if not r["d_c_equal"]],
            "max_abs_R_diff_overall": max(rmax) if rmax else None,
            "median_max_abs_R_diff": sorted(rmax)[len(rmax) // 2] if rmax else None,
            "eligibility_flips": [r["question_id"] for r in rows if r["eligibility_flip"]],
            "gap_sign_flips": [r["question_id"] for r in rows if r["gap_sign_flip"]],
            "gap_1nat_flips": [r["question_id"] for r in rows if r["gap_1nat_flip"]],
            "r0_flag_flips": [r["question_id"] for r in both if r.get("r0_flag_flip")],
            "rank_clean_unequal": sum(not r["rank_clean_equal"] for r in rows),
            "rank_corrupt_unequal": sum(not r["rank_corrupt_equal"] for r in rows),
            "max_abs_lp_clean_diff": max((r["abs_lp_clean_diff"] for r in rows), default=None),
            "max_abs_lp_corrupt_diff": max((r["abs_lp_corrupt_diff"] for r in rows), default=None),
            "max_abs_gap_diff": max((r["abs_gap_diff"] for r in rows), default=None)}


def markdown(s: dict, rows: list[dict], laptop: Path, gpu: Path) -> str:
    f = lambda x: "—" if x is None else (f"{x:.4g}" if isinstance(x, float) else str(x))  # noqa: E731
    L = ["# §44 rev 8: 7B device consistency (laptop CPU bf16 vs pod CUDA bf16)", "",
         f"Laptop rows {laptop} ({s['n_laptop_rows']} on folds 2-4), GPU rows {gpu} ({s['n_gpu_rows']}); common "
         f"{s['n_common']}, both patched {s['n_both_patched']}, patched on one side only "
         f"{s['n_patched_on_one_side_only']}.", "",
         f"- d_c equal on {s['d_c_equal']} / {s['n_both_patched']} rows; unequal: {s['d_c_unequal']}",
         f"- max |R difference|: overall {f(s['max_abs_R_diff_overall'])}, median over rows "
         f"{f(s['median_max_abs_R_diff'])}",
         f"- eligibility flips (rev 5/7 filter, re-derived): {s['eligibility_flips']}",
         f"- gap-sign flips: {s['gap_sign_flips']}; 1-nat gap flips: {s['gap_1nat_flips']}; R(0)-flag flips: "
         f"{s['r0_flag_flips']}",
         f"- rank changes: clean {s['rank_clean_unequal']}, cut {s['rank_corrupt_unequal']}; max |lp diff| clean "
         f"{f(s['max_abs_lp_clean_diff'])}, cut {f(s['max_abs_lp_corrupt_diff'])}, gap {f(s['max_abs_gap_diff'])}", "",
         "| row | fold | eligible (laptop, gpu) | patched | d_c equal | max abs R diff | abs gap diff |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['question_id']} | {r['fold']} | {r['eligible']} | {r['patched']} | {r.get('d_c_equal', '—')} | "
                 f"{f(r.get('max_abs_R_diff'))} | {f(r['abs_gap_diff'])} |")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--laptop", default="runs/exp44_patch/qwen7b")
    ap.add_argument("--gpu", default="runs/exp44_patch/qwen7b_gpu")
    ap.add_argument("--out", default="runs/exp44_patch/device_check_qwen7b", help="writes <out>.json and <out>.md")
    a = ap.parse_args(argv)
    la, gp = load(Path(a.laptop)), load(Path(a.gpu))
    common = sorted(set(la) & set(gp), key=lambda q: (la[q]["fold"], q))
    rows = [compare(la[q], gp[q]) for q in common]
    s = summarise(rows, len(la), len(gp))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps({"summary": s, "rows": rows}, indent=1))
    text = markdown(s, rows, Path(a.laptop), Path(a.gpu))
    out.with_suffix(".md").write_text(text)
    print("\n".join(text.splitlines()[:9]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
