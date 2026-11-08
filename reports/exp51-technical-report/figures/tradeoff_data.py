"""Data for Figure 2 (recovery vs unsupported answering on H), from stored per-row outputs only.

No model is run. The script reads the frozen design (runs/exp51/gpu/design.json), the stored B0 and arm records of
the H split (runs/exp51/gpu_podHT), and the sealed H report (h_report.json, sha 51bbbd2e..., commit 321a4f1).

For each repair arm it recomputes the base-weighted recovery (Rel, on B0 false abstentions) and unsupported-answer
rate (Uns, on unanswerable rows) with the frozen functions in memctl/analysis/mech51.py, and asserts that both match
`R.rates_summary` in h_report.json to 1e-12. It then adds marginal 95% percentile intervals from the frozen bootstrap
(memctl/analysis/stats51.py: paired base resampling stratified by task, B = 10,000, frozen seed).

These marginal intervals are descriptive. They were not part of the frozen analysis and do not replace the
Holm-corrected O1/O2 tests, which are computed on paired differences.

Run from the repository root:  .venv/bin/python reports/exp51-technical-report/figures/tradeoff_data.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import exp51_common as CM  # noqa: E402
import exp51_stage4 as ST  # noqa: E402
from memctl.analysis import mech51 as MK  # noqa: E402
from memctl.analysis import stats51 as S  # noqa: E402

POD = ROOT / "runs/exp51/gpu_podHT"
OUT = Path(__file__).resolve().parent / "tradeoff_data.json"


def marginal(bases: S.Bases, per_base: dict[str, float]) -> dict:
    def stat(idx):
        return S.bw_mean(per_base, bases.resampled(idx))
    res = S.bootstrap(stat, bases.n_by_task())
    lo, hi = np.percentile(res["draws"], [2.5, 97.5])
    return {"est": res["est"], "lo": float(lo), "hi": float(hi), "B_valid": res["B_valid"]}


def main() -> None:
    design = json.loads((ROOT / "runs/exp51/gpu/design.json").read_text())
    report = json.loads((POD / "h_report.json").read_text())["R"]["rates_summary"]
    rows_H = CM.rows("H", allow_sealed=True)
    labels, meta = ST.labels_meta(POD, "H", rows_H)
    bases = MK.bases_of(meta)

    rc = design["R"]["candidate"]
    specs = {"B0": {"arm": "B0"},
             "candidate": {"arm": "EDIT", "edit_id": rc["edit_id"], "dose": rc["dose"], "tag": f"sub{rc['index']}"}}
    specs.update({k: v["spec"] for k, v in design["R"]["comparator_doses"].items()})

    out = {}
    for name, spec in specs.items():
        graded = ST.total_graded(POD, "H", spec, rows_H)
        rates = MK.rates(graded, labels, meta)
        entry = {}
        for r in ("Rel", "Uns", "Dam"):
            m = marginal(bases, rates[r])
            assert abs(m["est"] - report[name][r]) < 1e-12, (name, r, m["est"], report[name][r])
            entry[r] = m
        fa = [c for c in graded if "FA" in labels[c]]
        una = [c for c in graded if "UNA" in labels[c]]
        entry["counts"] = {"FA": len(fa), "FA_gold": sum(graded[c]["cls"] == "gold" for c in fa),
                           "UNA": len(una), "FA_ab": sum("FA-ab" in labels[c] for c in fa)}
        out[name] = entry
        print(name, {r: round(entry[r]["est"], 3) for r in ("Rel", "Uns", "Dam")}, entry["counts"])
    OUT.write_text(json.dumps(out, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
