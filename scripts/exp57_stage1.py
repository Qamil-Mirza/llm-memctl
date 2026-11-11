#!/usr/bin/env python
"""§57 stage 1 (proposal rev 0, 9e51a1b, §5 stage 1): Goodfire-style mapping on the stored §56 dev-A reads
(runs/exp56/run/stage1/*.npz: activation a and direct push DLA of all 396,288 neurons at P_ans). No model calls.
Descriptive only.

Fixed here:
- Strong-firing share, row threshold: on each FA row, the neurons whose |a| is at or above the row's own 97th
  percentile of |a| (over its 396,288 neurons). Share = their summed positive DLA / the row's summed positive DLA.
  Reported as the mean over FA rows and pooled (sums over rows).
- Strong-firing share, neuron threshold: a neuron is strong on a row if its |a| exceeds its own 97th percentile of |a|
  over all 743 dev-A rows. Same shares.
- Differential firing: d_i = (mean_FA − mean_C) / pooled SD (n−1 weights) of a_i at P_ans; counts at |d| >= 0.5 and
  >= 1.0, split by sign, with layer histograms. Neurons with pooled SD 0 get d = 0.
- Base × hardness matched: cells (base_key, hardness) with at least one FA and one C row; matched difference = the
  mean over cells of (cell FA mean − cell C mean), scaled by the same pooled SD (cells hold about 1 FA and 1 C row, so
  no within-cell SD exists). Same counts, plus the agreement with the unmatched d.
Output: runs/exp57/stage1_report.json.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import exp56_common as X56  # noqa: E402

OUT = ROOT / "runs/exp57"
IN = ROOT / "runs/exp56/run/stage1"
INTER, NL = 11008, 36


def load():
    import exp56_stage1 as S1
    rows = S1.dev_a_rows()
    hard = {c["case_id"]: c["hardness"] for c in X56.cases()}
    metas, A, DL = [], np.empty((len(rows), NL * INTER), np.float32), np.empty((len(rows), NL * INTER), np.float32)
    for k, x in enumerate(rows):
        z = np.load(IN / f"{x['case_id']}.npz")
        m = json.loads(str(z["meta"]))
        assert m["case_id"] == x["case_id"] and m["cls"] == x["cls"]
        m["hardness"] = hard[x["case_id"]]
        metas.append(m)
        A[k] = z["a"].reshape(-1); DL[k] = z["dla"].reshape(-1)
    return metas, A, DL


def counts(d, thr):
    sel = np.abs(d) >= thr
    return {"n": int(sel.sum()), "FA_higher": int((d >= thr).sum()), "C_higher": int((d <= -thr).sum()),
            "layer_hist": np.bincount(np.where(sel)[0] // INTER, minlength=NL).tolist()}


def main():
    metas, A, DL = load()
    cls = np.array([m["cls"] for m in metas])
    fa, cc = np.where(cls == "FA")[0], np.where(cls == "C")[0]
    rep: dict = {"n_rows": {c: int((cls == c).sum()) for c in X56.CLASSES}}

    # strong-firing share (row threshold and neuron threshold)
    absA = np.abs(A)
    thr_n = np.percentile(absA, 97, axis=0).astype(np.float32)               # per neuron, over all 743 rows
    sh_r, sh_n, num_r, num_n, den, frac_n = [], [], 0.0, 0.0, 0.0, []
    for i in fa:
        pos = np.clip(DL[i].astype(np.float64), 0, None)
        s_row = absA[i] >= np.percentile(absA[i], 97)
        s_neu = absA[i] > thr_n
        P = pos.sum()
        sh_r.append(pos[s_row].sum() / P); sh_n.append(pos[s_neu].sum() / P)
        num_r += pos[s_row].sum(); num_n += pos[s_neu].sum(); den += P
        frac_n.append(s_neu.mean())
    rep["strong_firing_share"] = {
        "row_p97": {"mean_over_FA_rows": float(np.mean(sh_r)), "pooled": float(num_r / den),
                    "q10_q50_q90": np.percentile(sh_r, [10, 50, 90]).round(4).tolist(), "neuron_fraction": 0.03},
        "neuron_p97": {"mean_over_FA_rows": float(np.mean(sh_n)), "pooled": float(num_n / den),
                       "q10_q50_q90": np.percentile(sh_n, [10, 50, 90]).round(4).tolist(),
                       "neuron_fraction_mean": float(np.mean(frac_n))},
        "note": "share of the row's summed positive DLA carried by strongly firing neurons; 'neuron_fraction' = the "
                "fraction of neurons counted strong on an FA row (a share equal to it = no enrichment)"}
    del absA

    # differential firing, unmatched
    mF, mC = A[fa].mean(0, dtype=np.float64), A[cc].mean(0, dtype=np.float64)
    vF, vC = A[fa].var(0, ddof=1, dtype=np.float64), A[cc].var(0, ddof=1, dtype=np.float64)
    sd = np.sqrt(((len(fa) - 1) * vF + (len(cc) - 1) * vC) / (len(fa) + len(cc) - 2))
    d = np.where(sd > 0, (mF - mC) / np.where(sd > 0, sd, 1), 0.0)
    rep["differential"] = {"d_ge_0.5": counts(d, 0.5), "d_ge_1.0": counts(d, 1.0),
                           "max_abs_d": float(np.abs(d).max()),
                           "top10": [{"layer": int(k // INTER), "neuron": int(k % INTER), "d": float(d[k]),
                                      "mean_FA": float(mF[k]), "mean_C": float(mC[k])}
                                     for k in np.argsort(-np.abs(d))[:10]]}

    # base x hardness matched
    cell = defaultdict(lambda: {"FA": [], "C": []})
    for i in np.concatenate([fa, cc]):
        cell[(metas[i]["base_key"], metas[i]["hardness"])][cls[i]].append(i)
    cells = [v for v in cell.values() if v["FA"] and v["C"]]
    diff = np.zeros(A.shape[1], np.float64)
    for v in cells:
        diff += A[v["FA"]].mean(0, dtype=np.float64) - A[v["C"]].mean(0, dtype=np.float64)
    diff /= len(cells)
    dm = np.where(sd > 0, diff / np.where(sd > 0, sd, 1), 0.0)
    top_u, top_m = set(np.argsort(-np.abs(d))[:1000]), set(np.argsort(-np.abs(dm))[:1000])
    rep["differential_matched"] = {
        "n_cells": len(cells), "n_FA_in_cells": sum(len(v["FA"]) for v in cells),
        "n_C_in_cells": sum(len(v["C"]) for v in cells),
        "d_ge_0.5": counts(dm, 0.5), "d_ge_1.0": counts(dm, 1.0), "max_abs_d": float(np.abs(dm).max()),
        "corr_with_unmatched_d": float(np.corrcoef(d, dm)[0, 1]),
        "top1000_overlap_with_unmatched": len(top_u & top_m),
        "scale": "pooled SD of the unmatched FA and C rows"}

    # link to §56: direct push of the differential neurons, and overlap with §56 K
    K56 = [x["flat"] for x in json.loads((ROOT / "runs/exp56/K.json").read_text())["K"]]
    m_dla = DL[fa].mean(0, dtype=np.float64)
    pos = np.clip(m_dla, 0, None)
    sel = np.abs(d) >= 0.5
    rep["link_exp56"] = {"K56_with_abs_d_ge_0.5": int(sel[K56].sum()),
                         "share_of_positive_mean_FA_push_on_abs_d_ge_0.5": float(pos[sel].sum() / pos.sum()),
                         "neuron_fraction_abs_d_ge_0.5": float(sel.mean())}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "stage1_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items()}, indent=1)[:6000])


if __name__ == "__main__":
    main()
