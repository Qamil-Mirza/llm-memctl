#!/usr/bin/env python
"""§47 report: B0-B4 computed from the per-row files, never typed.

    .venv/bin/python scripts/exp47_report.py --root runs/exp47_bridge/cuda-float32

Spec: research/proposals/2026-10-31-exp47-bridge.md rev 0 (be2b9e5) §3-§4 + rev 1 (a9b70ff). Pooled read: every row
was read for other measurements under §44 / §45 (rev 0 §1 honesty clause); the thresholds are §46's. Writes
<root>/report.json and report.md.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from memctl.analysis import bridge47 as B  # noqa: E402


def load(root: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted((root / "rows").glob("*.json"))]


def flags_by_layer(recs, kind: str, pred) -> dict[int, list[bool]]:
    out = defaultdict(list)
    for r in recs:
        for x in r[kind]:
            out[x["layer"]].append(bool(pred(x)))
    return dict(out)


def transferred(x) -> bool:
    return x["T"] is not None and x["T"] >= B.T_THRESHOLD


def class_read(recs: list[dict]) -> dict:
    t = B.share_profile(flags_by_layer(recs, "patch", transferred))
    k = B.share_profile(flags_by_layer(recs, "k1", transferred))
    status = B.class_status(len(recs))
    return {"n": len(recs), "status": status, "transfer": t, "k1": k,
            "l_star": B.l_star(t, k) if status != "descriptive" and recs else None}


def outcome_shift(recs: list[dict], target) -> dict:
    """Per layer: share of patched generations with outcome in `target`, minus the same share under k1."""
    p = B.share_profile(flags_by_layer(recs, "patch", lambda x: x.get("outcome") in target))
    k = B.share_profile(flags_by_layer(recs, "k1", lambda x: x.get("outcome") in target))
    return {l: {"patch": p[l]["share"], "k1": k[l]["share"],
                "diff": None if p[l]["share"] is None else p[l]["share"] - k[l]["share"]} for l in p}


def build(root: Path) -> dict:
    recs = load(root)
    ent = [r for r in recs if r["entered"]]
    suc = [r for r in ent if r["label"] == "success"]
    fail = [r for r in ent if r["label"] != "success"]
    k3 = [r["k3"]["reproduces_swap"] for r in ent]
    k3p = B.wilson(sum(k3), len(k3))
    rep = {"n_rows": len(recs), "n_entered": len(ent), "pooled_read": "pre-registered thresholds (§46), pooled rows "
           "previously read for other measurements (§44, §45)",
           "B0": {"k": sum(k3), "n": len(k3), "share": k3p[0], "lower": k3p[1], "pass": k3p[0] is not None and k3p[0] >= B.K3_MIN},
           "sanity": {"t_last_fail": sum(not r["sanity"]["t_last_ok"] for r in recs),
                      "t_k3_fail": sum(not r["sanity"]["t_k3_ok"] for r in recs),
                      "delta_fail": sum(not r["sanity"]["delta_ok"] for r in recs)}}
    cs, cf = class_read(suc), class_read(fail)
    b1_yes = cs["status"] != "descriptive" and cs["l_star"] is not None and cs["l_star"] >= B.MID_STACK_L
    no_multi = class_read([r for r in suc if not r["multi_occurrence"]])
    rep["B1"] = {**cs, "reading": B.b1_reading(cs["l_star"], cs["status"]),
                 "without_multi_occurrence": {"n": no_multi["n"], "l_star_descriptive": B.l_star(no_multi["transfer"], no_multi["k1"]) if no_multi["n"] else None}}
    rep["B2"] = {**cf, "reading": B.b2_reading(cs["l_star"], cf["l_star"], cf["status"], b1_yes)}
    ref = [r for r in recs if r["label"] == "refusal"]
    ref_ent = [r for r in ref if r["entered"]]
    ref_ent_unknown = [r for r in ref_ent if r["baseline"]["clean"]["outcome"] == "unknown"]
    rep["B3"] = {"status": "read" if len(ref_ent) >= B.MIN_ROWS else "descriptive",
                 "refusal_rows_entered": len(ref_ent), "refusal_rows_all": len(ref),
                 "refusal_rows_entered_clean_unknown": len(ref_ent_unknown),
                 "refusal_to_answer_entered": outcome_shift(ref_ent, {"G", "S"}),
                 "refusal_to_answer_entered_clean_unknown": outcome_shift(ref_ent_unknown, {"G", "S"}),
                 "refusal_to_answer_all": outcome_shift(ref, {"G", "S"}),
                 "success_induced_abstention": outcome_shift(suc, {"unknown"})}
    per = defaultdict(list)
    for r in ent:
        lr = B.row_l_star([x["T"] for x in r["patch"]])
        per[("label", r["label"])].append(lr)
        per[("cls1", r["cls1"])].append(lr)
    rep["B4"] = {"row_l_star_median": {f"{a}:{b}": (statistics.median([v if v is not None else -1 for v in vals]), len(vals))
                                       for (a, b), vals in sorted(per.items())},
                 "s2": s2_split(ent)}
    rep["baseline_outcomes"] = {f"{lab}|{'entered' if e else 'not_entered'}|{kind}":
                                dict(Counter(r["baseline"][kind]["outcome"] for r in recs
                                             if r["label"] == lab and r["entered"] == e))
                                for lab in ("success", "refusal", "wrong") for e in (True, False)
                                for kind in ("clean", "swap")}

    def share(rs, want):
        k = sum(r["baseline"]["clean"]["outcome"] == want for r in rs)
        return {"k": k, "n": len(rs), "share": k / len(rs) if rs else None}
    rep["label_consistency"] = {
        "refusal_clean_unknown_all": share(ref, "unknown"), "refusal_clean_unknown_entered": share(ref_ent, "unknown"),
        "success_clean_G_all": share([r for r in recs if r["label"] == "success"], "G"),
        "success_clean_G_entered": share(suc, "G")}
    rep["outcomes_by_layer_entered"] = {lab: {l: dict(Counter(x["outcome"] for r in ent if r["label"] == lab
                                                              for x in r["patch"] if x["layer"] == l))
                                              for l in range(B.N_LAYERS)}
                                        for lab in ("success", "refusal", "wrong")}
    rep["not_entered_outcomes"] = {l: dict(Counter(x["outcome"] for r in recs if not r["entered"] for x in r["patch"]
                                                   if x["layer"] == l)) for l in range(B.N_LAYERS)}
    return rep


def s2_split(ent):
    out = {}
    for kind in ("patch", "k1"):
        fl = defaultdict(list)
        for r in ent:
            for x in r["s2"]:
                if x["kind"] == kind:
                    fl[x["layer"]].append(transferred(x))
        out[kind] = {l: v["share"] for l, v in B.share_profile(dict(fl)).items()}
    fwd = defaultdict(list)
    for r in ent:
        t = r.get("T_fwd44") or []
        for l in B.S2_LAYERS:
            if l < len(t) and t[l] is not None:
                fwd[l].append(t[l] >= B.T_THRESHOLD)
    out["s44_fwd"] = {l: v["share"] for l, v in B.share_profile(dict(fwd)).items()}
    return out


def markdown(rep: dict) -> str:
    f = lambda v: "—" if v is None else f"{v:.3f}"
    b0 = rep["B0"]
    lines = ["# §47 report (pooled read; §46 thresholds)", "",
             f"Rows {rep['n_rows']}, entered {rep['n_entered']}. Sanity fails: T(35) {rep['sanity']['t_last_fail']}, "
             f"T(k3) {rep['sanity']['t_k3_fail']}, Δ vs §44 {rep['sanity']['delta_fail']}.", "",
             f"**B0** k3 {b0['k']}/{b0['n']} = {f(b0['share'])} (LB {f(b0['lower'])}), pass {b0['pass']}.",
             f"**B1** success entered n {rep['B1']['n']} ({rep['B1']['status']}), L* {rep['B1']['l_star']}: {rep['B1']['reading']}. "
             f"Without the multi-occurrence rows (descriptive): n {rep['B1']['without_multi_occurrence']['n']}, "
             f"L* {rep['B1']['without_multi_occurrence']['l_star_descriptive']}.",
             f"**B2** failure entered n {rep['B2']['n']} ({rep['B2']['status']}), L* {rep['B2']['l_star']}: {rep['B2']['reading']}.",
             f"**B3** {rep['B3']['status']} (refusal rows entered {rep['B3']['refusal_rows_entered']} of {rep['B3']['refusal_rows_all']}; "
             f"of those, clean baseline 'unknown' under this readout: {rep['B3']['refusal_rows_entered_clean_unknown']}).",
             "", "Label consistency (this readout vs the §23 label): " + "; ".join(
                 f"{k} {v['k']}/{v['n']}" for k, v in rep["label_consistency"].items()),
             "", "Baseline outcomes (clean / swap generations) by label and entry:", "```",
             json.dumps(rep["baseline_outcomes"], indent=1), "```",
             "", "| layer | T succ | LB | k1 succ | T fail | LB | k1 fail |", "|---|---|---|---|---|---|---|"]
    for l in range(B.N_LAYERS):
        s, sk = rep["B1"]["transfer"].get(l, {}), rep["B1"]["k1"].get(l, {})
        g, gk = rep["B2"]["transfer"].get(l, {}), rep["B2"]["k1"].get(l, {})
        lines.append(f"| {l} | {f(s.get('share'))} | {f(s.get('lower'))} | {f(sk.get('share'))} | {f(g.get('share'))} | "
                     f"{f(g.get('lower'))} | {f(gk.get('share'))} |")
    lines += ["", "## B3 (descriptive unless >= 8 refusal rows enter)", "",
              "| layer | refusal -> G/S, patch - k1 (entered) | same, clean baseline unknown only | success -> unknown, patch - k1 |",
              "|---|---|---|---|"]
    for l in range(B.N_LAYERS):
        a = rep["B3"]["refusal_to_answer_entered"].get(l, {})
        c = rep["B3"]["refusal_to_answer_entered_clean_unknown"].get(l, {})
        b = rep["B3"]["success_induced_abstention"].get(l, {})
        lines.append(f"| {l} | {f(a.get('diff'))} | {f(c.get('diff'))} | {f(b.get('diff'))} |")
    s2 = rep["B4"]["s2"]
    lines += ["", "## B4 (descriptive)", "", "Per-row L* medians (-1 = none) and n:", "```",
              json.dumps(rep["B4"]["row_l_star_median"], indent=1), "```", "",
              "S2 at P1 (share T >= 0.5): layer: patch / k1 / §44 forward patch",
              " ".join(f"{l}:{f(s2['patch'].get(l))}/{f(s2['k1'].get(l))}/{f(s2['s44_fwd'].get(l))}" for l in B.S2_LAYERS)]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    args = ap.parse_args()
    rep = build(args.root)
    (args.root / "report.json").write_text(json.dumps(rep, indent=1, default=str))
    (args.root / "report.md").write_text(markdown(rep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
