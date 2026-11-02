#!/usr/bin/env python
"""§49 report: baseline outcomes, M1 decoding, Q-P and Q-2 to Q-6, all computed from the per-trial files.

    .venv/bin/python scripts/exp49_report.py --dev        # dev baselines + dev design only (no held file is opened)
    .venv/bin/python scripts/exp49_report.py              # the held read (refuses without the committed design)

Spec: research/proposals/2026-11-01-exp49-legible-abstentions.md rev 0a §3-§5 (D-63). Every interval is a paired,
base-clustered percentile bootstrap (10k resamples, seed 0; abst49.cluster_boot). Pools (held, baseline-defined):
false abstentions = ANS-F2 rows whose baseline is "unknown"; genuine = UNA-F2 rows whose baseline is "unknown"; the
abstention pool for U is their union at its natural held proportions. "Wrong" for decoding and the F1 arm = another
pool city (wrong or competitor). The §5 reading is computed from the Q-P intervals (rule in `reading`).
Writes <out>/report.json and report.md (or report_dev.*).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from memctl.analysis import abst49 as A  # noqa: E402
import exp49_run as R  # noqa: E402

WRONGS = ("wrong", "competitor")


def baseline_table(rows, root, split) -> dict:
    t = {}
    for r in rows:
        if r["split"] != split:
            continue
        b = R.baseline(root, r["case_id"])
        if b is None:
            continue
        key = f"{r['set']}|{r['format']}|{r['task']}|{r['hardness']}"
        t.setdefault(key, Counter())[b["outcome"]] += 1
    tot = {}
    for k, c in t.items():
        s, f = k.split("|")[:2]
        tot.setdefault(f"{s}|{f}", Counter()).update(c)
    return {"cells": {k: dict(v) for k, v in sorted(t.items())}, "totals": {k: dict(v) for k, v in sorted(tot.items())}}


def rate(keys, hits) -> dict:
    return A.cluster_boot(keys, [float(h) for h in hits])


def diff(keys, a, b) -> dict:
    return A.cluster_boot(keys, [float(x) for x in a], [float(x) for x in b])


def arm(root, cid, name) -> dict | None:
    return R.read_json(root / "arms_held" / f"{cid}__{name}.json")


def reading(gold_gain: dict, unsup_gain: dict, r1_estimable: bool = True) -> str:
    """Rev 0b B6, computed: R1 not estimable -> said so; R1 − R3 interval not above 0 -> 'no recovery detected at this
    site and strength'; else direction-specific recovery, with the unsupported-answer cost reported descriptively."""
    if not r1_estimable:
        return "R1 not estimable (fewer than 20 dev abstentions for μ_abst); R2-R4 are read"
    if gold_gain["lo"] is None or gold_gain["lo"] <= 0:
        return "no recovery detected at this site and strength"
    u = "n/a" if unsup_gain.get("est") is None else f"{unsup_gain['est']:+.3f} [{unsup_gain['lo']:+.3f}, {unsup_gain['hi']:+.3f}]"
    return ("direction-specific recovery (R1 against random) on false abstentions; unsupported-answer change on "
            f"genuine abstentions, descriptive: {u}")


def q6_reading(lf_gain: dict) -> str:
    """B6: Q-6 is interpretable only if the held Legible Failures F1 repair gain has its 95% CI above 0."""
    if lf_gain.get("lo") is None or lf_gain["lo"] <= 0:
        return "contrast not interpretable: calibration did not reproduce"
    return "interpretable (calibration reproduced); the difference is descriptive"


def exclusions(rows, root) -> dict:
    """B3: per split x set x format, the baseline class of every trial with no P_ans, by reason."""
    out = {}
    for r in rows:
        if r["format"] != "F2":
            continue
        b = R.baseline(root, r["case_id"])
        if b is None or b["p_ans_ok"]:
            continue
        key = f"{r['split']}|{r['set']}|{b['outcome']}|{b['p_ans_why']}"
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


def decoding(rows, root, P, design) -> dict:
    """pAcc on held subsets at ℓ*_pre (P_pre), ℓ*_F1 (F1 rows) and ℓ*_ans (P_ans, split by note-gold), with the
    1/12, in-prompt candidate and label-shuffled floors; acc − floor intervals paired per trial."""
    held = [r for r in rows if r["split"] == "held" and r["set"] == "ANS"]
    out = {}
    subsets = {"F2-abstain": ("F2", ("abstain",)), "F2-gold": ("F2", ("gold",)), "F2-wrong": ("F2", WRONGS),
               "F1-wrong": ("F1", WRONGS), "F1-gold": ("F1", ("gold",))}
    for site, model, shuf, layer, idx in (("P_pre", "pre", "pre", design["layer_pre"], 0),
                                          ("P_pre_F1", "f1", None, design["layer_f1"], 0),
                                          ("P_ans", "ans", "ans", design["layer_ans"], 1)):
        for name, (fmt, outs) in subsets.items():
            if site == "P_ans" and fmt == "F1":
                continue
            if site == "P_pre_F1" and fmt == "F2":
                continue
            for flag in ((None,) if site != "P_ans" else (None, True, False)):
                rs = []
                for r in held:
                    b = R.baseline(root, r["case_id"])
                    if r["format"] != fmt or b["outcome"] not in outs:
                        continue
                    if site == "P_ans" and not b["p_ans_ok"]:
                        continue
                    if flag is not None and b.get("note_gold") != flag:
                        continue
                    rs.append(r)
                if not rs:
                    continue
                X = np.stack([R.resid(root, r["case_id"])[idx, layer] for r in rs])
                pred = P["models"][model].predict(X)
                ok = [str(p) == r["gold"] for p, r in zip(pred, rs)]
                keys = [r["base_key"] for r in rs]
                cand = [1.0 / max(1, len(r["cities_in_prompt"])) for r in rs]
                rec = {"n": len(rs), "layer": layer, "acc": rate(keys, ok),
                       "minus_chance": A.cluster_boot(keys, [float(o) - 1 / 12 for o in ok]),
                       "minus_candidate": A.cluster_boot(keys, [float(o) - c for o, c in zip(ok, cand)]),
                       "candidate_floor": float(np.mean(cand))}
                if shuf:
                    sp = P["shuffled"][shuf].predict(X)
                    sok = [str(p) == r["gold"] for p, r in zip(sp, rs)]
                    rec["shuffled_acc"] = float(np.mean(sok))
                    rec["minus_shuffled"] = diff(keys, ok, sok)
                label = name + ("" if flag is None else f"|note_gold={flag}")
                out[f"{site}|{label}"] = rec
    return out


def held_report(rows, root) -> dict:
    design = A.require_design(root)
    P = R.load_probes(root)
    by = {r["case_id"]: r for r in rows}
    held = [r for r in rows if r["split"] == "held"]
    B = {r["case_id"]: R.baseline(root, r["case_id"]) for r in held}
    noops = [arm(root, r["case_id"], "noop") for r in R.noop_rows(rows, root, "held")]
    false_ab = [r for r in held if r["set"] == "ANS" and r["format"] == "F2" and B[r["case_id"]]["outcome"] == "abstain"
                and B[r["case_id"]]["p_ans_ok"]]
    genuine = [r for r in held if r["set"] == "UNA" and B[r["case_id"]]["outcome"] == "abstain"
               and B[r["case_id"]]["p_ans_ok"]]
    rep: dict = {"design": design, "design_sha256": A.design_sha(design),
                 "baselines_held": baseline_table(rows, root, "held"), "exclusions": exclusions(rows, root),
                 "noop_held": f"{sum(x['reproduces'] for x in noops)}/{len(noops)}",
                 "n_false_abstentions": len(false_ab), "n_genuine_abstentions": len(genuine),
                 "false_abst_under_min": len(false_ab) < A.HELD_FALSE_ABST_MIN}

    def outs(pop, name):
        return [arm(root, r["case_id"], name)["outcome"] for r in pop]
    # per-arm outcome rates on both populations
    rates = {}
    for popname, pop, arms in (("false", false_ab, A.ARMS), ("genuine", genuine, ("R1", "R2", "R3", "R4"))):
        keys = [r["base_key"] for r in pop]
        for a in arms:
            o = outs(pop, a)
            st = Counter(arm(root, r["case_id"], a).get("status", "ok") for r in pop)
            rates[f"{popname}|{a}"] = {c: rate(keys, [x == c for x in o])["est"] for c in A.CLASSES} | {
                "status": dict(st),
                "n": len(o), "unsupported": rate(keys, [x in ("wrong", "competitor", "degenerate") for x in o]),
                "gold": rate(keys, [x == "gold" for x in o])}
    rep["arm_rates"] = rates
    # Q-P and Q-2
    kf, kg = [r["base_key"] for r in false_ab], [r["base_key"] for r in genuine]
    gold = {a: [x == "gold" for x in outs(false_ab, a)] for a in A.ARMS}
    unsup = {a: [x in ("wrong", "competitor", "degenerate") for x in outs(genuine, a)] for a in ("R1", "R2", "R3", "R4")}
    qp_gold, qp_uns = diff(kf, gold["R1"], gold["R3"]), diff(kg, unsup["R1"], unsup["R3"])
    rep["Q-P"] = {"gold_R1_minus_R3_false": qp_gold, "unsupported_R1_minus_R3_genuine": qp_uns,
                  "reading": reading(qp_gold, qp_uns, design.get("r1_estimable", True))}
    rep["Q-2"] = {f"gold_{a}_minus_{b}_false": diff(kf, gold[a], gold[b]) for a, b in
                  (("R1", "R2"), ("R1", "R4"), ("R2", "R3"), ("R4", "R3"))} | {
                  f"unsupported_{a}_minus_{b}_genuine": diff(kg, unsup[a], unsup[b]) for a, b in
                  (("R1", "R2"), ("R1", "R4"), ("R2", "R3"), ("R4", "R3"))}
    correct = [r for r in held if r["set"] == "ANS" and r["format"] == "F2" and B[r["case_id"]]["outcome"] == "gold"]
    dmg = [arm(root, r["case_id"], "R4")["outcome"] != "gold" for r in correct]
    rep["Q-2"]["R4_damage_on_correct"] = rate([r["base_key"] for r in correct], dmg) | {
        "outcomes": dict(Counter(arm(root, r["case_id"], "R4")["outcome"] for r in correct))}
    rep["Q-2"]["Dgold_gold_false"] = rate(kf, gold["Dgold"])
    # Q-3 tracking (B2): H1 bases whose ANS-H1 and CF-H1 baselines both abstain; own denominators, no R1-R2 compare
    cf_ids = R.m2b_cf(held, B)
    q3 = {"n_ans_h1": sum(1 for r in held if r["set"] == "ANS" and r["format"] == "F2" and r["hardness"] == "H1"),
          "n_cf_h1": sum(1 for r in held if r["set"] == "CF"), "n_bases_population": len(cf_ids),
          "note": "R1 and R2 have different conditional populations and are not compared"}
    ans_h1 = {r["base_key"]: r for r in held if r["set"] == "ANS" and r["format"] == "F2" and r["hardness"] == "H1"}
    for a in ("R1", "R2"):
        recs = [(by[c], arm(root, c, a)) for c in cf_ids]
        released = [(r, x) for r, x in recs if x["outcome"] != "abstain"]
        hits = [x["item"] == r["cf_city"] for r, x in released]
        ans_gold = [arm(root, ans_h1[r["base_key"]]["case_id"], a)["outcome"] == "gold" for r, _ in recs]
        q3[f"{a} (one-position)" if a == "R2" else a] = {
            "n_bases": len(recs), "n_released": len(released), "n_cf_city": sum(hits), "n_ans_gold": sum(ans_gold),
            "tracking_among_released": rate([r["base_key"] for r, _ in released], hits),
            "cf_city_rate_all": rate([r["base_key"] for r, _ in recs], [x["item"] == r["cf_city"] for r, x in recs])}
    rep["Q-3"] = q3
    # Q-4 gates on the union pool
    f_h1 = [r for r in false_ab if r["hardness"] == "H1"]
    g_h1 = [r for r in genuine if r["hardness"] == "H1"]
    pool = f_h1 + g_h1                                   # B1: gates on H1 abstentions only
    kp = [r["base_key"] for r in pool]
    la, ls = design["layer_ans"], design["layer_suff"]
    Xa = np.stack([R.resid(root, r["case_id"])[1, la] for r in pool])
    Xs = np.stack([R.resid(root, r["case_id"])[0, ls] for r in pool])
    cmax = P["models"]["ans"].predict_proba(Xa).max(1)
    sm = P["models"]["suff"]
    sp = sm.predict_proba(Xs)[:, list(sm.classes_).index(1)]
    r1 = [arm(root, r["case_id"], "R1")["outcome"] for r in pool]
    never = ["abstain"] * len(pool)
    pol = {"never": never, "always_R1": r1,
           "G_content": A.policy_outcomes(never, r1, cmax >= design["tau_c"]),
           "G_suff": A.policy_outcomes(never, r1, sp >= design["tau_s"])}
    nf = len(f_h1)
    q4 = {"n_false_h1": nf, "n_genuine_h1": len(g_h1),
          "fire_rate": {"G_content": {"false": float(np.mean(cmax[:nf] >= design["tau_c"])) if nf else None,
                                      "genuine": float(np.mean(cmax[nf:] >= design["tau_c"])) if g_h1 else None},
                        "G_suff": {"false": float(np.mean(sp[:nf] >= design["tau_s"])) if nf else None,
                                   "genuine": float(np.mean(sp[nf:] >= design["tau_s"])) if g_h1 else None}},
          "suff_auroc_h1_abstentions": A.auroc(sp, [1] * nf + [0] * len(g_h1)),
          "claim_limit": "templates only; not general recognition of evidence sufficiency"}
    for lam in A.LAMBDAS:
        u = {k: [A.utility(o, lam) for o in v] for k, v in pol.items()}
        q4[f"lambda_{lam:g}"] = {"U": {k: float(np.mean(v)) for k, v in u.items()},
                                 **{f"{g}_minus_{h}": diff(kp, u[g], u[h]) for g, h in
                                    (("G_suff", "never"), ("G_content", "never"), ("G_suff", "always_R1"),
                                     ("G_content", "always_R1"), ("always_R1", "never"))}}
    rep["Q-4"] = q4
    # Q-5 F1 replication
    f1 = [r for r in held if r["format"] == "F1"]
    f1w = [r for r in f1 if B[r["case_id"]]["outcome"] != "gold"]      # B3: no-city trials stay in, unrepaired
    f1g = [r for r in f1 if B[r["case_id"]]["outcome"] == "gold"]
    lf = {r["case_id"]: arm(root, r["case_id"], "LF") for r in f1}
    lr = {r["case_id"]: arm(root, r["case_id"], "LFrand") for r in f1}
    kw = [r["base_key"] for r in f1w]
    lf_gain = diff(kw, [lf[r["case_id"]]["outcome"] == "gold" for r in f1w],
                   [lr[r["case_id"]]["outcome"] == "gold" for r in f1w])
    rep["Q-5"] = {"n_f1_wrong": len(f1w), "n_f1_gold": len(f1g),
                  "fired_wrong": sum(lf[r["case_id"]]["fired"] for r in f1w),
                  "fired_gold": sum(lf[r["case_id"]]["fired"] for r in f1g),
                  "LF_minus_random_gold_on_wrong": lf_gain,
                  "LF_gold_on_wrong": rate(kw, [lf[r["case_id"]]["outcome"] == "gold" for r in f1w]),
                  "kept_correct_on_gold": rate([r["base_key"] for r in f1g],
                                               [lf[r["case_id"]]["outcome"] == "gold" for r in f1g])}
    # decoding
    dec = decoding(rows, root, P, design)
    rep["M1"] = dec
    # Q-6
    a_keys, a_vals = kf, [float(x) - float(y) for x, y in zip(gold["R1"], gold["R3"])]
    b_vals = [float(lf[r["case_id"]]["outcome"] == "gold") - float(lr[r["case_id"]]["outcome"] == "gold") for r in f1w]
    rep["Q-6"] = {"reading": q6_reading(lf_gain), "repair_gain_false_abst": qp_gold, "repair_gain_f1_wrong": lf_gain,
                  "difference": A.boot_diff_unpaired(a_keys, a_vals, kw, b_vals),
                  "pAcc_P_pre_F2_abstain": dec.get("P_pre|F2-abstain", {}).get("acc"),
                  "pAcc_P_pre_F1_wrong": dec.get("P_pre|F1-wrong", {}).get("acc")}
    # breakdowns (descriptive)
    bd = {}
    for field in ("task", "hardness"):
        for v in sorted({r[field] for r in false_ab}):
            idx = [i for i, r in enumerate(false_ab) if r[field] == v]
            bd[f"{field}={v}"] = {"n": len(idx), **{a: float(np.mean([gold[a][i] for i in idx])) for a in A.ARMS}}
    for flag in (True, False):
        idx = [i for i, r in enumerate(false_ab) if B[r["case_id"]].get("note_gold") == flag]
        if idx:
            bd[f"note_gold={flag}"] = {"n": len(idx), **{a: float(np.mean([gold[a][i] for i in idx])) for a in A.ARMS}}
    rep["breakdowns_false_gold_rate"] = bd
    return rep


def fmt_ci(d) -> str:
    if not d or d.get("est") is None:
        return "n/a"
    return f"{d['est']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}] (n {d.get('n', '?')}, bases {d.get('n_bases', '?')})"


def markdown(rep: dict) -> str:
    L = ["# §49 legible abstentions: held read (computed)", "",
         f"Design sha256 `{rep['design_sha256']}`; α {rep['design']['alpha']}, ℓ* P_pre {rep['design']['layer_pre']}, "
         f"P_ans {rep['design']['layer_ans']}, F1 {rep['design']['layer_f1']}, suff {rep['design']['layer_suff']}; "
         f"τ_c {rep['design']['tau_c']}, τ_s {rep['design']['tau_s']}.", "",
         f"Held false abstentions {rep['n_false_abstentions']}, genuine {rep['n_genuine_abstentions']}"
         + (" (under 30: read with its width, no rescue)" if rep["false_abst_under_min"] else "") + ".", "",
         f"No-op replays (held): {rep['noop_held']}. Exclusions (no P_ans): {json.dumps(rep['exclusions'])}.", "",
         "## Q-P (primary)", f"- gold R1 − R3 on false abstentions: {fmt_ci(rep['Q-P']['gold_R1_minus_R3_false'])}",
         f"- unsupported R1 − R3 on genuine abstentions: {fmt_ci(rep['Q-P']['unsupported_R1_minus_R3_genuine'])}",
         f"- reading: {rep['Q-P']['reading']}", "", "## Q-2"]
    L += [f"- {k}: {fmt_ci(v)}" for k, v in rep["Q-2"].items() if isinstance(v, dict) and "est" in v]
    L += ["", "## Q-3 tracking (H1; own denominators; R1 and R2 not compared)"]
    for a in ("R1", "R2 (one-position)"):
        q = rep["Q-3"][a]
        L.append(f"- {a}: bases {q['n_bases']}, released {q['n_released']}, CF city {q['n_cf_city']}, ANS gold "
                 f"{q['n_ans_gold']}; tracking among released {fmt_ci(q['tracking_among_released'])}")
    L += ["", "## Q-4 gates (secondary; H1 abstentions; templates only)"]
    for lam in A.LAMBDAS:
        q = rep["Q-4"][f"lambda_{lam:g}"]
        L.append(f"- λ {lam:g}: U " + ", ".join(f"{k} {v:+.3f}" for k, v in q["U"].items()))
        L += [f"  - {k}: {fmt_ci(v)}" for k, v in q.items() if k != "U"]
    L += ["", "## Q-5 F1 replication (labelled)"] + [f"- {k}: {fmt_ci(v) if isinstance(v, dict) else v}"
                                                     for k, v in rep["Q-5"].items()]
    L += ["", "## Q-6 (descriptive)", f"- {rep['Q-6']['reading']}",
          f"- difference of repair gains: {fmt_ci(rep['Q-6']['difference'])}", "",
          "## M1 decoding (diagnostics)"]
    L += [f"- {k}: acc {fmt_ci(v['acc'])}; − candidate {fmt_ci(v['minus_candidate'])}" for k, v in rep["M1"].items()]
    L += ["", "## Per-arm rates", "```", json.dumps(rep["arm_rates"], indent=1)[:6000], "```"]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", type=Path, default=R.CASES)
    ap.add_argument("--out", type=Path, default=R.OUT)
    ap.add_argument("--dev", action="store_true")
    args = ap.parse_args(argv)
    rows = R.load_jsonl(args.cases)
    if args.dev:
        rep = {"baselines_dev": baseline_table(rows, args.out, "dev"),
               "probe_dev": R.read_json(args.out / "probe" / "probe_dev.json"),
               "design": R.read_json(args.out / "design.json")}
        R.write_json(args.out / "report_dev.json", rep)
        print(json.dumps(rep["baselines_dev"]["totals"]))
        return 0
    try:
        rep = held_report(rows, args.out)
    except PermissionError as e:
        print(f"design guard: {e}", file=sys.stderr)
        return R.EXIT_DESIGN
    R.write_json(args.out / "report.json", rep)
    (args.out / "report.md").write_text(markdown(rep))
    print(f"wrote {args.out / 'report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
