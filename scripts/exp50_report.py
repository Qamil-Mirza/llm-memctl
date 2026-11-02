#!/usr/bin/env python
"""§50 report: stage 1 (held; estimands Δ_S / Δ_M per task, bootstrap, states, profiles, Q-2 to Q-5, F1 diagnostic,
the stage-2 opening rule) and the final report (stage 2 arms and contrasts with Holm). Every number is computed from
the per-row files; nothing is typed. Called by scripts/exp50_run.py (--stage stage1-report / report), both guarded.

Spec: research/proposals/2026-11-02-exp50-why-false-abstention.md rev 1 §4-§5; helpers memctl/analysis/why50.py.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from memctl.analysis import abst49 as A  # noqa: E402
from memctl.analysis import why50 as W  # noqa: E402

TASKS = ("DE", "RC", "TU")


def _R():
    import exp50_run as R
    return R


def joint_rows(rows, root, split="held") -> list[dict]:
    R = _R()
    out = []
    for r in rows:
        if r["split"] != split or r["set"] != "ANS" or r["format"] != "F2":
            continue
        x = R.rec_of(root, r["case_id"])
        if not x or not x.get("answer_bearing") or not x["p_ans_ok"] or x["outcome"] not in ("gold", "abstain"):
            continue
        out.append({"case_id": r["case_id"], "task": r["task"], "hardness": r["hardness"], "gold": r["gold"],
                    "base_key": r["base_key"], "cls": "fa" if x["outcome"] == "abstain" else "ans",
                    "s": x["S"]["s"] if x["S"]["valid"] else None, "m": x["margins"]["m_city"],
                    "m_commit": x["margins"]["m_commit"], "kv": (x.get("KV") or {}).get("kv"),
                    "q5": (x.get("q5") or {}).get("producing_m_city")})
    return out


def una_rel_rows(rows, root, split="held") -> list[dict]:
    R = _R()
    out = []
    for r in rows:
        if r["split"] == split and r["set"] == "UNA-rel":
            x = R.rec_of(root, r["case_id"])
            if x and x["S"]["valid"]:
                out.append({"task": r["task"], "hardness": r["hardness"], "base_key": r["base_key"], "s": x["S"]["s"]})
    return out


def task_estimands(task, jr, ur, s_valid: bool) -> dict:
    J = [r for r in jr if r["task"] == task]
    U = [r for r in ur if r["task"] == task]
    fa, an = [r for r in J if r["cls"] == "fa"], [r for r in J if r["cls"] == "ans"]
    js = [r for r in J if r["s"] is not None]
    feas = W.feasible(fa, an)
    by_base = {}
    for r in J:
        by_base.setdefault(r["base_key"], {"j": [], "u": []})["j"].append(r)
    for r in U:
        by_base.setdefault(r["base_key"], {"j": [], "u": []})["u"].append(r)

    def rs_of(jj, uu):
        return W.reference_s([r for r in jj if r["cls"] == "ans" and r["s"] is not None], uu,
                             [r for r in jj if r["cls"] == "fa"])
    ds = W.cell_delta(js, "s")
    dm = W.cell_delta(J, "m")
    rs = rs_of(J, U)
    pick = lambda smp, k: [x for b in smp for x in by_base[b][k]]
    ci_s = W.boot_bases(list(by_base), lambda smp: W.cell_delta([r for r in pick(smp, "j") if r["s"] is not None], "s",
                                                                  ds["coarse"])["delta"])
    ci_m = W.boot_bases(list(by_base), lambda smp: W.cell_delta(pick(smp, "j"), "m", dm["coarse"])["delta"])
    ci_r = W.boot_bases(list(by_base), lambda smp: rs_of(pick(smp, "j"), pick(smp, "u")))
    r_def = rs is not None and ci_r["lo95"] is not None and ci_r["lo95"] >= W.R_S_MIN
    delta_s = W.DELTA_S_FRAC * rs if r_def else None
    s_state = W.state(ds["delta"], ci_s, delta_s, valid=s_valid and feas)
    m_state = W.state(dm["delta"], ci_m, W.DELTA_M, valid=feas)
    auc_m = W.auroc([r["m"] for r in J], [1 if r["cls"] == "ans" else 0 for r in J]) if fa and an else None
    return {"n_fa": len(fa), "n_ans": len(an), "fa_bases": len({r["base_key"] for r in fa}),
            "ans_bases": len({r["base_key"] for r in an}), "n_with_valid_s": len(js), "feasible": feas,
            "s_gate_valid": s_valid, "delta_s": ds, "ci_s": ci_s, "delta_m": dm, "ci_m": ci_m, "auroc_m": auc_m,
            "R_S": rs, "ci_R_S": ci_r, "r_S": (-ds["delta"] / rs) if (r_def and ds["delta"] is not None) else "undefined",
            "delta_s_margin": delta_s, "state_s": s_state, "state_m": m_state,
            "profile": W.profile(s_state, m_state) if feas else "underpowered",
            "m_commit_near_tie_share_fa": float(np.mean([abs(r["m_commit"]) < 1 for r in fa])) if fa else None,
            "m_city_iqr_answered": (lambda v: [float(x) for x in np.percentile(v, [25, 75])])(
                [r["m"] for r in an]) if an else None}


def stage1(rows, root) -> dict:
    R = _R()
    design = A.require_design(root)
    held = [r for r in rows if r["split"] == "held"]
    oc = Counter((r["task"], r["set"] + (r["rep"] if r["set"] == "ANS" else ""), r["format"], r["hardness"],
                  R.rec_of(root, r["case_id"])["outcome"]) for r in held)
    jr, ur = joint_rows(rows, root), una_rel_rows(rows, root)
    tasks = {t: task_estimands(t, jr, ur, bool(design["s_gate"].get(t))) for t in TASKS}
    # S on all gold / abstain ANS rows (not only answer-bearing), point only
    s_all = {}
    for t in TASKS:
        xs = []
        for r in held:
            if r["task"] == t and r["set"] == "ANS" and r["format"] == "F2":
                x = R.rec_of(root, r["case_id"])
                if x["outcome"] in ("gold", "abstain") and x["S"]["valid"]:
                    xs.append({"task": t, "hardness": r["hardness"], "gold": r["gold"],
                               "cls": "fa" if x["outcome"] == "abstain" else "ans", "s": x["S"]["s"]})
        s_all[t] = W.cell_delta(xs, "s")
    # Q-2 flip pairs
    pairs = R.flip_pairs(rows, root, "held")
    q2 = {"n_pairs": len(pairs), "n_bases": len({p["base_key"] for p in pairs}),
          "by_task": dict(Counter(p["task"] for p in pairs))}
    ds, dm, ks = [], [], []
    for p in pairs:
        a, b = R.rec_of(root, p["abstained"]), R.rec_of(root, p["answered"])
        dm.append(a["margins"]["m_city"] - b["margins"]["m_city"])
        if a["S"]["valid"] and b["S"]["valid"]:
            ds.append((p["base_key"], a["S"]["s"] - b["S"]["s"]))
    q2["paired_m_city"] = A.cluster_boot([p["base_key"] for p in pairs], dm)
    q2["paired_s"] = A.cluster_boot([k for k, _ in ds], [v for _, v in ds])
    # Q-3 descriptive
    q3 = {}
    for key in ("hardness", "gold"):
        for v in sorted({r[key] for r in jr}):
            sub = [r for r in jr if r[key] == v]
            q3[f"{key}={v}"] = {c: {"n": sum(r["cls"] == c for r in sub),
                                    "s_mean": _m([r["s"] for r in sub if r["cls"] == c]),
                                    "m_mean": _m([r["m"] for r in sub if r["cls"] == c])} for c in ("fa", "ans")}
    # Q-4, Q-5
    ans = [r for r in jr if r["cls"] == "ans"]
    sv = [r for r in ans if r["s"] is not None]
    kv = [r for r in ans if r["kv"] is not None]
    q5 = [r for r in jr if r["q5"] is not None]
    q4 = {"spearman_s_m_answered": W.spearman(np.array([r["s"] for r in sv]), np.array([r["m"] for r in sv])) if len(sv) > 2 else None,
          "spearman_kv_m_answered": W.spearman(np.array([r["kv"] for r in kv]), np.array([r["m"] for r in kv])) if len(kv) > 2 else None,
          "n": len(sv), "n_kv": len(kv),
          "kv_mean": {c: _m([r["kv"] for r in jr if r["cls"] == c and r["kv"] is not None]) for c in ("fa", "ans")}}
    q5o = {"spearman_producing_vs_p_ans": W.spearman(np.array([r["q5"] for r in q5]), np.array([r["m"] for r in q5]))
           if len(q5) > 2 else None, "n": len(q5), "note": "exploratory read-site dependence; no caching claim"}
    # F1 diagnostic
    f1 = {}
    for r in held:
        if r["format"] == "F1":
            x = R.rec_of(root, r["case_id"])
            f1.setdefault(x["outcome"], []).append(x["f1_margins"]["m_city"])
    f1d = {k: {"n": len(v), "mean": _m(v)} for k, v in f1.items()}
    # opening rule
    pt = Counter(p["task"] for p in pairs)
    pb = {t: len({p["base_key"] for p in pairs if p["task"] == t}) for t in TASKS}
    eligible = [t for t in TASKS if tasks[t]["profile"] in ("B-compatible", "C-compatible", "Joint")
                and pt[t] >= W.OPEN_PAIRS and pb[t] >= W.OPEN_BASES]
    opens = bool(eligible) and design.get("layer") is not None and bool(design.get("noop_ok"))
    rep = {"design_sha256": A.design_sha(design), "outcomes_held": {"|".join(k): n for k, n in sorted(oc.items())},
           "tasks": tasks, "s_all_gold_abstain": s_all, "Q-2": q2, "Q-3": q3, "Q-4": q4, "Q-5": q5o,
           "F1_diagnostic": f1d, "eligible_tasks": eligible, "stage2_opens": opens,
           "stage2_open_note": ("opens" if opens else "does not open: " + ("no eligible task" if not eligible else
                                "no layer / no-op failed on dev"))}
    R.write_json(root / "stage1_report.json", rep)
    return rep


def _m(v):
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if v else None


def stage2_tables(rows, root) -> dict:
    R = _R()
    st = R.read_json(root / "s2held" / "status.json") or {}
    if not st.get("run"):
        return {"run": False, "why": st.get("why", "not run (time budget or not opened)")}
    files = [R.read_json(p) for p in sorted((root / "s2held").glob("*__*__*.json"))]
    pairs = {p["pair"]: p for p in R.flip_pairs(rows, root, "held")}
    out = {"run": True, "rates": {}, "contrasts": {}, "norms": {}}
    val = {}
    for f in files:
        tgt = "gold" if f["direction"] == "forward" else "abstain"
        v = float(np.mean([o == tgt for o in f["outcomes"]]))
        val[(f["pair"], f["direction"], f["arm"])] = v
        k = f"{f['direction']}|{f['arm']}"
        out["rates"].setdefault(k, {c: [] for c in A.CLASSES})
        for c in A.CLASSES:
            out["rates"][k][c].append(float(np.mean([o == c for o in f["outcomes"]])))
        if f.get("norm") is not None:
            out["norms"].setdefault(k, []).append(f["norm"])
    out["rates"] = {k: {c: _m(v) for c, v in d.items()} | {"n": len(d["gold"])} for k, d in out["rates"].items()}
    out["norms"] = {k: _m(v) for k, v in out["norms"].items()}
    pvals = {}
    for direction in ("forward", "reverse"):
        for name, a1, a2 in (("P-S - Rand-S", "P-S", "Rand-S"), ("P-M - Rand-M", "P-M", "Rand-M"), ("P-S - P-M", "P-S", "P-M")):
            for scope in ("pooled",) + TASKS:
                ks = [p for p in pairs if scope == "pooled" or pairs[p]["task"] == scope]
                xs = [(pairs[p]["base_key"], val[(p, direction, a1)] - val[(p, direction, a2)]) for p in ks
                      if (p, direction, a1) in val and (p, direction, a2) in val]
                if not xs:
                    continue
                keys, d = [k for k, _ in xs], np.array([v for _, v in xs])
                ci = A.cluster_boot(keys, d)
                rec = {**ci}
                if scope == "pooled":
                    ub = sorted(set(keys))
                    g = np.array([ub.index(k) for k in keys])
                    sums, cnt = np.bincount(g, d, len(ub)), np.bincount(g, minlength=len(ub)).astype(float)
                    dr = np.random.default_rng(0).integers(0, len(ub), (A.BOOT_N, len(ub)))
                    reps = sums[dr].sum(1) / cnt[dr].sum(1)
                    rec["p_boot"] = W.boot_p(reps)
                    pvals[f"{direction}|{name}"] = rec["p_boot"]
                out["contrasts"][f"{direction}|{name}|{scope}"] = rec
    out["holm"] = W.holm(pvals) if pvals else {}
    return out


def final(rows, root) -> dict:
    R = _R()
    s1 = R.read_json(root / "stage1_report.json") or stage1(rows, root)
    rep = {"stage1": s1, "stage2": stage2_tables(rows, root), "design": R.read_json(root / "design.json")}
    R.write_json(root / "report.json", rep)
    (root / "report.md").write_text(markdown(rep))
    return rep


def _ci(d):
    if not d or d.get("est") is None:
        return "n/a"
    return f"{d['est']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}] (n {d['n']}, bases {d['n_bases']})"


def markdown(rep) -> str:
    s1 = rep["stage1"]
    L = ["# §50 why false abstention: held read (computed)", "", f"Design sha256 `{s1['design_sha256']}`.", "",
         "## Stage 1 per task (joint population: answer-bearing note, gold or abstain, P_ans)"]
    for t, x in s1["tasks"].items():
        ds, dm = x["delta_s"].get("delta"), x["delta_m"].get("delta")
        L.append(f"- **{t}**: FA {x['n_fa']} ({x['fa_bases']} bases), answered {x['n_ans']} ({x['ans_bases']} bases); "
                 f"S gate {'valid' if x['s_gate_valid'] else 'INVALID'}; "
                 f"Δ_S {ds if ds is None else round(ds, 4)} [{x['ci_s']['lo95']}, {x['ci_s']['hi95']}] -> {x['state_s']}; "
                 f"Δ_M {dm if dm is None else round(dm, 4)} [{x['ci_m']['lo95']}, {x['ci_m']['hi95']}] -> {x['state_m']}; "
                 f"R_S {x['R_S']}, r_S {x['r_S']}; AUROC_M {x['auroc_m']}; **profile: {x['profile']}**")
    L += ["", "## Q-2 flip pairs (selected, conditional)", f"- pairs {s1['Q-2']['n_pairs']} from {s1['Q-2']['n_bases']} bases",
          f"- paired M_city (abstained − answered): {_ci(s1['Q-2']['paired_m_city'])}",
          f"- paired s: {_ci(s1['Q-2']['paired_s'])}", "", "## Q-4 / Q-5", f"- {json.dumps(s1['Q-4'])}",
          f"- {json.dumps(s1['Q-5'])}", "", f"## Stage 2: {s1['stage2_open_note']}"]
    s2 = rep["stage2"]
    if s2.get("run"):
        L += [f"- {k}: {_ci(v)}" + (f"; p {v['p_boot']:.4f}, Holm {s2['holm'].get(k.rsplit('|', 1)[0])}" if 'p_boot' in v else "")
              for k, v in s2["contrasts"].items()]
        L += ["", "Rates:", "```", json.dumps(s2["rates"], indent=1), "```"]
    else:
        L.append(f"- stage 2 not run: {s2.get('why')}")
    return "\n".join(L) + "\n"
