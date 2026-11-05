"""§41 report (development only): fit the conditional-steering rule on the dev rows and freeze it.

    .venv/bin/python scripts/exp41_report.py --run-dir runs/exp41_dev/qwen3b [--out runs/exp41_report_dev.md]

Pre-registration: research/proposals/2026-10-23-exp41-conditional-steering.md (rev 0; rev 1, 877f86d). Logic in
memctl/analysis/conditional.py. Everything is recomputed from <run-dir> ({prefill,steer,steer_all}.jsonl, resid/,
manifest.json), the §38 dev rows (<source>/repro.jsonl, directions.npz) and the §35 cache (cached A and B final
answers, cached judge verdicts). Refused (exit 2): rows of folds other than 0-1, more than one engine among the laptop
rows and the manifest, a refusal-start id set that differs from the §38 file's, missing rows.

Policies on the same 67 rows: A = the run's unsteered intact output; B = the §38 dev B row on R, the cached §35 B row
on C (no laptop B exists on C); Steer = Steer-all (§38's rows on R, this run's on C); Cond = Steer where the rule
fires, else A. Labels: option (a) of rev 1 (ii): the cached §35 verdict where the final answer is byte-equal to the
cached A (first) or B final answer of the row, else D1 (PROVISIONAL); the source is counted per policy. Option (b)
(the 7B judge for the new texts on the laptop) is a hook: `--judge-7b` is refused in this build.

Writes <run-dir>/rule.json (the frozen rule; commit it before §40) and the .md / .json report. The dev utility is
optimistic by construction (the rule is fitted on the rows it is scored on); the out-of-sample test is §40.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from memctl.analysis import conditional as cd  # noqa: E402
from memctl.analysis import steering as st  # noqa: E402
from scripts.exp35_generate import arm_file, judge_file, read_jsonl  # noqa: E402
from scripts.exp38_dev import load_source  # noqa: E402
from scripts.exp39_dev import DEV, LAYER, READER, STEER_ALL_KEY, file_sha256, load_direction  # noqa: E402

POLICIES = ("A", "B", "Steer", "Cond")
OPTION_A = "option (a): cached §35 verdicts where the text is byte-equal to the cached A or B answer; D1 otherwise"


class Refused(SystemExit):
    pass


def refuse(why: str):
    print(f"REFUSED: {why}")
    raise Refused(2)


# --- loading --------------------------------------------------------------------------------------------------

def load_rows(run_dir: Path) -> dict:
    out = {}
    for stage in ("prefill", "steer", "steer_all"):
        p = run_dir / f"{stage}.jsonl"
        if not p.exists():
            refuse(f"{p} missing (run scripts/exp41_dev.py --run first)")
        out[stage] = {st.row_key(r): r for r in read_jsonl(p)}
    return out


def check_rows(rows: dict, manifest: dict) -> str:
    """Folds 0-1 only and one engine across every laptop row and the manifest; returns the engine."""
    engines = {manifest.get("engine")}
    for stage, table in rows.items():
        for r in table.values():
            if r["fold"] not in DEV:
                refuse(f"{stage} row {r['question_id']} is fold {r['fold']}; the §41 fit is folds 0-1 only")
            engines.add(r["engine"])
    if len(engines) != 1:
        refuse(f"more than one engine among the rows and the manifest: {sorted(map(str, engines))}")
    return engines.pop()


def v29_for(v: np.ndarray, h: np.ndarray, engine: str) -> np.ndarray:
    """v_29 at the residual's size: itself on the 3B; its first components on the stub (as exp41_dev adds them)."""
    if h.shape[0] == v.shape[0]:
        return v
    if engine != "stub":
        refuse(f"residual size {h.shape[0]} != v_29 size {v.shape[0]} on a non-stub engine")
    return np.ascontiguousarray(v[:h.shape[0]])


def cached_b_row(d, q: str) -> dict:
    """The cached §35 B row as a policy output (C rows): phrase, v2 class without ids, final answer."""
    c = d.rows["B"][q]
    phrase = st.phrase_of(c["generated"])
    return {"question_id": q, "phrase": phrase, "final_answer": c["final_answer"], "cls": st.classify(phrase),
            "engine": c["engine"], "source": "§35 cache B"}


def make_judge(enabled: bool):
    """Option (b): the 7B judge on the laptop. Not enabled in this build (the 7B is never loaded here)."""
    if not enabled:
        return None
    refuse("option (b) (the 7B judge on the laptop) is a hook only in this build; it needs the 5-call timing gate "
           "(conditional.judge_timing_gate, projection < 45 min) and the Advisor's go before the 7B is loaded")


# --- the analysis ---------------------------------------------------------------------------------------------

def build(run_dir: Path, source: Path | None = None, exp35: Path = Path("runs/exp35"), runs: str = "runs",
          own_root: Path = Path("runs/exp34_jlens_own"), judge=None) -> dict:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    source = Path(source or manifest["source"])
    rows = load_rows(run_dir)
    engine = check_rows(rows, manifest)
    folds = sorted({r["fold"] for t in rows.values() for r in t.values()})
    d, pop = load_source(list(DEV), exp35, runs, own_root)
    R, C = list(pop.R), list(pop.C)
    src_repro = read_jsonl(source / "repro.jsonl")
    ids = cd.refusal_start_ids(src_repro)
    if manifest.get("refusal_start_ids") != ids:
        refuse(f"refusal-start ids {manifest.get('refusal_start_ids')} in the manifest, {ids} from {source}")
    src = {st.row_key(r): r for r in src_repro}
    direction = load_direction(source)
    pops = {q: "R" for q in R} | {q: "C" for q in C}
    key = st.Setting("steer", manifest["layer"], manifest["alpha"], manifest["unit"]).key   # the stub maps layer 29
    if engine != "stub" and key != STEER_ALL_KEY:
        refuse(f"the run's Steer-all setting is {key}, not {STEER_ALL_KEY}")

    def get(stage, pop_, kind, key, q):
        r = rows[stage].get((pop_, kind, key, q))
        if r is None:
            refuse(f"no {stage} row {pop_}/{kind}/{key}/{q}")
        return r

    out_rows, sig = {}, {n: {} for n in cd.SIGNALS}
    proj_diff = 0.0
    for q in R + C:
        p = pops[q]
        a, cut = get("prefill", p, "A", "none", q), get("prefill", p, "cut", "none", q)
        for r in (a, cut):
            if r["step0"]["refusal_ids"] != ids:
                refuse(f"{r['kind']}/{q}: the row's refusal-start ids differ from {ids}")
        h = np.load(run_dir / "resid" / "A" / f"{q}.npy")
        if st.vector_sha(h) != a["h29_sha256"]:
            refuse(f"{q}: resid/A file does not match the row's h29 sha")
        v = v29_for(direction["v"], h, engine)
        s = cd.signals_of(a["step0"]["margin"], cut["step0"]["margin"], h, v)
        proj_diff = max(proj_diff, abs(s["s_proj"] - a["s_proj"]))
        for n in cd.SIGNALS:
            sig[n][q] = s[n]
        steer = get("steer_all", "R", "A", key, q) if p == "R" else get("steer", "C", "A", key, q)
        b = src.get(("R", "B", "none", q)) if p == "R" else cached_b_row(d, q)
        if b is None:
            refuse(f"no §38 dev B row for R/{q}")
        out_rows[q] = {"A": a, "B": b, "Steer": steer}
    if proj_diff > 1e-6:
        refuse(f"s_proj recomputed from resid/ differs from the rows' by {proj_diff}")

    # labels and outcomes per policy (A, B, Steer); Cond takes Steer's or A's
    lab, outc, src_count = {}, {}, {}
    for pol in ("A", "B", "Steer"):
        lab[pol], outc[pol] = {}, {}
        src_count[pol] = {s: 0 for s in cd.LABEL_SOURCES}
        for q in R + C:
            r = out_rows[q][pol]
            ca, cb = d.rows["A"].get(q), d.rows["B"].get(q)
            ok, how = cd.label(r["final_answer"], r["phrase"], r["cls"], d.by_q[q].gold,
                               ca and ca["final_answer"], d.correct("A", q), cb and cb["final_answer"],
                               d.correct("B", q), judge=(lambda t, q=q: judge(q, t)) if judge else None)
            lab[pol][q] = {"correct": ok, "source": how}
            src_count[pol][how] += 1
            outc[pol][q] = cd.outcome(r["cls"], r["phrase"], ok)
    w = cd.weights(len(R), len(C))
    fitted = cd.fit(sig, outc["A"], outc["Steer"], pops, w, cd.LAMBDA)
    ch = fitted["chosen"]
    fire = {q: cd.fires(sig[ch["signal"]][q], ch["sign"], ch["tau"]) for q in R + C}
    outc["Cond"] = {q: outc["Steer"][q] if fire[q] else outc["A"][q] for q in R + C}
    src_count["Cond"] = {s: 0 for s in cd.LABEL_SOURCES}
    for q in R + C:
        src_count["Cond"][lab["Steer" if fire[q] else "A"][q]["source"]] += 1

    def util(pol, weighted, lam, qs=None):
        o = {q: outc[pol][q] for q in (qs or R + C)}
        return cd.total(o, pops, w if weighted else None, lam)

    policies = {}
    for pol in POLICIES:
        policies[pol] = {"weighted_l1": util(pol, True, 1), "weighted_l2": util(pol, True, 2),
                         "unweighted_l1": util(pol, False, 1), "unweighted_l2": util(pol, False, 2),
                         "R": counts(outc[pol], R, out_rows, pol, fire), "C": counts(outc[pol], C, out_rows, pol, fire),
                         **cd.coverage_risk(outc[pol]), "label_sources": src_count[pol]}
    gate = cd.gate(policies["Cond"]["weighted_l1"], policies["Steer"]["weighted_l1"])
    gate["unweighted_diff"] = policies["Cond"]["unweighted_l1"] - policies["Steer"]["unweighted_l1"]
    gold_steer = [outc["Steer"][q] == "correct" for q in R + C]
    gold_steer_R = [outc["Steer"][q] == "correct" for q in R]
    aurocs = {n: {"all": cd.auroc([sig[n][q] for q in R + C], gold_steer),
                  "R": cd.auroc([sig[n][q] for q in R], gold_steer_R)} for n in cd.SIGNALS}
    curves = {f"{n}{g}": cd.risk_coverage_curve(sig[n], g, outc["A"], outc["Steer"], pops, w)
              for n in cd.SIGNALS for g in cd.SIGNS}
    shas = {"prefill.jsonl": file_sha256(run_dir / "prefill.jsonl"), "steer.jsonl": file_sha256(run_dir / "steer.jsonl"),
            "steer_all.jsonl": file_sha256(run_dir / "steer_all.jsonl"),
            "manifest.json": file_sha256(run_dir / "manifest.json"),
            "source repro.jsonl": file_sha256(source / "repro.jsonl"),
            "source directions.npz": direction["npz_sha256"]}
    for f in DEV:
        for arm in ("A", "B"):
            shas[f"exp35 {arm} fold{f}"] = file_sha256(arm_file(exp35, READER, arm, f))
        shas[f"exp35 judge fold{f}"] = file_sha256(judge_file(exp35, READER, f))
    rule = {"experiment": "§41 conditional steering (dev fit, frozen before §40)",
            "signal": ch["signal"], "sign": ch["sign"], "tau": tau_json(ch["tau"]),
            "fires_when": f"{ch['signal']} {'>' if ch['sign'] == '+' else '<'} tau",
            "signal_definitions": {
                "s_margin": "max logit over refusal_start_ids - max logit over all other ids, at ':' (step 0), unsteered",
                "s_proj": f"<h_{LAYER}(':'), v_{LAYER}/||v_{LAYER}||>, decoder layer output pre final norm, unsteered",
                "s_evid": "s_margin(intact prompt) - s_margin(cut prompt)"},
            "refusal_start_ids": ids, "layer": LAYER, "v_sha256": direction["v_sha256"],
            "weights": {"R": "1", "C": str(w["C"]), "C_float": float(w["C"]), "C_to_R_total": str(cd.CR_RATIO)},
            "lambda": cd.LAMBDA, "tie_break": cd.TIE_BREAK, "gate_rule": cd.GATE_RULE,
            "gate": jsonable(gate), "scored_on_40": bool(gate["pass"]),
            "dev_utility": jsonable({p: {k: policies[p][k] for k in ("weighted_l1", "unweighted_l1")} for p in POLICIES}),
            "label_option": OPTION_A if judge is None else "option (b): the 7B judge for new texts",
            "label_sources": src_count, "engine": engine, "folds": folds, "n_R": len(R), "n_C": len(C),
            "input_sha256": shas, "note": "dev utility optimistic by construction (fitted on the rows it is scored on)"}
    return {"engine": engine, "folds": folds, "n_R": len(R), "n_C": len(C), "refusal_start_ids": ids,
            "weights": w, "fit": fitted, "best_per_signal": cd.best_per_signal(fitted), "policies": policies,
            "gate": gate, "auroc_gold_steer": aurocs, "curves": curves, "rule": rule, "fires": fire,
            "signals": sig, "outcomes": outc, "labels": lab, "manifest": manifest,
            "label_option": rule["label_option"], "d1_bias": cd.D1_BIAS}


def counts(outc: dict, qs: list[str], out_rows: dict, pol: str, fire: dict) -> dict:
    out = {k: sum(outc[q] == k for q in qs) for k in ("correct", "wrong", "soft", "refusal", "degenerate")}
    out["n"] = len(qs)
    if pol == "Cond":
        out["ans_v2"] = sum((out_rows[q]["Steer"] if fire[q] else out_rows[q]["A"])["cls"] == "answer" for q in qs)
        out["fires"] = sum(fire[q] for q in qs)
    else:
        out["ans_v2"] = sum(out_rows[q][pol]["cls"] == "answer" for q in qs)
    return out


def tau_json(t: float):
    return "inf" if t == math.inf else "-inf" if t == -math.inf else t


def jsonable(x):
    if isinstance(x, Fraction):
        return {"exact": str(x), "value": float(x)}
    if isinstance(x, float) and math.isinf(x):
        return tau_json(x)
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    return x


# --- rendering ------------------------------------------------------------------------------------------------

def render(n: dict) -> str:
    f = cd.fmt
    w, ch, g = n["weights"], n["fit"]["chosen"], n["gate"]
    L = [f"# §41 report: conditional steering, development fit (folds {', '.join(map(str, n['folds']))})", "",
         "**DEVELOPMENT ONLY.** D1 labels are PROVISIONAL; the dev utility is optimistic by construction (the rule is "
         "fitted on the rows it is scored on). The out-of-sample test is §40.", "",
         f"- Engine: `{n['engine']}`; rows R {n['n_R']} + C {n['n_C']} (the 8 damage rows are the first 8 of C).",
         f"- Refusal-start ids (first ids of the refusal-class §38 dev A and B rows): {n['refusal_start_ids']}.",
         f"- Weights: R 1, C w_C = (n_R/3)/n_C = {f(w['C'])} (C total {f(w['C'] * n['n_C'])} : R total {n['n_R']} = 1 : 3).",
         f"- Labels: {n['label_option']}. {n['d1_bias']}.", "",
         "## Frozen rule", "",
         f"**{ch['signal']} {'>' if ch['sign'] == '+' else '<'} {tau_json(ch['tau'])}** (fires on {ch['fires']} of "
         f"{n['n_R'] + n['n_C']} rows: R {n['policies']['Cond']['R']['fires']}, C {n['policies']['Cond']['C']['fires']}). "
         f"Tie-break: {cd.TIE_BREAK}. Candidates tied at the top weighted utility: {n['fit']['n_tied_at_top']}; tied "
         f"after the order: {n['fit']['n_tied_final']}.", "",
         f"**Gate** ({cd.GATE_RULE}): U(Cond) {f(g['u_cond'])} - U(Steer) {f(g['u_steer'])} = {f(g['diff'])} "
         f"(need >= {g['need']}) -> **{'PASS: the rule is scored on §40' if g['pass'] else 'FAIL: §41 closes development-only; §40 still logs the signals'}**. "
         f"Unweighted difference {f(g['unweighted_diff'])}.", "",
         "## Policies (same rows)", "",
         "| policy | weighted U l=1 | weighted U l=2 | unweighted U l=1 | unweighted U l=2 | R c/w/soft/ref/deg (ans v2) | C c/w/soft/ref/deg (ans v2) | coverage | risk | label sources |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for pol in POLICIES:
        p = n["policies"][pol]
        cell = lambda b: f"{b['correct']}/{b['wrong']}/{b['soft']}/{b['refusal']}/{b['degenerate']} ({b['ans_v2']})"  # noqa: E731
        srcs = ", ".join(f"{k} {v}" for k, v in p["label_sources"].items() if v)
        L.append(f"| {pol} | {f(p['weighted_l1'])} | {f(p['weighted_l2'])} | {p['unweighted_l1']} | {p['unweighted_l2']} | "
                 f"{cell(p['R'])} | {cell(p['C'])} | {p['coverage']:.3f} | {f(p['risk']) if p['risk'] is not None else '-'} | {srcs} |")
    L += ["", "B on R is the §38 dev B row; on C the cached §35 B row (classified v2 without ids).", "",
          "## Best rule per signal (descriptive)", "", "| signal | sign | tau | fires | weighted U l=1 | unweighted U l=1 |",
          "|---|---|---|---|---|---|"]
    for name, c in n["best_per_signal"].items():
        L.append(f"| {name} | {c['sign']} | {tau_json(c['tau'])} | {c['fires']} | {f(c['weighted'])} | {c['unweighted']} |")
    L += ["", "## AUROC of each signal for gold(Steer)", "", "| signal | all rows | R rows |", "|---|---|---|"]
    for name, a in n["auroc_gold_steer"].items():
        L.append(f"| {name} | {f(a['all']) if a['all'] is not None else '-'} | {f(a['R']) if a['R'] is not None else '-'} |")
    L += ["", "AUROC < 0.5 means the signal ranks the other way (the fitted sign '-').", "",
          f"## Risk-coverage curve of the chosen signal ({ch['signal']}, sign {ch['sign']})", "",
          "| tau | fires | coverage | risk | weighted U l=1 | weighted U l=2 | unweighted U l=1 |", "|---|---|---|---|---|---|---|"]
    for e in n["curves"][f"{ch['signal']}{ch['sign']}"]:
        L.append(f"| {tau_json(e['tau']) if isinstance(e['tau'], float) and math.isinf(e['tau']) else f(e['tau'])} | "
                 f"{e['fires']} | {e['coverage']:.3f} | {f(e['risk']) if e['risk'] is not None else '-'} | "
                 f"{f(e['weighted_l1'])} | {f(e['weighted_l2'])} | {e['unweighted_l1']} |")
    L += ["", "Every (signal, sign) curve is in the .json beside this file.", ""]
    return "\n".join(L)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="runs/exp41_dev/qwen3b")
    parser.add_argument("--source", default=None, help="the §38 dev run directory (default: from the manifest)")
    parser.add_argument("--exp35", default="runs/exp35")
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--own-root", default="runs/exp34_jlens_own")
    parser.add_argument("--out", default="runs/exp41_report_dev.md")
    parser.add_argument("--rule", default=None, help="where to write the frozen rule (default <run-dir>/rule.json)")
    parser.add_argument("--judge-7b", action="store_true", help="option (b) labels (a hook; refused in this build)")
    args = parser.parse_args(argv)
    run_dir = Path(args.run_dir)
    try:
        judge = make_judge(args.judge_7b)
        n = build(run_dir, Path(args.source) if args.source else None, Path(args.exp35), args.runs,
                  Path(args.own_root), judge)
    except Refused as e:
        return int(e.code)
    text = render(n)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    keep = {k: n[k] for k in ("engine", "folds", "n_R", "n_C", "refusal_start_ids", "weights", "fit", "best_per_signal",
                              "policies", "gate", "auroc_gold_steer", "curves", "fires", "signals", "outcomes",
                              "labels", "label_option", "d1_bias")}
    out.with_suffix(".json").write_text(json.dumps(jsonable(keep), indent=1))
    rule_path = Path(args.rule) if args.rule else run_dir / "rule.json"
    rule_path.write_text(json.dumps(n["rule"], indent=1))
    print(text)
    print(f"wrote {out}, {out.with_suffix('.json')} and the frozen rule {rule_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
