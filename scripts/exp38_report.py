"""§38 report: the design gate (development) and, with --confirm, Claims 1, 1', 2 and Controls X, C, T.

    .venv/bin/python scripts/exp38_report.py --run-dir runs/exp38_dev/qwen3b [--out runs/exp38_report_dev.md]
    .venv/bin/python scripts/exp38_report.py --run-dir runs/exp38_pod/qwen3b --confirm --design runs/exp38_dev/qwen3b/gate.json

Pre-registration: research/proposals/2026-10-22-exp38-refusal-direction.md (rev 1, bb758c3; amendment 1, c367d28).
Everything is recomputed from the rows (the gate.json written by the run is only cross-checked):
reproduction against the §35 cache, sanity S, per-setting ans / degenerate / refusal rates on the reproduced R rows
and on the unseen ones (R minus the estimation pairs), the design gate arithmetic and the chosen (L, alpha), and the
controls at the chosen setting. Gold: the §35 judge's verdicts when <run-dir>/judge.jsonl holds them (the pod),
otherwise the D1 string rule on the phrase, always labelled PROVISIONAL (the laptop).

Confirm mode (folds 2-4; refused without --confirm): Claim 1 = paired P(gold(Steer)) - P(gold(Steer-cut)) on the
reproduced R rows (10,000 resamples), SHOWN if the lower bound > 0, meaningful if P(gold(Steer)) >= 0.30, UNDERPOWERED
below 30 rows; Claim 1' = the same on the rows where the gold fails D1 in both the cut prompt and the N-cut note;
Claim 2 (only if Claim 1 SHOWN); Control X = ans(Random) <= 0.2 x ans(Steer) on R (Reverse is descriptive, amendment
1.5); Control C = paired gold(Steer) - gold(A) on the C rows, a drop over 0.10 flagged; Control T by question type.
The report refuses a run directory with more than one engine.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from memctl.analysis import answer_selection as sel  # noqa: E402
from memctl.analysis import steering as st  # noqa: E402
from memctl.analysis.jacobian_stats import bootstrap_proportion, paired_claim  # noqa: E402
from scripts.exp35_generate import read_jsonl  # noqa: E402

DEV_FOLDS = {0, 1}
CONFIRM_FOLDS = {2, 3, 4}
STAGE_FILES = ("repro", "sanity", "sweep", "controls")
UNDERPOWERED_BELOW = 30
MEANINGFUL = 0.30
DAMAGE_DROP = 0.10


class Refused(SystemExit):
    pass


def read_rows(run_dir: Path) -> dict:
    rows = {}
    for stage in STAGE_FILES:
        for r in read_jsonl(run_dir / f"{stage}.jsonl"):
            rows[st.row_key(r)] = r
    return rows


def check_rows(rows: dict, run_dir: Path, confirm: bool) -> tuple[list[str], set[int]]:
    engines = sorted({r["engine"] for r in rows.values()})
    manifest = run_dir / "manifest.json"
    if manifest.exists():
        engines = sorted(set(engines) | {json.loads(manifest.read_text())["engine"]})
    if len(engines) > 1:
        raise Refused(f"REFUSED: the run directory mixes engines {engines}; one engine per run directory")
    folds = {int(r["fold"]) for r in rows.values()}
    if folds & CONFIRM_FOLDS and not confirm:
        raise Refused(f"REFUSED: rows of confirmation folds {sorted(folds & CONFIRM_FOLDS)}; pass --confirm (only on the pod "
                      "run, once, after the design gate is fixed in the record)")
    if confirm and not folds & CONFIRM_FOLDS:
        raise Refused("REFUSED: --confirm on a run directory without confirmation-fold rows")
    shas = sorted({r["wording_sha256"] for r in rows.values()})
    if len(shas) > 1:
        raise Refused(f"REFUSED: the run directory mixes wordings {shas}")
    return engines, folds


class Gold:
    """Judge verdicts when present (key: pop, kind, setting key, question id), else the provisional D1 rule."""

    def __init__(self, run_dir: Path) -> None:
        self.judge = {(r["pop"], r["kind"], r["setting_key"], r["question_id"]): bool(r["verdict"])
                      for r in read_jsonl(run_dir / "judge.jsonl")}
        self.sources: dict[str, int] = defaultdict(int)

    def __call__(self, row: dict) -> bool:
        key = (row["pop"], row["kind"], row["setting"]["key"], row["question_id"])
        if key in self.judge:
            self.sources["judge"] += 1
            return self.judge[key] and row["cls"] == "answer"
        self.sources["D1 provisional"] += 1
        return bool(row["gold_d1_provisional"])

    @property
    def label(self) -> str:
        return "judge" if self.judge else "D1 string rule, PROVISIONAL"


def repro_from_rows(rows: dict, d, pop) -> dict:
    result = {"R": {}, "damage": {}}
    for q in pop.R:
        a, b = rows.get(("R", "A", "none", q)), rows.get(("R", "B", "none", q))
        if a is None or b is None:
            continue
        ra = st.reproduces(a, d.rows["A"][q], d.by_q[q].gold)
        rb = st.reproduces(b, d.rows["B"][q], d.by_q[q].gold)
        result["R"][q] = {"A": ra, "B": rb, "ok": ra["ok"] and rb["ok"]}
    for q in sorted(set(pop.C)):
        a = rows.get(("C", "A", "none", q))
        if a is None:
            continue
        ra = st.reproduces(a, d.rows["A"][q], d.by_q[q].gold)
        result["damage"][q] = {"A": ra, "ok": ra["ok"]}
    from scripts.exp38_dev import summarise_repro
    return summarise_repro(result)


def sanity_from_rows(rows: dict, pairs: list[str]) -> dict | None:
    keys = [k for k in rows if k[0] == "pairs"]
    if not keys:
        return None
    settings = sorted({k[2] for k in keys})
    finals = sorted({int(s.split(":L")[1]) for s in settings if s.startswith("overwrite_B")})
    final, shallow = max(finals), min(finals)
    fin, zero, self_ex, shallow_m = [], [], [], []
    for q in pairs:
        a0, b0 = rows.get(("R", "A", "none", q)), rows.get(("R", "B", "none", q))
        if a0 is None or b0 is None or ("pairs", "A", f"overwrite_B:L{final}", q) not in rows:
            continue
        same = lambda r: r["generated"] == a0["generated"] and r["gen_ids"] == a0["gen_ids"]  # noqa: E731
        fin.append(rows[("pairs", "A", f"overwrite_B:L{final}", q)]["first_token"] == b0["first_token"])
        shallow_m.append(rows[("pairs", "A", f"overwrite_B:L{shallow}", q)]["first_token"] == b0["first_token"])
        self_ex += [same(rows[("pairs", "A", s, q)]) for s in settings if s.startswith("overwrite_self")
                    and ("pairs", "A", s, q) in rows]
        zero += [same(rows[("pairs", "A", s, q)]) for s in settings if s.startswith("zero") and ("pairs", "A", s, q) in rows]
    out = st.sanity_summary(fin, zero, self_ex, shallow_m)
    out["final_layer"], out["shallow_layer"] = final, shallow
    return out


def setting_table(rows: dict, R: list[str], unseen: list[str]) -> dict:
    by_setting: dict[str, dict] = {}
    for (pop, kind, key, q), r in rows.items():
        if pop != "R" or kind != "A" or r["setting"]["mode"] not in ("steer", "random", "reverse"):
            continue
        by_setting.setdefault(key, {"setting": r["setting"], "rows": {}})["rows"][q] = r
    out = {}
    for key, v in sorted(by_setting.items()):
        s = v["setting"]
        out[key] = {"mode": s["mode"], "layer": s["layer"], "alpha": s["alpha"],
                    "all": st.rates([v["rows"][q] for q in R if q in v["rows"]]),
                    "unseen": st.rates([v["rows"][q] for q in unseen if q in v["rows"]])}
    return out


def gate_from_table(table: dict, R: list[str], sanity_pass: bool, repro_rate) -> dict:
    steer = {k: {"layer": v["layer"], "alpha": v["alpha"], **v["all"]} for k, v in table.items()
             if v["mode"] == "steer" and v["all"]["n"] == len(R) and R}
    random_ans = {}
    for k, v in steer.items():
        rk = k.replace("steer:", "random:", 1)
        if rk in table and table[rk]["all"]["n"] == len(R):
            random_ans[k] = table[rk]["all"]["ans"]
    return st.design_gate(steer, random_ans, sanity_pass, repro_rate)


def pick(rows: dict, pop: str, kind: str, key: str, qs: list[str]) -> dict[str, dict]:
    return {q: rows[(pop, kind, key, q)] for q in qs if (pop, kind, key, q) in rows}


def rate_block(sel_rows: dict[str, dict], gold: Gold, samples: int) -> dict:
    rs = list(sel_rows.values())
    out = st.rates(rs)
    out["gold"] = bootstrap_proportion([gold(r) for r in rs], samples=samples)
    out["ans_ci"] = bootstrap_proportion([r["cls"] == "answer" for r in rs], samples=samples)
    return out


def ratio_bootstrap(num: dict[str, float], den: dict[str, float], samples: int, seed: int = 0) -> dict:
    keys = sorted(set(num) & set(den))
    if not keys or not sum(den[k] for k in keys):
        return {"n": len(keys), "ratio": None, "ci_low": None, "ci_high": None}
    rng = random.Random(seed)
    draws = []
    for _ in range(samples):
        sample = rng.choices(keys, k=len(keys))
        d = sum(den[k] for k in sample)
        if d:
            draws.append(sum(num[k] for k in sample) / d)
    draws.sort()
    return {"n": len(keys), "ratio": sum(num[k] for k in keys) / sum(den[k] for k in keys),
            "ci_low": draws[int(0.025 * len(draws))], "ci_high": draws[min(len(draws) - 1, int(0.975 * len(draws)))]}


def clean_subset(d, R: list[str]) -> list[str]:
    """Claim 1' rows: the gold fails D1 in the cut prompt and in the N-cut note (the §36 Control B restriction)."""
    out = []
    for q in R:
        rec = d.by_q[q]
        cut_prompt = sel.build_arm_input(rec, "Ncut").prompt
        note, _ = sel.split_note(d.rows["Ncut"][q]["generated"])
        if sel.d1(cut_prompt, rec.gold) is not True and sel.d1(note, rec.gold) is not True:
            out.append(q)
    return out


def analyse(run_dir: Path, d, pop, confirm: bool, samples: int, design: Path | None) -> dict:
    rows = read_rows(run_dir)
    engines, folds = check_rows(rows, run_dir, confirm)
    gold = Gold(run_dir)
    rep = repro_from_rows(rows, d, pop)
    if confirm:   # claims and controls on the confirmation folds only (development rows are bracketed context)
        for k in ("R_ok", "damage_ok"):
            rep[k] = [q for q in rep[k] if d.by_q[q].fold in CONFIRM_FOLDS]
    R = rep["R_ok"]
    pairs = [q for q in pop.pairs if q in R]
    unseen = [q for q in pop.unseen if q in R]
    C = rep["damage_ok"]
    n = {"engines": engines, "folds": sorted(folds), "gold_source": gold.label, "repro": {
        k: rep[k] for k in ("rate", "rate_R", "R_verdict_match", "pass", "dropped", "R_ok", "damage_ok")},
        "populations": {"R": len(pop.R), "pairs": len(pop.pairs), "unseen": len(pop.unseen), "C": len(pop.C),
                        "R_reproduced": len(R), "pairs_reproduced": len(pairs), "unseen_reproduced": len(unseen),
                        "C_reproduced": len(C)}}
    dpath = run_dir / "directions.json"
    n["directions"] = json.loads(dpath.read_text())["layers"] if dpath.exists() else None
    san = sanity_from_rows(rows, pairs)
    n["sanity"] = san
    sanity_pass = bool(san and san["pass"])
    table = setting_table(rows, R, unseen)
    n["settings"] = table
    if confirm:
        if design is None or not design.exists():
            raise Refused("REFUSED: --confirm needs --design (the development gate.json fixing L and alpha)")
        fixed = json.loads(design.read_text())
        if not fixed.get("chosen"):
            raise Refused("REFUSED: the design gate chose no setting; there is nothing to confirm")
        chosen_key = fixed["chosen"]
        n["gate"] = {"fixed_from": str(design), "chosen": chosen_key, "pass": fixed.get("pass")}
    else:
        gate = gate_from_table(table, R, sanity_pass, rep["rate"])
        n["gate"] = gate
        chosen_key = gate["chosen"]
        on_disk = run_dir / "gate.json"
        if on_disk.exists():
            written = json.loads(on_disk.read_text()).get("chosen")
            n["gate"]["run_gate_json_agrees"] = written == chosen_key
    if chosen_key is None:
        n["controls"] = None
        return n
    rnd, rev = chosen_key.replace("steer:", "random:", 1), chosen_key.replace("steer:", "reverse:", 1)
    steer = pick(rows, "R", "A", chosen_key, R)
    ctl = {"A": pick(rows, "R", "A", "none", R), "B": pick(rows, "R", "B", "none", R), "Steer": steer,
           "Random": pick(rows, "R", "A", rnd, R), "Reverse": pick(rows, "R", "A", rev, R),
           "A-cut": pick(rows, "R", "cut", "none", R), "Steer-cut": pick(rows, "R", "cut", chosen_key, R)}
    n["controls"] = {k: rate_block(v, gold, samples) for k, v in ctl.items()}
    n["controls_unseen"] = {k: rate_block({q: r for q, r in v.items() if q in unseen}, gold, samples)
                            for k, v in ctl.items()}
    n["cut_no_marker"] = sum(r["cut_no_marker"] for r in ctl["Steer-cut"].values())
    dmg = {"A": pick(rows, "C", "A", "none", C), "Steer": pick(rows, "C", "A", chosen_key, C),
           "Random": pick(rows, "C", "A", rnd, C), "Reverse": pick(rows, "C", "A", rev, C)}
    n["damage"] = {k: rate_block(v, gold, samples) for k, v in dmg.items()}
    g = lambda rs: {q: 1.0 if gold(r) else 0.0 for q, r in rs.items()}  # noqa: E731
    cd = paired_claim(g(dmg["Steer"]), g(dmg["A"]), samples=samples)
    cd.pop("verdict", None)
    cd["damages"] = cd.get("mean") is not None and cd["mean"] < -DAMAGE_DROP
    n["control_C"] = cd
    s_ans, r_ans = n["controls"]["Steer"]["ans"], n["controls"]["Random"]["ans"]
    a_ans, v_ans = n["controls"]["A"]["ans"], n["controls"]["Reverse"]["ans"]
    n["control_X"] = {"random_ans": r_ans, "steer_ans": s_ans,
                      "bound": None if s_ans is None else st.RANDOM_SHARE * s_ans,
                      "pass": None if s_ans is None or r_ans is None else r_ans <= st.RANDOM_SHARE * s_ans,
                      "reverse_ans": v_ans, "A_ans": a_ans,
                      "reverse_descriptive": None if v_ans is None or a_ans is None else v_ans <= a_ans + 0.05}
    if not confirm:
        return n
    # --- confirmation claims ---
    keys = sorted(set(steer) & set(ctl["Steer-cut"]))
    c1 = paired_claim({q: g(steer)[q] for q in keys}, {q: g(ctl["Steer-cut"])[q] for q in keys}, samples=samples)
    c1["n"] = len(keys)
    c1["p_steer"] = sum(g(steer)[q] for q in keys) / len(keys) if keys else None
    c1["p_steer_cut"] = sum(g(ctl["Steer-cut"])[q] for q in keys) / len(keys) if keys else None
    c1["meaningful"] = c1["p_steer"] is not None and c1["p_steer"] >= MEANINGFUL
    c1["underpowered"] = len(keys) < UNDERPOWERED_BELOW
    if not sanity_pass:
        c1["verdict"] = "NOT READ (sanity S failed)"
    elif not n["control_X"]["pass"]:
        c1["verdict"] = "NOT INTERPRETABLE (Control X failed)"
    n["claim1"] = c1
    clean = [q for q in clean_subset(d, keys)]
    c1p = paired_claim({q: g(steer)[q] for q in clean}, {q: g(ctl["Steer-cut"])[q] for q in clean}, samples=samples)
    c1p["n"] = len(clean)
    if c1["verdict"] == "NOT SHOWN" and c1p["verdict"] == "SHOWN":
        c1p["record"] = "shown on the clean subset only (descriptive)"
    n["claim1_prime"] = c1p
    if c1["verdict"] == "SHOWN":
        b_gold = g(ctl["B"])
        cond = lambda kind: [q for q in keys if pop.b_outcome.get(q) == kind]  # noqa: E731
        n["claim2"] = {"ratio": ratio_bootstrap(g(steer), b_gold, samples),
                       "p_gold_steer_given_B_correct": bootstrap_proportion([gold(steer[q]) for q in cond("correct")], samples=samples),
                       "p_gold_steer_given_B_wrong": bootstrap_proportion([gold(steer[q]) for q in cond("wrong")], samples=samples),
                       "p_ans_steer_given_B_refuses": bootstrap_proportion([steer[q]["cls"] == "answer" for q in cond("refusal")],
                                                                           samples=samples)}
    types: dict[str, list[str]] = defaultdict(list)
    for q in keys:
        types[d.by_q[q].question_type].append(q)
    n["control_T"] = {t: {"n": len(qs), "steer": sum(g(steer)[q] for q in qs) / len(qs),
                          "steer_cut": sum(g(ctl["Steer-cut"])[q] for q in qs) / len(qs)} for t, qs in sorted(types.items())}
    return n


def _r(x) -> str:
    return "-" if x is None else f"{x:.3f}"


def _p(s: dict) -> str:
    return "-" if not s.get("n") else f"{s['p']:.3f} ({s['ci_low']:.3f}, {s['ci_high']:.3f})"


def _d(s: dict) -> str:
    if s.get("mean") is None:
        return "-"
    return f"{s['mean']:+.3f} ({s['ci_low']:+.3f}, {s['ci_high']:+.3f})" if s.get("ci_low") is not None else f"{s['mean']:+.3f}"


def render(n: dict, confirm: bool) -> str:
    rp, pops = n["repro"], n["populations"]
    L = [f"# §38 report ({'confirmation' if confirm else 'design run'}; folds {', '.join(map(str, n['folds']))})", "",
         f"Engine {n['engines']}. Gold: {n['gold_source']}.", "",
         "## Reproduction", "",
         f"Rows reproduce (refusal flag, and the D1 string verdict when answering) on {_r(rp['rate'])} of the R and damage "
         f"rows (R alone {_r(rp['rate_R'])}; D1 verdict also matches on {rp.get('R_verdict_match')} R rows, descriptive; bound {st.REPRO_GATE}): **{'PASS' if rp['pass'] else 'FAIL'}**. "
         f"Dropped: {', '.join(rp['dropped']) or 'none'}.", "",
         f"Populations: R {pops['R']} (reproduced {pops['R_reproduced']}), pairs {pops['pairs']} ({pops['pairs_reproduced']}), "
         f"unseen {pops['unseen']} ({pops['unseen_reproduced']}), C rows run {pops['C_reproduced']}.", ""]
    if n["directions"]:
        L += ["## Directions", "", "| layer | pairs | ‖v_L‖ | mean ‖resid_A‖ | ratio |", "|---|---|---|---|---|"]
        for k, s in n["directions"].items():
            L.append(f"| {k} | {s['n_pairs']} | {s['v_norm']:.3f} | {s['mean_resid_A_norm']:.3f} | {_r(s['v_over_resid'])} |")
        L.append("")
    s = n["sanity"]
    if s:
        L += ["## Sanity S (plumbing)", "",
              f"Final layer (L{s['final_layer']}) overwrite with B's state gives B's first token on "
              f"{_r(s['final_overwrite_rate'])} of {s['n_final']} pairs (bound {st.SANITY_FINAL_GATE}); zero-vector add "
              f"byte-exact on all {s['n_zero']}: {s['zero_exact_all']}; self-overwrite byte-exact on all {s['n_self']}: "
              f"{s['self_exact_all']}; 0.2 depth (L{s['shallow_layer']}) rate {_r(s['shallow_overwrite_rate'])} (descriptive). "
              f"**{'PASS' if s['pass'] else 'FAIL'}**.", ""]
    L += ["## Settings on R (reproduced rows; unseen = R minus the estimation pairs)", "",
          "| setting | n | ans | degenerate | refusal | gold (D1, provisional) | unseen n | unseen ans | unseen degenerate |",
          "|---|---|---|---|---|---|---|---|---|"]
    for k, v in n["settings"].items():
        a, u = v["all"], v["unseen"]
        L.append(f"| {k} | {a['n']} | {_r(a['ans'])} | {_r(a['degenerate'])} | {_r(a['refusal'])} | {_r(a['gold_d1'])} | "
                 f"{u['n']} | {_r(u['ans'])} | {_r(u['degenerate'])} |")
    gate = n["gate"]
    L += ["", "## Design gate", ""]
    if confirm:
        L.append(f"Fixed from {gate['fixed_from']}: {gate['chosen']} (development gate pass {gate['pass']}).")
    else:
        L.append(f"Rules: sanity S passes; reproduction >= {st.REPRO_GATE}; settings in order of ans(Steer) (ties: shallower "
                 f"layer, then smaller alpha); eligible if degenerate <= {st.DEGENERATE_MAX} and ans(Random) <= "
                 f"{st.RANDOM_SHARE} x ans(Steer); the chosen setting needs ans(Steer) >= {st.ANS_GATE}.")
        L.append("")
        for e in gate["order"]:
            L.append(f"- {e['setting']}: ans {_r(e['ans'])}, degenerate {_r(e['degenerate'])}"
                     + (f", Random {_r(e.get('random_ans'))} vs bound {_r(e.get('random_bound'))}" if "random_ans" in e else "")
                     + f" -> {e.get('why')}")
        L += ["", f"Chosen: **{gate['chosen']}**; gate **{'PASS' if gate['pass'] else 'FAIL' if gate['pass'] is False else 'PENDING'}** "
              f"(status {gate['status']}; sanity {gate['sanity_pass']}, reproduction {_r(gate['repro_rate'])})."
              + ("" if gate.get("run_gate_json_agrees", True) else " WARNING: the run's gate.json chose differently.")]
    if n.get("controls"):
        L += ["", "## At the chosen setting (R rows)", "",
              "| arm | n | ans (95% CI) | degenerate | refusal | gold (95% CI) | unseen ans | unseen gold |", "|---|---|---|---|---|---|---|---|"]
        for k, v in n["controls"].items():
            u = n["controls_unseen"][k]
            L.append(f"| {k} | {v['n']} | {_p(v['ans_ci'])} | {_r(v['degenerate'])} | {_r(v['refusal'])} | {_p(v['gold'])} | "
                     f"{_r(u['ans'])} | {_r(u['gold']['p'])} |")
        L += ["", f"Cut prompts without an N-cut marker (no note): {n['cut_no_marker']}. A-cut (alpha 0) is the Steer-cut baseline.", "",
              "Damage / C rows:", "", "| arm | n | ans | degenerate | refusal | gold |", "|---|---|---|---|---|---|"]
        for k, v in n["damage"].items():
            L.append(f"| {k} | {v['n']} | {_r(v['ans'])} | {_r(v['degenerate'])} | {_r(v['refusal'])} | {_p(v['gold'])} |")
        cx, cc = n["control_X"], n["control_C"]
        L += ["", f"Control X: ans(Random) {_r(cx['random_ans'])} <= {st.RANDOM_SHARE} x ans(Steer) {_r(cx['steer_ans'])} = "
              f"{_r(cx['bound'])}: **{cx['pass']}**. Reverse (descriptive): ans {_r(cx['reverse_ans'])} vs A {_r(cx['A_ans'])} + 0.05: "
              f"{cx['reverse_descriptive']}.",
              f"Control C: gold(Steer) − gold(A) on C rows {_d(cc)}" + (" — the direction damages correct answers." if cc["damages"] else ".")]
    if confirm and "claim1" in n:
        c1, c1p = n["claim1"], n["claim1_prime"]
        L += ["", "## Claims", "",
              f"Claim 1: P(gold(Steer)) − P(gold(Steer-cut)) on R, n {c1['n']}: {_d(c1)} — **{c1['verdict']}**"
              + (" UNDERPOWERED" if c1["underpowered"] else "")
              + f"; P(gold(Steer)) {_r(c1['p_steer'])} (meaningful at {MEANINGFUL}: {c1['meaningful']}), "
              f"P(gold(Steer-cut)) {_r(c1['p_steer_cut'])}.",
              f"Claim 1′ (gold fails D1 in the cut prompt and the N-cut note), n {c1p['n']}: {_d(c1p)} — {c1p['verdict']}"
              + (f" ({c1p['record']})" if c1p.get("record") else "") + "."]
        if "claim2" in n:
            c2 = n["claim2"]
            L.append(f"Claim 2: P(gold(Steer)) / P(gold(B)) {_r(c2['ratio']['ratio'])} ({_r(c2['ratio']['ci_low'])}, "
                     f"{_r(c2['ratio']['ci_high'])}); P(gold(Steer) | B correct) {_p(c2['p_gold_steer_given_B_correct'])}; "
                     f"| B wrong {_p(c2['p_gold_steer_given_B_wrong'])}; P(ans(Steer) | B refuses) {_p(c2['p_ans_steer_given_B_refuses'])}.")
        L += ["", "Control T (descriptive):", "", "| type | n | gold(Steer) | gold(Steer-cut) |", "|---|---|---|---|"]
        for t, v in n["control_T"].items():
            L.append(f"| {t} | {v['n']} | {v['steer']:.3f} | {v['steer_cut']:.3f} |")
    if n.get("costs"):
        L += ["", f"Costs: {n['costs']}."]
    return "\n".join(L) + "\n"


def costs(run_dir: Path) -> dict:
    secs, gens, toks = 0.0, 0, 0
    seen_prefill = set()
    for stage in STAGE_FILES:
        for r in read_jsonl(run_dir / f"{stage}.jsonl"):
            t = r["timing"]
            secs += t["copy_s"] + t["first_step_s"] + t["decode_s"]
            pk = (stage, r["pop"], r["kind"], r["question_id"], round(t["prefill_s"], 6))
            if pk not in seen_prefill:
                seen_prefill.add(pk)
                secs += t["prefill_s"]
            gens += 1
            toks += t["n_new"]
    return {"generations": gens, "prefills": len(seen_prefill), "tokens_generated": toks, "seconds": round(secs, 1)}


def build(run_dir: Path, confirm: bool, samples: int = 10000, exp35: Path = Path("runs/exp35"), runs: str = "runs",
          own_root: Path = Path("runs/exp34_jlens_own"), design: Path | None = None) -> tuple[str, dict]:
    rows = read_rows(run_dir)
    _, folds = check_rows(rows, run_dir, confirm)
    from scripts.exp38_dev import load_source
    d, pop = load_source(sorted(folds) or [0, 1], exp35, runs, own_root)
    numbers = analyse(run_dir, d, pop, confirm, samples, design)
    numbers["costs"] = costs(run_dir)
    return render(numbers, confirm), numbers


def out_paths(out: Path) -> tuple[Path, Path]:
    return out.with_suffix(".md"), out.with_suffix(".json")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="runs/exp38_dev/qwen3b")
    parser.add_argument("--exp35", default="runs/exp35")
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--own-root", default="runs/exp34_jlens_own")
    parser.add_argument("--design", default=None, help="confirm mode: the development gate.json")
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--confirm", action="store_true", help="required for any of folds 2, 3, 4")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    try:
        text, numbers = build(Path(args.run_dir), args.confirm, args.samples, Path(args.exp35), args.runs,
                              Path(args.own_root), Path(args.design) if args.design else None)
    except Refused as exc:
        print(exc.code if isinstance(exc.code, str) else exc)
        return 2
    md, js = out_paths(Path(args.out or f"runs/exp38_report_{'confirm' if args.confirm else 'dev'}.md"))
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text(text)
    js.write_text(json.dumps(numbers, indent=1, default=float))
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
