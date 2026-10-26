"""§40 report: the fixed layer-29 setting on LoCoMo. Claim 1, Control X, Control C, Claim 2 (descriptive), by type,
the adversarial block, the §41 block, costs.

    python scripts/exp40_report.py --run-dir runs/exp40_pod/qwen3b --confirm [--out runs/exp40_report_confirm.md]

Pre-registration: research/proposals/2026-10-23-exp40-fixed-setting-validation.md §4 (rev 0) with rev 1 (877f86d) and
the Advisor's selection ruling; §41 rev 0 §5 / rev 1 for the conditional block. The LoCoMo outcomes are read only with
--confirm (exit 2 otherwise). Everything is recomputed from the rows; the run's selection.json is cross-checked.

- gold(row) = this run's 7B verdict is yes AND the row's class is 'answer' (§38's rule); a row without a verdict is
  left out of every gold figure and counted.
- Claim 1: P(gold(Steer-all)) - P(gold(Steer-all-cut)) on R, paired bootstrap 10,000 over rows; SHOWN if the lower
  bound > 0; meaningful if P(gold(Steer-all)) >= 0.30. R < 40: descriptive, no verdict (rev 1).
- Control X: ans(Random) <= 0.2 x ans(Steer-all) on R, else NOT INTERPRETABLE.
- Control C: P(gold(Steer-all)) - P(gold(A)) on C, paired; harmful if the upper bound < -0.10.
- Claim 2 (descriptive): P(gold(Steer-all)) against P(gold(B)) on R, paired, and by question type.
- Adversarial (40 category-5 rows with a marker): answer rates of A, B, Steer-all (no judge), and Cond's (§41).
- §41 (rule file shipped and scored_on_40 true): signals from the A and A-cut prefills, Cond = Steer-all where the
  frozen rule fires else A; U = +1 correct, -lambda wrong, 0 refusal / degenerate / soft refusal; U(Cond) - U(Steer)
  and U(Cond) - U(B) on R at lambda 1 (claims) and 2 (beside), paired bootstrap 10,000; coverage and risk; the rule's
  AUROC for gold(Steer). Rule missing or gate failed: the signals are reported only (descriptive AUROCs).
- Partial rows: an arm cut by the deadline is reported on the rows it has, with its count against the selection.

§42 (`--exp exp42`, proposal 2026-10-25 rev 1 at 8fd2613; confirm mode only): the same readouts on the fresh rows
(descriptive context, labelled), then the §42 block that carries the verdicts. Cond = Steer-all iff A refuses (step-0
argmax == rule42's refusal id on the unsteered A pass) AND s_proj >= tau (rule42.json, sha checked against the run's
manifest), else A's output. On R, lambda 1 (lambda 2 beside), paired bootstrap 10,000: Claim 1 U(Cond) - U(B), Claim 2
U(Cond) - U(A), Claim 3 U(Cond) - U(Steer-all) with U(Steer) - U(A) and the §41 reading rule beside, each SHOWN if the
lower bound > 0; Claim 4 AUROC(s_proj) for gold(Steer-all), SHOWN if the lower bound > 0.5. Control D on C: gold(Cond)
- 1 with the fire count (0 by construction; harmful if UB < -0.10). Control U: Cond's answer rate on the adversarial
rows beside A, B, Steer-all. Descriptive: precision of fires, coverage / risk, the frozen q curve (tau_q from §40's R
rows; only q = 0.3 carries a verdict) with the post hoc top-q on these rows beside, by type. Utility, soft-refusal
pattern, classifier, judge and lambda are §40/§41's (memctl/analysis/conditional.py).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from memctl.analysis import conditional as cd  # noqa: E402
from memctl.analysis import exp40 as ex  # noqa: E402
from memctl.analysis import steering as st  # noqa: E402
from memctl.analysis.jacobian_stats import bootstrap_proportion, paired_claim  # noqa: E402
from scripts.exp35_generate import read_jsonl  # noqa: E402

MEANINGFUL = 0.30
HARM = -0.10
PRICE_PER_H = 0.59
CATS = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop"}


class Refused(SystemExit):
    pass


class Data:
    def __init__(self, run_dir: Path) -> None:
        self.dir = run_dir
        self.manifest = json.loads((run_dir / "manifest.json").read_text())
        self.selection = json.loads((run_dir / "selection.json").read_text())
        self.view = json.loads((run_dir / "view.json").read_text())
        self.layer = self.manifest["layer"]
        self.rows: dict[tuple, dict] = {}
        engines = set()
        for f in ("apass.jsonl", "arms.jsonl"):
            for r in read_jsonl(run_dir / f):
                self.rows[st.row_key(r)] = r
                engines.add(r["engine"])
        if len(engines) > 1:
            raise Refused(f"REFUSED: more than one steering engine in {run_dir}: {sorted(engines)}")
        jr = read_jsonl(run_dir / "judge.jsonl")
        jengines = {r["judge_engine"] for r in jr}
        if len(jengines) > 1:
            raise Refused(f"REFUSED: more than one judge engine in {run_dir}: {sorted(jengines)}")
        self.judge_engine = next(iter(jengines), None)
        self.verdict = {(r["pop"], r["kind"], r["setting_key"], r["question_id"]): st.parse_verdict(r["reply"])
                        for r in jr}
        self.missing: set = set()
        self.key = {n: ex.setting_of(n, self.layer).key for n in ("steer_all", "steer0", "random", "none")}

    def row(self, pop, kind, name, q):
        return self.rows.get((pop, kind, self.key[name], q))

    def gold(self, pop, kind, name, q):
        r = self.row(pop, kind, name, q)
        if r is None:
            return None
        v = self.verdict.get((pop, kind, self.key[name], q))
        if v is None:
            self.missing.add((pop, kind, name, q))
            return None
        return bool(v) and r["cls"] == "answer"

    def a(self, q, pop="all"):
        """The A arm: the A-pass row of the 600 ('all') or the adversarial block ('adv')."""
        return self.rows.get((pop, "A", "none", q))


def golds(d: Data, spec: tuple, qs: list[str]) -> dict[str, float]:
    """{q: 1.0 / 0.0} for the rows with a verdict; the unsteered A arm is the A-pass row of the 600."""
    pop, kind, name = spec
    out = {}
    for q in qs:
        g = d.gold("all", "A", "none", q) if (kind == "A" and name == "none") else d.gold(pop, kind, name, q)
        if g is not None:
            out[q] = 1.0 if g else 0.0
    return out


def gold_A(d: Data, qs):
    out = {}
    for q in qs:
        g = d.gold("all", "A", "none", q)
        if g is not None:
            out[q] = 1.0 if g else 0.0
    return out


def ans_of(rows: list[dict]) -> dict:
    n = len(rows)
    return {"n": n, "ans": sum(r["cls"] == "answer" for r in rows) / n if n else None,
            "refusal": sum(r["cls"] == "refusal" for r in rows) / n if n else None,
            "degenerate": sum(r["cls"] == "degenerate" for r in rows) / n if n else None,
            "soft": sum(cd.soft_refusal(r["phrase"], r["cls"]) for r in rows) / n if n else None}


def arm_rows(d: Data, pop: str, kind: str, name: str, qs: list[str]) -> dict[str, dict]:
    if kind == "A" and name == "none":
        return {q: d.a(q, "adv" if pop == "adv" else "all") for q in qs if d.a(q, "adv" if pop == "adv" else "all")}
    return {q: r for q in qs if (r := d.row(pop, kind, name, q)) is not None}


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


# --- §41 --------------------------------------------------------------------------------------------------------

def outcome_of(d: Data, pop, kind, name, q):
    r = d.a(q) if (kind == "A" and name == "none") else d.row(pop, kind, name, q)
    if r is None:
        return None
    v = d.verdict.get(("all", "A", "none", q) if (kind == "A" and name == "none") else (pop, kind, d.key[name], q))
    if v is None and r["cls"] == "answer" and not cd.soft_refusal(r["phrase"], r["cls"]):
        return None
    return cd.outcome(r["cls"], r["phrase"], bool(v))


def signals_for(d: Data, q: str, pop: str = "all", cut_pop: str = "R") -> dict | None:
    a = d.a(q, pop)
    if a is None or "step0" not in a:
        return None
    cut = d.rows.get((cut_pop, "cut", "none", q))
    m_cut = cut["step0"]["margin"] if cut and "step0" in cut else None
    return {"s_margin": a["step0"]["margin"], "s_proj": a["s_proj"],
            "s_evid": (a["step0"]["margin"] - m_cut) if m_cut is not None else None}


def auroc_bootstrap(scores: dict[str, float], labels: dict[str, bool], samples: int = 10000, seed: int = 0) -> dict:
    """AUROC with a 95% percentile interval from resampling rows (resamples without both classes are skipped and
    counted); §41 amendment 1 Claim 4: SHOWN if the lower bound > 0.5."""
    keys = sorted(set(scores) & set(labels))
    point = cd.auroc([scores[k] for k in keys], [labels[k] for k in keys])
    rng, vals, skipped = random.Random(seed), [], 0
    for _ in range(samples if keys else 0):
        pick = rng.choices(keys, k=len(keys))
        a = cd.auroc([scores[k] for k in pick], [labels[k] for k in pick])
        if a is None:
            skipped += 1
        else:
            vals.append(a)
    vals.sort()
    lo = vals[int(0.025 * len(vals))] if vals else None
    hi = vals[min(int(0.975 * len(vals)), len(vals) - 1)] if vals else None
    return {"n": len(keys), "auroc": point, "ci_low": lo, "ci_high": hi, "skipped_resamples": skipped,
            "verdict": "SHOWN" if lo is not None and lo > 0.5 else "NOT SHOWN"}


def conditional_block(d: Data, R: list[str], rule: dict | None, rule_info: dict, adv: list[str], samples: int) -> dict:
    out = {"rule": rule_info}
    sig = {q: signals_for(d, q) for q in R}
    gs = golds(d, ("R", "A", "steer_all"), R)
    desc = {}
    for name in cd.SIGNALS:
        vals = {q: s[name] for q, s in sig.items() if s and s.get(name) is not None and q in gs}
        desc[name] = {"n": len(vals), "mean": mean(vals.values()),
                      "auroc_gold_steer_higher": cd.auroc([vals[q] for q in vals], [bool(gs[q]) for q in vals])}
    out["signals"] = desc
    # §41 amendment 1 Claim 4: AUROC of s_proj for gold(Steer-all) on R, bootstrap over rows; the others beside
    out["claim4"] = {name: auroc_bootstrap({q: s[name] for q, s in sig.items() if s and s.get(name) is not None},
                                           {q: bool(g) for q, g in gs.items()}, samples=samples)
                     for name in cd.SIGNALS}
    if rule is None or not rule_info.get("scored"):
        out["scored"] = False
        return out
    if list(rule.get("refusal_start_ids", [])) != list(d.manifest["refusal_ids"]) and d.manifest["engine"] != "stub":
        raise Refused("REFUSED: the run's refusal ids differ from the rule's; the margins are not the rule's signal")
    pol = {"A": ("R", "A", "none"), "B": ("R", "B", "none"), "Steer": ("R", "A", "steer_all")}
    oc: dict[str, dict] = {p: {} for p in ("A", "B", "Steer", "Cond")}
    fires, skipped = {}, defaultdict(int)
    for q in R:
        s = sig.get(q)
        o = {p: outcome_of(d, *spec, q) for p, spec in pol.items()}
        if s is None or s.get(rule["signal"]) is None:
            skipped["no signal"] += 1
            continue
        if any(v is None for v in o.values()):
            skipped["an outcome without a verdict or a row"] += 1
            continue
        f = cd.apply_rule(rule, s)
        fires[q] = f
        for p in pol:
            oc[p][q] = o[p]
        oc["Cond"][q] = o["Steer"] if f else o["A"]
    out["scored"], out["n"], out["skipped"], out["fires"] = True, len(fires), dict(skipped), sum(fires.values())
    for lam in (cd.LAMBDA, cd.LAMBDA_BESIDE):
        u = {p: {q: cd.utility(o, lam) for q, o in oc[p].items()} for p in oc}
        blk = {"U": {p: sum(u[p].values()) for p in u},
               "cond_minus_steer": paired_claim(u["Cond"], u["Steer"], samples=samples),
               "cond_minus_B": paired_claim(u["Cond"], u["B"], samples=samples),
               "cond_minus_A": paired_claim(u["Cond"], u["A"], samples=samples),        # amendment 1 Claim 3
               "steer_minus_A": paired_claim(u["Steer"], u["A"], samples=samples)}      # the Claim 1 reading rule
        blk["claim1_reading"] = ("expected under a net-negative steer; not evidence of a selective signal"
                                 if blk["U"]["Steer"] <= blk["U"]["A"] else "steer is net positive here")
        # descriptive, post hoc: steer the top fraction q of R rows by s_proj, A elsewhere
        ranked = sorted((q for q in oc["A"] if sig.get(q) and sig[q].get("s_proj") is not None),
                        key=lambda q: -sig[q]["s_proj"])
        curve = []
        for i in range(11):
            k = round(len(ranked) * i / 10)
            top = set(ranked[:k])
            curve.append({"q": i / 10, "steered": k,
                          "U": sum(cd.utility(oc["Steer"][q] if q in top else oc["A"][q], lam) for q in ranked)})
        blk["s_proj_top_q"] = curve
        out[f"lambda{lam}"] = blk
    out["coverage_risk"] = {p: cd.coverage_risk(oc[p]) for p in oc}
    fired = [q for q, f in fires.items() if f and q in gs]
    out["fire_precision"] = {"fired": len(fired), "gold_rate_fired": mean([gs[q] for q in fired]) if fired else None,
                             "gold_rate_steer_all": mean([gs[q] for q in fires if q in gs]) if fires else None}
    sign = 1.0 if rule["sign"] == "+" else -1.0
    qs = [q for q in fires if q in gs]
    out["auroc_rule_gold_steer"] = cd.auroc([sign * sig[q][rule["signal"]] for q in qs], [bool(gs[q]) for q in qs])
    # adversarial: Cond's answer rate beside Steer-all's (descriptive)
    ans_c, n_c = 0, 0
    for q in adv:
        s = signals_for(d, q, "adv", "adv")
        a, sa = d.a(q, "adv"), d.row("adv", "A", "steer_all", q)
        if s is None or s.get(rule["signal"]) is None or a is None or sa is None:
            continue
        r = sa if cd.apply_rule(rule, s) else a
        ans_c += r["cls"] == "answer"
        n_c += 1
    out["adversarial_cond_ans"] = {"n": n_c, "ans": ans_c / n_c if n_c else None}
    return out


# --- analysis -----------------------------------------------------------------------------------------------------

# --- §42 -------------------------------------------------------------------------------------------------------

def load_rule42(path: Path, d: Data) -> tuple[dict, dict]:
    if not path.exists():
        raise Refused(f"REFUSED: §42 rule {path} missing")
    rule = json.loads(path.read_text())
    info = dict(d.manifest.get("rule42") or {})
    sha = ex_file_sha(path)
    if not info:
        raise Refused("REFUSED: the run's manifest has no §42 rule (not an exp42 run)")
    if info.get("sha256") != sha:
        raise Refused("REFUSED: the §42 rule file differs from the one the run recorded")
    if d.manifest["engine"] != "stub":
        ex.check_rule42(rule)
        if info.get("refusal_id_used") != rule["refusal_id"]:
            raise Refused("REFUSED: the run's refusal id differs from the rule's")
    info["report_rule_sha256"] = sha
    return rule, info


def cond42_fire(d: Data, q: str, pop: str, tau: float, rid: int) -> bool | None:
    a = d.a(q, pop)
    if a is None or "step0" not in a or a.get("s_proj") is None:
        return None
    return ex.cond42_fires(a["step0"]["argmax"], a["s_proj"], tau, rid)


def selective_block(d: Data, R: list[str], C: list[str], adv: list[str], rule: dict, info: dict,
                    samples: int) -> dict:
    tau, rid = float(rule["tau"]), int(info["refusal_id_used"])
    out: dict = {"rule": info, "tau": tau, "tau_method": rule.get("tau_method"), "refusal_id": rid,
                 "rule_text": rule.get("rule"), "distribution": ex.IN_DISTRIBUTION}
    # the run's logged fires against the report's recomputation
    logged = {r["question_id"]: r["cond42"]["fires"] for (pop, kind, key, q), r in d.rows.items()
              if kind == "A" and key == "none" and "cond42" in r}
    pol = {"A": ("R", "A", "none"), "B": ("R", "B", "none"), "Steer": ("R", "A", "steer_all")}
    oc: dict[str, dict] = {p: {} for p in ("A", "B", "Steer", "Cond")}
    fires, skipped, mismatch = {}, defaultdict(int), []
    for q in R:
        f = cond42_fire(d, q, "all", tau, rid)
        if f is None:
            skipped["no A row or signal"] += 1
            continue
        if q in logged and logged[q] != f:
            mismatch.append(q)
        o = {p: outcome_of(d, *spec, q) for p, spec in pol.items()}
        if any(v is None for v in o.values()):
            skipped["an outcome without a verdict or a row"] += 1
            continue
        fires[q] = f
        for p in pol:
            oc[p][q] = o[p]
        oc["Cond"][q] = o["Steer"] if f else o["A"]
    out.update({"n": len(fires), "fires": sum(fires.values()), "skipped": dict(skipped),
                "refuse_step0_R": sum(d.a(q)["step0"]["argmax"] == rid for q in R if d.a(q) and "step0" in d.a(q)),
                "logged_mismatch": mismatch})
    for lam in (cd.LAMBDA, cd.LAMBDA_BESIDE):
        u = {p: {q: cd.utility(o, lam) for q, o in oc[p].items()} for p in oc}
        blk = {"U": {p: sum(u[p].values()) for p in u},
               "claim1_cond_minus_B": paired_claim(u["Cond"], u["B"], samples=samples),
               "claim2_cond_minus_A": paired_claim(u["Cond"], u["A"], samples=samples),
               "claim3_cond_minus_steer": paired_claim(u["Cond"], u["Steer"], samples=samples),
               "steer_minus_A": paired_claim(u["Steer"], u["A"], samples=samples)}
        blk["claim3_reading"] = ("expected under a net-negative steer; not evidence of a selective signal"
                                 if blk["U"]["Steer"] <= blk["U"]["A"] else "steer is net positive here")
        # the frozen q curve: tau_q from §40's R rows (rule42.json); only q 0.3 carries a verdict
        curve = []
        for qk, tq in sorted(rule.get("tau_curve", {}).items(), key=lambda kv: float(kv[0])):
            fq = {q: ex.cond42_fires(d.a(q)["step0"]["argmax"], d.a(q)["s_proj"], tq, rid) for q in oc["A"]}
            curve.append({"q": float(qk), "tau": tq, "fires": sum(fq.values()), "verdict": float(qk) == ex.Q42,
                          "U": sum(cd.utility(oc["Steer"][q] if fq[q] else oc["A"][q], lam) for q in oc["A"])})
        blk["frozen_q_curve"] = curve
        # post hoc, descriptive: steer the top fraction q of these R rows by s_proj (among step-0 refusals)
        ranked = sorted((q for q in oc["A"] if d.a(q)["step0"]["argmax"] == rid), key=lambda q: -d.a(q)["s_proj"])
        ph = []
        for i in range(1, 10):
            top = set(ranked[:round(len(ranked) * i / 10)])
            ph.append({"q": i / 10, "steered": len(top),
                       "U": sum(cd.utility(oc["Steer"][q] if q in top else oc["A"][q], lam) for q in oc["A"])})
        blk["posthoc_top_q"] = ph
        if len(R) < ex.R_MIN_FOR_CLAIMS:
            for k in ("claim1_cond_minus_B", "claim2_cond_minus_A", "claim3_cond_minus_steer"):
                blk[k]["verdict"] = f"descriptive (R = {len(R)} < {ex.R_MIN_FOR_CLAIMS}), no verdict"
        out[f"lambda{lam}"] = blk
    # Claim 4 (replication): AUROC(s_proj) for gold(Steer-all) on R
    gs = golds(d, ("R", "A", "steer_all"), R)
    sp = {q: d.a(q)["s_proj"] for q in R if d.a(q) and d.a(q).get("s_proj") is not None}
    out["claim4"] = auroc_bootstrap(sp, {q: bool(g) for q, g in gs.items()}, samples=samples)
    if len(R) < ex.R_MIN_FOR_CLAIMS:
        out["claim4"]["verdict"] = f"descriptive (R = {len(R)} < {ex.R_MIN_FOR_CLAIMS}), no verdict"
    # Control D (damage) on C: gold(Cond) - 1; Cond fires only on step-0 refusals, so 0 fires by construction
    fC = {q: cond42_fire(d, q, "all", tau, rid) for q in C}
    gA, gS = gold_A(d, C), golds(d, ("C", "A", "steer_all"), C)
    gcond = {}
    for q, f in fC.items():
        if f is None:
            continue
        src = gS if f else gA
        if q in src:
            gcond[q] = src[q]
    dd = paired_claim(gcond, gA, samples=samples)
    dd.update({"fires": sum(bool(f) for f in fC.values()), "rows": len(gcond),
               "p_gold_cond": mean(gcond.values()), "p_gold_A": mean(gA[q] for q in gcond if q in gA)})
    dd["harmful"] = dd["ci_high"] is not None and dd["ci_high"] < HARM
    out["controlD"] = dd
    # Control U (unanswerable): answer rates on the marked adversarial rows
    rows_u = {"A": [], "B": [], "Steer-all": [], "Cond": []}
    adv_fires = 0
    for q in adv:
        a, b, sa = d.a(q, "adv"), d.row("adv", "B", "none", q), d.row("adv", "A", "steer_all", q)
        f = cond42_fire(d, q, "adv", tau, rid)
        if a is None or sa is None or f is None:
            continue
        adv_fires += f
        rows_u["A"].append(a)
        rows_u["Steer-all"].append(sa)
        rows_u["Cond"].append(sa if f else a)
        if b is not None:
            rows_u["B"].append(b)
    out["controlU"] = {**{k: ans_of(v) for k, v in rows_u.items()}, "fires": adv_fires, "n_marked": len(adv)}
    # descriptive
    fired = [q for q, f in fires.items() if f]
    out["fire_precision"] = {"fired": len(fired), "gold_rate_fired": mean(gs[q] for q in fired if q in gs),
                             "outcomes_fired": dict(sorted(Counter(oc["Steer"][q] for q in fired).items())),
                             "gold_rate_steer_all": mean(gs[q] for q in fires if q in gs)}
    out["coverage_risk"] = {p: cd.coverage_risk(oc[p]) for p in oc}
    by = defaultdict(list)
    for q in fires:
        by[d.a(q).get("category")].append(q)
    out["by_type"] = {CATS.get(c, str(c)): {"n": len(qs), "fires": sum(fires[q] for q in qs),
                                            **{f"U_{p}": sum(cd.utility(oc[p][q], 1) for q in qs) for p in oc}}
                      for c, qs in sorted(by.items(), key=lambda kv: (kv[0] is None, kv[0]))}
    return out


def analyse(run_dir: Path, samples: int, rule_path: Path | None, exp: str = "exp40",
            rule42_path: Path | None = None) -> dict:
    d = Data(run_dir)
    sel_ = d.selection
    R, C = sel_["R"], sel_["C"]
    order = d.view["rows"]
    a_rows = {q: r for (pop, kind, key, q), r in d.rows.items() if pop == "all" and kind == "A" and key == "none"}
    verdict = {q: d.verdict.get(("all", "A", "none", q)) for q in a_rows}
    re_sel = ex.select(order, a_rows, verdict)
    n: dict = {"engine": d.manifest["engine"], "judge_engine": d.judge_engine, "layer": d.layer,
               "view": d.view["counts"], "view_shas": d.view["shas"], "counts_600": sel_["counts"],
               "selection_recomputed_equal": re_sel == {"R": R, "C": C}, "n_R": len(R), "n_C": len(C),
               "descriptive": len(R) < ex.R_MIN_FOR_CLAIMS, "manifest": d.manifest, "exp": exp,
               "seed": d.manifest.get("seed", ex.EXPS[exp]["seed"])}
    if d.manifest.get("exp", "exp40") != exp:
        raise Refused(f"REFUSED: the run is {d.manifest.get('exp', 'exp40')}, the report was asked for {exp}")
    plan = json.loads((run_dir / "plan.json").read_text()) if (run_dir / "plan.json").exists() else {}
    n["plan"] = plan
    n["partial"] = bool(plan.get("cut"))
    ag = run_dir / "agreement.json"
    n["agreement"] = json.loads(ag.read_text()) if ag.exists() else None
    if n["agreement"]:
        n["agreement"].pop("rows", None)
    sm = run_dir / "smoke" / "smoke.json"
    if sm.exists():
        s = json.loads(sm.read_text())
        n["smoke"] = {"notes_check": s["notes_check"], "verdicts": s["verdicts"], "problems": s["problems"],
                      "projection": s.get("projection")}

    # arms on R
    arms = {"A": ("R", "A", "none"), "B": ("R", "B", "none"), "Steer-all": ("R", "A", "steer_all"),
            "Steer-all-cut": ("R", "cut", "steer_all"), "A-cut": ("R", "cut", "none"), "Random": ("R", "A", "random"),
            "Steer-0": ("R", "A", "steer0")}
    table = {}
    G = {}
    for name, (pop, kind, s) in arms.items():
        rows = arm_rows(d, pop, kind, s, R)
        G[name] = gold_A(d, R) if name == "A" else golds(d, (pop, kind, s), R)
        table[name] = {**ans_of(list(rows.values())), "rows": len(rows), "of": len(R),
                       "gold": bootstrap_proportion([bool(v) for v in G[name].values()], samples=samples),
                       "no_verdict": len(rows) - len(G[name])}
    n["R_table"] = table
    c1 = paired_claim(G["Steer-all"], G["Steer-all-cut"], samples=samples)
    p_s = table["Steer-all"]["gold"]["p"]
    c1["p_gold_steer"] = p_s
    c1["meaningful"] = p_s is not None and p_s >= MEANINGFUL
    if n["descriptive"]:
        c1["verdict"] = f"descriptive (R = {len(R)} < {ex.R_MIN_FOR_CLAIMS}), no verdict"
    n["claim1"] = c1
    a_s, a_r = table["Steer-all"]["ans"], table["Random"]["ans"]
    n["controlX"] = {"ans_steer": a_s, "ans_random": a_r, "bound": None if a_s is None else st.RANDOM_SHARE * a_s,
                     "pass": a_s is not None and a_r is not None and a_r <= st.RANDOM_SHARE * a_s}
    n["controlX"]["verdict"] = "passes" if n["controlX"]["pass"] else "NOT INTERPRETABLE"
    gC_s = golds(d, ("C", "A", "steer_all"), C)
    gC_a = gold_A(d, C)
    cc = paired_claim(gC_s, gC_a, samples=samples)
    cc["p_gold_steer"] = mean(gC_s.values())
    cc["p_gold_A"] = mean(gC_a.values())
    cc["harmful"] = cc["ci_high"] is not None and cc["ci_high"] < HARM
    cc["rows"] = len(arm_rows(d, "C", "A", "steer_all", C))
    cc["C_ans_steer"] = ans_of(list(arm_rows(d, "C", "A", "steer_all", C).values()))
    n["controlC"] = cc
    c2 = paired_claim(G["Steer-all"], G["B"], samples=samples)
    c2["verdict"] = "descriptive"
    n["claim2"] = c2
    by = defaultdict(list)
    for q in R:
        by[a_rows[q].get("category") if q in a_rows else None].append(q)
    n["by_type"] = {CATS.get(c, str(c)): {"n": len(qs), **{name: {"gold": mean(G[name][q] for q in qs if q in G[name]),
                                                                    "n_gold": sum(q in G[name] for q in qs)}
                                                             for name in ("Steer-all", "Steer-all-cut", "B", "A-cut")},
                                          "ans_random": mean(float(r["cls"] == "answer") for r in
                                                             arm_rows(d, "R", "A", "random", qs).values())}
                    for c, qs in sorted(by.items(), key=lambda kv: (kv[0] is None, kv[0]))}
    adv = d.selection.get("adversarial_marked", [])
    n["adversarial"] = {name: ans_of(list(arm_rows(d, "adv", kind, s, adv).values()))
                        for name, (kind, s) in {"A": ("A", "none"), "B": ("B", "none"),
                                                "Steer-all": ("A", "steer_all")}.items()}
    n["adversarial"]["n_marked"] = len(adv)
    n["adversarial"]["n_block"] = len(d.view.get("adversarial", []))
    rule, rule_info = None, {"present": False, "scored": False, "why": "no rule file"}
    if rule_path and rule_path.exists():
        rule = json.loads(rule_path.read_text())
        rule_info = dict(d.manifest.get("rule", {}))
        rule_info.setdefault("scored", bool(rule.get("scored_on_40")))
        rule_info["report_rule_sha256"] = ex_file_sha(rule_path)
        if rule_info.get("sha256") and rule_info["sha256"] != rule_info["report_rule_sha256"]:
            raise Refused("REFUSED: the rule file differs from the one the run recorded")
    else:
        rule_info = dict(d.manifest.get("rule", rule_info))
        rule_info["scored"] = False
    n["s41"] = conditional_block(d, R, rule, rule_info, adv, samples)
    if exp == "exp42":
        r42, i42 = load_rule42(rule42_path or Path(ex.RULE42_PATH), d)
        n["s42"] = selective_block(d, R, C, adv, r42, i42, samples)
    n["gold_missing"] = len(d.missing)
    n["costs"] = costs(run_dir, d)
    return n


def ex_file_sha(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def costs(run_dir: Path, d: Data) -> dict:
    out = {}
    tp = run_dir / "timing_pod.json"
    if tp.exists():
        out["stage_s"] = json.loads(tp.read_text()).get("stage_s")
    jm = run_dir / "judge_manifest.json"
    if jm.exists():
        j = json.loads(jm.read_text())
        out["judge"] = {k: j.get(k) for k in ("n_items", "n_judged", "n_prompts_called", "seconds", "unparseable", "cut")}
    gens = defaultdict(lambda: [0, 0, 0.0])
    for r in d.rows.values():
        g = gens[r["stage"]]
        t = r["timing"]
        g[0] += 1
        g[1] += t["n_new"]
        g[2] += t["copy_s"] + t["first_step_s"] + t["decode_s"]
    out["generations"] = {k: {"rows": v[0], "tokens": v[1], "gen_seconds": round(v[2], 1)} for k, v in gens.items()}
    total = (out.get("stage_s") or {}).get("total")
    out["run_minutes"] = None if total is None else total / 60
    out["price_per_h"] = PRICE_PER_H
    return out


# --- rendering ----------------------------------------------------------------------------------------------------

def _p(x) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def _ci(c: dict) -> str:
    return f"{_p(c.get('mean'))} [{_p(c.get('ci_low'))}, {_p(c.get('ci_high'))}], n {c.get('n')}"


def render(n: dict) -> str:
    if n.get("exp") == "exp42":
        L = ["# §42: selective steering by s_proj on fresh LoCoMo rows", "",
             "The frozen rule from §40's rows (rule42.json) on rows §40 never used; " + ex.IN_DISTRIBUTION + ".", ""]
    else:
        L = ["# §40: the fixed layer-29 setting (alpha 2) on LoCoMo", "",
             "The §38 setting, fixed in advance, on a set it had never seen.", ""]
    v, c6 = n["view"], n["counts_600"]
    pools = (f"; pools after excluding §40's {v['exclude_ids']} ids: {v['pool_answerable']} answerable / "
             f"{v['pool_adversarial']} adversarial" if "pool_answerable" in v else "")
    L.append(f"Engine {n['engine']}; judge {n['judge_engine']}; layer {n['layer']}. Set: {v['rows']} answerable rows "
             f"+ {v['adversarial']} adversarial (seed {n['seed']}){pools}; excluded {v['excluded']}; rows by category "
             f"{v['rows_by_category']}; zero-fill views {v['zero_fill_rows']}. locomo10.json sha "
             f"{n['view_shas']['locomo10.json'][:12]}.")
    L.append(f"On the {c6['n']}: marker rate {_p(c6['marker_rate'])} ({c6['marked']}), refusal rate {_p(c6['refusal_rate'])} "
             f"({c6['refusals']}), judged-correct rate {_p(c6['judged_correct_rate'])} ({c6['judged_correct']}). "
             f"R {n['n_R']} (cap {ex.R_CAP}), C {n['n_C']} (cap {ex.C_CAP}); selection recomputed equal: "
             f"{n['selection_recomputed_equal']}.")
    ag = n.get("agreement")
    if ag:
        L.append(f"Judge agreement on {ag['n']} cached §35 rows: {ag['agree']}/{ag['n']} = {_p(ag['rate'])} (bar "
                 f"{ag['bar']}; {'met' if ag['pass'] else 'NOT MET'}).")
    bc = n["manifest"].get("bf16_cast", {}).get("steer", {})
    L.append(f"Direction: v_29 sha {n['manifest']['vectors']['v_sha256'][:8]}..., steer 2 v_29 sha "
             f"{n['manifest']['vectors']['steer_sha256'][:8]}...; bf16 cast cosine {bc.get('cosine_bf16_fp32', float('nan')):.6f}.")
    if n["partial"]:
        L.append(f"**PARTIAL**: cut by the deadline: {n['plan'].get('cut')}; every figure below is on the rows present.")
    if n["descriptive"]:
        L.append(f"**R = {n['n_R']} < {ex.R_MIN_FOR_CLAIMS}: the claims are descriptive, no verdict (rev 1).**")
    if n.get("s42"):
        L += render42(n)
        L += ["", "## §40's readouts on the fresh rows (context; §42's verdicts are in the block above)"]
    L += ["", "## Arms on R", "", "| arm | rows | ans | refusal | degenerate | soft | P(gold) [95% CI] | no verdict |",
          "|---|---|---|---|---|---|---|---|"]
    for name, t in n["R_table"].items():
        g = t["gold"]
        L.append(f"| {name} | {t['rows']}/{t['of']} | {_p(t['ans'])} | {_p(t['refusal'])} | {_p(t['degenerate'])} | "
                 f"{_p(t['soft'])} | {_p(g['p'])} [{_p(g['ci_low'])}, {_p(g['ci_high'])}] | {t['no_verdict']} |")
    c1 = n["claim1"]
    L += ["", f"**Claim 1** P(gold(Steer-all)) - P(gold(Steer-all-cut)) = {_ci(c1)}: **{c1['verdict']}**; "
              f"P(gold(Steer-all)) {_p(c1['p_gold_steer'])} ({'meaningful' if c1['meaningful'] else 'below 0.30'})."]
    cx = n["controlX"]
    L.append(f"**Control X** ans(Random) {_p(cx['ans_random'])} against 0.2 x ans(Steer-all) = {_p(cx['bound'])}: "
             f"**{cx['verdict']}**.")
    cc = n["controlC"]
    L.append(f"**Control C** (C, {cc['rows']} rows) P(gold(Steer-all)) {_p(cc['p_gold_steer'])} - P(gold(A)) "
             f"{_p(cc['p_gold_A'])} = {_ci(cc)}; upper bound {_p(cc['ci_high'])}: "
             f"**{'HARMFUL' if cc['harmful'] else 'not harmful'}** (harmful if UB < -0.10).")
    c2 = n["claim2"]
    L.append(f"**Claim 2** (descriptive) P(gold(Steer-all)) - P(gold(B)) on R = {_ci(c2)}.")
    L += ["", "## By question type (R)", "", "| type | n | Steer-all | Steer-all-cut | B | A-cut | ans(Random) |",
          "|---|---|---|---|---|---|---|"]
    for t, b in n["by_type"].items():
        L.append(f"| {t} | {b['n']} | {_p(b['Steer-all']['gold'])} | {_p(b['Steer-all-cut']['gold'])} | "
                 f"{_p(b['B']['gold'])} | {_p(b['A-cut']['gold'])} | {_p(b['ans_random'])} |")
    a = n["adversarial"]
    L += ["", f"## Adversarial block ({a['n_marked']} marked of {a['n_block']}; answer rate only, no judge)", ""]
    for name in ("A", "B", "Steer-all"):
        L.append(f"- {name}: ans {_p(a[name]['ans'])} (n {a[name]['n']}), soft {_p(a[name]['soft'])}")
    s = n["s41"]
    L += ["", "## §41: the frozen conditional rule", "", f"Rule: {json.dumps(s['rule'])}."]
    L.append("Signals on R (descriptive): " + "; ".join(
        f"{k} n {v['n']}, mean {_p(v['mean'])}, AUROC(gold Steer, higher) {_p(v['auroc_gold_steer_higher'])}"
        for k, v in s["signals"].items()) + ".")
    c4 = s["claim4"]
    L.append(f"- Claim 4 (amendment 1): AUROC of s_proj for gold(Steer-all) on R {_p(c4['s_proj']['auroc'])} "
             f"({_p(c4['s_proj']['ci_low'])}, {_p(c4['s_proj']['ci_high'])}), n {c4['s_proj']['n']}: "
             f"**{c4['s_proj']['verdict']}** (bar: lower bound > 0.5); beside, descriptive: "
             + "; ".join(f"{k} {_p(v['auroc'])} ({_p(v['ci_low'])}, {_p(v['ci_high'])})"
                         for k, v in c4.items() if k != "s_proj") + ".")
    if s.get("scored"):
        for lam in (1, 2):
            b = s[f"lambda{lam}"]
            tag = " (claims)" if lam == 1 else " (beside)"
            L.append(f"- lambda {lam}{tag}: U {b['U']}; Claim 1 U(Cond) - U(Steer) {_ci(b['cond_minus_steer'])} "
                     f"**{b['cond_minus_steer']['verdict']}**, beside U(Steer) - U(A) {_ci(b['steer_minus_A'])}: "
                     f"{b['claim1_reading']}; Claim 2 U(Cond) - U(B) {_ci(b['cond_minus_B'])} "
                     f"**{b['cond_minus_B']['verdict']}**; Claim 3 U(Cond) - U(A) {_ci(b['cond_minus_A'])} "
                     f"**{b['cond_minus_A']['verdict']}**")
            L.append(f"  - post hoc (descriptive): steer the top fraction q by s_proj: " + ", ".join(
                f"q {c['q']:.1f} U {c['U']:g}" for c in b["s_proj_top_q"]))
        fp = s["fire_precision"]
        L.append(f"- rule fires: gold rate among fired rows {_p(fp['gold_rate_fired'])} (n {fp['fired']}) against "
                 f"Steer-all's overall {_p(fp['gold_rate_steer_all'])}")
        L.append(f"- rows {s['n']}, fires {s['fires']}, skipped {s['skipped']}; rule AUROC for gold(Steer) "
                 f"{_p(s['auroc_rule_gold_steer'])}; adversarial ans Cond {_p(s['adversarial_cond_ans']['ans'])} "
                 f"(n {s['adversarial_cond_ans']['n']}) beside Steer-all {_p(n['adversarial']['Steer-all']['ans'])}")
        L.append("- coverage / risk: " + "; ".join(f"{p} {_p(c['coverage'])} / {_p(c['risk'])}"
                                                 for p, c in s["coverage_risk"].items()))
    else:
        L.append("Not scored: " + str(s["rule"].get("why", "rule missing or gate failed")) + "; signals logged only.")
    L += ["", "## Costs", ""]
    c = n["costs"]
    if c.get("stage_s"):
        L.append("Stage seconds: " + ", ".join(f"{k} {v:.0f}" for k, v in c["stage_s"].items()) + ".")
    if c.get("judge"):
        L.append(f"Judge: {c['judge']}.")
    L.append(f"Generations by file: {c['generations']}. Rows without a verdict left out: {n['gold_missing']}.")
    if c.get("run_minutes") is not None:
        L.append(f"Run {c['run_minutes']:.1f} min of model work (pod minutes include install and loads; price "
                 f"${c['price_per_h']}/h).")
    if n.get("smoke"):
        L.append(f"Smoke: {n['smoke']['notes_check']}.")
    return "\n".join(L) + "\n"


def render42(n: dict) -> list[str]:
    s = n["s42"]
    L = ["", "## §42: the frozen selective policy (the verdicts)", "",
         f"Rule: {s['rule_text']}. tau = {s['tau']!r} ({s['tau_method']}); refusal id {s['refusal_id']}; rule42.json "
         f"sha {s['rule']['report_rule_sha256'][:12]}... (equal to the run's). {s['distribution'][:1].upper()}"
         f"{s['distribution'][1:]}.",
         f"R rows scored {s['n']} (step-0 refusals among R {s['refuse_step0_R']}); Cond fires on {s['fires']}; skipped "
         f"{s['skipped']}; logged fires that differ from the recomputation: {len(s['logged_mismatch'])}."]
    if n["descriptive"]:
        L.append(f"**R = {n['n_R']} < {ex.R_MIN_FOR_CLAIMS}: descriptive, no verdict.**")
    for lam in (1, 2):
        b = s[f"lambda{lam}"]
        tag = "claims" if lam == 1 else "beside"
        L += ["", f"lambda {lam} ({tag}): U {b['U']}",
              f"- **Claim 1** U(Cond) - U(B) {_ci(b['claim1_cond_minus_B'])}: **{b['claim1_cond_minus_B']['verdict']}**",
              f"- **Claim 2** U(Cond) - U(A) {_ci(b['claim2_cond_minus_A'])}: **{b['claim2_cond_minus_A']['verdict']}**",
              f"- **Claim 3** U(Cond) - U(Steer-all) {_ci(b['claim3_cond_minus_steer'])}: "
              f"**{b['claim3_cond_minus_steer']['verdict']}**; beside U(Steer) - U(A) {_ci(b['steer_minus_A'])}: "
              f"{b['claim3_reading']}",
              "- frozen q curve (tau_q from §40's R rows; only q = 0.3 carries a verdict, the rest is descriptive): "
              + ", ".join(f"q {c['q']:.1f} (tau {c['tau']:.3f}, fires {c['fires']}) U {c['U']:g}"
                          + (" [verdict]" if c["verdict"] else "") for c in b["frozen_q_curve"]),
              "- post hoc top-q by s_proj among these R rows' step-0 refusals (descriptive): "
              + ", ".join(f"q {c['q']:.1f} ({c['steered']}) U {c['U']:g}" for c in b["posthoc_top_q"])]
    c4 = s["claim4"]
    L += ["", f"**Claim 4** (replication) AUROC(s_proj) for gold(Steer-all) on R {_p(c4['auroc'])} "
              f"({_p(c4['ci_low'])}, {_p(c4['ci_high'])}), n {c4['n']}: **{c4['verdict']}** (bar: lower bound > 0.5)."]
    cd_ = s["controlD"]
    L.append(f"**Control D** (C, {cd_['rows']} rows) gold(Cond) - 1 = {_ci(cd_)}; C rows that fire {cd_['fires']} "
             f"(0 by construction); upper bound {_p(cd_['ci_high'])}: "
             f"**{'HARMFUL' if cd_['harmful'] else 'not harmful'}** (harmful if UB < -0.10).")
    cu = s["controlU"]
    L.append(f"**Control U** (adversarial, {cu['n_marked']} marked; a cost whatever the claims say): answer rate Cond "
             f"{_p(cu['Cond']['ans'])} (n {cu['Cond']['n']}, fires {cu['fires']}) beside A {_p(cu['A']['ans'])}, "
             f"B {_p(cu['B']['ans'])} (n {cu['B']['n']}), Steer-all {_p(cu['Steer-all']['ans'])}.")
    fp = s["fire_precision"]
    L += ["", f"Precision of fires: gold rate of Steer-all among fired rows {_p(fp['gold_rate_fired'])} (n "
              f"{fp['fired']}; outcomes {fp['outcomes_fired']}) against Steer-all's overall {_p(fp['gold_rate_steer_all'])}.",
          "Coverage / risk: " + "; ".join(f"{p} {_p(c['coverage'])} / {_p(c['risk'])} (correct {c['correct']}, wrong "
                                          f"{c['wrong']}, of {c['n']})" for p, c in s["coverage_risk"].items()) + ".",
          "", "| type | n | fires | U(Cond) | U(A) | U(B) | U(Steer) |", "|---|---|---|---|---|---|---|"]
    for t, b in s["by_type"].items():
        L.append(f"| {t} | {b['n']} | {b['fires']} | {b['U_Cond']} | {b['U_A']} | {b['U_B']} | {b['U_Steer']} |")
    return L


def build(run_dir: Path, confirm: bool, samples: int = 10000, rule: Path | None = Path("runs/exp41_dev/qwen3b/rule.json"),
          exp: str = "exp40", rule42: Path | None = None):
    if not confirm:
        raise Refused(f"REFUSED: the {'§42' if exp == 'exp42' else '§40'} rows are the new set; the report reads them "
                      f"only with --confirm")
    n = analyse(run_dir, samples, rule, exp, rule42)
    return render(n), n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="exp40", choices=tuple(ex.EXPS))
    ap.add_argument("--run-dir", default=None, help="default runs/<exp>_pod/qwen3b")
    ap.add_argument("--rule42", default=ex.RULE42_PATH, help="§42 only")
    ap.add_argument("--rule", default="runs/exp41_dev/qwen3b/rule.json")
    ap.add_argument("--samples", type=int, default=10000)
    ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--out", default=None, help="default runs/<exp>_report_confirm.md")
    args = ap.parse_args(argv)
    args.run_dir = args.run_dir or f"{ex.EXPS[args.exp]['out']}/qwen3b"
    args.out = args.out or f"runs/{args.exp}_report_confirm.md"
    try:
        text, n = build(Path(args.run_dir), args.confirm, args.samples, Path(args.rule) if args.rule else None,
                        args.exp, Path(args.rule42))
    except Refused as exc:
        print(exc.code if isinstance(exc.code, str) else exc)
        return 2
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    out.with_suffix(".json").write_text(json.dumps(n, indent=1, default=float))
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
