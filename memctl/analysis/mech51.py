"""§51 analysis kit: per-base values from graded rows, the R family (repair, O1 / O2 per comparator, Holm, the B1
equivalence), M1 (revert, B6 seed-mean), M2 (transplant, clamp), M3 (activity tests, the Commit interaction, the
pooled AUROC predictor) and the nuisance baseline. Pure functions over graded records (no model); the model-side
interventions are in scripts/exp51_mech.py. Spec: proposal rev 1a §4.3, §6, §7, §8, §13.1, §13.4 (D-72).

Graded records: dicts with case_id, base_key, task, set, cls (grade51 classes) and, for populations, the B0 labels.
An endpoint is (population, outcome): population in POPS (FA, FA-ab, C, W, UNA, CF, ALL) and outcome a grade51
RATE_HITS key or a class name (indicator of that class).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np

from memctl.analysis import grade51 as GR
from memctl.analysis import stats51 as S

POPS = ("FA", "FA-ab", "C", "W", "UNA", "CF", "ALL")
NI_MARGIN, CEIL, MIN_BASES, DEG_M_CAP = 0.05, 0.95, 20, 0.10
R_COMPARATORS = ("B1", "B2", "B3", "B4", "B5")


def hit_fn(outcome: str):
    if outcome in GR.RATE_HITS:
        return GR.RATE_HITS[outcome]
    if outcome in GR.CLASSES:
        return lambda r: r["cls"] == outcome
    raise ValueError(f"unknown outcome {outcome!r}")


def in_pop(labels: Sequence[str], pop: str) -> bool:
    return pop == "ALL" or pop in labels


def per_base(graded: dict[str, dict], labels: dict[str, list[str]], meta: dict[str, dict], pop: str,
             outcome: str) -> dict[str, float]:
    """Base-weighted values: base -> share of its rows in `pop` with the outcome. graded: case_id -> record with cls;
    labels: case_id -> B0 population labels (fixed from B0, §4.1); meta: case_id -> row (base_key, set)."""
    h = hit_fn(outcome)
    by = defaultdict(list)
    for cid, rec in graded.items():
        if rec is None or "cls" not in rec or not in_pop(labels.get(cid, []), pop):
            continue
        by[meta[cid]["base_key"]].append(1.0 if h(rec) else 0.0)
    return {b: float(np.mean(v)) for b, v in by.items()}


def mixture(graded: dict[str, dict], meta: dict[str, dict], lam: float = 1.0) -> dict[str, float]:
    rows = [{"base_key": meta[c]["base_key"], "set": meta[c]["set"], "cls": r["cls"]} for c, r in graded.items()
            if r is not None and "cls" in r]
    return GR.base_mixture(rows, lam)


def bases_of(meta: dict[str, dict]) -> S.Bases:
    return S.Bases({r["base_key"]: r["task"] for r in meta.values()})


def paired(bases: S.Bases, a: dict, b: dict, B: int = S.B_DEFAULT, seed: int = S.SEED) -> dict:
    return S.bootstrap(S.diff_stat(bases, a, b), bases.n_by_task(), B, seed)


def summary(res: dict) -> dict:
    return {"est": res["est"], **S.bounds(res), "B_valid": res["B_valid"], "n_fail": res["n_fail"],
            "fail_ok": res["fail_ok"]}


# ---------------------------------------------------------------------------------------------- family R

def rates(graded, labels, meta) -> dict:
    return {"Rel": per_base(graded, labels, meta, "FA", "Rel"), "Rel_ab": per_base(graded, labels, meta, "FA-ab", "Rel"),
            "Uns": per_base(graded, labels, meta, "UNA", "Uns"), "Dam": per_base(graded, labels, meta, "C", "Dam"),
            "CF_follow": per_base(graded, labels, meta, "CF", "CF_follow"),
            "Deg": per_base(graded, labels, meta, "ALL", "Deg"), "U1": mixture(graded, meta, 1.0),
            "U2": mixture(graded, meta, 2.0)}


def family_r(cand: dict, comps: dict[str, dict], b0: dict, bases: S.Bases, B: int = S.B_DEFAULT) -> dict:
    """cand / comps[k] / b0: rates() outputs (on H, the frozen doses). Returns the repair checks, O1 / O2 per
    comparator with Holm over all 10, and the B1 equivalence (TOST ±0.05 on ΔRel and ΔUns)."""
    out = {"repair": {}, "tests": {}, "holm": None}
    u = paired(bases, cand["U1"], b0["U1"], B)
    dam = paired(bases, cand["Dam"], b0["Dam"], B)
    out["repair"] = {"U_minus_B0": {**summary(u), "p": S.p_greater(u, 0.0)},
                     "Dam_minus_B0": summary(dam), "Deg": S.bw_mean(cand["Deg"], list(cand["Deg"])),
                     "CF_follow": S.bw_mean(cand["CF_follow"], list(cand["CF_follow"]))}
    rp = out["repair"]
    rp["is_repair"] = bool(rp["U_minus_B0"]["p"] is not None and rp["U_minus_B0"]["p"] <= 0.025 and
                           dam["est"] is not None and dam["est"] <= NI_MARGIN and rp["Deg"] is not None and
                           rp["Deg"] <= 0.02 and rp["CF_follow"] is not None and rp["CF_follow"] >= 0.90)
    ps = {}
    for k in R_COMPARATORS:
        if k not in comps:
            continue
        drel = paired(bases, cand["Rel"], comps[k]["Rel"], B)
        duns = paired(bases, cand["Uns"], comps[k]["Uns"], B)
        o1 = S.iut([S.p_greater(drel, 0.0), S.p_less(duns, NI_MARGIN)])
        o2 = S.iut([S.p_less(duns, 0.0), S.p_greater(drel, -NI_MARGIN)])
        out["tests"][k] = {"dRel": summary(drel), "dUns": summary(duns), "O1_p": o1, "O2_p": o2}
        ps[f"{k}:O1"], ps[f"{k}:O2"] = o1, o2
        if k == "B1":
            out["tests"][k]["equiv_B1"] = {"dRel_tost_p": S.tost(drel, NI_MARGIN), "dUns_tost_p": S.tost(duns, NI_MARGIN)}
    out["holm"] = S.holm(ps)
    better = {k: any(out["holm"].get(f"{k}:{o}", {}).get("reject") for o in ("O1", "O2")) for k in out["tests"]}
    out["better_than"] = {k: bool(v and rp["is_repair"]) for k, v in better.items()}
    out["better_than_every"] = bool(rp["is_repair"] and len(better) == len(R_COMPARATORS) and all(better.values()))
    return out


def r_entry(cand_S: dict, b0_S: dict, bases: S.Bases, fa_bases_by_task: dict[str, int]) -> dict:
    """§8.1 entry on S: U − B0 one-sided 97.5% lower bound > 0 and >= 20 FA bases in DE or RC."""
    u = paired(bases, cand_S["U1"], b0_S["U1"])
    lo = S.bounds(u)["lo"]
    support = max(fa_bases_by_task.get("DE", 0), fa_bases_by_task.get("RC", 0)) >= MIN_BASES
    return {"U_minus_B0": summary(u), "entry": bool(lo is not None and lo > 0 and support), "support_ok": support}


# ---------------------------------------------------------------------------------------------- dose rules

def pick_dose(scores: dict, closeness: dict) -> object:
    """§5.3 / §13.5: max score; ties to the setting closest to no-op (smaller closeness), then the lexicographic id."""
    best = max(scores.values())
    tied = [d for d, s in scores.items() if s == best]
    return sorted(tied, key=lambda d: (closeness[d], str(d)))[0]


def m_dose(effects: dict, degs: dict, sign: int, closeness: dict) -> object | None:
    """§5.3 M dose: the largest effect in the declared sign with Deg <= 0.10; ties as pick_dose."""
    ok = {d: sign * e for d, e in effects.items() if e is not None and degs.get(d, 1.0) <= DEG_M_CAP}
    return pick_dose(ok, closeness) if ok else None


def closest_b1(target: float, b1_effects: dict[float, float]) -> float:
    """§6.3b: the B1 signed-grid value whose S endpoint effect is closest to the f-intervention's (ties: smaller
    |bias|, then the lexicographic value)."""
    return sorted(b1_effects, key=lambda v: (abs(b1_effects[v] - target), abs(v), str(v)))[0]


# ---------------------------------------------------------------------------------------------- family M

def oriented_test(bases: S.Bases, a: dict, b: dict, sign: int, B: int = S.B_DEFAULT) -> dict:
    res = paired(bases, a, b, B)
    return {**summary(res), "p": S.p_sided(res, sign, 0.0)}


def m1(bases, endpoint_star: dict, endpoint_revert: dict, endpoint_b6_mean: dict, sign: int, B=S.B_DEFAULT) -> dict:
    a = oriented_test(bases, endpoint_star, endpoint_revert, sign, B)
    b = oriented_test(bases, endpoint_star, endpoint_b6_mean, sign, B)
    return {"M1a": a, "M1b": b, "p": S.iut([a["p"], b["p"]])}


def m2(bases, transplant: dict, b0: dict, star: dict, clamp: dict, sign: int, B=S.B_DEFAULT) -> dict:
    a = oriented_test(bases, transplant, b0, sign, B)
    b = oriented_test(bases, star, clamp, sign, B)
    return {"M2a": a, "M2b": b, "p": S.iut([a["p"], b["p"]])}


def seed_mean(per_seed: list[dict]) -> dict:
    """B6: the per-base mean over the seeds (bases present in every seed)."""
    keys = set.intersection(*(set(d) for d in per_seed)) if per_seed else set()
    return {k: float(np.mean([d[k] for d in per_seed])) for k in keys}


def activity_directional(bases, delta_f: dict[str, float], sign: int, B=S.B_DEFAULT) -> dict:
    """§6.3a directional: the mean per-base Δf has the stated sign (one-sided)."""
    res = S.bootstrap(lambda idx: S.bw_mean(delta_f, bases.resampled(idx)), bases.n_by_task(), B)
    return {**summary(res), "p": S.p_sided(res, sign, 0.0)}


def activity_equivalence(bases, delta_f: dict[str, float], sd_S: float, B=S.B_DEFAULT) -> dict:
    """§6.3a / §13.4a equivalence: TOST with margin 2 x (SD on S of the per-base mean Δf); SD 0 -> unestimable (p 1)
    with the |Δf| <= 1e-6 check reported."""
    margin = 2.0 * sd_S
    res = S.bootstrap(lambda idx: S.bw_mean(delta_f, bases.resampled(idx)), bases.n_by_task(), B)
    out = {**summary(res), "margin": margin, "p": S.tost(res, margin)}
    if sd_S == 0:
        out["unestimable"] = True
        out["all_within_1e-6"] = bool(all(abs(v) <= 1e-6 for v in delta_f.values()))
    return out


def sd_of_base_means(delta_f_S: dict[str, float]) -> float:
    return float(np.std(list(delta_f_S.values()), ddof=1)) if len(delta_f_S) > 1 else 0.0


def interaction(bases, fint_fa, b1_fa, fint_una, b1_una, sign: int, B=S.B_DEFAULT) -> dict:
    """§13.4b Commit interaction: [(f-int − B1) on FA] − [(f-int − B1) on UNA], one-sided in the declared sign."""
    res = S.bootstrap(S.interaction_stat(bases, fint_fa, b1_fa, fint_una, b1_una), bases.n_by_task(), B)
    return {**summary(res), "p": S.p_sided(res, sign, 0.0)}


# ---------------------------------------------------------------------------------------------- predictor (M3c)

def design_matrix(rows: list[dict], cols: dict | None = None) -> tuple[np.ndarray, dict]:
    """One-hot categoricals + standardised numerics. rows: dicts with "cat" (name -> value) and "num" (name -> float).
    `cols` = the encoding fitted on S (categories, means, SDs), reused unchanged on H."""
    if cols is None:
        cats = {k: sorted({str(r["cat"][k]) for r in rows}) for k in sorted(rows[0]["cat"])}
        nums = sorted(rows[0]["num"])
        mu = {k: float(np.mean([r["num"][k] for r in rows])) for k in nums}
        sd = {k: float(np.std([r["num"][k] for r in rows])) or 1.0 for k in nums}
        cols = {"cats": cats, "nums": nums, "mu": mu, "sd": sd}
    X = []
    for r in rows:
        x = []
        for k, vals in cols["cats"].items():
            x += [1.0 if str(r["cat"][k]) == v else 0.0 for v in vals]
        x += [(r["num"][k] - cols["mu"][k]) / cols["sd"][k] for k in cols["nums"]]
        X.append(x)
    return np.asarray(X), cols


def cell_encoding(rows: list[dict], min_n: int = 5) -> dict:
    """§13.4c encoding rule: baseline city x task cells with fewer than 5 rows of a class are encoded at task x
    hardness. Returns {(task, city): "city" | "task_hardness"} (frozen at F2)."""
    cnt = defaultdict(lambda: [0, 0])
    for r in rows:
        cnt[(r["task"], r["city"])][r["label"]] += 1
    return {f"{t}|{c}": ("city" if min(v) >= min_n else "task_hardness") for (t, c), v in cnt.items()}


def fit_logistic(X: np.ndarray, y: np.ndarray):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=1.0, max_iter=5000).fit(X, y)      # L2 penalty (the default), C = 1


def predictor_rows(recs: list[dict], enc: dict, with_f: bool) -> list[dict]:
    """recs: dicts with task, hardness, city, label (1 = FA, 0 = answered gold), m_city, unk_logodds, f."""
    out = []
    for r in recs:
        cell = enc.get(f"{r['task']}|{r['city']}", "task_hardness")
        cat = {"task": r["task"], "hardness": r["hardness"],
               "city": f"{r['task']}|{r['city']}" if cell == "city" else f"{r['task']}|{r['hardness']}"}
        num = {"m_city": r["m_city"], "unk_logodds": r["unk_logodds"], **({"f": r["f"]} if with_f else {})}
        out.append({"cat": cat, "num": num, "label": r["label"], "base_key": r["base_key"]})
    return out


def m3c(train: list[dict], test: list[dict], bases_test: S.Bases, B=S.B_DEFAULT) -> dict:
    """Fit baseline and full on S (train), score H (test) without refitting; paired pooled AUROC difference, one-sided
    (full > baseline), bootstrap stratified by task. Both classes must be present overall."""
    enc = cell_encoding(train)
    fitted = {}
    for name, wf in (("baseline", False), ("full", True)):
        tr = predictor_rows(train, enc, wf)
        X, cols = design_matrix(tr)
        m = fit_logistic(X, np.array([r["label"] for r in tr]))
        te = predictor_rows(test, enc, wf)
        Xt, _ = design_matrix(te, cols)
        fitted[name] = {"cols": cols, "coef": m.coef_.tolist(), "intercept": m.intercept_.tolist(),
                        "scores": m.decision_function(Xt)}
    by = defaultdict(list)
    for r, sf, sb in zip(test, fitted["full"]["scores"], fitted["baseline"]["scores"]):
        by[r["base_key"]].append((float(sf), float(sb), int(r["label"])))
    res = S.bootstrap(S.auroc_diff_stat(bases_test, by), bases_test.n_by_task(), B)
    return {**summary(res), "p": S.p_greater(res, 0.0), "encoding": enc,
            "auroc_full": S.auroc(fitted["full"]["scores"], [r["label"] for r in test]),
            "auroc_baseline": S.auroc(fitted["baseline"]["scores"], [r["label"] for r in test]),
            "coef": {k: {"coef": v["coef"], "intercept": v["intercept"], "cols": v["cols"]} for k, v in fitted.items()}}


def nuisance_features(row: dict) -> dict:
    """§4.3 nuisance baseline features (exclude the answerability-defining relation)."""
    facts = row["facts"]
    q = row["query_entity"] if row["set"] != "UNA-swap" else row["A"]
    first = next((i for i, f in enumerate(facts) if q in f), len(facts))
    rel = defaultdict(int)
    for d in row.get("distractors", []):
        rel[d["relation"]] += 1
    return {"cat": {"task": row["task"], "hardness": row["hardness"], "template": row["set"] + row["rep"]},
            "num": {"n_distractors": len(row.get("distractors", [])), "rel_lives": rel["lives"], "rel_job": rel["job"],
                    "rel_works_in": rel["works_in"], "memory_len": len(row["prompt"].split("## Memory")[1]),
                    "first_line_of_asked": first}}


def nuisance_auroc(train_rows: list[dict], test_rows: list[dict]) -> dict:
    """ANS (1) vs UNA (0) from the nuisance features, fitted on S, reported on H. Note: the template feature uses
    set + rep, which encodes ANS vs UNA directly, so it is replaced by the hardness slot only (see build notes)."""
    def enc(rs):
        out = []
        for r in rs:
            f = nuisance_features(r)
            f["cat"].pop("template")
            out.append({**f, "label": 1 if r["set"] == "ANS" else 0})
        return out
    tr, te = enc(train_rows), enc(test_rows)
    X, cols = design_matrix(tr)
    m = fit_logistic(X, np.array([r["label"] for r in tr]))
    Xt, _ = design_matrix(te, cols)
    a = S.auroc(m.decision_function(Xt), [r["label"] for r in te])
    return {"auroc": a, "limitation": bool(a is not None and a >= CEIL)}
