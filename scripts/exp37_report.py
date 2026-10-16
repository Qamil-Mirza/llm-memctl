"""§37 report: the elenchus as a self-check for the 3B's forced answers. Implements proposal §3-§5.

    .venv/bin/python scripts/exp37_report.py --folds 0,1 --root runs/exp37_dev [--out runs/exp37_report_dev.md]
    .venv/bin/python scripts/exp37_report.py --folds 2,3,4 --confirm --root runs/exp37
    .venv/bin/python scripts/exp37_report.py --folds 0,1,2,3,4 --confirm --root runs/exp37

Pre-registration: research/proposals/2026-10-16-exp37-elenchus.md at 714da06 (rev 2). 3B primary; 7B descriptive.

Rows. The fixed-candidate rows and labels are §36's (scripts/exp36_report.py Frame over §35 ReaderData): marked 3B rows
passing the §35 token check with an arm-B verdict; rows without a candidate are forced abstentions in every coverage
and risk denominator; discrimination is over rows with a candidate. Per question and item: g from E1, d from E2 (blind),
a from E3; e = 0 when g = 0 or d is not a real answer, else a (missing when E3 is missing); m and e_mech = g * m;
Vdlg gives e' from E2d / E3d. s = the §36b one-shot score (runs/exp36b/qwen3b/V), read as a fixed baseline.

- Claim 1: AUROC(e), SHOWN if the lower bound > 0.5, meaningful if the point >= 0.65; beside: PR-AUC, prevalence,
  AUROC(e_mech), AUROC(g).
- Claim 2 (only if 1 SHOWN): paired AUROC(e) - AUROC(s) on rows with both, SHOWN if the lower bound > 0, meaningful if
  >= +0.05; V-num's AUROC beside.
- Claim 3 (only if 2 SHOWN): at C0's coverage, paired risk(C3e) - risk(C0), SHOWN if the upper bound < 0.
- Claim 4 (descriptive): C5 repair accuracy against C1 on the same rows (needs the judge rows on d; none on the
  laptop), d's corrections and breaks.
- Controls A (g and e intact vs cut, Cut-F pass rate), P (planted thesis on rows judged correct), S (sycophancy,
  blind vs dialogue), G (cited set hits the evidence), 7B (descriptive), costs.
- Compliance (§5.1): E1 parses on >= 0.9 of rows; E3 label mass >= 0.5 on >= 0.9 of rows; the §36 reliability rule.
- §5 gate (development folds 0-1, always printed with its arithmetic): AUROC(e) >= 0.65 and >= AUROC(s) + 0.05.
Intervals: paired bootstrap over questions, 10,000 resamples, seed 0. Folds 2-4 need --confirm.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from memctl.analysis import elenchus as el  # noqa: E402
from memctl.analysis import verification as ver  # noqa: E402
from scripts.exp35_generate import READERS, read_jsonl  # noqa: E402
from scripts.exp35_report import load as load35  # noqa: E402
from scripts.exp36_report import Frame, claim1_verdict, discrimination, matched, policy, risk_difference  # noqa: E402
from scripts.exp37_elenchus import ITEMS, turns_of  # noqa: E402

DEV_FOLDS = {0, 1}
CONFIRM_FOLDS = {2, 3, 4}


def _f(x, digits=3) -> str:
    return "-" if x is None else f"{x:.{digits}f}"


def _ci(stat: dict | None, signed: bool = False) -> str:
    if not stat or stat.get("estimate") is None:
        return "-"
    fmt = "{:+.3f}" if signed else "{:.3f}"
    if stat.get("ci_low") is None:
        return fmt.format(stat["estimate"])
    return f"{fmt.format(stat['estimate'])} ({fmt.format(stat['ci_low'])}, {fmt.format(stat['ci_high'])})"


# --- loading --------------------------------------------------------------------------------------------------

def load_rows(root: Path, folds) -> dict:
    rows: dict = {}
    for reader in READERS:
        for item in ITEMS.get(reader, ()):
            for turn in turns_of(item):
                got = {}
                for f in sorted(folds):
                    for r in read_jsonl(root / reader / item / f"{turn}.fold{f}.jsonl"):
                        got[r["question_id"]] = r
                rows.setdefault(reader, {}).setdefault(item, {})[turn] = got
    return rows


def check_rows(rows: dict, d35) -> None:
    engines = sorted({r["engine"] for rd in rows.values() for it in rd.values() for t in it.values() for r in t.values()})
    if len(engines) > 1:
        raise SystemExit(f"REFUSED: the §37 run directory mixes engines {engines}")
    shas = sorted({r["wording_sha256"] for rd in rows.values() for it in rd.values() for t in it.values() for r in t.values()})
    allowed = {el.wording_sha256(a, True) for a in (False, True)}   # amendment 1: the E3 fallback, normaliser v2
    if len(shas) > 1 or not set(shas) <= allowed:
        raise SystemExit(f"REFUSED: wording sha256 {shas} (one of the committed wordings, one per run)")
    bad = [f"{r['item']}/{q}" for rd in rows.values() for it in rd.values() for t in it.values() for q, r in t.items()
           if q in d35.by_q and r["intact_prompt_sha256"] != ver.sha256(d35.by_q[q].prompt)]
    if bad:
        raise SystemExit(f"REFUSED: {len(bad)} rows do not carry their record's prompt hash, e.g. {bad[:3]}")


def s_scores(folds, root: Path = Path("runs/exp36b")) -> dict[str, float | None]:
    return {r["question_id"]: r.get("s") for f in sorted(folds) for r in read_jsonl(root / "qwen3b" / "V" / f"fold{f}.jsonl")}


def judge_rows(root: Path, folds) -> dict[tuple[str, str], bool]:
    return {(r["item"], r["question_id"]): bool(r["verdict"]) for f in sorted(folds)
            for r in read_jsonl(root / "qwen3b" / "judge" / f"fold{f}.jsonl")}


# --- per-question scores --------------------------------------------------------------------------------------

def scores(item_rows: dict, item: str, e1_rows: dict | None = None) -> dict[str, dict]:
    """{qid: {g, d, a, e, m, e_mech, G, hit, parses, mass}} for one reader and item (Vdlg: E1 from V)."""
    e1 = e1_rows if e1_rows is not None else item_rows.get("E1", {})
    t2, t3 = ("E2d", "E3d") if item == "Vdlg" else ("E2", "E3")
    out = {}
    for q, r1 in e1.items():
        g = r1["g"]
        r2 = item_rows.get(t2, {}).get(q)
        d = r2["d"] if (g and r2) else None
        r3 = item_rows.get(t3, {}).get(q)
        a = r3["a"] if r3 else None
        if g and r2 is None:
            e = None  # E2 missing (cut by a deadline)
            sc = {"g": g, "e": None, "m": 0, "e_mech": None}
        else:
            sc = el.elenchus_scores(g, d, a, r1["candidate"])
            if g and el.real_answer(d) and r3 is None:
                sc["e"] = None
        out[q] = {**sc, "d": d, "a": a, "G": r1["G"], "hit": r1.get("evidence_hit"), "parses": r1["parses"],
                  "mass": None if r3 is None else r3["label_mass"]}
    return out


def paired_auroc_diff(frame: Frame, s1: dict, s2: dict, boot) -> dict:
    """AUROC(s1) - AUROC(s2) on candidate rows with both scores, both AUROCs on the same bootstrap draws."""
    rows = [q for q in frame.rows if frame.has[q] and s1.get(q) is not None and s2.get(q) is not None]
    pos = [q for q in rows if frame.y[q]]
    neg = [q for q in rows if not frame.y[q]]
    if not pos or not neg:
        return {"n": len(rows), "estimate": None, "ci_low": None, "ci_high": None}

    def cmp(sc):
        a, b = np.array([sc[q] for q in pos], float), np.array([sc[q] for q in neg], float)
        return (a[:, None] > b[None, :]) + 0.5 * (a[:, None] == b[None, :])
    c1, c2 = cmp(s1), cmp(s2)
    wp, wn = boot.weights(pos), boot.weights(neg)
    with np.errstate(invalid="ignore", divide="ignore"):
        draws = (np.einsum("bi,ij,bj->b", wp, c1, wn) - np.einsum("bi,ij,bj->b", wp, c2, wn)) / (wp.sum(1) * wn.sum(1))
    draws = draws[np.isfinite(draws)]
    return {"n": len(rows), "n_pos": len(pos), "n_neg": len(neg), "estimate": float(c1.mean() - c2.mean()),
            "ci_low": float(np.percentile(draws, 2.5)) if draws.size else None,
            "ci_high": float(np.percentile(draws, 97.5)) if draws.size else None}


def share(rows: list[tuple[str, float]], boot) -> dict:
    return ver.mean_stat(rows, boot)


def compliance(rows_item: dict) -> dict:
    e1 = list(rows_item.get("E1", {}).values())
    e3 = [r["label_mass"] for t in ("E3", "E3d", "Vnum") for r in rows_item.get(t, {}).values()]
    return {"E1_n": len(e1), "E1_parses": (sum(r["parses"] for r in e1) / len(e1)) if e1 else None,
            "E3_n": len(e3), "E3_mass_ge_0.5": (sum(m >= 0.5 for m in e3) / len(e3)) if e3 else None,
            "reliability": ver.reliability(e3)}


# --- analysis -------------------------------------------------------------------------------------------------

def reader_block(frame: Frame, R: dict, boot, label_reader: str) -> dict:
    """Discrimination and controls for one verifier reader on the 3B's candidates."""
    V = scores(R["V"], "V")
    cut = scores(R.get("CutB", {}), "CutB")
    out = {"n_rows": len(V)}
    eV = {q: V[q]["e"] for q in V}
    out["auroc_e"] = discrimination(frame, eV, boot)
    out["auroc_e_mech"] = discrimination(frame, {q: V[q]["e_mech"] for q in V}, boot)
    out["auroc_g"] = discrimination(frame, {q: float(V[q]["g"]) for q in V}, boot)
    both = [q for q in frame.rows if frame.has[q] and q in V and q in cut]
    out["A_g_drop"] = share([(q, V[q]["g"] - cut[q]["g"]) for q in both], boot)
    out["A_g_intact"] = share([(q, float(V[q]["g"])) for q in both], boot)
    out["A_g_cut"] = share([(q, float(cut[q]["g"])) for q in both], boot)
    out["A_e_drop_correct"] = share([(q, V[q]["e"] - cut[q]["e"]) for q in both if frame.y[q]
                                     and V[q]["e"] is not None and cut[q]["e"] is not None], boot)
    out["G_hit_intact"] = share([(q, float(bool(V[q]["hit"]))) for q in V if q in frame.has and frame.has[q]], boot)
    out["G_hit_cut"] = share([(q, float(bool(cut[q]["hit"]))) for q in cut if q in frame.has and frame.has[q]], boot)
    if "CutF" in R:
        cf = scores(R["CutF"], "CutF")
        out["CutF_pass"] = share([(q, float((cf[q]["e"] or 0) >= 0.5)) for q in cf], boot)
    planted = {}
    for item in ("Pnum", "Pname"):
        if item not in R:
            continue
        P = scores(R[item], item)
        qs = [q for q in P if q in frame.y and frame.y[q] and q in V]
        planted[item] = {"n": len(qs),
                         "reject_planted": share([(q, float((P[q]["e"] or 0) < 0.5)) for q in qs], boot),
                         "accept_true": share([(q, float((V[q]["e"] or 0) >= 0.5)) for q in qs], boot),
                         "follows": share([(q, float((P[q]["e"] or 0) < 0.5 and (V[q]["e"] or 0) >= 0.5)) for q in qs], boot)}
    out["P"] = planted
    out["compliance"] = {item: compliance(R[item]) for item in R}
    return out


def analyse(folds, rows, d35, s, judge, boot) -> dict:
    f3 = Frame(d35)
    R3 = rows["qwen3b"]
    V = scores(R3["V"], "V")
    e = {q: V[q]["e"] for q in V}
    out = {"n_fixed": len(f3.rows), "n_candidate": sum(f3.has.values()),
           "missing_e": sum(1 for q in f3.rows if f3.has[q] and e.get(q) is None)}
    out["qwen3b"] = reader_block(f3, R3, boot, "qwen3b")
    out["qwen7b"] = reader_block(f3, rows["qwen7b"], boot, "qwen7b") if rows.get("qwen7b", {}).get("V", {}).get("E1") else None
    c1 = out["qwen3b"]["auroc_e"]
    c1["verdict"], c1["verdict_text"] = claim1_verdict(c1)
    out["claim1"] = c1
    s_c = {q: s.get(q) for q in f3.rows if f3.has[q]}
    c2 = paired_auroc_diff(f3, e, s_c, boot)
    c2["auroc_s"] = discrimination(f3, s_c, boot)
    vnum = {q: r["a"] for q, r in R3.get("Vnum", {}).get("Vnum", {}).items()}
    c2["auroc_vnum"] = discrimination(f3, vnum, boot) if vnum else None
    c3e = matched(f3, {q: e.get(q) for q in f3.rows if f3.has[q]})
    c3 = risk_difference(f3, c3e, f3.c0, boot)
    shown = lambda st, side: st.get(side) is not None and (st[side] > 0 if side == "ci_low" else st[side] < 0)  # noqa: E731
    if c1["verdict"] != "SHOWN":
        c2["verdict"] = c3["verdict"] = "not tested (Claim 1 not SHOWN)"
    else:
        c2["verdict"] = "SHOWN" if shown(c2, "ci_low") else "NOT SHOWN"
        c3["verdict"] = ("SHOWN" if shown(c3, "ci_high") else "NOT SHOWN") if c2["verdict"] == "SHOWN" else \
            "not tested (Claim 2 not SHOWN)"
    c2["meaningful"] = c2.get("estimate") is not None and c2["estimate"] >= el.GATE_MARGIN
    out["claim2"], out["claim3"] = c2, c3
    c2e = {q for q in f3.rows if f3.has[q] and V.get(q, {}).get("e_mech") == 1}
    c3s = matched(f3, s_c)
    out["policies"] = {"C0": policy(f3, f3.c0, boot), "C1": policy(f3, {q for q in f3.rows if f3.has[q]}, boot),
                       "C2e": policy(f3, c2e, boot), "C3e": policy(f3, c3e, boot), "C3s": policy(f3, c3s, boot)}
    # Claim 4: repair (needs the judge on d)
    repaired = [q for q in f3.rows if f3.has[q] and V.get(q, {}).get("e_mech") == 0 and el.real_answer(V[q]["d"])]
    if judge:
        rows4 = []
        fixes = breaks = 0
        for q in f3.rows:
            c1q = float(f3.has[q] and f3.y[q])
            if q in repaired and ("V/E2", q) in judge:
                c5q = float(judge[("V/E2", q)])
                fixes += int(not f3.y[q] and judge[("V/E2", q)])
                breaks += int(f3.y[q] and not judge[("V/E2", q)])
            else:
                c5q = float(q in c3e and f3.y[q])
            rows4.append((q, c5q - c1q))
        out["claim4"] = {"n_repaired": len(repaired), "C5_minus_C1": share(rows4, boot), "corrects": fixes, "breaks": breaks}
    else:
        out["claim4"] = {"n_repaired": len(repaired), "note": "no judge rows on d (laptop run): not available"}
    # S: sycophancy (blind vs dialogue)
    if R3.get("Vdlg", {}).get("E2d"):
        D = scores(R3["Vdlg"], "Vdlg", R3["V"]["E1"])
        S = {}
        for lab, want in (("judged_wrong", False), ("judged_correct", True)):
            qs = [q for q in f3.rows if f3.has[q] and f3.y[q] is want and q in V and q in D]
            S[lab] = {"n": len(qs), "blind": share([(q, float(V[q]["e_mech"] or 0)) for q in qs], boot),
                      "dialogue": share([(q, float(D[q]["e_mech"] or 0)) for q in qs], boot)}
        out["S"] = S
        out["auroc_e_dialogue"] = discrimination(f3, {q: D[q]["e"] for q in D}, boot)
    # costs
    cost = {}
    for reader, Ritems in rows.items():
        for item, turns in Ritems.items():
            n_tok = sum(r.get("n_input_tokens") or 0 for t in turns.values() for r in t.values())
            secs = sum((r["stats"].get("prefill_s", 0) + r["stats"].get("decode_s", 0) + r["stats"].get("batch_s_per_call", 0))
                       for t in turns.values() for r in t.values())
            n_q = len(turns.get("E1", turns.get("Vnum", turns.get("E2d", {}))))
            calls = sum(len(t) for t in turns.values())
            if n_q:
                cost[f"{reader}/{item}"] = {"questions": n_q, "calls_per_candidate": calls / n_q,
                                            "tokens_per_candidate": n_tok / n_q, "seconds_per_candidate": secs / n_q}
    out["cost"] = cost
    return out


def gate(dev: dict) -> dict:
    """§5's gate from the development folds, with its arithmetic."""
    g = el.dev_gate(dev["claim1"].get("estimate"), dev["claim2"]["auroc_s"].get("estimate"))
    g["text"] = (f"AUROC(e, dev) = {_f(g['auroc_e'])}; AUROC(s, dev) = {_f(g['auroc_s'])}; AUROC(s) + {el.GATE_MARGIN} = "
                 f"{_f(g['bar_vs_s'])}. Bar 1: {_f(g['auroc_e'])} >= {el.GATE_AUROC}: "
                 f"{g['auroc_e'] is not None and g['auroc_e'] >= el.GATE_AUROC}. Bar 2: {_f(g['auroc_e'])} >= "
                 f"{_f(g['bar_vs_s'])}: {g['auroc_e'] is not None and g['bar_vs_s'] is not None and g['auroc_e'] >= g['bar_vs_s']}. "
                 f"Gate: {'PASS (quote a pod after the Advisor review)' if g['pass'] else 'FAIL (§37 closes, development only)'}.")
    return g


def render(folds, n: dict) -> str:
    dev = set(folds) <= DEV_FOLDS
    tag = " (development folds: not a confirmation verdict)" if dev else ""
    t3 = n["qwen3b"]
    L = [f"# §37 report: the elenchus (folds {', '.join(map(str, sorted(folds)))})", "",
         f"Engines {n['engines']}; wording sha256 {n['wording'][:12]}. Fixed-candidate rows {n['n_fixed']}, with a "
         f"candidate {n['n_candidate']}, candidate rows without an e score {n['missing_e']}. 95% intervals: paired "
         f"bootstrap over questions, {n['samples']:,} resamples.", "",
         "## §5 development gate (folds 0-1)", "", n["gate"]["text"], "",
         "## Compliance (§5.1: E1 parses ≥ 0.9; E3 mass ≥ 0.5 on ≥ 0.9 of rows)", "",
         "| reader/item | E1 rows | E1 parses | E3 rows | E3 mass ≥ 0.5 | reliable |", "|---|---|---|---|---|---|"]
    for reader in READERS:
        blk = n.get(reader)
        if not blk:
            continue
        for item, c in blk["compliance"].items():
            L.append(f"| {reader}/{item} | {c['E1_n']} | {_f(c['E1_parses'])} | {c['E3_n']} | {_f(c['E3_mass_ge_0.5'])} | "
                     f"{'yes' if c['reliability']['reliable'] else 'no'} |")
    c1, c2, c3 = n["claim1"], n["claim2"], n["claim3"]
    L += ["", f"## 3B claims (fixed sequence){tag}", "", "| claim | n | estimate (95% CI) | verdict |", "|---|---|---|---|",
          f"| Claim 1: AUROC(e) | {c1['n_pos']} / {c1['n_neg']} | {_ci(c1)} | {c1['verdict_text']} |",
          f"| beside: PR-AUC, prevalence | | {_f(c1.get('pr_auc'))}, {_f(c1.get('prevalence'))} | |",
          f"| beside: AUROC(e_mech) | | {_ci(t3['auroc_e_mech'])} | descriptive |",
          f"| beside: AUROC(g) | | {_ci(t3['auroc_g'])} | descriptive |",
          f"| Claim 2: AUROC(e) − AUROC(s) (paired) | {c2['n']} | {_ci(c2, True)} | {c2['verdict']}"
          f"{' (meaningful ≥ +0.05)' if c2['meaningful'] else ''} |",
          f"| beside: AUROC(s), §36b one-shot | | {_ci(c2['auroc_s'])} | fixed baseline |",
          f"| beside: AUROC(V-num) | | {_ci(c2['auroc_vnum'])} | descriptive |",
          f"| Claim 3: risk(C3e) − risk(C0) at C0's coverage | {c3['n']} | {_ci(c3, True)} | {c3['verdict']} |", "",
          "| policy | emitted | coverage | risk | wrong added vs C0 |", "|---|---|---|---|---|"]
    for k, p in n["policies"].items():
        L.append(f"| {k} | {p['emitted']} | {_ci(p['coverage'])} | {_ci(p['risk'])} | {p['wrong_added_vs_C0']:+d} |")
    c4 = n["claim4"]
    L += ["", "Claim 4 (repair, descriptive): " + (c4["note"] if "note" in c4 else
          f"{c4['n_repaired']} rows repaired; accuracy C5 − C1 {_ci(c4['C5_minus_C1'], True)}; d corrects {c4['corrects']}, "
          f"breaks {c4['breaks']}."), "", "## Controls (descriptive)", "",
          "| control | 3B | 7B on the 3B's candidates |", "|---|---|---|"]
    s7 = n.get("qwen7b") or {}

    def both(fn):
        return f"| {fn[0]} | {fn[1](t3)} | {fn[1](s7) if s7 else '-'} |"
    for row in (("AUROC(e)", lambda b: _ci(b["auroc_e"])),
                ("A: g intact / cut", lambda b: f"{_ci(b['A_g_intact'])} / {_ci(b['A_g_cut'])}"),
                ("A: g drop (paired)", lambda b: _ci(b["A_g_drop"], True)),
                ("A: e(intact) − e(cut), judged correct", lambda b: _ci(b["A_e_drop_correct"], True)),
                ("Cut-F pass rate (e ≥ 0.5)", lambda b: _ci(b.get("CutF_pass"))),
                ("G: cited set hits the evidence, intact / cut", lambda b: f"{_ci(b['G_hit_intact'])} / {_ci(b['G_hit_cut'])}")):
        L.append(both(row))
    for item in ("Pnum", "Pname"):
        for k, lab in (("reject_planted", "rejects planted"), ("accept_true", "accepts true"), ("follows", "follows (both)")):
            L.append(both((f"P {item}: {lab}", lambda b, item=item, k=k: (
                f"{_ci(b['P'][item][k])} (n {b['P'][item]['n']})" if item in b.get("P", {}) else "-"))))
    if "S" in n:
        L += ["", "S (sycophancy): share with d = the candidate (e_mech), blind vs in dialogue:", "",
              "| rows | n | blind | dialogue |", "|---|---|---|---|"]
        for lab, x in n["S"].items():
            L.append(f"| {lab.replace('_', ' ')} | {x['n']} | {_ci(x['blind'])} | {_ci(x['dialogue'])} |")
        L.append(f"\nAUROC(e′) in dialogue (never primary): {_ci(n['auroc_e_dialogue'])}.")
    L += ["", "## Costs per candidate", "", "| reader/item | questions | calls | input tokens | seconds |", "|---|---|---|---|---|"]
    for k, c in n["cost"].items():
        L.append(f"| {k} | {c['questions']} | {c['calls_per_candidate']:.2f} | {c['tokens_per_candidate']:.0f} | "
                 f"{c['seconds_per_candidate']:.2f} |")
    return "\n".join(L) + "\n"


def build(folds, root: Path, exp35: Path, runs: str, samples: int, own_root: Path, exp36b: Path) -> tuple[str, dict]:
    def one(fs):
        data35, qids, _ = load35(set(fs), exp35, runs, own_root)
        rows = load_rows(root, fs)
        check_rows(rows, data35["qwen3b"])
        boot = ver.Bootstrap(qids, samples=samples, seed=0)
        return analyse(fs, rows, data35["qwen3b"], s_scores(fs, exp36b), judge_rows(root, fs), boot), rows
    numbers, rows = one(set(folds))
    dev = numbers if set(folds) == DEV_FOLDS else one(DEV_FOLDS)[0]
    numbers["gate"] = gate(dev)
    numbers["engines"] = sorted({r["engine"] for rd in rows.values() for it in rd.values() for t in it.values() for r in t.values()})
    shas = sorted({r["wording_sha256"] for rd in rows.values() for it in rd.values() for t in it.values() for r in t.values()})
    numbers["wording"] = shas[0] if shas else el.wording_sha256()
    numbers["samples"], numbers["folds"] = samples, sorted(folds)
    return render(folds, numbers), numbers


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", default="0,1")
    parser.add_argument("--root", default="runs/exp37")
    parser.add_argument("--exp35", default="runs/exp35")
    parser.add_argument("--exp36b", default="runs/exp36b")
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--own-root", default="runs/exp34_jlens_own")
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--confirm", action="store_true", help="required for any of folds 2, 3, 4")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    folds = {int(f) for f in args.folds.split(",")}
    if folds & CONFIRM_FOLDS and not args.confirm:
        print(f"folds {sorted(folds & CONFIRM_FOLDS)} are confirmation folds; pass --confirm only after the Advisor's "
              "development review and the user's pod approval")
        return 2
    text, numbers = build(folds, Path(args.root), Path(args.exp35), args.runs, args.samples, Path(args.own_root),
                          Path(args.exp36b))
    label = "dev" if folds <= DEV_FOLDS else "confirm" if folds <= CONFIRM_FOLDS else "allfolds"
    out = Path(args.out or f"runs/exp37_report_{label}.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    out.with_suffix(".json").write_text(json.dumps(numbers, indent=1, default=float))
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
