"""§36 report: can the 3B judge evidence support for its own forced answers? Implements proposal §4.

    .venv/bin/python scripts/exp36_report.py --folds 0,1 [--out runs/exp36_report_dev.md]
    .venv/bin/python scripts/exp36_report.py --folds 2,3,4 --confirm [--out runs/exp36_report_confirm.md]
    .venv/bin/python scripts/exp36_report.py --folds 0,1,2,3,4 --confirm --out runs/exp36_report_allfolds.md

Pre-registration: research/proposals/2026-10-15-exp36-verification.md (frozen at d2a67f2 + a3a3f9c). 3B primary; every
7B row is descriptive (its verdicts share weights with the judge).

Rows. Fixed-candidate rows: marked 3B rows that pass the §35 token check and have an arm-B verdict
(scripts/exp35_report.py ReaderData.frozen). A row whose B output has no usable candidate (a refusal) stays in every
coverage and risk denominator as a forced abstention: every condition abstains on it.

Reading taken for Claim 1 (the proposal is open on it): the AUROC is computed over the candidate rows, where s exists.
Putting the no-candidate rows into the AUROC at the lowest score hands any score free correctly-ranked negatives (all
of them are judged wrong): on the stub run a random s passed Claim 1 that way (0.679, lower bound 0.571; 0.529 on the
candidate rows). That all-rows AUROC is printed beside, labelled, with the natural flag on both row sets (the all-rows
flag AUROC reproduces scripts/exp36_devcheck.py, the proposal's 0.620).
Label = the §35 judge verdict on B. C0 emits iff there is a candidate and the cached §23 answer was not a refusal.

- Claim 1: AUROC of s (V) for judged correct vs wrong over the candidate rows; SHOWN if the lower bound > 0.5;
  practically meaningful if the estimate >= 0.65. PR-AUC and prevalence beside; the natural flag's AUROC beside.
- Claim 2 (tested only if Claim 1 SHOWN): C3 emits as many candidates as C0, ranked by s (ties by question id);
  paired risk(C3) - risk(C0); SHOWN if the upper bound < 0.
- Claim 3 (tested only if Claim 2 SHOWN): the same against C4 (m = candidate - unknown prompt logprob after the arm-A
  prefix) at the same coverage; SHOWN if the upper bound < 0; meaningful if the estimate <= -0.05.
- Claim 4 (descriptive): tau = the lowest threshold with development-fold (0-1) risk <= 0.30, fitted on folds 0-1 and
  applied to the report's folds; coverage, risk, cached refusals recovered correctly, new wrong answers, wrong answers
  added relative to C0. C4 gets the same procedure.
- Controls (both readers as verifiers of the 3B candidates): A, s(intact) - s(cut) for the same candidate on rows
  judged correct, and the 7B-minus-3B difference of that drop; A', AUROC of intact correct-B rows vs Cut-F rows; B, the
  counterfactual: primary row = number-type substitutions whose value has no other occurrence in memory (coordinator
  decision 2026-10-15 after the dev hand check: name substitutions are not internally consistent); name-type rows,
  numbers with other occurrences and all rows are descriptive only (all generated; descriptive below 20 rows); the
  length control; C1, C2, C4'.
- Label-mass rule per verifier and item: if the mass is below 0.5 on more than 0.1 of rows, the readout is "not
  reliable" and only its argmax label is reported.
Intervals: paired bootstrap over questions (every variant of a question in one unit), 10,000 resamples, 95%.

Folds 2, 3 and 4 are confirmation folds: without --confirm the report refuses them. Claim 4's tau always comes from
folds 0-1.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from memctl.analysis import verification as ver  # noqa: E402
from memctl.jacobian_data import load_records  # noqa: E402
from scripts.exp35_generate import READERS, read_jsonl  # noqa: E402
from scripts.exp35_report import load as load35  # noqa: E402

DEV_FOLDS = {0, 1}
CONFIRM_FOLDS = {2, 3, 4}
PRIMARY = "qwen3b"
SHARED_ITEMS = ("V", "Vn", "P2", "P3", "CutB", "CutF", "Abs", "CF", "LC")


def _f(x, digits=3) -> str:
    return "-" if x is None else f"{x:.{digits}f}"


def _ci(stat: dict, signed: bool = False) -> str:
    if not stat or stat.get("estimate") is None:
        return "-"
    fmt = "{:+.3f}" if signed else "{:.3f}"
    if stat.get("ci_low") is None:
        return fmt.format(stat["estimate"]) + " (no interval)"
    return f"{fmt.format(stat['estimate'])} ({fmt.format(stat['ci_low'])}, {fmt.format(stat['ci_high'])})"


# --- loading --------------------------------------------------------------------------------------------------

def load_rows(root: Path, folds: set[int], rescore: bool = False) -> dict[str, dict[str, dict[tuple[str, str], dict]]]:
    """Rows per reader and item; with `rescore`, label fields are recomputed from the stored top-20 (decoded)."""
    rows: dict = {}
    for reader in READERS:
        for item in ver.ITEMS:
            got = {}
            for fold in sorted(folds):
                for row in read_jsonl(root / reader / item / f"fold{fold}.jsonl"):
                    got[ver.row_key(row)] = ver.rescore_row(row) if rescore else row
            rows.setdefault(reader, {})[item] = got
    return rows


def abs_records(folds: set[int], runs: str) -> dict:
    return {x.question_id: x for f in sorted(folds) for x in load_records("qwen3b", f, runs_dir=runs)[0] if not x.answerable}


def check_rows(rows: dict, data35: dict, absr: dict) -> None:
    """Refuse a mixed-engine run, a wording change, a prompt that is not its record's or a candidate that is not the
    current §35 candidate."""
    engines = sorted({r["engine"] for reader in rows.values() for item in reader.values() for r in item.values()})
    if len(engines) > 1:
        raise SystemExit(f"REFUSED: the §36 run directory mixes engines {engines}")
    frames = sorted({str(r.get("frame")) for reader in rows.values() for item, got in reader.items() if item != "C4"
                     for r in got.values()})
    if len(frames) > 1:
        raise SystemExit(f"REFUSED: the §36 run directory mixes verifier wordings {frames}")
    problems = []
    for reader, items in rows.items():
        for item, got in items.items():
            for (q, variant), r in got.items():
                rec = absr.get(q) if item == "Abs" else data35["qwen7b" if item in ("V_own",) or (item == "C4" and reader == "qwen7b") else "qwen3b"].by_q.get(q)
                if rec is None:
                    continue
                expected = (ver.frame_wording_sha256(r["frame"], r.get("labels", "frame")) if r.get("frame") and item != "C4"
                            else ver.wording_sha256())
                if r.get("wording_sha256") != expected or r["intact_prompt_sha256"] != ver.sha256(rec.prompt):
                    problems.append(f"{reader}/{item}/{q}")
                    continue
                if item in ("V", "Vn", "P2", "P3", "P2p", "P3p", "CutB", "LC") or (item == "C4" and reader == "qwen3b"):
                    cand = ver.candidate_of_row(data35["qwen3b"].rows["B"].get(q))
                elif item == "CutF":
                    cand = ver.candidate_of_row(data35["qwen3b"].rows["Fcut"].get(q))
                elif item == "V_own" or item == "C4":
                    cand = ver.candidate_of_row(data35["qwen7b"].rows["B"].get(q))
                else:
                    continue
                if cand != r["candidate"]:
                    problems.append(f"{reader}/{item}/{q} (candidate)")
    if problems:
        raise SystemExit(f"REFUSED: {len(problems)} rows do not match their record, wording or §35 candidate, e.g. {problems[:3]}")


# --- per-reader frames ----------------------------------------------------------------------------------------

class Frame:
    """The fixed-candidate rows of one candidate source (the 3B's, or the 7B's own) with labels and C0."""

    def __init__(self, d35) -> None:
        self.d = d35
        self.rows = [q for q in d35.frozen if d35.correct("B", q) is not None]
        self.cand = {q: ver.candidate_of_row(d35.rows["B"].get(q)) for q in self.rows}
        self.has = {q: self.cand[q] is not None for q in self.rows}
        self.y = {q: bool(d35.correct("B", q)) for q in self.rows}
        self.wrong = {q: not self.y[q] for q in self.rows}
        self.refusal = {q: d35.cached_refusal(q) for q in self.rows}
        self.c0 = {q for q in self.rows if self.has[q] and not self.refusal[q]}

    def scores(self, got: dict, field: str = "s") -> dict[str, float | None]:
        """Score per candidate row (None when the row is missing or has no score); rows without a candidate omitted."""
        out = {}
        for q in self.rows:
            if not self.has[q]:
                continue
            r = got.get((q, ""))
            out[q] = None if r is None else r.get(field)
        return out


def _auroc_rows(frame: Frame, scores: dict) -> tuple[list, list]:
    full = {q: (scores.get(q) if frame.has[q] and scores.get(q) is not None else -1.0) for q in frame.rows}
    pos = [(q, full[q]) for q in frame.rows if frame.y[q]]
    neg = [(q, full[q]) for q in frame.rows if not frame.y[q]]
    return pos, neg


def discrimination(frame: Frame, scores: dict, boot) -> dict:
    """AUROC over the candidate rows (where the score exists; primary), PR-AUC and prevalence on the same rows, and,
    beside it, the AUROC over all fixed rows with the no-candidate rows at the lowest score (-1)."""
    cand = [q for q in frame.rows if frame.has[q] and scores.get(q) is not None]
    pos = [(q, scores[q]) for q in cand if frame.y[q]]
    neg = [(q, scores[q]) for q in cand if not frame.y[q]]
    out = ver.auroc_stat(pos, neg, boot)
    out["pr_auc"] = ver.average_precision([s for _, s in pos], [s for _, s in neg])
    out["prevalence"] = len(pos) / len(cand) if cand else None
    out["all_fixed_rows"] = ver.auroc_stat(*_auroc_rows(frame, scores), boot)
    out["missing_scores"] = sum(1 for q in frame.rows if frame.has[q] and scores.get(q) is None)
    return out


def policy(frame: Frame, emitted: set[str], boot) -> dict:
    rows = frame.rows
    cov = ver.ratio_stat([(q, float(q in emitted), 1.0) for q in rows], boot)
    risk = ver.ratio_stat([(q, float(q in emitted and frame.wrong[q]), float(q in emitted)) for q in rows], boot)
    c0_wrong = sum(frame.wrong[q] for q in frame.c0)
    return {"emitted": len(emitted), "coverage": cov, "risk": risk,
            "wrong": sum(frame.wrong[q] for q in emitted), "wrong_added_vs_C0": sum(frame.wrong[q] for q in emitted) - c0_wrong,
            "refusals_recovered_correct": sum(1 for q in emitted if frame.refusal[q] and frame.y[q]),
            "refusals_new_wrong": sum(1 for q in emitted if frame.refusal[q] and frame.wrong[q])}


def risk_difference(frame: Frame, a: set[str], b: set[str], boot) -> dict:
    return ver.ratio_difference([(q, float(q in a and frame.wrong[q]), float(q in a), float(q in b and frame.wrong[q]),
                                  float(q in b)) for q in frame.rows], boot)


def matched(frame: Frame, scores: dict) -> set[str]:
    return ver.top_k({q: s for q, s in scores.items()}, len(frame.c0))


def unknown_ranks(reader: str, qids: list[str], own_root: Path) -> dict[str, float | None]:
    """C4': the §34b final-layer rank of ' unknown'/'unknown' at the reader's own answer position (as
    scripts/exp36_devcheck.py); a higher rank = 'unknown' less likely = answer. None when there is no read."""
    out = {}
    for q in qids:
        path = own_root / reader / f"{q}.json"
        if not path.exists():
            out[q] = None
            continue
        m = json.loads(path.read_text())["measures"]
        m = json.loads(m) if isinstance(m, str) else m
        x = m["logit"]["P1"]
        x = json.loads(x) if isinstance(x, str) else x
        out[q] = float(min(x["unknown_sp"]["final_rank"], x["unknown"]["final_rank"]))
    return out


def reliability_table(rows: dict) -> dict:
    out = {}
    for reader, items in rows.items():
        for item, got in items.items():
            masses = [r["label_mass"] for r in got.values() if "label_mass" in r]
            if masses:
                out[f"{reader}/{item}"] = ver.reliability(masses)
    return out


def argmax_policy(frame: Frame, got: dict) -> set[str]:
    return {q for q in frame.rows if frame.has[q] and (got.get((q, "")) or {}).get("argmax") == "A"}


# --- analysis -------------------------------------------------------------------------------------------------

def controls(verifier: str, rows: dict, frame3: Frame, fcut: dict, boot, tau: float | None) -> dict:
    """Controls A, A', B (counterfactual), the length control, Abs and Cut-F emission, for one verifier reader."""
    got = rows[verifier]
    sV, sCut, sLC = (frame3.scores(got[i]) for i in ("V", "CutB", "LC"))
    correct = [q for q in frame3.rows if frame3.has[q] and frame3.y[q]]
    out = {}
    out["A_drop"] = ver.mean_stat([(q, sV[q] - sCut[q]) for q in correct if sV.get(q) is not None and sCut.get(q) is not None], boot)
    cutf = {k[0]: r for k, r in got["CutF"].items()}
    pos = [(q, sV[q]) for q in correct if sV.get(q) is not None]
    neg = [(q, r["s"]) for q, r in cutf.items() if r.get("s") is not None]
    out["A_prime_auroc"] = ver.auroc_stat(pos, neg, boot)
    split = {}
    for label, want in (("fcut_judged_correct", True), ("fcut_judged_wrong", False)):
        xs = [r["s"] for q, r in cutf.items() if r.get("s") is not None and fcut.get(q) is want]
        split[label] = {"n": len(xs), "mean_s": sum(xs) / len(xs) if xs else None,
                        "emitted_at_tau": sum(x >= tau for x in xs) if tau is not None else None,
                        "argmax_A": sum(1 for q, r in cutf.items() if fcut.get(q) is want and r.get("argmax") == "A")}
    out["cutf_split"] = split
    out["cutf_unsupported_rate"] = {
        "n": len(neg), "at_tau": (sum(s >= tau for _, s in neg) / len(neg)) if neg and tau is not None else None,
        "argmax_A": (sum(r.get("argmax") == "A" for r in cutf.values()) / len(cutf)) if cutf else None}
    absr = list(got["Abs"].values())
    out["abs"] = {"n": len(absr), "mean_s": (sum(r["s"] for r in absr if r.get("s") is not None) / len(absr)) if absr else None,
                  "at_tau": (sum((r.get("s") or -1) >= tau for r in absr) / len(absr)) if absr and tau is not None else None,
                  "argmax_A": (sum(r.get("argmax") == "A" for r in absr) / len(absr)) if absr else None}
    out["LC"] = {"intact_minus_lc": ver.mean_stat([(q, sV[q] - sLC[q]) for q in correct
                                                    if sV.get(q) is not None and sLC.get(q) is not None], boot),
                 "lc_minus_cut": ver.mean_stat([(q, sLC[q] - sCut[q]) for q in correct
                                                if sLC.get(q) is not None and sCut.get(q) is not None], boot)}
    cf: dict[str, dict] = {}
    for (q, variant), r in got["CF"].items():
        cf.setdefault(q, {})[variant] = r
    full = {q: v for q, v in cf.items() if all(k in v and v[k].get("s") is not None for k in ver.CF_VARIANTS)}

    def cf_stats(qs):
        flags_cf = [(q, float(full[q]["sub@cf"]["s"] > full[q]["orig@cf"]["s"])) for q in qs]
        flags_in = [(q, float(full[q]["orig@intact"]["s"] > full[q]["sub@intact"]["s"])) for q in qs]
        both = [(q, a * b) for (q, a), (_, b) in zip(flags_cf, flags_in)]
        return {"n": len(qs), "follows": ver.mean_stat(both, boot), "follows_cf_context": ver.mean_stat(flags_cf, boot),
                "follows_intact_context": ver.mean_stat(flags_in, boot),
                "verdict": "descriptive" + (f" (fewer than {ver.CF_DESCRIPTIVE_BELOW} rows)" if len(qs) < ver.CF_DESCRIPTIVE_BELOW else "")}
    subsets = cf_subsets({q: full[q]["orig@cf"]["extra"] for q in full})
    out["CF"] = cf_stats(subsets["primary"])
    out["CF"]["verdict"] = "primary row" + ("; descriptive (fewer than 20 rows)" if out["CF"]["n"] < ver.CF_DESCRIPTIVE_BELOW else "")
    for key in ("number_with_other_copies", "name", "all_rows"):
        out[f"CF_{key}"] = cf_stats(subsets[key])
        out[f"CF_{key}"]["verdict"] = "descriptive only"
    out["CF_incomplete"] = len(cf) - len(full)
    return out


def claim1_verdict(stat: dict) -> tuple[str, str]:
    """(decision for the fixed sequence, text naming the rule that decided)."""
    if stat.get("ci_low") is None or stat["ci_low"] <= 0.5:
        return "NOT SHOWN", "NOT SHOWN (lower bound ≤ 0.5)"
    if stat["estimate"] < ver.CLAIM1_MEANINGFUL:
        return "SHOWN", f"SHOWN, not meaningful (point < {ver.CLAIM1_MEANINGFUL})"
    return "SHOWN", "SHOWN, meaningful"


def subframe(frame: "Frame", keep: set[str]) -> "Frame":
    """A copy of `frame` restricted to the fixed rows in `keep` (C0 restricted with it)."""
    sub = object.__new__(Frame)
    sub.d = frame.d
    sub.rows = [q for q in frame.rows if q in keep]
    for name in ("cand", "has", "y", "wrong", "refusal"):
        setattr(sub, name, {q: getattr(frame, name)[q] for q in sub.rows})
    sub.c0 = {q for q in frame.c0 if q in keep}
    return sub


def _norm_token(token) -> str:
    return "" if token is None else token.strip().lower()


def mass_diagnostics(got: dict) -> dict:
    """For one reader's V rows: share with label mass < 0.5, share whose top-1 token is 'unknown' (case- and
    space-insensitive), and the top-1 tokens of the low-mass rows (10 most common), read from the stored top-20."""
    rows = list(got.values())
    def top1(r):
        if r.get("top1") is not None:
            return r["top1"]
        top = r.get("top_logprobs") or []
        return top[0][0] if top else None
    low = [r for r in rows if r.get("label_mass", 1.0) < ver.MASS_FLOOR]
    counts: dict[str, int] = {}
    for r in low:
        t = repr(top1(r))
        counts[t] = counts.get(t, 0) + 1
    n = len(rows)
    return {"n": n, "low_mass_share": len(low) / n if n else None,
            "unknown_top1_share": sum(_norm_token(top1(r)) == "unknown" for r in rows) / n if n else None,
            "low_mass_top1": sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:10]}


def cf_subsets(extras: dict[str, dict]) -> dict[str, list[str]]:
    """Control B row sets (coordinator decision 2026-10-15, from the dev hand check: name substitutions are not
    internally consistent, e.g. "a guy named Memrise"). Primary: number-type substitutions whose value has no other
    occurrence in memory. Descriptive only: numbers with other occurrences, all name-type rows, and all rows."""
    qs = sorted(extras)
    return {"primary": [q for q in qs if extras[q]["kind"] == "number" and extras[q].get("outside") == 0],
            "number_with_other_copies": [q for q in qs if extras[q]["kind"] == "number" and extras[q].get("outside") != 0],
            "name": [q for q in qs if extras[q]["kind"] == "name"],
            "all_rows": qs}


def analyse(folds: set[int], rows: dict, data35: dict, dev: dict, boot, own_root: Path, samples: int) -> dict:
    f3, f7 = Frame(data35["qwen3b"]), Frame(data35["qwen7b"])
    rel = reliability_table(rows)
    has_s = any(r.get("s") is not None for r in rows["qwen3b"]["V"].values())
    reliable = rel.get("qwen3b/V", {"reliable": False})["reliable"] and has_s
    g3 = rows["qwen3b"]
    s3 = f3.scores(g3["V"])
    m3 = f3.scores(g3["C4"], "m")
    out: dict = {"n_fixed": len(f3.rows), "n_no_candidate": sum(not v for v in f3.has.values()),
                 "n_candidate": sum(f3.has.values()), "missing_V": sum(1 for v in s3.values() if v is None),
                 "reliability": rel, "V_reliable": reliable}
    # dev-fold tau (Claim 4), always from folds 0-1
    fd = Frame(dev["data35"]["qwen3b"])
    tau = ver.fit_tau(fd.scores(dev["rows"]["qwen3b"]["V"]), fd.wrong, ver.R_STAR)
    tau_c4 = ver.fit_tau(fd.scores(dev["rows"]["qwen3b"]["C4"], "m"), fd.wrong, ver.R_STAR)
    out["tau"], out["tau_c4"] = tau, tau_c4
    out["C0"] = policy(f3, f3.c0, boot)
    out["C1"] = policy(f3, {q for q in f3.rows if f3.has[q]}, boot)
    c2 = argmax_policy(f3, g3["V"])
    out["C2"] = policy(f3, c2, boot) | {"minus_C0": risk_difference(f3, c2, f3.c0, boot)}
    flag = {q: (1.0 if q in f3.c0 else 0.0) for q in f3.rows}
    out["natural_flag_auroc"] = discrimination(f3, flag, boot)
    # as scripts/exp36_devcheck.py (the proposal's dev 0.620): the cached flag alone, over all fixed rows
    out["natural_flag_auroc_devcheck"] = ver.auroc_stat(
        [(q, 0.0 if f3.refusal[q] else 1.0) for q in f3.rows if f3.y[q]],
        [(q, 0.0 if f3.refusal[q] else 1.0) for q in f3.rows if not f3.y[q]], boot)
    ranks = unknown_ranks("qwen3b", f3.rows, own_root)
    r_scores = {q: ranks[q] for q in f3.rows if f3.has[q]}
    c4p = matched(f3, r_scores)
    out["C4prime"] = {"auroc": discrimination(f3, r_scores, boot), "matched": policy(f3, c4p, boot),
                      "missing": sum(1 for v in r_scores.values() if v is None)}
    if reliable:
        c1 = discrimination(f3, s3, boot)
        c1["verdict"], c1["verdict_text"] = claim1_verdict(c1)
        c1["meaningful"] = c1["estimate"] is not None and c1["estimate"] >= ver.CLAIM1_MEANINGFUL
        c3set = matched(f3, s3)
        c4set = matched(f3, m3)
        c2s = risk_difference(f3, c3set, f3.c0, boot)
        c3s = risk_difference(f3, c3set, c4set, boot)
        c2s["verdict"], c3s["verdict"] = ver.fixed_sequence(c1["verdict"], c2s, c3s)
        c3s["meaningful"] = c3s["estimate"] is not None and c3s["estimate"] <= -ver.CLAIM3_MEANINGFUL
        c2s["C3"] = policy(f3, c3set, boot)
        c3s["C4"] = policy(f3, c4set, boot)
        c3s["C4_auroc"] = discrimination(f3, m3, boot)
        c3s["missing_C4"] = sum(1 for v in m3.values() if v is None)
        if c3s["missing_C4"]:
            keep = {q for q in f3.rows if not f3.has[q] or (s3.get(q) is not None and m3.get(q) is not None)}
            fb = subframe(f3, keep)
            both = risk_difference(fb, matched(fb, {q: s3[q] for q in fb.rows if fb.has[q]}),
                                   matched(fb, {q: m3[q] for q in fb.rows if fb.has[q]}), boot)
            both["n_candidate"] = sum(fb.has.values())
            c3s["both_scores"] = both
        out["claim1"], out["claim2"], out["claim3"] = c1, c2s, c3s
        out["claim4"] = {"C3": policy(f3, ver.emit_at(s3, tau), boot), "C4": policy(f3, ver.emit_at(m3, tau_c4), boot)}
    else:
        msg = ("NOT RELIABLE (label mass rule): argmax only, see C2" if has_s else
               "argmax only (wording e: generated letter, s undefined): Claims 1, 3, 4 not testable; see C2")
        out["claim1"] = out["claim2"] = out["claim3"] = out["claim4"] = {"verdict": msg}
    out["mass_diagnostics"] = {reader: mass_diagnostics(rows[reader]["V"]) for reader in READERS}
    out["robustness"] = {}
    for item in [i for i in ("Vn", "P2", "P3", "P2p", "P3p") if g3.get(i)]:
        sc = f3.scores(g3[item])
        ok = rel.get(f"qwen3b/{item}", {"reliable": False})["reliable"]
        out["robustness"][item] = (discrimination(f3, sc, boot) if ok else {"verdict": "not reliable"}) | {
            "C2": policy(f3, argmax_policy(f3, g3[item]), boot)}
    fcut = {q: data35["qwen3b"].correct("Fcut", q) for q in data35["qwen3b"].qids}
    out["controls"] = {v: controls(v, rows, f3, fcut, boot, tau) for v in READERS}
    v3, cut3 = f3.scores(g3["V"]), f3.scores(g3["CutB"])
    s7, c7 = f3.scores(rows["qwen7b"]["V"]), f3.scores(rows["qwen7b"]["CutB"])
    correct = [q for q in f3.rows if f3.has[q] and f3.y[q]]
    out["controls"]["A_drop_7B_minus_3B"] = ver.mean_stat(
        [(q, (s7[q] - c7[q]) - (v3[q] - cut3[q])) for q in correct
         if None not in (s7.get(q), c7.get(q), v3.get(q), cut3.get(q))], boot)
    # 7B verifying the 3B candidates (descriptive)
    seven = {"auroc": discrimination(f3, s7, boot), "C2": policy(f3, argmax_policy(f3, rows["qwen7b"]["V"]), boot)}
    m7 = matched(f3, s7)
    seven["matched"] = policy(f3, m7, boot) | {"minus_C0": risk_difference(f3, m7, f3.c0, boot)}
    seven["tau_3B_dev"] = policy(f3, ver.emit_at(s7, tau), boot)
    out["seven_on_3B"] = seven
    # 7B on its own candidates (descriptive)
    so, mo = f7.scores(rows["qwen7b"]["V_own"]), f7.scores(rows["qwen7b"]["C4"], "m")
    own = {"n_fixed": len(f7.rows), "n_no_candidate": sum(not v for v in f7.has.values()), "C0": policy(f7, f7.c0, boot),
           "auroc": discrimination(f7, so, boot), "natural_flag_auroc": discrimination(f7, {q: float(q in f7.c0) for q in f7.rows}, boot)}
    mo_set, so_set = matched(f7, mo), matched(f7, so)
    own["matched_V_own"] = policy(f7, so_set, boot) | {"minus_C0": risk_difference(f7, so_set, f7.c0, boot)}
    own["matched_C4"] = policy(f7, mo_set, boot) | {"V_own_minus_C4": risk_difference(f7, so_set, mo_set, boot)}
    out["seven_own"] = own
    out["cost"] = {}
    for reader in READERS:
        for item, got in rows[reader].items():
            if got:
                n = [r["n_input_tokens"] for r in got.values() if r.get("n_input_tokens") is not None]
                t = [r["seconds"] for r in got.values() if r.get("seconds") is not None]
                out["cost"][f"{reader}/{item}"] = {"rows": len(got), "tokens_per_row": sum(n) / len(n) if n else None,
                                                    "seconds_per_row": sum(t) / len(t) if t else None}
    return out


def render(folds: set[int], numbers: dict) -> str:
    fl = ", ".join(map(str, sorted(folds)))
    dev = folds <= DEV_FOLDS
    tag = " (development folds: not a confirmation verdict)" if dev else ""
    n = numbers
    lines = [f"# §36 report: can the 3B judge support for its own forced answers? (folds {fl})", "",
             ("Label fields RESCORED from the stored top-20 logprobs (byte-level tokens decoded). " if n.get("rescored") else "")
             + f"Verifier wording: {', '.join('run-1 frozen' if f == 'None' else '§36b frame ' + f for f in n.get('frames', []))}. "
             + f"Engines: {n['engines']}. Wording sha256 {n['wording_sha256'][:12]}. Labels: the §35 judge (Qwen2.5-7B) on "
             "arm B. 95% intervals: paired bootstrap over questions (all variants of a question in one unit), "
             f"{n['samples']:,} resamples.", "",
             "## Rows", "",
             f"Fixed-candidate rows (3B): {n['n_fixed']}; with a candidate {n['n_candidate']}; no candidate (B still says "
             f"unknown; forced abstentions, kept in every denominator) {n['n_no_candidate']}; candidate rows missing a V "
             f"score {n['missing_V']}.", "",
             "Label-mass rule (mass < 0.5 on more than 0.1 of rows = not reliable, argmax only):", "",
             "| verifier/item | rows | low-mass rows | share | reliable |", "|---|---|---|---|---|"]
    for key, r in n["reliability"].items():
        lines.append(f"| {key} | {r['n']} | {r['low_mass']} | {_f(r['low_share'])} | {'yes' if r['reliable'] else '**no**'} |")

    lines += ["", "V readout diagnostics (top-1 token from the stored top-20 logprobs):", "",
              "| verifier | V rows | share label mass < 0.5 | share top-1 'unknown' | top-1 tokens on low-mass rows (count) |",
              "|---|---|---|---|---|"]
    for reader, m in n["mass_diagnostics"].items():
        toks = ", ".join(f"{t} ({c})" for t, c in m["low_mass_top1"]) or "-"
        lines.append(f"| {reader} | {m['n']} | {_f(m['low_mass_share'])} | {_f(m['unknown_top1_share'])} | {toks} |")

    def pol(name, p):
        return (f"| {name} | {p['emitted']} | {_ci(p['coverage'])} | {_ci(p['risk'])} | {p['wrong_added_vs_C0']:+d} | "
                f"{p['refusals_recovered_correct']} | {p['refusals_new_wrong']} |")
    head = ["| policy | emitted | coverage (95% CI) | risk (95% CI) | wrong added vs C0 | cached refusals recovered "
            "correct | cached refusals turned wrong |", "|---|---|---|---|---|---|---|"]
    lines += ["", f"## 3B claims (primary; fixed sequence){tag}", ""]
    c1, c2, c3, c4 = n["claim1"], n["claim2"], n["claim3"], n["claim4"]
    if "estimate" in c1:
        lines += ["| claim | n | estimate (95% CI) | verdict |", "|---|---|---|---|",
                  f"| Claim 1: AUROC of s (V) for B judged correct, candidate rows | {c1['n_pos']} pos / {c1['n_neg']} neg | {_ci(c1)} | "
                  f"{c1['verdict_text']} |",
                  f"| beside: PR-AUC, prevalence (candidate rows) | | {_f(c1['pr_auc'])}, {_f(c1['prevalence'])} | |",
                  f"| beside: AUROC over all fixed rows, no-candidate rows lowest | {c1['all_fixed_rows']['n_pos']} / "
                  f"{c1['all_fixed_rows']['n_neg']} | {_ci(c1['all_fixed_rows'])} | descriptive (inflated by the forced abstentions) |",
                  f"| beside: natural flag (C0) AUROC, candidate rows | | {_ci(n['natural_flag_auroc'])} | descriptive |",
                  f"| beside: natural flag AUROC, all fixed rows (as exp36_devcheck.py) | | {_ci(n['natural_flag_auroc_devcheck'])} | descriptive |",
                  f"| Claim 2: risk(C3) − risk(C0) at C0's coverage | {c2['n']} | {_ci(c2, True)} | {c2['verdict']} |",
                  f"| Claim 3: risk(C3) − risk(C4) at C0's coverage | {c3['n']} | {_ci(c3, True)} | {c3['verdict']}"
                  f"{' (meaningful: ≤ −0.05)' if c3['meaningful'] else ''} |",
                  f"| beside: candidate rows with no C4 score | {c3['missing_C4']} | | |",
                  *([f"| beside: Claim 3 on rows with both a V and a C4 score ({c3['both_scores']['n_candidate']} candidate rows) | "
                     f"{c3['both_scores']['n']} | {_ci(c3['both_scores'], True)} | DESCRIPTIVE |"] if c3.get("both_scores") else []),
                  f"| beside: C4 AUROC (m) | | {_ci(c3['C4_auroc'])} | descriptive |", "",
                  f"Claim 4 (descriptive): τ fitted on folds 0-1 at r* = {ver.R_STAR}: s ≥ {_f(n['tau'])}; C4: m ≥ {_f(n['tau_c4'])}"
                  + (" (in-sample on these folds)" if dev else "") + ".", "", *head,
                  pol("C0 natural", n["C0"]), pol("C1 forced (all candidates)", n["C1"]),
                  pol("C2 self-check (argmax A)", n["C2"]), pol("C3 at C0's coverage", c2["C3"]),
                  pol("C4 at C0's coverage", c3["C4"]), pol("C4′ (§34b unknown rank) at C0's coverage", n["C4prime"]["matched"]),
                  pol("Claim 4: C3 at τ", c4["C3"]), pol("Claim 4: C4 at τ", c4["C4"]), "",
                  f"C2 − C0 risk: {_ci(n['C2']['minus_C0'], True)}. C4′ AUROC {_ci(n['C4prime']['auroc'])} "
                  f"({n['C4prime']['missing']} rows without a §34b read rank lowest)."]
    else:
        lines += [f"Claims 1-4: **{c1['verdict']}**.", "", *head, pol("C0 natural", n["C0"]),
                  pol("C1 forced (all candidates)", n["C1"]), pol("C2 self-check (argmax A)", n["C2"])]
    lines += ["", "Robustness (never selected; V stays primary):", "",
              "| wording | AUROC (95% CI) | C2 coverage | C2 risk |", "|---|---|---|---|"]
    for item, r in n["robustness"].items():
        lines.append(f"| {item} | {_ci(r) if 'estimate' in r else r.get('verdict')} | {_ci(r['C2']['coverage'])} | {_ci(r['C2']['risk'])} |")
    lines += ["", "## Evidence dependence (controls; both readers verify the same 3B candidates)", "",
              "| control | 3B verifier | 7B verifier (descriptive) |", "|---|---|---|"]
    a, b = n["controls"]["qwen3b"], n["controls"]["qwen7b"]
    rowsC = [("A: s(intact) − s(cut), B judged correct", lambda c: f"{_ci(c['A_drop'], True)} (n {c['A_drop']['n']})"),
             ("A′: AUROC intact correct-B vs Cut-F", lambda c: f"{_ci(c['A_prime_auroc'])} ({c['A_prime_auroc']['n_pos']} / {c['A_prime_auroc']['n_neg']})"),
             ("Length control: s(intact) − s(LC)", lambda c: f"{_ci(c['LC']['intact_minus_lc'], True)} (n {c['LC']['intact_minus_lc']['n']})"),
             ("Length control: s(LC) − s(cut)", lambda c: f"{_ci(c['LC']['lc_minus_cut'], True)}"),
             ("B (primary): number substitutions, no other copy in memory; follows in both contexts",
              lambda c: f"{_ci(c['CF']['follows'])} (n {c['CF']['n']}; {c['CF']['verdict']})"),
             ("B (primary): follows in the counterfactual context", lambda c: _ci(c["CF"]["follows_cf_context"])),
             ("B (primary): follows in the intact context", lambda c: _ci(c["CF"]["follows_intact_context"])),
             ("B, DESCRIPTIVE ONLY: numbers with other copies in memory, follows in both",
              lambda c: f"{_ci(c['CF_number_with_other_copies']['follows'])} (n {c['CF_number_with_other_copies']['n']})"),
             ("B, DESCRIPTIVE ONLY: name substitutions (not internally consistent), follows in both",
              lambda c: f"{_ci(c['CF_name']['follows'])} (n {c['CF_name']['n']})"),
             ("B, DESCRIPTIVE ONLY: all counterfactual rows, follows in both",
              lambda c: f"{_ci(c['CF_all_rows']['follows'])} (n {c['CF_all_rows']['n']})"),
             ("Cut-F emitted at τ (unsupported-answer rate)", lambda c: _f(c["cutf_unsupported_rate"]["at_tau"])),
             ("Cut-F argmax A", lambda c: _f(c["cutf_unsupported_rate"]["argmax_A"])),
             ("Abs: n, emitted at τ, argmax A", lambda c: f"{c['abs']['n']}, {_f(c['abs']['at_tau'])}, {_f(c['abs']['argmax_A'])}")]
    for label, fn in rowsC:
        lines.append(f"| {label} | {fn(a)} | {fn(b)} |")
    lines += ["", f"7B − 3B difference of the Control A drop (paired): {_ci(n['controls']['A_drop_7B_minus_3B'], True)} "
              f"(n {n['controls']['A_drop_7B_minus_3B']['n']}).", "",
              "Cut-F split by the §35 judge verdict on the Fcut answer (judged-correct answers are 'unsupported' by construction):", "",
              "| verifier | Fcut judged | n | mean s | emitted at τ | argmax A |", "|---|---|---|---|---|---|"]
    for v in READERS:
        for k, x in n["controls"][v]["cutf_split"].items():
            lines.append(f"| {v} | {k.replace('fcut_judged_', '')} | {x['n']} | {_f(x['mean_s'])} | {x['emitted_at_tau']} | {x['argmax_A']} |")
    s7, own = n["seven_on_3B"], n["seven_own"]
    lines += ["", "## 7B (descriptive; the judge shares the 7B's weights)", "",
              "The 7B verifying the 3B's candidates (labels: the 3B's B verdicts):", "",
              f"AUROC {_ci(s7['auroc'])}; at C0's coverage risk {_ci(s7['matched']['risk'])}, minus C0 "
              f"{_ci(s7['matched']['minus_C0'], True)}; argmax-A coverage {_ci(s7['C2']['coverage'])}, risk {_ci(s7['C2']['risk'])}; "
              f"at the 3B's dev τ coverage {_ci(s7['tau_3B_dev']['coverage'])}, risk {_ci(s7['tau_3B_dev']['risk'])}.", "",
              f"The 7B on its own B candidates (V_own; labels: the 7B's B verdicts): fixed rows {own['n_fixed']}, no candidate "
              f"{own['n_no_candidate']}; AUROC {_ci(own['auroc'])}; natural flag AUROC {_ci(own['natural_flag_auroc'])}; "
              f"C0 coverage {_ci(own['C0']['coverage'])}, risk {_ci(own['C0']['risk'])}; V_own at C0's coverage risk "
              f"{_ci(own['matched_V_own']['risk'])} (minus C0 {_ci(own['matched_V_own']['minus_C0'], True)}); C4 at C0's coverage "
              f"risk {_ci(own['matched_C4']['risk'])} (V_own − C4 {_ci(own['matched_C4']['V_own_minus_C4'], True)}). "
              "Never merged with the comparison above into one size conclusion.", "",
              "## Cost per verification", "", "| verifier/item | rows | tokens per row | seconds per row |", "|---|---|---|---|"]
    for key, c in n["cost"].items():
        lines.append(f"| {key} | {c['rows']} | {_f(c['tokens_per_row'], 0)} | {_f(c['seconds_per_row'], 4)} |")
    lines += ["", "Verdict rules (pre-registered, 3B only): Claim 1 SHOWN if the AUROC lower bound > 0.5 (meaningful ≥ 0.65); "
              "Claim 2 tested only if Claim 1 SHOWN, SHOWN if the upper bound of risk(C3) − risk(C0) < 0; Claim 3 tested only "
              "if Claim 2 SHOWN, SHOWN if the upper bound of risk(C3) − risk(C4) < 0 (meaningful ≤ −0.05); Claim 4 and every "
              "control descriptive. C4 for the 7B is on its own candidates and own note."]
    return "\n".join(lines) + "\n"


def build(folds: set[int], root: Path, exp35: Path, runs: str = "runs", samples: int = 10000,
          own_root: Path = Path("runs/exp34_jlens_own"), rescore: bool = False) -> tuple[str, dict]:
    data35, qids, _ = load35(folds, exp35, runs, own_root)
    rows = load_rows(root, folds, rescore)
    absr = abs_records(folds, runs)
    check_rows(rows, data35, absr)
    if folds == DEV_FOLDS:
        dev = {"data35": data35, "rows": rows}
    else:
        d35, _, _ = load35(DEV_FOLDS, exp35, runs, own_root)
        drows = load_rows(root, DEV_FOLDS, rescore)
        check_rows(drows, d35, abs_records(DEV_FOLDS, runs))
        dev = {"data35": d35, "rows": drows}
    boot = ver.Bootstrap(list(qids) + list(absr), samples=samples, seed=0)
    numbers = analyse(folds, rows, data35, dev, boot, own_root, samples)
    numbers.update({"frames": sorted({str(r.get("frame")) for reader in rows.values() for i, got in reader.items()
                                      if i != "C4" for r in got.values()}), "rescored": rescore, "folds": sorted(folds), "samples": samples,
                    "engines": sorted({r["engine"] for reader in rows.values() for item in reader.values() for r in item.values()})})
    frames = [f for f in numbers["frames"] if f != "None"]
    numbers["wording_sha256"] = ver.frame_wording_sha256(frames[0]) if frames else ver.wording_sha256()
    return render(folds, numbers), numbers


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", default="0,1")
    parser.add_argument("--root", default="runs/exp36")
    parser.add_argument("--exp35", default="runs/exp35")
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--own-root", default="runs/exp34_jlens_own", help="§34b own-position reads (token check, C4')")
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--confirm", action="store_true", help="required for any of folds 2, 3, 4")
    parser.add_argument("--out", default=None)
    parser.add_argument("--rescore", action="store_true",
                        help="recompute p_A/p_B/p_C, mass, s, argmax and top1 from each row's stored top-20 (no regeneration)")
    args = parser.parse_args(argv)
    folds = {int(f) for f in args.folds.split(",")}
    if folds & CONFIRM_FOLDS and not args.confirm:
        print(f"folds {sorted(folds & CONFIRM_FOLDS)} are confirmation folds; pass --confirm once the §36 pre-registration "
              "is accepted")
        return 2
    text, numbers = build(folds, Path(args.root), Path(args.exp35), args.runs, args.samples, Path(args.own_root),
                          args.rescore)
    label = "dev" if folds <= DEV_FOLDS else "confirm" if folds <= CONFIRM_FOLDS else "allfolds"
    out = Path(args.out or f"runs/exp36_report_{label}.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    out.with_suffix(".json").write_text(json.dumps(numbers, indent=1, default=float))
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
