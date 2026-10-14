"""§35 report: derivation vs answer selection (arms N, A, B, F, Fcut, Ncut), as pre-registered after review round 1.

    .venv/bin/python scripts/exp35_report.py --folds 0,1 [--out runs/exp35_report_dev.md]
    .venv/bin/python scripts/exp35_report.py --folds 2,3,4 --confirm [--out runs/exp35_report_confirm.md]
    .venv/bin/python scripts/exp35_report.py --folds 0,1,2,3,4 --confirm --out runs/exp35_report_allfolds.md

Pre-registration: research/proposals/2026-10-13-exp35-answer-selection.md at commit 12c9168 (reviewer round 2, conditional ACCEPT)
(3B primary; the 7B rows are descriptive).
Every verdict compares arms of the same engine and run: B against A, F against N; cached answers only define the
population, the labels, the notes and the conditioning sets. D1 (string) decides; D2 (judge on the note) is beside it.

- Token check: arm A's input (chat text + note + "Answer:") must have the token count of the §34b own-position read of
  the same question (runs/exp34_jlens_own/<reader>/<qid>.json: n_tokens - n_leading_ws). Mismatches are listed and
  excluded from the frozen-note arms (A and B); questions without a §34b read are listed and kept.
- Gate G: arm A gives the cached answer's correctness verdict and refusal flag on at least 0.8 of marked rows; below it
  Claims 1-2 are "not interpretable". Printed on all marked rows and again on S (the verdict uses all marked rows).
- Claim 1: on S (marked cached refusal, D1 true), paired P(B correct) - P(A correct), SHOWN if the lower bound is above
  0; UNDERPOWERED if |S| < 15 (with --confirm on folds 2-4 the all-folds pooled estimate is printed beside, labelled as
  mixing development and confirmation questions). s = P(B correct | S) with bootstrap and Wilson intervals, descriptive.
- Claim 2: P(B correct | S) - P(B correct | S'), S' = marked cached refusal with D1 false, SHOWN if the lower bound > 0.
- Claim 3 (no threshold): acc(B) - acc(A) and acc(F) - acc(N) on marked rows, gap share against the 7B's minus the 3B's
  cached accuracy on the same rows; acc(N) - cached, the cross-engine reproduction figure.
- Costs, both readers: (i) on S' (and on all marked cached refusals) the share B turns into judged-wrong answers;
  (ii) on the evidence-cut prompts, the share Fcut is judged correct and the share Ncut refuses.
Intervals: 10,000 bootstrap resamples of questions, 95%.

Folds 2, 3 and 4 are confirmation folds: without --confirm the report refuses them, and a development report opens only
the fold files and §34b reads of the folds it was asked for.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from memctl.analysis import answer_selection as sel  # noqa: E402
from memctl.analysis.jacobian_stats import bootstrap_mean_difference, bootstrap_proportion, paired_claim  # noqa: E402
from scripts.exp35_generate import READERS, arm_file, judge_file, load_population, read_jsonl  # noqa: E402

DEV_FOLDS = {0, 1}
CONFIRM_FOLDS = {2, 3, 4}
ALL_FOLDS = DEV_FOLDS | CONFIRM_FOLDS
PRIMARY = "qwen3b"
OTHER = {"qwen3b": "qwen7b", "qwen7b": "qwen3b"}


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float | None, float | None]:
    if n == 0:
        return None, None
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def _p(stat: dict) -> str:
    if not stat.get("n") or stat.get("p") is None:
        return "- (n 0)"
    return f"{stat['p']:.3f} ({stat['ci_low']:.3f}, {stat['ci_high']:.3f})"


def _d(stat: dict) -> str:
    if stat.get("mean") is None:
        return "-"
    if stat.get("ci_low") is None:
        return f"{stat['mean']:+.3f} (no interval)"
    return f"{stat['mean']:+.3f} ({stat['ci_low']:+.3f}, {stat['ci_high']:+.3f})"


def _f(x, digits=3) -> str:
    return "-" if x is None else f"{x:.{digits}f}"


class ReaderData:
    """Rows, verdicts and the token check for one reader on the chosen folds."""

    def __init__(self, reader: str, folds: set[int], root: Path, by_q: dict, qids: list[str], own_root: Path) -> None:
        self.reader, self.by_q, self.qids = reader, by_q, qids
        self.rows: dict[str, dict[str, dict]] = {arm: {} for arm in sel.ARMS}
        self.errors: dict[str, int] = {}
        for arm in sel.ARMS:
            self.errors[arm] = 0
            for fold in sorted(folds):
                for row in read_jsonl(arm_file(root, reader, arm, fold)):
                    if row["question_id"] in by_q:
                        self.rows[arm][row["question_id"]] = row
                self.errors[arm] += len(read_jsonl(root / reader / arm / f"errors.fold{fold}.jsonl"))
        self.verdict: dict[tuple[str, str], bool] = {}
        self.sources: dict[str, int] = defaultdict(int)
        for fold in sorted(folds):
            for row in read_jsonl(judge_file(root, reader, fold)):
                self.verdict[(row["item"], row["question_id"])] = bool(row["verdict"])
                self.sources[str(row.get("source"))] += 1
        self.marked = [q for q in qids if sel.split_note(by_q[q].output)[0] is not None]
        self.engines = sorted({row["engine"] for arm in sel.ARMS for row in self.rows[arm].values()})
        self.token_mismatch, self.token_missing, self.token_unchecked = self._token_check(own_root / reader)
        self.frozen = [q for q in self.marked if q not in self.token_mismatch]

    def _token_check(self, own_dir: Path) -> tuple[dict[str, tuple], list[str], int]:
        mismatch, missing, unchecked = {}, [], 0
        for q in self.marked:
            row = self.rows["A"].get(q)
            if row is None:
                continue
            if row.get("n_input_tokens") is None:
                unchecked += 1
                continue
            path = own_dir / f"{q}.json"
            if not path.exists():
                missing.append(q)
                continue
            own = json.loads(path.read_text())
            expected = own["n_tokens"] - own.get("n_leading_ws", 0)
            if own.get("answer_marker") is False or expected != row["n_input_tokens"]:
                mismatch[q] = (row["n_input_tokens"], expected)
        return mismatch, missing, unchecked

    def hash_problems(self) -> list[str]:
        return [f"{self.reader}/{arm}/{q}" for arm in sel.ARMS for q, row in self.rows[arm].items()
                if row["intact_prompt_sha256"] != sel.sha256(self.by_q[q].prompt)]

    def cached_refusal(self, q: str) -> bool:
        record = self.by_q[q]
        return record.correct is False and record.unknown is True

    def d1(self, q: str) -> bool | None:
        return sel.gold_in_note(self.by_q[q])

    def d2(self, q: str) -> bool | None:
        return self.verdict.get(("D2", q))

    def correct(self, arm: str, q: str) -> bool | None:
        return None if q not in self.rows[arm] else self.verdict.get((arm, q))

    def outcome(self, arm: str, q: str) -> str | None:
        c = self.correct(arm, q)
        return None if c is None else sel.outcome(c, self.rows[arm][q]["final_answer"])


def _acc(data: ReaderData, arm: str, qs: list[str]) -> dict[str, float]:
    return {q: 1.0 if data.correct(arm, q) else 0.0 for q in qs if data.correct(arm, q) is not None}


def _lift(new: dict, base: dict, samples: int) -> dict:
    keys = sorted(set(new) & set(base))
    stat = paired_claim({k: new[k] for k in keys}, {k: base[k] for k in keys}, samples=samples)
    stat.pop("verdict", None)
    stat["new_acc"] = sum(new[k] for k in keys) / len(keys) if keys else None
    stat["base_acc"] = sum(base[k] for k in keys) / len(keys) if keys else None
    stat["keys"] = keys
    return stat


def gate_g(data: ReaderData, samples: int, qids: list[str] | None = None) -> dict:
    """Share of rows (default: all marked rows after the token check) where arm A gives the cached answer's verdict
    and refusal flag."""
    flags = [data.correct("A", q) == bool(data.by_q[q].correct)
             and data.rows["A"][q]["refusal"] == bool(data.by_q[q].unknown)
             for q in (data.frozen if qids is None else qids) if data.correct("A", q) is not None]
    stat = bootstrap_proportion(flags, samples=samples)
    stat["pass"] = stat["p"] is not None and stat["p"] >= sel.GATE_G
    return stat


def claim1(data: ReaderData, samples: int, gate_pass: bool, crit: str = "D1") -> dict:
    fn = data.d1 if crit == "D1" else data.d2
    s_set = [q for q in data.frozen if data.cached_refusal(q) and fn(q) is True
             and data.correct("A", q) is not None and data.correct("B", q) is not None]
    b, a = _acc(data, "B", s_set), _acc(data, "A", s_set)
    stat = paired_claim(b, a, samples=samples)
    if not gate_pass:
        stat["verdict"] = "not interpretable (gate G failed)"
    elif len(s_set) < sel.UNDERPOWERED_BELOW:
        stat["verdict"] = f"UNDERPOWERED (|S| = {len(s_set)} < {sel.UNDERPOWERED_BELOW})"
    flags = [bool(b[q]) for q in s_set]
    stat["s"] = bootstrap_proportion(flags, samples=samples)
    stat["s_wilson"] = wilson(sum(flags), len(flags))
    stat["b_outcomes"] = {k: sum(data.outcome("B", q) == k for q in s_set) for k in ("correct", "refusal", "wrong")}
    stat["a_outcomes"] = {k: sum(data.outcome("A", q) == k for q in s_set) for k in ("correct", "refusal", "wrong")}
    stat["n_S"] = len(s_set)
    return stat


def analyse(data: ReaderData, other: ReaderData, samples: int) -> dict:
    out: dict = {}
    out["gate_g"] = g = gate_g(data, samples)
    out["claim1"] = claim1(data, samples, g["pass"], "D1")
    s_ids = [q for q in data.frozen if data.cached_refusal(q) and data.d1(q) is True]
    out["gate_g_on_S"] = gate_g(data, samples, s_ids)
    out["beside_D2"] = claim1(data, samples, g["pass"], "D2")
    out["beside_D2"]["verdict"] = "descriptive (D2 is reported beside D1, never deciding)"
    refusals = [q for q in data.frozen if data.cached_refusal(q) and data.correct("B", q) is not None]
    s_set = [q for q in refusals if data.d1(q) is True]
    s_prime = [q for q in refusals if data.d1(q) is False]
    c2 = bootstrap_mean_difference([1.0 if data.correct("B", q) else 0.0 for q in s_set],
                                   [1.0 if data.correct("B", q) else 0.0 for q in s_prime], samples=samples)
    c2["verdict"] = ("not interpretable (gate G failed)" if not g["pass"] else
                     "SHOWN" if c2["ci_low"] is not None and c2["ci_low"] > 0 else "NOT SHOWN")
    out["claim2"] = c2
    out["n_marked_cached_refusals"] = len(refusals)
    out["n_cached_refusals_all"] = sum(data.cached_refusal(q) for q in data.qids)
    out["S_share_of_population"] = len(s_set) / len(data.qids) if data.qids else None
    cached = {q: 1.0 if data.by_q[q].correct else 0.0 for q in data.qids}
    other_cached = {q: 1.0 if other.by_q[q].correct else 0.0 for q in data.qids}
    c3 = {"B_minus_A": _lift(_acc(data, "B", data.frozen), _acc(data, "A", data.frozen), samples),
          "F_minus_N": _lift(_acc(data, "F", data.marked), _acc(data, "N", data.marked), samples),
          "F_minus_N_all": _lift(_acc(data, "F", data.qids), _acc(data, "N", data.qids), samples),
          "N_minus_cached": _lift(_acc(data, "N", data.marked), cached, samples),
          "N_minus_cached_all": _lift(_acc(data, "N", data.qids), cached, samples),
          "A_minus_cached": _lift(_acc(data, "A", data.frozen), cached, samples)}
    for key in ("B_minus_A", "F_minus_N", "F_minus_N_all"):
        stat = c3[key]
        rows = stat["keys"]
        gap = (sum(other_cached[q] for q in rows) - sum(cached[q] for q in rows)) / len(rows) if rows else None
        stat["matched_gap"] = gap
        ok = data.reader == PRIMARY and gap and stat["mean"] is not None
        stat["gap_share"] = stat["mean"] / gap if ok else None
        stat["gap_share_ci"] = (stat["ci_low"] / gap, stat["ci_high"] / gap) if ok and stat["ci_low"] is not None else None
    for stat in c3.values():
        stat["n_rows"] = len(stat.pop("keys"))
    out["claim3"] = c3
    out["cost_i"] = {"S_prime": bootstrap_proportion([data.outcome("B", q) == "wrong" for q in s_prime], samples=samples),
                     "all_marked_refusals": bootstrap_proportion([data.outcome("B", q) == "wrong" for q in refusals],
                                                                 samples=samples)}
    fcut = [q for q in data.qids if data.correct("Fcut", q) is not None]
    ncut = [q for q in data.qids if q in data.rows["Ncut"]]
    out["cost_ii"] = {
        "fcut_judged_correct": bootstrap_proportion([bool(data.correct("Fcut", q)) for q in fcut], samples=samples),
        "ncut_refusal": bootstrap_proportion([data.rows["Ncut"][q]["refusal"] for q in ncut], samples=samples),
        "fcut_refusal": bootstrap_proportion([data.rows["Fcut"][q]["refusal"] for q in fcut], samples=samples),
        "ncut_judged_correct": bootstrap_proportion([bool(data.correct("Ncut", q)) for q in ncut
                                                     if data.correct("Ncut", q) is not None], samples=samples),
    }
    with_a = [q for q in data.frozen if q in data.rows["A"]]
    with_n = [q for q in data.qids if q in data.rows["N"]]
    cached_answer = {q: sel.agent_answer(data.by_q[q].output)[0] for q in data.qids}
    out["reproduction"] = {
        "A_same_answer": bootstrap_proportion([data.rows["A"][q]["final_answer"] == cached_answer[q] for q in with_a], samples=samples),
        "N_same_answer": bootstrap_proportion([data.rows["N"][q]["final_answer"] == cached_answer[q] for q in with_n], samples=samples),
        "N_same_output": bootstrap_proportion([data.rows["N"][q]["generated"].strip() == data.by_q[q].output.strip()
                                               for q in with_n], samples=samples),
    }
    table = {"both": 0, "D1 only": 0, "D2 only": 0, "neither": 0}
    for q in data.marked:
        x, y = data.d1(q), data.d2(q)
        if x is None or y is None:
            continue
        table["both" if x and y else "D1 only" if x else "D2 only" if y else "neither"] += 1
    n = sum(table.values())
    agree = (table["both"] + table["neither"]) / n if n else None
    p1 = (table["both"] + table["D1 only"]) / n if n else 0
    p2 = (table["both"] + table["D2 only"]) / n if n else 0
    expected = p1 * p2 + (1 - p1) * (1 - p2)
    out["d1_d2"] = {"table": table, "n": n, "agreement": agree,
                    "kappa": (agree - expected) / (1 - expected) if n and expected < 1 else None,
                    "on_marked_refusals": {"D1": sum(data.d1(q) is True for q in refusals),
                                           "D2": sum(data.d2(q) is True for q in refusals), "n": len(refusals)}}
    labelled = [q for q in data.qids if ("cached", q) in data.verdict]
    flips = [q for q in labelled if data.verdict[("cached", q)] != bool(data.by_q[q].correct)]
    out["judge_engine_check"] = {"n": len(labelled), "flips": len(flips),
                                 "agreement": 1 - len(flips) / len(labelled) if labelled else None}
    out["token_check"] = {"mismatch": data.token_mismatch, "missing_own_read": data.token_missing,
                          "unchecked_no_tokenizer": data.token_unchecked}
    out["types"] = type_split(data)
    out["judge_sources"] = dict(data.sources)
    out["rows"] = {a: len(data.rows[a]) for a in sel.ARMS}
    out["errors"] = data.errors
    return out


def type_split(data: ReaderData) -> dict:
    groups: dict[str, list[str]] = defaultdict(list)
    for q in data.qids:
        groups[data.by_q[q].question_type].append(q)
    out = {}
    for qtype, members in sorted(groups.items()):
        def acc(arm, pool):
            vals = [data.correct(arm, q) for q in pool if data.correct(arm, q) is not None]
            return (sum(vals) / len(vals), len(vals)) if vals else (None, 0)
        frozen = [q for q in members if q in data.frozen]
        s_set = [q for q in frozen if data.cached_refusal(q) and data.d1(q) is True]
        out[qtype] = {"n": len(members), "frozen": len(frozen), "flagged": qtype in sel.FLAGGED_TYPES,
                      "cached_acc": sum(bool(data.by_q[q].correct) for q in members) / len(members),
                      "N": acc("N", members), "A": acc("A", frozen), "B": acc("B", frozen), "F": acc("F", members),
                      "S": len(s_set), "S_b_correct": sum(bool(data.correct("B", q)) for q in s_set),
                      "S_a_correct": sum(bool(data.correct("A", q)) for q in s_set)}
    return out


def natural_control(folds: set[int], runs: str) -> dict:
    """Answerable, labelled questions with zero evidence lines in view: cached refusal count (descriptive, §1)."""
    from memctl.jacobian_data import load_records

    out = {}
    for reader in READERS:
        rows = [x for f in sorted(folds) for x in load_records(reader, f, runs_dir=runs)[0]]
        labelled = [x for x in rows if x.answerable and x.correct is not None]
        zero = [x for x in labelled if not x.evidence_lines]
        out[reader] = {"labelled": len(labelled), "zero_evidence": len(zero),
                       "cached_refusals": sum(x.unknown is True and x.correct is False for x in zero)}
    return out


def load(folds: set[int], root: Path, runs: str, own_root: Path) -> tuple[dict, list[str], list[str]]:
    by_reader, qids, mismatched = load_population(sorted(folds), runs)
    data = {r: ReaderData(r, folds, root, by_reader[r], qids, own_root) for r in READERS}
    engines = sorted({e for d in data.values() for e in d.engines})
    if len(engines) > 1:
        raise SystemExit(f"REFUSED: the run directory mixes engines {engines}; one engine for every arm of both readers")
    problems = [p for d in data.values() for p in d.hash_problems()]
    if problems:
        raise SystemExit(f"REFUSED: {len(problems)} rows do not carry their record's intact prompt hash, e.g. {problems[:3]}")
    return data, qids, mismatched


def render(folds: set[int], numbers: dict, data: dict) -> str:
    fl = ", ".join(map(str, sorted(folds)))
    lines = [f"# §35 report: derivation vs answer selection (folds {fl}; population {numbers['population']} in-view "
             "answerable questions, paired by question id and prompt)", "",
             f"Engines: {numbers['engines']}. Wording sha256 {numbers['wording_sha256'][:12]}. Every verdict compares arms of "
             "the same engine and run (B vs A, F vs N). Judge: the §23 prompt and model (Qwen2.5-7B); an identical judge "
             "prompt already judged by §23 reuses its verdict. D1 decides; D2 beside. 95% intervals: 10,000 bootstrap "
             "resamples of questions.", "",
             "## Population and rows", "",
             "| reader | population | marked | no marker | frozen-note rows after token check | " + " | ".join(sel.ARMS)
             + " | replacement / cut errors | token mismatches (excluded) | no §34b read (kept) |",
             "|---|---|---|---|---|" + "---|" * len(sel.ARMS) + "---|---|---|"]
    for reader, d in data.items():
        n = numbers["readers"][reader]
        lines.append(f"| {reader} | {len(d.qids)} | {len(d.marked)} | {len(d.qids) - len(d.marked)} | {len(d.frozen)} | "
                     + " | ".join(str(len(d.rows[a])) for a in sel.ARMS)
                     + f" | {sum(d.errors.values())} | {len(n['token_check']['mismatch'])} | {len(n['token_check']['missing_own_read'])} |")
    unchecked = {r: numbers["readers"][r]["token_check"]["unchecked_no_tokenizer"] for r in data}
    if any(unchecked.values()):
        lines += ["", f"Token check not possible (engine without a tokenizer) on {unchecked} rows."]
    for reader in data:
        mm = numbers["readers"][reader]["token_check"]["mismatch"]
        if mm:
            lines += ["", f"{reader} token mismatches (arm A input vs §34b): " + ", ".join(f"{q} {a} vs {b}" for q, (a, b) in sorted(mm.items()))]
    lines.append("")
    for reader in data:
        n = numbers["readers"][reader]
        role = "primary" if reader == PRIMARY else "descriptive"
        v = (lambda s: s) if reader == PRIMARY else (lambda s: "descriptive")
        g, gs, c1, c1d2, c2, c3 = n["gate_g"], n["gate_g_on_S"], n["claim1"], n["beside_D2"], n["claim2"], n["claim3"]
        s = c1["s"]
        wl = c1["s_wilson"]
        lines += [f"## {reader} ({role})", "",
                  f"Gate G: arm A gives the cached verdict and refusal flag on {_p(g)} of {g['n']} marked rows "
                  f"(bound {sel.GATE_G}): **{'PASS' if g['pass'] else 'FAIL'}**"
                  + ("" if g["pass"] else "; frozen-note claims are not interpretable") + f". On S: {_p(gs)} of {gs['n']}.", "",
                  f"Marked cached refusals {n['n_marked_cached_refusals']} (all cached refusals {n['n_cached_refusals_all']}); "
                  f"|S| (D1 true) {c1['n_S']}, S as a share of the population {_f(n['S_share_of_population'])}.", "",
                  "| claim | n | estimate (95% CI) | verdict | B outcomes (correct / refusal / wrong) | A outcomes |",
                  "|---|---|---|---|---|---|",
                  f"| Claim 1: P(B correct) − P(A correct) on S (D1) | {c1['n_S']} | {_d(c1)} | {v(c1['verdict'])} | "
                  f"{c1['b_outcomes']['correct']} / {c1['b_outcomes']['refusal']} / {c1['b_outcomes']['wrong']} | "
                  f"{c1['a_outcomes']['correct']} / {c1['a_outcomes']['refusal']} / {c1['a_outcomes']['wrong']} |",
                  f"| s = P(B correct \\| S), descriptive (Wilson {_f(wl[0])}, {_f(wl[1])}) | {s['n']} | {_p(s)} | descriptive | | |",
                  f"| beside, not deciding: the same difference on the D2 set | {c1d2['n_S']} | {_d(c1d2)} | descriptive | "
                  f"{c1d2['b_outcomes']['correct']} / {c1d2['b_outcomes']['refusal']} / {c1d2['b_outcomes']['wrong']} | "
                  f"{c1d2['a_outcomes']['correct']} / {c1d2['a_outcomes']['refusal']} / {c1d2['a_outcomes']['wrong']} |",
                  f"| Claim 2: P(B correct \\| S) − P(B correct \\| S') | {c2['n_a']} vs {c2['n_b']} | {_d(c2)} | {v(c2['verdict'])} | | |"]
        if "pooled" in n:
            p = n["pooled"]
            lines.append(f"| Claim 1 pooled over folds 0-4 (MIXES development and confirmation questions) | {p['n_S']} | "
                         f"{_d(p)} | not a verdict | {p['b_outcomes']['correct']} / {p['b_outcomes']['refusal']} / "
                         f"{p['b_outcomes']['wrong']} | {p['a_outcomes']['correct']} / {p['a_outcomes']['refusal']} / {p['a_outcomes']['wrong']} |")
        lines += ["", "Claim 3 (descriptive, no threshold): paired accuracy differences within the engine.", "",
                  "| comparison | rows | arm accuracy | baseline accuracy | difference (95% CI) | 7B − 3B cached accuracy on the same rows | gap share (95% CI) |",
                  "|---|---|---|---|---|---|---|"]
        for key, label in (("B_minus_A", "acc(B) − acc(A), marked rows after token check"),
                           ("F_minus_N", "acc(F) − acc(N), marked rows"),
                           ("F_minus_N_all", "acc(F) − acc(N), all rows (descriptive)"),
                           ("N_minus_cached", "acc(N) − cached, marked rows (cross-engine reproduction)"),
                           ("N_minus_cached_all", "acc(N) − cached, all rows"),
                           ("A_minus_cached", "acc(A) − cached, marked rows after token check")):
            st = c3[key]
            share = "-" if st.get("gap_share") is None else f"{st['gap_share']:+.2f}" + (
                "" if st.get("gap_share_ci") is None else f" ({st['gap_share_ci'][0]:+.2f}, {st['gap_share_ci'][1]:+.2f})")
            lines.append(f"| {label} | {st['n_rows']} | {_f(st['new_acc'])} | {_f(st['base_acc'])} | {_d(st)} | "
                         f"{_f(st.get('matched_gap'))} | {share} |")
        ci, cii = n["cost_i"], n["cost_ii"]
        lines += ["", "Costs (pre-registered, reported with the claims).", "",
                  "| cost | n | share (95% CI) |", "|---|---|---|",
                  f"| (i) S' (cached refusal, D1 false): B judged wrong | {ci['S_prime']['n']} | {_p(ci['S_prime'])} |",
                  f"| (i) all marked cached refusals: B judged wrong | {ci['all_marked_refusals']['n']} | {_p(ci['all_marked_refusals'])} |",
                  f"| (ii) evidence cut: Fcut judged correct (plausibility floor) | {cii['fcut_judged_correct']['n']} | {_p(cii['fcut_judged_correct'])} |",
                  f"| (ii) evidence cut: Ncut refuses (unknown is right) | {cii['ncut_refusal']['n']} | {_p(cii['ncut_refusal'])} |",
                  f"| evidence cut: Fcut refuses (descriptive) | {cii['fcut_refusal']['n']} | {_p(cii['fcut_refusal'])} |",
                  f"| evidence cut: Ncut judged correct (descriptive) | {cii['ncut_judged_correct']['n']} | {_p(cii['ncut_judged_correct'])} |", ""]
        r, dd, je = n["reproduction"], n["d1_d2"], n["judge_engine_check"]
        lines += [f"Reproduction: arm A gives the cached final answer exactly on {_p(r['A_same_answer'])}; arm N gives the "
                  f"cached final answer on {_p(r['N_same_answer'])} and the whole cached output on {_p(r['N_same_output'])}.", "",
                  f"D1 vs D2 on marked notes (n {dd['n']}): both {dd['table']['both']}, D1 only {dd['table']['D1 only']}, "
                  f"D2 only {dd['table']['D2 only']}, neither {dd['table']['neither']}; agreement {_f(dd['agreement'])}, "
                  f"Cohen's kappa {_f(dd['kappa'])}. On marked cached refusals: D1 {dd['on_marked_refusals']['D1']}, "
                  f"D2 {dd['on_marked_refusals']['D2']} of {dd['on_marked_refusals']['n']}.", "",
                  f"Judge engine check: the cached answers re-judged in this run agree with the §23 labels on "
                  f"{je['n'] - je['flips']} of {je['n']} ({_f(je['agreement'])}).", "",
                  "Per question type (descriptive; accuracy with n in brackets; A and B on marked rows after the token check; "
                  "flagged types have a different judge prompt):", "",
                  "| type | n | frozen | cached | N | A | B | F | \\|S\\| | S: A correct | S: B correct |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        fmt = lambda x: "-" if x[0] is None else f"{x[0]:.3f} ({x[1]})"  # noqa: E731
        for qtype, t in n["types"].items():
            name = qtype + (" (flagged)" if t["flagged"] else "")
            lines.append(f"| {name} | {t['n']} | {t['frozen']} | {t['cached_acc']:.3f} | {fmt(t['N'])} | {fmt(t['A'])} | "
                         f"{fmt(t['B'])} | {fmt(t['F'])} | {t['S']} | {t['S_a_correct']} | {t['S_b_correct']} |")
        lines += ["", f"Judge verdict sources: {n['judge_sources']}.", ""]
    nc = numbers["natural_control"]
    lines += ["## Natural control (cached, descriptive)", "",
              "Answerable labelled questions with zero evidence lines in view: " + "; ".join(
                  f"{r}: {x['zero_evidence']} of {x['labelled']}, cached refusals {x['cached_refusals']}" for r, x in nc.items()) + ".", "",
              f"Verdict rules (pre-registered, qwen3b only): gate G at {sel.GATE_G}; Claim 1 SHOWN if the lower bound of "
              f"P(B correct) − P(A correct) on S is above 0, UNDERPOWERED if |S| < {sel.UNDERPOWERED_BELOW}; Claim 2 SHOWN if "
              "the lower bound is above 0; Claim 3 has no threshold (a lift under 0.01 is within the reader noise floor); its gap "
              "share is against the 7B's minus the 3B's cached accuracy on the same rows."]
    return "\n".join(lines) + "\n"


def build(folds: set[int], root: Path, runs: str = "runs", samples: int = 10000, own_root: Path = Path("runs/exp34_jlens_own"),
          pooled: bool = False) -> tuple[str, dict]:
    data, qids, mismatched = load(folds, root, runs, own_root)
    numbers = {"folds": sorted(folds), "population": len(qids), "prompt_mismatches": mismatched,
               "engines": sorted({e for d in data.values() for e in d.engines}), "wording_sha256": sel.wording_sha256(),
               "readers": {r: analyse(data[r], data[OTHER[r]], samples) for r in READERS}}
    if pooled and numbers["readers"][PRIMARY]["claim1"]["verdict"].startswith("UNDERPOWERED") and folds != ALL_FOLDS:
        all_data, _, _ = load(ALL_FOLDS, root, runs, own_root)
        g = gate_g(all_data[PRIMARY], samples)
        numbers["readers"][PRIMARY]["pooled"] = claim1(all_data[PRIMARY], samples, g["pass"])
    numbers["natural_control"] = natural_control(folds, runs)
    return render(folds, numbers, data), numbers


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", default="0,1")
    parser.add_argument("--root", default="runs/exp35")
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--own-root", default="runs/exp34_jlens_own", help="§34b own-position reads (token check)")
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--confirm", action="store_true", help="required for any of folds 2, 3, 4")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    folds = {int(f) for f in args.folds.split(",")}
    if folds & CONFIRM_FOLDS and not args.confirm:
        print(f"folds {sorted(folds & CONFIRM_FOLDS)} are confirmation folds; pass --confirm once the §35 pre-registration "
              "is accepted")
        return 2
    text, numbers = build(folds, Path(args.root), args.runs, args.samples, Path(args.own_root), pooled=args.confirm)
    label = "dev" if folds <= DEV_FOLDS else "confirm" if folds <= CONFIRM_FOLDS else "allfolds"
    out = Path(args.out or f"runs/exp35_report_{label}.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    out.with_suffix(".json").write_text(json.dumps(numbers, indent=1, default=float))
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
