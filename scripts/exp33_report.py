"""§33 report: the pre-registered claims from runs/exp33_jacobian/<reader>/ outputs.

    .venv/bin/python scripts/exp33_report.py --small qwen3b --large qwen7b --folds 0,1 --variant l2 --out runs/exp33_report_dev.md

Folds default to the development folds 0,1. Confirmation folds (2,3,4) are read only after the
pre-registration is committed; the script refuses them unless --confirm is passed. Questions are paired
across readers only when both gold jsons carry the same prompt_sha256; otherwise the report stops
(exit 4) and lists the mismatched or missing hashes."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memctl.analysis.jacobian_stats import (bootstrap_auc, bootstrap_mean_difference, bootstrap_spearman,  # noqa: E402
                                            paired_claim)
from memctl.jacobian_data import PromptRecord, load_records  # noqa: E402

DEV_FOLDS = {0, 1}
TARGETS = ("gold", "margin", "own")
AUC_BAR = 0.5  # Claim 2: SHOWN when the AUC interval lies above chance
RHO_BAR = 0.3  # ablation check (spec §7): PASS when rho >= 0.3 and its interval lies above 0


class PromptIdentityError(ValueError):
    """The two readers' gold jsons for a question do not carry the same prompt hash."""


def load_outputs(out_dir: Path, folds: set[int], target: str) -> dict[str, dict]:
    found = {}
    for path in sorted(Path(out_dir).glob(f"*.{target}.json")):
        entry = json.loads(path.read_text())
        if entry["fold"] in folds:
            found[entry["question_id"]] = entry["measures"]
    return found


def load_ablation(out_dir: Path, folds: set[int]) -> dict[str, float]:
    """question_id -> delta for questions whose gold output is in the chosen folds."""
    folds_of = {}
    for path in Path(out_dir).glob("*.gold.json"):
        entry = json.loads(path.read_text())
        folds_of[entry["question_id"]] = entry["fold"]
    deltas = {}
    for path in Path(out_dir).glob("*.ablation.json"):
        entry = json.loads(path.read_text())
        if folds_of.get(entry["question_id"]) in folds:
            deltas[entry["question_id"]] = entry["delta"]
    return deltas


def load_hashes(out_dir: Path, folds: set[int]) -> dict[str, str | None]:
    """question_id -> prompt_sha256 of its gold json (None for a json written before the hash was stored)."""
    hashes = {}
    for path in Path(out_dir).glob("*.gold.json"):
        entry = json.loads(path.read_text())
        if entry["fold"] in folds:
            hashes[entry["question_id"]] = entry.get("prompt_sha256")
    return hashes


def prompt_mismatches(qids: list[str], small: dict[str, str | None], large: dict[str, str | None]) -> list[str]:
    """One line per question with both gold jsons whose hashes differ or are missing."""
    problems = []
    for q in qids:
        if q not in small or q not in large:
            continue
        a, b = small[q], large[q]
        if a is None or b is None:
            which = " and ".join(name for name, h in (("small", a), ("large", b)) if h is None)
            problems.append(f"{q}: prompt_sha256 missing from the {which} reader's gold json")
        elif a != b:
            problems.append(f"{q}: prompt_sha256 differs ({a[:12]} vs {b[:12]})")
    return problems


def population(records_by_reader: dict[str, list[PromptRecord]]) -> list[str]:
    sets = []
    for records in records_by_reader.values():
        sets.append({r.question_id for r in records if r.answerable and r.all_in_view and r.evidence_lines})
    return sorted(set.intersection(*sets)) if sets else []


def _fmt(stat: dict, key: str = "mean") -> str:
    if stat.get("ci_low") is None:
        return f"{stat.get(key)}"
    return f"{stat[key]:+.3f} ({stat['ci_low']:+.3f}, {stat['ci_high']:+.3f})"


def build_report(readers: tuple[str, str], folds: set[int], variant: str, k_label: str, out_root: Path,
                 samples: int = 10000, records_from: str | None = None) -> tuple[str, dict]:
    small, large = readers
    records_by_reader, dropped = {}, {}
    for reader in readers:
        rows = []
        dropped[reader] = {}
        for fold in sorted(folds):
            fold_rows, fold_dropped = load_records(records_from or reader, fold)
            rows.extend(fold_rows)
            dropped[reader][str(fold)] = fold_dropped
        records_by_reader[reader] = rows
    qids = population(records_by_reader)
    by_q = {reader: {r.question_id: r for r in rows} for reader, rows in records_by_reader.items()}
    outputs = {reader: {t: load_outputs(out_root / reader, folds, t) for t in TARGETS} for reader in readers}
    problems = prompt_mismatches(qids, load_hashes(out_root / small, folds), load_hashes(out_root / large, folds))
    if problems:
        raise PromptIdentityError(f"{len(problems)} question(s) cannot be paired: the {small} and {large} gold jsons "
                                  "do not carry the same prompt_sha256:\n  " + "\n  ".join(problems))
    numbers: dict = {"folds": sorted(folds), "variant": variant, "population": len(qids), "dropped": dropped,
                     "claim1": {}, "claim2": {}, "ablation": {}, "shares": {}, "cells": {}}
    lines = [f"# §33 Jacobian lens report ({small} vs {large}; folds: {', '.join(map(str, sorted(folds)))}; "
             f"variant {variant}; population {len(qids)} in-view answerable questions)", ""]

    lines += [f"## Dropped records (no cached prompt; complete and answerable but evidence not found)"
              + (f"; records from {records_from}" if records_from else ""), "",
              "| reader | fold | no cached prompt | evidence not found |", "|---|---|---|---|"]
    for reader in readers:
        for fold, counts in dropped[reader].items():
            lines.append(f"| {reader} | {fold} | {counts.get('no_cached_prompt')} | {counts.get('evidence_not_found')} |")
    lines.append("")

    lines += ["## Claim 1: evidence share, large minus small (paired by question)", "",
              "| target | n | mean difference (95% CI) | verdict |", "|---|---|---|---|"]
    for target in ("gold", "margin"):
        a = {q: outputs[large][target][q][variant]["shares"]["evidence"] for q in qids if q in outputs[large][target]}
        b = {q: outputs[small][target][q][variant]["shares"]["evidence"] for q in qids if q in outputs[small][target]}
        stat = paired_claim(a, b, samples=samples)
        numbers["claim1"][target] = stat
        lines.append(f"| {target} | {stat['n']} | {_fmt(stat)} | {stat['verdict']} |")

    lines += ["", "## Claim 2: within each reader, evidence share on correct vs wrong (gold target, AUC)", "",
              "| reader | n right | n wrong | AUC (95% CI) | mean right | mean wrong | right minus wrong (95% CI) | verdict |",
              "|---|---|---|---|---|---|---|---|"]
    for reader in readers:
        shares = outputs[reader]["gold"]
        right = [shares[q][variant]["shares"]["evidence"] for q in qids if q in shares and by_q[reader][q].correct is True]
        wrong = [shares[q][variant]["shares"]["evidence"] for q in qids if q in shares and by_q[reader][q].correct is False]
        stat = bootstrap_auc(right, wrong, samples=samples) if right and wrong else {"n_pos": len(right), "n_neg": len(wrong),
                                                                                    "auc": None, "ci_low": None, "ci_high": None}
        stat["difference"] = bootstrap_mean_difference(right, wrong, samples=samples)
        stat["verdict"] = "SHOWN" if stat["ci_low"] is not None and stat["ci_low"] > AUC_BAR else "NOT SHOWN"
        numbers["claim2"][reader] = stat
        lines.append(f"| {reader} | {len(right)} | {len(wrong)} | {_fmt(stat, 'auc')} | "
                     f"{mean(right) if right else float('nan'):.3f} | {mean(wrong) if wrong else float('nan'):.3f} | "
                     f"{_fmt(stat['difference'])} | {stat['verdict']} |")

    lines += ["", "## Ablation check: Spearman(evidence share, Δ gold log-prob when evidence is removed)", "",
              "| reader | n | rho (95% CI) | mean Δ | verdict |", "|---|---|---|---|---|"]
    for reader in readers:
        deltas = load_ablation(out_root / reader, folds)
        shares = outputs[reader]["gold"]
        keys = [q for q in qids if q in deltas and q in shares]
        if len(keys) >= 3:
            stat = bootstrap_spearman([shares[q][variant]["shares"]["evidence"] for q in keys], [deltas[q] for q in keys], samples=samples)
            stat["mean_delta"] = mean(deltas[q] for q in keys)
        else:
            stat = {"n": len(keys), "rho": None, "ci_low": None, "ci_high": None, "mean_delta": None}
        stat["verdict"] = ("PASS" if stat["rho"] is not None and stat["rho"] >= RHO_BAR and stat["ci_low"] is not None
                           and stat["ci_low"] > 0 else "FAIL")
        numbers["ablation"][reader] = stat
        lines.append(f"| {reader} | {stat['n']} | {_fmt(stat, 'rho')} | {stat['mean_delta']} | {stat['verdict']} |")

    lines += ["", "## Descriptive: mean region shares and evidence lift per reader and target", "",
              "| reader | target | instruction | evidence | distractor | question | lift | top-k evidence |",
              "|---|---|---|---|---|---|---|---|"]
    for reader in readers:
        refusals = [q for q in qids if by_q[reader][q].unknown is True]
        for target, chosen in [(t, qids) for t in TARGETS] + [("own (refusals)", refusals)]:
            source = outputs[reader][target.split(" ")[0]]
            rows = [source[q][variant] for q in chosen if q in source]
            if not rows:
                continue
            avg = {name: mean(r["shares"][name] for r in rows) for name in ("instruction", "evidence", "distractor", "question")}
            lifts = [r["lift"] for r in rows if r["lift"] is not None]
            lift = mean(lifts) if lifts else float("nan")
            topk = mean(r["topk"] for r in rows)
            numbers["shares"][f"{reader}/{target}"] = {**avg, "lift": lift, "topk": topk, "n": len(rows)}
            lines.append(f"| {reader} | {target} | {avg['instruction']:.3f} | {avg['evidence']:.3f} | {avg['distractor']:.3f} | "
                         f"{avg['question']:.3f} | {lift:.2f} | {topk:.3f} |")

    lines += ["", f"## Descriptive: the four outcome cells ({small} right?, {large} right?) and mean evidence share (gold)", "",
              f"| {small} | {large} | n | {small} share | {large} share |", "|---|---|---|---|---|"]
    cells = defaultdict(list)
    for q in qids:
        if q in outputs[small]["gold"] and q in outputs[large]["gold"]:
            cells[(by_q[small][q].correct, by_q[large][q].correct)].append(q)
    for key in sorted(cells, key=str):
        qs = cells[key]
        s = mean(outputs[small]["gold"][q][variant]["shares"]["evidence"] for q in qs)
        l = mean(outputs[large]["gold"][q][variant]["shares"]["evidence"] for q in qs)
        numbers["cells"][str(key)] = {"n": len(qs), small: s, large: l}
        lines.append(f"| {key[0]} | {key[1]} | {len(qs)} | {s:.3f} | {l:.3f} |")

    lines += ["", f"Top-k label: {k_label}. Verdict rules: Claim 1 SHOWN if the lower bound is above 0; Claim 2 SHOWN "
              f"if the AUC lower bound is above {AUC_BAR}; ablation PASS if rho >= {RHO_BAR} and its lower bound is above 0. "
              "`own (refusals)` is the own target on the questions the reader answered unknown. "
              "Correct/unknown labels are the §23 single-judge verdicts (STATE.md caveat 3)."]
    return "\n".join(lines) + "\n", numbers


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--small", default="qwen3b")
    parser.add_argument("--large", default="qwen7b")
    parser.add_argument("--folds", default="0,1")
    parser.add_argument("--variant", default="l2", choices=("l2", "gxi"))
    parser.add_argument("--root", default="runs/exp33_jacobian")
    parser.add_argument("--out", default=None)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--records-from", default=None,
                        help="reader whose §23 records to use for every reader (local development with the 0.5B)")
    parser.add_argument("--confirm", action="store_true", help="allow folds outside the development set")
    args = parser.parse_args(argv)
    folds = {int(f) for f in args.folds.split(",")}
    if not folds <= DEV_FOLDS and not args.confirm:
        print(f"folds {sorted(folds - DEV_FOLDS)} are confirmation folds; pass --confirm after the pre-registration is committed")
        return 2
    try:
        markdown, numbers = build_report((args.small, args.large), folds, args.variant, "k50", Path(args.root), args.samples,
                                         records_from=args.records_from)
    except PromptIdentityError as error:
        print(f"REFUSED: {error}")
        return 4
    out = Path(args.out or f"runs/exp33_report_{'dev' if folds <= DEV_FOLDS else 'confirm'}.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(markdown)
    out.with_suffix(".json").write_text(json.dumps(numbers, indent=1))
    print(markdown)
    return 0


if __name__ == "__main__":
    sys.exit(main())
