"""§33 report: the pre-registered claims from runs/exp33_jacobian/<reader>/ outputs.

    .venv/bin/python scripts/exp33_report.py --small qwen3b --large qwen7b --folds 0,1 --variant l2 --out runs/exp33_report_dev.md

Folds default to the development folds 0,1. Confirmation folds (2,3,4) are read only after the
pre-registration is committed; the script refuses them unless --confirm is passed."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memctl.analysis.jacobian_stats import bootstrap_auc, bootstrap_spearman, paired_claim  # noqa: E402
from memctl.jacobian_data import PromptRecord, load_records  # noqa: E402

DEV_FOLDS = {0, 1}
TARGETS = ("gold", "margin", "own")


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
                 samples: int = 10000) -> tuple[str, dict]:
    small, large = readers
    records_by_reader = {}
    for reader in readers:
        rows = []
        for fold in sorted(folds):
            fold_rows, _ = load_records(reader, fold)
            rows.extend(fold_rows)
        records_by_reader[reader] = rows
    qids = population(records_by_reader)
    by_q = {reader: {r.question_id: r for r in rows} for reader, rows in records_by_reader.items()}
    numbers: dict = {"folds": sorted(folds), "variant": variant, "population": len(qids), "claim1": {}, "claim2": {},
                     "ablation": {}, "shares": {}, "cells": {}}
    lines = [f"# §33 Jacobian lens report ({small} vs {large}; folds: {', '.join(map(str, sorted(folds)))}; "
             f"variant {variant}; population {len(qids)} in-view answerable questions)", ""]
    outputs = {reader: {t: load_outputs(out_root / reader, folds, t) for t in TARGETS} for reader in readers}

    lines += ["## Claim 1: evidence share, large minus small (paired by question)", "",
              "| target | n | mean difference (95% CI) | verdict |", "|---|---|---|---|"]
    for target in ("gold", "margin"):
        a = {q: outputs[large][target][q][variant]["shares"]["evidence"] for q in qids if q in outputs[large][target]}
        b = {q: outputs[small][target][q][variant]["shares"]["evidence"] for q in qids if q in outputs[small][target]}
        stat = paired_claim(a, b, samples=samples)
        numbers["claim1"][target] = stat
        lines.append(f"| {target} | {stat['n']} | {_fmt(stat)} | {stat['verdict']} |")

    lines += ["", "## Claim 2: within each reader, evidence share on correct vs wrong (gold target, AUC)", "",
              "| reader | n right | n wrong | AUC (95% CI) | mean right | mean wrong |", "|---|---|---|---|---|---|"]
    for reader in readers:
        shares = outputs[reader]["gold"]
        right = [shares[q][variant]["shares"]["evidence"] for q in qids if q in shares and by_q[reader][q].correct is True]
        wrong = [shares[q][variant]["shares"]["evidence"] for q in qids if q in shares and by_q[reader][q].correct is False]
        stat = bootstrap_auc(right, wrong, samples=samples) if right and wrong else {"n_pos": len(right), "n_neg": len(wrong),
                                                                                    "auc": None, "ci_low": None, "ci_high": None}
        numbers["claim2"][reader] = stat
        lines.append(f"| {reader} | {len(right)} | {len(wrong)} | {_fmt(stat, 'auc')} | "
                     f"{mean(right) if right else float('nan'):.3f} | {mean(wrong) if wrong else float('nan'):.3f} |")

    lines += ["", "## Ablation check: Spearman(evidence share, Δ gold log-prob when evidence is removed)", "",
              "| reader | n | rho (95% CI) | mean Δ |", "|---|---|---|---|"]
    for reader in readers:
        deltas = load_ablation(out_root / reader, folds)
        shares = outputs[reader]["gold"]
        keys = [q for q in qids if q in deltas and q in shares]
        if len(keys) >= 3:
            stat = bootstrap_spearman([shares[q][variant]["shares"]["evidence"] for q in keys], [deltas[q] for q in keys], samples=samples)
            stat["mean_delta"] = mean(deltas[q] for q in keys)
        else:
            stat = {"n": len(keys), "rho": None, "ci_low": None, "ci_high": None, "mean_delta": None}
        numbers["ablation"][reader] = stat
        lines.append(f"| {reader} | {stat['n']} | {_fmt(stat, 'rho')} | {stat['mean_delta']} |")

    lines += ["", "## Descriptive: mean region shares and evidence lift per reader and target", "",
              "| reader | target | instruction | evidence | distractor | question | lift | top-k evidence |",
              "|---|---|---|---|---|---|---|---|"]
    for reader in readers:
        for target in TARGETS:
            rows = [outputs[reader][target][q][variant] for q in qids if q in outputs[reader][target]]
            if not rows:
                continue
            avg = {name: mean(r["shares"][name] for r in rows) for name in ("instruction", "evidence", "distractor", "question")}
            lift = mean(r["lift"] for r in rows if r["lift"] is not None)
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

    lines += ["", f"Top-k label: {k_label}. Verdict rule: SHOWN if the lower bound is above 0. "
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
    parser.add_argument("--confirm", action="store_true", help="allow folds outside the development set")
    args = parser.parse_args(argv)
    folds = {int(f) for f in args.folds.split(",")}
    if not folds <= DEV_FOLDS and not args.confirm:
        print(f"folds {sorted(folds - DEV_FOLDS)} are confirmation folds; pass --confirm after the pre-registration is committed")
        return 2
    markdown, numbers = build_report((args.small, args.large), folds, args.variant, "k50", Path(args.root), args.samples)
    out = Path(args.out or f"runs/exp33_report_{'dev' if folds <= DEV_FOLDS else 'confirm'}.md")
    out.write_text(markdown)
    out.with_suffix(".json").write_text(json.dumps(numbers, indent=1))
    print(markdown)
    return 0


if __name__ == "__main__":
    sys.exit(main())
