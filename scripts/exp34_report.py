"""§34 report: lens gate, pre-registered claims, A/B/C decomposition and descriptives from runs/exp34_jlens/<reader>/.

    .venv/bin/python scripts/exp34_report.py --small qwen3b --large qwen7b --folds 0,1 --out runs/exp34_report_dev.md

Folds default to the development folds 0,1. Confirmation folds (2,3,4) are read only after the
pre-registration is committed; the script refuses them unless --confirm is passed (exit 2, nothing written).
Questions are paired across readers only when both readers' <qid>.json carry the same prompt_sha256;
otherwise the report stops (exit 4, nothing written) and lists the mismatched or missing hashes.

Spec §3: if the lens gate fails for a reader, that reader's claims are also reported with the logit lens
(rows marked `logit (gate fallback)`); the J-lens rows stay in the report."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import mean, median, quantiles

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memctl.analysis.jacobian_stats import (bootstrap_auc, bootstrap_mean_difference,  # noqa: E402
                                            bootstrap_proportion, paired_claim)
from memctl.jacobian_data import PromptRecord, load_records  # noqa: E402

DEV_FOLDS = {0, 1}
VARIANTS = ("jlens", "logit")
TRACKED = ("gold", "unknown_sp", "unknown")
OUTCOMES = ("A", "B", "C")
AUC_BAR = 0.5  # Claim 2: SHOWN when the AUC interval lies above chance


class PromptIdentityError(ValueError):
    """The two readers' outputs for a question do not carry the same prompt hash."""


# --- loading ---------------------------------------------------------------------------------------

def load_outputs(out_dir: Path, folds: set[int]) -> dict[str, dict]:
    """question_id -> the <qid>.json entry, for the chosen folds (underscore files are bookkeeping)."""
    found = {}
    for path in sorted(Path(out_dir).glob("*.json")):
        if path.name.startswith("_"):
            continue
        entry = json.loads(path.read_text())
        if "measures" in entry and entry["fold"] in folds:
            found[entry["question_id"]] = entry
    return found


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def prompt_mismatches(qids: list[str], small: dict[str, dict], large: dict[str, dict]) -> list[str]:
    """One line per question present for both readers whose prompt hashes differ or are missing."""
    problems = []
    for q in qids:
        if q not in small or q not in large:
            continue
        a, b = small[q].get("prompt_sha256"), large[q].get("prompt_sha256")
        if a is None or b is None:
            which = " and ".join(name for name, h in (("small", a), ("large", b)) if h is None)
            problems.append(f"{q}: prompt_sha256 missing from the {which} reader's json")
        elif a != b:
            problems.append(f"{q}: prompt_sha256 differs ({a[:12]} vs {b[:12]})")
    return problems


def population(records_by_reader: dict[str, list[PromptRecord]]) -> list[str]:
    sets = [{r.question_id for r in records if r.answerable and r.all_in_view and r.evidence_lines}
            for records in records_by_reader.values()]
    return sorted(set.intersection(*sets)) if sets else []


# --- measures --------------------------------------------------------------------------------------

def layer_band(layers) -> tuple[int, int]:
    """The middle third of depth, [n//3, 2n//3) in layer units, with n = last layer + 1."""
    n = int(np.asarray(layers)[-1]) + 1
    return n // 3, 2 * n // 3


def band_best_logrank(npz_path: Path, variant: str, position_idx: int, tracked_idx: int, band: tuple[int, int]) -> float:
    """log10 of the best (lowest) rank over the layers inside the band."""
    with np.load(npz_path) as z:
        layers = z["layers"]
        mask = (layers >= band[0]) & (layers < band[1])
        return math.log10(int(z[f"ranks_{variant}"][mask, position_idx, tracked_idx].min()))


def gold(entry: dict, variant: str, position: str = "P1") -> dict:
    return entry["measures"][variant][position]["gold"]


def best_logrank(entry: dict, variant: str, position: str = "P1") -> float:
    return math.log10(gold(entry, variant, position)["best_rank"])


def _fmt(stat: dict, key: str = "mean") -> str:
    if stat.get("ci_low") is None:
        return f"{stat.get(key)}"
    return f"{stat[key]:+.3f} ({stat['ci_low']:+.3f}, {stat['ci_high']:+.3f})"


def _prop(stat: dict) -> str:
    if stat["p"] is None:
        return "-"
    return f"{stat['p']:.3f} ({stat['ci_low']:.3f}, {stat['ci_high']:.3f})"


def _num(value, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


# --- sections --------------------------------------------------------------------------------------

def gate_for(reader: str, outputs: dict[str, dict], qids: list[str], root: Path, samples: int) -> dict:
    jl, lg, band, missing = {}, {}, None, 0
    for q in qids:
        npz = root / reader / f"{q}.npz"
        if q not in outputs or not npz.exists():
            missing += q in outputs
            continue
        with np.load(npz) as z:
            band = layer_band(z["layers"])
        jl[q] = band_best_logrank(npz, "jlens", 0, 0, band)
        lg[q] = band_best_logrank(npz, "logit", 0, 0, band)
    a = paired_claim(lg, jl, samples=samples)
    a["verdict"] = "PASS" if a["ci_low"] is not None and a["ci_low"] > 0 else "FAIL"
    a["n_missing_npz"] = missing
    ev = read_json(root / "eval" / f"{reader}.json")
    if ev is None:
        b = {"verdict": "MISSING"}
    else:
        m = ev["mean_pass_at_k"]
        b = {"sets": ev["sets"], "mean_pass_at_k": m, "mean_pass_at_k_all": ev.get("mean_pass_at_k_all"),
             "band": ev.get("band"), "verdict": "PASS" if m["jlens"] >= m["logit"] else "FAIL"}
    side = read_json(root / "lenses" / f"{reader}.json") or {}
    return {"a": a, "b": b, "band": list(band) if band else None, "n_prompts": side.get("n_prompts"),
            "below_minimum": side.get("below_minimum"),
            "verdict": "PASS" if a["verdict"] == "PASS" and b["verdict"] == "PASS" else "FAIL"}


def gate_section(readers, outputs, qids, root, samples, numbers) -> list[str]:
    lines = ["## Lens gate (spec §3): J-lens must beat the logit lens", "",
             "(a) gold at P1, log10(best rank over the middle-third band), logit lens minus J-lens, paired; PASS if the "
             "lower bound is above 0. (b) lens-eval pass@k over the band, mean over sets; PASS if J-lens >= logit.", "",
             "| reader | lens prompts | below minimum | band | n | (a) logit minus J-lens (95% CI) | (a) | "
             "(b) mean pass@k J-lens / logit | (b) all layers J-lens / logit | (b) | gate |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for reader in readers:
        g = gate_for(reader, outputs[reader], qids, root, samples)
        numbers["gate"][reader] = g
        b = g["b"]
        mb = b.get("mean_pass_at_k") or {}
        ma = b.get("mean_pass_at_k_all") or {}
        lines.append(f"| {reader} | {g['n_prompts']} | {g['below_minimum']} | {g['band']} | {g['a']['n']} | {_fmt(g['a'])} | "
                     f"{g['a']['verdict']} | {_num(mb.get('jlens'))} / {_num(mb.get('logit'))} | "
                     f"{_num(ma.get('jlens'))} / {_num(ma.get('logit'))} | {b['verdict']} | **{g['verdict']}** |")
    lines += ["", "| reader | eval set | items | skipped | pass@k J-lens | pass@k logit | all layers J-lens | all layers logit |",
              "|---|---|---|---|---|---|---|---|"]
    for reader in readers:
        for name, s in (numbers["gate"][reader]["b"].get("sets") or {}).items():
            pa, pl = s["pass_at_k"], s.get("pass_at_k_all") or {}
            lines.append(f"| {reader} | {name} | {s['n_items']} | {s['n_skipped']} | {_num(pa['jlens'])} | {_num(pa['logit'])} | "
                         f"{_num(pl.get('jlens'))} | {_num(pl.get('logit'))} |")
    return lines + [""]


def _label(variant: str) -> str:
    return "J-lens" if variant == "jlens" else "logit (gate fallback)"


def claim1_section(small, large, outputs, qids, samples, variants, numbers) -> list[str]:
    lines = [f"## Claim 1: at P1, {small} minus {large} (positive = {large} better), paired by question", "",
             "| measure | lens | n | mean difference (95% CI) | verdict |", "|---|---|---|---|---|"]
    both = [q for q in qids if q in outputs[small] and q in outputs[large]]
    for variant in variants:
        suffix = "" if variant == "jlens" else "_logit"
        for key, value in (("best_logrank", best_logrank),
                           ("outcome_A", lambda e, v: 1.0 if gold(e, v)["outcome"] == "A" else 0.0)):
            a = {q: value(outputs[small][q], variant) for q in both}
            b = {q: value(outputs[large][q], variant) for q in both}
            stat = paired_claim(a, b, samples=samples)
            numbers["claim1"][key + suffix] = stat
            name = "log10 best rank of gold" if key == "best_logrank" else "share of outcome A (secondary)"
            lines.append(f"| {name} | {_label(variant)} | {stat['n']} | {_fmt(stat)} | {stat['verdict']} |")
    return lines + [""]


def claim2_for(entries: dict[str, dict], by_q: dict[str, PromptRecord], qids, variant, samples) -> dict:
    right = [best_logrank(entries[q], variant) for q in qids if q in entries and by_q[q].correct is True]
    wrong = [best_logrank(entries[q], variant) for q in qids if q in entries and by_q[q].correct is False]
    if right and wrong:
        stat = bootstrap_auc(wrong, right, samples=samples)
    else:
        stat = {"n_pos": len(wrong), "n_neg": len(right), "auc": None, "ci_low": None, "ci_high": None}
    stat["difference"] = bootstrap_mean_difference(wrong, right, samples=samples)
    stat["mean_wrong"] = mean(wrong) if wrong else None
    stat["mean_right"] = mean(right) if right else None
    stat["verdict"] = "SHOWN" if stat["ci_low"] is not None and stat["ci_low"] > AUC_BAR else "NOT SHOWN"
    return stat


def claim2_section(readers, outputs, by_q, qids, samples, reader_variants, numbers) -> list[str]:
    lines = ["## Claim 2: within each reader, log10 best rank at P1 on wrong vs right answers (AUC > 0.5: failures rank worse)", "",
             "| reader | lens | n wrong | n right | AUC (95% CI) | mean wrong | mean right | wrong minus right (95% CI) | verdict |",
             "|---|---|---|---|---|---|---|---|---|"]
    for reader in readers:
        for variant in reader_variants[reader]:
            stat = claim2_for(outputs[reader], by_q[reader], qids, variant, samples)
            numbers["claim2"][reader if variant == "jlens" else f"{reader}/logit"] = stat
            lines.append(f"| {reader} | {_label(variant)} | {stat['n_pos']} | {stat['n_neg']} | {_fmt(stat, 'auc')} | "
                         f"{_num(stat['mean_wrong'])} | {_num(stat['mean_right'])} | {_fmt(stat['difference'])} | {stat['verdict']} |")
    return lines + [""]


def classes_of(by_q: dict[str, PromptRecord], qids: list[str]) -> dict[str, list[str]]:
    failures = [q for q in qids if by_q[q].correct is False]
    return {"refusal": [q for q in failures if by_q[q].unknown is True],
            "answered wrong": [q for q in failures if by_q[q].unknown is not True],
            "all failures": failures,
            "success": [q for q in qids if by_q[q].correct is True]}


def decomposition_section(readers, outputs, by_q, qids, samples, reader_variants, numbers) -> list[str]:
    lines = ["## Decomposition: share of outcomes A (never present), B (present then lost), C (present to the end) at P1", "",
             "| reader | lens | class | n | A (95% CI) | B (95% CI) | C (95% CI) |", "|---|---|---|---|---|---|---|"]
    for reader in readers:
        present = [q for q in qids if q in outputs[reader]]
        for variant in reader_variants[reader]:
            table = {}
            for cls, members in classes_of(by_q[reader], present).items():
                found = [gold(outputs[reader][q], variant)["outcome"] for q in members]
                if found:
                    table[cls] = {o: bootstrap_proportion([x == o for x in found], samples=samples) for o in OUTCOMES}
                cells = [_prop(table[cls][o]) if found else "-" for o in OUTCOMES]
                lines.append(f"| {reader} | {_label(variant)} | {cls} | {len(found)} | " + " | ".join(cells) + " |")
            numbers["decomposition"][reader if variant == "jlens" else f"{reader}/logit"] = table
    return lines + ["", "Refusal = the §23 judge marked the answer `unknown`; answered wrong = any other incorrect answer.", ""]


def _quartiles(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean": None, "q1": None, "median": None, "q3": None}
    q1, q2, q3 = quantiles(values, n=4, method="inclusive") if len(values) > 1 else (values[0],) * 3
    return {"n": len(values), "mean": mean(values), "q1": q1, "median": q2, "q3": q3}


def position_best(entry: dict, position: str) -> int:
    m = entry["measures"]["jlens"][position]
    return min(p["gold"]["best_rank"] for p in m) if isinstance(m, list) else m["gold"]["best_rank"]


def descriptive_depth_positions(readers, outputs, qids, desc) -> list[str]:
    lines = ["## Descriptive: emergence depth of the gold at P1 (J-lens; questions where it emerges)", "",
             "| reader | n emerged | n read | mean | q1 | median | q3 |", "|---|---|---|---|---|---|---|"]
    for reader in readers:
        present = [q for q in qids if q in outputs[reader]]
        depths = [gold(outputs[reader][q], "jlens")["emergence_depth"] for q in present]
        stat = _quartiles([d for d in depths if d is not None])
        desc["emergence"][reader] = {**stat, "n_read": len(present)}
        lines.append(f"| {reader} | {stat['n']} | {len(present)} | {_num(stat['mean'])} | {_num(stat['q1'])} | "
                     f"{_num(stat['median'])} | {_num(stat['q3'])} |")
    lines += ["", "## Descriptive: gold best rank by position (J-lens; E = best over the evidence lines)", "",
              "| reader | position | n | mean log10 best rank | median best rank | share best rank <= k |",
              "|---|---|---|---|---|---|"]
    for reader in readers:
        present = [q for q in qids if q in outputs[reader]]
        for position in ("P1", "P0", "E", "Q"):
            ranks = [position_best(outputs[reader][q], position) for q in present if outputs[reader][q]["measures"]["jlens"][position]]
            k = outputs[reader][present[0]]["measures"]["jlens"]["k"] if present else None
            row = {"n": len(ranks), "mean_logrank": mean(math.log10(r) for r in ranks) if ranks else None,
                   "median_rank": median(ranks) if ranks else None,
                   "share_topk": mean(r <= k for r in ranks) if ranks else None}
            desc["positions"][f"{reader}/{position}"] = row
            lines.append(f"| {reader} | {position} | {row['n']} | {_num(row['mean_logrank'])} | {row['median_rank']} | "
                         f"{_num(row['share_topk'])} |")
    return lines + [""]


def descriptive_unknown_types(readers, outputs, by_q, qids, desc) -> list[str]:
    lines = ["## Descriptive: the `unknown` trajectory at P1 on refusals (J-lens)", "",
             "| reader | token | refusals | reaches top-1 | mean first top-1 depth |", "|---|---|---|---|---|"]
    for reader in readers:
        refusals = [q for q in qids if q in outputs[reader] and by_q[reader][q].unknown is True]
        for token in ("unknown_sp", "unknown"):
            depths = [outputs[reader][q]["measures"]["jlens"]["P1"][token]["first_top1_depth"] for q in refusals]
            hit = [d for d in depths if d is not None]
            row = {"n": len(refusals), "n_top1": len(hit), "mean_depth": mean(hit) if hit else None}
            desc["unknown_on_refusals"][f"{reader}/{token}"] = row
            lines.append(f"| {reader} | {token} | {row['n']} | {row['n_top1']} | {_num(row['mean_depth'])} |")
    lines += ["", "## Descriptive: outcome shares by question type (J-lens, P1, all in-view questions)", "",
              "| reader | question type | n | A | B | C |", "|---|---|---|---|---|---|"]
    for reader in readers:
        desc["question_type"][reader] = {}
        present = [q for q in qids if q in outputs[reader]]
        for qtype in sorted({by_q[reader][q].question_type for q in present}):
            found = [gold(outputs[reader][q], "jlens")["outcome"] for q in present if by_q[reader][q].question_type == qtype]
            row = {"n": len(found), **{o: sum(x == o for x in found) / len(found) for o in OUTCOMES}}
            desc["question_type"][reader][qtype] = row
            lines.append(f"| {reader} | {qtype} | {row['n']} | {row['A']:.3f} | {row['B']:.3f} | {row['C']:.3f} |")
    return lines + [""]


def top_jaccard(ours: Path, theirs: Path, k: int = 10) -> list[float]:
    """Per layer, the Jaccard overlap of the two lenses' top-k token sets at P1."""
    with np.load(ours) as a, np.load(theirs) as b:
        x, y = a["top_ids_jlens"][:, 0, :k], b["top_ids_jlens"][:, 0, :k]
    return [len(set(p) & set(q)) / len(set(p) | set(q)) for p, q in zip(x.tolist(), y.tolist())]


def community_section(small, large, outputs, qids, root, community_root, folds, samples, desc) -> list[str]:
    theirs = load_outputs(community_root / large, folds)
    problems = prompt_mismatches(qids, outputs[small], theirs)
    if problems:
        raise PromptIdentityError(f"{len(problems)} question(s): the community-lens outputs do not match {small}:\n  "
                                  + "\n  ".join(problems))
    both = [q for q in qids if q in outputs[large] and q in theirs
            and (root / large / f"{q}.npz").exists() and (community_root / large / f"{q}.npz").exists()]
    per_layer = [top_jaccard(root / large / f"{q}.npz", community_root / large / f"{q}.npz") for q in both]
    bands = {}
    if per_layer:
        lo, hi = layer_band(outputs[large][both[0]]["measures"]["jlens"]["layers"])
        n = len(per_layer[0])
        for name, (s, e) in (("early", (0, lo)), ("middle", (lo, hi)), ("late", (hi, n))):
            bands[name] = mean(mean(row[s:e]) for row in per_layer) if e > s else None
    pair = [q for q in qids if q in outputs[small] and q in theirs]
    c1 = paired_claim({q: best_logrank(outputs[small][q], "jlens") for q in pair},
                      {q: best_logrank(theirs[q], "jlens") for q in pair}, samples=samples)
    desc["community"] = {"n": len(both), "jaccard_top10": bands, "claim1_best_logrank": c1}
    return ["## Descriptive: cross-check against the community lens for " + large, "",
            "| n | Jaccard top-10 at P1, early | middle | late | Claim 1 with the community lens (95% CI) | verdict |",
            "|---|---|---|---|---|---|",
            f"| {len(both)} | {_num(bands.get('early'))} | {_num(bands.get('middle'))} | {_num(bands.get('late'))} | "
            f"{_fmt(c1)} | {c1['verdict']} |", ""]


def bookkeeping_section(readers, dropped, records_by_reader, outputs, root, qids, records_from) -> list[str]:
    lines = ["## Dropped and failed" + (f" (records from {records_from})" if records_from else ""), "",
             "| reader | fold | no cached prompt | evidence not found |", "|---|---|---|---|"]
    for reader in readers:
        for fold, counts in dropped[reader].items():
            lines.append(f"| {reader} | {fold} | {counts.get('no_cached_prompt')} | {counts.get('evidence_not_found')} |")
    lines += ["", "| reader | records in folds | failed reads (_failed.json) | in-view questions without an output |",
              "|---|---|---|---|"]
    for reader in readers:
        ids = {r.question_id for r in records_by_reader[reader]}
        failed = read_json(root / reader / "_failed.json") or {}
        lines.append(f"| {reader} | {len(ids)} | {len(set(failed) & ids)} | {sum(q not in outputs[reader] for q in qids)} |")
    return lines + [""]


def build_report(readers: tuple[str, str], folds: set[int], k_label: str, root: Path, samples: int = 10000,
                 records_from: str | None = None, community_root: Path | None = None) -> tuple[str, dict]:
    small, large = readers
    root = Path(root)
    records_by_reader, dropped = {}, {}
    for reader in readers:
        rows, dropped[reader] = [], {}
        for fold in sorted(folds):
            fold_rows, dropped[reader][str(fold)] = load_records(records_from or reader, fold)
            rows.extend(fold_rows)
        records_by_reader[reader] = rows
    qids = population(records_by_reader)
    by_q = {reader: {r.question_id: r for r in rows} for reader, rows in records_by_reader.items()}
    outputs = {reader: load_outputs(root / reader, folds) for reader in readers}
    problems = prompt_mismatches(qids, outputs[small], outputs[large])
    if problems:
        raise PromptIdentityError(f"{len(problems)} question(s) cannot be paired: the {small} and {large} outputs "
                                  "do not carry the same prompt_sha256:\n  " + "\n  ".join(problems))
    paired = sum(q in outputs[small] and q in outputs[large] for q in qids)
    numbers: dict = {"folds": sorted(folds), "k_label": k_label, "population": len(qids), "paired": paired,
                     "dropped": dropped, "gate": {}, "claim1": {}, "claim2": {}, "decomposition": {},
                     "descriptive": {"emergence": {}, "positions": {}, "unknown_on_refusals": {}, "question_type": {}}}
    lines = [f"# §34 J-lens report ({small} vs {large}; folds: {', '.join(map(str, sorted(folds)))}; "
             f"population {len(qids)} in-view answerable questions; {paired} paired)", "",
             f"Prompt identity: all {paired} questions with outputs for both readers carry the same prompt_sha256.", ""]
    lines += gate_section(readers, outputs, qids, root, samples, numbers)
    failed_gate = [r for r in readers if numbers["gate"][r]["verdict"] != "PASS"]
    reader_variants = {r: VARIANTS if r in failed_gate else ("jlens",) for r in readers}
    if failed_gate:
        lines += [f"**Gate failed for {', '.join(failed_gate)}**: claims are also shown with the logit lens (spec §3).", ""]
    lines += claim1_section(small, large, outputs, qids, samples, VARIANTS if failed_gate else ("jlens",), numbers)
    lines += claim2_section(readers, outputs, by_q, qids, samples, reader_variants, numbers)
    lines += decomposition_section(readers, outputs, by_q, qids, samples, reader_variants, numbers)
    desc = numbers["descriptive"]
    lines += descriptive_depth_positions(readers, outputs, qids, desc)
    lines += descriptive_unknown_types(readers, outputs, by_q, qids, desc)
    if community_root is not None:
        lines += community_section(small, large, outputs, qids, root, Path(community_root), folds, samples, desc)
    lines += bookkeeping_section(readers, dropped, records_by_reader, outputs, root, qids, records_from)
    lines += [f"Top-k label: {k_label}. Verdict rules: gate (a) PASS if the lower bound is above 0, (b) PASS if J-lens >= logit "
              f"on the mean over sets; Claim 1 SHOWN if the lower bound is above 0; Claim 2 SHOWN if the AUC lower bound is "
              f"above {AUC_BAR}. Correct/unknown labels are the §23 single-judge verdicts (STATE.md caveat 3)."]
    return "\n".join(lines) + "\n", numbers


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--small", default="qwen3b")
    parser.add_argument("--large", default="qwen7b")
    parser.add_argument("--folds", default="0,1")
    parser.add_argument("--root", default="runs/exp34_jlens")
    parser.add_argument("--out", default=None)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--records-from", default=None,
                        help="reader whose §23 records to use for every reader (local development with the 0.5B)")
    parser.add_argument("--community-root", default=None,
                        help="root holding <large>/<qid>.{json,npz} read with a community lens (descriptive cross-check)")
    parser.add_argument("--confirm", action="store_true", help="allow folds outside the development set")
    args = parser.parse_args(argv)
    folds = {int(f) for f in args.folds.split(",")}
    if not folds <= DEV_FOLDS and not args.confirm:
        print(f"folds {sorted(folds - DEV_FOLDS)} are confirmation folds; pass --confirm after the pre-registration is committed")
        return 2
    try:
        markdown, numbers = build_report((args.small, args.large), folds, "k10", Path(args.root), args.samples,
                                         records_from=args.records_from,
                                         community_root=Path(args.community_root) if args.community_root else None)
    except PromptIdentityError as error:
        print(f"REFUSED: {error}")
        return 4
    out = Path(args.out or f"runs/exp34_report_{'dev' if folds <= DEV_FOLDS else 'confirm'}.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(markdown)
    out.with_suffix(".json").write_text(json.dumps(numbers, indent=1))
    print(markdown)
    return 0


if __name__ == "__main__":
    sys.exit(main())
