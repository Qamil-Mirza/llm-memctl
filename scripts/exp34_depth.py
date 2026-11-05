"""§34 exploratory analysis 1 (not pre-registered): does the depth at which the gold emerges predict the outcome?

    .venv/bin/python scripts/exp34_depth.py --folds 0,1 [--only-qids FILE] [--out runs/exp34_depth_dev.md]

From the existing read files only. For each reader and lens: emergence depth (first layer with gold rank <= k, over
n_layers) by outcome class (success / refusal / answered wrong); share emerged; AUC of depth for wrong-vs-right with
never-emerged coded as depth 1.05 (later than the last layer); and, per refusal, the order of two events along the
layers: the gold entering the top-k and ` unknown` reaching top-1. Also the final-layer rank of the gold and of
` unknown` on refusals, which the main report does not give.
"""
import argparse
import sys
from pathlib import Path
from statistics import mean, median

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from memctl.analysis.jacobian_stats import bootstrap_auc, bootstrap_mean_difference, bootstrap_proportion  # noqa: E402
from scripts.exp34_report import DEV_FOLDS, _paired_outputs, _quartiles, classes_of, gold  # noqa: E402

NEVER = 1.05
POS_P1 = 0   # position_index order in the npz follows POSITION_NAMES with P1 first


def _layers_and_ranks(npz_path: Path, variant: str):
    with np.load(npz_path) as z:
        return z["layers"], z[f"ranks_{variant}"]


def refusal_order(npz_path: Path, variant: str, k: int, n_layers: int) -> dict:
    layers, ranks = _layers_and_ranks(npz_path, variant)
    g = ranks[:, POS_P1, 0]          # gold
    u = ranks[:, POS_P1, 1]          # unknown_sp
    g_in = [int(l) for l, r in zip(layers, g) if r <= k]
    u_top = [int(l) for l, r in zip(layers, u) if r == 1]
    gl = g_in[0] if g_in else None
    ul = u_top[0] if u_top else None
    if gl is None and ul is None:
        order = "neither (gold never top-k, unknown never top-1)"
    elif gl is None:
        order = "gold never enters top-k; unknown reaches top-1"
    elif ul is None:
        order = "gold enters top-k; unknown never top-1"
    elif ul < gl:
        order = "unknown top-1 BEFORE gold enters top-k"
    elif ul == gl:
        order = "same layer"
    else:
        order = "gold enters top-k BEFORE unknown top-1"
    return {"order": order, "gold_final": int(g[-1]), "unknown_final": int(u[-1]),
            "gold_layer": gl, "unknown_layer": ul, "n_layers": n_layers}


def depth_of(entry: dict, variant: str) -> float | None:
    return gold(entry, variant)["emergence_depth"]


def section(reader, outputs, by_q, qids, root, variant, samples, k) -> list[str]:
    present = [q for q in qids if q in outputs[reader]]
    classes = classes_of(by_q[reader], present)
    lines = [f"### {reader}, {variant}", "",
             "| class | n | emerged | share emerged (95% CI) | depth mean | q1 | median | q3 |", "|---|---|---|---|---|---|---|---|"]
    depths = {}
    for cls, members in classes.items():
        d = [depth_of(outputs[reader][q], variant) for q in members]
        em = [x for x in d if x is not None]
        depths[cls] = (d, em)
        share = bootstrap_proportion([x is not None for x in d], samples=samples) if d else None
        qs = _quartiles(em) if em else {}
        lines.append(f"| {cls} | {len(d)} | {len(em)} | {share['p']:.3f} ({share['ci_low']:.3f}, {share['ci_high']:.3f}) | "
                     f"{mean(em):.3f} | {qs.get('q1', float('nan')):.3f} | {qs.get('median', float('nan')):.3f} | {qs.get('q3', float('nan')):.3f} |"
                     if em else f"| {cls} | {len(d)} | 0 | - | - | - | - | - |")
    lines.append("")
    # failures vs successes: depth among emerged, and with never-emerged coded as NEVER
    fail_d, fail_em = depths["all failures"]; succ_d, succ_em = depths["success"]
    if fail_em and succ_em:
        diff = bootstrap_mean_difference(fail_em, succ_em, samples=samples)
        lines.append(f"Depth, failures minus successes, among emerged: {diff['mean']:+.3f} ({diff['ci_low']:+.3f}, {diff['ci_high']:+.3f}) "
                     f"(n {len(fail_em)} vs {len(succ_em)}).")
        coded_f = [x if x is not None else NEVER for x in fail_d]
        coded_s = [x if x is not None else NEVER for x in succ_d]
        auc = bootstrap_auc(coded_f, coded_s, samples=samples)
        lines.append(f"AUC(depth, wrong vs right; never-emerged coded {NEVER}): {auc['auc']:.3f} ({auc['ci_low']:.3f}, {auc['ci_high']:.3f}) "
                     f"(n {len(coded_f)} vs {len(coded_s)}); 0.5 = depth says nothing about the outcome.")
        for cls in ("refusal", "answered wrong"):
            cd, cem = depths[cls]
            if cem:
                diff = bootstrap_mean_difference(cem, succ_em, samples=samples)
                lines.append(f"Depth, {cls} minus successes, among emerged: {diff['mean']:+.3f} ({diff['ci_low']:+.3f}, {diff['ci_high']:+.3f}).")
    lines.append("")
    # refusal ordering
    refusals = classes["refusal"]
    if refusals:
        n_layers = len(outputs[reader][refusals[0]]["measures"][variant]["layers"])
        orders = {}
        gold_final, unk_final = [], []
        for q in refusals:
            r = refusal_order(root / reader / f"{q}.npz", variant, k, n_layers)
            orders[r["order"]] = orders.get(r["order"], 0) + 1
            gold_final.append(r["gold_final"]); unk_final.append(r["unknown_final"])
        lines += [f"Refusals ({len(refusals)}), order of events along the layers at P1:", "",
                  "| order | n | share |", "|---|---|---|"]
        for name, n in sorted(orders.items(), key=lambda x: -x[1]):
            lines.append(f"| {name} | {n} | {n / len(refusals):.3f} |")
        lines += ["", f"Final-layer rank on refusals: gold median {median(gold_final):.0f} (q1 {np.percentile(gold_final, 25):.0f}, "
                  f"q3 {np.percentile(gold_final, 75):.0f}); ` unknown` median {median(unk_final):.0f} "
                  f"(q1 {np.percentile(unk_final, 25):.0f}, q3 {np.percentile(unk_final, 75):.0f}).", ""]
    return lines


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", default="0,1")
    parser.add_argument("--root", default="runs/exp34_jlens")
    parser.add_argument("--only-qids", default=None)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    folds = {int(f) for f in args.folds.split(",")}
    root = Path(args.root)
    only = set(Path(args.only_qids).read_text().split()) if args.only_qids else None
    readers = ("qwen3b", "qwen7b")
    records_by_reader, _, qids, outputs, both = _paired_outputs(readers, folds, root, None, only)
    by_q = {r: {x.question_id: x for x in rows} for r, rows in records_by_reader.items()}
    label = "development folds" if folds <= DEV_FOLDS else "ALL FOLDS (confirmation folds included; exploratory, after the pre-registered analysis)"
    subset = f"; subset {Path(args.only_qids).stem}" if only else ""
    lines = [f"# §34 exploratory analysis 1: emergence depth vs outcome ({', '.join(map(str, sorted(folds)))}: {label}; "
             f"{len(both)} paired questions{subset}; k = {args.k}; NOT pre-registered)", "",
             "Depth = first layer at which the gold's first token is inside the top-k at P1, divided by the number of layers. "
             "Labels are the §23 single-judge verdicts. Depth is defined only on questions where the gold emerges; the AUC line "
             f"codes never-emerged as {NEVER}.", ""]
    for reader in readers:
        for variant in ("jlens", "logit"):
            lines += section(reader, outputs, by_q, both, root, variant, args.samples, args.k)
    text = "\n".join(lines) + "\n"
    out = Path(args.out or f"runs/exp34_depth_{'dev' if folds <= DEV_FOLDS else 'allfolds'}{'_' + Path(args.only_qids).stem if only else ''}.md")
    out.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
