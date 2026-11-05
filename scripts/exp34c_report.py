"""§34c report: the forced-position readout with the evidence lines cut out (ablated) against the intact §34a reads.

    .venv/bin/python scripts/exp34c_report.py --folds 0,1 [--confirm] [--out runs/exp34c_report_dev.md]

Pre-registration: research/proposals/2026-10-13-exp34c-evidence-ablation.md. Claims (3B on its deciding logit row;
7B descriptive): A, share where the gold ever enters the top-k, intact − ablated, SHOWN if the lower bound is above 0;
B, share where the gold is in the final-layer top-k (outcome C, model logits), intact − ablated, SHOWN likewise.
Plausibility share: among questions C with evidence, the share still C without it, by class and gold-token kind.
Reads pair by question id; each ablated read must carry the intact read's prompt_sha256 as intact_prompt_sha256.
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from memctl.analysis.jacobian_stats import bootstrap_proportion, paired_claim  # noqa: E402
from scripts.exp34_gold_kinds import kind as gold_kind  # noqa: E402
from scripts.exp34_report import DEV_FOLDS, _paired_outputs, classes_of, gold, load_outputs  # noqa: E402

READERS = ("qwen3b", "qwen7b")
DECIDES = {"qwen3b": "logit", "qwen7b": "jlens"}


def _fmt(stat: dict) -> str:
    return f"{stat['mean']:+.3f} ({stat['ci_low']:+.3f}, {stat['ci_high']:+.3f})"


def _prop(flags: list[bool], samples: int) -> str:
    if not flags:
        return "-"
    s = bootstrap_proportion(flags, samples=samples)
    return f"{s['p']:.3f} ({s['ci_low']:.3f}, {s['ci_high']:.3f})"


def label(reader: str, variant: str) -> str:
    base = "J-lens" if variant == "jlens" else "logit"
    if variant == DECIDES[reader]:
        return base + (" (gate fallback), **decides**" if reader == "qwen3b" else ", **decides**")
    return base + " (beside)"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", default="0,1")
    parser.add_argument("--intact-root", default="runs/exp34_jlens")
    parser.add_argument("--ablated-root", default="runs/exp34_jlens_ablated")
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    folds = {int(f) for f in args.folds.split(",")}
    if not folds <= DEV_FOLDS and not args.confirm:
        print(f"folds {sorted(folds - DEV_FOLDS)} are confirmation folds; pass --confirm after the pre-registration is committed")
        return 2
    rec, _, qids, intact, both = _paired_outputs(READERS, folds, Path(args.intact_root), None)
    ablated = {r: load_outputs(Path(args.ablated_root) / r, folds) for r in READERS}
    both = [q for q in both if all(q in ablated[r] for r in READERS)]
    bad = [q for q in both for r in READERS if ablated[r][q].get("intact_prompt_sha256") != intact[r][q]["prompt_sha256"]]
    if bad:
        print(f"REFUSED: {len(bad)} ablated reads do not carry their intact read's prompt hash"); return 4
    by_q = {r: {x.question_id: x for x in rows} for r, rows in rec.items()}
    kinds = {q: gold_kind(intact[READERS[0]][q]["gold_token_text"]) for q in both}
    lines = [f"# §34c report: evidence cut out vs intact, forced position (folds: {', '.join(map(str, sorted(folds)))}; "
             f"{len(both)} questions paired; k = {args.k})", "",
             "Ablated = every evidence line removed from the prompt by its exact span; everything else identical. Lenses, tracked "
             "tokens, labels and population as §34. Outcome C uses the final layer (model logits), so Claim B is lens-free.", ""]
    numbers = {"folds": sorted(folds), "paired": len(both), "claims": {}, "plausibility": {}}
    for reader in READERS:
        lines += [f"## {reader}", "",
                  "| lens | n | ever in top-k: intact | ablated | Claim A intact − ablated (95% CI) | verdict | final-layer top-k (C): intact | ablated | "
                  "Claim B intact − ablated (95% CI) | verdict | log10 best rank intact − ablated (evidence lift) |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for variant in ("jlens", "logit"):
            ever_i = {q: 1.0 if gold(intact[reader][q], variant)["emergence_depth"] is not None else 0.0 for q in both}
            ever_a = {q: 1.0 if gold(ablated[reader][q], variant)["emergence_depth"] is not None else 0.0 for q in both}
            c_i = {q: 1.0 if gold(intact[reader][q], variant)["outcome"] == "C" else 0.0 for q in both}
            c_a = {q: 1.0 if gold(ablated[reader][q], variant)["outcome"] == "C" else 0.0 for q in both}
            a = paired_claim(ever_i, ever_a, samples=args.samples)
            b = paired_claim(c_i, c_a, samples=args.samples)
            lift = paired_claim({q: math.log10(gold(ablated[reader][q], variant)["best_rank"]) for q in both},
                                {q: math.log10(gold(intact[reader][q], variant)["best_rank"]) for q in both}, samples=args.samples)
            decides = variant == DECIDES[reader] and reader == "qwen3b"
            va = a["verdict"] if decides else "descriptive"
            vb = b["verdict"] if decides else "descriptive"
            numbers["claims"][f"{reader}/{variant}"] = {"A": a, "B": b, "lift": lift}
            lines.append(f"| {label(reader, variant)} | {len(both)} | {sum(ever_i.values()) / len(both):.3f} | {sum(ever_a.values()) / len(both):.3f} | "
                         f"{_fmt(a)} | {va} | {sum(c_i.values()) / len(both):.3f} | {sum(c_a.values()) / len(both):.3f} | {_fmt(b)} | {vb} | {_fmt(lift)} |")
        lines += ["", "Plausibility share: among questions C (final-layer top-k) WITH the evidence, the share still C WITHOUT it; "
                  "and among questions where the gold ever emerges with the evidence, the share where it still emerges without it.", "",
                  "| lens | class | n C with evidence | still C without (95% CI) | n ever-emerges with | still emerges without (95% CI) |",
                  "|---|---|---|---|---|---|"]
        for variant in ("jlens", "logit"):
            groups = dict(classes_of(by_q[reader], both))
            groups["all"] = list(both)
            groups["strict gold token"] = [q for q in both if kinds[q] == "strict"]
            groups["function-word gold token"] = [q for q in both if kinds[q] == "function"]
            for cls, members in groups.items():
                c_with = [q for q in members if gold(intact[reader][q], variant)["outcome"] == "C"]
                still_c = [gold(ablated[reader][q], variant)["outcome"] == "C" for q in c_with]
                e_with = [q for q in members if gold(intact[reader][q], variant)["emergence_depth"] is not None]
                still_e = [gold(ablated[reader][q], variant)["emergence_depth"] is not None for q in e_with]
                numbers["plausibility"][f"{reader}/{variant}/{cls}"] = {"n_c": len(c_with), "still_c": sum(still_c), "n_e": len(e_with), "still_e": sum(still_e)}
                lines.append(f"| {label(reader, variant).split(',')[0].replace(' (gate fallback)', '').replace(' (beside)', '')} | {cls} | {len(c_with)} | "
                             f"{_prop(still_c, args.samples)} | {len(e_with)} | {_prop(still_e, args.samples)} |")
        lines.append("")
        # unknown trajectory without evidence
        for variant in ("jlens", "logit"):
            top1_i = [intact[reader][q]["measures"][variant]["P1"]["unknown_sp"]["first_top1_depth"] is not None for q in both]
            top1_a = [ablated[reader][q]["measures"][variant]["P1"]["unknown_sp"]["first_top1_depth"] is not None for q in both]
            lines.append(f"` unknown` reaches top-1 at P1 ({variant}): with evidence {sum(top1_i)} of {len(both)} ({sum(top1_i) / len(both):.3f}); "
                         f"without {sum(top1_a)} ({sum(top1_a) / len(both):.3f}).")
        lines.append("")
    lines.append("Verdict rules: Claims A and B SHOWN if the lower bound of (intact − ablated) is above 0; pre-registered for qwen3b on its deciding "
                 "(logit) row. Labels are the §23 single-judge verdicts.")
    text = "\n".join(lines) + "\n"
    out = Path(args.out or f"runs/exp34c_report_{'dev' if folds <= DEV_FOLDS else 'confirm'}.md")
    out.write_text(text); out.with_suffix(".json").write_text(json.dumps(numbers, indent=1, default=float))
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
