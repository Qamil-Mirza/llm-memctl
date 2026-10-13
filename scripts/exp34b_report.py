"""§34b report: the J-lens readout at the reader's OWN answer position against the forced position of §34, paired.

    .venv/bin/python scripts/exp34b_report.py --folds 0,1 [--confirm] [--out runs/exp34b_report_dev.md]

Pre-registration: research/proposals/2026-10-13-exp34b-own-position.md. Claims (3B; 7B descriptive):
  A (presence): share of questions where the gold emerges (enters the top-k at some layer), own minus forced, paired;
    SHOWN if the lower bound is above 0.
  B (depth): among questions emerged in both modes, emergence depth forced minus own (positive = own earlier), paired;
    SHOWN if the lower bound is above 0.
Verdict rows follow §34: the 3B's gate failed, so its verdicts are on the logit rows (J-lens rows beside); the 7B's
J-lens rows stand (logit beside). Folds outside {0,1} need --confirm (exit 2).
"""
import argparse
import json
import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from memctl.analysis.jacobian_stats import bootstrap_proportion, paired_claim  # noqa: E402
from scripts.exp34_depth import refusal_order  # noqa: E402
from scripts.exp34_report import DEV_FOLDS, OUTCOMES, _paired_outputs, _quartiles, classes_of, gold  # noqa: E402

READERS = ("qwen3b", "qwen7b")
DECIDES = {"qwen3b": "logit", "qwen7b": "jlens"}      # §34 gate outcome, fixed on the development folds


def _fmt(stat: dict) -> str:
    return f"{stat['mean']:+.3f} ({stat['ci_low']:+.3f}, {stat['ci_high']:+.3f})"


def _label(reader: str, variant: str) -> str:
    base = "J-lens" if variant == "jlens" else "logit"
    if variant == DECIDES[reader]:
        return base + (" (gate fallback), **decides**" if reader == "qwen3b" else ", **decides**")
    return base + (" (shown beside)" if reader == "qwen7b" else "")


def depth(entry: dict, variant: str):
    return gold(entry, variant)["emergence_depth"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", default="0,1")
    parser.add_argument("--forced-root", default="runs/exp34_jlens")
    parser.add_argument("--own-root", default="runs/exp34_jlens_own")
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    folds = {int(f) for f in args.folds.split(",")}
    if not folds <= DEV_FOLDS and not args.confirm:
        print(f"folds {sorted(folds - DEV_FOLDS)} are confirmation folds; pass --confirm after the pre-registration is committed")
        return 2
    froot, oroot = Path(args.forced_root), Path(args.own_root)
    rec_f, _, qids_f, out_f, both_f = _paired_outputs(READERS, folds, froot, None)
    rec_o, _, qids_o, out_o, both_o = _paired_outputs(READERS, folds, oroot, None)
    both = [q for q in both_f if q in set(both_o)]
    by_q = {r: {x.question_id: x for x in rows} for r, rows in rec_f.items()}
    mismatch = [q for q in both if any(out_f[r][q]["prompt_sha256"] != out_o[r][q]["prompt_sha256"] for r in READERS)]
    if mismatch:
        print(f"REFUSED: {len(mismatch)} questions differ in prompt_sha256 between modes"); return 4
    lines = [f"# §34b report: own answer position vs forced position (folds: {', '.join(map(str, sorted(folds)))}; "
             f"{len(both)} questions paired across readers and modes; k = {args.k})", "",
             "Own position = the cached output teacher-forced up to its last `Answer:`; forced = `Answer:` right after the prompt (§34). "
             "Lenses, tracked tokens and labels as in §34. Verdict rows follow the §34 gate (3B on the logit lens, 7B on the J-lens).", ""]
    numbers = {"folds": sorted(folds), "paired": len(both), "claims": {}}
    for reader in READERS:
        markers = sum(1 for q in both if out_o[reader][q].get("answer_marker"))
        lines += [f"## {reader}: {markers} of {len(both)} cached outputs carry an `Answer:` marker; the rest are read at their first token", ""]
        lines += ["| lens | n | emerged forced | emerged own | Claim A: own − forced, share emerged (95% CI) | verdict | n both emerged | "
                  "depth forced (median) | depth own (median) | Claim B: forced − own depth (95% CI) | verdict |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for variant in ("jlens", "logit"):
            ef = {q: 1.0 if depth(out_f[reader][q], variant) is not None else 0.0 for q in both}
            eo = {q: 1.0 if depth(out_o[reader][q], variant) is not None else 0.0 for q in both}
            a = paired_claim(eo, ef, samples=args.samples)
            emerged_both = [q for q in both if ef[q] and eo[q]]
            df = {q: depth(out_f[reader][q], variant) for q in emerged_both}
            do = {q: depth(out_o[reader][q], variant) for q in emerged_both}
            b = paired_claim(df, do, samples=args.samples) if emerged_both else {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "verdict": "-"}
            decides = variant == DECIDES[reader]
            va = a["verdict"] if (reader == "qwen3b" and decides) else ("descriptive" if reader == "qwen7b" else a["verdict"] + " (beside)")
            vb = b["verdict"] if (reader == "qwen3b" and decides) else ("descriptive" if reader == "qwen7b" else b["verdict"] + " (beside)")
            numbers["claims"][f"{reader}/{variant}"] = {"A": a, "B": b, "n": len(both), "n_both_emerged": len(emerged_both)}
            lines.append(f"| {_label(reader, variant)} | {len(both)} | {sum(ef.values()) / len(both):.3f} | {sum(eo.values()) / len(both):.3f} | "
                         f"{_fmt(a)} | {va} | {len(emerged_both)} | {_quartiles(list(df.values()))['median'] if df else float('nan'):.3f} | "
                         f"{_quartiles(list(do.values()))['median'] if do else float('nan'):.3f} | {_fmt(b)} | {vb} |")
        lines.append("")
        # decomposition own vs forced by class
        lines += ["| lens | mode | class | n | A | B | C |", "|---|---|---|---|---|---|---|"]
        for variant in ("jlens", "logit"):
            for mode, outs in (("forced", out_f), ("own", out_o)):
                for cls, members in classes_of(by_q[reader], both).items():
                    found = [gold(outs[reader][q], variant)["outcome"] for q in members]
                    if not found:
                        continue
                    cells = " | ".join(f"{sum(x == o for x in found) / len(found):.3f}" for o in OUTCOMES)
                    lines.append(f"| {_label(reader, variant).replace(', **decides**', '').replace(' **decides**', '')} | {mode} | {cls} | {len(found)} | {cells} |")
        lines.append("")
        # refusal ordering at the own position
        refusals = classes_of(by_q[reader], both)["refusal"]
        for variant in ("jlens", "logit"):
            if not refusals:
                continue
            n_layers = len(out_o[reader][refusals[0]]["measures"][variant]["layers"])
            orders: dict[str, int] = {}
            gf, uf = [], []
            for q in refusals:
                r = refusal_order(oroot / reader / f"{q}.npz", variant, args.k, n_layers)
                orders[r["order"]] = orders.get(r["order"], 0) + 1
                gf.append(r["gold_final"]); uf.append(r["unknown_final"])
            lines += [f"Refusals ({len(refusals)}) at the own position, {variant}: " +
                      "; ".join(f"{k}: {v} ({v / len(refusals):.2f})" for k, v in sorted(orders.items(), key=lambda x: -x[1])) +
                      f". Final-layer rank: gold median {sorted(gf)[len(gf) // 2]}, ` unknown` median {sorted(uf)[len(uf) // 2]}.", ""]
        # marker-less subset
        nomark = [q for q in both if not out_o[reader][q].get("answer_marker")]
        if nomark:
            for variant in ("jlens", "logit"):
                em = [depth(out_o[reader][q], variant) is not None for q in nomark]
                corr = [by_q[reader][q].correct for q in nomark]
                lines.append(f"Marker-less outputs ({len(nomark)}; {sum(1 for c in corr if c)} judged correct), {variant}: gold emerges on "
                             f"{sum(em)} ({bootstrap_proportion(em, samples=args.samples)['p']:.3f}).")
            lines.append("")
    lines.append("Verdict rules: Claim A SHOWN if the lower bound of (own − forced) share emerged is above 0; Claim B SHOWN if the lower bound of "
                 "(forced − own) depth among questions emerged in both modes is above 0. Pre-registered for qwen3b on its deciding (logit) row.")
    text = "\n".join(lines) + "\n"
    out = Path(args.out or f"runs/exp34b_report_{'dev' if folds <= DEV_FOLDS else 'confirm'}.md")
    out.write_text(text); out.with_suffix(".json").write_text(json.dumps(numbers, indent=1, default=float))
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
