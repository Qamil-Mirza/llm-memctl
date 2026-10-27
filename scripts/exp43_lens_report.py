#!/usr/bin/env python
"""§43 Part B report (amendment 1; rev 1 B2), run ON THE LAPTOP from the rows copied back from one family's pod.

    .venv/bin/python scripts/exp43_lens_report.py --family llama --root runs/exp43_lens_pod_llama
    .venv/bin/python scripts/exp43_lens_report.py --family llama --root runs/exp43_lens_pod_llama --confirm

Without --confirm only the development folds 0-1 are loaded (every number is development-only and descriptive; the
J-lens gate is read there). With --confirm the confirmation folds 2-4 are loaded once, after the choices are
committed, and the pre-registered claims are read on them:
- Claim B1 (size): log10 best rank of the gold's first token at P1 (logit lens), small minus large, paired by
  question; SHOWN when the paired-bootstrap lower bound is > 0. Secondary: share of outcome A, small minus large, LB > 0.
- Claim B2 (within the small model): AUC of the gold's P1 log10 best rank (logit lens) for wrong against right
  answers; SHOWN when the lower bound is > 0.5.
- The logit lens decides. The small model's J-lens rows are shown beside and take a verdict role only if the lens
  passes the §34 gate (a) and (b) on folds 0-1 (exp34_report.gate_for).
- Descriptive, no threshold: A/B/C of the small model's in-view failures split refusal / answered wrong (and the
  large model's), the median emergence depth (first layer with the gold in the top-10, layer / n_layers) for both
  sizes beside Qwen's 0.889 (3B) and 0.464 (7B), the article-first split and the digit caveat (exp43_lens_profile).
Rows (scripts/exp43_lens_pod.sh layout): <root>/logit/<small|large>/<qid>.{json,npz}, <root>/jl/<small>/,
<root>/jl/lenses/<small>.json, <root>/jl/eval/<small>.json, <root>/stage_times.json.
Exit 2: folds 2-4 asked for without --confirm. Exit 4: a read's prompt hash differs from the population's or between
the sizes. Exit 6: rows missing (listed). Writes <root>/report_{dev,confirm}.{md,json}.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import median

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memctl.analysis import families  # noqa: E402
from memctl.analysis.jacobian_stats import bootstrap_proportion, paired_claim  # noqa: E402
from memctl.analysis.jlens_readout import OUTCOMES, outcome  # noqa: E402
from scripts.exp34_report import _quartiles, claim2_for, classes_of, gate_for  # noqa: E402
from scripts.exp43_lens_pod_run import EST_S, WEIGHTS_GB, pop34_pairs  # noqa: E402
from scripts.exp43_lens_profile import ARTICLES  # noqa: E402

DEV_FOLDS = (0, 1)
CONFIRM_FOLDS = (2, 3, 4)
K = 10
QWEN_EMERGENCE = {"qwen3b": 0.889, "qwen7b": 0.464}
REFUSED, MISMATCH, MISSING = 2, 4, 6


def check_folds(folds, confirm: bool) -> None:
    bad = sorted(set(folds) - set(DEV_FOLDS))
    if bad and not confirm:
        raise SystemExit(REFUSED)


# --- loading ----------------------------------------------------------------------------------------------------

def load_dir(read_dir: Path, qids, folds) -> dict[str, dict]:
    """qid -> the read json, for the given ids (rows of other folds are never opened)."""
    found = {}
    for q in qids:
        path = Path(read_dir) / f"{q}.json"
        if not path.exists():
            continue
        entry = json.loads(path.read_text())
        if entry["fold"] not in folds:
            raise ValueError(f"{q}: read fold {entry['fold']} is outside {folds}")
        found[q] = entry
    return found


def p1_gold_ranks(read_dir: Path, qid: str, variant: str = "logit") -> np.ndarray:
    with np.load(Path(read_dir) / f"{qid}.npz") as z:
        names = ["gold", "unknown_sp", "Unknown_sp", "unknown"]
        return z[f"ranks_{variant}"][:, 0, names.index("gold")].astype(int)


def logrank(entry: dict, variant: str = "logit") -> float:
    return math.log10(entry["measures"][variant]["P1"]["gold"]["best_rank"])


def is_a(entry: dict, variant: str = "logit") -> float:
    return 1.0 if entry["measures"][variant]["P1"]["gold"]["outcome"] == "A" else 0.0


# --- claims and descriptives (pure: entries and records in, numbers out) -------------------------------------------

def claim_b1(small: dict, large: dict, qids, samples: int) -> dict:
    both = [q for q in qids if q in small and q in large]
    return {"best_logrank": paired_claim({q: logrank(small[q]) for q in both}, {q: logrank(large[q]) for q in both}, samples=samples),
            "outcome_A": paired_claim({q: is_a(small[q]) for q in both}, {q: is_a(large[q]) for q in both}, samples=samples)}


def claim_b2(small: dict, by_q: dict, qids, samples: int, variant: str = "logit") -> dict:
    return claim2_for(small, by_q, [q for q in qids if q in small], variant, samples)


def abc(entries: dict, by_q: dict, qids, samples: int, variant: str = "logit") -> dict:
    present = [q for q in qids if q in entries]
    table = {}
    for cls, members in classes_of(by_q, present).items():
        found = [entries[q]["measures"][variant]["P1"]["gold"]["outcome"] for q in members]
        table[cls] = {"n": len(found), **{o: bootstrap_proportion([x == o for x in found], samples=samples) for o in OUTCOMES}}
    return table


def emergence(entries: dict, qids, variant: str = "logit") -> dict:
    present = [q for q in qids if q in entries]
    depths = [entries[q]["measures"][variant]["P1"]["gold"]["emergence_depth"] for q in present]
    return {**_quartiles([d for d in depths if d is not None]), "n_read": len(present)}


def article_split(read_dir: Path, entries: dict, qids) -> dict:
    def part(rs):
        if not rs:
            return {"n": 0}
        ranks = [p1_gold_ranks(read_dir, q) for q in rs]
        outs = [outcome(r, K) for r in ranks]
        return {"n": len(rs), "final_top10": float(np.mean([r[-1] <= K for r in ranks])),
                "final_top1": float(np.mean([r[-1] == 1 for r in ranks])), "A": outs.count("A") / len(rs),
                "B": outs.count("B") / len(rs)}
    present = [q for q in qids if q in entries]
    art = [q for q in present if entries[q]["gold_token_text"].strip() in ARTICLES]
    b_rows = [q for q in present if entries[q]["measures"]["logit"]["P1"]["gold"]["outcome"] == "B"]
    return {"article": part(art), "non_article": part([q for q in present if q not in set(art)]),
            "B_rows": len(b_rows), "B_rows_article_first": sum(q in set(art) for q in b_rows)}


def digit_subset(entries: dict, by_q: dict, qids) -> dict:
    numeric = [q for q in qids if q in entries and by_q[q].gold.strip()[:1].isdigit()]
    tops = [entries[q]["measures"]["logit"]["P1"]["gold"]["final_rank"] <= K for q in numeric]
    return {"n": len(numeric), "final_top10": float(np.mean(tops)) if tops else None,
            "gold_token_texts": sorted({entries[q]["gold_token_text"] for q in numeric})[:12]}


def cross_family_flags(abc_small: dict, em_small: dict, em_large: dict) -> dict:
    """The amendment's cross-family reading for one family: failures mostly C (share > 0.5) with an A floor (A > 0),
    and the small model's median emergence later than the large model's."""
    fail = abc_small.get("all failures", {})
    c = (fail.get("C") or {}).get("p")
    a = (fail.get("A") or {}).get("p")
    ms, ml = em_small.get("median"), em_large.get("median")
    return {"failures_C_share": c, "failures_A_share": a, "C_majority_with_A_floor": (c is not None and c > 0.5 and (a or 0) > 0),
            "small_median_emergence": ms, "large_median_emergence": ml,
            "small_later": (ms is not None and ml is not None and ms > ml)}


# --- formatting -------------------------------------------------------------------------------------------------

def _ci(stat: dict, key: str = "mean", signed: bool = True) -> str:
    if stat.get("ci_low") is None:
        return "-"
    f = "+.3f" if signed else ".3f"
    return f"{stat[key]:{f}} ({stat['ci_low']:{f}}, {stat['ci_high']:{f}})"


def _p(stat: dict) -> str:
    return "-" if stat.get("p") is None else f"{stat['p']:.3f} ({stat['ci_low']:.3f}, {stat['ci_high']:.3f})"


def _n(x, d=3) -> str:
    return "-" if x is None else f"{x:.{d}f}"


def time_table(family: str, stage_times: dict | None) -> list[str]:
    est = EST_S[family]
    observed = {s["stage"]: s for s in (stage_times or {}).get("stages", [])}
    rows = [("install", "install"), ("download", "download"), ("smoke_small + smoke_large", "smoke"),
            ("logit_small", "logit_small"), ("logit_large", "logit_large"), ("fit", "fit"), ("jread", "jread"),
            ("eval", "eval")]
    lines = ["| stage | expected (s, A40) | observed (s) | status |", "|---|---|---|---|"]
    for label, key in rows:
        stages = [observed[s] for s in label.split(" + ") if s in observed]
        obs = sum(s["elapsed_s"] for s in stages) if stages else None
        status = ", ".join(s["status"] for s in stages) if stages else "-"
        lines.append(f"| {label} | {est[key] * (2 if key == 'smoke' else 1)} | {obs if obs is not None else '-'} | {status} |")
    total = sum(v * (2 if k == "smoke" else 1) for k, v in est.items() if k != "fit_min")
    lines.append(f"| total | {total} | {sum(s['elapsed_s'] for s in observed.values()) if observed else '-'} | |")
    return lines


# --- build ------------------------------------------------------------------------------------------------------

def build(family: str, root: str, confirm: bool = False, samples: int = 10000, pairs=None) -> tuple[str, dict]:
    small_cfg, large_cfg = families.pair(family)
    folds = DEV_FOLDS + (CONFIRM_FOLDS if confirm else ())
    check_folds(folds, confirm)
    pairs = pop34_pairs(small_cfg, large_cfg, folds) if pairs is None else [p for p in pairs if p[0].fold in folds]
    root = Path(root)
    by_small = {x.question_id: x for x, _ in pairs}
    by_large = {y.question_id: y for _, y in pairs}
    want_sha = {x.question_id: __import__("hashlib").sha256(x.prompt.encode()).hexdigest() for x, _ in pairs}
    dirs = {"small": root / "logit" / small_cfg.key, "large": root / "logit" / large_cfg.key, "jlens": root / "jl" / small_cfg.key}
    qids_all = sorted(by_small)
    out = {name: load_dir(d, qids_all, folds) for name, d in dirs.items()}
    bad = [f"{name}/{q}: prompt_sha256 {e.get('prompt_sha256', '')[:12]} != population {want_sha[q][:12]}"
           for name, entries in out.items() for q, e in entries.items() if e.get("prompt_sha256") != want_sha[q]]
    if bad:
        print("\n".join(bad[:20]), file=sys.stderr)
        raise SystemExit(MISMATCH)
    missing = {name: sorted(set(qids_all) - set(out[name])) for name in ("small", "large")}
    if any(missing.values()):
        print(f"missing logit rows: small {len(missing['small'])}, large {len(missing['large'])} "
              f"(e.g. {(missing['small'] + missing['large'])[:4]})", file=sys.stderr)
        raise SystemExit(MISSING)
    dev = [q for q in qids_all if by_small[q].fold in DEV_FOLDS]
    conf = [q for q in qids_all if by_small[q].fold in CONFIRM_FOLDS]
    claim_qids = conf if confirm else dev
    claim_label = "confirmation folds 2-4" if confirm else "development folds 0-1 (descriptive; claims need --confirm)"

    # J-lens gate on folds 0-1 (exp34_report.gate_for: (a) logit minus J-lens band best log-rank, LB > 0; (b) eval)
    jl_dev = {q: e for q, e in out["jlens"].items() if q in set(dev)}
    gate = gate_for(small_cfg.key, jl_dev, dev, root / "jl", samples) if jl_dev else {"verdict": "NOT RUN", "a": {}, "b": {}}
    gate_pass = gate.get("verdict") == "PASS"

    numbers = {"family": family, "small": small_cfg.key, "large": large_cfg.key, "confirm": confirm,
               "claim_folds": list(CONFIRM_FOLDS if confirm else DEV_FOLDS), "n": {"dev": len(dev), "confirm": len(conf)},
               "jlens_rows": {"dev": len(jl_dev), "all": len(out["jlens"])}, "gate": gate}
    b1 = claim_b1(out["small"], out["large"], claim_qids, samples)
    b2 = claim_b2(out["small"], by_small, claim_qids, samples)
    b2_jl = claim_b2(out["jlens"], by_small, claim_qids, samples, "jlens") if all(q in out["jlens"] for q in claim_qids) and claim_qids else None
    if not confirm:
        for stat in (b1["best_logrank"], b1["outcome_A"], b2):
            stat["verdict_role"] = "development, descriptive"
    numbers.update({"B1": b1, "B2": b2, "B2_jlens": b2_jl})
    d_qids = claim_qids
    abc_s, abc_l = abc(out["small"], by_small, d_qids, samples), abc(out["large"], by_large, d_qids, samples)
    abc_j = abc(out["jlens"], by_small, d_qids, samples, "jlens") if b2_jl is not None else None
    em = {"small": emergence(out["small"], d_qids), "large": emergence(out["large"], d_qids)}
    if b2_jl is not None:
        em["small_jlens"] = emergence(out["jlens"], d_qids, "jlens")
    numbers.update({"abc": {"small": abc_s, "large": abc_l, "small_jlens": abc_j}, "emergence": em,
                    "article_split": {"small": article_split(dirs["small"], out["small"], d_qids),
                                      "large": article_split(dirs["large"], out["large"], d_qids)},
                    "digits": {"small": digit_subset(out["small"], by_small, d_qids),
                               "large": digit_subset(out["large"], by_large, d_qids)},
                    "same_gold_token_text": sum(out["small"][q]["gold_token_text"] == out["large"][q]["gold_token_text"]
                                                for q in d_qids),
                    "cross_family": cross_family_flags(abc_s, em["small"], em["large"])})
    stage_times = json.loads((root / "stage_times.json").read_text()) if (root / "stage_times.json").exists() else None
    numbers["stage_times"] = stage_times

    S, L = small_cfg.key, large_cfg.key
    md = [f"# §43 Part B: where does the small reader lose the answer? {family} ({S} vs {L})", "",
          f"Rows from `{root}`. Population: the §34 rule on the §24 `fixed8` caches (answerable, all evidence in view, "
          f"both sizes, same prompt sha256, judge label on both): {len(dev)} development rows, "
          f"{len(conf) if confirm else 'not loaded'} confirmation rows. Forced mode (an appended `Answer:`), P1, k = {K}.",
          f"Claims read on: **{claim_label}**. The logit lens decides (amendment 1).", "",
          "## J-lens gate (§34 (a) and (b), development folds 0-1)", "",
          "| lens prompts | below minimum | band | n | (a) logit minus J-lens (95% CI) | (a) | (b) mean pass@k J-lens / logit | (b) | gate |",
          "|---|---|---|---|---|---|---|---|---|"]
    if jl_dev:
        a, b = gate["a"], gate["b"]
        mb = b.get("mean_pass_at_k") or {}
        md.append(f"| {gate.get('n_prompts')} | {gate.get('below_minimum')} | {gate.get('band')} | {a.get('n')} | {_ci(a)} | "
                  f"{a.get('verdict')} | {_n(mb.get('jlens'))} / {_n(mb.get('logit'))} | {b.get('verdict')} | **{gate['verdict']}** |")
        if gate.get("eval_warnings"):
            md.append(f"\nEval warnings: {'; '.join(gate['eval_warnings'])}")
    else:
        md.append("| - | - | - | 0 | - | - | - | - | **NOT RUN** (no J-lens rows) |")
    md += ["", f"## Claim B1 (size): at P1, {S} minus {L}, paired by question (positive = {L} better)", "",
           "| measure | lens | n | mean difference (95% CI) | verdict |", "|---|---|---|---|---|",
           f"| log10 best rank of the gold | logit | {b1['best_logrank']['n']} | {_ci(b1['best_logrank'])} | "
           f"{b1['best_logrank']['verdict'] if confirm else 'descriptive (dev)'} |",
           f"| share of outcome A (secondary) | logit | {b1['outcome_A']['n']} | {_ci(b1['outcome_A'])} | "
           f"{b1['outcome_A']['verdict'] if confirm else 'descriptive (dev)'} |", "",
           f"## Claim B2 (within {S}): AUC of the gold's P1 log10 best rank, wrong vs right (> 0.5: failures rank worse)", "",
           "| lens | n wrong | n right | AUC (95% CI) | mean wrong | mean right | verdict |", "|---|---|---|---|---|---|---|",
           f"| logit | {b2['n_pos']} | {b2['n_neg']} | {_ci(b2, 'auc', False)} | {_n(b2['mean_wrong'])} | {_n(b2['mean_right'])} | "
           f"{b2['verdict'] if confirm else 'descriptive (dev)'} |"]
    if b2_jl is not None:
        role = ("verdict role (gate PASS)" if gate_pass else "shown beside (gate not passed)") if confirm else "descriptive (dev)"
        md.append(f"| J-lens | {b2_jl['n_pos']} | {b2_jl['n_neg']} | {_ci(b2_jl, 'auc', False)} | {_n(b2_jl['mean_wrong'])} | "
                  f"{_n(b2_jl['mean_right'])} | {b2_jl['verdict'] + ', ' + role if confirm else role} |")
    md += ["", f"## Descriptive: A / B / C at P1 ({claim_label})", "",
           "A never in the top-10, B in it at some layer but not the last, C in it at the last layer. Refusal = the §24 "
           "judge marked the answer `unknown`.", "", "| reader | lens | class | n | A (95% CI) | B (95% CI) | C (95% CI) |",
           "|---|---|---|---|---|---|---|"]
    for name, table, lens in ((S, abc_s, "logit"), (L, abc_l, "logit"), (S, abc_j, "J-lens")):
        if table is None:
            continue
        for cls in ("success", "refusal", "answered wrong", "all failures"):
            row = table[cls]
            md.append(f"| {name} | {lens} | {cls} | {row['n']} | " + " | ".join(_p(row[o]) for o in OUTCOMES) + " |")
    md += ["", "## Descriptive: emergence depth of the gold at P1 (first layer in the top-10, layer / n_layers)", "",
           "| reader | lens | n emerged | n read | q1 | median | q3 |", "|---|---|---|---|---|---|---|"]
    for name, key, lens in ((S, "small", "logit"), (L, "large", "logit"), (S, "small_jlens", "J-lens")):
        if key in em:
            e = em[key]
            md.append(f"| {name} | {lens} | {e['n']} | {e['n_read']} | {_n(e['q1'])} | {_n(e['median'])} | {_n(e['q3'])} |")
    md += [f"| qwen3b (§34) | - | - | - | - | {QWEN_EMERGENCE['qwen3b']} | - |",
           f"| qwen7b (§34) | - | - | - | - | {QWEN_EMERGENCE['qwen7b']} | - |", "",
           "Depth compares across families only as a fraction of depth and only descriptively.", "",
           "## Caveats: article-first golds and digits", ""]
    for name, key in ((S, "small"), (L, "large")):
        sp = numbers["article_split"][key]
        a, n_ = sp["article"], sp["non_article"]
        md.append(f"- {name}: article-first golds n={a['n']} (final top-10 {_n(a.get('final_top10'))}, A {_n(a.get('A'))}, "
                  f"B {_n(a.get('B'))}); others n={n_['n']} (final top-10 {_n(n_.get('final_top10'))}, A {_n(n_.get('A'))}, "
                  f"B {_n(n_.get('B'))}); {sp['B_rows_article_first']} of {sp['B_rows']} B rows are article-first.")
    for name, key in ((S, "small"), (L, "large")):
        dg = numbers["digits"][key]
        md.append(f"- {name}: {dg['n']} digit-first golds, final-layer top-10 {_n(dg['final_top10'])}; gold first tokens "
                  f"e.g. {dg['gold_token_texts']}.")
    md += [f"- Llama and Phi put up to three digits in one token, Qwen and Gemma one, so gold first tokens are not like "
           f"for like across families; the A/B/C shares are the cross-family quantity of record. Gold first-token text "
           f"identical across the two sizes on {numbers['same_gold_token_text']} of {len(d_qids)} rows"
           + (" (phi4mini and phi4 have different vocabularies)." if family == "phi" else "."), "",
           "## Cross-family reading (descriptive, this family)", "",
           f"- Small model's failures mostly C (share > 0.5) with an A floor: "
           f"{numbers['cross_family']['C_majority_with_A_floor']} (C {_n(numbers['cross_family']['failures_C_share'])}, "
           f"A {_n(numbers['cross_family']['failures_A_share'])}).",
           f"- Small model's median emergence later than the large model's: {numbers['cross_family']['small_later']} "
           f"({_n(em['small']['median'])} vs {_n(em['large']['median'])}).", "",
           "## Time per stage", ""] + time_table(family, stage_times) + [
           "", f"Weights (bf16): {S} {WEIGHTS_GB[S]} GB, {L} {WEIGHTS_GB[L]} GB; each stage loads one model alone on the GPU."]
    return "\n".join(md) + "\n", numbers


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--family", choices=sorted(families.PAIRS), required=True)
    p.add_argument("--root", default=None, help="default runs/exp43_lens_pod_<family>")
    p.add_argument("--confirm", action="store_true", help="load folds 2-4 and read the claims there")
    p.add_argument("--samples", type=int, default=10000)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)
    root = args.root or f"runs/exp43_lens_pod_{args.family}"
    try:
        md, numbers = build(args.family, root, args.confirm, args.samples)
    except SystemExit as e:
        return int(e.code) if isinstance(e.code, int) else 1
    stem = Path(args.out) if args.out else Path(root) / f"report_{'confirm' if args.confirm else 'dev'}"
    stem.with_suffix(".md").write_text(md)
    stem.with_suffix(".json").write_text(json.dumps(numbers, indent=1, default=str))
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
