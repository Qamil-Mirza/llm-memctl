#!/usr/bin/env python
"""§44 report: claims C1-C3 and the descriptives of the spec from runs/exp44_patch/<model>/.

    .venv/bin/python scripts/exp44_report.py                       # dev folds 0-1, descriptive
    .venv/bin/python scripts/exp44_report.py --root runs/exp44_stage0 --tag stage0
    .venv/bin/python scripts/exp44_report.py --confirm              # folds 2-4, once (refused until authorised)
    .venv/bin/python scripts/exp44_report.py --models qwen3b --out runs/exp44_report_dev_qwen3b   # one model, own path

Spec: research/proposals/2026-10-29-exp44-patching.md rev 0. Without --confirm only rows of folds 0-1 are opened (a
row file of another fold is skipped); every number is development-only and descriptive, and the §3 filter choice (strict
unless fewer than 40 of 126 dev rows survive, then corrupt rank > 3) is computed there. With --confirm the claims are
read on folds 2-4 with the filter fixed in `FIXED_FILTER` (None until the Advisor records it in rev 1: refused).
The §34 per-row depths (first layer with the gold in the top-10 at P1, logit lens and J-lens) are read from
runs/exp34_jlens/<model>/<qid>.json (a prompt hash that differs from the patch row's stops the report, exit 4).
Rev 5 (filter version pt.FILTER_VERSION): every row's derived fields are recomputed here from its recorded lp / rank /
R (`pt.derive`); the stored ones are never read. The filter is the rank filter AND lp_clean - lp_corrupt >= 1 nat (both
filters); rows inside the rank filter with a smaller gap are counted, never in a claim. An eligible row with |R(0)| >=
0.05 is R(0)-flagged: out of the claim statistics, counted beside every claim, and each claim's headline is repeated
with the flagged rows included (robustness line). If more than 10% of a model's eligible rows (or of its filter rows)
are flagged, the report says that the Advisor reviews it before any confirmation read.
Exit 2: confirmation refused. Exit 4: prompt mismatch. Exit 6: a filter row was not patched.
Writes <root>/report_<tag>.{md,json}, or <out>.{md,json} with --out.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import median

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memctl.analysis import patching as pt  # noqa: E402

MODELS = ("qwen3b", "qwen7b")
FIXED_FILTER: dict | None = {"qwen3b": 3, "qwen7b": 3}   # rev 7 (6f457fd): relaxed (corrupt rank > 3) + 1-nat gap, both models
REFUSED, MISMATCH, UNPATCHED = 2, 4, 6
EXP34 = Path("runs/exp34_jlens")


def load_rows(root: Path, model: str, folds) -> dict[str, dict]:
    out = {}
    for path in sorted((root / model).glob("*.json")):
        if path.name.startswith("_"):
            continue
        row = json.loads(path.read_text())
        if row["fold"] not in folds:
            continue
        row.update(pt.derive(row))                      # never trust the stored (write-time) derived fields
        out[row["question_id"]] = row
    return out


def lens_depths(model: str, rows: dict[str, dict], exp34: Path = EXP34) -> dict[str, dict]:
    """qid -> {'logit': d, 'jlens': d} from §34 (None: the gold never entered the top-10)."""
    out = {}
    for q, row in rows.items():
        path = exp34 / model / f"{q}.json"
        if not path.exists():
            continue
        entry = json.loads(path.read_text())
        if entry.get("prompt_sha256") != row["stamp"]["clean_prompt_sha256"]:
            print(f"exp44_report: {model}/{q}: §34 prompt sha differs from the patch row's", file=sys.stderr)
            raise SystemExit(MISMATCH)
        out[q] = {v: entry["measures"][v]["P1"]["gold"]["emergence_depth"] for v in ("logit", "jlens")}
    return out


def _q(values) -> dict:
    if not values:
        return {"n": 0, "median": None, "q1": None, "q3": None}
    a = np.asarray(values, dtype=float)
    return {"n": len(values), "median": float(np.median(a)), "q1": float(np.percentile(a, 25)),
            "q3": float(np.percentile(a, 75))}


def _f(x, d=3):
    return "—" if x is None else f"{x:.{d}f}"


def _stat(s):
    return f"{_f(s['value'])} [{_f(s['ci_low'])}, {_f(s['ci_high'])}] (n={s['n']})"


def classes(rows: list[dict]) -> dict[str, list[dict]]:
    """§34 classes from the model's own §23 judge labels: success / refusal / wrong."""
    return {"success": [r for r in rows if r.get("correct") is True],
            "refusal": [r for r in rows if r.get("correct") is False and r.get("unknown") is True],
            "wrong": [r for r in rows if r.get("correct") is False and r.get("unknown") is not True]}


def _claim_stats(sel: list[dict], lay: int | None, samples: int) -> dict:
    dc = [r["d_c"] for r in sel if r["d_c"] is not None]
    out = {"n": len(sel), "median_dc": pt.bootstrap(dc, median, samples)}
    if lay is not None:
        out["mean_R_c2_layer"] = pt.bootstrap([r["R"][lay] for r in sel], pt.mean, samples)
    return out


def model_section(model: str, rows: dict[str, dict], k_corrupt: int, depths: dict, samples: int) -> dict:
    in_rank = [r for r in rows.values() if pt.in_rank_filter(r["rank_clean"], r["rank_corrupt"], k_corrupt)]
    with_gap = [r for r in in_rank if pt.gap_ok(r["lp_clean"], r["lp_corrupt"])]
    gap_excluded = [r for r in in_rank if not pt.gap_ok(r["lp_clean"], r["lp_corrupt"])]
    unpatched = [r["question_id"] for r in with_gap if not r.get("patched")]
    if unpatched:
        print(f"exp44_report: {model}: filter rows not patched: {unpatched}", file=sys.stderr)
        raise SystemExit(UNPATCHED)
    with_gap.sort(key=lambda r: (r["fold"], r["question_id"]))
    flagged = [r for r in with_gap if r["r0_flag"]]
    sel = [r for r in with_gap if not r["r0_flag"]]                 # the claim rows
    eligible = [r for r in rows.values() if r["eligible"]]
    flagged_eligible = [r for r in eligible if r["r0_flag"]]
    frac_elig = len(flagged_eligible) / len(eligible) if eligible else 0.0
    frac_filter = len(flagged) / len(with_gap) if with_gap else 0.0
    L = with_gap[0]["n_layers"] if with_gap else None
    patched = [r for r in rows.values() if r.get("patched")]
    asserted = [r for r in patched if r["sanity_asserted"]]
    elig_patched = [r for r in eligible if r.get("patched") and r["R"]]
    out = {"model": model, "filter_version": pt.FILTER_VERSION, "gap_min": pt.GAP_MIN, "n_rows": len(rows),
           "n_patched": len(patched), "k_corrupt": k_corrupt, "n_rank_filter": len(in_rank),
           "n_rank_filter_gap": len(with_gap), "n_gap_excluded": len(gap_excluded),
           "gap_excluded": [{"question_id": r["question_id"], "fold": r["fold"], "rank_clean": r["rank_clean"],
                             "rank_corrupt": r["rank_corrupt"], "gap": r["gap"]} for r in gap_excluded],
           "n_filter": len(sel), "n_r0_flagged": len(flagged),
           "r0_flagged": [{"question_id": r["question_id"], "fold": r["fold"], "r_zero": r["R"][0], "gap": r["gap"]}
                          for r in flagged],
           "n_eligible": len(eligible), "n_r0_flagged_eligible": len(flagged_eligible),
           "r0_flag_fraction_eligible": frac_elig, "r0_flag_fraction_filter": frac_filter,
           "advisor_review_required": frac_elig > pt.FLAG_REVIEW_FRACTION or frac_filter > pt.FLAG_REVIEW_FRACTION,
           "n_layers": L, "n_dc_undefined": sum(r["d_c"] is None for r in sel),
           "sanity": {"n_patched": len(patched), "n_asserted": len(asserted),
                      "n_asserted_final_pass": sum(r["sanity"]["final_ok"] is True for r in asserted),
                      "n_all_pass": sum(r["sanity"]["pass"] is True for r in patched),
                      "max_abs_r0_eligible": max((abs(r["R"][0]) for r in elig_patched), default=None),
                      "max_abs_r0": max((abs(r["sanity"]["r_zero"]) for r in patched if r["R"]), default=None),
                      "max_abs_rfinal_minus_1": max((abs(r["sanity"]["r_final"] - 1) for r in patched if r["R"]),
                                                    default=None)}}
    lay = pt.c2_layer(L) if L else None
    claim = _claim_stats(sel, lay if sel else None, samples)
    out["median_dc"] = claim["median_dc"]
    out["robust_with_flagged"] = _claim_stats(with_gap, lay if with_gap else None, samples)
    if not sel:
        return out
    R = np.array([r["R"] for r in sel], dtype=float)
    out["mean_R_curve"] = R.mean(0).tolist()
    out["mean_R_curve_clipped"] = np.clip(R, *pt.CLIP).mean(0).tolist()
    out["mean_lp_unknown_curve"] = np.array([r["lp_unknown_patched"] for r in sel]).mean(0).tolist()
    out["lp_unknown_corrupt_mean"] = float(np.mean([r["lp_unknown_corrupt"] for r in sel]))
    out["lp_unknown_clean_mean"] = float(np.mean([r["lp_unknown_clean"] for r in sel]))
    flips = [r for r in sel if r["top1_corrupt"] != r["gold_id"]
             and any(t == r["gold_id"] for t in r["top1_patched"][:-1])]
    out["n_flip_to_gold_before_final"] = len(flips)
    out["flip_fraction_by_layer"] = np.mean([[t == r["gold_id"] for t in r["top1_patched"]] for r in sel], 0).tolist()
    diffs = {v: [r["d_c"] - depths[r["question_id"]][v] for r in sel
                 if r["d_c"] is not None and r["question_id"] in depths and depths[r["question_id"]][v] is not None]
             for v in ("jlens", "logit")}
    out["dc_minus_lens"] = {v: _q(d) for v, d in diffs.items()}
    out["lens_never"] = {v: sum(1 for r in sel if r["question_id"] in depths and depths[r["question_id"]][v] is None)
                         for v in ("jlens", "logit")}
    out["lens_missing"] = sum(r["question_id"] not in depths for r in sel)
    out["c2_layer"] = lay
    out["mean_R_c2_layer"] = claim["mean_R_c2_layer"]
    out["by_class"] = {name: {"n": len(rs), "median_dc": _q([r["d_c"] for r in rs if r["d_c"] is not None])["median"],
                              "mean_R_c2_layer": float(np.mean([r["R"][lay] for r in rs])) if rs else None}
                       for name, rs in classes(sel).items()}
    out["rows"] = [{"question_id": r["question_id"], "fold": r["fold"], "rank_clean": r["rank_clean"],
                    "rank_corrupt": r["rank_corrupt"], "gap": r["gap"], "d_c": r["d_c"],
                    "d_logit": depths.get(r["question_id"], {}).get("logit"),
                    "d_jlens": depths.get(r["question_id"], {}).get("jlens"),
                    "R_zero": r["R"][0], "R_c2_layer": r["R"][lay], "r_final_ok": r["sanity"]["final_ok"]}
                   for r in sel]
    return out


def build(root: Path, confirm: bool, samples: int = pt.BOOT_SAMPLES, exp34: Path = EXP34,
          models=MODELS) -> tuple[str, dict]:
    if confirm and FIXED_FILTER is None:
        print("exp44_report: --confirm refused: the filter is not yet fixed (FIXED_FILTER; spec §6.4)", file=sys.stderr)
        raise SystemExit(REFUSED)
    folds = pt.CONFIRM_FOLDS if confirm else pt.DEV_FOLDS
    numbers = {"confirm": confirm, "folds": list(folds), "method_sha256": pt.method_sha256(),
               "filter_version": pt.FILTER_VERSION, "gap_min": pt.GAP_MIN, "samples": samples, "models": {}}
    for model in models:
        rows = load_rows(root, model, folds)
        if not rows:
            continue
        dev = rows if not confirm else load_rows(root, model, pt.DEV_FOLDS)
        choice = pt.choose_filter(list(dev.values()))
        k = FIXED_FILTER[model] if confirm else choice["k_corrupt"]
        sec = model_section(model, rows, k, lens_depths(model, rows, exp34), samples)
        sec["filter_choice_dev"] = choice
        numbers["models"][model] = sec
    m7, m3 = numbers["models"].get("qwen7b"), numbers["models"].get("qwen3b")
    claims = {}

    def flags(m):
        return {"n_r0_flagged": m["n_r0_flagged"], "advisor_review_required": m["advisor_review_required"]}

    if m7:
        rob = m7["robust_with_flagged"]
        claims["C1"] = {"stat": m7["median_dc"], "reading": pt.c1_reading(m7["median_dc"]), **flags(m7),
                        "robust": {"stat": rob["median_dc"], "reading": pt.c1_reading(rob["median_dc"])}}
        if "mean_R_c2_layer" in m7:
            claims["C2"] = {"layer": m7["c2_layer"], "stat": m7["mean_R_c2_layer"],
                            "reading": pt.c2_reading(m7["mean_R_c2_layer"]), **flags(m7),
                            "robust": {"stat": rob["mean_R_c2_layer"],
                                       "reading": pt.c2_reading(rob["mean_R_c2_layer"])}}
    if m3:
        rob = m3["robust_with_flagged"]
        claims["C3"] = {"stat": m3["median_dc"], "reading": pt.c3_reading(m3["median_dc"]),
                        "ci_overlaps_0.85-0.92": pt.c3_overlaps_band(m3["median_dc"]), **flags(m3),
                        "robust": {"stat": rob["median_dc"], "reading": pt.c3_reading(rob["median_dc"])}}
    numbers["claims"] = claims
    return render(numbers), numbers


def render(n: dict) -> str:
    label = "confirmation folds 2-4" if n["confirm"] else "development folds 0-1 (descriptive; claims need --confirm)"
    lines = [f"# §44 activation patching report: {label}", "",
             "Spec research/proposals/2026-10-29-exp44-patching.md rev 0, filter amended in rev 5 (filter version "
             f"{n['filter_version']}: rank filter AND lp_clean - lp_corrupt >= {n['gap_min']:g} nat; R(0) a flag, not a "
             "stop; derived fields recomputed from the recorded lp / rank / R). Clean -> evidence-cut at the forced P1; "
             "R(l) = (m_patched(l) - m_corrupt) / (m_clean - m_corrupt); d_c = first l with R >= 0.5, / L. Percentile "
             f"bootstrap, {n['samples']} draws. The C2 layer row of the 3B (layer round(0.464 L)) is descriptive only.",
             ""]
    for m, s in n["models"].items():
        if s["advisor_review_required"]:
            lines += [f"**ADVISOR REVIEW REQUIRED BEFORE ANY CONFIRMATION READ: {m} has {s['n_r0_flagged_eligible']} of "
                      f"{s['n_eligible']} eligible rows ({100 * s['r0_flag_fraction_eligible']:.1f}%) and "
                      f"{s['n_r0_flagged']} of {s['n_rank_filter_gap']} filter rows "
                      f"({100 * s['r0_flag_fraction_filter']:.1f}%) R(0)-flagged (> 10%).**", ""]
    lines += ["## Rows and filter (dev counts; rank = clean rank <= 10 and corrupt rank > k; gap = lp_clean - "
              f"lp_corrupt >= {n['gap_min']:g} nat)", "",
              "| model | rows | patched | dev strict rank (>10) | strict + gap | dev relaxed rank (>3) | relaxed + gap | "
              "gap < 1 in strict rank | gap < 1 in relaxed rank | filter used (corrupt >) | in used rank filter | "
              "gap < 1 there (excluded) | R(0) flagged (excluded) | claim rows | d_c undefined |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for m, s in n["models"].items():
        c = s["filter_choice_dev"]
        lines.append(f"| {m} | {s['n_rows']} | {s['n_patched']} | {c['n_strict_rank']} / {c['n_dev']} | {c['n_strict']} | "
                     f"{c['n_relaxed_rank']} | {c['n_relaxed']} | {c['n_gap_excluded_strict']} | "
                     f"{c['n_gap_excluded_relaxed']} | "
                     f"{s['k_corrupt']}{' (relaxed)' if c['relaxed'] and not n['confirm'] else ''} | "
                     f"{s['n_rank_filter']} | {s['n_gap_excluded']} | {s['n_r0_flagged']} | {s['n_filter']} | "
                     f"{s['n_dc_undefined']} |")
    for m, s in n["models"].items():
        if s["gap_excluded"]:
            lines += ["", f"{m}: rows inside the used rank filter with gap < 1 nat (recorded, never in a claim): " +
                      ", ".join(f"{g['question_id']} (f{g['fold']}, ranks {g['rank_clean']}/{g['rank_corrupt']}, gap "
                                f"{g['gap']:.2f})" for g in s["gap_excluded"])]
        if s["r0_flagged"]:
            lines += ["", f"{m}: R(0)-flagged filter rows (out of the claim statistics): " +
                      ", ".join(f"{g['question_id']} (f{g['fold']}, R(0) {g['r_zero']:.3f}, gap {g['gap']:.2f})"
                                for g in s["r0_flagged"])]
    lines += ["", "## Sanity (asserted: |R(L-1) - 1| < 0.01 on eligible patched rows; |R(0)| < 0.05 recorded, a flag)",
              "", "| model | patched | asserted | asserted R(L-1) pass | all pass both bounds | eligible | eligible "
              "R(0)-flagged | max abs R(0) eligible | max abs R(0) all | max abs R(L-1)-1 |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for m, s in n["models"].items():
        z = s["sanity"]
        lines.append(f"| {m} | {z['n_patched']} | {z['n_asserted']} | {z['n_asserted_final_pass']} | {z['n_all_pass']} | "
                     f"{s['n_eligible']} | {s['n_r0_flagged_eligible']} ({100 * s['r0_flag_fraction_eligible']:.1f}%) | "
                     f"{_f(z['max_abs_r0_eligible'], 4)} | {_f(z['max_abs_r0'], 4)} | "
                     f"{_f(z['max_abs_rfinal_minus_1'], 5)} |")
    c = n["claims"]
    lines += ["", "## Claims" + ("" if n["confirm"] else " (dev: descriptive only)"), "",
              "| claim | statistic | value [95% CI] | reading | R(0)-flagged excluded |", "|---|---|---|---|---|"]

    def claim_lines(name, stat_name, cl, extra=""):
        out = [f"| {name} | {stat_name} | {_stat(cl['stat'])} | {cl['reading']}{extra} | {cl['n_r0_flagged']} |",
               f"| {name} robustness (flagged included) | {stat_name} | {_stat(cl['robust']['stat'])} | "
               f"{cl['robust']['reading']} | none ({cl['n_r0_flagged']} included) |"]
        if cl["advisor_review_required"]:
            out.append(f"| {name} | — | — | ADVISOR REVIEW REQUIRED BEFORE ANY CONFIRMATION READ (> 10% flagged) | "
                       f"{cl['n_r0_flagged']} |")
        return out

    if "C1" in c:
        lines += claim_lines("C1 (7B)", "median d_c", c["C1"])
    if "C2" in c:
        lines += claim_lines("C2 (7B)", f"mean R at layer {c['C2']['layer']}", c["C2"])
    if "C3" in c:
        lines += claim_lines("C3 (3B)", "median d_c", c["C3"],
                             f" (CI overlaps 0.85-0.92: {c['C3']['ci_overlaps_0.85-0.92']})")
    for m, s in n["models"].items():
        if "mean_R_curve" not in s:
            continue
        lines += ["", f"## {m}: descriptives (filter rows)", "",
                  "| d_c minus | n | median | q1 | q3 | lens never |", "|---|---|---|---|---|---|"]
        for v in ("jlens", "logit"):
            q = s["dc_minus_lens"][v]
            lines.append(f"| {v} | {q['n']} | {_f(q['median'])} | {_f(q['q1'])} | {_f(q['q3'])} | {s['lens_never'][v]} |")
        lines += ["", f"Rows where patching alone flips top-1 to the gold before the final layer: "
                  f"{s['n_flip_to_gold_before_final']} of {s['n_filter']}. Mean log p(' unknown'): clean "
                  f"{_f(s['lp_unknown_clean_mean'])}, corrupt {_f(s['lp_unknown_corrupt_mean'])}.", "",
                  "| class | n | median d_c | mean R at C2 layer |", "|---|---|---|---|"]
        for name, b in s["by_class"].items():
            lines.append(f"| {name} | {b['n']} | {_f(b['median_dc'])} | {_f(b['mean_R_c2_layer'])} |")
        lines += ["", "| layer | depth | mean R | mean R clipped | top-1 = gold | mean log p(' unknown') |",
                  "|---|---|---|---|---|---|"]
        L = s["n_layers"]
        for l in range(L):
            lines.append(f"| {l} | {l / L:.3f} | {_f(s['mean_R_curve'][l])} | {_f(s['mean_R_curve_clipped'][l])} | "
                         f"{_f(s['flip_fraction_by_layer'][l], 2)} | {_f(s['mean_lp_unknown_curve'][l], 2)} |")
        lines += ["", "| qid | fold | rank clean | rank corrupt | gap | d_c | d_logit | d_jlens | R(0) | R(C2 layer) | "
                  "R(L-1) ok |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in s["rows"]:
            lines.append(f"| {r['question_id']} | {r['fold']} | {r['rank_clean']} | {r['rank_corrupt']} | {_f(r['gap'], 2)} | "
                         f"{_f(r['d_c'])} | {_f(r['d_logit'])} | {_f(r['d_jlens'])} | {_f(r['R_zero'], 4)} | "
                         f"{_f(r['R_c2_layer'])} | {r['r_final_ok']} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="runs/exp44_patch")
    p.add_argument("--confirm", action="store_true")
    p.add_argument("--samples", type=int, default=pt.BOOT_SAMPLES)
    p.add_argument("--tag", default=None)
    p.add_argument("--models", default=",".join(MODELS), help="comma-separated subset of qwen3b,qwen7b")
    p.add_argument("--out", default=None, help="output path prefix (writes <out>.md and <out>.json)")
    args = p.parse_args(argv)
    root = Path(args.root)
    models = tuple(m for m in args.models.split(",") if m)
    if set(models) - set(MODELS):
        p.error(f"unknown models {sorted(set(models) - set(MODELS))}")
    text, numbers = build(root, args.confirm, args.samples, models=models)
    tag = args.tag or ("confirm" if args.confirm else "dev")
    prefix = Path(args.out) if args.out else root / f"report_{tag}"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_name(prefix.name + ".md").write_text(text)
    prefix.with_name(prefix.name + ".json").write_text(json.dumps(numbers, indent=1))
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
