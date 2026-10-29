#!/usr/bin/env python
"""§44 report: claims C1-C3 and the descriptives of the spec from runs/exp44_patch/<model>/.

    .venv/bin/python scripts/exp44_report.py                       # dev folds 0-1, descriptive
    .venv/bin/python scripts/exp44_report.py --root runs/exp44_stage0 --tag stage0
    .venv/bin/python scripts/exp44_report.py --confirm              # folds 2-4, once (refused until authorised)

Spec: research/proposals/2026-10-29-exp44-patching.md rev 0. Without --confirm only rows of folds 0-1 are opened (a
row file of another fold is skipped); every number is development-only and descriptive, and the §3 filter choice (strict
unless fewer than 40 of 126 dev rows survive, then corrupt rank > 3) is computed there. With --confirm the claims are
read on folds 2-4 with the filter fixed in `FIXED_FILTER` (None until the Advisor records it in rev 1: refused).
The §34 per-row depths (first layer with the gold in the top-10 at P1, logit lens and J-lens) are read from
runs/exp34_jlens/<model>/<qid>.json (a prompt hash that differs from the patch row's stops the report, exit 4).
Exit 2: confirmation refused. Exit 4: prompt mismatch. Exit 6: a filter row was not patched.
Writes <root>/report_<tag>.{md,json}.
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
FIXED_FILTER: dict | None = None        # e.g. {"qwen3b": 10, "qwen7b": 10}: k_corrupt per model, set in rev 1
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


def model_section(model: str, rows: dict[str, dict], k_corrupt: int, depths: dict, samples: int) -> dict:
    sel = [r for r in rows.values() if pt.evidence_dependent(r["rank_clean"], r["rank_corrupt"], k_corrupt)]
    unpatched = [r["question_id"] for r in sel if not r.get("patched")]
    if unpatched:
        print(f"exp44_report: {model}: filter rows not patched: {unpatched}", file=sys.stderr)
        raise SystemExit(UNPATCHED)
    sel.sort(key=lambda r: (r["fold"], r["question_id"]))
    L = sel[0]["n_layers"] if sel else None
    patched = [r for r in rows.values() if r.get("patched")]
    asserted = [r for r in patched if r.get("sanity_asserted")]
    dc = [r["d_c"] for r in sel if r["d_c"] is not None]
    out = {"model": model, "n_rows": len(rows), "n_patched": len(patched), "k_corrupt": k_corrupt, "n_filter": len(sel),
           "n_layers": L, "n_dc_undefined": sum(r["d_c"] is None for r in sel),
           "sanity": {"n_patched": len(patched), "n_asserted": len(asserted),
                      "n_asserted_pass": sum(r["sanity"]["pass"] is True for r in asserted),
                      "n_all_pass": sum(r["sanity"]["pass"] is True for r in patched),
                      "max_abs_r0": max((abs(r["sanity"]["r_zero"]) for r in patched if r["R"]), default=None),
                      "max_abs_rfinal_minus_1": max((abs(r["sanity"]["r_final"] - 1) for r in patched if r["R"]),
                                                    default=None)},
           "median_dc": pt.bootstrap(dc, median, samples)}
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
    lay = pt.c2_layer(L)
    out["c2_layer"] = lay
    out["mean_R_c2_layer"] = pt.bootstrap([r["R"][lay] for r in sel], pt.mean, samples)
    out["by_class"] = {name: {"n": len(rs), "median_dc": _q([r["d_c"] for r in rs if r["d_c"] is not None])["median"],
                              "mean_R_c2_layer": float(np.mean([r["R"][lay] for r in rs])) if rs else None}
                       for name, rs in classes(sel).items()}
    out["rows"] = [{"question_id": r["question_id"], "fold": r["fold"], "rank_clean": r["rank_clean"],
                    "rank_corrupt": r["rank_corrupt"], "d_c": r["d_c"],
                    "d_logit": depths.get(r["question_id"], {}).get("logit"),
                    "d_jlens": depths.get(r["question_id"], {}).get("jlens"),
                    "R_c2_layer": r["R"][lay], "sanity_pass": r["sanity"]["pass"]} for r in sel]
    return out


def build(root: Path, confirm: bool, samples: int = pt.BOOT_SAMPLES, exp34: Path = EXP34) -> tuple[str, dict]:
    if confirm and FIXED_FILTER is None:
        print("exp44_report: --confirm refused: the filter is not yet fixed (FIXED_FILTER; spec §6.4)", file=sys.stderr)
        raise SystemExit(REFUSED)
    folds = pt.CONFIRM_FOLDS if confirm else pt.DEV_FOLDS
    numbers = {"confirm": confirm, "folds": list(folds), "method_sha256": pt.method_sha256(), "samples": samples,
               "models": {}}
    for model in MODELS:
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
    if m7:
        claims["C1"] = {"stat": m7["median_dc"], "reading": pt.c1_reading(m7["median_dc"])}
        if "mean_R_c2_layer" in m7:
            claims["C2"] = {"layer": m7["c2_layer"], "stat": m7["mean_R_c2_layer"],
                            "reading": pt.c2_reading(m7["mean_R_c2_layer"])}
    if m3:
        claims["C3"] = {"stat": m3["median_dc"], "reading": pt.c3_reading(m3["median_dc"]),
                        "ci_overlaps_0.85-0.92": pt.c3_overlaps_band(m3["median_dc"])}
    numbers["claims"] = claims
    return render(numbers), numbers


def render(n: dict) -> str:
    label = "confirmation folds 2-4" if n["confirm"] else "development folds 0-1 (descriptive; claims need --confirm)"
    lines = [f"# §44 activation patching report: {label}", "",
             "Spec research/proposals/2026-10-29-exp44-patching.md rev 0. Clean -> evidence-cut at the forced P1; R(l) = "
             "(m_patched(l) - m_corrupt) / (m_clean - m_corrupt); d_c = first l with R >= 0.5, / L. Percentile bootstrap, "
             f"{n['samples']} draws. The C2 layer row of the 3B (layer round(0.464 L)) is descriptive only.", ""]
    lines += ["## Rows and filter", "", "| model | rows | patched | dev strict (rank_c<=10, corrupt>10) | dev relaxed (>3) | "
              "filter used (corrupt >) | rows in filter | d_c undefined |", "|---|---|---|---|---|---|---|---|"]
    for m, s in n["models"].items():
        c = s["filter_choice_dev"]
        lines.append(f"| {m} | {s['n_rows']} | {s['n_patched']} | {c['n_strict']} / {c['n_dev']} | {c['n_relaxed']} | "
                     f"{s['k_corrupt']}{' (relaxed)' if c['relaxed'] and not n['confirm'] else ''} | {s['n_filter']} | "
                     f"{s['n_dc_undefined']} |")
    lines += ["", "## Sanity (|R(L-1) - 1| < 0.01, |R(0)| < 0.05)", "",
              "| model | patched | asserted | asserted pass | all pass | max abs R(0) | max abs R(L-1)-1 |",
              "|---|---|---|---|---|---|---|"]
    for m, s in n["models"].items():
        z = s["sanity"]
        lines.append(f"| {m} | {z['n_patched']} | {z['n_asserted']} | {z['n_asserted_pass']} | {z['n_all_pass']} | "
                     f"{_f(z['max_abs_r0'], 4)} | {_f(z['max_abs_rfinal_minus_1'], 5)} |")
    c = n["claims"]
    lines += ["", "## Claims" + ("" if n["confirm"] else " (dev: descriptive only)"), "",
              "| claim | statistic | value [95% CI] | reading |", "|---|---|---|---|"]
    if "C1" in c:
        lines.append(f"| C1 (7B) | median d_c | {_stat(c['C1']['stat'])} | {c['C1']['reading']} |")
    if "C2" in c:
        lines.append(f"| C2 (7B) | mean R at layer {c['C2']['layer']} | {_stat(c['C2']['stat'])} | {c['C2']['reading']} |")
    if "C3" in c:
        lines.append(f"| C3 (3B) | median d_c | {_stat(c['C3']['stat'])} | {c['C3']['reading']} "
                     f"(CI overlaps 0.85-0.92: {c['C3']['ci_overlaps_0.85-0.92']}) |")
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
        lines += ["", "| qid | fold | rank clean | rank corrupt | d_c | d_logit | d_jlens | R(C2 layer) | sanity |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for r in s["rows"]:
            lines.append(f"| {r['question_id']} | {r['fold']} | {r['rank_clean']} | {r['rank_corrupt']} | {_f(r['d_c'])} | "
                         f"{_f(r['d_logit'])} | {_f(r['d_jlens'])} | {_f(r['R_c2_layer'])} | {r['sanity_pass']} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="runs/exp44_patch")
    p.add_argument("--confirm", action="store_true")
    p.add_argument("--samples", type=int, default=pt.BOOT_SAMPLES)
    p.add_argument("--tag", default=None)
    args = p.parse_args(argv)
    root = Path(args.root)
    text, numbers = build(root, args.confirm, args.samples)
    tag = args.tag or ("confirm" if args.confirm else "dev")
    (root / f"report_{tag}.md").write_text(text)
    (root / f"report_{tag}.json").write_text(json.dumps(numbers, indent=1))
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
