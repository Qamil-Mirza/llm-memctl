#!/usr/bin/env python
"""§43 logit-lens profile of the gold answer at P1, one family against Qwen 2.5 3B on the SAME dev rows (folds 0-1).

    .venv/bin/python scripts/exp43_lens_profile.py llama3b

Reads runs/exp43_lens_dev/<family>/ (scripts/exp43_lens_dev.py; identity lens = logit lens) and the logit variant of
runs/exp34_jlens/qwen3b/ restricted to the family's dev population question ids. Per layer: the share of questions
with the gold first token in the top-10 and at top-1, the median gold rank among the reader's successes (judge
correct), and the share of the reader's refusals (judge `unknown`) where any of its refusal ids (" unknown",
" Unknown") is top-1. Plus §34's A/B/C split at P1 (A never in the top-10, B in it then lost, C in it at the final
layer; `jlens_readout.outcome`, §34 classes `exp34_report.classes_of`) and the logit-lens emergence depth.
Depth is §34's convention, layer / n_layers (`jlens_readout.emergence_depth`), so the final layer is (n-1)/n.
Writes runs/exp43_lens_dev/<family>/profile.md and profile.json. Folds 2-4 are refused. Descriptive only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import median

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memctl.analysis import families  # noqa: E402
from memctl.analysis.jlens_readout import OUTCOMES, emergence_depth, outcome  # noqa: E402
from memctl.jacobian_data import family_population, load_records  # noqa: E402
from scripts.exp34_report import _quartiles, classes_of  # noqa: E402
from scripts.exp43_lens_dev import DEV_FOLDS, READERS, check_folds, dev_population  # noqa: E402

QWEN_DIR = "runs/exp34_jlens/qwen3b"
K = 10
CLASS_ORDER = ("success", "refusal", "answered wrong", "all failures")


# --- pure arithmetic (tested in tests/test_exp43_lens_profile.py) --------------------------------------------------

ARTICLES = {"The", "the", "A", "a", "An", "an"}


def layer_profile(rows: list[dict], cls: dict[str, str], n_layers: int, refusal_ids, k: int = K) -> list[dict]:
    """rows: {qid, gold_ranks [L], top1 [L]}; cls: qid -> 'success' | 'refusal' | 'answered wrong'."""
    refusal_ids = set(int(i) for i in refusal_ids)
    out = []
    successes = [r for r in rows if cls[r["qid"]] == "success"]
    refusals = [r for r in rows if cls[r["qid"]] == "refusal"]
    for layer in range(n_layers):
        ranks = [int(r["gold_ranks"][layer]) for r in rows]
        out.append({
            "layer": layer, "depth": layer / n_layers, "n": len(rows),
            "gold_topk": float(np.mean([x <= k for x in ranks])) if rows else None,
            "gold_top1": float(np.mean([x == 1 for x in ranks])) if rows else None,
            "median_rank_successes": float(median(int(r["gold_ranks"][layer]) for r in successes)) if successes else None,
            "n_refusals": len(refusals),
            "refusal_unknown_top1": (float(np.mean([int(r["top1"][layer]) in refusal_ids for r in refusals]))
                                     if refusals else None),
        })
    return out


def abc_split(rows: list[dict], cls: dict[str, str], k: int = K) -> dict[str, dict]:
    groups = {"all": [r["qid"] for r in rows]}
    for name in CLASS_ORDER:
        groups[name] = [r["qid"] for r in rows if (cls[r["qid"]] == name or (name == "all failures" and cls[r["qid"]] != "success"))]
    by_q = {r["qid"]: r for r in rows}
    table = {}
    for name, qids in groups.items():
        found = [outcome(np.asarray(by_q[q]["gold_ranks"]), k) for q in qids]
        table[name] = {"n": len(found), **{o: (sum(x == o for x in found) / len(found)) if found else None for o in OUTCOMES}}
    return table


def emergence(rows: list[dict], n_layers: int, k: int = K) -> dict:
    depths = [emergence_depth(np.asarray(r["gold_ranks"]), list(range(n_layers)), n_layers, k) for r in rows]
    return {**_quartiles([d for d in depths if d is not None]), "n_read": len(rows)}


# --- loading ----------------------------------------------------------------------------------------------------

def load_row(read_dir: Path, qid: str, variant: str = "logit") -> dict:
    meta = json.loads((read_dir / f"{qid}.json").read_text())
    if meta["fold"] not in DEV_FOLDS:
        raise ValueError(f"{qid}: fold {meta['fold']} is not a dev fold")
    names = meta.get("tracked_names", ["gold", "unknown_sp", "unknown"])
    with np.load(read_dir / f"{qid}.npz") as z:
        layers = z["layers"].tolist()
        p1 = 0                                            # POSITION_NAMES order: P1 is the first slot
        assert int(z["position_index"][p1]) == meta["positions"]["P1"][0]
        gold_ranks = z[f"ranks_{variant}"][:, p1, names.index("gold")].astype(int)
        top1 = z[f"top_ids_{variant}"][:, p1, 0].astype(int)
    assert layers == list(range(len(layers))), f"{qid}: layers are not 0..L-1"
    return {"qid": qid, "gold_ranks": gold_ranks, "top1": top1, "gold_token_text": meta["gold_token_text"],
            "prompt_sha256": meta.get("intact_prompt_sha256", meta.get("prompt_sha256")), "n_layers": len(layers)}


def qwen_dev_records(qids: set[str]):
    records = [r for f in DEV_FOLDS for r in load_records("qwen3b", f)[0]]
    return {r.question_id: r for r in family_population(records) if r.question_id in qids}


def profile_reader(rows, by_q, n_layers, refusal_ids) -> dict:
    qids = [r["qid"] for r in rows]
    classes = classes_of(by_q, qids)
    cls = {}
    for name in ("success", "refusal", "answered wrong"):
        for q in classes[name]:
            cls[q] = name
    return {"n": len(rows), "n_layers": n_layers, "refusal_ids": list(refusal_ids),
            "class_counts": {name: len(classes[name]) for name in CLASS_ORDER},
            "layers": layer_profile(rows, cls, n_layers, refusal_ids),
            "abc": abc_split(rows, cls), "emergence": emergence(rows, n_layers)}


# --- report -----------------------------------------------------------------------------------------------------

def _f(x, d=2):
    return "-" if x is None else f"{x:.{d}f}"


def _rank(x):
    return "-" if x is None else f"{x:,.0f}"


def layer_table(name: str, prof: dict) -> list[str]:
    n = prof["n_layers"]
    lines = [f"| depth | {name} layer (of {n}) | logit: gold in top-10 | logit: gold top-1 | logit: median gold rank, "
             f"successes (n={prof['class_counts']['success']}) | any unknown id top-1 on refusals "
             f"(n={prof['class_counts']['refusal']}) |", "|---|---|---|---|---|---|"]
    for row in prof["layers"]:
        lines.append(f"| {row['depth']:.3f} | {row['layer']} | {_f(row['gold_topk'])} | {_f(row['gold_top1'])} | "
                     f"{_rank(row['median_rank_successes'])} | {_f(row['refusal_unknown_top1'])} |")
    return lines


def abc_table(profiles: dict[str, dict]) -> list[str]:
    lines = ["| reader | class | n | A | B | C |", "|---|---|---|---|---|---|"]
    for reader, prof in profiles.items():
        for cls in ("all",) + CLASS_ORDER:
            row = prof["abc"][cls]
            lines.append(f"| {reader} | {cls} | {row['n']} | {_f(row['A'], 3)} | {_f(row['B'], 3)} | {_f(row['C'], 3)} |")
    return lines


def build(family: str, read_root: str = "runs/exp43_lens_dev", qwen_dir: str = QWEN_DIR) -> tuple[str, dict]:
    cfg = families.get(family)
    check_folds(DEV_FOLDS)
    pop, _ = dev_population(cfg)
    by_q = {r.question_id: r for r in pop}
    read_dir = Path(read_root) / family
    missing = [q for q in by_q if not (read_dir / f"{q}.json").exists()]
    if missing:
        raise SystemExit(f"{family}: {len(missing)} of {len(by_q)} dev rows not read yet (e.g. {missing[:3]})")
    rows = [load_row(read_dir, q) for q in sorted(by_q)]
    n_layers = rows[0]["n_layers"]
    assert n_layers == cfg.n_layers
    qwen_by_q = qwen_dev_records(set(by_q))
    if set(qwen_by_q) != set(by_q):
        raise SystemExit(f"Qwen 3B dev population differs from {family}'s: {len(set(by_q) ^ set(qwen_by_q))} ids")
    qrows = [load_row(Path(qwen_dir), q) for q in sorted(by_q)]
    qcfg = families.QWEN3B
    profiles = {family: profile_reader(rows, by_q, n_layers, cfg.refusal_ids),
                "qwen3b": profile_reader(qrows, qwen_by_q, qrows[0]["n_layers"], qcfg.refusal_ids)}
    # like-for-like checks: same prompt text, same gold first-token text
    same_prompt = sum(a["prompt_sha256"] == b["prompt_sha256"] for a, b in zip(rows, qrows))
    same_tok = [(a, b) for a, b in zip(rows, qrows) if a["gold_token_text"].strip() == b["gold_token_text"].strip()]
    numeric = [(a, b) for a, b in zip(rows, qrows) if by_q[a["qid"]].gold.strip()[:1].isdigit()]
    final_topk = lambda rs: float(np.mean([int(r["gold_ranks"][-1]) <= K for r in rs])) if rs else None
    checks = {"same_prompt_sha256": same_prompt, "n": len(rows),
              "same_gold_first_token_text": len(same_tok), "numeric_gold": len(numeric),
              "final_topk_same_token_subset": {family: final_topk([a for a, _ in same_tok]),
                                               "qwen3b": final_topk([b for _, b in same_tok])},
              "final_topk_numeric_subset": {family: final_topk([a for a, _ in numeric]),
                                            "qwen3b": final_topk([b for _, b in numeric])}}
    # article-first golds (Advisor's caveat on the Gemma profile, 88fa7b5): the §34 target is the gold's first token,
    # and when that is an article the final layer often predicts the content word instead, so the article drops out of
    # the top-10 late (a B row) or never enters it (an A row). A reporting split; the read protocol is unchanged.
    def art_split(rs):
        if not rs:
            return {"n": 0}
        outs = [outcome(np.asarray(r["gold_ranks"]), K) for r in rs]
        return {"n": len(rs), "final_top10": final_topk(rs),
                "final_top1": float(np.mean([int(r["gold_ranks"][-1]) == 1 for r in rs])),
                "A": outs.count("A") / len(rs), "B": outs.count("B") / len(rs)}
    is_art = lambda r: r["gold_token_text"].strip() in ARTICLES  # noqa: E731
    checks["article_split"] = {name: {"article": art_split([r for r in rs if is_art(r)]),
                                      "non_article": art_split([r for r in rs if not is_art(r)])}
                               for name, rs in ((family, rows), ("qwen3b", qrows))}
    b_rows = [r for r in rows if outcome(np.asarray(r["gold_ranks"]), K) == "B"]
    checks["B_rows_article_first"] = {"B": len(b_rows), "article": sum(is_art(r) for r in b_rows)}
    payload = {"family": family, "folds": list(DEV_FOLDS), "k": K, "lens": "logit (identity; no J-lens fitted)",
               "depth_convention": "layer / n_layers (§34 emergence_depth)", "profiles": profiles, "checks": checks}

    md = [f"# §43 logit-lens profile at P1: {family} vs Qwen 2.5 3B, dev folds 0-1 only", "",
          f"Descriptive; no claims. {len(rows)} dev population rows (§35 rule: answerable, all evidence in view, judge "
          f"label), the same question ids for both readers. Forced mode (an appended `Answer:`), position P1, k = {K}, "
          f"logit lens only (identity lens; no J-lens fitted). Qwen rows: the logit variant of `{qwen_dir}`.",
          f"Depth = layer / n_layers (§34's convention); the final layer is (n-1)/n. The 2026-10-26 briefing appendix "
          f"labels depth layer / (n-1), so its depth column runs to 1.00; compare by layer index or by this column.",
          f"Classes (§34 `classes_of`, judge labels): success / refusal (judge `unknown`) / answered wrong. Refusal ids: "
          f"{family} {list(cfg.refusal_ids)}, qwen3b {list(qcfg.refusal_ids)} (\" unknown\", \" Unknown\").", ""]
    for name in (family, "qwen3b"):
        md += [f"## {name}", ""] + layer_table(name, profiles[name]) + [""]
    md += ["## A/B/C at P1 (logit lens; A never in top-10, B in top-10 at some layer but not the last, C in top-10 at "
           "the last layer)", ""] + abc_table(profiles) + [""]
    md += ["## Logit-lens emergence depth of the gold at P1 (first layer with gold in top-10; questions where it emerges)",
           "", "| reader | n emerged | n read | q1 | median | q3 |", "|---|---|---|---|---|---|"]
    for name, prof in profiles.items():
        e = prof["emergence"]
        md.append(f"| {name} | {e['n']} | {e['n_read']} | {_f(e['q1'], 3)} | {_f(e['median'], 3)} | {_f(e['q3'], 3)} |")
    md += ["", "## Like-for-like checks", "",
           f"- Prompt text identical (sha256) for {same_prompt} of {len(rows)} rows.",
           f"- Gold first-token text identical (stripped) for {len(same_tok)} of {len(rows)} rows; final-layer gold in "
           f"top-10 on that subset: {family} {_f(checks['final_topk_same_token_subset'][family])}, qwen3b "
           f"{_f(checks['final_topk_same_token_subset']['qwen3b'])}.",
           f"- Digit tokenisation caveat: Llama 3.2 and Phi-4-mini put up to 3 digits in one token, Qwen 2.5 and Gemma 3 "
           f"one digit per token. On the {len(numeric)} rows whose gold starts with a digit the 'gold first token' is "
           f"the whole number (up to 3 digits) for Llama/Phi and only its first digit for Qwen/Gemma, so a top-k hit is "
           f"a harder target for Llama/Phi there. Final-layer gold in top-10 on numeric golds: {family} "
           f"{_f(checks['final_topk_numeric_subset'][family])}, qwen3b {_f(checks['final_topk_numeric_subset']['qwen3b'])}.",
           f"- Article-first golds (caveat, a reporting split; the §34 target token is unchanged): when the gold's first "
           f"token is an article (' The', ' A', ' a', ...), the final layer often predicts the content word instead and "
           f"the article drops out of the top-10, which shows up as a B (or A) row. Of {family}'s "
           f"{checks['B_rows_article_first']['B']} B rows, {checks['B_rows_article_first']['article']} have an "
           f"article-first gold. Split (final top-10 / final top-1 / A / B):",
           "", "| reader | article n | top-10 | top-1 | A | B | non-article n | top-10 | top-1 | A | B |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, sp in checks["article_split"].items():
        a, n_ = sp["article"], sp["non_article"]
        cells = lambda d: [str(d["n"])] + ([_f(d.get(x), 3) for x in ("final_top10", "final_top1", "A", "B")]  # noqa: E731
                                          if d["n"] else ["-"] * 4)
        md.append("| " + " | ".join([name] + cells(a) + cells(n_)) + " |")
    md += [""]
    return "\n".join(md), payload


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("family", choices=READERS)
    p.add_argument("--read-root", default="runs/exp43_lens_dev")
    p.add_argument("--qwen-dir", default=QWEN_DIR)
    args = p.parse_args(argv)
    md, payload = build(args.family, args.read_root, args.qwen_dir)
    out = Path(args.read_root) / args.family
    (out / "profile.md").write_text(md)
    (out / "profile.json").write_text(json.dumps(payload, indent=1))
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
