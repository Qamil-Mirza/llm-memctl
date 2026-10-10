"""§32 reader-free report: per fold and pooled, each arm against fixed8 (paired 95% bootstrap by question, 10,000
resamples, seed 0), plus the sensitivity row (test questions sharing no answer session with any training question).
    PYTHONPATH=. python configs/sweeps/exp32/free_report.py > runs/exp32_rerankers/free_report.md
Reads runs/exp32_free_f{k}/<arm>__fraction0.05/episodes.jsonl, runs/exp32_latency (select_ms per question),
runs/exp32_rerankers/split_audit.json."""
import argparse
import json
import random
import statistics
from pathlib import Path

from memctl.envs.longmemeval import _index
from memctl.splits import fold_indices

ARMS = ["fixed8", "bm25_top8", "rrf_top8", "lr_pointwise", "gbdt_pointwise", "cross_encoder_zero"]
MEASURES = [("all", "P(all in view)"), ("recall", "requirement recall"), ("precision", "precision of the 8"),
            ("tokens", "prompt tokens")]

parser = argparse.ArgumentParser()
parser.add_argument("--runs", default="runs")
args = parser.parse_args()
runs = Path(args.runs)
index = _index("data/longmemeval/longmemeval_s_cleaned.json")
data = {arm: {} for arm in ARMS}
fold_of = {}
for arm in ARMS:
    for k in range(5):
        chosen = fold_indices(index, 5, k)
        for line in open(runs / f"exp32_free_f{k}" / f"{arm}__fraction0.05" / "episodes.jsonl"):
            e = json.loads(line)
            qid = index[chosen[e["seed"] % len(chosen)]]["question_id"]
            fold_of[qid] = k
            data[arm][qid] = {"all": float((e.get("evidence_complete_rate") or 0) >= 1),
                              "recall": e.get("needed_hit_rate") or 0.0,
                              "precision": e.get("retrieval_precision") or 0.0,
                              "tokens": float(e["tokens_processed"]), "forced": e["forced_evictions"],
                              "labelled": e.get("evidence_complete_rate") is not None}
audit = json.load(open(runs / "exp32_rerankers" / "split_audit.json"))
clean = {q for q, flags in audit["questions"].items() if not flags["shares_answer_session"]}


def boot(values, n=10000):
    rng = random.Random(0)
    d = sorted(sum(rng.choices(values, k=len(values))) / len(values) for _ in range(n))
    return sum(values) / len(values), d[int(0.025 * n)], d[int(0.975 * n) - 1]


def table(questions, title):
    labelled = [q for q in questions if data["fixed8"][q]["labelled"]]
    print(f"\n**{title}** (n = {len(labelled)} questions with evidence labels; Δ = arm minus fixed8, paired 95% CI)\n")
    print("| arm | " + " | ".join(f"{name} | Δ" for _, name in MEASURES) + " | forced removals / q |")
    print("|---|" + "---|---|" * len(MEASURES) + "---|")
    for arm in ARMS:
        cells = []
        for key, _ in MEASURES:
            mean = statistics.fmean(data[arm][q][key] for q in labelled)
            fmt = (lambda x: f"{x:,.0f}") if key == "tokens" else (lambda x: f"{x:.3f}")
            if arm == "fixed8":
                cells.append(f"{fmt(mean)} | —")
            else:
                d = boot([data[arm][q][key] - data["fixed8"][q][key] for q in labelled])
                sign = (lambda x: f"{x:+,.0f}") if key == "tokens" else (lambda x: f"{x:+.3f}")
                cells.append(f"{fmt(mean)} | {sign(d[0])} ({sign(d[1])}, {sign(d[2])})")
        forced = statistics.fmean(data[arm][q]["forced"] for q in labelled)
        print(f"| {arm} | " + " | ".join(cells) + f" | {forced:.3f} |")


everything = sorted(data["fixed8"])
table(everything, "Pooled, all five test folds")
table([q for q in everything if q in clean], "Sensitivity: test questions sharing no answer session with a training question")
print("\n**Per fold, P(all in view)**\n")
print("| arm | " + " | ".join(f"fold {k}" for k in range(5)) + " |\n|---|" + "---|" * 5)
for arm in ARMS:
    row = [statistics.fmean(data[arm][q]["all"] for q in everything if fold_of[q] == k and data[arm][q]["labelled"])
           for k in range(5)]
    print(f"| {arm} | " + " | ".join(f"{x:.3f}" for x in row) + " |")
latency = runs / "exp32_latency"
if latency.exists():
    print("\n**Selection latency** (fold 0's test part, one process, one thread, fresh caches; first question dropped as"
          " warm-up; ms per question; BM25 search excluded, as it is common to all arms)\n")
    print("| arm | median ms | mean ms | p90 ms | BM25 search median ms |\n|---|---|---|---|---|")
    for cell in sorted(latency.glob("*__fraction0.05")):
        steps = [json.loads(line) for line in open(cell / "steps.jsonl")]
        info = [s["controller_info"] for s in steps if s.get("requires_response") and "select_ms" in s["controller_info"]]
        ms = sorted(i["select_ms"] for i in info[1:])
        search = statistics.median(i["search_ms"] for i in info[1:])
        print(f"| {cell.name.split('__')[0]} | {statistics.median(ms):.2f} | {statistics.fmean(ms):.2f} | "
              f"{ms[int(0.9 * len(ms))]:.2f} | {search:.2f} |")
