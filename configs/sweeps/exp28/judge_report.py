"""Experiment 28 (§28): the one blind Qwen 7B judge pass, and the pre-registered report.
    python configs/sweeps/exp28/judge_report.py judge --base-url URL/v1 --out runs/exp28_verdicts.json
    python configs/sweeps/exp28/judge_report.py report --verdicts runs/exp28_verdicts.json
The judge is the §23 judge (Qwen2.5-7B, official LongMemEval prompts). It sees the question, the gold answer and the
answer text, never the arm. Claims (margin -0.03, 95% paired bootstrap by question): fixed8_rewrite minus fixed8 on
the 470 answerable questions, and on the temporal-reasoning subset (the primary subgroup)."""
import argparse
import json
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from memctl.envs.longmemeval import _index, _instance
from memctl.splits import fold_indices

PATH = "data/longmemeval/longmemeval_s_cleaned.json"
ARMS = ("fixed8", "fixed8_rewrite")
BASE = "fixed8"


def rows_of(prefix: str, runs: str = "runs") -> list[dict]:
    index, rows = _index(PATH), []
    for k in range(5):
        chosen = fold_indices(index, 5, k)
        for arm in ARMS:
            cell = Path(runs) / f"{prefix}_qwen7b_f{k}" / f"{arm}__fraction0.05"
            if not (cell / "steps.jsonl").exists():
                continue
            in_view = {}
            for line in open(cell / "episodes.jsonl"):
                e = json.loads(line)
                in_view[e["seed"]] = (e.get("evidence_complete_rate") or 0) >= 1
            for line in open(cell / "steps.jsonl"):
                step = json.loads(line)
                if not step.get("scored"):
                    continue
                number = chosen[step["seed"] % len(chosen)]
                meta, instance = index[number], _instance(PATH, number)
                answer = step["agent_action"] or ""
                rows.append({"arm": arm, "fold": k, "question_id": meta["question_id"], "type": meta["question_type"],
                             "abstention": meta["question_id"].endswith("_abs"),
                             "question": f"(asked on {instance.get('question_date', '')}) {instance['question']}",
                             "gold": str(instance["answer"]), "answer": answer,
                             "prompt_tokens": step["agent_prompt_tokens"], "in_view": in_view.get(step["seed"]),
                             "unknown": answer.strip().lower().startswith("unknown")})
    return rows


def judge(args) -> None:
    from memctl.judge import Judge

    rows = rows_of(args.prefix)
    graded = [r for r in rows if not r["abstention"]]
    j = Judge({"backend": args.backend, "name": "qwen2.5-7b-instruct", "base_url": args.base_url, "timeout_s": 300,
               "cache_dir": args.cache, "style": "longmemeval"})
    with ThreadPoolExecutor(32) as pool:
        verdicts = list(pool.map(lambda r: j.is_correct(r["question"], r["gold"], r["answer"], r["type"]), graded))
    for r, v in zip(graded, verdicts):
        r["correct"] = bool(v)
    Path(args.out).write_text(json.dumps(rows))
    print(len(graded), "answers judged;", len(rows) - len(graded), "abstention answers kept unjudged")


def boot(values, n=10000):
    rng = random.Random(0)
    d = sorted(sum(rng.choices(values, k=len(values))) / len(values) for _ in range(n))
    return sum(values) / len(values), d[int(0.025 * n)], d[int(0.975 * n) - 1]


def verdict(d) -> str:
    return "BETTER" if d[1] > 0 else "NON-INFERIOR" if d[1] > -0.03 else ("fixed8 BETTER" if d[2] < 0 else "NOT SHOWN")


def report(args) -> None:
    rows = json.load(open(args.verdicts))
    cell = defaultdict(dict)
    for r in rows:
        if not r["abstention"]:
            cell[r["arm"]][r["question_id"]] = r
    mean = lambda v: sum(map(float, v)) / len(v) if v else float("nan")
    print("| arm | accuracy (95% CI) | temporal accuracy | unknown | P(all in view) | prompt tokens | false answers on unanswerable |")
    print("|---|---|---|---|---|---|---|")
    for arm in ARMS:
        rs = list(cell[arm].values())
        if not rs:
            continue
        a = boot([float(r["correct"]) for r in rs])
        t = [float(r["correct"]) for r in rs if r["type"] == "temporal-reasoning"]
        false = sum(1 for r in rows if r["arm"] == arm and r["abstention"] and not r["unknown"])
        print(f"| {arm} | {a[0]:.3f} ({a[1]:.3f}, {a[2]:.3f}) | {mean(t):.3f} (n={len(t)}) | "
              f"{mean([r['unknown'] for r in rs]):.3f} | {mean([r['in_view'] for r in rs]):.3f} | "
              f"{mean([r['prompt_tokens'] for r in rs]):,.0f} | {false} of 30 |")
    for arm in ARMS:
        if arm == BASE or not cell[arm]:
            continue
        for name, keep in (("all answerable", lambda r: True), ("temporal-reasoning (primary subgroup)",
                                                               lambda r: r["type"] == "temporal-reasoning")):
            qs = sorted(q for q in set(cell[arm]) & set(cell[BASE]) if keep(cell[BASE][q]))
            d = boot([float(cell[arm][q]["correct"]) - float(cell[BASE][q]["correct"]) for q in qs])
            print(f"- {arm} minus {BASE}, {name}, n={len(qs)}: {d[0]:+.3f} ({d[1]:+.3f}, {d[2]:+.3f}) → **{verdict(d)}**")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("judge")
    p.add_argument("--base-url"); p.add_argument("--backend", default="openai"); p.add_argument("--out", required=True)
    p.add_argument("--prefix", default="exp28"); p.add_argument("--cache", default="cache/judge_exp28")
    p = sub.add_parser("report"); p.add_argument("--verdicts", default="runs/exp28_verdicts.json")
    args = parser.parse_args()
    judge(args) if args.command == "judge" else report(args)
