"""A second-family judge over answers already generated (EXPERIMENTS.md §18, pre-registered).

Samples LongMemEval answers from finished sweep cells, stratified by question type, and asks a second judge
(official LongMemEval prompts) to grade (a) the answer, (b) a planted wrong answer, another question's gold of
the same type, and (c) the question's own gold. Reports agreement with the first judge (share and Cohen's
kappa), the false-accept and false-reject rates, and each controller's accuracy under both judges.

    python -m memctl.judge_check --base-url URL/v1 --model ibm-granite/granite-3.1-8b-instruct --out runs/exp18_judge.json
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from memctl.envs.longmemeval import _index, _instance
from memctl.judge import Judge
from memctl.runlog import open_trace
from memctl.splits import fold_indices
from memctl.sysinfo import collect_metadata

PATH = "data/longmemeval/longmemeval_s_cleaned.json"
CELLS = {  # controller -> (sweep template, cell)
    "keep_last0_top5": ("exp15_paid_f{k}", "keep_last0_top5__fraction0.05"),
    "head_listsum": ("exp15_paid_f{k}", "head_listsum__fraction0.05"),
    "fifo_top5_5pct": ("exp13_lme_f{k}", "fifo_top5__fraction0.05"),
    "oracle_2pct": ("exp13_lme_f{k}", "oracle-approx__fraction0.02"),
}


def answers(runs: Path, controller: str) -> list[dict]:
    """Every scored, non-abstention answer of a controller with the first judge's verdict."""
    index, out = _index(PATH), []
    template, cell = CELLS[controller]
    for k in range(5):
        folder = runs / template.format(k=k) / cell
        chosen = fold_indices(index, 5, k)
        failed = {json.loads(line)["seed"] for line in open(folder / "failures.jsonl")}
        for line in open_trace(folder / "steps.jsonl"):
            step = json.loads(line)
            if not step.get("scored"):
                continue
            number = chosen[step["seed"] % len(chosen)]
            meta = index[number]
            if meta["question_id"].endswith("_abs"):
                continue
            instance = _instance(PATH, number)
            out.append({"controller": controller, "question_id": meta["question_id"], "type": meta["question_type"],
                        "question": f"(asked on {instance.get('question_date', '')}) {instance['question']}",
                        "gold": str(instance["answer"]), "answer": step["agent_action"] or "",
                        "first_judge": step["seed"] not in failed})
    return out


def stratified(rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    by_type = defaultdict(list)
    for row in rows:
        by_type[row["type"]].append(row)
    picked = []
    for kind, members in sorted(by_type.items()):
        share = max(1, round(n * len(members) / len(rows)))
        picked += rng.sample(members, min(share, len(members)))
    rng.shuffle(picked)
    return picked[:n]


def kappa(a: list[bool], b: list[bool]) -> float:
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--per-controller", type=int, default=75)
    parser.add_argument("--cache-dir", default="cache/judge_granite")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--backend", default="openai", help="stub for a dry run with no server")
    args = parser.parse_args()
    rng = random.Random(0)
    # The same questions for every controller, so controllers compare paired under each judge.
    per = {controller: {row["question_id"]: row for row in answers(args.runs, controller)} for controller in CELLS}
    common = sorted(set.intersection(*[set(rows) for rows in per.values()]))
    questions = stratified([per["keep_last0_top5"][qid] for qid in common], args.per_controller, rng)
    sample = [dict(per[controller][q["question_id"]]) for controller in CELLS for q in questions]
    golds_by_type = defaultdict(list)
    for row in sample:
        golds_by_type[row["type"]].append((row["question_id"], row["gold"]))
    for row in sample:  # a planted wrong answer: another question's gold, same type
        others = [g for qid, g in golds_by_type[row["type"]] if qid != row["question_id"] and g != row["gold"]]
        row["planted"] = rng.choice(others)
    judge = Judge({"backend": args.backend, "name": args.model, "base_url": args.base_url, "timeout_s": 300,
                   "cache_dir": args.cache_dir, "style": "longmemeval"})
    jobs = [(row, field) for row in sample for field in ("answer", "planted", "gold")]
    with ThreadPoolExecutor(32) as pool:
        verdicts = list(pool.map(lambda job: judge.is_correct(job[0]["question"], job[0]["gold"], job[0][job[1]], job[0]["type"]), jobs))
    for (row, field), verdict in zip(jobs, verdicts):
        row[f"second_{field}"] = bool(verdict)
    first = [r["first_judge"] for r in sample]
    second = [r["second_answer"] for r in sample]
    by_controller = {}
    for controller in CELLS:
        rows = [r for r in sample if r["controller"] == controller]
        by_controller[controller] = {"n": len(rows), "first_judge_accuracy": sum(r["first_judge"] for r in rows) / len(rows),
                                     "second_judge_accuracy": sum(r["second_answer"] for r in rows) / len(rows)}
    def paired(field: str) -> dict:
        head = {r["question_id"]: r[field] for r in sample if r["controller"] == "head_listsum"}
        rule = {r["question_id"]: r[field] for r in sample if r["controller"] == "keep_last0_top5"}
        d = [float(head[q]) - float(rule[q]) for q in sorted(head)]
        draws = sorted(sum(random.Random(i).choices(d, k=len(d))) / len(d) for i in range(4000))
        return {"difference": sum(d) / len(d), "ci": [draws[100], draws[3899]]}

    report = {
        "judge": args.model, "n": len(sample),
        "agreement": sum(a == b for a, b in zip(first, second)) / len(sample), "kappa": kappa(first, second),
        "false_accept_rate": sum(r["second_planted"] for r in sample) / len(sample),
        "false_reject_rate": 1 - sum(r["second_gold"] for r in sample) / len(sample),
        "by_controller": by_controller,
        "head_minus_rule": {"first_judge": paired("first_judge"), "second_judge": paired("second_answer")},
        "metadata": collect_metadata(),
    }
    # Output 4 on every question: the head and the rule re-judged on all their common answers (the 74-question
    # sample has too little power for a paired interval). Added before the second judge ran (EXPERIMENTS §18).
    full = [per[c][qid] for c in ("head_listsum", "keep_last0_top5") for qid in common]
    with ThreadPoolExecutor(32) as pool:
        full_verdicts = list(pool.map(lambda r: judge.is_correct(r["question"], r["gold"], r["answer"], r["type"]), full))
    head = {r["question_id"]: (r["first_judge"], v) for r, v in zip(full, full_verdicts) if r["controller"] == "head_listsum"}
    rule = {r["question_id"]: (r["first_judge"], v) for r, v in zip(full, full_verdicts) if r["controller"] == "keep_last0_top5"}

    def full_paired(position: int) -> dict:
        d = [float(head[q][position]) - float(rule[q][position]) for q in sorted(head)]
        draws = sorted(sum(random.Random(i).choices(d, k=len(d))) / len(d) for i in range(4000))
        return {"n": len(d), "difference": sum(d) / len(d), "ci": [draws[100], draws[3899]],
                "head_accuracy": sum(head[q][position] for q in head) / len(head),
                "rule_accuracy": sum(rule[q][position] for q in rule) / len(rule)}

    report["head_minus_rule_all_questions"] = {"first_judge": full_paired(0), "second_judge": full_paired(1),
                                               "agreement": sum(a == b for a, b in list(head.values()) + list(rule.values()))
                                               / (len(head) + len(rule))}
    args.out.write_text(json.dumps({**report, "rows": sample}, indent=2))
    print(json.dumps(report | {"metadata": None}, indent=2))


if __name__ == "__main__":
    main()
