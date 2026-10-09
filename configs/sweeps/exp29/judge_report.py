"""§29 judge pass and grading (pre-registered; margin -0.03; 95% paired bootstrap by question, as §23).
    PYTHONPATH=. python configs/sweeps/exp29/judge_report.py judge --base-url URL/v1 [--backend stub] [--prefix exp29]
    PYTHONPATH=. python configs/sweeps/exp29/judge_report.py report [--prefix exp29]
The judge is Qwen2.5-7B with the official LongMemEval prompts (the §23/§27 judge). It sees the question, the gold
answer and the answer text, never which arm wrote it; the cache is keyed by those texts. Answerable questions only.
The report is run once, after the judge pass: P(all in view) per arm is read there, from the evaluation's own
episodes, and nowhere earlier."""
import argparse
import json
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from memctl.envs.longmemeval import _index, _instance
from memctl.splits import fold_indices

PATH = "data/longmemeval/longmemeval_s_cleaned.json"
ARMS = ("fifo_top5_t3000", "fixed8", "utility_head", "blend_head", "evidence_rows")
MARGIN = -0.03

parser = argparse.ArgumentParser()
parser.add_argument("stage", choices=("judge", "report"))
parser.add_argument("--base-url")
parser.add_argument("--backend", default="openai")
parser.add_argument("--runs", default="runs")
parser.add_argument("--prefix", default="exp29")
parser.add_argument("--cache", default="cache/judge_exp29")
parser.add_argument("--out", type=Path)
args = parser.parse_args()
out = args.out or Path(args.runs) / f"{args.prefix}_verdicts.json"
index = _index(PATH)


def cells():
    for k in range(5):
        chosen = fold_indices(index, 5, k)
        for arm in ARMS:
            yield k, chosen, arm, Path(args.runs) / f"{args.prefix}_qwen7b_f{k}" / f"{arm}__fraction0.05"


if args.stage == "judge":
    from memctl.judge import Judge

    rows = []
    for k, chosen, arm, cell in cells():
        if not (cell / "steps.jsonl").exists():
            continue
        for line in open(cell / "steps.jsonl"):
            step = json.loads(line)
            if not step.get("scored"):
                continue
            number = chosen[step["seed"] % len(chosen)]
            meta = index[number]
            answer = step["agent_action"] or ""
            row = {"arm": arm, "fold": k, "question_id": meta["question_id"], "type": meta["question_type"],
                   "prompt_tokens": step["agent_prompt_tokens"], "answer": answer,
                   "unknown": answer.strip().lower().startswith("unknown"),
                   "abstention": meta["question_id"].endswith("_abs")}
            if not row["abstention"]:
                instance = _instance(PATH, number)
                row["question"] = f"(asked on {instance.get('question_date', '')}) {instance['question']}"
                row["gold"] = str(instance["answer"])
            rows.append(row)
    judge = Judge({"backend": args.backend, "name": "qwen2.5-7b-instruct", "base_url": args.base_url,
                   "timeout_s": 300, "cache_dir": args.cache, "style": "longmemeval"})
    graded = [r for r in rows if not r["abstention"]]
    with ThreadPoolExecutor(32) as pool:
        verdicts = list(pool.map(lambda r: judge.is_correct(r["question"], r["gold"], r["answer"], r["type"]), graded))
    for r, v in zip(graded, verdicts):
        r["correct"] = bool(v)
    out.write_text(json.dumps(rows))
    print(len(graded), "answers judged;", len(rows) - len(graded), "abstention answers kept unjudged")
    raise SystemExit

V = json.load(open(out))
cell = defaultdict(dict)
false_answers = defaultdict(int)
for r in V:
    if r["abstention"]:
        false_answers[r["arm"]] += not r["unknown"]
    else:
        cell[r["arm"]][r["question_id"]] = r
in_view = defaultdict(dict)
for k, chosen, arm, folder in cells():
    for line in open(folder / "episodes.jsonl"):
        e = json.loads(line)
        in_view[arm][index[chosen[e["seed"] % len(chosen)]]["question_id"]] = (e.get("evidence_complete_rate") or 0) >= 1
m = lambda v: sum(map(float, v)) / len(v) if v else float("nan")


def boot(values, n=10000):
    rng = random.Random(0)
    d = sorted(sum(rng.choices(values, k=len(values))) / len(values) for _ in range(n))
    return sum(values) / len(values), d[int(0.025 * n)], d[int(0.975 * n) - 1]


def diff(a, b):
    qs = sorted(set(cell[a]) & set(cell[b]))
    return boot([float(cell[a][q]["correct"]) - float(cell[b][q]["correct"]) for q in qs])


f = lambda d: f"{d[0]:+.3f} ({d[1]:+.3f}, {d[2]:+.3f})"


def verdict(d):
    if d[1] > 0:
        return "BETTER"
    if d[2] < 0:
        return "WORSE"
    return "NON-INFERIOR" if d[1] > MARGIN else "UNDETERMINED"


print("| arm | accuracy (95% CI) | unknown | P(correct \\| answered) | P(all in view) | P(correct \\| in view) | "
      "prompt tokens | false answers on the unanswerable |\n|---|---|---|---|---|---|---|---|")
for arm in ARMS:
    rows = list(cell[arm].values())
    ans = [r for r in rows if not r["unknown"]]
    iv = [r for r in rows if in_view[arm][r["question_id"]]]
    a = boot([float(r["correct"]) for r in rows])
    print(f"| {arm} | {a[0]:.3f} ({a[1]:.3f}, {a[2]:.3f}) | {m([r['unknown'] for r in rows]):.3f} | "
          f"{m([r['correct'] for r in ans]):.3f} | {len(iv) / len(rows):.3f} | {m([r['correct'] for r in iv]):.3f} | "
          f"{m([r['prompt_tokens'] for r in rows]):,.0f} | {false_answers[arm]} |")
claims = [
    ("C1, utility head − fixed8 (direction undetermined)", "utility_head", "fixed8"),
    ("C2, blend head − fixed8 (expected ≥)", "blend_head", "fixed8"),
    ("C3, utility head − FIFO + floor (expected better)", "utility_head", "fifo_top5_t3000"),
    ("C4, blend head − FIFO + floor (expected better)", "blend_head", "fifo_top5_t3000"),
]
print()
for name, a, b in claims:
    d = diff(a, b)
    print(f"- {name}: {f(d)} → **{verdict(d)}**")
print("\nDescriptive (no claim):")
for name, a, b in [("evidence-rows head − fixed8 (the rows trainer against §19's)", "evidence_rows", "fixed8"),
                   ("utility head − evidence-rows head (the signal alone, same trainer)", "utility_head", "evidence_rows"),
                   ("fixed8 − FIFO + floor (pairing check; §23 +0.102, §27 +0.100)", "fixed8", "fifo_top5_t3000")]:
    print(f"- {name}: {f(diff(a, b))}")
