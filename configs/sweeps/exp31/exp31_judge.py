"""Experiment 31: one blind judge pass over the qTTT run's answers and the §23 rows it is compared with (§31).
    python configs/sweeps/exp31/exp31_judge.py --base-url URL/v1 --out runs/exp31_verdicts.json [--backend stub]
              [--prune] [--exp23-runs /home/qamil-mirza/Code/llm-memctl/runs]
The judge is Qwen2.5-7B with the official LongMemEval prompts (as §23). It sees the question, the gold answer and the
answer text, never which reader, arm or adaptation wrote it; the cache is keyed by those texts. Only the 470
answerable questions are judged; the 30 abstention rows are kept unjudged for the false-answer count. §23's Qwen 7B FIFO + floor and Qwen 3B `fixed8` answers are judged again in the same pass, so every
row of the claim sits on one judge pass; agreement with §23's verdicts is reported as a check.
--prune rewrites each finished cell's steps.jsonl to its scored rows only (disk: full step logs are ~1.7 MB a
question)."""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from memctl.envs.longmemeval import _index, _instance
from memctl.judge import Judge
from memctl.splits import fold_indices

PATH = "data/longmemeval/longmemeval_s_cleaned.json"
CELLS = {  # (reader row, folder pattern under --runs or --exp23-runs)
    ("qwen3b-hf", "fifo"): ("runs", "exp31_qwen3b_f{k}/fifo_top5_t3000__base__fraction0.05"),
    ("qwen3b-hf-qttt", "fifo"): ("runs", "exp31_qwen3b_f{k}/fifo_top5_t3000__qttt__fraction0.05"),
    ("qwen3b-hf", "fixed8"): ("runs", "exp31_qwen3b_f{k}/fixed8__base__fraction0.05"),
    ("qwen3b-hf-qttt", "fixed8"): ("runs", "exp31_qwen3b_f{k}/fixed8__qttt__fraction0.05"),
    ("qwen7b-exp23", "fifo"): ("exp23", "exp23_qwen7b_f{k}/fifo_top5_t3000__fraction0.05"),
    ("qwen3b-exp23", "fixed8"): ("exp23", "exp23_qwen3b_f{k}/fixed8__fraction0.05"),
}

parser = argparse.ArgumentParser()
parser.add_argument("--base-url")
parser.add_argument("--backend", default="openai")
parser.add_argument("--runs", default="runs")
parser.add_argument("--exp23-runs", default="/home/qamil-mirza/Code/llm-memctl/runs")
parser.add_argument("--cache-dir", default="cache/judge_exp31")
parser.add_argument("--prune", action="store_true")
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
roots = {"runs": Path(args.runs), "exp23": Path(args.exp23_runs)}
index = _index(PATH)
rows, missing = [], []
for (reader, arm), (root, pattern) in CELLS.items():
    for k in range(5):
        chosen = fold_indices(index, 5, k)
        cell = roots[root] / pattern.format(k=k)
        steps = cell / "steps.jsonl"
        if not steps.exists():
            missing.append(str(cell))
            continue
        scored = [json.loads(line) for line in open(steps) if '"scored": true' in line]
        scored = [step for step in scored if step.get("scored")]
        if args.prune and root == "runs" and (cell / "episodes.jsonl").exists():
            if sum(1 for _ in open(cell / "episodes.jsonl")) == len(chosen):  # finished cells only
                steps.write_text("".join(json.dumps(step) + "\n" for step in scored))
        for step in scored:
            number = chosen[step["seed"] % len(chosen)]
            meta = index[number]
            instance = _instance(PATH, number)
            rows.append({"reader": reader, "arm": arm, "fold": k, "question_id": meta["question_id"],
                         "type": meta["question_type"],
                         "question": f"(asked on {instance.get('question_date', '')}) {instance['question']}",
                         "gold": str(instance["answer"]), "answer": step["agent_action"] or "",
                         "prompt_tokens": step["agent_prompt_tokens"],
                         "unknown": (step["agent_action"] or "").strip().lower().startswith("unknown"),
                         "abstention": meta["question_id"].endswith("_abs")})
judge = Judge({"backend": args.backend, "name": "qwen2.5-7b-instruct", "base_url": args.base_url, "timeout_s": 300,
               "cache_dir": args.cache_dir, "style": "longmemeval"})
answerable = [r for r in rows if not r["abstention"]]  # abstention rows are counted, never judged
with ThreadPoolExecutor(32) as pool:
    verdicts = list(pool.map(lambda r: judge.is_correct(r["question"], r["gold"], r["answer"], r["type"]), answerable))
for r, v in zip(answerable, verdicts):
    r["correct"] = bool(v)
args.out.write_text(json.dumps(rows))
counts = {}
for r in rows:
    counts[(r["reader"], r["arm"])] = counts.get((r["reader"], r["arm"]), 0) + 1
print(len(answerable), "answers judged;", {f"{a}/{b}": n for (a, b), n in sorted(counts.items())},
      f"; {len(missing)} cells missing" if missing else "")
