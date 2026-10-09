"""Experiment 27: one blind judge pass over the answers (§27; the §23/§26 judge, prompts and route).
    PYTHONPATH=. python configs/sweeps/exp27/exp27_judge.py --base-url URL/v1 --out runs/exp27_verdicts.json [--backend stub]
The judge is Qwen2.5-7B with the official LongMemEval prompts. It sees the question, the gold answer and the answer
text, never which arm wrote it; the cache (cache/judge_exp27) is keyed by those texts. Answerable questions only (470).
A copy of runs/_pipelines/exp26_judge.py with the §27 arms."""
import argparse, json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from memctl.envs.longmemeval import _index, _instance
from memctl.judge import Judge
from memctl.splits import fold_indices

PATH = "data/longmemeval/longmemeval_s_cleaned.json"
ARMS = ("fifo_top5_t3000", "fixed8", "lre_native", "lre_slot")

parser = argparse.ArgumentParser()
parser.add_argument("--base-url")
parser.add_argument("--backend", default="openai")
parser.add_argument("--runs", default="runs")
parser.add_argument("--cache-dir", default="cache/judge_exp27")
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
index = _index(PATH)
rows = []
for k in range(5):
    chosen = fold_indices(index, 5, k)
    for arm in ARMS:
        cell = Path(args.runs) / f"exp27_qwen7b_f{k}" / f"{arm}__fraction0.05"
        if not (cell / "steps.jsonl").exists():
            continue
        for line in open(cell / "steps.jsonl"):
            step = json.loads(line)
            if not step.get("scored"):
                continue
            number = chosen[step["seed"] % len(chosen)]
            meta = index[number]
            if meta["question_id"].endswith("_abs"):
                continue
            instance = _instance(PATH, number)
            rows.append({"arm": arm, "fold": k, "question_id": meta["question_id"], "type": meta["question_type"],
                         "question": f"(asked on {instance.get('question_date', '')}) {instance['question']}",
                         "gold": str(instance["answer"]), "answer": step["agent_action"] or "",
                         "prompt_tokens": step["agent_prompt_tokens"],
                         "unknown": (step["agent_action"] or "").strip().lower().startswith("unknown")})
judge = Judge({"backend": args.backend, "name": "qwen2.5-7b-instruct", "base_url": args.base_url, "timeout_s": 300,
               "cache_dir": args.cache_dir, "style": "longmemeval"})
with ThreadPoolExecutor(32) as pool:
    verdicts = list(pool.map(lambda r: judge.is_correct(r["question"], r["gold"], r["answer"], r["type"]), rows))
for r, v in zip(rows, verdicts):
    r["correct"] = bool(v)
args.out.write_text(json.dumps(rows))
print(len(rows), "answers judged")
