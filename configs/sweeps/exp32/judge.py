"""Experiment 32: one blind judge pass over the reader's answers (the §23/§26/§27 judge, prompts and route).
    PYTHONPATH=. python configs/sweeps/exp32/judge.py --base-url URL/v1 --out runs/exp32_verdicts.json [--backend stub]
The judge is Qwen2.5-7B with the official LongMemEval prompts. It sees the question, the gold answer and the answer
text, never which arm wrote it; the cache (cache/judge_exp32) is keyed by those texts. Answerable questions only (470
per arm); abstention answers are kept aside for the "false answers on unanswerable" column (not judged)."""
import argparse, json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from memctl.analysis.shares import is_unknown
from memctl.envs.longmemeval import _index, _instance
from memctl.judge import Judge
from memctl.splits import fold_indices

PATH = "data/longmemeval/longmemeval_s_cleaned.json"
ARMS = ("fixed8", "bm25_top8", "rrf_top8", "lr_pointwise", "gbdt_pointwise", "cross_encoder_zero",
        "cross_encoder_tuned")

parser = argparse.ArgumentParser()
parser.add_argument("--base-url")
parser.add_argument("--backend", default="openai")
parser.add_argument("--runs", default="runs")
parser.add_argument("--prefix", default="exp32_qwen7b")
parser.add_argument("--cache-dir", default="cache/judge_exp32")
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
index = _index(PATH)
rows, abstention = [], []
for k in range(5):
    chosen = fold_indices(index, 5, k)
    for arm in ARMS:
        cell = Path(args.runs) / f"{args.prefix}_f{k}" / f"{arm}__fraction0.05"
        if not (cell / "episodes.jsonl").exists():
            continue
        with open(cell / "steps.jsonl") as handle:
            for line in handle:
                step = json.loads(line)
                if not step.get("scored"):
                    continue
                number = chosen[step["seed"] % len(chosen)]
                meta = index[number]
                answer = step["agent_action"] or ""
                if meta["question_id"].endswith("_abs"):
                    abstention.append({"arm": arm, "fold": k, "question_id": meta["question_id"],
                                       "unknown": is_unknown(answer)})
                    continue
                instance = _instance(PATH, number)
                rows.append({"arm": arm, "fold": k, "question_id": meta["question_id"], "type": meta["question_type"],
                             "question": f"(asked on {instance.get('question_date', '')}) {instance['question']}",
                             "gold": str(instance["answer"]), "answer": answer,
                             "prompt_tokens": step["agent_prompt_tokens"], "unknown": is_unknown(answer)})
judge = Judge({"backend": args.backend, "name": "qwen2.5-7b-instruct", "base_url": args.base_url, "timeout_s": 300,
               "cache_dir": args.cache_dir, "style": "longmemeval"})
with ThreadPoolExecutor(32) as pool:
    verdicts = list(pool.map(lambda r: judge.is_correct(r["question"], r["gold"], r["answer"], r["type"]), rows))
for r, v in zip(rows, verdicts):
    r["correct"] = bool(v)
args.out.write_text(json.dumps({"answerable": rows, "abstention": abstention}))
print(len(rows), "answers judged;", len(abstention), "abstention answers kept aside")
