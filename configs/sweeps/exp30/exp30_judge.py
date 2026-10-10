"""§30: one blind judge pass with the paper's Appendix F rubric (Qwen2.5-7B on the same pod, after all answers).
The judge sees the evidence sessions (cut at 1,000 tokens a turn, as the reader saw them), the question, the
reference (the oracle arm's answer) and the answer; never the arm. The cache is keyed by the prompt text.
    python -I configs/sweeps/exp30/exp30_judge.py <worktree> --data FILE --runs runs --prefix exp30_eval \
        --out runs/exp30_verdicts.json [--base-url URL/v1] [--backend stub]"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, sys.argv[1])
from memctl.envs.utilmem import JUDGE_PROMPT, _bundle, _index, is_refusal, parse_question, parse_score  # noqa: E402
from memctl.llm import build_llm  # noqa: E402
from memctl.memory.items import count_tokens, label_prefix  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("worktree")
parser.add_argument("--data", required=True)
parser.add_argument("--runs", default="runs")
parser.add_argument("--prefix", default="exp30_eval")
parser.add_argument("--base-url", default="")
parser.add_argument("--backend", default="openai")
parser.add_argument("--cache", default="cache/judge_exp30")
parser.add_argument("--workers", type=int, default=32)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--arms", default="fifo_top5_t3000,fixed8,fixed16",
                    help="triage: judge the claim's two arms first, fixed16 last (the cache keeps finished verdicts)")
args = parser.parse_args()
ARMS = tuple(args.arms.split(","))
bundle_of = {q["sample_id"]: q["bundle"] for q in _index(args.data)["questions"]}


def answers(cell: Path) -> dict:
    """question_id -> (answer, reader prompt tokens) for one cell."""
    question_of = {}
    for line in open(cell / "episodes.jsonl"):
        episode = json.loads(line)
        question_of[episode["episode_id"]] = episode["env_stats"]["answers"][0]["question_id"]
    found = {}
    for line in open(cell / "steps.jsonl"):
        step = json.loads(line)
        if step["requires_response"]:
            found[question_of[step["episode_id"]]] = (step["agent_action"] or "", step["agent_prompt_tokens"])
    return found


reference, by_arm = {}, {arm: {} for arm in ARMS}
for cell in sorted(Path(args.runs).glob(f"{args.prefix}_oracle_s*/oracle__*/")):
    reference.update(answers(cell))
for cell in sorted(Path(args.runs).glob(f"{args.prefix}_qwen7b_s*/*/")):
    arm = cell.name.split("__")[0]
    if arm in by_arm:
        by_arm[arm].update(answers(cell))
questions = sorted(reference)
for arm in ARMS:
    assert sorted(by_arm[arm]) == questions, (arm, len(by_arm[arm]), len(questions))


def context(question_id: str) -> str:
    sample_id, domain, number = question_id.split("/")
    episode, _ = parse_question(_bundle(args.data, bundle_of[sample_id]), domain, int(number[1:]), oracle=True,
                                max_turn_tokens=1000)
    return "\n".join(label_prefix(t.metadata, t.source_type) + t.content for t in episode.turns)


rows = []
for question_id in questions:
    sample_id, domain, number = question_id.split("/")
    question = _bundle(args.data, bundle_of[sample_id])["question_pool"][domain][int(number[1:])]
    text = context(question_id)
    for arm in ARMS:
        answer, prompt_tokens = by_arm[arm][question_id]
        prompt = JUDGE_PROMPT.format(context=text, question=question, answer_oracle=reference[question_id][0],
                                     answer_noisy=answer)
        rows.append({"arm": arm, "question_id": question_id, "domain": domain, "answer_tokens": count_tokens(answer),
                     "reader_prompt_tokens": prompt_tokens, "refusal": is_refusal(answer),
                     "reference_refusal": is_refusal(reference[question_id][0]),
                     "judge_prompt_tokens": count_tokens(prompt), "_prompt": prompt})
judge = build_llm({"backend": args.backend, "name": "qwen2.5-7b-instruct", "base_url": args.base_url,
                   "timeout_s": 600, "cache_dir": args.cache})
with ThreadPoolExecutor(args.workers) as pool:
    outputs = list(pool.map(lambda r: judge.generate(r.pop("_prompt"), 400), rows))
for row, output in zip(rows, outputs):
    row["score"] = parse_score(output)
    row["judge_output_tokens"] = count_tokens(output)
args.out.write_text(json.dumps(rows))
failed = sum(r["score"] is None for r in rows)
print(len(rows), "answers judged;", failed, "unparsable;", sum(r["reference_refusal"] for r in rows) // len(ARMS),
      "references are refusals")
