"""§40 laptop preparation (no model): the 60 cached §35 rows of the judge agreement check, frozen into a small file the
pod bundle ships (so the pod needs neither the LongMemEval file nor the §35 cache).

    .venv/bin/python scripts/exp40_prepare.py            # writes runs/exp40_inputs/agreement60.jsonl

Rule (fixed here, before any LoCoMo outcome exists): the pool is the §38 pod's same-text check items (pop 'cached35' of
runs/exp38_pod/qwen3b/judge.jsonl: the cached §35 3B final answers, A on R and C, B on R, folds 2-4), minus refusals
(the §23 rule: a refusal's verdict is 'no' by construction, so it would only pad the agreement); sorted by
(question_id, arm); random.Random(40) shuffle; the first 60. Each row stores the exact §35 judge prompt (rebuilt with
`answer_selection.judge_prompt` and asserted equal, by sha256, to the §35 judge row's prompt sha) and the cached §35
verdict (vLLM 0.8.5). The pod re-scores the 60 prompts with its own 7B judge; agreement < 0.90 stops the run before any
arm (the Advisor's ruling of 2026-10-23). The §38 pod's HF bf16 verdicts on the same 60 are printed beside (descriptive;
they are why the bar is expected to pass, not part of the rule).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from memctl.analysis import answer_selection as sel  # noqa: E402

N = 60
SEED = 40
OUT = "runs/exp40_inputs/agreement60.jsonl"
POOL = "runs/exp38_pod/qwen3b/judge.jsonl"
DATA = "data/longmemeval/longmemeval_s_cleaned.json"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args(argv)
    from memctl.jacobian_data import question_meta
    from scripts.exp35_generate import judge_questions

    pool = {(r["question_id"], r["kind"]): r for r in map(json.loads, Path(POOL).read_text().splitlines())
            if r["pop"] == "cached35"}
    cached = {}
    for f in (2, 3, 4):
        for r in map(json.loads, Path(f"runs/exp35/qwen3b/judge/fold{f}.jsonl").read_text().splitlines()):
            if r["item"] in ("A", "B"):
                cached[(r["question_id"], r["item"])] = r
    keys = sorted(k for k, r in pool.items() if not sel.is_refusal_answer(r["answer"]))
    random.Random(SEED).shuffle(keys)
    keys = keys[:N]
    questions = judge_questions(sorted({q for q, _ in keys}))
    rows = []
    for q, arm in keys:
        c = cached[(q, arm)]
        gold, qtype, _ = question_meta(DATA, q)
        prompt = sel.judge_prompt(questions[q], gold, c["answer"], qtype)
        if sel.sha256(prompt) != c["judge_prompt_sha256"]:
            raise SystemExit(f"{q}/{arm}: rebuilt judge prompt differs from the §35 judge row's")
        rows.append({"question_id": q, "arm": arm, "fold": c["fold"], "question_type": qtype, "answer": c["answer"],
                     "judge_prompt": prompt, "judge_prompt_sha256": c["judge_prompt_sha256"],
                     "cached_verdict": bool(c["verdict"]), "cached_engine": c["judge_engine"],
                     "exp38_hf_verdict": bool(pool[(q, arm)]["verdict"])})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    agree = sum(r["cached_verdict"] == r["exp38_hf_verdict"] for r in rows)
    print(f"pool {len(pool)} cached35 items, {sum(1 for k in pool if k in cached)} with a §35 verdict; non-refusal "
          f"{len([k for k, r in pool.items() if not sel.is_refusal_answer(r['answer'])])}; took {len(rows)} (seed {SEED})")
    print(f"cached verdicts yes {sum(r['cached_verdict'] for r in rows)} / no {sum(not r['cached_verdict'] for r in rows)}; "
          f"§38 HF bf16 judge agreed on {agree}/{len(rows)} (descriptive)")
    print(f"wrote {out} sha256 {sel.sha256(out.read_text())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
