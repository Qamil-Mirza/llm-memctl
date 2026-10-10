"""§32 split audit (free; reads the data file and the fold split only, no run output and no label of any arm).
    PYTHONPATH=. python configs/sweeps/exp32/split_audit.py --out runs/exp32_rerankers/split_audit.json

For each fold k (test part = fold k, training part = the other four):
(a) per test question, the share of its filler sessions (haystack sessions that are not answer sessions, by session
    id) that appear in some training question's haystack; mean per fold and pooled;
(b) the test questions that share an answer session (by id) with some training question's answer sessions.
    Also counted: test answer sessions found anywhere in a training haystack (as filler or answer).
The per-question flags are written out, so the §32 report can give the sensitivity row (test questions with no
shared answer session)."""
import argparse
import json
from pathlib import Path

from memctl.envs.longmemeval import _index, _instance
from memctl.splits import fold_indices

PATH = "data/longmemeval/longmemeval_s_cleaned.json"

parser = argparse.ArgumentParser()
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
index = _index(PATH)
sessions, answers = {}, {}
for number in range(len(index)):
    instance = _instance(PATH, number)
    sessions[number] = set(instance["haystack_session_ids"])
    answers[number] = set(instance.get("answer_session_ids", []))
per_question, folds = {}, []
for k in range(5):
    test, train = fold_indices(index, 5, k, "test"), fold_indices(index, 5, k, "train")
    train_hay = set().union(*(sessions[n] for n in train))
    train_ans = set().union(*(answers[n] for n in train))
    shares, shared, anywhere = [], 0, 0
    for n in test:
        filler = sessions[n] - answers[n]
        share = len(filler & train_hay) / len(filler) if filler else 0.0
        shares.append(share)
        s_ans, s_any = bool(answers[n] & train_ans), bool(answers[n] & train_hay)
        shared += s_ans
        anywhere += s_any
        per_question[index[n]["question_id"]] = {"fold": k, "filler_share_in_train": round(share, 4),
                                                 "shares_answer_session": s_ans,
                                                 "answer_session_in_train_haystack": s_any}
    folds.append({"fold": k, "test": len(test), "mean_filler_share": round(sum(shares) / len(shares), 4),
                  "shares_answer_session": shared, "answer_session_in_train_haystack": anywhere})
pooled = {"test": sum(f["test"] for f in folds),
          "mean_filler_share": round(sum(q["filler_share_in_train"] for q in per_question.values()) / len(per_question), 4),
          "shares_answer_session": sum(f["shares_answer_session"] for f in folds),
          "answer_session_in_train_haystack": sum(f["answer_session_in_train_haystack"] for f in folds)}
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps({"folds": folds, "pooled": pooled, "questions": per_question}, indent=1))
for row in folds + [{"fold": "pooled", **pooled}]:
    print(row)
