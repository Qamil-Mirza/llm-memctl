"""§30 report: the paper's metrics (RS, NR, DR) beside our three shares and P(correct | all in view), per arm, with
the pre-registered claim (fixed8 minus FIFO + floor on NR, 95% paired bootstrap by question; by bundle as a check).
    python -I configs/sweeps/exp30/exp30_report.py <worktree> <verdicts.json> <inview.json>"""
import json
import random
import sys

sys.path.insert(0, sys.argv[1])
from memctl.envs.utilmem import DEGRADED_AT  # noqa: E402

rows = json.loads(open(sys.argv[2]).read())
inview = json.loads(open(sys.argv[3]).read())["rows"]
ARMS = ("fifo_top5_t3000", "fixed8", "fixed16")
BASE = "fifo_top5_t3000"
score = {(r["arm"], r["question_id"]): r for r in rows}
questions = sorted({r["question_id"] for r in rows})
bundles = sorted({q.split("/")[0] for q in questions})
rng = random.Random(0)
by_question = [[rng.randrange(len(questions)) for _ in questions] for _ in range(2000)]
members = {b: [n for n, q in enumerate(questions) if q.startswith(b + "/")] for b in bundles}
by_bundle = [[n for _ in bundles for n in members[bundles[rng.randrange(len(bundles))]]] for _ in range(2000)]


def nr(values):
    kept = [v for v in values if v is not None]
    return (sum(kept) / len(kept) - 1) / 9 if kept else float("nan")


def interval(values, draws):
    means = sorted(sum(values[i] for i in d) / len(d) for d in draws)
    return sum(values) / len(values), means[49], means[1949]


print("| arm | RS | NR | DR | answered-correct | answered-wrong | unknown | P(correct \\| all in view) | "
      "P(correct \\| not all) | all in view | unparsable |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for arm in ARMS:
    got = [score[(arm, q)] for q in questions]
    scored = [r for r in got if r["score"] is not None]
    unknown = [r["refusal"] for r in got]
    correct = [(not r["refusal"]) and r["score"] is not None and r["score"] > DEGRADED_AT for r in got]
    wrong = [(not r["refusal"]) and not c for r, c in zip(got, correct)]
    allv = [inview[arm][q]["all"] for q in questions]
    p_in = sum(c for c, a in zip(correct, allv) if a) / max(1, sum(allv))
    p_out = sum(c for c, a in zip(correct, allv) if not a) / max(1, len(allv) - sum(allv))
    rs = sum(r["score"] for r in scored) / len(scored) if scored else float("nan")
    dr = sum(r["score"] <= DEGRADED_AT for r in scored) / len(scored) if scored else float("nan")
    print(f"| {arm} | {rs:.2f} | {nr([r['score'] for r in got]):.3f} | {dr:.3f} | {sum(correct) / len(got):.3f} | "
          f"{sum(wrong) / len(got):.3f} | {sum(unknown) / len(got):.3f} | {p_in:.3f} | {p_out:.3f} | "
          f"{sum(allv) / len(allv):.3f} | {len(got) - len(scored)} |")
# Paired NR differences: per question, (score_arm - score_base) / 9, over questions both arms scored.
for arm in ("fixed8", "fixed16"):
    keep = [n for n, q in enumerate(questions) if score[(arm, q)]["score"] is not None and score[(BASE, q)]["score"] is not None]
    diff = [0.0] * len(questions)
    for n in keep:
        q = questions[n]
        diff[n] = (score[(arm, q)]["score"] - score[(BASE, q)]["score"]) / 9
    keep_set = set(keep)
    dq = [[i for i in d if i in keep_set] for d in by_question]
    db = [[i for i in d if i in keep_set] for d in by_bundle]
    _, lo, hi = interval(diff, dq)
    _, blo, bhi = interval(diff, db)
    mean = sum(diff[i] for i in keep) / len(keep)
    verdict = ("BETTER" if lo > 0 else "NON-INFERIOR" if lo > -0.03 else "WORSE" if hi < 0 else "UNDETERMINED")
    print(f"{arm} minus FIFO + floor, NR: {mean:+.3f} ({lo:+.3f}, {hi:+.3f}) by question; ({blo:+.3f}, {bhi:+.3f}) "
          f"by bundle; n = {len(keep)}; {verdict if arm == 'fixed8' else 'descriptive'}")
