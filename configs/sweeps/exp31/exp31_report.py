"""Experiment 31 grading (§31, pre-registered): python configs/sweeps/exp31/exp31_report.py runs/exp31_verdicts.json
Accuracy on the 470 answerable LongMemEval questions under one blind Qwen 7B judge pass; differences paired by
question with a bootstrap 95% interval (10,000 resamples, seed 0), as §23."""
import json
import random
import sys
from collections import defaultdict

V = json.load(open(sys.argv[1]))
EXP23 = sys.argv[2] if len(sys.argv) > 2 else "/home/qamil-mirza/Code/llm-memctl/runs/exp23_verdicts.json"
cell = defaultdict(dict)
for r in V:
    cell[r["reader"], r["arm"]][r["question_id"]] = r
mean = lambda v: sum(map(float, v)) / len(v) if v else float("nan")


def boot(values, n=10000):
    if not values:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(0)
    d = sorted(sum(rng.choices(values, k=len(values))) / len(values) for _ in range(n))
    return sum(values) / len(values), d[int(0.025 * n)], d[int(0.975 * n) - 1]


def answerable(key):
    return {q: r for q, r in cell[key].items() if not r["abstention"]}


def diff(a, b):
    A, B = answerable(a), answerable(b)
    qs = sorted(set(A) & set(B))
    return boot([float(A[q]["correct"]) - float(B[q]["correct"]) for q in qs]), len(qs)


f = lambda d: f"{d[0]:+.3f} ({d[1]:+.3f}, {d[2]:+.3f})"
verdict = lambda d: "BETTER" if d[1] > 0 else "NON-INFERIOR" if d[1] > -0.03 else "NOT SHOWN"
print("| reader | arm | n | accuracy (95% CI) | unknown | P(correct \\| answered) | false answers on unanswerable |")
print("|---|---|---|---|---|---|---|")
for key in sorted(cell):
    rows = list(answerable(key).values())
    abst = [r for r in cell[key].values() if r["abstention"]]
    acc = boot([float(r["correct"]) for r in rows])
    print(f"| {key[0]} | {key[1]} | {len(rows)} | {acc[0]:.3f} ({acc[1]:.3f}, {acc[2]:.3f}) | "
          f"{mean([r['unknown'] for r in rows]):.3f} | {mean([r['correct'] for r in rows if not r['unknown']]):.3f} | "
          f"{sum(not r['unknown'] for r in abst)} of {len(abst)} |")
print("\nPre-registered (margin −0.03):")
for name, a, b in (
    ("Claim 1 again: 3B fixed8 + qTTT − 7B FIFO + floor (§23 answers, rejudged)", ("qwen3b-hf-qttt", "fixed8"), ("qwen7b-exp23", "fifo")),
    ("TTT effect at fixed8 (primary paired): qTTT − no adaptation", ("qwen3b-hf-qttt", "fixed8"), ("qwen3b-hf", "fixed8")),
    ("TTT effect at FIFO + floor (control): qTTT − no adaptation", ("qwen3b-hf-qttt", "fifo"), ("qwen3b-hf", "fifo")),
    ("Reference: 3B fixed8 (HF, no adaptation) − 7B FIFO + floor", ("qwen3b-hf", "fixed8"), ("qwen7b-exp23", "fifo")),
):
    d, n = diff(a, b)
    print(f"- {name}: {f(d)}, n = {n} → {verdict(d)}")
for name, a, b in (("controller x TTT: (fixed8 TTT gain) − (FIFO TTT gain)", None, None),):
    A1, A0 = answerable(("qwen3b-hf-qttt", "fixed8")), answerable(("qwen3b-hf", "fixed8"))
    B1, B0 = answerable(("qwen3b-hf-qttt", "fifo")), answerable(("qwen3b-hf", "fifo"))
    qs = sorted(set(A1) & set(A0) & set(B1) & set(B0))
    d = boot([(A1[q]["correct"] - A0[q]["correct"]) - (B1[q]["correct"] - B0[q]["correct"]) for q in qs])
    print(f"- {name} (descriptive): {f(d)}, n = {len(qs)}")
print("\nChecks:")
A, B = answerable(("qwen3b-hf", "fixed8")), answerable(("qwen3b-exp23", "fixed8"))
qs = sorted(set(A) & set(B))
if qs:
    same = mean([A[q]["answer"] == B[q]["answer"] for q in qs])
    print(f"- HF serving vs §23 vLLM, 3B fixed8, no adaptation: identical answers {same:.3f}; accuracy "
          f"{mean([A[q]['correct'] for q in qs]):.3f} vs {mean([B[q]['correct'] for q in qs]):.3f} (n = {len(qs)})")
try:
    old = {(r["question_id"], r["answer"]): r["correct"] for r in json.load(open(EXP23))
           if r["reader"] == "qwen7b" and r["arm"] == "fifo_top5_t3000"}
    rows = [r for r in answerable(("qwen7b-exp23", "fifo")).values() if (r["question_id"], r["answer"]) in old]
    print(f"- judge pass vs §23's pass on the 7B FIFO answers: verdict agreement "
          f"{mean([r['correct'] == old[r['question_id'], r['answer']] for r in rows]):.3f} (n = {len(rows)})")
except FileNotFoundError:
    print("- §23 verdicts not found; judge agreement not computed")
