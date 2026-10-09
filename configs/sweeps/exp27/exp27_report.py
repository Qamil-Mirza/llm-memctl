"""§27 grading (pre-registered; margin -0.03; 95% paired bootstrap by question, as §23):
    PYTHONPATH=. python configs/sweeps/exp27/exp27_report.py [--verdicts runs/exp27_verdicts.json]
Run once, after the judge pass. P(all in view) per arm is read here, from the evaluation's own episodes, and nowhere
earlier."""
import argparse, json, random
from collections import defaultdict
from memctl.envs.longmemeval import _index
from memctl.splits import fold_indices

parser = argparse.ArgumentParser()
parser.add_argument("--verdicts", default="runs/exp27_verdicts.json")
parser.add_argument("--runs", default="runs")
args = parser.parse_args()
V = json.load(open(args.verdicts))
ARMS = ["fifo_top5_t3000", "fixed8", "lre_native", "lre_slot"]
idx = _index("data/longmemeval/longmemeval_s_cleaned.json")
cell = defaultdict(dict)
for r in V:
    cell[r["arm"]][r["question_id"]] = r
in_view = defaultdict(dict)
for arm in ARMS:
    for k in range(5):
        chosen = fold_indices(idx, 5, k)
        for line in open(f"{args.runs}/exp27_qwen7b_f{k}/{arm}__fraction0.05/episodes.jsonl"):
            e = json.loads(line)
            in_view[arm][idx[chosen[e["seed"] % len(chosen)]]["question_id"]] = (e.get("evidence_complete_rate") or 0) >= 1
m = lambda v: sum(map(float, v)) / len(v)
def boot(values, n=10000):
    rng = random.Random(0)
    d = sorted(sum(rng.choices(values, k=len(values))) / len(values) for _ in range(n))
    return sum(values) / len(values), d[int(0.025 * n)], d[int(0.975 * n) - 1]
def diff(a, b):
    qs = sorted(set(cell[a]) & set(cell[b]))
    return boot([float(cell[a][q]["correct"]) - float(cell[b][q]["correct"]) for q in qs])
f = lambda d: f"{d[0]:+.3f} ({d[1]:+.3f}, {d[2]:+.3f})"
print("| arm | accuracy (95% CI) | unknown | P(correct \\| answered) | P(all in view) | P(correct \\| in view) | prompt tokens |\n|---|---|---|---|---|---|---|")
for arm in ARMS:
    rows = list(cell[arm].values()); ans = [r for r in rows if not r["unknown"]]
    iv = [r for r in rows if in_view[arm][r["question_id"]]]
    a = boot([float(r["correct"]) for r in rows])
    print(f"| {arm} | {a[0]:.3f} ({a[1]:.3f}, {a[2]:.3f}) | {m([r['unknown'] for r in rows]):.3f} | {m([r['correct'] for r in ans]):.3f} | "
          f"{len(iv)/len(rows):.3f} | {m([r['correct'] for r in iv]):.3f} | {m([r['prompt_tokens'] for r in rows]):,.0f} |")
c1 = diff("fixed8", "lre_slot"); c2 = diff("lre_slot", "fifo_top5_t3000"); c3 = diff("lre_native", "fifo_top5_t3000")
c4 = diff("fixed8", "fifo_top5_t3000")
print(f"\n- C1, fixed8 − LRE-slot: {f(c1)} → **{'fixed8 BETTER' if c1[1] > 0 else 'LRE-slot BETTER' if c1[2] < 0 else 'LRE-slot NON-INFERIOR' if c1[2] < 0.03 else 'NOT SHOWN'}**")
print(f"- C2, LRE-slot − FIFO + floor: {f(c2)} → **{'BETTER' if c2[1] > 0 else 'WORSE' if c2[2] < 0 else 'UNDETERMINED'}**")
print(f"- C3, LRE-native − FIFO + floor: {f(c3)} → **{'BETTER' if c3[1] > 0 else 'WORSE' if c3[2] < 0 else 'NON-INFERIOR' if c3[1] > -0.03 else 'UNDETERMINED'}**")
print(f"- fixed8 − FIFO + floor (pairing check against §23's +0.102 and §26's +0.081): {f(c4)}")
