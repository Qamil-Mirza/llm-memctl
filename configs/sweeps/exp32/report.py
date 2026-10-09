"""§32 reader grading (pre-registered; margin -0.03; 95% paired bootstrap by question, 10,000 resamples, as §23/§27).
    PYTHONPATH=. python configs/sweeps/exp32/report.py [--verdicts runs/exp32_verdicts.json]
Run once, after the judge pass. The best classical reranker is read from runs/exp32_rerankers/inner_selection.json,
fixed there from the training part before any reader call.

Primary:   D1 = fixed8 minus the best classical reranker, accuracy on the 470 answerable questions.
           Pre-declared direction: the head better (D1 > 0).
           - "head BETTER" if the lower bound is above 0;
           - "head NON-INFERIOR" if the lower bound is above -0.03 (and not better);
           - "head WORSE" if the upper bound is below 0; otherwise "NOT SHOWN".
           Read the other way (descriptive): the baseline is within the margin of the head if the upper bound < +0.03.
Secondary: D2 = fixed8 minus cross_encoder_zero, same thresholds and wording.
Answer shares are disjoint (memctl/analysis/shares.py); an "unknown" answer judged correct is flagged."""
import argparse
import json
import random
from collections import defaultdict

from memctl.analysis.shares import answer_shares

ARMS = ["fixed8", "bm25_top8", "rrf_top8", "lr_pointwise", "gbdt_pointwise", "cross_encoder_zero",
        "cross_encoder_tuned"]
parser = argparse.ArgumentParser()
parser.add_argument("--verdicts", default="runs/exp32_verdicts.json")
parser.add_argument("--selection", default="runs/exp32_rerankers/inner_selection.json")
args = parser.parse_args()
V = json.load(open(args.verdicts))
best = {"lr": "lr_pointwise", "gbdt": "gbdt_pointwise"}[json.load(open(args.selection))["best_classical"]]
cell = defaultdict(dict)
for r in V["answerable"]:
    cell[r["arm"]][r["question_id"]] = r


def boot(values, n=10000):
    rng = random.Random(0)
    d = sorted(sum(rng.choices(values, k=len(values))) / len(values) for _ in range(n))
    return sum(values) / len(values), d[int(0.025 * n)], d[int(0.975 * n) - 1]


def diff(a, b):
    qs = sorted(set(cell[a]) & set(cell[b]))
    return boot([float(cell[a][q]["correct"]) - float(cell[b][q]["correct"]) for q in qs]), len(qs)


def verdict(d):
    mean, low, high = d
    if low > 0:
        return "head BETTER"
    if high < 0:
        return "head WORSE"
    return "head NON-INFERIOR" if low > -0.03 else "NOT SHOWN"


f = lambda d: f"{d[0]:+.3f} ({d[1]:+.3f}, {d[2]:+.3f})"
print("| arm | accuracy (95% CI) | answered-correct | answered-wrong | unknown | P(correct \\| answered) | prompt tokens"
      " | false answers on the 30 unanswerable |\n|---|---|---|---|---|---|---|---|")
for arm in ARMS:
    rows = list(cell[arm].values())
    if not rows:
        continue
    s = answer_shares(rows)
    a = boot([float(r["correct"]) for r in rows])
    false = sum(1 for r in V["abstention"] if r["arm"] == arm and not r["unknown"])
    flag = f" (FLAG: {len(s['flagged'])} unknown judged correct)" if s["flagged"] else ""
    print(f"| {arm} | {a[0]:.3f} ({a[1]:.3f}, {a[2]:.3f}){flag} | {s['answered_correct']:.3f} | "
          f"{s['answered_wrong']:.3f} | {s['unknown']:.3f} | {s['p_correct_given_answered']:.3f} | "
          f"{sum(r['prompt_tokens'] for r in rows) / len(rows):,.0f} | {false} |")
(d1, n1), (d2, n2) = diff("fixed8", best), diff("fixed8", "cross_encoder_zero")
print(f"\n- **Primary, fixed8 − {best} (best classical, chosen on the training part):** {f(d1)}, n = {n1} → "
      f"**{verdict(d1)}**; baseline within 0.03 of the head: {'yes' if d1[2] < 0.03 else 'no'}")
print(f"- **Secondary, fixed8 − cross_encoder_zero:** {f(d2)}, n = {n2} → **{verdict(d2)}**; "
      f"baseline within 0.03 of the head: {'yes' if d2[2] < 0.03 else 'no'}")
for arm in ARMS[1:]:
    if arm not in (best, "cross_encoder_zero") and cell[arm]:
        d, n = diff("fixed8", arm)
        print(f"- descriptive, fixed8 − {arm}: {f(d)}")
