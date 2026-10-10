"""§32 human spot check: 50 judged answers, stratified by arm (balanced) and question type (spread within each arm),
drawn with a fixed seed. The sample file hides the arm and the judge's verdict; both are in a separate key file."""
import csv, json, random
from collections import defaultdict

SEED, N = 32, 50
rows = json.load(open("runs/exp32_verdicts.json"))["answerable"]
rng = random.Random(SEED)
by_arm = defaultdict(lambda: defaultdict(list))
for r in rows:
    by_arm[r["arm"]][r["type"]].append(r)
arms = sorted(by_arm)
quota = {a: N // len(arms) + (i < N % len(arms)) for i, a in enumerate(arms)}  # 9, 9, 8, 8, 8, 8
picked = []
for a in arms:
    types = sorted(by_arm[a]); rng.shuffle(types)
    pools = {t: rng.sample(by_arm[a][t], len(by_arm[a][t])) for t in types}
    i = 0
    while sum(1 for p in picked if p["arm"] == a) < quota[a]:  # round-robin over question types
        t = types[i % len(types)]; i += 1
        if pools[t]:
            picked.append(pools[t].pop())
rng.shuffle(picked)
with open("runs/exp32_spotcheck/sample.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["sample_id", "question_type", "question", "gold_answer", "model_answer", "your_verdict (correct/wrong)"])
    for i, r in enumerate(picked, 1):
        w.writerow([i, r["type"], r["question"], r["gold"], r["answer"], ""])
with open("runs/exp32_spotcheck/key.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["sample_id", "arm", "fold", "question_id", "unknown", "judge_correct"])
    for i, r in enumerate(picked, 1):
        w.writerow([i, r["arm"], r["fold"], r["question_id"], r["unknown"], r["correct"]])
print(len(picked), "answers;", {a: quota[a] for a in arms})
