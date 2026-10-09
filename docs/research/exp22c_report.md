# §22c report (held-out, exact simulator)

Copied from `runs/exp22c_report.md` (runs/ is gitignored), produced by `runs/_pipelines/exp22_report.py runs/exp22c_eval.json runs/exp21_gate.json runs/exp22c --noninferior 0.01`. See EXPERIMENTS.md §22c-a.

## Held-out old-evidence accuracy (exact simulator; 1249 questions, 10 conversations)

| policy | accuracy | share of the oracle gap closed over knapsack | age 1–2 | age 3–5 | age 6+ |
|---|---|---|---|---|---|
| oracle (§21) | 0.286 |  | 0.306 | 0.355 | 0.179 |
| (ii) per-write hindsight | 0.144 | -1% | 0.141 | 0.189 | 0.083 |
| (i) episode GRPO | 0.147 | 2% | 0.140 | 0.195 | 0.087 |
| learned-knapsack (init) | 0.145 |  | 0.145 | 0.192 | 0.080 |
| first turns (§21) | 0.109 |  | 0.129 | 0.144 | 0.050 |

- (ii) per-write hindsight − (i) episode GRPO: -0.004 (-0.014, +0.008)
- (ii) per-write hindsight − learned-knapsack (init): -0.001 (-0.012, +0.011)
- (i) episode GRPO − learned-knapsack (init): +0.002 (-0.001, +0.006)
- oracle (§21) − (ii) per-write hindsight: +0.142 (+0.107, +0.177)

**Clause (pre-registered, §22c): (ii)+KL − learned-knapsack lower bound > −0.01 → FAIL (-0.012)**

**Training curves** (surrogate train old-evidence accuracy, mean over folds and seeds; first and last 20 updates):

- (i) episode GRPO: 0.125 → 0.124 (15 runs)
- (ii) per-write hindsight: 0.091 → 0.100 (15 runs)
