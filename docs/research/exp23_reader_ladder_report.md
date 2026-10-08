# Experiment 23: reader size ladder: grading report

Copied from `runs/exp23_report.md` (runs/ is gitignored), produced by `runs/_pipelines/exp23_report.py` from `runs/exp23_verdicts.json`. In the first table, accuracy, answered-wrong and unknown are the three disjoint shares of the 470 answerable questions (they sum to 1). See EXPERIMENTS.md §23a.

## Accuracy on 470 answerable questions (one blind Qwen 7B judge)

| reader | arm | accuracy | answered-wrong | unknown | P(correct \| answered) | P(correct \| in view) | P(all in view) | prompt tokens |
|---|---|---|---|---|---|---|---|---|
| qwen3b | fifo_top5_t3000 | 0.162 | 0.130 | 0.709 | 0.555 | 0.282 | 0.498 | 2,964 |
| qwen3b | fixed8 | 0.315 | 0.313 | 0.372 | 0.502 | 0.397 | 0.702 | 1,049 |
| qwen3b | fixed16 | 0.202 | 0.228 | 0.570 | 0.470 | 0.242 | 0.747 | 2,569 |
| qwen7b | fifo_top5_t3000 | 0.436 | 0.236 | 0.328 | 0.649 | 0.692 | 0.498 | 2,964 |
| qwen7b | fixed8 | 0.538 | 0.311 | 0.151 | 0.634 | 0.667 | 0.702 | 1,049 |
| qwen7b | fixed16 | 0.487 | 0.311 | 0.202 | 0.611 | 0.578 | 0.747 | 2,569 |
| qwen14b | fifo_top5_t3000 | 0.504 | 0.226 | 0.270 | 0.691 | 0.803 | 0.498 | 2,964 |
| qwen14b | fixed8 | 0.606 | 0.211 | 0.183 | 0.742 | 0.773 | 0.702 | 1,049 |
| qwen14b | fixed16 | 0.581 | 0.279 | 0.140 | 0.676 | 0.698 | 0.747 | 2,569 |
| granite2b | fifo_top5_t3000 | 0.368 | 0.551 | 0.081 | 0.400 | 0.573 | 0.498 | 2,964 |
| granite2b | fixed8 | 0.364 | 0.455 | 0.181 | 0.444 | 0.452 | 0.702 | 1,049 |
| granite2b | fixed16 | 0.353 | 0.472 | 0.174 | 0.428 | 0.430 | 0.747 | 2,569 |
| granite8b | fifo_top5_t3000 | 0.421 | 0.317 | 0.262 | 0.571 | 0.650 | 0.498 | 2,964 |
| granite8b | fixed8 | 0.470 | 0.370 | 0.160 | 0.559 | 0.573 | 0.702 | 1,049 |
| granite8b | fixed16 | 0.489 | 0.385 | 0.126 | 0.560 | 0.587 | 0.747 | 2,569 |

## Pre-registered claims (non-inferiority margin −0.03; 'better' if the lower bound > 0)

- Claim 1: fixed8 @ Qwen 3B vs FIFO @ Qwen 7B: -0.121 (-0.170, -0.070) → **NOT SHOWN**; share of the size gap closed 0.56 (0.41, 0.72)
- Claim 2: fixed8 @ Qwen 7B vs FIFO @ Qwen 14B: +0.034 (-0.013, +0.079) → **NON-INFERIOR**; share of the size gap closed 1.50 (0.83, 2.85)
- Secondary: fixed8 @ Granite 2B vs FIFO @ Granite 8B: -0.057 (-0.102, -0.013) → **NOT SHOWN**; share of the size gap closed -0.08 (-1.67, 0.70)

## fixed8 − FIFO within each reader, and where the gain comes from

- qwen3b: +0.153 (+0.109, +0.200); unknown 0.709 → 0.372; P(correct | answered) 0.555 → 0.502
- qwen7b: +0.102 (+0.060, +0.145); unknown 0.328 → 0.151; P(correct | answered) 0.649 → 0.634
- qwen14b: +0.102 (+0.060, +0.145); unknown 0.270 → 0.183; P(correct | answered) 0.691 → 0.742
- granite2b: -0.004 (-0.047, +0.038); unknown 0.081 → 0.181; P(correct | answered) 0.400 → 0.444
- granite8b: +0.049 (+0.002, +0.096); unknown 0.262 → 0.160; P(correct | answered) 0.571 → 0.559

## fixed16 − fixed8 within each reader (context-rot probe, descriptive)

- qwen3b: -0.113 (-0.160, -0.068)
- qwen7b: -0.051 (-0.089, -0.015)
- qwen14b: -0.026 (-0.062, +0.009)
- granite2b: -0.011 (-0.051, +0.030)
- granite8b: +0.019 (-0.023, +0.064)

## Accuracy intervals and the 30 unanswerable questions (correct abstentions = said unknown)

| reader | arm | accuracy (95% CI) | correct abstentions of 30 |
|---|---|---|---|
| qwen3b | fifo_top5_t3000 | 0.162 (0.130, 0.196) | 29 |
| qwen3b | fixed8 | 0.315 (0.274, 0.357) | 23 |
| qwen3b | fixed16 | 0.202 (0.166, 0.238) | 28 |
| qwen7b | fifo_top5_t3000 | 0.436 (0.391, 0.481) | 24 |
| qwen7b | fixed8 | 0.538 (0.494, 0.583) | 23 |
| qwen7b | fixed16 | 0.487 (0.443, 0.532) | 24 |
| qwen14b | fifo_top5_t3000 | 0.504 (0.460, 0.551) | 25 |
| qwen14b | fixed8 | 0.606 (0.564, 0.651) | 27 |
| qwen14b | fixed16 | 0.581 (0.536, 0.626) | 28 |
| granite2b | fifo_top5_t3000 | 0.368 (0.326, 0.413) | 5 |
| granite2b | fixed8 | 0.364 (0.321, 0.409) | 15 |
| granite2b | fixed16 | 0.353 (0.311, 0.398) | 14 |
| granite8b | fifo_top5_t3000 | 0.421 (0.377, 0.466) | 21 |
| granite8b | fixed8 | 0.470 (0.426, 0.515) | 16 |
| granite8b | fixed16 | 0.489 (0.445, 0.534) | 18 |

## 7B pairing check against §19c (same 470 questions)

- fixed8: §19c 0.545 vs §23 0.538; per-question agreement 0.968
- fifo_top5_t3000: §19c 0.432 vs §23 0.436; per-question agreement 0.979

## Judge outputs (cache/judge_exp23): {'no': 2422, 'yes': 1321}
