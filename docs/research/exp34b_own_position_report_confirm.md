# §34b report: own answer position vs forced position (folds: 2, 3, 4; 204 questions paired across readers and modes; k = 10)

Own position = the cached output teacher-forced up to its last `Answer:`; forced = `Answer:` right after the prompt (§34). Lenses, tracked tokens and labels as in §34. Verdict rows follow the §34 gate (3B on the logit lens, 7B on the J-lens).

## qwen3b: 165 of 204 cached outputs carry an `Answer:` marker; the rest are read at their first token

| lens | n | emerged forced | emerged own | Claim A: own − forced, share emerged (95% CI) | verdict | n both emerged | depth forced (median) | depth own (median) | Claim B: forced − own depth (95% CI) | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| J-lens | 204 | 0.676 | 0.676 | +0.000 (-0.059, +0.059) | NOT SHOWN (beside) | 119 | 0.889 | 0.917 | -0.163 (-0.209, -0.118) | NOT SHOWN (beside) |
| logit (gate fallback), **decides** | 204 | 0.701 | 0.711 | +0.010 (-0.049, +0.069) | NOT SHOWN | 124 | 0.917 | 0.917 | +0.015 (-0.001, +0.033) | NOT SHOWN |

| lens | mode | class | n | A | B | C |
|---|---|---|---|---|---|---|
| J-lens | forced | refusal | 65 | 0.431 | 0.000 | 0.569 |
| J-lens | forced | answered wrong | 62 | 0.290 | 0.000 | 0.710 |
| J-lens | forced | all failures | 127 | 0.362 | 0.000 | 0.638 |
| J-lens | forced | success | 77 | 0.260 | 0.013 | 0.727 |
| J-lens | own | refusal | 65 | 0.369 | 0.000 | 0.631 |
| J-lens | own | answered wrong | 62 | 0.371 | 0.000 | 0.629 |
| J-lens | own | all failures | 127 | 0.370 | 0.000 | 0.630 |
| J-lens | own | success | 77 | 0.247 | 0.026 | 0.727 |
| logit (gate fallback) | forced | refusal | 65 | 0.400 | 0.031 | 0.569 |
| logit (gate fallback) | forced | answered wrong | 62 | 0.290 | 0.000 | 0.710 |
| logit (gate fallback) | forced | all failures | 127 | 0.346 | 0.016 | 0.638 |
| logit (gate fallback) | forced | success | 77 | 0.221 | 0.052 | 0.727 |
| logit (gate fallback) | own | refusal | 65 | 0.308 | 0.062 | 0.631 |
| logit (gate fallback) | own | answered wrong | 62 | 0.339 | 0.032 | 0.629 |
| logit (gate fallback) | own | all failures | 127 | 0.323 | 0.047 | 0.630 |
| logit (gate fallback) | own | success | 77 | 0.234 | 0.039 | 0.727 |

Refusals (65) at the own position, jlens: gold never enters top-k; unknown reaches top-1: 23 (0.35); gold enters top-k; unknown never top-1: 20 (0.31); unknown top-1 BEFORE gold enters top-k: 15 (0.23); same layer: 5 (0.08); neither (gold never top-k, unknown never top-1): 1 (0.02); gold enters top-k BEFORE unknown top-1: 1 (0.02). Final-layer rank: gold median 4, ` unknown` median 1.

Refusals (65) at the own position, logit: gold never enters top-k; unknown reaches top-1: 19 (0.29); unknown top-1 BEFORE gold enters top-k: 17 (0.26); gold enters top-k; unknown never top-1: 17 (0.26); gold enters top-k BEFORE unknown top-1: 7 (0.11); same layer: 4 (0.06); neither (gold never top-k, unknown never top-1): 1 (0.02). Final-layer rank: gold median 4, ` unknown` median 1.

Marker-less outputs (39; 23 judged correct), jlens: gold emerges on 17 (0.436).
Marker-less outputs (39; 23 judged correct), logit: gold emerges on 17 (0.436).

## qwen7b: 204 of 204 cached outputs carry an `Answer:` marker; the rest are read at their first token

| lens | n | emerged forced | emerged own | Claim A: own − forced, share emerged (95% CI) | verdict | n both emerged | depth forced (median) | depth own (median) | Claim B: forced − own depth (95% CI) | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| J-lens, **decides** | 204 | 0.931 | 0.907 | -0.025 (-0.059, +0.010) | descriptive | 181 | 0.429 | 0.536 | -0.020 (-0.042, +0.002) | descriptive |
| logit (shown beside) | 204 | 0.902 | 0.868 | -0.034 (-0.069, +0.000) | descriptive | 174 | 0.821 | 0.821 | -0.057 (-0.091, -0.028) | descriptive |

| lens | mode | class | n | A | B | C |
|---|---|---|---|---|---|---|
| J-lens | forced | refusal | 9 | 0.222 | 0.111 | 0.667 |
| J-lens | forced | answered wrong | 54 | 0.111 | 0.130 | 0.759 |
| J-lens | forced | all failures | 63 | 0.127 | 0.127 | 0.746 |
| J-lens | forced | success | 141 | 0.043 | 0.007 | 0.950 |
| J-lens | own | refusal | 9 | 0.333 | 0.222 | 0.444 |
| J-lens | own | answered wrong | 54 | 0.241 | 0.111 | 0.648 |
| J-lens | own | all failures | 63 | 0.254 | 0.127 | 0.619 |
| J-lens | own | success | 141 | 0.021 | 0.014 | 0.965 |
| logit (shown beside) | forced | refusal | 9 | 0.222 | 0.111 | 0.667 |
| logit (shown beside) | forced | answered wrong | 54 | 0.222 | 0.019 | 0.759 |
| logit (shown beside) | forced | all failures | 63 | 0.222 | 0.032 | 0.746 |
| logit (shown beside) | forced | success | 141 | 0.043 | 0.007 | 0.950 |
| logit (shown beside) | own | refusal | 9 | 0.556 | 0.000 | 0.444 |
| logit (shown beside) | own | answered wrong | 54 | 0.333 | 0.019 | 0.648 |
| logit (shown beside) | own | all failures | 63 | 0.365 | 0.016 | 0.619 |
| logit (shown beside) | own | success | 141 | 0.028 | 0.007 | 0.965 |

Refusals (9) at the own position, jlens: gold enters top-k BEFORE unknown top-1: 5 (0.56); gold never enters top-k; unknown reaches top-1: 3 (0.33); gold enters top-k; unknown never top-1: 1 (0.11). Final-layer rank: gold median 12, ` unknown` median 1.

Refusals (9) at the own position, logit: gold never enters top-k; unknown reaches top-1: 5 (0.56); gold enters top-k; unknown never top-1: 2 (0.22); same layer: 1 (0.11); gold enters top-k BEFORE unknown top-1: 1 (0.11). Final-layer rank: gold median 12, ` unknown` median 1.

Verdict rules: Claim A SHOWN if the lower bound of (own − forced) share emerged is above 0; Claim B SHOWN if the lower bound of (forced − own) depth among questions emerged in both modes is above 0. Pre-registered for qwen3b on its deciding (logit) row.
