# §34b report: own answer position vs forced position (folds: 0, 1, 2, 3, 4; 330 questions paired across readers and modes; k = 10)

Own position = the cached output teacher-forced up to its last `Answer:`; forced = `Answer:` right after the prompt (§34). Lenses, tracked tokens and labels as in §34. Verdict rows follow the §34 gate (3B on the logit lens, 7B on the J-lens).

## qwen3b: 265 of 330 cached outputs carry an `Answer:` marker; the rest are read at their first token

| lens | n | emerged forced | emerged own | Claim A: own − forced, share emerged (95% CI) | verdict | n both emerged | depth forced (median) | depth own (median) | Claim B: forced − own depth (95% CI) | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| J-lens | 330 | 0.691 | 0.685 | -0.006 (-0.048, +0.036) | NOT SHOWN (beside) | 200 | 0.889 | 0.917 | -0.138 (-0.174, -0.104) | NOT SHOWN (beside) |
| logit (gate fallback), **decides** | 330 | 0.703 | 0.715 | +0.012 (-0.033, +0.058) | NOT SHOWN | 204 | 0.917 | 0.889 | +0.011 (-0.001, +0.024) | NOT SHOWN |

| lens | mode | class | n | A | B | C |
|---|---|---|---|---|---|---|
| J-lens | forced | refusal | 97 | 0.412 | 0.021 | 0.567 |
| J-lens | forced | answered wrong | 102 | 0.284 | 0.010 | 0.706 |
| J-lens | forced | all failures | 199 | 0.347 | 0.015 | 0.638 |
| J-lens | forced | success | 131 | 0.252 | 0.015 | 0.733 |
| J-lens | own | refusal | 97 | 0.381 | 0.010 | 0.608 |
| J-lens | own | answered wrong | 102 | 0.343 | 0.010 | 0.647 |
| J-lens | own | all failures | 199 | 0.362 | 0.010 | 0.628 |
| J-lens | own | success | 131 | 0.244 | 0.015 | 0.740 |
| logit (gate fallback) | forced | refusal | 97 | 0.392 | 0.041 | 0.567 |
| logit (gate fallback) | forced | answered wrong | 102 | 0.284 | 0.010 | 0.706 |
| logit (gate fallback) | forced | all failures | 199 | 0.337 | 0.025 | 0.638 |
| logit (gate fallback) | forced | success | 131 | 0.237 | 0.031 | 0.733 |
| logit (gate fallback) | own | refusal | 97 | 0.309 | 0.082 | 0.608 |
| logit (gate fallback) | own | answered wrong | 102 | 0.324 | 0.029 | 0.647 |
| logit (gate fallback) | own | all failures | 199 | 0.317 | 0.055 | 0.628 |
| logit (gate fallback) | own | success | 131 | 0.237 | 0.023 | 0.740 |

Refusals (97) at the own position, jlens: gold never enters top-k; unknown reaches top-1: 34 (0.35); gold enters top-k; unknown never top-1: 27 (0.28); unknown top-1 BEFORE gold enters top-k: 22 (0.23); same layer: 9 (0.09); neither (gold never top-k, unknown never top-1): 3 (0.03); gold enters top-k BEFORE unknown top-1: 2 (0.02). Final-layer rank: gold median 5, ` unknown` median 1.

Refusals (97) at the own position, logit: gold never enters top-k; unknown reaches top-1: 27 (0.28); unknown top-1 BEFORE gold enters top-k: 24 (0.25); gold enters top-k; unknown never top-1: 24 (0.25); gold enters top-k BEFORE unknown top-1: 13 (0.13); same layer: 6 (0.06); neither (gold never top-k, unknown never top-1): 3 (0.03). Final-layer rank: gold median 5, ` unknown` median 1.

Marker-less outputs (65; 38 judged correct), jlens: gold emerges on 24 (0.369).
Marker-less outputs (65; 38 judged correct), logit: gold emerges on 24 (0.369).

## qwen7b: 330 of 330 cached outputs carry an `Answer:` marker; the rest are read at their first token

| lens | n | emerged forced | emerged own | Claim A: own − forced, share emerged (95% CI) | verdict | n both emerged | depth forced (median) | depth own (median) | Claim B: forced − own depth (95% CI) | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| J-lens, **decides** | 330 | 0.924 | 0.903 | -0.021 (-0.048, +0.003) | descriptive | 292 | 0.429 | 0.482 | -0.011 (-0.028, +0.005) | descriptive |
| logit (shown beside) | 330 | 0.891 | 0.867 | -0.024 (-0.048, +0.000) | descriptive | 281 | 0.821 | 0.857 | -0.074 (-0.103, -0.048) | descriptive |

| lens | mode | class | n | A | B | C |
|---|---|---|---|---|---|---|
| J-lens | forced | refusal | 18 | 0.222 | 0.056 | 0.722 |
| J-lens | forced | answered wrong | 92 | 0.130 | 0.109 | 0.761 |
| J-lens | forced | all failures | 110 | 0.145 | 0.100 | 0.755 |
| J-lens | forced | success | 220 | 0.041 | 0.018 | 0.941 |
| J-lens | own | refusal | 18 | 0.278 | 0.222 | 0.500 |
| J-lens | own | answered wrong | 92 | 0.239 | 0.098 | 0.663 |
| J-lens | own | all failures | 110 | 0.245 | 0.118 | 0.636 |
| J-lens | own | success | 220 | 0.023 | 0.014 | 0.964 |
| logit (shown beside) | forced | refusal | 18 | 0.222 | 0.056 | 0.722 |
| logit (shown beside) | forced | answered wrong | 92 | 0.228 | 0.011 | 0.761 |
| logit (shown beside) | forced | all failures | 110 | 0.227 | 0.018 | 0.755 |
| logit (shown beside) | forced | success | 220 | 0.050 | 0.009 | 0.941 |
| logit (shown beside) | own | refusal | 18 | 0.444 | 0.056 | 0.500 |
| logit (shown beside) | own | answered wrong | 92 | 0.315 | 0.022 | 0.663 |
| logit (shown beside) | own | all failures | 110 | 0.336 | 0.027 | 0.636 |
| logit (shown beside) | own | success | 220 | 0.032 | 0.005 | 0.964 |

Refusals (18) at the own position, jlens: gold enters top-k BEFORE unknown top-1: 11 (0.61); gold never enters top-k; unknown reaches top-1: 5 (0.28); gold enters top-k; unknown never top-1: 1 (0.06); unknown top-1 BEFORE gold enters top-k: 1 (0.06). Final-layer rank: gold median 12, ` unknown` median 1.

Refusals (18) at the own position, logit: gold never enters top-k; unknown reaches top-1: 8 (0.44); gold enters top-k; unknown never top-1: 3 (0.17); unknown top-1 BEFORE gold enters top-k: 3 (0.17); same layer: 2 (0.11); gold enters top-k BEFORE unknown top-1: 2 (0.11). Final-layer rank: gold median 12, ` unknown` median 1.

Verdict rules: Claim A SHOWN if the lower bound of (own − forced) share emerged is above 0; Claim B SHOWN if the lower bound of (forced − own) depth among questions emerged in both modes is above 0. Pre-registered for qwen3b on its deciding (logit) row.
