# §34b report: own answer position vs forced position (folds: 0, 1; 126 questions paired across readers and modes; k = 10)

Own position = the cached output teacher-forced up to its last `Answer:`; forced = `Answer:` right after the prompt (§34). Lenses, tracked tokens and labels as in §34. Verdict rows follow the §34 gate (3B on the logit lens, 7B on the J-lens).

## qwen3b: 100 of 126 cached outputs carry an `Answer:` marker; the rest are read at their first token

| lens | n | emerged forced | emerged own | Claim A: own − forced, share emerged (95% CI) | verdict | n both emerged | depth forced (median) | depth own (median) | Claim B: forced − own depth (95% CI) | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| J-lens | 126 | 0.714 | 0.698 | -0.016 (-0.079, +0.048) | NOT SHOWN (beside) | 81 | 0.889 | 0.917 | -0.102 (-0.157, -0.048) | NOT SHOWN (beside) |
| logit (gate fallback), **decides** | 126 | 0.706 | 0.722 | +0.016 (-0.056, +0.087) | NOT SHOWN | 80 | 0.917 | 0.889 | +0.005 (-0.014, +0.019) | NOT SHOWN |

| lens | mode | class | n | A | B | C |
|---|---|---|---|---|---|---|
| J-lens | forced | refusal | 32 | 0.375 | 0.062 | 0.562 |
| J-lens | forced | answered wrong | 40 | 0.275 | 0.025 | 0.700 |
| J-lens | forced | all failures | 72 | 0.319 | 0.042 | 0.639 |
| J-lens | forced | success | 54 | 0.241 | 0.019 | 0.741 |
| J-lens | own | refusal | 32 | 0.406 | 0.031 | 0.562 |
| J-lens | own | answered wrong | 40 | 0.300 | 0.025 | 0.675 |
| J-lens | own | all failures | 72 | 0.347 | 0.028 | 0.625 |
| J-lens | own | success | 54 | 0.241 | 0.000 | 0.759 |
| logit (gate fallback) | forced | refusal | 32 | 0.375 | 0.062 | 0.562 |
| logit (gate fallback) | forced | answered wrong | 40 | 0.275 | 0.025 | 0.700 |
| logit (gate fallback) | forced | all failures | 72 | 0.319 | 0.042 | 0.639 |
| logit (gate fallback) | forced | success | 54 | 0.259 | 0.000 | 0.741 |
| logit (gate fallback) | own | refusal | 32 | 0.312 | 0.125 | 0.562 |
| logit (gate fallback) | own | answered wrong | 40 | 0.300 | 0.025 | 0.675 |
| logit (gate fallback) | own | all failures | 72 | 0.306 | 0.069 | 0.625 |
| logit (gate fallback) | own | success | 54 | 0.241 | 0.000 | 0.759 |

Refusals (32) at the own position, jlens: gold never enters top-k; unknown reaches top-1: 11 (0.34); gold enters top-k; unknown never top-1: 7 (0.22); unknown top-1 BEFORE gold enters top-k: 7 (0.22); same layer: 4 (0.12); neither (gold never top-k, unknown never top-1): 2 (0.06); gold enters top-k BEFORE unknown top-1: 1 (0.03). Final-layer rank: gold median 6, ` unknown` median 1.

Refusals (32) at the own position, logit: gold never enters top-k; unknown reaches top-1: 8 (0.25); gold enters top-k; unknown never top-1: 7 (0.22); unknown top-1 BEFORE gold enters top-k: 7 (0.22); gold enters top-k BEFORE unknown top-1: 6 (0.19); neither (gold never top-k, unknown never top-1): 2 (0.06); same layer: 2 (0.06). Final-layer rank: gold median 6, ` unknown` median 1.

Marker-less outputs (26; 15 judged correct), jlens: gold emerges on 7 (0.269).
Marker-less outputs (26; 15 judged correct), logit: gold emerges on 7 (0.269).

## qwen7b: 126 of 126 cached outputs carry an `Answer:` marker; the rest are read at their first token

| lens | n | emerged forced | emerged own | Claim A: own − forced, share emerged (95% CI) | verdict | n both emerged | depth forced (median) | depth own (median) | Claim B: forced − own depth (95% CI) | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| J-lens, **decides** | 126 | 0.913 | 0.897 | -0.016 (-0.056, +0.024) | descriptive | 111 | 0.429 | 0.464 | +0.002 (-0.023, +0.026) | descriptive |
| logit (shown beside) | 126 | 0.873 | 0.865 | -0.008 (-0.040, +0.024) | descriptive | 107 | 0.821 | 0.857 | -0.100 (-0.153, -0.054) | descriptive |

| lens | mode | class | n | A | B | C |
|---|---|---|---|---|---|---|
| J-lens | forced | refusal | 9 | 0.222 | 0.000 | 0.778 |
| J-lens | forced | answered wrong | 38 | 0.158 | 0.079 | 0.763 |
| J-lens | forced | all failures | 47 | 0.170 | 0.064 | 0.766 |
| J-lens | forced | success | 79 | 0.038 | 0.038 | 0.924 |
| J-lens | own | refusal | 9 | 0.222 | 0.222 | 0.556 |
| J-lens | own | answered wrong | 38 | 0.237 | 0.079 | 0.684 |
| J-lens | own | all failures | 47 | 0.234 | 0.106 | 0.660 |
| J-lens | own | success | 79 | 0.025 | 0.013 | 0.962 |
| logit (shown beside) | forced | refusal | 9 | 0.222 | 0.000 | 0.778 |
| logit (shown beside) | forced | answered wrong | 38 | 0.237 | 0.000 | 0.763 |
| logit (shown beside) | forced | all failures | 47 | 0.234 | 0.000 | 0.766 |
| logit (shown beside) | forced | success | 79 | 0.063 | 0.013 | 0.924 |
| logit (shown beside) | own | refusal | 9 | 0.333 | 0.111 | 0.556 |
| logit (shown beside) | own | answered wrong | 38 | 0.289 | 0.026 | 0.684 |
| logit (shown beside) | own | all failures | 47 | 0.298 | 0.043 | 0.660 |
| logit (shown beside) | own | success | 79 | 0.038 | 0.000 | 0.962 |

Refusals (9) at the own position, jlens: gold enters top-k BEFORE unknown top-1: 6 (0.67); gold never enters top-k; unknown reaches top-1: 2 (0.22); unknown top-1 BEFORE gold enters top-k: 1 (0.11). Final-layer rank: gold median 6, ` unknown` median 1.

Refusals (9) at the own position, logit: unknown top-1 BEFORE gold enters top-k: 3 (0.33); gold never enters top-k; unknown reaches top-1: 3 (0.33); gold enters top-k; unknown never top-1: 1 (0.11); gold enters top-k BEFORE unknown top-1: 1 (0.11); same layer: 1 (0.11). Final-layer rank: gold median 6, ` unknown` median 1.

Verdict rules: Claim A SHOWN if the lower bound of (own − forced) share emerged is above 0; Claim B SHOWN if the lower bound of (forced − own) depth among questions emerged in both modes is above 0. Pre-registered for qwen3b on its deciding (logit) row.
