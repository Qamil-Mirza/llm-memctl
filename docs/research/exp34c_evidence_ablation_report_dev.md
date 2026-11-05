# §34c report: evidence cut out vs intact, forced position (folds: 0, 1; 126 questions paired; k = 10)

Ablated = every evidence line removed from the prompt by its exact span; everything else identical. Lenses, tracked tokens, labels and population as §34. Outcome C uses the final layer (model logits), so Claim B is lens-free.

## qwen3b

| lens | n | ever in top-k: intact | ablated | Claim A intact − ablated (95% CI) | verdict | final-layer top-k (C): intact | ablated | Claim B intact − ablated (95% CI) | verdict | log10 best rank intact − ablated (evidence lift) |
|---|---|---|---|---|---|---|---|---|---|---|
| J-lens (beside) | 126 | 0.714 | 0.476 | +0.238 (+0.167, +0.310) | descriptive | 0.683 | 0.437 | +0.246 (+0.175, +0.325) | descriptive | +0.580 (+0.445, +0.719) |
| logit (gate fallback), **decides** | 126 | 0.706 | 0.468 | +0.238 (+0.167, +0.317) | SHOWN | 0.683 | 0.437 | +0.246 (+0.175, +0.325) | SHOWN | +0.663 (+0.511, +0.820) |

Plausibility share: among questions C (final-layer top-k) WITH the evidence, the share still C WITHOUT it; and among questions where the gold ever emerges with the evidence, the share where it still emerges without it.

| lens | class | n C with evidence | still C without (95% CI) | n ever-emerges with | still emerges without (95% CI) |
|---|---|---|---|---|---|
| J-lens | refusal | 18 | 0.500 (0.278, 0.722) | 20 | 0.600 (0.400, 0.800) |
| J-lens | answered wrong | 28 | 0.786 (0.643, 0.929) | 29 | 0.759 (0.586, 0.897) |
| J-lens | all failures | 46 | 0.674 (0.543, 0.804) | 49 | 0.694 (0.551, 0.816) |
| J-lens | success | 40 | 0.600 (0.450, 0.750) | 41 | 0.634 (0.488, 0.780) |
| J-lens | all | 86 | 0.640 (0.535, 0.744) | 90 | 0.667 (0.567, 0.756) |
| J-lens | strict gold token | 75 | 0.693 (0.587, 0.800) | 79 | 0.722 (0.620, 0.823) |
| J-lens | function-word gold token | 11 | 0.273 (0.000, 0.545) | 11 | 0.273 (0.000, 0.545) |
| logit | refusal | 18 | 0.500 (0.278, 0.722) | 20 | 0.550 (0.350, 0.750) |
| logit | answered wrong | 28 | 0.786 (0.643, 0.929) | 29 | 0.759 (0.586, 0.897) |
| logit | all failures | 46 | 0.674 (0.543, 0.804) | 49 | 0.673 (0.531, 0.796) |
| logit | success | 40 | 0.600 (0.450, 0.750) | 40 | 0.650 (0.500, 0.800) |
| logit | all | 86 | 0.640 (0.535, 0.744) | 89 | 0.663 (0.562, 0.753) |
| logit | strict gold token | 75 | 0.693 (0.587, 0.800) | 77 | 0.714 (0.610, 0.818) |
| logit | function-word gold token | 11 | 0.273 (0.000, 0.545) | 12 | 0.333 (0.083, 0.583) |

` unknown` reaches top-1 at P1 (jlens): with evidence 74 of 126 (0.587); without 79 (0.627).
` unknown` reaches top-1 at P1 (logit): with evidence 74 of 126 (0.587); without 77 (0.611).

## qwen7b

| lens | n | ever in top-k: intact | ablated | Claim A intact − ablated (95% CI) | verdict | final-layer top-k (C): intact | ablated | Claim B intact − ablated (95% CI) | verdict | log10 best rank intact − ablated (evidence lift) |
|---|---|---|---|---|---|---|---|---|---|---|
| J-lens, **decides** | 126 | 0.913 | 0.635 | +0.278 (+0.190, +0.365) | descriptive | 0.865 | 0.579 | +0.286 (+0.198, +0.373) | descriptive | +0.708 (+0.536, +0.890) |
| logit (beside) | 126 | 0.873 | 0.619 | +0.254 (+0.167, +0.341) | descriptive | 0.865 | 0.579 | +0.286 (+0.198, +0.373) | descriptive | +0.746 (+0.574, +0.928) |

Plausibility share: among questions C (final-layer top-k) WITH the evidence, the share still C WITHOUT it; and among questions where the gold ever emerges with the evidence, the share where it still emerges without it.

| lens | class | n C with evidence | still C without (95% CI) | n ever-emerges with | still emerges without (95% CI) |
|---|---|---|---|---|---|
| J-lens | refusal | 7 | 0.714 (0.429, 1.000) | 7 | 0.857 (0.571, 1.000) |
| J-lens | answered wrong | 29 | 0.724 (0.552, 0.862) | 32 | 0.750 (0.594, 0.906) |
| J-lens | all failures | 36 | 0.722 (0.583, 0.861) | 39 | 0.769 (0.641, 0.897) |
| J-lens | success | 73 | 0.603 (0.493, 0.712) | 76 | 0.632 (0.526, 0.737) |
| J-lens | all | 109 | 0.642 (0.550, 0.734) | 115 | 0.678 (0.591, 0.757) |
| J-lens | strict gold token | 92 | 0.652 (0.554, 0.750) | 95 | 0.674 (0.579, 0.768) |
| J-lens | function-word gold token | 17 | 0.588 (0.353, 0.824) | 20 | 0.700 (0.500, 0.900) |
| logit | refusal | 7 | 0.714 (0.429, 1.000) | 7 | 0.857 (0.571, 1.000) |
| logit | answered wrong | 29 | 0.724 (0.552, 0.862) | 29 | 0.793 (0.621, 0.931) |
| logit | all failures | 36 | 0.722 (0.583, 0.861) | 36 | 0.806 (0.667, 0.917) |
| logit | success | 73 | 0.603 (0.493, 0.712) | 74 | 0.608 (0.500, 0.716) |
| logit | all | 109 | 0.642 (0.550, 0.734) | 110 | 0.673 (0.582, 0.755) |
| logit | strict gold token | 92 | 0.652 (0.554, 0.750) | 92 | 0.663 (0.565, 0.761) |
| logit | function-word gold token | 17 | 0.588 (0.353, 0.824) | 18 | 0.722 (0.500, 0.889) |

` unknown` reaches top-1 at P1 (jlens): with evidence 11 of 126 (0.087); without 76 (0.603).
` unknown` reaches top-1 at P1 (logit): with evidence 12 of 126 (0.095); without 76 (0.603).

Verdict rules: Claims A and B SHOWN if the lower bound of (intact − ablated) is above 0; pre-registered for qwen3b on its deciding (logit) row. Labels are the §23 single-judge verdicts.
