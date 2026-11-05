# §34c report: evidence cut out vs intact, forced position (folds: 2, 3, 4; 204 questions paired; k = 10)

Ablated = every evidence line removed from the prompt by its exact span; everything else identical. Lenses, tracked tokens, labels and population as §34. Outcome C uses the final layer (model logits), so Claim B is lens-free.

## qwen3b

| lens | n | ever in top-k: intact | ablated | Claim A intact − ablated (95% CI) | verdict | final-layer top-k (C): intact | ablated | Claim B intact − ablated (95% CI) | verdict | log10 best rank intact − ablated (evidence lift) |
|---|---|---|---|---|---|---|---|---|---|---|
| J-lens (beside) | 204 | 0.676 | 0.441 | +0.235 (+0.176, +0.294) | descriptive | 0.672 | 0.431 | +0.240 (+0.181, +0.299) | descriptive | +0.528 (+0.425, +0.639) |
| logit (gate fallback), **decides** | 204 | 0.701 | 0.466 | +0.235 (+0.176, +0.294) | SHOWN | 0.672 | 0.431 | +0.240 (+0.181, +0.299) | SHOWN | +0.631 (+0.507, +0.763) |

Plausibility share: among questions C (final-layer top-k) WITH the evidence, the share still C WITHOUT it; and among questions where the gold ever emerges with the evidence, the share where it still emerges without it.

| lens | class | n C with evidence | still C without (95% CI) | n ever-emerges with | still emerges without (95% CI) |
|---|---|---|---|---|---|
| J-lens | refusal | 37 | 0.568 (0.405, 0.730) | 37 | 0.568 (0.405, 0.730) |
| J-lens | answered wrong | 44 | 0.636 (0.500, 0.773) | 44 | 0.659 (0.523, 0.795) |
| J-lens | all failures | 81 | 0.605 (0.494, 0.716) | 81 | 0.617 (0.506, 0.716) |
| J-lens | success | 56 | 0.696 (0.571, 0.804) | 57 | 0.702 (0.579, 0.825) |
| J-lens | all | 137 | 0.642 (0.562, 0.723) | 138 | 0.652 (0.572, 0.732) |
| J-lens | strict gold token | 116 | 0.707 (0.621, 0.784) | 117 | 0.718 (0.641, 0.795) |
| J-lens | function-word gold token | 21 | 0.286 (0.095, 0.476) | 21 | 0.286 (0.095, 0.476) |
| logit | refusal | 37 | 0.568 (0.405, 0.730) | 39 | 0.590 (0.436, 0.744) |
| logit | answered wrong | 44 | 0.636 (0.500, 0.773) | 44 | 0.636 (0.500, 0.773) |
| logit | all failures | 81 | 0.605 (0.494, 0.716) | 83 | 0.614 (0.506, 0.723) |
| logit | success | 56 | 0.696 (0.571, 0.804) | 60 | 0.717 (0.600, 0.833) |
| logit | all | 137 | 0.642 (0.562, 0.723) | 143 | 0.657 (0.580, 0.734) |
| logit | strict gold token | 116 | 0.707 (0.621, 0.784) | 121 | 0.702 (0.620, 0.785) |
| logit | function-word gold token | 21 | 0.286 (0.095, 0.476) | 22 | 0.409 (0.227, 0.636) |

` unknown` reaches top-1 at P1 (jlens): with evidence 117 of 204 (0.574); without 132 (0.647).
` unknown` reaches top-1 at P1 (logit): with evidence 118 of 204 (0.578); without 129 (0.632).

## qwen7b

| lens | n | ever in top-k: intact | ablated | Claim A intact − ablated (95% CI) | verdict | final-layer top-k (C): intact | ablated | Claim B intact − ablated (95% CI) | verdict | log10 best rank intact − ablated (evidence lift) |
|---|---|---|---|---|---|---|---|---|---|---|
| J-lens, **decides** | 204 | 0.931 | 0.696 | +0.235 (+0.172, +0.294) | descriptive | 0.887 | 0.627 | +0.260 (+0.191, +0.328) | descriptive | +0.597 (+0.473, +0.730) |
| logit (beside) | 204 | 0.902 | 0.662 | +0.240 (+0.172, +0.309) | descriptive | 0.887 | 0.627 | +0.260 (+0.191, +0.328) | descriptive | +0.644 (+0.510, +0.784) |

Plausibility share: among questions C (final-layer top-k) WITH the evidence, the share still C WITHOUT it; and among questions where the gold ever emerges with the evidence, the share where it still emerges without it.

| lens | class | n C with evidence | still C without (95% CI) | n ever-emerges with | still emerges without (95% CI) |
|---|---|---|---|---|---|
| J-lens | refusal | 6 | 0.833 (0.500, 1.000) | 7 | 0.857 (0.571, 1.000) |
| J-lens | answered wrong | 41 | 0.732 (0.585, 0.854) | 48 | 0.750 (0.625, 0.875) |
| J-lens | all failures | 47 | 0.745 (0.617, 0.851) | 55 | 0.764 (0.655, 0.873) |
| J-lens | success | 134 | 0.657 (0.575, 0.739) | 135 | 0.726 (0.652, 0.800) |
| J-lens | all | 181 | 0.680 (0.608, 0.746) | 190 | 0.737 (0.674, 0.795) |
| J-lens | strict gold token | 144 | 0.694 (0.618, 0.771) | 145 | 0.745 (0.676, 0.814) |
| J-lens | function-word gold token | 37 | 0.622 (0.459, 0.784) | 45 | 0.711 (0.578, 0.844) |
| logit | refusal | 6 | 0.833 (0.500, 1.000) | 7 | 0.714 (0.429, 1.000) |
| logit | answered wrong | 41 | 0.732 (0.585, 0.854) | 42 | 0.738 (0.595, 0.857) |
| logit | all failures | 47 | 0.745 (0.617, 0.851) | 49 | 0.735 (0.612, 0.857) |
| logit | success | 134 | 0.657 (0.575, 0.739) | 135 | 0.681 (0.600, 0.756) |
| logit | all | 181 | 0.680 (0.608, 0.746) | 184 | 0.696 (0.630, 0.761) |
| logit | strict gold token | 144 | 0.694 (0.618, 0.771) | 145 | 0.703 (0.628, 0.779) |
| logit | function-word gold token | 37 | 0.622 (0.459, 0.784) | 39 | 0.667 (0.513, 0.821) |

` unknown` reaches top-1 at P1 (jlens): with evidence 16 of 204 (0.078); without 108 (0.529).
` unknown` reaches top-1 at P1 (logit): with evidence 21 of 204 (0.103); without 107 (0.525).

Verdict rules: Claims A and B SHOWN if the lower bound of (intact − ablated) is above 0; pre-registered for qwen3b on its deciding (logit) row. Labels are the §23 single-judge verdicts.
