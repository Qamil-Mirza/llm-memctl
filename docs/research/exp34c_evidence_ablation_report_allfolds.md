# §34c report: evidence cut out vs intact, forced position (folds: 0, 1, 2, 3, 4; 330 questions paired; k = 10)

Ablated = every evidence line removed from the prompt by its exact span; everything else identical. Lenses, tracked tokens, labels and population as §34. Outcome C uses the final layer (model logits), so Claim B is lens-free.

## qwen3b

| lens | n | ever in top-k: intact | ablated | Claim A intact − ablated (95% CI) | verdict | final-layer top-k (C): intact | ablated | Claim B intact − ablated (95% CI) | verdict | log10 best rank intact − ablated (evidence lift) |
|---|---|---|---|---|---|---|---|---|---|---|
| J-lens (beside) | 330 | 0.691 | 0.455 | +0.236 (+0.191, +0.282) | descriptive | 0.676 | 0.433 | +0.242 (+0.197, +0.288) | descriptive | +0.548 (+0.465, +0.636) |
| logit (gate fallback), **decides** | 330 | 0.703 | 0.467 | +0.236 (+0.191, +0.282) | SHOWN | 0.676 | 0.433 | +0.242 (+0.197, +0.288) | SHOWN | +0.643 (+0.546, +0.745) |

Plausibility share: among questions C (final-layer top-k) WITH the evidence, the share still C WITHOUT it; and among questions where the gold ever emerges with the evidence, the share where it still emerges without it.

| lens | class | n C with evidence | still C without (95% CI) | n ever-emerges with | still emerges without (95% CI) |
|---|---|---|---|---|---|
| J-lens | refusal | 55 | 0.545 (0.418, 0.673) | 57 | 0.579 (0.456, 0.702) |
| J-lens | answered wrong | 72 | 0.694 (0.583, 0.806) | 73 | 0.699 (0.589, 0.808) |
| J-lens | all failures | 127 | 0.630 (0.551, 0.709) | 130 | 0.646 (0.562, 0.723) |
| J-lens | success | 96 | 0.656 (0.562, 0.750) | 98 | 0.673 (0.582, 0.765) |
| J-lens | all | 223 | 0.641 (0.578, 0.704) | 228 | 0.658 (0.592, 0.719) |
| J-lens | strict gold token | 191 | 0.702 (0.639, 0.764) | 196 | 0.719 (0.658, 0.781) |
| J-lens | function-word gold token | 32 | 0.281 (0.125, 0.438) | 32 | 0.281 (0.125, 0.438) |
| logit | refusal | 55 | 0.545 (0.418, 0.673) | 59 | 0.576 (0.441, 0.695) |
| logit | answered wrong | 72 | 0.694 (0.583, 0.806) | 73 | 0.685 (0.575, 0.795) |
| logit | all failures | 127 | 0.630 (0.551, 0.709) | 132 | 0.636 (0.553, 0.720) |
| logit | success | 96 | 0.656 (0.562, 0.750) | 100 | 0.690 (0.600, 0.780) |
| logit | all | 223 | 0.641 (0.578, 0.704) | 232 | 0.659 (0.595, 0.720) |
| logit | strict gold token | 191 | 0.702 (0.639, 0.764) | 198 | 0.707 (0.641, 0.768) |
| logit | function-word gold token | 32 | 0.281 (0.125, 0.438) | 34 | 0.382 (0.235, 0.559) |

` unknown` reaches top-1 at P1 (jlens): with evidence 191 of 330 (0.579); without 211 (0.639).
` unknown` reaches top-1 at P1 (logit): with evidence 192 of 330 (0.582); without 206 (0.624).

## qwen7b

| lens | n | ever in top-k: intact | ablated | Claim A intact − ablated (95% CI) | verdict | final-layer top-k (C): intact | ablated | Claim B intact − ablated (95% CI) | verdict | log10 best rank intact − ablated (evidence lift) |
|---|---|---|---|---|---|---|---|---|---|---|
| J-lens, **decides** | 330 | 0.924 | 0.673 | +0.252 (+0.203, +0.303) | descriptive | 0.879 | 0.609 | +0.270 (+0.215, +0.324) | descriptive | +0.640 (+0.536, +0.745) |
| logit (beside) | 330 | 0.891 | 0.645 | +0.245 (+0.191, +0.300) | descriptive | 0.879 | 0.609 | +0.270 (+0.215, +0.324) | descriptive | +0.683 (+0.574, +0.793) |

Plausibility share: among questions C (final-layer top-k) WITH the evidence, the share still C WITHOUT it; and among questions where the gold ever emerges with the evidence, the share where it still emerges without it.

| lens | class | n C with evidence | still C without (95% CI) | n ever-emerges with | still emerges without (95% CI) |
|---|---|---|---|---|---|
| J-lens | refusal | 13 | 0.769 (0.538, 1.000) | 14 | 0.857 (0.643, 1.000) |
| J-lens | answered wrong | 70 | 0.729 (0.629, 0.829) | 80 | 0.750 (0.650, 0.838) |
| J-lens | all failures | 83 | 0.735 (0.639, 0.831) | 94 | 0.766 (0.681, 0.851) |
| J-lens | success | 207 | 0.638 (0.570, 0.700) | 211 | 0.692 (0.630, 0.754) |
| J-lens | all | 290 | 0.666 (0.610, 0.717) | 305 | 0.715 (0.662, 0.764) |
| J-lens | strict gold token | 236 | 0.678 (0.619, 0.737) | 240 | 0.717 (0.658, 0.771) |
| J-lens | function-word gold token | 54 | 0.611 (0.481, 0.741) | 65 | 0.708 (0.600, 0.815) |
| logit | refusal | 13 | 0.769 (0.538, 1.000) | 14 | 0.786 (0.571, 1.000) |
| logit | answered wrong | 70 | 0.729 (0.629, 0.829) | 71 | 0.761 (0.662, 0.859) |
| logit | all failures | 83 | 0.735 (0.639, 0.831) | 85 | 0.765 (0.671, 0.847) |
| logit | success | 207 | 0.638 (0.570, 0.700) | 209 | 0.656 (0.593, 0.718) |
| logit | all | 290 | 0.666 (0.610, 0.717) | 294 | 0.687 (0.633, 0.741) |
| logit | strict gold token | 236 | 0.678 (0.619, 0.737) | 237 | 0.688 (0.629, 0.747) |
| logit | function-word gold token | 54 | 0.611 (0.481, 0.741) | 57 | 0.684 (0.561, 0.807) |

` unknown` reaches top-1 at P1 (jlens): with evidence 27 of 330 (0.082); without 184 (0.558).
` unknown` reaches top-1 at P1 (logit): with evidence 33 of 330 (0.100); without 183 (0.555).

Verdict rules: Claims A and B SHOWN if the lower bound of (intact − ablated) is above 0; pre-registered for qwen3b on its deciding (logit) row. Labels are the §23 single-judge verdicts.
