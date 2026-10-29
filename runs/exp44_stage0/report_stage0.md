# §44 activation patching report: development folds 0-1 (descriptive; claims need --confirm)

Spec research/proposals/2026-10-29-exp44-patching.md rev 0. Clean -> evidence-cut at the forced P1; R(l) = (m_patched(l) - m_corrupt) / (m_clean - m_corrupt); d_c = first l with R >= 0.5, / L. Percentile bootstrap, 1000 draws. The C2 layer row of the 3B (layer round(0.464 L)) is descriptive only.

## Rows and filter

| model | rows | patched | dev strict (rank_c<=10, corrupt>10) | dev relaxed (>3) | filter used (corrupt >) | rows in filter | d_c undefined |
|---|---|---|---|---|---|---|---|
| qwen3b | 5 | 5 | 0 / 5 | 1 | 3 (relaxed) | 1 | 0 |
| qwen7b | 2 | 2 | 0 / 2 | 0 | 3 (relaxed) | 0 | 0 |

## Sanity (|R(L-1) - 1| < 0.01, |R(0)| < 0.05)

| model | patched | asserted | asserted pass | all pass | max abs R(0) | max abs R(L-1)-1 |
|---|---|---|---|---|---|---|
| qwen3b | 5 | 1 | 1 | 4 | 0.1801 | 0.00000 |
| qwen7b | 2 | 0 | 0 | 2 | 0.0076 | 0.00000 |

## Claims (dev: descriptive only)

| claim | statistic | value [95% CI] | reading |
|---|---|---|---|
| C1 (7B) | median d_c | — [—, —] (n=0) | None |
| C3 (3B) | median d_c | 0.583 [0.583, 0.583] (n=1) | lenses under-read the 3B (late picture a readout artefact) (CI overlaps 0.85-0.92: False) |

## qwen3b: descriptives (filter rows)

| d_c minus | n | median | q1 | q3 | lens never |
|---|---|---|---|---|---|
| jlens | 1 | 0.250 | 0.250 | 0.250 | 0 |
| logit | 1 | -0.361 | -0.361 | -0.361 | 0 |

Rows where patching alone flips top-1 to the gold before the final layer: 0 of 1. Mean log p(' unknown'): clean -28.249, corrupt -10.316.

| class | n | median d_c | mean R at C2 layer |
|---|---|---|---|
| success | 1 | 0.583 | 0.022 |
| refusal | 0 | — | — |
| wrong | 0 | — | — |

| layer | depth | mean R | mean R clipped | top-1 = gold | mean log p(' unknown') |
|---|---|---|---|---|---|
| 0 | 0.000 | 0.001 | 0.001 | 0.00 | -10.26 |
| 1 | 0.028 | -0.000 | -0.000 | 0.00 | -10.25 |
| 2 | 0.056 | 0.002 | 0.002 | 0.00 | -10.34 |
| 3 | 0.083 | 0.001 | 0.001 | 0.00 | -10.28 |
| 4 | 0.111 | 0.002 | 0.002 | 0.00 | -10.22 |
| 5 | 0.139 | -0.000 | -0.000 | 0.00 | -10.19 |
| 6 | 0.167 | -0.006 | -0.006 | 0.00 | -10.23 |
| 7 | 0.194 | -0.014 | -0.014 | 0.00 | -10.29 |
| 8 | 0.222 | -0.011 | -0.011 | 0.00 | -10.29 |
| 9 | 0.250 | -0.020 | -0.020 | 0.00 | -10.24 |
| 10 | 0.278 | -0.001 | -0.001 | 0.00 | -10.23 |
| 11 | 0.306 | -0.019 | -0.019 | 0.00 | -10.35 |
| 12 | 0.333 | -0.034 | -0.034 | 0.00 | -10.48 |
| 13 | 0.361 | -0.031 | -0.031 | 0.00 | -10.62 |
| 14 | 0.389 | -0.049 | -0.049 | 0.00 | -10.57 |
| 15 | 0.417 | -0.101 | -0.101 | 0.00 | -10.58 |
| 16 | 0.444 | -0.042 | -0.042 | 0.00 | -10.72 |
| 17 | 0.472 | 0.022 | 0.022 | 0.00 | -10.35 |
| 18 | 0.500 | 0.044 | 0.044 | 0.00 | -10.66 |
| 19 | 0.528 | 0.111 | 0.111 | 0.00 | -9.41 |
| 20 | 0.556 | 0.242 | 0.242 | 0.00 | -9.44 |
| 21 | 0.583 | 0.790 | 0.790 | 0.00 | -9.70 |
| 22 | 0.611 | 1.209 | 1.209 | 0.00 | -9.00 |
| 23 | 0.639 | 1.069 | 1.069 | 0.00 | -8.95 |
| 24 | 0.667 | 0.247 | 0.247 | 0.00 | -9.58 |
| 25 | 0.694 | 1.760 | 1.250 | 0.00 | -10.49 |
| 26 | 0.722 | 1.185 | 1.185 | 0.00 | -10.26 |
| 27 | 0.750 | 0.833 | 0.833 | 0.00 | -12.15 |
| 28 | 0.778 | 0.958 | 0.958 | 0.00 | -24.28 |
| 29 | 0.806 | 0.982 | 0.982 | 0.00 | -24.83 |
| 30 | 0.833 | 0.971 | 0.971 | 0.00 | -24.90 |
| 31 | 0.861 | 0.950 | 0.950 | 0.00 | -26.69 |
| 32 | 0.889 | 0.970 | 0.970 | 0.00 | -26.66 |
| 33 | 0.917 | 0.963 | 0.963 | 0.00 | -26.76 |
| 34 | 0.944 | 0.985 | 0.985 | 0.00 | -27.26 |
| 35 | 0.972 | 1.000 | 1.000 | 0.00 | -28.25 |

| qid | fold | rank clean | rank corrupt | d_c | d_logit | d_jlens | R(C2 layer) | sanity |
|---|---|---|---|---|---|---|---|---|
| 00ca467f | 0 | 3 | 4 | 0.583 | 0.944 | 0.333 | 0.022 | True |
