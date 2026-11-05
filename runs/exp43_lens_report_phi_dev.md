# §43 Part B: where does the small reader lose the answer? phi (phi4mini vs phi4)

Rows from `runs/exp43_lens_pod_phi`. Population: the §34 rule on the §24 `fixed8` caches (answerable, all evidence in view, both sizes, same prompt sha256, judge label on both): 126 development rows, not loaded confirmation rows. Forced mode (an appended `Answer:`), P1, k = 10.
Claims read on: **development folds 0-1 (descriptive; claims need --confirm)**. The logit lens decides (amendment 1).

## J-lens gate (§34 (a) and (b), development folds 0-1)

| lens prompts | below minimum | band | n | (a) logit minus J-lens (95% CI) | (a) | (b) mean pass@k J-lens / logit | (b) | gate |
|---|---|---|---|---|---|---|---|---|
| 56 | False | [10, 21] | 126 | +2.380 (+2.175, +2.582) | PASS | 0.066 / 0.025 | PASS | **PASS** |

## Claim B1 (size): at P1, phi4mini minus phi4, paired by question (positive = phi4 better)

| measure | lens | n | mean difference (95% CI) | verdict |
|---|---|---|---|---|
| log10 best rank of the gold | logit | 126 | +0.134 (+0.048, +0.217) | descriptive (dev) |
| share of outcome A (secondary) | logit | 126 | +0.103 (+0.040, +0.175) | descriptive (dev) |

## Claim B2 (within phi4mini): AUC of the gold's P1 log10 best rank, wrong vs right (> 0.5: failures rank worse)

| lens | n wrong | n right | AUC (95% CI) | mean wrong | mean right | verdict |
|---|---|---|---|---|---|---|
| logit | 67 | 59 | 0.729 (0.643, 0.809) | 0.619 | 0.189 | descriptive (dev) |
| J-lens | 67 | 59 | 0.699 (0.618, 0.778) | 0.454 | 0.127 | descriptive (dev) |

## Descriptive: A / B / C at P1 (development folds 0-1 (descriptive; claims need --confirm))

A never in the top-10, B in it at some layer but not the last, C in it at the last layer. Refusal = the §24 judge marked the answer `unknown`.

| reader | lens | class | n | A (95% CI) | B (95% CI) | C (95% CI) |
|---|---|---|---|---|---|---|
| phi4mini | logit | success | 59 | 0.102 (0.034, 0.186) | 0.017 (0.000, 0.051) | 0.881 (0.797, 0.949) |
| phi4mini | logit | refusal | 3 | 0.000 (0.000, 0.000) | 0.000 (0.000, 0.000) | 1.000 (1.000, 1.000) |
| phi4mini | logit | answered wrong | 64 | 0.281 (0.172, 0.391) | 0.016 (0.000, 0.047) | 0.703 (0.594, 0.812) |
| phi4mini | logit | all failures | 67 | 0.269 (0.164, 0.373) | 0.015 (0.000, 0.045) | 0.716 (0.612, 0.821) |
| phi4 | logit | success | 100 | 0.080 (0.030, 0.140) | 0.110 (0.050, 0.170) | 0.810 (0.730, 0.880) |
| phi4 | logit | refusal | 13 | 0.154 (0.000, 0.385) | 0.000 (0.000, 0.000) | 0.846 (0.615, 1.000) |
| phi4 | logit | answered wrong | 13 | 0.077 (0.000, 0.231) | 0.154 (0.000, 0.385) | 0.769 (0.538, 1.000) |
| phi4 | logit | all failures | 26 | 0.115 (0.000, 0.231) | 0.077 (0.000, 0.192) | 0.808 (0.654, 0.962) |
| phi4mini | J-lens | success | 59 | 0.017 (0.000, 0.051) | 0.102 (0.034, 0.186) | 0.881 (0.797, 0.949) |
| phi4mini | J-lens | refusal | 3 | 0.000 (0.000, 0.000) | 0.000 (0.000, 0.000) | 1.000 (1.000, 1.000) |
| phi4mini | J-lens | answered wrong | 64 | 0.172 (0.078, 0.266) | 0.125 (0.047, 0.203) | 0.703 (0.594, 0.812) |
| phi4mini | J-lens | all failures | 67 | 0.164 (0.075, 0.254) | 0.119 (0.045, 0.209) | 0.716 (0.612, 0.821) |

## Descriptive: emergence depth of the gold at P1 (first layer in the top-10, layer / n_layers)

| reader | lens | n emerged | n read | q1 | median | q3 |
|---|---|---|---|---|---|---|
| phi4mini | logit | 102 | 126 | 0.789 | 0.938 | 0.969 |
| phi4 | logit | 115 | 126 | 0.225 | 0.450 | 0.675 |
| phi4mini | J-lens | 114 | 126 | 0.539 | 0.656 | 0.844 |
| qwen3b (§34) | - | - | - | - | 0.889 | - |
| qwen7b (§34) | - | - | - | - | 0.464 | - |

Depth compares across families only as a fraction of depth and only descriptively.

## Caveats: article-first golds and digits

- phi4mini: article-first golds n=15 (final top-10 0.800, A 0.200, B 0.000); others n=111 (final top-10 0.793, A 0.189, B 0.018); 0 of 2 B rows are article-first.
- phi4: article-first golds n=15 (final top-10 0.867, A 0.000, B 0.133); others n=111 (final top-10 0.802, A 0.099, B 0.099); 2 of 13 B rows are article-first.
- phi4mini: 53 digit-first golds, final-layer top-10 0.811; gold first tokens e.g. ['1', '10', '100', '11', '12', '120', '132', '14', '140', '15', '17', '18'].
- phi4: 53 digit-first golds, final-layer top-10 0.906; gold first tokens e.g. ['1', '10', '100', '11', '12', '120', '132', '14', '140', '15', '17', '18'].
- Llama and Phi put up to three digits in one token, Qwen and Gemma one, so gold first tokens are not like for like across families; the A/B/C shares are the cross-family quantity of record. Gold first-token text identical across the two sizes on 121 of 126 rows (phi4mini and phi4 have different vocabularies).

## Cross-family reading (descriptive, this family)

- Small model's failures mostly C (share > 0.5) with an A floor: True (C 0.716, A 0.269).
- Small model's median emergence later than the large model's: True (0.938 vs 0.450).

## Time per stage

| stage | expected (s, A40) | observed (s) | status |
|---|---|---|---|
| install | 60 | 34 | ok |
| download | 270 | 26 | ok |
| smoke_small + smoke_large | 560 | 45 | ok, ok |
| logit_small | 270 | 162 | ok |
| logit_large | 540 | 220 | ok |
| fit | 4380 | 3591 | ok |
| jread | 330 | 258 | ok |
| eval | 160 | 120 | ok |
| total | 6570 | 4477 | |

Weights (bf16): phi4mini 7.7 GB, phi4 29.3 GB; each stage loads one model alone on the GPU.
