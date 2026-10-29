# §43 Part B: where does the small reader lose the answer? phi (phi4mini vs phi4)

Rows from `runs/exp43_lens_pod_phi`. Population: the §34 rule on the §24 `fixed8` caches (answerable, all evidence in view, both sizes, same prompt sha256, judge label on both): 126 development rows, 204 confirmation rows. Forced mode (an appended `Answer:`), P1, k = 10.
Claims read on: **confirmation folds 2-4**. The logit lens decides (amendment 1).

## J-lens gate (§34 (a) and (b), development folds 0-1)

| lens prompts | below minimum | band | n | (a) logit minus J-lens (95% CI) | (a) | (b) mean pass@k J-lens / logit | (b) | gate |
|---|---|---|---|---|---|---|---|---|
| 56 | False | [10, 21] | 126 | +2.380 (+2.175, +2.582) | PASS | 0.066 / 0.025 | PASS | **PASS** |

## Claim B1 (size): at P1, phi4mini minus phi4, paired by question (positive = phi4 better)

| measure | lens | n | mean difference (95% CI) | verdict |
|---|---|---|---|---|
| log10 best rank of the gold | logit | 204 | +0.157 (+0.085, +0.230) | SHOWN |
| share of outcome A (secondary) | logit | 204 | +0.059 (+0.005, +0.113) | SHOWN |

## Claim B2 (within phi4mini): AUC of the gold's P1 log10 best rank, wrong vs right (> 0.5: failures rank worse)

| lens | n wrong | n right | AUC (95% CI) | mean wrong | mean right | verdict |
|---|---|---|---|---|---|---|
| logit | 104 | 100 | 0.636 (0.562, 0.707) | 0.545 | 0.315 | SHOWN |
| J-lens | 104 | 100 | 0.598 (0.528, 0.669) | 0.368 | 0.204 | SHOWN, verdict role (gate PASS) |

## Descriptive: A / B / C at P1 (confirmation folds 2-4)

A never in the top-10, B in it at some layer but not the last, C in it at the last layer. Refusal = the §24 judge marked the answer `unknown`.

| reader | lens | class | n | A (95% CI) | B (95% CI) | C (95% CI) |
|---|---|---|---|---|---|---|
| phi4mini | logit | success | 100 | 0.100 (0.050, 0.160) | 0.000 (0.000, 0.000) | 0.900 (0.840, 0.950) |
| phi4mini | logit | refusal | 10 | 0.000 (0.000, 0.000) | 0.000 (0.000, 0.000) | 1.000 (1.000, 1.000) |
| phi4mini | logit | answered wrong | 94 | 0.223 (0.138, 0.309) | 0.000 (0.000, 0.000) | 0.777 (0.691, 0.862) |
| phi4mini | logit | all failures | 104 | 0.202 (0.125, 0.279) | 0.000 (0.000, 0.000) | 0.798 (0.721, 0.875) |
| phi4 | logit | success | 165 | 0.091 (0.048, 0.139) | 0.115 (0.073, 0.170) | 0.794 (0.733, 0.855) |
| phi4 | logit | refusal | 16 | 0.125 (0.000, 0.312) | 0.062 (0.000, 0.188) | 0.812 (0.625, 1.000) |
| phi4 | logit | answered wrong | 23 | 0.087 (0.000, 0.217) | 0.174 (0.043, 0.348) | 0.739 (0.565, 0.913) |
| phi4 | logit | all failures | 39 | 0.103 (0.026, 0.205) | 0.128 (0.026, 0.256) | 0.769 (0.641, 0.897) |
| phi4mini | J-lens | success | 100 | 0.030 (0.000, 0.070) | 0.070 (0.020, 0.120) | 0.900 (0.840, 0.950) |
| phi4mini | J-lens | refusal | 10 | 0.000 (0.000, 0.000) | 0.000 (0.000, 0.000) | 1.000 (1.000, 1.000) |
| phi4mini | J-lens | answered wrong | 94 | 0.117 (0.053, 0.191) | 0.106 (0.043, 0.170) | 0.777 (0.691, 0.862) |
| phi4mini | J-lens | all failures | 104 | 0.106 (0.048, 0.163) | 0.096 (0.048, 0.154) | 0.798 (0.721, 0.875) |

## Descriptive: emergence depth of the gold at P1 (first layer in the top-10, layer / n_layers)

| reader | lens | n emerged | n read | q1 | median | q3 |
|---|---|---|---|---|---|---|
| phi4mini | logit | 173 | 204 | 0.875 | 0.938 | 0.969 |
| phi4 | logit | 185 | 204 | 0.250 | 0.525 | 0.700 |
| phi4mini | J-lens | 190 | 204 | 0.531 | 0.672 | 0.844 |
| qwen3b (§34) | - | - | - | - | 0.889 | - |
| qwen7b (§34) | - | - | - | - | 0.464 | - |

Depth compares across families only as a fraction of depth and only descriptively.

## Caveats: article-first golds and digits

- phi4mini: article-first golds n=31 (final top-10 0.806, A 0.194, B 0.000); others n=173 (final top-10 0.855, A 0.145, B 0.000); 0 of 0 B rows are article-first.
- phi4: article-first golds n=31 (final top-10 0.677, A 0.097, B 0.226); others n=173 (final top-10 0.809, A 0.092, B 0.098); 7 of 24 B rows are article-first.
- phi4mini: 84 digit-first golds, final-layer top-10 0.917; gold first tokens e.g. ['1', '10', '100', '12', '130', '14', '15', '16', '17', '18', '190', '2'].
- phi4: 84 digit-first golds, final-layer top-10 0.976; gold first tokens e.g. ['1', '10', '100', '12', '130', '14', '15', '16', '17', '18', '190', '2'].
- Llama and Phi put up to three digits in one token, Qwen and Gemma one, so gold first tokens are not like for like across families; the A/B/C shares are the cross-family quantity of record. Gold first-token text identical across the two sizes on 198 of 204 rows (phi4mini and phi4 have different vocabularies).

## Cross-family reading (descriptive, this family)

- Small model's failures mostly C (share > 0.5) with an A floor: True (C 0.798, A 0.202).
- Small model's median emergence later than the large model's: True (0.938 vs 0.525).

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
