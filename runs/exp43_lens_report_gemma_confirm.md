# §43 Part B: where does the small reader lose the answer? gemma (gemma4b vs gemma12b)

Rows from `runs/exp43_lens_pod_gemma`. Population: the §34 rule on the §24 `fixed8` caches (answerable, all evidence in view, both sizes, same prompt sha256, judge label on both): 126 development rows, 204 confirmation rows. Forced mode (an appended `Answer:`), P1, k = 10.
Claims read on: **confirmation folds 2-4**. The logit lens decides (amendment 1).

## J-lens gate (§34 (a) and (b), development folds 0-1)

| lens prompts | below minimum | band | n | (a) logit minus J-lens (95% CI) | (a) | (b) mean pass@k J-lens / logit | (b) | gate |
|---|---|---|---|---|---|---|---|---|
| 56 | False | [11, 22] | 126 | -0.671 (-0.907, -0.450) | FAIL | 0.126 / 0.171 | FAIL | **FAIL** |

## Claim B1 (size): at P1, gemma4b minus gemma12b, paired by question (positive = gemma12b better)

| measure | lens | n | mean difference (95% CI) | verdict |
|---|---|---|---|---|
| log10 best rank of the gold | logit | 204 | +0.019 (-0.020, +0.059) | NOT SHOWN |
| share of outcome A (secondary) | logit | 204 | -0.005 (-0.039, +0.034) | NOT SHOWN |

## Claim B2 (within gemma4b): AUC of the gold's P1 log10 best rank, wrong vs right (> 0.5: failures rank worse)

| lens | n wrong | n right | AUC (95% CI) | mean wrong | mean right | verdict |
|---|---|---|---|---|---|---|
| logit | 80 | 124 | 0.698 (0.631, 0.764) | 0.435 | 0.136 | SHOWN |
| J-lens | 80 | 124 | 0.714 (0.645, 0.779) | 0.605 | 0.197 | SHOWN, shown beside (gate not passed) |

## Descriptive: A / B / C at P1 (confirmation folds 2-4)

A never in the top-10, B in it at some layer but not the last, C in it at the last layer. Refusal = the §24 judge marked the answer `unknown`.

| reader | lens | class | n | A (95% CI) | B (95% CI) | C (95% CI) |
|---|---|---|---|---|---|---|
| gemma4b | logit | success | 124 | 0.048 (0.016, 0.089) | 0.056 (0.024, 0.105) | 0.895 (0.839, 0.944) |
| gemma4b | logit | refusal | 4 | 0.000 (0.000, 0.000) | 0.500 (0.000, 1.000) | 0.500 (0.000, 1.000) |
| gemma4b | logit | answered wrong | 76 | 0.132 (0.066, 0.211) | 0.092 (0.039, 0.158) | 0.776 (0.684, 0.868) |
| gemma4b | logit | all failures | 80 | 0.125 (0.062, 0.200) | 0.113 (0.050, 0.188) | 0.762 (0.662, 0.850) |
| gemma12b | logit | success | 149 | 0.034 (0.007, 0.067) | 0.060 (0.027, 0.101) | 0.906 (0.859, 0.953) |
| gemma12b | logit | refusal | 4 | 0.250 (0.000, 0.750) | 0.250 (0.000, 0.750) | 0.500 (0.000, 1.000) |
| gemma12b | logit | answered wrong | 51 | 0.216 (0.118, 0.333) | 0.039 (0.000, 0.098) | 0.745 (0.627, 0.863) |
| gemma12b | logit | all failures | 55 | 0.218 (0.109, 0.327) | 0.055 (0.000, 0.127) | 0.727 (0.600, 0.836) |
| gemma4b | J-lens | success | 124 | 0.081 (0.040, 0.129) | 0.024 (0.000, 0.056) | 0.895 (0.839, 0.944) |
| gemma4b | J-lens | refusal | 4 | 0.500 (0.000, 1.000) | 0.000 (0.000, 0.000) | 0.500 (0.000, 1.000) |
| gemma4b | J-lens | answered wrong | 76 | 0.197 (0.118, 0.289) | 0.026 (0.000, 0.066) | 0.776 (0.684, 0.868) |
| gemma4b | J-lens | all failures | 80 | 0.212 (0.125, 0.300) | 0.025 (0.000, 0.062) | 0.762 (0.662, 0.850) |

## Descriptive: emergence depth of the gold at P1 (first layer in the top-10, layer / n_layers)

| reader | lens | n emerged | n read | q1 | median | q3 |
|---|---|---|---|---|---|---|
| gemma4b | logit | 188 | 204 | 0.559 | 0.618 | 0.676 |
| gemma12b | logit | 187 | 204 | 0.542 | 0.667 | 0.792 |
| gemma4b | J-lens | 177 | 204 | 0.382 | 0.676 | 0.765 |
| qwen3b (§34) | - | - | - | - | 0.889 | - |
| qwen7b (§34) | - | - | - | - | 0.464 | - |

Depth compares across families only as a fraction of depth and only descriptively.

## Caveats: article-first golds and digits

- gemma4b: article-first golds n=31 (final top-10 0.484, A 0.065, B 0.452); others n=173 (final top-10 0.908, A 0.081, B 0.012); 14 of 16 B rows are article-first.
- gemma12b: article-first golds n=31 (final top-10 0.645, A 0.226, B 0.129); others n=173 (final top-10 0.896, A 0.058, B 0.046); 4 of 12 B rows are article-first.
- gemma4b: 84 digit-first golds, final-layer top-10 1.000; gold first tokens e.g. ['1', '2', '3', '4', '5', '6', '7', '8', '9'].
- gemma12b: 84 digit-first golds, final-layer top-10 1.000; gold first tokens e.g. ['1', '2', '3', '4', '5', '6', '7', '8', '9'].
- Llama and Phi put up to three digits in one token, Qwen and Gemma one, so gold first tokens are not like for like across families; the A/B/C shares are the cross-family quantity of record. Gold first-token text identical across the two sizes on 204 of 204 rows.

## Cross-family reading (descriptive, this family)

- Small model's failures mostly C (share > 0.5) with an A floor: True (C 0.762, A 0.125).
- Small model's median emergence later than the large model's: False (0.618 vs 0.667).

## Time per stage

| stage | expected (s, A40) | observed (s) | status |
|---|---|---|---|
| install | 60 | 30 | ok |
| download | 240 | 21 | ok |
| smoke_small + smoke_large | 480 | 46 | ok, ok |
| logit_small | 270 | 208 | ok |
| logit_large | 420 | 367 | ok |
| fit | 3060 | 3317 | ok |
| jread | 330 | 350 | ok |
| eval | 160 | 151 | ok |
| total | 5020 | 4510 | |

Weights (bf16): gemma4b 8.6 GB, gemma12b 24.4 GB; each stage loads one model alone on the GPU.
