# §43 Part B: where does the small reader lose the answer? gemma (gemma4b vs gemma12b)

Rows from `runs/exp43_lens_pod_gemma`. Population: the §34 rule on the §24 `fixed8` caches (answerable, all evidence in view, both sizes, same prompt sha256, judge label on both): 126 development rows, not loaded confirmation rows. Forced mode (an appended `Answer:`), P1, k = 10.
Claims read on: **development folds 0-1 (descriptive; claims need --confirm)**. The logit lens decides (amendment 1).

## J-lens gate (§34 (a) and (b), development folds 0-1)

| lens prompts | below minimum | band | n | (a) logit minus J-lens (95% CI) | (a) | (b) mean pass@k J-lens / logit | (b) | gate |
|---|---|---|---|---|---|---|---|---|
| 56 | False | [11, 22] | 126 | -0.671 (-0.907, -0.450) | FAIL | 0.126 / 0.171 | FAIL | **FAIL** |

## Claim B1 (size): at P1, gemma4b minus gemma12b, paired by question (positive = gemma12b better)

| measure | lens | n | mean difference (95% CI) | verdict |
|---|---|---|---|---|
| log10 best rank of the gold | logit | 126 | +0.003 (-0.056, +0.065) | descriptive (dev) |
| share of outcome A (secondary) | logit | 126 | -0.016 (-0.063, +0.032) | descriptive (dev) |

## Claim B2 (within gemma4b): AUC of the gold's P1 log10 best rank, wrong vs right (> 0.5: failures rank worse)

| lens | n wrong | n right | AUC (95% CI) | mean wrong | mean right | verdict |
|---|---|---|---|---|---|---|
| logit | 43 | 83 | 0.714 (0.626, 0.801) | 0.538 | 0.155 | descriptive (dev) |
| J-lens | 43 | 83 | 0.723 (0.632, 0.811) | 0.677 | 0.214 | descriptive (dev) |

## Descriptive: A / B / C at P1 (development folds 0-1 (descriptive; claims need --confirm))

A never in the top-10, B in it at some layer but not the last, C in it at the last layer. Refusal = the §24 judge marked the answer `unknown`.

| reader | lens | class | n | A (95% CI) | B (95% CI) | C (95% CI) |
|---|---|---|---|---|---|---|
| gemma4b | logit | success | 83 | 0.036 (0.000, 0.084) | 0.048 (0.012, 0.096) | 0.916 (0.855, 0.964) |
| gemma4b | logit | refusal | 2 | 0.000 (0.000, 0.000) | 0.500 (0.000, 1.000) | 0.500 (0.000, 1.000) |
| gemma4b | logit | answered wrong | 41 | 0.171 (0.073, 0.293) | 0.171 (0.073, 0.293) | 0.659 (0.512, 0.805) |
| gemma4b | logit | all failures | 43 | 0.163 (0.070, 0.279) | 0.186 (0.070, 0.302) | 0.651 (0.512, 0.791) |
| gemma12b | logit | success | 93 | 0.043 (0.011, 0.086) | 0.022 (0.000, 0.054) | 0.935 (0.882, 0.978) |
| gemma12b | logit | refusal | 1 | 0.000 (0.000, 0.000) | 0.000 (0.000, 0.000) | 1.000 (1.000, 1.000) |
| gemma12b | logit | answered wrong | 32 | 0.250 (0.094, 0.406) | 0.125 (0.031, 0.250) | 0.625 (0.469, 0.781) |
| gemma12b | logit | all failures | 33 | 0.242 (0.121, 0.394) | 0.121 (0.030, 0.242) | 0.636 (0.455, 0.788) |
| gemma4b | J-lens | success | 83 | 0.072 (0.024, 0.133) | 0.012 (0.000, 0.036) | 0.916 (0.855, 0.964) |
| gemma4b | J-lens | refusal | 2 | 0.500 (0.000, 1.000) | 0.000 (0.000, 0.000) | 0.500 (0.000, 1.000) |
| gemma4b | J-lens | answered wrong | 41 | 0.268 (0.146, 0.415) | 0.073 (0.000, 0.171) | 0.659 (0.512, 0.805) |
| gemma4b | J-lens | all failures | 43 | 0.279 (0.140, 0.419) | 0.070 (0.000, 0.163) | 0.651 (0.512, 0.791) |

## Descriptive: emergence depth of the gold at P1 (first layer in the top-10, layer / n_layers)

| reader | lens | n emerged | n read | q1 | median | q3 |
|---|---|---|---|---|---|---|
| gemma4b | logit | 116 | 126 | 0.559 | 0.647 | 0.676 |
| gemma12b | logit | 114 | 126 | 0.583 | 0.667 | 0.812 |
| gemma4b | J-lens | 108 | 126 | 0.559 | 0.676 | 0.765 |
| qwen3b (§34) | - | - | - | - | 0.889 | - |
| qwen7b (§34) | - | - | - | - | 0.464 | - |

Depth compares across families only as a fraction of depth and only descriptively.

## Caveats: article-first golds and digits

- gemma4b: article-first golds n=15 (final top-10 0.267, A 0.133, B 0.600); others n=111 (final top-10 0.901, A 0.072, B 0.027); 9 of 12 B rows are article-first.
- gemma12b: article-first golds n=15 (final top-10 0.533, A 0.400, B 0.067); others n=111 (final top-10 0.901, A 0.054, B 0.045); 1 of 6 B rows are article-first.
- gemma4b: 53 digit-first golds, final-layer top-10 1.000; gold first tokens e.g. ['1', '2', '3', '4', '5', '6', '7', '8'].
- gemma12b: 53 digit-first golds, final-layer top-10 0.981; gold first tokens e.g. ['1', '2', '3', '4', '5', '6', '7', '8'].
- Llama and Phi put up to three digits in one token, Qwen and Gemma one, so gold first tokens are not like for like across families; the A/B/C shares are the cross-family quantity of record. Gold first-token text identical across the two sizes on 126 of 126 rows.

## Cross-family reading (descriptive, this family)

- Small model's failures mostly C (share > 0.5) with an A floor: True (C 0.651, A 0.163).
- Small model's median emergence later than the large model's: False (0.647 vs 0.667).

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
