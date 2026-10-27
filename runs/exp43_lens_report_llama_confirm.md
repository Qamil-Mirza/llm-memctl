# §43 Part B: where does the small reader lose the answer? llama (llama3b vs llama8b)

Rows from `runs/exp43_lens_pod_llama`. Population: the §34 rule on the §24 `fixed8` caches (answerable, all evidence in view, both sizes, same prompt sha256, judge label on both): 126 development rows, 204 confirmation rows. Forced mode (an appended `Answer:`), P1, k = 10.
Claims read on: **confirmation folds 2-4**. The logit lens decides (amendment 1).

## J-lens gate (§34 (a) and (b), development folds 0-1)

| lens prompts | below minimum | band | n | (a) logit minus J-lens (95% CI) | (a) | (b) mean pass@k J-lens / logit | (b) | gate |
|---|---|---|---|---|---|---|---|---|
| 56 | False | [9, 18] | 126 | +1.458 (+1.285, +1.623) | PASS | 0.270 / 0.217 | PASS | **PASS** |

## Claim B1 (size): at P1, llama3b minus llama8b, paired by question (positive = llama8b better)

| measure | lens | n | mean difference (95% CI) | verdict |
|---|---|---|---|---|
| log10 best rank of the gold | logit | 204 | +0.181 (+0.106, +0.259) | SHOWN |
| share of outcome A (secondary) | logit | 204 | +0.078 (+0.029, +0.132) | SHOWN |

## Claim B2 (within llama3b): AUC of the gold's P1 log10 best rank, wrong vs right (> 0.5: failures rank worse)

| lens | n wrong | n right | AUC (95% CI) | mean wrong | mean right | verdict |
|---|---|---|---|---|---|---|
| logit | 95 | 109 | 0.677 (0.605, 0.746) | 0.658 | 0.322 | SHOWN |
| J-lens | 95 | 109 | 0.682 (0.610, 0.751) | 0.518 | 0.246 | SHOWN, verdict role (gate PASS) |

## Descriptive: A / B / C at P1 (confirmation folds 2-4)

A never in the top-10, B in it at some layer but not the last, C in it at the last layer. Refusal = the §24 judge marked the answer `unknown`.

| reader | lens | class | n | A (95% CI) | B (95% CI) | C (95% CI) |
|---|---|---|---|---|---|---|
| llama3b | logit | success | 109 | 0.110 (0.055, 0.174) | 0.018 (0.000, 0.046) | 0.872 (0.807, 0.927) |
| llama3b | logit | refusal | 32 | 0.219 (0.094, 0.375) | 0.031 (0.000, 0.094) | 0.750 (0.594, 0.875) |
| llama3b | logit | answered wrong | 63 | 0.302 (0.190, 0.413) | 0.048 (0.000, 0.111) | 0.651 (0.524, 0.762) |
| llama3b | logit | all failures | 95 | 0.274 (0.189, 0.368) | 0.042 (0.011, 0.084) | 0.684 (0.589, 0.779) |
| llama8b | logit | success | 134 | 0.075 (0.037, 0.119) | 0.015 (0.000, 0.037) | 0.910 (0.858, 0.955) |
| llama8b | logit | refusal | 13 | 0.000 (0.000, 0.000) | 0.077 (0.000, 0.231) | 0.923 (0.769, 1.000) |
| llama8b | logit | answered wrong | 57 | 0.211 (0.105, 0.316) | 0.000 (0.000, 0.000) | 0.789 (0.684, 0.895) |
| llama8b | logit | all failures | 70 | 0.171 (0.086, 0.257) | 0.014 (0.000, 0.043) | 0.814 (0.714, 0.900) |
| llama3b | J-lens | success | 109 | 0.046 (0.009, 0.092) | 0.083 (0.037, 0.138) | 0.872 (0.807, 0.927) |
| llama3b | J-lens | refusal | 32 | 0.094 (0.000, 0.219) | 0.156 (0.031, 0.281) | 0.750 (0.594, 0.875) |
| llama3b | J-lens | answered wrong | 63 | 0.206 (0.111, 0.302) | 0.143 (0.063, 0.238) | 0.651 (0.524, 0.762) |
| llama3b | J-lens | all failures | 95 | 0.168 (0.095, 0.253) | 0.147 (0.084, 0.221) | 0.684 (0.589, 0.779) |

## Descriptive: emergence depth of the gold at P1 (first layer in the top-10, layer / n_layers)

| reader | lens | n emerged | n read | q1 | median | q3 |
|---|---|---|---|---|---|---|
| llama3b | logit | 166 | 204 | 0.679 | 0.750 | 0.884 |
| llama8b | logit | 182 | 204 | 0.688 | 0.750 | 0.844 |
| llama3b | J-lens | 183 | 204 | 0.464 | 0.714 | 0.786 |
| qwen3b (§34) | - | - | - | - | 0.889 | - |
| qwen7b (§34) | - | - | - | - | 0.464 | - |

Depth compares across families only as a fraction of depth and only descriptively.

## Caveats: article-first golds and digits

- llama3b: article-first golds n=31 (final top-10 0.516, A 0.484, B 0.000); others n=173 (final top-10 0.832, A 0.133, B 0.035); 0 of 6 B rows are article-first.
- llama8b: article-first golds n=31 (final top-10 0.935, A 0.065, B 0.000); others n=173 (final top-10 0.867, A 0.116, B 0.017); 0 of 3 B rows are article-first.
- llama3b: 84 digit-first golds, final-layer top-10 0.845; gold first tokens e.g. ['1', '10', '100', '12', '130', '14', '15', '16', '17', '18', '190', '2'].
- llama8b: 84 digit-first golds, final-layer top-10 0.905; gold first tokens e.g. ['1', '10', '100', '12', '130', '14', '15', '16', '17', '18', '190', '2'].
- Llama and Phi put up to three digits in one token, Qwen and Gemma one, so gold first tokens are not like for like across families; the A/B/C shares are the cross-family quantity of record. Gold first-token text identical across the two sizes on 204 of 204 rows.

## Cross-family reading (descriptive, this family)

- Small model's failures mostly C (share > 0.5) with an A floor: True (C 0.684, A 0.274).
- Small model's median emergence later than the large model's: False (0.750 vs 0.750).

## Time per stage

| stage | expected (s, A40) | observed (s) | status |
|---|---|---|---|
| install | 60 | 31 | ok |
| download | 170 | 16 | ok |
| smoke_small + smoke_large | 400 | 39 | ok, ok |
| logit_small | 240 | 114 | ok |
| logit_large | 300 | 169 | ok |
| fit | 3780 | 2929 | ok |
| jread | 300 | 172 | ok |
| eval | 150 | 77 | ok |
| total | 5400 | 3568 | |

Weights (bf16): llama3b 6.4 GB, llama8b 16.1 GB; each stage loads one model alone on the GPU.
