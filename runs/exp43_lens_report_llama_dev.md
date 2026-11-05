# §43 Part B: where does the small reader lose the answer? llama (llama3b vs llama8b)

Rows from `runs/exp43_lens_pod_llama`. Population: the §34 rule on the §24 `fixed8` caches (answerable, all evidence in view, both sizes, same prompt sha256, judge label on both): 126 development rows, not loaded confirmation rows. Forced mode (an appended `Answer:`), P1, k = 10.
Claims read on: **development folds 0-1 (descriptive; claims need --confirm)**. The logit lens decides (amendment 1).

## J-lens gate (§34 (a) and (b), development folds 0-1)

| lens prompts | below minimum | band | n | (a) logit minus J-lens (95% CI) | (a) | (b) mean pass@k J-lens / logit | (b) | gate |
|---|---|---|---|---|---|---|---|---|
| 56 | False | [9, 18] | 126 | +1.458 (+1.285, +1.623) | PASS | 0.270 / 0.217 | PASS | **PASS** |

## Claim B1 (size): at P1, llama3b minus llama8b, paired by question (positive = llama8b better)

| measure | lens | n | mean difference (95% CI) | verdict |
|---|---|---|---|---|
| log10 best rank of the gold | logit | 126 | +0.163 (+0.074, +0.254) | descriptive (dev) |
| share of outcome A (secondary) | logit | 126 | +0.079 (+0.024, +0.135) | descriptive (dev) |

## Claim B2 (within llama3b): AUC of the gold's P1 log10 best rank, wrong vs right (> 0.5: failures rank worse)

| lens | n wrong | n right | AUC (95% CI) | mean wrong | mean right | verdict |
|---|---|---|---|---|---|---|
| logit | 55 | 71 | 0.847 (0.774, 0.909) | 0.927 | 0.188 | descriptive (dev) |
| J-lens | 55 | 71 | 0.821 (0.745, 0.890) | 0.695 | 0.150 | descriptive (dev) |

## Descriptive: A / B / C at P1 (development folds 0-1 (descriptive; claims need --confirm))

A never in the top-10, B in it at some layer but not the last, C in it at the last layer. Refusal = the §24 judge marked the answer `unknown`.

| reader | lens | class | n | A (95% CI) | B (95% CI) | C (95% CI) |
|---|---|---|---|---|---|---|
| llama3b | logit | success | 71 | 0.056 (0.014, 0.113) | 0.000 (0.000, 0.000) | 0.944 (0.887, 0.986) |
| llama3b | logit | refusal | 13 | 0.462 (0.231, 0.769) | 0.000 (0.000, 0.000) | 0.538 (0.231, 0.769) |
| llama3b | logit | answered wrong | 42 | 0.357 (0.214, 0.500) | 0.000 (0.000, 0.000) | 0.643 (0.500, 0.786) |
| llama3b | logit | all failures | 55 | 0.382 (0.255, 0.509) | 0.000 (0.000, 0.000) | 0.618 (0.491, 0.745) |
| llama8b | logit | success | 90 | 0.067 (0.022, 0.122) | 0.011 (0.000, 0.033) | 0.922 (0.867, 0.978) |
| llama8b | logit | refusal | 6 | 0.167 (0.000, 0.500) | 0.000 (0.000, 0.000) | 0.833 (0.500, 1.000) |
| llama8b | logit | answered wrong | 30 | 0.267 (0.133, 0.433) | 0.067 (0.000, 0.167) | 0.667 (0.500, 0.833) |
| llama8b | logit | all failures | 36 | 0.250 (0.111, 0.389) | 0.056 (0.000, 0.139) | 0.694 (0.556, 0.833) |
| llama3b | J-lens | success | 71 | 0.028 (0.000, 0.070) | 0.028 (0.000, 0.070) | 0.944 (0.887, 0.986) |
| llama3b | J-lens | refusal | 13 | 0.385 (0.154, 0.615) | 0.077 (0.000, 0.231) | 0.538 (0.231, 0.769) |
| llama3b | J-lens | answered wrong | 42 | 0.190 (0.071, 0.310) | 0.167 (0.071, 0.286) | 0.643 (0.500, 0.786) |
| llama3b | J-lens | all failures | 55 | 0.236 (0.127, 0.345) | 0.145 (0.055, 0.236) | 0.618 (0.491, 0.745) |

## Descriptive: emergence depth of the gold at P1 (first layer in the top-10, layer / n_layers)

| reader | lens | n emerged | n read | q1 | median | q3 |
|---|---|---|---|---|---|---|
| llama3b | logit | 101 | 126 | 0.714 | 0.750 | 0.893 |
| llama8b | logit | 111 | 126 | 0.688 | 0.719 | 0.812 |
| llama3b | J-lens | 111 | 126 | 0.536 | 0.714 | 0.821 |
| qwen3b (§34) | - | - | - | - | 0.889 | - |
| qwen7b (§34) | - | - | - | - | 0.464 | - |

Depth compares across families only as a fraction of depth and only descriptively.

## Caveats: article-first golds and digits

- llama3b: article-first golds n=15 (final top-10 0.667, A 0.333, B 0.000); others n=111 (final top-10 0.820, A 0.180, B 0.000); 0 of 0 B rows are article-first.
- llama8b: article-first golds n=15 (final top-10 0.933, A 0.067, B 0.000); others n=111 (final top-10 0.847, A 0.126, B 0.027); 0 of 3 B rows are article-first.
- llama3b: 53 digit-first golds, final-layer top-10 0.868; gold first tokens e.g. ['1', '10', '100', '11', '12', '120', '132', '14', '140', '15', '17', '18'].
- llama8b: 53 digit-first golds, final-layer top-10 0.887; gold first tokens e.g. ['1', '10', '100', '11', '12', '120', '132', '14', '140', '15', '17', '18'].
- Llama and Phi put up to three digits in one token, Qwen and Gemma one, so gold first tokens are not like for like across families; the A/B/C shares are the cross-family quantity of record. Gold first-token text identical across the two sizes on 126 of 126 rows.

## Cross-family reading (descriptive, this family)

- Small model's failures mostly C (share > 0.5) with an A floor: True (C 0.618, A 0.382).
- Small model's median emergence later than the large model's: True (0.750 vs 0.719).

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
