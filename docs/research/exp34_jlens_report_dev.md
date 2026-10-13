# §34 J-lens report (qwen3b vs qwen7b; folds: 0, 1; population 126 in-view answerable questions; 126 paired)

Prompt identity: all 126 questions with outputs for both readers carry the same prompt_sha256.

## Lens gate (spec §3): J-lens must beat the logit lens

(a) gold at P1, log10(best rank over the middle-third band), logit lens minus J-lens, paired; PASS if the lower bound is above 0. (b) lens-eval pass@k over the band, mean over sets; PASS if J-lens > logit (strict).

| reader | lens prompts | below minimum | band | n | (a) logit minus J-lens (95% CI) | (a) | (b) mean pass@k J-lens / logit | (b) all layers J-lens / logit | (b) | gate | eval warnings |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3b | 56 | False | [12, 24] | 126 | +1.219 (+0.907, +1.531) | PASS | 0.014 / 0.039 | 0.107 / 0.162 | FAIL | **FAIL** | - |
| qwen7b | 56 | False | [9, 18] | 126 | +1.685 (+1.476, +1.896) | PASS | 0.022 / 0.018 | 0.177 / 0.170 | PASS | **PASS** | - |

| reader | eval set | items | skipped | pass@k J-lens | pass@k logit | all layers J-lens | all layers logit |
|---|---|---|---|---|---|---|---|
| qwen3b | association | 102 | 0 | 0.000 | 0.000 | 0.000 | 0.000 |
| qwen3b | multihop | 93 | 0 | 0.011 | 0.000 | 0.285 | 0.274 |
| qwen3b | multilingual | 107 | 0 | 0.021 | 0.002 | 0.252 | 0.248 |
| qwen3b | order-ops | 55 | 0 | 0.000 | 0.000 | 0.000 | 0.000 |
| qwen3b | poetry | 98 | 0 | 0.000 | 0.000 | 0.000 | 0.010 |
| qwen3b | typo | 96 | 0 | 0.052 | 0.229 | 0.104 | 0.438 |
| qwen7b | association | 102 | 0 | 0.010 | 0.000 | 0.010 | 0.020 |
| qwen7b | multihop | 93 | 0 | 0.011 | 0.000 | 0.441 | 0.317 |
| qwen7b | multilingual | 107 | 0 | 0.028 | 0.002 | 0.374 | 0.304 |
| qwen7b | order-ops | 55 | 0 | 0.000 | 0.000 | 0.009 | 0.036 |
| qwen7b | poetry | 98 | 0 | 0.000 | 0.000 | 0.020 | 0.010 |
| qwen7b | typo | 96 | 0 | 0.083 | 0.104 | 0.208 | 0.333 |

**Gate failed for qwen3b**: claims are also shown with the logit lens (spec §3).

## Claim 1: at P1, qwen3b minus qwen7b (positive = qwen7b better), paired by question

| measure | lens | n | mean difference (95% CI) | verdict |
|---|---|---|---|---|
| log10 best rank of gold | J-lens | 126 | +0.480 (+0.367, +0.601) | SHOWN |
| share of outcome A (secondary) | J-lens | 126 | +0.198 (+0.127, +0.270) | SHOWN |
| log10 best rank of gold | logit (gate fallback) | 126 | +0.451 (+0.326, +0.581) | SHOWN |
| share of outcome A (secondary) | logit (gate fallback) | 126 | +0.167 (+0.103, +0.230) | SHOWN |

## Claim 2: within each reader, log10 best rank at P1 on wrong vs right answers (AUC > 0.5: failures rank worse)

| reader | lens | n wrong | n right | AUC (95% CI) | mean wrong | mean right | wrong minus right (95% CI) | verdict |
|---|---|---|---|---|---|---|---|---|
| qwen3b | J-lens | 72 | 54 | 0.598 (0.499, 0.693) | 0.877 | 0.559 | +0.318 (+0.040, +0.599) | NOT SHOWN |
| qwen3b | logit (gate fallback) | 72 | 54 | 0.598 (0.502, 0.692) | 0.905 | 0.542 | +0.363 (+0.067, +0.668) | SHOWN |
| qwen7b | J-lens | 47 | 79 | 0.659 (0.571, 0.746) | 0.453 | 0.145 | +0.308 (+0.119, +0.520) | descriptive (no verdict) |

## Decomposition: share of outcomes A (never present), B (present then lost), C (present to the end) at P1

| reader | lens | class | n | A (95% CI) | B (95% CI) | C (95% CI) |
|---|---|---|---|---|---|---|
| qwen3b | J-lens | refusal | 32 | 0.375 (0.219, 0.531) | 0.062 (0.000, 0.156) | 0.562 (0.375, 0.719) |
| qwen3b | J-lens | answered wrong | 40 | 0.275 (0.150, 0.400) | 0.025 (0.000, 0.075) | 0.700 (0.550, 0.825) |
| qwen3b | J-lens | all failures | 72 | 0.319 (0.208, 0.431) | 0.042 (0.000, 0.097) | 0.639 (0.528, 0.750) |
| qwen3b | J-lens | success | 54 | 0.241 (0.130, 0.352) | 0.019 (0.000, 0.056) | 0.741 (0.611, 0.852) |
| qwen3b | logit (gate fallback) | refusal | 32 | 0.375 (0.219, 0.562) | 0.062 (0.000, 0.156) | 0.562 (0.375, 0.719) |
| qwen3b | logit (gate fallback) | answered wrong | 40 | 0.275 (0.150, 0.400) | 0.025 (0.000, 0.075) | 0.700 (0.550, 0.825) |
| qwen3b | logit (gate fallback) | all failures | 72 | 0.319 (0.222, 0.431) | 0.042 (0.000, 0.097) | 0.639 (0.528, 0.750) |
| qwen3b | logit (gate fallback) | success | 54 | 0.259 (0.148, 0.389) | 0.000 (0.000, 0.000) | 0.741 (0.611, 0.852) |
| qwen7b | J-lens | refusal | 9 | 0.222 (0.000, 0.556) | 0.000 (0.000, 0.000) | 0.778 (0.444, 1.000) |
| qwen7b | J-lens | answered wrong | 38 | 0.158 (0.053, 0.289) | 0.079 (0.000, 0.158) | 0.763 (0.632, 0.895) |
| qwen7b | J-lens | all failures | 47 | 0.170 (0.064, 0.277) | 0.064 (0.000, 0.149) | 0.766 (0.638, 0.872) |
| qwen7b | J-lens | success | 79 | 0.038 (0.000, 0.089) | 0.038 (0.000, 0.089) | 0.924 (0.861, 0.975) |

Refusal = the §23 judge marked the answer `unknown`; answered wrong = any other incorrect answer.

## Descriptive: emergence depth of the gold at P1 (J-lens; questions where it emerges)

| reader | n emerged | n read | mean | q1 | median | q3 |
|---|---|---|---|---|---|---|
| qwen3b | 90 | 126 | 0.799 | 0.833 | 0.889 | 0.917 |
| qwen7b | 115 | 126 | 0.556 | 0.250 | 0.464 | 0.857 |

## Descriptive: gold best rank by position (J-lens; E = best over the evidence lines)

| reader | position | n | mean log10 best rank | median best rank | share best rank <= k |
|---|---|---|---|---|---|
| qwen3b | P1 | 126 | 0.740 | 2.0 | 0.714 |
| qwen3b | P0 | 126 | 2.417 | 229.0 | 0.040 |
| qwen3b | E | 126 | 2.603 | 477.5 | 0.056 |
| qwen3b | Q | 126 | 2.530 | 389.5 | 0.063 |
| qwen7b | P1 | 126 | 0.260 | 1.0 | 0.913 |
| qwen7b | P0 | 126 | 2.762 | 419.5 | 0.087 |
| qwen7b | E | 126 | 2.941 | 737.0 | 0.024 |
| qwen7b | Q | 126 | 2.717 | 223.5 | 0.032 |

## Descriptive: the `unknown` trajectory at P1 on refusals (J-lens)

| reader | token | refusals | reaches top-1 | mean first top-1 depth |
|---|---|---|---|---|
| qwen3b | unknown_sp | 32 | 25 | 0.907 |
| qwen3b | unknown | 32 | 3 | 0.907 |
| qwen7b | unknown_sp | 9 | 5 | 0.857 |
| qwen7b | unknown | 9 | 0 | - |

## Descriptive: outcome shares by question type (J-lens, P1, all in-view questions)

| reader | question type | n | A | B | C |
|---|---|---|---|---|---|
| qwen3b | knowledge-update | 25 | 0.200 | 0.040 | 0.760 |
| qwen3b | multi-session | 21 | 0.143 | 0.095 | 0.762 |
| qwen3b | single-session-assistant | 18 | 0.222 | 0.056 | 0.722 |
| qwen3b | single-session-preference | 4 | 1.000 | 0.000 | 0.000 |
| qwen3b | single-session-user | 25 | 0.360 | 0.000 | 0.640 |
| qwen3b | temporal-reasoning | 33 | 0.333 | 0.000 | 0.667 |
| qwen7b | knowledge-update | 25 | 0.080 | 0.000 | 0.920 |
| qwen7b | multi-session | 21 | 0.000 | 0.000 | 1.000 |
| qwen7b | single-session-assistant | 18 | 0.056 | 0.056 | 0.889 |
| qwen7b | single-session-preference | 4 | 0.000 | 0.750 | 0.250 |
| qwen7b | single-session-user | 25 | 0.040 | 0.000 | 0.960 |
| qwen7b | temporal-reasoning | 33 | 0.212 | 0.061 | 0.727 |

## Descriptive: cross-check against the community lens for qwen7b

| n | Jaccard top-10 at P1, early | middle | late | Claim 1 with the community lens (95% CI) | verdict |
|---|---|---|---|---|---|
| 126 | 0.408 | 0.538 | 0.704 | +0.465 (+0.353, +0.584) | SHOWN |

## Dropped and failed

| reader | fold | no cached prompt | evidence not found |
|---|---|---|---|
| qwen3b | 0 | 0 | 0 |
| qwen3b | 1 | 0 | 0 |
| qwen7b | 0 | 0 | 0 |
| qwen7b | 1 | 0 | 0 |

| reader | records in folds | failed reads (_failed.json) | in-view questions without an output | paired questions without a correct/unknown label |
|---|---|---|---|---|
| qwen3b | 200 | 0 | 0 | 0 |
| qwen7b | 200 | 0 | 0 | 0 |

Top-k label: k10. Verdict rules: gate (a) PASS if the lower bound is above 0, (b) PASS if J-lens > logit on the mean over sets (strict: a tie (e.g. both 0.0) is FAIL); Claim 1 SHOWN if the lower bound is above 0; Claim 2 SHOWN if the AUC lower bound is above 0.5 (pre-registered for qwen3b; the qwen7b row is descriptive, no verdict). Correct/unknown labels are the §23 single-judge verdicts (STATE.md caveat 3).
