# §34 J-lens report (qwen3b vs qwen7b; folds: 2, 3, 4; population 204 in-view answerable questions; 204 paired)

Prompt identity: all 204 questions with outputs for both readers carry the same prompt_sha256.

## Lens gate (spec §3): J-lens must beat the logit lens; gate (development folds 0,1), fixed before the confirmation folds

(a) gold at P1, log10(best rank over the middle-third band), logit lens minus J-lens, paired; PASS if the lower bound is above 0. (b) lens-eval pass@k over the band, mean over sets; PASS if J-lens > logit (strict).

| reader | lens prompts | below minimum | band | n | (a) logit minus J-lens (95% CI) | (a) | (b) mean pass@k J-lens / logit | (b) all layers J-lens / logit | (b) | gate (development folds 0,1) | eval warnings |
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

Logit-lens rows are also shown beside the J-lens rows for qwen7b (gate passed; `logit (shown beside)` rows carry no verdict role).

## Claim 1: at P1, qwen3b minus qwen7b (positive = qwen7b better), paired by question

| measure | lens | n | mean difference (95% CI) | verdict |
|---|---|---|---|---|
| log10 best rank of gold | J-lens | 204 | +0.535 (+0.435, +0.641) | SHOWN |
| share of outcome A (secondary) | J-lens | 204 | +0.255 (+0.191, +0.319) | SHOWN |
| log10 best rank of gold | logit (gate fallback) | 204 | +0.486 (+0.376, +0.600) | SHOWN |
| share of outcome A (secondary) | logit (gate fallback) | 204 | +0.201 (+0.137, +0.265) | SHOWN |

## Claim 2: within each reader, log10 best rank at P1 on wrong vs right answers (AUC > 0.5: failures rank worse)

| reader | lens | n wrong | n right | AUC (95% CI) | mean wrong | mean right | wrong minus right (95% CI) | verdict |
|---|---|---|---|---|---|---|---|---|
| qwen3b | J-lens | 127 | 77 | 0.629 (0.547, 0.707) | 0.897 | 0.603 | +0.294 (+0.067, +0.525) | SHOWN |
| qwen3b | logit (gate fallback) | 127 | 77 | 0.654 (0.576, 0.727) | 0.915 | 0.495 | +0.420 (+0.190, +0.645) | SHOWN |
| qwen7b | J-lens | 63 | 141 | 0.646 (0.571, 0.721) | 0.406 | 0.182 | +0.224 (+0.088, +0.367) | descriptive (no verdict) |
| qwen7b | logit (shown beside) | 63 | 141 | 0.692 (0.617, 0.766) | 0.512 | 0.162 | +0.350 (+0.204, +0.503) | descriptive (no verdict) |

## Decomposition: share of outcomes A (never present), B (present then lost), C (present to the end) at P1

| reader | lens | class | n | A (95% CI) | B (95% CI) | C (95% CI) |
|---|---|---|---|---|---|---|
| qwen3b | J-lens | refusal | 65 | 0.431 (0.308, 0.554) | 0.000 (0.000, 0.000) | 0.569 (0.446, 0.692) |
| qwen3b | J-lens | answered wrong | 62 | 0.290 (0.177, 0.403) | 0.000 (0.000, 0.000) | 0.710 (0.597, 0.823) |
| qwen3b | J-lens | all failures | 127 | 0.362 (0.283, 0.449) | 0.000 (0.000, 0.000) | 0.638 (0.551, 0.717) |
| qwen3b | J-lens | success | 77 | 0.260 (0.169, 0.364) | 0.013 (0.000, 0.039) | 0.727 (0.623, 0.818) |
| qwen3b | logit (gate fallback) | refusal | 65 | 0.400 (0.277, 0.523) | 0.031 (0.000, 0.077) | 0.569 (0.446, 0.692) |
| qwen3b | logit (gate fallback) | answered wrong | 62 | 0.290 (0.177, 0.403) | 0.000 (0.000, 0.000) | 0.710 (0.597, 0.823) |
| qwen3b | logit (gate fallback) | all failures | 127 | 0.346 (0.268, 0.433) | 0.016 (0.000, 0.039) | 0.638 (0.551, 0.717) |
| qwen3b | logit (gate fallback) | success | 77 | 0.221 (0.130, 0.312) | 0.052 (0.013, 0.104) | 0.727 (0.623, 0.818) |
| qwen7b | J-lens | refusal | 9 | 0.222 (0.000, 0.556) | 0.111 (0.000, 0.333) | 0.667 (0.333, 0.889) |
| qwen7b | J-lens | answered wrong | 54 | 0.111 (0.037, 0.204) | 0.130 (0.056, 0.222) | 0.759 (0.648, 0.870) |
| qwen7b | J-lens | all failures | 63 | 0.127 (0.048, 0.206) | 0.127 (0.048, 0.222) | 0.746 (0.635, 0.841) |
| qwen7b | J-lens | success | 141 | 0.043 (0.014, 0.078) | 0.007 (0.000, 0.021) | 0.950 (0.915, 0.986) |
| qwen7b | logit (shown beside) | refusal | 9 | 0.222 (0.000, 0.556) | 0.111 (0.000, 0.333) | 0.667 (0.333, 0.889) |
| qwen7b | logit (shown beside) | answered wrong | 54 | 0.222 (0.111, 0.333) | 0.019 (0.000, 0.056) | 0.759 (0.648, 0.870) |
| qwen7b | logit (shown beside) | all failures | 63 | 0.222 (0.127, 0.333) | 0.032 (0.000, 0.079) | 0.746 (0.635, 0.841) |
| qwen7b | logit (shown beside) | success | 141 | 0.043 (0.014, 0.078) | 0.007 (0.000, 0.021) | 0.950 (0.915, 0.986) |

Refusal = the §23 judge marked the answer `unknown`; answered wrong = any other incorrect answer.

## Descriptive: emergence depth of the gold at P1 (J-lens; questions where it emerges)

| reader | n emerged | n read | mean | q1 | median | q3 |
|---|---|---|---|---|---|---|
| qwen3b | 138 | 204 | 0.761 | 0.590 | 0.889 | 0.944 |
| qwen7b | 190 | 204 | 0.559 | 0.250 | 0.464 | 0.857 |

## Descriptive: gold best rank by position (J-lens; E = best over the evidence lines)

| reader | position | n | mean log10 best rank | median best rank | share best rank <= k |
|---|---|---|---|---|---|
| qwen3b | P1 | 204 | 0.786 | 3.0 | 0.676 |
| qwen3b | P0 | 204 | 2.329 | 209.0 | 0.064 |
| qwen3b | E | 204 | 2.521 | 398.5 | 0.059 |
| qwen3b | Q | 204 | 2.516 | 394.0 | 0.049 |
| qwen7b | P1 | 204 | 0.251 | 1.0 | 0.931 |
| qwen7b | P0 | 204 | 2.644 | 315.5 | 0.103 |
| qwen7b | E | 204 | 3.052 | 640.0 | 0.015 |
| qwen7b | Q | 204 | 2.798 | 330.0 | 0.049 |

## Descriptive: the `unknown` trajectory at P1 on refusals (J-lens)

| reader | token | refusals | reaches top-1 | mean first top-1 depth |
|---|---|---|---|---|
| qwen3b | unknown_sp | 65 | 47 | 0.901 |
| qwen3b | unknown | 65 | 9 | 0.914 |
| qwen7b | unknown_sp | 9 | 5 | 0.886 |
| qwen7b | unknown | 9 | 0 | - |

## Descriptive: outcome shares by question type (J-lens, P1, all in-view questions)

| reader | question type | n | A | B | C |
|---|---|---|---|---|---|
| qwen3b | knowledge-update | 35 | 0.400 | 0.029 | 0.571 |
| qwen3b | multi-session | 36 | 0.194 | 0.000 | 0.806 |
| qwen3b | single-session-assistant | 30 | 0.267 | 0.000 | 0.733 |
| qwen3b | single-session-preference | 9 | 0.889 | 0.000 | 0.111 |
| qwen3b | single-session-user | 36 | 0.333 | 0.000 | 0.667 |
| qwen3b | temporal-reasoning | 58 | 0.293 | 0.000 | 0.707 |
| qwen7b | knowledge-update | 35 | 0.114 | 0.000 | 0.886 |
| qwen7b | multi-session | 36 | 0.056 | 0.000 | 0.944 |
| qwen7b | single-session-assistant | 30 | 0.067 | 0.000 | 0.933 |
| qwen7b | single-session-preference | 9 | 0.000 | 0.778 | 0.222 |
| qwen7b | single-session-user | 36 | 0.083 | 0.028 | 0.889 |
| qwen7b | temporal-reasoning | 58 | 0.052 | 0.017 | 0.931 |

## Descriptive: cross-check against the community lens for qwen7b

| n | Jaccard top-10 at P1, early | middle | late | Claim 1 with the community lens (95% CI) | verdict |
|---|---|---|---|---|---|
| 0 | - | - | - | None | NOT SHOWN |

## Dropped and failed

| reader | fold | no cached prompt | evidence not found |
|---|---|---|---|
| qwen3b | 2 | 0 | 0 |
| qwen3b | 3 | 0 | 0 |
| qwen3b | 4 | 0 | 0 |
| qwen7b | 2 | 0 | 0 |
| qwen7b | 3 | 0 | 0 |
| qwen7b | 4 | 0 | 0 |

| reader | records in folds | failed reads (_failed.json) | in-view questions without an output | paired questions without a correct/unknown label |
|---|---|---|---|---|
| qwen3b | 300 | 1 | 0 | 0 |
| qwen7b | 300 | 1 | 0 | 0 |

Top-k label: k10. Verdict rules: gate (a) PASS if the lower bound is above 0, (b) PASS if J-lens > logit on the mean over sets (strict: a tie (e.g. both 0.0) is FAIL); Claim 1 SHOWN if the lower bound is above 0; Claim 2 SHOWN if the AUC lower bound is above 0.5 (pre-registered for qwen3b; the qwen7b row is descriptive, no verdict). Correct/unknown labels are the §23 single-judge verdicts (STATE.md caveat 3).
