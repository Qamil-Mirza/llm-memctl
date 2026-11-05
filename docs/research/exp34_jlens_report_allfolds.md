# §34 J-lens report (qwen3b vs qwen7b; folds: 0, 1, 2, 3, 4; population 330 in-view answerable questions; 330 paired)

Prompt identity: all 330 questions with outputs for both readers carry the same prompt_sha256.

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
| log10 best rank of gold | J-lens | 330 | +0.514 (+0.437, +0.595) | SHOWN |
| share of outcome A (secondary) | J-lens | 330 | +0.233 (+0.188, +0.282) | SHOWN |
| log10 best rank of gold | logit (gate fallback) | 330 | +0.473 (+0.388, +0.560) | SHOWN |
| share of outcome A (secondary) | logit (gate fallback) | 330 | +0.188 (+0.142, +0.236) | SHOWN |

## Claim 2: within each reader, log10 best rank at P1 on wrong vs right answers (AUC > 0.5: failures rank worse)

| reader | lens | n wrong | n right | AUC (95% CI) | mean wrong | mean right | wrong minus right (95% CI) | verdict |
|---|---|---|---|---|---|---|---|---|
| qwen3b | J-lens | 199 | 131 | 0.620 (0.558, 0.679) | 0.890 | 0.585 | +0.305 (+0.123, +0.478) | SHOWN |
| qwen3b | logit (gate fallback) | 199 | 131 | 0.632 (0.571, 0.689) | 0.911 | 0.514 | +0.397 (+0.212, +0.575) | SHOWN |
| qwen7b | J-lens | 110 | 220 | 0.651 (0.594, 0.708) | 0.426 | 0.169 | +0.257 (+0.142, +0.375) | descriptive (no verdict) |
| qwen7b | logit (shown beside) | 110 | 220 | 0.710 (0.655, 0.765) | 0.539 | 0.152 | +0.388 (+0.266, +0.513) | descriptive (no verdict) |

## Decomposition: share of outcomes A (never present), B (present then lost), C (present to the end) at P1

| reader | lens | class | n | A (95% CI) | B (95% CI) | C (95% CI) |
|---|---|---|---|---|---|---|
| qwen3b | J-lens | refusal | 97 | 0.412 (0.320, 0.515) | 0.021 (0.000, 0.052) | 0.567 (0.464, 0.670) |
| qwen3b | J-lens | answered wrong | 102 | 0.284 (0.196, 0.373) | 0.010 (0.000, 0.029) | 0.706 (0.618, 0.794) |
| qwen3b | J-lens | all failures | 199 | 0.347 (0.281, 0.412) | 0.015 (0.000, 0.035) | 0.638 (0.573, 0.704) |
| qwen3b | J-lens | success | 131 | 0.252 (0.176, 0.328) | 0.015 (0.000, 0.038) | 0.733 (0.656, 0.809) |
| qwen3b | logit (gate fallback) | refusal | 97 | 0.392 (0.299, 0.485) | 0.041 (0.010, 0.082) | 0.567 (0.464, 0.670) |
| qwen3b | logit (gate fallback) | answered wrong | 102 | 0.284 (0.196, 0.373) | 0.010 (0.000, 0.029) | 0.706 (0.618, 0.794) |
| qwen3b | logit (gate fallback) | all failures | 199 | 0.337 (0.271, 0.402) | 0.025 (0.005, 0.050) | 0.638 (0.573, 0.704) |
| qwen3b | logit (gate fallback) | success | 131 | 0.237 (0.168, 0.313) | 0.031 (0.008, 0.061) | 0.733 (0.656, 0.809) |
| qwen7b | J-lens | refusal | 18 | 0.222 (0.056, 0.444) | 0.056 (0.000, 0.167) | 0.722 (0.500, 0.889) |
| qwen7b | J-lens | answered wrong | 92 | 0.130 (0.065, 0.207) | 0.109 (0.054, 0.174) | 0.761 (0.674, 0.848) |
| qwen7b | J-lens | all failures | 110 | 0.145 (0.082, 0.218) | 0.100 (0.045, 0.155) | 0.755 (0.673, 0.836) |
| qwen7b | J-lens | success | 220 | 0.041 (0.018, 0.068) | 0.018 (0.005, 0.036) | 0.941 (0.909, 0.968) |
| qwen7b | logit (shown beside) | refusal | 18 | 0.222 (0.056, 0.444) | 0.056 (0.000, 0.167) | 0.722 (0.500, 0.889) |
| qwen7b | logit (shown beside) | answered wrong | 92 | 0.228 (0.141, 0.315) | 0.011 (0.000, 0.033) | 0.761 (0.674, 0.848) |
| qwen7b | logit (shown beside) | all failures | 110 | 0.227 (0.155, 0.309) | 0.018 (0.000, 0.045) | 0.755 (0.673, 0.836) |
| qwen7b | logit (shown beside) | success | 220 | 0.050 (0.023, 0.082) | 0.009 (0.000, 0.023) | 0.941 (0.909, 0.968) |

Refusal = the §23 judge marked the answer `unknown`; answered wrong = any other incorrect answer.

## Descriptive: emergence depth of the gold at P1 (J-lens; questions where it emerges)

| reader | n emerged | n read | mean | q1 | median | q3 |
|---|---|---|---|---|---|---|
| qwen3b | 228 | 330 | 0.776 | 0.611 | 0.889 | 0.944 |
| qwen7b | 305 | 330 | 0.558 | 0.250 | 0.464 | 0.857 |

## Descriptive: gold best rank by position (J-lens; E = best over the evidence lines)

| reader | position | n | mean log10 best rank | median best rank | share best rank <= k |
|---|---|---|---|---|---|
| qwen3b | P1 | 330 | 0.769 | 3.0 | 0.691 |
| qwen3b | P0 | 330 | 2.363 | 212.0 | 0.055 |
| qwen3b | E | 330 | 2.553 | 430.0 | 0.058 |
| qwen3b | Q | 330 | 2.521 | 390.0 | 0.055 |
| qwen7b | P1 | 330 | 0.254 | 1.0 | 0.924 |
| qwen7b | P0 | 330 | 2.689 | 364.5 | 0.097 |
| qwen7b | E | 330 | 3.010 | 713.5 | 0.018 |
| qwen7b | Q | 330 | 2.767 | 275.5 | 0.042 |

## Descriptive: the `unknown` trajectory at P1 on refusals (J-lens)

| reader | token | refusals | reaches top-1 | mean first top-1 depth |
|---|---|---|---|---|
| qwen3b | unknown_sp | 97 | 72 | 0.903 |
| qwen3b | unknown | 97 | 12 | 0.912 |
| qwen7b | unknown_sp | 18 | 10 | 0.871 |
| qwen7b | unknown | 18 | 0 | - |

## Descriptive: outcome shares by question type (J-lens, P1, all in-view questions)

| reader | question type | n | A | B | C |
|---|---|---|---|---|---|
| qwen3b | knowledge-update | 60 | 0.317 | 0.033 | 0.650 |
| qwen3b | multi-session | 57 | 0.175 | 0.035 | 0.789 |
| qwen3b | single-session-assistant | 48 | 0.250 | 0.021 | 0.729 |
| qwen3b | single-session-preference | 13 | 0.923 | 0.000 | 0.077 |
| qwen3b | single-session-user | 61 | 0.344 | 0.000 | 0.656 |
| qwen3b | temporal-reasoning | 91 | 0.308 | 0.000 | 0.692 |
| qwen7b | knowledge-update | 60 | 0.100 | 0.000 | 0.900 |
| qwen7b | multi-session | 57 | 0.035 | 0.000 | 0.965 |
| qwen7b | single-session-assistant | 48 | 0.062 | 0.021 | 0.917 |
| qwen7b | single-session-preference | 13 | 0.000 | 0.769 | 0.231 |
| qwen7b | single-session-user | 61 | 0.066 | 0.016 | 0.918 |
| qwen7b | temporal-reasoning | 91 | 0.110 | 0.033 | 0.857 |

## Descriptive: cross-check against the community lens for qwen7b

| n | Jaccard top-10 at P1, early | middle | late | Claim 1 with the community lens (95% CI) | verdict |
|---|---|---|---|---|---|
| 126 | 0.408 | 0.538 | 0.704 | +0.465 (+0.353, +0.584) | SHOWN |

## Dropped and failed

| reader | fold | no cached prompt | evidence not found |
|---|---|---|---|
| qwen3b | 0 | 0 | 0 |
| qwen3b | 1 | 0 | 0 |
| qwen3b | 2 | 0 | 0 |
| qwen3b | 3 | 0 | 0 |
| qwen3b | 4 | 0 | 0 |
| qwen7b | 0 | 0 | 0 |
| qwen7b | 1 | 0 | 0 |
| qwen7b | 2 | 0 | 0 |
| qwen7b | 3 | 0 | 0 |
| qwen7b | 4 | 0 | 0 |

| reader | records in folds | failed reads (_failed.json) | in-view questions without an output | paired questions without a correct/unknown label |
|---|---|---|---|---|
| qwen3b | 500 | 1 | 0 | 0 |
| qwen7b | 500 | 1 | 0 | 0 |

Top-k label: k10. Verdict rules: gate (a) PASS if the lower bound is above 0, (b) PASS if J-lens > logit on the mean over sets (strict: a tie (e.g. both 0.0) is FAIL); Claim 1 SHOWN if the lower bound is above 0; Claim 2 SHOWN if the AUC lower bound is above 0.5 (pre-registered for qwen3b; the qwen7b row is descriptive, no verdict). Correct/unknown labels are the §23 single-judge verdicts (STATE.md caveat 3).
