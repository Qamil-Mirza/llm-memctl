# §35 report: derivation vs answer selection (folds 0, 1, 2, 3, 4; population 330 in-view answerable questions, paired by question id and prompt)

Engines: ['vllm-0.8.5']. Wording sha256 1c45f71a94df. Every verdict compares arms of the same engine and run (B vs A, F vs N). Judge: the §23 prompt and model (Qwen2.5-7B) on this run's engine (verdict sources listed per reader). D1 decides; D2 beside. 95% intervals: 10,000 bootstrap resamples of questions.

## Population and rows

| reader | population | marked | no marker | frozen-note rows after token check | N | A | B | F | Fcut | Ncut | replacement / cut errors | token mismatches (excluded) | no §34b read (kept) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3b | 330 | 265 | 65 | 265 | 330 | 265 | 265 | 330 | 330 | 330 | 0 | 0 | 0 |
| qwen7b | 330 | 330 | 0 | 330 | 330 | 330 | 330 | 330 | 330 | 330 | 0 | 0 | 0 |

## qwen3b (primary)

Gate G: arm A gives the cached verdict and refusal flag on 0.966 (0.943, 0.985) of 265 marked rows (bound 0.8): **PASS**. On S: 1.000 (1.000, 1.000) of 29.

Marked cached refusals 92 (all cached refusals 97); |S| (D1 true) 29, S as a share of the population 0.088.

| claim | n | estimate (95% CI) | verdict | B outcomes (correct / refusal / wrong) | A outcomes |
|---|---|---|---|---|---|
| Claim 1: P(B correct) − P(A correct) on S (D1) | 29 | +0.621 (+0.448, +0.793) | SHOWN | 18 / 8 / 3 | 0 / 29 / 0 |
| s = P(B correct \| S), descriptive (Wilson 0.440, 0.773) | 29 | 0.621 (0.448, 0.793) | descriptive | | |
| beside, not deciding: the same difference on the D2 set | 34 | +0.647 (+0.471, +0.794) | descriptive | 24 / 8 / 2 | 2 / 32 / 0 |
| Claim 2: P(B correct \| S) − P(B correct \| S') | 29 vs 63 | +0.335 (+0.123, +0.536) | SHOWN | | |

Claim 3 (descriptive, no threshold): paired accuracy differences within the engine.

| comparison | rows | arm accuracy | baseline accuracy | difference (95% CI) | 7B − 3B cached accuracy on the same rows | gap share (95% CI) |
|---|---|---|---|---|---|---|
| acc(B) − acc(A), marked rows after token check | 265 | 0.506 | 0.355 | +0.151 (+0.102, +0.204) | 0.287 | +0.53 (+0.36, +0.71) |
| acc(F) − acc(N), marked rows | 265 | 0.536 | 0.343 | +0.192 (+0.132, +0.253) | 0.287 | +0.67 (+0.46, +0.88) |
| acc(F) − acc(N), all rows (descriptive) | 330 | 0.561 | 0.373 | +0.188 (+0.136, +0.239) | 0.270 | +0.70 (+0.51, +0.89) |
| acc(N) − cached, marked rows (cross-engine reproduction) | 265 | 0.343 | 0.351 | -0.008 (-0.053, +0.038) | - | - |
| acc(N) − cached, all rows | 330 | 0.373 | 0.397 | -0.024 (-0.067, +0.018) | - | - |
| acc(A) − cached, marked rows after token check | 265 | 0.355 | 0.351 | +0.004 (-0.015, +0.023) | - | - |

Costs (pre-registered, reported with the claims).

| cost | n | share (95% CI) |
|---|---|---|
| (i) S' (cached refusal, D1 false): B judged wrong | 63 | 0.238 (0.143, 0.349) |
| (i) all marked cached refusals: B judged wrong | 92 | 0.196 (0.120, 0.283) |
| (ii) evidence cut: Fcut judged correct (plausibility floor) | 330 | 0.152 (0.115, 0.191) |
| (ii) evidence cut: Ncut refuses (unknown is right) | 330 | 0.630 (0.579, 0.682) |
| evidence cut: Fcut refuses (descriptive) | 330 | 0.342 (0.291, 0.394) |
| evidence cut: Ncut judged correct (descriptive) | 330 | 0.088 (0.058, 0.121) |

Reproduction: arm A gives the cached final answer exactly on 0.958 (0.932, 0.981); arm N gives the cached final answer on 0.630 (0.579, 0.682) and the whole cached output on 0.336 (0.285, 0.388).

D1 vs D2 on marked notes (n 265): both 80, D1 only 20, D2 only 44, neither 121; agreement 0.758, Cohen's kappa 0.509. On marked cached refusals: D1 29, D2 34 of 92.

Judge engine check: the cached answers re-judged in this run agree with the §23 labels on 329 of 330 (0.997).

Per question type (descriptive; accuracy with n in brackets; A and B on marked rows after the token check; flagged types have a different judge prompt):

| type | n | frozen | cached | N | A | B | F | \|S\| | S: A correct | S: B correct |
|---|---|---|---|---|---|---|---|---|---|---|
| knowledge-update (flagged) | 60 | 40 | 0.483 | 0.417 (60) | 0.400 (40) | 0.475 (40) | 0.450 (60) | 3 | 0 | 1 |
| multi-session | 57 | 51 | 0.404 | 0.351 (57) | 0.412 (51) | 0.569 (51) | 0.561 (57) | 2 | 0 | 1 |
| single-session-assistant | 48 | 33 | 0.521 | 0.542 (48) | 0.515 (33) | 0.515 (33) | 0.812 (48) | 0 | 0 | 0 |
| single-session-preference (flagged) | 13 | 11 | 0.077 | 0.000 (13) | 0.000 (11) | 0.000 (11) | 0.077 (13) | 0 | 0 | 0 |
| single-session-user | 61 | 47 | 0.574 | 0.590 (61) | 0.468 (47) | 0.723 (47) | 0.852 (61) | 11 | 0 | 9 |
| temporal-reasoning | 91 | 83 | 0.198 | 0.176 (91) | 0.217 (83) | 0.422 (83) | 0.374 (91) | 13 | 0 | 7 |

Judge verdict sources: {'exp35_cache': 2445}.

## qwen7b (descriptive)

Gate G: arm A gives the cached verdict and refusal flag on 0.988 (0.976, 0.997) of 330 marked rows (bound 0.8): **PASS**. On S: 0.750 (0.250, 1.000) of 4.

Marked cached refusals 18 (all cached refusals 18); |S| (D1 true) 4, S as a share of the population 0.012.

| claim | n | estimate (95% CI) | verdict | B outcomes (correct / refusal / wrong) | A outcomes |
|---|---|---|---|---|---|
| Claim 1: P(B correct) − P(A correct) on S (D1) | 4 | +0.250 (+0.000, +0.750) | descriptive | 1 / 2 / 1 | 0 / 3 / 1 |
| s = P(B correct \| S), descriptive (Wilson 0.046, 0.699) | 4 | 0.250 (0.000, 0.750) | descriptive | | |
| beside, not deciding: the same difference on the D2 set | 1 | +1.000 (no interval) | descriptive | 1 / 0 / 0 | 0 / 1 / 0 |
| Claim 2: P(B correct \| S) − P(B correct \| S') | 4 vs 14 | +0.107 (-0.286, +0.607) | descriptive | | |

Claim 3 (descriptive, no threshold): paired accuracy differences within the engine.

| comparison | rows | arm accuracy | baseline accuracy | difference (95% CI) | 7B − 3B cached accuracy on the same rows | gap share (95% CI) |
|---|---|---|---|---|---|---|
| acc(B) − acc(A), marked rows after token check | 330 | 0.688 | 0.661 | +0.027 (+0.012, +0.045) | - | - |
| acc(F) − acc(N), marked rows | 330 | 0.700 | 0.676 | +0.024 (-0.006, +0.055) | - | - |
| acc(F) − acc(N), all rows (descriptive) | 330 | 0.700 | 0.676 | +0.024 (-0.006, +0.055) | - | - |
| acc(N) − cached, marked rows (cross-engine reproduction) | 330 | 0.676 | 0.667 | +0.009 (-0.015, +0.036) | - | - |
| acc(N) − cached, all rows | 330 | 0.676 | 0.667 | +0.009 (-0.015, +0.036) | - | - |
| acc(A) − cached, marked rows after token check | 330 | 0.661 | 0.667 | -0.006 (-0.015, +0.000) | - | - |

Costs (pre-registered, reported with the claims).

| cost | n | share (95% CI) |
|---|---|---|
| (i) S' (cached refusal, D1 false): B judged wrong | 14 | 0.429 (0.143, 0.714) |
| (i) all marked cached refusals: B judged wrong | 18 | 0.389 (0.167, 0.611) |
| (ii) evidence cut: Fcut judged correct (plausibility floor) | 330 | 0.221 (0.179, 0.267) |
| (ii) evidence cut: Ncut refuses (unknown is right) | 330 | 0.627 (0.573, 0.679) |
| evidence cut: Fcut refuses (descriptive) | 330 | 0.282 (0.233, 0.333) |
| evidence cut: Ncut judged correct (descriptive) | 330 | 0.170 (0.130, 0.212) |

Reproduction: arm A gives the cached final answer exactly on 0.912 (0.882, 0.939); arm N gives the cached final answer on 0.782 (0.736, 0.827) and the whole cached output on 0.403 (0.352, 0.458).

D1 vs D2 on marked notes (n 330): both 152, D1 only 28, D2 only 64, neither 86; agreement 0.721, Cohen's kappa 0.426. On marked cached refusals: D1 4, D2 1 of 18.

Judge engine check: the cached answers re-judged in this run agree with the §23 labels on 330 of 330 (1.000).

Per question type (descriptive; accuracy with n in brackets; A and B on marked rows after the token check; flagged types have a different judge prompt):

| type | n | frozen | cached | N | A | B | F | \|S\| | S: A correct | S: B correct |
|---|---|---|---|---|---|---|---|---|---|---|
| knowledge-update (flagged) | 60 | 60 | 0.617 | 0.633 (60) | 0.617 (60) | 0.650 (60) | 0.717 (60) | 1 | 0 | 0 |
| multi-session | 57 | 57 | 0.702 | 0.702 (57) | 0.702 (57) | 0.737 (57) | 0.789 (57) | 1 | 0 | 0 |
| single-session-assistant | 48 | 48 | 0.938 | 0.917 (48) | 0.917 (48) | 0.938 (48) | 0.917 (48) | 0 | 0 | 0 |
| single-session-preference (flagged) | 13 | 13 | 0.154 | 0.077 (13) | 0.077 (13) | 0.154 (13) | 0.231 (13) | 0 | 0 | 0 |
| single-session-user | 61 | 61 | 0.836 | 0.885 (61) | 0.836 (61) | 0.852 (61) | 0.885 (61) | 1 | 0 | 0 |
| temporal-reasoning | 91 | 91 | 0.495 | 0.505 (91) | 0.495 (91) | 0.516 (91) | 0.462 (91) | 1 | 0 | 1 |

Judge verdict sources: {'exp35_cache': 2640}.

## Natural control (cached, descriptive)

Answerable labelled questions with zero evidence lines in view: qwen3b: 43 of 470, cached refusals 30; qwen7b: 43 of 470, cached refusals 29.

Verdict rules (pre-registered, qwen3b only): gate G at 0.8; Claim 1 SHOWN if the lower bound of P(B correct) − P(A correct) on S is above 0, UNDERPOWERED if |S| < 15; Claim 2 SHOWN if the lower bound is above 0; Claim 3 has no threshold (a lift under 0.01 is within the reader noise floor); its gap share is against the 7B's minus the 3B's cached accuracy on the same rows.
