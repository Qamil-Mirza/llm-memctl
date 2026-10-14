# §35 report: derivation vs answer selection (folds 0, 1; population 126 in-view answerable questions, paired by question id and prompt)

Engines: ['vllm-0.8.5']. Wording sha256 1c45f71a94df. Every verdict compares arms of the same engine and run (B vs A, F vs N). Judge: the §23 prompt and model (Qwen2.5-7B) on this run's engine (verdict sources listed per reader). D1 decides; D2 beside. 95% intervals: 10,000 bootstrap resamples of questions.

## Population and rows

| reader | population | marked | no marker | frozen-note rows after token check | N | A | B | F | Fcut | Ncut | replacement / cut errors | token mismatches (excluded) | no §34b read (kept) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3b | 126 | 100 | 26 | 100 | 126 | 100 | 100 | 126 | 126 | 126 | 0 | 0 | 0 |
| qwen7b | 126 | 126 | 0 | 126 | 126 | 126 | 126 | 126 | 126 | 126 | 0 | 0 | 0 |

## qwen3b (primary)

Gate G: arm A gives the cached verdict and refusal flag on 0.970 (0.930, 1.000) of 100 marked rows (bound 0.8): **PASS**. On S: 1.000 (1.000, 1.000) of 7.

Marked cached refusals 30 (all cached refusals 32); |S| (D1 true) 7, S as a share of the population 0.056.

| claim | n | estimate (95% CI) | verdict | B outcomes (correct / refusal / wrong) | A outcomes |
|---|---|---|---|---|---|
| Claim 1: P(B correct) − P(A correct) on S (D1) | 7 | +0.571 (+0.143, +0.857) | UNDERPOWERED (|S| = 7 < 15) | 4 / 2 / 1 | 0 / 7 / 0 |
| s = P(B correct \| S), descriptive (Wilson 0.250, 0.842) | 7 | 0.571 (0.143, 0.857) | descriptive | | |
| beside, not deciding: the same difference on the D2 set | 11 | +0.636 (+0.364, +0.909) | descriptive | 8 / 3 / 0 | 1 / 10 / 0 |
| Claim 2: P(B correct \| S) − P(B correct \| S') | 7 vs 23 | +0.354 (-0.062, +0.739) | NOT SHOWN | | |

Claim 3 (descriptive, no threshold): paired accuracy differences within the engine.

| comparison | rows | arm accuracy | baseline accuracy | difference (95% CI) | 7B − 3B cached accuracy on the same rows | gap share (95% CI) |
|---|---|---|---|---|---|---|
| acc(B) − acc(A), marked rows after token check | 100 | 0.500 | 0.390 | +0.110 (+0.050, +0.180) | 0.210 | +0.52 (+0.24, +0.86) |
| acc(F) − acc(N), marked rows | 100 | 0.520 | 0.410 | +0.110 (+0.030, +0.190) | 0.210 | +0.52 (+0.14, +0.90) |
| acc(F) − acc(N), all rows (descriptive) | 126 | 0.548 | 0.421 | +0.127 (+0.056, +0.198) | 0.198 | +0.64 (+0.28, +1.00) |
| acc(N) − cached, marked rows (cross-engine reproduction) | 100 | 0.410 | 0.390 | +0.020 (-0.060, +0.100) | - | - |
| acc(N) − cached, all rows | 126 | 0.421 | 0.429 | -0.008 (-0.087, +0.071) | - | - |
| acc(A) − cached, marked rows after token check | 100 | 0.390 | 0.390 | +0.000 (-0.030, +0.030) | - | - |

Costs (pre-registered, reported with the claims).

| cost | n | share (95% CI) |
|---|---|---|
| (i) S' (cached refusal, D1 false): B judged wrong | 23 | 0.174 (0.043, 0.348) |
| (i) all marked cached refusals: B judged wrong | 30 | 0.167 (0.033, 0.300) |
| (ii) evidence cut: Fcut judged correct (plausibility floor) | 126 | 0.159 (0.095, 0.222) |
| (ii) evidence cut: Ncut refuses (unknown is right) | 126 | 0.603 (0.516, 0.690) |
| evidence cut: Fcut refuses (descriptive) | 126 | 0.381 (0.294, 0.468) |
| evidence cut: Ncut judged correct (descriptive) | 126 | 0.087 (0.040, 0.143) |

Reproduction: arm A gives the cached final answer exactly on 0.970 (0.930, 1.000); arm N gives the cached final answer on 0.611 (0.524, 0.698) and the whole cached output on 0.341 (0.262, 0.421).

D1 vs D2 on marked notes (n 100): both 33, D1 only 4, D2 only 14, neither 49; agreement 0.820, Cohen's kappa 0.634. On marked cached refusals: D1 7, D2 11 of 30.

Judge engine check: the cached answers re-judged in this run agree with the §23 labels on 125 of 126 (0.992).

Per question type (descriptive; accuracy with n in brackets; A and B on marked rows after the token check; flagged types have a different judge prompt):

| type | n | frozen | cached | N | A | B | F | \|S\| | S: A correct | S: B correct |
|---|---|---|---|---|---|---|---|---|---|---|
| knowledge-update (flagged) | 25 | 16 | 0.440 | 0.320 (25) | 0.375 (16) | 0.438 (16) | 0.360 (25) | 0 | 0 | 0 |
| multi-session | 21 | 19 | 0.524 | 0.476 (21) | 0.474 (19) | 0.632 (19) | 0.714 (21) | 1 | 0 | 0 |
| single-session-assistant | 18 | 13 | 0.500 | 0.667 (18) | 0.385 (13) | 0.385 (13) | 0.778 (18) | 0 | 0 | 0 |
| single-session-preference (flagged) | 4 | 4 | 0.000 | 0.000 (4) | 0.000 (4) | 0.000 (4) | 0.000 (4) | 0 | 0 | 0 |
| single-session-user | 25 | 19 | 0.640 | 0.640 (25) | 0.579 (19) | 0.789 (19) | 0.920 (25) | 3 | 0 | 3 |
| temporal-reasoning | 33 | 29 | 0.212 | 0.212 (33) | 0.276 (29) | 0.379 (29) | 0.242 (33) | 3 | 0 | 1 |

Judge verdict sources: {'exp35_cache': 930}.

## qwen7b (descriptive)

Gate G: arm A gives the cached verdict and refusal flag on 0.976 (0.944, 1.000) of 126 marked rows (bound 0.8): **PASS**. On S: 1.000 (1.000, 1.000) of 3.

Marked cached refusals 9 (all cached refusals 9); |S| (D1 true) 3, S as a share of the population 0.024.

| claim | n | estimate (95% CI) | verdict | B outcomes (correct / refusal / wrong) | A outcomes |
|---|---|---|---|---|---|
| Claim 1: P(B correct) − P(A correct) on S (D1) | 3 | +0.333 (+0.000, +1.000) | descriptive | 1 / 2 / 0 | 0 / 3 / 0 |
| s = P(B correct \| S), descriptive (Wilson 0.061, 0.792) | 3 | 0.333 (0.000, 1.000) | descriptive | | |
| beside, not deciding: the same difference on the D2 set | 1 | +1.000 (no interval) | descriptive | 1 / 0 / 0 | 0 / 1 / 0 |
| Claim 2: P(B correct \| S) − P(B correct \| S') | 3 vs 6 | +0.167 (-0.333, +0.833) | descriptive | | |

Claim 3 (descriptive, no threshold): paired accuracy differences within the engine.

| comparison | rows | arm accuracy | baseline accuracy | difference (95% CI) | 7B − 3B cached accuracy on the same rows | gap share (95% CI) |
|---|---|---|---|---|---|---|
| acc(B) − acc(A), marked rows after token check | 126 | 0.651 | 0.611 | +0.040 (+0.008, +0.079) | - | - |
| acc(F) − acc(N), marked rows | 126 | 0.675 | 0.619 | +0.056 (+0.016, +0.103) | - | - |
| acc(F) − acc(N), all rows (descriptive) | 126 | 0.675 | 0.619 | +0.056 (+0.016, +0.103) | - | - |
| acc(N) − cached, marked rows (cross-engine reproduction) | 126 | 0.619 | 0.627 | -0.008 (-0.056, +0.040) | - | - |
| acc(N) − cached, all rows | 126 | 0.619 | 0.627 | -0.008 (-0.056, +0.040) | - | - |
| acc(A) − cached, marked rows after token check | 126 | 0.611 | 0.627 | -0.016 (-0.040, +0.000) | - | - |

Costs (pre-registered, reported with the claims).

| cost | n | share (95% CI) |
|---|---|---|
| (i) S' (cached refusal, D1 false): B judged wrong | 6 | 0.333 (0.000, 0.667) |
| (i) all marked cached refusals: B judged wrong | 9 | 0.222 (0.000, 0.556) |
| (ii) evidence cut: Fcut judged correct (plausibility floor) | 126 | 0.190 (0.127, 0.262) |
| (ii) evidence cut: Ncut refuses (unknown is right) | 126 | 0.627 (0.548, 0.714) |
| evidence cut: Fcut refuses (descriptive) | 126 | 0.341 (0.262, 0.429) |
| evidence cut: Ncut judged correct (descriptive) | 126 | 0.143 (0.087, 0.206) |

Reproduction: arm A gives the cached final answer exactly on 0.905 (0.849, 0.952); arm N gives the cached final answer on 0.802 (0.730, 0.873) and the whole cached output on 0.421 (0.333, 0.508).

D1 vs D2 on marked notes (n 126): both 49, D1 only 12, D2 only 25, neither 40; agreement 0.706, Cohen's kappa 0.416. On marked cached refusals: D1 3, D2 1 of 9.

Judge engine check: the cached answers re-judged in this run agree with the §23 labels on 126 of 126 (1.000).

Per question type (descriptive; accuracy with n in brackets; A and B on marked rows after the token check; flagged types have a different judge prompt):

| type | n | frozen | cached | N | A | B | F | \|S\| | S: A correct | S: B correct |
|---|---|---|---|---|---|---|---|---|---|---|
| knowledge-update (flagged) | 25 | 25 | 0.560 | 0.560 (25) | 0.560 (25) | 0.640 (25) | 0.720 (25) | 0 | 0 | 0 |
| multi-session | 21 | 21 | 0.619 | 0.619 (21) | 0.619 (21) | 0.619 (21) | 0.762 (21) | 1 | 0 | 0 |
| single-session-assistant | 18 | 18 | 0.944 | 0.889 (18) | 0.889 (18) | 0.944 (18) | 0.889 (18) | 0 | 0 | 0 |
| single-session-preference (flagged) | 4 | 4 | 0.250 | 0.000 (4) | 0.000 (4) | 0.250 (4) | 0.250 (4) | 0 | 0 | 0 |
| single-session-user | 25 | 25 | 0.840 | 0.920 (25) | 0.840 (25) | 0.840 (25) | 0.920 (25) | 1 | 0 | 0 |
| temporal-reasoning | 33 | 33 | 0.394 | 0.364 (33) | 0.394 (33) | 0.424 (33) | 0.333 (33) | 1 | 0 | 1 |

Judge verdict sources: {'exp35_cache': 1008}.

## Natural control (cached, descriptive)

Answerable labelled questions with zero evidence lines in view: qwen3b: 15 of 188, cached refusals 10; qwen7b: 15 of 188, cached refusals 9.

Verdict rules (pre-registered, qwen3b only): gate G at 0.8; Claim 1 SHOWN if the lower bound of P(B correct) − P(A correct) on S is above 0, UNDERPOWERED if |S| < 15; Claim 2 SHOWN if the lower bound is above 0; Claim 3 has no threshold (a lift under 0.01 is within the reader noise floor); its gap share is against the 7B's minus the 3B's cached accuracy on the same rows.
