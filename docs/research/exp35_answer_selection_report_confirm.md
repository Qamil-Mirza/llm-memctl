# §35 report: derivation vs answer selection (folds 2, 3, 4; population 204 in-view answerable questions, paired by question id and prompt)

Engines: ['vllm-0.8.5']. Wording sha256 1c45f71a94df. Every verdict compares arms of the same engine and run (B vs A, F vs N). Judge: the §23 prompt and model (Qwen2.5-7B) on this run's engine (verdict sources listed per reader). D1 decides; D2 beside. 95% intervals: 10,000 bootstrap resamples of questions.

## Population and rows

| reader | population | marked | no marker | frozen-note rows after token check | N | A | B | F | Fcut | Ncut | replacement / cut errors | token mismatches (excluded) | no §34b read (kept) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3b | 204 | 165 | 39 | 165 | 204 | 165 | 165 | 204 | 204 | 204 | 0 | 0 | 0 |
| qwen7b | 204 | 204 | 0 | 204 | 204 | 204 | 204 | 204 | 204 | 204 | 0 | 0 | 0 |

## qwen3b (primary)

Gate G: arm A gives the cached verdict and refusal flag on 0.964 (0.933, 0.988) of 165 marked rows (bound 0.8): **PASS**. On S: 1.000 (1.000, 1.000) of 22.

Marked cached refusals 62 (all cached refusals 65); |S| (D1 true) 22, S as a share of the population 0.108.

| claim | n | estimate (95% CI) | verdict | B outcomes (correct / refusal / wrong) | A outcomes |
|---|---|---|---|---|---|
| Claim 1: P(B correct) − P(A correct) on S (D1) | 22 | +0.636 (+0.455, +0.818) | SHOWN | 14 / 6 / 2 | 0 / 22 / 0 |
| s = P(B correct \| S), descriptive (Wilson 0.430, 0.803) | 22 | 0.636 (0.455, 0.818) | descriptive | | |
| beside, not deciding: the same difference on the D2 set | 23 | +0.652 (+0.435, +0.826) | descriptive | 16 / 5 / 2 | 1 / 22 / 0 |
| Claim 2: P(B correct \| S) − P(B correct \| S') | 22 vs 40 | +0.311 (+0.055, +0.552) | SHOWN | | |

Claim 3 (descriptive, no threshold): paired accuracy differences within the engine.

| comparison | rows | arm accuracy | baseline accuracy | difference (95% CI) | 7B − 3B cached accuracy on the same rows | gap share (95% CI) |
|---|---|---|---|---|---|---|
| acc(B) − acc(A), marked rows after token check | 165 | 0.509 | 0.333 | +0.176 (+0.109, +0.242) | 0.333 | +0.53 (+0.33, +0.73) |
| acc(F) − acc(N), marked rows | 165 | 0.545 | 0.303 | +0.242 (+0.164, +0.321) | 0.333 | +0.73 (+0.49, +0.96) |
| acc(F) − acc(N), all rows (descriptive) | 204 | 0.569 | 0.343 | +0.225 (+0.157, +0.294) | 0.314 | +0.72 (+0.50, +0.94) |
| acc(N) − cached, marked rows (cross-engine reproduction) | 165 | 0.303 | 0.327 | -0.024 (-0.079, +0.030) | - | - |
| acc(N) − cached, all rows | 204 | 0.343 | 0.377 | -0.034 (-0.078, +0.010) | - | - |
| acc(A) − cached, marked rows after token check | 165 | 0.333 | 0.327 | +0.006 (-0.018, +0.036) | - | - |

Costs (pre-registered, reported with the claims).

| cost | n | share (95% CI) |
|---|---|---|
| (i) S' (cached refusal, D1 false): B judged wrong | 40 | 0.275 (0.150, 0.425) |
| (i) all marked cached refusals: B judged wrong | 62 | 0.210 (0.113, 0.306) |
| (ii) evidence cut: Fcut judged correct (plausibility floor) | 204 | 0.147 (0.098, 0.196) |
| (ii) evidence cut: Ncut refuses (unknown is right) | 204 | 0.647 (0.578, 0.711) |
| evidence cut: Fcut refuses (descriptive) | 204 | 0.319 (0.255, 0.382) |
| evidence cut: Ncut judged correct (descriptive) | 204 | 0.088 (0.054, 0.127) |

Reproduction: arm A gives the cached final answer exactly on 0.952 (0.915, 0.982); arm N gives the cached final answer on 0.642 (0.574, 0.706) and the whole cached output on 0.333 (0.270, 0.397).

D1 vs D2 on marked notes (n 165): both 47, D1 only 16, D2 only 30, neither 72; agreement 0.721, Cohen's kappa 0.433. On marked cached refusals: D1 22, D2 23 of 62.

Judge engine check: the cached answers re-judged in this run agree with the §23 labels on 204 of 204 (1.000).

Per question type (descriptive; accuracy with n in brackets; A and B on marked rows after the token check; flagged types have a different judge prompt):

| type | n | frozen | cached | N | A | B | F | \|S\| | S: A correct | S: B correct |
|---|---|---|---|---|---|---|---|---|---|---|
| knowledge-update (flagged) | 35 | 24 | 0.514 | 0.486 (35) | 0.417 (24) | 0.500 (24) | 0.514 (35) | 3 | 0 | 1 |
| multi-session | 36 | 32 | 0.333 | 0.278 (36) | 0.375 (32) | 0.531 (32) | 0.472 (36) | 1 | 0 | 1 |
| single-session-assistant | 30 | 20 | 0.533 | 0.467 (30) | 0.600 (20) | 0.600 (20) | 0.833 (30) | 0 | 0 | 0 |
| single-session-preference (flagged) | 9 | 7 | 0.111 | 0.000 (9) | 0.000 (7) | 0.000 (7) | 0.111 (9) | 0 | 0 | 0 |
| single-session-user | 36 | 28 | 0.528 | 0.556 (36) | 0.393 (28) | 0.679 (28) | 0.806 (36) | 8 | 0 | 6 |
| temporal-reasoning | 58 | 54 | 0.190 | 0.155 (58) | 0.185 (54) | 0.444 (54) | 0.448 (58) | 10 | 0 | 6 |

Judge verdict sources: {'exp35_cache': 1515}.

## qwen7b (descriptive)

Gate G: arm A gives the cached verdict and refusal flag on 0.995 (0.985, 1.000) of 204 marked rows (bound 0.8): **PASS**. On S: 0.000 (0.000, 0.000) of 1.

Marked cached refusals 9 (all cached refusals 9); |S| (D1 true) 1, S as a share of the population 0.005.

| claim | n | estimate (95% CI) | verdict | B outcomes (correct / refusal / wrong) | A outcomes |
|---|---|---|---|---|---|
| Claim 1: P(B correct) − P(A correct) on S (D1) | 1 | +0.000 (no interval) | descriptive | 0 / 0 / 1 | 0 / 0 / 1 |
| s = P(B correct \| S), descriptive (Wilson 0.000, 0.793) | 1 | 0.000 (0.000, 0.000) | descriptive | | |
| beside, not deciding: the same difference on the D2 set | 0 | - | descriptive | 0 / 0 / 0 | 0 / 0 / 0 |
| Claim 2: P(B correct \| S) − P(B correct \| S') | 1 vs 8 | -0.125 (-0.375, +0.000) | descriptive | | |

Claim 3 (descriptive, no threshold): paired accuracy differences within the engine.

| comparison | rows | arm accuracy | baseline accuracy | difference (95% CI) | 7B − 3B cached accuracy on the same rows | gap share (95% CI) |
|---|---|---|---|---|---|---|
| acc(B) − acc(A), marked rows after token check | 204 | 0.711 | 0.691 | +0.020 (+0.005, +0.039) | - | - |
| acc(F) − acc(N), marked rows | 204 | 0.716 | 0.711 | +0.005 (-0.034, +0.044) | - | - |
| acc(F) − acc(N), all rows (descriptive) | 204 | 0.716 | 0.711 | +0.005 (-0.034, +0.044) | - | - |
| acc(N) − cached, marked rows (cross-engine reproduction) | 204 | 0.711 | 0.691 | +0.020 (-0.010, +0.049) | - | - |
| acc(N) − cached, all rows | 204 | 0.711 | 0.691 | +0.020 (-0.010, +0.049) | - | - |
| acc(A) − cached, marked rows after token check | 204 | 0.691 | 0.691 | +0.000 (+0.000, +0.000) | - | - |

Costs (pre-registered, reported with the claims).

| cost | n | share (95% CI) |
|---|---|---|
| (i) S' (cached refusal, D1 false): B judged wrong | 8 | 0.500 (0.125, 0.875) |
| (i) all marked cached refusals: B judged wrong | 9 | 0.556 (0.222, 0.889) |
| (ii) evidence cut: Fcut judged correct (plausibility floor) | 204 | 0.240 (0.181, 0.299) |
| (ii) evidence cut: Ncut refuses (unknown is right) | 204 | 0.627 (0.564, 0.691) |
| evidence cut: Fcut refuses (descriptive) | 204 | 0.245 (0.186, 0.304) |
| evidence cut: Ncut judged correct (descriptive) | 204 | 0.186 (0.137, 0.240) |

Reproduction: arm A gives the cached final answer exactly on 0.917 (0.877, 0.951); arm N gives the cached final answer on 0.770 (0.711, 0.828) and the whole cached output on 0.392 (0.324, 0.461).

D1 vs D2 on marked notes (n 204): both 103, D1 only 16, D2 only 39, neither 46; agreement 0.730, Cohen's kappa 0.423. On marked cached refusals: D1 1, D2 0 of 9.

Judge engine check: the cached answers re-judged in this run agree with the §23 labels on 204 of 204 (1.000).

Per question type (descriptive; accuracy with n in brackets; A and B on marked rows after the token check; flagged types have a different judge prompt):

| type | n | frozen | cached | N | A | B | F | \|S\| | S: A correct | S: B correct |
|---|---|---|---|---|---|---|---|---|---|---|
| knowledge-update (flagged) | 35 | 35 | 0.657 | 0.686 (35) | 0.657 (35) | 0.657 (35) | 0.714 (35) | 1 | 0 | 0 |
| multi-session | 36 | 36 | 0.750 | 0.750 (36) | 0.750 (36) | 0.806 (36) | 0.806 (36) | 0 | 0 | 0 |
| single-session-assistant | 30 | 30 | 0.933 | 0.933 (30) | 0.933 (30) | 0.933 (30) | 0.933 (30) | 0 | 0 | 0 |
| single-session-preference (flagged) | 9 | 9 | 0.111 | 0.111 (9) | 0.111 (9) | 0.111 (9) | 0.222 (9) | 0 | 0 | 0 |
| single-session-user | 36 | 36 | 0.833 | 0.861 (36) | 0.833 (36) | 0.861 (36) | 0.861 (36) | 0 | 0 | 0 |
| temporal-reasoning | 58 | 58 | 0.552 | 0.586 (58) | 0.552 (58) | 0.569 (58) | 0.534 (58) | 0 | 0 | 0 |

Judge verdict sources: {'exp35_cache': 1632}.

## Natural control (cached, descriptive)

Answerable labelled questions with zero evidence lines in view: qwen3b: 28 of 282, cached refusals 20; qwen7b: 28 of 282, cached refusals 20.

Verdict rules (pre-registered, qwen3b only): gate G at 0.8; Claim 1 SHOWN if the lower bound of P(B correct) − P(A correct) on S is above 0, UNDERPOWERED if |S| < 15; Claim 2 SHOWN if the lower bound is above 0; Claim 3 has no threshold (a lift under 0.01 is within the reader noise floor); its gap share is against the 7B's minus the 3B's cached accuracy on the same rows.
