# §36 report: can the 3B judge support for its own forced answers? (folds 0, 1, 2, 3, 4)

Verifier wording: §36b frame a. Engines: ['vllm-0.8.5']. Wording sha256 3089973f27ae. Labels: the §35 judge (Qwen2.5-7B) on arm B. 95% intervals: paired bootstrap over questions (all variants of a question in one unit), 10,000 resamples.

## Rows

Fixed-candidate rows (3B): 265; with a candidate 224; no candidate (B still says unknown; forced abstentions, kept in every denominator) 41; candidate rows missing a V score 0.

Label-mass rule (mass < 0.5 on more than 0.1 of rows = not reliable, argmax only):

| verifier/item | rows | low-mass rows | share | reliable |
|---|---|---|---|---|
| qwen3b/V | 224 | 0 | 0.000 | yes |
| qwen3b/Vn | 224 | 0 | 0.000 | yes |
| qwen3b/CutB | 224 | 0 | 0.000 | yes |
| qwen3b/CutF | 180 | 0 | 0.000 | yes |
| qwen3b/Abs | 7 | 0 | 0.000 | yes |
| qwen3b/CF | 216 | 0 | 0.000 | yes |
| qwen3b/LC | 224 | 0 | 0.000 | yes |
| qwen3b/P2p | 224 | 0 | 0.000 | yes |
| qwen3b/P3p | 224 | 0 | 0.000 | yes |
| qwen7b/V | 224 | 0 | 0.000 | yes |
| qwen7b/Vn | 224 | 0 | 0.000 | yes |
| qwen7b/CutB | 224 | 0 | 0.000 | yes |
| qwen7b/CutF | 180 | 0 | 0.000 | yes |
| qwen7b/Abs | 7 | 0 | 0.000 | yes |
| qwen7b/CF | 216 | 0 | 0.000 | yes |
| qwen7b/LC | 224 | 0 | 0.000 | yes |
| qwen7b/V_own | 322 | 0 | 0.000 | yes |
| qwen7b/P2p | 224 | 0 | 0.000 | yes |
| qwen7b/P3p | 224 | 0 | 0.000 | yes |

V readout diagnostics (top-1 token from the stored top-20 logprobs):

| verifier | V rows | share label mass < 0.5 | share top-1 'unknown' | top-1 tokens on low-mass rows (count) |
|---|---|---|---|---|
| qwen3b | 224 | 0.000 | 0.000 | - |
| qwen7b | 224 | 0.000 | 0.000 | - |

## 3B claims (primary; fixed sequence)

| claim | n | estimate (95% CI) | verdict |
|---|---|---|---|
| Claim 1: AUROC of s (V) for B judged correct, candidate rows | 134 pos / 90 neg | 0.567 (0.492, 0.644) | NOT SHOWN (lower bound ≤ 0.5) |
| beside: PR-AUC, prevalence (candidate rows) | | 0.689, 0.598 | |
| beside: AUROC over all fixed rows, no-candidate rows lowest | 134 / 131 | 0.703 (0.639, 0.763) | descriptive (inflated by the forced abstentions) |
| beside: natural flag (C0) AUROC, candidate rows | | 0.466 (0.410, 0.523) | descriptive |
| beside: natural flag AUROC, all fixed rows (as exp36_devcheck.py) | | 0.579 (0.522, 0.635) | descriptive |
| Claim 2: risk(C3) − risk(C0) at C0's coverage | 265 | -0.024 (-0.072, +0.025) | not tested (Claim 1 not SHOWN) |
| Claim 3: risk(C3) − risk(C4) at C0's coverage | 265 | +0.012 (-0.036, +0.060) | not tested (Claim 1 not SHOWN) |
| beside: candidate rows with no C4 score | 0 | | |
| beside: C4 AUROC (m) | | 0.518 (0.439, 0.596) | descriptive |

Claim 4 (descriptive): τ fitted on folds 0-1 at r* = 0.3: s ≥ 0.513; C4: m ≥ 10.995.

| policy | emitted | coverage (95% CI) | risk (95% CI) | wrong added vs C0 | cached refusals recovered correct | cached refusals turned wrong |
|---|---|---|---|---|---|---|
| C0 natural | 170 | 0.642 (0.584, 0.698) | 0.424 (0.351, 0.500) | +0 | 0 | 0 |
| C1 forced (all candidates) | 224 | 0.845 (0.801, 0.888) | 0.402 (0.339, 0.466) | +18 | 36 | 18 |
| C2 self-check (argmax A) | 81 | 0.306 (0.250, 0.361) | 0.309 (0.211, 0.415) | -47 | 11 | 3 |
| C3 at C0's coverage | 170 | 0.642 (0.583, 0.699) | 0.400 (0.327, 0.476) | -4 | 22 | 13 |
| C4 at C0's coverage | 170 | 0.642 (0.584, 0.698) | 0.388 (0.317, 0.463) | -6 | 12 | 2 |
| C4′ (§34b unknown rank) at C0's coverage | 170 | 0.642 (0.584, 0.697) | 0.382 (0.312, 0.458) | -7 | 13 | 7 |
| Claim 4: C3 at τ | 93 | 0.351 (0.294, 0.410) | 0.333 (0.240, 0.433) | -41 | 13 | 4 |
| Claim 4: C4 at τ | 20 | 0.075 (0.045, 0.109) | 0.350 (0.143, 0.571) | -65 | 0 | 0 |

C2 − C0 risk: -0.115 (-0.202, -0.028). C4′ AUROC 0.469 (0.392, 0.546) (0 rows without a §34b read rank lowest).

Robustness (never selected; V stays primary):

| wording | AUROC (95% CI) | C2 coverage | C2 risk |
|---|---|---|---|
| Vn | 0.587 (0.509, 0.663) | 0.623 (0.565, 0.680) | 0.382 (0.311, 0.459) |
| P2p | 0.571 (0.495, 0.647) | 0.302 (0.246, 0.357) | 0.312 (0.214, 0.419) |
| P3p | 0.582 (0.507, 0.657) | 0.287 (0.233, 0.341) | 0.276 (0.179, 0.382) |

## Evidence dependence (controls; both readers verify the same 3B candidates)

| control | 3B verifier | 7B verifier (descriptive) |
|---|---|---|
| A: s(intact) − s(cut), B judged correct | +0.240 (+0.189, +0.291) (n 134) | +0.493 (+0.420, +0.565) (n 134) |
| A′: AUROC intact correct-B vs Cut-F | 0.704 (0.653, 0.754) (134 / 180) | 0.748 (0.700, 0.793) (134 / 180) |
| Length control: s(intact) − s(LC) | +0.250 (+0.199, +0.300) (n 134) | +0.503 (+0.430, +0.575) (n 134) |
| Length control: s(LC) − s(cut) | -0.010 (-0.035, +0.017) | -0.010 (-0.024, +0.003) |
| B (primary): number substitutions, no other copy in memory; follows in both contexts | 0.875 (0.684, 1.000) (n 16; primary row; descriptive (fewer than 20 rows)) | 0.812 (0.600, 1.000) (n 16; primary row; descriptive (fewer than 20 rows)) |
| B (primary): follows in the counterfactual context | 0.938 (0.786, 1.000) | 0.875 (0.688, 1.000) |
| B (primary): follows in the intact context | 0.938 (0.789, 1.000) | 0.875 (0.692, 1.000) |
| B, DESCRIPTIVE ONLY: numbers with other copies in memory, follows in both | 0.778 (0.444, 1.000) (n 9) | 0.333 (0.000, 0.693) (n 9) |
| B, DESCRIPTIVE ONLY: name substitutions (not internally consistent), follows in both | 0.621 (0.441, 0.800) (n 29) | 0.586 (0.400, 0.762) (n 29) |
| B, DESCRIPTIVE ONLY: all counterfactual rows, follows in both | 0.722 (0.600, 0.842) (n 54) | 0.611 (0.479, 0.738) (n 54) |
| Cut-F emitted at τ (unsupported-answer rate) | 0.206 | 0.333 |
| Cut-F argmax A | 0.161 | 0.289 |
| Abs: n, emitted at τ, argmax A | 7, 0.000, 0.000 | 7, 0.429, 0.571 |

7B − 3B difference of the Control A drop (paired): +0.253 (+0.189, +0.317) (n 134).

Cut-F split by the §35 judge verdict on the Fcut answer (judged-correct answers are 'unsupported' by construction):

| verifier | Fcut judged | n | mean s | emitted at τ | argmax A |
|---|---|---|---|---|---|
| qwen3b | correct | 44 | 0.683 | 19 | 18 |
| qwen3b | wrong | 136 | 0.496 | 18 | 11 |
| qwen7b | correct | 44 | 0.675 | 29 | 26 |
| qwen7b | wrong | 136 | 0.286 | 31 | 26 |

## 7B (descriptive; the judge shares the 7B's weights)

The 7B verifying the 3B's candidates (labels: the 3B's B verdicts):

AUROC 0.706 (0.636, 0.772); at C0's coverage risk 0.335 (0.266, 0.408), minus C0 -0.088 (-0.145, -0.033); argmax-A coverage 0.445 (0.384, 0.504), risk 0.254 (0.177, 0.336); at the 3B's dev τ coverage 0.487 (0.425, 0.546), risk 0.271 (0.197, 0.350).

The 7B on its own B candidates (V_own; labels: the 7B's B verdicts): fixed rows 330, no candidate 8; AUROC 0.724 (0.662, 0.783); natural flag AUROC 0.530 (0.505, 0.560); C0 coverage 0.945 (0.920, 0.969), risk 0.282 (0.233, 0.333); V_own at C0's coverage risk 0.285 (0.237, 0.337) (minus C0 +0.003 (-0.014, +0.020)); C4 at C0's coverage risk 0.285 (0.236, 0.337) (V_own − C4 +0.000 (-0.016, +0.016)). Never merged with the comparison above into one size conclusion.

## Cost per verification

| verifier/item | rows | tokens per row | seconds per row |
|---|---|---|---|
| qwen3b/V | 224 | 1075 | 0.0659 |
| qwen3b/Vn | 224 | 1128 | 0.0629 |
| qwen3b/CutB | 224 | 890 | 0.0436 |
| qwen3b/CutF | 180 | 922 | 0.0209 |
| qwen3b/Abs | 7 | 1077 | 0.0764 |
| qwen3b/CF | 216 | 1117 | 0.0317 |
| qwen3b/LC | 224 | 1065 | 0.0561 |
| qwen3b/C4 | 224 | 2401 | 0.2151 |
| qwen3b/P2p | 224 | 1075 | 0.0117 |
| qwen3b/P3p | 224 | 1078 | 0.0079 |
| qwen7b/V | 224 | 1075 | 0.1311 |
| qwen7b/Vn | 224 | 1128 | 0.1190 |
| qwen7b/CutB | 224 | 890 | 0.0866 |
| qwen7b/CutF | 180 | 922 | 0.0395 |
| qwen7b/Abs | 7 | 1077 | 0.1524 |
| qwen7b/CF | 216 | 1117 | 0.0567 |
| qwen7b/LC | 224 | 1065 | 0.1110 |
| qwen7b/V_own | 322 | 1108 | 0.1317 |
| qwen7b/C4 | 322 | 2438 | 0.3977 |
| qwen7b/P2p | 224 | 1075 | 0.1379 |
| qwen7b/P3p | 224 | 1078 | 0.0117 |

Verdict rules (pre-registered, 3B only): Claim 1 SHOWN if the AUROC lower bound > 0.5 (meaningful ≥ 0.65); Claim 2 tested only if Claim 1 SHOWN, SHOWN if the upper bound of risk(C3) − risk(C0) < 0; Claim 3 tested only if Claim 2 SHOWN, SHOWN if the upper bound of risk(C3) − risk(C4) < 0 (meaningful ≤ −0.05); Claim 4 and every control descriptive. C4 for the 7B is on its own candidates and own note.
