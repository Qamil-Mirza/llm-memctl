# §36 report: can the 3B judge support for its own forced answers? (folds 0, 1)

Verifier wording: §36b frame a. Engines: ['vllm-0.8.5']. Wording sha256 3089973f27ae. Labels: the §35 judge (Qwen2.5-7B) on arm B. 95% intervals: paired bootstrap over questions (all variants of a question in one unit), 10,000 resamples.

## Rows

Fixed-candidate rows (3B): 100; with a candidate 84; no candidate (B still says unknown; forced abstentions, kept in every denominator) 16; candidate rows missing a V score 0.

Label-mass rule (mass < 0.5 on more than 0.1 of rows = not reliable, argmax only):

| verifier/item | rows | low-mass rows | share | reliable |
|---|---|---|---|---|
| qwen3b/V | 84 | 0 | 0.000 | yes |
| qwen3b/Vn | 84 | 0 | 0.000 | yes |
| qwen3b/CutB | 84 | 0 | 0.000 | yes |
| qwen3b/CutF | 60 | 0 | 0.000 | yes |
| qwen3b/Abs | 3 | 0 | 0.000 | yes |
| qwen3b/CF | 84 | 0 | 0.000 | yes |
| qwen3b/LC | 84 | 0 | 0.000 | yes |
| qwen3b/P2p | 84 | 0 | 0.000 | yes |
| qwen3b/P3p | 84 | 0 | 0.000 | yes |
| qwen7b/V | 84 | 0 | 0.000 | yes |
| qwen7b/Vn | 84 | 0 | 0.000 | yes |
| qwen7b/CutB | 84 | 0 | 0.000 | yes |
| qwen7b/CutF | 60 | 0 | 0.000 | yes |
| qwen7b/Abs | 3 | 0 | 0.000 | yes |
| qwen7b/CF | 84 | 0 | 0.000 | yes |
| qwen7b/LC | 84 | 0 | 0.000 | yes |
| qwen7b/V_own | 121 | 0 | 0.000 | yes |
| qwen7b/P2p | 84 | 0 | 0.000 | yes |
| qwen7b/P3p | 84 | 0 | 0.000 | yes |

V readout diagnostics (top-1 token from the stored top-20 logprobs):

| verifier | V rows | share label mass < 0.5 | share top-1 'unknown' | top-1 tokens on low-mass rows (count) |
|---|---|---|---|---|
| qwen3b | 84 | 0.000 | 0.000 | - |
| qwen7b | 84 | 0.000 | 0.000 | - |

## 3B claims (primary; fixed sequence) (development folds: not a confirmation verdict)

| claim | n | estimate (95% CI) | verdict |
|---|---|---|---|
| Claim 1: AUROC of s (V) for B judged correct, candidate rows | 50 pos / 34 neg | 0.559 (0.434, 0.682) | NOT SHOWN (lower bound ≤ 0.5) |
| beside: PR-AUC, prevalence (candidate rows) | | 0.683, 0.595 | |
| beside: AUROC over all fixed rows, no-candidate rows lowest | 50 / 50 | 0.700 (0.593, 0.800) | descriptive (inflated by the forced abstentions) |
| beside: natural flag (C0) AUROC, candidate rows | | 0.484 (0.403, 0.566) | descriptive |
| beside: natural flag AUROC, all fixed rows (as exp36_devcheck.py) | | 0.620 (0.532, 0.707) | descriptive |
| Claim 2: risk(C3) − risk(C0) at C0's coverage | 100 | +0.014 (-0.045, +0.072) | not tested (Claim 1 not SHOWN) |
| Claim 3: risk(C3) − risk(C4) at C0's coverage | 100 | +0.043 (-0.022, +0.108) | not tested (Claim 1 not SHOWN) |
| beside: candidate rows with no C4 score | 0 | | |
| beside: C4 AUROC (m) | | 0.555 (0.432, 0.680) | descriptive |

Claim 4 (descriptive): τ fitted on folds 0-1 at r* = 0.3: s ≥ 0.513; C4: m ≥ 10.995 (in-sample on these folds).

| policy | emitted | coverage (95% CI) | risk (95% CI) | wrong added vs C0 | cached refusals recovered correct | cached refusals turned wrong |
|---|---|---|---|---|---|---|
| C0 natural | 70 | 0.700 (0.608, 0.788) | 0.414 (0.299, 0.532) | +0 | 0 | 0 |
| C1 forced (all candidates) | 84 | 0.840 (0.765, 0.908) | 0.405 (0.300, 0.511) | +5 | 9 | 5 |
| C2 self-check (argmax A) | 34 | 0.340 (0.248, 0.434) | 0.294 (0.147, 0.455) | -19 | 3 | 0 |
| C3 at C0's coverage | 70 | 0.700 (0.606, 0.787) | 0.429 (0.313, 0.545) | +1 | 5 | 4 |
| C4 at C0's coverage | 70 | 0.700 (0.610, 0.788) | 0.386 (0.271, 0.500) | -2 | 4 | 1 |
| C4′ (§34b unknown rank) at C0's coverage | 70 | 0.700 (0.609, 0.788) | 0.400 (0.286, 0.514) | -1 | 3 | 3 |
| Claim 4: C3 at τ | 37 | 0.370 (0.277, 0.465) | 0.297 (0.154, 0.452) | -18 | 4 | 0 |
| Claim 4: C4 at τ | 12 | 0.120 (0.059, 0.186) | 0.250 (0.000, 0.533) | -26 | 0 | 0 |

C2 − C0 risk: -0.120 (-0.250, +0.006). C4′ AUROC 0.424 (0.302, 0.548) (0 rows without a §34b read rank lowest).

Robustness (never selected; V stays primary):

| wording | AUROC (95% CI) | C2 coverage | C2 risk |
|---|---|---|---|
| Vn | 0.614 (0.482, 0.742) | 0.690 (0.598, 0.779) | 0.377 (0.262, 0.492) |
| P2p | 0.561 (0.435, 0.683) | 0.330 (0.240, 0.424) | 0.303 (0.152, 0.467) |
| P3p | 0.606 (0.483, 0.728) | 0.340 (0.248, 0.433) | 0.265 (0.125, 0.419) |

## Evidence dependence (controls; both readers verify the same 3B candidates)

| control | 3B verifier | 7B verifier (descriptive) |
|---|---|---|
| A: s(intact) − s(cut), B judged correct | +0.249 (+0.170, +0.334) (n 50) | +0.469 (+0.351, +0.589) (n 50) |
| A′: AUROC intact correct-B vs Cut-F | 0.681 (0.588, 0.770) (50 / 60) | 0.730 (0.642, 0.812) (50 / 60) |
| Length control: s(intact) − s(LC) | +0.269 (+0.191, +0.354) (n 50) | +0.480 (+0.362, +0.599) (n 50) |
| Length control: s(LC) − s(cut) | -0.021 (-0.062, +0.014) | -0.011 (-0.034, +0.006) |
| B (primary): number substitutions, no other copy in memory; follows in both contexts | 1.000 (1.000, 1.000) (n 6; primary row; descriptive (fewer than 20 rows)) | 0.833 (0.429, 1.000) (n 6; primary row; descriptive (fewer than 20 rows)) |
| B (primary): follows in the counterfactual context | 1.000 (1.000, 1.000) | 1.000 (1.000, 1.000) |
| B (primary): follows in the intact context | 1.000 (1.000, 1.000) | 0.833 (0.429, 1.000) |
| B, DESCRIPTIVE ONLY: numbers with other copies in memory, follows in both | 1.000 (1.000, 1.000) (n 2) | 0.000 (0.000, 0.000) (n 2) |
| B, DESCRIPTIVE ONLY: name substitutions (not internally consistent), follows in both | 0.462 (0.182, 0.750) (n 13) | 0.462 (0.182, 0.750) (n 13) |
| B, DESCRIPTIVE ONLY: all counterfactual rows, follows in both | 0.667 (0.455, 0.867) (n 21) | 0.524 (0.296, 0.739) (n 21) |
| Cut-F emitted at τ (unsupported-answer rate) | 0.250 | 0.367 |
| Cut-F argmax A | 0.183 | 0.350 |
| Abs: n, emitted at τ, argmax A | 3, 0.000, 0.000 | 3, 0.667, 0.667 |

7B − 3B difference of the Control A drop (paired): +0.220 (+0.128, +0.318) (n 50).

Cut-F split by the §35 judge verdict on the Fcut answer (judged-correct answers are 'unsupported' by construction):

| verifier | Fcut judged | n | mean s | emitted at τ | argmax A |
|---|---|---|---|---|---|
| qwen3b | correct | 16 | 0.732 | 9 | 8 |
| qwen3b | wrong | 44 | 0.496 | 6 | 3 |
| qwen7b | correct | 16 | 0.782 | 12 | 11 |
| qwen7b | wrong | 44 | 0.314 | 10 | 10 |

## 7B (descriptive; the judge shares the 7B's weights)

The 7B verifying the 3B's candidates (labels: the 3B's B verdicts):

AUROC 0.726 (0.611, 0.829); at C0's coverage risk 0.329 (0.219, 0.439), minus C0 -0.086 (-0.159, -0.017); argmax-A coverage 0.450 (0.353, 0.550), risk 0.222 (0.105, 0.348); at the 3B's dev τ coverage 0.480 (0.382, 0.577), risk 0.271 (0.151, 0.400).

The 7B on its own B candidates (V_own; labels: the 7B's B verdicts): fixed rows 126, no candidate 5; AUROC 0.716 (0.615, 0.808); natural flag AUROC 0.513 (0.479, 0.555); C0 coverage 0.929 (0.881, 0.969), risk 0.316 (0.233, 0.402); V_own at C0's coverage risk 0.316 (0.234, 0.400) (minus C0 +0.000 (-0.026, +0.025)); C4 at C0's coverage risk 0.316 (0.233, 0.402) (V_own − C4 +0.000 (-0.026, +0.025)). Never merged with the comparison above into one size conclusion.

## Cost per verification

| verifier/item | rows | tokens per row | seconds per row |
|---|---|---|---|
| qwen3b/V | 84 | 1027 | 0.0609 |
| qwen3b/Vn | 84 | 1082 | 0.0585 |
| qwen3b/CutB | 84 | 852 | 0.0416 |
| qwen3b/CutF | 60 | 892 | 0.0214 |
| qwen3b/Abs | 3 | 1098 | 0.0774 |
| qwen3b/CF | 84 | 1041 | 0.0307 |
| qwen3b/LC | 84 | 1026 | 0.0537 |
| qwen3b/C4 | 84 | 2309 | 0.2064 |
| qwen3b/P2p | 84 | 1027 | 0.0115 |
| qwen3b/P3p | 84 | 1030 | 0.0081 |
| qwen7b/V | 84 | 1027 | 0.1215 |
| qwen7b/Vn | 84 | 1082 | 0.1052 |
| qwen7b/CutB | 84 | 852 | 0.0821 |
| qwen7b/CutF | 60 | 892 | 0.0402 |
| qwen7b/Abs | 3 | 1098 | 0.1574 |
| qwen7b/CF | 84 | 1041 | 0.0517 |
| qwen7b/LC | 84 | 1026 | 0.1068 |
| qwen7b/V_own | 121 | 1088 | 0.1226 |
| qwen7b/C4 | 121 | 2395 | 0.3912 |
| qwen7b/P2p | 84 | 1027 | 0.1314 |
| qwen7b/P3p | 84 | 1030 | 0.0122 |

Verdict rules (pre-registered, 3B only): Claim 1 SHOWN if the AUROC lower bound > 0.5 (meaningful ≥ 0.65); Claim 2 tested only if Claim 1 SHOWN, SHOWN if the upper bound of risk(C3) − risk(C0) < 0; Claim 3 tested only if Claim 2 SHOWN, SHOWN if the upper bound of risk(C3) − risk(C4) < 0 (meaningful ≤ −0.05); Claim 4 and every control descriptive. C4 for the 7B is on its own candidates and own note.
