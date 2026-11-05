# §36 report: can the 3B judge support for its own forced answers? (folds 2, 3, 4)

Verifier wording: §36b frame a. Engines: ['vllm-0.8.5']. Wording sha256 3089973f27ae. Labels: the §35 judge (Qwen2.5-7B) on arm B. 95% intervals: paired bootstrap over questions (all variants of a question in one unit), 10,000 resamples.

## Rows

Fixed-candidate rows (3B): 165; with a candidate 140; no candidate (B still says unknown; forced abstentions, kept in every denominator) 25; candidate rows missing a V score 0.

Label-mass rule (mass < 0.5 on more than 0.1 of rows = not reliable, argmax only):

| verifier/item | rows | low-mass rows | share | reliable |
|---|---|---|---|---|
| qwen3b/V | 140 | 0 | 0.000 | yes |
| qwen3b/Vn | 140 | 0 | 0.000 | yes |
| qwen3b/CutB | 140 | 0 | 0.000 | yes |
| qwen3b/CutF | 120 | 0 | 0.000 | yes |
| qwen3b/Abs | 4 | 0 | 0.000 | yes |
| qwen3b/CF | 132 | 0 | 0.000 | yes |
| qwen3b/LC | 140 | 0 | 0.000 | yes |
| qwen3b/P2p | 140 | 0 | 0.000 | yes |
| qwen3b/P3p | 140 | 0 | 0.000 | yes |
| qwen7b/V | 140 | 0 | 0.000 | yes |
| qwen7b/Vn | 140 | 0 | 0.000 | yes |
| qwen7b/CutB | 140 | 0 | 0.000 | yes |
| qwen7b/CutF | 120 | 0 | 0.000 | yes |
| qwen7b/Abs | 4 | 0 | 0.000 | yes |
| qwen7b/CF | 132 | 0 | 0.000 | yes |
| qwen7b/LC | 140 | 0 | 0.000 | yes |
| qwen7b/V_own | 201 | 0 | 0.000 | yes |
| qwen7b/P2p | 140 | 0 | 0.000 | yes |
| qwen7b/P3p | 140 | 0 | 0.000 | yes |

V readout diagnostics (top-1 token from the stored top-20 logprobs):

| verifier | V rows | share label mass < 0.5 | share top-1 'unknown' | top-1 tokens on low-mass rows (count) |
|---|---|---|---|---|
| qwen3b | 140 | 0.000 | 0.000 | - |
| qwen7b | 140 | 0.000 | 0.000 | - |

## 3B claims (primary; fixed sequence)

| claim | n | estimate (95% CI) | verdict |
|---|---|---|---|
| Claim 1: AUROC of s (V) for B judged correct, candidate rows | 84 pos / 56 neg | 0.573 (0.478, 0.666) | NOT SHOWN (lower bound ≤ 0.5) |
| beside: PR-AUC, prevalence (candidate rows) | | 0.698, 0.600 | |
| beside: AUROC over all fixed rows, no-candidate rows lowest | 84 / 81 | 0.705 (0.623, 0.781) | descriptive (inflated by the forced abstentions) |
| beside: natural flag (C0) AUROC, candidate rows | | 0.455 (0.381, 0.532) | descriptive |
| beside: natural flag AUROC, all fixed rows (as exp36_devcheck.py) | | 0.555 (0.481, 0.629) | descriptive |
| Claim 2: risk(C3) − risk(C0) at C0's coverage | 165 | -0.060 (-0.132, +0.012) | not tested (Claim 1 not SHOWN) |
| Claim 3: risk(C3) − risk(C4) at C0's coverage | 165 | -0.020 (-0.090, +0.049) | not tested (Claim 1 not SHOWN) |
| beside: candidate rows with no C4 score | 0 | | |
| beside: C4 AUROC (m) | | 0.502 (0.398, 0.603) | descriptive |

Claim 4 (descriptive): τ fitted on folds 0-1 at r* = 0.3: s ≥ 0.513; C4: m ≥ 10.995.

| policy | emitted | coverage (95% CI) | risk (95% CI) | wrong added vs C0 | cached refusals recovered correct | cached refusals turned wrong |
|---|---|---|---|---|---|---|
| C0 natural | 100 | 0.606 (0.530, 0.681) | 0.430 (0.333, 0.529) | +0 | 0 | 0 |
| C1 forced (all candidates) | 140 | 0.848 (0.792, 0.902) | 0.400 (0.320, 0.481) | +13 | 27 | 13 |
| C2 self-check (argmax A) | 47 | 0.285 (0.218, 0.354) | 0.319 (0.191, 0.457) | -28 | 8 | 3 |
| C3 at C0's coverage | 100 | 0.606 (0.531, 0.679) | 0.370 (0.275, 0.468) | -6 | 17 | 9 |
| C4 at C0's coverage | 100 | 0.606 (0.531, 0.680) | 0.390 (0.294, 0.485) | -4 | 8 | 1 |
| C4′ (§34b unknown rank) at C0's coverage | 100 | 0.606 (0.532, 0.682) | 0.370 (0.276, 0.466) | -6 | 10 | 4 |
| Claim 4: C3 at τ | 56 | 0.339 (0.269, 0.412) | 0.357 (0.233, 0.484) | -23 | 9 | 4 |
| Claim 4: C4 at τ | 8 | 0.048 (0.018, 0.084) | 0.500 (0.125, 0.875) | -39 | 0 | 0 |

C2 − C0 risk: -0.111 (-0.229, +0.011). C4′ AUROC 0.502 (0.404, 0.603) (0 rows without a §34b read rank lowest).

Robustness (never selected; V stays primary):

| wording | AUROC (95% CI) | C2 coverage | C2 risk |
|---|---|---|---|
| Vn | 0.570 (0.472, 0.664) | 0.582 (0.506, 0.656) | 0.385 (0.287, 0.485) |
| P2p | 0.579 (0.483, 0.671) | 0.285 (0.218, 0.354) | 0.319 (0.191, 0.457) |
| P3p | 0.571 (0.474, 0.664) | 0.255 (0.190, 0.321) | 0.286 (0.152, 0.429) |

## Evidence dependence (controls; both readers verify the same 3B candidates)

| control | 3B verifier | 7B verifier (descriptive) |
|---|---|---|
| A: s(intact) − s(cut), B judged correct | +0.235 (+0.170, +0.302) (n 84) | +0.508 (+0.419, +0.596) (n 84) |
| A′: AUROC intact correct-B vs Cut-F | 0.715 (0.652, 0.774) (84 / 120) | 0.760 (0.703, 0.816) (84 / 120) |
| Length control: s(intact) − s(LC) | +0.238 (+0.174, +0.306) (n 84) | +0.516 (+0.426, +0.606) (n 84) |
| Length control: s(LC) − s(cut) | -0.003 (-0.039, +0.034) | -0.009 (-0.028, +0.008) |
| B (primary): number substitutions, no other copy in memory; follows in both contexts | 0.800 (0.500, 1.000) (n 10; primary row; descriptive (fewer than 20 rows)) | 0.800 (0.500, 1.000) (n 10; primary row; descriptive (fewer than 20 rows)) |
| B (primary): follows in the counterfactual context | 0.900 (0.667, 1.000) | 0.800 (0.500, 1.000) |
| B (primary): follows in the intact context | 0.900 (0.667, 1.000) | 0.900 (0.667, 1.000) |
| B, DESCRIPTIVE ONLY: numbers with other copies in memory, follows in both | 0.714 (0.333, 1.000) (n 7) | 0.429 (0.000, 0.833) (n 7) |
| B, DESCRIPTIVE ONLY: name substitutions (not internally consistent), follows in both | 0.750 (0.500, 0.941) (n 16) | 0.688 (0.438, 0.917) (n 16) |
| B, DESCRIPTIVE ONLY: all counterfactual rows, follows in both | 0.758 (0.607, 0.897) (n 33) | 0.667 (0.500, 0.821) (n 33) |
| Cut-F emitted at τ (unsupported-answer rate) | 0.183 | 0.317 |
| Cut-F argmax A | 0.150 | 0.258 |
| Abs: n, emitted at τ, argmax A | 4, 0.000, 0.000 | 4, 0.250, 0.500 |

7B − 3B difference of the Control A drop (paired): +0.273 (+0.187, +0.359) (n 84).

Cut-F split by the §35 judge verdict on the Fcut answer (judged-correct answers are 'unsupported' by construction):

| verifier | Fcut judged | n | mean s | emitted at τ | argmax A |
|---|---|---|---|---|---|
| qwen3b | correct | 28 | 0.655 | 10 | 10 |
| qwen3b | wrong | 92 | 0.496 | 12 | 8 |
| qwen7b | correct | 28 | 0.614 | 17 | 15 |
| qwen7b | wrong | 92 | 0.273 | 21 | 16 |

## 7B (descriptive; the judge shares the 7B's weights)

The 7B verifying the 3B's candidates (labels: the 3B's B verdicts):

AUROC 0.692 (0.600, 0.777); at C0's coverage risk 0.340 (0.248, 0.432), minus C0 -0.090 (-0.171, -0.010); argmax-A coverage 0.442 (0.366, 0.518), risk 0.274 (0.174, 0.381); at the 3B's dev τ coverage 0.491 (0.415, 0.566), risk 0.272 (0.176, 0.372).

The 7B on its own B candidates (V_own; labels: the 7B's B verdicts): fixed rows 204, no candidate 3; AUROC 0.731 (0.651, 0.805); natural flag AUROC 0.541 (0.507, 0.583); C0 coverage 0.956 (0.926, 0.981), risk 0.262 (0.201, 0.324); V_own at C0's coverage risk 0.267 (0.206, 0.328) (minus C0 +0.005 (-0.017, +0.028)); C4 at C0's coverage risk 0.262 (0.200, 0.324) (V_own − C4 +0.005 (-0.017, +0.027)). Never merged with the comparison above into one size conclusion.

## Cost per verification

| verifier/item | rows | tokens per row | seconds per row |
|---|---|---|---|
| qwen3b/V | 140 | 1104 | 0.0689 |
| qwen3b/Vn | 140 | 1156 | 0.0655 |
| qwen3b/CutB | 140 | 914 | 0.0448 |
| qwen3b/CutF | 120 | 937 | 0.0207 |
| qwen3b/Abs | 4 | 1061 | 0.0757 |
| qwen3b/CF | 132 | 1166 | 0.0324 |
| qwen3b/LC | 140 | 1089 | 0.0575 |
| qwen3b/C4 | 140 | 2457 | 0.2203 |
| qwen3b/P2p | 140 | 1104 | 0.0118 |
| qwen3b/P3p | 140 | 1107 | 0.0077 |
| qwen7b/V | 140 | 1104 | 0.1368 |
| qwen7b/Vn | 140 | 1156 | 0.1272 |
| qwen7b/CutB | 140 | 914 | 0.0893 |
| qwen7b/CutF | 120 | 937 | 0.0391 |
| qwen7b/Abs | 4 | 1061 | 0.1486 |
| qwen7b/CF | 132 | 1166 | 0.0598 |
| qwen7b/LC | 140 | 1089 | 0.1135 |
| qwen7b/V_own | 201 | 1121 | 0.1372 |
| qwen7b/C4 | 201 | 2464 | 0.4017 |
| qwen7b/P2p | 140 | 1104 | 0.1417 |
| qwen7b/P3p | 140 | 1107 | 0.0114 |

Verdict rules (pre-registered, 3B only): Claim 1 SHOWN if the AUROC lower bound > 0.5 (meaningful ≥ 0.65); Claim 2 tested only if Claim 1 SHOWN, SHOWN if the upper bound of risk(C3) − risk(C0) < 0; Claim 3 tested only if Claim 2 SHOWN, SHOWN if the upper bound of risk(C3) − risk(C4) < 0 (meaningful ≤ −0.05); Claim 4 and every control descriptive. C4 for the 7B is on its own candidates and own note.
