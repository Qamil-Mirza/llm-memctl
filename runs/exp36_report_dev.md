# §36 report: can the 3B judge support for its own forced answers? (folds 0, 1)

Engines: ['vllm-0.8.5']. Wording sha256 653f03b22b1f. Labels: the §35 judge (Qwen2.5-7B) on arm B. 95% intervals: paired bootstrap over questions (all variants of a question in one unit), 10,000 resamples.

## Rows

Fixed-candidate rows (3B): 100; with a candidate 84; no candidate (B still says unknown; forced abstentions, kept in every denominator) 16; candidate rows missing a V score 70.

Label-mass rule (mass < 0.5 on more than 0.1 of rows = not reliable, argmax only):

| verifier/item | rows | low-mass rows | share | reliable |
|---|---|---|---|---|
| qwen3b/V | 84 | 84 | 1.000 | **no** |
| qwen3b/Vn | 84 | 84 | 1.000 | **no** |
| qwen3b/P2 | 84 | 84 | 1.000 | **no** |
| qwen3b/P3 | 84 | 84 | 1.000 | **no** |
| qwen3b/CutB | 84 | 84 | 1.000 | **no** |
| qwen3b/CutF | 60 | 60 | 1.000 | **no** |
| qwen3b/Abs | 3 | 3 | 1.000 | **no** |
| qwen3b/CF | 84 | 84 | 1.000 | **no** |
| qwen3b/LC | 84 | 84 | 1.000 | **no** |
| qwen7b/V | 84 | 84 | 1.000 | **no** |
| qwen7b/Vn | 84 | 84 | 1.000 | **no** |
| qwen7b/P2 | 84 | 84 | 1.000 | **no** |
| qwen7b/P3 | 84 | 84 | 1.000 | **no** |
| qwen7b/CutB | 84 | 84 | 1.000 | **no** |
| qwen7b/CutF | 60 | 60 | 1.000 | **no** |
| qwen7b/Abs | 3 | 3 | 1.000 | **no** |
| qwen7b/CF | 84 | 84 | 1.000 | **no** |
| qwen7b/LC | 84 | 84 | 1.000 | **no** |
| qwen7b/V_own | 121 | 121 | 1.000 | **no** |

V readout diagnostics (top-1 token from the stored top-20 logprobs):

| verifier | V rows | share label mass < 0.5 | share top-1 'unknown' | top-1 tokens on low-mass rows (count) |
|---|---|---|---|---|
| qwen3b | 84 | 1.000 | 0.000 | 'ĠA' (16), '1' (11), '2' (10), '5' (9), '3' (8), '4' (7), 'æľªçŁ¥' (5), 'Ġunknown' (4), '6' (3), '7' (2) |
| qwen7b | 84 | 1.000 | 0.000 | 'ĠA' (22), 'ĠC' (15), 'ĠAnswer' (8), 'æľªçŁ¥' (7), '3' (6), '5' (6), 'Ġunknown' (6), '2' (4), '4' (4), '<|im_end|>' (2) |

## 3B claims (primary; fixed sequence) (development folds: not a confirmation verdict)

Claims 1-4: **NOT RELIABLE (label mass rule): argmax only, see C2**.

| policy | emitted | coverage (95% CI) | risk (95% CI) | wrong added vs C0 | cached refusals recovered correct | cached refusals turned wrong |
|---|---|---|---|---|---|---|
| C0 natural | 70 | 0.700 (0.608, 0.788) | 0.414 (0.299, 0.532) | +0 | 0 | 0 |
| C1 forced (all candidates) | 84 | 0.840 (0.765, 0.908) | 0.405 (0.300, 0.511) | +5 | 9 | 5 |
| C2 self-check (argmax A) | 4 | 0.040 (0.010, 0.082) | 0.250 (0.000, 1.000) | -28 | 0 | 0 |

Robustness (never selected; V stays primary):

| wording | AUROC (95% CI) | C2 coverage | C2 risk |
|---|---|---|---|
| Vn | not reliable | 0.060 (0.019, 0.111) | 0.500 (0.000, 1.000) |
| P2 | not reliable | 0.030 (0.000, 0.068) | 0.333 (0.000, 1.000) |
| P3 | not reliable | 0.010 (0.000, 0.032) | 1.000 (1.000, 1.000) |

## Evidence dependence (controls; both readers verify the same 3B candidates)

| control | 3B verifier | 7B verifier (descriptive) |
|---|---|---|
| A: s(intact) − s(cut), B judged correct | +0.100 (+0.000, +0.333) (n 5) | - (n 0) |
| A′: AUROC intact correct-B vs Cut-F | 0.718 (0.528, 0.902) (7 / 20) | - (1 / 0) |
| Length control: s(intact) − s(LC) | +0.083 (+0.000, +0.250) (n 6) | - (n 0) |
| Length control: s(LC) − s(cut) | +0.004 (-0.001, +0.013) | - |
| B (primary): number substitutions, no other copy in memory; follows in both contexts | - (n 0; primary row; descriptive (fewer than 20 rows)) | - (n 0; primary row; descriptive (fewer than 20 rows)) |
| B (primary): follows in the counterfactual context | - | - |
| B (primary): follows in the intact context | - | - |
| B, DESCRIPTIVE ONLY: numbers with other copies in memory, follows in both | - (n 0) | - (n 0) |
| B, DESCRIPTIVE ONLY: name substitutions (not internally consistent), follows in both | - (n 0) | - (n 0) |
| B, DESCRIPTIVE ONLY: all counterfactual rows, follows in both | - (n 0) | - (n 0) |
| Cut-F emitted at τ (unsupported-answer rate) | 0.050 | - |
| Cut-F argmax A | 0.017 | 0.000 |
| Abs: n, emitted at τ, argmax A | 3, 0.000, 0.000 | 3, 0.000, 0.000 |

7B − 3B difference of the Control A drop (paired): - (n 0).

Cut-F split by the §35 judge verdict on the Fcut answer (judged-correct answers are 'unsupported' by construction):

| verifier | Fcut judged | n | mean s | emitted at τ | argmax A |
|---|---|---|---|---|---|
| qwen3b | correct | 2 | 0.500 | 0 | 0 |
| qwen3b | wrong | 18 | 0.513 | 1 | 1 |
| qwen7b | correct | 0 | - | 0 | 0 |
| qwen7b | wrong | 0 | - | 0 | 0 |

## 7B (descriptive; the judge shares the 7B's weights)

The 7B verifying the 3B's candidates (labels: the 3B's B verdicts):

AUROC 0.000 (0.000, 0.000); at C0's coverage risk 0.357 (0.247, 0.471), minus C0 -0.057 (-0.119, +0.000); argmax-A coverage 0.010 (0.000, 0.032), risk 1.000 (1.000, 1.000); at the 3B's dev τ coverage 0.010 (0.000, 0.032), risk 1.000 (1.000, 1.000).

The 7B on its own B candidates (V_own; labels: the 7B's B verdicts): fixed rows 126, no candidate 5; AUROC -; natural flag AUROC 0.513 (0.479, 0.555); C0 coverage 0.929 (0.881, 0.969), risk 0.316 (0.233, 0.402); V_own at C0's coverage risk 0.308 (0.224, 0.393) (minus C0 -0.009 (-0.036, +0.018)); C4 at C0's coverage risk 0.316 (0.233, 0.402) (V_own − C4 -0.009 (-0.037, +0.018)). Never merged with the comparison above into one size conclusion.

## Cost per verification

| verifier/item | rows | tokens per row | seconds per row |
|---|---|---|---|
| qwen3b/V | 84 | 1021 | 0.0642 |
| qwen3b/Vn | 84 | 1076 | 0.0588 |
| qwen3b/P2 | 84 | 1030 | 0.0122 |
| qwen3b/P3 | 84 | 1032 | 0.0090 |
| qwen3b/CutB | 84 | 846 | 0.0415 |
| qwen3b/CutF | 60 | 886 | 0.0214 |
| qwen3b/Abs | 3 | 1092 | 0.0747 |
| qwen3b/CF | 84 | 1035 | 0.0305 |
| qwen3b/LC | 84 | 1020 | 0.0536 |
| qwen3b/C4 | 84 | 2309 | 0.2075 |
| qwen7b/V | 84 | 1021 | 0.1292 |
| qwen7b/Vn | 84 | 1076 | 0.1058 |
| qwen7b/P2 | 84 | 1030 | 0.1332 |
| qwen7b/P3 | 84 | 1032 | 0.0153 |
| qwen7b/CutB | 84 | 846 | 0.0827 |
| qwen7b/CutF | 60 | 886 | 0.0400 |
| qwen7b/Abs | 3 | 1092 | 0.1557 |
| qwen7b/CF | 84 | 1035 | 0.0519 |
| qwen7b/LC | 84 | 1020 | 0.1071 |
| qwen7b/V_own | 121 | 1082 | 0.1231 |
| qwen7b/C4 | 121 | 2395 | 0.3944 |

Verdict rules (pre-registered, 3B only): Claim 1 SHOWN if the AUROC lower bound > 0.5 (meaningful ≥ 0.65); Claim 2 tested only if Claim 1 SHOWN, SHOWN if the upper bound of risk(C3) − risk(C0) < 0; Claim 3 tested only if Claim 2 SHOWN, SHOWN if the upper bound of risk(C3) − risk(C4) < 0 (meaningful ≤ −0.05); Claim 4 and every control descriptive. C4 for the 7B is on its own candidates and own note.
