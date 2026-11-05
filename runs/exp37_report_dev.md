# §37 report: the elenchus (folds 0, 1)

Engines ['hf-transformers-5.17.0-cpu-float32']; wording sha256 7ef8e0336b7f. Fixed-candidate rows 100, with a candidate 84, candidate rows without an e score 0. 95% intervals: paired bootstrap over questions, 10,000 resamples.

## §5 development gate (folds 0-1)

AUROC(e, dev) = 0.505; AUROC(s, dev) = 0.559; AUROC(s) + 0.05 = 0.609. Bar 1: 0.505 >= 0.65: False. Bar 2: 0.505 >= 0.609: False. Gate: FAIL (§37 closes, development only).

## Compliance (§5.1: E1 parses ≥ 0.9; E3 mass ≥ 0.5 on ≥ 0.9 of rows)

| reader/item | E1 rows | E1 parses | E3 rows | E3 mass ≥ 0.5 | reliable |
|---|---|---|---|---|---|
| qwen3b/V | 84 | 1.000 | 10 | 1.000 | yes |
| qwen3b/CutB | 84 | 0.988 | 5 | 1.000 | yes |
| qwen3b/CutF | 60 | 0.983 | 3 | 1.000 | yes |
| qwen3b/Pnum | 14 | 1.000 | 1 | 1.000 | yes |
| qwen3b/Pname | 13 | 1.000 | 0 | - | no |
| qwen3b/Vnum | 0 | - | 84 | 1.000 | yes |
| qwen3b/Vdlg | 0 | - | 6 | 1.000 | yes |

## 3B claims (fixed sequence) (development folds: not a confirmation verdict)

| claim | n | estimate (95% CI) | verdict |
|---|---|---|---|
| Claim 1: AUROC(e) | 50 / 34 | 0.505 (0.431, 0.574) | NOT SHOWN (lower bound ≤ 0.5) |
| beside: PR-AUC, prevalence | | 0.620, 0.595 | |
| beside: AUROC(e_mech) | | 0.521 (0.460, 0.578) | descriptive |
| beside: AUROC(g) | | 0.576 (0.481, 0.675) | descriptive |
| Claim 2: AUROC(e) − AUROC(s) (paired) | 84 | -0.055 (-0.181, +0.071) | not tested (Claim 1 not SHOWN) |
| beside: AUROC(s), §36b one-shot | | 0.559 (0.433, 0.681) | fixed baseline |
| beside: AUROC(V-num) | | 0.548 (0.422, 0.675) | descriptive |
| Claim 3: risk(C3e) − risk(C0) at C0's coverage | 100 | -0.057 (-0.121, +0.004) | not tested (Claim 1 not SHOWN) |

| policy | emitted | coverage | risk | wrong added vs C0 |
|---|---|---|---|---|
| C0 | 70 | 0.700 (0.608, 0.787) | 0.414 (0.300, 0.531) | +0 |
| C1 | 84 | 0.840 (0.765, 0.908) | 0.405 (0.302, 0.511) | +5 |
| C2e | 7 | 0.070 (0.022, 0.124) | 0.286 (0.000, 0.700) | -27 |
| C3e | 70 | 0.700 (0.610, 0.787) | 0.357 (0.250, 0.472) | -4 |
| C3s | 70 | 0.700 (0.608, 0.788) | 0.429 (0.313, 0.548) | +1 |

Claim 4 (repair, descriptive): no judge rows on d (laptop run): not available

## Controls (descriptive)

| control | 3B | 7B on the 3B's candidates |
|---|---|---|
| AUROC(e) | 0.505 (0.431, 0.574) | - |
| A: g intact / cut | 0.738 (0.639, 0.829) / 0.262 (0.171, 0.359) | - |
| A: g drop (paired) | +0.476 (+0.369, +0.582) | - |
| A: e(intact) − e(cut), judged correct | +0.038 (-0.055, +0.135) | - |
| Cut-F pass rate (e ≥ 0.5) | 0.050 (0.000, 0.113) | - |
| G: cited set hits the evidence, intact / cut | 0.560 (0.455, 0.667) / 0.000 (0.000, 0.000) | - |
| P Pnum: rejects planted | 1.000 (1.000, 1.000) (n 7) | - |
| P Pnum: accepts true | 0.143 (0.000, 0.500) (n 7) | - |
| P Pnum: follows (both) | 0.143 (0.000, 0.500) (n 7) | - |
| P Pname: rejects planted | 1.000 (1.000, 1.000) (n 9) | - |
| P Pname: accepts true | 0.333 (0.000, 0.667) (n 9) | - |
| P Pname: follows (both) | 0.333 (0.000, 0.667) (n 9) | - |

S (sycophancy): share with d = the candidate (e_mech), blind vs in dialogue:

| rows | n | blind | dialogue |
|---|---|---|---|
| judged wrong | 34 | 0.059 (0.000, 0.150) | 0.000 (0.000, 0.000) |
| judged correct | 50 | 0.100 (0.021, 0.190) | 0.100 (0.021, 0.192) |

AUROC(e′) in dialogue (never primary): 0.537 (0.486, 0.589).

## Costs per candidate

| reader/item | questions | calls | input tokens | seconds |
|---|---|---|---|---|
| qwen3b/V | 84 | 1.86 | 1231 | 26.42 |
| qwen3b/CutB | 84 | 1.32 | 890 | 18.05 |
| qwen3b/CutF | 60 | 1.47 | 966 | 20.59 |
| qwen3b/Pnum | 14 | 1.64 | 1134 | 24.93 |
| qwen3b/Pname | 13 | 1.00 | 1013 | 19.73 |
| qwen3b/Vnum | 84 | 1.00 | 1051 | 19.21 |
| qwen3b/Vdlg | 62 | 1.10 | 1050 | 22.23 |
