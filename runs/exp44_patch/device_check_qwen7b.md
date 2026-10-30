# §44 rev 8: 7B device consistency (laptop CPU bf16 vs pod CUDA bf16)

Laptop rows runs/exp44_patch/qwen7b (73 on folds 2-4), GPU rows runs/exp44_patch/qwen7b_gpu (204); common 73, both patched 36, patched on one side only 1.

- d_c equal on 34 / 36 rows; unequal: [{'question_id': '1de5cff2', 'diff_layers': -1, 'one_none': False}, {'question_id': 'a40e080f', 'diff_layers': 2, 'one_none': False}]
- max |R difference|: overall 0.1417, median over rows 0.04238
- eligibility flips (rev 5/7 filter, re-derived): ['91b15a6e']
- gap-sign flips: ['18bc8abd', '184da446']; 1-nat gap flips: ['bcbe585f']; R(0)-flag flips: ['16c90bf4']
- rank changes: clean 10, cut 26; max |lp diff| clean 1.375, cut 1.062, gap 1.125

| row | fold | eligible (laptop, gpu) | patched | d_c equal | max abs R diff | abs gap diff |
|---|---|---|---|---|---|---|
| 0100672e | 2 | [False, False] | [False, False] | — | — | 0.009355 |
| 07741c45 | 2 | [False, False] | [False, False] | — | — | 0.3973 |
| 08e075c7 | 2 | [True, True] | [True, True] | True | 0.06642 | 0.1205 |
| 10d9b85a | 2 | [False, False] | [False, False] | — | — | 0.244 |
| 16c90bf4 | 2 | [True, True] | [True, True] | True | 0.1279 | 0.1242 |
| 18bc8abd | 2 | [False, False] | [False, False] | — | — | 7.774e-06 |
| 1d4da289 | 2 | [False, False] | [False, False] | — | — | 0.1707 |
| 1de5cff2 | 2 | [True, True] | [True, True] | False | 0.04566 | 0.5926 |
| 1faac195 | 2 | [True, True] | [True, True] | True | 0.02214 | 0.3976 |
| 2311e44b | 2 | [True, True] | [True, True] | True | 0.1166 | 0.01548 |
| 2318644b | 2 | [True, True] | [True, True] | True | 0.01915 | 0.2267 |
| 3d86fd0a | 2 | [False, False] | [False, False] | — | — | 0.2382 |
| 3e321797 | 2 | [False, False] | [False, False] | — | — | 0.0002286 |
| 45dc21b6 | 2 | [True, True] | [True, True] | True | 0.04297 | 0.1339 |
| 4b24c848 | 2 | [True, True] | [True, True] | True | 0.01881 | 0.1844 |
| 4baee567 | 2 | [False, False] | [False, False] | — | — | 0.25 |
| 4fd1909e | 2 | [False, False] | [False, False] | — | — | 7.839e-06 |
| 51a45a95 | 2 | [False, False] | [False, False] | — | — | 0.3181 |
| 5a7937c8 | 2 | [True, True] | [True, True] | True | 0.06222 | 0.1882 |
| 5c40ec5b | 2 | [False, False] | [False, False] | — | — | 0.01155 |
| 60036106 | 2 | [True, True] | [True, True] | True | 0.05862 | 0.2726 |
| 6222b6eb | 2 | [False, False] | [False, False] | — | — | 0.1366 |
| 6aeb4375 | 2 | [True, True] | [True, True] | True | 0.01995 | 0.05461 |
| 6b168ec8 | 2 | [False, False] | [False, False] | — | — | 0.02334 |
| 6c49646a | 2 | [True, True] | [True, True] | True | 0.1417 | 0.4688 |
| 6e984301 | 2 | [True, True] | [True, True] | True | 0.05131 | 0.01884 |
| 6f9b354f | 2 | [False, False] | [False, False] | — | — | 0.01099 |
| 70b3e69b | 2 | [True, True] | [True, True] | True | 0.01592 | 0.1826 |
| 71315a70 | 2 | [False, False] | [False, False] | — | — | 0.000564 |
| 72e3ee87 | 2 | [False, False] | [False, False] | — | — | 0.003989 |
| 7405e8b1 | 2 | [False, False] | [False, False] | — | — | 1.125 |
| 75f70248 | 2 | [True, True] | [True, True] | True | 0.06619 | 0.125 |
| 853b0a1d | 2 | [False, False] | [False, False] | — | — | 3.349e-08 |
| 86b68151 | 2 | [True, True] | [True, True] | True | 0.01747 | 0.06251 |
| 87f22b4a | 2 | [True, True] | [True, True] | True | 0.02242 | 0.2486 |
| 8e9d538c | 2 | [True, True] | [True, True] | True | 0.1091 | 0.1895 |
| 91b15a6e | 2 | [True, False] | [True, False] | — | — | 0.6651 |
| 9a707b81 | 2 | [True, True] | [True, True] | True | 0.1009 | 0.02482 |
| 9aaed6a3 | 2 | [False, False] | [False, False] | — | — | 0.0005558 |
| a1eacc2a | 2 | [True, True] | [True, True] | True | 0.01818 | 0.06349 |
| a3045048 | 2 | [True, True] | [True, True] | True | 0.0448 | 0.4053 |
| a40e080f | 2 | [True, True] | [True, True] | False | 0.02017 | 0.1237 |
| a89d7624 | 2 | [False, False] | [False, False] | — | — | 0.6089 |
| aae3761f | 2 | [False, False] | [False, False] | — | — | 0.001638 |
| ad7109d1 | 2 | [True, True] | [True, True] | True | 0.02773 | 0.001913 |
| bc8a6e93 | 2 | [True, True] | [True, True] | True | 0.02055 | 0.5282 |
| bcbe585f | 2 | [False, False] | [False, False] | — | — | 0.2989 |
| c960da58 | 2 | [False, False] | [False, False] | — | — | 0.04223 |
| ceb54acb | 2 | [False, False] | [False, False] | — | — | 0.05198 |
| cf22b7bf | 2 | [False, False] | [False, False] | — | — | 0.125 |
| e9327a54 | 2 | [False, False] | [False, False] | — | — | 0.2138 |
| ed4ddc30 | 2 | [True, True] | [True, True] | True | 0.05083 | 0.1864 |
| ef66a6e5 | 2 | [True, True] | [True, True] | True | 0.01467 | 0.1654 |
| f685340e | 2 | [True, True] | [True, True] | True | 0.06287 | 0.06384 |
| gpt4_0a05b494 | 2 | [True, True] | [True, True] | True | 0.03442 | 0.9876 |
| gpt4_0b2f1d21 | 2 | [True, True] | [True, True] | True | 0.0769 | 0.02528 |
| gpt4_1916e0ea | 2 | [True, True] | [True, True] | True | 0.04238 | 0.02627 |
| gpt4_2487a7cb | 2 | [True, True] | [True, True] | True | 0.03183 | 0.3814 |
| gpt4_2c50253f | 2 | [True, True] | [True, True] | True | 0.02584 | 0.1648 |
| gpt4_4edbafa2 | 2 | [False, False] | [False, False] | — | — | 0.1179 |
| gpt4_4fc4f797 | 2 | [False, False] | [False, False] | — | — | 0.061 |
| gpt4_68e94287 | 2 | [True, True] | [True, True] | True | 0.03685 | 0.3236 |
| gpt4_93f6379c | 2 | [True, True] | [True, True] | True | 0.02451 | 0.1019 |
| gpt4_c27434e8 | 2 | [False, False] | [False, False] | — | — | 0.03612 |
| gpt4_fa19884c | 2 | [False, False] | [False, False] | — | — | 0.04086 |
| gpt4_fa19884d | 2 | [False, False] | [False, False] | — | — | 0.4057 |
| 099778bb | 3 | [True, True] | [True, True] | True | 0.04304 | 0.2369 |
| 10e09553 | 3 | [False, False] | [False, False] | — | — | 0.125 |
| 1192316e | 3 | [False, False] | [False, False] | — | — | 0.1178 |
| 184da446 | 3 | [False, False] | [False, False] | — | — | 1.278e-06 |
| 18dcd5a5 | 3 | [True, True] | [True, True] | True | 0.03952 | 0.06955 |
| 1f2b8d4f | 3 | [False, False] | [False, False] | — | — | 0.02651 |
| 27016adc | 3 | [False, False] | [False, False] | — | — | 0.03044 |
