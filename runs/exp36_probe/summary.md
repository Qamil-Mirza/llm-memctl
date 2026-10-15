# §36b wording probe (3B, CPU fp32, development folds 0-1; format compliance only)

518 rows. Gate: label mass ≥ 0.5 on ≥ 0.9 of rows, on V and on CutB (primary spellings only). Run-1 frozen wording on vLLM, rescored: V 0.107, CutB 0.083. Order a, b, c, d, f, g, h, e. No AUROC, risk or correctness-linked number is computed here.

**Pick (first in the order to pass on all 84 rows, every earlier one having failed on all rows): a.** Per §36b (30415c3) the probe stopped running later wordings once (a) was passing; those are tabled as descriptive on the questions already done.

| wording | item | n | share mass ≥ 0.5 | median mass | share top-1 is a label | (tab-inclusive mass ≥ 0.5) | gate (V and CutB) | top-1 on failures (count) | s/prompt | tokens/prompt |
|---|---|---|---|---|---|---|---|---|---|---|
| a | V | 84 | 1.000 | 1.000 | 1.000 | 1.000 | PASS | - | 18.1 | 1027 |
| a | CutB | 84 | 1.000 | 1.000 | 1.000 | 1.000 | PASS | - | 14.8 | 852 |
| b | V | 25 | 1.000 | 1.000 | 1.000 | 1.000 | descriptive (25 of 84 questions) | - | 16.2 | 985 |
| b | CutB | 25 | 1.000 | 1.000 | 1.000 | 1.000 | descriptive (25 of 84 questions) | - | 13.3 | 788 |
| c | V | 25 | 1.000 | 1.000 | 1.000 | 1.000 | descriptive (25 of 84 questions) | - | 16.3 | 981 |
| c | CutB | 25 | 1.000 | 1.000 | 1.000 | 1.000 | descriptive (25 of 84 questions) | - | 13.0 | 784 |
| d | V | 25 | 0.960 | 0.893 | 1.000 | 1.000 | descriptive (25 of 84 questions) | - | 17.0 | 984 |
| d | CutB | 25 | 0.960 | 0.893 | 1.000 | 1.000 | descriptive (25 of 84 questions) | - | 13.7 | 787 |
| f | V | 25 | 1.000 | 1.000 | 1.000 | 1.000 | descriptive (25 of 84 questions) | - | 16.0 | 963 |
| f | CutB | 25 | 1.000 | 1.000 | 1.000 | 1.000 | descriptive (25 of 84 questions) | - | 12.8 | 766 |
| g | V | 25 | 1.000 | 1.000 | 1.000 | 1.000 | descriptive (25 of 84 questions) | - | 17.0 | 1035 |
| g | CutB | 25 | 1.000 | 1.000 | 1.000 | 1.000 | descriptive (25 of 84 questions) | - | 14.0 | 838 |
| h | V | 25 | 1.000 | 1.000 | 1.000 | 1.000 | descriptive (25 of 84 questions) | - | 17.0 | 1033 |
| h | CutB | 25 | 1.000 | 1.000 | 1.000 | 1.000 | descriptive (25 of 84 questions) | - | 13.7 | 836 |
| e | V | 25 | 1.000 | 1.000 | 1.000 | - | descriptive (25 of 84 questions) | - | 17.0 | 981 |
| e | CutB | 25 | 1.000 | 1.000 | 1.000 | - | descriptive (25 of 84 questions) | - | 13.7 | 784 |

e: the 'mass' is 1 when the greedy 8-token reply contains a standalone A/B/C, else 0; its seconds include c's prefill. d's seconds include c's prefill (d continues c's cache).

Wording sha256: a 225330734fa0, b f81ba116520e, c db0be071aa7c, d 35aac1299ae5, f d8607efa8852, g f0709042de66, h 156398ccabae, e dcca9b05d0ab
