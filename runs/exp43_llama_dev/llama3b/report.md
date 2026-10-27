Gate on 13 reproduced in-view dev refusals (rev 1 A2): Steer answers >= ceil(0.5 x 13) = 7; gold(Steer) - gold(Steer-cut) >= 2; Random answers <= 0.2 x Steer's; degenerate <= 1. Rows in rule order (deeper layer, then smaller alpha). Gold: D1, provisional.

| setting | L | alpha | Steer ans | ok | degen | ok | gold Steer | gold cut | diff | ok | Random ans | bound | ok | gate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| steer:L25:a1:vnorm | 25 | 1 | 4/13 | no | 0 | yes | 1 | — | — | — | — | 0.8 | — | no |
| steer:L25:a2:vnorm | 25 | 2 | 9/13 | yes | 0 | yes | 3 | 0 | 3 | yes | 2 | 1.8 | no | no |
| steer:L25:a4:vnorm | 25 | 4 | 13/13 | yes | 0 | yes | 4 | 0 | 4 | yes | 4 | 2.6 | no | no |
| steer:L22:a1:vnorm | 22 | 1 | 3/13 | no | 0 | yes | 0 | — | — | — | — | 0.6 | — | no |
| steer:L22:a2:vnorm | 22 | 2 | 6/13 | no | 0 | yes | 1 | — | — | — | — | 1.2 | — | no |
| steer:L22:a4:vnorm | 22 | 4 | 13/13 | yes | 0 | yes | 1 | 0 | 1 | no | — | 2.6 | — | no |
| steer:L20:a1:vnorm | 20 | 1 | 3/13 | no | 0 | yes | 0 | — | — | — | — | 0.6 | — | no |
| steer:L20:a2:vnorm | 20 | 2 | 7/13 | yes | 0 | yes | 1 | 0 | 1 | no | — | 1.4 | — | no |
| steer:L20:a4:vnorm | 20 | 4 | 12/13 | yes | 0 | yes | 1 | 0 | 1 | no | — | 2.4 | — | no |
| steer:L17:a1:vnorm | 17 | 1 | 3/13 | no | 0 | yes | 1 | — | — | — | — | 0.6 | — | no |
| steer:L17:a2:vnorm | 17 | 2 | 7/13 | yes | 0 | yes | 2 | 0 | 2 | yes | 0 | 1.4 | yes | yes |
| steer:L17:a4:vnorm | 17 | 4 | 12/13 | yes | 0 | yes | 3 | 0 | 3 | yes | — | 2.4 | — | — |

A-cut (unsteered, cut prompt): {'n': 13, 'ans': 0, 'degenerate': 0, 'refusal': 13, 'gold': 0}
Chosen (rule order): steer:L17:a2:vnorm
Highest-Steer-answers alternative (descriptive, §38-style): steer:L17:a2:vnorm

## Gate rows: gold in the cached note (descriptive)

Gold (D1) in Llama's cached note on 2 of 13 G rows. Steer column: the chosen setting steer:L17:a2:vnorm.

| question | reproduced | gold in cached note | Steer class | Steer gold (D1) |
|---|---|---|---|---|
| 0862e8bf | yes | yes | answer | yes |
| 2ebe6c90 | yes | no | answer | no |
| 37f165cf | yes | no | refusal | no |
| 58470ed2 | yes | no | refusal | no |
| 60d45044 | yes | no | refusal | no |
| 71017276 | yes | no | answer | no |
| 7401057b | yes | no | answer | no |
| 76d63226 | yes | no | refusal | no |
| 7e00a6cb | yes | no | refusal | no |
| b86304ba | yes | no | answer | no |
| c14c00dd | yes | yes | answer | yes |
| c9f37c46 | yes | no | answer | no |
| fca70973 | yes | no | refusal | no |

