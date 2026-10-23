# §40: the fixed layer-29 setting (alpha 2) on LoCoMo

The §38 setting, fixed in advance, on a set it had never seen.

Engine hf-transformers-5.17.0-cuda-bfloat16; judge hf-transformers-5.17.0-cuda-bfloat16; layer 29. Set: 600 answerable rows + 40 adversarial (seed 40); excluded {'none:cat3': 4, 'too_many:cat1': 5, 'too_many:cat3': 3, 'unresolved:cat1': 3, 'unresolved:cat2': 1, 'unresolved:cat4': 1}; rows by category {'1': 99, '2': 114, '3': 34, '4': 353}; zero-fill views 2. locomo10.json sha 79fa87e90f04.
On the 600: marker rate 0.737 (442), refusal rate 0.207 (124), judged-correct rate 0.347 (208). R 120 (cap 120), C 40 (cap 40); selection recomputed equal: True.
Judge agreement on 60 cached §35 rows: 60/60 = 1.000 (bar 0.9; met).
Direction: v_29 sha a7f967ec..., steer 2 v_29 sha e0000366...; bf16 cast cosine 0.999999.

## Arms on R

| arm | rows | ans | refusal | degenerate | soft | P(gold) [95% CI] | no verdict |
|---|---|---|---|---|---|---|---|
| A | 120/120 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0 |
| B | 120/120 | 0.400 | 0.600 | 0.000 | 0.000 | 0.217 [0.150, 0.292] | 0 |
| Steer-all | 120/120 | 0.758 | 0.242 | 0.000 | 0.050 | 0.358 [0.275, 0.450] | 0 |
| Steer-all-cut | 120/120 | 0.525 | 0.475 | 0.000 | 0.117 | 0.025 [0.000, 0.058] | 0 |
| A-cut | 120/120 | 0.075 | 0.925 | 0.000 | 0.000 | 0.017 [0.000, 0.042] | 0 |
| Random | 120/120 | 0.058 | 0.942 | 0.000 | 0.000 | 0.017 [0.000, 0.042] | 0 |
| Steer-0 | 120/120 | 0.758 | 0.242 | 0.000 | 0.033 | 0.283 [0.200, 0.367] | 0 |

**Claim 1** P(gold(Steer-all)) - P(gold(Steer-all-cut)) = 0.333 [0.250, 0.425], n 120: **SHOWN**; P(gold(Steer-all)) 0.358 (meaningful).
**Control X** ans(Random) 0.058 against 0.2 x ans(Steer-all) = 0.152: **passes**.
**Control C** (C, 40 rows) P(gold(Steer-all)) 0.800 - P(gold(A)) 1.000 = -0.200 [-0.325, -0.075], n 40; upper bound -0.075: **not harmful** (harmful if UB < -0.10).
**Claim 2** (descriptive) P(gold(Steer-all)) - P(gold(B)) on R = 0.142 [0.058, 0.225], n 120.

## By question type (R)

| type | n | Steer-all | Steer-all-cut | B | A-cut | ans(Random) |
|---|---|---|---|---|---|---|
| multi-hop | 25 | 0.320 | 0.040 | 0.280 | 0.040 | 0.120 |
| temporal | 28 | 0.250 | 0.000 | 0.107 | 0.000 | 0.000 |
| open-domain | 20 | 0.200 | 0.050 | 0.000 | 0.000 | 0.050 |
| single-hop | 47 | 0.511 | 0.021 | 0.340 | 0.021 | 0.064 |

## Adversarial block (37 marked of 40; answer rate only, no judge)

- A: ans 0.324 (n 37), soft 0.000
- B: ans 0.541 (n 37), soft 0.000
- Steer-all: ans 0.730 (n 37), soft 0.027

## §41: the frozen conditional rule

Rule: {"path": "runs/exp41_dev/qwen3b/rule.json", "present": true, "sha256": "af0cbe9866ebba9fb882db80710e99bfb1a16a56db59e2592c13e1b79ae10e9f", "signal": "s_evid", "sign": "+", "tau": 2.2183780670166016, "scored_on_40": true, "refusal_start_ids": [9788], "commit": "30f891a963354ac05436f4b138d16885a061a6cb", "scored": true, "report_rule_sha256": "af0cbe9866ebba9fb882db80710e99bfb1a16a56db59e2592c13e1b79ae10e9f"}.
Signals on R (descriptive): s_margin n 120, mean 6.486, AUROC(gold Steer, higher) 0.185; s_proj n 120, mean -28.562, AUROC(gold Steer, higher) 0.803; s_evid n 120, mean -2.719, AUROC(gold Steer, higher) 0.232.
- Claim 4 (amendment 1): AUROC of s_proj for gold(Steer-all) on R 0.803 (0.724, 0.875), n 120: **SHOWN** (bar: lower bound > 0.5); beside, descriptive: s_margin 0.185 (0.111, 0.269); s_evid 0.232 (0.140, 0.332).
- lambda 1 (claims): U {'A': 0, 'B': 4, 'Steer': -1, 'Cond': 1}; Claim 1 U(Cond) - U(Steer) 0.017 [-0.125, 0.158], n 120 **NOT SHOWN**, beside U(Steer) - U(A) -0.008 [-0.150, 0.142], n 120: expected under a net-negative steer; not evidence of a selective signal; Claim 2 U(Cond) - U(B) -0.025 [-0.142, 0.092], n 120 **NOT SHOWN**; Claim 3 U(Cond) - U(A) 0.008 [-0.033, 0.050], n 120 **NOT SHOWN**
  - post hoc (descriptive): steer the top fraction q by s_proj: q 0.0 U 0, q 0.1 U 9, q 0.2 U 9, q 0.3 U 9, q 0.4 U 7, q 0.5 U 7, q 0.6 U 4, q 0.7 U 2, q 0.8 U 2, q 0.9 U 1, q 1.0 U -1
- lambda 2 (beside): U {'A': 0, 'B': -18, 'Steer': -44, 'Cond': -2}; Claim 1 U(Cond) - U(Steer) 0.350 [0.133, 0.567], n 120 **SHOWN**, beside U(Steer) - U(A) -0.367 [-0.592, -0.133], n 120: expected under a net-negative steer; not evidence of a selective signal; Claim 2 U(Cond) - U(B) 0.133 [-0.042, 0.308], n 120 **NOT SHOWN**; Claim 3 U(Cond) - U(A) -0.017 [-0.083, 0.042], n 120 **NOT SHOWN**
  - post hoc (descriptive): steer the top fraction q by s_proj: q 0.0 U 0, q 0.1 U 8, q 0.2 U 2, q 0.3 U -4, q 0.4 U -13, q 0.5 U -19, q 0.6 U -29, q 0.7 U -36, q 0.8 U -37, q 0.9 U -40, q 1.0 U -44
- rule fires: gold rate among fired rows 0.364 (n 11) against Steer-all's overall 0.358
- rows 120, fires 11, skipped {}; rule AUROC for gold(Steer) 0.232; adversarial ans Cond 0.324 (n 37) beside Steer-all 0.730
- coverage / risk: A 0.000 / n/a; B 0.400 / 0.458; Steer 0.708 / 0.506; Cond 0.058 / 0.429

## Costs

Stage seconds: load_3b 11, load_judge 11, smoke 11, agreement 2, notes 176, apass 131, judge_apass 13, arms 236, judge_arms 12, total 611.
Judge: {'n_items': 1242, 'n_judged': 1242, 'n_prompts_called': 391, 'seconds': 12.429652713006362, 'unparseable': 0, 'cut': False}.
Generations by file: {'apass.jsonl': {'rows': 479, 'tokens': 3128, 'gen_seconds': 110.7}, 'arms.jsonl': {'rows': 911, 'tokens': 4918, 'gen_seconds': 171.0}}. Rows without a verdict left out: 0.
Run 10.2 min of model work (pod minutes include install and loads; price $0.59/h).
Smoke: {'same_first': 5, 'marker_agree': 5, 'marker_batched': 3, 'marker_single': 3, 'exact_text': 4, 'n': 5, 'need': 4, 'problems': [], 'pass': True}.
