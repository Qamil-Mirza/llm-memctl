# §45 report (j56-jrefit-logit, half: held)

Descriptive. Readable (fixed, rev 1) = the entity's space-form id in the lens top-10 at a counted position (the last prompt token and the answer span: F1 every generated token, F2 / stage 3 the tokens after 'Answer:'), or its bare-form id at the last prompt token; criterion (a) before the final layer. Rows marked '| ...' are descriptive beside. A miss is 'not recovered by this lens', never 'not computed'. **The held half decides**; dev is shown first for presentation, pooled beside.

Caveat: in F2 the memory lines carry dates that increase with line order, so in TU the timestamp order is a second update cue beside the wording (a format effect, which F2 is there to show).

## Stage 1: solved prompts (246 of 432)

Outcomes: {"gold": 246, "other": 88, "abstain": 78, "competitor": 20}
Answer marker present (format|present): {"F1|False": 216, "F2|False": 2, "F2|True": 214}

### held (decides)

| task | lens | criterion | n | readable | share | Wilson 95% | verdict | first depth q1/q2/q3 (fixed-rule depths: inflated by echo, not shown) |
|---|---|---|---|---|---|---|---|---|
| DE | j56 | a_gold | 95 | 95 | 1.00 | [0.96, 1.00] | recovers (inflated by echo) | inflated by echo |
| DE | j56 | a_gold | last prompt token only | 95 | 23 | 0.24 | [0.17, 0.34] | not recovered | 0.92/0.92/0.92 |
| DE | j56 | a_gold | any generated token | 95 | 95 | 1.00 | [0.96, 1.00] | recovers (inflated by echo) | inflated by echo |
| DE | j56 | a_gold | either form anywhere | 95 | 95 | 1.00 | [0.96, 1.00] | recovers (inflated by echo) | inflated by echo |
| DE | jrefit | a_gold | 95 | 95 | 1.00 | [0.96, 1.00] | recovers (inflated by echo) | inflated by echo |
| DE | jrefit | a_gold | last prompt token only | 95 | 23 | 0.24 | [0.17, 0.34] | not recovered | 0.92/0.92/0.92 |
| DE | jrefit | a_gold | any generated token | 95 | 95 | 1.00 | [0.96, 1.00] | recovers (inflated by echo) | inflated by echo |
| DE | jrefit | a_gold | either form anywhere | 95 | 95 | 1.00 | [0.96, 1.00] | recovers (inflated by echo) | inflated by echo |
| DE | logit | a_gold | 95 | 95 | 1.00 | [0.96, 1.00] | recovers (inflated by echo) | inflated by echo |
| DE | logit | a_gold | last prompt token only | 95 | 21 | 0.22 | [0.15, 0.31] | not recovered | 0.92/0.92/0.94 |
| DE | logit | a_gold | any generated token | 95 | 95 | 1.00 | [0.96, 1.00] | recovers (inflated by echo) | inflated by echo |
| DE | logit | a_gold | either form anywhere | 95 | 95 | 1.00 | [0.96, 1.00] | recovers (inflated by echo) | inflated by echo |
| RC | j56 | a_gold | 79 | 79 | 1.00 | [0.95, 1.00] | recovers (inflated by echo) | inflated by echo |
| RC | j56 | a_gold | last prompt token only | 79 | 24 | 0.30 | [0.21, 0.41] | not recovered | 0.92/0.92/0.92 |
| RC | j56 | a_gold | any generated token | 79 | 79 | 1.00 | [0.95, 1.00] | recovers (inflated by echo) | inflated by echo |
| RC | j56 | a_gold | either form anywhere | 79 | 79 | 1.00 | [0.95, 1.00] | recovers (inflated by echo) | inflated by echo |
| RC | j56 | b_intermediate | 50 | 47 | 0.94 | [0.84, 0.98] | recovers (inflated by echo) | inflated by echo |
| RC | j56 | b_intermediate | last prompt token only | 50 | 43 | 0.86 | [0.74, 0.93] | recovers | 0.92/0.92/0.92 |
| RC | j56 | b_intermediate | any generated token | 50 | 50 | 1.00 | [0.93, 1.00] | recovers (inflated by echo) | inflated by echo |
| RC | j56 | b_intermediate | either form anywhere | 50 | 50 | 1.00 | [0.93, 1.00] | recovers (inflated by echo) | inflated by echo |
| RC | jrefit | a_gold | 79 | 79 | 1.00 | [0.95, 1.00] | recovers (inflated by echo) | inflated by echo |
| RC | jrefit | a_gold | last prompt token only | 79 | 24 | 0.30 | [0.21, 0.41] | not recovered | 0.92/0.92/0.92 |
| RC | jrefit | a_gold | any generated token | 79 | 79 | 1.00 | [0.95, 1.00] | recovers (inflated by echo) | inflated by echo |
| RC | jrefit | a_gold | either form anywhere | 79 | 79 | 1.00 | [0.95, 1.00] | recovers (inflated by echo) | inflated by echo |
| RC | jrefit | b_intermediate | 50 | 47 | 0.94 | [0.84, 0.98] | recovers (inflated by echo) | inflated by echo |
| RC | jrefit | b_intermediate | last prompt token only | 50 | 43 | 0.86 | [0.74, 0.93] | recovers | 0.92/0.92/0.92 |
| RC | jrefit | b_intermediate | any generated token | 50 | 50 | 1.00 | [0.93, 1.00] | recovers (inflated by echo) | inflated by echo |
| RC | jrefit | b_intermediate | either form anywhere | 50 | 50 | 1.00 | [0.93, 1.00] | recovers (inflated by echo) | inflated by echo |
| RC | logit | a_gold | 79 | 79 | 1.00 | [0.95, 1.00] | recovers (inflated by echo) | inflated by echo |
| RC | logit | a_gold | last prompt token only | 79 | 16 | 0.20 | [0.13, 0.30] | not recovered | 0.92/0.92/0.92 |
| RC | logit | a_gold | any generated token | 79 | 79 | 1.00 | [0.95, 1.00] | recovers (inflated by echo) | inflated by echo |
| RC | logit | a_gold | either form anywhere | 79 | 79 | 1.00 | [0.95, 1.00] | recovers (inflated by echo) | inflated by echo |
| RC | logit | b_intermediate | 50 | 44 | 0.88 | [0.76, 0.94] | recovers (inflated by echo) | inflated by echo |
| RC | logit | b_intermediate | last prompt token only | 50 | 40 | 0.80 | [0.67, 0.89] | recovers | 0.92/0.92/0.92 |
| RC | logit | b_intermediate | any generated token | 50 | 50 | 1.00 | [0.93, 1.00] | recovers (inflated by echo) | inflated by echo |
| RC | logit | b_intermediate | either form anywhere | 50 | 50 | 1.00 | [0.93, 1.00] | recovers (inflated by echo) | inflated by echo |
| TU | j56 | a_gold | 72 | 72 | 1.00 | [0.95, 1.00] | recovers (inflated by echo) | inflated by echo |
| TU | j56 | a_gold | last prompt token only | 72 | 0 | 0.00 | [0.00, 0.05] | not recovered | - |
| TU | j56 | a_gold | any generated token | 72 | 72 | 1.00 | [0.95, 1.00] | recovers (inflated by echo) | inflated by echo |
| TU | j56 | a_gold | either form anywhere | 72 | 72 | 1.00 | [0.95, 1.00] | recovers (inflated by echo) | inflated by echo |
| TU | j56 | c_competitor | 72 | 34 | 0.47 | [0.36, 0.59] | not recovered (inflated by echo) | inflated by echo |
| TU | j56 | c_competitor | last prompt token only | 72 | 0 | 0.00 | [0.00, 0.05] | not recovered | - |
| TU | j56 | c_competitor | any generated token | 72 | 63 | 0.88 | [0.78, 0.93] | recovers (inflated by echo) | inflated by echo |
| TU | j56 | c_competitor | either form anywhere | 72 | 63 | 0.88 | [0.78, 0.93] | recovers (inflated by echo) | inflated by echo |
| TU | jrefit | a_gold | 72 | 72 | 1.00 | [0.95, 1.00] | recovers (inflated by echo) | inflated by echo |
| TU | jrefit | a_gold | last prompt token only | 72 | 0 | 0.00 | [0.00, 0.05] | not recovered | - |
| TU | jrefit | a_gold | any generated token | 72 | 72 | 1.00 | [0.95, 1.00] | recovers (inflated by echo) | inflated by echo |
| TU | jrefit | a_gold | either form anywhere | 72 | 72 | 1.00 | [0.95, 1.00] | recovers (inflated by echo) | inflated by echo |
| TU | jrefit | c_competitor | 72 | 35 | 0.49 | [0.37, 0.60] | not recovered (inflated by echo) | inflated by echo |
| TU | jrefit | c_competitor | last prompt token only | 72 | 0 | 0.00 | [0.00, 0.05] | not recovered | - |
| TU | jrefit | c_competitor | any generated token | 72 | 64 | 0.89 | [0.80, 0.94] | recovers (inflated by echo) | inflated by echo |
| TU | jrefit | c_competitor | either form anywhere | 72 | 64 | 0.89 | [0.80, 0.94] | recovers (inflated by echo) | inflated by echo |
| TU | logit | a_gold | 72 | 72 | 1.00 | [0.95, 1.00] | recovers (inflated by echo) | inflated by echo |
| TU | logit | a_gold | last prompt token only | 72 | 0 | 0.00 | [0.00, 0.05] | not recovered | - |
| TU | logit | a_gold | any generated token | 72 | 72 | 1.00 | [0.95, 1.00] | recovers (inflated by echo) | inflated by echo |
| TU | logit | a_gold | either form anywhere | 72 | 72 | 1.00 | [0.95, 1.00] | recovers (inflated by echo) | inflated by echo |
| TU | logit | c_competitor | 72 | 31 | 0.43 | [0.32, 0.55] | not recovered (inflated by echo) | inflated by echo |
| TU | logit | c_competitor | last prompt token only | 72 | 0 | 0.00 | [0.00, 0.05] | not recovered | - |
| TU | logit | c_competitor | any generated token | 72 | 59 | 0.82 | [0.72, 0.89] | recovers (inflated by echo) | inflated by echo |
| TU | logit | c_competitor | either form anywhere | 72 | 59 | 0.82 | [0.72, 0.89] | recovers (inflated by echo) | inflated by echo |

## Post hoc echo rules (§45 results review, 1a6e1e5)

Every number in this section is **post hoc**. Rule A = readable anywhere in the answer span (post hoc, rule A): a hit on target X at counted position p does not count when the token fed in at p is one of X's pinned ids (either form). Rule B = readable before the answer is produced (post hoc, rule B): count only the counted positions up to and including the one that produces X's first occurrence: p is excluded once X is among input_ids[lp+1..p] (lp the last prompt token). The bare form at the last prompt token is unchanged. The fixed column is the pre-registered rule, inflated by echo (§45 results review); not evidence of readability before production. In F2, rule B mostly measures whether the note already held the target. Verdicts against the 0.70 / 0.55 bar are descriptive. Depth = first readable layer / 36. Rules A and B read every criterion before the final layer (for (b) and (c) the pre-registered call counts the final layer too; its before-final value is in brackets, the rules with the final layer are in the JSON). 'Last prompt token only' = both forms at the last prompt token (echo-free by construction).

### held (decides): post hoc

| task | lens | criterion | format | n | fixed (inflated by echo) | rule A share [Wilson] | rule A verdict | rule A depth q1/q2/q3 | rule B share [Wilson] | rule B verdict | rule B depth q1/q2/q3 | last prompt token only |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DE | j56 | a_gold | pooled | 95 | 1.00 | 1.00 [0.96, 1.00] | recovers (descriptive, post hoc) | 0.83/0.89/0.89 | 0.62 [0.52, 0.71] | not recovered (descriptive, post hoc) | 0.83/0.86/0.89 | 0.24 |
| DE | j56 | a_gold | F1 | 49 | 1.00 | 1.00 [0.93, 1.00] | recovers (descriptive, post hoc) | 0.83/0.86/0.89 | 1.00 [0.93, 1.00] | recovers (descriptive, post hoc) | 0.83/0.86/0.89 | 0.27 |
| DE | j56 | a_gold | F2 | 46 | 1.00 | 1.00 [0.92, 1.00] | recovers (descriptive, post hoc) | 0.89/0.89/0.89 | 0.22 [0.12, 0.36] | not recovered (descriptive, post hoc) | 0.92/0.92/0.94 | 0.22 |
| DE | jrefit | a_gold | pooled | 95 | 1.00 | 1.00 [0.96, 1.00] | recovers (descriptive, post hoc) | 0.83/0.89/0.89 | 0.62 [0.52, 0.71] | not recovered (descriptive, post hoc) | 0.83/0.86/0.89 | 0.24 |
| DE | jrefit | a_gold | F1 | 49 | 1.00 | 1.00 [0.93, 1.00] | recovers (descriptive, post hoc) | 0.83/0.86/0.89 | 1.00 [0.93, 1.00] | recovers (descriptive, post hoc) | 0.83/0.86/0.89 | 0.27 |
| DE | jrefit | a_gold | F2 | 46 | 1.00 | 1.00 [0.92, 1.00] | recovers (descriptive, post hoc) | 0.89/0.89/0.89 | 0.22 [0.12, 0.36] | not recovered (descriptive, post hoc) | 0.92/0.92/0.94 | 0.22 |
| DE | logit | a_gold | pooled | 95 | 1.00 | 1.00 [0.96, 1.00] | recovers (descriptive, post hoc) | 0.86/0.86/0.89 | 0.60 [0.50, 0.69] | not recovered (descriptive, post hoc) | 0.86/0.86/0.92 | 0.22 |
| DE | logit | a_gold | F1 | 49 | 1.00 | 1.00 [0.93, 1.00] | recovers (descriptive, post hoc) | 0.86/0.86/0.86 | 1.00 [0.93, 1.00] | recovers (descriptive, post hoc) | 0.86/0.86/0.89 | 0.27 |
| DE | logit | a_gold | F2 | 46 | 1.00 | 1.00 [0.92, 1.00] | recovers (descriptive, post hoc) | 0.86/0.89/0.89 | 0.17 [0.09, 0.31] | not recovered (descriptive, post hoc) | 0.92/0.94/0.94 | 0.17 |
| RC | j56 | a_gold | pooled | 79 | 1.00 | 1.00 [0.95, 1.00] | recovers (descriptive, post hoc) | 0.86/0.89/0.89 | 0.61 [0.50, 0.71] | not recovered (descriptive, post hoc) | 0.83/0.89/0.90 | 0.30 |
| RC | j56 | a_gold | F1 | 37 | 1.00 | 1.00 [0.91, 1.00] | recovers (descriptive, post hoc) | 0.83/0.86/0.89 | 1.00 [0.91, 1.00] | recovers (descriptive, post hoc) | 0.83/0.89/0.89 | 0.35 |
| RC | j56 | a_gold | F2 | 42 | 1.00 | 1.00 [0.92, 1.00] | recovers (descriptive, post hoc) | 0.89/0.89/0.89 | 0.26 [0.15, 0.41] | not recovered (descriptive, post hoc) | 0.92/0.92/0.93 | 0.26 |
| RC | j56 | b_intermediate | pooled | 50 | 0.94 (before final 0.84) | 0.84 [0.71, 0.92] | recovers (descriptive, post hoc) | 0.89/0.89/0.92 | 0.84 [0.71, 0.92] | recovers (descriptive, post hoc) | 0.89/0.89/0.92 | 0.86 |
| RC | j56 | b_intermediate | F1 | 23 | 1.00 (before final 1.00) | 1.00 [0.86, 1.00] | recovers (descriptive, post hoc) | 0.89/0.89/0.89 | 1.00 [0.86, 1.00] | recovers (descriptive, post hoc) | 0.89/0.89/0.89 | 1.00 |
| RC | j56 | b_intermediate | F2 | 27 | 0.89 (before final 0.70) | 0.70 [0.52, 0.84] | not recovered (descriptive, post hoc) | 0.92/0.92/0.93 | 0.70 [0.52, 0.84] | not recovered (descriptive, post hoc) | 0.92/0.92/0.93 | 0.74 |
| RC | jrefit | a_gold | pooled | 79 | 1.00 | 1.00 [0.95, 1.00] | recovers (descriptive, post hoc) | 0.86/0.89/0.89 | 0.61 [0.50, 0.71] | not recovered (descriptive, post hoc) | 0.83/0.89/0.90 | 0.30 |
| RC | jrefit | a_gold | F1 | 37 | 1.00 | 1.00 [0.91, 1.00] | recovers (descriptive, post hoc) | 0.83/0.86/0.89 | 1.00 [0.91, 1.00] | recovers (descriptive, post hoc) | 0.83/0.86/0.89 | 0.35 |
| RC | jrefit | a_gold | F2 | 42 | 1.00 | 1.00 [0.92, 1.00] | recovers (descriptive, post hoc) | 0.89/0.89/0.89 | 0.26 [0.15, 0.41] | not recovered (descriptive, post hoc) | 0.92/0.92/0.93 | 0.26 |
| RC | jrefit | b_intermediate | pooled | 50 | 0.94 (before final 0.84) | 0.84 [0.71, 0.92] | recovers (descriptive, post hoc) | 0.89/0.89/0.92 | 0.84 [0.71, 0.92] | recovers (descriptive, post hoc) | 0.89/0.89/0.92 | 0.86 |
| RC | jrefit | b_intermediate | F1 | 23 | 1.00 (before final 1.00) | 1.00 [0.86, 1.00] | recovers (descriptive, post hoc) | 0.89/0.89/0.89 | 1.00 [0.86, 1.00] | recovers (descriptive, post hoc) | 0.89/0.89/0.89 | 1.00 |
| RC | jrefit | b_intermediate | F2 | 27 | 0.89 (before final 0.70) | 0.70 [0.52, 0.84] | not recovered (descriptive, post hoc) | 0.92/0.92/0.92 | 0.70 [0.52, 0.84] | not recovered (descriptive, post hoc) | 0.92/0.92/0.92 | 0.74 |
| RC | logit | a_gold | pooled | 79 | 1.00 | 1.00 [0.95, 1.00] | recovers (descriptive, post hoc) | 0.86/0.89/0.89 | 0.56 [0.45, 0.66] | not recovered (descriptive, post hoc) | 0.86/0.89/0.92 | 0.20 |
| RC | logit | a_gold | F1 | 37 | 1.00 | 1.00 [0.91, 1.00] | recovers (descriptive, post hoc) | 0.86/0.89/0.89 | 1.00 [0.91, 1.00] | recovers (descriptive, post hoc) | 0.86/0.89/0.89 | 0.24 |
| RC | logit | a_gold | F2 | 42 | 1.00 | 1.00 [0.92, 1.00] | recovers (descriptive, post hoc) | 0.86/0.89/0.89 | 0.17 [0.08, 0.31] | not recovered (descriptive, post hoc) | 0.92/0.92/0.92 | 0.17 |
| RC | logit | b_intermediate | pooled | 50 | 0.88 (before final 0.72) | 0.72 [0.58, 0.83] | recovers (descriptive, post hoc) | 0.89/0.89/0.92 | 0.70 [0.56, 0.81] | recovers (descriptive, post hoc) | 0.89/0.89/0.92 | 0.80 |
| RC | logit | b_intermediate | F1 | 23 | 1.00 (before final 1.00) | 1.00 [0.86, 1.00] | recovers (descriptive, post hoc) | 0.89/0.89/0.89 | 1.00 [0.86, 1.00] | recovers (descriptive, post hoc) | 0.89/0.89/0.89 | 1.00 |
| RC | logit | b_intermediate | F2 | 27 | 0.78 (before final 0.48) | 0.48 [0.31, 0.66] | not recovered (descriptive, post hoc) | 0.92/0.92/0.92 | 0.44 [0.28, 0.63] | not recovered (descriptive, post hoc) | 0.92/0.92/0.92 | 0.63 |
| TU | j56 | a_gold | pooled | 72 | 1.00 | 1.00 [0.95, 1.00] | recovers (descriptive, post hoc) | 0.86/0.89/0.89 | 0.35 [0.25, 0.46] | not recovered (descriptive, post hoc) | 0.83/0.83/0.86 | 0.00 |
| TU | j56 | a_gold | F1 | 25 | 1.00 | 1.00 [0.87, 1.00] | recovers (descriptive, post hoc) | 0.83/0.83/0.86 | 1.00 [0.87, 1.00] | recovers (descriptive, post hoc) | 0.83/0.83/0.86 | 0.00 |
| TU | j56 | a_gold | F2 | 47 | 1.00 | 1.00 [0.92, 1.00] | recovers (descriptive, post hoc) | 0.89/0.89/0.89 | 0.00 [0.00, 0.08] | not recovered (descriptive, post hoc) | - | 0.00 |
| TU | j56 | c_competitor | pooled | 72 | 0.47 (before final 0.29) | 0.29 [0.20, 0.41] | not recovered (descriptive, post hoc) | 0.83/0.89/0.92 | 0.22 [0.14, 0.33] | not recovered (descriptive, post hoc) | 0.83/0.89/0.89 | 0.00 |
| TU | j56 | c_competitor | F1 | 25 | 0.64 (before final 0.64) | 0.64 [0.45, 0.80] | not recovered (descriptive, post hoc) | 0.83/0.89/0.89 | 0.64 [0.45, 0.80] | not recovered (descriptive, post hoc) | 0.83/0.89/0.89 | 0.00 |
| TU | j56 | c_competitor | F2 | 47 | 0.38 (before final 0.11) | 0.11 [0.05, 0.23] | not recovered (descriptive, post hoc) | 0.92/0.92/0.92 | 0.00 [0.00, 0.08] | not recovered (descriptive, post hoc) | - | 0.00 |
| TU | jrefit | a_gold | pooled | 72 | 1.00 | 1.00 [0.95, 1.00] | recovers (descriptive, post hoc) | 0.86/0.89/0.89 | 0.35 [0.25, 0.46] | not recovered (descriptive, post hoc) | 0.83/0.83/0.86 | 0.00 |
| TU | jrefit | a_gold | F1 | 25 | 1.00 | 1.00 [0.87, 1.00] | recovers (descriptive, post hoc) | 0.83/0.83/0.86 | 1.00 [0.87, 1.00] | recovers (descriptive, post hoc) | 0.83/0.83/0.86 | 0.00 |
| TU | jrefit | a_gold | F2 | 47 | 1.00 | 1.00 [0.92, 1.00] | recovers (descriptive, post hoc) | 0.89/0.89/0.89 | 0.00 [0.00, 0.08] | not recovered (descriptive, post hoc) | - | 0.00 |
| TU | jrefit | c_competitor | pooled | 72 | 0.49 (before final 0.29) | 0.29 [0.20, 0.41] | not recovered (descriptive, post hoc) | 0.83/0.89/0.92 | 0.24 [0.15, 0.35] | not recovered (descriptive, post hoc) | 0.83/0.89/0.89 | 0.00 |
| TU | jrefit | c_competitor | F1 | 25 | 0.68 (before final 0.68) | 0.68 [0.48, 0.83] | not recovered (descriptive, post hoc) | 0.83/0.89/0.89 | 0.68 [0.48, 0.83] | not recovered (descriptive, post hoc) | 0.83/0.89/0.89 | 0.00 |
| TU | jrefit | c_competitor | F2 | 47 | 0.38 (before final 0.09) | 0.09 [0.03, 0.20] | not recovered (descriptive, post hoc) | 0.91/0.92/0.92 | 0.00 [0.00, 0.08] | not recovered (descriptive, post hoc) | - | 0.00 |
| TU | logit | a_gold | pooled | 72 | 1.00 | 1.00 [0.95, 1.00] | recovers (descriptive, post hoc) | 0.86/0.86/0.89 | 0.35 [0.25, 0.46] | not recovered (descriptive, post hoc) | 0.86/0.86/0.86 | 0.00 |
| TU | logit | a_gold | F1 | 25 | 1.00 | 1.00 [0.87, 1.00] | recovers (descriptive, post hoc) | 0.86/0.86/0.86 | 1.00 [0.87, 1.00] | recovers (descriptive, post hoc) | 0.86/0.86/0.86 | 0.00 |
| TU | logit | a_gold | F2 | 47 | 1.00 | 1.00 [0.92, 1.00] | recovers (descriptive, post hoc) | 0.86/0.89/0.89 | 0.00 [0.00, 0.08] | not recovered (descriptive, post hoc) | - | 0.00 |
| TU | logit | c_competitor | pooled | 72 | 0.43 (before final 0.26) | 0.26 [0.18, 0.38] | not recovered (descriptive, post hoc) | 0.89/0.89/0.92 | 0.18 [0.11, 0.28] | not recovered (descriptive, post hoc) | 0.89/0.89/0.89 | 0.00 |
| TU | logit | c_competitor | F1 | 25 | 0.52 (before final 0.52) | 0.52 [0.33, 0.70] | not recovered (descriptive, post hoc) | 0.89/0.89/0.89 | 0.52 [0.33, 0.70] | not recovered (descriptive, post hoc) | 0.89/0.89/0.89 | 0.00 |
| TU | logit | c_competitor | F2 | 47 | 0.38 (before final 0.13) | 0.13 [0.06, 0.25] | not recovered (descriptive, post hoc) | 0.90/0.93/0.94 | 0.00 [0.00, 0.08] | not recovered (descriptive, post hoc) | - | 0.00 |

### Fit stability (j56 vs jrefit, before the final layer)

- solved prompts: agreement 0.998 over 939 calls -> **stable**; Spearman median 0.980 (quartiles [0.9725945625850406, 0.980105355459875, 0.9880563183389721], n=939)
- all prompts (beside): agreement 0.999 over 1656 calls -> **stable**; Spearman median 0.980 (quartiles [0.9704025487420083, 0.9795438837103732, 0.9878703619798364], n=1656)

## Stage 2: matched failures (186, 78 at H2, 150 paired)

- j56: on the matched failures the gold is not readable before the final layer on >= 0.70 of them (j56) (share 0.65, Wilson [0.58, 0.72], n=186)
- jrefit: on the matched failures the gold is not readable before the final layer on >= 0.70 of them (jrefit) (share 0.65, Wilson [0.58, 0.72], n=186)
- logit: on the matched failures the gold is not readable before the final layer on >= 0.70 of them (logit) (share 0.64, Wilson [0.57, 0.71], n=186)

### Stage 2, post hoc (rules A/B; fixed departure layers struck as an echo artefact)

| lens | rule | not readable before the final layer | share [Wilson] | departure layer buckets (paired) |
|---|---|---|---|---|
| j56 | fixed (inflated by echo) | 65/186 | 0.35 [0.28, 0.42] | 0: 88, 1-9: 26, 10-29: 4, 30+: 16, none: 16 |
| j56 | rule A (post hoc) | 65/186 | 0.35 [0.28, 0.42] | 0: 0, 1-9: 0, 10-29: 0, 30+: 100, none: 50 |
| j56 | rule B (post hoc) | 124/186 | 0.67 [0.60, 0.73] | 0: 0, 1-9: 0, 10-29: 0, 30+: 67, none: 83 |
| jrefit | fixed (inflated by echo) | 65/186 | 0.35 [0.28, 0.42] | 0: 86, 1-9: 22, 10-29: 7, 30+: 16, none: 19 |
| jrefit | rule A (post hoc) | 65/186 | 0.35 [0.28, 0.42] | 0: 0, 1-9: 0, 10-29: 2, 30+: 96, none: 52 |
| jrefit | rule B (post hoc) | 124/186 | 0.67 [0.60, 0.73] | 0: 0, 1-9: 0, 10-29: 2, 30+: 63, none: 85 |
| logit | fixed (inflated by echo) | 67/186 | 0.36 [0.29, 0.43] | 0: 8, 1-9: 4, 10-29: 4, 30+: 98, none: 36 |
| logit | rule A (post hoc) | 67/186 | 0.36 [0.29, 0.43] | 0: 0, 1-9: 1, 10-29: 4, 30+: 106, none: 39 |
| logit | rule B (post hoc) | 127/186 | 0.68 [0.61, 0.75] | 0: 0, 1-9: 0, 10-29: 4, 30+: 61, none: 85 |

Failures by task, hardness and outcome: {"DE|H0|abstain": 6, "DE|H1|abstain": 11, "DE|H1|other": 10, "DE|H2|abstain": 8, "DE|H2|other": 14, "RC|H0|abstain": 6, "RC|H0|other": 2, "RC|H1|abstain": 13, "RC|H1|other": 17, "RC|H2|abstain": 11, "RC|H2|other": 16, "TU|H0|competitor": 2, "TU|H0|other": 6, "TU|H1|abstain": 17, "TU|H1|competitor": 9, "TU|H1|other": 9, "TU|H2|abstain": 6, "TU|H2|competitor": 9, "TU|H2|other": 14}

| failure | outcome | ambiguous | marker | pair | departure layer per lens | generated |
|---|---|---|---|---|---|---|
| DE01_v0_F1_H2 | other | False | False | DE01_v0_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 31} | Based on the information provided, there is no mention of where Paul lives. The |
| DE01_v1_F1_H2 | other | False | False | DE01_v1_F1_H0 | {"j56": 9, "jrefit": 9, "logit": 31} | Based on the information provided, there is no mention of where Paul lives. The |
| DE01_v1_F2_H0 | abstain | False | True | - | {} | user stated that Paul lives in Norwich. Answer: unknown |
| DE01_v1_F2_H1 | abstain | False | True | - | {} | Kate works as a cook. Sarah works as a designer. Paul lives in Norwich. Answer:  |
| DE01_v1_F2_H2 | abstain | False | True | - | {} | Kate works as a cook. Sarah works as a designer. Rachel works in Chester. Alice  |
| DE01_v2_F1_H1 | other | False | False | DE01_v2_F1_H0 | {"j56": null, "jrefit": null, "logit": null} | Based on the information provided:  - Kate works as a cook. - Sarah works |
| DE01_v3_F1_H2 | other | False | False | DE01_v3_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 31} | Based on the information provided, there is no mention of where Paul lives. The |
| DE01_v3_F2_H2 | other | True | False | DE01_v3_F2_H0 | {"j56": 0, "jrefit": 0, "logit": 32} | Kate - works as a cook   James - works in York   Henry - lives in Manchester   D |
| DE03_v0_F1_H1 | other | False | False | DE03_v0_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 30} | Based on the information provided:  - James works in Cambridge. - Laura works in |
| DE03_v1_F1_H1 | other | False | False | DE03_v1_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 31} | Based on the information provided:  - James works in Cambridge. - Laura works in |
| DE03_v1_F1_H2 | other | False | False | DE03_v1_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 31} | Based on the information provided:  - James works in Cambridge. - Laura works in |
| DE03_v1_F2_H0 | abstain | False | True | DE03_v1_F2_H2 | {"j56": 0, "jrefit": 0, "logit": null} | user stated that Ben lives in Durham. Answer: unknown |
| DE03_v1_F2_H1 | abstain | False | True | DE03_v1_F2_H2 | {"j56": 0, "jrefit": 0, "logit": 32} | James works in Cambridge, Laura works in Liverpool, and Ben lives in Durham base |
| DE03_v2_F1_H1 | other | False | False | DE03_v2_F1_H0 | {"j56": 31, "jrefit": 31, "logit": 31} | Based on the information provided:  - James works in Cambridge. - Laura works in |
| DE03_v2_F2_H1 | abstain | False | True | DE03_v2_F2_H0 | {"j56": 33, "jrefit": 33, "logit": null} | James works in Cambridge, Laura works in Liverpool, and no information is given  |
| DE03_v3_F1_H1 | other | False | False | DE03_v3_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 30} | Based on the information provided, there is no mention of where Ben lives. The |
| DE03_v3_F2_H1 | abstain | False | True | DE03_v3_F2_H0 | {"j56": 0, "jrefit": 0, "logit": 31} | James works in Cambridge. Laura works in Liverpool. Ben lives in Cardiff.   Answ |
| DE03_v3_F2_H2 | abstain | False | True | DE03_v3_F2_H0 | {"j56": 0, "jrefit": 0, "logit": 31} | James works in Cambridge, Laura works in Liverpool, Paul lives in Derby, Rachel  |
| DE05_v0_F1_H1 | other | False | False | DE05_v0_F1_H0 | {"j56": 32, "jrefit": 32, "logit": 31} | Based on the information provided:  - Alice lives in Liverpool. - Sarah works as |
| DE05_v0_F1_H2 | other | False | False | DE05_v0_F1_H0 | {"j56": 32, "jrefit": 32, "logit": 31} | Based on the information provided:  - Alice lives in Liverpool and works in Leed |
| DE05_v1_F1_H1 | other | False | False | DE05_v1_F1_H0 | {"j56": 9, "jrefit": 9, "logit": 31} | Based on the information provided:  - Alice lives in Liverpool. - Sarah works as |
| DE05_v1_F1_H2 | other | False | False | DE05_v1_F1_H0 | {"j56": 9, "jrefit": 9, "logit": 31} | Based on the information provided:  - Alice lives in Liverpool and works in Leed |
| DE05_v1_F2_H2 | abstain | False | True | DE05_v1_F2_H0 | {"j56": 8, "jrefit": 8, "logit": 32} | David lives in Brighton. Answer: unknown |
| DE05_v2_F1_H1 | other | False | False | DE05_v2_F1_H0 | {"j56": null, "jrefit": null, "logit": null} | Based on the information provided:  - Alice lives in Liverpool. - Sarah works as |
| DE05_v2_F1_H2 | other | False | False | DE05_v2_F1_H0 | {"j56": null, "jrefit": null, "logit": null} | Based on the information provided:  - Alice lives in Liverpool. - Sarah works as |
| DE05_v2_F2_H2 | abstain | False | True | DE05_v2_F2_H0 | {"j56": 0, "jrefit": 32, "logit": null} | Alice - lives in York Answer: unknown |
| DE05_v3_F1_H2 | other | False | False | DE05_v3_F1_H0 | {"j56": null, "jrefit": null, "logit": null} | Based on the information provided:  - Alice lives in Liverpool. - David lives in |
| DE07_v0_F1_H1 | other | False | False | DE07_v0_F1_H0 | {"j56": 0, "jrefit": 0, "logit": null} | Based on the information provided:  - Peter lives in Cambridge. - James lives in |
| DE07_v0_F2_H0 | abstain | False | True | DE07_v0_F2_H1 | {"j56": 0, "jrefit": 0, "logit": 31} | user stated that James lives in Manchester. Answer: unknown |
| DE07_v1_F2_H1 | abstain | False | True | DE07_v1_F2_H0 | {"j56": 32, "jrefit": 32, "logit": 32} | James lives in Chester. Answer: unknown |
| DE07_v3_F2_H1 | abstain | False | True | DE07_v3_F2_H0 | {"j56": 0, "jrefit": 0, "logit": 31} | James lives in Manchester. Answer: unknown |
| DE09_v0_F1_H2 | other | False | False | DE09_v0_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 30} | Based on the information provided:  - Emma works in Leeds. - Laura works in |
| DE09_v0_F2_H1 | abstain | False | True | DE09_v0_F2_H0 | {"j56": 0, "jrefit": 0, "logit": 32} | Emma works in Leeds, Laura works in Cardiff, Alice lives in Cambridge, David liv |
| DE09_v0_F2_H2 | abstain | False | True | DE09_v0_F2_H0 | {"j56": 0, "jrefit": 0, "logit": 32} | Emma works in Leeds and Alice lives in Cambridge. Answer: unknown |
| DE09_v1_F1_H2 | other | False | False | DE09_v1_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 32} | Based on the information provided:  - Emma works in Leeds. - Laura works in |
| DE09_v1_F2_H0 | abstain | False | True | - | {} | user stated that Alice lives in Derby. Answer: unknown |
| DE09_v1_F2_H1 | abstain | False | True | - | {} | Emma works in Leeds, Laura works in Cardiff, Alice lives in Derby, David lives i |
| DE09_v1_F2_H2 | abstain | False | True | - | {} | Emma works in Leeds and Alice lives in Derby. Answer: unknown |
| DE09_v2_F1_H1 | other | False | False | DE09_v2_F1_H0 | {"j56": null, "jrefit": null, "logit": null} | Based on the information provided:  - Emma works in Leeds. - Laura works in |
| DE09_v3_F1_H2 | other | False | False | DE09_v3_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 31} | Based on the information provided:  - Emma works in Leeds. - Laura works in |
| DE09_v3_F2_H0 | abstain | False | True | DE09_v3_F2_H2 | {"j56": 0, "jrefit": 0, "logit": 31} | Kate's information is about Kate working as a scientist and Alice's location is  |
| DE09_v3_F2_H1 | abstain | False | True | DE09_v3_F2_H2 | {"j56": 0, "jrefit": 0, "logit": 31} | user mentioned that Alice lives in Cambridge on 2023/03/05 (Sun) 10:00. Answer:  |
| DE11_v0_F1_H2 | other | False | False | DE11_v0_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 31} | Based on the information provided:  - Kate works as a doctor. - Ben lives |
| DE11_v0_F2_H0 | abstain | False | True | DE11_v0_F2_H1 | {"j56": 0, "jrefit": 0, "logit": 31} | Sarah lives in Liverpool. Answer: unknown |
| DE11_v0_F2_H2 | abstain | False | True | DE11_v0_F2_H1 | {"j56": 0, "jrefit": 0, "logit": 31} | Kate works as a doctor, Ben lives in Chester, Alice works as a chef, James lives |
| DE11_v1_F1_H2 | other | False | False | DE11_v1_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 31} | Based on the information provided:  - Ben lives in Chester. - Alice works as |
| DE11_v2_F2_H1 | abstain | False | True | DE11_v2_F2_H0 | {"j56": 32, "jrefit": 32, "logit": 31} | Kate works as a doctor, so she does not live in Liverpool. Sarah lives in Liverp |
| DE11_v2_F2_H2 | abstain | False | True | DE11_v2_F2_H0 | {"j56": null, "jrefit": null, "logit": null} | Kate works as a doctor on 2023/03/01 (Wed), and Sarah lives in Liverpool on 2023 |
| DE11_v3_F2_H1 | abstain | False | True | DE11_v3_F2_H0 | {"j56": 0, "jrefit": 0, "logit": 31} | Sarah lives in Liverpool. Answer: unknown |
| RC01_v0_F1_H0 | other | False | False | - | {} | Based on the information provided:  - Laura's supervisor is David. - David works |
| RC01_v0_F1_H1 | other | False | False | - | {} | Based on the information provided:  - Laura's supervisor is David. - David works |
| RC01_v0_F1_H2 | other | False | False | - | {} | Based on the information provided, we know that Laura's supervisor is David. Fro |
| RC01_v0_F2_H1 | abstain | False | True | RC01_v0_F2_H0 | {"j56": 0, "jrefit": 0, "logit": 31} | user mentioned that Laura's supervisor is David, and David works in Cambridge on |
| RC01_v1_F1_H1 | other | False | False | RC01_v1_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 31} | Based on the information provided:  - Laura's supervisor is Peter. - Peter works |
| RC01_v1_F1_H2 | other | False | False | RC01_v1_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 31} | To determine the city where Laura's supervisor works, we need to identify who La |
| RC01_v1_F2_H1 | abstain | False | True | RC01_v1_F2_H0 | {"j56": 0, "jrefit": 0, "logit": 31} | user mentioned that Laura's supervisor is Peter, and Peter works in Cambridge on |
| RC01_v2_F1_H1 | other | False | False | RC01_v2_F1_H0 | {"j56": 0, "jrefit": 0, "logit": null} | Based on the information provided:  - Paul lives in Chester. - Kate works in |
| RC01_v3_F2_H0 | abstain | False | True | RC01_v3_F2_H1 | {"j56": 0, "jrefit": 0, "logit": 31} | user mentioned David works in Cambridge and Laura's supervisor is David. Answer: |
| RC03_v0_F1_H0 | other | False | False | - | {} | Based on the information provided, we know that Peter's supervisor is James and  |
| RC03_v0_F1_H1 | other | False | False | - | {} | Based on the information provided:  - Ben works as a teacher. - Peter's |
| RC03_v0_F1_H2 | other | False | False | - | {} | Based on the information provided:  - James is Peter's supervisor. - James works |
| RC03_v0_F2_H1 | abstain | False | True | RC03_v0_F2_H0 | {"j56": 0, "jrefit": 0, "logit": 31} | Ben works as a teacher, so he is not Peter's supervisor. James works in Liverpoo |
| RC03_v0_F2_H2 | abstain | False | True | RC03_v0_F2_H0 | {"j56": 0, "jrefit": 0, "logit": 31} | Ben works as a teacher, Emma works as a driver, Kate works in Derby, Ben works i |
| RC03_v1_F1_H1 | other | False | False | RC03_v1_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 31} | Based on the information provided:  - Ben works as a teacher. - Peter's |
| RC03_v1_F1_H2 | other | False | False | RC03_v1_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 31} | Based on the information provided:  - Sarah is Peter's supervisor. - Sarah works |
| RC03_v1_F2_H1 | abstain | False | True | RC03_v1_F2_H0 | {"j56": 0, "jrefit": 0, "logit": 31} | Ben works as a teacher, so he is not Peter's supervisor. Sarah works in Liverpoo |
| RC03_v1_F2_H2 | abstain | False | True | RC03_v1_F2_H0 | {"j56": 0, "jrefit": 0, "logit": null} | Ben works as a teacher, Emma works as a driver, Kate works in Derby, Ben works i |
| RC03_v2_F1_H1 | other | False | False | RC03_v2_F1_H0 | {"j56": null, "jrefit": null, "logit": 32} | Based on the information provided:  - Ben works as a teacher. - Peter's |
| RC03_v2_F2_H2 | abstain | False | True | RC03_v2_F2_H0 | {"j56": 32, "jrefit": 32, "logit": 32} | Emma works as a driver on 2023/03/02 (Thu), so she cannot be Peter's supervisor. |
| RC03_v3_F1_H1 | other | False | False | RC03_v3_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 31} | To determine the city where Peter's supervisor works, we need to follow the info |
| RC03_v3_F1_H2 | other | False | False | RC03_v3_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 31} | To determine the city where Peter's supervisor works, we need to identify who Pe |
| RC03_v3_F2_H0 | abstain | False | True | - | {} | James works in Liverpool. Peter's supervisor is James. Answer: unknown |
| RC03_v3_F2_H1 | abstain | False | True | - | {} | Ben works as a teacher and James works in Liverpool. Peter's supervisor is James |
| RC03_v3_F2_H2 | abstain | False | True | - | {} | Ben works as a teacher and Kate works in Derby. However, no information is provi |
| RC05_v0_F1_H1 | other | False | False | RC05_v0_F1_H0 | {"j56": 9, "jrefit": 10, "logit": 32} | Based on the information provided:  - Laura works in Liverpool. - Rachel works i |
| RC05_v0_F1_H2 | other | False | False | RC05_v0_F1_H0 | {"j56": 9, "jrefit": 10, "logit": 33} | Based on the information provided, we know that Emma works in Chester and that E |
| RC05_v0_F2_H1 | abstain | False | True | RC05_v0_F2_H0 | {"j56": 9, "jrefit": 9, "logit": 31} | Sarah's supervisor is Paul and Paul works in Norwich. Answer: unknown |
| RC05_v0_F2_H2 | abstain | False | True | RC05_v0_F2_H0 | {"j56": 9, "jrefit": 9, "logit": 31} | Sarah's supervisor is Paul and Paul works in Norwich. Answer: unknown |
| RC05_v1_F1_H2 | other | False | False | RC05_v1_F1_H0 | {"j56": 9, "jrefit": 31, "logit": 32} | Based on the information provided, we know that Alice is Sarah's supervisor and  |
| RC05_v1_F2_H1 | abstain | False | True | RC05_v1_F2_H0 | {"j56": 9, "jrefit": 9, "logit": 31} | Sarah's supervisor is Alice. Alice works in Norwich. Answer: unknown |
| RC05_v1_F2_H2 | abstain | False | True | RC05_v1_F2_H0 | {"j56": 9, "jrefit": 9, "logit": 31} | Sarah's supervisor is Alice. Alice works in Norwich on 2023/03/08 (Wed). Answer: |
| RC05_v2_F1_H1 | other | False | False | RC05_v2_F1_H0 | {"j56": 32, "jrefit": 32, "logit": 32} | Based on the information provided:  - Laura works in Liverpool. - Rachel works i |
| RC05_v3_F1_H2 | other | False | False | RC05_v3_F1_H0 | {"j56": 9, "jrefit": 9, "logit": 31} | Based on the information provided:  - Laura works in Liverpool. - Rachel works i |
| RC05_v3_F2_H0 | abstain | False | True | RC05_v3_F2_H1 | {"j56": 9, "jrefit": 9, "logit": 31} | user mentioned Paul works in Norwich and Sarah's supervisor is Paul. Answer: unk |
| RC05_v3_F2_H2 | abstain | False | True | RC05_v3_F2_H1 | {"j56": 9, "jrefit": 9, "logit": null} | Laura works in Liverpool. Rachel works in Manchester. Paul works in Norwich. Sar |
| RC07_v0_F1_H1 | other | False | False | RC07_v0_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 32} | Based on the information provided:  - Alice's supervisor is Rachel. - Rachel wor |
| RC07_v0_F1_H2 | other | False | False | RC07_v0_F1_H0 | {"j56": 1, "jrefit": 2, "logit": 32} | Based on the information provided, we know that Alice's supervisor is Rachel. Fr |
| RC07_v1_F1_H1 | other | False | False | RC07_v1_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 32} | Based on the information provided:  - Alice's supervisor is Ben. - Ben works |
| RC07_v1_F1_H2 | other | False | False | RC07_v1_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 32} | To determine the city where Alice's supervisor works, we need to identify who Al |
| RC07_v3_F1_H2 | other | False | False | RC07_v3_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 32} | Based on the information provided:  - Alice's supervisor is Rachel. - Rachel wor |
| RC07_v3_F2_H0 | abstain | False | True | - | {} | Rachel works in Derby. Alice's supervisor is Rachel. Answer: unknown |
| RC07_v3_F2_H1 | abstain | False | True | - | {} | Rachel works in Derby and is Alice's supervisor. Answer: unknown |
| RC07_v3_F2_H2 | abstain | False | True | - | {} | Sarah lives in Liverpool and is mentioned to be Rachel's supervisor. Rachel work |
| RC09_v0_F1_H1 | other | False | False | RC09_v0_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 1} | Based on the information provided:  - Kate lives in Liverpool. - James lives in |
| RC09_v0_F1_H2 | other | False | False | RC09_v0_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 0} | Based on the information provided, we know that Henry is Ben's supervisor and th |
| RC09_v1_F1_H1 | other | False | False | RC09_v1_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 1} | Based on the information provided:  - Kate lives in Liverpool. - James lives in |
| RC09_v1_F1_H2 | other | False | False | RC09_v1_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 1} | Based on the information provided, we know that Emma is Ben's supervisor and she |
| RC09_v1_F2_H2 | abstain | False | True | RC09_v1_F2_H0 | {"j56": 0, "jrefit": 0, "logit": null} | Kate lives in Liverpool and James lives in Manchester. Ben's supervisor is Emma, |
| RC09_v2_F2_H1 | abstain | False | True | RC09_v2_F2_H0 | {"j56": 0, "jrefit": 9, "logit": 32} | Kate, James, and Rachel live in different cities, so they are not Ben's supervis |
| RC09_v3_F1_H2 | other | False | False | RC09_v3_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 0} | To determine the city where Ben's supervisor works, we need to identify who Ben |
| RC09_v3_F2_H0 | abstain | False | True | RC09_v3_F2_H1 | {"j56": 0, "jrefit": 0, "logit": 0} | Henry works in Leeds. Ben's supervisor is Henry. Answer: unknown |
| RC09_v3_F2_H2 | abstain | False | True | RC09_v3_F2_H1 | {"j56": 0, "jrefit": 0, "logit": 0} | Kate lives in Liverpool. James lives in Manchester. Rachel lives in Norwich. Hen |
| RC11_v0_F1_H1 | other | False | False | RC11_v0_F1_H0 | {"j56": 32, "jrefit": 32, "logit": 32} | Based on the information provided:  - James lives in Cardiff. - Emma's superviso |
| RC11_v0_F1_H2 | other | False | False | RC11_v0_F1_H0 | {"j56": null, "jrefit": null, "logit": null} | Based on the information provided, Emma's supervisor is Kate, and Kate works in |
| RC11_v0_F2_H1 | abstain | False | True | RC11_v0_F2_H0 | {"j56": null, "jrefit": null, "logit": 32} | Emma's supervisor is Kate and Kate works in Chester. Answer: unknown |
| RC11_v1_F1_H1 | other | False | False | RC11_v1_F1_H0 | {"j56": 32, "jrefit": 32, "logit": 32} | Based on the information provided:  - James lives in Cardiff. - Emma's superviso |
| RC11_v1_F1_H2 | other | False | False | RC11_v1_F1_H0 | {"j56": 32, "jrefit": 32, "logit": 32} | Based on the information provided, Emma's supervisor is Laura. We know that Laur |
| RC11_v1_F2_H1 | abstain | False | True | RC11_v1_F2_H0 | {"j56": 32, "jrefit": 32, "logit": 32} | Emma's supervisor is Laura and Laura works in Chester. Answer: unknown |
| RC11_v2_F1_H1 | other | False | False | RC11_v2_F1_H0 | {"j56": 0, "jrefit": 0, "logit": null} | Based on the information provided:  - James lives in Cardiff. - Emma's superviso |
| RC11_v2_F2_H1 | abstain | False | True | RC11_v2_F2_H0 | {"j56": 0, "jrefit": 0, "logit": 32} | Emma's supervisor is Kate. Answer: unknown |
| RC11_v3_F1_H1 | other | False | False | RC11_v3_F1_H0 | {"j56": null, "jrefit": null, "logit": 7} | Based on the information provided:  - James lives in Cardiff. - Kate works in |
| RC11_v3_F2_H0 | abstain | False | True | - | {} | Kate works in Chester. Emma's supervisor is Kate. Answer: unknown |
| RC11_v3_F2_H1 | abstain | False | True | - | {} | Kate works in Chester. Answer: unknown |
| RC11_v3_F2_H2 | abstain | False | True | - | {} | James lives in Cardiff, Peter lives in Cambridge, Henry lives in Brighton, Sarah |
| TU01_v0_F1_H1 | competitor | False | False | TU01_v0_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 31} | Based on the information provided, we know that Peter initially lived in Norwich |
| TU01_v0_F1_H2 | competitor | False | False | TU01_v0_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 31} | Based on the information provided:  - Initially, Peter lived in Norwich. - Later |
| TU01_v0_F2_H1 | abstain | False | True | TU01_v0_F2_H0 | {"j56": 0, "jrefit": 0, "logit": null} | Emma lives in Leeds, Peter lived in Norwich, and later Peter moved to Liverpool. |
| TU01_v0_F2_H2 | abstain | False | True | TU01_v0_F2_H0 | {"j56": 0, "jrefit": 0, "logit": 32} | Emma lives in Leeds, Alice lives in Cardiff, Kate works as a chef, Peter lived i |
| TU01_v1_F1_H1 | competitor | False | False | TU01_v1_F1_H0 | {"j56": 9, "jrefit": 10, "logit": 31} | Based on the information provided:  - Initially, Peter lived in Liverpool. - Lat |
| TU01_v1_F1_H2 | competitor | False | False | TU01_v1_F1_H0 | {"j56": 9, "jrefit": 10, "logit": 31} | Based on the information provided:  - Initially, Peter lived in Liverpool. - Lat |
| TU01_v1_F2_H1 | abstain | False | True | TU01_v1_F2_H0 | {"j56": 9, "jrefit": 9, "logit": null} | Emma lives in Leeds, Peter lived in Liverpool, later Peter moved to Norwich. Ans |
| TU01_v1_F2_H2 | abstain | False | True | TU01_v1_F2_H0 | {"j56": 9, "jrefit": 9, "logit": null} | Emma lives in Leeds, Alice lives in Cardiff, Kate works as a chef, Peter lived i |
| TU01_v2_F1_H0 | other | True | False | TU01_v2_F1_H1 | {"j56": 11, "jrefit": 9, "logit": null} | Before moving to Liverpool, Peter lived in Norwich. |
| TU01_v2_F2_H2 | other | True | False | TU01_v2_F2_H0 | {"j56": 9, "jrefit": 9, "logit": 32} | Emma lives in Leeds. Alice lives in Cardiff. Peter lived in Norwich earlier. Kat |
| TU01_v3_F1_H1 | other | False | False | TU01_v3_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 31} | Based on the information provided:  - Emma lives in Leeds. - Peter used to |
| TU01_v3_F1_H2 | other | False | False | TU01_v3_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 31} | Based on the information provided:  - Emma lives in Leeds. - Alice lives in |
| TU01_v3_F2_H1 | abstain | False | True | TU01_v3_F2_H0 | {"j56": 0, "jrefit": 0, "logit": 32} | Peter now lives in Liverpool. Answer: unknown |
| TU03_v0_F1_H1 | competitor | False | False | TU03_v0_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 30} | Based on the information provided:  1. Initially, Ben lived in Leeds. 2 |
| TU03_v0_F1_H2 | competitor | False | False | TU03_v0_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 30} | Based on the information provided:  - Initially, Ben lived in Leeds. - Later |
| TU03_v0_F2_H1 | abstain | False | True | TU03_v0_F2_H0 | {"j56": 0, "jrefit": 0, "logit": null} | Ben moved from Leeds to Edinburgh on 2023/03/04. Answer: unknown |
| TU03_v0_F2_H2 | abstain | False | True | TU03_v0_F2_H0 | {"j56": 0, "jrefit": 0, "logit": null} | Kate works as a driver. Henry lives in Chester. Peter lives in Manchester. Laura |
| TU03_v1_F1_H1 | competitor | False | False | TU03_v1_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 0} | Based on the information provided:  1. Initially, Ben lived in Edinburgh. 2 |
| TU03_v1_F1_H2 | competitor | False | False | TU03_v1_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 0} | Based on the information provided:  - Initially, Ben lived in Edinburgh. - Later |
| TU03_v1_F2_H1 | abstain | False | True | TU03_v1_F2_H0 | {"j56": 0, "jrefit": 0, "logit": 0} | Ben moved from Edinburgh to Leeds. Answer: unknown |
| TU03_v2_F1_H0 | other | True | False | TU03_v2_F1_H1 | {"j56": 18, "jrefit": 19, "logit": null} | Before moving to Edinburgh, Ben lived in Leeds. |
| TU03_v2_F1_H2 | other | True | False | TU03_v2_F1_H1 | {"j56": 12, "jrefit": 19, "logit": null} | Before moving to Edinburgh, Ben lived in Leeds. |
| TU03_v2_F2_H1 | abstain | False | True | TU03_v2_F2_H0 | {"j56": 0, "jrefit": 0, "logit": 0} | Ben lived in Leeds before moving to Edinburgh. Answer: unknown |
| TU03_v3_F1_H2 | competitor | False | False | TU03_v3_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 30} | Based on the information provided:  - Before that, Ben lived in Leeds. - |
| TU03_v3_F2_H1 | abstain | False | True | TU03_v3_F2_H0 | {"j56": 0, "jrefit": 0, "logit": 30} | Ben now lives in Edinburgh. Answer: unknown |
| TU05_v0_F1_H0 | competitor | False | False | - | {} | Based on the information provided:  1. Initially, David lived in Brighton. 2 |
| TU05_v0_F1_H1 | competitor | False | False | - | {} | Based on the information provided:  1. Initially, David lived in Brighton. 2 |
| TU05_v0_F1_H2 | competitor | False | False | - | {} | Based on the information provided:  - Initially, David lived in Brighton. - Late |
| TU05_v0_F2_H1 | abstain | False | True | TU05_v0_F2_H0 | {"j56": 0, "jrefit": 0, "logit": null} | David lived in Brighton initially. He later moved to Derby. Answer: unknown |
| TU05_v0_F2_H2 | abstain | False | True | TU05_v0_F2_H0 | {"j56": 0, "jrefit": 0, "logit": null} | David lived in Brighton before moving to Derby later. Answer: unknown |
| TU05_v1_F1_H1 | competitor | False | False | TU05_v1_F1_H0 | {"j56": 9, "jrefit": 9, "logit": 31} | Based on the information provided:  - Initially, David lived in Derby. - Later |
| TU05_v1_F1_H2 | competitor | False | False | TU05_v1_F1_H0 | {"j56": 9, "jrefit": 9, "logit": null} | Based on the information provided:  - Initially, David lived in Derby. - Later |
| TU05_v1_F2_H1 | abstain | False | True | TU05_v1_F2_H0 | {"j56": 8, "jrefit": 9, "logit": 31} | user mentioned that David lived in Derby on 2023/03/02 (Thu) 10:00 and later mov |
| TU05_v2_F1_H0 | other | True | False | - | {} | Before moving to Derby, David lived in Brighton. |
| TU05_v2_F1_H1 | other | True | False | - | {} | Before moving to Derby, David lived in Brighton. |
| TU05_v2_F1_H2 | other | True | False | - | {} | Before moving to Derby, David lived in Brighton. |
| TU05_v2_F2_H1 | abstain | False | True | TU05_v2_F2_H0 | {"j56": 8, "jrefit": 9, "logit": null} | user mentioned David lived in Brighton on 2023/03/02 (Thu). Answer: unknown |
| TU05_v2_F2_H2 | abstain | False | True | TU05_v2_F2_H0 | {"j56": 8, "jrefit": 9, "logit": 32} | Henry, Laura, Paul, Rachel, David, Peter, Alice, Kate, Ben - all mentioned their |
| TU05_v3_F1_H1 | competitor | False | False | TU05_v3_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 32} | Based on the information provided:  1. David used to live in Brighton. 2 |
| TU05_v3_F1_H2 | competitor | False | False | TU05_v3_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 32} | Based on the information provided:  - Before that, David lived in Brighton. - |
| TU05_v3_F2_H1 | abstain | False | True | TU05_v3_F2_H0 | {"j56": 0, "jrefit": 0, "logit": 32} | David now lives in Derby. Answer: unknown |
| TU07_v0_F1_H1 | competitor | False | False | TU07_v0_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 31} | Based on the information provided:  - Initially, Sarah lived in Manchester. - La |
| TU07_v0_F1_H2 | other | True | False | TU07_v0_F1_H0 | {"j56": 1, "jrefit": null, "logit": null} | Sarah originally lived in Manchester. Later, it was mentioned that Sarah moved t |
| TU07_v0_F2_H1 | abstain | False | True | TU07_v0_F2_H0 | {"j56": 0, "jrefit": 0, "logit": 32} | Sarah lived in Manchester until 2023/03/02. She later moved to Durham. Answer: u |
| TU07_v1_F1_H0 | competitor | False | False | - | {} | Based on the information provided:  1. Initially, Sarah lived in Durham. 2 |
| TU07_v1_F1_H1 | other | True | False | - | {} | Based on the information provided, Sarah originally lived in Durham but later mo |
| TU07_v1_F1_H2 | other | True | False | - | {} | Sarah originally lived in Durham, but later moved to Manchester. Therefore, Sara |
| TU07_v1_F2_H1 | abstain | False | True | TU07_v1_F2_H0 | {"j56": 0, "jrefit": 0, "logit": null} | Sarah lived in Durham before moving to Manchester. Answer: unknown |
| TU07_v2_F1_H0 | other | True | False | - | {} | Before moving to Durham, Sarah lived in Manchester. |
| TU07_v2_F1_H1 | other | True | False | - | {} | Before moving to Durham, Sarah lived in Manchester. |
| TU07_v2_F1_H2 | other | True | False | - | {} | Before moving to Durham, Sarah lived in Manchester. |
| TU07_v2_F2_H1 | abstain | False | True | TU07_v2_F2_H0 | {"j56": 0, "jrefit": 0, "logit": null} | Sarah lived in Manchester according to the conversation timeline. Answer: unknow |
| TU07_v3_F1_H1 | other | False | False | TU07_v3_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 31} | Based on the information provided:  1. Ben lives in Chester. 2. Sarah |
| TU07_v3_F2_H1 | abstain | False | True | TU07_v3_F2_H0 | {"j56": 0, "jrefit": 0, "logit": 32} | Sarah now lives in Durham. Answer: unknown |
| TU09_v0_F1_H1 | other | True | False | TU09_v0_F1_H0 | {"j56": 30, "jrefit": 30, "logit": 29} | Based on the information provided, Laura originally lived in Cardiff but later m |
| TU09_v0_F1_H2 | other | True | False | TU09_v0_F1_H0 | {"j56": 18, "jrefit": 18, "logit": 29} | Laura moved from Cardiff to Cambridge later, so she now lives in Cambridge. |
| TU09_v0_F2_H1 | abstain | False | True | TU09_v0_F2_H0 | {"j56": 0, "jrefit": 0, "logit": 31} | user mentioned Laura lived in Cardiff on 2023/03/01 (Wed). user later stated Lau |
| TU09_v0_F2_H2 | abstain | False | True | TU09_v0_F2_H0 | {"j56": 0, "jrefit": 0, "logit": 31} | Sarah lives in Manchester, Kate works in Leeds, Paul works as a driver, Laura li |
| TU09_v1_F1_H1 | competitor | False | False | TU09_v1_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 12} | Based on the information provided:  1. Initially, Laura lived in Cambridge. 2 |
| TU09_v1_F1_H2 | other | True | False | TU09_v1_F1_H0 | {"j56": null, "jrefit": null, "logit": 12} | Laura moved from Cambridge to Cardiff later, so she now lives in Cardiff. |
| TU09_v2_F1_H0 | other | True | False | - | {} | Before moving to Cambridge, Laura lived in Cardiff. |
| TU09_v2_F1_H1 | other | True | False | - | {} | Before moving to Cambridge, Laura lived in Cardiff. |
| TU09_v2_F1_H2 | other | True | False | - | {} | Before moving to Cambridge, Laura lived in Cardiff, as stated in the information |
| TU09_v2_F2_H2 | other | False | True | TU09_v2_F2_H0 | {"j56": 0, "jrefit": 0, "logit": 31} | Sarah lives in Manchester and later Laura moved to Cambridge. Laura lived in Man |
| TU11_v0_F1_H1 | other | True | False | TU11_v0_F1_H0 | {"j56": null, "jrefit": null, "logit": null} | Rachel moved from Chester to York, so she no longer lives in Chester. According |
| TU11_v0_F1_H2 | other | True | False | TU11_v0_F1_H0 | {"j56": null, "jrefit": null, "logit": null} | Rachel originally lived in Chester, but later moved to York. Therefore, Rachel n |
| TU11_v1_F1_H1 | other | True | False | TU11_v1_F1_H0 | {"j56": null, "jrefit": null, "logit": null} | Rachel moved from York to Chester, so she no longer lives in York. According |
| TU11_v1_F1_H2 | other | True | False | TU11_v1_F1_H0 | {"j56": null, "jrefit": null, "logit": null} | Rachel moved from York to Chester later, so she now lives in Chester. |
| TU11_v2_F1_H0 | other | True | False | TU11_v2_F1_H1 | {"j56": 30, "jrefit": null, "logit": 31} | Before moving to York, Rachel lived in Chester. |
| TU11_v2_F1_H2 | other | True | False | TU11_v2_F1_H1 | {"j56": 30, "jrefit": null, "logit": null} | Before moving to York, Rachel lived in Chester. |
| TU11_v2_F2_H1 | abstain | False | True | TU11_v2_F2_H0 | {"j56": null, "jrefit": null, "logit": 32} | Sarah lives in Norwich, which was mentioned on 2023/03/01 (Wed). Rachel lived in |
| TU11_v3_F1_H2 | competitor | False | False | TU11_v3_F1_H0 | {"j56": 31, "jrefit": 31, "logit": 31} | Based on the information provided:  - Before that, Rachel lived in Chester. - |

## Stage 3: the §44 rev 8 rows (0 prompts)

| key | §23 | §44 class (1 nat) | d_T fwd | outcome | marker | lens: gold / substitute / unknown best rank at the last prompt token | generated |
|---|---|---|---|---|---|---|---|
