# §45 report (j56-jrefit-logit, half: all)

Descriptive. Readable (fixed, rev 1) = the entity's space-form id in the lens top-10 at a counted position (the last prompt token and the answer span: F1 every generated token, F2 / stage 3 the tokens after 'Answer:'), or its bare-form id at the last prompt token; criterion (a) before the final layer. Rows marked '| ...' are descriptive beside. A miss is 'not recovered by this lens', never 'not computed'. **The held half decides**; dev is shown first for presentation, pooled beside.

Caveat: in F2 the memory lines carry dates that increase with line order, so in TU the timestamp order is a second update cue beside the wording (a format effect, which F2 is there to show).

## Stage 1: solved prompts (502 of 864)

Outcomes: {"gold": 502, "other": 161, "abstain": 154, "competitor": 47}
Answer marker present (format|present): {"F1|False": 432, "F2|False": 3, "F2|True": 429}

### dev (presentation only)

| task | lens | criterion | n | readable | share | Wilson 95% | verdict | first depth q1/q2/q3 |
|---|---|---|---|---|---|---|---|---|
| DE | j56 | a_gold | 93 | 93 | 1.00 | [0.96, 1.00] | recovers | 0.00/0.00/0.86 |
| DE | j56 | a_gold | last prompt token only | 93 | 20 | 0.22 | [0.14, 0.31] | not recovered | 0.92/0.92/0.92 |
| DE | j56 | a_gold | any generated token | 93 | 93 | 1.00 | [0.96, 1.00] | recovers | 0.00/0.00/0.86 |
| DE | j56 | a_gold | either form anywhere | 93 | 93 | 1.00 | [0.96, 1.00] | recovers | 0.00/0.00/0.86 |
| DE | jrefit | a_gold | 93 | 93 | 1.00 | [0.96, 1.00] | recovers | 0.00/0.11/0.86 |
| DE | jrefit | a_gold | last prompt token only | 93 | 20 | 0.22 | [0.14, 0.31] | not recovered | 0.92/0.92/0.92 |
| DE | jrefit | a_gold | any generated token | 93 | 93 | 1.00 | [0.96, 1.00] | recovers | 0.00/0.11/0.83 |
| DE | jrefit | a_gold | either form anywhere | 93 | 93 | 1.00 | [0.96, 1.00] | recovers | 0.00/0.11/0.83 |
| DE | logit | a_gold | 93 | 93 | 1.00 | [0.96, 1.00] | recovers | 0.86/0.89/0.89 |
| DE | logit | a_gold | last prompt token only | 93 | 18 | 0.19 | [0.13, 0.29] | not recovered | 0.92/0.92/0.92 |
| DE | logit | a_gold | any generated token | 93 | 93 | 1.00 | [0.96, 1.00] | recovers | 0.86/0.86/0.89 |
| DE | logit | a_gold | either form anywhere | 93 | 93 | 1.00 | [0.96, 1.00] | recovers | 0.86/0.86/0.89 |
| RC | j56 | a_gold | 88 | 88 | 1.00 | [0.96, 1.00] | recovers | 0.00/0.25/0.89 |
| RC | j56 | a_gold | last prompt token only | 88 | 29 | 0.33 | [0.24, 0.43] | not recovered | 0.92/0.92/0.92 |
| RC | j56 | a_gold | any generated token | 88 | 88 | 1.00 | [0.96, 1.00] | recovers | 0.00/0.25/0.89 |
| RC | j56 | a_gold | either form anywhere | 88 | 88 | 1.00 | [0.96, 1.00] | recovers | 0.00/0.25/0.89 |
| RC | j56 | b_intermediate | 55 | 51 | 0.93 | [0.83, 0.97] | recovers | 0.89/0.89/0.92 |
| RC | j56 | b_intermediate | last prompt token only | 55 | 50 | 0.91 | [0.80, 0.96] | recovers | 0.92/0.92/0.92 |
| RC | j56 | b_intermediate | any generated token | 55 | 55 | 1.00 | [0.93, 1.00] | recovers | 0.64/0.89/0.89 |
| RC | j56 | b_intermediate | either form anywhere | 55 | 55 | 1.00 | [0.93, 1.00] | recovers | 0.49/0.89/0.89 |
| RC | jrefit | a_gold | 88 | 88 | 1.00 | [0.96, 1.00] | recovers | 0.00/0.25/0.89 |
| RC | jrefit | a_gold | last prompt token only | 88 | 29 | 0.33 | [0.24, 0.43] | not recovered | 0.92/0.92/0.92 |
| RC | jrefit | a_gold | any generated token | 88 | 88 | 1.00 | [0.96, 1.00] | recovers | 0.00/0.25/0.87 |
| RC | jrefit | a_gold | either form anywhere | 88 | 88 | 1.00 | [0.96, 1.00] | recovers | 0.00/0.25/0.87 |
| RC | jrefit | b_intermediate | 55 | 52 | 0.95 | [0.85, 0.98] | recovers | 0.89/0.89/0.92 |
| RC | jrefit | b_intermediate | last prompt token only | 55 | 51 | 0.93 | [0.83, 0.97] | recovers | 0.92/0.92/0.92 |
| RC | jrefit | b_intermediate | any generated token | 55 | 55 | 1.00 | [0.93, 1.00] | recovers | 0.86/0.89/0.89 |
| RC | jrefit | b_intermediate | either form anywhere | 55 | 55 | 1.00 | [0.93, 1.00] | recovers | 0.83/0.89/0.89 |
| RC | logit | a_gold | 88 | 88 | 1.00 | [0.96, 1.00] | recovers | 0.86/0.86/0.89 |
| RC | logit | a_gold | last prompt token only | 88 | 20 | 0.23 | [0.15, 0.33] | not recovered | 0.92/0.92/0.92 |
| RC | logit | a_gold | any generated token | 88 | 88 | 1.00 | [0.96, 1.00] | recovers | 0.86/0.86/0.89 |
| RC | logit | a_gold | either form anywhere | 88 | 88 | 1.00 | [0.96, 1.00] | recovers | 0.86/0.86/0.89 |
| RC | logit | b_intermediate | 55 | 47 | 0.85 | [0.74, 0.92] | recovers | 0.89/0.92/0.92 |
| RC | logit | b_intermediate | last prompt token only | 55 | 45 | 0.82 | [0.70, 0.90] | recovers | 0.92/0.92/0.92 |
| RC | logit | b_intermediate | any generated token | 55 | 55 | 1.00 | [0.93, 1.00] | recovers | 0.89/0.89/0.89 |
| RC | logit | b_intermediate | either form anywhere | 55 | 55 | 1.00 | [0.93, 1.00] | recovers | 0.89/0.89/0.89 |
| TU | j56 | a_gold | 75 | 75 | 1.00 | [0.95, 1.00] | recovers | 0.00/0.00/0.25 |
| TU | j56 | a_gold | last prompt token only | 75 | 0 | 0.00 | [0.00, 0.05] | not recovered | - |
| TU | j56 | a_gold | any generated token | 75 | 75 | 1.00 | [0.95, 1.00] | recovers | 0.00/0.00/0.25 |
| TU | j56 | a_gold | either form anywhere | 75 | 75 | 1.00 | [0.95, 1.00] | recovers | 0.00/0.00/0.25 |
| TU | j56 | c_competitor | 75 | 36 | 0.48 | [0.37, 0.59] | not recovered | 0.89/0.92/0.94 |
| TU | j56 | c_competitor | last prompt token only | 75 | 0 | 0.00 | [0.00, 0.05] | not recovered | - |
| TU | j56 | c_competitor | any generated token | 75 | 65 | 0.87 | [0.77, 0.93] | recovers | 0.00/0.25/0.86 |
| TU | j56 | c_competitor | either form anywhere | 75 | 65 | 0.87 | [0.77, 0.93] | recovers | 0.00/0.25/0.86 |
| TU | jrefit | a_gold | 75 | 75 | 1.00 | [0.95, 1.00] | recovers | 0.00/0.00/0.26 |
| TU | jrefit | a_gold | last prompt token only | 75 | 0 | 0.00 | [0.00, 0.05] | not recovered | - |
| TU | jrefit | a_gold | any generated token | 75 | 75 | 1.00 | [0.95, 1.00] | recovers | 0.00/0.00/0.26 |
| TU | jrefit | a_gold | either form anywhere | 75 | 75 | 1.00 | [0.95, 1.00] | recovers | 0.00/0.00/0.26 |
| TU | jrefit | c_competitor | 75 | 35 | 0.47 | [0.36, 0.58] | not recovered | 0.89/0.92/0.94 |
| TU | jrefit | c_competitor | last prompt token only | 75 | 0 | 0.00 | [0.00, 0.05] | not recovered | - |
| TU | jrefit | c_competitor | any generated token | 75 | 64 | 0.85 | [0.76, 0.92] | recovers | 0.00/0.25/0.86 |
| TU | jrefit | c_competitor | either form anywhere | 75 | 64 | 0.85 | [0.76, 0.92] | recovers | 0.00/0.25/0.86 |
| TU | logit | a_gold | 75 | 75 | 1.00 | [0.95, 1.00] | recovers | 0.86/0.89/0.89 |
| TU | logit | a_gold | last prompt token only | 75 | 0 | 0.00 | [0.00, 0.05] | not recovered | - |
| TU | logit | a_gold | any generated token | 75 | 75 | 1.00 | [0.95, 1.00] | recovers | 0.86/0.86/0.86 |
| TU | logit | a_gold | either form anywhere | 75 | 75 | 1.00 | [0.95, 1.00] | recovers | 0.86/0.86/0.86 |
| TU | logit | c_competitor | 75 | 36 | 0.48 | [0.37, 0.59] | not recovered | 0.89/0.92/0.97 |
| TU | logit | c_competitor | last prompt token only | 75 | 0 | 0.00 | [0.00, 0.05] | not recovered | - |
| TU | logit | c_competitor | any generated token | 75 | 64 | 0.85 | [0.76, 0.92] | recovers | 0.86/0.86/0.89 |
| TU | logit | c_competitor | either form anywhere | 75 | 64 | 0.85 | [0.76, 0.92] | recovers | 0.86/0.86/0.89 |

### held (decides)

| task | lens | criterion | n | readable | share | Wilson 95% | verdict | first depth q1/q2/q3 |
|---|---|---|---|---|---|---|---|---|
| DE | j56 | a_gold | 95 | 95 | 1.00 | [0.96, 1.00] | recovers | 0.00/0.00/0.86 |
| DE | j56 | a_gold | last prompt token only | 95 | 23 | 0.24 | [0.17, 0.34] | not recovered | 0.92/0.92/0.92 |
| DE | j56 | a_gold | any generated token | 95 | 95 | 1.00 | [0.96, 1.00] | recovers | 0.00/0.00/0.83 |
| DE | j56 | a_gold | either form anywhere | 95 | 95 | 1.00 | [0.96, 1.00] | recovers | 0.00/0.00/0.83 |
| DE | jrefit | a_gold | 95 | 95 | 1.00 | [0.96, 1.00] | recovers | 0.00/0.00/0.86 |
| DE | jrefit | a_gold | last prompt token only | 95 | 23 | 0.24 | [0.17, 0.34] | not recovered | 0.92/0.92/0.92 |
| DE | jrefit | a_gold | any generated token | 95 | 95 | 1.00 | [0.96, 1.00] | recovers | 0.00/0.00/0.86 |
| DE | jrefit | a_gold | either form anywhere | 95 | 95 | 1.00 | [0.96, 1.00] | recovers | 0.00/0.00/0.85 |
| DE | logit | a_gold | 95 | 95 | 1.00 | [0.96, 1.00] | recovers | 0.86/0.86/0.89 |
| DE | logit | a_gold | last prompt token only | 95 | 21 | 0.22 | [0.15, 0.31] | not recovered | 0.92/0.92/0.94 |
| DE | logit | a_gold | any generated token | 95 | 95 | 1.00 | [0.96, 1.00] | recovers | 0.86/0.86/0.86 |
| DE | logit | a_gold | either form anywhere | 95 | 95 | 1.00 | [0.96, 1.00] | recovers | 0.86/0.86/0.86 |
| RC | j56 | a_gold | 79 | 79 | 1.00 | [0.95, 1.00] | recovers | 0.00/0.00/0.86 |
| RC | j56 | a_gold | last prompt token only | 79 | 24 | 0.30 | [0.21, 0.41] | not recovered | 0.92/0.92/0.92 |
| RC | j56 | a_gold | any generated token | 79 | 79 | 1.00 | [0.95, 1.00] | recovers | 0.00/0.00/0.83 |
| RC | j56 | a_gold | either form anywhere | 79 | 79 | 1.00 | [0.95, 1.00] | recovers | 0.00/0.00/0.83 |
| RC | j56 | b_intermediate | 50 | 47 | 0.94 | [0.84, 0.98] | recovers | 0.89/0.89/0.92 |
| RC | j56 | b_intermediate | last prompt token only | 50 | 43 | 0.86 | [0.74, 0.93] | recovers | 0.92/0.92/0.92 |
| RC | j56 | b_intermediate | any generated token | 50 | 50 | 1.00 | [0.93, 1.00] | recovers | 0.00/0.86/0.89 |
| RC | j56 | b_intermediate | either form anywhere | 50 | 50 | 1.00 | [0.93, 1.00] | recovers | 0.00/0.86/0.89 |
| RC | jrefit | a_gold | 79 | 79 | 1.00 | [0.95, 1.00] | recovers | 0.00/0.00/0.86 |
| RC | jrefit | a_gold | last prompt token only | 79 | 24 | 0.30 | [0.21, 0.41] | not recovered | 0.92/0.92/0.92 |
| RC | jrefit | a_gold | any generated token | 79 | 79 | 1.00 | [0.95, 1.00] | recovers | 0.00/0.00/0.85 |
| RC | jrefit | a_gold | either form anywhere | 79 | 79 | 1.00 | [0.95, 1.00] | recovers | 0.00/0.00/0.85 |
| RC | jrefit | b_intermediate | 50 | 47 | 0.94 | [0.84, 0.98] | recovers | 0.89/0.89/0.92 |
| RC | jrefit | b_intermediate | last prompt token only | 50 | 43 | 0.86 | [0.74, 0.93] | recovers | 0.92/0.92/0.92 |
| RC | jrefit | b_intermediate | any generated token | 50 | 50 | 1.00 | [0.93, 1.00] | recovers | 0.00/0.89/0.89 |
| RC | jrefit | b_intermediate | either form anywhere | 50 | 50 | 1.00 | [0.93, 1.00] | recovers | 0.00/0.89/0.89 |
| RC | logit | a_gold | 79 | 79 | 1.00 | [0.95, 1.00] | recovers | 0.86/0.89/0.89 |
| RC | logit | a_gold | last prompt token only | 79 | 16 | 0.20 | [0.13, 0.30] | not recovered | 0.92/0.92/0.92 |
| RC | logit | a_gold | any generated token | 79 | 79 | 1.00 | [0.95, 1.00] | recovers | 0.86/0.86/0.89 |
| RC | logit | a_gold | either form anywhere | 79 | 79 | 1.00 | [0.95, 1.00] | recovers | 0.86/0.86/0.89 |
| RC | logit | b_intermediate | 50 | 44 | 0.88 | [0.76, 0.94] | recovers | 0.89/0.92/0.92 |
| RC | logit | b_intermediate | last prompt token only | 50 | 40 | 0.80 | [0.67, 0.89] | recovers | 0.92/0.92/0.92 |
| RC | logit | b_intermediate | any generated token | 50 | 50 | 1.00 | [0.93, 1.00] | recovers | 0.89/0.89/0.89 |
| RC | logit | b_intermediate | either form anywhere | 50 | 50 | 1.00 | [0.93, 1.00] | recovers | 0.89/0.89/0.89 |
| TU | j56 | a_gold | 72 | 72 | 1.00 | [0.95, 1.00] | recovers | 0.00/0.00/0.25 |
| TU | j56 | a_gold | last prompt token only | 72 | 0 | 0.00 | [0.00, 0.05] | not recovered | - |
| TU | j56 | a_gold | any generated token | 72 | 72 | 1.00 | [0.95, 1.00] | recovers | 0.00/0.00/0.25 |
| TU | j56 | a_gold | either form anywhere | 72 | 72 | 1.00 | [0.95, 1.00] | recovers | 0.00/0.00/0.25 |
| TU | j56 | c_competitor | 72 | 34 | 0.47 | [0.36, 0.59] | not recovered | 0.89/0.92/0.97 |
| TU | j56 | c_competitor | last prompt token only | 72 | 0 | 0.00 | [0.00, 0.05] | not recovered | - |
| TU | j56 | c_competitor | any generated token | 72 | 63 | 0.88 | [0.78, 0.93] | recovers | 0.00/0.28/0.85 |
| TU | j56 | c_competitor | either form anywhere | 72 | 63 | 0.88 | [0.78, 0.93] | recovers | 0.00/0.28/0.85 |
| TU | jrefit | a_gold | 72 | 72 | 1.00 | [0.95, 1.00] | recovers | 0.00/0.00/0.26 |
| TU | jrefit | a_gold | last prompt token only | 72 | 0 | 0.00 | [0.00, 0.05] | not recovered | - |
| TU | jrefit | a_gold | any generated token | 72 | 72 | 1.00 | [0.95, 1.00] | recovers | 0.00/0.00/0.26 |
| TU | jrefit | a_gold | either form anywhere | 72 | 72 | 1.00 | [0.95, 1.00] | recovers | 0.00/0.00/0.26 |
| TU | jrefit | c_competitor | 72 | 35 | 0.49 | [0.37, 0.60] | not recovered | 0.89/0.92/0.97 |
| TU | jrefit | c_competitor | last prompt token only | 72 | 0 | 0.00 | [0.00, 0.05] | not recovered | - |
| TU | jrefit | c_competitor | any generated token | 72 | 64 | 0.89 | [0.80, 0.94] | recovers | 0.00/0.44/0.84 |
| TU | jrefit | c_competitor | either form anywhere | 72 | 64 | 0.89 | [0.80, 0.94] | recovers | 0.00/0.44/0.84 |
| TU | logit | a_gold | 72 | 72 | 1.00 | [0.95, 1.00] | recovers | 0.86/0.86/0.89 |
| TU | logit | a_gold | last prompt token only | 72 | 0 | 0.00 | [0.00, 0.05] | not recovered | - |
| TU | logit | a_gold | any generated token | 72 | 72 | 1.00 | [0.95, 1.00] | recovers | 0.86/0.86/0.86 |
| TU | logit | a_gold | either form anywhere | 72 | 72 | 1.00 | [0.95, 1.00] | recovers | 0.86/0.86/0.86 |
| TU | logit | c_competitor | 72 | 31 | 0.43 | [0.32, 0.55] | not recovered | 0.89/0.92/0.97 |
| TU | logit | c_competitor | last prompt token only | 72 | 0 | 0.00 | [0.00, 0.05] | not recovered | - |
| TU | logit | c_competitor | any generated token | 72 | 59 | 0.82 | [0.72, 0.89] | recovers | 0.86/0.86/0.89 |
| TU | logit | c_competitor | either form anywhere | 72 | 59 | 0.82 | [0.72, 0.89] | recovers | 0.86/0.86/0.89 |

### pooled (beside)

| task | lens | criterion | n | readable | share | Wilson 95% | verdict | first depth q1/q2/q3 |
|---|---|---|---|---|---|---|---|---|
| DE | j56 | a_gold | 188 | 188 | 1.00 | [0.98, 1.00] | recovers | 0.00/0.00/0.86 |
| DE | j56 | a_gold | last prompt token only | 188 | 43 | 0.23 | [0.17, 0.29] | not recovered | 0.92/0.92/0.92 |
| DE | j56 | a_gold | any generated token | 188 | 188 | 1.00 | [0.98, 1.00] | recovers | 0.00/0.00/0.84 |
| DE | j56 | a_gold | either form anywhere | 188 | 188 | 1.00 | [0.98, 1.00] | recovers | 0.00/0.00/0.83 |
| DE | jrefit | a_gold | 188 | 188 | 1.00 | [0.98, 1.00] | recovers | 0.00/0.00/0.86 |
| DE | jrefit | a_gold | last prompt token only | 188 | 43 | 0.23 | [0.17, 0.29] | not recovered | 0.92/0.92/0.92 |
| DE | jrefit | a_gold | any generated token | 188 | 188 | 1.00 | [0.98, 1.00] | recovers | 0.00/0.00/0.83 |
| DE | jrefit | a_gold | either form anywhere | 188 | 188 | 1.00 | [0.98, 1.00] | recovers | 0.00/0.00/0.83 |
| DE | logit | a_gold | 188 | 188 | 1.00 | [0.98, 1.00] | recovers | 0.86/0.86/0.89 |
| DE | logit | a_gold | last prompt token only | 188 | 39 | 0.21 | [0.16, 0.27] | not recovered | 0.92/0.92/0.92 |
| DE | logit | a_gold | any generated token | 188 | 188 | 1.00 | [0.98, 1.00] | recovers | 0.86/0.86/0.89 |
| DE | logit | a_gold | either form anywhere | 188 | 188 | 1.00 | [0.98, 1.00] | recovers | 0.86/0.86/0.89 |
| RC | j56 | a_gold | 167 | 167 | 1.00 | [0.98, 1.00] | recovers | 0.00/0.00/0.88 |
| RC | j56 | a_gold | last prompt token only | 167 | 53 | 0.32 | [0.25, 0.39] | not recovered | 0.92/0.92/0.92 |
| RC | j56 | a_gold | any generated token | 167 | 167 | 1.00 | [0.98, 1.00] | recovers | 0.00/0.00/0.86 |
| RC | j56 | a_gold | either form anywhere | 167 | 167 | 1.00 | [0.98, 1.00] | recovers | 0.00/0.00/0.86 |
| RC | j56 | b_intermediate | 105 | 98 | 0.93 | [0.87, 0.97] | recovers | 0.89/0.89/0.92 |
| RC | j56 | b_intermediate | last prompt token only | 105 | 93 | 0.89 | [0.81, 0.93] | recovers | 0.92/0.92/0.92 |
| RC | j56 | b_intermediate | any generated token | 105 | 105 | 1.00 | [0.96, 1.00] | recovers | 0.00/0.89/0.89 |
| RC | j56 | b_intermediate | either form anywhere | 105 | 105 | 1.00 | [0.96, 1.00] | recovers | 0.00/0.89/0.89 |
| RC | jrefit | a_gold | 167 | 167 | 1.00 | [0.98, 1.00] | recovers | 0.00/0.00/0.86 |
| RC | jrefit | a_gold | last prompt token only | 167 | 53 | 0.32 | [0.25, 0.39] | not recovered | 0.92/0.92/0.92 |
| RC | jrefit | a_gold | any generated token | 167 | 167 | 1.00 | [0.98, 1.00] | recovers | 0.00/0.00/0.86 |
| RC | jrefit | a_gold | either form anywhere | 167 | 167 | 1.00 | [0.98, 1.00] | recovers | 0.00/0.00/0.86 |
| RC | jrefit | b_intermediate | 105 | 99 | 0.94 | [0.88, 0.97] | recovers | 0.89/0.89/0.92 |
| RC | jrefit | b_intermediate | last prompt token only | 105 | 94 | 0.90 | [0.82, 0.94] | recovers | 0.92/0.92/0.92 |
| RC | jrefit | b_intermediate | any generated token | 105 | 105 | 1.00 | [0.96, 1.00] | recovers | 0.44/0.89/0.89 |
| RC | jrefit | b_intermediate | either form anywhere | 105 | 105 | 1.00 | [0.96, 1.00] | recovers | 0.44/0.89/0.89 |
| RC | logit | a_gold | 167 | 167 | 1.00 | [0.98, 1.00] | recovers | 0.86/0.86/0.89 |
| RC | logit | a_gold | last prompt token only | 167 | 36 | 0.22 | [0.16, 0.28] | not recovered | 0.92/0.92/0.92 |
| RC | logit | a_gold | any generated token | 167 | 167 | 1.00 | [0.98, 1.00] | recovers | 0.86/0.86/0.89 |
| RC | logit | a_gold | either form anywhere | 167 | 167 | 1.00 | [0.98, 1.00] | recovers | 0.86/0.86/0.89 |
| RC | logit | b_intermediate | 105 | 91 | 0.87 | [0.79, 0.92] | recovers | 0.89/0.92/0.92 |
| RC | logit | b_intermediate | last prompt token only | 105 | 85 | 0.81 | [0.72, 0.87] | recovers | 0.92/0.92/0.92 |
| RC | logit | b_intermediate | any generated token | 105 | 105 | 1.00 | [0.96, 1.00] | recovers | 0.89/0.89/0.89 |
| RC | logit | b_intermediate | either form anywhere | 105 | 105 | 1.00 | [0.96, 1.00] | recovers | 0.89/0.89/0.89 |
| TU | j56 | a_gold | 147 | 147 | 1.00 | [0.97, 1.00] | recovers | 0.00/0.00/0.25 |
| TU | j56 | a_gold | last prompt token only | 147 | 0 | 0.00 | [0.00, 0.03] | not recovered | - |
| TU | j56 | a_gold | any generated token | 147 | 147 | 1.00 | [0.97, 1.00] | recovers | 0.00/0.00/0.25 |
| TU | j56 | a_gold | either form anywhere | 147 | 147 | 1.00 | [0.97, 1.00] | recovers | 0.00/0.00/0.25 |
| TU | j56 | c_competitor | 147 | 70 | 0.48 | [0.40, 0.56] | not recovered | 0.89/0.92/0.97 |
| TU | j56 | c_competitor | last prompt token only | 147 | 0 | 0.00 | [0.00, 0.03] | not recovered | - |
| TU | j56 | c_competitor | any generated token | 147 | 128 | 0.87 | [0.81, 0.92] | recovers | 0.00/0.25/0.86 |
| TU | j56 | c_competitor | either form anywhere | 147 | 128 | 0.87 | [0.81, 0.92] | recovers | 0.00/0.25/0.86 |
| TU | jrefit | a_gold | 147 | 147 | 1.00 | [0.97, 1.00] | recovers | 0.00/0.00/0.26 |
| TU | jrefit | a_gold | last prompt token only | 147 | 0 | 0.00 | [0.00, 0.03] | not recovered | - |
| TU | jrefit | a_gold | any generated token | 147 | 147 | 1.00 | [0.97, 1.00] | recovers | 0.00/0.00/0.26 |
| TU | jrefit | a_gold | either form anywhere | 147 | 147 | 1.00 | [0.97, 1.00] | recovers | 0.00/0.00/0.26 |
| TU | jrefit | c_competitor | 147 | 70 | 0.48 | [0.40, 0.56] | not recovered | 0.89/0.92/0.97 |
| TU | jrefit | c_competitor | last prompt token only | 147 | 0 | 0.00 | [0.00, 0.03] | not recovered | - |
| TU | jrefit | c_competitor | any generated token | 147 | 128 | 0.87 | [0.81, 0.92] | recovers | 0.00/0.28/0.86 |
| TU | jrefit | c_competitor | either form anywhere | 147 | 128 | 0.87 | [0.81, 0.92] | recovers | 0.00/0.28/0.86 |
| TU | logit | a_gold | 147 | 147 | 1.00 | [0.97, 1.00] | recovers | 0.86/0.86/0.89 |
| TU | logit | a_gold | last prompt token only | 147 | 0 | 0.00 | [0.00, 0.03] | not recovered | - |
| TU | logit | a_gold | any generated token | 147 | 147 | 1.00 | [0.97, 1.00] | recovers | 0.86/0.86/0.86 |
| TU | logit | a_gold | either form anywhere | 147 | 147 | 1.00 | [0.97, 1.00] | recovers | 0.86/0.86/0.86 |
| TU | logit | c_competitor | 147 | 67 | 0.46 | [0.38, 0.54] | not recovered | 0.89/0.92/0.97 |
| TU | logit | c_competitor | last prompt token only | 147 | 0 | 0.00 | [0.00, 0.03] | not recovered | - |
| TU | logit | c_competitor | any generated token | 147 | 123 | 0.84 | [0.77, 0.89] | recovers | 0.86/0.86/0.89 |
| TU | logit | c_competitor | either form anywhere | 147 | 123 | 0.84 | [0.77, 0.89] | recovers | 0.86/0.86/0.89 |

### Fit stability (j56 vs jrefit, before the final layer)

- solved prompts: agreement 0.997 over 1910 calls -> **stable**; Spearman median 0.980 (quartiles [0.9728321481894842, 0.9801501829017221, 0.9879227463900553], n=1910)
- all prompts (beside): agreement 0.997 over 3312 calls -> **stable**; Spearman median 0.980 (quartiles [0.9706553005984945, 0.9797916381753577, 0.9880798970153453], n=3312)

## Stage 2: matched failures (362, 152 at H2, 287 paired)

- j56: on the matched failures the gold is not readable before the final layer on >= 0.70 of them (j56) (share 0.64, Wilson [0.58, 0.68], n=362)
- jrefit: on the matched failures the gold is not readable before the final layer on >= 0.70 of them (jrefit) (share 0.64, Wilson [0.58, 0.68], n=362)
- logit: on the matched failures the gold is not readable before the final layer on >= 0.70 of them (logit) (share 0.62, Wilson [0.57, 0.67], n=362)

Failures by task, hardness and outcome: {"DE|H0|abstain": 17, "DE|H1|abstain": 22, "DE|H1|other": 14, "DE|H2|abstain": 19, "DE|H2|other": 28, "RC|H0|abstain": 12, "RC|H0|other": 4, "RC|H1|abstain": 22, "RC|H1|other": 35, "RC|H2|abstain": 19, "RC|H2|other": 29, "TU|H0|abstain": 2, "TU|H0|competitor": 5, "TU|H0|other": 12, "TU|H1|abstain": 28, "TU|H1|competitor": 21, "TU|H1|other": 16, "TU|H2|abstain": 13, "TU|H2|competitor": 21, "TU|H2|other": 23}

| failure | outcome | ambiguous | marker | pair | departure layer per lens | generated |
|---|---|---|---|---|---|---|
| DE00_v0_F1_H2 | other | False | False | DE00_v0_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 31} | Based on the information provided:  - Kate works as a teacher. - Ben lives |
| DE00_v0_F2_H1 | abstain | False | True | DE00_v0_F2_H0 | {"j56": 0, "jrefit": 0, "logit": 31} | Kate works as a teacher, Emma lives in Edinburgh, and there is no information ab |
| DE00_v0_F2_H2 | abstain | False | True | DE00_v0_F2_H0 | {"j56": 0, "jrefit": 0, "logit": 31} | Emma lives in Edinburgh. Answer: unknown |
| DE00_v1_F1_H2 | other | False | False | DE00_v1_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 0} | Based on the information provided:  - Ben lives in Cambridge. - Henry lives in |
| DE00_v1_F2_H1 | abstain | False | True | DE00_v1_F2_H0 | {"j56": 0, "jrefit": 0, "logit": 32} | Emma lives in Leeds. Answer: unknown |
| DE00_v1_F2_H2 | abstain | False | True | DE00_v1_F2_H0 | {"j56": 0, "jrefit": 0, "logit": 32} | Emma lives in Leeds. Answer: unknown |
| DE00_v2_F1_H2 | other | False | False | DE00_v2_F1_H0 | {"j56": null, "jrefit": null, "logit": null} | Based on the information provided:  - Ben lives in Cambridge. - Henry lives in |
| DE00_v2_F2_H1 | abstain | False | True | DE00_v2_F2_H0 | {"j56": 32, "jrefit": 32, "logit": null} | Kate does not live in Edinburgh. Emma lives in Edinburgh.   Answer: unknown |
| DE00_v3_F2_H0 | abstain | False | True | DE00_v3_F2_H2 | {"j56": 0, "jrefit": 0, "logit": 31} | user mentioned that Emma lives in Edinburgh on 2023/03/02 (Thu). Answer: unknown |
| DE00_v3_F2_H1 | abstain | False | True | DE00_v3_F2_H2 | {"j56": 0, "jrefit": 0, "logit": 31} | user mentioned that Emma lives in Edinburgh on 2023/03/04 (Sat). Answer: unknown |
| DE01_v0_F1_H2 | other | False | False | DE01_v0_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 31} | Based on the information provided, there is no mention of where Paul lives. The |
| DE01_v1_F1_H2 | other | False | False | DE01_v1_F1_H0 | {"j56": 9, "jrefit": 9, "logit": 31} | Based on the information provided, there is no mention of where Paul lives. The |
| DE01_v1_F2_H0 | abstain | False | True | - | {} | user stated that Paul lives in Norwich. Answer: unknown |
| DE01_v1_F2_H1 | abstain | False | True | - | {} | Kate works as a cook. Sarah works as a designer. Paul lives in Norwich. Answer:  |
| DE01_v1_F2_H2 | abstain | False | True | - | {} | Kate works as a cook. Sarah works as a designer. Rachel works in Chester. Alice  |
| DE01_v2_F1_H1 | other | False | False | DE01_v2_F1_H0 | {"j56": null, "jrefit": null, "logit": null} | Based on the information provided:  - Kate works as a cook. - Sarah works |
| DE01_v3_F1_H2 | other | False | False | DE01_v3_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 31} | Based on the information provided, there is no mention of where Paul lives. The |
| DE01_v3_F2_H2 | other | True | False | DE01_v3_F2_H0 | {"j56": 0, "jrefit": 0, "logit": 32} | Kate - works as a cook   James - works in York   Henry - lives in Manchester   D |
| DE02_v0_F1_H1 | other | False | False | DE02_v0_F1_H0 | {"j56": 9, "jrefit": 10, "logit": 31} | Based on the information provided:  - Kate works as an editor. - Paul lives |
| DE02_v0_F1_H2 | other | False | False | DE02_v0_F1_H0 | {"j56": 9, "jrefit": 10, "logit": 31} | Based on the information provided:  - Kate works as an editor and lives in Brigh |
| DE02_v0_F2_H0 | abstain | False | True | DE02_v0_F2_H1 | {"j56": 9, "jrefit": 9, "logit": null} | user stated that Peter lives in Norwich. Answer: unknown |
| DE02_v1_F1_H1 | other | False | False | DE02_v1_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 30} | Based on the information provided:  - Kate works as an editor. - Paul lives |
| DE02_v1_F1_H2 | other | False | False | DE02_v1_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 30} | Based on the information provided:  - Kate works as an editor and lives in Brigh |
| DE02_v2_F1_H1 | other | False | False | DE02_v2_F1_H0 | {"j56": null, "jrefit": null, "logit": 32} | Based on the information provided:  - Kate works as an editor. - Paul lives |
| DE02_v2_F1_H2 | other | False | False | DE02_v2_F1_H0 | {"j56": null, "jrefit": null, "logit": 32} | Based on the information provided:  - Paul lives in Durham. - Emma works in |
| DE02_v2_F2_H0 | abstain | False | True | DE02_v2_F2_H1 | {"j56": null, "jrefit": null, "logit": null} | user mentioned that Peter lives in Norwich. Answer: unknown |
| DE02_v3_F1_H2 | other | False | False | DE02_v3_F1_H0 | {"j56": 9, "jrefit": 10, "logit": 31} | Based on the information provided:  - Paul lives in Durham. - Sarah lives in |
| DE02_v3_F2_H0 | abstain | False | True | - | {} | user mentioned that Peter lives in Norwich on 2023/03/01 (Wed). Answer: unknown |
| DE02_v3_F2_H1 | abstain | False | True | - | {} | Peter lives in Norwich. Answer: unknown |
| DE02_v3_F2_H2 | abstain | False | True | - | {} | Paul lives in Durham. Sarah lives in Liverpool. Peter lives in Norwich. Laura li |
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
| DE04_v0_F2_H2 | abstain | False | True | DE04_v0_F2_H0 | {"j56": 0, "jrefit": 0, "logit": 32} | Henry lives in Durham based on the information provided. Answer: unknown |
| DE04_v2_F2_H0 | abstain | False | True | DE04_v2_F2_H1 | {"j56": 0, "jrefit": 4, "logit": null} | user mentioned that Henry lives in Durham. Answer: unknown |
| DE04_v3_F1_H2 | other | False | False | DE04_v3_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 31} | Based on the information provided, James works as a cook and James lives in Camb |
| DE04_v3_F2_H1 | abstain | False | True | DE04_v3_F2_H0 | {"j56": 0, "jrefit": 0, "logit": 33} | Henry lives in Durham. Answer: unknown |
| DE05_v0_F1_H1 | other | False | False | DE05_v0_F1_H0 | {"j56": 32, "jrefit": 32, "logit": 31} | Based on the information provided:  - Alice lives in Liverpool. - Sarah works as |
| DE05_v0_F1_H2 | other | False | False | DE05_v0_F1_H0 | {"j56": 32, "jrefit": 32, "logit": 31} | Based on the information provided:  - Alice lives in Liverpool and works in Leed |
| DE05_v1_F1_H1 | other | False | False | DE05_v1_F1_H0 | {"j56": 9, "jrefit": 9, "logit": 31} | Based on the information provided:  - Alice lives in Liverpool. - Sarah works as |
| DE05_v1_F1_H2 | other | False | False | DE05_v1_F1_H0 | {"j56": 9, "jrefit": 9, "logit": 31} | Based on the information provided:  - Alice lives in Liverpool and works in Leed |
| DE05_v1_F2_H2 | abstain | False | True | DE05_v1_F2_H0 | {"j56": 8, "jrefit": 8, "logit": 32} | David lives in Brighton. Answer: unknown |
| DE05_v2_F1_H1 | other | False | False | DE05_v2_F1_H0 | {"j56": null, "jrefit": null, "logit": null} | Based on the information provided:  - Alice lives in Liverpool. - Sarah works as |
| DE05_v2_F1_H2 | other | False | False | DE05_v2_F1_H0 | {"j56": null, "jrefit": null, "logit": null} | Based on the information provided:  - Alice lives in Liverpool. - Sarah works as |
| DE05_v2_F2_H2 | abstain | False | True | DE05_v2_F2_H0 | {"j56": 0, "jrefit": 32, "logit": null} | Alice - lives in York Answer: unknown |
| DE05_v3_F1_H2 | other | False | False | DE05_v3_F1_H0 | {"j56": null, "jrefit": null, "logit": null} | Based on the information provided:  - Alice lives in Liverpool. - David lives in |
| DE06_v0_F2_H1 | abstain | False | True | DE06_v0_F2_H0 | {"j56": 8, "jrefit": 8, "logit": 32} | Sarah works as a doctor and Laura lives in Brighton. Answer: unknown |
| DE06_v0_F2_H2 | abstain | False | True | DE06_v0_F2_H0 | {"j56": 8, "jrefit": 8, "logit": null} | user - Laura lives in Brighton Answer: unknown |
| DE06_v1_F2_H0 | abstain | False | True | - | {} | user stated that Laura lives in Manchester. Answer: unknown |
| DE06_v1_F2_H1 | abstain | False | True | - | {} | Sarah works as a doctor and Laura lives in Manchester. Answer: unknown |
| DE06_v1_F2_H2 | abstain | False | True | - | {} | user - Laura lives in Manchester Answer: unknown |
| DE06_v2_F2_H1 | abstain | False | True | DE06_v2_F2_H0 | {"j56": null, "jrefit": null, "logit": null} | Sarah works as a doctor and Laura lives in Brighton. Answer: unknown |
| DE06_v3_F2_H0 | abstain | False | True | - | {} | user mentioned that Laura lives in Brighton on 2023/03/01 (Wed). Answer: unknown |
| DE06_v3_F2_H1 | abstain | False | True | - | {} | Sarah works as a doctor and Laura lives in Brighton. Answer: unknown |
| DE06_v3_F2_H2 | abstain | False | True | - | {} | user mentioned Laura lives in Brighton on 2023/03/05 (Sun). Answer: unknown |
| DE07_v0_F1_H1 | other | False | False | DE07_v0_F1_H0 | {"j56": 0, "jrefit": 0, "logit": null} | Based on the information provided:  - Peter lives in Cambridge. - James lives in |
| DE07_v0_F2_H0 | abstain | False | True | DE07_v0_F2_H1 | {"j56": 0, "jrefit": 0, "logit": 31} | user stated that James lives in Manchester. Answer: unknown |
| DE07_v1_F2_H1 | abstain | False | True | DE07_v1_F2_H0 | {"j56": 32, "jrefit": 32, "logit": 32} | James lives in Chester. Answer: unknown |
| DE07_v3_F2_H1 | abstain | False | True | DE07_v3_F2_H0 | {"j56": 0, "jrefit": 0, "logit": 31} | James lives in Manchester. Answer: unknown |
| DE08_v0_F1_H2 | other | False | False | DE08_v0_F1_H0 | {"j56": 32, "jrefit": 32, "logit": 32} | Based on the information provided:  - Laura lives in Cardiff. - David lives in |
| DE08_v1_F1_H2 | other | False | False | DE08_v1_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 31} | Based on the information provided:  - Laura lives in Cardiff. - David lives in |
| DE08_v2_F1_H2 | other | False | False | DE08_v2_F1_H0 | {"j56": null, "jrefit": null, "logit": 32} | Based on the information provided:  - Laura lives in Cardiff. - David lives in |
| DE08_v3_F1_H2 | other | False | False | DE08_v3_F1_H0 | {"j56": 31, "jrefit": 31, "logit": 32} | Based on the information provided:  - David lives in Liverpool. - Alice works as |
| DE08_v3_F2_H2 | abstain | False | True | DE08_v3_F2_H0 | {"j56": 32, "jrefit": 32, "logit": 32} | Rachel lives in Chester. Answer: unknown |
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
| DE10_v0_F1_H2 | other | False | False | DE10_v0_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 32} | Based on the information provided, there is no mention of where Kate lives. The |
| DE10_v0_F2_H0 | abstain | False | True | DE10_v0_F2_H1 | {"j56": 0, "jrefit": 0, "logit": 32} | Kate lives in Derby. Answer: unknown |
| DE10_v0_F2_H2 | abstain | False | True | DE10_v0_F2_H1 | {"j56": 0, "jrefit": 0, "logit": 32} | user mentioned that Kate lives in Derby on 2023/03/07 (Tue). Answer: unknown |
| DE10_v1_F1_H2 | other | False | False | DE10_v1_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 31} | Based on the information provided, there is no mention of where Kate lives. The |
| DE10_v1_F2_H0 | abstain | False | True | DE10_v1_F2_H1 | {"j56": 0, "jrefit": 0, "logit": 31} | Kate lives in Liverpool. Answer: unknown |
| DE10_v1_F2_H2 | abstain | False | True | DE10_v1_F2_H1 | {"j56": 0, "jrefit": 0, "logit": 31} | user mentioned Kate lives in Liverpool on 2023/03/07 (Tue). Answer: unknown |
| DE10_v2_F1_H1 | other | False | False | DE10_v2_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 32} | Based on the information provided:  - Peter works as a doctor. - James works |
| DE10_v2_F2_H0 | abstain | False | True | DE10_v2_F2_H1 | {"j56": 0, "jrefit": 0, "logit": null} | user mentioned that Kate lives in Derby. Answer: unknown |
| DE10_v3_F2_H0 | abstain | False | True | - | {} | Kate lives in Derby. Answer: unknown |
| DE10_v3_F2_H1 | abstain | False | True | - | {} | Kate lives in Derby. Answer: unknown |
| DE10_v3_F2_H2 | abstain | False | True | - | {} | Kate lives in Derby. Answer: unknown |
| DE11_v0_F1_H2 | other | False | False | DE11_v0_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 31} | Based on the information provided:  - Kate works as a doctor. - Ben lives |
| DE11_v0_F2_H0 | abstain | False | True | DE11_v0_F2_H1 | {"j56": 0, "jrefit": 0, "logit": 31} | Sarah lives in Liverpool. Answer: unknown |
| DE11_v0_F2_H2 | abstain | False | True | DE11_v0_F2_H1 | {"j56": 0, "jrefit": 0, "logit": 31} | Kate works as a doctor, Ben lives in Chester, Alice works as a chef, James lives |
| DE11_v1_F1_H2 | other | False | False | DE11_v1_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 31} | Based on the information provided:  - Ben lives in Chester. - Alice works as |
| DE11_v2_F2_H1 | abstain | False | True | DE11_v2_F2_H0 | {"j56": 32, "jrefit": 32, "logit": 31} | Kate works as a doctor, so she does not live in Liverpool. Sarah lives in Liverp |
| DE11_v2_F2_H2 | abstain | False | True | DE11_v2_F2_H0 | {"j56": null, "jrefit": null, "logit": null} | Kate works as a doctor on 2023/03/01 (Wed), and Sarah lives in Liverpool on 2023 |
| DE11_v3_F2_H1 | abstain | False | True | DE11_v3_F2_H0 | {"j56": 0, "jrefit": 0, "logit": 31} | Sarah lives in Liverpool. Answer: unknown |
| RC00_v0_F1_H1 | other | False | False | RC00_v0_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 31} | Based on the information provided:  - Laura is Kate's supervisor. - Laura works |
| RC00_v0_F1_H2 | other | False | False | RC00_v0_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 31} | Based on the information provided:  - Sarah lives in Liverpool and works in Brig |
| RC00_v1_F1_H0 | other | False | False | - | {} | Based on the information provided:  - Kate's supervisor is David. - David works |
| RC00_v1_F1_H1 | other | False | False | - | {} | Based on the information provided:  - Sarah lives in Liverpool. - Paul lives in |
| RC00_v1_F1_H2 | other | False | False | - | {} | Based on the information provided, we know that Kate's supervisor is David. Howe |
| RC00_v1_F2_H1 | abstain | False | True | RC00_v1_F2_H0 | {"j56": 0, "jrefit": 0, "logit": 32} | Kate's supervisor is David and David works in Durham. Answer: unknown |
| RC00_v1_F2_H2 | abstain | False | True | RC00_v1_F2_H0 | {"j56": 0, "jrefit": 0, "logit": 32} | Sarah lives in Liverpool and works in Brighton on March 11th, but her location c |
| RC00_v3_F1_H2 | other | False | False | RC00_v3_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 31} | To determine the city where Kate's supervisor works, we need to identify who Kat |
| RC00_v3_F2_H0 | abstain | False | True | RC00_v3_F2_H1 | {"j56": 0, "jrefit": 0, "logit": 32} | user mentioned Laura works in Durham and Kate's supervisor is Laura. Answer: unk |
| RC01_v0_F1_H0 | other | False | False | - | {} | Based on the information provided:  - Laura's supervisor is David. - David works |
| RC01_v0_F1_H1 | other | False | False | - | {} | Based on the information provided:  - Laura's supervisor is David. - David works |
| RC01_v0_F1_H2 | other | False | False | - | {} | Based on the information provided, we know that Laura's supervisor is David. Fro |
| RC01_v0_F2_H1 | abstain | False | True | RC01_v0_F2_H0 | {"j56": 0, "jrefit": 0, "logit": 31} | user mentioned that Laura's supervisor is David, and David works in Cambridge on |
| RC01_v1_F1_H1 | other | False | False | RC01_v1_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 31} | Based on the information provided:  - Laura's supervisor is Peter. - Peter works |
| RC01_v1_F1_H2 | other | False | False | RC01_v1_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 31} | To determine the city where Laura's supervisor works, we need to identify who La |
| RC01_v1_F2_H1 | abstain | False | True | RC01_v1_F2_H0 | {"j56": 0, "jrefit": 0, "logit": 31} | user mentioned that Laura's supervisor is Peter, and Peter works in Cambridge on |
| RC01_v2_F1_H1 | other | False | False | RC01_v2_F1_H0 | {"j56": 0, "jrefit": 0, "logit": null} | Based on the information provided:  - Paul lives in Chester. - Kate works in |
| RC01_v3_F2_H0 | abstain | False | True | RC01_v3_F2_H1 | {"j56": 0, "jrefit": 0, "logit": 31} | user mentioned David works in Cambridge and Laura's supervisor is David. Answer: |
| RC02_v0_F1_H1 | other | False | False | RC02_v0_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 31} | Based on the information provided:  - David's supervisor is Peter. - Peter works |
| RC02_v0_F1_H2 | other | False | False | RC02_v0_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 31} | Based on the information provided, we know that David's supervisor is Peter. Fro |
| RC02_v1_F1_H1 | other | False | False | RC02_v1_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 31} | Based on the information provided:  - David's supervisor is James. - James works |
| RC02_v1_F1_H2 | other | False | False | RC02_v1_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 31} | Based on the information provided, we know that David's supervisor is James. Fro |
| RC02_v3_F1_H1 | other | False | False | RC02_v3_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 30} | To determine the city where David's supervisor works, we need to identify who Da |
| RC02_v3_F2_H0 | abstain | False | True | - | {} | user mentioned Peter works in Manchester and David's supervisor is Peter. Answer |
| RC02_v3_F2_H1 | abstain | False | True | - | {} | user mentioned that David's supervisor is Peter, and user stated that Peter work |
| RC02_v3_F2_H2 | abstain | False | True | - | {} | Sarah works in Durham and Peter works in Manchester. David's supervisor is Peter |
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
| RC04_v0_F1_H1 | other | False | False | RC04_v0_F1_H0 | {"j56": 9, "jrefit": 9, "logit": 32} | Based on the information provided:  - Alice works as a writer. - James's |
| RC04_v0_F2_H1 | abstain | False | True | RC04_v0_F2_H0 | {"j56": 8, "jrefit": 8, "logit": null} | James's supervisor is Sarah and Sarah works in Brighton. Answer: unknown |
| RC04_v0_F2_H2 | abstain | False | True | RC04_v0_F2_H0 | {"j56": 8, "jrefit": 8, "logit": null} | Alice - works as a writer   Henry - works as a designer   Kate - works as a doct |
| RC04_v1_F1_H1 | other | False | False | RC04_v1_F1_H0 | {"j56": 9, "jrefit": 9, "logit": 31} | Based on the information provided:  - Alice works as a writer. - James's |
| RC04_v1_F2_H1 | abstain | False | True | RC04_v1_F2_H0 | {"j56": 8, "jrefit": 8, "logit": null} | James's supervisor is Paul and Paul works in Brighton. Answer: unknown |
| RC04_v2_F1_H1 | other | False | False | RC04_v2_F1_H0 | {"j56": 16, "jrefit": 16, "logit": 31} | Based on the information provided:  - Alice works as a writer. - James's |
| RC04_v3_F1_H2 | other | False | False | RC04_v3_F1_H0 | {"j56": 9, "jrefit": 9, "logit": 31} | Based on the information provided:  - James's supervisor is Sarah. - Sarah works |
| RC04_v3_F2_H0 | abstain | False | True | RC04_v3_F2_H1 | {"j56": 9, "jrefit": 9, "logit": 31} | Sarah works in Brighton. James's supervisor is Sarah. Answer: unknown |
| RC04_v3_F2_H2 | abstain | False | True | RC04_v3_F2_H1 | {"j56": 9, "jrefit": 9, "logit": 31} | Alice works as a writer and is mentioned as James's supervisor on 2023/03/10 (Fr |
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
| RC06_v0_F1_H1 | other | False | False | RC06_v0_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 31} | To determine in which city Paul's supervisor works, we need to identify who Paul |
| RC06_v0_F1_H2 | other | False | False | RC06_v0_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 31} | Based on the information provided, we know that Alice is David's supervisor. Fro |
| RC06_v0_F2_H2 | abstain | False | True | RC06_v0_F2_H0 | {"j56": 0, "jrefit": 0, "logit": 31} | Emma works as a chef and is mentioned earlier, but there is no information about |
| RC06_v1_F1_H1 | other | False | False | RC06_v1_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 31} | Based on the information provided:  - Emma works as a chef. - Paul's |
| RC06_v1_F2_H1 | abstain | False | True | RC06_v1_F2_H0 | {"j56": 0, "jrefit": 0, "logit": null} | Emma works as a chef and Paul's supervisor is Rachel. Rachel works in Cardiff. A |
| RC06_v1_F2_H2 | abstain | False | True | RC06_v1_F2_H0 | {"j56": 0, "jrefit": 0, "logit": 31} | Emma works as a chef and Henry works as an editor, but no information is given a |
| RC06_v2_F1_H1 | other | False | False | RC06_v2_F1_H0 | {"j56": 0, "jrefit": 0, "logit": null} | Based on the information provided:  - Emma works as a chef. - Paul's |
| RC06_v3_F1_H1 | other | False | False | RC06_v3_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 31} | To determine the city where Paul's supervisor works, we need to identify who Pau |
| RC06_v3_F2_H0 | abstain | False | True | RC06_v3_F2_H2 | {"j56": 0, "jrefit": 0, "logit": null} | user mentioned Alice works in Cardiff and Paul's supervisor is Alice. Answer: un |
| RC06_v3_F2_H1 | abstain | False | True | RC06_v3_F2_H2 | {"j56": 0, "jrefit": 0, "logit": 32} | Emma works as a chef and there is no information about Paul's supervisor working |
| RC07_v0_F1_H1 | other | False | False | RC07_v0_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 32} | Based on the information provided:  - Alice's supervisor is Rachel. - Rachel wor |
| RC07_v0_F1_H2 | other | False | False | RC07_v0_F1_H0 | {"j56": 1, "jrefit": 2, "logit": 32} | Based on the information provided, we know that Alice's supervisor is Rachel. Fr |
| RC07_v1_F1_H1 | other | False | False | RC07_v1_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 32} | Based on the information provided:  - Alice's supervisor is Ben. - Ben works |
| RC07_v1_F1_H2 | other | False | False | RC07_v1_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 32} | To determine the city where Alice's supervisor works, we need to identify who Al |
| RC07_v3_F1_H2 | other | False | False | RC07_v3_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 32} | Based on the information provided:  - Alice's supervisor is Rachel. - Rachel wor |
| RC07_v3_F2_H0 | abstain | False | True | - | {} | Rachel works in Derby. Alice's supervisor is Rachel. Answer: unknown |
| RC07_v3_F2_H1 | abstain | False | True | - | {} | Rachel works in Derby and is Alice's supervisor. Answer: unknown |
| RC07_v3_F2_H2 | abstain | False | True | - | {} | Sarah lives in Liverpool and is mentioned to be Rachel's supervisor. Rachel work |
| RC08_v0_F1_H1 | other | False | False | RC08_v0_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 31} | To determine the city where Rachel's supervisor works, we need to identify who R |
| RC08_v0_F1_H2 | other | False | False | RC08_v0_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 31} | Based on the information provided, we know that Rachel's supervisor is Ben. Howe |
| RC08_v0_F2_H1 | abstain | False | True | RC08_v0_F2_H0 | {"j56": 0, "jrefit": 0, "logit": 31} | Rachel's supervisor is Ben and Ben works in Edinburgh. Answer: unknown |
| RC08_v0_F2_H2 | abstain | False | True | RC08_v0_F2_H0 | {"j56": 0, "jrefit": 0, "logit": null} | Rachel's supervisor is Ben. Ben works in Edinburgh. Answer: unknown |
| RC08_v1_F1_H1 | other | False | False | RC08_v1_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 31} | To determine in which city Rachel's supervisor works, we need to identify who Ra |
| RC08_v1_F1_H2 | other | False | False | RC08_v1_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 31} | Based on the information provided, we know that Rachel's supervisor is Henry. Ho |
| RC08_v1_F2_H1 | abstain | False | True | RC08_v1_F2_H0 | {"j56": 0, "jrefit": 0, "logit": 31} | Rachel's supervisor is Henry and Henry works in Edinburgh. Answer: unknown |
| RC08_v1_F2_H2 | abstain | False | True | RC08_v1_F2_H0 | {"j56": 0, "jrefit": 0, "logit": 31} | Rachel's supervisor is Henry and his location is not specified; only that he sup |
| RC08_v2_F1_H1 | other | False | False | RC08_v2_F1_H0 | {"j56": null, "jrefit": null, "logit": null} | Based on the information provided:  - Alice lives in Liverpool. - Paul works as |
| RC08_v3_F1_H1 | other | False | False | RC08_v3_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 31} | To determine where Rachel's supervisor works, we need to refer to the informatio |
| RC08_v3_F1_H2 | other | False | False | RC08_v3_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 31} | Based on the information provided, we know that Rachel's supervisor is Ben. We |
| RC08_v3_F2_H0 | abstain | False | True | RC08_v3_F2_H2 | {"j56": 0, "jrefit": 0, "logit": 31} | Ben works in Edinburgh. Rachel's supervisor is Ben. Answer: unknown |
| RC08_v3_F2_H1 | abstain | False | True | RC08_v3_F2_H2 | {"j56": 0, "jrefit": 0, "logit": 31} | Rachel's supervisor is Ben and Ben works in Edinburgh. Answer: unknown |
| RC09_v0_F1_H1 | other | False | False | RC09_v0_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 1} | Based on the information provided:  - Kate lives in Liverpool. - James lives in |
| RC09_v0_F1_H2 | other | False | False | RC09_v0_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 0} | Based on the information provided, we know that Henry is Ben's supervisor and th |
| RC09_v1_F1_H1 | other | False | False | RC09_v1_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 1} | Based on the information provided:  - Kate lives in Liverpool. - James lives in |
| RC09_v1_F1_H2 | other | False | False | RC09_v1_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 1} | Based on the information provided, we know that Emma is Ben's supervisor and she |
| RC09_v1_F2_H2 | abstain | False | True | RC09_v1_F2_H0 | {"j56": 0, "jrefit": 0, "logit": null} | Kate lives in Liverpool and James lives in Manchester. Ben's supervisor is Emma, |
| RC09_v2_F2_H1 | abstain | False | True | RC09_v2_F2_H0 | {"j56": 0, "jrefit": 9, "logit": 32} | Kate, James, and Rachel live in different cities, so they are not Ben's supervis |
| RC09_v3_F1_H2 | other | False | False | RC09_v3_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 0} | To determine the city where Ben's supervisor works, we need to identify who Ben |
| RC09_v3_F2_H0 | abstain | False | True | RC09_v3_F2_H1 | {"j56": 0, "jrefit": 0, "logit": 0} | Henry works in Leeds. Ben's supervisor is Henry. Answer: unknown |
| RC09_v3_F2_H2 | abstain | False | True | RC09_v3_F2_H1 | {"j56": 0, "jrefit": 0, "logit": 0} | Kate lives in Liverpool. James lives in Manchester. Rachel lives in Norwich. Hen |
| RC10_v0_F1_H1 | other | False | False | RC10_v0_F1_H0 | {"j56": 32, "jrefit": 32, "logit": 31} | Based on the information provided:  - Henry's supervisor is Emma. - Emma works |
| RC10_v0_F1_H2 | other | False | False | RC10_v0_F1_H0 | {"j56": 32, "jrefit": 32, "logit": 31} | To determine the city where Henry's supervisor works, we need to identify who He |
| RC10_v1_F1_H0 | other | False | False | - | {} | Based on the information provided, we know that Henry's supervisor is Kate and t |
| RC10_v1_F1_H1 | other | False | False | - | {} | Based on the information provided:  - Henry's supervisor is Kate. - Kate works |
| RC10_v1_F1_H2 | other | False | False | - | {} | To determine the city where Henry's supervisor works, we need to identify who He |
| RC10_v3_F1_H2 | other | False | False | RC10_v3_F1_H0 | {"j56": 31, "jrefit": 31, "logit": 31} | Based on the information provided:  - Henry's supervisor is Emma. - Emma works |
| RC10_v3_F2_H0 | abstain | False | True | RC10_v3_F2_H1 | {"j56": 9, "jrefit": 9, "logit": 31} | Emma works in York. Henry's supervisor is Emma. Answer: unknown |
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
| TU00_v0_F1_H1 | other | True | False | TU00_v0_F1_H0 | {"j56": 11, "jrefit": 31, "logit": 31} | After Kate moved to Norwich, she no longer lives in York. Therefore, the |
| TU00_v0_F1_H2 | competitor | False | False | TU00_v0_F1_H0 | {"j56": 9, "jrefit": 10, "logit": 31} | Based on the information provided, Kate originally lived in York. Later, it is |
| TU00_v0_F2_H0 | abstain | False | True | - | {} | Kate moved from York to Norwich. Answer: unknown |
| TU00_v0_F2_H1 | abstain | False | True | - | {} | user - Kate lived in York on 2023/03/02 (Thu) user - Later, Kate moved to Norwic |
| TU00_v0_F2_H2 | abstain | False | True | - | {} | Ben works in Brighton, Alice works as a scientist, Kate lived in York, later she |
| TU00_v1_F1_H1 | competitor | False | False | TU00_v1_F1_H0 | {"j56": 31, "jrefit": 10, "logit": 31} | Based on the information provided:  1. Initially, Kate lived in Norwich. 2 |
| TU00_v1_F1_H2 | competitor | False | False | TU00_v1_F1_H0 | {"j56": 31, "jrefit": 10, "logit": 31} | Based on the information provided, Kate originally lived in Norwich. Later, it i |
| TU00_v2_F1_H0 | other | True | False | TU00_v2_F1_H1 | {"j56": null, "jrefit": null, "logit": null} | Before moving to Norwich, Kate lived in York. |
| TU00_v2_F1_H2 | other | True | False | TU00_v2_F1_H1 | {"j56": null, "jrefit": null, "logit": null} | Before moving to Norwich, Kate lived in York. |
| TU00_v2_F2_H2 | abstain | False | True | TU00_v2_F2_H0 | {"j56": 10, "jrefit": 10, "logit": 32} | Ben, Alice, Kate, James, David, Laura, James, Henry, Emma, Rachel, Sarah - none  |
| TU00_v3_F1_H1 | other | False | False | TU00_v3_F1_H0 | {"j56": 9, "jrefit": 10, "logit": 31} | Based on the information provided:  1. Ben works in Brighton. 2. Kate |
| TU00_v3_F2_H0 | abstain | False | True | TU00_v3_F2_H2 | {"j56": 9, "jrefit": 9, "logit": null} | Kate lived in York before moving to Norwich on 2023/03/01. Answer: unknown |
| TU00_v3_F2_H1 | abstain | False | True | TU00_v3_F2_H2 | {"j56": 9, "jrefit": 9, "logit": 32} | Kate now lives in Norwich. Answer: unknown |
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
| TU02_v0_F1_H1 | competitor | False | False | TU02_v0_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 34} | Based on the information provided:  - Initially, Paul lived in Liverpool. - Late |
| TU02_v0_F1_H2 | competitor | False | False | TU02_v0_F1_H0 | {"j56": 0, "jrefit": 0, "logit": null} | Based on the information provided:  - Initially, Paul lived in Liverpool. - Late |
| TU02_v0_F2_H1 | abstain | False | True | TU02_v0_F2_H0 | {"j56": 0, "jrefit": 0, "logit": 31} | Emma lives in York, Paul lived in Liverpool, later Paul moved to Leeds. Answer:  |
| TU02_v1_F1_H1 | competitor | False | False | TU02_v1_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 31} | Based on the information provided:  - Initially, Paul lived in Leeds. - Later |
| TU02_v1_F1_H2 | competitor | False | False | TU02_v1_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 31} | Based on the information provided:  - Initially, Paul lived in Leeds. - Later |
| TU02_v1_F2_H1 | abstain | False | True | TU02_v1_F2_H0 | {"j56": 0, "jrefit": 0, "logit": null} | Emma lives in York, Paul lived in Leeds, later Paul moved to Liverpool. Answer:  |
| TU02_v2_F1_H0 | other | True | False | TU02_v2_F1_H2 | {"j56": 23, "jrefit": 23, "logit": null} | Before moving to Leeds, Paul lived in Liverpool. |
| TU02_v2_F1_H1 | other | True | False | TU02_v2_F1_H2 | {"j56": 23, "jrefit": 23, "logit": null} | Before moving to Leeds, Paul lived in Liverpool. |
| TU02_v3_F1_H1 | other | False | False | TU02_v3_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 1} | Based on the information provided:  - Emma lives in York. - Paul used to |
| TU02_v3_F1_H2 | competitor | False | False | TU02_v3_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 1} | Based on the information provided:  - Before that, Paul lived in Liverpool. - |
| TU02_v3_F2_H1 | abstain | False | True | TU02_v3_F2_H0 | {"j56": 0, "jrefit": 0, "logit": 0} | Emma lives in York, Paul now lives in Leeds, David works in Chester, before that |
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
| TU04_v0_F1_H1 | competitor | False | False | TU04_v0_F1_H0 | {"j56": 9, "jrefit": 9, "logit": 31} | Based on the information provided:  - Initially, James lived in Edinburgh. - Lat |
| TU04_v0_F1_H2 | competitor | False | False | TU04_v0_F1_H0 | {"j56": 9, "jrefit": 9, "logit": null} | Based on the information provided:  - Initially, James lived in Edinburgh. - Lat |
| TU04_v0_F2_H2 | abstain | False | True | TU04_v0_F2_H0 | {"j56": 8, "jrefit": 8, "logit": 31} | James lived in Edinburgh initially, but later moved to Brighton. Answer: unknown |
| TU04_v1_F1_H1 | competitor | False | False | TU04_v1_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 30} | Based on the information provided:  - Initially, James lived in Brighton. - Late |
| TU04_v1_F1_H2 | competitor | False | False | TU04_v1_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 30} | Based on the information provided:  - Initially, James lived in Brighton. - Late |
| TU04_v1_F2_H2 | abstain | False | True | TU04_v1_F2_H0 | {"j56": 0, "jrefit": 0, "logit": null} | Peter lives in York. Kate works as a designer. Laura lives in Liverpool. Henry w |
| TU04_v2_F1_H0 | other | True | False | - | {} | Before moving to Brighton, James lived in Edinburgh. |
| TU04_v2_F1_H1 | other | True | False | - | {} | Before moving to Brighton, James lived in Edinburgh. |
| TU04_v2_F1_H2 | other | True | False | - | {} | Before moving to Brighton, James lived in Edinburgh. |
| TU04_v2_F2_H1 | abstain | False | True | TU04_v2_F2_H0 | {"j56": 0, "jrefit": 0, "logit": null} | James lived in Edinburgh before moving to Brighton. Answer: unknown |
| TU04_v3_F1_H1 | competitor | False | False | TU04_v3_F1_H0 | {"j56": 9, "jrefit": 9, "logit": 31} | Based on the information provided:  1. Before that, James lived in Edinburgh.  |
| TU04_v3_F1_H2 | other | False | False | TU04_v3_F1_H0 | {"j56": 9, "jrefit": 9, "logit": 31} | Based on the information provided:  - Peter lives in York. - Kate works as |
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
| TU06_v0_F1_H1 | competitor | False | False | TU06_v0_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 30} | Based on the information provided:  - Initially, Emma lived in Derby. - Later |
| TU06_v0_F1_H2 | competitor | False | False | TU06_v0_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 30} | Based on the information provided:  - Initially, Emma lived in Derby. - Later |
| TU06_v0_F2_H1 | abstain | False | True | TU06_v0_F2_H0 | {"j56": 0, "jrefit": 0, "logit": null} | user mentioned that Emma lived in Derby on 2023/03/02 (Thu) 10:00 and later move |
| TU06_v0_F2_H2 | abstain | False | True | TU06_v0_F2_H0 | {"j56": 0, "jrefit": 0, "logit": null} | Kate works as an artist, Alice works in Chester, Laura works in Cardiff, Paul li |
| TU06_v1_F1_H1 | other | True | False | TU06_v1_F1_H0 | {"j56": 17, "jrefit": 31, "logit": 31} | After Emma moved to Derby, she no longer lives in Manchester. Therefore, Emma |
| TU06_v1_F1_H2 | competitor | False | False | TU06_v1_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 31} | Based on the information provided:  - Initially, Emma lived in Manchester. - Lat |
| TU06_v2_F1_H0 | other | True | False | - | {} | Before moving to Manchester, Emma lived in Derby. |
| TU06_v2_F1_H1 | other | True | False | - | {} | Before moving to Manchester, Emma lived in Derby. |
| TU06_v2_F1_H2 | other | True | False | - | {} | Before moving to Manchester, Emma lived in Derby. |
| TU06_v2_F2_H1 | abstain | False | True | TU06_v2_F2_H0 | {"j56": 0, "jrefit": 0, "logit": null} | user mentioned Emma lived in Derby on 2023/03/02 (Thu). Answer: unknown |
| TU06_v2_F2_H2 | abstain | False | True | TU06_v2_F2_H0 | {"j56": 0, "jrefit": 0, "logit": null} | Kate works as an artist, Alice works in Chester, Laura works in Cardiff, Paul li |
| TU06_v3_F1_H1 | competitor | False | False | TU06_v3_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 30} | Based on the information provided:  1. Emma used to live in Derby. 2 |
| TU06_v3_F1_H2 | competitor | False | False | TU06_v3_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 30} | Based on the information provided:  - Before that, Emma lived in Derby. - |
| TU06_v3_F2_H1 | abstain | False | True | TU06_v3_F2_H0 | {"j56": 0, "jrefit": 0, "logit": null} | Emma used to live in Derby before moving to Manchester. Answer: unknown |
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
| TU08_v0_F1_H0 | competitor | False | False | - | {} | Based on the information provided:  1. Initially, Henry lived in Durham. 2 |
| TU08_v0_F1_H1 | competitor | False | False | - | {} | Based on the information provided:  1. Initially, Henry lived in Durham. 2 |
| TU08_v0_F1_H2 | competitor | False | False | - | {} | Based on the information provided:  - Initially, Henry lived in Durham. - Later |
| TU08_v1_F1_H0 | competitor | False | False | - | {} | Based on the information provided:  1. Initially, Henry lived in Cardiff. 2 |
| TU08_v1_F1_H1 | competitor | False | False | - | {} | Based on the information provided:  1. Initially, Henry lived in Cardiff. 2 |
| TU08_v1_F1_H2 | competitor | False | False | - | {} | Based on the information provided:  - Initially, Henry lived in Cardiff. - Later |
| TU08_v1_F2_H1 | abstain | False | True | TU08_v1_F2_H0 | {"j56": 0, "jrefit": 0, "logit": null} | Henry moved from Cardiff to Durham on 2023/03/02 (Thu). Answer: unknown |
| TU08_v2_F1_H0 | other | True | False | TU08_v2_F1_H1 | {"j56": 23, "jrefit": 23, "logit": null} | Before moving to Cardiff, Henry lived in Durham. |
| TU08_v2_F1_H2 | other | True | False | TU08_v2_F1_H1 | {"j56": 30, "jrefit": null, "logit": null} | Before moving to Cardiff, Henry lived in Durham. |
| TU08_v2_F2_H1 | abstain | False | True | TU08_v2_F2_H0 | {"j56": 0, "jrefit": 0, "logit": null} | Henry lived in Durham before moving to Cardiff. Answer: unknown |
| TU08_v2_F2_H2 | other | False | False | TU08_v2_F2_H0 | {"j56": 0, "jrefit": 0, "logit": 32} | Emma - 2023/03/01 (Wed) Peter - 2023/03/02 (Thu) Sarah - 2023/03/03 (Fri) James  |
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
| TU10_v0_F1_H0 | competitor | False | False | - | {} | Based on the information provided:  1. Initially, Alice lived in Cambridge. 2 |
| TU10_v0_F1_H1 | competitor | False | False | - | {} | Based on the information provided:  1. Initially, Alice lived in Cambridge. 2 |
| TU10_v0_F1_H2 | other | True | False | - | {} | Alice moved from Cambridge to Chester, so she now lives in Chester. |
| TU10_v1_F1_H1 | competitor | False | False | TU10_v1_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 30} | Based on the information provided:  1. Initially, Alice lived in Chester. 2 |
| TU10_v1_F1_H2 | other | True | False | TU10_v1_F1_H0 | {"j56": 20, "jrefit": 20, "logit": 30} | Alice moved from Chester to Cambridge later, so she now lives in Cambridge. |
| TU10_v1_F2_H2 | abstain | False | True | TU10_v1_F2_H0 | {"j56": 0, "jrefit": 0, "logit": 32} | Alice lived in Chester earlier but moved to Cambridge on 2023/03/07 (Tue). Answe |
| TU10_v2_F1_H0 | other | True | False | TU10_v2_F1_H1 | {"j56": 22, "jrefit": 21, "logit": null} | Before moving to Chester, Alice lived in Cambridge. |
| TU10_v2_F1_H2 | other | True | False | TU10_v2_F1_H1 | {"j56": 30, "jrefit": 30, "logit": null} | Based on the information provided, Alice lived in Cambridge before she moved to  |
| TU11_v0_F1_H1 | other | True | False | TU11_v0_F1_H0 | {"j56": null, "jrefit": null, "logit": null} | Rachel moved from Chester to York, so she no longer lives in Chester. According |
| TU11_v0_F1_H2 | other | True | False | TU11_v0_F1_H0 | {"j56": null, "jrefit": null, "logit": null} | Rachel originally lived in Chester, but later moved to York. Therefore, Rachel n |
| TU11_v1_F1_H1 | other | True | False | TU11_v1_F1_H0 | {"j56": null, "jrefit": null, "logit": null} | Rachel moved from York to Chester, so she no longer lives in York. According |
| TU11_v1_F1_H2 | other | True | False | TU11_v1_F1_H0 | {"j56": null, "jrefit": null, "logit": null} | Rachel moved from York to Chester later, so she now lives in Chester. |
| TU11_v2_F1_H0 | other | True | False | TU11_v2_F1_H1 | {"j56": 30, "jrefit": null, "logit": 31} | Before moving to York, Rachel lived in Chester. |
| TU11_v2_F1_H2 | other | True | False | TU11_v2_F1_H1 | {"j56": 30, "jrefit": null, "logit": null} | Before moving to York, Rachel lived in Chester. |
| TU11_v2_F2_H1 | abstain | False | True | TU11_v2_F2_H0 | {"j56": null, "jrefit": null, "logit": 32} | Sarah lives in Norwich, which was mentioned on 2023/03/01 (Wed). Rachel lived in |
| TU11_v3_F1_H2 | competitor | False | False | TU11_v3_F1_H0 | {"j56": 31, "jrefit": 31, "logit": 31} | Based on the information provided:  - Before that, Rachel lived in Chester. - |

## Stage 3: the §44 rev 8 rows (140 prompts)

| key | §23 | §44 class (1 nat) | d_T fwd | outcome | marker | lens: gold / substitute / unknown best rank at the last prompt token | generated |
|---|---|---|---|---|---|---|---|
| 01493427_clean | refusal | answer_production | 0.89 | abstain | True | j56 27/76/630; jrefit 27/76/658; logit 27/76/727 | user added 17 new postcards since starting to collect again  |
| 01493427_swap | refusal | answer_production | 0.89 | abstain | True | j56 26/82/617; jrefit 26/82/628; logit 26/82/687 | user added 17 new postcards since starting to collect again  |
| 0862e8bf_clean | success | answer_production | 0.92 | gold | True | j56 117/5037/101; jrefit 117/5037/101; logit 117/3840/101 | Facts bearing on the question: - User mentioned their cat's  |
| 0862e8bf_swap | success | answer_production | 0.92 | abstain | True | j56 369/1280/84; jrefit 369/1280/84; logit 339/831/84 | Brief notes: - User asked about their cat's name on 2023/05/ |
| 08e075c7_clean | success | answer_production | 0.89 | gold | True | j56 35/24/1144; jrefit 35/24/1213; logit 28/17/1405 | user mentioned they have been using their Fitbit Charge 3 fo |
| 08e075c7_swap | success | answer_production | 0.89 | gold | True | j56 39/23/1397; jrefit 39/23/1457; logit 36/17/1717 | user mentioned they have been using their Fitbit Charge 3 fo |
| 099778bb_clean | refusal | answer_production | 0.86 | abstain | False | j56 9/6/1; jrefit 9/6/1; logit 9/6/1 | unknown |
| 099778bb_swap | refusal | answer_production | 0.86 | abstain | False | j56 12/5/1; jrefit 12/5/1; logit 12/5/1 | unknown |
| 0e4e4c46_clean | wrong | answer_production | 0.86 | gold | False | j56 32/57/8; jrefit 32/57/8; logit 32/57/8 | user, 2023/05/23 (Tue) 03:08: I'm interested in trying out C |
| 0e4e4c46_swap | wrong | answer_production | 0.86 | competitor | False | j56 34/61/19; jrefit 34/61/19; logit 34/61/19 | user, 2023/05/23 (Tue) 03:08: I'm interested in trying out C |
| 0e5e2d1a_clean | wrong | answer_production | 0.89 | other | False | j56 4/5/9; jrefit 4/5/8; logit 4/5/13 | 18 subjects participated in the study published in the journ |
| 0e5e2d1a_swap | wrong | answer_production | 0.89 | other | False | j56 4/5/13; jrefit 4/5/12; logit 4/5/16 | 15 subjects were in the study published in the journal Music |
| 0f05491a_clean | wrong | indeterminate | - | other | True | j56 12/204/172; jrefit 12/204/177; logit 11/50/221 | facts: user needing gold level stars; user reaching gold lev |
| 0f05491a_swap | wrong | indeterminate | - | abstain | True | j56 12/200/171; jrefit 12/200/172; logit 11/52/212 | facts: user needing gold level stars; user reaching gold lev |
| 15745da0_clean | success | answer_production | 0.92 | gold | False | j56 545/331/1473; jrefit 353/331/1519; logit 849/331/1681 | user has been collecting vintage cameras for three months ba |
| 15745da0_swap | success | answer_production | 0.92 | competitor | False | j56 545/44/1726; jrefit 353/44/1786; logit 1945/44/1925 | user has been collecting vintage cameras for Two weeks now b |
| 18dcd5a5_clean | wrong | answer_production | 0.86 | other | False | j56 59/36/552; jrefit 59/36/569; logit 44/31/610 | name: number of mummies in the temple |
| 18dcd5a5_swap | wrong | answer_production | 0.86 | other | False | j56 63/28/598; jrefit 63/28/617; logit 46/28/658 | name: number of mummies in the temple |
| 19b5f2b3_clean | success | indeterminate | - | other | True | j56 664/530/1877; jrefit 299/348/1935; logit 547/5036/2204 | user mentioned spending two weeks traveling solo around Japa |
| 19b5f2b3_swap | success | indeterminate | - | competitor | True | j56 660/533/1802; jrefit 295/348/1895; logit 533/4943/2090 | user mentioned spending three months traveling solo around J |
| 1b9b7252_clean | success | answer_production | 0.89 | gold | False | j56 90/3238/108; jrefit 90/3011/128; logit 90/1084/150 | To remind you of the website that had free guided imagery ex |
| 1b9b7252_swap | success | answer_production | 0.89 | competitor | False | j56 95/1368/136; jrefit 95/1368/136; logit 95/1105/167 | To remind you of the website that had free guided imagery ex |
| 1de5cff2_clean | success | answer_production | 0.92 | gold | True | j56 115/1559/608; jrefit 115/291/644; logit 115/3716/637 | Indian culture and traditions, guitar humidifier recommendat |
| 1de5cff2_swap | success | answer_production | 0.92 | competitor | True | j56 337/1583/587; jrefit 337/291/608; logit 278/3715/614 | Furniture or footwear brands are being discussed, specifical |
| 1e043500_clean | refusal | answer_production | 0.92 | abstain | True | j56 1960/376/2314; jrefit 1771/166/2422; logit 2428/1172/2815 | user mentioned creating a playlist called "Summer Vibes" on  |
| 1e043500_swap | refusal | answer_production | 0.92 | competitor | False | j56 2466/354/2066; jrefit 2264/163/2128; logit 3099/598/2321 | user mentioned creating a Spotify playlist called "Tom" that |
| 1faac195_clean | refusal | indeterminate | - | abstain | True | j56 26/38/239; jrefit 42/83/255; logit 4430/357/272 | user mentioned that they were thinking of visiting their sis |
| 1faac195_swap | refusal | indeterminate | - | abstain | True | j56 33/47/94; jrefit 48/101/94; logit 4339/360/94 | user mentioned that they were thinking of visiting their sis |
| 26bdc477_clean | success | evidence_use | - | gold | False | j56 968/622/672; jrefit 565/381/702; logit 6024/10138/810 | user, 2023/03/16 (Thu) 19:05: I've taken my camera on quite  |
| 26bdc477_swap | success | evidence_use | - | competitor | False | j56 968/626/825; jrefit 548/383/870; logit 5475/8871/1024 | user, 2023/03/16 (Thu) 19:05: I've taken my camera on three  |
| 29f2956b_clean | success | answer_production | 0.86 | gold | False | j56 94/415/432; jrefit 94/340/462; logit 94/415/517 | user mentioned practicing guitar for 30 minutes daily. |
| 29f2956b_swap | success | answer_production | 0.86 | competitor | False | j56 83/342/328; jrefit 83/342/350; logit 83/342/411 | user mentioned practicing guitar for 4 days daily. |
| 3249768e_clean | wrong | indeterminate | - | other | False | j56 4268/491/2; jrefit 4050/155/2; logit 4268/672/3 | user, 2023/05/23 (Tue) 03:26: The fifth bottle recommended w |
| 3249768e_swap | wrong | indeterminate | - | other | False | j56 5067/491/6; jrefit 3938/155/6; logit 4402/684/6 | user, 2023/05/23 (Tue) 03:26: The fifth bottle recommended w |
| 3e321797_clean | success | answer_production | 0.86 | gold | True | j56 4/16/2; jrefit 4/16/2; logit 2/13/1 | The relevant information provided was about applying tomato  |
| 3e321797_swap | success | answer_production | 0.86 | gold | True | j56 4/16/2; jrefit 4/16/2; logit 2/13/2 | user, 2023/05/28 (Sun) 14:59: 10-15 minutes Answer: 10-15 mi |
| 45dc21b6_clean | success | answer_production | 0.86 | gold | True | j56 195/73/1037; jrefit 195/73/1079; logit 195/64/920 | user mentioned trying out 3 of Emma's recipes so far. Answer |
| 45dc21b6_swap | success | answer_production | 0.86 | competitor | True | j56 193/67/1178; jrefit 193/67/1239; logit 193/60/1068 | user mentioned trying out 2 of Emma's recipes so far. Answer |
| 4b24c848_clean | success | answer_production | 0.86 | gold | False | j56 907/873/749; jrefit 523/514/782; logit 1606/2719/844 | user mentioned buying three tops from H&M initially and then |
| 4b24c848_swap | success | answer_production | 0.86 | other | False | j56 931/896/841; jrefit 524/513/889; logit 1540/2395/932 | user mentioned buying three tops from H&M initially and then |
| 577d4d32_clean | refusal | answer_production | 0.89 | abstain | True | j56 250/110/537; jrefit 250/110/564; logit 213/70/555 | user mentioned stopping work emails and messages by 7 pm to  |
| 577d4d32_swap | refusal | answer_production | 0.89 | abstain | True | j56 235/103/466; jrefit 235/103/494; logit 197/67/481 | user mentioned stopping work emails and messages by 2 AM to  |
| 5809eb10_clean | success | indeterminate | - | gold | True | j56 12/17/1; jrefit 12/17/1; logit 12/17/2 | Answer: 2014 |
| 5809eb10_swap | success | indeterminate | - | other | True | j56 12/13/1; jrefit 12/13/1; logit 12/13/2 | Answer: 2015 |
| 5831f84d_clean | wrong | answer_production | 0.89 | other | True | j56 17/93/971; jrefit 17/93/1021; logit 17/90/1153 | user watched 10 Crash Course videos in the past few weeks be |
| 5831f84d_swap | wrong | answer_production | 0.89 | competitor | True | j56 17/74/807; jrefit 17/74/823; logit 17/74/965 | user watched 5-2 Crash Course videos in the past few weeks a |
| 58bf7951_clean | success | answer_production | 0.92 | gold | False | j56 2195/6913/449; jrefit 2195/6249/479; logit 2077/3234/491 | user, 2023/05/26 (Fri) 07:31: The play I attended at the loc |
| 58bf7951_swap | success | answer_production | 0.92 | competitor | False | j56 3614/6611/357; jrefit 3281/6091/369; logit 4586/3712/416 | user, 2023/05/26 (Fri) 07:31: The play I attended at the loc |
| 59524333_clean | wrong | answer_production | 0.75 | gold | True | j56 240/616/1557; jrefit 240/335/1622; logit 73/249/1866 | user mentioned going to the gym on Mondays, Wednesdays, and  |
| 59524333_swap | wrong | answer_production | 0.75 | gold | True | j56 249/616/1324; jrefit 249/341/1366; logit 73/255/1579 | user mentioned going to the gym on Mondays, Wednesdays, and  |
| 5a4f22c0_clean | refusal | answer_production | 0.92 | abstain | True | j56 2584/418/1; jrefit 2584/337/1; logit 2584/1586/1 | unknown Answer: unknown |
| 5a4f22c0_swap | refusal | answer_production | 0.92 | abstain | True | j56 4436/369/1; jrefit 4436/290/1; logit 3740/518/1 | unknown Answer: unknown |
| 5e1b23de_clean | success | evidence_use | - | gold | True | j56 71/99/2238; jrefit 71/99/2368; logit 71/99/2582 | user attended the photography workshop on 11 November 2023.  |
| 5e1b23de_swap | success | evidence_use | - | gold | True | j56 79/92/2128; jrefit 79/92/2247; logit 79/92/2364 | user attended the photography workshop on 2023/11/01 (Wed).  |
| 60472f9c_clean | success | indeterminate | - | gold | True | j56 22/83/1815; jrefit 22/83/1883; logit 22/83/2181 | user mentioned working on a Master's thesis and a group proj |
| 60472f9c_swap | success | indeterminate | - | gold | True | j56 27/38/1661; jrefit 27/38/1705; logit 27/37/1933 | user, 2023/05/25 (Thu) 11:44: Created separate boards for th |
| 618f13b2_clean | wrong | indeterminate | - | other | False | j56 328/1123/122; jrefit 328/922/122; logit 328/818/122 | user, 2023/05/30 (Tue) 22:16: I've worn my new black Convers |
| 618f13b2_swap | wrong | indeterminate | - | other | False | j56 384/1127/145; jrefit 384/912/145; logit 384/842/145 | user, 2023/05/30 (Tue) 22:16: I've worn my new black Convers |
| 6aeb4375_clean | success | evidence_use | - | other | True | j56 622/1011/342; jrefit 376/578/342; logit 5342/3223/342 | user, 2023/09/30 (Sat) 15:06: I've tried four different Kore |
| 6aeb4375_swap | success | evidence_use | - | other | True | j56 643/1008/341; jrefit 380/577/341; logit 5758/3351/341 | user, 2023/09/30 (Sat) 15:06: user has tried five different  |
| 70b3e69b_clean | wrong | answer_production | 0.89 | other | False | j56 263/20387/5; jrefit 263/22732/5; logit 263/42157/5 | user, 2023/05/21 (Sun) 10:54: But how do you think the curre |
| 70b3e69b_swap | wrong | answer_production | 0.89 | competitor | True | j56 1242/1463/3; jrefit 1091/1463/3; logit 449/1463/3 | user, 2023/05/21 (Sun) 10:54: The Spanish-Catalan singer/son |
| 7401057b_clean | success | answer_production | 0.86 | gold | False | j56 438/881/38; jrefit 438/495/38; logit 438/2486/38 | user, 2023/05/29 (Mon) 16:25: I've accumulated enough points |
| 7401057b_swap | success | answer_production | 0.86 | competitor | False | j56 475/881/50; jrefit 475/500/50; logit 475/1913/50 | user, 2023/05/29 (Mon) 16:25: I've accumulated enough points |
| 7527f7e2_clean | success | answer_production | 0.78 | gold | True | j56 158/367/352; jrefit 158/367/352; logit 111/132/352 | user, 2023/05/23 (Tue) 08:15: I remember buying a designer h |
| 7527f7e2_swap | success | answer_production | 0.78 | competitor | True | j56 117/315/245; jrefit 117/315/245; logit 93/112/245 | user, 2023/05/23 (Tue) 08:15: I remember buying a designer h |
| 76d63226_clean | success | answer_production | 0.89 | gold | True | j56 121/68/252; jrefit 121/68/261; logit 98/36/272 | user mentioned setting up a new Samsung 55-inch 4K smart TV  |
| 76d63226_swap | success | answer_production | 0.89 | abstain | True | j56 117/60/189; jrefit 117/60/197; logit 102/39/204 | user mentioned setting up their new Samsung 10 pounds 4K sma |
| 7a87bd0c_clean | success | answer_production | 0.89 | gold | False | j56 101/48/523; jrefit 101/48/553; logit 101/44/574 | user mentioned sticking to their daily tidying routine for 4 |
| 7a87bd0c_swap | success | answer_production | 0.89 | competitor | False | j56 74/34/661; jrefit 74/34/693; logit 74/34/704 | user mentioned sticking to their daily tidying routine for 5 |
| 7a8d0b71_clean | success | answer_production | 0.86 | gold | True | j56 4/16/4; jrefit 4/16/4; logit 4/16/4 | Answer: 2000 |
| 7a8d0b71_swap | success | answer_production | 0.86 | competitor | True | j56 9/12/4; jrefit 9/12/4; logit 9/12/4 | Answer: 400,000 |
| 7e00a6cb_clean | refusal | answer_production | 0.92 | other | False | j56 1633/1217/780; jrefit 1230/1217/802; logit 1023/1123/761 | user asked for recommendations for must-see places and thing |
| 7e00a6cb_swap | refusal | answer_production | 0.92 | other | False | j56 1650/1133/882; jrefit 1244/1133/924; logit 1002/1106/892 | user asked for recommendations for must-see places and thing |
| 8464fc84_clean | success | answer_production | 0.92 | gold | False | j56 7149/7988/476; jrefit 2382/7988/483; logit 2949/2656/557 | user mentioned a famous deli named Roscioli near the Vatican |
| 8464fc84_swap | success | answer_production | 0.92 | competitor | True | j56 7953/6024/442; jrefit 2346/6024/455; logit 2766/2659/544 | user mentioned a famous deli named TechCorp near the Vatican |
| 852ce960_clean | wrong | answer_production | 0.92 | other | True | j56 222/25/1456; jrefit 222/25/1501; logit 222/25/1783 | user mentioned being pre-approved for $350,000 from Wells Fa |
| 852ce960_swap | wrong | answer_production | 0.92 | other | True | j56 225/24/1590; jrefit 225/24/1673; logit 225/24/1955 | user mentioned being pre-approved for $350,000 from Wells Fa |
| 86b68151_clean | success | answer_production | 0.89 | gold | True | j56 5764/22/117; jrefit 5764/29/117; logit 5764/4404/117 | name: new bookshelf brand Answer: IKEA |
| 86b68151_swap | success | answer_production | 0.89 | abstain | True | j56 6632/25/147; jrefit 6632/34/147; logit 6632/4400/147 | name: where new bookshelf purchased answer: unknown |
| 86f00804_clean | success | answer_production | 0.89 | gold | False | j56 2628/8731/826; jrefit 2783/8819/834; logit 2670/9115/837 | user mentioned they are currently devouring "The Seven Husba |
| 86f00804_swap | success | answer_production | 0.89 | competitor | False | j56 2622/8437/802; jrefit 2877/8437/830; logit 2793/8437/848 | user mentioned they are currently devouring "The Sugar Facto |
| 8b9d4367_clean | success | answer_production | 0.89 | gold | True | j56 7000/3239/1016; jrefit 7630/3422/1046; logit 7630/558/1043 | Private sector business that employs over 40,000 people in t |
| 8b9d4367_swap | success | answer_production | 0.89 | competitor | True | j56 6898/3237/1013; jrefit 8902/3440/1051; logit 10806/526/1021 | Private sector business that employs over 40,000 people in t |
| 95bcc1c8_clean | success | answer_production | 0.94 | gold | True | j56 47/148/581; jrefit 47/148/588; logit 39/148/588 | user mentioned seeing 10 amateur comedians perform at the op |
| 95bcc1c8_swap | success | answer_production | 0.94 | competitor | True | j56 56/159/684; jrefit 56/159/684; logit 42/159/684 | user mentioned seeing 15 amateur comedians perform at the op |
| a96c20ee_clean | refusal | indeterminate | - | abstain | True | j56 2806/1259/1317; jrefit 2462/1109/1385; logit 1005/2001/1407 | user mentioned presenting a poster on their thesis research  |
| a96c20ee_swap | refusal | indeterminate | - | abstain | True | j56 2744/1336/1068; jrefit 2371/1188/1080; logit 1072/2318/1080 | user mentioned presenting a poster on their thesis research  |
| ad7109d1_clean | refusal | answer_production | 0.86 | abstain | True | j56 76/35/64; jrefit 76/35/64; logit 76/35/64 | user mentioned upgrading to 500 Mbps about three weeks ago,  |
| ad7109d1_swap | refusal | answer_production | 0.86 | abstain | True | j56 85/33/73; jrefit 85/33/73; logit 85/32/73 | user mentioned upgrading their internet speed to 132 points  |
| affe2881_clean | success | answer_production | 0.89 | gold | True | j56 67/27/878; jrefit 67/27/909; logit 62/23/1018 | user, 2023/05/24 (Wed) 01:31: managed to spot 27 different s |
| affe2881_swap | success | answer_production | 0.89 | other | True | j56 93/42/1007; jrefit 93/42/1044; logit 83/27/1170 | user, 2023/05/24 (Wed) 01:31: managed to spot 27 different s |
| b6019101_clean | success | answer_production | 0.86 | gold | True | j56 125/63/580; jrefit 125/63/605; logit 125/56/658 | user mentioned watching 5 MCU films in the last 3 months. An |
| b6019101_swap | success | answer_production | 0.86 | competitor | True | j56 126/80/484; jrefit 126/80/506; logit 126/60/556 | user mentioned watching 2 MCU films in the last 3 months. An |
| ba61f0b9_clean | wrong | answer_production | 0.89 | other | True | j56 6/8/2; jrefit 6/8/2; logit 6/8/3 | 2023/01/18 (Wed) 13:09: Rachel's team consists of 5 women ou |
| ba61f0b9_swap | wrong | answer_production | 0.89 | other | True | j56 11/5/2; jrefit 11/5/2; logit 9/5/3 | 2023/01/18 (Wed) 13:09: Rachel's team consists of 5 women ou |
| c14c00dd_clean | success | answer_production | 0.92 | gold | False | j56 12561/2478/1246; jrefit 12561/2351/1305; logit 12561/1436/1615 | user mentioned using a lavender scented shampoo at Trader Jo |
| c14c00dd_swap | success | answer_production | 0.92 | competitor | False | j56 14943/1679/1005; jrefit 14943/1679/1022; logit 14943/1299/1273 | user mentioned using a lavender scented shampoo that was pic |
| c19f7a0b_clean | refusal | answer_production | 0.56 | abstain | True | j56 117/281/215; jrefit 117/281/226; logit 86/80/325 | - Homecoming time mentioned: 6:30 pm on weekdays Answer: unk |
| c19f7a0b_swap | refusal | answer_production | 0.56 | abstain | True | j56 117/291/246; jrefit 117/291/262; logit 88/85/371 | - Homecoming time mentioned: 6:00 pm on weekdays Answer: unk |
| c5e8278d_clean | success | answer_production | 0.89 | gold | True | j56 758/10577/265; jrefit 1074/12852/265; logit 1547/18129/265 | user mentioned their old name was Johnson before changing it |
| c5e8278d_swap | success | answer_production | 0.89 | competitor | True | j56 670/10147/262; jrefit 949/10904/262; logit 1464/10904/262 | user mentioned their old name was IKEA before changing it to |
| c7cf7dfd_clean | success | answer_production | 0.92 | gold | True | j56 492/7429/1623; jrefit 194/8277/1716; logit 265/8277/1697 | name_of_online_store = Nostalgia Answer: Nostalgia |
| c7cf7dfd_swap | success | answer_production | 0.92 | competitor | True | j56 518/3968/1351; jrefit 193/3968/1452; logit 504/3968/1418 | name_of_online_store = Jaipur Rugs Answer: Jaipur Rugs |
| c7dc5443_clean | refusal | answer_production | 0.86 | abstain | True | j56 84/29/873; jrefit 84/29/925; logit 84/22/1062 | user mentioned their volleyball team's record as 3-2 on 2023 |
| c7dc5443_swap | refusal | answer_production | 0.86 | other | False | j56 95/35/563; jrefit 95/35/600; logit 92/24/669 | user mentioned their volleyball team, the Net Ninjas, has a  |
| cc06de0d_clean | refusal | indeterminate | - | abstain | True | j56 161/142/342; jrefit 161/142/349; logit 153/101/352 | facts: taxi cost; train fare; train expense calculation Answ |
| cc06de0d_swap | refusal | indeterminate | - | abstain | True | j56 171/137/318; jrefit 171/137/318; logit 153/112/320 | facts: taxi cost; train fare; train expense calculation Answ |
| ccb36322_clean | success | indeterminate | - | gold | True | j56 40/193/118; jrefit 40/321/118; logit 40/2538/118 | user has been using Spotify recently based on the conversati |
| ccb36322_swap | success | indeterminate | - | abstain | True | j56 43/275/11; jrefit 43/365/11; logit 43/1060/11 | user mentioned using Netflix for a while now and has been wa |
| cf22b7bf_clean | wrong | answer_production | 0.86 | gold | True | j56 23/92/596; jrefit 23/92/620; logit 23/92/569 | user mentioned losing 5 pounds in the past month and 10 poun |
| cf22b7bf_swap | wrong | answer_production | 0.86 | abstain | True | j56 26/95/466; jrefit 26/95/494; logit 26/95/472 | user mentioned losing about 5 pounds in the past month and 5 |
| d52b4f67_clean | wrong | answer_production | 0.92 | other | True | j56 12707/3374/1108; jrefit 11131/3052/1187; logit 3683/6147/1101 | user attended my cousin's wedding at the Grand Ballroom on 2 |
| d52b4f67_swap | wrong | answer_production | 0.92 | other | True | j56 12722/3384/1106; jrefit 11146/3085/1170; logit 3862/5671/1107 | user attended my cousin's wedding at The Glass Menagerie on  |
| d596882b_clean | wrong | answer_production | 0.92 | other | False | j56 766/828/345; jrefit 630/846/352; logit 344/191/381 | user asked about vegan restaurants in New York City last tim |
| d596882b_swap | wrong | answer_production | 0.92 | other | False | j56 764/137/360; jrefit 632/137/367; logit 550/137/396 | user asked about vegan restaurants in New York City last tim |
| dd2973ad_clean | wrong | answer_production | 0.89 | other | True | j56 69/330/2174; jrefit 69/330/2296; logit 47/173/2776 | user mentioned having a doctor's appointment on Thursday, 20 |
| dd2973ad_swap | wrong | answer_production | 0.89 | other | True | j56 74/333/2071; jrefit 74/333/2203; logit 49/182/2581 | user mentioned having a doctor's appointment on Thursday, 20 |
| e01b8e2f_clean | refusal | answer_production | 0.92 | abstain | False | j56 57/1941/1; jrefit 57/1896/1; logit 57/1675/1 | unknown |
| e01b8e2f_swap | refusal | answer_production | 0.92 | abstain | False | j56 256/142/1; jrefit 256/142/1; logit 241/142/1 | unknown |
| e8a79c70_clean | success | answer_production | 0.86 | gold | False | j56 4/13/5; jrefit 4/13/5; logit 4/8/7 | Ingredients: 2-3 eggs |
| e8a79c70_swap | success | answer_production | 0.86 | abstain | True | j56 8/6/7; jrefit 8/6/8; logit 8/6/10 | Ingredients: 10 pounds of eggs (which seems incorrect for a  |
| e982271f_clean | wrong | answer_production | 0.92 | other | True | j56 2441/50/626; jrefit 2441/93/626; logit 91/4059/626 | user mentioned the last venue recommended was The Fillmore f |
| e982271f_swap | wrong | answer_production | 0.92 | other | True | j56 2286/39/580; jrefit 2286/81/580; logit 92/3825/580 | user mentioned the last venue recommended was The Fillmore f |
| ec81a493_clean | success | answer_production | 0.86 | gold | False | j56 198/84/533; jrefit 198/84/555; logit 153/35/577 | user mentioned having a signed poster from their favorite ar |
| ec81a493_swap | success | answer_production | 0.86 | competitor | False | j56 211/81/624; jrefit 211/81/642; logit 156/32/677 | user mentioned having a signed poster from their favorite ar |
| ed4ddc30_clean | refusal | answer_production | 0.89 | other | True | j56 9/26/365; jrefit 9/26/388; logit 9/25/405 | user mentioned having 30 dozen eggs stocked up in the fridge |
| ed4ddc30_swap | refusal | answer_production | 0.89 | abstain | True | j56 10/21/369; jrefit 10/21/389; logit 10/20/402 | user mentioned having 30 dozen eggs stocked up in the fridge |
| f523d9fe_clean | success | answer_production | 0.92 | abstain | True | j56 3613/21873/27; jrefit 3390/21873/27; logit 8138/13301/27 | user, 2023/05/26 (Fri) 18:07: I used "doc martin" as an exam |
| f523d9fe_swap | success | answer_production | 0.92 | competitor | True | j56 3581/14942/30; jrefit 3261/14942/30; logit 13587/13439/30 | user, 2023/05/26 (Fri) 18:07: The user mentioned "Trader Joe |
| f9e8c073_clean | success | answer_production | 0.89 | gold | False | j56 957/937/134; jrefit 539/539/134; logit 2721/4069/134 | user, 2023/10/30 (Mon) 07:19: I attended five sessions of th |
| f9e8c073_swap | success | answer_production | 0.89 | competitor | False | j56 954/909/135; jrefit 539/539/135; logit 2857/3412/135 | user, 2023/10/30 (Mon) 07:19: I attended six sessions of the |
| faba32e5_clean | success | answer_production | 0.89 | gold | False | j56 49/352/1750; jrefit 49/343/1864; logit 49/352/1979 | user mentioned that Alex marinated the BBQ ribs in a special |
| faba32e5_swap | success | answer_production | 0.89 | competitor | False | j56 43/251/2056; jrefit 43/251/2155; logit 43/251/2371 | user mentioned that Alex marinated the BBQ ribs in a special |
| fca762bc_clean | wrong | answer_production | 0.92 | other | True | j56 226/3618/871; jrefit 226/3618/911; logit 226/3618/878 | facts: language learning apps recommendation, mnemonic-based |
| fca762bc_swap | wrong | answer_production | 0.92 | competitor | True | j56 173/2203/1010; jrefit 173/2203/1031; logit 173/2203/1055 | - user asked about language learning apps - user specificall |
| fea54f57_clean | success | answer_production | 0.92 | gold | False | j56 2725/1210/844; jrefit 2396/1241/874; logit 2725/696/916 | user mentioned that the song "Evolution" best exemplifies th |
| fea54f57_swap | success | answer_production | 0.92 | competitor | False | j56 3061/1059/1047; jrefit 2379/1300/1087; logit 4248/710/1120 | user mentioned that the song "Johnson" best exemplifies the  |
| gpt4_468eb064_clean | wrong | answer_production | 0.92 | gold | True | j56 202/4914/44; jrefit 202/5204/44; logit 202/7593/44 | user, 2023/04/11 (Tue) 22:13: I was talking with Emma, a fre |
| gpt4_468eb064_swap | wrong | answer_production | 0.92 | competitor | True | j56 270/3666/30; jrefit 329/3666/30; logit 1121/3666/30 | user, 2023/04/11 (Tue) 22:13: I caught up with Absinthe, a f |
