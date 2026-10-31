# §45 report (j56-jrefit-logit, half: dev)

Descriptive. Readable (fixed, rev 1) = the entity's space-form id in the lens top-10 at a counted position (the last prompt token and the answer span: F1 every generated token, F2 / stage 3 the tokens after 'Answer:'), or its bare-form id at the last prompt token; criterion (a) before the final layer. Rows marked '| ...' are descriptive beside. A miss is 'not recovered by this lens', never 'not computed'. **The held half decides**; dev is shown first for presentation, pooled beside.

Caveat: in F2 the memory lines carry dates that increase with line order, so in TU the timestamp order is a second update cue beside the wording (a format effect, which F2 is there to show).

## Stage 1: solved prompts (256 of 432)

Outcomes: {"gold": 256, "other": 73, "abstain": 76, "competitor": 27}
Answer marker present (format|present): {"F1|False": 216, "F2|False": 1, "F2|True": 215}

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

### Fit stability (j56 vs jrefit, before the final layer)

- solved prompts: agreement 0.996 over 971 calls -> **stable**; Spearman median 0.980 (quartiles [0.9732096875500564, 0.980179487828752, 0.9877608723196698], n=971)
- all prompts (beside): agreement 0.996 over 1656 calls -> **stable**; Spearman median 0.980 (quartiles [0.9707820042409199, 0.9798851418700795, 0.9882207327849615], n=1656)

## Stage 2: matched failures (176, 74 at H2, 137 paired)

- j56: on the matched failures the gold is not readable before the final layer on >= 0.70 of them (j56) (share 0.62, Wilson [0.55, 0.69], n=176)
- jrefit: on the matched failures the gold is not readable before the final layer on >= 0.70 of them (jrefit) (share 0.62, Wilson [0.55, 0.69], n=176)
- logit: on the matched failures the gold is not readable before the final layer on >= 0.70 of them (logit) (share 0.61, Wilson [0.53, 0.68], n=176)

Failures by task, hardness and outcome: {"DE|H0|abstain": 11, "DE|H1|abstain": 11, "DE|H1|other": 4, "DE|H2|abstain": 11, "DE|H2|other": 14, "RC|H0|abstain": 6, "RC|H0|other": 2, "RC|H1|abstain": 9, "RC|H1|other": 18, "RC|H2|abstain": 8, "RC|H2|other": 13, "TU|H0|abstain": 2, "TU|H0|competitor": 3, "TU|H0|other": 6, "TU|H1|abstain": 11, "TU|H1|competitor": 12, "TU|H1|other": 7, "TU|H2|abstain": 7, "TU|H2|competitor": 12, "TU|H2|other": 9}

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
| DE04_v0_F2_H2 | abstain | False | True | DE04_v0_F2_H0 | {"j56": 0, "jrefit": 0, "logit": 32} | Henry lives in Durham based on the information provided. Answer: unknown |
| DE04_v2_F2_H0 | abstain | False | True | DE04_v2_F2_H1 | {"j56": 0, "jrefit": 4, "logit": null} | user mentioned that Henry lives in Durham. Answer: unknown |
| DE04_v3_F1_H2 | other | False | False | DE04_v3_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 31} | Based on the information provided, James works as a cook and James lives in Camb |
| DE04_v3_F2_H1 | abstain | False | True | DE04_v3_F2_H0 | {"j56": 0, "jrefit": 0, "logit": 33} | Henry lives in Durham. Answer: unknown |
| DE06_v0_F2_H1 | abstain | False | True | DE06_v0_F2_H0 | {"j56": 8, "jrefit": 8, "logit": 32} | Sarah works as a doctor and Laura lives in Brighton. Answer: unknown |
| DE06_v0_F2_H2 | abstain | False | True | DE06_v0_F2_H0 | {"j56": 8, "jrefit": 8, "logit": null} | user - Laura lives in Brighton Answer: unknown |
| DE06_v1_F2_H0 | abstain | False | True | - | {} | user stated that Laura lives in Manchester. Answer: unknown |
| DE06_v1_F2_H1 | abstain | False | True | - | {} | Sarah works as a doctor and Laura lives in Manchester. Answer: unknown |
| DE06_v1_F2_H2 | abstain | False | True | - | {} | user - Laura lives in Manchester Answer: unknown |
| DE06_v2_F2_H1 | abstain | False | True | DE06_v2_F2_H0 | {"j56": null, "jrefit": null, "logit": null} | Sarah works as a doctor and Laura lives in Brighton. Answer: unknown |
| DE06_v3_F2_H0 | abstain | False | True | - | {} | user mentioned that Laura lives in Brighton on 2023/03/01 (Wed). Answer: unknown |
| DE06_v3_F2_H1 | abstain | False | True | - | {} | Sarah works as a doctor and Laura lives in Brighton. Answer: unknown |
| DE06_v3_F2_H2 | abstain | False | True | - | {} | user mentioned Laura lives in Brighton on 2023/03/05 (Sun). Answer: unknown |
| DE08_v0_F1_H2 | other | False | False | DE08_v0_F1_H0 | {"j56": 32, "jrefit": 32, "logit": 32} | Based on the information provided:  - Laura lives in Cardiff. - David lives in |
| DE08_v1_F1_H2 | other | False | False | DE08_v1_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 31} | Based on the information provided:  - Laura lives in Cardiff. - David lives in |
| DE08_v2_F1_H2 | other | False | False | DE08_v2_F1_H0 | {"j56": null, "jrefit": null, "logit": 32} | Based on the information provided:  - Laura lives in Cardiff. - David lives in |
| DE08_v3_F1_H2 | other | False | False | DE08_v3_F1_H0 | {"j56": 31, "jrefit": 31, "logit": 32} | Based on the information provided:  - David lives in Liverpool. - Alice works as |
| DE08_v3_F2_H2 | abstain | False | True | DE08_v3_F2_H0 | {"j56": 32, "jrefit": 32, "logit": 32} | Rachel lives in Chester. Answer: unknown |
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
| RC00_v0_F1_H1 | other | False | False | RC00_v0_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 31} | Based on the information provided:  - Laura is Kate's supervisor. - Laura works |
| RC00_v0_F1_H2 | other | False | False | RC00_v0_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 31} | Based on the information provided:  - Sarah lives in Liverpool and works in Brig |
| RC00_v1_F1_H0 | other | False | False | - | {} | Based on the information provided:  - Kate's supervisor is David. - David works |
| RC00_v1_F1_H1 | other | False | False | - | {} | Based on the information provided:  - Sarah lives in Liverpool. - Paul lives in |
| RC00_v1_F1_H2 | other | False | False | - | {} | Based on the information provided, we know that Kate's supervisor is David. Howe |
| RC00_v1_F2_H1 | abstain | False | True | RC00_v1_F2_H0 | {"j56": 0, "jrefit": 0, "logit": 32} | Kate's supervisor is David and David works in Durham. Answer: unknown |
| RC00_v1_F2_H2 | abstain | False | True | RC00_v1_F2_H0 | {"j56": 0, "jrefit": 0, "logit": 32} | Sarah lives in Liverpool and works in Brighton on March 11th, but her location c |
| RC00_v3_F1_H2 | other | False | False | RC00_v3_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 31} | To determine the city where Kate's supervisor works, we need to identify who Kat |
| RC00_v3_F2_H0 | abstain | False | True | RC00_v3_F2_H1 | {"j56": 0, "jrefit": 0, "logit": 32} | user mentioned Laura works in Durham and Kate's supervisor is Laura. Answer: unk |
| RC02_v0_F1_H1 | other | False | False | RC02_v0_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 31} | Based on the information provided:  - David's supervisor is Peter. - Peter works |
| RC02_v0_F1_H2 | other | False | False | RC02_v0_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 31} | Based on the information provided, we know that David's supervisor is Peter. Fro |
| RC02_v1_F1_H1 | other | False | False | RC02_v1_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 31} | Based on the information provided:  - David's supervisor is James. - James works |
| RC02_v1_F1_H2 | other | False | False | RC02_v1_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 31} | Based on the information provided, we know that David's supervisor is James. Fro |
| RC02_v3_F1_H1 | other | False | False | RC02_v3_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 30} | To determine the city where David's supervisor works, we need to identify who Da |
| RC02_v3_F2_H0 | abstain | False | True | - | {} | user mentioned Peter works in Manchester and David's supervisor is Peter. Answer |
| RC02_v3_F2_H1 | abstain | False | True | - | {} | user mentioned that David's supervisor is Peter, and user stated that Peter work |
| RC02_v3_F2_H2 | abstain | False | True | - | {} | Sarah works in Durham and Peter works in Manchester. David's supervisor is Peter |
| RC04_v0_F1_H1 | other | False | False | RC04_v0_F1_H0 | {"j56": 9, "jrefit": 9, "logit": 32} | Based on the information provided:  - Alice works as a writer. - James's |
| RC04_v0_F2_H1 | abstain | False | True | RC04_v0_F2_H0 | {"j56": 8, "jrefit": 8, "logit": null} | James's supervisor is Sarah and Sarah works in Brighton. Answer: unknown |
| RC04_v0_F2_H2 | abstain | False | True | RC04_v0_F2_H0 | {"j56": 8, "jrefit": 8, "logit": null} | Alice - works as a writer   Henry - works as a designer   Kate - works as a doct |
| RC04_v1_F1_H1 | other | False | False | RC04_v1_F1_H0 | {"j56": 9, "jrefit": 9, "logit": 31} | Based on the information provided:  - Alice works as a writer. - James's |
| RC04_v1_F2_H1 | abstain | False | True | RC04_v1_F2_H0 | {"j56": 8, "jrefit": 8, "logit": null} | James's supervisor is Paul and Paul works in Brighton. Answer: unknown |
| RC04_v2_F1_H1 | other | False | False | RC04_v2_F1_H0 | {"j56": 16, "jrefit": 16, "logit": 31} | Based on the information provided:  - Alice works as a writer. - James's |
| RC04_v3_F1_H2 | other | False | False | RC04_v3_F1_H0 | {"j56": 9, "jrefit": 9, "logit": 31} | Based on the information provided:  - James's supervisor is Sarah. - Sarah works |
| RC04_v3_F2_H0 | abstain | False | True | RC04_v3_F2_H1 | {"j56": 9, "jrefit": 9, "logit": 31} | Sarah works in Brighton. James's supervisor is Sarah. Answer: unknown |
| RC04_v3_F2_H2 | abstain | False | True | RC04_v3_F2_H1 | {"j56": 9, "jrefit": 9, "logit": 31} | Alice works as a writer and is mentioned as James's supervisor on 2023/03/10 (Fr |
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
| RC10_v0_F1_H1 | other | False | False | RC10_v0_F1_H0 | {"j56": 32, "jrefit": 32, "logit": 31} | Based on the information provided:  - Henry's supervisor is Emma. - Emma works |
| RC10_v0_F1_H2 | other | False | False | RC10_v0_F1_H0 | {"j56": 32, "jrefit": 32, "logit": 31} | To determine the city where Henry's supervisor works, we need to identify who He |
| RC10_v1_F1_H0 | other | False | False | - | {} | Based on the information provided, we know that Henry's supervisor is Kate and t |
| RC10_v1_F1_H1 | other | False | False | - | {} | Based on the information provided:  - Henry's supervisor is Kate. - Kate works |
| RC10_v1_F1_H2 | other | False | False | - | {} | To determine the city where Henry's supervisor works, we need to identify who He |
| RC10_v3_F1_H2 | other | False | False | RC10_v3_F1_H0 | {"j56": 31, "jrefit": 31, "logit": 31} | Based on the information provided:  - Henry's supervisor is Emma. - Emma works |
| RC10_v3_F2_H0 | abstain | False | True | RC10_v3_F2_H1 | {"j56": 9, "jrefit": 9, "logit": 31} | Emma works in York. Henry's supervisor is Emma. Answer: unknown |
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
| TU10_v0_F1_H0 | competitor | False | False | - | {} | Based on the information provided:  1. Initially, Alice lived in Cambridge. 2 |
| TU10_v0_F1_H1 | competitor | False | False | - | {} | Based on the information provided:  1. Initially, Alice lived in Cambridge. 2 |
| TU10_v0_F1_H2 | other | True | False | - | {} | Alice moved from Cambridge to Chester, so she now lives in Chester. |
| TU10_v1_F1_H1 | competitor | False | False | TU10_v1_F1_H0 | {"j56": 0, "jrefit": 0, "logit": 30} | Based on the information provided:  1. Initially, Alice lived in Chester. 2 |
| TU10_v1_F1_H2 | other | True | False | TU10_v1_F1_H0 | {"j56": 20, "jrefit": 20, "logit": 30} | Alice moved from Chester to Cambridge later, so she now lives in Cambridge. |
| TU10_v1_F2_H2 | abstain | False | True | TU10_v1_F2_H0 | {"j56": 0, "jrefit": 0, "logit": 32} | Alice lived in Chester earlier but moved to Cambridge on 2023/03/07 (Tue). Answe |
| TU10_v2_F1_H0 | other | True | False | TU10_v2_F1_H1 | {"j56": 22, "jrefit": 21, "logit": null} | Before moving to Chester, Alice lived in Cambridge. |
| TU10_v2_F1_H2 | other | True | False | TU10_v2_F1_H1 | {"j56": 30, "jrefit": 30, "logit": null} | Based on the information provided, Alice lived in Cambridge before she moved to  |

## Stage 3: the §44 rev 8 rows (0 prompts)

| key | §23 | §44 class (1 nat) | d_T fwd | outcome | marker | lens: gold / substitute / unknown best rank at the last prompt token | generated |
|---|---|---|---|---|---|---|---|
