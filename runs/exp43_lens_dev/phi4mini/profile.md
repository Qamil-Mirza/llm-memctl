# §43 logit-lens profile at P1: phi4mini vs Qwen 2.5 3B, dev folds 0-1 only

Descriptive; no claims. 126 dev population rows (§35 rule: answerable, all evidence in view, judge label), the same question ids for both readers. Forced mode (an appended `Answer:`), position P1, k = 10, logit lens only (identity lens; no J-lens fitted). Qwen rows: the logit variant of `runs/exp34_jlens/qwen3b`.
Depth = layer / n_layers (§34's convention); the final layer is (n-1)/n. The 2026-10-26 briefing appendix labels depth layer / (n-1), so its depth column runs to 1.00; compare by layer index or by this column.
Classes (§34 `classes_of`, judge labels): success / refusal (judge `unknown`) / answered wrong. Refusal ids: phi4mini [17927, 41855], qwen3b [9788, 21693] (" unknown", " Unknown").

## phi4mini

| depth | phi4mini layer (of 32) | logit: gold in top-10 | logit: gold top-1 | logit: median gold rank, successes (n=59) | any unknown id top-1 on refusals (n=3) |
|---|---|---|---|---|---|
| 0.000 | 0 | 0.00 | 0.00 | 193,141 | 0.00 |
| 0.031 | 1 | 0.00 | 0.00 | 190,892 | 0.00 |
| 0.062 | 2 | 0.00 | 0.00 | 193,587 | 0.00 |
| 0.094 | 3 | 0.00 | 0.00 | 191,868 | 0.00 |
| 0.125 | 4 | 0.00 | 0.00 | 190,537 | 0.00 |
| 0.156 | 5 | 0.00 | 0.00 | 178,751 | 0.00 |
| 0.188 | 6 | 0.00 | 0.00 | 173,089 | 0.00 |
| 0.219 | 7 | 0.00 | 0.00 | 173,323 | 0.00 |
| 0.250 | 8 | 0.00 | 0.00 | 165,626 | 0.00 |
| 0.281 | 9 | 0.00 | 0.00 | 168,888 | 0.00 |
| 0.312 | 10 | 0.00 | 0.00 | 175,540 | 0.00 |
| 0.344 | 11 | 0.00 | 0.00 | 182,766 | 0.00 |
| 0.375 | 12 | 0.00 | 0.00 | 181,429 | 0.00 |
| 0.406 | 13 | 0.00 | 0.00 | 117,951 | 0.00 |
| 0.438 | 14 | 0.00 | 0.00 | 41,991 | 0.00 |
| 0.469 | 15 | 0.00 | 0.00 | 157,588 | 0.00 |
| 0.500 | 16 | 0.00 | 0.00 | 65,175 | 0.00 |
| 0.531 | 17 | 0.00 | 0.00 | 43,460 | 0.00 |
| 0.562 | 18 | 0.01 | 0.00 | 31,429 | 0.00 |
| 0.594 | 19 | 0.02 | 0.00 | 16,537 | 0.00 |
| 0.625 | 20 | 0.02 | 0.00 | 12,105 | 0.00 |
| 0.656 | 21 | 0.03 | 0.02 | 14,294 | 0.00 |
| 0.688 | 22 | 0.06 | 0.02 | 4,665 | 0.00 |
| 0.719 | 23 | 0.10 | 0.03 | 5,707 | 0.00 |
| 0.750 | 24 | 0.11 | 0.03 | 5,361 | 0.00 |
| 0.781 | 25 | 0.18 | 0.05 | 2,580 | 0.00 |
| 0.812 | 26 | 0.18 | 0.06 | 3,454 | 0.00 |
| 0.844 | 27 | 0.21 | 0.08 | 756 | 0.00 |
| 0.875 | 28 | 0.24 | 0.11 | 610 | 0.00 |
| 0.906 | 29 | 0.28 | 0.16 | 69 | 0.33 |
| 0.938 | 30 | 0.45 | 0.25 | 3 | 0.33 |
| 0.969 | 31 | 0.78 | 0.44 | 1 | 0.00 |

## qwen3b

| depth | qwen3b layer (of 36) | logit: gold in top-10 | logit: gold top-1 | logit: median gold rank, successes (n=54) | any unknown id top-1 on refusals (n=32) |
|---|---|---|---|---|---|
| 0.000 | 0 | 0.00 | 0.00 | 53,466 | 0.00 |
| 0.028 | 1 | 0.00 | 0.00 | 99,995 | 0.00 |
| 0.056 | 2 | 0.00 | 0.00 | 81,379 | 0.00 |
| 0.083 | 3 | 0.00 | 0.00 | 69,430 | 0.00 |
| 0.111 | 4 | 0.00 | 0.00 | 54,512 | 0.00 |
| 0.139 | 5 | 0.00 | 0.00 | 58,813 | 0.00 |
| 0.167 | 6 | 0.00 | 0.00 | 60,434 | 0.00 |
| 0.194 | 7 | 0.00 | 0.00 | 59,190 | 0.00 |
| 0.222 | 8 | 0.00 | 0.00 | 44,740 | 0.00 |
| 0.250 | 9 | 0.00 | 0.00 | 21,605 | 0.00 |
| 0.278 | 10 | 0.00 | 0.00 | 24,344 | 0.00 |
| 0.306 | 11 | 0.00 | 0.00 | 20,456 | 0.00 |
| 0.333 | 12 | 0.02 | 0.00 | 9,527 | 0.00 |
| 0.361 | 13 | 0.02 | 0.00 | 41,846 | 0.00 |
| 0.389 | 14 | 0.05 | 0.00 | 49,910 | 0.00 |
| 0.417 | 15 | 0.00 | 0.00 | 84,770 | 0.00 |
| 0.444 | 16 | 0.02 | 0.02 | 60,048 | 0.00 |
| 0.472 | 17 | 0.02 | 0.00 | 57,645 | 0.00 |
| 0.500 | 18 | 0.02 | 0.02 | 50,153 | 0.00 |
| 0.528 | 19 | 0.02 | 0.02 | 27,434 | 0.00 |
| 0.556 | 20 | 0.02 | 0.00 | 48,674 | 0.00 |
| 0.583 | 21 | 0.02 | 0.00 | 41,861 | 0.00 |
| 0.611 | 22 | 0.01 | 0.00 | 66,212 | 0.00 |
| 0.639 | 23 | 0.00 | 0.00 | 83,718 | 0.00 |
| 0.667 | 24 | 0.00 | 0.00 | 109,266 | 0.00 |
| 0.694 | 25 | 0.00 | 0.00 | 101,955 | 0.00 |
| 0.722 | 26 | 0.02 | 0.00 | 80,006 | 0.00 |
| 0.750 | 27 | 0.02 | 0.00 | 57,362 | 0.00 |
| 0.778 | 28 | 0.02 | 0.00 | 65,531 | 0.00 |
| 0.806 | 29 | 0.03 | 0.01 | 34,152 | 0.00 |
| 0.833 | 30 | 0.04 | 0.02 | 8,994 | 0.03 |
| 0.861 | 31 | 0.10 | 0.02 | 2,074 | 0.38 |
| 0.889 | 32 | 0.23 | 0.08 | 34 | 0.59 |
| 0.917 | 33 | 0.39 | 0.13 | 8 | 0.72 |
| 0.944 | 34 | 0.63 | 0.22 | 3 | 0.78 |
| 0.972 | 35 | 0.68 | 0.31 | 2 | 0.72 |

## A/B/C at P1 (logit lens; A never in top-10, B in top-10 at some layer but not the last, C in top-10 at the last layer)

| reader | class | n | A | B | C |
|---|---|---|---|---|---|
| phi4mini | all | 126 | 0.206 | 0.016 | 0.778 |
| phi4mini | success | 59 | 0.102 | 0.017 | 0.881 |
| phi4mini | refusal | 3 | 0.000 | 0.000 | 1.000 |
| phi4mini | answered wrong | 64 | 0.312 | 0.016 | 0.672 |
| phi4mini | all failures | 67 | 0.299 | 0.015 | 0.687 |
| qwen3b | all | 126 | 0.294 | 0.024 | 0.683 |
| qwen3b | success | 54 | 0.259 | 0.000 | 0.741 |
| qwen3b | refusal | 32 | 0.375 | 0.062 | 0.562 |
| qwen3b | answered wrong | 40 | 0.275 | 0.025 | 0.700 |
| qwen3b | all failures | 72 | 0.319 | 0.042 | 0.639 |

## Logit-lens emergence depth of the gold at P1 (first layer with gold in top-10; questions where it emerges)

| reader | n emerged | n read | q1 | median | q3 |
|---|---|---|---|---|---|
| phi4mini | 100 | 126 | 0.805 | 0.938 | 0.969 |
| qwen3b | 89 | 126 | 0.889 | 0.917 | 0.944 |

## Like-for-like checks

- Prompt text identical (sha256) for 126 of 126 rows.
- Gold first-token text identical (stripped) for 95 of 126 rows; final-layer gold in top-10 on that subset: phi4mini 0.81, qwen3b 0.61.
- Digit tokenisation caveat: Llama 3.2 and Phi-4-mini put up to 3 digits in one token, Qwen 2.5 and Gemma 3 one digit per token. On the 53 rows whose gold starts with a digit the 'gold first token' is the whole number (up to 3 digits) for Llama/Phi and only its first digit for Qwen/Gemma, so a top-k hit is a harder target for Llama/Phi there. Final-layer gold in top-10 on numeric golds: phi4mini 0.77, qwen3b 0.98.
- Article-first golds (caveat, a reporting split; the §34 target token is unchanged): when the gold's first token is an article (' The', ' A', ' a', ...), the final layer often predicts the content word instead and the article drops out of the top-10, which shows up as a B (or A) row. Of phi4mini's 2 B rows, 0 have an article-first gold. Split (final top-10 / final top-1 / A / B):

| reader | article n | top-10 | top-1 | A | B | non-article n | top-10 | top-1 | A | B |
|---|---|---|---|---|---|---|---|---|---|---|
| phi4mini | 15 | 0.800 | 0.133 | 0.200 | 0.000 | 111 | 0.775 | 0.486 | 0.207 | 0.018 |
| qwen3b | 15 | 0.467 | 0.000 | 0.467 | 0.067 | 111 | 0.712 | 0.351 | 0.270 | 0.018 |
