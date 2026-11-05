# §43 logit-lens profile at P1: llama3b vs Qwen 2.5 3B, dev folds 0-1 only

Descriptive; no claims. 126 dev population rows (§35 rule: answerable, all evidence in view, judge label), the same question ids for both readers. Forced mode (an appended `Answer:`), position P1, k = 10, logit lens only (identity lens; no J-lens fitted). Qwen rows: the logit variant of `runs/exp34_jlens/qwen3b`.
Depth = layer / n_layers (§34's convention); the final layer is (n-1)/n. The 2026-10-26 briefing appendix labels depth layer / (n-1), so its depth column runs to 1.00; compare by layer index or by this column.
Classes (§34 `classes_of`, judge labels): success / refusal (judge `unknown`) / answered wrong. Refusal ids: llama3b [9987, 22435], qwen3b [9788, 21693] (" unknown", " Unknown").

## llama3b

| depth | llama3b layer (of 28) | logit: gold in top-10 | logit: gold top-1 | logit: median gold rank, successes (n=71) | any unknown id top-1 on refusals (n=13) |
|---|---|---|---|---|---|
| 0.000 | 0 | 0.00 | 0.00 | 12,893 | 0.00 |
| 0.036 | 1 | 0.00 | 0.00 | 8,329 | 0.00 |
| 0.071 | 2 | 0.00 | 0.00 | 6,125 | 0.00 |
| 0.107 | 3 | 0.00 | 0.00 | 11,435 | 0.00 |
| 0.143 | 4 | 0.00 | 0.00 | 14,926 | 0.00 |
| 0.179 | 5 | 0.00 | 0.00 | 25,609 | 0.00 |
| 0.214 | 6 | 0.00 | 0.00 | 32,055 | 0.00 |
| 0.250 | 7 | 0.00 | 0.00 | 59,487 | 0.00 |
| 0.286 | 8 | 0.00 | 0.00 | 69,693 | 0.00 |
| 0.321 | 9 | 0.00 | 0.00 | 50,674 | 0.00 |
| 0.357 | 10 | 0.00 | 0.00 | 59,681 | 0.00 |
| 0.393 | 11 | 0.00 | 0.00 | 51,869 | 0.00 |
| 0.429 | 12 | 0.00 | 0.00 | 54,576 | 0.00 |
| 0.464 | 13 | 0.00 | 0.00 | 50,409 | 0.00 |
| 0.500 | 14 | 0.02 | 0.00 | 29,658 | 0.08 |
| 0.536 | 15 | 0.03 | 0.01 | 21,981 | 0.08 |
| 0.571 | 16 | 0.02 | 0.00 | 6,917 | 0.00 |
| 0.607 | 17 | 0.06 | 0.00 | 2,038 | 0.08 |
| 0.643 | 18 | 0.09 | 0.00 | 327 | 0.15 |
| 0.679 | 19 | 0.17 | 0.03 | 120 | 0.54 |
| 0.714 | 20 | 0.32 | 0.15 | 14 | 0.62 |
| 0.750 | 21 | 0.41 | 0.17 | 3 | 0.62 |
| 0.786 | 22 | 0.46 | 0.14 | 3 | 0.62 |
| 0.821 | 23 | 0.49 | 0.21 | 2 | 0.54 |
| 0.857 | 24 | 0.55 | 0.28 | 2 | 0.62 |
| 0.893 | 25 | 0.65 | 0.36 | 1 | 0.54 |
| 0.929 | 26 | 0.70 | 0.30 | 2 | 0.62 |
| 0.964 | 27 | 0.80 | 0.38 | 1 | 0.62 |

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
| llama3b | all | 126 | 0.198 | 0.000 | 0.802 |
| llama3b | success | 71 | 0.056 | 0.000 | 0.944 |
| llama3b | refusal | 13 | 0.462 | 0.000 | 0.538 |
| llama3b | answered wrong | 42 | 0.357 | 0.000 | 0.643 |
| llama3b | all failures | 55 | 0.382 | 0.000 | 0.618 |
| qwen3b | all | 126 | 0.294 | 0.024 | 0.683 |
| qwen3b | success | 54 | 0.259 | 0.000 | 0.741 |
| qwen3b | refusal | 32 | 0.375 | 0.062 | 0.562 |
| qwen3b | answered wrong | 40 | 0.275 | 0.025 | 0.700 |
| qwen3b | all failures | 72 | 0.319 | 0.042 | 0.639 |

## Logit-lens emergence depth of the gold at P1 (first layer with gold in top-10; questions where it emerges)

| reader | n emerged | n read | q1 | median | q3 |
|---|---|---|---|---|---|
| llama3b | 101 | 126 | 0.714 | 0.750 | 0.893 |
| qwen3b | 89 | 126 | 0.889 | 0.917 | 0.944 |

## Like-for-like checks

- Prompt text identical (sha256) for 126 of 126 rows.
- Gold first-token text identical (stripped) for 100 of 126 rows; final-layer gold in top-10 on that subset: llama3b 0.80, qwen3b 0.61.
- Digit tokenisation caveat: Llama 3.2 and Phi-4-mini put up to 3 digits in one token, Qwen 2.5 and Gemma 3 one digit per token. On the 53 rows whose gold starts with a digit the 'gold first token' is the whole number (up to 3 digits) for Llama/Phi and only its first digit for Qwen/Gemma, so a top-k hit is a harder target for Llama/Phi there. Final-layer gold in top-10 on numeric golds: llama3b 0.87, qwen3b 0.98.
- Article-first golds (caveat, a reporting split; the §34 target token is unchanged): when the gold's first token is an article (' The', ' A', ' a', ...), the final layer often predicts the content word instead and the article drops out of the top-10, which shows up as a B (or A) row. Of llama3b's 0 B rows, 0 have an article-first gold. Split (final top-10 / final top-1 / A / B):

| reader | article n | top-10 | top-1 | A | B | non-article n | top-10 | top-1 | A | B |
|---|---|---|---|---|---|---|---|---|---|---|
| llama3b | 15 | 0.667 | 0.000 | 0.333 | 0.000 | 111 | 0.820 | 0.432 | 0.180 | 0.000 |
| qwen3b | 15 | 0.467 | 0.000 | 0.467 | 0.067 | 111 | 0.712 | 0.351 | 0.270 | 0.018 |
