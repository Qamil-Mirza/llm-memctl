## Accuracy on 470 answerable questions (one blind Qwen 7B judge)

| reader | arm | accuracy | answered-wrong | unknown | P(correct \| answered) | P(correct \| in view) | P(all in view) | prompt tokens | judge-wrong with refusal phrase | empty / loop / special-token |
|---|---|---|---|---|---|---|---|---|---|---|
| llama3b | fifo_top5_t3000 | 0.266 | 0.304 | 0.430 | 0.466 | 0.410 | 0.498 | 2,964 | 0 | 0 / 0 / 0 |
| llama3b | fixed8 | 0.428 | 0.323 | 0.249 | 0.569 | 0.545 | 0.702 | 1,049 | 0 | 0 / 0 / 0 |
| llama3b | fixed16 | 0.306 | 0.326 | 0.368 | 0.485 | 0.368 | 0.747 | 2,569 | 2 | 0 / 0 / 0 |
| llama8b | fifo_top5_t3000 | 0.479 | 0.257 | 0.264 | 0.650 | 0.756 | 0.498 | 2,964 | 3 | 0 / 0 / 0 |
| llama8b | fixed8 | 0.540 | 0.328 | 0.132 | 0.623 | 0.679 | 0.702 | 1,049 | 2 | 0 / 0 / 0 |
| llama8b | fixed16 | 0.534 | 0.328 | 0.138 | 0.620 | 0.638 | 0.747 | 2,569 | 6 | 0 / 0 / 0 |
| gemma4b | fifo_top5_t3000 | 0.357 | 0.457 | 0.185 | 0.439 | 0.538 | 0.498 | 2,964 | 0 | 0 / 0 / 0 |
| gemma4b | fixed8 | 0.504 | 0.409 | 0.087 | 0.552 | 0.627 | 0.702 | 1,049 | 0 | 0 / 0 / 0 |
| gemma4b | fixed16 | 0.421 | 0.481 | 0.098 | 0.467 | 0.499 | 0.747 | 2,569 | 0 | 0 / 0 / 0 |
| gemma12b | fifo_top5_t3000 | 0.489 | 0.347 | 0.164 | 0.585 | 0.748 | 0.498 | 2,964 | 0 | 0 / 0 / 0 |
| gemma12b | fixed8 | 0.596 | 0.319 | 0.085 | 0.651 | 0.733 | 0.702 | 1,049 | 0 | 0 / 0 / 0 |
| gemma12b | fixed16 | 0.574 | 0.353 | 0.072 | 0.619 | 0.681 | 0.747 | 2,569 | 0 | 0 / 0 / 0 |
| phi4mini | fifo_top5_t3000 | 0.253 | 0.560 | 0.187 | 0.312 | 0.410 | 0.498 | 2,964 | 3 | 2 / 39 / 0 |
| phi4mini | fixed8 | 0.385 | 0.532 | 0.083 | 0.420 | 0.482 | 0.702 | 1,049 | 3 | 12 / 13 / 29 |
| phi4mini | fixed16 | 0.362 | 0.530 | 0.109 | 0.406 | 0.433 | 0.747 | 2,569 | 3 | 0 / 40 / 1 |
| phi4 | fifo_top5_t3000 | 0.502 | 0.168 | 0.330 | 0.749 | 0.803 | 0.498 | 2,964 | 0 | 0 / 0 / 0 |
| phi4 | fixed8 | 0.630 | 0.164 | 0.206 | 0.794 | 0.803 | 0.702 | 1,049 | 0 | 0 / 0 / 0 |
| phi4 | fixed16 | 0.591 | 0.200 | 0.209 | 0.747 | 0.721 | 0.747 | 2,569 | 0 | 0 / 0 / 0 |

## Pre-registered claims per family (non-inferiority margin −0.03; 'better' if the lower bound > 0)

- Llama (a) fixed8 − FIFO @ llama3b: +0.162 (+0.115, +0.209) → **LIFTS**
- Llama (a) fixed8 − FIFO @ llama8b: +0.062 (+0.015, +0.111) → **LIFTS**
- Llama (b) fixed8 @ llama3b vs FIFO @ llama8b: -0.051 (-0.100, -0.004) → **NOT SHOWN**; share of the size gap closed 0.76 (0.56, 0.98)
- Gemma 3 (a) fixed8 − FIFO @ gemma4b: +0.147 (+0.102, +0.194) → **LIFTS**
- Gemma 3 (a) fixed8 − FIFO @ gemma12b: +0.106 (+0.062, +0.153) → **LIFTS**
- Gemma 3 (b) fixed8 @ gemma4b vs FIFO @ gemma12b: +0.015 (-0.028, +0.057) → **NON-INFERIOR**; share of the size gap closed 1.11 (0.81, 1.55)
- Phi-4 (a) fixed8 − FIFO @ phi4mini: +0.132 (+0.079, +0.183) → **LIFTS**
- Phi-4 (a) fixed8 − FIFO @ phi4: +0.128 (+0.085, +0.172) → **LIFTS**
- Phi-4 (b) fixed8 @ phi4mini vs FIFO @ phi4: -0.117 (-0.168, -0.066) → **NOT SHOWN**; share of the size gap closed 0.53 (0.34, 0.72)

## Cross-family test, declared in advance: does the smaller size gain most?

(lift at small − lift at mid, paired by question; HOLDS if the lower bound > 0, REVERSED if the upper bound < 0)

- Llama: +0.100 (+0.038, +0.164) → **HOLDS**
- Gemma 3: +0.040 (-0.017, +0.098) → **UNDETERMINED**
- Phi-4: +0.004 (-0.060, +0.066) → **UNDETERMINED**

## fixed8 − FIFO within each reader, and where the gain comes from

- llama3b: +0.162 (+0.115, +0.209); unknown 0.430 → 0.249; P(correct | answered) 0.466 → 0.569
- llama8b: +0.062 (+0.015, +0.111); unknown 0.264 → 0.132; P(correct | answered) 0.650 → 0.623
- gemma4b: +0.147 (+0.102, +0.194); unknown 0.185 → 0.087; P(correct | answered) 0.439 → 0.552
- gemma12b: +0.106 (+0.062, +0.153); unknown 0.164 → 0.085; P(correct | answered) 0.585 → 0.651
- phi4mini: +0.132 (+0.079, +0.183); unknown 0.187 → 0.083; P(correct | answered) 0.312 → 0.420
- phi4: +0.128 (+0.085, +0.172); unknown 0.330 → 0.206; P(correct | answered) 0.749 → 0.794

## fixed16 − fixed8 within each reader (context-rot probe, descriptive)

- llama3b: -0.121 (-0.166, -0.077)
- llama8b: -0.006 (-0.053, +0.040)
- gemma4b: -0.083 (-0.126, -0.040)
- gemma12b: -0.021 (-0.060, +0.017)
- phi4mini: -0.023 (-0.070, +0.023)
- phi4: -0.038 (-0.077, +0.000)
