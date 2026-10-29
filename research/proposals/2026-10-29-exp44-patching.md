# §44 proposal (rev 0): is the answer causally in the residual stream at the answer position, and at what depth?

**Author:** Advisor, 2026-10-29. **Order:** the user, Advisor pane, 2026-10-29: "do the patching experiment" (D-44).
**Status:** pre-registered before any run. Development on folds 0–1 only; confirmation folds 2–4 read once after the
choices below are committed. Laptop, $0, unless the 7B does not fit (then one quote, ≤ $0.59, the user's line first).

## 1. Question

§34 and §43 read *when the gold's first token becomes readable* at the answer position P1 with two lenses, and the
lenses disagree on the Qwen 7B: J-lens median emergence depth 0.464, logit lens 0.821 (same 330 rows, depth =
layer / n_layers). On the Qwen 3B they agree (0.889 vs 0.917). A lens is a readout; whether the mid-stack state
*carries* the answer is a causal question. The user's framing: are the lens results trivial, and are our tools
limited? This experiment answers the one resolvable piece: **at which depth is the P1 residual sufficient for the
model to produce the gold, given no evidence anywhere else in the context?**

## 2. Method: single-position activation patching, clean → evidence-cut

For a row with the evidence in view ("clean") build the evidence-cut prompt as in §38 / §34c (`ablate_evidence`:
evidence lines removed, the rest identical, `Answer:` appended, forced whitespace as §34). Three forward passes:

- **clean:** record the residual at P1 after every layer, h_clean(l), l = 0..L−1 (L = n_layers).
- **corrupt:** the cut prompt, KV cache up to P1−1, then the P1 token; m_corrupt = log p(gold first token) at the
  final layer; also the gold's final rank.
- **patched(l), for each l:** the corrupt run, but at layer l's output the P1 residual is overwritten with
  h_clean(l) (the existing `overwrite_spec` hook; layers < l at P1 are irrelevant since they are overwritten; layers
  > l run on the corrupt cache). m_patched(l) = log p(gold) at the final layer.

Recovery: R(l) = (m_patched(l) − m_corrupt) / (m_clean − m_corrupt), clipped to [−0.25, 1.25] for plotting, unclipped
for the statistics. By construction R(L−1) = 1 (the final residual determines the logits) and R(0) ≈ 0 (the P1 token's
embedding is the same in both runs); both are asserted on every row (|R(L−1) − 1| < 0.01; |R(0)| < 0.05) as the
sanity bound, exit on failure. **Causal emergence depth** d_c = (first l with R(l) ≥ 0.5) / L. For comparison the
row's **logit-lens depth** d_logit and **J-lens depth** d_jlens are the §34 numbers already on disk
(`runs/exp34_jlens/{qwen3b,qwen7b}`; first layer with the gold in the top-10, same convention).

Cost per row: two full forwards plus L single-token forwards on the cache, so about 30 s for the 3B in fp32 on the
laptop; the 7B in bf16 on the CPU is measured at stage 0 (an overnight laptop run is acceptable; otherwise a pod).

## 3. Population and rows

The §34 population (330 paired rows; folds 0–1 = 126 dev, folds 2–4 = 204 confirmation), both Qwen sizes. The
claims are read on the **evidence-dependent rows**: clean final gold rank ≤ 10 and corrupt final gold rank > 10
(there is something to recover, and the cut removes it). This filter is outcome-free with respect to the judge and is
fixed here; its count is reported per model and fold set. On dev, if fewer than 40 of 126 rows survive for a model,
the filter is relaxed to corrupt final rank > 3 **before** any confirmation read, and the change is recorded in rev 1.

## 4. Claims (confirmation folds 2–4, evidence-dependent rows, percentile bootstrap, 10,000 draws)

- **C1 (the 7B, decisive).** Median d_c with its 95% CI. Readings fixed in advance:
  - **J-lens right:** the CI's upper bound < 0.65 (the state at mid-depth suffices on most rows).
  - **Logit lens right:** the CI's lower bound > 0.70.
  - Otherwise **both partly right**: the mid-stack state carries part of the answer; C2 quantifies it.
- **C2 (the 7B, continuous).** Mean R at the J-lens median depth (layer round(0.464 × 28) = 13) with CI. Above 0.5
  with the lower bound > 0.5: the mid-stack state carries most of the answer. Below 0.25 with the upper bound < 0.25:
  it carries little.
- **C3 (the 3B, consistency).** Median d_c with CI, expected near the lens medians (0.89–0.92). If the CI's upper
  bound is < 0.75 the lenses *under*-read the 3B as well, and the "late" picture of §34 was a readout artefact; if it
  contains 0.85–0.92 the method agrees with both lenses where they agree.
- **Descriptive.** Per row, d_c − d_jlens and d_c − d_logit (medians, quartiles); the full mean R(l) curve for both
  models; R by outcome class (success / refusal / wrong) and the "unknown" token's log p along the curve; the count of
  rows where patching alone flips the final top-1 to the gold.

## 5. What settles what

- C1 J-lens right + C2 high: the §34 "the 7B has it by mid-stack" sentence is causal, and the logit lens is the
  limited tool. C1 logit right + C2 low: the J-lens mid-stack bump is a lens artefact, and §34/§43 should cite logit
  depths only. Either way the user's "are our tools limited" question gets a concrete answer for the one case where
  the tools disagree.
- C3 guards against the method itself reading early: a single-position patch can only recover what the P1 state
  holds, so an early 3B recovery would be a real finding, not a bug, but it would also mean the lenses are blind to
  non-linearly-encoded answers.

## 6. Build and run order

1. Stage 0 (laptop, free): the patching loop on 5 dev rows of the 3B; the sanity bounds; timing; the 7B's bf16 CPU
   feasibility (peak RSS, seconds per row). Report to the Advisor before any full run.
2. Build review (Advisor): the hook, the cut builder reuse, the filter, the report script with `--confirm` guarded
   as in §43.
3. Dev run (folds 0–1), both models, the dev report; the filter check of §3.
4. Confirmation read, once, after the dev report is committed and the Advisor has recorded the fixed filter.
5. Results review, EXPERIMENTS §44, record changes to the user.

The Phi Part B pod (D-43) finishes first; the laptop build does not touch it. No pod for §44 without a separate quote.

## 7. Caveats fixed in advance

One prompt family, one position (P1, forced), one corruption (cut evidence; the corrupt run may still hold the gold
by plausibility, which the filter removes). Patching P1 only tests what the P1 state holds; answer information sitting
at evidence positions at mid-depth and moved late is "late at P1", which is what the lenses claim to read. The digit
and article token caveats of §43 apply to the gold's first token.
