# Contributions statement (draft, 2026-10-29)

Ranked by how well each survives scrutiny; prior art beside each; status as of the Gemma pod (Phi running, §44 running).

1. **Two kinds of in-view failure, told apart causally.** Qwen 3B: refusal is a late decision; a content-free direction
   nudged at one position opens refusals, about half come out right (§38 0.517 gold with evidence, 0.183 without),
   more than the note held (§38 pod 31/60 right vs 23/60 note-gold); transfers to LoCoMo with the setting fixed (§40,
   replicated smaller in §42). Llama 3B: the same nudge opens refusals but none come out right beyond the 2/13 whose
   note held the gold; the failure is upstream (§43 stage 1). Prior art: Arditi et al. 2024 (safety refusal direction);
   Ferrando et al. 2025 (entity-knowledge features drive abstention on parametric facts). Ours: in-context evidence
   present; the direction encodes extractability, not answerability (§42). Status: Qwen confirmed; Llama 13 rows.
2. **Sealed-fold, cross-family test of the lens picture.** C-majority failures and rank-predicts-correctness (B2) in
   3/3 families; larger-holds-more-clearly (B1) 2/3; small-later 1/3 (Qwen, 0.10 of depth on the logit lens). Prior
   art: hidden-knowledge results on parametric facts; logit/tuned-lens readability depth. Status: Phi pending.
3. **"Present" deflated to "plausible"; readouts disagree.** §34c: ~60% of top-10 presence survives evidence removal.
   J-lens vs logit on the Qwen 7B: 0.464 vs 0.821. §44 (patching) adjudicates. Status: running.
4. **The reader is the ceiling for small-model memory systems.** 3B + perfect evidence < 7B + none; the learned
   controller ≈ a 31-parameter logistic regression (F3). Status: confirmed.
5. **Negatives.** Self-verification fails (§36b, §37); selective steering fails (§41, §42); Gemma and Phi do not refuse
   in view (§43 stage 0).
6. **Methodological.** Pre-registration + sealed folds + independent recount at ≈ $19 total; dev→confirm shrinkage
   (B2 0.847 → 0.677); article and digit token artefacts as cross-family caveats.

**Not novel:** the layer-profile shape; a fitted lens reading earlier than the logit lens.

**Gap:** no literature note for §33–§44 (abstention directions, hidden knowledge, patching). Advisor to write one
before the §44 result, positioning 1–3.
