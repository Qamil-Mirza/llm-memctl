# Review: §43 Llama 3.2 3B stage 1 (steering design gate), run 1057a0d

**Verdict: the gate PASSES on its bars at L17 α2, and the pass is uninformative on the half that matters. Part A for
Llama is eligible for a pod but is not recommended ahead of the Part B lens pods. The user decides.**

Checked from `runs/exp43_llama_dev/llama3b/gate.json` and `report.md`: reproduction 43/43; G_dropped none; ans_min 7
(ceil(0.5 × 13)); n_pairs 8 ≥ 5; sanity (zero-add, self-overwrite, final-layer overwrite) pass; A-cut 0/13. Every cell
of the Executor's table matches the file.

## What the table says

1. **The direction exists in Llama and is content-free, as in Qwen.** At L17 α2 it opens 7 of 13 in-view refusals with
   the evidence present and 3 of 13 with it cut; at α4, 12 and 7; at L22 α4, 13 and 13. The Random vector opens 0 at
   L17 α2, but 2–4 at L25 (a specificity failure at the deep layer, the same pattern as Qwen's L25 α4 primary in §38).
2. **The gold half of the gate has no headroom.** Llama's own cached note holds the gold on 2 of the 13 rows. Those two
   are exactly the two the chosen setting gets right; the 5 other opened answers are wrong, and no setting at any layer
   gets a row right whose note lacks the gold. The bar "gold − cut ≥ 2" is met at its ceiling of 2. A pass that cannot
   have failed by more than it passed is not evidence that the direction reveals reading.
3. **Mechanism reading (descriptive, 13 rows):** Llama's in-view refusals are mostly upstream of the answer decision.
   On 11 of 13 the note the model answers from does not contain the answer, so a switch at the answer step cannot
   recover them. In Qwen the push recovered about half of opened refusals with evidence in view (§38 0.517). The
   comparison needs Qwen's gold-in-note rate on its §38 gate rows, requested from the Executor.

## Ruling on Part A (Llama)

- Stage 2 as designed (1,000 fresh LoCoMo rows, Claim 1 = gold(Steer) − gold(cut)) would test a claim whose ceiling is
  Llama's note quality, not the direction. If run, Claim 1 must be re-framed before launch as the **answer rate**
  (Steer − Random on fresh refusals, with the §38 specificity control) and gold reported descriptively beside the
  note-gold ceiling. That is a mechanism question ("is the content-free switch Qwen-only?") worth about $0.15–0.60.
- Order: the three Part B lens pods first (they address the user's current question, depth and presence on sealed
  folds with the large partner). Llama Part A after them, only if the user wants the switch question answered and the
  claim is re-framed as above in a rev 2 of the proposal.
- Gemma and Phi Part A stay closed at stage 0.

Advisor, 2026-10-26.
