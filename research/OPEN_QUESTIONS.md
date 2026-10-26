# Open scientific questions

Questions the record leaves unresolved. Each one says what would settle it and whether that costs money.
Anyone may propose an addition, but only the Advisor edits this file. Mark an item **answered**, with
a pointer to the EXPERIMENTS.md section, instead of deleting it.

## Toward the pivot (sequential memory and delayed credit)

| ID | Question | Why it is open | What would settle it | Cost |
|---|---|---|---|---|
| Q1 | Can any learner close the lossy-memory headroom of §21 (oracle − supervised writer +0.139)? | §22–22c tried RL with uniform, per-write, counterfactual and time-forward credit; none beat the supervised writer | A pre-registered learner on the §21 simulator that clears learned-knapsack with CI > 0 | free (scripted reader) |
| Q2 | Is the limit information per sample, not the credit rule? | §22 (ii) − (iv) +0.012: the gain needs labels at training time; MLP and linear policies overfit 8 conversations | Same credit rules with more conversations, or a task with more decisions per episode | free first |
| Q3 | Does the §21 headroom exist outside LoCoMo? | The LongMemEval corroboration row of §21 was not run; LoCoMo has 10 conversations | Run the §21 gate on another stream with in-stream questions | free |
| Q4 | Which benchmark has genuine mid-stream dependence (later questions depend on earlier decisions or answers)? | §21 B questions score 1.000 (uninformative); StreamMemBench is recency-dominated (§25) | A literature and data check before any build | free |
| Q5 | Does the leave-one-out set-level utility signal (declared in §29, 16,920 calls) credit turns that only help together? | Declared, costed, never run | §29's declared run | NEEDS SPEND |

## About the existing head and its claims

| ID | Question | What would settle it | Cost |
|---|---|---|---|
| Q6 | How large is `fixed8`'s true training variance, given 3 distinct heads, not 5? | Retrain with 5 distinct half-A/B dealings (fix `memctl/splits.py`); compare per-fold spread | free (stub reader); reader pass NEEDS SPEND |
| Q7 | Do the §26–§32 rankings hold under a stronger or second-family judge? | Rejudge cached answers with a frontier or second-family judge | small spend (API or pod) |
| Q8 | Do the positive results hold on a second benchmark with a non-Qwen-tuned prompt? | One reader pass on another benchmark with the frozen protocol | NEEDS SPEND |
| Q9 | Is context rot (`fixed16` < `fixed8`) real at 7B? | §19c and §23 significant, §19d not. A pooled or third pre-registered test | NEEDS SPEND |
| Q10 | Can a head + recent-window arm transfer to StreamMemBench? | Separate pre-registration, designed on A1/A2 only (§25) | free gate first |
| Q11 | Does fixing the designated-carrier label quirk change the head? | Relabel; retrain; free gate | free |
| Q12 | Do `cross_encoder_tuned` or lambdarank beat LR? | Dropped in §32 (memory kills, 73 min/epoch) | compute only |
| Q13 | OpenJev arm A (memory manager) accuracy, claims A1/A2 | "not tested, not dropped" (§26); about $15 per 1,000 questions | NEEDS SPEND |
| Q14 | Can a 3B reader be lifted to 7B (claim 1 of §23)? | Neither the controller (§23) nor qTTT (§31, closed) did it; LoRA deferred. **Partly answered by §35 (2026-10-14):** about half of the matched in-view gap is the 3B's answering policy (frozen note + "do not reply unknown" +0.176 (+0.109, +0.242); forced choice +0.242), at the cost of a fifth of refusals turning wrong and double the answered-and-wrong share without evidence; the other half stays with the reading. **§36b (2026-10-15):** the 3B cannot sort its own forced answers by checking them against the evidence (AUROC 0.573, NOT SHOWN), so self-verification does not recover the policy half without cost. **§37 (2026-10-16):** the self-verification route closes with a second negative (one shot §36b, three turns §37); extraction is the bottleneck; only a larger checker (Q15) remains. **§38 (2026-10-23):** the policy half is a late, content-free disposition: a layer-29 direction opens it at 0.517 gold with the evidence in view (0.183 without), beating the §35 line's 0.433, with no measurable cost on correct answers; the reading half is still open | the reading half still NEEDS SPEND (LoRA) |
| Q15 | Does a larger checker over a small generator beat the larger model answering alone, and at what cost? | §36b descriptive: the 7B checking the 3B's forced answers reaches AUROC 0.692 (0.600, 0.777) and, at C0's coverage, 0.090 lower risk than the 3B's own refusals; but the 7B also produced the labels, so this is circular | needs an independent label source first (human or a third model), then NEEDS SPEND |
| Q16 | Is the answer already computed under the 3B's refusal, or does lifting the refusal let the model read the evidence? Does the §38 direction transfer to the 7B, to other prompts, or to a fresh confirmation with the setting fixed in advance? | §38 Claim 1 SHOWN only at a setting chosen on development labels, with the add acting at every generated token | **First half answered on development data by §39 (2026-10-23):** one push at the ':' position does as much as the push at every token (ans 26/28 vs 26/28, gold 7 vs 8, random 0/28); the answer is available at the answer position. **§39b (2026-10-23):** the first token carries it on development data (a forced first token on a clean cache reproduces the pushed run 26/28). **§40 (2026-10-23): transfer shown** on LoCoMo with the setting fixed in advance (+0.333 (+0.250, +0.425); 0.358 vs 0.025 cut vs 0.217 for the §35 line), with larger costs than on LongMemEval (−0.200 on correct rows; unanswerable questions answered 0.32 → 0.73) and a qualification: on LoCoMo the every-token push beats the one-position push on content by +0.075 (+0.017, +0.142) with the same first token. Still open: the 7B, other prompt families | answered on both halves for the 3B; transfer to other models or prompts NEEDS SPEND |
| Q17 | Can steering be made selective? §41 (2026-10-23): the projection of the unsteered ':' state on the direction ranks which opened refusals come out right (AUROC 0.803 (0.724, 0.875)), but the rule fitted on 67 development rows was useless, and unconditional steering loses to the §35 line once wrong answers are penalised (−1 vs +4 of 120) | A pre-registered threshold on s_proj (fixed from §40's rows) scored on a fresh set against the §35 line, unconditional steering and abstention, with the damage and adversarial rates (draft §42) | one small pod, about $0.15–0.60 |

## Record-keeping gaps (not science, but they block clean citation)

- `docs/research/THESIS_RESULTS_OUTLINE.md` is dated 2026-10-07. It has no §29–§32 rows. Its headline
  must be reconciled with F3 (head ≈ LR) and must carry the 3-head and ±0.007 caveats.
- The cost table ends at §27–§28 ($12.58); §29 and §32 rows are missing.
- Head A fold 0 and head B fold 4 are the same checkpoint (sha256 prefix `9f0be108`), and head B folds 0 = 1,
  2 = 3: five distinct §19 networks across both heads. Found by the 2026-10-16 audit
  (`research/audits/2026-10-16-thesis-direction-audit.md` §1.8); record notes under §19a and §19d in
  EXPERIMENTS.md at 9289b52 (Executor, hashes re-verified). Resolved.
- No interpretability or abstention literature note exists for §33–§37 (`docs/literature/` covers memory
  controllers only).
- The §32 spot-check file times (Oct 9, about 21:xx local) and the text's date (2026-10-10) differ,
  probably by timezone. Not resolved.
- EXPERIMENTS.md still names `runs/_aborted*` and `runs/exp22c_peek_quarantine/`, which were deleted
  on 2026-10-10 (see `runs/_deleted_2026-10-10.txt`).
