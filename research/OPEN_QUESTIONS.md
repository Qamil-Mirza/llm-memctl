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
| Q14 | Can a 3B reader be lifted to 7B (claim 1 of §23)? | Neither the controller (§23) nor qTTT (§31, closed) did it; LoRA deferred | NEEDS SPEND |

## Record-keeping gaps (not science, but they block clean citation)

- `docs/research/THESIS_RESULTS_OUTLINE.md` is dated 2026-10-07. It has no §29–§32 rows. Its headline
  must be reconciled with F3 (head ≈ LR) and must carry the 3-head and ±0.007 caveats.
- The cost table ends at §27–§28 ($12.58); §29 and §32 rows are missing.
- The §32 spot-check file times (Oct 9, about 21:xx local) and the text's date (2026-10-10) differ,
  probably by timezone. Not resolved.
- EXPERIMENTS.md still names `runs/_aborted*` and `runs/exp22c_peek_quarantine/`, which were deleted
  on 2026-10-10 (see `runs/_deleted_2026-10-10.txt`).
