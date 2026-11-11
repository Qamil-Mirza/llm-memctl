# §57 proposal: which neurons behave differently on false "unknown", and does resetting them fix it? (Qwen 2.5 3B)

**Status: proposed, rev 0 (Advisor, 2026-11-10). Not built, not run, nothing spent.** All stages 0–4 are laptop-only,
$0, and use development data only. A sealed confirmation (stage 5) needs a separate approval. The section number and
D-85 are provisional (D-84, the §56 close, waits for the user's "Go").

**Rev 0a (Advisor, at the stage-0 gate, 2026-11-10 ~20:30 PT; before stage 2, before K_P/K_N exist, before any
dev-B row is touched).** Stage 0 passed (3501a72): memory fits at all 36 layers; validity P Spearman 0.998, N 0.950;
sets 10/10 sign; equality 440/440. One change, to R4 only, and it makes R4 harder for the sets:
- **Why.** Each set is chosen on dev-A FA rows, so its dev-A release is in-sample and will tend to exceed its dev-B
  release. A b calibrated to the dev-A release then overshoots on dev-B, which raises Uns(B1) and biases R4 toward
  BETTER THAN THE DIAL.
- **R4, primary from now on: matched release on dev-B.** Run B1 on dev-B FA and TA rows at the frozen grid
  b ∈ {0, 1, 2, 3, 4, 5, 6, 8, 10, 12}. Linearly interpolate Uns(B1) at the set's own dev-B Rel between the two
  bracketing grid values. Statistic: Uns(set) − Uns_B1@Rel(set). 95% interval: base-clustered bootstrap (2,000 draws),
  recomputing both Rel(set) and the B1 curve in each draw. Verdicts as in §6. If Rel(set) falls outside the grid's
  release range, R4 is NOT READ for that set.
- The original R4 (b calibrated on dev-A, §5 stage 3 step 3) is still run and reported beside it, labelled
  "dev-A-calibrated".
- **Disclosure.** This was written after the Executor reported, incidentally, that in the stage-0 equality test (all
  dev-A; stand-in sets from 5 dev-A rows) the stand-in N set changed 9 of 10 FA outputs and random sets 2 of 400.
  No stage-2 or dev-B quantity existed. The change can only make a set's R4 harder to pass.
- ā^C_N was built in stage 0 rather than stage 2 because the validity gate needed it; it is a fixed function of dev-A
  C rows and is frozen in 3501a72. Accepted.
- **Order.** Stage 2 and the dev-B UNA generation may run back to back (the generation does not depend on the sets).
  The Executor stops after both, before stage 3, for the stage-2 review.

**Authority.** The user, Advisor pane, 2026-11-10, after the plan was described to them (find neurons that differ
between false-"unknown" and correct rows at the note and the answer; rank by total effect with gradient attribution;
reset the top set against random sets and the dial; describe what survives): "Go for, it's free right". This covers
stages 0–4 at $0 on development data. It does not cover stage 5, any pod, or any held-out read.

## 1. The question

§56 asked which neurons push toward "unknown" and found the push spread over ~15,000 neurons, mostly not specific to
false abstention (C/FA 0.92 on its top 50). It measured only the direct path at the answer position. This study asks
the question the user cares about, with Goodfire-style tools:

1. **Differential:** which neurons fire differently on false-"unknown" rows (FA) than on correctly answered rows (C)?
2. **Total effect:** of those, which ones would lower the "unknown" margin D if reset to their C-row values, counting
   indirect paths (through attention and later layers), at the answer ':' (P_ans) and over the model's own note?
3. **Causal and selective:** does resetting the top set turn false "unknown"s into gold answers more than random sets
   do, and more selectively than a plain logit bias (the dial)?

## 2. What we already know

| Source | Fact | What it implies here |
|---|---|---|
| §56 / F23 (draft) | Direct push toward "unknown" at P_ans: K50 = 14,966 neurons; top 50 not FA-specific (C/FA 0.92); net push on the direct path through attention (FA +14.5 vs C +3.8), MLPs net −7.7 | Rank by a FA−C difference and by total effect, not by direct push |
| §50 dev (descriptive) | On 60 twin flips, copying the whole residual at P_ans from the answering twin releases gold on ≥ 50% from layer 26 | The decision-relevant difference is present at P_ans by L26 |
| §51 / F18 | Attention pathway L32h2 + L33h10 → L35; "mover" heads write the gold ~15–20% weaker on FA; L35 MLP compensates | Neurons may act by feeding the movers; indirect effects matter |
| §53 / F19 | Logit bias B1 (−8) ties both §51 weight edits | The bar for "selective" |
| §54 / F20 | 67% of twin flips are note-driven; ~0.89 of FA notes hold the gold | The note is read, then overridden. Test the note positions as well as P_ans |
| Outside | Attribution patching (Nanda 2023; Syed et al. 2023) and AtP* (Kramár et al. 2024): one backward pass estimates every unit's patching effect; known failures are attention saturation and cancellation. Goodfire use a gradient method (RelP) for the same purpose | The linear estimate must be checked against real patches before it is trusted (stage 0 gate, stage 3 descriptive) |

## 3. Definitions

Unchanged from §56 rev 0a unless stated: Qwen 2.5 3B Instruct, fp32, laptop CPU, prompt P0 (7b5fc30), greedy, the
row's own P0 note forced, then the marker. Neurons a_i = silu(gate_i)·up_i (396,288). D = logit(" unknown") −
logit(t*), t* = the best city token of the row's pool (rev 0a). Classes FA, C, TA, UA as in §56.

- **Regions.** P = the P_ans position alone. N = the note tokens and the "Answer" marker before P_ans (the model's
  own note, forced).
- **Reset values.** For region P: the neuron's mean activation over dev-A C rows at P_ans (`c_mean_act.npy`, already
  committed at 7d78ee0). For region N: the neuron's mean over all note tokens of dev-A C rows (new, stage 2).
- **Predicted effect (attribution patching).** For row r, neuron i, region R:
  Δ̂_i(r) = Σ_{t∈R} (ā^C_i − a_i(r,t)) · ∂D/∂a_i(r,t), from one forward and one backward pass with no intervention.
  Negative Δ̂ means resetting the neuron lowers D (toward answering).
- **Score.** S_i^R = mean over dev-A FA rows of −Δ̂_i. Specificity: the same mean over TA rows (would the reset also
  cut true abstentions?) and C rows (does it move rows that already answer?).

## 4. Data

The §56 rows, unchanged (freeze 55d3d87b at 3690b0f): W split, seed 55. Dev-A (even bases: FA 140, C 305, TA 289,
UA 9) chooses everything; dev-B (odd bases: FA 148, C 298, plus ~300 UNA rows to generate under P0) tests. No held-out
split is read. Seeds 56 and 57 stay unused.

## 5. Stages

### Stage 0: freeze and feasibility ($0)

1. Commit the code for Δ̂ (both regions) and the stage-3 patching code, with a dummy-model test that Δ̂ equals the
   exact patching effect for a linear toy model.
2. **Memory and time.** Time one forward + backward on one long dev-A row with gradients at all 36 layers. If peak
   RSS exceeds 21 GB (23 GB machine), restrict gradients to layers 16–35 (detach below L16) and record it; if that
   still does not fit, stop and report.
3. **Validity gate.** On 5 dev-A FA rows, compare Δ̂ with the real patching effect on D (one forward with the reset):
   - 40 single neurons per region (the 20 top-|Δ̂| on that row and 20 random), so 400 pairs in all;
   - the top-50 set by Δ̂ on each row, per region (10 set patches).

   **Pass** if the Spearman correlation of Δ̂ with the real effect over the single-neuron pairs is ≥ 0.6 in each
   region, and the top-50 set patch has the predicted sign on ≥ 8 of 10. If one region fails, that region becomes
   descriptive only. If both fail, stop and report under D-45.
4. **Equality test** for the batched-arm engine (scripts/exp56_arms.py) on 10 dev-A rows × the stage-3 arms, as in
   §56 rev 0a (≥ 99% identical greedy outputs, |ΔD| ≤ 1e-3). If it fails, stage 3 uses batch 1 and the random sets
   drop to 10, recorded before any stage-3 read.

### Stage 1: Goodfire mapping on stored data (minutes, descriptive only)

From the §56 stage-1 reads (activations and DLA at P_ans, dev-A), with no new model calls:
- **Strong-firing share:** the fraction of the positive "unknown" push (FA rows) carried by neurons whose |a_i| on that
  row is in the row's top 3% (Goodfire's p97 tail), and the same with a per-neuron threshold (above the neuron's own
  97th percentile over all dev-A rows).
- **Differential firing:** per neuron, the FA − C difference in mean activation at P_ans, in pooled-SD units; the
  number of neurons beyond |d| ≥ 0.5 and ≥ 1.0, and their layer histogram.
- Base-matched version: the same within base × hardness cells holding both an FA and a C row.

### Stage 2: rank by total effect (dev-A only)

1. One forward + backward per dev-A row (FA, C, TA; UA descriptive), storing Δ̂_i for regions P and N.
2. **R1 concentration** (frozen below), per region, on S^R.
3. **Candidate sets.** K_P = the top 50 neurons by S^P; K_N = the top 50 by S^N. Frozen by commit with their reset
   values before any dev-B row is touched.
4. Descriptive: specificity ratios (TA/FA and C/FA on each set); overlap of K_P with §56's K and with the top
   differential neurons of stage 1; layer histograms; write-vector cosines with the §51 rank-1 direction and §38 v_29;
   for K_N, which note tokens carry the effect (gold-city mention, hedging words, others).

### Stage 3: causal test and selectivity (dev-B)

1. **Arms.** Reset K_P at P_ans; reset K_N over the note positions; each to its dev-A C-row values. The note text is
   fixed (forced), so the reset changes how the note is read, not what it says. Generate the answer greedily and grade
   with the §51 grader (head + tail).
2. **Random controls.** 20 random sets per region, matched to the set's per-layer counts, on dev-B FA rows only.
3. **Dial B1.** A bias −b on the " unknown" first token at P_ans, with b calibrated on dev-A so that its FA release
   equals the better set's dev-A release (dev-A runs of K_P and K_N on FA rows only, for calibration). b is frozen
   before any dev-B read.
4. **Measures on dev-B:** Rel (FA rows now gold), Uns (TA rows now answered), Dam (C rows now wrong), U(λ = 1) as in
   §51. Sets and B1 run on FA, C and TA rows; random sets on FA rows only.
5. Descriptive: Δ̂-predicted vs real ΔD for K_P and K_N on dev-B FA rows.

### Stage 4: what the survivors are (only for a set with R3 SHOWN; descriptive)

For the top 10 neurons of that set, the §56 stage-3 procedure: upstream contributions (heads, earlier MLPs) to the
neuron's input at the relevant positions, and an Advisor-written one-line description from the 20 highest- and
20 lowest-activating dev rows and 200 LongMemEval dev prompts, followed by the 10-pair synthesis check (pass ≥ 8/10).
Descriptions are written before the Executor runs the synthesis inputs.

### Stage 5: sealed confirmation (separate approval)

Re-read R3 and R4 on a new seed-57 split with sets, reset values and b frozen.

## 6. Readings, fixed now

| Reading | Statistic | Verdict |
|---|---|---|
| **R1 concentration** (stage 2, per region) | K50 on the positive part of S^R: the fewest neurons whose summed S reaches half the positive total. Base-clustered bootstrap, 2,000 draws, seed 56 for resampling only | **CONCENTRATED** ≤ 50; **DISTRIBUTED** > 500; **PARTLY** otherwise. Descriptive for the plan: stage 3 runs whatever R1 says |
| **R2 specificity** (stage 2, per set) | Σ_K S on TA rows ÷ Σ_K S on FA rows | **FA-SPECIFIC** if the 95% interval is wholly ≤ 0.5; **SHARED** if wholly ≥ 0.8; **MIXED** otherwise |
| **R3 causal** (stage 3, dev-B, per set) | Rel of the set vs its 20 random sets | **SHOWN** if the set ranks 1 of 21 and Rel ≥ 0.20; **NOT SHOWN** otherwise |
| **R4 selectivity vs the dial** (stage 3, dev-B, per set with R3 SHOWN) | Uns(set) − Uns(B1) at matched dev-A release; 95% interval clustered by base | **BETTER THAN THE DIAL** if wholly < 0; **WORSE** if wholly > 0; **NO DIFFERENCE SHOWN** otherwise. Rel − Rel(B1) and Dam reported beside it |

Two sets are tested, so every claim names its set and region. No set is re-chosen on dev-B.

## 7. Stopping rules (decided now)

| Where | Result | What happens |
|---|---|---|
| Stage 0 | Backward does not fit in memory at L16–35, or both regions fail the validity gate | Report and stop (D-45) |
| Stage 0 | One region fails the validity gate | That region is descriptive; stage 3 tests the other set only |
| Stage 2 | Any R1 | No stop. §56 stopped on concentration and never reached the causal test; the causal test is the question here |
| Stage 3 | R3 NOT SHOWN for both sets | Report "resetting the top total-effect neurons, at the answer or over the note, does not release false abstentions beyond random sets." Stop; the next unit is attention heads (new proposal) |
| Stage 3 | R3 SHOWN for a set | Read R4; run stage 4 for that set; then propose stage 5 to the user |

## 8. What each outcome would mean for the mechanism

- **A set passes R3 and is FA-specific, better than the dial.** We have found neurons whose state differs on false
  abstentions and whose reset fixes them without raising made-up answers: the first selective lever in the
  programme, and a concrete mechanism to describe in stage 4.
- **A set passes R3 but is shared / no better than the dial.** We have found where the dial physically lives in the
  neurons. The cause of false (vs true) abstention is upstream of them, and stage 4 says what feeds them.
- **K_N passes, K_P does not.** The decision is made while the note is read, not at the answer: consistent with F20.
- **Both NOT SHOWN.** Neither the answer step nor the reading of the note has a small neuron-level switch. Together
  with §56 and the attention-heavy direct path, attention heads become the next unit, and the same recipe moves to the
  self-doubt project only at the head level.

## 9. Threats and how the design handles them

| Threat | Handling |
|---|---|
| **The linear estimate (attribution patching) is wrong** for large resets, attention saturation, or cancelling neurons (Kramár et al. 2024) | Stage 0 validity gate against real patches; stage 3 reports predicted vs real ΔD; R3 itself uses real patches |
| **C-row means are a crude baseline** (different prompts, different notes) | Natural class means, not zeros. Stage 1's base-matched contrast is reported beside it. Zero-reset is a sensitivity check on K_P only |
| **The note is forced**, so a neuron that changes what the model *writes* is not tested | Stated in every reading. The claim is about reading the note and answering, not writing the note |
| **Memory** (forward alone peaks at 17.7 GB of 23 GB) | Stage 0 measures; fallback to gradients at L16–35, recorded |
| **Two sets, two tests** | Both reported, each named; no pooled claim |
| **Same rows choose and test** | Dev-A chooses sets, reset values and b; dev-B tests |
| **Synthetic templates only** | Any claim is scoped to W until stage 5 or a natural-data replication |

## 10. Cost, roles, schedule

- $0, laptop, development data only. The Executor builds and runs; the Advisor reviews at the stage-0, stage-2 and
  stage-3 gates. No STATE or DECISIONS change until results are reviewed and the user says "Go".
- Rough times, to be replaced by stage-0 measurements:

  | Step | Estimate |
  |---|---|
  | Stage 0 | ~1 h (one backward timing, 5-row validity gate, 10-row equality test) |
  | Stage 1 | Minutes (stored data) |
  | Stage 2 | ~730 rows × (forward + backward), likely 4–6 h |
  | Dev-B UNA generation | ~1.5 h |
  | Stage 3 | FA rows: 2 sets + 40 random + B1 + baseline; C and TA rows: 4 arms. ~9,000 row-arms: ~3 h batched, ~20 h batch-1 |
  | Stage 4 | Hours |

## 11. References

- Nanda, N. (2023). Attribution patching: activation patching at industrial scale.
- Syed, A., Rager, C. and Conmy, A. (2023). Attribution patching outperforms automated circuit discovery. arXiv:2310.10348.
- Kramár, J., Lieberum, T., Shah, R. and Nanda, N. (2024). AtP*: an efficient and scalable method for localizing LLM
  behaviour to components. arXiv:2403.00745.
- Tigges, C., Kowal, M., McGrath, T., Lewis, O. and Cammarata, N. (2026). Neurons as an interpretable basis. Goodfire.
- Cammarata, N. (2026). Neural networks are already clean. Goodfire.
- §56 proposal and record (this repo) for the shared definitions.
