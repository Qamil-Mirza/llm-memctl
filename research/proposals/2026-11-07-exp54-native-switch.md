# §54 proposal: looking for the native switch behind false "unknown" (Qwen 2.5 3B)

**Status: rev 0 APPROVED for stages 0–4 (user, 2026-11-07: "approved, go ahead with stages 0–4"; D-78).** Stages
0–4 cost $0 and use no held-out data. Stage 5 needs a separate approval.

**Rev 0a (Advisor, at the stage-0 freeze review, before any model call;
`research/reviews/2026-11-07-exp54-freeze-review.md`):**
- **Criterion 1** now compares the within-twin difference with the same quantity on *stable* twins: same base and
  hardness, both answer gold under the same forced note. The bar is a ratio of at least 2, with at least 20 stable
  pairs required, instead of the spread across C rows from different bases. That spread is dominated by content, so
  every component would have failed mechanically.
- **Criterion 4** is not evaluable, and K cannot be called the switch, if patching all components itself closes less
  than half the gap.
- **The stage-2 pool** becomes every component if the measured one-pair timing fits the 24 h budget.

---

## 1. The question in one paragraph

§51 and §53 showed that we can *repair* false "unknown" answers with weight edits. They also showed *where* a push
works. They did not find the part of the **unedited** model that decides, on a given question, to say "unknown" when
the answer is in view. This proposal looks for that part directly.

The design compares near-identical twin questions. The memory differs only in its timestamp strings, and the model
answers one twin correctly and says "unknown" on the other. We copy internal values from one twin into the other and
ask which components carry the decision. A component counts as the switch only if it passes five pre-set tests,
listed in §4.

The most likely result, given what we already know, is a **competition**: "mover" heads that carry the answer lose to
a near-constant "unknown" default. That would still be a mechanism, and it would explain why a plain logit bias did
as well as both edits.

## 2. Why now, and what we already know

What earlier work tells us, and where it points:

| Source | Finding | What it implies for the switch |
|---|---|---|
| F16 (§49) | 152 of 165 false abstentions name the gold city in the model's own note, then answer "unknown" | The answer is available. The failure is in committing to it |
| F15 (§47) | The gold value is carried at the evidence tokens through layer 30. Patching the answer position moves the readout only from layer 33 | The decision is late, at the answer position |
| §50 dev, descriptive | On 60 dev flip pairs (rep a vs rep b, timestamps differ only), copying the **whole residual** at the answer position from the answering twin releases gold on ≥ 50% of pairs from **layer 26** (0.58) | By layer 26 the decision-relevant difference is in the residual at P_ans. The components that put it there are not known |
| F17 (§50) | On refusals the gold-vs-rival-city margin is lower than on answers. Within 47 flip pairs the difference is −2.50 | Weaker commitment is a correlate, not shown to be a cause |
| §51 editor notes (exploratory, `runs/exp51/editor_run/workspace/notes/hypothesis.md`) | "Prior" heads L32h2 and L33h10 push toward "unknown" about equally on every row (cosine 0.99 to their mean). "Mover" heads L32h13, L32h3, L33h3, L33h15 and L33h11 write the gold city, about 15–20% weaker on false abstentions. The L35 MLP is the largest city writer and compensates for upstream changes | A concrete, untested account: a competition between movers and a constant prior, with the last MLP absorbing changes |
| F19 (§53) C1/C2 | Zeroing the prior heads releases 0.751 of false abstentions; doubling them makes 0.381 of correct answers abstain | The prior heads are a causal lever. But their output barely varies between rows, so they set the default rather than flip the decision |
| §53 B1-ref | A constant logit bias against "unknown" matches both edits on utility | Consistent with the failure being a threshold on a near-constant default |

**What is missing.** No test has asked which components *differ naturally* between an answering twin and an
abstaining twin, and whether copying exactly that natural difference flips the decision in both directions. That is
the test here. It differs from the editor's failed "double the movers" edit in two ways:
- it copies the twin's own values (natural size), rather than amplifying;
- it measures downstream compensation explicitly, rather than being defeated by it.

## 3. Definitions

- **P_ans:** the final ":" token of the "Answer:" marker, as in §50 and §51. Its output predicts the first answer
  token.
- **Twin pair:** two rows of the same base and hardness, rep a and rep b, whose memory blocks differ only in
  timestamp strings (§50's construction).
  - **Primary pairs (controlled note):** both twins receive the *same* note, the answering twin's own note copied
    token for token. Their first answer token differs: one is the gold (or the gold-bearing first word), the other is
    "unknown". In these pairs only the timestamp tokens differ in the whole prefix.
  - **Secondary pairs (own notes):** each twin keeps its own note, as in §50. These are reported beside the primary
    pairs, because notes can differ.
- **A** = the answering twin, **F** = the abstaining twin.
- **Decision score D:** at P_ans, D = logit("unknown" first token) − logit(t*), where t* is A's first answer token.
  - Positive D means "unknown" wins.
  - Using A's own first token avoids §50's DE problem, where the answer begins with a name rather than the city.
  - D_A < 0 < D_F by construction of the pair.
- **Gap closed by a patch:** (D_F − D_F,patched) / (D_F − D_A), for a patch from A into F. The reverse direction is
  defined symmetrically. 1 means the patch fully reproduces the other twin's decision; 0 means no effect.
- **Component:** one attention head's output at P_ans (its slice of the `o_proj` input), or one MLP's output at
  P_ans. Layers 20–35. That is 16 layers × 16 heads + 16 MLPs = 272 components.
- **Patch:** replace a component's output in the recipient's forward pass with the donor twin's value at the same
  position. Downstream components recompute unless stated otherwise.
- **Direct and indirect effect.** The *direct* effect of a patch is its effect with every downstream component held
  at the recipient's values (only the patched output reaches the logits). The *indirect* effect is the remainder.
  Splitting the two measures how much compensation or amplification happens downstream.

## 4. What counts as "the switch": five pre-set criteria

A set K of at most 5 components is called the **switch candidate** if it passes all five tests on the stage-3 dev
pairs:

| # | Criterion | Test | Bar |
|---|---|---|---|
| 1 | **It differs within twins** | Size of the A−F difference in each component of K, against the spread of that component across C rows of the same cell | Median standardised difference ≥ 0.5 in every component of K. Reported for every component |
| 2 | **Sufficient forward** | Patch K from A into F | Median gap closed ≥ 0.5, **and** gold released in greedy generation on ≥ 50% of pairs |
| 3 | **Sufficient in reverse** | Patch K from F into A | Median gap closed ≥ 0.5, and "unknown" produced on ≥ 30% of pairs. The bar is lower because correct answers are expected to be more robust |
| 4 | **Necessary** | Start from F and patch *every* component in layers 20–35 from A, except K, which stays at F's values | Gap closed drops below 0.5 of the all-component patch's gap closed |
| 5 | **Specific** | Same patch with 20 random component sets of the same size and type mix (same number of heads and MLPs, drawn from layers 20–35) | K ranks 1 of 21 on forward gap closed, as in §53 |

How these criteria relate to the earlier results:
- **Criterion 1 separates a switch from a default.** If the prior heads are as constant as the editor's reads say,
  they fail criterion 1. They would then be reported as the default the switch acts against, not as the switch.
- **Criteria 2 and 3 copy natural values.** Neither amplifies or zeroes anything. This is the key difference from
  the §51 edits and from the output-push baselines. A logit bias has no natural counterpart in the twin pair, so it
  cannot pass criterion 1 by construction.
- **Criterion 4 asks whether K is needed** for the difference between twins, given everything else in the late
  stack.
- **Criterion 5 controls for patching "any 5 things"** in late layers.

**Reported for every candidate, with no bar:**
- the direct vs indirect split;
- the effect of transplanting the *mean* A−F difference of K (the same vector for all pairs) instead of each pair's
  own difference. If the mean transplant works as well, the switch is a shift along one fixed direction;
- the effect on UNA-rel rows of transplanting that mean difference. This is descriptive: a real commit signal may
  well create answers on unanswerable rows. That would be a property of the switch, not a failure.

## 5. Stages

### Stage 0: freeze this design before any measurement ($0)
- The user approves this proposal, or a revision of it.
- The Advisor records the approval as a DECISIONS entry.
- Before any model call, the Executor commits:
  - the criteria in §4;
  - the stopping rules in §6;
  - the candidate lists in stage 3;
  - the seeds in stage 1.

### Stage 1: build the twin pairs ($0, laptop)
- **Data.** New cases from the §51 F2 generator (same templates, note format and instruction as §51 and §53, so
  results connect to the edits) on a **new seed, 55**.
  - Seeds 50–54 are used: §50, E/S, H, T-city and N. None of their held splits is touched.
  - Size: about 1,950 rows, as N had, aiming for at least 60 primary pairs.
- **Procedure.**
  - Run the unedited model on rep a and rep b of every answerable base × hardness, with its own notes.
  - For each pair where the decisions differ, force A's note into F's prompt and rerun only the answer step.
  - Keep the pair as primary if F still says "unknown". If F now answers, it becomes a "note-driven" pair, reported
    separately.
- **Feasibility gate.**
  - Pass: at least 40 primary pairs from at least 20 bases, covering at least 2 of the 3 tasks.
  - Fail: report the counts and stop (§6).
- **Split.** Pairs are divided by base, 50/50 at random: dev-A for screening (stage 2) and dev-B for verification
  (stage 3). The candidate set K is chosen on dev-A only and verified on dev-B. This prevents the same pairs choosing
  and testing K.

### Stage 2: screen all 272 components (dev-A, $0)
- **Attribution patching (AtP\*).** For each pair and each component, a first-order estimate of the gap a patch
  would close, from one forward and one backward pass per pair and direction (Kramár et al. 2024).
- **Check the estimate.**
  - AtP is known to misjudge components whose attention saturates. So the top 30 by AtP estimate are also patched
    for real, along with a random 30 as a calibration check.
  - The correlation between estimated and real gap closed is reported.
- **Always included,** whatever their rank, so the editor's account is tested directly:
  - the prior heads (L32h2, L33h10);
  - the five mover heads (L32h13, L32h3, L33h3, L33h15, L33h11);
  - the L33, L34 and L35 MLPs;
  - L35h12 and L35h13, which the §51 gate reads.
- **Choosing K.** The smallest set of at most 5 components, built greedily from the real-patch results, whose joint
  forward patch closes the most gap on dev-A. The choice is frozen and committed before dev-B is read.

### Stage 3: verify K on dev-B ($0)
- Run criteria 1–5 on dev-B pairs for K.
- Run the same five criteria on two **named comparison sets**, reported as descriptive:
  - **"movers"**, the editor's five mover heads;
  - **"movers + L35 MLP"**, which tests whether compensation hides the movers' effect.
- **Generation check** (criteria 2 and 3): greedy, controlled prefix, with the patch applied at P_ans only.
  - Patching only at P_ans means the prefix KV cache from the recipient's own run can be reused. Each patched pass is
    then a single-token forward, which makes the full grid cheap on the laptop.

### Stage 4: if K passes, one step upstream (dev-A and dev-B, $0, descriptive)
For each attention head in K, separate *where it looks* from *what it carries*:
- **Pattern-only patch:** copy A's attention pattern from P_ans into F, keeping F's values.
- **Value-only patch:** keep F's pattern, copy A's values.
- **Source positions:** split the head's output by source position: evidence line, note tokens, timestamp tokens,
  instruction.

Since only timestamps differ in a primary pair, this asks how a timestamp change reaches the switch. For example,
through slightly different attention to the evidence line. For an MLP in K, report which of its inputs (the residual
written by earlier heads in K or outside it) carry the difference.

### Stage 5: confirmation on sealed cases (separate approval)
- **Data.** A fresh seed, 56, sealed until the K, criteria and analysis code are committed.
- **Test.** Criteria 1–5 for the frozen K, with the same bars and rank tests. No family-wise claim beyond what is
  stated; the multiplicity rule is as in §53.
- **Cost.** Run it on the laptop if one-pair timing shows it fits in about 24 h. Otherwise one small pod with a quote
  and a hard stop below the remaining budget (about $2.25 under the $36 cap, D-73).
  - The user approves the spend in the Executor's pane before any pod is created.
  - Stage 5 is proposed only if stage 3 passes.

## 6. Stopping rules (decided now)

| Where | Result | What happens |
|---|---|---|
| Stage 1 | Fewer than 40 primary pairs or fewer than 20 bases | Report the counts and the share of note-driven pairs. Stop. No rescue (D-45) |
| Stage 2 | No set of ≤ 5 components closes ≥ 0.5 of the forward gap on dev-A | Report: "at head/MLP resolution in layers 20–35 the decision is distributed". Report the top components' shares. Stop |
| Stage 3 | K fails any of criteria 2–5 on dev-B | Report which criterion failed and the named comparison sets. Stop. K is not re-chosen on dev-B |
| Stage 3 | K passes 2, 3 and 5 but fails 1 (it moves the decision but barely differs between twins) | Report K as a lever, not a switch, the same status as the prior heads. Stop |
| Stage 3 | K passes all five | Run stage 4, then propose stage 5 to the user |
| Stage 5 | Fails | Report under D-45. The dev result stays exploratory |

## 7. What each outcome would mean

- **K passes, and K is mostly mover heads.** The editor's untested account is right in kind. False abstention is a
  competition: the answer-carrying heads are slightly weaker on the failing twin, and a near-constant "unknown"
  default wins.
  - This explains why removing the default (C1, B1) works as well as any edit.
  - The thesis can then say *why* the failure happens, at the component level.
- **K passes but needs the L35 MLP.** The last MLP is part of the switch, not just a compensator. Stage 4 says what
  it reads.
- **K passes and the mean-difference transplant works as well as pair-specific patches.** The switch is a single
  direction, a scalar commit signal. That is the closest analogue to Ferrando et al. (2025), who found internal
  "known vs unknown entity" features that steer refusals.
- **The decision is distributed (stage 2 stop).** Not a dead end. It says no small set of late components flips
  false abstention, so mechanistic claims about this failure must be made at the level of directions or features, not
  heads. That is also why the head-level edits needed a fitted read direction to be selective.
- **Few primary pairs (stage 1 stop).** Most twin differences come from the notes, so the decision depends more on
  what the model wrote than on a late switch. That is useful for the thesis's account of "note, then refuse".

In every case except the stage 1 stop, the result fills the gap stated in the technical report's claims audit: no
native computation has yet been shown to be necessary, sufficient and specific for false abstention.

## 8. Threats and how the design handles them

| Threat | Handling |
|---|---|
| **Twin pairs are near-threshold rows.** F17: only 0.23–0.35 of refusals are near ties | Claims are limited to near-threshold false abstentions. The share of all false abstentions that form twin pairs is reported. A switch found on twins may not explain robust refusals; the mean-transplant check on non-twin false abstentions (descriptive) gives a first look |
| **The same pairs choose and test K** | Split by base: dev-A chooses, dev-B tests (stage 1) |
| **Compensation hides real effects** (the editor saw the L35 MLP give back +4 of −6) | The direct/indirect split is reported for every candidate. "Movers + L35 MLP" is a named set. Criterion 4 patches everything else jointly |
| **Patching out-of-distribution values** | Only natural values from the twin are copied, never zeroed or amplified (Zhang & Nanda 2024; Heimersheim & Nanda 2024) |
| **The metric choice changes conclusions** (Zhang & Nanda 2024) | D is a logit difference with a pre-set target token. The generation endpoint is reported beside it |
| **AtP misses components** | Real-patch check on the top 30 plus a random 30; named candidates are always included |
| **Timestamp tokens are an odd cause** | Expected. They are nuisance variation that happens to push a near-threshold row across. The switch is whatever carries that push to the decision, and stage 4 traces how |
| **Using a fitted probe as the instrument** (the weakness of §51's gate) | No probe is fitted. Every quantity is a natural activation or a patch of one |
| **Held-out leakage** | Seeds 55 (dev) and 56 (sealed, stage 5) are new. No §50–§53 held split is read |
| **Laptop time** (the §53 laptop B0 run swapped at 15–19 GB of 23 GB) | Time one real pair first (memory rule "measure before ETA"). Reuse the prefix KV cache so patches cost one token. If the full grid exceeds about 24 h, restrict the screen to layers 26–35 (§50's ℓ = 26) before running, and record the restriction |

## 9. Cost and roles

- **Stages 0–4:** $0, laptop, development data only. Built and run by the Executor; reviewed by the Advisor at each
  stage gate (pair counts, frozen K, stage-3 results).
- **Stage 5:** laptop if it fits, otherwise one small pod quoted in advance with a hard stop. User approval required
  in the Executor's pane. About $2.25 remains under the $36 cap.
- **No change to STATE or DECISIONS** until results are reviewed and the user says "Go".

## 10. Rough schedule (to be replaced by measured times)

| Step | Estimate |
|---|---|
| Stage 0 freeze | Same day as approval |
| Stage 1 generation of seed-55 rows and notes | Unknown until timed; §53's laptop N-dev B0 took 2 h 25 min for its rows. Expect a similar order |
| Stage 2 screen | One forward and backward pass per pair and direction, plus about 60 real patches per pair, each a single token. Expected hours, not days, but timed first |
| Stage 3 verification | Similar to stage 2 |
| Stage 4 | Hours |

## 11. References

- Zhang, F. and Nanda, N. (2024). Towards best practices of activation patching in language models: metrics and
  methods. ICLR.
- Heimersheim, S. and Nanda, N. (2024). How to use and interpret activation patching. arXiv:2404.15255.
- Kramár, J., Lieberum, T., Shah, R. and Nanda, N. (2024). AtP\*: an efficient and scalable method for localizing
  LLM behaviour to components. arXiv:2403.00745.
- Wang, K. et al. (2023). Interpretability in the wild: a circuit for indirect object identification in GPT-2 small.
  ICLR.
- McDougall, C. et al. (2024). Copy suppression: comprehensively understanding a motif in language model attention
  heads. BlackboxNLP.
- Rushing, C. and Nanda, N. (2024). Explorations of self-repair in language models. ICML.
- McGrath, T. et al. (2023). The hydra effect: emergent self-repair in language model computations. arXiv:2307.15771.
- Ferrando, J., Obeso, O., Rajamanoharan, S. and Nanda, N. (2025). Do I know this entity? Knowledge awareness and
  hallucinations in language models. ICLR.

## 12. Decision requested

Approve stages 0–4 ($0, development data only), approve with changes, or decline. Stage 5 would come back to you as
a separate decision with a measured time and, if a pod is needed, a price and a hard stop.
