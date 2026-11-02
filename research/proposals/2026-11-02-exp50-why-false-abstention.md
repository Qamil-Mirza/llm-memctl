# §50 proposal, rev 1: what separates a false "unknown" after an answer-bearing note from an answer? Elicited sufficiency and city-margin profiles, with conditional subspace-restricted interventions (2026-11-02, pre-registration draft)

## Origin, authority and revision history

**The user's orders, Advisor pane, 2026-11-02:**
- "So what's your final recommendation? What do you think is the next best step towards novelty?"
- After the recommendation (test misjudged sufficiency against sub-threshold commitment on the city tasks): "Yes, and draft the proposal".
- "Send it to Codex and commit after rev 1".

**Revision history.**
- **Rev 0** (same day) asked whether false abstentions are caused by misjudged sufficiency (B) or by sub-threshold commitment (C).
- **Codex review** (`/tmp/exp50-rev0-codex-review.md`, 14 points) recommended revising before any build. Rev 0 named its measures as if they were the model's internal sufficiency and confidence. They are elicited or output-side observables.
- **Rev 1** keeps the question and narrows the claims. Point-by-point resolution is in §9.

**Status.**
- Draft. Nothing is built, run or spent.
- Before any build it needs the user's "Go" and a D-number (proposed D-70). Spend needs its own approval under the two-pane rule.

**Relation to D-45.**
- §49 is closed (D-68), and this is not a rescue of §49's selectivity or calibration.
- It asks what *distinguishes* the failure §49 described ("the model writes the answer down, then refuses to commit to it") from success on the same templates.
- No §49 verdict is revisited.

## Status of the claim (fixed before the build)

**Candidate framing (adopted from Codex point 13).** A controlled test of whether signals associated with sufficiency and with city discrimination predict, and causally influence, false abstentions after an answer-bearing note. Outcomes are graded, and limits on identifying those signals are explicit.

**What this framing commits us to.**
- Novelty is conditional on the result, the controls and the final comparison to prior art.
- No first-in-setting claim is made.
- Stage 1 describes **association profiles compatible with accounts**. It does not identify the model's reason for abstaining.
- Stage 2, if it opens, shows **causal sensitivity of the decision to a projection**. It does not show that the projection means "sufficiency" or "confidence".

**Prior art and the boundary (corrected in rev 1, Codex point 13).**

| Paper | What it already covers | The boundary for §50 |
|---|---|---|
| Wagner, *Two Axes of LLM Abstention*, arXiv 2607.08456 | Answer confidence and answerability are separate axes; a hidden-state probe sees answerability. §6 adds probe-routed premise checking with random-gate controls and graded correction | False premises and unanswerable questions, not false abstention after an answer-bearing note. Its repair is behavioural, not an internal intervention contrasting the two signals. Its elicitation and surface-confound analyses bear directly on our S |
| Kumaran, Daw, Osindero, Veličković & Pătrăucean, Nature MI 2026 (arXiv 2603.22161) | Confidence and a threshold drive abstention: GPT-4o behaviour and threshold manipulation; Gemma 3 27B activation steering with option-balanced contrasts; mediation analysis | **Substantial overlap with the confidence-to-abstention account.** Closed-book. Contextual sufficiency against confidence is an extension only if the two are identifiable here |
| Kumaran, Conmy, Barbero et al., arXiv 2603.17839 | Verbal confidence is cached at the first post-answer position; routing is tested with swaps and attention blocking | Makes the choice of position consequential. §50 does not test caching (Q-5 is exploratory) |
| Joren et al., *Sufficient Context*, arXiv 2411.06037 | Sufficient-context failures; selective generation combining external sufficiency with reader confidence | **No novelty is claimed** for combining the two signals or for sufficient-context failures. The distinction is the reader's own elicited judgement against an external rater |
| Nguyen et al., *Commit-Abstain Circuit*, arXiv 2609.32964 | A circuit for commit against abstain, with interventions; limited to unanswerable inputs and decision-level outcomes | Absence of Qwen2.5 is a replication opportunity, not a gap. Possible gap: separating sufficiency-associated from discrimination-associated signals in false abstention, with graded outcomes |
| Lavi, Milo & Geva (EACL 2026); Xiros et al. 2607.05013; Liu et al. 2508.18760; Ling et al. 2507.16199 | Unanswerability directions; fabrication on unsolvable problems; the "unknown" option as a prompt artefact | Mostly the opposite failure, or descriptive |

**What would be mere reproduction:**
- "the answer is present but the model abstains" (Ling, Liu, our §49);
- "steering changes abstention" (Lavi, CAC, Kumaran, our §38);
- "confidence drives abstention" (Kumaran).

## 1. Stage-0 facts from §49 dev (post hoc, dev only; they motivate, they do not decide)

These are recounted from `runs/exp49/run/baselines/` on even (dev) bases, with Qwen2.5-3B-Instruct and the F2 format. They were not pre-registered, and no held row was used.

**False abstentions.** There are 149 false abstentions among 432 dev ANS-F2 rows. The gold city string occurs in the note in 140 of them (DE 83/87, RC 36/40, TU 21/22). This is a lexical condition, not a check that the note asserts the answer.

**Abstaining notes can read like answering notes:**

| Outcome | Note |
|---|---|
| abstain | "user mentioned that Ben lives in Manchester on 2023/03/01 (Wed) 10:00. / Answer: unknown" |
| gold | "user mentioned that Ben lives in York … / Answer: York" |

**By task and hardness.** False abstentions per 48 rows:

| Task | H0 | H1 | H2 |
|---|---|---|---|
| DE | 34 | 28 | 25 (plus 4 degenerate) |
| RC | 5 | 14 | 21 |
| TU | 3 | 6 | 13 |

**Flips across hardness.**
- 79 of 144 dev bases have both a gold and an abstain row across hardness.
- Gold-at-easier/abstain-at-harder pairs: 98. The reverse: 58.

**City effect.**
- The abstention rate by gold city runs from 5/36 (York, Cardiff) to 20/36 (Norwich).
- All 12 cities and " unknown" are single Qwen tokens.
- Not significance-tested.

**Use.** These motivate stratifying by task and adjusting for city (§4.4). They are not evidence for any account. Different profiles across tasks would be reported as **heterogeneity of association profiles**, not as "two causes" (Codex point 2).

## 2. Accounts and what each would predict (descriptive targets, not mutually exclusive)

| Label | Account | Profile it would predict in stage 1 |
|---|---|---|
| **B** | Misjudged sufficiency | Lower elicited sufficiency on false abstentions than on answered rows, toward the level of unanswerable twins |
| **C** | Weaker commitment to the found city | Elicited sufficiency similar to answered rows, but a lower gold-versus-rival-city margin |
| **B+C** | Successive parts of one route (sufficiency feeds commitment) | Both lower |
| **Neither** | Neither measure separates the classes | Both similar (inside equivalence regions). This does not show a late veto: timing and veto are causal claims that §50 does not test |

## 3. Population and data

**Model and format.**
- Qwen2.5-3B-Instruct, F2 only, using the §45 and §49 template and renderer verbatim.
- Precision as on the §49 pod, fixed in the build.

**Generator: newly generated cases** (Codex point 7).
- `scripts/exp49_cases.py` with seed 50, extended as below.
- A new seed does not make the content new. The name and city pools are finite (for DE, 12 × 12 core tuples).
- **Overlap with §49:**
  - The build reports overlap in core tuples and exact prompts with all of `runs/exp49/cases49.jsonl`.
  - Exact prompt duplicates of any §49 row are dropped before the split.
  - Core-tuple overlap is reported, not removed.
- **Splits:**
  - Even base index is dev, odd is held.
  - Dev is split by base into **dev-fit** (base index ≡ 0 mod 4) and **dev-check** (≡ 2 mod 4). Calibration and evaluation of every gate and direction stay separate (Codex points 4 and 8).
- Gold is balanced over the 12 cities per split × task before selection.

**Sets per split** (48 bases per task, 3 tasks):

| Set | Hardness | Rows | Purpose |
|---|---|---|---|
| ANS-F2 rep a | H0, H1, H2 | 432 | answerable |
| ANS-F2 rep b | H0, H1, H2 | 432 | the same facts and the same line order. **Every timestamp, including the question's, is shifted by +7 days** (weekday labels preserved, order and relative dates preserved). No line is reordered. Nothing else changes. Fixed before generation; no search for flip-causing changes |
| UNA-rel | H0, H1, H2 | 432 | §49's relation-replaced twin, now at every hardness |
| UNA-swap | H1 | 144 | the same memory lines as ANS rep a, with a question about another mentioned entity whose queried relation is absent. Construction is per task, below |
| ANS-F1 | H1 | 72 | output-consistency diagnostic for the margin (§4.2), not a validation |

**UNA-swap construction, per task** (Codex point 4).
- **DE:** "Where does {D} live?", where D is an H1 distractor person who appears only with a job line.
- **RC:** "In which city does {D}'s supervisor work?", where D appears with a job but no supervisor edge.
- **TU:** "Where does {D} live now?", where D appears with a job but no city line.
- **If a base has no suitable D,** that base contributes no UNA-swap row. The count is reported.
- **Known scope.** "D has only a job line" may itself predict answerability. UNA-swap therefore narrows but does not remove template cues. The result is reported as template-scoped.

**Generator checks:**
- pet and job nuisance lines in equal proportions across ANS and UNA;
- rep b differs from rep a only in timestamp strings;
- UNA-swap's memory block is byte-identical to rep a's.

**Outcome classes** (the §45 and §49 classifier): gold, abstain, competitor, wrong-other, degenerate. Every count is reported for every set, task and hardness.

## 4. Stage 1: measures, populations, estimands (held; one read)

### 4.1 S, elicited sufficiency (renamed, Codex point 1)

**What S is.** S is **elicited in a separate prompt, without using the observed F2 decision as its label**. It is the model's answer to a new question, which may involve fresh processing. It is not a readout of the sufficiency judgement made during the F2 run.

**Prompt.** The same memory block, then one of:
- DE: "Does the memory give enough information to tell where {A} lives? Reply yes or no."
- RC: "Does the memory give enough information to tell in which city {A}'s supervisor works? Reply yes or no."
- TU: "Does the memory give enough information to tell where {A} lives now? Reply yes or no."

The "enough information to tell" wording names inferential sufficiency, not explicit mention. That ambiguity for RC and TU is resolved toward inference (Codex point 4).

**Score.**
- s = P_yes / (P_yes + P_no) at the first answer position, with no generation.
- P_yes sums the token variants {"yes", " yes", "Yes", " Yes"}, and P_no likewise for "no".
- Reported beside s:
  - the summed yes+no mass;
  - the yes/no decision (s ≥ 0.5).
- A row with yes+no mass < 0.5 is counted and excluded from S.

**Polarity control (dev-check only).** The reversed question "Is information needed to tell … missing from the memory? Reply yes or no." The agreement between the two decisions is reported.

**Validity gate.** Computed on dev-check, per task, separately against UNA-rel and UNA-swap at matched hardness. All three conditions are required:
1. AUROC(ANS vs UNA) with a lower 95% CI bound ≥ 0.75;
2. a yes rate ≥ 0.70 on answered ANS rows and ≤ 0.30 on UNA rows (absolute judgement, not only ranking);
3. polarity agreement ≥ 0.70.

A task failing the gate gets S = **invalid** for that task. Its margin results are still reported.

### 4.2 M_city, the gold-versus-rival-city margin (renamed, Codex points 1 and 3)

**Definition.** M_city = z_gold − max over the other 11 cities of z_c, from the logits at P_ans: the last token of the model's own teacher-forced F2 output up to and including `Answer:`.

**What M_city is not.**
- It is **not** a validated confidence measure.
- It is **not** independent of the commit decision. It shares the hidden state and z_gold.
- A common shift in all city logits, or a shift in z_unknown alone, can change the decision with M_city unchanged.

**Reported beside it: M_commit** = z_gold − z_" unknown", as an outcome-proximal descriptive diagnostic. It is not evidence. A "near tie" is claimed only from M_commit's distribution, reported as the share of false abstentions with |M_commit| < 1 logit, with no reading attached.

**ANS-F1 diagnostic.** M_city at the F1 answer position is reported separately for gold, wrong-city, competitor and degenerate outcomes. This is an output-consistency check in the F1 format, not validation of a confidence construct. There is no gate.

**Feasibility, not validity.** The spread of M_city among answered rows (IQR) is reported. No IQR threshold is used.

**Secondary K_v.** An elicited 0–9 verbal rating of the noted city. Separate prompt, expected digit, descriptive only; self-report may be unfaithful.

### 4.3 Joint population (Codex point 6)

**Rule: answer-bearing note**, audited. The gold string occurs in the note and:
- no negator ("not", "no longer", "never", "moved from") occurs within the 6 words before it;
- for TU, the gold is the last city mentioned in the note.

**Audit.** The Advisor hand-checks 40 dev-check rows against this rule before the design is committed. The agreement is reported. The text always says "answer-bearing note (rule above)", never "found the answer".

**Primary joint population.** ANS-F2 rows of rep a and rep b with:
- an answer-bearing note;
- outcome gold (answered) or abstain (false abstention);
- valid S (yes+no mass ≥ 0.5).

S and M_city are compared on **the same rows**. S on all gold or abstain rows is reported separately.

**Feasibility per task.**
- ≥ 30 false abstentions from ≥ 15 distinct held bases;
- ≥ 30 answered rows from ≥ 15 distinct held bases.

A task below this is **underpowered**. It is reported descriptively and gets no profile label.

### 4.4 Estimands (continuous contrasts are primary, Codex point 5)

**Cells.**
- Cells are task × hardness × gold city.
- Only cells with both classes present enter.
- Cells without overlap are counted and reported, not extrapolated.
- Weights are proportional to the false-abstention count per cell.
- If fewer than 50% of a task's false abstentions fall in overlapping cells, fall back to task × hardness cells. The fallback is reported.

**Primary quantities, per task:**
- **Δ_S** = weighted mean of (s̄_falseAbst − s̄_answered) over cells.
- **Δ_M** = weighted mean of (M̄_falseAbst − M̄_answered) over cells, in logits.
- Also reported: Δ_M standardised by the pooled within-cell SD of answered rows, and AUROC_M.

**Reference separation for S.** R_S = weighted mean of (s̄_answered − s̄_UNA-rel) over task × hardness cells (UNA-rel exists at all hardness levels).
- The ratio r_S = −Δ_S / R_S is reported **only if R_S's lower CI bound ≥ 0.20**. Otherwise r_S is "undefined".
- Numerator and denominator are always reported.
- r_S is not clamped: values outside [0, 1] are reported as they are.

**Bootstrap** (Codex point 11).
- 10k resamples, seed 0.
- Bases are resampled within task, with all rows of a base (both replicates, all sets) kept together.
- Cell weights are recomputed in each replicate.
- A replicate in which a needed cell or reference is empty is dropped and counted. If more than 5% are dropped, the CI is reported as unstable.

### 4.5 Profile labels (operational, after the continuous estimates)

Each measure gets one of five states per task. The equivalence regions are fixed now.

| State | Rule for Δ_S | Rule for Δ_M |
|---|---|---|
| **lower** | upper CI bound < 0 and point ≤ −δ_S | upper CI bound < 0 and point ≤ −δ_M |
| **similar** | 90% CI inside ±δ_S (equivalence) | 90% CI inside ±δ_M |
| **higher** | lower CI bound > 0 and point ≥ +δ_S | lower CI bound > 0 and point ≥ +δ_M |
| **inconclusive** | anything else | anything else |
| **invalid / underpowered** | gate failed, or feasibility not met | feasibility not met |

**Equivalence margins (operational choices, not mechanism boundaries).**
- δ_S = 0.25 × R_S (a quarter of the answerable-to-unanswerable gap). If R_S is undefined, S can only be "lower", "higher" or "inconclusive".
- δ_M = 0.5 logit.

**Profiles.**
- **B-compatible:** S lower, M similar.
- **C-compatible:** S similar, M lower.
- **Joint:** both lower.
- **Neither separates:** both similar.
- **Other:** any "higher" or "inconclusive" state is reported as it stands.

A failed or invalid measure does not block reporting of the other measure.

**Q-2, within-content flips.**
- Pairs: rep a and rep b of the same base × hardness, with an answer-bearing note in both and different decisions.
- Paired differences in s and M_city, abstained minus answered, with a base-clustered CI.
- This controls base identity and nominal hardness, not all processing (the dates and the generated notes differ).
- It is a selected, conditional population, and is reported as such.

**Q-3.** S and M_city by hardness and by gold city: descriptive. Does either track the stage-0 city effect?

**Q-4.** The correlation of s with M_city among answered rows; K_v beside M_city.

**Q-5 (exploratory, read-site dependence; not a caching test, Codex point 12).**
- In the note, the *producing position* is the token before the gold's first occurrence; its output logits predict the city. M_city is computed there with the same formula.
- The *receiving position* is the gold token itself. Only the hidden-state norm and the projection on u_M (if §5 fits it) are recorded there; logits at that position predict the next token and are not interpreted.
- Positions are taken at the first occurrence.
- The correlation with M_city at P_ans is descriptive. No claim about the Kumaran et al. 2603.17839 mechanism is made.

## 5. Stage 2: subspace-restricted residual interventions (conditional)

**Opening rule.** For at least one task, all of the following hold:
- a B-compatible, C-compatible or joint profile in stage 1;
- ≥ 20 held Q-2 flip pairs from ≥ 10 bases.

**Conditioning (Codex point 11).**
- Stage 2 runs on the same held cases that decided eligibility. It is a **prespecified conditional analysis, not an independent confirmation**, and is reported that way.
- Every candidate layer, direction, donor rule, threshold and classifier is frozen in the design file before any held access.
- Results are reported for every eligible task, not only the most favourable one.

**Pairs.**
- **Forward:** the recipient is the abstained row, the donor is the answered row of the same Q-2 pair.
- **Reverse:** the donor is the abstained row, the recipient is the answered row.
- Cross-hardness pairs are not used.

**Site and hook.**
- Residual stream output of decoder block ℓ at P_ans (the patch46 hook), recipient's own frozen prefix, KV cache rebuilt from the patched forward pass.
- Greedy continuation to 16 tokens, scored by the §49 first-city rule.
- **No-op replay:** patching the recipient with its own state must reproduce its baseline outcome on ≥ 0.95 of dev pairs, or the stage stops as a build fault.

**Layer ℓ (one global layer; Codex point 8).**
- Eligible layers: 1–35. The embedding output is excluded.
- ℓ = the earliest layer at which the whole-residual forward patch W releases gold on ≥ 50% of **dev** flip pairs.
- Denominator: all dev flip pairs with valid P_ans in both rows.
- Minimum support: 20 dev pairs from ≥ 10 bases. Otherwise stage 2 is "not feasible".
- If no layer reaches 50%: **no qualifying single-site transfer was found under this protocol**. The stage stops.
- The 50% value is an operational feasibility rule. §49's R2 result is a different intervention and is not a calibration for it.

**Directions (global, task-balanced; fitted on dev-fit, checked on dev-check).**
- **u_S:** at layer ℓ, at the S prompt's answer position, the difference of means between ANS rows and UNA rows.
  - UNA-rel and UNA-swap get equal weight.
  - Rows are city-balanced within task by subsampling to the smallest city count.
  - Normalised to unit length.
- **u_M:** at layer ℓ, at F2 P_ans, among **answered** answer-bearing rows only, the difference of means between the top and bottom M_city terciles.
  - Computed within task × city cells, then averaged, so that city identity does not dominate.
  - Normalised.
  - Fitted only on rows whose decision did not vary, so the F2 decision is not its label.
- A zero or undefined direction makes its arm untestable.

**Transfer checks (dev-check, at F2 P_ans).**
- u_S projection separates ANS from UNA-swap: AUROC lower CI bound ≥ 0.60.
- u_M projection has Spearman correlation with M_city ≥ 0.30, lower CI bound > 0.
- A direction that fails is "untestable". The failure is reported and not tuned. Passing establishes association, not meaning (Codex point 10).

**Geometry.**
- The angle between u_S and u_M is reported.
- If |cos| > 0.8, the S-against-M comparison is **not identified** and only the span arm is read.

**Arms** (Δ = h_donor − h_recipient; U = [u_S, u_M]; Π = U(UᵀU)⁺Uᵀ):

| Arm | Patched state |
|---|---|
| W | h_recipient + Δ |
| P-S | h_recipient + u_S u_Sᵀ Δ |
| P-M | h_recipient + u_M u_Mᵀ Δ |
| P-span | h_recipient + Π Δ |
| P-perp | h_recipient + (I − Π) Δ |
| Rand-X, for X ∈ {S, M, span} | h_recipient + ‖patch_X‖ · r̂ · sign(r̂ᵀΔ) for each of 5 random unit vectors r̂ (seeds 0–4 per pair); the outcome is the mean over the 5. Norm-matched to arm X on every pair |
| Same-outcome | donor and recipient both answered (forward) or both abstained (reverse), same task × hardness, nearest in M_commit. Interpreted empirically, with no "should be zero" claim |

Patch norms are reported per arm.

**Outcomes.**
- Rates of gold, wrong city, abstain and degenerate per arm, forward and reverse.

**Primary paired contrasts, per eligible task and pooled:**
- (P-S − Rand-S);
- (P-M − Rand-M);
- (P-S − P-M).

Each uses a base-clustered paired CI.

**Multiplicity.**
- Holm across the three pooled contrasts × two directions.
- Per-task contrasts are descriptive.
- The release-rate ratio P-X / W is descriptive only and undefined when W releases nothing.

**Readings, bounded.**
- A positive P-X − Rand-X shows the decision is causally sensitive to the donor's displacement along u_X beyond a norm-matched random displacement, in the selected flip population.
- It does not identify u_X as sufficiency or confidence, and it does not localise components (Codex point 10).
- A null does not show the concept is unused.
- No repair claim is made. Effects on unanswerable or already-correct cases are not measured, so the claim stays limited to flip pairs.

## 6. Limits (fixed)

- **Scope:**
  - one model (Qwen2.5-3B-Instruct);
  - synthetic city templates with a finite pool;
  - the F2 format only;
  - a one-position forced readout in stage 2.
- **S is elicited in a separate prompt.** M_city is an output-side margin. Neither is validated as an internal construct.
- **The "answer-bearing note" rule is lexical plus an audit.**
- **Q-2 and stage 2 use selected flip pairs.** Stage 2 is conditional on stage 1's held data.
- **Equivalence margins and gates are operational.** Pre-registration stops them being changed after the data; it does not validate their interpretation.
- **The literature check was not exhaustive.** Most papers were read at abstract level, except Codex's primary-text reading of Wagner, Kumaran (both), Joren and CAC.
- **Out of scope:** math, code and the Qwen3 thinking toggle.

## 7. Compute, cost, authorisation, order

**Rows.**
- About 1,512 rows per split: 1,440 F2 generations of up to 160 tokens, and 72 F1.
- One forward pass per row for S, plus dev-check polarity prompts, plus K_v on answer-bearing rows.
- Teacher-forced captures at P_ans.
- Stage 2: about 35 layers × the dev pairs (for ℓ), then about (pairs × 12 arms, where each Rand arm counts its 5 seeds as one arm × 5) × 2 directions of 16-token continuations, on dev (checks) and held.

**Placement.**
- Laptop ($0): build, tests, a one-unit timing.
- One pod per stage if the timing projects the laptop above 24 h.
- **No cost quote until the one-unit timing is measured.**
- **Proposed hard cap for all of §50: $3.00.** Spend to date is about $22.85, so this stays inside the $30 cap (D-40).

**Pod conditions:**
- the D-22 smoke and the D-30 fresh-venv bundle test;
- never two pods at once;
- list-pods empty after each pod, verified from both panes;
- the Executor creates a pod only after the user's own approval line appears in the Executor pane.

**Order.**
1. The user's "Go" (D-70).
2. Generator and build, with tests:
   - nuisance balance;
   - the rep b diff contains timestamps only;
   - UNA-swap memory byte-identical;
   - §49 overlap report;
   - the no-op replay unit test;
   - projector algebra tests (Π idempotent; P-S + P-perp ≠ W when cos ≠ 0 is handled by Π).
3. Advisor build review.
4. Timing, then quote.
5. Dev run: S gate; the answer-bearing audit (40 rows, Advisor); ℓ; u_S, u_M; transfer checks; no-op replay.
6. Commit the design file (sha256; the guard from §49 build fix 3).
7. Held stage 1, then a single read.
8. If stage 2 opens: held stage 2, then a single read.
9. Advisor recount from per-trial files.
10. Results review.
11. Record changes on the user's "Go".

**Stopping (D-45).**
- One held read per stage. The result is documented, whatever it is.
- No rescue variant.
- Nothing new follows from §50 without a separate decision.

## 8. Answers to the rev 0 self-review (from Codex point 14)

1. **S gate:** AUROC alone misses uniform "yes". The gate now also requires absolute yes rates and polarity agreement (§4.1).
2. **M_city ceiling:** spread is reported as feasibility. No IQR threshold; the construct is renamed rather than "validated".
3. **Rep b:** the timestamp shift only (+7 days; weekdays, order and relative dates preserved), with no reordering. Acknowledged to change processing (the dates and the notes differ).
4. **Cut-points:** the r_S thresholds are removed. Continuous contrasts are primary, and equivalence margins are fixed as operational choices.
5. **ℓ ≥ 50%:** kept as an operational feasibility rule, with denominators and support fixed. Failure supports only a bounded negative.
6. **Task heterogeneity:** pre-registered and worded as "different association/intervention profiles".

## 9. Resolution of Codex's rev 0 review (`/tmp/exp50-rev0-codex-review.md`)

| Point | Change in rev 1 |
|---|---|
| 1 | S renamed "elicited in a separate prompt…". K renamed to M_city, with the independence claim withdrawn. M_commit reported as outcome-proximal. "Near tie" only from M_commit's distribution, descriptive (§4.1, §4.2) |
| 2 | The accounts are non-exclusive (§2). Stage 1 describes profiles. Heterogeneity is not called "two causes" |
| 3 | The F1 gate is removed; F1 is a per-outcome diagnostic. No confidence claim; no IQR threshold (§4.2) |
| 4 | Absolute yes rates, token variants and normalisation; a polarity control; inferential wording; UNA-swap construction per task, with its template scope stated; dev-fit/dev-check separation; transfer checked against UNA-swap (§3, §4.1, §5) |
| 5 | Continuous Δ_S and Δ_M primary; r_S only with a defined reference and unclamped; equivalence regions; "Neither separates" in place of V; a joint state; gate precedence (§4.4, §4.5) |
| 6 | A common joint population; the answer-bearing rule with an audit; feasibility in bases; city-adjusted cells with the non-overlap reported (§4.3, §4.4) |
| 7 | "Newly generated cases"; overlap with §49 reported and exact duplicates dropped; rep b's transformation fixed; Q-2 wording corrected (§3, §4.5) |
| 8 | ℓ fully specified (global, hook, range, denominator, support, no-op replay); the failure wording; §49 R2 not used as calibration; directions specified with fit and check separation (§5) |
| 9 | Projector Π for the joint span; P-perp via I − Π; the angle and an identifiability rule; norm-matched random arms per tested patch; same-outcome controls interpreted empirically (§5) |
| 10 | "Subspace-restricted residual interventions"; causal sensitivity, not semantics; the release-rate ratio descriptive only; paired P-S − P-M CI; all outcome categories; no repair claim (§5) |
| 11 | Stage 2 labelled conditional; everything frozen before held; all eligible tasks reported; Holm on the pooled contrasts; the bootstrap fully specified (§4.4, §5) |
| 12 | Q-5 redefined as exploratory read-site dependence, with producing and receiving positions separated; no caching claim (§4.5) |
| 13 | Literature table corrected for Wagner §6, Kumaran 2603.22161's overlap, Joren and CAC; the candidate framing adopted (Status) |
| 14 | §8 |

## Suggested one-line summary for meetings

"We have a reviewed draft of a follow-up. It asks what separates the cases where the model writes the right answer in its note and then says 'unknown' from the cases where it answers. Does its own yes/no judgement of the evidence differ, or does its preference for the right city over the others differ? A conditional causal check follows if either does. It is not yet built or funded, and there are no results."
