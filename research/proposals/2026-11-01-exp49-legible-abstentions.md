# §49 proposal, rev 0a: legible abstentions. Can information decoded from a failing model guide correct answering on answerable trials, while keeping abstention where the evidence is insufficient? (2026-11-01, pre-registration draft)

## Origin and authority

**The user's orders, Advisor pane, 2026-11-01:**
- "let's write out option D and then let's also run option A. Make the appropriate changes to the slides"
- "You have my approval for ALL pod runs as long as the total does not exceed $5"

The $5 is read as the **total pod spend for this work**. It is inside the programme cap of $30 (D-40), with spend at about $22.00. It is recorded as D-63.

**Scope change from A to A′ (material, documented here before any build).** Option A asked whether the latent answer tracks counterfactual evidence on controlled failures. *Legible Failures* (Ravulapalli, Chadha & Hari, arXiv 2609.11216) already does this:
- randomised bindings;
- failure-conditioned probes;
- a query counterfactual;
- gated repair with harm counted on correct trials.

So A became A′, a test of the **boundary of decoded-target repair**: does it work on **false abstentions** as on wrong answers, and does it keep **genuine** abstentions? The basis is `/tmp/thesis-contribution-audit.md` Addendum 1 and Codex's reviews (`/tmp/thesis-contribution-audit-codex-review.md`, `/tmp/thesis-a-prime-codex-review.md`).

**What this does not cover.** This proposal does not cover Option B, Option C or any spend beyond D-63.

## Status of the claim (fixed before the build)

- **A contribution candidate, not established novelty.**
- Adding an abstention centroid to *Legible Failures*' repair is an implementation step, not novelty.
- The scientific value, if any, comes from testing one **contrast**: decoded-target repair on false abstentions versus on wrong answers, with the cost on genuinely insufficient evidence measured.
- The decoding arms partly replicate:
  - *Legible Failures* §3;
  - Ling et al. 2507.16199 v8 ("mid-layer representations retain the right answer" under an "Unknown" option).
- The wrong-answer arms are labelled as replications of *Legible Failures* §3 and §5.

## 1. Background facts, qualified

**§45 recount** (stage-1 maps, `runs/exp45_maps/j56-jrefit-logit/stage1/`):
- In F2 the Qwen 3B answers "unknown" on 154 of 432 answerable prompts. In F1 it does so on 0 of 432.
- **Bundled and exploratory.** F2 differs from F1 in several things at once:
  - the "unknown" instruction;
  - the memory framing;
  - the timestamps;
  - a required note;
  - the output format;
  - the generation budget (160 against 16 tokens) and the scoring span.
- So the contrast does not isolate the instruction or locate a mechanism.
- F1's failures are wrong answers (161) and competitors (47). Not abstaining is not the same as answering correctly.

**The 432 rows per format are not independent.** They reuse 36 base cases across 4 variants and 3 hardness levels.
- The F2 abstention counts by hardness are 31/144 (H0), 72/144 (H1) and 51/144 (H2), as rows, not bases.
- H0 means *no distractors*, not *no reasoning*: RC is two-step and TU is temporal.

**"Prior-free" is shorthand.**
- Independently resampled assignments remove stable *entity–value* associations.
- They do not remove token-frequency or answer-set priors.
- A correct output alone does not prove that the answer came from the evidence. Hence the tracking check in §3, M2b.

## 2. Population and data (new generator; dev chooses, held decides)

**Generator** (new script; §45 templates and F1/F2 renderers reused verbatim):
- **Independent bases with balanced labels.** Each *base* draws its names, cities, jobs and distractors independently (seed 49, base index).
  - Within each split × task, the gold city is balanced across the 12 cities (each city is gold on an equal share of bases).
  - The label is independent of everything else, so labels are not confounded with base case (the §46 lesson).
- **Matched variants stay in one split.** One base yields all of:
  - its hardness levels H0, H1 and H2;
  - its counterfactual twin (the gold city replaced by another, balanced city; used in M2b);
  - its unanswerable twin.
- **Splits are by base.** Even base index is dev, odd is held. Uncertainty is computed with **bootstrap clustered by base**.
- **City-answer questions only** (K = 12 classes):
  - DE "Where does {A} live?";
  - RC "In which city does {A}'s supervisor work?";
  - TU "Where does {A} live now?" (the competitor is the old city).
- **Unanswerable twin (UNA) without a superficial marker.** The queried entity is still mentioned, but the needed relation is absent:
  - DE: A's city fact becomes "{A} works as {a J}.";
  - RC: "{A}'s supervisor is {B}." is kept, and B's workplace fact becomes "{B} works as {a J}.";
  - TU: both of A's city facts become one job fact and one "{A} has a {pet}" fact, from a fixed small pool.
  - Same number of lines, same hardness slots.

**Sizes per split.**

| Set | Format | Bases per task | Rows per split | Purpose |
|---|---|---|---|---|
| ANS-F2 | F2 | 48 (H0/H1/H2 each) | 432 | answerable; false abstentions come from here |
| CF-F2 | F2 | the same 48 bases, H1 only | 144 | counterfactual twins (M2b) |
| UNA-F2 | F2 | the same 48 bases, H1 only | 144 | genuinely insufficient evidence |
| ANS-F1 | F1 | 24, H1 only | 72 | wrong-answer calibration (replication) |

- 48 bases per task is 4 per gold city.
- The Executor may halve the bases (24 per task) only if the measured one-unit timing projects the whole run above the $5 cap. Any such cut is decided before the dev read and recorded.

**Outcome classes** (baseline generation; the §45/patch46 classifier):
- gold;
- abstain ("unknown");
- competitor;
- wrong-other;
- degenerate.

**Reporting.**
- Every baseline outcome is reported for every set, task and hardness.
- All conditional analyses (for example, on abstentions) are labelled as subsets.

## 3. Read sites, arms and outcomes

**Read sites (kept separate; never pooled).**
- **P_pre:** the last prompt token (the "Note:" token in F2), before the model writes anything. This is the pre-note site. **Primary for decoding.**
- **P_ans:** the last token of the model's own teacher-forced output up to and including `Answer:`. This is the post-note site, labelled as such.
  - The note may already state the city, so decoding at P_ans is **not** evidence of an unexpressed answer.
  - Each F2 trial carries a **note-gold flag** (the gold city's string appears in the generated note) and a note-substitute flag. Every P_ans analysis is split by this flag.
- **Echo control.** The §45 rule B is applied, the input-token echo is excluded, and no teacher-forced prefix ever contains a gold that the model did not itself write.

**M1. Decoding (calibration diagnostics; restricted pool).**
- **Probe.** Multinomial logistic regression over the 12 cities, on standardised residuals at P_pre and, separately, at P_ans, per layer. C = 0.5 (*Legible Failures*' value, not tuned).
- **Fit and layer choice.** Fit on dev ANS (F1 + F2, all outcomes). Choose ℓ* per site on a dev holdout third of the bases, by overall accuracy.
- **Read on held.** pAcc on F2-abstain, F2-gold, F2-wrong, F1-wrong and F1-gold, against three floors:
  - 1/12;
  - the in-prompt candidate floor;
  - a label-shuffled probe.
- **Caveat.** A restricted-pool probe excludes abstention by construction and always picks a city. These numbers are diagnostics, not answerability.

**M2. Generated outcomes under repair (primary).**
- Each arm acts at P_ans on the model's own frozen prefix, followed by greedy continuation of up to 16 tokens.
- R4 is the exception: it regenerates from scratch.

| Arm | What it does | Uses gold? | Set on dev |
|---|---|---|---|
| R1 decoded-target | add α·‖h‖·unit(μ_ĉ − μ_abst) at ℓ*(P_ans). ĉ is the P_ans probe decode; μ_c is the dev class mean of gold-outcome trials with gold c; μ_abst is the dev mean over F2 abstentions | no | α ∈ {0.5, 1, 2} |
| R2 refusal direction | the §38 layer-29 direction, α 2 (the §40 fixed setting) | no | fixed |
| R3 random | matched-norm random vector at ℓ*, seeded per trial | no | none |
| R4 instruction change | the F2 prompt without the "If the memory does not contain the answer, reply: unknown" sentence, regenerated | no | none |
| D-gold (diagnostic only) | R1 with ĉ replaced by the gold | **yes** | α as R1; never a deployed arm |

**Populations for M2 (held).**
- All F2 trials whose baseline abstains: ANS-F2-abstain (false abstentions) and UNA-F2-abstain (genuine abstentions).
- Plus the CF-F2 twins of false-abstention bases (M2b).

**Scoring (no judge).** The first city on the answer line (patch46 `first_city`) gives one of:
- correct;
- wrong city or other answer;
- abstain;
- degenerate.

**M2a outcomes.**
- On false abstentions: correct, wrong and abstain rates per arm.
- On genuine abstentions: the unsupported-answer rate per arm (any city is an error).

**M2b, evidence tracking.**
- Only for released-correct ANS trials under R1 and R2.
- The same arm runs on the base's CF twin.
- Tracking means the released answer is the CF city.
- Correct output alone is not provenance.

**M3. Gates (secondary; no gold used to trigger or target).**
- **G-content.** Fire R1 iff the baseline output is "unknown" and the P_ans probe's max probability is ≥ τ_c. This is expected to be a poor sufficiency signal: a forced-pool probe chooses even on unanswerable inputs. It is tested as such.
- **G-suff.** Fire R1 iff the baseline output is "unknown" and a binary **sufficiency probe** at P_pre says "sufficient" with probability ≥ τ_s.
  - The sufficiency probe is logistic regression, trained on dev ANS-F2 against dev UNA-F2.
  - It is evaluated on held as the independent test.
- τ_c and τ_s are chosen on dev to maximise U.
- **Utility U.** +1 correct, −λ wrong or unsupported, 0 for an abstention, on the held abstention pool (false + genuine) at its natural held proportions. λ 1 is primary; λ 2 is reported beside it.
- The pool mix is set by design (ANS 432 : UNA 144 rows), so per-class rates are reported beside U.

**Calibration replication (labelled).**
- On held F1-wrong, *Legible Failures*' own repair μ_ĉ − μ_e at ℓ*(P_pre on F1), gated on probe ≠ output.
- Harm: the same gate on F1-gold, reporting the share of correct answers kept.

**Damage accounting.**
- R1, R2 and the gates fire only on abstentions, so they cannot damage correct F2 answers by construction (stated).
- R4 changes the prompt for every trial. So R4 is also run on all held ANS-F2 trials whose baseline is correct, and its correct-to-wrong damage is reported.

## 4. Primary comparison (frozen) and secondary quantities

**Primary contrast (Q-P).**
- On held false abstentions: the gold rate under R1 minus R3.
- Beside it, the unsupported-answer rate under R1 minus R3 on held genuine abstentions.
- Both with paired, base-clustered bootstrap 95% CIs (10k resamples, seed 0).

**Secondary quantities.**

| Label | Quantity |
|---|---|
| Q-2 | R2 and R4 on the same two populations; R1 − R2; R1 − R4 |
| Q-3 | M2b tracking rate under R1 and R2 (with CI), against the chance rate of naming the CF city |
| Q-4 | U(G-suff) − U(never), U(G-content) − U(never), and each against U(always R1) |
| Q-5 (replication) | F1 pAcc\|wrong − floors; *Legible Failures* repair gain on F1-wrong; correct kept on F1-gold |
| Q-6 (contrast, descriptive) | the repair gain on false abstentions (R1 − R3) against the repair gain on F1 wrong answers (*Legible Failures* arm − random), each with a CI; and pAcc at P_pre on F2-abstain against F1-wrong |

Breakdowns by task, hardness and note-gold flag are descriptive.

**No novelty threshold.** Every quantity is an estimate with an interval against its floors.

## 5. Readings (fixed)

- **Q-P gold gain above 0, with a small unsupported-answer cost.** Decoded-target repair releases correct answers on false abstentions on these templates, with a measured cost on insufficient evidence.
  - Positive M2b tracking supports the claim that the released answers follow the supplied value.
  - None of this shows the original failure's mechanism or location, or that a completed answer was "released".
- **Q-P gain above 0 but unsupported answers rise as much.** Repair is an answer switch, not a selective one. This replicates our §40/§42 picture on prior-free templates.
- **Q-P gain not above 0.** Decoded-target repair does not transfer from wrong answers to abstentions at this site and strength. This is not evidence that the value is absent.
- **Q-4.** G-content failing is expected. G-suff succeeding would be a candidate selective-repair result on templates only, needing a naturalistic test under a separate decision. G-suff failing replicates the selective-steering negative.
- **Q-6.** A difference between the repair gains on abstentions and on wrong answers is descriptive. No equivalence is claimed from overlapping intervals.

## 6. Limits (fixed)

- Readout-limited conclusions:
  - restricted-pool probes are diagnostics;
  - a forced or prefixed release shows an effect at P_ans, not the source of the natural failure;
  - negative decoding does not show absence;
  - no difference does not mean equivalence.
- Synthetic templates only. New names from the §45 generator are not a fresh task distribution.
- The F1/F2 contrast is bundled and descriptive.
- The post-note site may decode what the note states (hence the note-gold split).

## 7. Compute, cost, authorisation, order

**Cost.**
- About 1,300–1,600 baseline generations per split (F2 up to 160 tokens).
- Teacher-forced captures at both sites.
- Release arms: about 5 arms × (false + genuine abstentions + CF twins) on held, plus the dev α and τ selection.
- R4 on all held correct ANS-F2.
- Order of 4,000–6,000 generations in all, most of them 16 tokens.
- **No time or cost quote until the one-unit timing is measured.**

**Placement.**
- Laptop fp32 CPU ($0) for the build, tests and a timing unit.
- Dev and held runs on one A40 pod under D-63, total pod spend for this work ≤ $5.00. The quote comes from measured per-unit times, with:
  - a hard stop that sums to ≤ $5.00 with any earlier §49 pod;
  - the D-22 smoke;
  - the D-30 fresh-venv bundle test;
  - never two pods at once;
  - list-pods empty after each pod, verified from both panes.
- The two-pane rule stands: the Executor creates a pod only after the user's own approval line appears in the Executor pane. The D-63 line can be pasted there once.

**Order.**
1. Generator and build, with tests.
2. Advisor build review.
3. Timing unit.
4. Quote.
5. Dev run.
6. Commit the design file (ℓ* per site, α, τ_c, τ_s, sha256).
7. Held run, then a single read.
8. Advisor recount from per-trial files.
9. Results review.
10. Record changes on the user's "Go".

**Stopping.** One held read. The result is documented, whatever it is (D-45). No rescue variant.

## 8. What would be mere reproduction

- F1 decoding above floor and *Legible Failures*-style repair of wrong answers (run, and labelled).
- Decodability under an "Unknown" option alone (Ling).
- Release above random without graded outcomes, the unsupported-answer cost and tracking (Kumaran, CAC and Lavi in other settings).

## 9. Advisor self-review (rev 0a)

- **Codex's A′ review, points 1–6,** is carried into:
  - §1 (bundled contrast, rows not bases);
  - §2 (labels balanced over independent bases, matched variants in one split, base-clustered intervals, UNA without a superficial marker);
  - §3 (P_pre versus P_ans with the note-gold flag; forced-pool decoder ≠ gate; generated outcomes primary; gold only in the diagnostic arm);
  - §4 (frozen primary contrast);
  - §6.
- **Remaining risks.**
  - The F2 abstention rate on the new generator may differ from §45. The dev baseline reports it before any held read. If the held false-abstention count is under 30 rows, the primary contrast is read and reported with its width, with no rescue.
  - μ_abst may encode format. R3 and R4 are the controls.
  - The TU unanswerable twin uses a new "pet" relation, which changes vocabulary slightly. This is stated.

---

## Rev 0b amendments (2026-11-01, before any build): Codex review `/tmp/exp49-rev0a-codex-review.md`

**Status of these amendments.** They override rev 0a wherever the two differ. Neither the user's authorisation nor the $5 cap changes (D-63). Every data-dependent choice below goes into the design file whose sha256 is committed before the held read.

### B1. Sufficiency gate without shortcut cues (review point 1)

- **Same fact inventory, different answerable relation.**
  - Every ANS prompt gains one *nuisance-relation* fact about the entity the query concerns:
    - DE: "{A} works as {a J}.";
    - RC: "{B} works as {a J}.";
    - TU: "{A} has a {pet}.".
  - The pet pool appears in ANS and UNA alike.
  - The UNA twin is the ANS prompt with the query's city fact(s) replaced by the same number of lines of ordinary distractor facts about *other* people, from the base's distractor list.
  - So both labels carry the queried entity, the nuisance relation and the same line count. The only difference is whether the queried relation exists.
  - This adds one line to every ANS prompt compared with §45, which is stated as a change from the §45 templates.
- **Hardness is matched.** The sufficiency probe is fitted, its layer selected, and it is evaluated **on H1 rows only** (ANS-H1 against UNA-H1). Gates G-content and G-suff are applied and scored only on H1 abstentions.
- **Claim limit (fixed).** Any gate result applies to these templates only. It is not called general recognition of evidence sufficiency.

### B2. Evidence-tracking check M2b: exact unit (review point 2)

- **Unit.** H1 only. For base b, ANS-H1(b) and CF-H1(b) differ only in the gold city. CF-H1(b) stays at H1, and M2b is restricted to H1. The ANS-H1 and CF-H1 counts are reported.
- **Population.** Bases where the ANS-H1 baseline abstains **and** the unsteered CF-H1 baseline also abstains. That is the CF baseline, not uniform chance.
- **Measure.** The same arm at the same site on CF-H1(b). Tracking means the released answer is the CF city.
- **Reported counts.** R1 and R2 are reported with their own denominators: n bases in the population, n released, n CF-city, n ANS-gold. **R1 and R2 tracking rates are not compared with each other**, because their conditional populations differ.
- **Sanity check.** A no-op replay (a zero vector at P_ans on the frozen prefix) must reproduce the baseline continuation token for token on 20 trials drawn per split.

### B3. Support and fallbacks, fixed before any read (review point 3)

- **μ_c** (the decoded-target centroid at P_ans, per site and layer).
  - Primary: the dev F2 gold-outcome trials with gold c, when there are at least 5.
  - Fallback: pooled dev F1 + F2 gold-outcome trials with gold c, when there are at least 5.
  - Otherwise class c is **unestimable**. An R1 trial whose decode is c is recorded as "R1 unestimable" and scored as **no release** (it stays an abstention) in the primary contrast. Counts are reported.
- **μ_abst.** At least 20 dev F2 abstentions with a defined P_ans. Otherwise R1 is **not estimable** as a whole, the verdict says so, and R2 to R4 are still read.
- **Missing `Answer:` or truncation at 160 tokens.** P_ans is undefined, so the trial is excluded from M2 and M3. It is kept in the baseline tables, and exclusions are counted per class and split. P_pre analyses keep the trial.
- **F1 trial that emits no city.** μ_e is undefined, so the *Legible Failures* repair is not applied. The trial stays in the F1-wrong denominator as unrepaired.
- **No selection of an easier population.** Every exclusion is reported beside every rate.

### B4. Every selection, frozen (review point 4)

- **Standardisation.** Mean and SD are fitted on the probe's own training portion only: the dev fit thirds of bases. They are never fitted on all states, and never on held states.
- **Dev bases are split three ways by base index** (dev index mod 3):
  - fit (0);
  - select (1);
  - gate-tune (2).
- **ℓ\* per site** (12-way probe): highest select-portion accuracy. Ties go to the lower layer.
- **Sufficiency probe.** C = 0.5. Layer at P_pre chosen by select-portion AUROC on H1, ties to the lower layer. Refitted on fit + select at the chosen layer.
- **α for R1** (grid {0.5, 1, 2}). Objective, on gate-tune portion H1 rows: (gold rate on false abstentions) − (unsupported-answer rate on genuine abstentions). Ties go to the smaller α.
- **τ_c and τ_s.** Maximise U at λ 1 on the gate-tune portion H1 abstentions. Ties go to the higher τ (fewer firings).
- **The *Legible Failures* F1 repair α** uses the same grid. Objective: the gate-tune F1 repair gain minus the correct-kept loss on gate-tune F1-gold.
- **Design file contents.** Every value above, the probe weights' sha256 and the centroids' sha256, all committed before any held output is read.

### B5. R2 is a one-position variant (review point 5)

- R2 is the **§38/§40 layer-29 vector at α 2, added at P_ans only**. It is the one-position variant, as in §39's Steer-0, and is not the §40 all-token schedule.
- It is labelled "R2 (one-position)".
- The §40 effect size is not borrowed for any comparison.

### B6. Bounded conclusions (review point 6)

- **The primary contrast establishes direction-specific recovery** (R1 against random) on false abstentions, with the unsupported-answer cost on genuine abstentions **reported descriptively**. There is no practical-threshold label such as "small".
- **Selectivity (M3) is secondary,** in the text and on any slide.
- **The abstention-versus-wrong-answer contrast (Q-6) is interpretable only if the calibration arm succeeds:** the *Legible Failures* repair gain on held F1-wrong must have its 95% CI above 0.
  - If it does not, Q-6 reads "contrast not interpretable: calibration did not reproduce".
  - A non-positive R1 − R3 interval alone is reported as "no recovery detected at this site and strength". It is never a binary "repair does not transfer".
- **A G-suff failure is a second, bounded negative for a different gate.** It is not a replication of the §42 policy.

### B7. Counts and cost arithmetic (review point 7)

- **Baseline rows per split:** ANS-F2 432 + CF-F2 144 + UNA-F2 144 + ANS-F1 72 = **792**, so **1,584** in all. Of these, 1,440 are F2 rows of up to 160 tokens.
- **Teacher-forced captures:** one pass per baseline row covering both sites, so 1,584.
- **Short (16-token) generations, estimated** (the abstention counts are assumptions from §45's rate, to be replaced by the dev counts):

| Item | Approximate count |
|---|---|
| Dev α grid: 3 α × gate-tune H1 abstentions | 3 × ~50 |
| Dev LF-repair grid: 3 × gate-tune F1 failures | 3 × ~25 |
| Held arms R1, R2, R3, D-gold × (false + genuine abstentions) | 4 × ~230 |
| R4 regenerations on held abstentions and held correct ANS-F2 (160-token) | ~230 + ~280 |
| M2b on CF-H1: R1 and R2 | ≤ 2 × 72 |
| Gates (replays of R1 outputs; no new generations) | — |
| *Legible Failures* repair on held F1 | ~72 |
| No-op replays | 40 |

- Total: about **1,600 long + ~1,500 short generations, plus 1,584 captures**. This replaces rev 0a's "4,000–6,000" and "1,300–1,600 per split" prose.
- The quote comes from the measured one-unit timing **for each kind of unit** (one long baseline, one capture, one short release). The $5 total is unchanged.

### B8. Meeting sentence (from Codex, adopted)

"We have approved a bounded follow-up comparing decoded-target repair of false abstentions with wrong-answer repair, while checking costs when evidence is insufficient. The protocol and implementation are being reviewed; there are no results from this study yet."
