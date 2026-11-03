# §51 proposal, rev 1a: an agent proposes weight edits for note-then-unknown in Qwen2.5-3B. Is the repair selective, and does its account survive tests on unseen cases? (2026-11-02, pre-registration draft)

## Origin, authority and revision history

**The user's request, relayed through Codex, 2026-11-02:**
- "what if we ask the frontier LLM like claude to edit the weights of the QWEN model such that it stops abstaining?
  WHat if we can use it to understand what activations cause the model to tell me unknown despite having the answer.
  … make a plan of attack (ask the advisor to do this)".
- Then, in the Advisor pane: "Yes, draft the formal proposal."

**Planning history.** Advisory only; nothing was built, run or spent.

| Step | File | Outcome |
|---|---|---|
| Advisor plan, rev 0 → rev 1a | `/tmp/weight-space-programming-advisor-plan.md` | §13 binding |
| Codex review of rev 0 | `/tmp/weight-space-programming-codex-review.md` | 12 points |
| Codex review of rev 1 | `/tmp/weight-space-programming-rev1-codex-review.md` | 5 specification points |
| Codex close-out of rev 1a | `/tmp/weight-space-programming-rev1a-codex-review.md` | 3 details; design accepted for formal drafting |
| §51 rev 0 (this file) | — | Consolidated proposal |
| Codex review of §51 rev 0 | `/tmp/exp51-rev0-codex-review.md` | 9 points: keep the design, fix the operational specification |
| §51 rev 1 (d9e67b7) | — | Resolves the 9 points (Appendix A, last table) |
| Codex review of §51 rev 1 | `/tmp/exp51-rev1-codex-review.md` | Design accepted; narrow build-contract fixes |
| **§51 rev 1a (this file)** | — | §13 build contract; where §13 differs from §1–§10, §13 wins |

This proposal consolidates that plan into one specification. Superseded text is dropped; the resolution tables are in
Appendix A.

**Status.**
- Draft. Nothing is built, run or spent.
- Before any build it needs:
  - the user's "Go" and a D-number (proposed D-72);
  - the user's decisions in §12.
- Any spend needs its own approval under the two-pane rule.

**Relation to D-45.**
- §49 (D-68) and §50 (D-71) are closed. This is a new line of work, not a rescue of either.
- It does not revisit any verdict.
- It does not revive the B/C ("misjudged sufficiency" / "weak commitment") labels that §50 could not identify.

## Status of the claim (fixed before the build)

**Framing.** A frontier model (Claude) proposes a localized explanation and a weight edit. We then test, independently
of its explanation:
- **Branch R (repair):** whether the edit improves the trade-off between releasing correct answers and inventing
  unsupported ones, against frozen comparators;
- **Branch M (mechanism):** whether its mechanistic predictions survive interventions and hold out of sample.

Successful editing alone is not an explanation of false abstention.

**What this framing commits us to.**
- **No first-in-setting claim.** Novelty is conditional on the result and on a final literature comparison.
- **Replications stay replications:**
  - "an LLM agent can find a useful weight edit" (Takehi et al. 2026);
  - "ablating a refusal-like direction increases answering" (Arditi et al. 2024, adapted; Lavi et al. 2025; our §38/§49);
  - the late-layer override (Liu 2026; Ling 2026, reported findings).
- **Candidate contributions, if the result supports them:**
  - an edit with a better release-versus-unsupported trade-off than every frozen comparator, on fresh content;
  - an account, found by the agent and verified by intervention, whose frozen predictions hold on unseen bases.
    Its wording ceiling is "an answerability-associated (or account-named) computation on these templates" (§4.3).

**Prior, stated narrowly.**
- **What §49 found.** The ungated pushes in §49, at their evaluated settings, released most held false abstentions,
  each at a cost in unsupported answers:
  - decoded-city push 0.903 (city named on 144/144 genuine abstentions);
  - §38 push 0.927 (unsupported 0.188);
  - no-"unknown" instruction 1.000 (unsupported 0.451).
- **The other §49 finding.** F16 also records a template-limited sufficiency probe (AUROC 0.999).
- **What a weight edit might still do.** It could improve the trade-off without classifying whether evidence is
  present, for example by changing how the instruction is handled.
- **Expectation.** We expect an observed trade-off more often than a clearly selective edit. That is an expectation,
  not a claim about weight edits in general.

**Prior art and the boundary.** These are hypotheses about the closest work, to be re-checked from primary pages at
rev 1.

| Work | What it covers | Boundary for §51 |
|---|---|---|
| Takehi, Teoh, Zhang & Khattab 2026, *Sparks of Weight-Space Programming* (blog/report, https://rikiyat.github.io/blog/weight-space-programming/; Exp 8 logs at github.com/RikiyaT/lm_understand-logs) | GPT-6 Astra edits Qwen3.5-0.8B with a weight kernel and a metered forward-pass tool. The final p50c edit is a *scaled* sparse update (α 6 in the released report) against a looping direction, on 64 `down_proj` columns in each of two layers, with controlled layer, dose and random comparisons. The Gemma 3 270M follow-up moved MATH-500 by at most about a point | Different behaviour (looping vs epistemic abstention) and editor (Claude). Our transplant, clamp and separating-intervention tests are additions. Not peer-reviewed |
| Arditi et al. 2024 (arXiv:2406.11717) | Safety refusal is mediated by one direction; weight orthogonalization removes it | Safety refusal, not "unknown". B5 is an *adaptation*, and nothing establishes a shared representation |
| Lavi, Milo & Geva 2025 (arXiv:2509.22449) | An unanswerability direction steers abstention in extractive QA | Its evaluations label responses as abstention vs attempted answer; they do not establish that released answers are factually correct. Framing rule: a refusal direction is not an answerability direction |
| Commit–Abstain Circuit (arXiv:2609.32964v2) | A commit vs abstain circuit across ten models from five families, including Qwen 3.5 4B and 9B (and 35B in an extension) | Not our exact setting (Qwen2.5-3B, evidence in view, note-then-unknown). Its stated limitation concerns incorrect answers to answerable questions; the factual grading of released answers is the boundary |
| Liu, Wang, Wang, Xiao & Lin 2026, *Broken Symmetry in LLM Refusal: Answer Release Is More Local Than Refusal Restoration* (arXiv:2608.15772; EMNLP 2026 main); Ling et al., *LLM Abstention Can Be a Prompt Artifact, in Addition to Genuine Uncertainty* (arXiv:2507.16199; EMNLP 2026 main) | The answer is recoverable after refusal; a single-position patch releases it; a later-layer override follows the "Unknown" option | Predicts that a late-layer edit releases answers: the expected, non-novel part |
| Kumaran, Daw, Osindero, Veličković & Pătrăucean 2026, Nature MI (arXiv:2603.22161) | Confidence and a threshold drive abstention; steering moves it | Closed-book |
| Ruscio & Thompson 2026, *Tip of the Tongue* (arXiv:2603.13911) | Late-MLP success-to-failure transfer in Qwen2.5-3B | Closed-book |

## 1. Questions

- **R1 (repair).** On sealed fresh content, does the selected candidate E\* give a better release-versus-unsupported
  trade-off than each frozen comparator? "Better" is tested in two orientations (§7.2):
  - higher release at non-inferior unsupported answering;
  - or lower unsupported answering at non-inferior release.

  Damage and degeneration are reported against fixed bounds.
- **M1 (pathway).** For a reproducible effect of the selected M candidate, does reverting its one pre-selected component
  remove the effect, and does the effect exceed a matched random edit? This tests the *edit's* pathway, not the
  original failure.
- **M2 (causal path).** On the exact shared token prefix, does the effect run through the named activation? Shown by
  transplant plus clamp.
- **M3 (account).** Do the account's frozen predictions hold on unseen bases?
  - the unedited model's activity under the specified manipulations;
  - a population interaction that a matched output bias does not produce;
  - an increment in predicting refusals.

R and M are independent branches with separate error-control scopes. No study-wide 5% guarantee is claimed (§7).

## 2. Model, task and inherited measurement rules

**Model and inference.**
- Qwen2.5-3B-Instruct, at a checkpoint revision frozen at build (the same snapshot as §49/§50, recorded with its hash),
  fp32, greedy decoding.
- The F2 memory-note format (Chain-of-Note), on the controlled city tasks:
  - DE: "Where does A live?";
  - RC: "In which city does A's supervisor work?";
  - TU: "Where does A live now?".
- `max_new_tokens` is 160 for a full F2 generation (as in §49/§50) and 16 for answer-only generation from a fixed prefix
  (`MAX_NEW_RELEASE`).

**Inherited rules.**
- **The note is a prompted note.** If it names the gold, the content was written down. That does not show an internal
  derivation was completed.
- **Elicited sufficiency is not used** (F17).
- **Construction labels are a diagnostic, not semantic validation** (§4.3).
- **The answer-line marker** is a newline followed by the token sequence of "Answer:", as the model generates it.
  - **B0's designated marker** is the *last* such marker in B0's full output.
  - **P_ans** is the position of that marker's final token (the ":" token). Its output predicts the first answer token.
  - P_ans is the common read site for every class. A margin read at the first city the model produces is descriptive
    only.
  - M_city at P_ans is "city preference before answer generation". It depends on the gold answer, so it is an oracle
    covariate, not a deployable signal.
- **Exclusions.** The §49 and §50 rows are excluded from every split, by exact prompt hash, and their core tuples are
  disclosed (§3).

## 3. Data

**Generator.** An extension of `scripts/exp50_cases.py`, reviewed at build:
- seeds 51 (E, S), 52 (H) and 53 (T);
- counterfactual twins;
- T-city pools.

**Rows per base: 13.**

| Rows | Count |
|---|---|
| ANS rep a and rep b × H0–H2 | 6 |
| UNA-rel × H0–H2 | 3 |
| UNA-swap at H1 | 1 |
| CF (evidence city changed) × H0–H2 | 3 |

**Content groups and closure.**
- **Canonical key:** the §50 core tuple (task, A, B for RC, base gold, old city), i.e. `key()` in `exp50_cases.py`.
- **Closure:** every variant of a base (replicates, hardness, UNA-rel, UNA-swap, CF, nuisance variants) inherits the
  base's split.
- **Assignment:** bases are assigned to splits by key. No key appears in two splits, and no exact memory text appears
  in two splits.
- **Partial overlap** (a shared person–city pair under different keys, shared template instances) is counted and
  disclosed, not removed.
- **Old rows:** excluded by exact prompt hash. Core-tuple overlap with §49/§50 is counted and disclosed.

**Splits.**

| Split | Content | Who sees it | Use |
|---|---|---|---|
| E | Seed 51 keys | Editor (curated workspace) | Exploration |
| S | Seed 51, disjoint keys | Analysts only | Doses, selection, freezing, predictor fitting |
| H | Seed 52 | Sealed until the pre-H freeze; read once | Confirmatory R and M |
| T-city | Seed 53, new single-token cities outside the 12-pool (tokenizer-verified) | Sealed; read once | Separate transfer report |
| T-format | Seed 53, one re-worded F2 instruction | Sealed; read once | Separate transfer report |
| LongMemEval 165 (§35), LoCoMo §40 rows | Previously read | — | Secondary, descriptive, lowest priority in the inventory |

- **What may be done to sealed sets before reading:** mechanical validation (schema, tokenization, hashes, overlap
  checks).
- **What may not:** any inference, baseline generation, or analyst inspection of H or T outputs before the pre-H freeze.

**Sizes (fixed before exploration, from timing, never from H outcomes).**
- **Base-level support from §50.** Joint-population false-abstention *bases* out of 48 held bases per task were DE 48,
  RC 30 and TU 9, i.e. rates of about 1.0, 0.63 and 0.19 per base.
- **To reach ≥ 20 FA bases per task with margin:**
  - DE and RC need about 32–40 bases per task;
  - TU would need about 110.
- **Plan:**
  - **H:** 48 bases per task. DE and RC are expected to be supported; TU is expected to be underpowered, and is reported
    without a state.
  - **S:** 32 per task.
  - **E:** 48 per task.
  - **T-city and T-format:** 24 per task each.
- If timing forces smaller sets, they are reduced uniformly and disclosed. H is never topped up after reading.
- **Minimum support:** 20 FA bases per task for any stated R or M state, on S for entry and on H for claims.

## 4. Measures

### 4.1 Populations, fixed from B0 under each estimand

| Population | Definition |
|---|---|
| **FA** | ANS rows where B0's answer span is graded abstention |
| **FA-ab** | FA rows whose B0 note is answer-bearing (§50 rule, 088af4a) |
| **C** | ANS rows B0 answers with gold |
| **W** | ANS rows B0 answers wrong |
| **UNA** | All unanswerable rows |
| **CF** | Counterfactual rows |
| **NM** | Rows where B0 has no answer-line marker. Reported separately; excluded from P_ans analyses |

Every arm is evaluated on these same rows.

### 4.2 Grader (frozen, with fixtures, before E is released)

**Answer span.** The text after B0's designated marker, or, for any arm's own generation, after the *last* marker in
that generation, to the end. The note is scored separately (answer-bearing rule).
- **No marker** → no-answer-line.
- **Repeated markers:** only the last span is graded; earlier spans count as note text.

**Step 1: mark every city mention in the span.** Each mention of a pool city, T-city, alias or other place is tagged by
frozen cue lists, applied within its clause:
- **asserted:** a bare city, "lives in X", "X.";
- **suggested:** "possibly / maybe / probably / might be X";
- **negated:** "not X", "X, not Y" (Y negated, X asserted), "never X";
- **doubted mention:** "I don't know whether X", "the memory does not establish X or Y";
- **quoted:** a mention inside quotation marks that repeats the memory or question.

Only asserted and suggested mentions count as an answer.

**Step 2: classify the span.**

| Asserted or suggested cities (A) | Abstention phrase present (P) | Class |
|---|---|---|
| 0 | yes | **abstention** |
| 0 | no; no content | **degenerate** (also repetition, or empty after stripping) |
| 0 | no; other text | **other content** |
| exactly 1 | no | That city: **gold** (via the alias list; T-city golds in T-city), **wrong pool city**, or **other content** (a non-pool place) |
| exactly 1 | yes | **mixed** |
| ≥ 2 distinct | — | **conflicting** |

- A city followed by an explanation is classed by the city.

**Fixtures (frozen, all must pass before E is released).**
- "York, not Leeds" → York asserted.
- "The memory does not establish Leeds or York." → abstention.
- "I don't know whether Leeds." → abstention.
- "unknown, possibly Leeds" → mixed.
- "Leeds. Or York." → conflicting.
- "Leeds — the note says she moved there in 2023." → Leeds.
- "'Leeds' is mentioned, but not for her." → abstention.
- A missing marker, and repeated markers.
- An alias of the gold.
- A non-pool city.
- A T-city gold.

**Utility, per row, by class:**

| Row type | Gold | Abstention | Mixed | Wrong / conflicting / other | Degenerate / no-answer-line |
|---|---|---|---|---|---|
| ANS (and CF) | +1 | 0 | 0 (not released) | −λ | −λ |
| UNA | n/a | 0 | −λ | −λ | −λ |

- **λ:** 1 is primary; 2 is reported beside it.
- **U (mixture):** each base has weight 1, with 0.5 to ANS (each of its 6 rows 1/12) and 0.5 to UNA (each of its 4 rows
  1/8). CF is reported separately.

**Rates.**

| Rate | Definition |
|---|---|
| Rel | Gold share on FA (and on FA-ab) |
| Uns | Share of UNA rows classed wrong, conflicting, other content or mixed, i.e. an asserted or suggested answer. A mere mention does not count |
| Dam | Non-gold share on C |
| CF-follow | New-city share on CF |
| Deg | Degenerate plus no-answer-line share |
| Sel | Rel − Uns: a summary index, not a test |

**Wording audit.** n = 40, stratified by arm group × changed vs unchanged relative to B0 × ANS vs UNA.
- Proportional allocation, at least 2 per non-empty stratum.
- Drawn with `numpy.random.default_rng(5101)` after H's outputs exist.
- The rater is blind to the arm; adjudication follows §4.2.
- Disagreements are reported. The grader is never changed after results.

### 4.3 Construction labels and the nuisance diagnostic

- **Construction AUROC.** Separating ANS from UNA is necessary for any claimed answerability signal, but not
  sufficient. It says nothing about confidence.
- **Nuisance baseline.** A logistic model on features that *exclude* the answerability-defining relation:
  - task, hardness, template id;
  - the number and relation types of distractor lines;
  - memory length;
  - the position of the asked person's first line.

  Encoding: one-hot categoricals, standardisation from S. It is fitted on S and reported on H.
- **If the nuisance AUROC is ≥ 0.95,** that is a dataset limitation. The account is then tested on the matched
  contrasts ANS vs UNA-rel and UNA-rel vs UNA-swap.
- **Invariance depends on the account.** Each account lists its activity predictions before the pre-H freeze, in the
  form of §6.3(a).

## 5. Arms

### 5.1 Estimands (never compared across) and exact prefixes

**Total effect (primary for R).** The arm generates the full note and answer from the prompt token ids.

**Controlled prefix (primary for M).**
- The prefix is B0's **stored token ids**: the prompt, plus B0's generated ids through the designated marker's ":"
  token. Nothing is re-tokenised from decoded text.
- The KV over the prefix is recomputed under the arm, then the answer is generated from P_ans
  (≤ `MAX_NEW_RELEASE` tokens).

**Hybrid answer-site (secondary).**
- The KV for positions ≤ P_ans − 1 comes from the unedited model on the same stored ids.
- P_ans and every later position are computed under the arm's weights.

### 5.2 Comparators (settings frozen at build)

| Id | Arm | Specification |
|---|---|---|
| B0 | Unedited | — |
| B1 | Logit bias against "unknown" | Token set = every vocabulary token whose decoded, lower-cased, stripped form begins "unknown", printed and frozen. **Online trigger:** in total-effect generation the bias switches on after the first generated answer-line marker and stays on to the end (a second marker changes nothing); with no marker it never switches on. In controlled-prefix generation it is on from P_ans. A whole-generation variant is reported beside it. R grid {−1, −2, −4, −8, −16, −∞} |
| B2 | §35 prompt | Total: "Do not reply unknown." appended to the instruction. Controlled prefix: B0's stored note ids, then the ids of "\nDo not reply unknown.\nAnswer:"; P_ans is that new marker's ":" (role alignment, as in §44) |
| B3 | §49 prompt | Total: the "unknown" option removed from the instruction (§49 wording). Controlled prefix: the modified instruction ids, plus B0's stored note ids, plus the marker; positions aligned by role |
| B4 | §38 steering | Layer 29, α grid of ≤ 4. Applied at the answer-line marker's ":" (online trigger as for B1 in total effect; P_ans in controlled prefix) and at every later generated token. Never at other colons such as dates |
| B5 | Arditi-style orthogonalization (adaptation) | Direction = mean difference at P_ans between B0-abstained and B0-answered ANS rows on E, frozen. Orthogonalized out of `o_proj` and `down_proj` in a layer set frozen from E; ≤ 4 doses |
| B6 | Random edit | Matched to the candidate's edited entries, per component: per column (Frobenius) for matrices; L2 on the same entries for RMSNorm vectors; Gaussian directions; seeds 1–5. The contrast uses the per-row mean over the 5 seeds; the seed spread is reported. Magnitude matching does not guarantee equal functional disruption |

**Caveat.** B2 and B3 are prompt interventions. Their prefixes differ from B0's, so they are not activation
interventions on an identical prefix.

### 5.3 Dose and selection rules (all on S, none on H)

- **R doses:** each R arm (B1–B5 and the R candidate) takes the dose with maximum U(λ = 1) on S; ties go to the lower
  dose.
- **M doses:** the M candidate, and any f-intervention, take the dose with the largest effect in the account's
  pre-declared *signed* direction, on its pre-declared endpoint population, subject to Deg ≤ 0.10; ties go to the lower
  dose.
- **M comparator:** B1 is matched by §6.3(b).
- **R candidate selection:** among the submissions meeting R entry (§8.1), the highest U(λ = 1) on S. Ties go to fewer
  edited parameters. If none meets entry, there is no R candidate.
- **M candidate selection:** among the submissions meeting M entry (§8.2), the largest standardised effect (effect
  divided by its bootstrap SE on S) on its declared endpoint. Ties go to fewer components. Insufficient support (< 20
  bases in the endpoint population) means not eligible.
- **The two choices are independent:** the R and M candidates may be the same submission or different ones.

### 5.4 The editor and candidates

**Editor.**
- One fresh Claude session, with the model and version recorded at the start.
- It is not the Advisor or the Executor, and has no repo, no records and no S/H/T files.
- It works in a curated workspace (read-only E, enforced by mounts and sandbox).

**Initial instruction.** The text is frozen before E access. It does not mention sufficiency, confidence, B/C,
steering or our results.
- It asks for up to 3 submissions. At least one should aim at a correct-answer repair. One *may* be a diagnostic edit
  that changes the behaviour in a way the editor can explain, even if it is not a repair.
- Before testing any edit, the editor writes down:
  - (a) the component and the hypothesised computation;
  - (b) its evidence;
  - (c) numeric predictions for each row type;
  - (d) which unedited rows should flip and which should not;
  - (e) falsifiable predictions about the unedited model's activity under manipulations it specifies.
- For each submission it also gives a signed endpoint (population and direction) and its single most important
  component.
- It explains every step before each code block.

**Components and the edit family.**
- **One component** is one (layer, parameter tensor) pair:
  - `o_proj` of layer l;
  - `down_proj` of layer l;
  - `input_layernorm` of layer l;
  - `post_attention_layernorm` of layer l.

  Each counts separately.
- A candidate has at most **4 components**. Within one component the edit is *either* a rank ≤ 4 dense update *or* a
  sparse row or column edit. Sparse edits total ≤ 512 rows or columns per candidate.
- **Refused:**
  - q/k/v, gate and up projections, the embedding, `lm_head` and the final norm;
  - scaling a whole tensor;
  - new parameters or biases;
  - the "unknown" tokens' embedding or unembedding rows.

**Analysis.**
- **Allowed:** means, covariances, projections, PCA/SVD, and closed-form solves computed once.
- **Forbidden:** autograd, optimisers, and iterative loss-based fitting.

**Search bounds** (final values fixed after timing, before exploration):

| Bound | Value |
|---|---|
| Forward passes | ≤ 30. A paired pass is ≤ 64 pairs (128 sequences); fixed-prefix ≤ 256 rows; teacher-forced read ≤ 512 texts |
| Tokens | ≤ 2.0M input, ≤ 0.4M generated |
| Captures on disk | ≤ 20 GB |
| Python runs | ≤ 100 |
| Turns | ≤ 120 |
| Wall-clock | ≤ 48 h |
| Saved edits | ≤ 40 |
| Retries | ≤ 10 failed requests, not counted as passes |

- **Ledger.** Failed requests are not scientific passes, but their tokens and time count in the resource ledger.
- **One run only.**
  - A tooling dry run uses a dummy model and synthetic responses.
  - Any editor turn on the real model or E counts inside the run.
- **Explanations.** They are recorded verbatim. They are hypotheses, scored (§6.4), and never used for selection.

## 6. Mechanism tests (branch M; controlled prefix; account frozen before H)

### 6.1 M1: pathway

**Pre-selected on S, before the freeze:** the M candidate's single primary component, which is the component whose
revert removes the most of the endpoint effect on S (ties go to the editor's named component), and its signed
endpoint.

**Elementary tests on H:**
- **M1a:** E\* minus E\*-with-that-component-reverted, on the endpoint, one-sided in the declared direction.
- **M1b:** E\* minus B6 (seed-mean), on the endpoint, one-sided.

**Reporting.** Isolate effects, other components' reverts and the dose curve are descriptive. There are no causal
shares, no additivity claim and no monotonicity criterion.

### 6.2 M2: causal path

- **Prefix:** the stored B0 ids (§5.1).
- **Hook:** at the primary component's output, at the account-named prefix positions frozen before H (default: P_ans
  only). During answer generation the hook is applied at those positions only; it is applied at later generated
  positions only if the account names them before the freeze.
- **Primary endpoint:** the graded answer-span outcome on the endpoint population (for example Rel on FA). The lexical
  "unknown" log-odds at P_ans is descriptive.

**Elementary tests on H, both one-sided in the declared direction:**
- **M2a, transplant:** (unedited + Δh) minus B0.
- **M2b, clamp:** E\* minus (E\* with the component clamped to its unedited output).

A dynamic intervention on divergent histories is exploratory.

### 6.3 M3: account

The variable f is the quantity the account says the component computes, on the unedited model, at P_ans or the stated
positions.

**(a) Activity predictions, at most 4, fixed before the freeze.** Each is either:
- **directional:** the mean Δf under a manipulation (evidence removed, city changed, distractor changed, UNA-rel vs
  UNA-swap) has a stated sign, one-sided paired test;
- **or equivalence:** |Δf| below a margin set to 2 × its S-estimated SD, tested with TOST at α 0.05.

They are tested on the shortcut-breaking contrasts if the nuisance AUROC is ≥ 0.95.

**(b) Separating intervention.**
- Set f in the unedited model through its frozen direction or site, at the §5.3 M dose.
- **Comparator:** B1 at the signed-grid value (8 values each side, fixed) whose effect on the endpoint, on S, is closest
  to the f-intervention's.
- **Test:** the predicted *interaction*, (f-int − B1) on FA minus (f-int − B1) on UNA, in the declared direction, by
  paired base bootstrap.
- Significance in one population and non-significance in the other is not used.
- On H, both arms' realised endpoint effects and their difference are reported. The comparison is at S-matched, not
  H-matched, release. There is no retuning, and an unmatchable effect is a limitation of the comparator.

**(c) Frozen prediction.**
- **Models, fitted on S:** logistic regressions with an L2 penalty, C = 1.0, one-hot categoricals, standardisation from
  S:
  - the baseline (task, hardness, city one-hot, M_city at P_ans as an oracle covariate, the lexical "unknown" log-odds
    at P_ans);
  - full = baseline + f.
- **Population:** pooled across tasks, as the FA vs answered-gold rows of ANS (NM excluded).
- **Test on H, without refitting:** the paired AUROC difference (full − baseline), one-sided, with the bootstrap
  stratified by task.
- **Edge cases:** cells with one class are counted and excluded; there is no refitting or cell redefinition on H.

**M3** is an intersection–union of its elementary tests: each activity prediction in (a), (b) and (c).

### 6.4 Scoring the editor (descriptive)

The editor's pre-test predictions (§5.4 c–e) are scored against outcomes on E, S and H.

## 7. Inference

### 7.1 Elementary p-values

**Bootstrap.**
- A paired, base-cluster bootstrap, stratified by task, with B = 10,000 and seed 5102.
- Statistics are pooled across tasks with the §4.2 base weights. Task-specific values are descriptive.

**Tests.** For a one-sided null H0: θ ≤ θ0, with the alternative θ > θ0:
- p = (1 + #{b : θ\*_b ≤ θ0}) / (B + 1), on the percentile bootstrap of the paired statistic.
- The mirror form is used for H0: θ ≥ θ0.
- Reported one-sided 97.5% bounds are the matching percentile bounds, so a bound beyond θ0 corresponds to p ≤ 0.025.

**Failed replicates.** A replicate is failed if a denominator is empty or an AUROC is undefined.
- Failed replicates are dropped, counted and reported.
- If more than 1% fail, the statistic gets no state.

**Equivalence (TOST).** The two one-sided p-values; the larger one is used.

**Intersection–union.** For a fixed set of elementary tests, the composite p is the maximum p, with no internal search.

### 7.2 Family R (its own error-control scope, FWER 0.05)

For each comparator k ∈ {B1, …, B5}, two orientations:
- **O1 (higher release):** an intersection–union of
  - superiority, ΔRel > 0;
  - non-inferiority, ΔUns < +0.05.
- **O2 (lower unsupported answering):** an intersection–union of
  - superiority, ΔUns < 0;
  - non-inferiority, ΔRel > −0.05.

**Holm runs across all 10 hypotheses.**
- "Better trade-off than k" means O1 or O2 is rejected for k under Holm.
- "Better than every comparator" needs one rejection for each k.
- O2 protects against the ceiling problem: §49's no-"unknown" continuation released 1.000. If B3 reaches a release
  ceiling, O1 against B3 is unattainable, and a selective candidate can still pass O2.

**Brute-force equivalence to B1.** TOST on ΔRel and ΔUns, each within ±0.05.

### 7.3 Family M (its own scope, FWER 0.05)

- M1 = max(p_M1a, p_M1b).
- M2 = max(p_M2a, p_M2b).
- M3 = the maximum over its elementary tests.

Holm runs across M1–M3.

**The R and M families are separate questions.** We do not claim a single study-wide 5% guarantee.

### 7.4 Conventions

The 0.05 margins, the 0.95 ceiling, the 20-base minimum, the 2-SD equivalence margins and the 0.10 Deg cap for M doses
are operational conventions, stated as such.

## 8. Entry, decision criteria and stopping rules

### 8.1 Branch R

**Entry, on S.** A submission has U(λ = 1) − B0 with its one-sided 97.5% lower bound > 0 under the total effect, with
≥ 20 FA bases in DE or RC.

**Outcomes on H:**

| Outcome | Criteria |
|---|---|
| **Repair** | U − B0 lower bound > 0; Dam point estimate ≤ +0.05 vs B0 (upper bound reported); Deg ≤ 0.02; CF-follow point estimate ≥ 0.90. T-city and T-format reported separately |
| **Better trade-off than k** | Repair, plus O1 or O2 rejected for k under Holm (§7.2) |
| **Better than every comparator** | Repair, plus a rejection for each of B1–B5 |
| **Brute-force equivalent** | §7.2 TOST against B1 |
| **Observed trade-off** | Rel and Uns both rise; both effects reported |

### 8.2 Branch M

**Entry, on S.** A submission's declared signed endpoint effect is ≥ 0.05 in absolute value, with its one-sided 97.5%
bound beyond 0, and ≥ 20 bases in the endpoint population. The effect may be helpful, non-selective or harmful.

**Outcomes on H:**

| Outcome | Criteria |
|---|---|
| **Pathway located** | M1 rejected under Holm |
| **Causal path through the activation** | M2 rejected |
| **Account supported on these templates** | M1, M2 and M3 rejected, under the wording ceiling of §4.3 |

### 8.3 Negatives, worded by what was tested

- **No candidate passes anywhere:** "no qualifying candidate found by this bounded search (budget …)".
- **Trade-off observed:** both effects, with doses and CIs.
- **Failed or underpowered M tests:** reported with their precision.

### 8.4 Stopping rules

1. **Stage 1:** unexplained no-op mismatches, peak RSS too large, or timing that cannot fit 10 passes → stop and re-plan
   with the user.
2. **Exploration:** at the end of the run, the harness (not the editor) replays every submission on all of E. If none
   meets the R or M entry rule computed on E → document it. S, H and T stay sealed.
3. **On S:** neither entry rule holds → no H read and no pod.
4. **Sealing:** H and T are each read once. No rescue variant and no second editor run (D-45).
5. **Spend:** hard stops as in §10.

## 9. Freeze schedule

| Freeze point | What is frozen | Record |
|---|---|---|
| **F0: before E access** | Checkpoint and tokenizer revisions with hashes; the editor's model and version; the instruction text; prompts and formats; `max_new_tokens`; generator rules and seeds; the partition key; the grader with its cue lists, alias list and fixtures; the comparator specifications and grids; the edit family; the search bounds; the allowed-analysis rule | Manifest hash committed |
| **F1: after timing, before exploration** | Split sizes; final search bounds; resource caps; the workload inventory | Manifest hash committed |
| **F2: before H/T are read** | The selected R and M candidates; the primary component; signed endpoints; doses; hooks and positions; the account's activity predictions and margins; the f definition; predictor encodings and coefficients; the matched B1 value; the elementary-test list; the audit seed | Manifest hash committed; H then run once |

Nothing frozen at F0–F2 changes afterwards. Any deviation is recorded as a deviation, never silently.

## 10. Feasibility, inventory and cost

**Precision and parity.**
- fp32 everywhere. Edits are applied bit-for-bit from a hashed manifest.
- **No-op gate:** a zero edit compared on the same path as B0, with every changed row investigated.
- **CPU↔pod replay:** 200 E rows, B0 plus one edit, before any pod evaluation of H. Every mismatch class is
  investigated; its pod time is in the quote.

**Laptop memory (16 cores, 23 GB RAM).**
- **Weights:** about 3.1B parameters, about 12.4 GB in fp32.
  - The tied embedding/output matrix is 151936 × 2048, about 1.245 GB, read-only.
  - The largest editable tensor is about 90 MB.
- **KV cache:** about 72 MiB per 1,024 tokens per sequence.
- **Logits:** about 622 MB per 1,024 positions at full vocabulary, so they are streamed, keeping only the requested
  positions.
- **Design:** disk-backed weights, lazy fp32 copies of edited tensors, one model in memory, streamed captures.
- **Peak:** about 16 GB is a hypothesis until peak RSS is measured on a full-size stub.

**Inventory and quote.**
- Before any quote: every split × row × arm × dose × seed × estimand × capture, each timed per row on its backend.
- **Secondary work last:** the transfer sets and the previously read natural-chat analyses are inventoried last, and
  are the first to drop.
- No cost figure is given here.

**Budget envelope.**
- **GPU:** about $5.8 remains of the $30 cap (D-40; spent ≈ $24.17).
  - Exploration and S are designed for the laptop at $0.
  - Any pod work (H/T, the replay check, or a GPU exploration if the laptop is too slow) needs an inventory-based
    quote, a hard stop, the user's approval, and the user's line in the Executor pane.
  - The cap needs raising only if the combined approved workload exceeds the remaining envelope.
  - One pod at a time, list-pods checked empty afterwards, D-22 smoke test, D-30 bundle test.
- **Editor:** a separate cost line, outside D-40. Either the user's Claude plan or the API under a user-set cap,
  authorised before the dry run.

**Human review.**
- Advisor: build review, S review and the F2 commit, the wording audit, the recount.
- Codex: proposal and source review.
- User: the gates.

## 11. Stages

| Stage | What | Cost | Gate |
|---|---|---|---|
| 0 | This proposal; Codex review of the changed specifications; user "Go" (D-72) and the §12 decisions | $0 | User |
| 1 | Laptop build (Executor): weight-edit layer; metered service (CPU fp32); curated workspace; evaluation runner; grader with fixtures; mechanism kit; generator extension and partition audit; no-op gate; peak RSS; per-pass timing; inventory; B5 fit on E. Advisor build review; F0 then F1 | $0 | §8.4.1 |
| 2 | Tooling dry run with a dummy model | Editor cap | Boundaries verified |
| 3 | One editor run on E; harness replay on E | $0 GPU; editor within its cap | §8.4.2 |
| 4 | S stage: arms at their grids; doses; R and M selection; F2 | $0 CPU | §8.1 / §8.2 entry |
| 5 | H and T read once: replay check, then all frozen arms | Inventory quote, hard stop, approvals | — |
| 6 | Recount, results review, records (owners) | $0 | User "Go" on record changes |

## 12. Decisions for the user

1. Open §51 under D-45 (proposed D-72)?
2. How the editor is paid (Claude plan or API) and its cap.
3. If the laptop cannot fit ≥ 10 passes in ≤ 48 h: accept a smaller search, or approve GPU exploration (and raise the
   cap only if the combined workload exceeds the envelope).

## 13. Rev 1a: build contract (Codex review of rev 1)

These amendments are binding. Where they differ from §1–§10, §13 wins.

### 13.1 Bootstrap accounting

- **Valid draws.** Failed replicates are dropped. Then:
  - p = (1 + #{tail}) / (B_valid + 1);
  - the 1% failure threshold is computed over the attempted B = 10,000;
  - the paired contrast and its bounds use the same valid draws.
- **These are approximate bootstrap tests** (percentile inversion), not exact Monte Carlo null tests. Holm controls
  multiplicity given valid elementary p-values; it does not repair bootstrap miscalibration. The p-value and its
  Holm-adjusted decision are authoritative; the bounds are reported beside them and agree only approximately at a
  boundary.
- **Degenerate distribution** (every valid draw equals θ̂):
  - p = 1/(B_valid + 1) if θ̂ lies on the alternative's side of θ0;
  - otherwise p = 1.
- **M selection ranking.** Each candidate's effect is first *oriented* by its declared sign, so beneficial and harmful
  effects both rank as positive. Candidates are then ranked by oriented effect divided by max(SE, 0.01).
  - Eligibility still needs an oriented effect ≥ 0.05 (§8.2).
  - Ties go to fewer components, then fewer changed parameters, then submission index.

### 13.2 Closure across counterfactual collisions

- **Effective key.** Every ANS and CF variant has one: (task, A, B for RC, the city stated in its evidence).
  - Bases are linked when any of their variants share an effective key.
  - Each connected group is a content group, and the whole group is assigned to one split.
  - The generator may instead resample a colliding base, with at most 1,000 attempts per base; then it aborts and
    reports.
- **Build steps.**
  - `key()` (currently a lambda inside `drop_overlap` in `exp50_cases.py`) is extracted into a reusable function.
  - Before any model run, a finite-pool feasibility check confirms the closed groups can meet the F1 split sizes. If
    not, sizes are reduced uniformly and disclosed; resampling never loops without bound.
- **Partial name/city overlap** that is not a full effective-key collision is counted and disclosed, as before.

### 13.3 M2 hooks stay on the shared prefix

- **Confirmatory M2 hooks** apply only at stored shared-prefix positions ≤ P_ans. The transplant and clamp values are
  computed on those same token ids and component outputs, and each arm then continues normally.
- **Hooks at later generated positions** belong to the exploratory dynamic analysis. The §6.2 clause allowing
  account-named later positions is withdrawn for M2.

### 13.4 M3 definitions

**(b) Interaction variable.**
- **Commit** = 1 if the answer span's class is gold, wrong pool city, other content, mixed or conflicting; 0 for
  abstention, degenerate or no-answer-line. It is defined on FA and on UNA.
- **Interaction:** [(f-int − B1) in Commit rate on FA] − [(f-int − B1) in Commit rate on UNA], in the declared
  direction.
- Rel and Uns are reported separately beside it.
- If the M candidate's declared population is not FA (for example C), the population pair for (b) is declared at F2,
  with UNA as the default second population.

**(a) Equivalence tolerance.**
- **Scaling statistic:** the SD across S bases of the per-base mean paired Δf for that manipulation.
- **Tolerance:** 2 × that SD, frozen at F2.
- **If the SD is 0:** the test becomes exact equality, with every |Δf| ≤ 1e-6 on H.
- **Wording:** "equivalent within the frozen tolerance". It is not evidence of near-invariance by itself. Each account
  justifies its tolerance when it is declared.

**(c) Prediction population.**
- The test uses the **pooled AUROC** on the fixed population (FA vs ANS rows answered with gold, NM excluded), and both
  classes must be present overall.
- Per-task AUROCs are descriptive.
- The §6.3(c) "one-class cell" exclusion is withdrawn. Cell pooling below remains an *encoding* rule only: baseline
  city × task cells with fewer than 5 rows of a class are encoded at task × hardness, frozen at F2.

### 13.5 Manifest and build fields

**Freeze order.**
- **F0, before the editor's access to E:** allowed maxima for every bound; a reduction rule (every bound × r, where
  r = min(1, 48 h divided by the projected time for 30 passes)); and everything else listed at F0 in §9.
- **Analyst work before F0 is allowed:** generating E/S data, B0 outputs on E and S, timing, and fitting B5 on E. No
  H/T outcomes are touched.
- **F1:** the final bound values (within F0's maxima) and the split sizes.

**Dose and candidate ties.**
- Dose ties go to the setting closest to no-op: smallest |bias| (so −1 before −∞), smallest α, smallest edit norm.
- Then the lexicographic arm-dose id decides.
- Remaining candidate ties go to submission index.

**RMSNorm.** An RMSNorm component may change at most 512 coordinates. Every changed coordinate counts against the
candidate's shared sparse budget of 512 rows, columns or coordinates in total.

**Grader.**
- **"Other place" recognizer is dropped.** Any non-pool, non-alias, non-abstention content is classed as "other
  content", with the same utility. Only gold aliases, pool cities and T-cities are recognised.
- **Quoted mention.** A quoted city counts as a quoted mention only if the quoted string is longer than the city name
  and matches a substring of the memory or question. A bare quoted answer ("Answer: 'York'") is asserted.
- **Repeats.** Repeated assertions of the same city count as one distinct city.
- **Fixtures added:**
  - `Answer: 'York'` → York;
  - quoted evidence followed by an actual answer → that answer;
  - "York. York." → York;
  - "York … actually Leeds" → conflicting.

**Audit arm groups (4).**
- B0;
- prompt baselines (B2, B3);
- logit, steering, weight and random baselines (B1, B4, B5, B6);
- candidates (R and M).

With 4 groups × changed/unchanged × ANS/UNA = 16 strata, two per non-empty stratum needs at most 32 of the 40 rows.

**Sources.** Codex verified the Liu (2608.15772v2) and Ling (2507.16199v10) identifiers against their primary records.
They support motivation, not a mechanism in our model.

## Appendix A. Resolutions

**Planning reviews.** Codex's review of plan rev 0 (12 points), its review of rev 1 (5 points) and its close-out of
rev 1a (3 details) are resolved in `/tmp/weight-space-programming-advisor-plan.md` §12–§13. Their content is
consolidated in §2–§10 above.

**Codex review of §51 rev 0 (`/tmp/exp51-rev0-codex-review.md`, 9 points).**

| # | Point | Resolution in rev 1 |
|---|---|---|
| 1 | O1 alone can be unattainable at a ceiling | O1 and O2 are both confirmatory per comparator, with Holm over 10 hypotheses; "better than k" = either orientation rejected (§7.2, §8.1) |
| 2 | Elementary tests; the M1 OR; M2 and M3 tests; family scopes | Bootstrap p-values defined (test inversion, the +1 rule, failed replicates); M1 fixed to one pre-selected component (revert) plus B6; M2 endpoint and two directional contrasts; M3 lists directional/TOST activity tests, an interaction test and the AUROC increment; IUT = max p; R and M have separate scopes (§6, §7) |
| 3 | Max-U dose vs harmful M; selection; "non-trivial"; E reproducibility | Separate R and M dose rules plus the B1 matching rule; selection rules with ties and support; the instruction allows a diagnostic submission; entry = \|effect\| ≥ 0.05 with a bound; E stopping uses a harness replay (§5.3, §5.4, §8) |
| 4 | Grader precedence vs assertion | Assertion/negation tagging first, then classification; Uns counts only asserted or suggested answers; fixtures listed; aliases, non-pool and T-city golds handled; missing and repeated markers (§4.2) |
| 5 | Exact prefixes; online B1 scope; §38 position; B2/B3 alignment; the M2 hook contradiction | Stored token ids; designated marker and P_ans defined; B1 and B4 online trigger after the first generated marker; B2/B3 role alignment, stated as prompt interventions; M2 hook positions frozen with the generation rule (§2, §5, §6.2) |
| 6 | Freeze schedule; partition key and closure; sizing from base counts; analysis manifest | F0/F1/F2 schedule with manifest hashes; key = the §50 core tuple, with closure; sizes from FA *base* rates (48/30/9 of 48), TU expected underpowered, no top-up; minimum support on S and H; pooling, encodings, edge cases (§3, §6.3, §7.1, §9) |
| 7 | CAC includes Qwen; Lavi wording; WSP scaling; exact references | CAC row corrected (Qwen 3.5 4B/9B/35B; boundary = exact setting and factual grading); Lavi row narrowed; WSP α 6 scaled update; titles and ids added (Liu arXiv:2608.15772, Ling arXiv:2507.16199, from search listings; to be re-read from primary pages at the build review) |
| 8 | Component definition; B6 for RMSNorm and seeds; ledger; GPU wording | Component = (layer, tensor), at most 4, one edit type each; B6 per-column and per-vector matching, seed-mean contrast; failed requests in the ledger; generation lengths fixed; GPU exploration needs a quote and approvals, with a cap raise only if exceeded (§5.2, §5.4, §10) |
| 9 | Scope | No design widening; secondary analyses inventoried last |

**Codex review of §51 rev 1 (`/tmp/exp51-rev1-codex-review.md`).** Narrow build-contract fixes, all carried into §13: bootstrap B_valid and the approximate-test wording; degenerate distributions and oriented ranking; closure across counterfactual effective-key collisions; M2 hooks on the shared prefix only; M3 Commit variable, SD definition and pooled AUROC; freeze order, ties, RMSNorm budget, grader fixtures, audit groups.
