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

## Rev 1 (2026-10-29, after the build review d1902f5 / stage 0; before any development run)

- §2 correction: R(0) ≈ 0 is an empirical check, not a construction (layer 0's output already attends over the
  context). Bound unchanged, |R(0)| < 0.05; stage 0 observed < 0.01 on rows with a real gap.
- §2 scope: sanity is asserted on patched rows inside the widest filter; recorded elsewhere. The 3B development run
  patches all rows (descriptive curves by class); the 7B and both confirmation runs patch eligible rows only.
- §4 C3 reading fixed: "agrees" = CI overlaps [0.85, 0.92] and median ≥ 0.75; "under-read" = upper bound < 0.75;
  else undetermined.
- §3 filter: strict (corrupt rank > 10) unless fewer than 40 of 126 development rows survive for that model, then
  relaxed (> 3), decided by the development run and recorded here as rev 2 before `FIXED_FILTER` is set. Stage 0's
  preview from the §34c reads: strict 31 / 39, relaxed 58 / 65 (3B / 7B).

## Rev 1 amendment (2026-10-29, adopted by the user; supersedes the "proposed" amendment)

**Thesis question adopted (D-45):** when a small model fails on a question whose evidence is in context, can a
failure of *evidence use* (the evidence does not reach the answer computation) be distinguished experimentally from a
failure of *answer production* (evidence-dependent information is available at the answer position but the response
decision suppresses or overrides it)? Rev 0 (clean → cut patching, C1–C3) is kept intact as the baseline and the lens
adjudication; this amendment adds the conditions that address the thesis question. Nothing else is added: no new
probes, no scaling, no routing (the user's scope rule). Models: Qwen 2.5 3B (primary), Llama 3.2 3B (the planned
contrast, already configured), Qwen 2.5 7B (rev 0 only, plus condition S if the 3B results are readable and time
allows). Laptop, $0.

### A. The swap subset (feasibility gate first, free, no model run)

**Definition (fixed now).** Rows of the §34 population (330 per family; 126 dev, 204 confirmation) where the gold
string (D1 normalisation) occurs verbatim in exactly one evidence line, and nowhere else in the prompt (not in the
question, not in a non-evidence memory line). **Substitute:** another row's gold of the same surface type (name /
date / number / other, by a fixed regex), with a different first token after `Answer:` under the family tokenizer,
within ±2 tokens of the gold's length, not occurring anywhere in the prompt; chosen deterministically (seed 44, first
match in fold-then-id order). The swap prompt replaces the gold string in that one evidence line with the substitute;
everything else is byte-identical.

**Exclusions (fixed now):** gold in zero or several evidence lines (aggregation and multi-hop rows are therefore out
by construction, and the subset is biased toward extractive single-hop questions; stated as a limit in §F); gold in
the question; no substitute satisfying the rules; alignment failure (`check_alignment` on clean vs swap: P1 last, same
token, identical question-to-P1 tail); a gold whose first token equals the substitute's first token.

**Feasibility gate (before any build of S/U):** the Executor counts the swap-eligible rows per family, per fold set
and per LongMemEval question type, from the data alone. **Proceed** if, for Qwen 3B, folds 2–4 hold ≥ 50 eligible rows
with at least two question types of ≥ 15 rows each, and the eligible rows contain ≥ 30 of the 3B's in-view failures
(§23 labels) on folds 2–4. If the failure count is between 15 and 29, the claims are read on all 330 rows (every
design choice is fixed here, no dev-fold choice is made for S/U, so pooling is legitimate) and the thinness is stated.
If the gate fails outright (< 50 eligible or < 15 failures), the result is **"the swap design cannot be read on
LongMemEval"**: recorded in EXPERIMENTS §44 and the programme stops here, per the user's stopping rule; no new
intervention is invented.

### B. Conditions and measures per swap row

- **clean** (gold G in the evidence), **swap** (substitute S in its place), **cut** (rev 0, no evidence). Forced
  P1 as §34. Per run: Δ = logit(G's first token) − logit(S's first token) at the final layer; also log p(G), log
  p(S), rank(G), rank(S), log p(unknown), top-1.
- **Evidence following (behavioural, no patch):** F = Δ_clean − Δ_swap. The row "follows the evidence" if F ≥ 1
  nat and Δ_clean > 0 > Δ_swap (the preferred first token flips with the evidence). Margin sensitivity reported at
  0.5, 1, 2 nats.
- **Per-row failure class (the central measurement), on the small model's in-view failures (§23 label refusal or
  wrong):** *answer-production failure* if the row follows the evidence (the answer computation tracks the evidence,
  the output does not); *evidence-use failure* if |F| < 1 nat (the evidence does not move the answer preference);
  *indeterminate* otherwise (F ≥ 1 without a sign flip). Successes are classified too, as the reference.
- **Forward patch (swap → clean):** overwrite the clean run's P1 residual at layer l with the swap run's;
  T_fwd(l) = (Δ_patched(l) − Δ_clean) / (Δ_swap − Δ_clean). **Reverse patch (clean → swap):** T_rev(l) =
  (Δ_patched(l) − Δ_swap) / (Δ_clean − Δ_swap). Bidirectional on every patched row (item 3). Transfer depth d_T =
  first l with T ≥ 0.5 over L; T(L−1) = 1 by construction (asserted); T(0) ≈ 0 empirical (bound 0.05).
- **Negative controls (every patched row):** (i) *mismatched donor*: the P1 residual of a different swap row's
  swap run at the same layer (carries a third answer); report Δ shift toward S, expected ≈ 0 at every layer; (ii)
  *self-overwrite*: the clean residual into the clean run, T = 0 by construction (sanity); (iii) *random vector* of
  the donor's norm at the same layer (seeded), reported like §38's Random. Rows are patched only if they follow
  the evidence (|F| ≥ 1 nat, sign flip): T is undefined otherwise.
- **Condition U (no "unknown" option), item 2:** the clean prompt with the instruction's "unknown" option removed
  (the §35 frozen-note wording, fixed), generated answers (max 64 new tokens, greedy) on the small model's in-view
  refusal rows of the swap subset and of the full population; the share that becomes gold (D1) is the
  **prompt-artefact share** (Ling et al. 2026). U is also run under the §38 Steer setting (layer 29, α 2, Qwen 3B
  only; L17 α 2 for Llama): Steer-under-U minus U alone is the push's contribution beyond the format. Both
  descriptive, pre-registered readings in §C.

### C. Claims and success criteria (confirmation folds 2–4, or all 330 under the §A pooling rule)

- **S0 (validity precondition).** Evidence-following rate among the small model's *successes* in the swap subset
  ≥ 0.80 (CI lower bound ≥ 0.70). If it fails, the swap manipulation does not control the answer on this prompt
  family and S1–S3 are not read; recorded as "swap not valid", stop.
- **S1 (separability, the thesis claim).** Among the small model's in-view failures in the swap subset: the share
  classified answer-production failure, with its 95% CI, and the indeterminate share. **A reliable distinction is
  established** if (a) the indeterminate share is ≤ 0.20, (b) the answer-production share changes by < 0.15 between
  the 0.5- and 2-nat margins, and (c) the CI half-width is ≤ 0.15. If (a)–(c) hold, the share itself is read:
  ≥ 0.50 "answer-production failures are the majority", ≤ 0.25 "evidence-use failures are the majority", else
  mixed. If any of (a)–(c) fails: **"not reliably distinguishable on these rows"**, documented, no new intervention.
- **S2 (the two classes differ on an independent measure).** Among refusal rows in the swap subset: the §38 push
  (Qwen 3B, layer 29 α 2; Llama L17 α 2) turns answer-production refusals into gold answers more often than
  evidence-use refusals: difference of proportions with CI, lower bound > 0. Prediction: yes for Qwen, undetermined
  for Llama (few refusals). Read only if each class has ≥ 8 refusal rows; else descriptive.
- **S3 (localisation).** Median forward transfer depth d_T with CI, beside rev 0's d_c and the lens depths, per
  model; the reverse depth beside it (asymmetry reported, Liu et al. 2026). Mismatched-donor Δ shift must stay
  within ±0.5 nat at every layer (else the transfer measure is confounded and S3 is not read). Descriptive readings:
  the depth at which evidence-specific content is transferable, compared across the two 3Bs.
- **U1 (prompt artefact).** Share of in-view refusals that become gold under U: ≥ 0.50 (LB > 0.40) → "the refusal
  disposition is mostly a prompt artefact; the §38 decision-failure claim is restated as format-induced"; ≤ 0.25
  (UB < 0.35) → "not the format"; else partial, with Steer-under-U − U reported.
- **Sample counts.** Expected (from the §43 dev profiles): Qwen 3B in-view failure rate ≈ 0.57, refusals ≈ 0.25;
  Llama 3B failures ≈ 0.44, refusals ≈ 0.10. With 50 eligible confirmation rows that is ≈ 28 Qwen failures and ≈ 12
  refusals, too few for S1(c) (half-width ≤ 0.15 needs ≈ 40 failures): the §A pooling rule exists for this reason,
  and S1(c) failing on thin rows is a legitimate "undetermined", not a reason to relax (c).

### D. Order and cost

1. Rev 0 development run continues unchanged (baseline; in progress).
2. Feasibility gate (§A) from the data, free; the Executor reports counts per family, fold set, question type and
   failure class **before** building S/U. The Advisor records proceed / pool / stop as rev 2.
3. Build S/U (clean, swap, cut, forward and reverse patches, three controls, U generation); build review; 10-row
   smoke of S0 on dev rows (sanity bounds, following rate) — the only dev-fold use, and it decides nothing.
4. Run on the pre-declared rows; report; results review; EXPERIMENTS §44; record changes.
   Laptop: Qwen 3B ≈ 3 passes × 2 patch sweeps + controls ≈ 3–4 min/row ≈ 3–4 h for 50 rows; Llama 3B similar; the
   7B ≈ 3× that if run. No pod.

### E. What the outcomes mean (fixed now)

| outcome | reading |
|---|---|
| S0 fails | the swap manipulation does not control the answer on this prompt family; the design is not readable here |
| S1 establishes a distinction, answer-production majority | the in-view failures of this model are mostly downstream of an evidence-sensitive computation (Qwen prediction) |
| S1 establishes a distinction, evidence-use majority | the failures are mostly upstream; the §38 push cannot help them (Llama prediction) |
| S1 not reliable | the two failure kinds are not separable with this manipulation at this sample; documented as the result |
| U1 ≥ 0.50 | the "decision failure" is a format effect; §38–§42 are restated as such |
| S3 forward transfer early, rev 0 d_c late | evidence-specific content is at P1 by mid-stack but is not the only thing the final layers need; both lenses partly right |

### F. Interpretation limits (fixed now)

The swap subset is biased toward extractive single-hop rows, so evidence-use failures on aggregation, temporal and
multi-hop questions are excluded by construction and the S1 share is an upper bound for answer-production failures
over all failures. One benchmark and prompt family; one forced position; first-token logit differences; a substitute
may be less plausible than the gold (the mismatched-donor control bounds this only for patching, not for F); the 7B
in bf16; the §38 push comparison in S2 reuses a setting chosen on development rows of §38, not of §44. Thirteen
Llama refusals in total make the Llama contrast descriptive whatever S1 says for it.

## Rev 2 (2026-10-29): swap-gate verdict and rulings on the gate build (63882c1, `runs/exp44_swapgate/`)

**Verdict: POOL for Qwen 3B.** Folds 2–4: 54 eligible (≥ 50), two types of ≥ 15 (single-session-assistant 16,
single-session-user 16), 27 in-view failures (15 refusals, 12 wrong): the 15–29 band. Claims S0–S3 and U1 are read
on all 330 rows: 90 eligible, 43 failures (21 refusals, 22 wrong), 47 successes. 43 failures give a binomial
half-width of ≈ 0.15 at p = 0.5, so S1(c) is at its edge; a failure of (c) is the documented outcome. No development
choice is made for S/U (the gate used labels only to count). **Llama 3B: descriptive only** (folds 2–4: 56 eligible
but 12 failures; pooled 95 / 24): S0 and the class shares are reported, S1–S2 are not read. **Qwen 7B:** the same 90
rows; condition S run after both 3Bs if time allows, descriptive beside. Counts recounted by the Advisor from
`eligible_qwen3b.jsonl` (90; 54 / 27 / 43 as reported).

Rulings on the builder's choices:
1. Verbatim = the gold's D1 tokens as a contiguous run of whole tokens: **accepted** (stricter than D1's substring
   branch; used for every absence check).
2. Substitute assignment: **changed.** One global shuffle concentrates the substitutes (19 distinct over 90 rows,
   "Nike" ×21, "8" ×18), which would let one substitute's own prior drive F and the mismatched-donor control. Rule:
   per-row draw with seed 44 + the row's index in (fold, id) order, walking that row's own shuffled candidate list,
   and a cap of 3 uses per substitute string across the family's eligible rows (round-robin: a candidate at its cap is
   skipped). Report the resulting distinct count and the maximum multiplicity.
3. Surface types: **tightened.** "number" splits into currency (leading $ or a currency word), percent, duration
   (number + time unit), bare number; substitutes must share the subtype. "other" substitutes are drawn from rows of
   the same LongMemEval question type with the same first-character capitalisation and ±1 token length. Implausible
   swaps that remain are what S0 is for.
4. Same leading-whitespace tokens as the gold: accepted.
5. First token = §34's tracked id; continuation-split as a consistency check only: accepted.
6. Every match in the one line replaced, affixes absorbed, outside the line byte-identical, alignment 0 failures:
   accepted.
7. Gate edge case (≥ 50 eligible, < 2 types of ≥ 15): **proceed with a type-diversity caveat**, not stop; the
   diversity condition is a reporting requirement. Did not arise.

After rulings 2 and 3 the eligible set is recomputed (it can only shrink by the "no substitute" exclusion); the
verdict stands if ≥ 85 eligible and ≥ 40 failures remain, else the Advisor re-reads the gate as rev 3.

## Rev 3 (2026-10-29): re-read of the recomputed gate (88b2a19) and one measurement correction

After rulings 2–3 the Qwen 3B swap set is 82 eligible / 39 failures (dev 32, folds 2–4 50; labels on 2–4: 24 / 14 /
12), under the rev 2 re-read bar (85 / 40). The loss is a measurement artefact, not a shortage of rows: under the
tighter subtypes every currency gold and its substitute start with the same "$" token, so the "different first token"
rule excluded them (Qwen lost $800, $2,000, $6 and $400,000; 7 first-token-equal exclusions against 4 in rev 1).

**Correction (applies uniformly to every row, fixed now).** Let the *shared prefix* be the longest common token
prefix of the gold and the substitute under the family tokenizer (after the leading-whitespace tokens). The
"different first token" rule becomes "different first token *after* the shared prefix", the shared prefix is
teacher-forced in every run of that row (clean, swap, cut, patched), and Δ, F, T and the ranks are read at the position
that predicts the first divergent token. For rows with an empty shared prefix (the large majority) nothing changes.
`check_alignment` extends to the forced prefix; the sanity bounds apply unchanged. The gate is recomputed once more
with this rule.

**Verdict rule for that recount: POOL regardless.** The 85 / 40 bar was a re-read trigger, not a stop; the stop
conditions remain §A's (< 50 eligible on folds 2–4 or < 15 failures), which are far from binding. The recounted
numbers are recorded here as rev 4 before `POOL_AUTHORISED` is set. With about 86 / 41 the S1(c) half-width sits at
its 0.15 edge; a (c) failure is the documented outcome, and no criterion moves.

Type diversity: on folds 2–4 only single-session-assistant keeps ≥ 15 rows; by ruling 7 the pooled read carries a
type-diversity caveat in the report.

### Rulings on the S/U build (dbd813e; 42 tests pass on the Advisor's machine)

1. U = §35 arm B exactly (cached note, the B instruction, `Answer:`, greedy 64 tokens; D1 on the phrase): **accepted**.
   U differs from the §38 arm A rows by the instruction only, which is the right contrast for the artefact share.
   No-note rows excluded and counted.
2. S2 push = §38 Steer on arm A, unsteered A beside; Steer-under-U at every step; vector hashes match: accepted.
3. Mismatched donor = nearest earlier swap row (cyclic) whose gold and substitute first ids both differ from the
   row's; patched into the clean run only; the ±0.5 nat bound on the mean shift per layer: **accepted**, with the
   per-row shifts kept in the rows for the review.
4. Random control seeded [44, row, layer] at the norm of h_swap(l): accepted.
5. Self-overwrite checked at layers 0, L//2, L−1: accepted.
6. Sanity |T(L−1) − 1| < 0.01, |T(0)| < 0.05, exit 3 on any patched row: accepted.
7. log p(unknown) = log-sum-exp over the two refusal ids, each logged: accepted.
8. Cap on D1-normalised strings, counted once eligible; unit-less numbers "bare": accepted.

Order: rev 0 7B finishes (laptop) → gate recount with the shared-prefix rule (data only, send counts) → rev 4 recorded,
`POOL_AUTHORISED` set by the Executor on the Advisor's word → 10-row S0 smoke → Qwen 3B S then U (≈ 4.3 h) → Llama 3B
(≈ 3.4 h) → report → results review.

## Rev 4 (2026-10-29): recount under the shared-prefix rule (0dcb155) — the fixed swap set; POOL authorised

**Qwen 3B: 89 eligible (dev 35, folds 2–4 54, all 89); 43 in-view failures (21 refusals, 22 wrong), 46 successes;
folds 2–4: 29 failures (16 / 13), 25 successes.** Recounted by the Advisor from `eligible_qwen3b.jsonl` (89 / 54 /
46-22-21; 71 distinct substitutes, maximum multiplicity 3; 12 rows with a non-empty shared prefix). Two types keep
≥ 15 rows on folds 2–4, so no diversity caveat. Exclusions: zero evidence lines 180, several 5, in the question 11,
elsewhere in the prompt 39, no substitute 6, first-token-equal 0, alignment 0. Seven rows gained against rev 2, none
lost; the substitute changed on 5 rows by the per-row draw. **Verdict: POOL, all 330 rows; `POOL_AUTHORISED` may be
set.** S1(c) at 43 failures: half-width ≈ 0.15 at p = 0.5, at the bar; documented outcome if it fails.

Prefix rows (12): " $" ×4 (currency), "6:" ×2 (clock times), a shared leading digit ×2 (Qwen splits digits), articles
×4 (" a" ×2, " The", " the"; the content word is what is read, which also resolves the §43 article artefact for these
rows). Llama 3B: 90 eligible, 24 failures (12 / 12), 66 successes, 9 prefix rows, descriptive only.

The swap set is now fixed. Any further change to the eligible rows, substitutes or prefixes after this point is a
protocol deviation and must be recorded as such.
