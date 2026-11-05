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

## Rev 5 (2026-10-29): rev 0 filter flaw — the gap requirement; R(0) becomes an exclusion, not a stop

**What happened.** The 3B development run finished (126/126, `--patch-all`, exit 0). The 7B stopped itself at row 11
of 126 (37f165cf, exit 3): clean rank 8 / corrupt rank 22, log p clean −16.58 against corrupt −14.88, R(0) = −0.110.
The §3 filter is rank-only: it never required the cut to *lower* the gold's log p. A row whose corrupt log p is above
its clean one has a negative gap, so R's denominator carries the wrong sign and the sanity bound is meaningless.

**Recount (Advisor, from `runs/exp44_patch/{qwen3b,qwen7b}/*.json`).** 3B: strict 30 / 126, relaxed 57 / 126; of the
57, five have a gap below 1 nat (71017276 −0.81, ac031881 −1.18, bbf86515 −1.09, gpt4_1e4a8aeb −3.91, gpt4_b0863698
−0.29), none in the strict 30; no sanity failure on any eligible row; |R(0)| ≤ 0.004 on eligible rows with gap ≥ 1.
The five 3B sanity failures in the log are all on rows outside the rank filter (gaps −0.36 to 1.85). 7B, 11 rows:
strict 4, relaxed 8; two under 1 nat (22d2cb42 −9.97, ranks 7 / 4; 37f165cf −1.70, ranks 8 / 22). **The Executor's
note that the strict filter rules this out is wrong: 37f165cf is a strict row.** Rank order and log p order disagree
when the two distributions differ in peakedness, so the gap has to be required explicitly on both filters.

Layer-0 patches move the gold's log p by a median 0.008 nat on the 3B, but up to 0.126 nat (6 of 126 rows above
0.05 nat); on the 7B a row with a real gap already reached |R(0)| = 0.043. The bound |R(0)| < 0.05 is empirical (rev 1
correction) and divides a fixed-size shift by the row's gap, so a legitimate row with a 1–3 nat gap can fail it. An
empirical bound must not halt a five-hour run.

**Rulings.**
1. **Gap requirement, both filters.** Evidence-dependent now means: clean final gold rank ≤ 10, corrupt final gold
   rank > k (k = 10 strict, 3 relaxed) **and lp_clean − lp_corrupt ≥ 1 nat**. The Executor's 1-nat margin is accepted
   over "corrupt rank > clean rank": rank alone cannot separate a negligible gap from a real one, and R needs a
   denominator the layer-0 noise cannot dominate. Rows inside the rank filter with gap < 1 nat are recorded, never
   asserted, never in a claim; their count is reported per model and fold set.
2. **R(L−1) stays a hard assertion** (|R(L−1) − 1| < 0.01, exit 3): it is a construction, and a failure is a bug.
3. **R(0) becomes a recorded exclusion.** |R(0)| ≥ 0.05 on an eligible row no longer stops the run. The row is kept on
   disk, flagged, excluded from the claim statistics, and the count of such rows is reported beside every claim; the
   report also gives each claim's headline with them included as a robustness line. If more than 10 % of a model's
   eligible rows are flagged the result is reviewed before any confirmation read.
4. **3B filter fixed:** strict 30 < 40, so the 3B reads the **relaxed filter with the gap requirement: 52 development
   rows**. Re-read of the 3B development rows under the amended filter is free (all 126 are patched) and is ordered.
5. **7B filter** is decided when its development run finishes (strict 4 of 11 so far projects to about 46 of 126, so
   the 7B may stay strict; that is the pre-declared per-model rule, and C1–C2 are 7B-only, C3 is 3B-only, so the two
   models need not share a filter). The 7B resumes keeping its 11 rows.
6. `FIXED_FILTER` and `CONFIRM_AUTHORISED` stay off until rulings 1–5 are built, the 3B dev report is re-read, the 7B
   dev report lands with its counts, and the Advisor records the 7B filter (rev 6). No confirmation fold is read before
   that.

This change was made on development rows only (folds 0–1); no confirmation row has been read by anyone. The
development d_c medians were seen during the recount and did not enter the choice of margin.

## Rev 6 (2026-10-29): the S3 negative controls are mis-scaled — difference-matched controls replace them

**What the S0 smoke showed** (Qwen 3B, 10 development rows, `runs/exp44_swap/smoke/qwen3b/`, decides nothing; Advisor
recount of the 7 patched rows). The forward patch at layer 0 shifts Δ by ≤ 0.006 nat on every row, and its median
shift across layers is ≤ 0.2 nat: the swap residual is a near copy of the clean one until the content arrives. The
two controls are not: the mismatched donor's median shift toward S per row runs from −2.7 to +17.6 nat and the random
overwrite's from −1.0 to +21.9 nat, at every layer including layer 0. From the stored residuals, the swap-minus-clean
difference at layer 18 has norm 0.2–8.6 against a residual norm of about 55 (0.3–13 %), while overwriting with a
donor row's residual moves the state by 10–68. The controls replace the whole P1 state with a foreign one and so
measure generic disruption at ten to a hundred times the scale of the real patch. The ±0.5 nat bound of §B cannot be
met by construction; on two rows the donor's shift (16.8, 17.6 nat) approaches the forward shift at d_T (23.3, 20.7),
so a full-state overwrite could by itself clear T ≥ 0.5. The §B controls, as built, test nothing about specificity.

**Rulings.**
1. **Mismatched donor becomes a difference patch.** At layer l the clean run's P1 residual is set to
   h_clean(l) + (h_swap_donor(l) − h_clean_donor(l)): the donor row's own swap-minus-clean difference, which carries the
   donor's substitute at the same scale as the real patch (donor difference norms 0.2–8.6 at layer 18, the same range
   as own differences). Donor choice unchanged (rev 3 ruling 3, nearest earlier cyclic, both read ids outside the
   row's); both donor residuals are already on disk (`resid/*.npz`, clean and swap). The full-overwrite donor is
   dropped, not kept beside, so the pass count per row does not change.
2. **Random control becomes a difference-norm perturbation.** h_clean(l) + ε with ε a seeded Gaussian direction
   ([44, row, layer], unchanged) of norm ‖h_swap(l) − h_clean(l)‖, added to the clean residual, not replacing it.
   Reported like §38's Random, no bound. The full-norm replacement is dropped.
3. **S3 validity.** Primary, unchanged: the donor control's mean shift toward S over following rows within ±0.5 nat
   at every layer. Pre-declared fallback if the absolute bound fails: S3 is still read if, at every layer from the
   median d_T onward, the donor's mean shift is at most 0.25 × the forward patch's mean shift at that layer (the
   specificity ratio), and the fallback is named in the report. Otherwise S3 is not read, as before. The absolute
   check is printed in both cases.
4. Executor options (a) and (c) are rejected: (a) subtracts a disruption measured at the wrong scale, (c) removes the
   only specificity check S3 has. (b) is what ruling 1 achieves without needing matched prompts.
5. Self-overwrite, the T sanity bounds, the following rule, the classes and S0–S2 / U1 are unchanged. The smoke's
   full-overwrite numbers stay on disk as the record of why the controls changed.
6. Tests to add before the S run: a donor difference of zero (donor = self) gives zero shift; the random vector's
   norm equals the difference norm at that layer; the pass count per patched row is unchanged.

Development-only observations, not readings: S0 following 4/6 among successes; d_T forward 0.889, reverse 0.889, on
rows whose rev 0 d_c is about 0.75; two non-following successes are indeterminate (Δ moves, no flip). All on
folds 0–1; no confirmation row has been read.

## Rev 7 (2026-10-30): the rev 0 7B filter fixed; confirmation read of rev 0 authorised

7B development run complete (7572906: 126 rows, 61 patched, 208 s/row bf16 CPU). Advisor recount from
`runs/exp44_patch/qwen7b/*.json`: strict rank 39, **strict + gap 38 < 40**, so the pre-declared relaxation applies;
relaxed rank 66, **relaxed + gap 59** (7 gap-excluded), 3 R(0)-flagged (0.050–0.067, gaps 2.5–4.7), **56 claim rows**.
Report and recount agree. Development C1: median d_c 0.679 [0.643, 0.714] (n 56; flagged included 0.679, n 59);
C2: mean R at layer 13 0.022 [0.001, 0.045]. Development readings only; they decide nothing.

**Rulings.** (1) Both models read the **relaxed filter with the 1-nat gap** (3B 52 development rows, 7B 59; R(0)
flags excluded with counts). (2) `FIXED_FILTER` and `CONFIRM_AUTHORISED` may now be flipped together, in one commit by
the Executor. (3) The confirmation read of rev 0 (folds 2–4, 204 rows per model, eligible rows patched) is a single
read per model, after which C1–C3 are read as written in §4 with the rev 1 and rev 5 amendments. (4) Order on the
laptop: Qwen 3B S then U first (the user has asked for the swap results), then Llama 3B, then the rev 0
confirmation reads (3B about 3 h, 7B about 7–12 h depending on the eligible share); the Executor reports the
projected 7B confirmation time before starting it. No pod is involved unless the Advisor asks the user.

## Rev 8 (2026-10-30): the repaired swap on Qwen 3B — user-approved; GPU pod permitted for speed

**Standing.** The S0 failure on Qwen 3B (34/46 = 0.739; review 6230ac3) and the same non-follower classes on
Llama 3B (review 8ec5bef, ruling 3) show a fault in the substitute construction, not in the models: "other"-type
substitutes drawn by LongMemEval question type are often answers the question cannot take; bare numbers were swapped
across digit count, digit/word form and unit presence; a leading determiner ("The", "A", "my") can be the read token
when the substitute lacks one. The user chose the repaired rev over closing the line (Advisor pane, 2026-10-30:
"You have approval for both"). D-45 stands: this is a repair of a measurement, not a new intervention, probe, scaling
or routing study.

**Honesty clause, fixed now.** All 89 Qwen 3B swap rows (folds 0–4, pooled by rev 4) were already read under rev 1,
and the repair was motivated by those reads. Rows whose substitute is unchanged under rev 8 give **no new
information** (their results are deterministic and already known: name 12/14, currency 2/2, duration 6/7 following
among successes). The fresh evidence in rev 8 is the set of rows whose substitute **changes** (every bare-number row
that gets a form-matched substitute; any name row re-drawn under the determiner rule). The report states S0 on the
whole rev 8 set *and* on the changed-substitute rows separately, and the §44 text calls rev 8 "a repaired design
tested on the same rows after the fault was seen". No claim from rev 8 is confirmatory; S1, if (a)–(c) pass, is
reported as "established on a post-hoc-repaired design" and F11 says so.

**Changes to the gate (`scripts/exp44_swap_gate.py`, rev 8; everything else rev 3 behaviour).**
1. **"other" type excluded.** Rows with `surface_type == "other"` are excluded as `type_other_rev8` (Qwen 3B: 18 rows).
2. **Bare numbers form-matched.** For `number:bare` rows the candidate must additionally satisfy: (i) same form, both
   digit-form or both number-word-form; (ii) digit-form: same count of digits in the leading number; (iii) unit
   presence matches: both have a trailing unit/noun after the number ("points", "Mbps", "eggs") or neither (the unit
   itself need not match). Ranges ("5-2", "2-3 eggs") count as digit-form with the digit count of the whole string.
   A row with no passing candidate is excluded as `no_substitute` (counted; the rev 2 cap rule unchanged).
3. **Leading determiner matched.** For every kept type: if the gold's first word is in {the, a, an, my, our, his, her,
   their, its} (case-insensitive) the candidate's first word must be the same word; if the gold has no such word the
   candidate must have none. The rev 3 shared-prefix rule then puts the read at the first divergent content token.
4. Candidate walk unchanged: row i's list shuffled by `random.Random(44 + i)`, first passing candidate taken, so the
   unchanged rows keep their rev 1 substitute (the honesty clause depends on this; the Executor asserts the identity
   of the substitute string on every unchanged row and reports the count of changed rows).
5. Percent rows (1) and date rows (4) are kept as they are; too few to matter, reported in the type table.

**Runs.** Qwen 3B S only (the U runs and the Llama runs do not change). Bidirectional patching and the rev 6 controls
as before. Laptop ≈ 2.7 h fp32, or a pod (below). Output under `runs/exp44_swap/qwen3b_rev8/S/`, report
`runs/exp44_swap/qwen3b_rev8/report_pooled.{md,json}`; the rev 1 directory is kept untouched.

**Reads (as §C, with the honesty clause).** S0 on the whole set and on changed rows; if S0 passes on the whole set,
S1 (a)(b)(c), S2 (≥ 8 refusal rows per class, else descriptive), S3 with the rev 6 validity rule; then the type
table. **Power, stated in advance:** after the "other" exclusion the set is ≤ 71 rows, ≈ 38 successes and ≈ 33
failures before any `no_substitute` loss; S1(c) (half-width ≤ 0.15) needs ≈ 40 failures, so S1(c) is expected to
fail on power even if the repair is sound. That outcome is "not reliably distinguishable on these rows, underpowered",
documented, no further rev. S0 on the repaired design is the question rev 8 can actually answer.

**GPU pod (speed only; the user's words 2026-10-30: "if there's a way to speed things up by renting out pods let's
do it").** Both remaining computations may run on one RunPod A40 instead of the laptop: (A) the rev 0 7B
confirmation, all 204 rows re-run on the GPU in bf16 (not the 171 remaining: one device per confirmation read), into
`runs/exp44_patch/qwen7b_gpu/`; the laptop run keeps going as the fallback until the GPU read is accepted, then it is
stopped; the rows both devices have done are reported as a device-consistency check (per-row d_c equal, max |R|
difference) before the GPU rows are read as the confirmation; (B) rev 8 S on Qwen 3B in fp32. The engine already
selects CUDA when present (`HFEngine.device`); `dtype` overrides stay as in `MODELS`. Pre-launch stub check and D-22
smoke on the pod (a 3-row run of each) before the full runs; D-30 fresh-venv bundle test. **Advisor's quote to the
user: about $0.50–1.00, hard stop $2.50 (≈ 4–5 h at $0.49–0.59/h); programme after this ≤ $22.56 of the $30 cap.**
The Executor creates no pod until the user types the approval in its pane; it never leaves a pod billing while
waiting on a question; it terminates and verifies with list-pods when both runs are copied back.

**Rev 8 gate review (e60c8b5; `research/reviews/2026-10-30-exp44-rev8-gate.md`):** counts confirmed (70 rows; 47
unchanged / 23 changed / 1 lost / 18 "other"). Determiner compared as written (accepted deviation). **Rule 5 added
before any run:** a candidate whose divergent read token for G or S has no letter or digit is rejected and the walk
continues (c7dc5443 "5-2" → "50" read "-" vs "0"); gate re-run, other 69 rows asserted identical.
