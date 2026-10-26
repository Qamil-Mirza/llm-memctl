# §42 proposal (rev 0, Advisor, 2026-10-25; Q17): selective steering by the projection signal, pre-registered on fresh rows

**Status: draft for the Executor's feasibility note and build. No pod without a fresh quote to the user.**

## 1. Question

§41 found, on §40's 120 LoCoMo refusal rows, that s_proj = ⟨h_29(`:`), v_29/‖v_29‖⟩ on the unsteered pass ranks which
steered refusals come out right (AUROC 0.803 (0.724, 0.875)), and a post hoc cut at the top 10–30% would have scored
+9 against the §35 line's +4 at a penalty of one per wrong answer. Does a threshold on s_proj, fixed now from §40's
rows, beat the §35 line, unconditional steering and abstention on rows it has never seen?

## 2. The frozen policy (fixed from §40's rows before any new row is read)

- Cond: steer (layer 29, α 2, Steer-all) iff s_proj ≥ τ, else the unsteered output. τ = the 70th percentile of s_proj
  over §40's 120 R rows (steer the top 30%), the widest cut of the post hoc plateau; the value is computed from
  `runs/exp40_pod/qwen3b/apass.jsonl` and written to `rule42.json` with the input sha before the build is reviewed.
  The whole curve (q = 0.1 … 0.9) is reported descriptively; only q = 0.3 carries a verdict.
- The direction, α, judge, classifier, soft-refusal pattern, utility (+1 correct, −λ wrong, 0 refusal / degenerate /
  soft refusal) and λ (1 primary, 2 beside) are §40/§41's, unchanged.
- Cond applies to every marked row, refusal or not: on a correct row that fires, the steered output replaces the
  correct one (that is the damage the policy must pay for).

## 3. The fresh set

LoCoMo rows not used by §40: a seeded shuffle (seed 42) of the eligible answerable rows minus §40's 600 (923 remain),
600 taken; 40 adversarial rows from the unused category-5 rows (406 remain). Same view builder, filters, template,
batched notes and selection (R = first 120 marked refusals, C = first 40 marked judged-correct). The set is in
distribution (same benchmark, disjoint rows); a cross-benchmark test is a separate step and is said so in the record.

## 4. Claims (R rows, λ = 1, paired bootstrap 10k; λ = 2 beside)

- **Claim 1:** U(Cond) − U(B) > 0 (LB > 0): the selective policy beats the instruction line.
- **Claim 2:** U(Cond) − U(A) > 0: it beats abstention.
- **Claim 3:** U(Cond) − U(Steer-all) > 0: it beats indiscriminate steering (trivial if Steer is net negative; U(Steer)
  − U(A) printed beside, the §41 reading rule applies).
- **Claim 4 (replication):** AUROC(s_proj) for gold(Steer-all) on the new R rows, LB > 0.5.
- **Control D (damage):** on C, gold(Cond) − 1, with the number of C rows that fire; harmful if UB < −0.10.
- **Control U (unanswerable):** Cond's answer rate on the 40 adversarial rows beside A, B and Steer-all; the record
  reports it as a cost whatever the claims say.
- Descriptive: precision of fires; coverage and risk of each policy; the full q curve; by type.
- **Prediction, recorded now:** Claim 1 SHOWN by a small margin (+3 to +7 units of 120), Claim 2 SHOWN, Claim 3 SHOWN,
  Claim 4 SHOWN at 0.70–0.80; Control D: 1–3 of 40 fire and most stay right; Control U: Cond near A (about 0.35–0.45).

## 5. Cost and rules

Same pipeline as §40 (`exp40_pod_run.py` with the seed, the exclusion list and `rule42.json`; the §40 code otherwise
unchanged). One A40 secure pod at the then-current price (last $0.59/h), hard stop 60 min, about 12 min expected
(about $0.12), at most about $0.59; programme spend about $17.82 → about $17.94–$18.41 of $20. D-22 smoke (batching
drift and judge compliance), D-30 bundle test in a fresh venv, the user's line in the Executor's pane before creation.

## 6. For the Executor's feasibility note

The τ value and its sha; that the exclusion of §40's rows is by question id and asserted; any §40 code path that
assumes the seed-40 set; the C-row firing path (steered output replaces the correct one); time.

## Rev 1 (Advisor, 2026-10-25): rulings on the Executor's feasibility note; design frozen for the build

- **F1, ruled: Cond = steer iff A refuses AND s_proj ≥ τ, else A's output.** The Executor is right that s_proj on all
  marked rows mostly separates "already answering" (mean −15.2) from "refusing" (−28.8), so §2 as written would steer
  33 of §40's 40 correct rows and fail Control D by construction. "A refuses" is the step-0 argmax being ` unknown`
  on the same unsteered pass that gives s_proj, so the policy is still one forward pass at `:`. It matches the
  population τ was fitted on (refusals). C rows cannot fire; Control D is 0 by construction and the C-row fire count
  (should be 0) is printed. **Prediction for Control D revised: 0 fires.** Control U unchanged: Cond's answer rate on
  the adversarial rows beside A, B and Steer-all (it fires only on the refusing adversarial rows that pass τ).
- **τ fixed: numpy linear 70th percentile of s_proj over §40's 120 R rows = −18.896762090918696** (input
  `runs/exp40_pod/qwen3b/apass.jsonl` sha 388e698b…; fires on 36 of the 120 §40 rows). The method and value go into
  `rule42.json` with the sha; the report prints both.
- **F2** (seed 42; exclusion list = §40's 600 + 40 question ids asserted disjoint; pools asserted 923 / 406; §40's
  filter exclusions kept) accepted.
- **F3:** keep the Random arm unchanged (descriptive, cheap); the report header's seed becomes a parameter; `--exp`
  tag so §42 writes to `runs/exp42_pod`. Accepted.
- **F4** (arms and order as §40; Cond computed in the report from `rule42.json`; per row, which arm supplied Cond's
  output is logged) accepted.
- **F5** time as §40 (about 12–14 min, hard stop 60 min). **F6** accepted; the §41 reading rule applies to Claim 3.
- **Quote to the user (separate, before any launch):** one A40 secure pod at the then-current price (last $0.59/h),
  hard stop 60 min, at most about $0.59, about $0.12–0.14 expected; programme spend about $17.82 → about $17.95–$18.41.
  The user's own line in the Executor's pane before creation. Build → Advisor build review → bundle → D-30 → quote.
