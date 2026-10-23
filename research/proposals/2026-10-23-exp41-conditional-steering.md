# §41 proposal (rev 0, Advisor, 2026-10-23; D-33 item 3): conditional steering — can a signal at ':' say when steering will help?

**Status: draft for the Executor's feasibility notes. Development on the laptop is free; the confirmation rides on §40's pod.**

## 1. Question

Unconditional steering opens almost every refusal, and about half of the opened answers are wrong (§38: 31 right,
23 wrong of 60 on the sealed folds). The prompt-only line (B) opens fewer and is more selective (26 right, 12 wrong).
Can a simple signal read off the unsteered forward pass at the `:` position predict, row by row, whether steering will
produce a correct answer, so that a conditional policy beats both unconditional steering and the prompt-only line
once wrong answers are penalised?

## 2. Signals (all from unsteered prefills; no generation)

- s_margin: logit(` unknown`) − max logit of any other token at `:` (the §34 refusal margin).
- s_proj: ⟨h_29(`:`), v_29/‖v_29‖⟩, the residual's projection on the direction at layer 29.
- s_evid: s_margin(intact) − s_margin(cut), the refusal's sensitivity to the evidence (§34c's quantity), one extra
  prefill on the cut prompt.
- Each signal gives one rule: steer iff signal > τ (or < τ; the sign is fitted). One (signal, τ) is chosen on
  development data and frozen before §40.

## 3. Scoring (fixed now)

Per row, utility U = +1 if the final answer is correct, −λ if an answer is given and wrong, 0 if the model abstains.
λ = 1 primary; λ = 2 beside; the full risk–coverage curve printed. Policies on the same rows: A (unsteered), B
(prompt-only line), Steer (unconditional), Cond (steer iff the frozen rule fires, else A's output).

## 4. Development (laptop, fp32, folds 0–1, free; D1 gold, provisional)

Rows: §38's dev R 28 (Steer-all rows exist), C 39 (Steer-all to be run, about 16 min), damage 8; B rows exist. Prefills:
intact and cut for all 75 rows (about 30 min). Fit τ for each signal by the dev utility at λ = 1 over a grid of row
values; pick the signal with the best dev utility; report the dev utility of all four policies; freeze (signal, sign, τ)
by commit before §40 starts. Development gate: Cond must beat Steer on dev utility at λ = 1 by at least 3 rows, else the
rule is not scored on §40 and §41 closes development-only (the signals are still logged).

## 5. Confirmation (on §40's new set, same pod, no extra arms)

- **Claim 1:** U(Cond) − U(Steer) at λ = 1 on §40's R rows, paired bootstrap 10k; SHOWN if LB > 0.
- **Claim 2:** U(Cond) − U(B) at λ = 1; SHOWN if LB > 0.
- **Beside:** the same at λ = 2; coverage and risk of each policy; the rule's AUROC for gold(Steer).
- **Prediction, recorded now:** Claim 1 SHOWN (the signal removes the worst third of opened rows); Claim 2 NOT SHOWN
  (B's selectivity is hard to beat with one threshold); AUROC 0.60–0.70.
- Cost: none beyond §40's pod (logging only). The rule must be in a commit named in §40's pre-registration before launch.

## 6. Open for the feasibility note (Executor; no build yet)

Whether `exp38_dev.py`'s prefill path exposes the `:` logits and the layer-29 residual without a second pass; the cut
prompt for C rows; the time of the 75 × 2 prefills on the laptop; anything that forces a choice not fixed above.

## Rev 1 (Advisor, 2026-10-23): rulings on the Executor's feasibility note; development may be built after §39b

- **(i) Rows.** 67 distinct development rows: R 28 and C 39 (the 8 damage rows are the first 8 of C). Prefills intact
  and cut on all 67 (the step-0 logits returned from the same pass; residual at layer 29 captured); Steer-all run on
  the 39 C rows (the 28 R rows reuse §38's). About 45 min on the laptop, free, `GUARD_VM_GB=22`.
- **(ii) Labels.** Option (a): the cached §35 judge verdict wherever a text equals the cached A or B text byte-exact;
  D1 only for new texts. Option (b), the 7B judge on the laptop for the new Steer-all texts, only if 5 timed calls
  project the whole set under 45 min; otherwise D1 for new texts with the bias stated in the report (D1 marks about half
  the correct paraphrases wrong, which tilts the fit toward steering). Which option ran is printed.
- **(iii) The refusal token set.** s_margin = max logit over the refusal-start ids minus the max logit over all other
  ids, where the refusal-start ids are the first generated ids of every refusal-class row in the §38 dev A and B rows
  (the list printed and frozen in the report; ` unknown` is expected to dominate).
- **(iv) Utility.** Per row: +1 correct; −λ for an answer judged wrong; 0 for refusal, degenerate, or a **soft refusal**
  by the frozen pattern `not (specified|mentioned|provided|determined|explicitly|contain|named|available|stated|given)
  | does not contain | cannot be determined | no (information|mention)` (case-insensitive; the §39 review's pattern).
  A hedge is not an unsupported answer. The v2 classes still give the ans counts; both are printed.
- **(v) Selection.** Three signals × two signs × a τ grid over the row values, fitted on the dev utility at λ = 1; the
  best single (signal, sign, τ) is frozen by commit. The dev utility is optimistic by construction and is reported as
  such; the 3-row development gate (Cond ≥ Steer + 3 on the weighted dev utility) stands; the out-of-sample test is §40.
- **(vi) Mix.** The fit uses a weighted utility with C rows weighted so that C : R = 1 : 3 (§40's caps 40 : 120); the
  unweighted dev utility is printed beside it.
- **Confirmation** on §40's rows as §5, unchanged; plus the descriptive adversarial answer rate of Cond against Steer-all.
- **Order.** Build after the §39b run; the Advisor reviews the build; the dev run is free; the frozen rule's commit hash
  goes into §40's launch message. If the gate fails, §41 closes development-only and §40 still logs the signals.
