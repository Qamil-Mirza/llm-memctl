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
