# §38 pre-registration: is the answer causally available under the 3B's refusal? A content-free "answer, don't refuse" direction, with an evidence-cut plausibility control (Advisor, 2026-10-22, rev 1)

**Question.** §34–§34c read the 3B's residual stream and found the gold token "present" at the answer position on about two
thirds of failures, with ` unknown` winning late and a ` unknown` disposition that barely moves when the evidence is cut.
§35 showed that one instruction line forbidding "unknown" turns 27 of 64 marked refusals into correct answers on the
confirmation folds. Both are readouts or prompt interventions; no representation-level intervention exists in the record
(audit §2.6). This experiment asks: **under the refusing prompt, is the correct answer already in the 3B's late state, so
that shifting only its disposition to answer, with nothing question-specific added, releases the gold?** And the
control: does the same shift release the gold just as often when the evidence is not in the prompt?

**Why not patch run B into run A.** The §35 arms A (frozen note + `Answer:`) and B (frozen note + the forbidding line +
`Answer:`) share every token up to the end of the note. Overwriting A's residual at the answer position with B's at the
last layer reproduces B's output by construction: B's state at that position *is* B's answer. It is a plumbing check,
kept here as sanity S, not evidence. The primary intervention must be unable to carry the answer. It is therefore a
single direction estimated as a mean difference over *other questions* (folds 0–1), applied to confirmation-fold rows
the estimation never saw.

## 1. Data (all cached; nothing new generated before the design run)

- Population and labels: §35's (`runs/exp35/qwen3b/{A,B,N,Ncut}`, judge rows in `runs/exp35/qwen3b/judge`, the §23
  records via `scripts/exp35_generate.py` / `scripts/exp35_report.py`). Marked rows only (the §35 token check).
- **Refusal rows R:** marked rows where arm A refuses (§23 rule: the text after the last `Answer:` starts with "unknown").
  Counts from the cache: folds 0–1: 28; folds 2–4: 64. Of these, B correct 8 / 27, B wrong 5 / 15, B refuses 15 / 22.
- **Direction-estimation pairs (folds 0–1 only):** the 13 development rows where A refuses and B answers (correct or
  wrong). The direction is `mean_over_pairs( resid_B(L, last) − resid_A(L, last) )`, where `last` is the final prompt
  token of each run (the token after `Answer:`), at layer L. It is fixed from these 13 rows and never re-estimated.
- **Intact-correct rows C:** marked rows where A is judged correct (39 dev; 55 confirmation). Used to check the direction
  does not break answers that were already right.
- **Evidence-cut control prompt:** the §34c cut view (`ablate_evidence`, evidence lines removed, filler per §34c) with the
  3B's own cached note *from the cut prompt* (the §35 N-cut arm's generated note, extracted by the §35 note rule), then
  `Answer:`. This is the same question with no evidence anywhere in the context; the intact frozen note cannot be used
  because it carries the gold (§34c's reason for the forced position). Rows whose N-cut note is empty keep an empty note.
- Engine: `transformers` with forward hooks (laptop fp32 CPU for the design run; pod bf16 on an A40 for confirmation).
  **Reproduction rule (fixed):** on the chosen engine, unsteered A and B are regenerated for every row used; a row counts
  only if A's refusal flag and B's refusal flag match the cache (verdict for B checked by string match on the laptop,
  by the §35 judge on the pod). Rows that fail are listed and dropped before any steered output is read. If fewer than
  0.8 of rows reproduce, the engine is not fit and §38 stops at that stage.

## 2. Interventions (wording and arithmetic fixed now)

| name | what is changed | role |
|---|---|---|
| **Steer (primary)** | add α·v_L to the residual stream at layer L, at the current position, for every generated token (the `Answer:` position and each new token), on run A's prompt | the content-free disposition shift |
| Steer-cut | the same α·v_L, same L, on the evidence-cut control prompt | plausibility control |
| Random | a random vector with the norm of v_L, same L and α, same rows | specificity control |
| Reverse | −α·v_L | the direction should push toward refusal: answer rate must not rise |
| Sanity S | overwrite A's residual at `last` with B's at layer L, L = final layer and L = 0.2 depth | plumbing: must reproduce B's first token at the final layer (≥ 0.9 of pairs) and mostly not at 0.2 depth (≤ 0.5) |

Qwen2.5-3B-Instruct has 36 layers and hidden size 2048. The design run sweeps L over depths {0.5, 0.6, 0.7, 0.8, 0.9} (layers
18, 22, 25, 29, 32) and α over {2, 4} in units of the mean residual norm at that layer and position over the 13 pairs
(α·v_L is rescaled so that its norm equals α × mean‖resid_A(L, last)‖ × ‖v̂‖, v̂ the unit direction). Greedy decoding,
temperature 0, max new tokens 16 on the laptop (a refusal is decided by the first token; answers are short phrases) and
64 on the pod (§35's frozen-note setting). Judging: pod rows by the §35 judge (Qwen2.5-7B-Instruct, per-type prompts,
cached by text); laptop rows by the §35 D1 string rule, reported as provisional and never as a claim.

## 3. Claims (confirmation folds 2–4, fixed sequence) and controls

Scores: gold(x) = 1 if the judged answer under intervention x is correct; ans(x) = 1 if it is not a refusal. Intervals:
paired bootstrap over questions, 10,000 resamples, 95%.

- **Claim 1 (primary).** On the reproduced refusal rows R (expected about 60): P(gold(Steer)) − P(gold(Steer-cut)),
  paired over questions. **SHOWN** if the lower bound > 0. **Meaningful** if P(gold(Steer)) ≥ 0.30 (B's own lift on the
  same population is 27/64 = 0.42). If fewer than 30 rows reproduce, UNDERPOWERED is written beside the verdict.
- **Claim 2 (only if Claim 1 SHOWN; descriptive with CI).** P(gold(Steer)) / P(gold(B)) on R: the share of the §35
  instruction line's effect that a content-free direction reproduces. Also P(gold(Steer) | B correct) and
  P(gold(Steer) | B wrong) and P(ans(Steer) | B refuses).
- **Control X (specificity, decides whether the result is read at all).** On R: P(ans(Random)) ≤ 0.2 × P(ans(Steer)) and
  P(ans(Reverse)) ≤ P(ans(A unsteered)) + 0.05. If X fails, the direction is not specific and Claim 1 is reported as
  NOT INTERPRETABLE regardless of its interval.
- **Control C (damage).** On the intact-correct rows C: P(gold(Steer)) − P(gold(A)) with CI. A drop of more than 0.10
  is reported as "the direction damages correct answers" beside Claim 1; it does not change the verdict.
- **Control T (type).** Claim 1's two rates by question type (temporal-reasoning, multi-session, knowledge-update,
  single-session-user, single-session-assistant); descriptive, small cells.
- **Sanity S** as in §2; a failure means the hooks are wrong and nothing is read.
- Costs: forward passes, seconds, and dollars, beside.

## 4. Readings, fixed before the data

- Claim 1 SHOWN and meaningful, X passes: the gold is causally available in the 3B's late state under the refusing prompt
  and the refusal is a late disposition; the §35 lift is, mechanistically, a disposition shift. The reading-vs-policy
  split of F6 gets its first causal support.
- Claim 1 NOT SHOWN with both rates high: steering makes the 3B answer plausibly whether or not the evidence is there (the
  §34c and §36 reading: the late state holds a guess, not the evidence). The "present" of §34 is then plausibility.
- Claim 1 NOT SHOWN with both rates low but ans(Steer) high: the direction makes the model answer, but what it says is
  not the gold; the answer is not in the late state and the instruction line of §35 worked by making the model re-read,
  not by releasing a stored answer.
- Design gate fails (no setting makes the 3B answer): no content-free direction unlocks answers; the §35 line works
  through its content. §38 closes at no cost.
- Control C shows damage: the direction is a general perturbation, not a refusal switch; Claim 1 is reported with that
  caveat and no mechanism is claimed.

**Prediction, recorded now.** Sanity S passes. A setting at depth 0.7–0.9, α 2–4, gives ans(Steer) 0.6–0.9 on the dev
pairs. On confirmation: P(gold(Steer)) 0.25–0.40, P(gold(Steer-cut)) 0.10–0.25 (the §34c plausibility share says the cut
rate will not be near zero); the chance Claim 1 is SHOWN is about 0.5, meaningful about 0.4. Random ≤ 0.1 answer rate.
Control C drop under 0.05. Temporal-reasoning rows release least.

## 5. Stage 1: the free design run (laptop, folds 0–1 only, the script refuses other folds)

1. Reproduction of A and B on the 28 dev refusal rows and 5-row timing first; the Executor reports the projection.
2. Direction estimation on the 13 pairs at each of the 5 layers.
3. Sanity S at the final layer and at 0.2 depth.
4. Sweep: Steer on the 28 dev refusal rows for the 10 (L, α) settings, max 16 new tokens; Random and Reverse at the
   best setting; Steer on 8 of the 39 dev intact-correct rows at the best setting (damage check).
5. **Design gate (label-light):** sanity S passes; reproduction ≥ 0.8; at the chosen setting ans(Steer) ≥ 0.5 on the
   28 rows and ans(Random) ≤ 0.2 × ans(Steer). The setting (L, α) is chosen by the highest ans(Steer) with Random
   passing, ties to the shallower layer; it is then fixed in the record before the pod. The dev gold rate (string rule)
   is printed but is not a gate (8 B-correct rows).
6. Time: about 75 forward passes of 1,100 tokens for reproduction and estimation plus about 330 short generations
   (prefill 15 s + 16 tokens at 0.95 s/token ≈ 30 s each) ≈ 3 h on 8 threads under `GUARD_VM_GB=22`. If the 5-row timing
   projects over 5 h, the α grid drops to {4} and the layer grid to {0.6, 0.8, 0.9}.
7. If the gate fails, §38 closes as "no content-free direction makes the 3B answer" in EXPERIMENTS.md, no spend.

## 6. Stage 2: the pod (NEEDS SPEND; quoted only if the design gate passes)

One A40 secure pod (about $0.59/h at the §36b price; the Executor re-reads the price at launch), **hard stop 60 min
(≈ $0.59)**, `transformers` bf16 for the 3B with hooks and the §35 judge model for verdicts. D-22 smoke: 5 rows through
reproduction, Steer and the judge before the full run. Run order: reproduction of A and B on all folds (the dev folds in
brackets), sanity S, Steer / Steer-cut / Random / Reverse on R (folds 2–4), Steer on C, judge, report with `--confirm`.
Programme spend would move from about $17.55 to about $18.15 of the $20 cap. The pre-registered quote goes to the user
with the design report; nothing is created before the user's yes.

## 7. Rules carried over

No held-out peeking: the direction, L and α come from folds 0–1 only; folds 2–4 are read once, on the pod, with
`--confirm`. One engine per run directory; wording and arithmetic sha-stamped per row; report refuses mixed engines.
No new wording is tried after the design run. Nothing outside `~/Code/llm-memctl`; `.env` never printed. The Executor
builds with a stub engine and tests, the Advisor reviews the build, the design report, and the EXPERIMENTS.md text.

## 8. What this cannot show

Nothing about the 7B (its residual width differs; no cross-model patch). Nothing about *why* temporal and multi-session
rows fail to derive (that is a separate attention or note study). A positive result shows availability and a late
override; it does not locate the circuit that decides to refuse. A negative result does not show the answer is absent,
only that one direction at one layer does not release it.
