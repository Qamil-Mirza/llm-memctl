# Paper contract: self-doubt

Draft 1, 2026-11-12. Not yet approved by the user. This page is the test every experiment and every claim must pass.
It fixes the goal and the stopping rules, not the steps.

**How it is used.**
- **Before** an experiment, its EXPERIMENTS.md entry names the line of this page it fills and the result that would
  change the plan. If no possible result would change the plan, the experiment is not run.
- **After** every result, the entry answers one line: *does this change the significance table (section 3), and do we
  continue, narrow or stop?* There are no calendar gates. A pivot can happen after any result, and this line makes
  sure the question is asked every time.
- The page changes only by a dated edit with the user's agreement.

## 1. The claim we hope to make

When a reasoning model ignores a valid update that arrives mid-thinking, it has read the update; the update's value
is lost afterwards, between being read and the answer being written. We aim to show where it is lost and whether one
internal quantity governs the loss (how long the update's value stays readable) or two (how firmly the trace has fixed
an answer, times how well the model carries an update forward).

**Fallback claim** if the mechanism does not resolve: a careful, controlled description of when the update is lost
(timing, doubt versus silent carrying-on), with the null results stated plainly.

The three research questions (timing, doubt versus carrying on, scale with a held-out prediction) were proposed on
2026-11-11 in `reports/Self doubt mechanism research questions.md`. **Not yet adopted by the user.**

## 2. Venue and the one fixed decision

| Date | What |
|---|---|
| 2027-01-04 | ARR deadline (ACL 2027). Cannot also go to ICML. |
| 2027-01-14 / 01-22 | ICML 2027 abstract / paper. Preferred venue: it published the original paper. |
| ~2027-02-01 | ICLR 2027 workshop fallback (list published after 2026-11-29). |

The only fixed decision is external: by early January the user picks one of ARR, ICML or the workshop route, since
the two main venues forbid dual submission. Realistic ceiling: a main-track poster.

## 3. The significance test (asked now, and again after every result)

| Question | Current answer | What would fail it |
|---|---|---|
| Does anyone care? | Partly. The behaviour was named in an ICML 2026 poster (scores 4;4;4;5, significance 2;2;3;4). No one has explained it. | A competitor explains it first, or reviewers see it as a prompt artefact. |
| Does it generalize? | Two sizes of one family. Stale rates rise with lateness on 1.7B MATH (16% → 59%) and 8B MATH (2% → 23%). No other family yet. | The effect exists only on 1.7B, or only on one benchmark. |
| Does the explanation beat a simple baseline? | Not yet tested. | A located component does no better than a logit bias, the paper's prompt fix, or a random set of the same size. This is how false abstention ended. |

## 4. What we already have (held-out, Qwen3-1.7B, GSM8K)

- The update is read on failures: the fact margin right after the update is +9.36 [+7.57, +11.03].
- It is present but not used: 128 tokens later, the fact margin is still > 0 while the answer margin is −11.17.
- Failures hold the value less from 32 tokens on, but the value also fades on successes, so "fades" is not yet
  failure-specific.

All three are read-outs, so they are correlational. None is yet a causal claim (rule E1 below).

## 5. The main figure

Update value readable after the update (y) against tokens since the update (x), for failures and successes, on both
models. Paired with the causal panel: patching success states into failures raises correct answers, beside the three
baselines.

## 6. Baselines every intervention is reported beside

1. A random set of components of the same shape.
2. A logit bias at matched effect.
3. The original paper's prompt fix.

## 7. The strongest objections, and the evidence that answers each

| Objection | What answers it |
|---|---|
| A small-model quirk. | RQ3: the held-out prediction on Qwen3 0.6B and 4B, and failures on 8B MATH. |
| The fix is no better than a prompt or a bias. | Section 6, for every intervention. |
| The read-outs are correlational. | Patching or ablation for every mechanism claim (E1). |
| "Hypothesis Anchoring" found every mechanistic test null. | Name where our setup differs (valid update vs injected answer; our models; a stale-answer check), and report our own nulls. |

## 8. Competitors to watch

- "Hypothesis Anchoring" (ICLR 2027, desk-rejected), OpenReview qXmWvPFkZP.
- "Who Owns the Answer?" (ARR, May 2026), OpenReview XAYzryI95u.
- "Learning to Revise When Updates Arrive Mid-Reasoning" (ARR, October 2026), OpenReview kKHmd2Q7ra.
- "The Termination Circuit" (LessWrong, unreviewed), Qwen3-1.7B stop-thinking circuit.

The ledger of closest papers is kept in `research/RELATED_WORK.md` (one line per paper: what it does, a quoted
passage, how we differ, a resolved ID).

## 9. What would kill the line

- RQ2 and the RQ1 tabulation show no dissociation worth a headline.
- Every located component ties the logit bias (the false-abstention ending).
- The 8B and the held-out models show no failures to explain.
- A competitor publishes the same mechanism first.

## 10. Evidence rules

- **E1. Causal test.** No mechanism claim is recorded without a patching or ablation result beside its baselines.
  Lens, probe and attention read-outs are labelled correlational.
- **E2. Alternatives.** Each figure names the alternative explanation it rules out, decided before the run.
- **E3. Numbers trace.** Every number in a draft carries a source tag to a committed result file;
  `python -m selfdoubt.check_numbers <draft>` must pass.
- **E4. References resolve.** Every reference has an arXiv ID or DOI; `python -m selfdoubt.check_refs <draft>` must pass.
- **E5. Fresh review.** Before any spend and before any claim reaches STATE.md, a separate session reviews the code, logs
  and data, not the summary. It answers pass/fail defect questions and never gives a score. No draft is revised to
  raise a reviewer's score.
- **E6. Raw outputs.** The user reads 10 randomly chosen raw outputs behind each headline number.
- **E7. Nulls.** Before reading a null, check that the intervention actually changed the activations. Report every
  seed.
- **E8. Debug cap.** After three failed fixes of the same error, stop and escalate to the user.
- **E9. Cost.** Each EXPERIMENTS.md entry records its cost in dollars and in wall-clock time.
