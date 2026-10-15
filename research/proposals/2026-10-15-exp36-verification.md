# §36 pre-registration: can the 3B tell which of its own forced answers the memory supports?

Date: 2026-10-15, Advisor draft, revision 1, before any pod run and before any new confirmation-fold number. Branch
`jacobian-lens`. Behavioural follow-up to §35. Prompted by an outside design note ("Experiment 36", pasted 2026-10-15)
that asks whether the 3B can judge evidence support for its own candidate answers, even though its default answering
policy does not use that judgment. This draft keeps the note's questions and drops or narrows the parts the record
already answers or that the data cannot support. **NEEDS Reviewer verdict and the user's spend approval.**

## 0. What the record already says, what the note gets right and wrong, and what this adds

- The note's background figures match the record: §23 P(correct | in view) 0.397 (3B) and 0.667 (7B); §35 frozen note +
  "do not reply unknown" +0.176 (+0.109, +0.242), 0.53 of the matched gap; 14 of 22 S-rows recovered; and its costs
  (B turns 0.275 of S' into wrong answers; F answers 0.68 of evidence-free prompts). One correction: §34c did not show
  that "much of the signal reflected plausibility" in general. It showed that ~0.6 of the 3B's C-failures stay C with
  the evidence cut, and that the 3B's ` unknown` disposition barely moves with the evidence (0.58 → 0.63) while the
  7B's does (0.08 → 0.53). That last fact is the main prior for this experiment. It predicts that the 3B's support
  judgments will follow plausibility more than the evidence.
- **Stage 1 already exists.** The note asks for candidates frozen before any verification instruction. §35 arm B is
  exactly this. It is the 3B's answer after its own cached note with "unknown" prohibited, greedy, vLLM 0.8.5. It was
  judged by the §23 judge, for every marked row of both readers on folds 0–4. No new candidate generation is needed.
  On the evidence-cut prompts, §35 arm F-cut gives the 3B's forced candidate when nothing is in view.
- **Category C (insufficient evidence)** exists in two forms. The first is the §34c exact-span evidence cut
  (`memctl.analysis.jacobian.ablate_evidence`). The second is the 30 LongMemEval abstention questions in the §23
  sample (`*_abs`). The note's caution applies to the cut: an exact-span cut can leave paraphrased or derivable
  content. F-cut answers that the judge marks correct (dev folds: plausibility share) are therefore kept in the
  "unsupported" class by construction and reported apart.
- **The labels are not independent of the 7B.** The note asks for an independent correctness evaluator. The only
  labels are the §23 judge, and that judge is Qwen2.5-7B-Instruct, the same weights as the 7B verifier in Stage 5. A
  7B verifier AUROC against 7B-judge labels can be inflated by shared errors. Mitigation: a hand audit (below). The 7B
  verifier rows stay descriptive.
- **Free check, development folds 0–1 only** (`scripts/exp36_devcheck.py`; it loads folds 0–1 only). The fixed-candidate set is the 3B's marked rows with an arm-B verdict:
  **100**. The B candidate is judged correct on **50 (0.50)**. **16 of 100** B outputs still say unknown even though
  unknown was prohibited (no candidate). The natural policy C0 applied to the same candidates (emit B's candidate iff
  the cached answer was not a refusal) gives coverage **0.70** and risk **0.414**. The natural answer/refuse flag, used
  as a 0/1 score, has **AUROC 0.620** for B-candidate correctness. A gold-free logit baseline uses the final-layer
  rank of ` unknown` at the 3B's own answer position, from the §34b logit rows. It has **AUROC 0.572**, and at C0's
  coverage its risk is **0.414**, the same as C0. For the 7B: 126 rows, B correct 0.65, C0 coverage 0.93, risk 0.316,
  natural-flag AUROC 0.567, unknown-rank AUROC 0.464. So the 3B's own refusals already carry some signal (0.62). A
  verifier is interesting only if it beats that, and the cheapest confidence readout beats nothing.
- What this adds: a gold-free, single-token support readout from the same checkpoint in a fresh context. It is tested
  (i) for discrimination, (ii) as a gate at the natural policy's own coverage (no threshold tuning), (iii) against
  simple confidence, and (iv) for whether it moves when the evidence is removed or changed.

## 1. Population and data

- The §35 population: the §23 `fixed8` in-view answerable questions with a judge label, paired by question id and
  prompt hash. Folds 0–1 are development (126). Folds 2–4 are confirmation (204). **Fixed-candidate rows:** marked 3B
  rows that passed the §35 token check and have an arm-B verdict (dev 100; confirmation 165 expected). Rows where B
  still says unknown stay in the denominator as "no candidate". Every condition must abstain on them. Their count is
  reported, so no condition is scored only on the rows where elicitation succeeded.
- **Candidate** = §35 arm-B `final_answer`, byte-exact. **Correctness label** = the §35 judge verdict on B. A
  **hand audit** of 60 development rows (stratified: 20 judged correct, 20 judged wrong, 20 F-cut) is done blind to
  every verifier score, before launch. It is free and reported as the judge's error rate. The 7B rows are quoted
  with it.
- **Unsupported rows (category C).** (a) **Cut-B:** the same B candidate on the evidence-cut prompt (Control A, paired).
  (b) **Cut-F:** the 3B's F-cut candidate on the evidence-cut prompt. (c) **Abs:** the 30 abstention questions, with
  the 3B's cached answer as the candidate where it is not unknown (descriptive; §23: 7 false answers on 30).
- **Clustering.** In LongMemEval every question carries its own haystack, so the conversation-level cluster is the
  question. All variants of a question (intact, cut, counterfactual, paraphrase) form one bootstrap unit. Paired
  bootstrap, 10,000 resamples of questions, 95% intervals.
- Folds 2–4 were the confirmation folds of §34–§35. Their §35 aggregates are published, but no verifier score exists
  on them. Every design number in this draft comes from folds 0–1 or is fixed without data
  ([[no-heldout-peeking]]).

## 2. The verifier readout (Stage 2)

Completion-style input on the §35 engine (vLLM 0.8.5, bf16, prefix caching, greedy, `max_tokens` 1, `logprobs` 20).
The chat template wraps one user message:

    <the §23 goal line and memory block, byte-exact from the cached prompt, with the reasoning instructions removed>
    Question: <question>
    Proposed answer: <candidate>
    Does the memory above support the proposed answer? Reply with one letter.
    A = the memory states or directly implies this answer
    B = the memory is partly relevant but does not settle it
    C = the memory does not support it, or contradicts it

followed by the generation prompt and the literal `Answer: `. The support score is
**s = (p_A + 0.5·p_B) / (p_A + p_B + p_C)**, from the first-token logprobs of ` A`/`A`, ` B`/`B`, ` C`/`C` (both
spellings summed). The label mass p_A + p_B + p_C is stored per row. If it falls below 0.5 on more than 0.1 of rows,
the readout is "not reliable" and only the argmax label is reported (the note's fallback). No free-text confidence is
used.

- **V (primary):** no note in view. The question is support by the memory. Showing the note would measure agreement
  with the reader's own note, which often already contains the candidate.
- **V+n (secondary):** the same, with `A previous reader wrote this note: <frozen note>` before the question.
- **Paraphrases P2, P3 (robustness, never selected):** two rewordings of the instruction and labels, fixed in the code
  before launch. They are reported on all folds beside V. V stays primary whatever they show.

## 3. Conditions (Stage 3), all on the same fixed-candidate rows

| | decision | threshold |
|---|---|---|
| C0 natural | emit B's candidate iff the cached answer was not a refusal | none |
| C1 forced | emit every candidate (B's own coverage) | none |
| C2 self-check | emit iff the V argmax label is A | none |
| C3 verifier gate | emit iff s ≥ τ | matched coverage (primary) or τ from dev (Claim 4) |
| C4 confidence | emit iff m ≥ τ, where m = log P(candidate tokens) − log P(` unknown`) after the arm-A prefix (note + `Answer:`), from vLLM prompt logprobs | same as C3 |
| C4′ | the §34b final-layer ` unknown` rank (free; already computed) | same as C3 |

C0 and C1 in their original end-to-end form are the §35 numbers. They are quoted apart and never pooled with the
fixed-candidate analysis.

## 4. Claims (3B, confirmation folds 2–4; fixed-sequence testing: each is tested only if the previous one is SHOWN)

- **Claim 1 (H1, discrimination).** AUROC of s for B-candidate judged correct vs judged wrong, over answerable
  fixed-candidate rows. SHOWN if the lower bound is above 0.5. Practically meaningful if the point estimate is
  ≥ 0.65 (fixed now, just above the natural flag's dev 0.62). The PR-AUC and prevalence are reported beside it.
- **Claim 2 (H2, better choices than the model's own refusals).** At C0's coverage on the same rows (C3 emits the
  same number of candidates as C0, ranked by s; ties broken by question id), paired risk(C3) − risk(C0). SHOWN if
  the upper bound is below 0. No threshold is fitted, so no tuning can leak.
- **Claim 3 (beyond simple confidence; the note's §9 criterion).** At the same coverage, paired risk(C3) −
  risk(C4). SHOWN if the upper bound is below 0. Minimum meaningful difference 0.05 (fixed now).
- **Claim 4 (H3, recovery under a risk cap; descriptive with intervals).** τ is fixed on folds 0–1 as the lowest
  threshold whose dev risk is ≤ **r\* = 0.30** (about the 7B's natural dev risk, 0.316: "answer as reliably as the
  7B"). On confirmation, report coverage, risk, the number of cached 3B refusals recovered correctly, and the number
  of new wrong answers. C4 gets the same τ procedure.
- **Evidence dependence (Controls; pre-registered, both readers).** (A) The paired drop s(intact) − s(cut) for the same
  B candidate, on rows judged correct. A positive lower bound means the readout uses the evidence. Compared 3B vs 7B.
  (A′) AUROC of intact correct-B rows vs Cut-F rows (supported vs unsupported). (B) The counterfactual. Rows whose
  gold is a number or a name that occurs verbatim exactly once in the evidence span (the Executor counts this on dev
  folds before launch). The span is substituted with another value of the same kind: number ± a fixed offset, or a
  name taken from another question's gold of the same type. Then the original value and the substitute are both
  verified in both contexts. "Follows the evidence" means s(substitute | counterfactual) > s(original |
  counterfactual) and the reverse in the original context. Descriptive if fewer than 20 confirmation rows qualify.
  Each counterfactual prompt is checked by the same exact-span assertion as `ablate_evidence`, and 10 are checked by
  hand.
- **Length control (descriptive).** Cut-B with the removed span replaced by a length-matched irrelevant memory line
  from another question.
- **Stage 5, 7B (descriptive, judge-circularity caveat).** The 7B verifies the **same 3B candidates** (V, Cut-B,
  Cut-F, counterfactual). That isolates verification from generation. Separately, the 7B's own B candidates go
  through its own V. The two comparisons are never merged into one size conclusion.
- **Costs beside every claim.** Wrong answers added relative to C0, the unsupported-answer rate on Cut-F and Abs,
  and tokens and seconds per verification.

**Prediction, recorded now.** My §35 prediction was too pessimistic by 2–3×, so this one carries wide uncertainty.
From §34c (the 3B's abstention signal ignores the evidence), I expect the following. Claim 1 SHOWN but weak: AUROC
about 0.60–0.68, near the natural flag. Claim 2 at the edge of power. Claim 3 NOT SHOWN: the 3B verifier is no better
than candidate-vs-unknown confidence. Control A small for the 3B (mean drop under 0.1) and clearly larger for the 7B.
The 7B verifying 3B candidates is clearly better (AUROC ≥ 0.75). If so, the reading is the note's Outcome B for the
3B and H4 in the 7B's favour: the 3B cannot check its own work against the evidence, and a stronger checker can. That
points to an architecture question (small generator + larger verifier), not a prompt fix.

**Readings, fixed before the data.**
- Claims 1–3 SHOWN and Control A positive: the note's Outcome A. The 3B holds an elicitable, evidence-grounded support
  judgment that its answering policy ignores. Stated as externally elicited (a new prompt). It does not show that the
  original reader "knew".
- Claim 1 SHOWN, Control A ≈ 0 for the 3B: the readout is plausibility, not verification. Gating is then a confidence
  trick that will fail on evidence-free prompts. Cut-F answers would be emitted at a high rate.
- Claim 1 NOT SHOWN: evidence evaluation is itself part of the 3B's bottleneck (Outcome B). Permissive policies like
  §35's cannot be made safe with the 3B's own judgment.
- 7B ≈ 3B as verifiers on the same candidates, and the 3B's natural policy is worse: the note's Outcome E, the
  strongest behavioural evidence for a translation gap.

## 5. Caveats fixed now

The verifier prompt is new computation. A positive result is "elicited verification", not a passive readout. The labels
come from a single 7B judge (understates accuracy by about 0.1; audited above), and that judge shares weights with the
7B verifier. B candidates exist only for marked rows (3B: 165 of 204 on confirmation); marker-less outputs are outside
and counted. The exact-span cut can leave derivable content, so "unsupported" is by construction. Counterfactual
substitution covers only strict-token, verbatim golds, a subset biased toward D1-type questions. Risk and coverage are
on fixed candidates. They are not the deployed end-to-end accuracy (§35 gives that). No confirmation-fold verifier
number is read before the Reviewer accepts this document. The report script refuses folds 2–4 without `--confirm`.

## 6. Commands and spend (NEEDS SPEND; approval pending)

The Executor builds `memctl/analysis/verification.py` (prompt builder, label scoring, counterfactual substitution),
`scripts/exp36_verify.py`, `scripts/exp36_pod.sh` (3B then 7B; V, V+n, P2, P3, Cut-B, Cut-F, Abs, counterfactual,
length control, and C4 prompt logprobs on folds 0–4) and `scripts/exp36_report.py` (`--folds 0,1`, then
`--folds 2,3,4 --confirm`, then all folds). Volume: about 330 rows × up to 9 single-token prompts per reader ≈ 3,000
prompts of ~1,000 tokens each, plus 330 prompt-logprob passes. These are all prefill-dominated with prefix caching, and
there is no judge pass (labels exist). The standing pre-launch rule applies: a full-size stub run, peak RSS, and a cost
from the measured per-call latency. One A40 secure pod at the then-current price (§35: $0.59/h), about 25 min
expected; **hard stop 60 min (≈ $0.59)**. Programme spend so far ≈ $17.17 of the $20 cap.

**Triage order if the stub latency exceeds the stop:** drop the paraphrases first, then V+n and the length control,
then the 7B's own-candidate pass. Never drop the 3B's V, Cut-B, Cut-F or C4, or the 7B's V on 3B candidates.
