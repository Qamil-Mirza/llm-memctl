# Advisor briefing: where the 3B reader loses the answer, and what moves it

Prepared 2026-10-25 for the meeting of 2026-10-26. Every figure is from EXPERIMENTS.md (§23, §34–§41) with a review
in `research/reviews/`. Intervals are 95% bootstrap. "3B" and "7B" are Qwen2.5-3B-Instruct and Qwen2.5-7B-Instruct,
reading LongMemEval questions with the gold evidence in view.

## 1. Starting point (§23): the controller is not the bottleneck, the reader is

With all the right evidence in the prompt, the 3B answers correctly on 0.397 of questions and the 7B on 0.667. The 3B
says "unknown" on 0.372 of answerable questions. A memory controller can only choose what to show; here the model is
shown the answer and still fails. So the question became: where inside the 3B does the answer get lost?

## 2. The two probes

**Logit lens.** A transformer keeps one running vector per token position (the residual stream) and updates it layer
by layer; only at the last layer is it turned into word scores by the unembedding matrix. The logit lens applies that
same unembedding to the vector at an earlier layer, as if the model had stopped there. It answers: "if the model had
to speak now, what would it say?" It needs no fitting. Its weakness is that early layers do not write in the final
layer's coordinates, so early readings are noisy.

**Jacobian lens (Anthropic's J-lens).** Instead of pretending layer l is the last one, it estimates how the final
vector moves when layer l's vector moves (the expected Jacobian J_l, averaged over ordinary text), and reads
unembed(J_l · h). It is a linear fast-forward from layer l to the output. We fitted one per model on 56 WikiText
paragraphs. Before using it we set a gate on development folds: the J-lens had to rank the gold token better than the
logit lens over the middle third of layers, and beat it on the lens library's own test sets. Both lenses passed the
first test by a wide margin (best rank about 17× better for the 3B, 48× for the 7B). The 3B's lens failed the
library's test sets, so by the pre-registration the 3B's claims were decided on the logit-lens rows with the J-lens
shown beside; the 7B's passed.

**How we used them.** Each cached prompt was rebuilt byte for byte and "Answer:" appended. At the position just before
the answer's first content token we read, at every layer, the rank of the gold token among all vocabulary tokens.
Each question was classed A (gold never in the top 10 at any layer), B (in the top 10 at some layer, then gone at the
last), or C (in the top 10 at the last layer). We also read the end of each evidence line and the question.

## 3. What the lenses showed (§34, §34b, §34c)

| finding | 3B | 7B |
|---|---|---|
| failures where the gold is in the final top 10 (class C) | 0.638 (0.551, 0.717) | 0.746 (0.635, 0.841) |
| failures where the gold never appears (class A) | 0.346 (0.268, 0.433) | 0.127 (0.048, 0.206) |
| "rises then gets overtaken" (class B) | 0.016 | 0.127 |
| median depth at which the gold enters the top 10 (fraction of layers) | 0.889 (q1 0.590) | 0.464 (q1 0.250) |
| gold rank separates right from wrong answers (AUC) | 0.654 (0.576, 0.727) | 0.692 (0.617, 0.766) |

- **The prediction was wrong.** We expected B: the answer rises mid-stack and "unknown" overtakes it. Instead most
  failures, including 0.569 of refusals, are C: the model has the gold in its top 10 at the very end and still says
  "unknown". A floor of about a third of failures is A: the answer never appears.
- **Emergence is late in the 3B.** Where the gold appears at all, the 7B has it by mid-stack (median depth 0.46) and
  the 3B only near the end (0.89). The answer is not readable at the evidence lines or the question in either model.
- **§34b: is the late emergence an artefact of forcing "Answer:" right away?** No. Read at the position where the 3B
  actually answered after its own note, the depth is the same (median 0.917 either way). Part of the absence is the
  forced format: presence rises from 0.69 to 0.78 on those rows. The lateness belongs to the model.
- **§34c: is "present at the end" real extraction?** Only partly. Cutting the evidence from the prompt removes the gold
  from the top 10 on about a quarter of questions and worsens its rank about 4×, so the readout does measure evidence
  use. But about 0.6 of the 3B's C failures keep the gold in the top 10 with no evidence at all: small numbers and
  common names are plausible for their questions. Of the 0.64 "present", only about 0.25 is present because of the
  evidence. The 3B–7B rank gap is the same with the evidence cut, so it is a ranking difference, not an extraction one.

## 4. Prompt-level evidence (§35–§37)

- **§35.** Freezing the 3B's own note and adding one line, "do not reply unknown", raises accuracy by +0.176 (+0.109,
  +0.242), closing 0.53 (0.33, 0.73) of the matched 3B–7B gap; forced choice gives +0.242. Cost: a fifth of the
  refusals turn into wrong answers, and without evidence the answered-and-wrong share doubles. The 7B gains almost
  nothing. So about half the gap is answering policy, not reading.
- **§36b, §37.** The 3B cannot sort its own forced answers by checking them against the evidence (AUROC 0.573, one
  shot; 0.505 in a three-turn Socratic chain). Its support score detects whether evidence is present, not whether the
  answer matches it. The 7B checking the 3B's answers reaches 0.692, but the 7B also produced the labels, so that is
  circular. Self-verification is closed as a route.

## 5. The refusal direction (§38–§40)

| experiment | what | opens refusals | right, evidence in | right, evidence cut | instruction line |
|---|---|---|---|---|---|
| §38, LongMemEval sealed folds (60 refusals, pod) | layer-29 direction, every token | 54/60 | 0.517 | 0.183 | 0.433 |
| §40, LoCoMo, setting fixed in advance (120 refusals, pod) | same vector, nothing re-chosen | 0.758 | 0.358 (0.275, 0.450) | 0.025 | 0.217 |

- **§38.** We took the average difference of the layer-29 residual between "answer" and "refuse" prompts on 13
  development pairs, a direction with nothing question-specific in it, and added twice its unit vector at the colon
  of "Answer:" and at every generated token. On the sealed folds it opened 54 of 60 refusals; 0.517 came out right
  against 0.183 with the evidence cut (+0.333 (+0.183, +0.483)), above the §35 line's 0.433. A random vector of the
  same size opened 2 of 60. Damage to correct answers −0.056 (−0.130, 0.000). Caveat: this working setting (layer 29)
  was chosen on development labels after the pre-registered one (layer 25, a push twice as large) failed its
  specificity control by producing word salad; two settings were tested and the record says so.
- **§39 and §39b (development only, free).** Pushing only at the colon, never again, gives the same answer rate as
  pushing at every token (26 of 28 both) with the same first word on every row. Throwing the push away and feeding the
  model only the first word the pushed version would have said reproduces the pushed answers almost word for word
  (26 of 28 texts identical). The refusal is a one-token decision at the answer position; the pushed state computes
  the answer's own first word, and the plain model finishes from there. My prediction was the opposite both times.
- **§40.** On LoCoMo, a different benchmark no model in this programme had touched, with the vector byte-identical and
  nothing re-selected, the direction opens 0.758 of refusals, 0.358 right against 0.025 cut (+0.333 (+0.250, +0.425)),
  random 0.058, 14 points above the instruction line. The costs are larger than on LongMemEval: 8 of 40 already-correct
  answers are lost (−0.200 (−0.325, −0.075)); on 37 unanswerable questions the answer rate goes from 0.32 to 0.73 (the
  instruction line 0.54). With a penalty of one per wrong answer, the instruction line scores +4 and steering −1 on the
  120 refusal rows. One qualification of §39: on LoCoMo the every-token push beats the one-position push on content by
  +0.075 (+0.017, +0.142), with the first token identical on all 120 rows.

## 6. Selectivity (§41)

A rule fitted on 67 development rows (steer when the refusal margin drops sharply with the evidence cut) was useless on
LoCoMo: it fired on 11 of 120 rows with no better precision than chance. But the projection of the unsteered state at
the colon onto the direction ranks which opened refusals will come out right at AUROC 0.803 (0.724, 0.875); the
refusal margin (0.19) and its evidence sensitivity (0.23) do not. Post hoc, steering only the top 10–30% by that
projection would have scored +9 against the instruction line's +4. That is a hypothesis; §42 pre-registers it on fresh
rows.

## 7. How it fits together

The 3B's failure with the evidence in view is not lossy memory. The lenses show a candidate answer present at the end
of the network in most failures, emerging late (around 0.9 of depth) and only partly because of the evidence. A
content-free direction at layer 29 of 36 (about 0.8 of depth, where the late emergence sits) switches the refusal off
in a single token, and the answers that follow are read from the evidence (they collapse when it is cut) with about
one-in-three precision on a new benchmark. The same direction's projection says in advance which refusals are worth
flipping. What remains: the reading half of the gap (§34c's plausibility share, the class-A floor), whether the 7B has
the same disposition, and whether selectivity turns the direction into a policy that beats a one-line prompt.

## 8. Caveats to raise, and questions for her

- The judge is the 7B (agreement 0.971 with cached verdicts, 60/60 on §40's check rows) and it also labelled §35;
  an independent label source is the standing gap (Q15).
- One model family, one prompt family, one layer and strength; the 3B's J-lens failed its gate, so its lens claims
  rest on the logit lens; small n (28–120 rows per arm).
- §38's working setting was chosen on development labels; §40 is the clean test of it.
- The direction removes abstention where abstention is right (unanswerable questions 0.32 → 0.73). Is a one-in-three
  precision lift worth that, and under what utility?
- Questions: is the "late disposition over a present candidate" reading a known pattern in small instruction-tuned
  models? Is the plausibility discount (§34c) the right way to separate extraction from prior? Would she accept a
  7B-judged result for a thesis chapter, or should we budget for a human-labelled subset?

Spend to date: about $17.82 of the $20 programme cap. Code, rows and reviews are in `~/Code/llm-memctl`.

## Appendix (added 2026-10-26): the layer-by-layer profile of the gold answer at the answer position

Recomputed from the §34 read files (`runs/exp34_jlens/`), all five folds, the §34 population (330 questions with
all evidence in view, paired across the two Qwen readers), position P1, k = 10. Share of questions whose gold first
token is inside the top-10 of the lens at that layer, and the median rank among successes.

| depth | Qwen 3B layer (of 36) | logit: gold in top-10 | logit: median rank, successes | J-lens: gold in top-10 | unknown is top-1 on refusals (logit) |
|---|---|---|---|---|---|
| 0.03–0.78 | 0–27 | 0.00–0.02 | tens of thousands | 0.00 | 0.00 |
| 0.83 | 29 | 0.03 | 50,711 | 0.01 | 0.00 |
| 0.86 | 30 | 0.06 | 11,749 | 0.04 | 0.04 |
| 0.89 | 31 | 0.09 | 4,287 | 0.06 | 0.41 |
| 0.92 | 32 | 0.19 | 61 | 0.25 | 0.63 |
| 0.94 | 33 | 0.32 | 11 | 0.43 | 0.66 |
| 0.97 | 34 | 0.60 | 3 | 0.53 | 0.74 |
| 1.00 | 35 | 0.68 | 2 | 0.68 | 0.66 |

| depth | Qwen 7B layer (of 28) | logit: gold in top-10 | logit: median rank, successes | J-lens: gold in top-10 | unknown is top-1 on refusals (logit) |
|---|---|---|---|---|---|
| 0.04–0.14 | 0–3 | 0.00 | tens of thousands | 0.00 | 0.00 |
| 0.29 | 7 | 0.00 | 8,431 | 0.28 | 0.00 |
| 0.43 | 11 | 0.00 | 11,853 | 0.22 | 0.00 |
| 0.57 | 15 | 0.00 | 22,596 | 0.21 | 0.00 |
| 0.71–0.79 | 19–21 | 0.01–0.05 | 15,080 → 4,000 | 0.00–0.02 | 0.00 |
| 0.82 | 22 | 0.20 | 162 | 0.03 | 0.00 |
| 0.86 | 23 | 0.38 | 14 | 0.16 | 0.50 |
| 0.89 | 24 | 0.51 | 4 | 0.35 | 0.39 |
| 0.93 | 25 | 0.68 | 2 | 0.74 | 0.44 |
| 0.96 | 26 | 0.83 | 1 | 0.78 | 0.50 |
| 1.00 | 27 | 0.88 | 1 | 0.88 | 0.50 |

Readings (descriptive):
- With the logit lens neither model shows the gold anywhere in the first four-fifths of its layers. Emergence is a
  steep climb over the last four to six layers: layers 31–35 in the 3B, 22–27 in the 7B. The 7B's climb starts
  about one tenth of the depth earlier (0.82 against 0.89) and ends higher (0.88 against 0.68 of questions with the
  gold in the final top-10; top-1: 0.56 against 0.30).
- Once the gold enters the top-10 it stays: on the logit lens it is in the top-10 at 0.94 (3B) / 0.88 (7B) of the
  remaining layers and at the final layer for 0.96 / 0.99 of the questions that emerged.
- The "7B has the answer by mid-stack" sentence of §34 (median J-lens depth 0.464, here 0.50 on all folds) rests on
  a J-lens-only bump at depths 0.29–0.57 that holds the gold in the top-10 for about a fifth to a quarter of
  questions, vanishes again at 0.71–0.82, and is not seen by the logit lens. The J-lens emergence distribution for
  the 7B is bimodal (q1 0.286, q3 0.893). The robust cross-size statement is the late climb, earlier and steeper in
  the 7B; the mid-stack presence is a lens-dependent minority signal.
- On the 3B's refusals ` unknown` reaches top-1 at layers 31–32 (0.41 → 0.63 of refusals), one or two layers before
  the gold enters the top-10 (0.10 at layer 32, 0.57 at layer 35). At the final layer the gold sits at median rank 5
  under ` unknown` at rank 1. In the 7B's 18 refusals, ` unknown` and the gold rise together at layer 23.
- In failures the gold is in the final top-10 nearly as often as in successes on the 3B (answered wrong 0.71,
  refusal 0.57, success 0.73): the answer is readable at the end in most failures and the final choice goes elsewhere,
  which is the C class of §34.
- Caveat: the median emergence depths quoted in the main text (0.889 against 0.464) are J-lens figures on the
  confirmation folds; the logit-lens medians are 0.944 against 0.857. Both orderings agree; the sizes differ by lens.
