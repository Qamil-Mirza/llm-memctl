# §53 arm C: hand audit of the LongMemEval releases (Advisor, 2026-11-07)

**Status: descriptive audit of stored outputs. $0, no run, no new model call.** Data: LongMemEval dev folds 0–1, so
nothing held out was touched. The audit does not revise STATE by itself. The decision rule below was fixed in
`research/briefings/2026-11-07-advisor-next-directions-response.md` §5 before any output was read.

## Result in one paragraph

The containment grader did miss some correct answers: 3 for each edit, plus one borderline for D-target. It also
credited some wrong ones: 2 for A-target, 1 for D-target. These are a minority of the non-gold releases (3 of 12 for
A-target, 3–4 of 12 for D-target), so **under the pre-set rule the arm C conclusion stands.** Most of the in-view
refusals the edits release do not become correct answers.

After the audit, the correct releases are:
- **A-target:** 9 of 20 releases (d1 said 8);
- **D-target:** 10 of 19 releases (d1 said 7; 9 without the borderline case).

That is 9–10 of the 26 in-view refusals. The rest are mostly degenerate or off-format outputs (5 per edit) and
reasoning errors in counts, dates and orderings (3–4 per edit). Hallucinated values and binding errors: none.

## Method

- **Rules.** Written before any released output was read (Advisor scratchpad `lmeaudit/RULES.md`). The categories:
  - **G:** correct and credited by d1.
  - **G-fp:** credited by d1 but wrong.
  - **(i):** correct, missed by containment.
  - **(ii):** binding error.
  - **(iii):** hallucinated.
  - **(v):** reasoning error on an aggregate or derivation.
  - **(iv):** degenerate, off-format, or a refusal in other words.
- **Population.** The 26 B0 in-view refusals. Every release was audited: 20 by A-target and 19 by D-target, 39 in
  all. This matches the report: 20/19 releases, 8/7 credited by d1.
- **Blinding.** The 39 items were shuffled (seed 531) and the arm labels moved to a separate key file. One reader
  classified all of them against the gold and the decoded prompt, then unblinded. Items and labels are in the
  scratchpad: `lmeaudit/items.json`, `labels.json`, `KEY_arms.json`.
- **Grader reminder.** d1 credits an answer when the gold string, or every gold token, occurs in it. A refusal is an
  answer that starts with "unknown".

## Counts

All 39 releases, by category:

| Edit | Releases | d1 gold | G | G-fp | (i) | (i)* borderline | (v) | (iv) | (ii) / (iii) |
|---|---|---|---|---|---|---|---|---|---|
| A-target (two-head ablation) | 20 | 8 | 6 | 2 | 3 | 0 | 4 | 5 | 0 |
| D-target (rank-1 L35) | 19 | 7 | 6 | 1 | 3 | 1 | 3 | 5 | 0 |

The decision rule applies to the non-gold releases only:

| Edit | Non-gold releases | (i) among them | Share |
|---|---|---|---|
| A-target | 12 | 3 | 0.25 |
| D-target | 12 | 3 (4 with the borderline) | 0.25–0.33 |

Both shares are below one half, so the rule says the conclusion stands.

## What the categories contain

- **(i) missed by containment:**
  - "25000" for "$25,000" (d1 splits on the comma);
  - "6" and "6 days" against the gold string "6 days. 7 days (including the last day) is also acceptable";
  - "has a blue scaly body" against "had a blue scaly body", twice;
  - a paraphrase of "untidy, stained white shirt".
  - The borderline (i)* is "Snaggletooth" for "a blue Snaggletooth".
- **G-fp, credited but wrong:**
  - "5.73 weeks" against 5.5. The gold tokens "5" and "weeks" both occur.
  - Two runaway dumps of memory lines in which the gold appears only inside a quoted line, with no answer given.
- **(v) reasoning errors:** day counts (21 for 29, 1 for 2, 35 for 1), a sum ($55 for $15 + $50), and a which-first
  comparison answered with the later event.
- **(iv) degenerate or off-format:**
  - the literal template placeholder "<short phrase>" (3 releases, two of them on preference questions);
  - "1, 2, 3" (twice);
  - "user";
  - a date given where an event was asked (twice; one is the question's own timestamp);
  - the correct order given only as three dates, without naming the events;
  - a wrong order followed by a runaway dump.
- **Credited gold (G) of note:** on one question B0's note said the rice type was "not mentioned", and an edit
  answered "Japanese short-grain rice", which is correct and in the history.

## The 12 abstention (`_abs`) rows

The report counts 6 answers for B0 and 4 for each edit. Read by hand, most of those "answers" are not inventions:
- note-like dumps with no answer marker;
- "user did not mention baking egg tarts", a refusal in other words.

The substantive unsupported answers are the same in all three arms: "Harvard University", and a count of 12 (10 for
D-target). The edits remove two dump-style outputs and add no new invention on these rows.

## Reading

- **Releasing the answer is not the bottleneck on natural data.** About a third of the in-view refusals become
  correct answers under either edit (9–10 of 26). The rest become reasoning errors or format breakdowns, not
  hallucinations or binding errors.
- **The format breakdowns are edit-induced.** These are the "<short phrase>" placeholder, "1, 2, 3" and runaway
  dumps.
  - This fits the edits' synthetic origin: they push the first answer token away from "unknown", but they cannot
    supply the multi-step reasoning these LongMemEval questions need.
- **The grader errors run both ways and roughly cancel.** For A-target the net change is +1 (+3 missed, −2 false
  credit). For D-target it is +2 or +3. No d1-based count in §53 changes its direction.
- **Small sample, one reader, dev rows only.** Descriptive.
  - The (i) shares have wide 95% Wilson intervals: 3/12 is 0.09–0.53, and 4/12 is 0.14–0.61. Both cross one half,
    so the decision rule's "minority" outcome is a point-estimate reading, not a statistically secure one.
  - The correct-release rate, 9 of 26 in-view refusals, is 0.19–0.54.
  - The two edits share the same 26 questions, so they are not independent samples.
  - What the sample does support is the *kind* of failure: no hallucinations or binding errors, and mostly
    format breakdowns and reasoning errors. An output-side push cannot repair those.
  - The closure of the line rests mainly on F18/F19, not on this audit: the B1 tie, selectivity explained by the
    fit, and too little headroom to measure.

## Consequence under the pre-set rule

- (i) is a minority for both edits, so the §53 arm C conclusion stands and the Qwen weight-editing line closes. No
  §54 is opened.
- **F19 wording.** F19's sentence "only 8 and 7 of those contain the gold" is accurate as a d1 count. If the user
  wants the audit reflected, an optional descriptive clause could follow it: "a blinded hand audit finds 9 and 10
  correct, with the rest mostly degenerate outputs and reasoning errors".
  - That would need the user's "Go". It is not required by the rule.
