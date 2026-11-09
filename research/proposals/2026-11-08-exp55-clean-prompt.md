# §55 (proposed): how much false abstention survives a clean, consistent abstention prompt?

**Status: proposed, rev 0 (Advisor, 2026-11-08). Not built, not run, nothing spent.** Laptop only, $0. Needs the
user's approval before any build. The section number is provisional until the Executor opens it in EXPERIMENTS.md.

## Why

Every false-abstention result since §35 uses one frozen reader prompt (7b5fc30). That prompt gives two different
conditions for saying "unknown":
- the task line: "If the memory does not contain the answer, reply: unknown";
- the note instruction: "Reply unknown only if the memory has nothing at all on the topic."

The record already shows that the option matters a great deal on controlled tasks:
- §53 B-opt: with both "unknown" sentences removed and the model's own note kept, the unedited model answers 145 of
  150 sampled false abstentions correctly (EXPERIMENTS §53).
- §51 B3 on the sealed split H: the same removal releases 0.977 of false abstentions, but the model then answers
  0.682 of unanswerable rows (B0 0.035).
- On real chats the picture differs: under the no-"unknown" control, the 3B gets the gold on only 0.228 of its
  in-view refusals (STATE F12).

Outside work names this effect "abstention inflation": adding an "Unknown" option makes models abstain on questions
they can answer, and the answer survives in mid layers before late layers override it (arXiv 2507.16199, EMNLP 2026).
That is close to our §47 / F15 and §49 / F16 picture.

So before more mechanism work, or a cross-model table built from cached outputs under the same prompt, the
programme needs to know how much of the failure is created by the inconsistent wording, how much by the option
itself, and what the option buys on questions that have no answer.

## Question

With the evidence, model and decoding fixed, how do the false-abstention rate on answerable questions and the
answer rate on unanswerable questions change between:
- **P0**, the current prompt (byte-identical, 7b5fc30);
- **P1**, a clean prompt with one consistent abstention rule;
- **P2**, the same prompt with no "unknown" instruction at all?

## Prompts (fixed now, before any output)

All three keep everything else in the reader prompt byte-identical: the memory format, the 80-word note, date and
advice sentences, and the `Answer: <short phrase>` line.

| Arm | Task line, last sentence | Note instruction, abstention sentence |
|---|---|---|
| P0 | "If the memory does not contain the answer, reply: unknown" | "Reply unknown only if the memory has nothing at all on the topic." |
| P1 | "If the memory does not state the answer, reply: unknown" | "Reply unknown only if the memory does not state the answer." |
| P2 | (removed) | (removed) |

P2 is the §51 `prompt_b3` text. The Executor builds P1 and P2 from `prompt` by exact string replacement and asserts
that nothing else differs (diff = the two sentences). Prompt shas are committed in the freeze.

## Data

**W, controlled (§54 split, seed 55, development).**
- Bases with even `base_index`: 75 bases, all three tasks.
- Rows: ANS (450), UNA-rel (225) and UNA-swap (75). CF rows are not used.
- P0 on ANS reuses §54 B0 (same engine, prompt and greedy decoding; byte-identical text is asserted on a 20-row
  rerun). P0 on UNA rows is new.

**LongMemEval dev folds 0–1 (real chats, descriptive).** The §52 rows (`runs/exp52/rows52.jsonl`): 126 answerable rows
with all evidence in view, 62 answerable but not all in view, 12 unanswerable. P0 is the cached Qwen 2.5 3B §23 output.
§52's pipeline check confirmed prompt equality on the 126 primary rows; the Executor checks the 74 others the same way,
and reruns P0 on any row without a cached output. P1 and P2 are new.

No held-out split is touched. Seed 56 stays sealed.

## Runs

1. **Free generation**, P1 and P2 on all listed rows; P0 on the new W UNA rows. Greedy, 256 new tokens, Qwen 2.5 3B
   Instruct fp32 on the laptop CPU, the §51 engine. The model writes its own note under each prompt.
2. **Controlled prefix** (cheap, one answer step per row), W only: each prompt's instruction followed by the model's
   own P0 note and marker, as §51 B3. This separates the instruction's effect at the answer step from its effect on
   what the note says.

## Measures

Graded with the §51 grader (head + tail where a prefix is forced):
- **FA:** the share of answerable rows answered "unknown".
- **Gold:** the share of answerable rows answered with the gold.
- **Uns:** the share of unanswerable rows (UNA-rel, UNA-swap; LongMemEval `_abs`) given an answer.
- **Dam (controlled prefix only):** P0-correct rows that become wrong.
- **U(λ = 1)**, as in §51.
- Note-then-refuse share among P1 abstentions (descriptive).

Paired differences between arms are reported with 95% bootstrap intervals clustered by base (W) or question
(LongMemEval).

## Readings, fixed now

**R1, the inconsistency (P1 against P0, W free generation):**
- FA(P1) ≤ ⅓ FA(P0): "the false abstention on controlled tasks is mostly created by the inconsistent wording."
- FA(P1) ≥ ⅔ FA(P0): "the inconsistency is not the main cause."
- Otherwise: "partly."

**R2, the option (P2 against P1, W free generation):** report FA, Gold, Uns and U(λ = 1) together, as a trade-off.
- If FA(P2) ≤ ⅓ FA(P1) and Uns(P2) − Uns(P1) has a lower bound > 0, the reading is "the false abstention is largely
  the price of having an abstention option".
- No reading uses FA alone. Removing the option is never called a fix unless Uns also stays within 0.05 of P1.

**LongMemEval:** the same quantities, descriptive only (12 unanswerable rows is too few for a claim).

## What each outcome changes (decided now)

- **R1 "mostly created":** every controlled-task result in §49–§54 gets a scope note ("under an inconsistent
  abstention instruction"), and any later controlled-task work uses P1.
- **R1 "not the main cause", with R2 as above:** false abstention on these tasks is framed as abstention inflation, a
  calibration trade-off, and the thesis cites arXiv 2507.16199 as the closest prior work.
- **In every case:** the cross-model table from cached outputs is not built on P0 alone. Whether to rerun it under
  P1 is a separate decision.

## Timing (to be measured before any duration is quoted)

New free-generation rows:
- W: 750 (P1) + 750 (P2) + 300 (P0 UNA);
- LongMemEval: 200 (P1) + 200 (P2).

§54 B0 ran at 18.1 s per W row. LongMemEval rows are longer and must be timed separately. The Executor times one row
of each kind before quoting. The controlled-prefix arm is one answer step per row. If the measured total exceeds 24 h,
LongMemEval P2 is dropped first, then W UNA-swap.

## Rules

- Laptop only, $0. No pod, no API.
- Dev data only. D-45: a failed check is documented, not rescued; no prompt is tuned after outputs are seen.
- The freeze (prompt shas, row lists, grader version, readings) is committed before any model output.

## Confounds stated in advance

- One model (Qwen 2.5 3B). The question here is the prompt, not the family.
- P1 is one clean wording out of many. A different consistent wording could give a different rate. P1 is chosen
  because it keeps P0's task-line condition and makes the note instruction agree with it.
- Free generation lets the note change with the prompt. The controlled-prefix arm is there to separate that.
