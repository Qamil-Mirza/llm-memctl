# §54 note audit: do note-driven twin flips drop the evidence? (Advisor, 2026-11-08)

**Status: descriptive audit of stored B0 outputs (71b9ce7). $0, no model call, dev seed 55 only.** Rules were written
before reading the notes (Advisor scratchpad `noteaudit/RULES.md`); one F note had been seen earlier while checking the
record format. One reader. No causal claim.

## Question

In the 43 note-driven pairs of §54, the abstaining twin (F) answers once given the answering twin's (A) note. Did F's
own note drop or corrupt the evidence (a failure of evidence use at the note step), or did it hold the evidence?

## Result

| F's own note (hand label, all 43 read) | Count | Share |
|---|---|---|
| N2 complete: the facts the answer needs are there, stated correctly | 29 | 0.67 |
| N0 omitted: a needed fact is missing (6 of 8 also say "not mentioned") | 8 | 0.19 |
| N3 present but denied: the fact is written, then the note says it is missing or unknown | 6 | 0.14 |
| N1 altered (wrong value) | 0 | 0 |

All 43 A notes hold the needed facts.

**Under the pre-set reading, N2 is the majority: in most note-driven flips the note holds the evidence, and the
refusal is still sensitive to the note's form.** That is not an evidence-use failure. A third (14 of 43) are
note-step failures: 8 drop the fact, 6 write it and then deny it.

## What the categories look like

- **N2.** Often a trivially different wording. F: "Sarah lives in Cardiff." → "unknown"; A: "Sarah lives in Cardiff
  as stated on 2023/03/08 (Wed) 10:00." → "Cardiff". Several F notes are a complete restatement of the gold fact
  alone.
- **N0.** Mostly relational and update questions. RC: "Sarah's supervisor is Kate …, but there is no information
  about the city where Kate works" (Kate works in Derby is in the memory). TU: "… all mentioned their current
  locations, but not Peter's."
- **N3.** Mostly update questions (TU 5 of 6). "James lived in Chester initially, but the memory does not specify
  where he lives now after moving to York."

By task: DE 13 N2 / 1 N0 / 1 N3; RC 8 / 4 / 0; TU 8 / 3 / 5. Note-step failures cluster in the two-hop and update
tasks.

## Wider picture (automatic labels, descriptive)

The automatic rule (the gold string is in the note; denial phrases flag N3) agrees with the hand labels on 42 of 43
(it misses one N3). It cannot see binding errors, so its N2 is an upper bound.

| Group | N2 | N0 | N3 |
|---|---|---|---|
| 21 primary pairs, F's own note | 18 | 2 | 1 |
| All 288 false-abstention rows in W | 255 (0.89) | 23 | 10 |
| By task, all FA: DE / RC / TU | 170 / 66 / 19 | 10 / 9 / 4 | 1 / 3 / 6 |

## Reading

- On this synthetic data, most false "unknown" answers come with the evidence written in the model's own note:
  about 9 in 10 by the automatic rule. The failure is mostly after the note, at the answer step, consistent with
  §49 (note-then-refuse) and F15.
- Within the twin flips, the answer step depends on the note's surface form, not just its content: the same fact
  in slightly different words flips the decision.
- Evidence dropped or denied at the note step does happen (about 1 in 9 FA rows by the automatic rule, 1 in 3
  note-driven flips). It is concentrated in two-hop (RC) and update (TU) questions. That is the clearest
  "evidence not used" population we have, and it is small.
- Caveats: one reader; F2 synthetic data on Qwen 2.5 3B only; the note is a self-written summary, so "the note
  holds the fact" does not show the model's internal computation used it.
