# Review: §44 rev 1 conditions S and U on Qwen 3B (pooled, 330 rows) — 1ada828

Advisor results review. Pre-registration: `research/proposals/2026-10-29-exp44-patching.md` rev 1 §C with revs 2–6.
Every number below was recounted from `runs/exp44_swap/qwen3b/{S,U}/*.json`; where my bootstrap differs from the
report's it is noted. Verdict: **ACCEPT the report as the pre-registered outcome. S0 FAILS, so the swap manipulation
is not valid on this design and S1–S3 are not read. U1 reads "not the format".** Per the user's instruction
(D-45) the result is documented, not repaired.

## Pre-registered readings

| claim | pre-registered rule | result (recount) | reading |
|---|---|---|---|
| S0 | following among the 3B's successes ≥ 0.80, LB ≥ 0.70 | 34/46 = 0.739 [0.609, 0.848]* at every margin (0.5 / 1 / 2 nat) | **FAIL → swap not valid** |
| S1 | read only if S0 holds | 43 failures: answer-production 0.674, evidence-use 0.070, indeterminate 0.256 | not read (descriptive: indeterminate > 0.20 would fail S1(a) too) |
| S2 | each class ≥ 8 refusal rows | answer-production refusals 11 (push opens 0.455), evidence-use 1 | not read (class too thin) |
| S3 | read only if S0 holds; donor bound | d_T forward 0.889 (n 63; all rows in [0.556, 0.917]), reverse 0.889; donor primary fails (max |mean| 1.22 nat, layers 27–35), fallback passes (ratio 0.04–0.07 from l0 = 32); random ≤ 1.9 | not read (descriptive) |
| U1 | ≥ 0.50 → format artefact; ≤ 0.25 with UB < 0.35 → not the format | 21/92 = 0.228 [0.152, 0.315]; still refuses 39/92 = 0.424; Steer-under-U 35/92 = 0.380; paired difference +0.152 [0.065, 0.239] | **"not the format"** |

*Report CI [0.609, 0.870]; mine [0.609, 0.848] with seed 0, 10,000 draws. Same lower bound; both fail.
Swap-subset U1 7/18 = 0.389 [0.167, 0.611] "partial", descriptive. No row has a missing marker (report's "5" refers to
the population scope; the 92 U rows all carry one).

## What U1 says

Removing the "unknown" option from the instruction, with the model's own note frozen, turns 23 % of the Qwen 3B's
in-view refusals into the gold and leaves 42 % still refusing. The §38 push under the same wording adds 15 points on
top. So the refusal is **not mainly a prompt artefact** of the "unknown" option (Ling et al. 2026's mechanism is at
most a quarter of it here), and the push does something the wording does not. This is the one clean reading of the
run. Pooled over all 330 rows as authorised in rev 4.

## Why S0 failed — post-hoc, exploratory, not a reading

The 12 non-following successes split three ways (surface type → following among successes: name 12/14, currency 2/2,
duration 6/7, bare number 10/15, **other 4/8**):

1. **Type-incompatible substitutes (7 rows).** The "other" rule drew the substitute from another row's gold of the
   same *question* type, which produced answers the question cannot take: "every week" → "under my bed",
   "Marketing specialist" → "Japanese short-grain", "Mod Podge" → "The GR-90 trail", "Chapter 4 of Book 1" → "I
   recommended learning", "The Glass Menagerie" → "Mindful.org.", "Spotify" → "Emma", and a year "2014." → "3,000
   miles" (bare-number subtype, but a year is not a distance). A model that refuses to copy an impossible answer is
   behaving correctly; these rows test the substitute, not evidence use.
2. **Generic first tokens (overlaps with 1).** The readout is the first divergent token. For "2014." it is "2", which
   any year shares; for "The Glass Menagerie" it is "The"; for "Chapter 4 …" it is "Chapter". On those rows the gold's
   log p under the swap is ≈ 0.0 (2bf43736, 5809eb10) because the first token is predicted by the question, not the
   evidence. The rev 3 prefix rule handles a prefix shared *with the substitute*, not a first token that is generic
   on its own.
3. **Small bare numbers (4 rows):** "five" → "four", "3" → "25", "2" → "18", "four" → "five"; Δ moves by 0.2–1.0 nat
   only. Plausibly counting answers or digits present elsewhere in the prompt in another form.

The one duration miss (19b5f2b3, "two weeks" → "three months") has Δ_clean = −0.3: the model's success was never
carried by the read token, so the row is uninformative either way.

On name, currency and duration rows alone, following is 20/23 = 0.87; **this is a selection after the outcome and
cannot rescue S0**. It says what a repaired design would need: substitutes of the same answer type with a
non-generic first content token, which the pre-registered rules did not guarantee. Whether to run such a rev is the
user's decision; nothing is proposed here.

## Descriptive notes for the record

- Classes at 1 nat: successes 34 / 4 / 8 (answer-production / evidence-use / indeterminate), refusals 13 / 1 / 7,
  wrong 16 / 2 / 4. Within the swap rows the 3B's refusals and wrong answers are mostly on rows where the model *does*
  follow the evidence (29 of 43), which is the direction the thesis predicts, but S0's failure means the following
  test itself is not trusted on this subset.
- d_T 0.889 on 63 rows with a degenerate CI (the median sits on a layer boundary, 32/36); rev 0 d_c on the same rows
  0.778 (n 26), logit lens 0.917, J-lens 0.889. Evidence-specific transfer completes later than clean→cut recovery,
  consistent with the "both partly right" row of the §C table; descriptive only.
- The rev 6 controls behaved: donor difference and random perturbation ≤ 0.05 × the forward shift at d_T; the donor's
  late-layer mean shift (up to 1.22 nat) exceeds the ±0.5 bound but is under the 2-nat "still disruptive" flag.
- Cost: S 2.7 h, U 1.4 h, laptop only, $0.

## Rulings

1. Report 1ada828 ACCEPTED as the pre-registered outcome. EXPERIMENTS §44 must state S0 FAIL / swap not valid /
   S1–S3 not read, and U1 "not the format", in that order, before any descriptive line.
2. The post-hoc diagnosis above goes into §44 under a heading that says post-hoc and exploratory, with the 20/23
   figure labelled as selected after the outcome.
3. No new intervention and no rerun of S without the user's decision (D-45 stopping rule).
4. Llama 3B S/U continues as ordered (its S0 is read the same way); then the rev 0 confirmation reads (3B, then 7B).

## Addendum: EXPERIMENTS.md §44 text at 117955b — ACCEPT with two wording edits

Status block order is as ruled (S0 FAIL → swap not valid → S1–S3 not read → U1 → the rest). The rev 0 development
table may stay above the rev 1 block: it is a different condition, and ruling 1 concerned the S/U result. Numbers
checked against the files: 34/46, 0.739, classes 34/4/8, 13/1/7, 16/2/4, 29/43, U 0.228 / 0.424 / 0.380 / +0.152,
rev 0 table (30/52, 38/59, 5/7, 0/3, 52/56, 0.736, 0.679, 0.022), d_T 0.889, d_c 0.778, lenses 0.917 / 0.889. Two
edits: (1) write "21 of 92" beside 0.228 in the Status block; (2) the "d_c 0.778, n 26" beside d_T is the median over
all patched development rows of the swap set, not filter rows (filter rows alone: n 17, same median 0.778) — say
"26 development rows, unfiltered". With those, §44 is accepted as the in-progress text; it closes when Llama 3B S/U
and the rev 0 confirmation reads are in.
