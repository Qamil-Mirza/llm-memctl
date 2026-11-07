# §53 (rev 1): are the two §51 edits specific, and do they carry over to real questions?

**Status: rev 1a (Advisor, 2026-11-07). Not reviewed, not built, not run, nothing spent.** Rev 0 (2026-11-05, D-75)
is in git history; the changes are listed under Revision. Decisions D-75, D-76.

## Opening (fixed now)

§51 closed on 2026-11-07 (D-76). The edit that met the repair criterion on H is submission 2 (rank-1 L35 `o_proj`,
`8364bf135ac59a83`, dose 2). Submission 1 (zero the `o_proj` columns of L32h2 and L33h10, `c64ea090781450cf`) passed
the branch M pathway tests (M1, M2) but is not selective at its M dose. Rev 0's opening rule ("submission 1 confirms
on H") is therefore read as not met, and the user chose this revision instead (D-76): §53 rev 1 tests both edits.

Nothing in §53 touches E, S, H or T, or changes §51's verdicts. §53 cannot turn §51's "B1 inconclusive" into a
result; any B1 number here is descriptive.

## Why

§51 tested overfitting to E (S, H, T), "another answer-more knob" (B1–B5) and "any change to these weights" (B6, for
the two heads only). Three alternatives are untested:

1. **General disruption (two heads).** Removing *any* two late heads might release false "unknown"s.
2. **The read side does not matter (L35 edit).** The L35 edit reads an "answer available" gate `w` and writes along
   city − unknown. If a rank-1 edit with the same write direction and size but a *random* read vector does as well,
   the gate is not what makes the edit selective, and the edit is a dressed-up output bias.
3. **Templates only.** Neither edit has been tried on natural questions.

A fourth gap: the editor's own activity predictions for submission 2 (how the gate responds to removed evidence,
distractors and so on) were never tested, because branch M covered submission 1 only.

## Data

- **N (new controlled split):** seed 54 (seeds 50–53 are used by §50 and §51). Same generator, tasks (DE, RC, TU),
  F2 format, prompt, grader and classes (FA, C, UNA-rel, UNA-swap, CF) as §51; B0 classes from the §51 grader.
  Never seen by the editor or the analysts.
  - **N-dev** (20% of bases): pipeline checks, timing and the random-edit dose calibration only.
  - **N-held** (80%): read once.
- **The §53 sample from N-held:** 150 FA, 150 C and 150 UNA rows (UNA-rel and UNA-swap in N's proportion),
  stratified by task, drawn with seed 5300 from the B0 labels before any edited output exists.
- **Activity manipulations (C5):** the four §51 manipulations (evidence removed, city changed, distractors added,
  unrelated vs absent) built on N-held bases exactly as §51's M3 builds them on H.
- **LongMemEval for arm C:** dev folds 0–1 only (the 126 answerable rows of the §35/§43 population plus the 12
  abstention rows of folds 0–1), `fixed8`, the frozen reader prompt. Folds 2–4 stay sealed.
- **N keys (rev 1a):** disjoint from E and, where the key pool allows, from S (where selection happened); any overlap
  with S, H or T-city keys is reported, and C1/C4 are also reported on the S-disjoint rows (descriptive). No N prompt
  equals any §49–§51 prompt, and no N memory text equals an E memory text.
- **Sample ids (rev 1a):** the manifest freezes the drawing rule, not the ids, because the sample needs B0 labels on
  N-held: seed 5300; per class a largest-remainder allocation across tasks by availability; UNA stratified by
  (task, set) in N's proportions; if fewer than 150 FA exist, all are taken. The pod writes the drawn ids before any
  edited run.

## Arm A: the two heads (submission 1)

Total-effect generation with §51's grader.

| Arm | Edit | Rows |
|---|---|---|
| A-target | submission 1, byte-identical (sha checked), dose 1 (the full zeroing) | FA, C, UNA |
| A-single | L32h2 alone; L33h10 alone (descriptive) | FA, C, UNA |
| A-rand | 20 random pairs, one head of L32 and one of L33, targets excluded, seed 5301, listed in the manifest | FA, UNA |
| A-late | 10 random pairs of any two heads in L30–L35, targets excluded, seed 5302 (descriptive) | FA, UNA |
| A-amp | §51 diagnostic `87224655b7a5361e` (the two heads × 2) and the A-rand pairs × 2 | C |

- **C1 (release is specific):** on FA, A-target's release rate (B0 abstention → gold) is above every A-rand pair's
  (rank p ≤ 1/21), and A-target − the median A-rand rate ≥ 0.30.
- **C2 (abstention creation is specific):** on C, A-amp target's C → abstention rate is above every amplified A-rand
  pair's (rank p ≤ 1/21).
- **C3 (cost, descriptive):** A-target's unsupported-answer rate on UNA against the A-rand distribution.

## Arm D: the L35 rank-1 edit (submission 2)

| Arm | Edit | Rows |
|---|---|---|
| D-target | submission 2, byte-identical (sha checked), dose 2 | FA, C, UNA |
| D-randread | 20 rank-1 L35 `o_proj` edits with submission 2's write vector `dm` and a random unit read vector (seed 5303, in the manifest), each scaled so the RMS of its gate over N-dev FA rows equals the target's (rev 1a; a median can sit near zero for a random read) | FA, C, UNA |
| D-randwrite | 10 rank-1 edits with submission 2's read vector `w` and a random unit write vector of the same norm (seed 5304) (descriptive) | FA, UNA |
| B1-ref | §51's B1 at −8 (descriptive reference only) | FA, C, UNA |

- **C4 (the read side matters):** A-target-style rank test on the utility. D-target's U(λ=1) on the sample (FA, C,
  UNA, §51 base weights) is above every D-randread edit's (rank p ≤ 1/21), *and* D-target's unsupported-answer rate
  is below the median D-randread rate.
- **Reading:** C4 shown means the gate `w` carries the selectivity. C4 not shown means a size-matched push along
  city − unknown, read from anywhere, does as well, and the report must say the edit acts like an output bias.

## Arm E: the editor's activity predictions for submission 2

- f = the projection of the L35 `o_proj` output at the answer ':' on `w/|w|`, in the unedited model, per base.
- **C5:** the four predictions exactly as the editor froze them in submission 2's `activity` field: evidence removed
  decreases f; city changed: equivalence; distractors added decreases f; unrelated vs absent increases f. Tested with
  §51's M3 activity code and margins (computed from N-dev as §51 computed them from S). C5 is shown if all four hold
  (intersection-union).
- This is teacher-forced and cheap. It tests the editor's description of the signal, not the edit's effect.

## Arm B (descriptive, cut to what fits)

- **B-DLA:** each target head's direct logit attribution at P_ans onto "unknown" − gold first token, on FA, C, UNA.
- **B-opt:** the option-removed prompt (§51 B3 wording) on FA rows that still abstain under it: do A-target and
  D-target still release them? n is expected to be small.
- Attention-mass by token class and head-output patching (rev 0 arm B items 1 and 4) move to a later laptop follow-up;
  they are not part of this run.

## Arm C: real questions (descriptive)

- On the 138 LongMemEval dev rows: B0, A-target and D-target, all run fresh on the pod with one HF fp32 engine.
  The reproduction check is agreement of §23's cached Qwen 3B `fixed8` outputs with the fresh B0 (refusal flag and
  first token after "Answer:"). Graded by `answer_selection.d1` containment; a refusal is an answer starting with
  "unknown". (Rev 1 named §24 in error.)
- **Reported:** in-view refusals released (n ≈ 30) and how many contain the gold; damage to correct rows; answers on
  the abstention questions in folds 0–1. Gold-string containment.
- **Reading:** releases with the gold at low damage mean the effect carries beyond the templates; little release means
  template-specific, and the report must say so.

## Multiplicity (rev 1a)

C1, C2, C4 and C5 are four separate claims, each judged at its own pre-registered bar: rank 1 of 21 (p ≤ 1/21) plus
its co-requirement for C1, C2 and C4; the §51 IUT at 0.05 for C5. No family-wise claim is made across them; under a
global null the chance that at least one passes is at most about 0.19, and the report says so. (Rev 1 put Holm over
the four, which no rank test with 20 draws could pass: p_min 1/21 > 0.0125. Found by the Executor before any data.)

## Compute, order and spend

- **Size:** about 24,000 generations (A ≈ 13,100; D ≈ 10,400 + B1-ref 450; B0 450) plus teacher-forced activity reads
  and 378 long-context LongMemEval generations.
- **Laptop:** about 230 rows/h measured in §51, so about 4–5 days. Too slow for the user's timeline.
- **Pod (recommended):** one A40 at the then-current price ($0.59/h in §51). §51's H+T pod generated about 9,000
  rows/h. Estimate about 3 h ≈ $1.80; **hard stop 4 h ≈ $2.40**. The Executor measures the rate on N-dev first and
  re-quotes before any held run. Inside the $36 cap (spent ≈ $31.16; $4.84 left). Needs the user's own approval line
  in the Executor's pane. One pod at a time.
- **Order:** manifest commit (random pairs, random vectors, sample ids, calibrated doses, code sha) → N-dev smoke and
  timing on the laptop → quote → pod → Advisor recount → report.

## Rules

- No held-out peeking. N-held and LongMemEval folds 2–4 are read once; nothing is chosen from them.
- D-45: a failed claim is documented, not rescued.
- The manifest is committed before any N-held output, and the pod checks its hash (as the §51 F2 guard does).

## Confounds stated in advance

- Zeroing a head is a large intervention. "Necessary in this pathway" is not "computes the decision".
- Random pairs at L32/L33 may include heads with real roles; A-rand is a null for "these two specifically".
- D-randread edits are calibrated on N-dev FA rows to match the target's size there; a random read vector can have a
  very different spread across rows, so C4 compares outcomes at a matched FA-median size, not a matched release.
- Arm C uses one prompt and the `fixed8` contexts, so it shares §43's prompt-format caveat.

## Revision

- **rev 1a (2026-11-07, before any N-held output):** Holm over four rank tests replaced by per-claim bars (rev 1
  could not reject); D-randread calibrated by RMS, not median; N key rule; sample-drawing rule frozen instead of ids;
  arm C B0 run fresh, §23 cache as the reproduction check, 12 abstention rows added. Executor's questions 1–4.
- **rev 1 (2026-11-07, D-76):** opening rule rewritten after §51 closed; arm D (L35 rank-1 specificity, C4) and arm E
  (submission 2's activity predictions, C5) added; arm B cut to DLA and option-removed; Holm over four claims; pod
  recommended for the user's timeline with a quote and hard stop.
- **rev 0 (2026-11-05, D-75):** two-head controls, arm B in full, arm C; laptop only.
