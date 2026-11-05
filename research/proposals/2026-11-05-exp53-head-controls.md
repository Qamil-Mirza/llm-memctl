# §53 (proposed): are the two §51 heads specific, what do they read, and does the edit carry over to real data?

**Status: proposed, rev 0 (Advisor, 2026-11-05). Not reviewed, not built, not run, nothing spent.** Decision D-75.
The section number is provisional. Written from `research/audits/2026-11-05-research-gap-check.md`.

## Opening rule (fixed now)

§53 opens only if §51 confirms submission 1 (`c64ea090781450cf`: zero the `o_proj` columns of L32h2 and L33h10) on H
under §51's own confirmatory rule.
- **Only submission 2 confirms** (the rank-1 L35 edit): §53 does not open as written. A rev 1 would be needed, because
  arms A and B are about heads.
- **Nothing confirms:** §53 closes unrun, documented under D-45. No rescue variant.

Nothing in §53 touches E, S, H or T after §51 reads them. §51's freeze is not changed.

## Why

§51's frozen design tests three alternatives: overfitting to E (S, H, T), "another answer-more knob" (B1–B5) and
"any change to these weights" (B6). Three alternatives are untested:

1. **General disruption.** Removing *any* two late heads might release false "unknown"s. B6 perturbs the same two
   heads' columns, not other heads.
2. **Prompt-option copying.** The heads might copy the prompt's "unknown" option rather than weigh the evidence.
3. **Templates only.** The effect might not carry over to natural questions.

## Data

- **N (new controlled split):**
  - Seed 54. Seeds 50–53 are used by §50 and §51.
  - The same generator, tasks (DE, RC, TU), F2 format, prompt and classes (FA, C, UNA-rel, UNA-swap) as §51.
  - B0 classes are assigned by the §51 grader, unchanged.
  - N has never been seen by the editor or the analysts.
  - Split into **N-dev** (20% of bases: pipeline checks and timing only) and **N-held** (80%, read once).
- **The §53 sample from N-held:** 150 FA, 150 C and 150 UNA rows (UNA-rel and UNA-swap in N's proportion),
  stratified by task. It is drawn with seed 5300 from the B0 class labels before any edited output exists.
- **LongMemEval for arm C:** dev folds 0–1 only (126 rows, the §35/§43 population), `fixed8`, the frozen reader
  prompt. Folds 2–4 stay sealed.

## Arm A: is the pair specific? (pre-registered claims)

Mode: §51's controlled prefix at P_ans, plus total-effect generation for unsupported answers, with §51's grader.

| Arm | Edit | Rows |
|---|---|---|
| A-target | §51 submission 1, byte-identical (sha checked) | FA, C, UNA |
| A-single | L32h2 alone; L33h10 alone (descriptive) | FA, C, UNA |
| A-rand | 20 random pairs: one head of L32 and one of L33, excluding the targets, drawn with seed 5301 and listed in the manifest before any output | FA, UNA |
| A-late | 10 random pairs of any two heads in L30–L35, excluding the targets, seed 5302 (descriptive) | FA, UNA |
| A-amp | §51 diagnostic `87224655b7a5361e` (the two heads × 2) and the A-rand pairs × 2 | C |

**Claims:**
- **C1 (release is specific):** on FA, the A-target release rate (B0 abstention → gold) is above the release rate of
  every one of the 20 A-rand pairs (rank p ≤ 1/21), and A-target − the median A-rand rate ≥ 0.30.
- **C2 (abstention creation is specific):** on C, the A-amp target pair's C → abstention rate is above every one of
  the 20 amplified A-rand pairs (rank p ≤ 1/21).
- **C3 (cost):** the A-target unsupported-answer rate on UNA is reported against the A-rand distribution. It is
  descriptive and claims nothing.

**Readings:**
- **C1 and C2 both shown:** the pair is specific in both directions on fresh data.
- **C1 not shown:** "general late-head disruption" is not ruled out, and the §51 write-up must say so.

## Arm B: what do the heads read? (descriptive, with predictions fixed in advance)

On the N-held sample, unedited (B0 runs), at P_ans:

1. **Attention mass by token class**, for each head. The classes are defined from the template before any output:
   - (o) the "unknown" option in the instruction;
   - (n) the model's own note, with (g) the gold-city tokens inside it;
   - (e) the memory and evidence span;
   - (q) the question;
   - (r) the rest, BOS included.

   This is compared across FA, C and UNA-rel.
2. **Direct logit attribution:** each head's output at P_ans, projected through the final norm (its scale frozen from
   the clean run) onto the difference between the first "unknown" token and the gold's first token.
3. **The option-removed prompt** (§51 B3 / §49 wording) on FA rows that still abstain under it: does A-target still
   release them? Descriptive; n is expected to be small.
4. **Head-output patching** on content-matched pairs from the same base (one variant answered, one abstained, as in
   §50): the two heads' outputs at P_ans are patched from answered into abstained, and the reverse. Descriptive. The
   pair count is whatever N supplies; it is reported, and no pair is sought elsewhere.

**Fixed predictions**, used only to name the reading:
- **Option copying:** attention to (o) is higher on FA than on C; the attribution favours "unknown"; A-target
  releases little under the option-removed prompt.
- **Evidence-weighing:** attention to (n), (g) or (e) separates FA from C more than (o) does.
- **Neither:** the result is reported as such. No new class is invented after the fact.

## Arm C: real questions (descriptive)

- On the 126 LongMemEval dev rows, A-target is compared with B0. B0 is §24's cached Qwen 3B outputs, after a
  reproduction check like §43's (first token ≥ 4/5).
- **Reported:**
  - the in-view refusals released (n ≈ 30), and how many of those contain the gold;
  - damage to rows that were correct;
  - answers on the abstention questions inside folds 0–1.
- Judged by gold-string containment. The Qwen 7B judge can be added later on a pod, with its own quote.
- **Reading:** releases with the gold, at low damage, mean the effect carries beyond the templates. Little release
  means a template-specific mechanism, and the §51 write-up must say so.

## Compute, order and spend

- **Laptop CPU, fp32, batch 1, as §51 at $0.**
- **Size:** about 13,500 controlled-prefix generations plus arm C.
  - At §51's measured 9 s/row prefix rate, that is about 34 h before savings.
  - The Executor may share computation across arms where it is exact (layers below 32 are unchanged by every arm A
    edit). This must be verified bit-equal on N-dev before use.
- **Time:** measure on N-dev before quoting. If the laptop time is too long, a pod is a separate quote and needs the
  user's own line, inside the $36 cap, never during a §51 pod.
- **Order:** after §51 H/T reports. §52 stage 0 (laptop) goes first if the two would overlap.

## Rules

- No held-out peeking. N-held and LongMemEval folds 2–4 are read once, and nothing is chosen from them.
- D-45: a failed claim is documented, not rescued.
- The manifest (random pairs, sample ids, token-class spans and the code sha) is committed before any N-held output.

## Confounds stated in advance

- Zeroing a head is a large intervention. "Necessary in this pathway" is not "computes the decision".
- Attention weights are not explanations. Arm B is descriptive for that reason, and patching is its causal check.
- Random pairs at L32/L33 may include heads with real roles. A-rand is a null for "these two specifically", not a
  null for "nothing in these layers matters".
- Arm C uses one prompt and the `fixed8` contexts, so it shares the prompt-format caveat of §43.
