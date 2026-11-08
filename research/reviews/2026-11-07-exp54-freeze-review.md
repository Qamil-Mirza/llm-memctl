# §54 stage-0 freeze review (Advisor, 2026-11-07)

**Input.** `runs/exp54/freeze.json` at 66fcafe, sha256 `5390513a…be76`. It was made before any model call.

**Verdict: ACCEPT with two required changes and one ruling.** The Executor commits an amended freeze (new hash)
before stage 1 starts. All of these changes come before any model output, so none of them is a deviation from data.

## Checked and fine

- **Seeds.** Dev 55; seed 56 is sealed and not generated.
- **Data.** Split W, 150 bases and 1,950 rows. Prompts are distinct from all rows in §49–§53. E and S keys are
  excluded. The size is fixed before counts are seen.
- **Key overlap with N and H.** W shares DE and TU (name, gold) keys with N, and through N with H. That is
  acceptable for development data, because only W outputs are read. It matters for stage 5 (see the note at the end).
- **D and t\*.** D uses id 9788 and A's own first token. This avoids §50's DE name-first problem.
- **Controlled prefixes.** F_primary is F.prompt_ids followed by A's note ids, copied without re-tokenisation.
  Requiring equal prompt length keeps P_ans aligned.
- **Primary-pair condition.** The forced-note F still abstains and D_A < 0 < D_F. The other categories are counted.
- **Split rule.** By base, seeded per task, alternating. K is committed before any dev-B patch.
- **Criteria 2, 3 and 5.** The bars match the proposal. Ties count against K.
- **Plain AtP.** Correct here. The AtP\* QK fix targets patched keys and queries, and only outputs at P_ans are
  patched. AtP's linear error is covered by the real-patch check.
- **Mean transplant.** Fit on dev-A, applied on dev-B. Good. This is better than the proposal text.
- **Time rule, stopping rules and hard rules.** As proposed.

## Required change 1: criterion 1 needs a different reference (the proposal's flaw, not the Executor's)

As written, criterion 1 compares the A−F difference with the spread of a component across C rows from *different
bases* in the same cell. Different bases have different names, cities and memories, so that spread is dominated by
content. Every component would then look like it barely differs within twins. Criterion 1 would fail mechanically and
turn any result into "a lever, not a switch". It would not test what it was meant to test: whether the component
differs between twins more than timestamp noise would make it differ.

**Replace the reference with stable twins.**
- **Stable twin pair.** Same base and hardness, rep a and rep b, both graded 'gold' under their own B0, equal prompt
  length.
- Build its controlled prefixes the same way: both twins get rep a's note ids, copied token for token, and B's prefix
  is B.prompt_ids + a.gen_ids[:a.j]. Keep the pair only if both still grade 'gold' under the forced note (greedy, ≤ 16
  tokens), so that its decision does not flip.
- **Score.** For component c, r_c = median over primary pairs of ||w_c(A) − w_c(F)||, divided by the median over
  stable pairs of ||w_c(a) − w_c(b)||. Both are taken at P_ans on the controlled prefixes, within the same half
  (dev-B for stage 3).
- **Bar.** r_c ≥ 2 for every component of K. r_c is reported for every component.
- **Minimum.** At least 20 stable pairs in the half. Otherwise criterion 1 is reported as not evaluable, and K cannot
  be called the switch.
- **What it means.** A switch component must move at least twice as much between twins whose decision flips as
  between twins whose decision holds. A near-constant prior, such as the editor's L32h2 and L33h10, should come out
  near 1. That is the intended reading.

Stable pairs need B0 on the same ANS rows already planned, plus one forced-note answer step per candidate pair. No
UNA rows or C rows from other bases are needed for criterion 1.

## Required change 2: criterion 4 needs a minimum for its reference patch

Criterion 4 is relative to the patch of *all* components ("all"). If "all" itself closes little of the gap, because
part of the difference enters before layer 20 (or before 26 under the time rule), the ratio is noise.

**Add:** if the median forward gap closed by "all" is below 0.5, criterion 4 is reported as not evaluable, and K
cannot be called the switch. Report the median gap closed by "all" in every case.

## Ruling on the full 272-component grid

**Allowed, under a rule fixed now.** At the one-pair timing step, before any stage-2 result is computed:
- If real single patches of all components (272, or 160 under the time rule), forward and reverse, together with the
  greedy steps over that full pool, fit inside the 24 h budget for stages 2 + 3, the greedy pool becomes **every
  component**. AtP is then still computed and reported, as a calibration of AtP against real patches.
- Otherwise the frozen pool stands: top 30 by AtP, a random 30 and the always-included list.
- Record which branch applied, with the measured time, in the amended freeze or a timing note committed before
  stage 2.

**Reason.** The full pool removes the dependence on AtP's approximation and adds no selection risk, because K is
still chosen on dev-A only and verified on dev-B.

## Note for a future stage 5 (not approved, no action now)

Seed 56 generated with the same deterministic DE/TU cell allocation would share keys with W, N and H. A stage-5
proposal must state the overlap and either exclude W keys or justify the overlap. Recorded here so it is not
forgotten.
