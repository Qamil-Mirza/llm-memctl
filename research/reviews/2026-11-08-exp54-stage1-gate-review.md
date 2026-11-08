# §54 stage-1 gate review (Advisor, 2026-11-08)

**Input.** Commit 71b9ce7 (`runs/exp54/pairs.json`, `pairs_pre_regrade.json`, `scripts/exp54_run.py`).

**Verdict: the gate FAILS, and the stop is correct.** Under the frozen §6 rule, §54 stops after stage 1. There is
no stage 2, no timing step, no enlargement of W, and no use of the secondary pairs to make up the count (D-45).

## Checked against pairs.json

| Gate item (frozen) | Bar | Measured | Result |
|---|---|---|---|
| Primary pairs | ≥ 40 | 21 | **fail** |
| Bases | ≥ 20 | 20 (10 per half) | pass, exactly |
| Tasks | ≥ 2 of 3 | 3 (DE 8, RC 8, TU 5) | pass |
| Stable pairs per half (criterion 1) | ≥ 20 | 126 / 131 | would pass |

Other counts: 450 twin candidates, 64 gold/abstention flips covering 64 of 288 false-abstention rows (22%). Under A's
forced note, 43 flips answer with the gold (note-driven) and 21 still abstain (primary). Forced-note grades of stable
candidates: 257 gold, 11 abstain.

## The grading bug fix

The first read gave 0 primary pairs because the forced tail was graded without the note + marker head. The fix
(`exp54_run.py`, `stage_regrade`) grades head + tail, as §51 does, from the stored generations and D values. No model
was rerun and nothing in the design changed. I accept it: it restores the grading the proposal specified, the
pre-regrade file is kept, and it cannot have rescued the gate because the gate still fails.

## What the record may say

- Most timestamp-twin flips (43 of 64, 67%) disappear when both twins see the same note. On this data, the
  near-threshold "unknown" decision is mostly set by what the model wrote in its own note, not by a late switch at
  the answer position.
- Scope: twin flips cover only 22% of false abstentions. The other 78% do not flip under a timestamp change, so this
  says nothing about where their decision is made.
- §54 found no native switch and did not test for one. It is not evidence that no switch exists.
- Median D_F on F's own note: 4.9 for note-driven pairs, 1.0 for primary pairs. Descriptive only.

## Next steps

- The Executor drafts EXPERIMENTS.md §54 (counts, the bug and its fix, the stop) for Advisor review.
- Proposed D-79: "§54 closed at the stage-1 feasibility gate (21 < 40 primary pairs); stopped under §6 / D-45."
  STATE/DECISIONS entries wait for the user's "Go".
- Seed 56 stays sealed and unused.
