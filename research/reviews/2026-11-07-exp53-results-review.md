# §53 results review: Advisor recount (2026-11-07)

**Verdict: ACCEPT.** All four pre-registered claims (C1, C2, C4, C5) are shown on N-held at their own bars. My
independent recount from the per-row records reproduces every number in the report. Each claim is judged at its own
bar. Do not write a family-wise "all four" headline (see Wording).

## Inputs checked
- **Report:** `runs/exp53/gpu_pod/runs/exp53/pod/report.json`, sha256 `f1ba6dad767ef65fca8bd7039a49d31e92e9059776394005e4fdee5b551ce04f`.
  It matches the Executor's notice.
- **Design:** manifest `b2b14bac` (§53 rev 1b, proposal `97edca99`, committed a177536 before any N-held output).
- **Recount script:** Advisor scratchpad `recount53/recount.py`. It uses only the stored per-row `cls`, the
  `cases53.jsonl` metadata (base_key, set) and the `sample.json` population ids. It does not import the report code's
  metric functions.

## Recount
- **Row counts:** all 86 manifest arms ran 150 rows per population (0 mismatches against the manifest `pops`).
  B0 has 450 sampled rows. No arm was cut by the deadline. B1-ref ran in full.
- **Rates:** Rel, Uns, Dam, Deg, U1 and C→abstention for all 87 arms (86 + B0) match `report.json`. The largest
  difference is 1.1e-16.
  - U1 follows the frozen `base_mixture` rule: a base counts only if it has both ANS and UNA rows.
  - My first pass averaged bases with one row type too. That shifted U1 for every arm (B0 0.331 instead of 0.262) but
    did not change the C4 ranking. The frozen rule is the right one, and the report uses it.
- **Claims, recomputed:**

| Claim | Target | Random draws (20) | Rank p | Co-requirement | Result |
|---|---|---|---|---|---|
| C1: ablation releases FA | Rel 0.751 | max 0.464, median 0.163 | 1/21 | diff 0.589 ≥ 0.30 | shown |
| C2: amplification abstains on C | C→abst 0.381 | max 0.345, median 0.094 | 1/21 | — | shown (closest of the four) |
| C4: read direction matters | U1 0.403 | max 0.300 | 1/21 | Uns 0.058 < draw median 0.228 | shown |
| C5: four activity predictions | — | — | IUT p ≈ 1e-4 | all four hold | shown |

- **C5 estimates.** I recomputed them independently from `f_held.json` with the frozen pairing: Δf = f(b) − f(a),
  averaged within base and then over bases.
  - evidence_removed −3.766 (120 bases);
  - city_changed −0.025 (120);
  - distractors_added −1.119 (119);
  - unrelated_vs_absent +0.221 (119).
  All four match the report.
  - My unstratified bootstrap intervals are wider than the report's task-stratified ones (evidence_removed
    [−4.05, −3.49] against [−3.89, −3.65]). Every directional test still has 100% of draws on the predicted side.
  - The city_changed 90% interval [−0.113, 0.062] lies well inside ±1.123.
- **f missing on 7 of 1,560 rows:** all are B0 no_answer_line rows, so there is no ':' read site. Their pairs drop,
  as in §51.

## The Executor's margin question
The margins are used correctly. Under §51 §6.3(a) and §13.4(a), a **directional** prediction is a one-sided paired
sign test with no margin. The 2×SD margin applies only to the **equivalence** prediction (city_changed, TOST at
±1.123).
- The manifest also stores margins for the three directional manipulations. The code ignores them
  (`activity_directional` takes only the sign). That is correct.
- The report should not say that distractors_added "cleared its margin". Its estimate is −1.12, smaller in size than
  the stored 2.96, and that comparison has no meaning in the test.

## Wording the write-up (EXPERIMENTS §53, report §7b) must keep
1. **No family-wise claim.** Each claim passed its own bar. Say: "Each of the four pre-registered claims passed its own
   test." Do not say "all four confirmed together".
   - Keep the bound: if the target were exchangeable with the random draws, the chance that at least one claim passes
     is at most about 0.19.
   - Rank tests with 20 draws cannot give a p below 1/21. Report z only as descriptive.
2. **C2 is narrow.** The target is 0.381 against a best random pair of 0.345. Say so.
3. **C4 is relative to random read vectors, not to the simplest baseline.**
   - B1-ref (§51's frozen logit bias at −8, never tuned on N) gives U1 0.406, against D-target 0.403 and A-target
     0.414. Its Rel is 0.698, Uns 0.075 and Dam 0.
   - On N, neither edit improves on a plain logit bias. This is descriptive and is consistent with §51's "B1
     inconclusive". It must appear next to the C4 result, not only in a footnote.
   - The proposal's line "C4 shown means the gate `w` carries the selectivity" is fine, as long as the sentence is
     limited to selectivity relative to random reads.
4. **C3 (descriptive).** The target pair ablation has the highest unsupported-answer rate of all 21 (0.075 against a
   random-pair max of 0.071; B0 0.031). Its release is specific, and so is its cost on UNA rows.
5. **C5 tests the agent's description of the signal on the unedited model.** It does not test the edit's effect.
   - city_changed: write "equivalent within the frozen tolerance". That is not near-invariance (§13.4a). The tolerance
     is 2 SD.
   - The predictions held on fresh bases. C5 is the first test of submission 2's account, and it covers only that
     account's activity predictions: no predictor or interaction test was run for submission 2. It does not establish
     the account.
   - **Correction (same day, before EXPERIMENTS §53 was committed):** an earlier version of this review said C5 "does
     not reverse §51's M3". That framing was wrong. §51's M3 tested submission 1's (the two-head edit's) account,
     whose predictions were different. C5 says nothing about M3.
6. **Arm C (LongMemEval) is descriptive and small.**
   - A-target released 20 of 26 in-view refusals, but only 8 contain the gold. Gold rose from 35 to 45; 2 of 35
     correct answers were damaged.
   - D-target released 19 (7 with gold); gold 45; 3 damaged.
   - On the 12 abstention questions, the edits answered fewer than B0 did (4 against 6).
   - "Released" must not be read as "answered correctly". The reproduction against the §23 cache is imperfect
     (refusal flag 121/138, first word 97/138), so it is the fresh B0, not the §23 numbers, that is the baseline.
7. **B-opt is not estimable.** No FA row abstains when the "unknown" option is removed (B0 gold 145/150), so n = 0.
8. **B-DLA** is descriptive direct-attribution only. It does not show mechanism.

9. **C4 failure modes (Executor's per-draw addition, checked).** Draw 07 releases but answers UNA rows (Rel 0.944,
   Uns 0.879). Draws 12 and 19 break generation (Deg 0.906 and 1.000; C→abstention 0). They do not turn answers into
   "unknown". The best draw is 13 (U1 0.300).

## Spend
- As reported by the Executor: about 15,779 s billed, 4.383 h × $0.59 ≈ $2.59 (quote $2.55, hard stop $3.10). The
  pod was deleted at 19:15:33 (HTTP 204), and list-pods shows 0.
- RunPod programme total ≈ $33.75 of the $36 cap (D-73). About $2.25 of headroom remains.

## Next
- The Executor may commit the small pod outputs (report.json, timing, sample, logs, pod.done) by explicit path, and
  draft EXPERIMENTS §53 for my review under the wording above.
- Advisor: proposed STATE/DECISIONS entries (D-77) go to the user for "Go". The report §7b, Appendix B and C slots are
  filled from this recount.
