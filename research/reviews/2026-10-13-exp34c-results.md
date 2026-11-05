# Review: §34c results (86197fa) and the §34a reviewer-question caveats (ae62ffc)

Reviewer, 2026-10-13. Authority: pre-registration `research/proposals/2026-10-13-exp34c-evidence-ablation.md` (ce9910c).

**Verdict: REVISE.** The numbers are right and the process is clean. Five sentences claim more than the data show. One
of them matters: the 3B-vs-7B gap of §34a Claim 1 is the same size with no evidence in the prompt, so the claim that it
is "unaffected" hides that it is not a gap in extraction.

## What I ran

1. **Leak and process.** ce9910c was committed 14:27:31 −0700 (21:27:31 UTC). The ablated reads have mtimes from 21:31:15
   to 21:39:31 UTC, and the reports were written at 21:40:30 (dev), 21:40:54 (confirm) and 21:41:13 (all folds). The
   pod ran 21:29 to 21:40 (ledger lines 103 to 106; the user's approval "do it" is at line 103). So no §34c number on a
   confirmation fold existed before the pre-registration. The section says "reviewed at the result stage". **OK.**
2. **The ablation.** I took three fold 2–4 records per reader that have evidence (seed 1: 72e3ee87, f0853d11,
   b759caee), plus 86b68151 and gpt4_f420262d. On every one: no evidence line is left. The text before the first
   memory line and after the last memory line is byte-identical. The memory block equals `"\n".join(non-evidence
   lines)`. `intact_prompt_sha256` equals the §34a `prompt_sha256` and the hash of `record.prompt`, and `prompt_sha256`
   equals the hash of the ablated text. The drop in `n_tokens` matches the tokens of the cut lines: 84 vs about 85,
   651 vs about 654, 163, 163 and 534. My Qwen2.5 tokenizer gives the same `n_tokens` exactly. Across the 204 paired
   questions: 0 hash mismatches, every `position_mode` is "ablated", and none has 0 cut lines. In all 427 records with
   evidence (3B, folds 0–4), no evidence line appears twice in memory, so `n_evidence_lines_cut` is exact. **OK.**
3. **Numbers against the reports.** Every cell matches `runs/exp34c_report_{dev,confirm,allfolds}.md`:
   - the claim table, the dev rows and the all-folds rows in brackets;
   - all seven plausibility rows in both columns, and the all-folds plausibility line;
   - the 7B presence, C and lift numbers, and the 7B shares;
   - the ` unknown` counts (118→129 logit, 16→108 J-lens) and the all-folds 0.582→0.624 and 0.082→0.558.

   "About 4×" holds: 10^0.631 = 4.3. The spend checks out: 11 min ≈ 0.18 h, ≈ $0.11, total ≈ $17.00.

   **The 0.25.** §34a's 0.638 is 81/127, the same population as §34c's "all failures, n C with evidence 81" (folds 2–4,
   logit). So using 1 − 0.605 = 0.395 on it is legitimate: 32/127 = 0.252. **OK.**
4. **Extra checks (paired bootstrap, 10,000 samples, folds 2–4, 204 questions):**
   - 3B − 7B log10 best rank, logit. Intact: +0.486 (+0.376, +0.600), as in §34a. **Ablated: +0.473 (+0.361, +0.584).**
     J-lens: intact +0.535, ablated +0.466 (+0.363, +0.570).
   - Evidence lift, 3B − 7B: −0.013 (−0.126, +0.097) on logit/logit, −0.069 (−0.187, +0.045) on J-lens/J-lens, +0.034
     (−0.078, +0.141) on the deciding rows.
   - ` unknown` top-1, ablated − intact: 3B +0.054 (+0.010, +0.103) on logit and +0.074 (+0.034, +0.118) on J-lens; 7B
     +0.451 (+0.382, +0.520) on J-lens and +0.422 (+0.353, +0.490) on logit.
5. **The §34a caveats.** I ran `scripts/exp34_alignment_audit.py --folds 2,3,4`. The output matches the text: 54 cases,
   38+2 that start with the gold, 8+6 paraphrases; the 7B has 103 of 141, all at rank 1; 65 refusals split 40 / 6 / 19,
   so 25 that the 7B also missed. One mismatch: the "by question type" line counts all 65 refusals, not the 25. Among
   the 25 the split is temporal 6, multi-session 5, knowledge-update 5, preference 5, single-session-user 3,
   assistant 1. Temporal is denser among the 40 the 7B gets right (14/40 = 0.35, vs 6/25 = 0.24).

## Required changes

1. **Reading it, the sentence "The same discount applies to the 7B, so the paired 3B-vs-7B gap of §34a Claim 1 is
   unaffected, and the evidence lift is the cleaner per-question measure."** Replace it with this: "The 3B-vs-7B gap of
   §34a Claim 1 does not depend on the evidence. Without it, the 3B's log10 best rank is still worse by +0.473 (+0.361,
   +0.584) on the logit row (+0.486 intact), and the two readers' evidence lifts do not differ (3B − 7B −0.013 (−0.126,
   +0.097), logit). Claim 1 is a difference in how highly each reader ranks the gold token at a forced position, with or
   without the evidence. It is not a difference in extraction from the evidence." Make the same change in "What this
   changes in §34a": "the size claims stand (paired)" becomes "the size claim stands as a ranking difference that is
   present without the evidence (see above); it is not an extraction difference". Do the same anywhere Claim 1 is
   quoted as "size" in STATE.md or the docs copies.
2. **"Cutting the evidence removes about a quarter of "present" in both readers."** The +0.235 is absolute. Relative
   to presence it is 0.34 for the 3B (0.235 / 0.701) and 0.25 for the 7B. Change it to: "Cutting the evidence lowers
   presence by about 0.24 (absolute) in both readers."
3. **"…about a third of which is evidence-driven…"** (What this changes in §34a). 0.25 / 0.64 = 0.39. Change it to
   "about two fifths of which (0.25 of 0.64; 32 of 127) is evidence-driven".
4. **"This is the clearest single description of the 3B's failure in the whole series, and it is a readout at a
   forced position, not an intervention."** Delete the ranking. Replace it with: "This is a descriptive contrast at a
   forced position, not an intervention. The 3B's ` unknown` reaches top-1 at some layer on 58% of questions, while it
   refuses on 32% (65 of 204), so the signal does not track its behaviour either." Also add the paired intervals to the
   ` unknown` paragraph: "3B +0.054 (+0.010, +0.103), 7B +0.451 (+0.382, +0.520), paired, ablated − intact". Then
   "close to a constant" has to stay honest: the 3B's shift is small but its interval excludes 0.
5. **§34a caveat (ae62ffc): "…the rest, temporal reasoning most of all, may be hard or arguable."** The audit's type
   count covers all 65 refusals. Among the 25 the 7B also missed, temporal is 6, only one ahead of the next three
   types (5 each), and temporal is denser among the 40 the 7B answers correctly. Change it to: "the remaining 25 are
   spread across types (temporal 6, multi-session 5, knowledge-update 5, preference 5, other 4) and may be hard or
   arguable". If the type split is to be quoted, change the audit to print it for the 25.

Not required: the 0.25 = 0.64 × 0.39 arithmetic is valid on the same 127 failures. The plausibility table, the claim
verdicts and the spend are correct as written.

## Round 2 (62fbb88)

**Verdict: ACCEPT.** I checked `git diff 86197fa..62fbb88 -- EXPERIMENTS.md scripts/exp34_alignment_audit.py` against
the five changes from round 1:
1. Applied. There is a new bullet: "§34a Claim 1 is a ranking difference that exists without the evidence". Its numbers
   match mine: the gap is +0.473 (+0.361, +0.584) against +0.486, and the lift difference is −0.013 (−0.126, +0.097).
   The J-lens numbers also match: the gap goes from +0.535 to +0.466, and the lift difference is −0.069 (−0.187, +0.045).
   The "What this changes" sentence and the §34a "Size" caveat now point to it. I grepped for other places that quote
   Claim 1. research/STATE.md does not mention §34a. The docs report copies are generated tables. The Claim 1 in
   THESIS_RESULTS_OUTLINE.md is a different claim (§23). So nothing else needs changing.
2. Applied: the drop is about 0.24 absolute, a relative drop of 0.34 for the 3B and 0.25 for the 7B.
3. Applied, in both places: "two fifths (0.25 of 0.64; 32 of 127)".
4. Applied. The ranking sentence is gone. The paired intervals are in: 3B +0.054 (+0.010, +0.103) and 7B +0.451
   (+0.382, +0.520). The note "58% vs 32% (65 of 204)" is there, and the text keeps "a descriptive contrast at a forced
   position, not an intervention".
5. Applied. The text now gives the split of the 25 (temporal 6, multi-session 5, knowledge-update 5, preference 5,
   other 4), and other 4 = single-session-user 3 + single-session-assistant 1. I re-ran the audit script and it now
   prints the split of the 25, with the same values.

No remaining changes.
