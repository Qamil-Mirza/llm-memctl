# Review: EXPERIMENTS.md §34b results (b970fbb)

Reviewer, 2026-10-13. Authority: pre-registration `research/proposals/2026-10-13-exp34b-own-position.md` (59d8b0d),
§34 pre-registration and its reviews (verdict rows carry over), `runs/exp34b_report_{dev,confirm,allfolds}.{md,json}`,
reads `runs/exp34_jlens_own/<reader>/`, code `scripts/exp34b_report.py`, `memctl/analysis/jlens_readout.py`,
`scripts/jlens_read.py`.

**Verdict: REVISE.** The pre-registered verdicts (A, B NOT SHOWN) are computed correctly. The reading is not
supported. The pooled null comes from the 39 outputs with no marker, which are not read at an answer position. On the
165 outputs that are read at the reader's own answer, presence goes up, with an interval that excludes 0.

## What I ran

1. **Leak and process.** 59d8b0d was committed at 2026-10-13T20:34:22Z. The pod was created at 20:34:24Z (progress.md
   l.96). The own-mode read files are dated from 20:41Z (`qwen3b/_dropped.json`), and `pod.log` was last written at
   20:51Z. No own-position number existed before the pre-registration. The only data it quotes are marker counts from
   the cached outputs, which are not outcomes. The section says that the pre-registration was not reviewed before the
   run. Confirmed. The `--confirm` gate in the report script is a flag, not a check against the commit time; noted only.
2. **Readout.** `readout_for(own_output=...)` builds chat ids + ids(output[: last "Answer:" + 7]) + the leading
   whitespace ids, with P1 = len - 1. With no marker, only the leading ids are appended, so P1 is the token after
   `<|im_start|>assistant\n`. I recomputed this with the Qwen2.5-0.5B tokenizer (same vocabulary; the 3B and 7B
   tokenizers are not cached here). It matches the files on fold 2:
   3B 0100672e (1588, P1 1587, marker true), 078150f1 (921, 920, true), 07741c45 (1132, 1131, false);
   7B 0100672e (1583, 1582, true), 07741c45 (1142, 1141, true), 2311e44b_abs (2120, 2119, false; outside the paired
   population). **Tokenisation boundary.** chat ids + ids(prefix) equals the joint tokenisation of chat + full output,
   truncated to the same length, on 83 of 83 (3B) and 96 of 96 (7B) fold-2 marker outputs. So splitting the text at
   the chat boundary or at "Answer:" changes no tokens. What cannot be checked is whether sampling produced
   non-canonical splits, because the generated ids are not cached. This belongs under the teacher-forcing caveat and
   does not change the claim. 499 of 500 reads per reader; `_failed.json` holds the over-length eac54adc in each.
3. **Numbers against the reports.** Matching: the Claim A and B rows (confirm, all folds, dev), the J-lens beside
   rows, every cell of the decomposition, the 7B rows (the logit "later by 0.057 (0.028, 0.091)" is the sign-flipped
   −0.057 (−0.091, −0.028)), the marker counts (165 of 204, 39 marker-less of which 23 correct, 17 emerged, 26 of 126 on
   dev), and $0.18 = 0.3 h × $0.59, with $16.70 + 0.18 ≈ $16.90. Not matching: see changes 2 and 3.
4. **Marker-only sensitivity** (exploratory, my run with the report's functions and paired bootstrap; own − forced
   presence, forced − own depth):

   | folds | lens | n | presence own − forced | n both | depth forced − own |
   |---|---|---|---|---|---|
   | 2–4 | logit (decides) | 165 | **+0.091 (+0.030, +0.152)**; 0.685 → 0.776 | 107 | +0.021 (+0.003, +0.041) |
   | 2–4 | J-lens | 165 | +0.073 (+0.012, +0.133) | 102 | −0.151 (−0.201, −0.103) |
   | 0–1 | logit | 100 | +0.110 (+0.050, +0.170) | 73 | +0.008 (−0.013, +0.024) |
   | 0–4 | logit | 265 | +0.098 (+0.057, +0.143) | 180 | +0.016 (+0.002, +0.029) |

   The 39 marker-less questions: the gold emerges on 30 at the forced position and on 17 at the output's first token
   (both lenses). That drop of 13 cancels the gain of about 15 on the marker rows. The 7B (all marker rows) stays null.
   The subset is fixed by the cached text before any read, but this analysis was not pre-registered, so it is
   exploratory. The pre-registration also understated the size of this group ("200 of 204" write `Answer:`; in fact 165).

## Required changes

1. **Heading and status.** "Is the late, often absent emergence an artefact of the forced answer position? No." →
   end with "Not as pre-registered; partly, on presence, among outputs that have an answer marker (exploratory)."
   Keep the status line's NOT SHOWN, but strike "which is the informative outcome".
2. **Refusal order (J-lens, 65)**: "same layer 5; gold first 1" → "same layer 5; neither 1; gold first 1" (the six
   counts sum to 65).
3. **"Final-layer rank on refusals: gold median 6, ` unknown` 1."** On folds 2–4 the gold median is **4** (both
   lenses). 6 is the dev value.
4. **Add a paragraph** after "Marker-less outputs", labelled exploratory, giving the table in item 4 above and the
   30 → 17 marker-less drop. Say that the pooled null on presence is the sum of a gain on the 165 marker rows and a loss
   on the 39 rows that are not read at an answer position. Note the pre-registration's "200 of 204" against the actual 165.
5. **"Reading it" must be rewritten.** These sentences are contradicted by item 4: "a property of the model at its
   answer position, not of the forced format"; "the gold is no more readable at the moment it answers than it was with
   no note at all"; "the share where the gold never enters the top-10 (about 0.30 …) is the same with or without it"
   (on the marker rows, never-enters falls from 0.315 to 0.224); "This closes the forced-position caveat of §34a in
   §34a's favour". Replacement, roughly: the pre-registered pooled test cannot tell the two positions apart. Its
   intervals rule out presence gains above about 0.07 and depth gains above about 0.03 layers-fraction on the mixed
   population, not small effects. Among outputs with an answer marker, the gold is more often in the top-10 at the own
   answer (+0.09, exploratory, replicated on dev). Depth hardly moves (logit +0.02; J-lens later). So lateness belongs
   to the model, while part of the absence is due to forcing. The forced-position caveat of §34a is narrowed, not closed.
6. **"Nothing moves beyond noise."** (decomposition) has no interval, and it mixes the two groups (refusals A
   0.400 → 0.308 on logit). Replace with "No class moves by more than 0.10; no intervals were computed; the cells mix
   marker and marker-less rows."
7. **Caveats: add** "The own position exists only for outputs with a marker; the 19% without one are read at the
   output's first token, which the pooled claims include." Also add the 7B's J-lens failures A 0.127 → 0.254 to
   "The 7B (descriptive)" (less present on failures at the own position), so that "does not make the gold … more
   present" is shown along with where it is less present.
8. **Run and spend.** The record (progress.md l.96) says the pod's hard stop was set at $4.00 (terminate by
   03:19Z), not the approved 60 min / $0.59. Write "approved hard stop $0.59; the pod's timer was set at $4.00".
   progress.md has no termination record for wl4hkwkji9m44p. Add the DELETE and its time (the section's 20:52Z),
   or say that it was not logged.

## Round 2 (92588cf)

**Verdict: REVISE (small).** Changes 1–4 and 6–8 are applied correctly. I checked the marker-split figures against my
round 1 run, the pre-registration's correction note (lines 20–21), and ledger line 98 (DELETE, 20:52Z, the 58 min
timer). Change 5 is mostly done. One new sentence does not hold for the marked rows, by class, which I computed with
the report's own functions (A forced → own on the marked 3B logit rows, folds 2–4): failures (111) 0.369 → 0.306,
refusals (62) 0.419 → 0.323, answered wrong (49) 0.306 → 0.286, successes (54) 0.204 → 0.056; failures own B 0.054, C 0.640.

Remaining required changes:
1. Under "Reading it", replace "§34a's A shares are overstated by roughly 0.09 on the marked outputs". The 0.09 is the
   change in presence over all marked questions, not a change in any class's A share. On the marked rows the failures'
   A share falls by 0.06 (0.369 → 0.306) and the successes' by 0.15 (0.204 → 0.056). Most of the gain is on successes,
   so the own position widens the failure/success presence gap. Say that too, since it supports analysis 1.
2. "The own-position decomposition of failures (A 0.32, B 0.05, C 0.63) is the one to quote from here on" quotes the
   pooled cells, which include the 39 marker-less reads that this same paragraph calls "not an answer position".
   Either quote the marked-row failures (n 111: A 0.31, B 0.05, C 0.64) or label the pooled cells as pooled.
3. The marker-less bullet ("the gold emerges on 30 at the forced position but on 17") is the logit count. Add
   "(logit; J-lens 29 → 17)".
