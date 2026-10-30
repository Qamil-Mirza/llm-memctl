# §44 review: rev 8 swap result (Qwen 3B) and the 7B device check (cbffc52)

**Verdicts.** (1) **Rev 8 S0 FAILS on its lower bound; the swap manipulation is not valid on this prompt family in
either design; the swap line closes with the documented result (D-45 / D-49: no rev 9).** (2) **The GPU 7B rows are
the confirmation read, under the pre-declared hold rule below, written before the GPU report exists.**
Pod verified: list-pods empty from the Advisor pane 2026-10-30 ≈ 21:20 UTC; ≈ 17 min ≈ $0.16; spend ≈ $20.22 of $30.

## 1. Rev 8 S, Qwen 3B — recount from `runs/exp44_swap/qwen3b_rev8/S/*.json`

| item | Executor | Advisor |
|---|---|---|
| rows / labels | 70; 38 / 14 / 18 | 70; success 38, refusal 14, wrong 18 |
| S0 following among successes | 31/38 = 0.816 [0.684, 0.921] | 31/38 = 0.816, bootstrap [0.684, 0.921] (seed 44) |
| honesty split: changed-substitute rows | 7/10 = 0.700 | 7/10 |
| honesty split: unchanged rows (known from rev 1) | 24/28 = 0.857 | 24/28 |
| failures by class at 1 nat | AP 0.812, indeterminate 0.188, EU 0 | 26 / 6 / 0 of 32 (margins 0.5 and 2 nat: 26/6/0 and 26/5/1) |
| classes (AP / EU / ind) success, refusal, wrong | 31/3/4, 11/0/3, 15/0/3 | same |
| S2 descriptive | AP refusals 9, opened 0.556 | not recounted (descriptive) |
| S3 | d_T 0.889 (n 57), fallback validity | not recounted (S3 not read: S0 failed) |

**Reading.** The point estimate clears 0.80 but the lower bound 0.684 misses 0.70; by the pre-registered rule S0
fails and S1–S3 are not read. The fresh evidence (changed rows) is 7/10, a thin, non-significant improvement over
the rev 1 bare-number rows (10/15). The seven non-followers are now almost all **small-number rows**: "four" → "five",
"five" → "four", "3" → "5", "2" → "6", "2014." → "1300", plus "Spotify" → "Emma" (a brand swapped for a person's
name under the name rule) and the duration row whose Δ_clean < 0. Small counts are plausibly answers the model
computes or reads off elsewhere in the prompt; a swap of the number string does not control them. This is the
residual fault of the design and it is documented, not repaired again.

**For §44 text (Executor; verbatim where quoted):** a rev 8 block stating: the repaired design (rules 1–5), 70 rows,
S0 "31 of 38 = 0.816 [0.684, 0.921]: not valid (lower bound below 0.70)"; the honesty split "changed-substitute rows
7 of 10, unchanged rows 24 of 28 (already known from rev 1)"; "a repaired design tested on the same rows after the
fault was seen; nothing here is confirmatory"; the non-follower list as above; the descriptive class line "among the
32 failures 26 answer-production, 6 indeterminate, 0 evidence-use at 1 nat; descriptive only, S0 failed"; and the
closing sentence: **"The swap manipulation did not reach validity on this prompt family in two designs; the
evidence-use / answer-production split is not established by §44."**

## 2. 7B device check (laptop CPU bf16 vs pod CUDA bf16; 73 common rows, 36 patched on both)

d_c equal 34/36 (the two differ by −1 and +2 layers of 28); max |R| difference 0.142, median over rows 0.042; one
eligibility flip (91b15a6e), two gap-sign flips, one 1-nat-gap flip, one R(0)-flag flip; rank changes clean 10 / cut
26; max |log p| difference ≈ 1.4 nat. This is bf16 drift between devices, small on the deciding statistic (a median
of d_c over ≈ 55 rows) and able to move a row or two across the filter.

**Ruling (pre-declared before the GPU report is produced).**
1. The GPU rows (`runs/exp44_patch/qwen7b_gpu/`, 204 rows, one device) are the confirmation read for C1 and C2.
2. The report carries a **sensitivity line**: C1 and C2 recomputed (a) excluding the rows whose eligibility, gap sign,
   1-nat gap or R(0) flag flipped between devices (5 rows), and (b) with the median shifted by ±1 layer (±0.036).
3. **Hold rule:** if the C1 verdict (J-lens right if UB < 0.65; logit right if LB > 0.70; else undetermined) changes
   under (a) or (b), the verdict is **held**, the laptop CPU read completes (≈ 8 h, $0) and both device reads are
   reported together, the CPU read deciding (the filter was developed on CPU reads). If the verdict is the same
   under (a) and (b), it stands and **the laptop run is stopped** (rows kept; the device check is the record).
4. C3 (3B) was read on the laptop CPU; no change.
5. The §44 text carries one device paragraph with the numbers above.
