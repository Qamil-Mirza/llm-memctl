# Review: §44 rev 1 conditions S and U on Llama 3.2 3B (pooled, 330 rows) — 587b6cc

Advisor results review; every number recounted from `runs/exp44_swap/llama3b/{S,U}/*.json` (all match the report).
Llama is **descriptive throughout** by the rev 2 gate (24 in-view failures < 30). Verdict: **ACCEPT.**

| claim | rule | result (recount) | reading |
|---|---|---|---|
| S0 | following among successes ≥ 0.80, LB ≥ 0.70 | 55/66 = 0.833 [0.742, 0.924], same at 0.5 / 1 / 2 nat | **valid** |
| S1 | (a) indeterminate ≤ 0.20, (b) margin stability < 0.15, (c) half-width ≤ 0.15 | 24 failures: answer-production 19 (0.792 [0.625, 0.958]), evidence-use 1 (0.042), indeterminate 4 (0.167); at 2 nat 0.750 / 0.125 / 0.125 | (a) yes, (b) yes (0.042), (c) **no** (0.167) → "not reliably distinguishable on these rows"; point estimate in the answer-production-majority range |
| S2 | each class ≥ 8 refusal rows | answer-production refusals 10, push (L17 α2) opens 0.300; evidence-use 1 | descriptive |
| S3 | donor bound | d_T forward 0.768 [0.750, 0.786] (n 74), reverse 0.786; donor primary fails (max |mean| 1.0 nat), fallback passes (ratio 0.04–0.05 from l0 = 22); no rev 0 / lens rows for this family | descriptive |
| U1 | ≥ 0.50 format artefact; ≤ 0.25 & UB < 0.35 not the format | 4/45 = 0.089 [0.022, 0.178]; still refuse 31/45 = 0.689; Steer-under-U 10/45 = 0.222; paired +0.133 [0.044, 0.244]; swap subset 1/12 | **"not the format"** |

## What it says

1. **The swap manipulation works on Llama.** Five of six successes follow the swapped evidence, clearing the bar
   Qwen 3B missed (0.739). So the manipulation is usable on this prompt family; Qwen's S0 failure is not a general
   property of the design.
2. **Where it works, Llama's in-view failures are answer-production failures.** On 19 of 24 failing rows the answer
   position tracks the evidence (gold preferred with the gold line, substitute preferred with the swapped line), yet
   the model refuses (10) or answers wrong (9). Only 1 of 24 looks like the evidence never reached the answer
   position. This is the **opposite of rev 1's prediction for Llama** (evidence-use majority, drawn from §43's
   note-ceiling reading). The interval is too wide for the pre-registered reliability bar, so it stays descriptive.
3. **Selection caveat on 2.** The swap subset is the verbatim-extractive end of the population (the gold sits in exactly
   one evidence line). §43's "upstream extraction" reading for Llama came from rows where the note had to carry the
   gold; those are under-represented here. The honest statement: on verbatim single-line rows, Llama's failures are
   decision failures, and the §43 contrast with Qwen is not reproduced on this subset.
4. **U1 again "not the format."** Removing the "unknown" option opens fewer than 1 in 10 Llama refusals (Qwen: 23 %);
   the push adds 13 points on top. Across both families the Ling et al. 2026 mechanism is a minority of in-view
   refusals.
5. **Same non-follower pattern as Qwen (post-hoc).** The 11 non-following successes: 5 "other" rows with
   type-incompatible substitutes (e.g. "a yellow dress" → "over a year", "Dark Souls 3 DLC" → "A mix of Irish and
   …"), 4 bare-number rows with mismatched forms ("2014." → "5", "2-3 eggs" → "2014.", "5-2" → "8 meals"), 2 names
   with a generic "The" read token. By type: date 4/4, name 20/22, currency 3/3, duration 5/5, bare 14/18, other 9/14.
   Two families now show the same failure on the same substitute classes, which corroborates the Qwen diagnosis as a
   design fault of the "other" rule and of bare-number form matching, not model behaviour.

## Descriptive notes

- Classes at 1 nat: successes 55 / 0 / 11, refusals 10 / 1 / 1, wrong 9 / 0 / 3.
- d_T 0.768 on Llama against 0.889 on Qwen 3B (28 and 36 layers); Llama's §43 logit-lens emergence median was 0.750.
- Controls behaved as on Qwen (donor and random ≤ 0.05 × the forward shift from l0 on; donor late-layer mean up to
  1.0 nat, below the 2-nat flag).
- Cost: laptop, $0.

## Rulings

1. 587b6cc ACCEPTED. §44 text: the Llama block states S0 valid; S1 "not reliably distinguishable (half-width 0.167 >
   0.15)" with the point estimate and the selection caveat of item 3; U1 "not the format"; everything descriptive by
   rev 2.
2. The rev 1 prediction for Llama is recorded as **not supported on the swap subset**, with the caveat, in §44 and in
   F11's Llama cell when the user approves the record changes.
3. **Decision for the user (not a proposal):** the swap design has a corroborated fault in two substitute classes. A
   repaired rev (drop the "other" type; require bare numbers to match in form; read the first content token) would be
   a pre-registered rev 8 on Qwen 3B only, about 2.7 h on the laptop, $0. Under D-45 the Advisor does not order it;
   the user chooses between that and closing the swap line with the documented outcome.
4. Rev 0 confirmation reads proceed as ordered (3B running; 7B after the projection).
