# Results review: §43 Part B, Gemma 3 (pod x7sj0wau67mp6i; rows c9b376d, confirm report 204e558)

**Verdict: ACCEPT. Claim B1 NOT SHOWN, Claim B2 SHOWN for Gemma on the confirmation folds; J-lens gate FAIL, so the
logit row alone decides. The Gemma paragraph may be added to §43.**

Recounted independently from `runs/exp43_lens_pod_gemma/{logit,jl}/*.json` and `load_family_records` labels, folds 2–4
(204 paired rows; prompt hashes equal across sizes), percentile bootstrap, 10,000 draws:

| claim | Executor | my recount | verdict |
|---|---|---|---|
| B1 log10 best rank, 4B − 12B | +0.019 (−0.020, +0.059) | +0.019 (−0.021, +0.057) | NOT SHOWN |
| B1 secondary, A-share 4B − 12B | −0.005 (−0.039, +0.034) | −0.005 (−0.044, +0.034) | NOT SHOWN |
| B2 logit AUC, wrong vs right (4B) | 0.698 (0.631, 0.764) | 0.698 (0.633, 0.764) | SHOWN |
| B2 J-lens AUC (gate FAIL, beside) | 0.714 (0.645, 0.779) | 0.714 (0.647, 0.777) | shown beside |

A/B/C of failures (4B n 80: 0.125 / 0.113 / 0.762; 12B n 55: 0.218 / 0.055 / 0.727), emergence medians (4B 0.618,
12B 0.667, quartiles as reported), article split (n 31: 4B 0.484, 12B 0.645) and digit subset (n 84: 1.000 both)
match. The 4B's B class is the article artefact again: 8 of its 9 B failures are article-first (the Executor's "14 of
16" counts B over all 204 rows, successes included; both readings agree). Dev values (B1 +0.003, B2 0.714) are
descriptive and consistent. Pod ≈ 77 min ≈ $0.75; balance delta $0.750; spend ≈ $19.31 of $30. No pod live.

## Reading against the proposal's fixed readings

- **C-majority with an A floor: yes** (0.762 of the 4B's failures have the gold in the final top-10; A 0.125, the
  lowest floor of the three families).
- **The larger reader holds the gold more clearly: no.** B1 is a null with a tight interval (±0.04 in log10 rank);
  the 12B's A share among failures is if anything higher. In Gemma, size does not move where or how clearly the gold
  sits at the answer position on in-view questions.
- **Later emergence in the small model: no.** 0.618 vs 0.667; the 4B is slightly earlier.
- **B2 holds** (0.698): within the 4B, a weak gold rank marks a wrong answer, as in Qwen and Llama.

Scorecard after three families (confirmation folds, logit lens): C-majority 3/3; B2 3/3; larger holds it more
clearly 2/3 (Qwen, Llama; not Gemma); small reader later 1/3 (Qwen, by 0.10). The "answer present, decision late"
half is a property of small readers under this prompt; the size half is Qwen's. Phi is the fourth point.

## Rulings

- Phi quote: 2.5 h shape as ruled (HARD_STOP_S 8100, SELF_STOP_S 8700, laptop DELETE 2.5 h), at most $1.48,
  **expected about $0.90** (the Gemma fit ran at 1.08× its estimate; the Llama-scaled 0.78× was optimistic).
- Add Gemma to the §43 text with the article note phrased on the failures table (8 of 9) and the gate FAIL stated.

Advisor, 2026-10-28.
