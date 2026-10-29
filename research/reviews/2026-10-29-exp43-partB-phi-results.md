# Results review: §43 Part B, Phi (pod pb6zhpzw85xmet; rows 4ee61dd, confirm report af1f429)

**Verdict: ACCEPT. Claims B1 and B2 SHOWN for Phi on the confirmation folds; J-lens gate PASS. Part B is complete
for all three new families; the Phi block and the final four-family scorecard may be added to §43.**

Recounted independently from `runs/exp43_lens_pod_phi/{logit,jl}/*.json` and `load_family_records` labels, folds
2–4 (204 paired rows; prompt hashes equal across sizes), percentile bootstrap, 10,000 draws:

| claim | Executor | my recount | verdict |
|---|---|---|---|
| B1 log10 best rank, mini − 14B | +0.157 (+0.085, +0.230) | +0.157 (+0.084, +0.231) | SHOWN |
| B1 secondary, A-share mini − 14B | +0.059 (+0.005, +0.113) | +0.059 (+0.005, +0.113) | SHOWN |
| B2 logit AUC, wrong vs right (mini) | 0.636 (0.562, 0.707) | 0.636 (0.561, 0.707) | SHOWN |
| B2 J-lens AUC (gate PASS, verdict role) | 0.598 (0.528, 0.669) | 0.598 (0.529, 0.668) | SHOWN |

A/B/C of failures (mini n 104: 0.202 / 0.000 / 0.798; 14B n 39: 0.103 / 0.128 / 0.769; 7 of the 14B's 24 B rows
article-first), emergence medians (mini 0.938, 14B 0.525, quartiles as reported), digit subset (n 84: 0.917 / 0.976)
match. The gold first token differs between the two tokenizers on 6 of 204 rows (paired by question, as the build
states). Pod ≈ 77 min ≈ $0.75; balance delta $0.723; **spend ≈ $20.06 of $30**. No pod live.

## Reading

- C-majority with an A floor: yes (0.798 / 0.202). B2 holds (0.636, the weakest of the four, lower bound 0.56).
- The larger reader holds the gold more clearly: yes.
- Later emergence in the small reader: yes, and by the largest margin of any family (0.938 vs 0.525), with the
  caveat that the 14B is a bigger size step than the 7B/8B/12B partners and a different tokenizer.

## Final scorecard, four families, confirmation folds, logit lens

| reading | Qwen | Llama | Gemma | Phi | count |
|---|---|---|---|---|---|
| failures C-majority with an A floor | yes | yes | yes | yes | 4/4 |
| B2: a weaker gold rank separates wrong from right in the small reader | yes | yes | yes | yes | 4/4 |
| B1: the larger reader holds the gold more clearly | yes | yes | no | yes | 3/4 |
| the small reader finds the gold later (logit) | yes (0.10) | no (0.00) | no (−0.05) | yes (0.41) | 2/4 |

The proposal's fixed reading is therefore **mixed, family-dependent** on both size halves, and "answer present,
decision late" is a property of the small readers under this prompt in every family. Descriptive beyond the B1/B2
verdicts; the depth numbers carry the article, digit and lens caveats of the earlier reviews.

Advisor, 2026-10-29.
