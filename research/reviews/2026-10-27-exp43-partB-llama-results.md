# Results review: §43 Part B, Llama (pod jvp7k2c1i6ngoh; rows 408c678, confirm report b1aa38f)

**Verdict: ACCEPT. Claims B1 and B2 are SHOWN for Llama on the confirmation folds, as pre-registered. The EXPERIMENTS
text may be drafted.**

Recounted independently from `runs/exp43_lens_pod_llama/{logit,jl}/*.json` and `load_family_records` labels for
folds 2–4 (204 paired rows; prompt hashes equal across sizes), with my own percentile bootstrap (10,000 draws):

| claim | Executor | my recount | verdict |
|---|---|---|---|
| B1 log10 best rank, 3B − 8B | +0.181 (+0.106, +0.259) | +0.181 (+0.107, +0.258) | SHOWN |
| B1 secondary, A-share 3B − 8B | +0.078 (+0.029, +0.132) | +0.078 (+0.029, +0.132) | SHOWN |
| B2 logit AUC, wrong vs right (3B) | 0.677 (0.605, 0.746) | 0.677 (0.607, 0.746) | SHOWN |
| B2 J-lens AUC (gate PASS, verdict role) | 0.682 (0.610, 0.751) | 0.682 (0.613, 0.748) | SHOWN |

A/B/C of failures (3B n 95: 0.274 / 0.042 / 0.684; 8B n 70: 0.171 / 0.014 / 0.814), the 3B refusal / wrong split
(A 0.219 / 0.302, C 0.750 / 0.651), emergence medians (0.750 both sizes; 3B quartiles 0.679–0.884, 8B 0.688–0.844)
and the article split (n 31: 3B 0.516 / A 0.484; 8B 0.935 / 0.065) all match. Dev values (B1 +0.163, B2 0.847) are as
reported. Pod ≈ 62 min ≈ $0.61; balance delta $0.576; spend ≈ $18.55 of $30. Every stage ok; no pod live.

## Reading against the proposal's fixed readings

- **C-majority with an A floor: yes.** 0.684 of the 3B's in-view failures have the gold in the final top-10, with
  0.274 never present. Same shape as Qwen (§34: C majority, A floor).
- **Later emergence in the small model: no.** Median logit emergence is 0.750 for both sizes (Qwen: 0.889 vs 0.464).
  The 8B is better on the gold's rank and on the A share, not on how early the gold appears. The proposal's reading
  is therefore **"mixed, family-dependent"** on the depth half, and "property of small readers" on the C half, with
  Gemma and Phi still to come.
- B2 shrinks from dev 0.847 to 0.677 on confirmation; both lower bounds clear 0.5, so the claim stands, but the
  dev value was flattering and the confirmation number is the one to cite.
- Caveat to carry: article-first golds (31 rows) account for much of the 3B's A class (A 0.484 there vs 0.274
  overall); the digit-token caveat applies to any depth comparison with Qwen or Gemma.

## Rulings

- Lens .pt files not copied back: fine for the next two families (the report does not need them).
- Gemma and Phi quote shape: see the Advisor message of 2026-10-27 (2 h and 2.5 h stops; the logit rows decide, so a
  cut fit costs the J-lens row only).

Advisor, 2026-10-27.
