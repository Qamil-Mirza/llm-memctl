# Draft record changes for §53 (awaiting the user's "Go"; not applied)

Applies to `research/DECISIONS.md` (new D-77) and `research/STATE.md` (new F19; F18's last sentence; Spend). The
EXPERIMENTS.md §53 commit hash goes in where marked 54a00e1.

---

## D-77 (2026-11-07) — §53 complete; STATE F19 added; closed under D-45

- **What.** §53 (rev 1b) ran once on N-held. Each of its four pre-registered claims passed its own test:
  - **C1:** the two-head ablation releases more false abstentions than 20 random L32×L33 pairs (0.751 against a max
    of 0.464).
  - **C2:** doubling the pair creates more abstention on correct rows than doubling 20 random pairs (0.381 against
    0.345; the narrowest pass).
  - **C4:** the L35 edit's U1 beats 20 sign-aligned, RMS-matched random read vectors (0.403 against 0.300), with Uns
    below their median.
  - **C5:** submission 2's four activity predictions hold on the unedited model (IUT p ≈ 1e-4).
  - No family-wise claim. Under exchangeability, P(at least one passes) ≤ about 0.19.
- **Beside it (descriptive).** §51's frozen logit bias (B1 −8) matches both edits on U1 (0.406 against 0.403 and
  0.414). The edits beat random edits of their own form, not the simplest baseline. This is consistent with §51's
  "B1 inconclusive".
  - LongMemEval: the edits release most in-view refusals (20 and 19 of 26), but only 8 and 7 of those contain the
    gold.
  - B-opt is not estimable.
- **Record.**
  - EXPERIMENTS.md §53 at 54a00e1; manifest b2b14bac at a177536 (before any N-held output).
  - Pod outputs at 7b9e271 (report.json f1ba6dad).
  - Advisor review `research/reviews/2026-11-07-exp53-results-review.md` (ACCEPT; independent recount of all 87 arms,
    0 mismatches; corrected at e3e051e).
- **Spend.** Pod A40 4.383 h ≈ $2.59 (quote $2.55, hard stop $3.10). RunPod total ≈ $33.75 of $36. The §51 editor
  API is unchanged at $3.58 of its $50.
- **Closure.** No rescue variant, no second N read, and no tuning of B1 or the edits on N. Open follow-ups go to
  OPEN_QUESTIONS only, if the user wants them: a predictor/interaction test of submission 2's account, and a
  head-to-head of the edits against B1 on fresh data with power for a difference.
- **Authority.** The user, Advisor pane, 2026-11-07: "do draft the decision entry and the state update". Applied
  after the user's "Go".

---

## STATE.md, new row F19 (after F18)

| F19 | On a new controlled split (N, seed 54; disjoint from E and S keys; 867 of 1,350 keyed rows share an H key, as reported), both §51 edits beat random edits of the same form. Zeroing the `o_proj` columns of L32h2 + L33h10 releases 0.751 of false abstentions, against at most 0.464 (median 0.163) for 20 random L32×L33 pairs. Doubling them turns 0.381 of correct answers into "unknown", against at most 0.345 for doubled random pairs. The ablation's unsupported-answer rate (0.075; B0 0.031) is also the highest of the 21. The rank-1 L35 edit's U(λ=1), 0.403, beats all 20 random read vectors with the same write vector and RMS gate (max 0.300, median −0.069), with Uns 0.058 against a median of 0.228. The random reads either answer unanswerable rows (Rel 0.94 with Uns 0.88) or break generation (degenerate on up to 100% of rows). The editor's four activity predictions for the L35 signal hold on the unedited model: evidence removed −3.77, distractors added −1.12, unrelated vs absent +0.22, and city changed equivalent within ±1.12. Only the activity part of that account was tested. Descriptive: §51's frozen logit bias on "unknown" (−8) gives U1 0.406, against 0.403 (L35) and 0.414 (two-head). On N neither edit improves on the simplest output bias. On 126 LongMemEval dev rows the edits release 20 and 19 of 26 in-view refusals, but only 8 and 7 of those contain the gold (damage 2 and 3 of 35) | §53 (2026-11-05 to 11-07): one A40 pod, 4.383 h ≈ $2.59; manifest b2b14bac committed a177536 before any N-held output; report f1ba6dad at 7b9e271; EXPERIMENTS.md §53 at 54a00e1 | Pre-registered (§53 rev 1, 1a, 1b; rev 1b sign-aligned the random reads before any held output). C1, C2, C4 and C5 each SHOWN at its own bar (rank 1 of 21 plus co-requirement; IUT 0.05); no family-wise claim (≤ ~0.19 under exchangeability); C2 is the narrowest pass; B1-ref, C3, A-single, A-late, D-randwrite, B-DLA and arm C descriptive; B-opt not estimable (n = 0). Advisor recount from per-row files 0 mismatches (review 2026-11-07, corrected e3e051e); closed under D-45 (D-77) |

## STATE.md, F18: last sentence

- Old: "One model, synthetic templates; no random-head or natural-data control yet (§53)"
- New: "One model, synthetic templates; random-edit and natural-data controls in F19 (§53)"

## STATE.md, Spend: first sentence

- Old: "About $31.16 on RunPod for all experiments to date (2026-11-07; §51 pods S $4.38 and H+T $2.57; the §51 editor
  API, $3.58, is billed separately); …"
- New: "About $33.75 on RunPod for all experiments to date (2026-11-07; the §53 A40 pod about $2.59; §51 pods S $4.38
  and H+T $2.57; the §51 editor API, $3.58, is billed separately); …"
- The rest of the sentence is unchanged.
- The header "findings table to 2026-11-02" → "findings table to 2026-11-07".
