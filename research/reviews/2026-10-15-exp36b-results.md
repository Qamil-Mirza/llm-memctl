# Review: §36b results (pod uul64th11xxqip, 2026-10-15; reports at 14e8478)

Advisor acting as Reviewer. Authority: pre-registration frozen at d2a67f2 + a3a3f9c, amendment 1 f14fe54, amendment 2
(§36b) 30415c3 with probe pick (a) at 84e1106; `runs/exp36b/{qwen3b,qwen7b}/<item>/fold{0..4}.jsonl`,
`runs/exp36b/pod.log`, `runs/exp36b_report_{dev,confirm,allfolds}.md`. My recompute: an independent implementation of the
AUROC, matched-coverage policies, paired risk differences and the question-level bootstrap (labels via the reviewed
§35 loader), run on folds 0–1 first and compared with the dev report, then on folds 2–4.

**Verdict: ACCEPT the reports and the verdicts.** Every number I recomputed matches to the third decimal, with interval
ends within bootstrap resampling noise (≤ 0.002). Every verdict follows the pre-registered rule. The section text, when
drafted, must carry the items in §4 below.

## 1. Process: clean
- Design fixed before any confirmation number: 30415c3 (21:0x UTC) and the probe pick 84e1106 (21:4x UTC) precede pod
  creation 22:26:26Z; chain 22:28:20Z–22:41:41Z (846 s: install 145, 3B 250, 7B 451); dev report, then `--confirm`,
  then all folds; pod deleted 22:42:34Z; about 16 min ≈ **$0.16** (programme ≈ $17.50). Account 0 pods after.
- Smokes PASS (3B 5/5, 7B 5/5 at mass ≥ 0.5). One engine `vllm-0.8.5`, frame (a), scorer `v2-bytedecode` in every row.
  No stage cut; counts as in run 1 (3B 224 V rows, 180 Cut-F, 216 CF, 7 Abs; 7B + 322 V_own/C4).
- Reliability rule: 0 low-mass rows on every item of both readers (my count on all five folds). Claims computable.
- Run 1's confirmation folds were never read beyond label-free mass (run-1 review §2.4) and stay sealed.

## 2. Numbers against rows (confirmation folds 2–4, 3B primary)
165 fixed rows, 140 with a candidate, 25 forced abstentions, |C0| = 100; no missing V or C4 score.

| quantity | report | my recompute |
|---|---|---|
| Claim 1 AUROC, candidate rows (84 pos / 56 neg) | 0.573 (0.478, 0.666) NOT SHOWN | 0.573 (0.476, 0.668) |
| Claim 2 risk(C3) − risk(C0) at |C0| | −0.060 (−0.132, +0.012) not tested | −0.060 (−0.131, +0.013) |
| Claim 3 risk(C3) − risk(C4) | −0.020 (−0.090, +0.049) not tested | −0.020 (−0.088, +0.050) |
| C0 / C3 / C4 risk at 100 emitted | 0.430 / 0.370 / 0.390 | 43 / 37 / 39 wrong of 100 |
| C4 (m) AUROC | 0.502 (0.398, 0.603) | 0.502 (0.399, 0.603) |
| natural flag AUROC, candidate rows | 0.455 (0.381, 0.532) | 0.455 (0.381, 0.533) |
| C2 argmax-A | 47 emitted, risk 0.319 | 47 emitted, 15 wrong |
| Control A, 3B drop on correct B (n 84) | +0.235 (+0.170, +0.302) | +0.235 (+0.170, +0.302) |
| Control A, 7B drop | +0.508 (+0.419, +0.596) | +0.508 (+0.418, +0.595) |
| 7B verifying the 3B's candidates, AUROC | 0.692 (0.600, 0.777) | 0.692 (0.605, 0.777) |
| Cut-F 3B mean s / argmax-A share | (0.655·28 + 0.496·92)/120 = 0.533 / 0.150 | 0.533 / 0.150 |

Dev folds (0–1) agree the same way (Claim 1 0.559; Claim 2 +0.014; Claim 3 +0.043; Control A +0.249 / +0.469; C2 34
emitted, risk 0.294).

## 3. Reading, against the pre-registered readings and the prediction
- **Claim 1 NOT SHOWN** → the pre-registered reading is "evidence evaluation is itself part of the 3B's bottleneck
  (Outcome B)". That stands, with a sharper statement the controls allow: the 3B's support score **does** respond to
  the evidence (Control A +0.235, length control rules out length, A′ 0.715: it tells an evidenced candidate from an
  unevidenced one), but with the evidence in view it barely tells a right candidate from a wrong one (0.573). It is an
  evidence-presence detector, not a verifier. This matches §34c ("present, not extracted").
- **The 3B's own refusals carry no signal about which forced candidates are right** (natural flag 0.455 on candidate
  rows, confirmation). This confirms the amendment-1 dev finding and sharpens §35: the refusals are not tracking
  correctness.
- **The 7B as checker of the 3B's candidates:** AUROC 0.692, Control A drop +0.508, 7B − 3B drop +0.273 (+0.187,
  +0.359); at matched coverage risk −0.090 (−0.171, −0.010) vs C0. Descriptive (judge shares the 7B's weights; the
  60-row hand audit at 58/60 limits, not removes, that). Direction as predicted (≥ 0.75 was predicted; 0.692 falls
  short). This is the architecture pointer the proposal named (small generator + larger checker), still unproven.
- **Prediction scorecard (recorded in the proposal):** Claim 1 "SHOWN but weak, 0.60–0.68" → wrong, NOT SHOWN at
  0.573. Claim 3 NOT SHOWN → not reached. Control A "small for the 3B (< 0.1)" → **wrong**, +0.235. "Clearly larger
  for the 7B" → right. 7B AUROC ≥ 0.75 → slightly short. The §34c-based prior (the 3B ignores the evidence) was wrong
  for this readout.
- **Claim 4 (descriptive):** the dev-fitted τ = 0.513 does not hold the risk cap on confirmation (risk 0.357 (0.233,
  0.484) at coverage 0.339 vs r* = 0.30). C2 (argmax A) halves coverage (0.285) for risk 0.319; C2 − C0 −0.111
  (−0.229, +0.011) crosses zero. No gating rule from the 3B's own readout is shown to beat its natural policy.
- **Control B (primary, numbers with no other copy):** 8/10 follow in both contexts for both readers; n 10 < 20,
  descriptive as pre-registered. Only the both-contexts row is to be quoted.

## 4. What EXPERIMENTS.md §36/§36b must carry (Executor; I review the text next)
1. Run 1 and run 2 both reported: run 1 "not reliable" with the two causes and $0.17; §36b with the probe (pick (a),
   84/84, order, label-free gate, later wordings descriptive on 25 questions) and $0.16.
2. The confirmation table as in the report; dev figures in brackets; Claims 2–3 labelled "not tested" with their point
   estimates, never "shown" or "narrowly missed".
3. The reading in §3 above, in that order: NOT SHOWN; presence-detector-not-verifier (with Control A, LC and A′);
   refusals carry no signal; the 7B checker descriptive with the circularity caveat; the prediction scorecard.
4. One descriptive add, free: the paired AUROC difference, 7B − 3B on the same 140 candidates, with its interval (the
   prediction named it; the report has only the two separate AUROCs).
5. Caveats: wording chosen for one-letter compliance on dev folds (a second elicitation; "elicited", never "the reader
   knew"); labels from one 7B judge (understates by ≈ 0.1; audit 58/60); exact-span cut leaves derivable content
   (Cut-F judged-correct rows get mean s 0.655 from the 3B: plausibility); Control B n 10; coverage/risk are on fixed
   candidates, not deployed accuracy.
6. Costs beside: C1 forced adds 13 wrong on 165; C3 at matched coverage −6 wrong (17 refusals recovered, 9 turned
   wrong); Cut-F emitted at τ 0.183 (3B) / 0.317 (7B); Abs 4 rows, 0 emitted by the 3B.
7. Run 1's confirmation folds: state that they were never read beyond label-free mass and stay sealed.

## 5. Proposed STATE.md change (for the user, after the section is accepted)
Add to F6's evidence cell: "§36b: the 3B cannot check its own forced answers against the evidence (AUROC 0.573 (0.478,
0.666), NOT SHOWN); its support score detects whether evidence is present (+0.235 drop when cut) but not whether the
candidate matches it; its refusals carry no signal about candidate correctness (0.455). The 7B checking the same
candidates reaches 0.692 (descriptive)." Open question: does a larger checker over a small generator beat the larger
model answering alone, at what cost? (new Q row). Spend ≈ $17.50 of $20.
