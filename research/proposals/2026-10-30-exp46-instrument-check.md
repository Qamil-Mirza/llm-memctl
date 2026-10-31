# §46 proposal (rev 0): instrument check — is the relevant intermediate present where the lenses cannot see it?

**Author:** Advisor, 2026-10-30. **Order:** the user, Advisor pane: "go on 2" (D-53), after the Codex measurement-audit
handoff and the Advisor's assessment `research/briefings/2026-10-30-measurement-audit.md` (b363636).
**Status:** pre-registered instrument check. It is NOT the causal-from-maps test that §45 declined to name (D-52); it
asks whether the §45 negative is a property of the readout family or of the model's route. Prior art predicts the
answer on toy tasks (Feng & Steinhardt 2023, binding IDs at entity positions; Geva et al. 2023, late attribute
extraction), so §46 is a confirmation on our model and our case list, worth doing once and cheaply, and its only path to
a thesis claim is the bridge in §6, which is a separate amendment.

## 1. Question and the three outcomes fixed in advance

§45 (F13): three vocabulary lenses read the RC intermediate (the supervisor's name) and the TU competitor/gold (the
superseded / current city) only at layers 30–33 at the producing position; at the last prompt token the intermediate's
median rank is 17k–113k through layer 29 (dev audit). Either (i) the variable "which name is A's supervisor" is linearly
present in the residual stream before layer 30 in directions the unembedding does not see (**instrument blind spot**),
or (ii) it is not there until the late attention step writes the answer (**route late, lenses adequate**), or (iii) it
is present but moving it does not move the answer (**represented, not used**). The decision rule (§5) maps the two
measurements below onto these three; nothing else is claimed.

## 2. Material (fixed, already on disk)

- Model: Qwen2.5-3B-Instruct, fp32 on the laptop, bf16 on a pod if approved (the §45 smoke pinned bf16 = fp32 final
  top-1 on 3 cases; the probe layers and the intervention windows are read on the same device for dev and held).
- Prompts: the §45 case list `runs/exp45_maps/cases.jsonl`, tasks RC and TU only, all variants, formats F1 and F2,
  hardness H0–H2 (2 tasks × 12 base × 4 × 2 × 3 = 576 prompts; dev = even base index, held = odd, as §45). Outcomes
  from `runs/exp45_maps/j56-jrefit-logit/stage1/*.json` (no regeneration; the chat template sha must match).
- Positions (per prompt, from `position_tokens`): P_last = last prompt token; P_B = the first content token of the
  supervisor's name in the "{A}'s supervisor is {B}" sentence (RC) / of each city in its fact sentence (TU); P_q = the
  subject's name token inside the question. Residuals captured at every layer 0–35 (output of decoder layer l, pre
  final norm, as `memctl/analysis/patching.py`).

## 3. Measurement A — contrastive probe (correlational)

- **Target.** RC: the identity of B (the supervisor named in the prompt), a 12-way label over the name pool, read from
  the residual at P_last and at P_q, per layer. TU: the identity of the *current* city (gold) and, separately, of the
  *superseded* city, 12-way over the city pool, at P_last, per layer.
- **Fit.** Multinomial logistic regression (scikit-learn, L2, C chosen by 5-fold CV on dev only), one probe per
  (task, position, layer), trained on **dev** prompts (all outcomes; outcome recorded), standardised features.
- **Read once on held.** Accuracy per layer with Wilson interval; held prompts whose (A, B) or (A, C1, C2) combination
  appears in dev are excluded before any read (the Executor reports the exclusion count from `cases.jsonl` first).
- **Controls, same features, same fit.** (c1) Hewitt–Liang control task: each base case is assigned a random name
  label (seed 46); selectivity = probe accuracy − control accuracy. (c2) **Presence control**: predict the identity of
  the *first distractor name* present in the prompt (H1/H2 only). A probe that decodes B but not an equally present
  distractor reads relevance, not mention. (c3) Chance = 1/(number of names present in the prompt), reported beside.
- **Readout criterion (fixed).** "Decodable at layer l" = held accuracy ≥ 0.70 with lower bound ≥ 0.55 AND
  selectivity ≥ 0.30 AND presence-control accuracy below the probe's lower bound. The **first decodable layer** per
  (task, position) is the result; "none before layer 30" is a legitimate result.

## 4. Measurement B — interchange intervention (causal, RC only)

- **Pairs.** Two solved dev prompts P and P′ of the same base case, variant family and hardness that differ only in
  the supervisor's name (v0 vs v1: B vs B′) and therefore in the gold (C1 vs C1′), with both "{B} works in {C1}" and
  "{B′} works in {C1′}" present in both prompts (H1/H2 rows where B′'s fact is a distractor; the Executor reports how
  many such pairs exist in dev and held; if fewer than 20 held pairs exist, the H0 rows are augmented by one extra
  distractor fact per the §45 pool, written to `cases46.jsonl` before any run, and marked as a §46 addition).
- **Site S1 (primary): P_B**, the supervisor-name token in sentence 1. For layer l in 0…35: run P with the residual at
  (P_B, l) replaced by P′'s (`steering.overwrite_spec`), layers > l on P's own cache, greedy ≤ 8 tokens; outcome
  classified by the §23 match as C1 / C1′ / other / abstain. **Transfer(l)** = share of pairs producing C1′.
- **Site S2 (secondary, descriptive): P_last.** Same, reported as a layer profile beside the §45 lens ranks at that
  position; a flip at layers ≤ 29 while the lens rank is in the thousands is informative about the readout, a flip
  only at layers ≥ 30 is answer injection and is labelled so.
- **Controls (§44 rev 6 lesson; scale-matched).** (k1) difference-matched random vector of norm ‖h_P′ − h_P‖ added at
  the same site and layer; (k2) **same-answer donor**: a prompt P″ with a third name B″ as supervisor but B″ working in
  C1 (answer unchanged) — Transfer must be ≈ 0 for C1′ here; (k3) sanity: patching at layer 0 at P_B must reproduce
  the re-tokenised prompt's output on ≥ 0.95 of pairs (the token-substitution floor), or the capture path is wrong.
- **Criterion (fixed).** "The binding is carried at S1 through layer L*" = Transfer(l) ≥ 0.70 with lower bound ≥ 0.55
  and k1 ≤ 0.10, k2 ≤ 0.10, at every l ≤ L*. L* ≥ 10 counts as mid-stack carriage. Read once on held pairs after the
  dev profile is committed.

## 5. Decision rule (fixed now)

| probe A: first decodable layer at P_last (RC) | intervention B: L* at S1 | verdict |
|---|---|---|
| ≤ 29 | ≥ 10 | **instrument blind spot**: the variable is linearly present mid-stack and causally carried; vocabulary lenses cannot see it (the expected result from prior art) |
| none ≤ 29 | < 10 or not shown | **route late, lenses adequate**: the §45 picture is the model's route |
| ≤ 29 | < 10 or not shown | **represented, not used** at those sites (Walters et al. framing); report the layer gap |
| none ≤ 29 | ≥ 10 | binding carried at the entity position but not decodable at the answer position before 30: route late at P_last, instrument blind at P_B; stated as such |

TU probe results are reported descriptively (current vs superseded city decodability by layer) and do not enter the
verdict. No sentence about "erased", "suppressed" or "overridden" is available from §46.

## 6. What follows, and what does not

- Only the first verdict (or the fourth) opens the bridge: the same probe and intervention on the 140 §44 rev 8 rows
  (is the gold's identity decodable mid-stack on refusals; does moving it change refuse vs answer). That is a separate
  amendment with its own criteria; it is the thesis test. §46 alone reproduces Feng & Steinhardt on a 3B.
- No DAS, SAEs or Patchscopes in §46. No new lens. No claim of a methods contribution (Codex point 6 stays conditional).
- D-45 stopping rule: whichever verdict, §46 is documented; no rescue variant without the user.

## 7. Compute, time and the pod question

Compute is small; the two-day figure in the briefing was build and review latency, not GPU time.

| step | laptop fp32 (3B) | one A40 bf16 |
|---|---|---|
| residual capture, 576 prompts × 36 layers × 3 positions (one forward each, no generation) | ≈ 1–2 h (≈ 6–12 s per prompt; §45 fallback estimate) | ≈ 10 min (§45 measured 1.78 s per prompt *with* generation) |
| probes: 2 tasks × 3 positions × 36 layers × (probe + 2 controls), sklearn | ≈ 20 min CPU, either way | same (CPU) |
| intervention: ≈ 60–120 pairs × 36 layers × (patched + k1 + k2) ≈ 6.5k–13k short generations | ≈ 6–9 h (§44: 206 s per 7B row for 38 passes; 3B ≈ 2.5× faster) | ≈ 1–1.5 h |
| **total** | **≈ 8–11 h, $0** | **≈ 1.5–2 h ≈ $0.90–1.20 at $0.59/h** |

Several pods in parallel would save under an hour of the A40 time, break the never-two-pods rule, and leave the
calendar time where it is: Executor build (≈ half a day), the Advisor's build review, the dev read, then the held read.
**Pod quote, if the user wants it: ≈ $1.00–1.50, hard stop $2.50 (HARD_STOP_S 14400); programme after ≈ $22.9 of
$30.** The laptop path is the default and costs nothing; the pod needs the user's approval line in both panes.

## 8. Build (the Executor; nothing runs on held before the Advisor's build review)

`scripts/exp46_capture.py` (residual capture at P_last, P_q, P_B/city positions, all layers, resumable, `.npz` per
prompt under `runs/exp46_probe/<device>/`), `scripts/exp46_probe.py` (dev CV fit, controls c1–c3, held read behind a
`--held` flag that refuses to run unless `design_committed.json` exists), `scripts/exp46_patch.py` (pairs, S1/S2, k1–k3,
reusing `memctl/analysis/patching.py` capture and `steering.overwrite_spec`), `scripts/exp46_report.py` (tables, Wilson,
the §5 verdict computed, never typed). Tests: pair construction yields distinct golds; the layer-0 sanity k3 on 3
pairs; probe shapes; the held flag refuses without the commit marker. Report the dev/held pair counts and the held
(A, B) overlap exclusions first, before any model run.

## Rev 1 (2026-10-30, Advisor rulings on the Executor's pre-run numbers; before any model run)

The Executor's three pre-run reads, checked by the Advisor against `runs/exp45_maps/cases.jsonl`: (1) the §4 pair
condition is met by **0** RC rows in either split, because v1 keeps the gold on 72 of 72 bases (the new supervisor
inherits the city) and the v1 name never occurs in the v0 prompt; (2) **0** held (A, B) or (A, C1, C2) combinations
occur in dev, so no probe exclusions; (3) laptop fp32 timings make the intervention 110–210 h on the laptop, so the
laptop path is dropped for Measurement B (capture and probes may still run anywhere). Every pool name and city is a
single token with a leading space; at H2 at least 3 names and 4 cities are unused in every prompt, so the triple
design below fits at every hardness. Rulings:

1. **Pairs (replaces §4 "Pairs").** New file `runs/exp46_probe/cases46.jsonl`, marked "§46 addition", never mixed into
   §45 tables. For every RC v0 and v3 row (two orders) × F1/F2 × H0–H2 (144 rows, 72 dev / 72 held), a **triple**
   P / P′ / P″ built from the same text: two extra facts "{B′} works in {C1′}" and "{B″} works in {C1}" are inserted in
   **all three** at one seeded slot (same slot in the triple; seed 46), with B′, B″ names and C1′ a city unused in that
   prompt; the supervisor named in sentence 1 is B (P, gold C1), B′ (P′, gold C1′) or B″ (P″, gold C1, the k2
   same-answer donor). Per triple the Executor asserts alignment (`check_alignment` style: equal token length, P_B at the
   same index, exactly the one token at P_B differing); a triple failing it is dropped and counted. **Solved filter:**
   one unpatched greedy generation per prompt; a pair enters Measurement B if P and P′ are both solved; k2 additionally
   requires P″ solved. If fewer than 30 held pairs survive, the §4 criterion is still computed but read as under-powered
   (stated in the text), not as a verdict row in §5; the expected count is 30–45 (§45 RC solve rate ≈ 0.64).
2. **Readout (replaces "greedy ≤ 8 tokens").** Greedy generation that stops at the first pool-city token, at ` unknown`,
   or at max_new (F1 48, F2 64); outcome C1 / C1′ / other / abstain by that first city. **Validation, dev baselines
   only:** the first-city rule must agree with the §45 outcome rule (full generation, §23 match) on ≥ 0.95 of the
   unpatched dev generations; if it does not, the §45 rule decides and max_new becomes 48 / 160 as in §45. Forced-prefix
   readouts are rejected (they change the prompt). Reported beside: whether the generation names the supervisor before
   the city (a name-copy route is a legitimate transfer result and is described as such).
3. **Presence control c2.** Its first-distractor labels on held (Rachel, Ben, Laura, Kate) do not occur in dev, so c2 is
   read by 5-fold cross-validation on dev only, against the main probe's dev-CV accuracy at the same layer; the §3
   criterion's presence clause is evaluated there. The held read is the main probe with c1 and chance. No held tuning.
4. **Pod budget and order (D-54).** All GPU work on the one A40: capture (≈ 10 min), baselines, interventions; probe
   fits on CPU. Batch the 36 layer variants of a pair into one batch and cache the shared prefix before P_B. Fixed
   order, so a time cut loses the least important part last-first: (i) baselines for all triples; (ii) S1 patch, all 36
   layers, dev then held; (iii) k1 and k2 at S1, all layers; (iv) S2 at layers {0, 3, …, 33} with k1 only (descriptive).
   The D-22 smoke (3 triples) measures the per-pass rate; the Executor projects the total before continuing and **stops
   and asks if the projection exceeds $1.80** (margin under the $2.50 hard stop). Quote revised to ≈ $0.90–1.80.
5. Everything else in rev 0 stands: probe design (§3), S1/S2 sites, k1/k3 controls, the §5 decision rule, no held read
   before `design_committed.json`, the §44-row bridge as a separate amendment.

## Rev 2 (2026-10-30, Advisor rulings on the Executor's build notes)

- **H2 names.** The Advisor's rev 1 count ("≥ 3 free names at H2") was wrong: it missed the "works as" distractor
  facts. Whole-word count: H2 rows have exactly 1 unused name (always the §45 v1 supervisor). Ruling: the Executor's
  option (b). At H2, B′ = the one unused name (the main P/P′ pair stays clean); B″, used only by the k2 same-answer
  donor, reuses a distractor name, preferring a name whose distractor facts carry no city at all ("works as"), else one
  with a "lives in" fact; each such triple is flagged `b2_reuses_distractor` (and `b2_has_rival_city` when a "lives in"
  fact exists), and k2 is reported split by the flag. 144 triples stand. Option (a) would leave ≈ 20 held pairs; option
  (c) alters §45 text.
- **Position-indexed hook.** Accepted (`steering.overwrite_spec` only patches the last position). Required test: with
  the site set to the last position, the new hook reproduces the `overwrite_spec` output on one prompt to within bf16
  tolerance, so the overwrite semantics are the same object.
- **k3.** Accepted: the token-substitution floor is the input-embedding overwrite at P_B (must reproduce P′'s output on
  ≥ 0.95 of pairs); the layer-0 *output* patch is part of the S1 profile, not the sanity.
- **c1 (Hewitt–Liang) and the probe's dev CV.** The Executor is right that per-base-case random labels are at chance on
  new held bases by construction. Ruling: the main probe's dev CV is **grouped by base case** (no base-identity leakage
  into the generalisation estimate); c1 is read by dev CV with prompt-level folds (where a representation that encodes
  base identity can be memorised, which is what the control measures); the presence control c2 by dev CV as rev 1.
  The **held** criterion is accuracy ≥ 0.70 with lower bound ≥ 0.55 and chance reported; the selectivity and presence
  clauses of §3 are dev-CV conditions, read before the held read. Held selectivity is reported as accuracy − 1/12 and
  labelled as such.

## Rev 3 (2026-10-30, Advisor ruling on the probe fold scheme)

Rev 2's "dev CV grouped by base case" gives accuracy 0 by construction: each probe label (RC supervisor, TU current,
TU superseded) occurs in exactly one dev base case (base i uses perm[i+1], perm[i+2]; dev = even i), so holding out a
base holds out its label. The held read is unaffected (every held label occurs in dev as some dev base's v1 B′; the
held bases are unseen name/city combinations). Ruling: the Executor's scheme. Dev CV folds grouped by **(format ×
hardness) cell**, 6 folds, every label present in train; identical folds for the main probe and for c1 (labels per base
case), so c1 measures how far base-case identity alone is decodable and selectivity = probe dev-CV − c1 dev-CV on the
same folds; c2 on the H1/H2 cells (4 folds). Dev CV is therefore a within-base estimate (generalisation across format
and hardness, base identity subtracted via c1); the across-base test is the single held read, criterion as rev 2.

## Rev 4 (2026-10-30, build review `research/reviews/2026-10-30-exp46-build.md`): readout = the conclusion

Rev 1 ruling 2 (first city, validated against the §45 rule) is withdrawn: the triple design puts C1′ in every prompt and
the model restates the facts before concluding, so first-city and the §45 "both cities → other" rule both misread the
model's correct conclusion (Executor's dev baselines, 1 of 21 rows, structural). Readout fixed now: F2 = the city on the
`Answer:` line (§45 stop rule, max_new 160); F1 = the last pool city or "unknown" in the generation to end of turn or
max_new 96. Classification C1 / C1′ / other / abstain by that item; no city → other. Used for the solved filter, k3,
Transfer and S2 alike. `validate` reports the three rules' agreement and solve rates on the dev baselines and lists
disagreements; it does not switch rules. Name-copy share computed on the text before the conclusion city.
