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
