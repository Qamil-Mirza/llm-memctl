# §39 pre-registration: is the answer already computed under the refusal? The §38 direction added at the `:` position only (Advisor, 2026-10-23, rev 1)

**Status: development only (folds 0–1, laptop, $0). A pod confirmation is a separate quote, not part of this proposal.**
Opened by Q16 and §38 caveat 2. User approval: "Sure go for it" (Advisor pane, 2026-10-23) to the record changes and
the offer to draft this as §39; recorded as D-31 (draft, build and free laptop run; no spend).

## 1. Question and the two mechanisms it separates

§38 showed that adding the content-free direction v_29 (α 2, unit ‖v‖) at the `:` token of `Answer:` **and at every
generated token** lifts the 3B's refusal and releases the gold on 0.517 of refusal rows with the evidence in view. Two
mechanisms fit that:

- **M1, answer in place.** Under the refusing prompt the 3B has already computed the answer path at the `:` position; the
  direction only flips the refuse/answer disposition there, and generation then runs on its own.
- **M2, disposition held off.** The refusal re-asserts itself at each generated token unless the push is maintained; the
  evidence is read during generation once the push keeps the disposition off.

§39 adds the same vector at **step 0 only** (the `:` position; `HookSpec.steps = {0}`, already in `steering.py` for the
overwrite sanity) and leaves the generated tokens unsteered. Everything else is §38's dev set-up.

## 2. Data and arms (folds 0–1 only; the script refuses other folds)

- Rows: the 28 development refusal rows R of §38 (reproduced in `runs/exp38_dev`), the same 13 estimation pairs and 15
  unseen rows; the 8 damage rows C of §38. The direction is **`runs/exp38_dev/qwen3b/directions.npz` layer 29**, reused
  byte-exact; nothing is re-estimated. Reproduction: A and B flags against the §38 dev rows (bound 0.8 as §38).
- Arms, all at layer 29, α 2, unit ‖v‖, max 16 new tokens, greedy, newline/EOS stop, laptop fp32:
  - **Steer-all**: §38's arm (cached in `runs/exp38_dev/qwen3b/sweep.jsonl`, `steer:L29:a2:vnorm`; reused, not rerun).
  - **Steer-0**: add at step 0 only, intact prompt. Primary arm.
  - **Steer-0-cut** and **Steer-all-cut**: the same two on the §38 evidence-cut prompt with the N-cut note (no marker →
    no note; §38 amendment 1.4). Steer-all-cut was not run at layer 29 on dev and is run now.
  - **Steer-k**, k ∈ {1, 2, 4}: add at steps 0..k, intact prompt; descriptive dose curve.
  - **Random-0**: the §38 seeded layer-29 random vector at step 0 only; specificity floor for Steer-0.
  - **Damage-0**: Steer-0 on the 8 C rows.
- Classes: refusal / degenerate / answer as §38 (CLASSIFIER_VERSION v2). Gold: D1 string rule, **PROVISIONAL** (no judge
  on the laptop); ans is the primary quantity because it is label-free.

## 3. Readings, fixed now (development only; no claim on folds 2–4)

Let a0 = ans(Steer-0), aA = ans(Steer-all) on the 28 rows; g0, gA the D1 gold rates.

- **M1 supported:** a0 ≥ 0.8 · aA and g0 ≥ 0.8 · gA, and ans(Random-0) ≤ 0.2 · a0. The `:` state carries both the switch
  and the answer path.
- **M2 supported:** a0 < 0.5 · aA. The refusal re-asserts at the generated tokens; the push has to be maintained.
- **Split:** a0 ≥ 0.8 · aA but g0 < 0.5 · gA. The first token is released at the `:` position, but the content needs the
  push during generation; read with the dose curve.
- Anything between is "undetermined on 28 rows"; the dose curve is printed in every case.
- Steer-0-cut beside Steer-0, Steer-all-cut beside Steer-all: the plausibility share of each, descriptive.
- Damage-0 on 8 rows, descriptive.

**Prediction, recorded now.** §38's texts are extracted from the evidence in view, and the §34 lens showed ` unknown`
winning late. I expect **M2 or split**: a0 about 0.3–0.6 against aA 0.93, g0 well under gA (8/28); the dose curve rises
with k and reaches Steer-all by k = 4. Random-0 ≤ 0.1. Damage-0 none.

## 4. Time and rules

- About 28 × 6 + 8 generations ≈ 180 short generations with prefill ≈ 30 s each ≈ 1.5 h on 8 threads under
  `GUARD_VM_GB=22`; 5-row timing first; over 5 h projected → drop Steer-k to k = 2 only.
- Folds 2–4 untouched; no pod, no spend; the result is **development only** and is written into EXPERIMENTS.md §39 as
  such. If the user later wants it on the sealed folds, it is a separate quote (about $0.15 on an A40).
- Executor may raise F-items at the build stage; rulings are appended as amendment 1 before `--run`. Build review by the
  Advisor before `--run`; the 5-row timing and the stage logs come to the Advisor as in §38.
