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

## Amendment 1 (Advisor, 2026-10-23, before `--run`; the Executor's F1–F6 at the build stage)

1. **Step semantics (F1, accepted; recorded).** Step 0 is the forward pass at `:`, which produces the first generated
   token; step j ≥ 1 is the pass at the j-th generated token. Steer-0 changes the first token **and leaves the `:`
   position's keys and values above layer 29 pushed**, which every later token attends to. M1 is therefore "the pushed
   `:` state carries the answer path", not "nothing downstream is touched". Steer-k = steps {0..k}.
2. **Reproduction (F2, accepted with the gate kept on flags).** The regenerated unsteered A and B must match the §38 dev
   rows by the §38 flag rule (bound 0.8, as pre-registered); byte-identity with the cached text is printed per row and
   as a rate. A row that matches flags but not bytes is kept and marked. If byte-identity is below 0.9 the engine is not
   the one the Steer-all rows came from and the run stops before any steered output is read.
3. **Small-n arithmetic (F3, accepted; recorded).** With gA = 8/28, M1 needs g0 ≥ 7/28 and the split needs g0 ≤ 3/28; a
   single row moves a verdict. The verdict is written with the counts, never the rate alone.
4. **Random-0 vector (F4, accepted).** The seeded layer-29 vector (rng seed [38, 29], rescaled to ‖2·v_29‖) is rebuilt
   deterministically and its sha printed; it is not the pod's layer-29 random vector (bf16 re-estimate, different norm).
5. **Time (F5, accepted).** About 50 min with one prefix cache per prompt shared across arms; the 5-row timing decides.
   The §4 fallback (drop Steer-k to k = 2) applies only if the projection exceeds 5 h.
6. **Text identity (F6, accepted, descriptive).** Per row: first-token agreement of Steer-0 with Steer-all, and whether
   the full text is identical. Identical text on most rows would show the step-0 push alone fixes the whole answer.
7. Reuse guard as built: Steer-all rows are reused only if engine, classifier version, wording sha, vector sha and max
   tokens all match the §38 dev manifest.

## Amendment 2 (Advisor, 2026-10-23, after the development run; D-33 item 1): the first-token-only clean-cache control (§39b)

**Question.** §39's Steer-0 leaves the pushed `:` key/value above layer 29 in the cache for every later token (caveat F1).
Is the effect carried by the first generated token, or by that pushed state? This arm removes the pushed state entirely.

**Arms** (the same 28 dev R rows, engine, wording sha, max 16 tokens, greedy; folds 0–1 only; laptop fp32; $0):
- **Steer-first (primary).** The first generated token of §39's Steer-0 row (`gen_ids[0]` from
  `runs/exp39_dev/qwen3b/arms.jsonl`, byte-exact, with a check below) is fed as a forced token on an **unpushed** cache:
  unpushed prefill through `:` (the A arm's cache), then the forced token, then greedy continuation with no hook at any
  step. No pushed state exists anywhere.
- **Steer-first-cut (descriptive).** The same on the evidence-cut prompt, forced token from the Steer-0-cut row.
- **Gold-first (descriptive ceiling).** The first token of the gold string (tokenised as the model would emit it after
  `Answer:`, with the leading space) forced on the unpushed cache: does any answer-start token let the model continue an
  answer from the evidence?
- **Unknown-first (sanity).** The token ` unknown` forced on the unpushed cache. Must equal the A row byte-exact on
  28/28, otherwise the forcing path is wrong: stop, exit 4, before any other arm.
- **Step-0 first-token check.** Steer-0 re-run for one new token on all 28 rows; its token must equal the §39 Steer-0
  first token on 28/28, otherwise stop, exit 5.
- Reproduction as in §39 (flag rule ≥ 0.8 against the §38 dev rows; byte identity ≥ 0.9, stop below); the unsteered
  prefills come first so both stops fire before any forced or steered output.

**Readings, fixed now** (exact counts; aF = ans(Steer-first), gF = D1 gold(Steer-first); a0 = 26, g0 = 7 from §39):
- **Token carries it:** 5·aF ≥ 4·a0 (aF ≥ 21) and 5·gF ≥ 4·g0 (gF ≥ 6).
- **State needed:** 2·aF < a0 (aF ≤ 12).
- **Split:** aF ≥ 21 and 2·gF < g0 (gF ≤ 3): the token keeps the model answering, the pushed state keeps it right.
- Otherwise undetermined on 28 rows. Per-row table in every case: forced token, Steer-first text, Steer-0 text, same
  text yes/no, divergence index. Gold-first and Unknown-first beside, descriptive. Soft-refusal count as in the §39 review.
- **Prediction, recorded now:** split or undetermined, aF about 0.6–0.9 of a0 and gF ≤ 4. In §39 the texts left
  Steer-all at token 1–2 on 11 rows, so later tokens do read the pushed state, and a leading token such as "The" or
  "Favorite" need not commit to content.

**Time and rules.** About 28 × 4 arms × ≤ 24 s plus reproduction, about 1.0–1.2 h conservative; `--timing-only` first
and the projection sent before `--run`; `GUARD_VM_GB=22`; rows under `runs/exp39b_dev/qwen3b/`, report
`runs/exp39b_report_dev.md`; the script refuses folds other than 0–1. Build, then the Advisor's review, then run.
Development only; no claim on folds 2–4; no pod.
