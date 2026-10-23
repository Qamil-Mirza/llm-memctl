# §40 proposal (rev 0, Advisor, 2026-10-23; D-33 item 2): independent validation of the fixed layer-29 intervention on a genuinely new set

**Status: draft for the Executor's feasibility notes. Nothing runs; the pod is quoted to the user after rev 1.**

## 1. Question

§38 showed its Claim 1 at a setting (layer 29, α 2) chosen on development labels, confirmed once on folds 2–4 of the
same prompt family. Does the **same, fixed** intervention lift the 3B's refusals into correct answers on data it has
never touched, with no layer, strength, direction or wording chosen from that data?

## 2. The fixed intervention (nothing is re-chosen)

- Direction: v_29 from `runs/exp38_dev/qwen3b/directions.npz` (sha a7f967ec…), byte-exact; cast to the pod dtype is
  the only change, cosine with the cast printed. Not re-estimated on the new set.
- Add 2·v_29 (unit ‖v_29‖, norm 64.47) at the `:` of `Answer:` and at every generated token (Steer-all, the setting
  §38 confirmed). Steer-0 beside, descriptive (§39 is development-only).
- The script accepts exactly one (layer, α) and refuses any other; no sweep exists in the run.
- Classifier v2; max 16 tokens; greedy; the §38 pod judge (Qwen2.5-7B-Instruct HF bf16, per-type prompt) with the
  §38 agreement check on a cached subset.

## 3. The new set: two candidates, the Executor's feasibility note decides which is primary

- **(a) LoCoMo** (`data/locomo/locomo10.json`, 10 conversations, about 1,986 questions with gold evidence turns).
  No LLM has run on it in this programme; the direction has never seen it. Prompt: the §35-family A prompt (gold
  evidence in view, the 3B's own note frozen, then `Answer:`) built for LoCoMo by `exp35_generate.py`'s note step;
  adversarial (unanswerable) questions excluded from R and kept as a separate descriptive block. Judge prompt: the
  LongMemEval per-type prompt mapped to LoCoMo's categories (map fixed in rev 1).
- **(b) LongMemEval folds 2–4 under the §23 in-view prompt family** (`fixed8` evidence, no note): a new prompt
  distribution on rows the direction never saw under that prompt; §34's caches may cover the A arm.
- Either way: no row of the new set is read before the pre-registration is frozen; R = the 3B's marked refusals under
  the A prompt; up to 120 R rows (first in file order after a seeded shuffle, seed 40); C = correct-answer rows, up to 40.

## 4. Arms and claims (fixed now; thresholds as §38)

Arms on R: A, B (the §35 "do not reply unknown" line, prompt-only), Steer-all, Steer-all-cut (evidence ablated, N-cut
note), Random (seeded rng [40, 29], norm-matched, one vector), Steer-0 (descriptive). On C: Steer-all (damage).
- **Claim 1:** P(gold(Steer)) − P(gold(Steer-cut)), paired bootstrap 10k over rows; SHOWN if LB > 0; meaningful if
  P(gold(Steer)) ≥ 0.30.
- **Control X:** ans(Random) ≤ 0.2·ans(Steer), else NOT INTERPRETABLE.
- **Control C:** P(gold(Steer)) − P(gold(A)) on C rows, UB printed; harmful if UB < −0.10.
- **Claim 2 (descriptive):** P(gold(Steer)) against P(gold(B)) on R, paired; by question type.
- **Prediction:** Claim 1 SHOWN with P(gold(Steer)) between 0.35 and 0.50 on (a); Control X passes (random ≤ 0.1).
- Signals at `:` (§41 §2) are logged on the A and A-cut prefills at no extra cost; §41's frozen rule is scored on the
  same rows. The §41 rule is frozen (committed) before this pod starts, or §41 is not scored here.

## 5. Cost, stop, rules

One A40 secure pod at the then-current price (last $0.59/h), hard stop 60 min (≈ $0.59), D-22 five-row smoke, D-30
bundle test in a fresh venv, D-08 resume. Programme spend ≈ $17.68 → ≈ $18.27 of $20. The new set is read once, on the
pod, with `--confirm`. No layer or strength selection of any kind on the new set; the record sentence is "the §38
setting, fixed in advance, on a set it had never seen".

## 6. Open for the feasibility note (Executor; no build yet)

LoCoMo evidence format and prompt length at the 3B's limit; note-generation time per row on the A40; whether §34 caches
cover (b)'s A arm; expected R size under each candidate; wall-clock inside 60 min with the judge; anything that would
force a choice the Advisor has not fixed above.

## Rev 1 (Advisor, 2026-10-23): rulings on the Executor's feasibility note; design frozen for the build

- **Set: LoCoMo (a) is primary; (b) is dropped.** The Executor's F1 stands: under the §23 prompt the 3B regenerates
  the same greedy note on the same folds 2–4 rows, so (b) would re-run §38's confirmation, not test a new distribution.
- **Rows.** Answerable questions (categories 1–4) with 1–8 resolvable evidence turns; a seeded shuffle (seed 40) takes
  600 of them for the note pass. R = marked refusals under A, capped at 120 in shuffle order; C = marked correct
  rows under A (judge), capped at 40. **Adversarial block (new):** 40 category-5 questions (seed 40), arms A, B and
  Steer-all; the readout is the answer rate alone (every answer to an unanswerable question is unsupported; no judge).
  If R < 40 after the 600, the arms still run and the §4 claims are reported descriptive, no verdict.
- **In-view memory (i).** All evidence turns plus BM25 top turns from the same conversation (the question as the
  query) to fill 8 items, in arrival order. Questions with more than 8 evidence turns, none, or an unresolved id are
  excluded before the shuffle; the counts are printed.
- **Line format (ii).** The §23 compact "speaker, date: content" line (`label_prefix`), the session date as the date.
- **Photo captions (iii).** Included as "[shares a photo: <blip_caption>]" inside the turn's content; some golds live
  only there, and the caption is part of the evidence the user saw.
- **Judge map (iv).** 4 → single-session-user, 1 → multi-session, 2 → temporal-reasoning, 3 → the default prompt.
  Agreement check as §38: the judge re-scores 60 cached §35 rows and must agree ≥ 0.90.
- **Notes (v, vi).** A fresh N pass on the 600 rows (the §23 prompt, 256 tokens, greedy), **batched** (batch 32, left
  padding); `split_note` gives the frozen note; rows without a marker dropped, count printed. N-cut notes for the R rows
  generated on the pod on the cut prompt, batched. Batched generation is new code: the D-22 five-row smoke on the pod
  compares batched against single-sequence notes on the same 5 rows and must show the same first token on ≥ 4 of 5 and a
  marker on ≥ 4 of 5, else abort before the 600. The D-30 bundle test in a fresh venv covers the batched path with the
  stub engine.
- **Direction.** The laptop fp32 `directions.npz` v_29 (sha a7f967ec…), bf16 cast, cosine with the fp32 vector printed;
  the §38 pod's own re-estimate is not used. One (layer, α) = (29, 2); the script refuses anything else; no sweep.
- **Arms on R:** A, A-cut, B, Steer-all (primary), Steer-all-cut, Random (seeded rng [40, 29], norm-matched), Steer-0
  (descriptive). On C: Steer-all. Steer-all and Steer-0 share one prefix cache. Signals for §41 (step-0 logits and the
  layer-29 residual at `:`) logged on the A and A-cut prefills.
- **Claims:** as §4, unchanged. Added descriptive readouts: the adversarial answer rates (A, B, Steer-all) and, for
  §41, Cond's answer rate on the adversarial block beside Steer-all's.
- **Time and stop.** Executor's projection 25–35 min with batched notes; hard stop 60 min; the pod is killed at the stop
  whatever the state, and the rows written so far are reported as partial.
- **Order.** §39b finishes first; §41's rule is frozen by commit (named in the launch message) before the pod; the
  bundle passes D-30; then the quote's conditions are re-read and the pod launches on the user's word.
- **Quote to the user:** one A40 secure pod at the then-current price (last $0.59/h), hard stop 60 min, ≈ $0.59;
  programme spend ≈ $17.68 → ≈ $18.27 of $20.

## Rev 2 (Advisor, 2026-10-23, after pod attempt 1): the smoke's marker bar

Attempt 1 (pod b0qnmah0utqvh2, 19:19–19:22 UTC, about $0.03, logs at 78f2c07) stopped at the D-22 smoke before any
note pass or arm: batched and single-sequence notes agreed on the first token 5/5 and the text 4/5, all 4 judge replies
were verdicts, but the notes carried the `Answer:` marker on only 3 of 5 rows (batched and single alike), and the
smoke as built required an absolute 4/5. Rev 1's wording ("same first token ≥ 4/5, marker ≥ 4/5") was meant as a
batched-against-single agreement check and was read as an absolute bar; the Advisor's wording was ambiguous.

**Ruling: option (a).** The smoke checks what it exists for, batching drift and judge compliance: same first token
≥ 4/5 and marker agreement (batched = single) ≥ 4/5; the absolute marker rate is printed, not gated. The marker
rate on LoCoMo is a result, not a fault: on 2 of the 5 smoke rows the 3B answered in one sentence without the
note-and-`Answer:` format. After the N pass the marker rate on the 600 and the 40 is printed; the pre-registered
R < 40 rule (claims descriptive) stands as the only population guard. The record states that R and C are drawn
from marked rows only, as in §35 and §38, and gives the unmarked share as a finding about the prompt family on LoCoMo.
Descriptive, only if it is a few lines: the unmarked rows' direct answers (the N output) judged, so the record can say
how often the 3B answers correctly on LoCoMo when it skips the format.

A relaunch is a new pod and needs the user's new line in the Executor's pane. Terms unchanged: A40 secure at the
then-current price (now $0.59/h), hard stop 60 min, at most about $0.59; programme spend about $17.71 after attempt 1,
at most about $18.30 after the relaunch.
