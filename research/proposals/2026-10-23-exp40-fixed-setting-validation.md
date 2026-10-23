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
