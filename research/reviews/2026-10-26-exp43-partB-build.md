# Build review: §43 Part B lens pod, commit 7444d11

**Verdict: ACCEPT. The Llama Part B quote may go to the user.** No code change required.

Checked: the diff of `memctl/analysis/families.py` (partner configs from cached tokenizer/config files; Llama 3.1's
date pinned to the template default; Phi-4's own tokenizer and no BOS; pairing by question, stated), `scripts/
exp43_lens_pod_run.py` (population 61/65/66/68/70 asserted with an outcome-free fingerprint; `read_row` is the §34
forced read with `check_bos` and the refusal-id assert; confirmation-fold outcomes never reach a log line; the smoke
compares the final-layer top-1 at P1, which is slot 0 of the array layout), `scripts/exp43_lens_pod.sh` (chain and
deadlines: HARD_STOP_S 9900, SELF_STOP_S 10500, skip-and-log, reads stop between rows, fit boxed after the J-lens
reads and eval are reserved, watchdog, pod.done on any exit), `scripts/exp43_lens_report.py` (exit 2 on folds 2–4
without `--confirm`; B1 = paired bootstrap of log10 best rank and of outcome A, LB > 0; B2 = `exp34_report.claim2_for`,
wrong (answered wrong or refused) vs right, AUC LB > 0.5; J-lens gate through `exp34_report.gate_for` on folds 0–1;
A/B/C, emergence, article and digit caveats descriptive). Tests: `tests/test_exp43_lens_pod.py` and
`tests/test_exp43_families.py`, 39 passed (2 min 7 s). D-30 for Llama: PASS as reported (fresh venv, stub chain,
STUB_BREAK=bos exits 5, HARD_STOP_S=45 exits 5).

## Rulings on the builder's points

1. Population = stage 0's pop34 plus a judge label on both sizes, outcome-free fingerprint: **as the proposal**.
2. **Confirmed.** The logit row decides B1 and B2. The J-lens B2 row is shown beside it; it is labelled "verdict role
   (gate PASS)" only when the §34 gate (a) and (b) pass on folds 0–1, else "shown beside (gate not passed)". A J-lens
   verdict never overrides the logit verdict; if the two disagree, the report says so and the logit row stands.
3. **Accepted.** The reads are single-sequence forward passes, so §40's batched-vs-single check has no object here.
   The D-22 analogue is "the pod reproduces the laptop": on 5 development rows the small model's final-layer top-1 at
   P1 must equal the laptop fp32 read on ≥ 4 of 5 (bf16 vs fp32 may flip a near tie). The large model has no laptop
   reference, so its smoke is the structural checks only; that is the best available. Any mismatching row is to be
   printed with both ids in `_smoke.json` (it is; no change).
4. Dev-only report prints B1/B2 as descriptive; D-30 never runs `--confirm`: **as the proposal**.
5. Bundle = memctl editable install plus imported scripts: **fine**.

## Notes for the launch and the next two quotes

- The pod's `runpodctl stop pod` is a stop, not a delete; the laptop DELETE at 3 h remains the billing backstop, as
  for the §38–§42 pods.
- Phi-4 needs ≥ 40 GB RAM and ≥ 60 GB disk on the pod; the Executor checks at create. Not Llama's concern.
- D-30 is per bundle: Gemma and Phi each need their own D-30 run before their quote.
- `stage_times.json` from the Llama pod sets the Gemma and Phi estimates.
- The held-out seal holds: the pod reads folds 2–4 but logs no outcome; the report opens them only with `--confirm`,
  which is run once per family, after this review and the gate on folds 0–1.

Advisor, 2026-10-26.
