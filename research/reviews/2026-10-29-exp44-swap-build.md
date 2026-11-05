# Build review: §44 rev 1 conditions S and U, commit dbd813e (gate recount 88b2a19)

**Verdict: ACCEPT with one change (the shared-prefix rule of rev 3), then the smoke and the runs may go in the order
rev 3 fixes. Confirmation / pooled rows stay locked until rev 4 records the recount.**

Checked: `memctl/analysis/swap.py` (readout of Δ = logit(G) − logit(S) with both refusal ids; `follows` and
`failure_class` at the 1-nat margin; `transfer` with the forward/reverse convention and MIN_GAP guard; `transfer_depth`
through `patching.causal_depth`; `transfer_sanity`; `shift_toward_s`; seeded `random_overwrite`; `choose_donor`
cyclic nearest-earlier with both first ids distinct; `self_layers` 0 / L//2 / L−1; `build_swap_inputs` with the
alignment check), `scripts/exp44_swap.py` (`POOL_AUTHORISED` lock, exit 2 on folds 2–4 or `--pooled` without it;
eligible rows loaded from the gate file by sha; resumable stamped rows; U rows from the §35 arm-B wording; input check:
all 82 / 85 rows build and align, tokens 1206 / 1206 / 1023), `scripts/exp44_swap_report.py` (S0–S3, U1; Llama S1/S2
not read). Tests: `tests/test_exp44_swap.py`, `tests/test_exp44_swap_gate.py`, `tests/test_patching.py`: 42 passed
here (21 s). Smoke deferred until the laptop is free: correct.

Change required before any run: implement the rev 3 shared-prefix rule (gate exclusion on the first token *after*
the longest common token prefix; the prefix teacher-forced in every run of the row; Δ / F / T / ranks read at the
first divergent position; alignment extended), recompute the gate, send the counts. Rulings 1–8 as in rev 3.

Advisor, 2026-10-29.
