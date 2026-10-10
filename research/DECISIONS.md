# Research decisions

Major decisions, in order, with the evidence each one rests on. Append only. Reversing a decision means
adding a new row that cites the old one, never editing it. Only the Advisor writes here, and only after
the user has made the decision.

| ID | Date | Decision | Evidence / reason | Made by |
|---|---|---|---|---|
| D-01 | 2026-09-29 | Clear `main` to a scaffold and rebuild as a controller framework (heuristic / learned / RL / prompted / oracle). Phase 2 kept on branch `phase2-locomo-baseline`, tag `phase2-complete` | Phase 2 design too narrow for the restated thesis goal (`docs/archive/restart.md`) | user |
| D-02 | 2026-10-05 | Replace every dirty run with a clean rerun at `ef57fc4`; archive the dirty ones | 654/654 runs stamped `dirty: true`; E16/E17 fixes (research note §7) | user + agent |
| D-03 | 2026-10-05 | Imitate a least-regret teacher, not the argmin oracle, once an archive exists | 4b: oracle imitation deletes what it should archive; D1 0.34 → 0.85 at 2% | agent, reviewed |
| D-04 | 2026-10-06 | Move from synthetic tasks to a real 7B reader on LoCoMo and LongMemEval (RunPod A40, vLLM) | Exp 8/9: the scripted-reader results needed a real reader | user |
| D-05 | 2026-10-07 | Adopt the reviewed protocol: 5 stratified folds, retrieval floor, reader prompt frozen at `7b5fc30`, pre-registration before learned cells | §13 | user |
| D-06 | 2026-10-07 | Close RQ3 (RL vs imitation) after §16a | §16 gate FAIL +0.0125 (−0.005, +0.033); labels were already a sufficient target | user |
| D-07 | 2026-10-07 | Stop adaptive k at its time box; carry `fixed8` to the reader | §19/§19b gate 4 fails; oracle-k ceiling is the finding | agent, per pre-registration |
| D-08 | 2026-10-07 | Standing rule: full-size stub run under the memory guard before any paid launch | §16 OOM incidents; later ≥ 15 GB free disk (§24) | user |
| D-09 | 2026-10-08 | Design choices come from training-side data only; a labelled leak is still a leak | §22c launch stopped at the user's objection; update count re-chosen on an inner 6/2 split | user |
| D-10 | 2026-10-08 | Re-close RQ3 "on this task" | §22c clause −0.001 (−0.012, +0.011) FAIL | user |
| D-11 | 2026-10-09 | Next benchmarks: StreamMemBench (§25), then OpenJev (§26). ALFWorld rejected; LoRA deferred | user order | user |
| D-12 | 2026-10-09 | Close §25 without a paid run | free label-only check: `fixed8` evidence in view far below FIFO (−0.61 to −0.90) | user, on reviewer advice |
| D-13 | 2026-10-09 | Spend rule: $20 programme cap, a UTC launch window, per-run advance approval; never leave a pod billing while waiting on a question | §24 idle 43 min ≈ $0.42 | user |
| D-14 | 2026-10-09 | LRE (2606.20954) must be a baseline | literature pass: closest published twin of the head | user |
| D-15 | 2026-10-10 | Close §30 (UtilMem) and §31 (qTTT) without pod runs | expected undetermined at about $1.75 with an untested judge and unlicensed data (§30); third-party re-creation shows little gain at small scale (§31) | user |
| D-16 | 2026-10-10 | Close §29 (reader-utility signal) as negative | C1 −0.013 (−0.051, +0.026), C2 −0.002; utility head picks longer turns | per pre-registration |
| D-17 | 2026-10-10 | Report the head as non-inferior to a 31-param LR and a zero-shot cross-encoder, not better | §32 D1 +0.013 (−0.017, +0.043) | per pre-registration |
| D-18 | 2026-10-10 | Reset: clean the repo, preserve every result, and consider a pivot toward sequential memory management and delayed credit assignment. No new architecture yet | this reset; F5, F10 in STATE.md | user |
| D-19 | 2026-10-10 | Delete the aborted / quarantined runs (`runs/_aborted*`, `runs/exp22c_peek_quarantine`); trim `runs/_archive_dirty_2026-09-30` to reports, summaries and configs; gzip per-step traces | none were ever reported; deletion log in `runs/_deleted_2026-10-10.txt` | user |
