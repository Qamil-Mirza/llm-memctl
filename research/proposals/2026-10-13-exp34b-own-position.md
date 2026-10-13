# §34b pre-registration: the J-lens readout at the reader's own answer position

Date: 2026-10-13, written before the pod run. Branch `jacobian-lens`, code 8ed915f (`--position-mode own` in
`scripts/jlens_read.py`, `own_output` in `memctl/analysis/jlens_readout.readout_for`). Follow-up to §34
(`research/proposals/2026-10-13-exp34-jlens.md`, results EXPERIMENTS.md §34a) and to its exploratory analysis 1
(`scripts/exp34_depth.py`, ledger 2026-10-13): at the forced position the 3B's gold emerges late in successes and
failures alike, and what separates failures is whether it emerges at all.

## 1. Question

§34 read every question at a **forced immediate answer**: `Answer:` appended straight after the prompt, no reasoning.
The cached 3B outputs almost always write a note first and `Answer:` later (200 of 204 confirmation-fold outputs). Is the
late and often absent emergence of the gold an artefact of forcing? §34b reads the same questions at the reader's **own
answer position**: the cached output teacher-forced up to and including its last `Answer:` marker, then the gold's
leading-whitespace tokens, P1 as before. Outputs without a marker are read from their first token and flagged
(`answer_marker: false`; on the development folds 26 of 126 for the 3B, 0 for the 7B). P0, E and Q are prompt positions
and are identical in both modes, so every question pairs with its §34 read by `prompt_sha256`.

## 2. Fixed

- Lenses: the §34 lens files, unchanged (7B `e3a07f9e…`, 3B `ddbdb32d…`, 56 prompts each). k = 10, same band, same
  tracked tokens (the gold's first token after whitespace, ` unknown`, `unknown`), same population rule, same labels.
- Verdict rows: as in §34, the 3B's gate failed, so its verdicts are taken on the logit-lens rows with the J-lens rows
  beside; the 7B's J-lens rows stand with logit rows beside.
- Statistics: paired bootstrap over questions, 10,000 resamples, 95% intervals. Development folds 0,1 are looked at
  first; verdicts are on folds 2,3,4 (`--confirm`), reported on all folds too.
- Command: `bash scripts/exp34b_pod.sh` (reads both readers, folds 0–4, own mode, existing lenses), then
  `.venv/bin/python scripts/exp34b_report.py --folds 2,3,4 --confirm --out runs/exp34b_report_confirm.md` and the
  same with `--folds 0,1,2,3,4 --out runs/exp34b_report_allfolds.md`.

## 3. Claims (3B; the 7B rows are descriptive)

- **Claim A (presence).** At the own position the gold emerges (enters the top-10 at some layer) on a larger share of
  questions than at the forced position: paired difference of proportions, own minus forced, SHOWN if the lower bound
  is above 0.
- **Claim B (depth).** Among questions where the gold emerges in both modes, emergence is earlier at the own position:
  paired mean difference of depth, own minus forced, SHOWN if the upper bound is below 0.
- **Descriptive.** The A/B/C decomposition at the own position by class (success / refusal / answered wrong) next to the
  forced one; the per-refusal order of gold-enters-top-10 against ` unknown`-reaches-top-1 at the own position; the
  marker-less questions separately; best-rank and final-rank changes; the same for the 7B.

## 4. Reading, fixed before the data

If both claims are SHOWN, the forced-position lateness is largely an artefact: the reader's own reasoning makes the
answer readable, and §34a's "late emergence" caveat becomes the main qualification of §34a. If neither is SHOWN, the
lateness belongs to the model and §34a stands as written. Mixed outcomes are reported as such. Nothing here is causal.

## 5. Spend

One A40 secure pod at $0.59/h, reads only, about 20 min of work; hard stop 60 min ($0.59). User approval 2026-10-13
(verbatim): "experiement 2 is approved", on the quote "$0.59 per hour, hard stop 60 minutes, at most $0.59".
