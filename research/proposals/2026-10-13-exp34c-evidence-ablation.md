# §34c pre-registration: does the gold's presence at the answer position depend on the evidence?

Date: 2026-10-13, written before the pod run. Branch `jacobian-lens`, code ae62ffc (`--position-mode ablated` in
`scripts/jlens_read.py`; `memctl.analysis.jacobian.ablate_evidence`, exact-span removal, from §33). Follow-up to §34a and
§34b. Raised by the reviewer questions of 2026-10-13 (2 and 4): a gold token can sit in the top-10 at the answer
position because it is *plausible* for the question ("2" for a counting question) or already known, not because the
reader extracted it from the evidence in view.

## 1. Design

Every §23 in-view question is read again at the **forced position** (prompt + `Answer:`, as §34a) on a prompt in which
every evidence line has been cut out by its exact span (`ablate_evidence`; the other memory lines, the instruction and
the question are unchanged). The forced position is used, not the own position of §34b, because the cached note was
written with the evidence and would leak it. The same lenses, k, tracked tokens, labels and population as §34; reads
pair with the §34a intact reads by question id (the prompt hash differs by construction; the intact hash is stored in
each ablated read as `intact_prompt_sha256` and must match the §34a read). Outcome and presence definitions as §34.

## 2. Claims (3B, deciding logit row as in §34; 7B descriptive; folds 2,3,4 after folds 0,1)

- **Claim A (presence depends on evidence).** The share of questions where the gold ever enters the top-10 is lower
  without the evidence: paired difference intact − ablated, SHOWN if the lower bound is above 0.
- **Claim B (final-layer presence depends on evidence).** The share of questions where the gold is in the final-layer
  top-10 (outcome C; model logits, lens-free) is lower without the evidence: paired difference intact − ablated, SHOWN
  if the lower bound is above 0.
- **The plausibility share (descriptive, the number this experiment exists for).** Among questions that are C with the
  evidence, the share that remain C without it, by class (success / refusal / answered wrong) and by gold-token kind
  (strict vs function word, `scripts/exp34_gold_kinds.py`). The same for "ever emerges". A question that stays C without
  evidence is one where "present at the end" says nothing about extraction.
- **Descriptive.** Paired log10 best rank intact − ablated ("evidence lift") per reader and class; the ` unknown`
  trajectory without evidence (does ` unknown` reach top-1 more often when the evidence is absent, as it should for a
  calibrated reader); the 7B rows.

## 3. Reading, fixed before the data

If Claims A and B are SHOWN and the plausibility share among the 3B's C-failures is small (say under a third), §34a's
"present at the end" is evidence-driven and stands. If the plausibility share is large, §34a's C share must be
discounted by it and the write-up says that outcome C overstates extraction. If Claim B is NOT SHOWN, the forced-position
readout does not measure evidence use at all, and §34a's decomposition is withdrawn as an extraction measure. Nothing
here is causal about the refusal; it is about what the readout measures.

## 4. Commands and spend

`bash scripts/exp34c_pod.sh` (reads only, existing lenses, both readers, folds 0–4, ablated mode), then
`.venv/bin/python scripts/exp34c_report.py --folds 0,1 --out runs/exp34c_report_dev.md`, then `--folds 2,3,4 --confirm
--out runs/exp34c_report_confirm.md` and `--folds 0,1,2,3,4 --confirm --out runs/exp34c_report_allfolds.md`.
One A40 secure pod at $0.59/h, about 20 min; hard stop 60 min ($0.59). Approval: pending.
