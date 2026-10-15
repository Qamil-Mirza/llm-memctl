# §36 amendment 2 (§36b): a compliant verifier wording, chosen on development folds by a label-free gate

Date: 2026-10-15, Advisor draft, after run 1 (review: `research/reviews/2026-10-15-exp36-run1.md`). Branch
`jacobian-lens`. Everything in `2026-10-15-exp36-verification.md` and amendment 1 stands except the verifier message
(§2), which run 1 showed both readers do not follow (label mass ≥ 0.5 on 16/84 dev rows for the 3B's V). **NEEDS the
user's spend approval for the rerun; the probe is free.**

## 0. Status of the data
- Run 1 rows (`runs/exp36/`) keep their value as the record of the failure. Their confirmation folds 2–4 are **sealed**:
  no report, no label-linked number, for any readout item. C4, C4′ and the natural flag are unaffected by the scorer bug
  but are also held until §36b reports every row once (one confirmation read).
- The §36b design is fixed by this document and the probe's rule-based pick before any confirmation-fold number exists.

## 1. The probe (free; development folds 0–1; 3B only; laptop CPU)
Candidates, tested in this order on the 84 dev candidate rows; the first to pass the gate is adopted and later ones are
not run. Everything after the memory block stays as in §2 of the pre-registration unless named here.

| id | change from run 1's V |
|---|---|
| (a) | P2 frame ("Answer with a single letter"), the trailing `Answer: ` literal removed; the assistant turn starts empty |
| (b) | (a) plus the §23 goal line replaced by one sentence: "You are checking whether the memory below supports a proposed answer." The memory block stays byte-exact |
| (c) | (b) plus the system message "Reply with exactly one letter: A, B or C." |
| (d) | (c) with the completion prefix `Label: ` |
| (f)–(h) | the Executor's W1–W3, run after (d) and before (e); their exact texts must be in a commit before the probe reaches them, and they are listed in EXPERIMENTS.md §36b whether or not they are reached |
| (e) | fallback (last): (c) generated for up to 8 tokens, greedy; the first A/B/C letter is the label and the readout's "mass" is the share of rows that contain one. Scored by argmax only (s undefined); Claims 1, 3 and 4 then cannot be tested and only Claim 2 (C2-style, at the natural coverage via the letter alone) and the controls' argmax forms are reported |

**Gate (label-free, fixed now):** decoded label mass over the two pre-registered spellings per label (`A` and ` A`, i.e. bare and `Ġ`-prefixed after byte decoding; the tab-prefixed spelling is stored beside as `label_mass_tab` and never used for the gate) ≥ 0.5 on at least 0.9 of the 84 rows. No
correctness-linked quantity is computed on the probe. If (a)–(e) all fail, §36 closes: "the 3B cannot be made to grade
support in a single-token readout under this prompt family", and the remaining question (can a larger checker grade the
3B's candidates?) is a separate proposal.

Paraphrases P2′ and P3′ for §36b are the adopted frame with the run-1 P2 and P3 label texts (the instruction sentence
is the adopted one). V+n, Cut-B, Cut-F, Abs, CF and LC use the adopted frame unchanged.

## 2. Rerun (NEEDS SPEND)
- Same pod script with two changes: (i) a real-model smoke before the main chain, 5 dev rows per reader through the
  adopted wording, abort the chain unless ≥ 4 of 5 rows have mass ≥ 0.5 (a failure costs about $0.05); (ii) export
  `RUNPOD_POD_ID` into the ssh shell so the in-pod self-stop can fire.
- The 7B cannot be probed on the laptop (bf16 weights ≈ 15 GB of 19 GB free); its compliance is tested by the smoke only.
  If the 7B fails its smoke while the 3B passes, the chain runs the 3B only and the 7B rows are reported as "not
  compliant" (descriptive).
- One A40 secure pod at the then-current price (run 1: $0.59/h), run 1 took 17 min; **hard stop 60 min (≈ $0.59)**.
  Programme spend so far ≈ $17.34 of the $20 cap (17.17 + run 1's 0.17).
- Claims, controls, statistics, fixed-sequence testing, τ from folds 0–1, the reliability rule, the judge audit, and the
  prediction are unchanged from the pre-registration and amendment 1. Run 1 and run 2 are both reported in §36.

## 3. Caveat added
A wording chosen for compliance is a second elicitation. "The 3B can judge support" would mean "under a prompt tuned
for one-letter compliance on development folds". A positive Claim 1 still does not show the original reader "knew".

## 4. For the user (DECISIONS rows, written only after the user decides)
- D-22 (proposed): standing pre-launch rule amended: any new prompt or readout gets a 5-row real-model smoke that must
  pass the readout's own reliability gate before a paid run; a stub run does not test compliance.
- D-23 (proposed): approve or decline the §36b rerun at the quoted price and hard stop.
