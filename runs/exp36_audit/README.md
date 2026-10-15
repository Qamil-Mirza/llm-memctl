# §36 hand audit of the §23 judge (folds 0-1 only), 2026-10-15

Sample: `scripts/exp36_audit_sample.py` (seed 36): 20 3B arm-B candidates judged correct, 20 judged wrong, 20 F-cut
candidates; judge verdict hidden in `sample.jsonl`, key in `key.jsonl`, hand labels in `hand.json` (done before any
verifier score exists).

- Agreement 58 of 60 (0.967). Disagreements: #26 "Memrise recommendation" for gold "Memrise" (judge wrong, hand
  correct); #32 "5 goals and 2 assists" for gold total 5 (judge correct, hand wrong: the implied total is 7).
  One error each way, so no direction of bias in this sample.
- Cut prompts: the gold string still occurs somewhere in the evidence-cut prompt on 5 of 20 F-cut rows (#7 "Yes",
  #12 "30", #19 "4", #33 "5", #42 "50"; short strings, several plausibly coincidental). The judge marks 4 F-cut
  answers correct (#12, #23, #24, #49). As pre-registered, F-cut rows stay "unsupported" by construction; this is the
  residual the proposal's caveat names.
