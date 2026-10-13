# §35 pre-registration draft: does the 3B derive the answer in its own note and still refuse? Separating derivation from answer selection

Date: 2026-10-13, Advisor draft, before any pod run and before any confirmation-fold number. Branch
`jacobian-lens` (the data loader `memctl/jacobian_data.py`, the evidence ablation `memctl.analysis.jacobian.ablate_evidence`
and the §34 pairing are reused). Behavioural follow-up to §34a–c. Prompted by an outside design note (2026-10-13) that
asks the §34 question at the level of the model's text instead of its activations: can the 3B derive the answer and still
not give it?

## 0. What the record already says, and what this adds

- The frozen §23 prompt (`memctl/agents/llm.py`, `REASONING_INSTRUCTIONS`) already makes every reader write a note
  ("name the facts ... then do any counting, adding or date arithmetic") and then one line `Answer: <phrase>`. It also
  already says "Reply unknown only if the memory has nothing at all on the topic". So the design note's "evidence
  commitment" condition **is the normal condition**, and its "normal, no intermediate step" condition is the §11e/§13
  short-phrase arm that was dropped because a 7B fell back to "unknown" without the note. Neither needs a run.
- The 3B still refuses 0.32 of in-view answerable questions under that prompt (§34a, 65 of 204 on folds 2–4).
- **Free check, development folds 0–1 only** (`scripts/exp35_note_check.py`, criterion D1 below): among the 3B's 30
  marked refusals, the gold is in the 3B's own note on **7 (0.23)**; among its 31 marked wrong answers, 4 (0.13); among
  its 39 marked successes, 26 (0.67). The 7B: refusals 3 of 9, wrong 8 of 38, successes 50 of 79. So "derived and still
  refused" exists, and it is a **minority of refusals, about a quarter**; three quarters of the 3B's refusals have a note
  that does not contain the answer. The design note's illustrative "75% correct derivations" is not what this reader does.
- §34c found that the 3B's ` unknown` disposition at the forced answer position barely moves when the evidence is cut
  (top-1 at some layer on 0.58 of questions with the evidence, 0.63 without), while the 7B's moves from 0.08 to 0.53.
  That is a readout. §35 is the behavioural intervention that §34's Stage 2 gate did not open: hold the 3B's own note
  fixed and change only the answering instruction.

## 1. Population and data

- The §23 `fixed8` (fraction 0.05) in-view answerable questions with a judge label, both readers, paired by prompt hash,
  exactly the §34 population: folds 0–1 for development (126), folds 2–4 for confirmation (204), all folds 330.
- **Derivation criterion D1 (string, free):** the normalised gold string, or every gold token, occurs in the note (the
  text before the output's last `Answer:`). Outputs without a marker (3B: 39 of 204 on folds 2–4) are excluded from the
  frozen-note arms and reported. D1 over-counts derivation: a note that copies the evidence line counts even when the
  question asks for a comparison or a count. **D2 (judge):** the §23 judge (`memctl/judge.py`) asked whether the *note
  alone* answers the question. D2 is the deciding criterion where the two disagree; D1 is reported beside it.
- **Control, evidence cut:** the §34c prompt with every evidence line removed by exact span (`ablate_evidence`), same
  question. On it the right answer is unknown from the reader's view. Also the natural control: answerable questions with
  zero evidence lines in view (15 of 188 labelled on the development folds; the 3B refuses 10 of them).

## 2. Arms (3B primary; 7B run beside, descriptive)

All generation at temperature 0 with the §23 vLLM settings. The reader noise floor applies (±0.007 accuracy on repeated
passes; §33 reproduced 16 of 20 final answers from the same prompt).

| arm | prompt | what it isolates |
|---|---|---|
| **A, reproduce** | §23 prompt + the cached note (teacher-forced, up to and including `Answer:`) → generate the phrase | reproduction check for the frozen-note arms; expected to give the cached answer on most questions |
| **B, frozen note, abstention prohibited** | §23 prompt + cached note + one line "The note above names the facts. Give the single best-supported short answer. Do not reply unknown." + `Answer:` | answer selection with the derivation held fixed (the design note's Experiment 3, condition B) |
| **F, forced choice, full regeneration** | §23 prompt with the instruction's unknown clause replaced by "Never reply unknown; give the single best-supported short answer" → note and answer regenerated | the deployable version: threshold shift with a new derivation (the design note's Experiment 2) |
| **F-cut** | arm F on the evidence-cut prompt | what forced choice invents when nothing is in view |
| **N-cut** | the §23 prompt on the evidence-cut prompt (normal instruction) | the refusal rate when unknown is right |

Not run: a multiple-choice condition (its candidate list would carry the gold into the prompt). The frozen-note arm
cannot be run on the evidence-cut prompt (the cached note was written with the evidence, §34c caveat).

Judging: the §23 judge on every generated answer (judged correct / refusal by the `REFUSALS` list / wrong).

## 3. Claims (3B, folds 2–4 after folds 0–1; paired bootstrap, 10,000 resamples, 95% intervals)

- **Claim 1 (a selection gap exists).** Among in-view questions where the cached 3B output was a refusal and its note
  meets D2, arm B is judged correct on share s. SHOWN if the lower bound of s is above 0.5 (a majority of
  derived-then-refused questions become correct when only the answering instruction changes). Descriptive beside it: the
  same with D1; arm A's reproduction rate.
- **Claim 2 (it is selection, not guessing).** Among the cached refusals, P(B correct | D2 true) − P(B correct | D2 false)
  > 0, SHOWN if the lower bound is above 0. If forced answers are right only when the note already holds the answer, B
  recovers derived answers rather than lucky guesses.
- **Claim 3 (the thesis stake).** In-view accuracy of the 3B under arm B minus the cached accuracy (paired, all marked
  questions). Reported with its interval and as the share of the 3B−7B in-view gap (0.397 vs 0.667, §23) that it closes.
  No verdict threshold: this number is what the experiment is for. The same for arm F.
- **Cost (pre-registered, must be reported with Claims 1–3).** (i) Among cached refusals with D2 false, the share arm B
  turns into judged-wrong answers. (ii) On the evidence-cut prompts, the share of questions where arm F produces the gold
  (the plausibility floor; §34c's final-layer plausibility share was 0.43 for top-10 presence, so a top-1 string match
  should be well below that) and the share where N-cut refuses (calibration when unknown is right). A selection fix is
  worth reporting only with (i) and (ii) beside it.

**Prediction, recorded now.** Derived-then-refused questions are about a quarter of the 3B's refusals (0.23 by D1 on the
development folds), so about 0.07 of in-view questions. Claim 1 SHOWN (most of those answer correctly once unknown is
prohibited). Claim 2 SHOWN. Claim 3: arm B lifts in-view accuracy by roughly +0.05 to +0.08 absolute, closing about a
quarter of the 3B−7B gap at most; arm F lifts more but with a larger (i) and a non-trivial (ii). The reading if this holds:
answer selection is a real, separable, but **minor** part of the 3B's shortfall; the larger part is that the note does not
contain the answer (derivation, or reading), which is §34a's A-share and the "present without evidence" discount of §34c.

**Reading, fixed before the data.** If Claim 1 is NOT SHOWN (derived-then-refused questions mostly stay wrong or refuse
under B), the 3B has a note-to-answer consistency problem that an instruction does not fix, and the design note's last
row applies. If Claim 2 is NOT SHOWN, B is a threshold shift and nothing is said about selection. If Claim 3's lift is
within the noise floor (under 0.01), the selection route to §23's claim 1 is closed.

## 4. Caveats fixed now

The note is text the model wrote, not its computation (chain-of-thought faithfulness); arms B and F do not need
faithfulness, they are interventions on the text. Labels are the §23 single judge (understates accuracy by about 0.1;
paired comparisons unaffected); D2 adds a second judge call per note. Teacher-forcing the cached note assumes the
model would write it again (§33: 16 of 20 final answers reproduced). The 39 marker-less 3B outputs are outside the
frozen-note arms. No held-out number is read before the Reviewer accepts this document; the development-fold D1 counts
above are the only numbers that exist.

## 5. Commands and spend (NEEDS SPEND; approval pending)

Executor builds `scripts/exp35_pod.sh` (3B then 7B via vLLM, arms A/B/F/F-cut/N-cut on folds 0–4, then the judge on
every answer) and `scripts/exp35_report.py` (`--folds 0,1` first, then `--folds 2,3,4 --confirm`, then all folds).
About 330 in-view + 330 cut prompts × up to 5 short arms per reader ≈ 3,000 generations plus about 3,000 judge calls;
3B and 7B load in turn on one A40. Full-size stub run, peak RSS and measured per-call latency before launch (standing
rule). One A40 secure pod at $0.59/h, about 40 min expected; **hard stop 90 min ($0.89)**. Programme spend so far
≈ $17.00 of the $20 cap.
