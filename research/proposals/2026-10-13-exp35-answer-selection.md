# §35 pre-registration: does the 3B derive the answer in its own note and still refuse? Separating derivation from answer selection

Date: 2026-10-13, Advisor draft, before any pod run and before any confirmation-fold number. Revision 2 after the
Reviewer's REVISE (`research/reviews/2026-10-13-exp35-prereg.md`); every change that review required is in this text.
Branch `jacobian-lens` (reuses `memctl/jacobian_data.py`, `memctl.analysis.jacobian.ablate_evidence` and the §34 pairing).
Behavioural follow-up to §34a–c. Prompted by an outside design note (2026-10-13) that asks the §34 question at the level
of the model's text instead of its activations: can the 3B derive the answer and still not give it?

## 0. What the record already says, and what this adds

- The frozen §23 prompt is one user message: the environment's goal line ("Below is your memory ... Answer with a short
  phrase that states the fact asked for. If the memory does not contain the answer, reply: unknown"), then
  `REASONING_INSTRUCTIONS` from `memctl/agents/llm.py` ("First write a brief note of at most 80 words ... then do any
  counting, adding or date arithmetic ... Reply unknown only if the memory has nothing at all on the topic. End with one
  line of the form 'Answer: <short phrase>'"). Every reader therefore writes a note before its answer, and the prompt
  already discourages unknown. The design note's "evidence commitment" condition is this normal condition. Its "no
  intermediate step" condition resembles the §11e short-phrase arm (a different prompt, 32 output tokens, an earlier
  judge, unpaired), which fell back to unknown and was replaced by the note; it is not re-run here.
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

- The §23 `fixed8` (fraction 0.05) in-view answerable questions with a judge label, both readers, paired by question id
  and prompt hash, exactly the §34 population: folds 0–1 for development (126), folds 2–4 for confirmation (204), all
  folds 330. LongMemEval's abstention questions are outside this population (the §23 sample scored them apart: under
  `fixed8` the cached 3B gave 7 false answers on the 30, so 23 correct abstentions, §23a cost column); arm F below
  answers them wrong by construction, which is noted, not measured.
- **Marked rows:** outputs whose text contains `Answer:`. Only marked rows enter the frozen-note arms (3B: 165 of 204
  on folds 2–4; 7B: all). The note is everything before the output's last `Answer:`, kept byte-exact.
- **Derivation criterion D1 (deciding; string, free):** the normalised gold string, or every gold token, occurs in the
  note (`memctl.analysis.answer_selection`, moved from `scripts/exp35_note_check.py`). D1 over-counts derivation: a note
  that copies the evidence line counts even when the question asks for a comparison or a count; this bias runs *against*
  Claims 1–2 (it adds questions whose note does not derive the answer to the "derived" set). **D2 (reported beside, not
  deciding):** the §23 judge asked whether the note alone answers the question. D2 is not deciding because the
  LongMemEval judge prompt says yes when a response "contains all the intermediate steps", so a note with the facts and
  no derived answer passes, and because the same judge scoring the note and the arm-B answer against the same gold would
  make D2 and "B correct" co-vary through judge false negatives.
- **Cached label rule (as §23):** a cached answer is a refusal when the text after the last `Answer:` starts with
  "unknown" (`runs/_pipelines/exp23_judge.py`); the same rule is applied to every generated answer here. A generated
  output with no `Answer:` marker (possible in arms N, F, F-cut, N-cut) is graded as the whole output, and is a refusal
  only if that whole text starts with "unknown", as `LLMAgent.act` does.
- **Control, evidence cut:** the §34c prompt with every evidence line removed by exact span (`ablate_evidence`), same
  question. On it the right answer is unknown from the reader's view. Natural control, descriptive: answerable questions
  with zero evidence lines in view (15 of 188 labelled on the development folds; the cached 3B refuses 10 of them).

## 2. Arms (3B primary; 7B run beside, descriptive)

Greedy decoding, temperature 0. **Engine (filled in before launch, Executor commit 2c04307 / 477a43a):** vLLM 0.8.5
offline (`vllm.LLM`), bf16, max_model_len 8192, prefix caching on, transformers 4.51.3 (vLLM 0.8.5's own pin); 0.8.5 is
the vLLM version that served every §23 answer and verdict. Sampling: greedy, temperature 0, top_p 1, seed 0;
max_new_tokens 256 for N, F, F-cut and N-cut (as §23), 64 for A and B, 4 for the judge. Fallback if `import vllm` fails
on the pod: transformers 5.17.0 batched greedy, for every arm of both readers and the judge. **One engine for every arm
of both readers and for the judge within the run**, recorded in every row (`vllm-0.8.5` or `hf-transformers-5.17.0`);
the generator refuses to resume rows from another engine, and a report over rows carrying two engines is "not
interpretable". The judge does not reuse §23 verdicts (one judge engine per run). The §23 cached answers came from a
vLLM chat-completions server at temperature 0; §33 found transformers greedy reproduces 10 of 20 cached outputs exactly
and 16 of 20 final answers. Therefore **every verdict compares arms generated on the same engine in the same run**: B
against A, F against N. The cached answers are used for the population, the labels, the notes and the conditioning sets,
never as a comparison arm.

**Teacher forcing (arms A, B).** Completion-style continuation on raw text: the chat template applied to the §23 user
message with the generation prompt (as `memctl/analysis/jacobian.py` does for §34), then the note byte-exact, then the
arm's suffix. Not a chat message. Per question the file stores the prompt sha256, the number of teacher-forced tokens and
the sha256 of the raw input text; the prefix (chat text + note) must tokenise to the same count as the §34b own-position
read of the same question (`runs/exp34_jlens_own/<reader>/`, `n_tokens` before the gold's leading whitespace), checked in
the report; mismatches are listed and excluded.

| arm | input | what it isolates |
|---|---|---|
| **N, normal** | the intact §23 prompt, regenerated | the within-engine baseline for F; reproduction check against the cached answer |
| **A, reproduce** | chat text + note + `Answer:` | reproduction check for the frozen-note arms; the within-engine baseline for B |
| **B, frozen note, abstention prohibited** | chat text + note + newline + `Based on the note above, give the answer as a short phrase. Do not reply unknown.` + newline + `Answer:` | answer selection with the derivation held fixed (the design note's Experiment 3, condition B) |
| **F, forced choice, full regeneration** | the §23 prompt with both unknown clauses replaced (below), note and answer regenerated | the deployable version: threshold shift with a new derivation (the design note's Experiment 2) |
| **F-cut** | arm F's prompt on the evidence-cut prompt | what forced choice invents when nothing is in view |
| **N-cut** | the intact prompt on the evidence-cut prompt | the refusal rate when unknown is right |

**Arm F's two replacements, verbatim; each original occurs exactly once in every cached prompt (asserted per prompt), and
the arm-F prompt sha256 is stored per row.**

| original (goal line) | replacement |
|---|---|
| `If the memory does not contain the answer, reply: unknown` | `Always answer with your best guess, even if the memory does not contain the answer; never reply unknown` |

| original (reasoning line) | replacement |
|---|---|
| `Reply unknown only if the memory has nothing at all on the topic.` | `Never reply unknown; if the memory has little on the topic, give your best guess anyway.` |

The intervention in B and F is "unknown is not allowed, answer anyway"; neither line introduces a confidence frame
beyond "best guess". Not run: a multiple-choice condition (its candidate list would carry the gold into the prompt). The
frozen-note arm cannot be run on the evidence-cut prompt (the cached note was written with the evidence, §34c caveat).

**Judging.** The §23 judge (Qwen2.5-7B-Instruct, `style: longmemeval`, per-type prompts) on every generated final
answer, and on every cached note (D2). Judge calls are text-keyed and cached under `cache/judge_exp35/`.

## 3. Claims (3B, folds 2–4 after folds 0–1; paired bootstrap, 10,000 resamples, 95% intervals)

Conditioning sets on marked rows: **S** = cached refusal and D1 true; **S'** = cached refusal and D1 false. On the
development folds S has 7 questions; on folds 2–4 it is expected to hold about 14 (62 marked refusals × 0.23).

- **Gate G (pre-stated, descriptive bound).** Arm A gives the same correctness verdict and the same refusal flag as the
  cached answer on at least 0.8 of marked rows (§33: 16 of 20 final answers reproduced). Below 0.8 the frozen-note arms
  are reported as **not interpretable** (teacher forcing or engine mismatch) and Claims 1–2 get no verdict.
- **Claim 1 (a selection gap exists).** On S, paired P(B correct) − P(A correct), SHOWN if the lower bound is above 0.
  The raw share P(B correct | S) = s is reported beside it, with its interval; "s above 0.5" is descriptive, not a test.
  **UNDERPOWERED** is the verdict, not NOT SHOWN, if |S| < 15 on folds 2–4; the all-folds pooled estimate is then given,
  labelled as mixing development and confirmation questions.
- **Claim 2 (it is selection, not guessing).** P(B correct | S) − P(B correct | S'), SHOWN if the lower bound is above 0.
  If forced answers are right only when the note already holds the answer, B recovers derived answers rather than lucky
  guesses.
- **Claim 3 (the thesis stake, descriptive, no threshold).** On the marked rows of folds 2–4: paired accuracy acc(B) −
  acc(A), and acc(F) − acc(N), each with its interval. Gap share: each lift divided by the 7B's minus the 3B's *cached*
  accuracy on the same rows (fold- and population-matched; the 330-question §23 figures 0.397 and 0.667 are context only).
  Also acc(N) − cached accuracy, the cross-engine reproduction figure.
- **Cost (pre-registered, reported with Claims 1–3, both readers).** (i) On S', the share arm B turns into judged-wrong
  answers (and the same on all marked cached refusals). (ii) On the evidence-cut prompts: the share where F-cut's answer is
  judged correct (the plausibility floor; for context, §34c's plausibility share was 0.605 of the 3B's C-failures and 0.642
  of all questions for top-10 presence, so a judged-correct top-1 answer should be well below that), and the share where
  N-cut refuses (calibration when unknown is right). A selection fix is worth reporting only with (i) and (ii) beside it.
- **Descriptive.** D1-vs-D2 agreement; per-question-type split (single-session-preference and knowledge-update flagged:
  their judge prompts differ and the preference type has 4 questions on the development folds); the 7B under A, B, F, N.

**Prediction, recorded now.** S is about a quarter of the 3B's refusals (0.23 by D1 on the development folds), about
0.07 of in-view questions. Gate G passes. Claim 1 SHOWN (most of S answer correctly once unknown is prohibited), probably
at the edge of power. Claim 2 SHOWN. Claim 3: arm B lifts accuracy by roughly +0.05 to +0.08 absolute over A, about a
quarter of the matched 3B−7B gap at most; F lifts more over N but with a larger (i) and a non-trivial (ii). Reading if
this holds: answer selection is a real, separable, but **minor** part of the 3B's shortfall; the larger part is that the
note does not contain the answer (derivation or reading), which is §34a's A-share and §34c's "present without evidence"
discount.

**Readings, fixed before the data.**
- Claim 1 SHOWN and Claim 2 NOT SHOWN: B is a threshold shift; nothing is said about selection.
- Claim 1 NOT SHOWN and gate G failed: teacher forcing or the engine failed; no verdict on selection; the design note's
  Experiment 3 needs the vLLM route.
- Claim 1 NOT SHOWN and gate G passed: the 3B has a note-to-answer consistency problem that an instruction does not fix
  (the design note's last row).
- Claim 3's lift within the noise floor (under 0.01, with acc(N) − cached also small): the selection route to §23's
  claim 1 is closed.

## 4. Caveats fixed now

The note is text the model wrote, not its computation (chain-of-thought faithfulness); arms B and F do not need
faithfulness, they are interventions on the text. Labels are the §23 single judge (understates accuracy by about 0.1;
paired comparisons unaffected). D1 is a string criterion with the bias stated above; D2 is beside it. Teacher-forcing
the cached note assumes the model would write it again; gate G measures how far that holds on this engine. Marker-less
outputs are outside the frozen-note arms (3B: 39 of 204 on folds 2–4) and reported. Arm F forfeits abstention questions
by construction. Engine, version and flags are recorded per row. No held-out number is read before the Reviewer accepts
this document; the development-fold D1 counts above are the only numbers that exist, and `scripts/exp35_note_check.py`
and `scripts/exp35_report.py` refuse folds 2–4 without an explicit `--confirm`. The pod log prints per-question lines
for folds 0–1 only.

## 5. Commands and spend (NEEDS SPEND; approval pending)

Executor builds `memctl/analysis/answer_selection.py`, `scripts/exp35_generate.py`, `scripts/exp35_pod.sh` (3B then 7B,
arms N/A/B/F/F-cut/N-cut on folds 0–4, then the judge on every answer and every note) and `scripts/exp35_report.py`
(`--folds 0,1` first, then `--folds 2,3,4 --confirm`, then all folds). About 330 intact + 330 cut prompts × up to 6 arms
per reader ≈ 3,300 generations (N, F, F-cut, N-cut full notes of up to 256 tokens; A and B a short phrase) plus about
4,000 judge calls; 3B and 7B load in turn on one A40. Full-size stub run, peak RSS and measured per-call latency before
launch (standing rule). One A40 secure pod at $0.59/h, about 40 min expected; **hard stop 90 min ($0.89)**. Programme
spend so far ≈ $17.00 of the $20 cap; account balance $12.65 on 2026-10-13.

**Triage order, declared now, if the stub-measured latency puts the chain over the 90-minute stop:** drop the 7B's arms
first, then the cut arms (F-cut, N-cut) of the 3B, never the 3B's N, A or B. Gate G is printed on all marked rows and
again on S.
