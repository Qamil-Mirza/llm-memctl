# §37 pre-registration: can the 3B be cross-examined into checking its own answer? (an elenchus in three turns)

Date: 2026-10-16, Advisor draft, revision 1, before any new verifier number on any fold. Branch `jacobian-lens`.
Follow-up to §36/§36b (`research/proposals/2026-10-15-exp36-verification.md`, results review
`research/reviews/2026-10-15-exp36b-results.md`, EXPERIMENTS.md §36a). Prompted by the user's question of 2026-10-16:
"Did we try putting the 3B model through Socrates's elenchus?" We did not. Every §36 check was one shot and one token.
**Two stages: a free development-fold run on the laptop CPU that decides whether a pod is worth quoting, then a pod
run that NEEDS the user's spend approval at the quoted price.** No confirmation-fold number is read before the user
approves the pod and the Advisor's review of the development run is in `research/reviews/`.

## 0. What the record says and what this adds

- §36b: the 3B cannot tell which of its own forced answers the memory supports when asked in one shot (AUROC 0.573
  (0.478, 0.666), NOT SHOWN; development 0.559). Its one-letter support score follows evidence *presence* (+0.235 when
  the evidence is cut; A′ 0.715) but not whether the candidate *matches* the evidence. Its refusals carry no signal
  (0.455). The 7B checking the same candidates reaches 0.692 (descriptive; it is also the judge).
- §34c: about 0.6 of the 3B's failures are "present without the evidence"; the 3B's trouble is partly extraction.
- The elenchus is a different question from §36b's. §36b asked for a verdict on the whole. An elenchus asks the model
  to state the grounds of its answer, derive what those grounds say on their own, and then face the comparison. Each
  step is smaller than the whole-cloth judgment: citing lines, reading two or three lines, comparing two short phrases.
  §36b says the 3B can do the first (presence). Whether the chain delivers a verdict that the one-shot could not is the
  open question. The design also carries its own refutation: if the model cannot ground its answer in any real line,
  that is a contradiction with the memory itself, found without a label.
- Known hazard, built in as a control: a model confronted with its own thesis tends to confirm it. The derivation step
  is therefore run **blind** (the candidate is not in view) as the primary form, and the in-dialogue form is run beside
  it as the sycophancy measurement. A **planted wrong thesis** control measures how often the chain passes an answer we
  know is wrong.
- Prior from the record, written down before the data: my predictions have over-estimated the 3B twice (§35 by 2–3×
  in the other direction, §36 Claim 1 and Control A). The honest prior for Claim 1 clearing 0.65 is about one in three.

## 1. Population and data (unchanged from §36, amendment 1)

- The §35 population: §23 `fixed8` in-view answerable questions with a judge label, paired by question id and prompt
  hash. **Development folds 0–1** (100 fixed-candidate 3B rows, 84 with a candidate). **Confirmation folds 2–4** (165
  fixed rows, 140 with a candidate). Rows where arm B still said unknown have no candidate; every condition abstains
  on them and they stay in every coverage and risk denominator. Discrimination claims are over rows with a candidate.
- **Candidate** = §35 arm-B `final_answer`, byte-exact. **Label** = the §35 judge verdict on B (7B judge; audit 58 of
  60, `runs/exp36_audit/README.md`). 7B rows are descriptive for the judge-circularity reason given in §36.
- **Unsupported rows:** Cut-B (the same candidate on the §34c exact-span evidence-cut prompt) and Cut-F (the 3B's
  evidence-free F-cut candidate on the cut prompt). The 30 abstention questions are descriptive as in §36.
- **Bootstrap unit** = the question (all variants of a question in one unit). Paired bootstrap, 10,000 resamples,
  95% intervals, seed 0.
- The §36b scores `s` on every fold already exist (`runs/exp36b/qwen3b/V/`). Claim 2 reads them as a fixed baseline;
  nothing about them is recomputed or re-chosen.

## 2. The elenchus (Stage 2), 3B first, same checkpoint, greedy

Engines: pod vLLM 0.8.5 bf16 (confirmation and the pod's development re-run); laptop `transformers` fp32 CPU
(development run, as `scripts/exp36_wording_probe.py`). Sampling greedy, temperature 0, seed 0. The chat template
wraps each user message; no system message; the assistant turn starts empty (the §36b lesson: never end a message
with a literal that names another task).

**The memory block is the cached §23 memory block with the reasoning instructions removed, byte-exact except that
each memory line is prefixed `[k] ` for k = 1..n in prompt order.** The numbering is what makes grounding mechanical.
Cut-B and Cut-F keep the numbering over the remaining lines (renumbered 1..n′). The §36b one-shot had no numbers, so a
one-shot **V-num** (the §36b frame (a) wording on the numbered block) runs beside as a descriptive comparison, to
separate "dialogue" from "numbering" in Claim 2.

| turn | what the model sees | what it returns | parse |
|---|---|---|---|
| **E1 grounds** | numbered memory block; `Question: <q>`; `You answered: <candidate>`; "Which memory lines is this answer based on? Reply with the line numbers only, for example [2] [5]. If no line bears on the question, reply NONE." | up to 16 tokens | G = the set of valid line numbers in the reply (numbers outside 1..n are dropped). **g = 1** if G is non-empty, else 0 (NONE, empty, or only invalid numbers) |
| **E2 derivation, blind (primary)** | a fresh context: only the cited lines, with their numbers, then `Question: <q>`, then "Using only the lines above, what is the answer? Answer with a short phrase. If the lines above do not determine the answer, reply: not determined." The candidate is **not** shown | up to 32 tokens | d = the reply's first line. Skipped when g = 0 (d := not determined) |
| **E3 confrontation** | no memory: `Question: <q>`; `Answer 1: <x>`; `Answer 2: <y>`; "Do Answer 1 and Answer 2 give the same answer to the question? Reply with one letter. A = same, B = partly the same or unclear, C = different". x, y are the candidate and d in an order fixed by the parity of the question id's first hex digit (recorded per row) | 1 token, logprobs 20 | a = (p_A + 0.5·p_B)/(p_A + p_B + p_C), v2 byte-decoded scorer, primary spellings bare and space-prefixed, `label_mass` stored; the §36 reliability rule (mass < 0.5 on more than 0.1 of rows = not reliable, argmax only) |

**Elenchus score e** = 0 if g = 0 or d is "not determined" (or a refusal by the §23 rule); otherwise e = a.
**Mechanical agreement m** = 1 if d and the candidate are the same after a normalisation fixed in code before the
development run (lower case; strip punctuation and the articles a/an/the; number words one..twenty to digits; two
numbers equal if numerically equal; otherwise one must equal the other or contain it as a whole-word phrase). **e_mech =
g·m**, binary, label- and model-free beyond the two generations. **Grounding alone, g,** is reported as a score too.

**E2 in dialogue (secondary; the sycophancy measurement).** The same derivation asked as a third user turn after E1,
with the E1 reply and the candidate in context: "Based only on lines <G>, what do those lines say the answer is?
Short phrase, or: not determined." Its d′ feeds its own E3 and gives e′. e′ is never primary.

**Wording fixed now.** The three instructions above are the texts; one pre-fixed fallback for E1 exists in case the
parse gate fails (E1′: "List the numbers of the memory lines that support your answer, separated by spaces, or write
NONE."). No other wording is tried. Texts, normalisation and the hash rule are committed before the development run
and their sha256 is in every row.

## 3. Conditions (Stage 3), on the same fixed-candidate rows

| | decision | threshold |
|---|---|---|
| C0 natural | emit the candidate iff the cached answer was not a refusal (§36) | none |
| C1 forced | emit every candidate | none |
| C2e self-check | emit iff e_mech = 1 | none |
| C3e elenchus gate | emit iff e ≥ τ | matched coverage (Claim 3) |
| C3s one-shot gate (§36b, fixed) | emit iff s ≥ τ | matched coverage; the baseline Claim 2 compares against |
| C5 repair (descriptive) | if e_mech = 0 and d is a real answer (not "not determined", not a refusal), emit **d** instead of the candidate; otherwise as C3e at C0's coverage | needs a judge pass on d (the §35 judge, cached by text) |

## 4. Claims (3B, confirmation folds 2–4, rows with a candidate; fixed sequence, each tested only if the previous is SHOWN)

- **Claim 1 (discrimination).** AUROC of e for B judged correct vs judged wrong. SHOWN if the lower bound is above
  0.5. Meaningful if the point is ≥ 0.65 (the §36 bar, unchanged). PR-AUC, prevalence, and the AUROC of e_mech and of
  g alone are printed beside it.
- **Claim 2 (the dialogue adds something).** Paired AUROC(e) − AUROC(s) on the same rows, s from `runs/exp36b`.
  SHOWN if the lower bound is above 0. Meaningful if the point is ≥ +0.05. V-num's AUROC is printed beside so the
  reader can see how much of any gain the numbering alone gives (descriptive, not a claim).
- **Claim 3 (better choices than the model's own refusals).** At C0's coverage, paired risk(C3e) − risk(C0). SHOWN if
  the upper bound is below 0. No threshold is fitted.
- **Claim 4 (repair; descriptive with intervals).** Accuracy of C5 against C1 on the same rows; how many judged-wrong
  candidates d corrects and how many judged-right ones it breaks.

**Controls (pre-registered, both readers, descriptive with intervals).**
- **(A) Evidence cut.** Grounding rate g on intact vs Cut-B (paired drop), and e(intact) − e(cut) on rows judged
  correct. Cut-F: the share of evidence-free candidates that pass (e ≥ 0.5), the unsupported-answer rate.
- **(P) Planted wrong thesis.** On rows judged correct whose gold is a number or a name (`gold_kind`), the candidate is
  replaced by a wrong value of the same kind (number: §36's `substitute_value` offset; name: another question's gold
  of kind name, from the same fold set) in the **intact** memory, and the elenchus runs on it. The memory is not
  edited, so the §36 consistency problem with names does not arise. Report the rejection rate (e < 0.5) of planted
  theses beside the acceptance rate (e ≥ 0.5) of the true candidates on the same rows, and "follows the evidence" =
  both at once. This is the confirmation-bias measurement. About 40 development and 70 confirmation rows qualify
  (counted on development before the run, as §36 did).
- **(S) Sycophancy.** On rows judged wrong: the share where d reproduces the candidate (m = 1), blind vs in dialogue.
  On rows judged correct: the same two shares. The gap is the pull of the model's own thesis.
- **(7B).** The 7B runs the same elenchus on the 3B's candidates (V, Cut-B, planted) and on its own B candidates. Never
  merged into one size conclusion; judge circularity as in §36.
- **Costs beside every claim.** Calls, tokens and seconds per candidate against §36b's one call (0.07 s on an A40);
  wrong answers added vs C0; Cut-F pass rate.

**Prediction, recorded now.** Claim 1: AUROC(e) 0.58–0.66; the chance that the point reaches 0.65 is about 0.35.
Claim 2: the interval includes 0 (the chain is not clearly better than the one-shot); V-num close to s. Claim 3 not
tested or not shown. Control A: g about 0.8 intact and about 0.4 on the cut prompt (the 3B will still cite some line
on about 0.4 of evidence-free prompts, consistent with A′ 0.715). Planted thesis: the 3B rejects 0.55–0.70 of planted
wrong answers, the 7B above 0.8. Sycophancy: in dialogue the 3B reproduces its own wrong candidate on at least half
the judged-wrong rows, blind on under a third. Repair: d corrects 0.10–0.20 of judged-wrong rows and breaks about as
many judged-right ones; net about zero. 7B on 3B candidates: 0.72–0.80.

**Readings, fixed before the data.**
- Claims 1–2 SHOWN and planted rejection ≥ 0.7: §36b's failure was the form of the question; decomposed, the 3B can
  check. The policy half of the gap (§35) is then recoverable with three cheap 3B calls, stated as elicited.
- Claim 1 SHOWN, Claim 2 NOT SHOWN: the chain is another presence detector. The limit is the judgment, not the format.
- Claim 1 NOT SHOWN, or the development gate fails: the 3B cannot even ground its own answers reliably; §34c's
  extraction failure is the bottleneck. The self-verification route of Q14 closes with a second negative and only a
  larger checker (Q15) remains.
- g alone matches or beats e: citation is the signal and the rest of the chain is decoration; the cheaper readout wins.
- Planted rejection low with Claim 1 SHOWN: the score tracks the model's prior for the answer, not the evidence;
  gating would be a confidence trick (the §36 reading).

## 5. Stage 1: the free development run and the gate that decides whether a pod is quoted

Laptop CPU, folds 0–1 only (the script refuses other folds), 3B only, `GUARD_VM_GB=22` (the §36b probe peaked at
18.5 GB RSS; its prefill cost 18 s per 1,000-token prompt at 8 threads).

1. **Compliance gate (label-free, first 5 rows, then all 84 intact and 84 Cut-B rows).** E1 parses (line numbers or
   NONE) on ≥ 0.9 of rows; E3 label mass ≥ 0.5 on ≥ 0.9 of rows. If E1 fails, E1′ once; if that fails, §37 closes on
   compliance ("the 3B cannot be made to cite lines in this format") with no spend.
2. **Development readout (correctness-linked, development folds only, permitted by the peeking rule).** AUROC(e) on
   the 84 candidate rows, Control A, and, time permitting, P and S.
3. **Gate to quote a pod:** development AUROC(e) point ≥ 0.65 **and** ≥ AUROC(s, development) + 0.05 = 0.609. Both are
   read off the laptop run. If the gate fails, §37 closes as "not promising enough to confirm" and the development
   result is recorded in EXPERIMENTS.md §37 as a development-only closure (as §25 and §30 were). No confirmation-fold
   number is ever read. If it passes, the Advisor reviews the development run and the Executor quotes the pod to the
   user.
4. **Time budget.** Estimated 60–90 s per candidate row on the laptop (prefill 1,000 tokens, ≤ 48 generated tokens,
   two short prefills): intact 84 rows about 2 h, Cut-B about 1.5 h, planted about 1 h, in-dialogue E2 about 0.5 h,
   about 5 h in all, in the background. **The Executor times the first 5 rows and reports the projection before
   running the rest.** If the projection exceeds 8 h, P and S move to the pod only and the laptop runs intact, Cut-B and
   V-num. The user is told how long the laptop will be busy before it starts.
5. **Engine caveat.** The laptop is fp32 `transformers`, the pod is bf16 vLLM. The pod re-runs the development folds and
   reports them in brackets; the gate decision stays the laptop's. A development AUROC that moves by more than 0.05
   between engines is reported as such.

## 6. Stage 2: the pod (NEEDS SPEND; quoted only if the gate passes)

- One A40 secure pod at the then-current price (§36b: $0.59/h). Work: 3B elenchus on all folds (intact, Cut-B, Cut-F,
  planted, in-dialogue E2, V-num; about 330 rows × 6 items × 3 short calls), the 7B on the 3B's candidates and its own,
  and the judge pass on d (and d′) for the 3B rows, cached by text as §35. §36b's 3B stage took 250 s for about 3,000
  one-token prompts; generation here is ≤ 48 tokens per call. Expected 20–30 min; **hard stop 60 min (≈ $0.59)**.
  Programme spend now ≈ $17.55 of $20; after this run about $17.85.
- **D-22 smoke in the pod:** 5 development rows per reader through E1–E3 before the main chain; abort unless ≥ 4 of 5
  parse at E1 and ≥ 4 of 5 have E3 mass ≥ 0.5. 7B smoke fail with 3B pass: 3B only, 7B rows "not compliant".
  `RUNPOD_POD_ID` exported for the in-pod self-stop; laptop DELETE as the backstop; the lab-broker sweep stopped first.
- **Triage order if the stub latency exceeds the stop:** drop the 7B's own-candidate pass, then in-dialogue E2, then
  V-num, then planted names (keep planted numbers). Never drop the 3B's intact, Cut-B, Cut-F, planted numbers, or the
  judge pass on d.
- Reports: `scripts/exp37_report.py --folds 0,1`, then `--folds 2,3,4 --confirm`, then all folds. The report refuses
  folds 2–4 without `--confirm`, and `--confirm` is run once, after the Advisor has the development review in
  `research/reviews/` and the user's pod approval is in DECISIONS.

## 7. Caveats fixed now

Three new prompts: everything is elicited verification, never "the reader knew". The line numbering helps grounding
and is absent from the §36b baseline; V-num is beside Claim 2 for that reason, and Claim 2 is a claim about the whole
procedure (numbering included), not about dialogue in the abstract. Labels are one 7B judge (understates accuracy by
about 0.1); the repair arm's d is judged by the same judge; the 7B elenchus rows share weights with it. The blind
derivation sees only the cited lines, so a correct candidate grounded in the wrong lines scores low: that is intended
(it is ungrounded), but it means e punishes some right answers. The normalisation behind m is a rule, fixed before
the run; e (model comparison) is primary so that m's rule errors do not decide a claim. The development gate rests on
84 rows and a wide interval; a pod can be refused for a true effect, which is the cheaper error. The planted-thesis
control uses only number and name golds (a subset biased toward D1-type questions, as §36 Control B). One benchmark,
one prompt family, one 3B.

## 8. Build (Executor), review (Advisor), decisions (user)

- Executor builds `memctl/analysis/elenchus.py` (numbered block, E1–E3 builders, parsers, normalisation, planted
  thesis, scores), `scripts/exp37_dev.py` (laptop, folds 0–1 only, resumable per question/item/turn, 5-row timing
  first), `scripts/exp37_pod.sh` (smoke, 3B, 7B, judge, self-stop) and `scripts/exp37_report.py`. Unit tests: the
  parser on NONE / `[2] [5]` / `2, 5` / out-of-range; the normaliser on the cases listed in §2; the planted substitute
  never equals the gold after normalisation; the blind E2 prompt never contains the candidate string.
- Advisor reviews the build before the laptop run starts (prompts byte-exact against §2, the refusal of folds 2–4,
  the gate arithmetic), reviews the development run against the gate, and reviews the confirmation results.
- User decides: (i) whether the laptop may be busy for about 5 h on the free development run; (ii) if the gate passes,
  approve or decline the pod at the quoted price and hard stop. DECISIONS rows are written only after each decision.
