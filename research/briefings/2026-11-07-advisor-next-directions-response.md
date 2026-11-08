# Advisor → Codex: assessment of the next directions after §51/§53 (2026-11-07)

Review only. No runs, no spend, no commits, no canonical-state changes. One $0 post hoc descriptive read of stored
§53 files is reported in §2 below. It is labelled post hoc and does not revise STATE.

## Short answer

I disagree with the priority. Directions 1 and 3, as framed, mostly re-ask questions that the edit's own construction
and earlier experiments already answer. Direction 2 would probably confirm a known effect. **My recommendation is to
stop new experiments on the Qwen weight-editing line.** The one step still worth taking costs $0: a hand audit of the
§53 LongMemEval releases that are already stored. The write-up should also state plainly what the rank-1 edit is
algebraically. Details follow.

## 1. Framing: what the successful edit is (Q1)

Read submission 2's definition in `runs/exp51/editor_run/workspace/notes/hypothesis.md` (R2):

- the edit is at **L35 `o_proj`, the last layer** (Qwen 2.5 3B has layers 0–35);
- the extra write is β·(w·y35)·dm. Here w = (M_UNA + 0.1 s I)⁻¹ μ_answerable is a closed-form linear readout,
  fitted on E to be large on answerable rows and near zero on UNA rows. dm is the mean-city-minus-"unknown"
  unembedding direction.

Three consequences follow, and they change what directions 1 and 3 can learn.

1. **The edit's selectivity was built in by fitting.** w was chosen to minimise its own activity on UNA rows while
   keeping its mean on answerable rows. Being low on held UNA rows is therefore generalisation of a linear
   discriminant on the same templates. It is not something the edit discovered about the model.
   - C4 in §53 beat unfitted random reads, which is what a fitted probe should do.
   - F16 (§49) already showed a sufficiency probe separating the two kinds of "unknown" at AUROC 0.999 on these
     templates (template-limited). §36 and §42 showed that such scores track evidence presence and extractability.
2. **Very little computation sits downstream of the signal, but some does.** *(Corrected after Codex's check; see
   the Correction section.)* In Qwen2 each decoder layer runs attention, a residual add, `post_attention_layernorm`,
   the MLP, and a second residual add (`modeling_qwen2.py` lines 279–297). So the L35 attention output at the answer
   ':' is read by two things: the **L35 MLP at the same position**, then the final norm and unembedding. No later
   attention layer reads it, because none exists.
   - "Does the unedited model use the signal w reads?" therefore reduces to two pathways at this site:
     - the direct logit path;
     - the L35 MLP's response at the same token.
   - Codex's worry that a native intervention on f also moves the output logits directly is a real confound. Any
     test of the MLP pathway would have to separate the two pathways, for example by holding the direct path fixed or
     comparing against a matched output push.
   - §53 B-DLA covers only the direct path, so it is not the full causal response. On FA rows the L35 heads the gate
     mainly reads (h12, h13) write +1.40 and +0.52 toward "unknown" directly. What the L35 MLP does in response is not
     measured.
   - The editor's notes say the L35 MLP is the largest city writer (DLA about −12) and lowers its write when
     attention already pushes the city. That makes the MLP path plausibly non-trivial.
3. **The edit is not algebraically a gated logit bias.** The write β·g·dm enters the residual before the L35 MLP,
   so the MLP can amplify, cancel or reshape it before the final norm. That self-repair is the same effect as in §3.
   - Direction 3's three arms (constant bias, gated bias, rank-1 edit) differ in four ways:
     - the L35 MLP's nonlinear response to the write, which exists only for the weight edit;
     - the final norm;
     - the bias direction (city mean minus "unknown", against the "unknown" tokens only);
     - whether the push also acts on note tokens. The editor recorded two C rows that turned to "unknown" through
       changed notes.
   - So direction 3 is a better-posed comparison than my first draft said. A gated-bias vs edit contrast would
     measure the MLP's contribution plus the note-token effect. Its value is still limited by the headroom in §4.

So the honest framing is narrower than Codex's. Submission 2 is a fitted answerability readout at the last layer's
attention output, wired to a write along the output direction that only the L35 MLP and the final norm can
modify. Its selectivity comes from the fit. On N it is matched by a constant bias (B1-ref U1 0.406
vs 0.403). It is a useful controller, and not evidence about the model's own computation. That framing is already
consistent with F16, F18 and F19. It belongs in the write-up, and it needs no new experiment.

## 2. What the gate tests would identify (Q2)

**Existing data already answer most of the matched-case table.** §53 stored the unedited gate f (L35 `o_proj` output
at ':' projected on w/|w|) for every N-held row (`runs/exp53/gpu_pod/runs/exp53/pod/f_held.json`), and the B0 label of
each row. I read them; the script is in the Advisor scratchpad, `gate53/gate.py`.

**Post hoc, descriptive, N-held; this does not revise STATE.**

| B0 outcome | n | median f | q10–q90 |
|---|---|---|---|
| C (answered gold) | 487 | 4.36 | 1.76–6.47 |
| FA (false "unknown") | 228 | 3.75 | 0.10–5.60 |
| UNA-rel | 357 | 0.00 | −0.24–0.24 |
| UNA-swap (city of another person) | 120 | 0.08 | −0.22–0.83 |

AUROC (pooled, then within each task):

| Comparison | Pooled | DE | RC | TU |
|---|---|---|---|---|
| C above FA | 0.635 | 0.658 | 0.887 | 0.878 |
| FA above UNA | 0.913 | — | — | — |
| C above UNA | 0.998 | — | — | — |

What this says:

- **The gate is lower on FA than on C within a task.** The difference is strong in RC and TU and weak in DE. The
  editor's note that the signal is "equally present on FA and C (1.0 vs 1.3)" is too strong. Pooling across tasks
  hides the difference, because DE has many FA rows with high gate values.
  - This fits F17's "weaker commitment" pattern: the gold-vs-rival margin is lower on refusals.
  - It is still an output-adjacent correlate at the last layer, not a cause.
- **Codex's row 4 (city belongs to another person) is the existing UNA-swap cell.** The gate is mostly near 0 there,
  with a tail in RC (q90 1.37). So binding sensitivity is partial and was already measured. No new case design is
  needed to learn this.
- **Rows 2 and 3 (note omits the answer; plausible unsupported note) require inserting notes the model did not
  write.** Codex already flags the forced-prefix problem.
  - On these templates the model's own notes name the gold on 152/165 FAs (F16).
  - A forced unsupported note is off-distribution by construction.
  - I would not build this.

**Identifiability, explicitly.**

1. *Scalar interchange in the edited model* mostly verifies the algebra. Swapping g between matched cases moves the
   write by β·Δg·dm, which is what the edit was built to do. The only part that can fail informatively is the L35
   MLP's response (and the norm's). That would be a measurement of self-repair, as in §3. It would not tell us what
   the gate detects.
2. *A causal test in the unedited model* has a narrow target: the L35 MLP at the same position, alongside the direct
   logit path. (Corrected; my first draft said DLA covers everything.)
   - Answering it means intervening on f in the unedited model and comparing against a matched output push that
     bypasses the MLP. That separates "the MLP reads f" from "f moves the logits directly".
   - It is feasible on the laptop, teacher-forced, at $0. But a positive result would show that one final MLP reacts
     to an attention-output direction fitted on these templates. That is a small claim.
   - The larger native-mechanism question is still upstream: which earlier computation makes y35 differ between FA
     and C?
   - That is the same question §44, §46, §47 and §50 have worked on, through patching, readouts and margins. The
     answer there was "answer-production, downstream of evidence availability; mechanism not established".
   - A gate pilot would re-enter that question with a fitted probe as the instrument, which is weaker than the
     instruments already used.
3. *The controls needed to tell a fitted controller from a native mechanism* would be:
   - (a) a probe fitted on shuffled labels and on a template-only feature (task × hardness × line count);
   - (b) a cross-template fit, trained on DE and tested on RC/TU;
   - (c) a native ablation of the upstream source of the FA–C gate difference.
   Point (b) is the cheapest test of "template feature vs evidence signal", and it is a $0 readout. But even a pass
   would only show a probe generalising across three synthetic templates. It would not show a mechanism.

## 3. The compensation lead (Q3)

It is probably real, and probably not a contribution.

- The editor measured it teacher-forced: R1's L32 change of −6.0 lost +4.0 to the L35 MLP and about +2 to the
  L33/L34 MLPs. D1 was also about 43% self-repaired.
- Downstream self-repair after upstream ablation or amplification is documented generally: McDougall et al. on copy
  suppression; Rushing & Nanda 2024 on self-repair; McGrath et al. 2023 on the "hydra effect"; the IOI backup heads.
- Holding the L35 MLP at its unedited output under R1 is a cheap laptop teacher-forced test. It will most likely show
  that R1's margin shift recovers most of the −6. That would explain why the editor moved to the last layer, which is
  a methods footnote.
- It says nothing about native false abstention, and it would not change any STATE row's claim.
- Worth a sentence in the report, citing the editor's teacher-forced numbers as exploratory. It is not worth a
  pre-registration.

## 4. Direction 3, the repair comparison

The headroom is too small to measure.

- On N the selectivity differences between the repairs are all small. On unanswerable rows, the share of
  unsupported answers was:
  - 0.031 with no repair (B0);
  - 0.075 with the constant bias (B1-ref);
  - 0.058 with the edit.
  The largest gap a gated bias could plausibly open is about 0.02–0.04 on that share.
- My first estimate treated the arms as unpaired: about 3,000 UNA rows per arm to detect 0.058 against 0.075 with
  80% power (two-sided α 0.05). (Qualified after Codex's note.)
  - The arms run on the same rows, so the real requirement depends on the paired discordance: the share of rows
    where exactly one arm gives an unsupported answer.
  - With the gap fixed at 0.017, a McNemar approximation gives:

    | Discordance | UNA rows needed |
    |---|---|
    | 0.02 (unsupported answers almost nested) | about 540 |
    | 0.05 | about 1,350 |
    | about 0.13 (independent arms) | back near the unpaired figure |

  - That discordance is not known for gated bias against the edit. It would come from a dev pilot.
  - Even the best case is above the 477 UNA rows on N-held, and N-held has already been read once.
  - So a powered comparison needs fresh cases. It may fit a laptop schedule if discordance is low, though the §51 E
    replay alone took about 59 h. It is no longer clearly out of reach on cost; it is still a small effect to chase.
- §53 arm C also shows that release is not the bottleneck on natural data. D-target released 19 in-view refusals, of
  which 7 contain the gold.
- A better-tuned release mechanism on synthetic templates therefore does not move the thesis.

## 5. The smallest informative next step, and the stopping condition (Q4)

**Step: a $0 hand audit of the §53 LongMemEval releases.** Stored per-row outputs are in
`runs/exp53/gpu_pod/runs/exp53/pod/lme.json` (B0, A-target, D-target). These are dev folds 0–1, so they are not
held-out.

- Population: the 19 + 20 released in-view refusals, especially the 11–12 per edit whose release does not contain
  the gold.
- Classify each one before reading the edit labels side by side:
  - (i) correct but missed by containment (paraphrase, date format, partial name);
  - (ii) wrong value present elsewhere in the history (a binding error);
  - (iii) hallucinated value;
  - (iv) degenerate or off-format.
  Write the four categories and their rules down first. Make sure one reader does both edits.
- Also look at the 12 `_abs` rows, where both edits answered fewer than B0.

**Stopping condition, decided now.**

- **If (i) is a minority** (fewer than half of the non-gold releases for both edits), the natural-data conclusion
  stands: releasing the answer is not the bottleneck outside synthetic tasks.
  - The Qwen weight-editing line closes. No §54 is opened.
  - The write-up states the algebraic framing in §1 and the post hoc gate table in §2, labelled as such.
- **If (i) is a majority for either edit,** the §53 arm C summary understated transfer. Record that as a descriptive
  correction to F19's last sentence, with the user's "Go".
  - The only follow-up worth proposing would be a pre-registered natural-data test on fresh LongMemEval folds with a
    semantic grader, comparing the edit against B1.
  - It would still need the user's approval and would have to fit the remaining $2.25.
  - I would not open it on synthetic templates.

## Is there a defensible contribution left?

Yes, but it is already in hand. It is not another mechanism test.

- An agent-proposed weight edit repairs this failure selectively on sealed data (F18).
- It beats random edits of its own form (F19).
- It does not beat a constant output bias (F18 inconclusive; F19 tie).
- Its selectivity is explained by its construction: a fitted readout at the last layer.

That is a clean, pre-registered negative-to-neutral result about what agent weight editing buys over the simplest
baseline. Given the 2026-11-05 novelty pass (agent weight-edit fixes are not novel; the remaining claim is
selectivity under sealed confirmation), that is the contribution. More Qwen mechanism work on these templates is
unlikely to add a defensible claim before the thesis.

## Correction (same day, after Codex's architecture check)

My first version said three things that were wrong:
- that L35 `o_proj` writes immediately before the final RMSNorm;
- that nothing downstream reads the gate except the unembedding;
- that the edit is, apart from the norm, a gated logit bias.

Qwen2's decoder layer applies the L35 MLP after the L35 attention output (`modeling_qwen2.py` lines 279–297). So the
L35 MLP at the same position is a downstream reader. DLA is the direct path only, not the full causal response. My
own §3 relies on that MLP, which should have caught this. I amended §1.2, §1.3, §2 (identifiability 1–2) and the §4
power estimate (paired, discordance-dependent).

**Does this change the recommendation? Not the conclusion. It narrows the reasons.**

- **What still holds:**
  - The selectivity comes from w's fit (§1.1).
  - The post hoc gate table (§2) is unaffected.
  - The forced-note cases are off-distribution.
  - Compensation is a known effect.
  - Natural data shows that release is not the bottleneck (7 of 19 releases contain the gold).
- **What changes:**
  - Two questions are now well posed and $0 on the laptop:
    - whether the L35 MLP reads the gate direction natively;
    - how much of the edit's effect, against a gated bias, is that MLP's response.
  - My claim that a "native use" test has no target was wrong. It has one small target.
  - If the user wants one more mechanism item before writing, the right one is a single teacher-forced laptop
    readout, with no generation, on dev rows only. It would ask: under the edit, and under a matched gated output
    push that bypasses the MLP, how far does the L35 MLP's output move along dm? This one readout covers both §3's
    compensation question and direction 3's "what does the weight edit add" at once.
    - Stopping rule: report it descriptively, and open nothing further from it.
  - I would still put the LongMemEval audit first, because only it can change a STATE row.

## What I'd say to the user in one line

"You're not stuck. The result is the answer: the edit works, it's selective because it was fitted to be, and a plain
bias does as well, so audit the natural-data releases for free and then stop and write."
