# Thesis results chapter: outline

Draft of 2026-10-07. Every number here is in EXPERIMENTS.md, cited by section.

## Contributions (mapped to research questions)

1. **(RQ1, RQ4) A query-aware selection head** that beats the best simple rule on LongMemEval with a 7B reader, at
   half the prompt (§15). It is the first learned controller beyond the rule frontier on real conversations.
2. **(RQ1, RQ2) The regret teacher and follow-the-clue search on synthetic tasks**, closing about half the gap to
   the hindsight oracle (D1, §10–11a).
3. **(RQ2, RQ4) Measured limits of query-blind memory control on conversational QA.**
   - Context rot: the reader degrades past about 3k prompt tokens even with all the evidence in view (§13).
   - The query-blind ceiling: before the question arrives, a turn's later need is barely predictable (AUC
     0.62–0.74, §14a).
4. **(RQ3) GRPO with exact credit adds at most about a point over imitation**, on both tasks. The limit is
   information per sample (§10, §16a).
5. **(Method) Pre-registered gating of paid cells without a reader.** Every model-graded cell first passed a free
   check, written down before it ran. Experiments 14–16 spent about $0.44 of an approved $1.70 and never chose
   rows after seeing the test answers. The method is reusable.

## Headline

On real long conversations with a searchable archive, the only learned decision that pays is query-aware: which
few retrieved turns the reader sees.
- A ~27k-parameter head, trained by imitation of gold evidence, picks 5 of 16 search candidates at the question.
- It beats the best simple rule, keep-last-0 + top 5, with the 7B reader: **0.513 against 0.466** judge accuracy
  at **745 against 1,554** prompt tokens (Experiment 15b).
- The arm was **selected as the best of three declared losses, then replicated across seeds and a host**:
  +0.044, seed SD 0.002 (15c).

**Headline figure:** accuracy against prompt tokens on both benchmarks (`docs/research/figures/exp13_frontier.png`).
Before the chapter is final, the Experiment 15 head row should be added to it.

**Headline table:** Experiment 15b (accuracy, the paired difference, the evidence decomposition, and the
"unknown" rate, since context rot is the mechanism the chapter argues).

## RQ1: Does a learned controller beat simple rules under the same budget?

- **Synthetic tasks (yes).** Regret imitation with follow-the-clue search beats every rule and closes about half
  the gap to the hindsight oracle, with the scripted reader and with Qwen2.5-7B (§10, §11a: 0.832 to 0.889 at 2%).
- **Real conversations, query-blind eviction (no).**
  - With a retrieval floor, learned eviction keeps more evidence at a 5% budget: +0.09 to +0.13 in view (§13
    gate).
  - It converts little: one paired win and one loss in five, none after Holm, none on the frontier (§13a).
  - LoCoMo: no win (§13b).
- **Real conversations, query-aware selection (yes, modestly).** The floor head (§15a–c).
- **Tables:** §11 status; §13a learned rows; §15b. **Figure:** the frontier.

## RQ2: Can a controller learn from hindsight?

- **On synthetic tasks, yes.** The regret teacher (D1) and its prices (D5).
- **On real conversations, hindsight about what is needed is not enough.**
  - With one question per episode the teacher has no signal: label starvation and tied regrets (§11e, review F2).
    Composed multi-question episodes and a measured retrieval risk restore it (§13, T4/D1).
  - The teacher rewards evidence per content token, but the reader pays per line, which is context rot (§13a
    decomposition).
  - Pricing tokens cannot fix this when the need itself is unpredictable before the question: the needed-model
    AUC is 0.62–0.74 (§14a). This is the query-blind ceiling, measured on real text.
- **Tables:** §13a decomposition; §14a calibration.

## RQ3: Does reinforcement learning (GRPO) add to imitation? (closed)

- **Synthetic task.** GRPO adds about one point over imitation; how credit is split makes no difference (§10).
- **Real conversations.** Set-valued GRPO on the head, rewarded by the reader, with exact credit. Held-out gain
  +0.0125, CI (−0.005, +0.033): consistent with about one point, inconsistent with more than about three (§16a).
- **Answer.** The limit is the information each sample carries (44–61% of groups tied; a noise floor of about 1
  answer in 50), not credit assignment across time.
- **Closed** by the user's decision of 2026-10-07. A declared but unrun §17 (leave-one-out set reward) is noted
  as future work.

## RQ4: Does it hold up with real models and conversations, and when is it worth it?

- **A learned retrieve gate is the wrong abstraction.** It almost never fires on real text (§11).
- **Context rot.**
  - The 7B reader degrades as the prompt grows even with all the evidence in view: "unknown" rises from 0.079 to
    0.296 between 399 and 6,223 tokens (§13a).
  - Onset is about 3k tokens on both benchmarks, for this reader and prompt (§13b).
  - So the controller's job is to keep the prompt small and right, not to fill the budget.
- **When a learned controller is worth it:** at the question, choosing which turns to show (§15).
- **The two benchmarks need different things.**
  - LongMemEval is a selection problem: about 0.63 of the evidence is already in BM25's top 5.
  - LoCoMo is a search problem: only 0.18–0.21 is, and multi-hop questions need all of it (§12, §13b).
  - The head's evidence gain transfers to LoCoMo (+0.13 all-found), but it fails the gate's token rule (§15b).
- **Transfer.**
  - Synthetic-trained eviction does not transfer to real conversations: −0.05 to −0.09 evidence almost
    everywhere on LongMemEval (§13 gate).
  - Training on one question per episode transfers negatively to LoCoMo at every budget. Training on composed
    multi-question episodes transfers on evidence at 25% (§13 LoCoMo gate). This motivates the composed episodes.
- **Tables:** §12 headroom; §13b; §15b LoCoMo transfer.

## Protocol section (methods)

- **Folds and scoring.** Stratified 5-fold LongMemEval; LoCoMo categories 1–4, with refusal-scored questions
  reported apart.
- **Reader and judge.** The reader is frozen at 7b5fc30; its one revision is disclosed. LongMemEval uses its
  official judge prompts.
- **Pre-registration.** Every gate and every success criterion was written before its run.
- **Statistics.** Paired bootstrap intervals, Holm where several arms are tested, the evidence decomposition, and
  prompt tokens beside every accuracy.
- **Run-to-run variation.** The judge's verdict is stable: 50/50 identical on repeated prompts. Parsed answers
  vary by 1 in 50 on one host and 6 in 50 across hosts. This is why accuracy is reported with intervals.
- **Costs.** About $3.70 of GPU in all, each run approved in advance.
- **Provenance.**
  - Every run records its commit and a dirty flag.
  - The sharded LongMemEval loader has a test showing it plays the same episodes as the full file.
  - An out-of-memory incident is recorded, and the partial runs it left are quarantined in `runs/_aborted/`.
  - The cache was scanned after a thread race: 0 entries under the wrong key.
- **Memory rules.**
  - Every laptop job runs under a memory guard.
  - Every paid launch is preceded by a full-size dry run with a stub model, with peak RSS recorded.

## Limitations

- **One reader and one frozen prompt** (Qwen2.5-7B-Instruct, the 7b5fc30 prompt). The ~3k-token onset of context
  rot is measured for that pair only.
- **The judge is the reader's own model.**
  - Every Experiment 13–16 cell is judged by Qwen2.5-7B on its own answers, with LongMemEval's official prompts.
  - The planned second-family judge check never ran: Llama-3.1-8B re-judging a stratified sample, with
    false-accept and false-reject tests. That would cost about $0.50–1 and is the user's call.
  - The ranking of controllers is what matters and probably holds, but it has not been shown.
- **The positive reader result is on one benchmark.** LoCoMo was not tested with the reader for the head: it
  failed the gate's token rule.
- **A selection step precedes the replication.** The `listsum` arm was the best of three declared arms.
- **LoCoMo intervals resample only 10 conversations**, so they undercover.

## Proposal next-steps (to update)

- **Done:** the retrieval floor and context rot (§13), the floor head (§15).
- **Closed:** RQ3.
- **Next:** adaptive k, then writing.
