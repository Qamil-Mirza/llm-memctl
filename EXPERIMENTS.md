# Experiments

The record of the honors thesis: every experiment, what it asked, what it found, and where its full text, code and
data are. Two eras are closed and condensed here (consolidation of 2026-11-11, D-86):

- **§1–§32, memory controllers.** Full text: `git show memctl-era-final:EXPERIMENTS.md`. Data: `archive/memctl-era/`.
- **§33–§57, false abstention.** Full text (3,604 lines, with every command, number and caveat):
  `git show false-abstention-era-final:EXPERIMENTS.md`. Code, proposals and reviews are at the same tag. Data:
  `archive/false-abstention/` (see [archive/README.md](archive/README.md)).

What the closed eras support, and how strongly, is in [research/STATE.md](research/STATE.md) (F1–F24). The command
that recomputes each finding is in [REPRODUCE.md](REPRODUCE.md). Where a condensed line here disagrees with the full
text at a tag, the full text wins and this file should be fixed. Live work (self-doubt) is numbered from §58.

## 1–32. The memory-controller era (condensed, 2026-09-30 to 2026-10-10)

**What was asked.** Whether a learned memory controller (heuristics, imitation of a hindsight teacher, RL, a
prompted LLM, a published scorer) helps a language model answer questions over long histories, first on scripted
tasks and then on LongMemEval and LoCoMo.

**What was found.**
- On scripted tasks, learned controllers matched but did not beat the best content rule, and an archive-aware regret
  teacher fixed imitation's archive failure (D1). RL never beat imitation; GRPO was reliable only on the workflow
  task.
- On real conversations, the only clear gain was query-aware selection at question time: the §19 head picking 8 of
  32 BM25 candidates (`fixed8`), +0.045 over the §15 head and well above FIFO. A 31-parameter logistic regression did
  as well as the learned head (§32). Writing notes, rewriting queries, reader-utility training, OpenJev and LRE did
  not beat it (§20, §26–§29).
- The ceiling was the reader. With the controller fixed, a 3B reader still trailed a 7B one (§23 Claim 1 not shown;
  §24 the same in Llama and Phi-4), and with the evidence in view the 3B often answered "unknown". That became the
  thesis question (§33 onward).

**Status of each experiment** (the table of the original record, unchanged):


|---|---|---|
| 1 | Do policies differ at equal budget? (first milestone) | **Yes.** PASSED |
| 1b | Does attribution recover an injected reasoning-failure rate? | **Yes.** PASSED |
| 1c | Does policy quality matter more as the horizon grows, at a fixed token budget? | Headroom grows. PASSED |
| 1d | Does policy quality matter more as dependencies get longer? | **Yes.** PASSED |
| 2 | What is archive + retrieval worth against deletion? | Large gain; retrieval becomes the bottleneck. PASSED |
| 3 | Do compaction and consolidation help? | Compaction: only if lossless. Consolidation: little. PASSED |
| 4 | Do learned controllers beat heuristics? | They beat every recency rule and match, but do not beat, the best content rule. PASSED |
| 4b | …with archive and retrieval? | Imitation of the hindsight expert deletes when it should archive. PASSED |
| D5 | …and when the archive has a price? | Imitation collapses at any price; cost-sensitive imitation wins at low price, over-archives at high. PARTIAL |
| D5b | Does a plug-in Bayes rule track the price? | Yes at 0.01 and 0.05 (+0.07 to +0.28); loses 0.06 to delete-all at 0.2 (D5c: not a value-calibration problem). PARTIAL |
| GRPO | Does GRPO fix PPO, and does an oracle warm start poison RL? | GRPO is reliable on the workflow task, not better on recall; the oracle warm start halves RL's result. PASSED |
| D2 | Can one controller serve both task structures? | **Yes**: matches each specialist on its task; every fixed rule fails on one. Holds with identical action sets (D2b). PASSED |
| D1 | Does an archive-aware regret teacher fix 4b? | **Yes**: 0.34 → 0.85 at 2%, above every heuristic at every budget. Clean commit, 3 seeds. PASSED |
| 5 | Does the ranking hold on a sequential task? | **No, it flips.** PASSED |
| 5b | Learned controllers on the sequential task | Task-trained imitation reaches 1.0 where the blind oracle does not; the recall-trained policy does not transfer. PASSED |
| 6 | LoCoMo and LongMemEval: evidence retention under a budget | Pipeline PASSED; QA accuracy not run (now feasible with the `openai` backend on a rented GPU) |
| 6c | Dense against lexical retrieval on LoCoMo | See §6c. PASSED |
| 7 | A real language model as task model and as controller | Smoke tests only; the task-model half is superseded by Experiments 8 and 9. PARTIAL |
| 8 | A real language model (qwen2.5:3b) as the task model | A small, well-chosen memory beats full context; the D1 ranking holds. PASSED (1 seed, 30 episodes) |
| 9 | …with Qwen2.5-7B as the task model, 100 episodes | Every D-series result holds; oracle-approx beats full context by +0.04–0.06; the remaining gap is archived-but-not-retrieved. PASSED |
| 10 | Diagnostics; a follow-the-clue search for two-hop questions | GRPO's signal was real but the needed fact was out of reach; the search lifts imitation from 0.851 to 0.917 at 2% (oracle 0.966). PASSED (scripted reader) |
| 11 | Follow-the-clue with a 7B reader; LoCoMo and LongMemEval QA | The 7B gain holds (0.832 → 0.889 at 2%). On real conversations the learned policies lose to simple rules: they barely retrieve. PASSED (negative on benchmarks) |
| 12 | Retrieval headroom: how much evidence search reaches (no reader) | Top 16 holds 0.12–0.16 more evidence than top 5; labels and BM25+bge fusion help LoCoMo, not LongMemEval. Gates for the pick-k head and better units passed. |
| 13 | LongMemEval (all 500, 5 folds) and LoCoMo under the reviewed protocol: retrieval floor, no EVICT, reasoning reader, pre-registered gates | The floor fixes retrieval. Learned controllers keep more evidence at 5% but convert little: one paired win, one loss, none after Holm, none on the frontier. This reader refuses more as prompts grow (peak near 3k tokens). NEGATIVE, with a mechanism |
| 14 | A token price for kept memory: labels counted in the budget, plug-in Bayes archiving at a pre-set price | No-go at the free gate: the priced rule never beats keep-last-0 + top 5 on evidence at equal memory; needed model AUC 0.62–0.74; a per-token price archives long turns, not lines. NEGATIVE (free) |
| 15 | A query-aware floor head: which 5 of 16 to retrieve, learned listwise | Free gate: +0.08 all-found evidence at half the tokens. Reader: +0.047 accuracy (0.513 vs 0.466) at 745 vs 1,554 tokens; two of three arms win; Holm-adjusted p 0.058. First learned row beyond the rule frontier; replicates over 3 training seeds on a second host (+0.044, seed SD 0.002). POSITIVE |
| 16 | Set-valued GRPO on the selection head with the reader's verdict as reward | Gate fails: +0.0125 (−0.005, +0.033) held-out over the imitation start; flat in 4 of 5 folds. Labels already a sufficient target for this head. NEGATIVE (pre-registered) |
| 18 | Second-family judge check; exploratory LoCoMo reader test | Granite agrees with Qwen on 97% (κ 0.95), similar error rates; ranking and the head's gain hold (+0.040, CI excludes 0). Exploratory: head transfers to LoCoMo, +0.087 (post hoc). CONFIRMS |
| 19 | Adaptive k (5, 8 or 16 of 32); reader test §19c | Adaptivity stopped at its time box (the ceiling is the finding). Reader: LME `fixed8` 0.545 vs §15 head 0.500 (+0.045, CI +0.011..+0.079) at 1,049 tokens, PASS; LoCoMo `fixed5` +0.023 (CI −0.001..+0.048), no pass. Exploratory: LME `fixed16` −0.057 vs `fixed8` (context rot on a controlled pair); LoCoMo `fixed16` 0.344 at 734 tokens beats FIFO 10% (0.295 at 3,087). $0.36. §19d confirmation (head B, second host): −0.026 (CI −0.066..+0.015), NOT CONFIRMED; the decomposition repeats (P(correct \| in view) −0.088); $0.10 |
| 20 | Write action: notes beside the top 8 (§20) | Gate FAILED for both note arms (containment vs `fixed8`: +0.028 retrieval-time notes, +0.043 stored session notes, bar +0.05; both CIs above 0); no reader stage. Session notes are the most token-efficient evidence (+0.043 for about 300 tokens vs +0.060 for 1,520 raw). Gate spend about $0.81 (four pods) |
| 21 | Sequential task, lossy memory (LoCoMo; free, scripted) | Oracle gate PASSES: oracle − best rule +0.177 (CI +0.126..+0.224) on old-evidence questions; cross-fitted learned notes reach +0.038 (AUC 0.74–0.84), leaving +0.139 for §22; unlimited-archive control gap +0.016. $0 |
| 22 | Credit for the note-writer (simulator, free) | Per-write hindsight credit beats uniform GRPO (MLP +0.015; linear +0.036); time-forward (label-free) ≈ uniform; counterfactual best RL (0.141). No RL variant beats the supervised writer (0.146); from the supervised init (§22b) RL makes it worse (−0.027). §22c (KL to init, update count chosen on an inner split): level with supervised, −0.001 (CI −0.012..+0.011), clause FAILS narrowly; RQ3 closed. $0 |
| 23 | Reader size ladder (Qwen 3B/7B/14B, Granite 2B/8B) | Claim 2 NON-INFERIOR: fixed8 at 7B minus FIFO at 14B +0.034 (CI −0.013..+0.079), closes 150% of the size gap. Claim 1 NOT SHOWN: fixed8 at 3B minus FIFO at 7B −0.121, closes 56%. Granite secondary not shown. $0.83 |
| 24 | Reader families (Llama, Gemma 3, Phi-4; two sizes each) | Lift (a) PASSES in all six readers (+0.06 to +0.16). Size step (b): Gemma 3 4B→12B NON-INFERIOR (+0.015, CI −0.028..+0.057; closes 111%); Llama −0.051 and Phi-4 −0.117 NOT SHOWN. "Smaller gains most" holds in Llama only. $1.44 |
| 25 | StreamMemBench zero-shot transfer of the §19 head (free close) | Closed without a paid run at the user's decision; $0. Evidence in view on 541 evaluation anchors: FIFO 0.93 / 0.91 / 1.00 (initial / revised / follow-up), `fixed8` 0.32 / 0.23 / 0.10: the head does not transfer, because recency wins on this benchmark |
| 26 | OpenJev (open Jev re-creation) as a controller | Arm B (selector, 8 of the head's 32 candidates): B − FIFO +0.045 (CI +0.004..+0.087), BETTER; B − `fixed8` −0.036 (CI −0.081..+0.009), not shown; B's prompts 2,373 tokens vs 1,049. Arm A (memory manager) cost only: ~94 GPU-s per question (~$15 per 1,000); accuracy not run by decision. $0.64 |
| 27 | LRE (published query-blind scorer) as a baseline controller | LRE-slot (8 of the head's 32): 0.109, far below FIFO (−0.328) and the head (−0.428), evidence in view 0.03. LRE-native (in place of recency): +0.028 vs FIFO (CI −0.002..+0.057), non-inferior. About $0.23 shared with §28 |
| 28 | Query rewriting for the head's 32 candidates | Rule time filter (b): NO GO at its free gate. LLM rewrite (a): training gate passed at v3 (v1 malformed, v2 invented ranges); test folds +0.006 overall (CI −0.017..+0.030) and +0.024 on temporal (n = 127, CI −0.016..+0.063), non-inferior, not better. About $0.23 shared with §27 |
| 29 | A reader-utility training signal for the §19 head | Utility head − `fixed8` −0.013 (CI −0.051..+0.026), blend − `fixed8` −0.002: both UNDETERMINED; both beat FIFO + floor (+0.09, +0.10). The utility head picks longer turns. Closed as a negative (§29b, branch `reader-utility`). About $1.02 ($0.87 partial run + half of the shared pod) |
| 30 | UtilMem: the §19 head zero-shot on spread-out evidence among look-alike distractors | Closed without a pod run at the user's decision (2026-10-10); the free part (adapter, checks, stub run, price) is the record. $0 |
| 31 | Test-time training (qTTT) on the 3B reader: claim 1 of §23 without LoRA | Closed without a pod run at the user's decision (2026-10-10): little gain in a third-party re-creation at small scale, and our prompts are far shorter than qTTT's target. Free part is the record. $0 |
| 32 | The §19 head against standard rerankers of the same 32 candidates | Reader: head − logistic regression (31 parameters) +0.013 (CI −0.017..+0.043), head − zero-shot cross-encoder +0.013 (CI −0.030..+0.053): NON-INFERIOR, not better; only BM25 order is clearly worse (+0.068). `fixed8` is 3 distinct heads, not 5. About $0.15 (half of the shared pod) |

**What the false-abstention work (§33–§57) inherited from this era.**
- **Protocol (§13, D-05).** LongMemEval-S, all 500 questions in 5 stratified folds; folds 0–1 are development and
  folds 2–4 confirmation. The reader prompt is frozen at `7b5fc30`: a brief note of at most 80 words, then
  `Answer: <short phrase>`, with "unknown" as the abstention reply; 256 output tokens, greedy.
- **Memory view (§19).** The `fixed8` arm: 8 turns chosen by the reader-free §19 head from 32 BM25 candidates. The
  controller does not depend on the reader, so every reader sees byte-identical prompts.
- **Readers and caches (§23, §24).** Qwen 2.5 3B/7B/14B and Granite 2B/8B (`runs/exp23_*`,
  `cache/generations_exp23_*`); Llama 3.2 3B / 3.1 8B, Gemma 3 4B/12B and Phi-4-mini / Phi-4 (`runs/exp24_*`,
  `cache/generations_exp24_*`; since 2026-11-11 all in tarball K01). `memctl.jacobian_data` rebuilds every prompt
  from these.
- **Results the thesis deck still cites (STATE F1, F2).** `fixed8` − FIFO + floor ≈ +0.10 to +0.11 accuracy on
  LongMemEval (470 answerable questions) at about 1,049 against about 2,960 prompt tokens: §19c +0.113 (+0.072,
  +0.155), §23 +0.102. The lift has a 95% interval above 0 in 10 of 11 readers (§23–§24; Granite 2B the exception).
  Sources: §19c `runs/exp19_paid*` and `runs/exp19_paid_report.md` (in `archive/memctl-era/memctl-era-data.tar.zst`); §23/§24
  `runs/exp23_*`, `runs/exp24_*` and the two verdict files (since 2026-11-11 in `archive/false-abstention/kept/K01`); full text at the tag.
- **Judge labels.** `runs/exp23_verdicts.json` and `runs/exp24_verdicts.json`: Qwen2.5-7B with LongMemEval's official
  judge prompts, one judge for every reader; a second-family check (Granite 8B) agreed on 97% (§18).

## 33–57. False abstention: the groundwork (condensed, 2026-10-12 to 2026-11-11)

**What was asked.** Why a small reader (mostly Qwen 2.5 3B) answers "unknown", or answers wrong, when the evidence it
needs is in view, and whether an intervention inside the model can fix that selectively.

**What was found.**
- The answer is usually there. When a small reader fails with the evidence in view, the gold answer's first token is
  in its final-layer top 10 on most failures, in four model families (§34, §43), and on controlled tasks the model's
  own note usually names the gold before it refuses (§49, §54).
- The refusal is a late decision at the answer position (§34, §39, §45), and offering the "unknown" option causes
  much of it (§55).
- Many interventions release it: a prompt line (§35), a content-free steering direction (§38–§40), a decoded target
  (§49), an agent-proposed weight edit (§51), zeroing two attention heads (§53), resetting 50 neurons (§57). Each
  beats its random control. None releases false "unknown"s more selectively than a plain logit bias against the
  "unknown" token (§51, §53, §57).
- The push toward "unknown" is distributed: no small set of neurons carries it, and on the direct path it comes
  through attention (§56, §57).
- In-view refusal is strongest in the Qwen 3B. It is low in Llama 3B, Gemma 4B and Phi-4-mini under the same prompt
  (§43) and not measurable below 3B (§52).

**Why it closed (D-86, 2026-11-11).** The user judged the result not enough for a paper. The line stays as thesis
groundwork. What carries forward is the localisation toolkit and two controls every later mechanism claim must pass:
random component sets of the same shape, and a logit bias at matched effect.

**Each experiment.** "Dev" means development data only. Costs are RunPod unless marked. There is no §48.

| § | Question | Result | STATE | Cost |
|---|---|---|---|---|
| 33 | Where does the gradient of the answer land in the prompt (3B, 7B)? | STOPPED at its regeneration sanity check; no gradient computed | – | share of §34 |
| 34 | Where does the 3B lose the answer? J-lens readout on the §23 in-view questions | CONFIRMED on folds 2–4: the answer is present and "unknown" wins late | F11 | ≈ $1.03 |
| 34b | Does the model's own answer position change the picture? | Both pre-registered claims NOT SHOWN on the pooled rows; a presence gain on marked outputs is exploratory | – | ≈ $0.18 |
| 34c | Does presence depend on the evidence? | Both claims SHOWN; the plausibility share is about 0.6 among the 3B's failures (the answer reads as present there even with the evidence removed); the 3B's "unknown" signal is insensitive to evidence | – | ≈ $0.11 |
| 35 | Does the 3B derive the answer in its note and still refuse? | Gate PASS, Claims 1–2 SHOWN. Frozen note plus "do not reply unknown": +0.176 (+0.109, +0.242), 0.53 of the matched 3B−7B gap; forced choice +0.242. Cost: 0.210 of marked refusals become wrong answers | F6 | ≈ $0.22 |
| 36, 36b | Can the 3B tell which of its forced answers the evidence supports? | Run 1 not reliable. §36b Claim 1 NOT SHOWN: AUROC 0.573 (0.478, 0.666); the score detects that evidence is present, not that the answer matches it | F6 | $0.33 |
| 37 | Can a three-turn cross-examination make it check? | CLOSED, dev: gate FAIL, AUROC 0.505 (0.431, 0.574) | F6 | $0 |
| 38 | Does one content-free direction open the refusal? | Primary setting (layer 25, α 4) NOT INTERPRETABLE. Disclosed secondary (layer 29, α 2) Claim 1 SHOWN: +0.333 (+0.183, +0.483) | F6 | ≈ $0.13 |
| 39, 39b | Is the refusal decided at one position, by one token? | Dev: a push at ':' alone equals a push at every token (26 of 28 answers each); forcing the pushed run's first token reproduces it (26 of 28) | F6 | $0 |
| 40 | Does the fixed push transfer to LoCoMo? | Claim 1 SHOWN: +0.333 (+0.250, +0.425). Costs: 8 of 40 correct answers lost; answers on unanswerable questions 0.32 → 0.73 | F6 | ≈ $0.15 |
| 41 | Can a signal at ':' say when to push? | Claims 1–3 NOT SHOWN; Claim 4 SHOWN: the projection ranks which opened refusals come out right, AUROC 0.803 (0.724, 0.875) | F6 | $0 |
| 42 | Does a threshold on that signal give a selective policy on fresh LoCoMo rows? | Claims 1–3 NOT SHOWN (utility: conditional −6, always-push −6, baseline 0, prompt line +1). The ranking replicates (0.765) but ranks extractability, not answerability. §40 Claim 1 replicates smaller, +0.233 (0.150, 0.317) | F6 | ≈ $0.12 |
| 43 | Does the picture hold in Llama 3.2, Gemma 3, Phi-4? | Part B: gold-present on failures in all four families; larger-sibling claim SHOWN for Llama and Phi, not Gemma. In-view refusal rate: Qwen 0.30, Llama 0.10, Gemma 0.02, Phi 0.03. Part A closed for Gemma and Phi, deferred for Llama | F11 | ≈ $2.11 |
| 44 | Is the answer at the answer position put there by the evidence? Patching and an evidence swap | Causal depth 0.750 (3B) and 0.679 (7B), claims C1 and C3 undetermined. The swap failed validity twice on Qwen, so evidence-use and answer-production failures were not told apart | F12 | ≈ $0.16 |
| 45 | What do the lenses show, layer by layer and token by token, on controlled tasks? | Descriptive: the answer is readable only at layers 30–33 of 36, at the position that produces it; the pre-registered failure sentence was not earned | F13 | ≈ $1.21 |
| 46 | Is an intermediate present where the lenses cannot see it? | The supervisor-name residual sets the concluded city through layer 15; the probe arm is NOT ESTABLISHED (dev confound) | F14 | ≈ $0.24 |
| 47 | Does the evidence's value reach the answer computation on in-view failures? | Exploratory, selected rows: the value is carried at its evidence tokens through layer 30, alike on 13 failures and 13 successes | F15 | $0.34 |
| 49 | Does a target decoded from the failing model release correct answers? | 152 of 165 false abstentions name the gold in the note. Decoded push − random +0.836 (+0.774, +0.894), but not selective and no better than the content-free push | F16 | $0.85 |
| 50 | What separates a false "unknown" after an answer-bearing note from an answer? | The elicited sufficiency measure is invalid in every task. The gold margin is lower on false abstentions (DE −1.21, RC −3.13) but stays large. Stage 2 not opened | F17 | $1.32 |
| 51 | Can an agent's weight edit repair it selectively, on sealed cases? | Repair SHOWN on held-out H: 212 of 261 released, unsupported answers 0.035 → 0.042. Better than the prompt line, the §38 push and orthogonalisation; against the logit bias inconclusive (Holm p 0.092) | F18 | editor API $3.58; pods ≈ $6.95 |
| 52 | Do readers below 3B refuse with the evidence in view? | Dev: gate FAIL for all six readers; the habit is not measurable there | F21 | $0 |
| 53 | Are the two §51 edits specific? Random-edit controls and real questions | C1, C2, C4 and C5 each SHOWN against 20 random edits. Descriptive: the logit bias scores 0.406 against 0.403 and 0.414 for the edits | F19 | ≈ $2.59 |
| 54 | Is there a native switch behind false "unknown"? Twin-pair patching | STOPPED at the stage-1 gate (21 pairs against a bar of 40). Note audit: the gold is in the model's own note on 255 of 288 false abstentions | F20 | $0 |
| 55 | How much survives a clean, consistent abstention prompt? | A consistent rule raises false abstention (0.311 → 0.380). Removing the option removes it, but the model then answers 0.653 of unanswerable rows | F22 | $0.59 |
| 56 | Which MLP neurons carry false "unknown" at the answer position? | Dev, STOPPED at stage 1: distributed; half the push needs 14,966 of 396,288 neurons. Descriptive: the direct push comes through attention (+14.5), not MLPs (−7.7) | F23 | $0 |
| 57 | Which neurons differ on false "unknown", and does resetting them fix it? | Dev, CLOSED after stage 3: resetting the top 50 releases 0.58 (answer position) and 0.83 (note) against at most 0.03 for random sets; neither beats the logit bias, and the note set's effect is an off-distribution push | F24 | $0 |

Programme spend to 2026-11-11: about $34.32 on RunPod against a $36 cap (STATE.md, Spend).

## 58 onward. Self-doubt (live)

No § entry yet. The thesis now asks which layers, attention heads and neurons cause self-doubt in reasoning models:
a model that receives a valid update in the middle of its thinking and still answers the old problem (Wu, Miroyan et
al., arXiv 2510.11713). Stages 0–2 were run on 2026-11-09 and 2026-11-10 as a separate project. Their
pre-registration and results sections are in [selfdoubt/PROPOSAL.md](selfdoubt/PROPOSAL.md), their summaries in
`selfdoubt/results/`, and their spend in [selfdoubt/SPEND.md](selfdoubt/SPEND.md).

New experiments take the next § number, with the self-doubt stage named in the title. Each entry states the question,
the pre-registration commit, the command, the frozen inputs with their sha256, the result, what it does and does not
show, and the cost. The conventions are in [REPRODUCE.md](REPRODUCE.md).
