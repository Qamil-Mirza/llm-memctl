# Experiments

The record of the honors thesis *A Mechanistic Investigation of Evidence-Use Failures in Language Models*: every
experiment, the command or script that reproduces it, the result, and what it does and does not show. `runs/` is not
in git; each section names the commit and the files its numbers come from. Sections §33–§55 are in full. The earlier
memory-controller work (§1–§32) is condensed below; its full text is at the git tag `memctl-era-final`
(`git show memctl-era-final:EXPERIMENTS.md`), and its run data is in the local archive
`archive/memctl-era-data.tar.zst` (not in git).

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

**What the mechanistic work (§33–§55) inherits from this era.**
- **Protocol (§13, D-05).** LongMemEval-S, all 500 questions in 5 stratified folds; folds 0–1 are development and
  folds 2–4 confirmation. The reader prompt is frozen at `7b5fc30`: a brief note of at most 80 words, then
  `Answer: <short phrase>`, with "unknown" as the abstention reply; 256 output tokens, greedy.
- **Memory view (§19).** The `fixed8` arm: 8 turns chosen by the reader-free §19 head from 32 BM25 candidates. The
  controller does not depend on the reader, so every reader sees byte-identical prompts.
- **Readers and caches (§23, §24).** Qwen 2.5 3B/7B/14B and Granite 2B/8B (`runs/exp23_*`,
  `cache/generations_exp23_*`); Llama 3.2 3B / 3.1 8B, Gemma 3 4B/12B and Phi-4-mini / Phi-4 (`runs/exp24_*`,
  `cache/generations_exp24_*`). `memctl.jacobian_data` rebuilds every prompt from these.
- **Results the thesis deck still cites (STATE F1, F2).** `fixed8` − FIFO + floor ≈ +0.10 to +0.11 accuracy on
  LongMemEval (470 answerable questions) at about 1,049 against about 2,960 prompt tokens: §19c +0.113 (+0.072,
  +0.155), §23 +0.102. The lift has a 95% interval above 0 in 10 of 11 readers (§23–§24; Granite 2B the exception).
  Sources: §19c `runs/exp19_paid*` and `runs/exp19_paid_report.md` (in `archive/memctl-era-data.tar.zst`); §23/§24
  `runs/exp23_*`, `runs/exp24_*` and the two verdict files (kept in `runs/`); full text at the tag.
- **Judge labels.** `runs/exp23_verdicts.json` and `runs/exp24_verdicts.json`: Qwen2.5-7B with LongMemEval's official
  judge prompts, one judge for every reader; a second-family check (Granite 8B) agreed on 97% (§18).

---


## 33. Input-gradient attribution on the 3B and 7B readers: where does the gradient of the answer land in the prompt? (2026-10-12, pre-registered in the spec; pod run 2026-10-13 stopped by its own sanity check)

**Status: STOPPED at the regeneration sanity check; no gradient was computed.** Secondary to §34 (the J-lens
readout), which ran to completion on the same pod. Spec: `docs/superpowers/specs/2026-10-12-jacobian-lens-design.md`;
code `scripts/jacobian_lens.py`, `memctl/analysis/jacobian.py`; pod script `scripts/exp33_pod.sh`.

**Why.** §23's reader-side number (P(correct | all evidence in view) 0.397 at 3B against 0.667 at 7B) says the 3B
ignores evidence that is in front of it. §33 was to measure, for every cached §23 prompt, how much of the gradient of
the teacher-forced answer (the gold's first content token, the margin to ` unknown`, and the reader's own output) lands
on the evidence lines against the template, instruction, distractor and question regions, and whether the share rises
with the reader size; with an exact-span evidence ablation as the causal check.

**The sanity check and why it stops the run (spec §33, "regeneration sanity check").** The targets are teacher-forced
on the cached vLLM outputs, so the transformers model must reproduce those outputs: 20 prompts per model are
regenerated greedily (bf16, 256 new tokens) and at least 15 must match the cached text exactly; fewer stops the run.

**What happened (pod 5eiga0da36l6o9, A40, 2026-10-13 ≈ 04:55–05:00 UTC).** Qwen2.5-3B-Instruct in bf16 reproduced
**10 of 20** cached outputs exactly, so the run stopped before any gradient. The pairs are in
`runs/exp33_jacobian/qwen3b/_regen_check.json`; summary (recomputed from that file):

| question | cached chars | exact match | common prefix chars | same final answer |
|---|---|---|---|---|
| 00ca467f | 231 | no | 139 | yes |
| 06db6396 | 79 | yes | 79 | yes |
| 07b6f563 | 205 | no | 191 | no |
| 0977f2af | 93 | no | 48 | yes |
| 0a995998 | 166 | no | 156 | yes |
| 0bb5a684 | 248 | yes | 248 | yes |
| 0db4c65d | 326 | yes | 326 | yes |
| 15745da0 | 90 | yes | 90 | yes |
| 15745da0_abs | 97 | yes | 97 | yes |
| 1a1907b4 | 93 | no | 56 | yes |
| 1a8a66a6 | 303 | yes | 303 | yes |
| 1b9b7252 | 150 | yes | 150 | yes |
| 21d02d0d | 327 | no | 65 | no |
| 22d2cb42 | 132 | no | 98 | yes |
| 2698e78f | 456 | yes | 456 | yes |
| 2698e78f_abs | 134 | no | 43 | yes |
| 28dc39ac | 845 | no | 64 | no |
| 37f165cf | 23 | yes | 23 | yes |
| 3a704032 | 257 | yes | 257 | yes |
| 3fe836c9 | 170 | no | 66 | no |

Exact match 10/20 (15 required); the final answer line (text after the last `Answer:`) agrees on 16/20. Every
mismatch is a divergence part-way through a chain-of-thought output (93–845 characters; common prefix 43–191
characters), consistent with, but not tested as, a kernel difference between bf16 transformers and vLLM on greedy
generation: once one token differs, the rest of the text differs.

**What this means for §33.** The spec's check is doing its job: teacher-forcing a gradient target on text the model
would not itself produce measures a hypothetical. Two honest ways forward, neither taken here because each is a design
change after seeing data: (i) teacher-force only up to the first content token of the gold (the P1 target of §34,
which does not depend on the cached chain of thought) and drop the "own output" target; (ii) regenerate the outputs
with transformers and use those as the targets, which breaks the identity with the §23 answers the judge scored. Both
need a new pod (about 30–45 min on an A40, under $0.50) and the user's approval. The 7B was never loaded for §33.

**Spend.** None beyond the §34 pod (the §33 stage took about five minutes of it).


## 34. Where does the 3B lose the answer? A Jacobian-lens readout of the Qwen 2.5 3B and 7B readers on the §23 in-view questions (2026-10-12, pre-registered; run 2026-10-13)

**Status: RUN and CONFIRMED on folds 2, 3, 4.** Spec `docs/superpowers/specs/2026-10-12-jlens-readers-design.md`
(d094acb); pre-registration `research/proposals/2026-10-13-exp34-jlens.md` (be32b33, reviewer ACCEPT at b42ba90,
`research/reviews/2026-10-13-exp34-prereg.md`); code `memctl/analysis/jlens_readout.py`, `scripts/jlens_fit.py`,
`scripts/jlens_read.py`, `scripts/jlens_eval.py`, `scripts/exp34_report.py`, `scripts/exp34_pod.sh`; lens library
`external/jacobian-lens` at 581d398 (Anthropic's J-lens, `jlens` 0.1.0, Apache-2.0). Reports:
`runs/exp34_report_confirm.md` (folds 2–4, the one quoted below), `runs/exp34_report_allfolds.md` (all five folds),
`runs/exp34_report_dev.md` (folds 0–1); copies under `docs/research/`.

**Why.** §23 found that with all evidence in view under `fixed8` the 3B answers correctly on 0.397 of questions and the
7B on 0.667, and that the 3B says "unknown" on 0.372 of answerable questions. The controller does its job; the reader
does not read. §34 asks *where* inside the 3B the gold answer gets lost, using a probe that reads, at every layer, which
token the residual stream is disposed to make the model say.

**The instrument.** Anthropic's Jacobian lens: `lens_l(h) = unembed(J_l · h)` with `J_l` the expected Jacobian of the
final residual with respect to layer `l`'s residual, averaged over 56 WikiText-103 paragraphs (target 128; the 7B's
90-minute box stopped at 56 and the 3B was then fitted on the same 56, so the two lenses average identical prompts).
The logit lens `unembed(h)` is the baseline and the fallback. Every cached §23 prompt (`fixed8`, fraction 0.05) is
rebuilt byte-exactly from the cache (sha256 identity checked per question) and run once through the reader; at the
position **P1**, the token right before the gold's first content token after `Answer:` (leading whitespace tokens
teacher-forced), the rank of the gold's first content token is read at every layer with both lenses. P1 is read on the
prompt with `Answer:` appended directly, with no reasoning text in between (P1 = P0 + 2 in every read file); 200 of the
204 cached 3B outputs on the confirmation folds reason first and write `Answer:` later, so every P1 figure describes a
**forced immediate answer**, not the position at which the cached answer was produced. With k = 10 each
question is **A** (gold never in the top-10 at any layer), **B** (gold in the top-10 at some layer, then out at the
final layer) or **C** (gold in the final-layer top-10). Positions P0
(last prompt token), E (end of each evidence line) and Q (end of the question) are read too. Population: answerable
questions with all evidence in view, present for both readers with a §23 judge label, paired by prompt hash.

**The lens gate (spec §3), fixed on the development folds 0–1 before the confirmation folds were read.** A reader's
J-lens carries its claims only if (a) it ranks the gold better than the logit lens over the middle third of the layers
(paired log10 best rank, lower bound above 0) and (b) it beats the logit lens on the lens library's six evaluation sets
(mean pass@10 over the band, strictly). On 126 development questions:

| reader | lens prompts | band | (a) logit − J-lens (95% CI) | (a) | (b) mean pass@10 J-lens / logit | (b) | gate |
|---|---|---|---|---|---|---|---|
| qwen3b | 56 | [12, 24) | +1.219 (+0.907, +1.531) | PASS | 0.014 / 0.039 | FAIL | **FAIL** → the `logit (gate fallback)` rows decide |
| qwen7b | 56 | [9, 18) | +1.685 (+1.476, +1.896) | PASS | 0.022 / 0.018 | PASS | **PASS** on (a); (b) is noise-sized (0.004, about two items of 550) |

Both J-lenses beat the logit lens on gate (a): the geometric-mean best rank over the middle band is about 17× (3B) and
48× (7B) better, development folds. The 3B lens fails the library's own `typo` set (0.052 against 0.229) and so
fails the gate as pre-registered; its claims are decided on the logit-lens rows, with the J-lens rows beside them. The
7B's logit rows are shown beside its J-lens rows (`logit (shown beside)`, no verdict role) because its gate rests on
(a) alone. Fifty-six prompts is a weak lens by the library's standard (the community 7B lens uses 100); the
cross-check against that community lens is in the descriptives.

**Prediction, recorded in the spec before any data.** "The 3B's refusals are mostly B: the gold rises into the top-k in
the middle third and ` unknown` overtakes it in the last third. The 7B shows C on the same questions. The 3B's wrong
answers look more like A. Emergence is at a similar fractional depth in both readers." **Falsified**, see below.

### 34a. Results (confirmation folds 2, 3, 4; 204 paired questions; all-folds rows, 330 questions, in brackets)

**Claim 1 (size), SHOWN.** At P1 the 7B ranks the gold higher than the 3B, paired by question. On the deciding
logit-lens row the 3B's log10 best rank is worse by +0.486 (+0.376, +0.600) [all folds +0.473 (+0.388, +0.560)], about
3× in rank; the 3B's share of outcome A exceeds the 7B's by +0.201 (+0.137, +0.265) [+0.188 (+0.142, +0.236)]. The
J-lens rows agree: +0.535 (+0.435, +0.641) and +0.255 (+0.191, +0.319).

| measure | lens | n | 3B minus 7B (95% CI) | verdict |
|---|---|---|---|---|
| log10 best rank of gold | J-lens | 204 | +0.535 (+0.435, +0.641) | SHOWN |
| share of outcome A (secondary) | J-lens | 204 | +0.255 (+0.191, +0.319) | SHOWN |
| log10 best rank of gold | logit (gate fallback), **decides** | 204 | +0.486 (+0.376, +0.600) | SHOWN |
| share of outcome A (secondary) | logit (gate fallback), **decides** | 204 | +0.201 (+0.137, +0.265) | SHOWN |

**Claim 2 (within the 3B), SHOWN, clear of the noise edge.** The 3B's gold rank at P1 is worse on the questions it
gets wrong than on those it gets right: AUC 0.654 (0.576, 0.727) on the deciding logit row [0.632 (0.571, 0.689)];
0.629 (0.547, 0.707) with the J-lens. On the development folds the lower bounds had straddled 0.5 (0.499 J-lens,
0.502 logit); the confirmation folds settle it. The 7B's descriptive row is 0.646 (0.571, 0.721) J-lens, 0.692 (0.617,
0.766) logit.

| reader | lens | n wrong | n right | AUC (95% CI) | mean log10 rank, wrong | right | verdict |
|---|---|---|---|---|---|---|---|
| qwen3b | J-lens | 127 | 77 | 0.629 (0.547, 0.707) | 0.897 | 0.603 | SHOWN |
| qwen3b | logit (gate fallback), **decides** | 127 | 77 | 0.654 (0.576, 0.727) | 0.915 | 0.495 | SHOWN |
| qwen7b | J-lens | 63 | 141 | 0.646 (0.571, 0.721) | 0.406 | 0.182 | descriptive |
| qwen7b | logit (shown beside) | 63 | 141 | 0.692 (0.617, 0.766) | 0.512 | 0.162 | descriptive |

**The decomposition (the interpretive result; no directional claim).** Among the 3B's 127 in-view failures, on the
deciding logit row: **A 0.346 (0.268, 0.433), B 0.016 (0.000, 0.039), C 0.638 (0.551, 0.717)**. Split: refusals (65)
A 0.400, B 0.031, C 0.569; answered wrong (62) A 0.290, B 0.000, C 0.710. The J-lens row gives the same picture (A
0.362, B 0.000, C 0.638). The 7B's 63 failures: A 0.127, B 0.127, C 0.746 with the J-lens; A 0.222, B 0.032, C 0.746
with the logit lens (the 7B's B share depends on the lens and is not quoted as a property of the model).

| reader | lens | class | n | A (95% CI) | B (95% CI) | C (95% CI) |
|---|---|---|---|---|---|---|
| qwen3b | logit (gate fallback) | refusal | 65 | 0.400 (0.277, 0.523) | 0.031 (0.000, 0.077) | 0.569 (0.446, 0.692) |
| qwen3b | logit (gate fallback) | answered wrong | 62 | 0.290 (0.177, 0.403) | 0.000 (0.000, 0.000) | 0.710 (0.597, 0.823) |
| qwen3b | logit (gate fallback) | all failures | 127 | 0.346 (0.268, 0.433) | 0.016 (0.000, 0.039) | 0.638 (0.551, 0.717) |
| qwen3b | logit (gate fallback) | success | 77 | 0.221 (0.130, 0.312) | 0.052 (0.013, 0.104) | 0.727 (0.623, 0.818) |
| qwen3b | J-lens | all failures | 127 | 0.362 (0.283, 0.449) | 0.000 (0.000, 0.000) | 0.638 (0.551, 0.717) |
| qwen3b | J-lens | success | 77 | 0.260 (0.169, 0.364) | 0.013 (0.000, 0.039) | 0.727 (0.623, 0.818) |
| qwen7b | J-lens | all failures | 63 | 0.127 (0.048, 0.206) | 0.127 (0.048, 0.222) | 0.746 (0.635, 0.841) |
| qwen7b | logit (shown beside) | all failures | 63 | 0.222 (0.127, 0.333) | 0.032 (0.000, 0.079) | 0.746 (0.635, 0.841) |
| qwen7b | J-lens | success | 141 | 0.043 (0.014, 0.078) | 0.007 (0.000, 0.021) | 0.950 (0.915, 0.986) |

**Reading it.** The prediction was B; what the readout shows is **C, the most common outcome among failures and successes alike, with a floor of A**.
- **B is rare.** The "extract mid-stack, then override late" outcome is 0.016 (0.000, 0.039) of the 3B's failures on
  the deciding row [all folds 0.025 (0.005, 0.050)], and no more common than among its successes (0.052 (0.013, 0.104)).
- **C is the most common outcome among failures (0.638), as among successes (0.727):** at a forced immediate answer
  the gold's first token is in the final-layer top-10. J-lens descriptives, from the 3B lens that failed its gate: on
  47 of the 65 refusals the ` unknown` token reaches top-1 (mean first top-1 depth 0.90 of the stack); the 3B's median
  best rank of the gold at P1 over layers is 3 and the 7B's is 1 (share of questions with best rank ≤ 10: 0.676
  against 0.931). No report row gives the gold's final-layer rank on refusals.
- **A has a floor among successes, cause untested.** A is 0.35 of failures, but 0.22–0.26 of the 3B's *successes*
  are also A (the 7B's success-A is 0.04). The hypothesis is paraphrase and tokenisation: the reviewer's check of the
  17 logit-A successes on folds 2–4 found 11 whose final answer does not start with the gold token's text (gold "four",
  answer "4") and 6 that do start with it case-insensitively and are still A ("IKEA" for the token "IKE", "doc martin"
  for "Doc"). The "never extracted" share net of the success floor is a rough bound that judge false negatives push
  upward. The clean signal is the paired Claim 1 gap and the success/failure contrast, not the raw A share.
- **Emergence is late in the 3B, not similar.** Where the gold enters the top-10 at all, the 3B's median fractional
  depth is 0.889 (q1 0.590, q3 0.944; 138 of 204 emerge) against 0.464 for the 7B (q1 0.250, q3 0.857; 190 of 204)
  [all folds 0.889 vs 0.464]. The 7B has the answer by mid-stack; the 3B's median is 0.889 (q1 0.590). Whether
  emergence and the ` unknown` rise happen in the same layers is not tested here (the ` unknown` depth is over 65
  refusals, the emergence depth over 138 emerged questions of all outcomes). This rests on the J-lens (the only lens
  with an emergence row) and, for the 3B, on a lens that failed its gate; it is descriptive.
- **The answer is not readable at the evidence or question positions, in either reader.** Median best rank of the gold
  over the evidence-line ends (E) is 398 for the 3B and 640 for the 7B; at the question end (Q) 394 and 330; at the
  last prompt token (P0) 209 and 316. The gold becomes sayable only once the model is asked to answer. This is the
  same for both readers and says nothing about size.
- **By question type (J-lens, all 204).** The 3B's A share is highest on single-session-preference (0.889 of 9) and
  knowledge-update (0.400 of 35), lowest on multi-session (0.194 of 36); the 7B is below 0.12 everywhere except the
  same preference set, where its outcome is mostly B (0.778 of 9): a small, odd cell.
- **Community cross-check (development folds only, 126 questions; no community reads exist on folds 2–4, so that row
  in the confirmation report is empty, not negative).** Against the public 100-prompt 7B lens, the top-10 overlap at
  P1 is 0.408 / 0.538 / 0.704 by layer third, and Claim 1 recomputed with the community lens is +0.465 (+0.353,
  +0.584), SHOWN. Our 56-prompt 7B lens and the public one agree where it matters.

**What this changes: a hypothesis for stage 2, not a result.** What is shown: Claims 1 and 2 say the gold is less
available at the forced answer position in the 3B than in the 7B, and less available on the 3B's failures than on its
successes. What is not shown, because no intervention was run and no evidence was varied: that the 3B "extracts" the
answer, or that better-placed evidence would not help. The readout is *consistent with* a late loss at a forced answer
position (C in two-thirds of failures, B rare, late emergence, ` unknown` reaching top-1 in the last layers), which
would point at the reader's final layers (the refusal rule) rather than at what the controller puts in view; that is
the hypothesis the spec's stage 2 was written to test (add the lens-transported gold direction or suppress the
` unknown` direction at the emergence layer). **Stage 2 is not opened by this run**, because its pre-registered gate
requires both readers' lenses to pass and the 3B's did not. A stage-2 spec would need a better 3B lens first (more
prompts, or the logit lens as the instrument, since the logit rows carry the same picture), and a P1 that matches the
model's own reasoning-then-answer format.

**Caveats.**
- Labels are the §23 single-judge verdicts (STATE.md caveat 3). Judge errors are false negatives, about 0.1 of
  answers (about 20 of the 204 here, roughly 0.16 of the 127 "failures"): correct answers counted as wrong or as
  refusals. They dilute Claim 2, and they carry the success profile (C 0.727, A 0.221) into the failure rows, so they
  inflate C and deflate A among failures; the size of C among failures is an upper bound.
- P1 is a forced immediate answer (prompt + `Answer:`), while 200 of 204 cached 3B outputs reason before answering;
  the readout does not describe the position at which those answers were produced.
- The 3B J-lens failed its gate; everything with a verdict is decided on the logit lens, and the J-lens rows agree.
  The emergence-depth, position, ` unknown`-trajectory and question-type tables exist only for the J-lens and are
  descriptive.
- Both lenses average 56 prompts, not the 128 planned (7B time box). The 7B gate (b) margin is noise-sized; gate (a)
  carries it. Lens hashes are recorded in every read and eval file.
- The A share carries a floor among successes (0.22–0.26, cause untested, described above); P1 tracks the gold's first content token only.
- The 7B's B share is lens-dependent (0.127 J-lens, 0.032 logit) and is not a finding.
- Process: spec §8 asked for the reviewer's signature before the paid run; it came after (the `--confirm` guard and
  dev-only pod logs protected the confirmation folds, as the reviewer verified). Stated in the pre-registration.
- The spec's descriptive "§33 evidence share against the §34 outcome" is dropped: §33 stopped at its sanity check.
- **Reviewer questions of 2026-10-13, with what the data say (audit script `scripts/exp34_alignment_audit.py`).**
  - *Right token, right position?* At the 3B's own answer position (§34b), on the 54 judged-correct outputs with an
    `Answer:` marker, 40 begin with the tracked gold token and the gold's final-layer rank is 1 on 38 of them and 2 on
    the other 2 (our greedy pass and the cached sample differ by one place); the 14 that do not begin with it are
    paraphrases ("4" for "four", a sentence for a name), 8 at rank 2–10 and 6 beyond 10. No case begins with the gold
    token and ranks it below 2. The 7B: 103 of 141 begin with it, all at rank 1. The token and the position are
    aligned; "success with no gold in the top-10" is the paraphrase floor, not a bug. At the forced position rank 1 is
    not expected even for correct answers, because it is not the context the cached answer was produced in (§34b).
  - *Does "in the top-10" mean extracted from the evidence, or merely plausible or already known?* Not tested here. A
    counting question makes "2" plausible without any reading. These are personal facts from the user's conversations,
    not world knowledge, so recall of the answer itself is unlikely, but plausibility of small numbers and common words
    is live. §34c (pre-registered, `research/proposals/2026-10-13-exp34c-evidence-ablation.md`) reads the same questions
    with the evidence lines cut out to measure it; until it runs, outcome C is "present", not "extracted".
  - *Is late emergence a lens artefact?* Depth is descriptive only and carries no verdict; it likely has a
    token-frequency component (the 7B's failures emerge *earlier* than its successes). Outcome C uses the final layer,
    the model's own logits, so the C share does not depend on any lens; only the A/B split does.
  - *Erroneous refusal or sensible abstention?* On the 65 questions the 3B refused (folds 2–4), the 7B answered 40
    correctly, refused 6 and got 19 wrong. About 0.6 are answerable by a model one size up; the remaining 25 are spread
    across types (temporal 6, multi-session 5, knowledge-update 5, preference 5, other 4) and may be hard or arguable.
    The claim is "refusals on questions a 7B answers correctly", not "on easy questions"; a human pass on those 25
    would settle the remainder.
  - *Size, or something else?* This is a two-model comparison inside one family (Qwen2.5-Instruct, same tokenizer and,
    as far as is public, the same recipe scaled), not a scaling law. A third point (the 14B, already at 0.77 in-view
    accuracy in §23) would make it a trend; it costs a lens fit, about two hours on an A40. §34c adds that Claim 1's
    rank gap is the same with the evidence cut out (+0.473 vs +0.486), so it is a difference in how the two readers
    rank the gold at the answer position, not in how much they extract from the evidence.

**Run and spend.** One A40 secure pod (CA-MTL-1, $0.59/h, `runpod/pytorch` 2.8 image), 2026-10-13 02:31–07:31 UTC:
install 29 s, 7B lens 5,419 s (56 prompts), 3B lens 1,931 s, reads 277 s + 297 s (499 of 500 questions per reader; one
read failed per reader on fold 2), evals 84 s + 101 s, community cross-check 130 s, §33 stage about 5 min; the chain
ended at about 05:00 UTC and the pod then sat idle about 2.5 h because the monitor's process check matched its own
command line (the same self-match bug as Exp 26). 5.0 h × $0.59 ≈ $2.95, of which about $1.47 idle. All experiments
≈ $16.70. Earlier on 2026-10-12/13 six pods were created and deleted within a minute of starting by a compute broker
from another project running on the same laptop against the same RunPod account; the billing API shows $0.034 for the
six together (none lived over 41 s); the broker is now disabled. The billing API shows $2.64 for the main pod at the time
of writing against $2.95 by uptime × price (billing lags).

### 34b. Is the late, often absent emergence an artefact of the forced answer position? Not as pre-registered; partly, on presence, among outputs that have an answer marker (exploratory). (pre-registered 2026-10-13, run and confirmed the same day)

**Status: RUN and CONFIRMED on folds 2, 3, 4; both pre-registered claims NOT SHOWN on the pooled population; an exploratory split by answer marker shows a presence gain on the marked outputs.**
Pre-registration `research/proposals/2026-10-13-exp34b-own-position.md` (59d8b0d, before the pod; not reviewed before
the run, the reviewer signs the result section); code `--position-mode own` in `scripts/jlens_read.py` (8ed915f),
`scripts/exp34b_pod.sh`, `scripts/exp34b_report.py` (14a958d). Reports `runs/exp34b_report_{dev,confirm,allfolds}.md`,
copies `docs/research/exp34b_own_position_report_*.md`. Reads `runs/exp34_jlens_own/<reader>/` (499 of 500 per reader,
one failed read each on fold 2, as in §34).

**Why.** §34a reads every question at a forced immediate answer (`Answer:` right after the prompt), but the cached 3B
outputs write a note first and answer later. The exploratory analysis 1 (`scripts/exp34_depth.py`, ledger) had shown
that within the 3B emergence depth is the same on failures and successes (both late) and that what separates failures
is whether the gold ever enters the top-10. If forcing was the cause of the lateness and the absence, reading at the
reader's own answer position, after its own reasoning, should move both.

**What was read.** The same prompts, lenses, tracked tokens, labels and population as §34; the cached output is
teacher-forced up to and including its last `Answer:`, then the gold's leading-whitespace tokens, and P1 is read there.
Outputs without the marker (3B: 39 of 204 on the confirmation folds, 26 of 126 on the development folds; 7B: none) are
read at their first token and reported separately. P0, E and Q are unchanged, so each question pairs with its §34 read
by prompt hash. Verdict rows as in §34 (3B on the logit lens, J-lens beside; 7B on the J-lens, logit beside).

**Claims (3B, deciding logit row, folds 2–4, 204 questions; all folds, 330, in brackets).**

| claim | measure | own vs forced | verdict |
|---|---|---|---|
| A (presence) | share of questions where the gold ever enters the top-10: forced 0.701, own 0.711 | +0.010 (−0.049, +0.069) [+0.012 (−0.033, +0.058)] | **NOT SHOWN** |
| B (depth) | emergence depth among the 124 questions emerged in both modes, forced − own: medians 0.917 vs 0.917 | +0.015 (−0.001, +0.033) [+0.011 (−0.001, +0.024)] | **NOT SHOWN** |

The J-lens rows say the same or lean the other way: presence +0.000 (−0.059, +0.059); depth −0.163 (−0.209, −0.118),
that is *later* at the own position (medians 0.889 forced, 0.917 own). Development folds: presence +0.016 (−0.056,
+0.087), depth +0.005 (−0.014, +0.019), both NOT SHOWN, so the result did not move between development and
confirmation.

**Decomposition at the own position (3B, logit, folds 2–4).** Failures (127): A 0.323, B 0.047, C 0.630 against
0.346 / 0.016 / 0.638 forced; refusals (65): A 0.308, B 0.062, C 0.631 against 0.400 / 0.031 / 0.569; successes (77):
A 0.234, B 0.039, C 0.727 against 0.221 / 0.052 / 0.727. No class moves by more than 0.10 (the refusals' A share
moves most, 0.400 to 0.308); no intervals were computed on these cells, and they mix marker and marker-less rows. Refusal
order at the own position
(J-lens, 65): gold never enters the top-10 while ` unknown` reaches top-1 in 23 (0.35); gold enters and ` unknown`
never reaches top-1 in 20 (0.31); ` unknown` top-1 before the gold enters in 15 (0.23); same layer 5; neither 1; gold first 1.
Final-layer rank on refusals: gold median 4, ` unknown` 1. The same shape as at the forced position.

**The 7B (descriptive).** Presence 0.931 forced, 0.907 own (−0.025 (−0.059, +0.010)); J-lens depth medians 0.429
forced, 0.536 own (−0.020 (−0.042, +0.002)); its logit-lens depth is later at the own position by 0.057 (0.028, 0.091).
Reading after the reader's own reasoning does not make the gold earlier or more present in the 7B either; on its
failures the J-lens A share (gold never in the top-10) goes from 0.127 forced to 0.254 own, so the gold is *less* present
there at the own position.

**Exploratory split by answer marker (not pre-registered; found by the reviewer, recomputed here).** The pooled null
mixes two groups that move in opposite directions.
- *Outputs with an `Answer:` marker* (165 of 204; the only ones actually read at an answer position): the gold's presence
  goes **up** at the own position, 0.685 to 0.776 on the logit row, paired +0.091 (+0.030, +0.152); J-lens 0.661 to
  0.733, +0.073 (+0.012, +0.133). Development folds: logit +0.110 (+0.050, +0.170), J-lens +0.070 (+0.020, +0.120).
  Logit depth among those emerged in both modes: +0.021 (+0.003, +0.041), a small shift earlier.
- *Outputs without the marker* (39 of 204, 23 of them judged correct): read at their first token, the gold emerges on 30
  at the forced position but on 17 at the own position (logit; J-lens 29 → 17). These outputs answer in their own words from the first token
  ("user has been collecting…" for a gold of "three months"), so their own-position read is the paraphrase floor of
  §34a, not an answer position.
- The loss of 13 on the marker-less rows cancels the gain of about 15 on the marked rows, which is why the
  pre-registered pooled Claim A is a null. The pre-registration also stated "200 of 204" outputs carry the marker; the
  count is 165 (the 200 came from a reviewer's note on a different check and was not verified before the run).

**Reading it.** The pooled, pre-registered test rules out only large effects of the answer position: a presence shift
above about 0.07 and a depth shift above about 0.03. Within that, two things can be said. The *lateness* of emergence
belongs to the model: where the gold emerges in both modes its depth is the same (median 0.917 either way; +0.015
(−0.001, +0.033) on the deciding row), and the 7B does not get earlier either. Part of the *absence* is due to the
forced format: on the outputs the 3B actually answered after a note, the gold is present at the own position on 0.78
against 0.69 forced (exploratory, both lenses, both fold sets), so the share where the gold never enters the top-10
falls from 0.315 to 0.224 on those rows. The §34a caveat is therefore narrowed, not closed. By class on the marked rows
(3B, logit, folds 2–4), the A share forced → own is: failures (111) 0.369 → 0.306, refusals (62) 0.419 → 0.323,
answered wrong (49) 0.306 → 0.286, successes (54) 0.204 → 0.056. Most of the presence gain is on the successes, so the
own position *widens* the gap between failures and successes in presence, which supports analysis 1 (presence, not
timing, separates failures) rather than weakening it; the C-over-B picture and the depth numbers stand. The failure
decomposition to quote from here on is the marked-row, own-position one: A 0.31, B 0.05, C 0.64 (n 111); the pooled
own-position cells (A 0.32, B 0.05, C 0.63) include the 39 marker-less reads. It is still a readout, not an intervention.

**Caveats.** 19% of the 3B's outputs (39 of 204) carry no `Answer:` marker; they are read at their first token, which is
not an answer position, and they are included in the pooled claims as pre-registered. The 3B's J-lens failed its gate
(§34), so the deciding rows are logit-lens rows; the J-lens rows agree on presence and on the marker split, and on depth
they lean later at the own position (−0.163 (−0.209, −0.118)), which does not change the reading. Emergence depth with k = 10 likely
carries a token-frequency component (analysis 1 found the 7B's failures emerging *earlier* than its successes), so depth
comparisons across readers are descriptive; the within-question paired comparison here is not affected. Labels are the
§23 single-judge verdicts. Teacher-forcing the cached output assumes the model would reproduce it; §33's regeneration
check found it does not always (10 of 20 exact), so the own position is the cached answer's position under the cached
text, which is what the judge scored.

**Run and spend.** One A40 secure pod (EU-SE-1, $0.59/h), 2026-10-13 20:34–20:52 UTC: install 8 s, own-position reads
294 s (3B) and 323 s (7B) with the §34 lenses, no fitting. Two false starts before the chain ran: the code bundle from
`git archive` omitted the untracked `external/jacobian-lens` (preflight exit 3, fixed by copying it), and the model
download hit Hugging Face's unauthenticated rate limit (fixed by placing the HF token on the pod). The local hard-stop
timer was armed at 58 min as approved (60 min, $0.59); the ledger's pod-creation line carries the creation script's
default "$4.00" text, corrected in a later ledger entry. Terminated 20:52 UTC by DELETE (list-pods empty). About 0.3 h
≈ $0.18 against the approved $0.59 hard stop. All experiments ≈ $16.90.

### 34c. Does "present at the answer position" depend on the evidence? Partly: the evidence matters, and more than half of the 3B's "present" failures are present without it (pre-registered 2026-10-13, run and confirmed the same day)

**Status: RUN and CONFIRMED on folds 2, 3, 4; both pre-registered claims SHOWN; the plausibility share, the number the
experiment exists for, is about 0.6 among the 3B's failures.** Pre-registration
`research/proposals/2026-10-13-exp34c-evidence-ablation.md` (ce9910c, before the pod; reviewed at the result stage);
code `--position-mode ablated` in `scripts/jlens_read.py` (ae62ffc; exact-span removal by
`memctl.analysis.jacobian.ablate_evidence`), `scripts/exp34c_pod.sh`, `scripts/exp34c_report.py`. Reports
`runs/exp34c_report_{dev,confirm,allfolds}.md`, copies `docs/research/exp34c_evidence_ablation_report_*.md`. Reads
`runs/exp34_jlens_ablated/<reader>/` (499 of 500 per reader, the same one failed read on fold 2 as §34), each carrying
the intact read's prompt hash.

**Why.** Reviewer questions 2 and 4 on §34a: a gold token in the final-layer top-10 may be there because it is
plausible for the question ("2" for a counting question) rather than because the reader extracted it from the evidence
in view. §34c reads every in-view question at the same forced position on a prompt with every evidence line cut out
and nothing else changed, and asks how much of "present" survives.

**Claims (3B, deciding logit row, folds 2–4, 204 questions; all folds, 330, in brackets).**

| claim | with evidence | without | intact − ablated, paired (95% CI) | verdict |
|---|---|---|---|---|
| A: gold ever enters the top-10 | 0.701 | 0.466 | +0.235 (+0.176, +0.294) [+0.236 (+0.191, +0.282)] | **SHOWN** |
| B: gold in the final-layer top-10 (outcome C; model logits, lens-free) | 0.672 | 0.431 | +0.240 (+0.181, +0.299) [+0.242 (+0.197, +0.288)] | **SHOWN** |
| evidence lift, log10 best rank intact − ablated (descriptive) | | | +0.631 (+0.507, +0.763), about 4× in rank | |

Development folds: +0.238 (+0.167, +0.317) and +0.246 (+0.175, +0.325), SHOWN; the J-lens rows agree throughout. The
7B (descriptive, J-lens): presence 0.931 → 0.696 (+0.235 (+0.172, +0.294)); C 0.887 → 0.627 (+0.260 (+0.191, +0.328));
lift +0.597 (+0.473, +0.730). Cutting the evidence lowers presence by about 0.24 (absolute) in both readers, a
relative drop of 0.34 for the 3B and 0.25 for the 7B.

**The plausibility share (3B, logit, folds 2–4): among questions that are C with the evidence, the share still C
without it.**

| class | n C with evidence | still C without (95% CI) | n ever-emerges with | still emerges without |
|---|---|---|---|---|
| refusals | 37 | 0.568 (0.405, 0.730) | 39 | 0.590 (0.436, 0.744) |
| answered wrong | 44 | 0.636 (0.500, 0.773) | 44 | 0.636 (0.500, 0.773) |
| all failures | 81 | **0.605 (0.494, 0.716)** | 83 | 0.614 (0.506, 0.723) |
| successes | 56 | 0.696 (0.571, 0.804) | 60 | 0.717 (0.600, 0.833) |
| all questions | 137 | 0.642 (0.562, 0.723) | 143 | 0.657 (0.580, 0.734) |
| strict gold token (number, name) | 116 | 0.707 (0.621, 0.784) | 121 | 0.702 (0.620, 0.785) |
| function-word gold token | 21 | 0.286 (0.095, 0.476) | 22 | 0.409 (0.227, 0.636) |

All folds: failures 0.630 (0.551, 0.709), successes 0.656, strict 0.702, function-word 0.281. The 7B's shares are of
the same size: failures 0.745 (0.617, 0.851), successes 0.657 (0.575, 0.739) on folds 2–4.

**The ` unknown` token without evidence (descriptive, the calibration check).** At the forced position, ` unknown`
reaches top-1 at some layer on 0.578 of questions for the 3B with the evidence and 0.632 without it (118 → 129 of
204, logit). For the 7B: 0.078 with and 0.529 without (16 → 108, J-lens). All folds: 3B 0.582 → 0.624; 7B 0.082 → 0.558.

**Reading it.**
- *The readout does measure evidence use.* Both claims are SHOWN with intervals far from zero, in both readers, on
  both lenses: cutting the evidence takes the gold out of the top-10 on about a quarter of questions and worsens its
  rank about 4×. "Present" is not only plausibility.
- *But "present at the end" overstates extraction by a lot.* About 0.6 of the 3B's C-failures, and 0.7 of its
  C-successes, keep the gold in the final-layer top-10 with no evidence in the prompt. The strict-token subset is the
  worst (0.71): small numbers and common names are plausible for their questions whether or not anything was read. The
  function-word subset is the only one where C mostly depends on the evidence (0.29). §34a's decomposition therefore
  has to be read with this discount: of the 0.64 "present at the end" among the 3B's failures, only about 0.25 (= 0.64
  × 0.39; 32 of 127) is present *because of the evidence*; the rest would be there anyway.
- *§34a Claim 1 is a ranking difference that exists without the evidence, not a difference in extraction.* With the
  evidence cut out, the 3B − 7B log10 best-rank gap is +0.473 (+0.361, +0.584) on the logit row against +0.486 intact;
  and the two readers' evidence lifts do not differ, 3B − 7B −0.013 (−0.126, +0.097) (J-lens: gap +0.466 ablated vs
  +0.535 intact; lift difference −0.069 (−0.187, +0.045)). The 7B ranks the gold higher than the 3B with or without
  the evidence in view, by the same margin; what the evidence adds is the same in both. Claim 1 should be quoted as a
  rank gap at the answer position, not as "the 7B extracts more".
- *The 3B's ` unknown` signal barely responds to the evidence; the 7B's does.* ` unknown` reaches top-1 at some layer
  on 58% of questions for the 3B with the evidence and 63% without, paired +0.054 (+0.010, +0.103) on the logit row:
  small, but the interval excludes 0. For the 7B it goes from 8% to 53%, +0.451 (+0.382, +0.520) on the J-lens. At the
  forced position the 7B's disposition to say "unknown" tracks whether the evidence is in front of it; the 3B's moves
  little. The 3B's signal does not track its own behaviour either: ` unknown` reaches top-1 on 58% of questions while
  the 3B refuses on 32% (65 of 204). This is a descriptive contrast at a forced position, not an intervention.
- *What this changes in §34a.* Outcome C is "present", not "extracted"; the interpretive sentence becomes: the 3B's
  answer token is in reach at the end in two thirds of failures, about two fifths of that (0.25 of 0.64; 32 of 127)
  evidence-driven, and the 3B's ` unknown` signal barely moves with the evidence. The C-over-B picture stands (B is
  rare with or without evidence); Claim 1 stands as a paired rank gap but, per the bullet above, not as an extraction
  difference; the "late emergence" descriptive is unchanged.

**Caveats.** Forced position only (the own position of §34b cannot be used: the cached note was written with the
evidence and would carry it). Cutting the evidence lines changes the prompt length and the positions of P0 and Q, so
the intact and ablated reads pair by question id, not prompt hash; the intact hash is stored and checked. The 3B's
J-lens failed its gate (§34), so the deciding rows are logit rows; the J-lens rows agree. Plausibility shares are
descriptive proportions with wide intervals on the smaller classes (refusals n 37). Labels are the §23 single-judge
verdicts; a false negative moves a success into the failure rows and carries its (higher) plausibility share with it.

**Run and spend.** One A40 secure pod (CA-MTL-1, $0.59/h), 2026-10-13 21:29–21:40 UTC, chain on the first attempt:
install 31 s, ablated reads 238 s (3B) and 272 s (7B) with the §34 lenses. About 0.18 h ≈ $0.11 against the approved
$0.59 hard stop. All experiments ≈ $17.00.

## 35. Does the 3B derive the answer in its own note and still refuse? Separating derivation from answer selection (2026-10-13, pre-registered; run 2026-10-14)

**Status: RUN and CONFIRMED on folds 2, 3, 4; gate passed; Claims 1 and 2 SHOWN; the Claim 3 lifts are two to three
times the pre-registered prediction.** Pre-registration `research/proposals/2026-10-13-exp35-answer-selection.md`
(c01aaf8 draft; revision e290a66 after reviewer round 1; 12c9168 and 26603cf for the reviewer's pre-launch items;
reviewer rounds 1–3 in `research/reviews/2026-10-13-exp35-prereg.md`, ACCEPT at 26603cf, recorded 289ebd3, all before
the pod). Code `memctl/analysis/answer_selection.py`, `scripts/exp35_generate.py`, `scripts/exp35_pod.sh`,
`scripts/exp35_report.py`, `scripts/exp35_note_check.py` (2c04307, 0d23082, 477a43a), tests
`tests/test_answer_selection.py`. Rows `runs/exp35/<reader>/<arm>/`, judge rows `runs/exp35/<reader>/judge/`, judge
cache `cache/judge_exp35/`. Reports `runs/exp35_report_{dev,confirm,allfolds}.md` (the confirmation one is quoted below),
copies `docs/research/exp35_answer_selection_report_*.md`.

**Why.** §34a–c read the 3B's residual stream and found the gold "present" at the answer position in two thirds of its
failures, with ` unknown` winning late, and a ` unknown` disposition that barely moves when the evidence is cut (§34c).
Those are readouts. An outside design note (2026-10-13) asked the same question at the level of the model's own text:
can the 3B derive the answer and still not give it? The frozen §23 prompt already makes every reader write a note
("name the facts ... then do any counting, adding or date arithmetic") before one line `Answer: <phrase>`, and already
says "Reply unknown only if the memory has nothing at all on the topic". So the note is a derivation the model wrote
without seeing the gold, and the answer step can be intervened on with the note held fixed.

**Design (3B primary, 7B beside; the §34 population: `fixed8` in-view answerable questions with a §23 label, paired by
question id and prompt hash; folds 0–1 development, 2–4 confirmation).** Marked rows are cached outputs that contain
`Answer:` (3B 165 of 204 on folds 2–4; 7B all). The note is everything before the last `Answer:`, byte-exact. **D1**
(deciding): the normalised gold string, or every gold token, occurs in the note. **D2** (beside): the §23 judge asked
whether the note alone answers the question. Six arms, all generated in one run on one engine (vLLM 0.8.5 offline, bf16,
greedy, temperature 0, seed 0; max_new_tokens 256 for the regenerated arms, 64 for the frozen-note arms, 4 for the
judge), completion-style on the chat-templated prompt:

| arm | input | role |
|---|---|---|
| N | the intact prompt, regenerated | within-engine baseline for F; reproduction check |
| A | chat text + cached note + `Answer:` | within-engine baseline for B; gate G |
| B | chat text + cached note + newline + "Based on the note above, give the answer as a short phrase. Do not reply unknown." + newline + `Answer:` | the answer step with the derivation held fixed |
| F | the prompt with both unknown clauses replaced ("Always answer with your best guess, even if the memory does not contain the answer; never reply unknown"; "Never reply unknown; if the memory has little on the topic, give your best guess anyway."), regenerated | forced choice with a new derivation |
| F-cut, N-cut | F and N on the §34c evidence-cut prompt | what forced choice invents, and how often the normal prompt refuses, when nothing is in view |

Every answer and every note was judged by the §23 judge (Qwen2.5-7B-Instruct, official LongMemEval per-type prompts) on
the run's own engine; the cached answers were re-judged as a check (204 of 204 agree with the §23 labels). Refusal rule
as §23: the text after the last `Answer:` starts with "unknown". Conditioning sets on marked rows: **S** = cached refusal
and D1 true; **S'** = cached refusal and D1 false. Verdicts: gate G (arm A reproduces the cached verdict and refusal flag
on at least 0.8 of marked rows); Claim 1 = paired P(B correct) − P(A correct) on S, lower bound above 0, UNDERPOWERED
if |S| < 15; Claim 2 = P(B correct | S) − P(B correct | S'), lower bound above 0; Claim 3 descriptive, acc(B) − acc(A)
and acc(F) − acc(N) with the share of the 7B-minus-3B *cached* gap on the same rows; costs (i) and (ii) beside. 10,000
paired bootstrap resamples, 95% intervals.

**Prediction, recorded before the data (proposal §3).** S about a quarter of the 3B's refusals (0.23 by D1 on the
development folds, where S had 7 of 30 marked refusals), about 0.07 of in-view questions; Claims 1 and 2 SHOWN, Claim 1
"probably at the edge of power"; arm B lifting accuracy by roughly +0.05 to +0.08, "about a quarter of the matched
3B−7B gap at most"; F more, with a larger cost. Reading if that held: selection "a real, separable, but minor part of the
3B's shortfall". **The size part of the prediction was wrong**, see below.

### 35a. Results (confirmation folds 2, 3, 4; 204 paired questions, 165 marked 3B rows; all folds, 330 and 265, in brackets)

**Gate G, PASS.** Arm A gives the cached verdict and refusal flag on 0.964 (0.933, 0.988) of marked rows [0.966], and
the cached final answer exactly on 0.952; on S, 1.000 of 22. Arm N (the whole prompt regenerated) gives the cached final
answer on 0.642 and the whole cached output on 0.333, with acc(N) − cached −0.024 (−0.079, +0.030): offline vLLM
reproduces the server's answers below what §33 found on 20 questions (16 of 20 final answers, 10 of 20 whole outputs),
consistent with it given n 20, and not better.

**Claim 1 (a selection gap exists), SHOWN.** |S| = 22 (0.108 of the population; D1 true on 22 of 62 marked refusals,
0.35, against 0.23 on the development folds). Under A every one of the 22 refuses again; under B 14 answer correctly, 6
still refuse, 2 answer wrong.

| claim | n | estimate (95% CI) | verdict |
|---|---|---|---|
| Claim 1: P(B correct) − P(A correct) on S | 22 | +0.636 (+0.455, +0.818) [+0.621 (+0.448, +0.793), n 29] | **SHOWN** |
| s = P(B correct \| S), Wilson | 22 | 0.636 (0.430, 0.803) | descriptive |
| the same difference on the D2 set (beside) | 23 | +0.652 (+0.435, +0.826) | descriptive |
| Claim 2: P(B correct \| S) − P(B correct \| S') | 22 vs 40 | +0.311 (+0.055, +0.552) [+0.335 (+0.123, +0.536)] | **SHOWN** |

Development folds (read first): Claim 1 +0.571 (+0.143, +0.857) with |S| = 7, UNDERPOWERED as pre-registered; Claim 2
+0.354 (−0.062, +0.739), not shown there; both point the same way as the confirmation folds.

**Claim 2 (selection, not guessing), SHOWN, but S' is not inert (the rest of this paragraph is exploratory).** B
answers correctly on 0.636 of S and on 0.325 of S' (13 of 40). A plain "lower threshold plus lucky guesses" account of
the 13 is ruled out by the run's own control: F-cut (same question, no evidence) is judged correct on only 2 of the 13
(0.15, the plausibility floor of 0.147) and never gives B's answer; among the 27 S' non-recoveries F-cut is right on 1.
About 8 of the 13 fit "the note holds the ingredients but not the gold string": "user spent $60 on coffee mugs ... bought
5 coffee mugs" → `$12`; "meeting up with Alex from Germany twice before" → `2`; "unique dessert shop, giant milkshakes" →
`The Sugar Factory`; there the last step of the arithmetic or the naming happens *at the answer step*, once unknown is
off the table. The other 5 do not: 3 recovered notes are literally "unknown" (7 of the 62 marked refusal notes are),
1 is off-topic, 1 names only the topic. On those, B answered from the evidence in the prompt, not from the note, so the
frozen-note arm does not freeze the derivation where the note is empty. D1 therefore under-counts derivation, and "the
answer step given the note" is what B changes: it includes a final computation, and where the note is empty a fresh
read, not only a choice between an answer the note states and unknown.

**Claim 3 (the thesis stake).** On the 165 marked rows, the 7B's cached accuracy exceeds the 3B's by 0.333 (0.660 vs
0.327).

| comparison | rows | arm | baseline | difference (95% CI) | share of the matched 3B−7B cached gap |
|---|---|---|---|---|---|
| acc(B) − acc(A), frozen note | 165 | 0.509 | 0.333 | **+0.176 (+0.109, +0.242)** [+0.151 (+0.102, +0.204)] | 0.53 (0.33, 0.73) [0.53] |
| acc(F) − acc(N), regenerated | 165 | 0.545 | 0.303 | **+0.242 (+0.164, +0.321)** [+0.192 (+0.132, +0.253)] | 0.73 (0.49, 0.96) [0.67] |
| acc(F) − acc(N), all 204 rows | 204 | 0.569 | 0.343 | +0.225 (+0.157, +0.294) [+0.188 (+0.136, +0.239)] | 0.72 (0.50, 0.94) [0.70] |

Development folds: +0.110 (+0.050, +0.180) and +0.110 (+0.030, +0.190). Descriptive: the 3B's refusal rate on in-view
questions goes from 0.358 under N to 0.015 under F. By type (3B, folds 2–4, accuracy): temporal-reasoning 0.185 (A) → 0.444 (B) on 54
rows and 0.155 (N) → 0.448 (F) on 58; single-session-user 0.393 → 0.679 and 0.556 → 0.806; single-session-assistant
0.600 → 0.600 and 0.467 → 0.833; multi-session 0.375 → 0.531 and 0.278 → 0.472; knowledge-update 0.417 → 0.500 and
0.486 → 0.514 (flagged type, different judge prompt); single-session-preference cached 0.111 and F 0.111, N, A and B 0 (9 questions, flagged).

**The 7B (descriptive).** Gate 0.995; 9 cached refusals, |S| = 1. acc(B) − acc(A) +0.020 (+0.005, +0.039), small but
above the noise floor; acc(F) − acc(N) +0.005 (−0.034, +0.044). The 7B has almost no refusals to recover and forced
choice does nothing for it.

**Costs (pre-registered).**

| cost | 3B, n | share (95% CI) | 7B |
|---|---|---|---|
| (i) S' refusals that B turns into judged-wrong answers | 40 | 0.275 (0.150, 0.425) [0.238] | 0.500 of 8 |
| (i) all marked refusals that B turns into wrong answers | 62 | 0.210 (0.113, 0.306) [0.196] | 0.556 of 9 |
| (ii) evidence cut: F-cut judged correct (plausibility floor) | 204 | 0.147 (0.098, 0.196) [0.152] | 0.240 |
| (ii) evidence cut: N-cut refuses (unknown is right) | 204 | 0.647 (0.578, 0.711) [0.630] | 0.627 |
| evidence cut: F-cut refuses (descriptive) | 204 | 0.319 (0.255, 0.382) [0.342] | 0.245 |
| evidence cut: N-cut judged correct (descriptive) | 204 | 0.088 (0.054, 0.127) | 0.186 |

Descriptive, derived from (ii): with nothing in view, the normal prompt already answers 0.35 of the time (0.09 of all
prompts "correct" by plausibility); forced choice answers 0.68 of the time (0.15 correct by plausibility), so the share
of prompts answered and judged wrong on evidence-free prompts roughly doubles, from 0.265 to 0.534 (no confidence is
measured; judge false negatives make these absolute shares upper bounds). Even told never to say unknown, the 3B still
refuses on 0.32 of evidence-free prompts and on 0.015 of in-view ones (descriptive): its refusal does track the
evidence under F.

**D1 against D2.** On 165 marked notes: both 47, D1 only 16, D2 only 30, neither 72; agreement 0.721, κ 0.433. On the
62 marked refusals D1 holds on 22 and D2 on 23; the D2-set result (+0.652) matches the D1 one.

**Reading it.**
- *The answer step, not the note, is where a large part of the 3B's in-view shortfall sits.* With its own note held
  fixed to the byte and one line added that forbids unknown, the 3B's accuracy rises by 0.18, about half of the matched
  gap to the 7B; with the two unknown clauses rewritten and everything regenerated, by 0.24, about three quarters. The
  pre-registered prediction (+0.05 to +0.08, "minor") was wrong by a factor of two to three, because D1 misses the
  refusals whose note holds the ingredients and the last step is done at the answer, and those whose note is empty and
  B reads the evidence afresh (exploratory, above). This is the behavioural
  counterpart of §34c: the 3B's ` unknown` disposition barely tracks the evidence at the forced position, and here its
  refusal is mostly removable by instruction.
- *It is not just a lower threshold.* Claim 2 holds (B is right twice as often where the note holds the gold), the 7B
  gains little from the same interventions (B − A +0.020 (+0.005, +0.039), twice the noise floor; F − N +0.005, nothing),
  and B's wrong answers on S' (0.275) are well below its correct ones on S (0.636). But the threshold moves too: a fifth of all marked refusals become wrong answers under B, and on evidence-free
  prompts F doubles the answered-and-wrong share. Forced choice is a trade of calibration for coverage; whether it pays
  depends on the cost of a wrong answer against an abstention, which this benchmark's metric does not price (and F
  forfeits LongMemEval's 30 abstention questions by construction, where the cached 3B abstained correctly on 23).
- *What it changes upstream.* §23's claim 1 (a controller cannot lift the 3B to the 7B's baseline) stands as stated: the
  controller changes what the reader sees, and the reader's in-view accuracy was the ceiling. About half of that ceiling
  is a *response policy*: the same 3B, same evidence, same note, closes 0.53 (0.33, 0.73) of the matched gap when told
  not to abstain, at the cost of a fifth of its refusals becoming wrong answers; under B it reaches 0.51 against the 7B's
  0.66, and the other half of the gap stays with the reading. A note-and-forced-answer prompt is an untested,
  zero-parameter alternative to the LoRA route of §23/Q14; it forfeits abstention and carries the costs above.

**Caveats.** D1 is a string criterion that under-counts derivation (shown by the S' recoveries); the headline Claim 3 rows
do not depend on it. B's line also invites the model to finish the computation, so "selection" here means the final
answer step given the note, not a pure choice between a stated answer and unknown. F changes the note too (full
regeneration), so F − N mixes derivation and selection; B − A is the clean one and (descriptive) amounts to 0.73 of
F's lift. Labels are
the §23 single judge (understates accuracy by about 0.1; paired rows unaffected; the judge ran on this run's engine and
agreed with the §23 labels on 204 of 204 cached answers). The 39 marker-less 3B outputs are outside the frozen-note arms;
the all-rows F − N row covers them. Offline vLLM reproduces the server's cached answers on 0.64 of questions (final
answer) and acc(N) − cached is −0.024, so every verdict is within-engine. Two flagged types (knowledge-update,
single-session-preference) have different judge prompts and are reported descriptively. One benchmark, one prompt family,
one 3B.

**Run and spend.** One A40 secure pod (CA-MTL-1, $0.59/h), created 2026-10-14 00:57 UTC, chain 01:01:16–01:17:18 UTC
on the first attempt (install 183 s with vLLM 0.8.5 on torch 2.6.0+cu124; 3B main arms 220 s; 3B cut arms 158 s; 7B all
arms 367 s; judge 94 s for 5,085 items), deleted 01:19 UTC; uptime about 0.37 h × $0.59 ≈ $0.22 (the account balance moved 12.654 → 12.488, $0.17, and
billing lags), against the approved $0.89 hard stop. All experiments ≈ $17.22.

## 36. Can the 3B tell which of its own forced answers the memory supports? (2026-10-15, pre-registered; run 1 not reliable, §36b rerun the same day)

**Status: RUN; readout not reliable in run 1; in §36b (a second elicitation) Claim 1 NOT SHOWN on folds 2, 3, 4, so
Claims 2 and 3 were not tested.** Pre-registration `research/proposals/2026-10-15-exp36-verification.md` (d2a67f2;
Reviewer step waived by the user, DECISIONS D-21, a3a3f9c; amendment 1 f14fe54, development folds only). §36b
amendment `research/proposals/2026-10-15-exp36b-verifier-wording.md` (30415c3; probe pick recorded 84e1106); rerun
approved by the user (D-23); launch confirmed in the Executor pane. Reviews `research/reviews/2026-10-15-exp36-run1.md` and
`research/reviews/2026-10-15-exp36b-results.md` (ACCEPT, 308c198, every primary figure recomputed independently).
Code `memctl/analysis/verification.py`, `scripts/exp36_verify.py`, `scripts/exp36_pod.sh`, `scripts/exp36_report.py`,
`scripts/exp36_rescore.py`, `scripts/exp36_wording_probe.py`, `scripts/exp36_devcheck.py`,
`scripts/exp36_audit_sample.py` (723bf29, 26f67f6, 7b3f249, baebd29, 189168c, f8c89e5, 5cba705, 2988a12, d850c51,
8c933f3, 57f25c6); tests `tests/test_verification.py`. Rows `runs/exp36/` (run 1) and `runs/exp36b/` (§36b); probe
`runs/exp36_probe/`; judge audit `runs/exp36_audit/`. Reports `runs/exp36b_report_{dev,confirm,allfolds}.md` (the
confirmation one is quoted below; 14e8478) and `runs/exp36_report_dev.md` (run 1).

**Why.** §35 showed that forbidding "unknown" with the 3B's own note held fixed recovers many answers (B − A +0.176),
at a cost: a fifth of refusals turn into wrong answers. An outside design note (2026-10-15) asked whether the 3B can
tell which of its forced answers the memory supports. If it can, its own judgment could gate a permissive policy
safely. If it cannot, a permissive policy needs a better checker.

**Design.** Candidates are the §35 arm-B answers (frozen note, "do not reply unknown", greedy vLLM 0.8.5), generated
before any verification instruction. Labels are the §35 judge verdicts on B. Fixed-candidate rows are the 3B's marked
rows (dev 100, confirmation 165). Rows where B still said unknown (dev 16, confirmation 25) have no candidate. They are
forced abstentions in every coverage and risk figure and are outside Claim 1's AUROC (amendment 1: scoring them lowest
hands any score free correct rankings). The verifier is the same checkpoint in a fresh prompt: the memory, the question,
"Proposed answer: <candidate>", and an A/B/C support question (A supported, B partial, C not supported). The score is
s = (p_A + 0.5·p_B)/(p_A + p_B + p_C) from first-token logprobs. A row is reliable when the label mass is at least 0.5,
and the item is "not reliable" if more than 0.1 of rows fall below that. Conditions on the same candidates: C0 natural
(emit iff the cached answer was not a refusal), C1 forced, C2 argmax A, C3 gate on s, C4 gate on log P(candidate) −
log P(" unknown") after the note, C4′ the §34b final-layer unknown rank. Claims in fixed sequence. Claim 1: AUROC of s,
SHOWN if the lower bound is above 0.5, meaningful at 0.65 or more. Claim 2: at C0's coverage, risk(C3) − risk(C0).
Claim 3: risk(C3) − risk(C4). Claim 4 (descriptive): τ fitted on folds 0–1 at r* = 0.30. Controls: A, the evidence cut
(same candidate); A′, intact correct candidates against the 3B's F-cut answers; length control, cut lines replaced by
length-matched irrelevant lines; B, counterfactual number substitution. The 7B verifies the same 3B candidates, and its
own (descriptive). Question-level paired bootstrap, 10,000 resamples. The judge audit (blind, 60 dev rows) agrees 58 of
60, one error each way.

**Run 1: readout not reliable.** On every item of both readers, the label mass fell below 0.5 on more than 0.1 of
rows. There were two causes. (i) A scorer bug: vLLM stores byte-level BPE token strings (`ĠA` for " A") and the scorer
matched " A". (ii) The design itself: the prompt kept the §23 goal line ("Answer with a short phrase ... reply: unknown")
and ended with a literal `Answer: `. So the readers answered the question instead of grading it. Decoded, only 0.11 of
the 3B's dev V rows reached mass 0.5 (the 7B 0.35). The top-1 tokens were digits, 未知 and " unknown". As pre-registered,
run 1 has no verdict. Its confirmation folds were read only for label-free mass (by the Advisor, disclosed in the run-1
review), never for any correctness-linked number, and they stay sealed. Run 1 pod: 19:13–19:30 UTC, chain 859 s.

**§36b: the wording fix.** A CPU probe (transformers fp32, laptop) on development folds 0–1 tested candidate wordings
in a fixed order, under a label-free gate: mass ≥ 0.5 on ≥ 0.9 of the 3B's V and Cut-B rows. No correctness-linked
number was computed. The first in the order, (a) = the run-1 P2 wording with the trailing `Answer: ` removed, passed on
84 of 84 rows for both items. The other seven wordings, (b)–(h), all at or near 1.0 on their first 25 questions, are
descriptive (`runs/exp36_probe/summary.md`). Removing the answer cue was the whole fix. On the pod, a 5-row real-model
smoke per reader gated the chain (3B 5/5, 7B 5/5). Every row of every item then had mass ≥ 0.5.

### 36a. Results (§36b, confirmation folds 2, 3, 4; 165 fixed-candidate 3B rows, 140 with a candidate; development folds in brackets)

| claim | n | estimate (95% CI) | verdict |
|---|---|---|---|
| Claim 1: AUROC of s (V) for B judged correct, candidate rows | 84 pos / 56 neg | 0.573 (0.478, 0.666) [0.559 (0.434, 0.682)] | **NOT SHOWN** (lower bound ≤ 0.5) |
| beside: PR-AUC, prevalence | | 0.698, 0.600 | descriptive |
| beside: the natural refusal flag (C0) as a score, candidate rows | | 0.455 (0.381, 0.532) [0.484] | descriptive |
| beside: C4 margin AUROC; C4′ unknown-rank AUROC | | 0.502 (0.398, 0.603); 0.502 (0.404, 0.603) | descriptive |
| Claim 2: risk(C3) − risk(C0) at C0's coverage (0.606) | 165 | −0.060 (−0.132, +0.012) [+0.014] | not tested (Claim 1 not SHOWN) |
| Claim 3: risk(C3) − risk(C4) at the same coverage | 165 | −0.020 (−0.090, +0.049) [+0.043] | not tested (Claim 1 not SHOWN) |

Claim 4 (descriptive): the development τ (s ≥ 0.513) emits 0.339 of rows with risk 0.357 (0.233, 0.484). It does not
hold the 0.30 cap on the confirmation folds. The C4 τ emits 0.048.

| control | 3B verifier | 7B verifier (descriptive) |
|---|---|---|
| A: s(intact) − s(cut), B judged correct | +0.235 (+0.170, +0.302) [+0.249] | +0.508 (+0.419, +0.596) [+0.469] |
| length control: s(intact) − s(LC); s(LC) − s(cut) | +0.238 (+0.174, +0.306); −0.003 (−0.039, +0.034) | +0.516; −0.009 |
| A′: AUROC, intact correct-B vs Cut-F (84 / 120) | 0.715 (0.652, 0.774) [0.681] | 0.760 (0.703, 0.816) |
| B: number substitution, no other copy in memory, follows in both contexts | 8 of 10 (descriptive, n < 20) | 8 of 10 |

7B − 3B difference of the Control A drop, paired: +0.273 (+0.187, +0.359). **The 7B checking the same 140 3B
candidates** (descriptive; the judge that made the labels is the 7B): AUROC 0.692 (0.600, 0.777). Paired 7B − 3B AUROC
on the same candidates: **+0.119 (+0.023, +0.216)** (computed after the report, same bootstrap unit; 10,000 resamples,
seed 0). At C0's coverage, its risk is 0.340, −0.090 (−0.171, −0.010) against the natural policy. The 7B on its own B
candidates: AUROC 0.731 (0.651, 0.805).

**Costs, beside.** C1 (emit every candidate) adds 13 wrong answers on 165 rows (27 refusals recovered correctly, 13
turned wrong). C3 at matched coverage: 6 fewer wrong (17 refusals recovered, 9 turned wrong). On evidence-free prompts,
Cut-F candidates emitted at the development τ: 0.183 (3B), 0.317 (7B). Abstention questions: 4 rows, none emitted by the
3B. Verification cost: one prefill of about 1,100 tokens per candidate, 0.07 s (3B) and 0.14 s (7B) on an A40.

**Reading.** (1) Claim 1 is NOT SHOWN. Given the memory, the 3B cannot reliably tell which of its own forced candidates
are right (0.573, interval includes chance), so the pre-registered gate questions (Claims 2–3) were not tested. Their
point estimates are listed only for completeness. (2) What the 3B's support score does carry is evidence presence. It
drops by 0.235 when the evidence lines are cut, the drop is the same when they are replaced by length-matched irrelevant
lines (so it is not prompt length), and it separates supported candidates from evidence-free F-cut answers at 0.715. It
detects whether the memory speaks to the question, not whether the candidate matches what the memory says. (3) The 3B's
own refusals carry no signal about which of its forced candidates are right (0.455 on candidate rows). The 0.62 in the
first draft came entirely from rows with no candidate (amendment 1). (4) The 7B is a better checker of the same
candidates (+0.119 paired AUROC; twice the evidence sensitivity, +0.508). This is descriptive, and the 7B is also the
judge, so shared errors can inflate it. (5) Prediction scorecard (pre-registered in d2a67f2). Claim 1 "SHOWN but weak,
0.60–0.68": wrong, not shown. Claim 3 NOT SHOWN: not tested. Control A "small for the 3B, under 0.1": wrong, +0.235; the
§34c prior (the 3B's ` unknown` disposition ignores the evidence) does not carry over to this readout. "Clearly larger
for the 7B": right (+0.508). "7B AUROC ≥ 0.75": wrong, 0.692. The broad reading predicted (the 3B cannot check its own
work; a stronger checker can, partly) holds, with a smaller 7B advantage than predicted.

**Caveats.** The readout is a second elicitation under a wording chosen for one-letter compliance on development folds.
It is "elicited verification", never "the reader knew". Run 1's wording failed for a format reason, so a third wording
could read differently, although (b)–(h) complied on their 25 dev questions and were not scored. Labels come from one 7B
judge (understates accuracy by about 0.1; the audit gives 58 of 60), and the 7B verifier shares its weights. The
exact-span cut leaves derivable content: Cut-F answers the judge marks correct get a mean 3B s of 0.655 against 0.496
for wrong ones (plausibility or leftover evidence). Control B has n 10. Coverage and risk are on fixed candidates, not
deployed accuracy (§35 gives that). One benchmark, one prompt family, one 3B.

**Run and spend.** Run 1: one A40 secure pod (EU-SE-1, $0.59/h), 19:13:34–19:30:14 UTC 2026-10-15, chain 859 s.
§36b: one A40 secure pod (EU-SE-1, $0.59/h), 22:26:26–22:42:34 UTC, chain 846 s (install 145 s, 3B 250 s, 7B 451 s), both smokes PASS,
in-pod self-stop wired (RUNPOD_POD_ID exported), laptop DELETE as the backstop. The account balance moved 12.452 →
12.126, $0.33 for both runs, against an approved hard stop of $0.59 each. All experiments ≈ $17.55.

## 37. Can the 3B be cross-examined into checking its own answer? An elenchus in three turns (2026-10-16, pre-registered; closed on the development folds)

**Status: CLOSED, development only. The development gate FAILED: AUROC(e) 0.505 (0.431, 0.574) on folds 0–1, 84
candidate rows. No pod was quoted and no confirmation-fold number exists.** Pre-registration
`research/proposals/2026-10-16-exp37-elenchus.md` (Advisor; d993eca, revision 2 714da06, amendment 1 9639186). The
user approved the free laptop run (D-25, b0f955d). Reviews: `research/reviews/2026-10-16-exp37-build.md` (ACCEPT on
condition of the normaliser fix) and `research/reviews/2026-10-16-exp37-dev.md` (gate FAIL confirmed, 06bc575). Code:
`memctl/analysis/elenchus.py`, `scripts/exp37_elenchus.py`, `scripts/exp37_dev.py`, `scripts/exp37_pod.sh` (built, never
run on a pod), `scripts/exp37_report.py` (213636e, 02332a2); tests `tests/test_elenchus.py` (73). Rows
`runs/exp37_dev/qwen3b/<item>/<turn>.fold<k>.jsonl`; report `runs/exp37_report_dev.md` (5c5c9ef).

**Why.** The user asked: "Did we try putting the 3B model through Socrates's elenchus?" Every §36 check was one shot and
one token, and the one-shot failed (§36b AUROC 0.573). An elenchus splits the judgment into smaller steps. First, state
the grounds. Second, derive what the grounds say on their own. Third, face the comparison.

**Design.** The candidates, labels and population are §36's: the §35 arm-B answers, the §35 judge verdicts, 84 dev rows
with a candidate. Each memory item is numbered `[k]`. Turn E1: "which memory lines is this answer based on?", parsed to
a set G, with g = 1 if G is non-empty. Turn E2 is blind: only the cited items and the question, with the candidate not
shown; it returns d, or "not determined". Turn E3: a one-letter same / partly / different comparison of the candidate
and d, scored by its logprobs as a. The score is e = 0 if g = 0 or d is "not determined", else a. Beside it: e_mech
(g × mechanical agreement, normaliser v2) and g alone. Controls: A (Cut-B), G (whether G hits an annotated evidence
item), P (a planted wrong thesis in the intact memory), S (blind against in-dialogue derivation), and V-num (§36b frame
(a) on the numbered block). The gate to quote a pod was: dev AUROC(e) ≥ 0.65 and ≥ AUROC(s, dev) + 0.05 = 0.609.
Laptop CPU, transformers fp32, greedy; 2 h 23 min wall time.

**Calibration first (label-free, 20 dev rows).** The primary E3 wording called identical pairs the same 17/20 times
(FAIL; the bar is 18/20). The one pre-fixed fallback ("Are Answer 1 and Answer 2 the same answer?") passed: 18/20
identical called the same, 19/20 different called different. It was re-run at the start of the run with identical
counts. Every miss was argmax B at label mass 1.0: the model is cautious, not off-format. The wording sha changed
b48048c2 → 7ef8e033 because NORMALISER_VERSION is part of the hash (v1 → v2 at 02332a2). Compliance: E1 parses on ≥ 0.98
of rows on every item; E3 mass ≥ 0.5 on every E3 row that ran.

**Result: the chain stops at the derivation.** Intact: g = 1 on 62 of 84 (0.738). Blind E2 returns "not determined" on
52 of those 62, so E3 runs on only 10 rows (6 judged correct, 4 judged wrong), and e = 0 on 74 of 84. In dialogue the
same thing happens: E2d returns "not determined" on 56 of 62, with 6 derivations. Cut-B: 22 grounded, 17 not determined,
5 derived. Cut-F: 25, 22, 3. Planted numbers: 8, 7, 1. Planted names: g = 0 on all 13.

| claim (dev folds 0–1, 84 candidate rows) | estimate (95% CI) | verdict |
|---|---|---|
| Claim 1: AUROC(e) | 0.505 (0.431, 0.574) | NOT SHOWN; gate FAIL (0.505 < 0.65 and < 0.609) |
| beside: AUROC(g); AUROC(e_mech); AUROC(V-num); §36b AUROC(s) | 0.576 (0.481, 0.675); 0.521 (0.460, 0.578); 0.548 (0.422, 0.675); 0.559 (0.433, 0.681) | descriptive |
| Claim 2: AUROC(e) − AUROC(s), paired | −0.055 (−0.181, +0.071) | not tested |
| Claim 3: risk(C3e) − risk(C0) at C0's coverage | −0.057 (−0.121, +0.004) | not tested |
| AUROC(e′), in-dialogue derivation | 0.537 (0.486, 0.589) | descriptive, never primary |

**Controls (descriptive).**
- A: g drops from 0.738 to 0.262 when the evidence is cut (+0.476 (+0.369, +0.582)), but e on judged-correct rows barely
  moves (+0.038 (−0.055, +0.135)). The Cut-F pass rate (e ≥ 0.5) is 0.050.
- G: the cited set contains an annotated evidence item on 0.560 (0.455, 0.667) of intact rows and 0.000 of Cut-B rows.
- P: the chain rejects the planted wrong number 7 of 7 times but accepts the true candidate only 1 of 7 ("follows the
  evidence" 1/7). For names it rejects the planted value 9 of 9 and accepts the true one 3 of 9 (3/9). Rejecting planted
  values means little here, because the chain rejects almost everything.
- S: the derivation reproduces a judged-wrong candidate 0.059 of the time blind and 0.000 in dialogue; for
  judged-correct candidates, 0.100 and 0.100. No pull of the model's own thesis is visible; the chain says "not
  determined" either way.

Costs: on the laptop, 1.86 calls, 1,231 input tokens and 26.4 s per intact candidate, against §36b's single call.

**Reading (the pre-registered third reading).** The 3B cannot ground and derive its own answers reliably. Its citations
miss the evidence about half the time. Given only the items it cited, it says the answer is not determined five times
out of six, including on rows where those items determine it. One observation, not a measurement: the review lists rows
00ca467f and 06db6396, where the cited items do state the answer and E2 still returned "not determined". So the
bottleneck is extraction (consistent with §34c), and the score again carries evidence presence (g drop +0.476; G 0.56
against 0.00), not candidate correctness. The self-verification route of Q14 closes with a second negative (§36b one
shot, §37 three turns). Only a larger checker (Q15) remains. No other E2 wording was tried: none was pre-registered, and
a retry would be a post-hoc wording search.

**Caveats.** Development folds only, 84 rows, wide intervals. The gate can refuse a true effect, the cheaper error.
Laptop fp32 transformers, not the pod's bf16 vLLM. Every score is elicited, three new prompts. Labels come from one 7B
judge (audit 58 of 60). P has n 7 and 9. Claim 4 (repair) has no number (no judge on the laptop). One benchmark, one
prompt family, one 3B.

**Spend.** $0 (laptop). All experiments ≈ $17.55.


## 38. Is the answer already under the 3B's refusal? A content-free "answer, don't refuse" direction with an evidence-cut control (2026-10-22, pre-registered; laptop design run and pod 2026-10-23)

**Status: RUN and CONFIRMED on folds 2–4, with a disclosed second setting. Primary setting (layer 25, α 4): NOT
INTERPRETABLE (Control X fails). Secondary setting (layer 29, α 2, chosen after reading development labels): Claim 1
SHOWN, +0.333 (+0.183, +0.483). Record: shown at the secondary setting only; two settings were tested.**
Pre-registration `research/proposals/2026-10-22-exp38-refusal-direction.md` (Advisor; rev 1 bb758c3, amendment 1
c367d28 with rulings on the Executor's F1–F9, amendment 2 at the development review). The user approved the draft, build
and free laptop design run (D-27, bb758c3) and the pod at the quote (D-28, 13214c0; the user's "Yes" also typed in the
Executor pane). Reviews: `research/reviews/2026-10-22-exp38-build.md` (08014ad), `research/reviews/2026-10-23-exp38-dev.md`
(4df512f, judge addendum 7e2c90d, C-row ruling fc58772), `research/reviews/2026-10-23-exp38-results.md` (ACCEPT,
ba9fc60). Code: `memctl/analysis/steering.py`, `scripts/exp38_dev.py`, `scripts/exp38_pod_run.py`,
`scripts/exp38_pod.sh`, `scripts/exp38_bundle.sh`, `scripts/exp38_report.py` (0943373, 468cd52, 377bd52); tests
`tests/test_steering.py` (26), `tests/test_exp38_pod.py` (13). Rows: `runs/exp38_dev/qwen3b/` and
`runs/exp38_report_dev.md` (c782dee); `runs/exp38_pod/qwen3b/` and `runs/exp38_report_confirm.md` (2263ca3, report
regenerated with the Control T and population-label changes of the results review).

**Why.** §34–§34c found the gold "present" in the 3B's late state on about two thirds of failures, with ` unknown`
winning late. §35 turned 27 of 64 marked refusals into correct answers with one line forbidding "unknown". Both were
readouts or prompt changes; nothing in the record had intervened on the representation. §38 asks: under the refusing
prompt, does shifting only the 3B's disposition to answer, with nothing question-specific added, release the gold? And
does the same shift release it as often when the evidence is not in the prompt?

**Design.** Arm A of §35 (frozen note + `Answer:`) refuses on R, the marked rows where A says "unknown" (28 dev, 64
confirmation). The direction v_L is the mean of resid_B − resid_A at the output of decoder layer L, at the ':' token of
`Answer:`, over the 13 development rows where A refuses and B (A plus the forbidding line) answers. It is never
re-estimated and never sees folds 2–4. Steer adds α·v_L (α in units of ‖v_L‖) at layer L at the ':' position and at
every generated token, on A's prompt. Steer-cut applies the same vector to the §34c evidence-cut prompt with the N-cut
arm's own note (9 confirmation rows have no N-cut marker and keep no note). A-cut is the same cut prompt unsteered.
Controls:
- X: a random vector with the same norm; ans(Random) ≤ 0.2 × ans(Steer), or Claim 1 is not interpretable;
- Reverse (−α·v_L), descriptive;
- C: damage on rows A already answers correctly;
- T: by question type;
- sanity S: overwriting A's ':' state with B's at layer 35 must give B's first token on ≥ 0.9 of pairs; a zero add and
  a self-overwrite must reproduce A byte-exact.

Outputs are classed as refusal (the §23 rule), degenerate (empty, no alphanumeric character, or a run of ≥ 8 identical
token ids), or answer. Claim 1 is P(gold(Steer)) − P(gold(Steer-cut)) on R, paired bootstrap over questions, 10,000
resamples; SHOWN if the lower bound > 0, meaningful if P(gold(Steer)) ≥ 0.30. Claim 1′ repeats it on the rows where the
gold fails the D1 string rule in both the cut prompt and the N-cut note (35 of 60).

**Stage 1, laptop design run (folds 0–1, fp32 CPU, 1.85 h, $0).** All 28 R rows reproduced. Sanity S passed (1.000 at
layer 35; 0.077 at layer 7). ‖v_L‖ / mean‖resid_A‖ was 0.25–0.31 at layers 18–32. Of the 15 settings (5 layers × α 1,
2, 4), the pre-registered rule (highest ans(Steer), Random passing, degenerate ≤ 0.2) chose layer 25, α 4: ans 0.964,
Random 0.179 against the bound 0.193. Its provisional D1 gold was 3/28, equal to Steer-cut and below B's 6/28; its
"answers" were mostly lists of topic words. Layer 29, α 2 had ans 0.929 and D1 gold 8/28. Amendment 2 added it as a
secondary setting, declared as chosen after seeing development labels; folds 2–4 stayed sealed.

**Stage 2, pod (A40, transformers 5.17 bf16 for the 3B and for the 7B judge; folds 2–4 read once with `--confirm`).**
Reproduction 0.973 over the R and C rows run (R 0.957; 60 of 64 confirmation R rows kept; 54 C rows). The pod's
directions match the laptop's (cosine 1.000 at layer 25, 0.999 at 29). Sanity S passed (1.000; layer 7 0.231,
descriptive). Judge: the same Qwen2.5-7B-Instruct with the §35 per-type prompts, on the HF engine instead of §35's vLLM.
It agreed with the cached §35 verdicts on the reproduced A and B rows at 0.971 (0.943, 0.994), above the 0.90 bar, and
at 1.000 on identical texts; it gave no non-verdict replies. Every gold figure below is this judge on this run's outputs.

| folds 2–4, 60 reproduced R rows | ans | gold | gold, 22 rows where B refuses |
|---|---|---|---|
| A (unsteered) | 0.000 | 0.000 | 0.000 |
| B (the §35 line) | 0.633 | 0.433 (0.317, 0.567) | 0.000 |
| A-cut | 0.067 | 0.017 | 0.000 |
| primary L25 α4: Steer / Steer-cut / Random | 0.950 / 0.683 / 0.283 | 0.150 / 0.100 / 0.033 | 0.000 / 0.045 / 0.045 |
| secondary L29 α2: Steer / Steer-cut / Random | 0.900 / 0.517 / 0.033 | 0.517 / 0.183 / 0.000 | 0.364 / 0.182 / 0.000 |

(The last column is the 22 rows where B itself refuses; no direction was estimated on them, and B's gold there is 0 by definition.)

| claim / control (folds 2–4) | primary L25 α4 | secondary L29 α2 (disclosed) |
|---|---|---|
| Control X: ans(Random) ≤ 0.2 × ans(Steer) | 0.283 > 0.190: FAIL | 0.033 ≤ 0.180: pass |
| Claim 1: gold(Steer) − gold(Steer-cut), n 60 | +0.050 (−0.067, +0.167): NOT INTERPRETABLE | +0.333 (+0.183, +0.483): SHOWN, meaningful (0.517 ≥ 0.30) |
| Claim 1′, clean subset, n 35 | +0.086 (−0.029, +0.200): NOT SHOWN | +0.429 (+0.257, +0.600): SHOWN |
| Control C: gold(Steer) − gold(A) on 54 C rows | −0.519 (−0.648, −0.389): damages correct answers | −0.056 (−0.130, 0.000) |
| Claim 2: gold(Steer) / gold(B) | not tested | 31/26 ≈ 1.19 (0.87, 1.68), descriptive |
| Reverse: ans on R; on C | 0.017; 0.185 | 0.000; not run |

By B's outcome on the 60 rows (new judge), secondary setting, gold under Steer / Steer-cut:
- B correct (26): 18 / 5;
- B wrong (12): 5 / 2;
- B refuses (22): 8 / 4.

(The report's Claim 2 conditionals split by the cached §35 B outcome instead and give 20/26 and 3/12; same 31 in total.)

Control T, secondary, gold counts Steer / Steer-cut / B:

| type | n | counts |
|---|---|---|
| temporal-reasoning | 20 | 12 / 5 / 11 |
| multi-session | 13 | 7 / 1 / 3 |
| single-session-user | 12 | 5 / 2 / 7 |
| knowledge-update | 6 | 3 / 1 / 2 |
| single-session-assistant | 4 | 3 / 2 / 2 |
| single-session-preference | 5 | 1 / 0 / 1 |

The primary's dev reading holds on folds 2–4: its outputs are mostly topic-word lists, and it wrecks about half the
correct answers. The secondary's correct outputs are short real answers, not candidate lists (review: 10.2 tokens and
0.42 separators on average, against 12.5 and 0.21 for the wrong ones). The 10 rows the judge marks correct while D1
fails are honest paraphrases. The 3 C rows it damages:
- "Veja" → "E-V-R (Eco-Vegan Rubber)…";
- "Roscioli" → "Pizzeria La Montecarlo";
- "Jaipur Rugs" → repeated "Rug-manufacturing industry".

**Found.** Pushing the 3B's internal state at layer 29, by the average "answer, don't refuse" difference from 13 other
questions, made it answer 9 of 10 of its refusals. 31 of the 60 rows came out right (0.517; 31 of the 54 answers), against 0.183 when the
evidence was cut from the prompt, and 0.433 for the §35 instruction line. A random push of the same size did almost
nothing. On the 22 questions where even the instruction line refused, the push got 8 right. The pre-registered setting
(layer 25, a push twice as large) did not work: it produced word salad, a random push also made the model "answer", and
it broke half the correct answers.

**Why it matters.** This is the first intervention in the record on the 3B's representation, not its prompt. A single
direction, with nothing question-specific in it, switches off the refusal, and the 3B then gets the gold mostly when the
evidence is in view. So on many of its "unknown"s the 3B can extract the answer; the refusal is a late disposition laid
over it. That gives the reading-vs-policy split of F6 its first causal support.

**Caveats, which bound the reading.**
1. Selection. The working setting was chosen on development labels, after the pre-registered ans-only rule had picked a
   setting that fails its specificity control. Folds 2–4 were sealed throughout and read once, but two settings were
   tested, and the result stands at the secondary setting only.
2. Mechanism. The add acts at the ':' token and at every generated token, so the generated tokens still attend to the
   evidence. Shown: one content-free direction at layer 29 switches off the refusal, after which the 3B extracts the gold
   from the evidence in view. It does so above the instruction line's rate and far above its rate without the evidence.
   Not shown: that the answer was already computed at the ':' position. That needs an add at step 0 only.
3. Narrow scope. One model, one benchmark, one prompt family, two layers. Control C at the secondary setting has n 54,
   with 3 damaged answers. Control T cells are small. The judge is one 7B, on a different engine from §35 (agreement
   0.971; no caveat line, since the bar was met).
4. A negative at other settings would not have shown the answer absent; this positive does not locate the circuit that
   decides to refuse.

**Next.** A free laptop variant on folds 0–1 adds the layer-29 vector at the ':' position only (step 0) and lets
generation run unsteered. It tells "the answer is already computed under the refusal" apart from "the refusal is lifted
and the model then reads the evidence". A fresh confirmation of the layer-29 setting, now fixed in advance, would remove
the selection caveat.

**Process note (attempt 1).** The first pod attempt died at the smoke stage, after about 2 minutes and with no rows
written: `scipy` was missing from the bundle's install line. It was installed by hand and the chain re-run, with its
deadlines shortened to stay inside the original hour; the install line is fixed (2263ca3). The stub end-to-end test had
run in the laptop venv, so it could not see a package missing from the pod's install line. Proposed to the user: run the
stub test from the extracted bundle in a fresh venv built from the bundle's own install line.

**Spend.** Laptop design run $0 (1.85 h). Pod ebfakbdujp2z7w ran 13 min at $0.59/h ≈ $0.13, including attempt 1; the
chain itself took 613 s. All experiments ≈ $17.68 of $20.

## 39. Is the answer already computed under the 3B's refusal? The §38 layer-29 direction at the ':' position only (2026-10-23, pre-registered; development only)

**Status: RUN, development only (folds 0–1, laptop, $0). Reading: M1 supported. The pushed ':' state carries the
answer path. Gold is provisional (D1 string rule; no judge ran). No claim on folds 2–4.** Pre-registration
`research/proposals/2026-10-23-exp39-one-position.md` (Advisor; rev 1 25d257f, amendment 1 7db734d with rulings on the
Executor's F1–F6). The user approved drafting, the build and the free laptop run (D-31, 25d257f). Reviews:
`research/reviews/2026-10-23-exp39-build.md` (ACCEPT, c895ed8) and `research/reviews/2026-10-23-exp39-dev.md` (ACCEPT,
980a06b). Code: `memctl/analysis/steering.py` (step-limited settings), `scripts/exp39_dev.py`, `scripts/exp39_report.py`
(28c908d); tests `tests/test_exp39.py` (16). Rows `runs/exp39_dev/qwen3b/`; report `runs/exp39_report_dev.md` (b743bd9).

**Why.** §38 added v_29 (α 2) at the ':' token of `Answer:` and at every generated token, and the 3B then answered and
got the gold mostly when the evidence was in view. Two mechanisms fit that.
- M1, answer in place: the pushed ':' state already carries the answer path.
- M2, disposition held off: the refusal comes back at each generated token unless the push is kept on.

§38's caveat 2 named the test: push at step 0 only (Q16).

**Design.** The 28 dev refusal rows R of §38 (13 estimation pairs, 15 unseen) and its 8 damage rows. The direction is
§38's layer-29 vector, reused byte-exact (`runs/exp38_dev/qwen3b/directions.npz` sha be1e3a09…, added vector 2·v_29
sha e0000366…, norm 64.47). It is not re-estimated. Arms, all at layer 29, α 2, max 16 tokens, greedy, laptop fp32:
- Steer-0 (primary): add at step 0 only, the forward pass at ':' that produces the first generated token;
- Steer-0..k, k ∈ {1, 2, 4}: add at steps 0..k, a dose curve;
- Steer-all: §38's rows, reused under a guard on engine, classifier, wording, vector sha and max tokens;
- Steer-0-cut and Steer-all-cut on §38's evidence-cut prompt;
- Random-0: the seeded layer-29 random vector of the same norm, at step 0 (sha 633bc895…; not the pod's layer-29
  random vector);
- Damage-0: Steer-0 on the 8 damage rows.

Readings, fixed in advance, with a0 = ans(Steer-0), aA = ans(Steer-all), and g0, gA their D1 gold counts:
- M1 if a0 ≥ 0.8·aA, g0 ≥ 0.8·gA and ans(Random-0) ≤ 0.2·a0;
- M2 if a0 < 0.5·aA;
- split if a0 ≥ 0.8·aA and g0 < 0.5·gA.

With gA = 8, M1 needs g0 ≥ 7 and the split needs g0 ≤ 3.

**Reproduction.** Flag rule 0.889, pass (R 28/28; the 4 damage rows dropped are §38's D1-paraphrase rows).
Byte-identity with the §38 dev rows: 64/64. Unsteered A and A-cut re-run in the arms stage matched on 28/28.

| 28 dev R rows | ans | D1 gold (provisional) | refusal |
|---|---|---|---|
| A (unsteered) | 0 | 0 | 28 |
| B (the §35 line) | 13 | 6 | 15 |
| Steer-all (§38) | 26 | 8 | 2 |
| Steer-0 | 26 | 7 | 2 |
| Steer-0..1 / 0..2 / 0..4 | 26 / 26 / 26 | 7 / 7 / 7 | 2 / 2 / 2 |
| Random-0 | 0 | 0 | 28 |
| A-cut (unsteered) | 4 | 2 | 24 |
| Steer-0-cut | 17 | 2 | 11 |
| Steer-all-cut | 17 | 3 | 11 |

Reading: a0 26 ≥ 21, g0 7 ≥ 7, Random-0 0 ≤ 5, so **M1 supported**. The row that sits on the gold bar is 89527b6b:
Steer-0 "The Plesiosaur has a blue scaly body in the image." against Steer-all "had …", gold "The Plesiosaur had a
blue scaly body." D1 accepts one and not the other. It is a tense paraphrase, and a judge would most likely give
g0 = gA = 8/28. Gold stays provisional. Damage-0: the 4 reproduced damage rows stay answered and gold, 4/4 (8/8 answered,
4/8 gold on all 8, as for A).

**Texts.**
- The first generated token is the same under Steer-0 and Steer-all on 28/28.
- Steer-0..k equals Steer-all on its first k + 1 tokens on 28/28 for each k, which confirms the step semantics.
- The whole text equals Steer-all on 11 / 13 / 16 / 19 of 28 at k = 0 / 1 / 2 / 4.
- The point where Steer-0 and Steer-all first differ is token 1 on 6 rows, 2 on 5, 3 on 2, 5 on 2 and 11–12 on 2; on
  the other 11 rows they never differ.
- So the dose curve is flat in counts, not in text: later pushes move the wording toward Steer-all without changing any
  class or D1 gold. Most differences are continuations ("21" against "21 days", "1998" against "1998 Views").

**Soft refusals (descriptive).** The v2 rule counts as answers some phrases that hedge: "not specified", "does not
contain information", "not provided", "not mentioned". A simple pattern finds 4 of Steer-0's 26, 3 of Steer-all's 26,
2 of Steer-0..4's 26 and 0 of B's 13. Removing them gives a0 22 against aA 23, and M1 still holds (5 × 22 = 110 ≥
4 × 23 = 92). Two of the four, 89941a94 and fca70973, are rows where Steer-0 hedges ("The memory does not contain
information…", "not provided") while Steer-all invents (a road bike, named parks).

**Found.** Pushing the 3B's state at one position, the ':' where it decides to refuse, does as much as pushing at every
generated token. It answers 26 of 28 refusals against 26, with provisional gold 7 against 8. That one-row gap is a
tense paraphrase. Pushing for 1, 2 or 4 more tokens changes no count, only wording. A random push of the same size at the
same position makes it answer 0 times. The pre-registered prediction (M2 or split, a dose curve rising with k) was wrong.

**Why it matters.** The refusal is decided at the answer position and can be lifted there. Once lifted, the 3B produces
the answer from its own computation on the evidence in view. With the evidence cut, the same push gives 2 of 28 gold.
This sharpens §38: the direction acts on a disposition at that one position; whether the §35 instruction line acts at the same point was not tested.

**Caveats.**
1. The F1 caveat, as amended. The pushed ':' key/value above layer 29 stays in the cache, so every later token can
   attend to the pushed state. "One position" means one input position, not one moment. This run does not separate "the
   first token commits the model" from "later tokens keep reading the pushed ':' state".
2. Development only: 28 rows, the same folds the direction was estimated on (13 of the 28 are estimation pairs). Gold
   is the D1 string rule, and the gold verdict sits on the bar by one row.
3. One layer, one α, one model, one prompt family.

**Next.** A first-token-only arm would generate token 0 under the push, then continue from the unpushed cache with that
token forced. That separates the two readings of caveat 1, free on the laptop; the Advisor will put it to the user. A
sealed-fold run of §39 would be a separate quote (about $0.15 on an A40).

**Spend.** $0 (laptop; 128 prefills, 324 generations). All experiments ≈ $17.68 of $20.

### 39b. Does the first generated token carry the effect, or the pushed ':' state? A first-token-only clean-cache control (2026-10-23, pre-registered as §39 amendment 2; development only)

**Status: RUN, development only (folds 0–1, laptop, $0). Reading: token carries it. The refusal is a one-token
decision; the token the push chooses is the answer's own first token, not a neutral one. Gold is provisional (D1).
No claim on folds 2–4.** Spec: §39 amendment 2 (3e7c349, D-33 item 1). Reviews:
`research/reviews/2026-10-23-exp39b-build.md` (ACCEPT, 64ac039) and `research/reviews/2026-10-23-exp39b-dev.md` (ACCEPT,
7d4e037). Code: `memctl/analysis/steering.py` (`run_forced`, which refuses if any hook is registered),
`scripts/exp39b_dev.py`, `scripts/exp39b_report.py` (68ac539, soft-refusal pattern 8bc073c); tests `tests/test_exp39b.py`
(18). Rows `runs/exp39b_dev/qwen3b/`; report `runs/exp39b_report_dev.md` (24eded6).

**Why.** §39's caveat 1: Steer-0 leaves the pushed ':' key/value above layer 29 in the cache for every later token, so
§39 could not separate "the first token commits the model" from "later tokens keep reading the pushed state".

**Design.** Same 28 dev R rows, engine and wording as §39. Steer-first (primary): the first generated token of §39's
Steer-0 row is forced as the first output token on an unpushed cache (':' run unpushed). Greedy continuation follows
with no hook at any step, so no pushed state exists anywhere. Beside it:
- Steer-first-cut: the same on the evidence-cut prompt, with Steer-0-cut's first token;
- Gold-first: the gold's first token forced, a descriptive ceiling;
- Unknown-first: ' unknown' forced, which must equal A byte-exact;
- a step-0 check: Steer-0 for one token must give §39's first token.

Readings, fixed in advance, with aF = ans(Steer-first) and gF = its D1 gold, against a0 = 26 and g0 = 7:
- token carries it if aF ≥ 21 and gF ≥ 6;
- state needed if aF ≤ 12;
- split if aF ≥ 21 and gF ≤ 3.

**Checks.**
- Reproduction: flag rule 0.889 (R 28/28; the same 4 D1-paraphrase damage drops as §38–§39); byte-identity 64/64.
- Unknown-first equals A byte-exact on 28/28, and the unpushed ':' argmax is ' unknown' on 28/28.
- The step-0 check gives §39's first token on 28/28.

| 28 dev R rows | ans | D1 gold (provisional) | refusal |
|---|---|---|---|
| A (unsteered) | 0 | 0 | 28 |
| Steer-0 (§39, pushed ':' state kept) | 26 | 7 | 2 |
| **Steer-first (clean cache, forced first token)** | **26** | **7** | 2 |
| Steer-0-cut (§39) / Steer-first-cut | 17 / 17 | 2 / 2 | 11 / 11 |
| Gold-first | 28 | 13 | 0 |
| Unknown-first | 0 | 0 | 28 |

Reading: aF 26 ≥ 21 and gF 7 ≥ 6, so **token carries it**.

**Texts.**
- Steer-first's text equals Steer-0's on 26/28. The two that diverge, 51b23612 at token 2 and b6025781 at token 1, change
  neither class nor gold.
- Steer-first-cut equals Steer-0-cut on 28/28.
- Soft refusals (the §41 rev 1 pattern): Steer-first 4/26, Steer-0 4/26, Steer-first-cut 1/17, Gold-first 2/28.

**Gold-first (descriptive).**
- Forced tokens: ' ' on 7 rows (numbers), ' The' on 6, ' a' on 2, ' I' on 2, and a content-bearing first word on the
  other 11 (' Nu', ' Japanese', ' International', ' Michael'…).
- It answers on 28/28, with gold 13/28 (5/15 unseen).
- The content-bearing tokens give part of the answer away, so 13/28 is an upper bound helped by the forced token.
- Generic starts (' The', ' I') often lead to hedges or off-topic text.

**Found.** Removing the pushed state entirely changes almost nothing. One forced first token on a clean cache reproduces
the pushed run almost token for token: 26 answers and 7 gold, identical text on 26 of 28. The push's whole effect passes
through its choice of the first generated token. After that token, the unpushed 3B continues on its own, from the
evidence when it is in view. The pre-registered prediction (split or undetermined, gF ≤ 4) was wrong.

**Why it matters.** The 3B's refusal is a one-token decision at the answer position: ' unknown' or the answer's first
token. The token the push chooses is not content-free. It is usually the answer's own first word ("Serenity" for
"Serenity Yoga") or ' ' before a number. So what the layer-29 push supplies is the choice of a non-refusal first token
that the model's own computation already ranks below ' unknown'. The content then follows without the push.

**Caveats.** Development only: 28 rows, 13 of them the direction's estimation pairs, the same folds the direction came
from. Gold is the D1 string rule. One layer and one α. That the forced token is "the answer's own first token" is read
off the texts, not measured as a rank.

**See §40.** On LoCoMo's 120 refusal rows, the first token under Steer-0 and Steer-all is the same on 120/120, but
Steer-all is right more often: gold 43 against 34, paired +0.075 (+0.017, +0.142). The refusal decision is still the
first token; keeping the push on improves the content there.

**Spend.** $0 (laptop; 176 prefills, 204 generations). All experiments ≈ $17.68 of $20.

## 40. Does the fixed layer-29 push transfer to a set it has never seen? The §38 setting on LoCoMo (2026-10-23, pre-registered; pod run 2026-10-23)

**Status: RUN and CONFIRMED on a new set. Claim 1 SHOWN, +0.333 (+0.250, +0.425). Record: the §38 setting, fixed in
advance, on a set it had never seen.** Pre-registration
`research/proposals/2026-10-23-exp40-fixed-setting-validation.md` (Advisor; rev 0 3e7c349, rev 1 877f86d after the
Executor's feasibility note, rev 2 61f9f6b after attempt 1). Decisions: D-33 (programme), D-34 (pod approval at $0.59;
the user's yes typed in the Executor pane for attempt 1, and again for attempt 2). Reviews:
`research/reviews/2026-10-23-exp40-build.md` (ACCEPT, 456dd11) and `research/reviews/2026-10-23-exp40-results.md`
(ACCEPT, 8b12f27). Code: `memctl/locomo_view.py`, `memctl/analysis/exp40.py`, `scripts/exp40_pod_run.py`,
`scripts/exp40_pod.sh`, `scripts/exp40_bundle.sh`, `scripts/exp40_d30.sh`, `scripts/exp40_report.py` (d4f7fc3,
64 tokens c4df0d1, §41 block 3c5f9d2, rev 2 smoke 7b37aba); tests `tests/test_exp40.py` (24). Rows `runs/exp40_pod/`;
report `runs/exp40_pod/exp40_report_confirm.md` (08fc63e); attempt-1 logs `runs/exp40_pod_attempt1/` (78f2c07).

**Why.** §38's working setting (layer 29, α 2) was chosen on development labels and confirmed once, on folds 2–4 of the
same prompt family. §40 asks whether the same intervention, with nothing re-chosen, lifts the 3B's refusals into
correct answers on data it has never touched.

**Design.**
- Direction and setting: v_29 from the laptop `directions.npz` byte-exact (sha a7f967ec…), 2·v_29 added at the ':' of
  `Answer:` and at every generated token, cast to bf16 (cosine 0.999999). The script accepts layer 29 and α 2 only.
- New set: LoCoMo (`data/locomo/locomo10.json`), on which no LLM had run in this programme. 600 answerable questions,
  drawn after excluding 17 with no, too many (> 8) or unresolved evidence turns, in a seed-40 shuffle (category 1 99,
  2 114, 3 34, 4 353), and 40 adversarial (category 5) questions.
- Prompt: the §23 fixed8 template, the memory being every evidence turn plus a BM25 fill to 8 lines from the same
  conversation, in arrival order, as "speaker, date: content" lines with the session date and photo captions as
  "[shares a photo: …]". The 3B writes its own note (batched greedy, 256 tokens); A = frozen note + `Answer:`; B adds
  §35's "do not reply unknown" line.
- Selection, from A outputs and the judge only, before any steered arm: R = the first 120 marked A refusals, C = the
  first 40 marked A answers the judge marks correct. Arms on R: A, B, Steer-all, Steer-all-cut (evidence ablated, own
  N-cut note), A-cut, Random (seeded rng [40, 29], same norm), Steer-0 (descriptive). On C: Steer-all.
- Judge: Qwen2.5-7B-Instruct, HF bf16, the LongMemEval per-type prompts mapped by category (4 single-session-user,
  1 multi-session, 2 temporal-reasoning, 3 default). Its agreement with 60 frozen cached §35 verdicts was checked
  before any arm (bar 0.90).
- Engine: transformers 5.17.0 bf16 on one A40, greedy, 64 new tokens, newline/EOS stop.

**Attempts.**
- Attempt 1 (pod b0qnmah0utqvh2, about 3 min, about $0.03) stopped at the D-22 smoke. The rev-1 check required an
  `Answer:` marker in at least 4 of 5 batched notes, and only 3 of 5 had one. Batched and single-sequence notes agreed
  on all 5 rows, so it was not batching drift; the 3B skipped the format on 2 rows and answered in one sentence.
- Rev 2 made the smoke a batching-drift check only (same first token, and the same marker outcome batched vs single,
  on ≥ 4 of 5) and printed the absolute marker rate.
- Attempt 2 (pod 0jurb616nv389v, about 12 min, about $0.12) ran to the end in 653 s with nothing cut. Smoke: first
  token 5/5, marker agreement 5/5 (absolute 3/5).

**Population.**
- On the 600: marker rate 0.737 (442), refusal rate 0.207 (124), judged-correct rate 0.347 (208). R 120 and C 40, both
  at cap.
- R and C are marked rows only, as in §35 and §38. The 3B skips the note-and-`Answer:` format on 26% of LoCoMo rows,
  a finding about the prompt family on this set.
- Judge agreement on the 60 cached rows: 60/60.

| 120 R rows | ans | gold (95% CI) |
|---|---|---|
| A (unsteered) | 0.000 | 0.000 |
| B (the §35 line) | 0.400 | 0.217 (0.150, 0.292) |
| Steer-all | 0.758 | 0.358 (0.275, 0.450) |
| Steer-all-cut | 0.525 | 0.025 (0.000, 0.058) |
| A-cut | 0.075 | 0.017 |
| Random | 0.058 | 0.017 |
| Steer-0 (descriptive) | 0.758 | 0.283 (0.200, 0.367) |

| claim / control | estimate | verdict |
|---|---|---|
| Claim 1: gold(Steer-all) − gold(Steer-all-cut), n 120 | +0.333 (+0.250, +0.425) | SHOWN, meaningful (0.358 ≥ 0.30) |
| Control X: ans(Random) ≤ 0.2 × ans(Steer-all) | 7 rows ≤ 18.2 (0.058 ≤ 0.152) | passes |
| Control C: gold(Steer-all) − gold(A) on 40 C rows | −0.200 (−0.325, −0.075) | a 20-point loss on correct rows, below the pre-registered harm bar (UB < −0.10) by a small margin |
| Claim 2 (descriptive): gold(Steer-all) − gold(B) on R | +0.142 (+0.058, +0.225) | descriptive |

By type, gold Steer-all / Steer-all-cut / B:

| type | n | Steer-all | Steer-all-cut | B |
|---|---|---|---|---|
| single-hop | 47 | 0.511 | 0.021 | 0.340 |
| multi-hop | 25 | 0.320 | 0.040 | 0.280 |
| temporal | 28 | 0.250 | 0.000 | 0.107 |
| open-domain | 20 | 0.200 | 0.050 | 0.000 |

Soft refusals (§41 rev 1 pattern) among the answers: Steer-all 0.050, Steer-all-cut 0.117.

**Steer-all against Steer-0 (a qualification of §39 and §39b).** On the 120 R rows:
- the first generated token is the same under both on 120/120, and the whole text on 53/120;
- both answer on 91 rows;
- gold is 43 against 34: paired +0.075 (+0.017, +0.142).

On LoCoMo the refusal decision is still the first token at ':', but keeping the push on during generation improves the
content by about 7 points. §39's "one position does as much" holds for whether the model answers, not, on this set, for
how often it is right.

See §42: on fresh LoCoMo rows the gap did not replicate (Steer-all − Steer-0 on gold +0.017 (−0.042, +0.075), first token
identical 120/120).

**Costs.**
- Correct rows: on 40 rows the 3B already had right, the push loses 8 (−0.200). The pre-registered bar was set when §38
  showed −0.056, and it is missed by 0.025.
- Unanswerable questions: on the 37 marked adversarial rows, the answer rate goes from 0.324 (A) to 0.730 (Steer-all),
  against 0.541 for the instruction line. The push removes abstention where abstention is right.
- Net: at +1 per correct and −1 per wrong answer on the refusal rows, unconditional steering scores −1 against the
  instruction line's +4 (the §41 block below).

**Found.** The §38 setting, fixed in advance with nothing chosen on LoCoMo, works on a set and a conversation style it
never saw. It opens 76% of the 3B's refusals, and 36% of them come out right, against 2.5% when the evidence is cut
from the prompt. A random push of the same size opens 6%. It beats the instruction line by 14 points. What comes out is
read from the evidence in view.

**Why it matters.** The answer-vs-refuse direction is not an artefact of the LongMemEval prompts or of the rows it was
chosen on. The 3B's refusals on LoCoMo hide answers it can extract once the refusal is lifted. But the push is blunt.
It also breaks a fifth of the correct answers and makes the model answer questions it should refuse, so as an
unconditional policy it does not beat the instruction line once wrong answers count.

**Caveats.**
1. One model, one layer, one α, one prompt family.
2. R and C are marked rows only (marker rate 0.737).
3. The judge is the same 7B that labelled §35 (agreement 60/60 on cached rows). LoCoMo golds are short strings, and the
   judge prompt is the LongMemEval one, mapped by category.
4. 120 refusal rows, 40 correct rows, 37 adversarial rows.

**Spend.** Attempt 1 ≈ $0.03, attempt 2 ≈ $0.12. All experiments ≈ $17.82 of $20.

## 41. Can a signal at ':' say when steering will help? Conditional steering (2026-10-23, pre-registered; development fit on the laptop, confirmation on §40's pod)

**Status: RUN. The frozen development rule failed on §40 (Claims 1–3 NOT SHOWN at λ 1). Claim 4 SHOWN: the projection
of the unsteered ':' state on the direction ranks which opened refusals come out right, AUROC 0.803 (0.724, 0.875). A
signal exists; the development fit did not find it.** Pre-registration
`research/proposals/2026-10-23-exp41-conditional-steering.md` (Advisor; rev 0 3e7c349, rev 1 877f86d, amendment 1
0bda490). Reviews: `research/reviews/2026-10-23-exp41-build.md` (ACCEPT, 288b93f),
`research/reviews/2026-10-23-exp41-dev.md` (ACCEPT of the frozen rule, 0bda490) and the §41 block of
`research/reviews/2026-10-23-exp40-results.md` (8b12f27). Code: `memctl/analysis/conditional.py`, `scripts/exp41_dev.py`,
`scripts/exp41_report.py` (73c3371); tests `tests/test_exp41.py` (19). Dev rows `runs/exp41_dev/qwen3b/`; frozen rule
`runs/exp41_dev/qwen3b/rule.json` (30f891a, sha af0cbe98…); dev report `runs/exp41_report_dev.md`.

**Why.** Unconditional steering opens almost every refusal, but about half the opened answers are wrong. Can a signal
read off the unsteered forward pass at ':' say, row by row, when steering will produce a correct answer?

**Design.** Three signals from unsteered prefills:
- s_margin: the ' unknown' logit (id 9788, the only refusal start in the §38 dev rows) minus the best other logit at ':';
- s_proj: the projection of the layer-29 residual at ':' on v_29;
- s_evid: s_margin with the evidence in view minus s_margin with it cut.

Utility per row: +1 correct, −λ wrong, 0 for a refusal, degenerate output or soft refusal; λ = 1 primary, λ = 2
beside. One (signal, sign, τ) was fitted on 67 development rows (R 28, C 39, C weighted to C:R = 1:3) and frozen by
commit before §40. Development gate: Cond ≥ Steer + 3 weighted units. Dev labels: the cached §35 verdict where a text
equals a cached A or B answer, D1 otherwise.

**Development fit (laptop, folds 0–1, $0).**
- Weighted utility at λ 1: A 9.094, B 11.855, Steer −0.060, Cond 9.855.
- Frozen rule: s_evid > 2.218, firing on 6 of 67. The gate passes (+9.915 ≥ 3).
- The dev utility is optimistic by construction (the best of 6 families × about 70 thresholds on the same rows). The
  gate was nearly vacuous because Steer scored about 0 under D1 labels.
- Amendment 1, written before any §40 row was read, added Claim 3 (Cond − A) and Claim 4 (AUROC of s_proj), and a
  reading rule for Claim 1.

**Confirmation on §40's 120 R rows (7B judge).**

| | λ 1 | λ 2 |
|---|---|---|
| U: A / B / Steer / Cond | 0 / +4 / −1 / +1 | 0 / −18 / −44 / −2 |
| Claim 1: Cond − Steer | +0.017 (−0.125, +0.158): NOT SHOWN | +0.350 (+0.133, +0.567): SHOWN, but explained by a net-negative steer (Steer − A −0.367) |
| Claim 2: Cond − B | −0.025 (−0.142, +0.092): NOT SHOWN | +0.133 (−0.042, +0.308): NOT SHOWN |
| Claim 3: Cond − A | +0.008 (−0.033, +0.050): NOT SHOWN | −0.017: NOT SHOWN |

At λ 1, U(Steer) − U(A) = −0.008, so the pre-registered reading applies to Claim 1: "expected under a net-negative
steer; not evidence of a selective signal".

- The rule fires on 11 of 120. Gold among the fired rows is 0.364, against Steer-all's overall 0.358, so it does not
  select.
- Adversarial answer rate under Cond: 0.324, the same as A.
- **Claim 4:** the AUROC of s_proj for gold(Steer-all) on R is 0.803 (0.724, 0.875), SHOWN. Beside it, s_margin 0.185
  (0.111, 0.269) and s_evid 0.232 (0.140, 0.332), both ranking the other way.

**Post hoc (descriptive; a hypothesis, not a result).** Steering only the top fraction q of R rows by s_proj would have
scored, at λ 1, +9 for q = 0.1–0.3, against +4 for B and −1 for Steer; at λ 2, +8 at q = 0.1. This is a hypothesis for
a pre-registration on another new set.

**Found.** The development rule (s_evid) is useless on new data. Its few fires are right at the base rate. One of the
logged signals, how far the 3B's unsteered state already lies along the push direction, ranks well which refusals the
push will turn into correct answers (0.80). A signal exists; the development fit did not find it, because 67 rows with
string-match labels chose the wrong signal.

**Why it matters.** It answers "can the model's own state say when to push?" with yes for ranking, not yet as a policy.
A threshold on s_proj, fixed in advance, is the next test, on rows neither §40 nor the fit has seen.

**Caveats.** The development labels were D1 for new texts, which biased the fit against steering. Claim 4 has 120 rows.
One model, one layer, one direction. The top-q curve is post hoc.

**Spend.** Development $0 (laptop); confirmation rode on §40's pod at no extra cost.

## 42. Can a threshold on the projection signal choose which refusals to push? Selective steering on fresh LoCoMo rows (2026-10-25, pre-registered; pod run 2026-10-26)

**Status: RUN. The selective policy fails: Claims 1–3 NOT SHOWN at λ 1 (U: Cond −6, Steer −6, A 0, B +1). Claim 4
replicates: AUROC of s_proj 0.765 (0.678, 0.843). Q17 closes negative for s_proj as a policy.** Pre-registration
`research/proposals/2026-10-25-exp42-selective-steering.md` (Advisor; rev 0 b2e0543, rev 1 8fd2613 after the Executor's
feasibility note). Decisions: D-36 (draft and build), D-37 (pod approval; the user's "Yes launch please" in the Executor
pane). Reviews: `research/reviews/2026-10-25-exp42-build.md` (ACCEPT, 604cc73) and
`research/reviews/2026-10-26-exp42-results.md` (ACCEPT, cab1a25). Code: the §40 pipeline with `--exp exp42`
(`memctl/locomo_view.py` seed and exclusion parameters, `scripts/exp40_pod_run.py`, `scripts/exp40_report.py`),
`scripts/exp42_rule.py` (e2c9798); tests `tests/test_exp42.py` (15). Rule `runs/exp42_inputs/rule42.json` (sha
7a6afa62…); rows `runs/exp42_pod/`; report `runs/exp42_pod/exp42_report_confirm.md` (909626a).

**Why.** §41 found that s_proj, the projection of the unsteered ':' state on v_29, ranks which steered refusals come out
right (AUROC 0.803 on §40's rows). A post hoc cut at the top 10–30% would have beaten the §35 line. §42 fixes that cut
in advance and tests it on rows nothing has seen (Q17).

**Design.**
- Frozen rule, fixed from §40's rows before any new row was read: Cond = steer (Steer-all, layer 29, α 2) iff A refuses
  (the step-0 argmax on the unsteered pass is ' unknown', id 9788) and s_proj ≥ τ; otherwise A's output.
  τ = −18.896762090918696, the numpy linear 70th percentile of s_proj over §40's 120 R rows (apass sha 388e698b…).
- The rule was restricted to refusals by the Executor's feasibility note (F1). On §40's rows, s_proj ≥ τ held on 242
  of the 318 rows where A already answers, including 33 of 40 correct rows. Firing there would have made the damage
  control fail by design.
- Fresh set: LoCoMo, seed 42, with §40's 600 + 40 question ids excluded before the shuffle. Disjointness and the pools
  (923 answerable, 406 adversarial) are asserted. 600 + 40 taken; by category 1 113, 2 132, 3 32, 4 323. This is
  in distribution (same benchmark, disjoint rows); a cross-benchmark test is a separate step.
- Pipeline, arms, judge and selection as §40 (R = the first 120 marked A refusals, C = the first 40 marked A answers
  judged correct).
- Utility: +1 correct, −λ wrong, 0 for a refusal, degenerate output or soft refusal; λ 1 primary, λ 2 beside; paired
  bootstrap 10k.

**Population.**
- On the 600: marker rate 0.772 (463), refusal rate 0.212 (127), judged-correct rate 0.378 (227). R 120, C 40.
- Judge agreement on the 60 cached rows: 60/60.
- All 120 R rows refuse at step 0.

**The policy (λ 1; the rule fires on 20 of 120 R rows).**

| claim | estimate (95% CI) | verdict |
|---|---|---|
| U: A / B / Steer-all / Cond | 0 / +1 / −6 / −6 | |
| Claim 1: U(Cond) − U(B) | −0.058 (−0.167, +0.042) | NOT SHOWN |
| Claim 2: U(Cond) − U(A) | −0.050 (−0.125, +0.025) | NOT SHOWN |
| Claim 3: U(Cond) − U(Steer-all) | 0.000 (−0.117, +0.117) | NOT SHOWN (beside, U(Steer) − U(A) −0.050) |
| Claim 4: AUROC(s_proj) for gold(Steer-all) on R | 0.765 (0.678, 0.843) | SHOWN (replicates §41's 0.803) |
| Control D: gold(Cond) − 1 on C | 0.000; 0 of 40 C rows fire | 0 by construction |
| Control U: answer rate on 36 adversarial rows | Cond 0.583 (5 fire); A 0.444, B 0.556, Steer-all 0.917 | a cost |

- At λ 2 (beside), U is A 0, B −15, Steer −44, Cond −19.
  - Claim 3 is SHOWN, +0.208 (+0.033, +0.392). The fixed reading applies: "expected under a net-negative steer; not
    evidence of a selective signal" (U(Steer) − U(A) −0.367).
  - Claims 1 and 2 are NOT SHOWN (Cond − A −0.158).
- Fire precision: the 20 fired rows come out 7 correct and 13 wrong (0.350), against Steer-all's 0.275 over all R. That
  is above the base rate but below the 0.5 a λ 1 policy needs.
- The frozen q curve (τ_q from §40's R rows; only q = 0.3 carries a verdict) at λ 1 reads −3, −3, −6, −5, −2, −4, −1,
  −3, −4 for q = 0.1 … 0.9. A post hoc top-q curve on these rows is negative throughout. No frozen or post hoc threshold
  is positive.
- By type: Cond fires 14 times on single-hop rows and scores −6 there, against B's +7.

**Mechanism: s_proj ranks extractability, not answerability.**
- On the 36 marked adversarial rows, the rule fires on 5 of the 20 step-0 refusals (0.25). On the answerable rows it
  fires on 20 of 127 (0.157).
- The AUROC of s_proj for answerable-refusal against unanswerable-refusal is 0.443.
- So the signal is no higher when an answer exists; it tracks how ready the state is to produce an answer-shaped
  continuation. That is why Control U moves the wrong way, and it is consistent with §34c, where the 3B's late state
  carried a plausible candidate with or without the evidence.

**§40's readouts on the fresh rows (an unplanned replication).**

| 120 R rows | ans | gold |
|---|---|---|
| A | 0 | 0 |
| B | 0.275 | 0.142 |
| Steer-all | 0.625 | 0.275 (0.200, 0.358) |
| Steer-all-cut | 0.442 | 0.042 |
| A-cut | 0.067 | 0.017 |
| Random | 0.050 | 0.008 |
| Steer-0 | 0.625 | 0.258 |

- §40's Claim 1 replicates: +0.233 (+0.150, +0.317). Steer-all's gold of 0.275 is below the rev 1 "meaningful" line of
  0.30.
- Control X passes (0.050 ≤ 0.125). Steer-all − B is +0.133 (+0.050, +0.217).
- Control C: −0.225 (−0.350, −0.100). This is a tie on the bar: the upper bound is exactly −0.100, and the rule is
  "harmful if UB < −0.10". Pooled over §40 and §42 (80 rows; descriptive, not pre-registered) it is −0.2125 (−0.300,
  −0.125).
- Steer-all − Steer-0 on gold is +0.017 (−0.042, +0.075), with the first token identical on 120/120. §40's +0.075
  qualification of §39/§39b did not replicate on these rows.
- By type, Steer-all gold: multi-hop 0.429 (n 21), single-hop 0.362 (47), open-domain 0.250 (20), temporal 0.062 (32).

**Found.** A threshold on s_proj, fixed in advance from §40's rows, does not make steering pay on fresh rows. The
refusals it picks come out right 35% of the time once pushed. That beats the 28% base rate, but loses to saying
"unknown" once each wrong answer costs as much as a right one gains. The ranking itself replicates (0.765). The push's
evidence-dependent effect replicates too, at a smaller size (+0.23 against §40's +0.33), with the same costs: about 22
points on correct rows, and the answer rate on unanswerable questions roughly doubled.

**Why it matters.** The 3B's unsteered state does tell which refusals hide an extractable answer, but not sharply
enough to act on: at its best, the push is wrong more often than right. The signal ranks extractability rather than
whether an answer exists, so it cannot protect the questions the model should refuse. For the 3B, Q17 closes negative:
s_proj is a ranking, not a policy.

**Caveats.** One model, one layer, one α; the same benchmark as §40, disjoint rows; 120 refusal rows, 40 correct rows,
36 marked adversarial rows. The 7B judge (60/60 on cached rows). Marked rows only (marker rate 0.772).

**Spend.** About $0.12 (12.4 min at $0.59/h). All experiments ≈ $17.94 of $20.

## 43. Does the refusal-direction result hold in other model families? Llama 3.2, Gemma 3, Phi-4 (2026-10-26, pre-registered; Part B done 2026-10-29)

**Status: Part A: Llama deferred, Gemma and Phi closed at stage 0. Part B: done.**
- **Part A (the refusal direction):** closed at stage 0 for Gemma 3 4B and Phi-4-mini as "little to open". Llama 3.2 3B
  passes its stage-1 gate on the bars, on development data only, and its Part A pod is not recommended.
- **Part B (the lens readout): DONE.** Llama (B1 and B2 SHOWN), Gemma (B1 NOT SHOWN, B2 SHOWN, J-lens gate FAIL), Phi
  (B1 and B2 SHOWN). The Llama Part A pod stays deferred under the user's scope rule (no new interventions beyond §44).

Pre-registration `research/proposals/2026-10-26-exp43-cross-family.md` (Advisor; rev 0, amendment 1 21e4e35, rev 1
9490d07). Decisions: D-39 (the user's order: "run it on at least 2 more model families") and D-40 (cap raised to $30;
Part B added). Reviews:
- `research/reviews/2026-10-26-exp43-build.md` (c7157f9, addendum e39bd62);
- `research/reviews/2026-10-26-exp43-llama-stage1.md` (66ecf2a);
- `research/reviews/2026-10-26-exp43-partB-build.md` (67f05fe);
- `research/reviews/2026-10-27-exp43-partB-llama-results.md` (ACCEPT, f951bf5).

Code: `memctl/analysis/families.py` (a per-family config beside the Qwen defaults; the Qwen outputs are byte-identical),
the lens BOS fix in `memctl/analysis/jlens_readout.py` (ccaff40), `scripts/exp43_llama_dev.py`, `scripts/exp43_llama_report.py`
(6342460, 511eb32), `scripts/exp43_lens_dev.py` and `scripts/exp43_lens_profile.py` (7c74932, 3698bc1),
`scripts/exp43_lens_pod.sh`, `scripts/exp43_lens_pod_run.py` and `scripts/exp43_lens_report.py` (7444d11).

**Why.** §38–§42 are one family: Qwen 2.5 3B, judged by Qwen 2.5 7B. The question is whether the refusal disposition and
the lens picture belong to small instruction-tuned readers or to Qwen.

**Stage 0: feasibility (laptop, $0, folds 0–1).** It used §24's cached `fixed8` outputs and judge labels for each
family. Every model is accessible on HF, including Llama 3.2 3B, which returned 403 in §24. The three small models load
in fp32 on the laptop and reproduce the §24 cache (A arm first token ≥ 4/5 for each). In-view refusal rate among marked
rows, on the 126 dev population rows (the §35 rule):

| model | in-view refusals | marker rate | judged correct |
|---|---|---|---|
| Qwen 2.5 3B (reference) | 0.300 (30/100) | 1.00 | 0.390 (39/100 marked rows) |
| Llama 3.2 3B | 0.104 (13/125) | 0.992 | 0.563 (71/126) |
| gemma-3-4b | 0.017 (2/121) | 0.960 | 0.659 (83/126) |
| Phi-4-mini | 0.034 (3/89) | 0.706 | 0.468 (59/126) |

On the same 126 rows Qwen 3B is judged correct on 54/126 = 0.429, so the four readers compare on one denominator.

All three are under the pre-registered ~15% line. **Gemma and Phi close Part A at stage 0 as "little to open"**, a
result in itself. The Qwen 3B habit of refusing with the evidence in view is not shared at this size: the other
families answer and get more right. Llama's high §24 unknown share came from rows where the evidence was not all in
view (0.219 over all rows).

**Llama Part A, stage 1 (laptop, $0, development only).**
- The direction was estimated from 8 pairs: dev rows with the evidence not all in view, where A refuses and the §35
  line answers.
- It was gated on the 13 in-view dev refusals the estimation never saw. Every row reproduced (43/43).
- Bars: Steer answers ≥ 7, gold(Steer) − gold(cut) ≥ 2, Random ≤ 0.2 × Steer, degenerate ≤ 1.
- The gate passes on the bars at layer 17, α 2: 7/13 answers, gold 2 against 0 with the evidence cut, Random 0/13.
- At layer 25 the direction opens 9–13 of 13, but a random vector of the same norm opens 2–4, the same specificity
  failure as Qwen's pre-registered §38 setting.
- The gold half is at its ceiling. Llama's own cached note holds the gold on 2 of the 13 gate rows, and those are
  exactly the 2 rows the push gets right; no setting gets a row right whose note lacks the gold. So the pass does not
  show that the direction lets Llama read the evidence.
- For contrast, Qwen 3B's notes held the gold on 7/28 dev and 23/60 pod refusal rows, and its §38 push got 31 of the 60
  right.
- A Llama Part A pod is not recommended. If run, its Claim 1 would be re-framed as the answer rate (Steer − Random),
  with gold beside the note-gold ceiling.

**Laptop logit-lens profiles (dev folds 0–1, 126 rows, logit lens only; descriptive).** Median emergence depth of the
gold (first layer in the top-10 at P1, depth = layer / n_layers) and the final-layer top-10 rate:

| model | layers | median emergence | final top-10 | A / B / C, all failures |
|---|---|---|---|---|
| gemma-3-4b | 34 | 0.647 | 0.83 | 0.163 / 0.186 / 0.651 |
| Llama 3.2 3B | 28 | 0.750 | 0.80 | 0.382 / 0.000 / 0.618 |
| Qwen 2.5 3B | 36 | 0.917 | 0.68 | 0.319 / 0.042 / 0.639 |
| Phi-4-mini | 32 | 0.938 | 0.78 | 0.299 / 0.015 / 0.687 |

- Gemma's apparent B class is mostly a target-token artefact. 9 of its 12 B rows have an article-first gold (' The',
  ' A', ' a'): the final layer predicts the content word, and the article drops out of the top-10. On content-first
  golds Gemma's B is 0.027, like the others'.
- Llama and Phi put up to three digits in one token, while Qwen and Gemma use one, so the "gold first token" is not like
  for like across families on digit-first golds (53 of 126).

**Llama Part B pod (A40, 62 min, about $0.61).**
- Logit reads at P1/P0/E/Q for Llama 3.2 3B and Llama 3.1 8B on the 330 §34-population rows (paired, the same prompts
  and judge labels for both sizes; 126 dev, 204 confirmation).
- A J-lens fitted on the 3B (56 paragraphs), plus its reads and the library evaluation sets.
- The J-lens gate on folds 0–1 passes: (a) logit − J-lens +1.458 (+1.285, +1.623); (b) pass@k 0.270 against 0.217. The
  J-lens row therefore takes its verdict role beside the logit row, which decides.
- Confirmation folds 2–4 were read once, after the development report was committed.

| claim (folds 2–4, 204 rows) | estimate (95% CI) | verdict |
|---|---|---|
| B1: log10 best rank of the gold at P1, 3B − 8B (positive = 8B better) | +0.181 (+0.106, +0.259) | SHOWN |
| B1 secondary: share of outcome A, 3B − 8B | +0.078 (+0.029, +0.132) | SHOWN |
| B2: AUC of the 3B's P1 log10 best rank, wrong against right (logit) | 0.677 (0.605, 0.746) | SHOWN |
| B2, J-lens (verdict role, gate PASS) | 0.682 (0.610, 0.751) | SHOWN |

The development-fold B2 (0.847) is higher; the confirmation figure is the one of record.

A/B/C on failures, logit lens:

| reader | n | A | B | C |
|---|---|---|---|---|
| Llama 3B | 95 | 0.274 | 0.042 | 0.684 |
| Llama 8B | 70 | 0.171 | 0.014 | 0.814 |

- By class, the 3B's refusals (n 32) are A 0.219 and C 0.750; its wrong answers (n 63) are A 0.302 and C 0.651.
- Median logit emergence depth is 0.750 for both sizes (quartiles 0.679–0.884 and 0.688–0.844); J-lens on the 3B, 0.714.
- Article caveat: on the 31 article-first golds, the 3B's final top-10 is 0.516 (A 0.484), against the 8B's 0.935.
- Digit caveat: Llama keeps up to three digits per token, so on the 84 digit-first golds of the 204 confirmation rows the target is the whole number.

**Found (Llama).**
- The 8B holds the answer more clearly than the 3B at the answer position, and the 3B is more often the reader in which
  the gold never reaches the top-10.
- Within the 3B, how weakly the gold ranks separates its wrong answers from its right ones (0.68). Both lenses agree.
- Most of the 3B's failures still have the gold in the final top-10 (C 0.68), with an A floor of about a quarter, as in
  Qwen.
- On the logit lens, at the same depth convention on the same 330-row population, Qwen's small reader finds the gold
  later than its large one (median 0.917 against 0.821, a gap of 0.10 of depth, about 3.5 layers; identical on the 204
  confirmation rows). The two Llama sizes do not differ (0.750 and 0.750). The 8B's advantage is in rank, not depth.
- Aside, J-lens (lens-dependent): Qwen's J-lens medians are 0.889 against 0.464; the 7B's low figure rests on the §34
  mid-stack bump the logit lens does not see.

**Why it matters.** "C-majority with an A floor" holds in a second family, and so does "the larger reader holds the
answer more clearly". The "small reader finds it later" half of the §34 picture is family-dependent so far: on the
logit lens Qwen's small reader is later by 0.10 of depth and Llama's by 0.00. Mixed; see the four-family scorecard below.

**Gemma Part B pod (A40, 77 min, about $0.75).** The same design: gemma-3-4b and gemma-3-12b-it on the 330 rows,
with a J-lens fitted on the 4B.
- **The J-lens gate on folds 0–1 FAILS:** (a) logit − J-lens −0.671 (−0.907, −0.450); (b) pass@k 0.126 against 0.171.
  The logit row alone decides, and the J-lens row is shown beside it only.
- Confirmation folds 2–4 were read once, after the development report was committed.

| claim (folds 2–4, 204 rows) | estimate (95% CI) | verdict |
|---|---|---|
| B1: log10 best rank of the gold at P1, 4B − 12B | +0.019 (−0.020, +0.059) | NOT SHOWN |
| B1 secondary: share of outcome A, 4B − 12B | −0.005 (−0.039, +0.034) | NOT SHOWN |
| B2: AUC of the 4B's P1 log10 best rank, wrong against right (logit) | 0.698 (0.631, 0.764) | SHOWN |
| B2, J-lens (beside only, gate FAIL) | 0.714 (0.645, 0.779) | not a verdict |

B1 is a tight null: in Gemma the 12B holds the gold no more clearly than the 4B.

A/B/C on failures, logit lens:

| reader | n | A | B | C |
|---|---|---|---|---|
| Gemma 4B | 80 | 0.125 | 0.113 | 0.762 |
| Gemma 12B | 55 | 0.218 | 0.055 | 0.727 |

- The 4B's B on failures is the article artefact again: 8 of its 9 B failure rows have an article-first gold.
- Median logit emergence: 4B 0.618 (0.559–0.676), 12B 0.667 (0.542–0.792). The small reader is earlier, not later, by
  0.05 of depth.
- Gemma splits digits like Qwen; digit-first golds are in the final top-10 for both sizes (1.000).
- Article-first golds (n 31): 4B final top-10 0.484, 12B 0.645.

**Phi Part B pod (A40, 77 min, about $0.75).** Phi-4-mini and Phi-4 (14B) on the 330 rows, with a J-lens fitted on
the mini.
- **The J-lens gate on folds 0–1 passes:** (a) logit − J-lens +2.380 (+2.175, +2.582); (b) pass@k 0.066 against
  0.025.
- Confirmation folds 2–4 were read once, after the development report was committed.

| claim (folds 2–4, 204 rows) | estimate (95% CI) | verdict |
|---|---|---|
| B1: log10 best rank of the gold at P1, mini − 14B | +0.157 (+0.085, +0.230) | SHOWN |
| B1 secondary: share of outcome A, mini − 14B | +0.059 (+0.005, +0.113) | SHOWN |
| B2: AUC of the mini's P1 log10 best rank, wrong against right (logit) | 0.636 (0.562, 0.707) | SHOWN |
| B2, J-lens (verdict role, gate PASS) | 0.598 (0.528, 0.669) | SHOWN |

A/B/C on failures, logit lens:

| reader | n | A | B | C |
|---|---|---|---|---|
| Phi-4-mini | 104 | 0.202 | 0.000 | 0.798 |
| Phi-4 14B | 39 | 0.103 | 0.128 | 0.769 |

- The 14B's B is partly the article artefact: 7 of its 24 B rows over all 204 have an article-first gold.
- Median logit emergence: mini 0.938 (0.875–0.969), 14B 0.525 (0.250–0.700). The small reader is later, by 0.41 of
  depth.
- Caveats:
  - Phi's large partner is 14B, a bigger size step than the other families' (3B→8B, 4B→12B, 3B→7B).
  - The mini and the 14B use different tokenizers, so the gold first token differs between them on 6 of the 204 rows;
    the rows are paired by question.
  - Digit-first golds, final top-10: mini 0.917, 14B 0.976.

**Scorecard after four families (Qwen, Llama, Gemma, Phi; descriptive).**

| reading | Qwen | Llama | Gemma | Phi | count |
|---|---|---|---|---|---|
| Failures C-majority (the gold is in the final top-10 but loses) | yes | yes | yes | yes | 4/4 |
| B2: a weaker gold rank separates wrong from right in the small reader | yes | yes | yes | yes | 4/4 |
| The larger reader holds the gold more clearly (B1) | yes | yes | no | yes | 3/4 |
| The small reader finds the gold later (logit lens; gap in depth) | yes (0.10) | no (0.00) | no (−0.05) | yes (0.41) | 2/4 |

**Found (Part B, four families).** Two readings hold in every family:
- When the small reader fails with the evidence in view, the gold is usually still in its final top-10 and loses at
  the end.
- How weakly the gold ranks at the answer position separates its wrong answers from its right ones.

That the larger reader holds the gold more clearly holds in three of four families (not Gemma). That the small reader
finds the gold later depends on the family: it holds in Qwen and Phi (whose size step is the largest), not in Llama or
Gemma.

**Caveats.** One prompt family (§23 `fixed8` on LongMemEval), one judge (Qwen 2.5 7B, from §24). The gold first token
differs by tokenizer (digits, articles). P1 is a forced position. The Llama Part A gate is on 13 rows.

**Spend so far.** Stage 0, stage 1 and the dev reads: $0 (laptop). Llama Part B pod: about $0.61; Gemma Part B pod:
about $0.75; Phi Part B pod: about $0.75. All experiments ≈ $20.06 of $30.

## 44. Is the answer at the answer position put there by the evidence? Activation patching and an evidence swap (2026-10-29, pre-registered; complete 2026-10-30)

**Status: COMPLETE.**
- **Rev 1, conditions S and U on Qwen 2.5 3B:** done.
  - **S0 FAILS** (34/46 = 0.739): the swap is not valid, and S1–S3 are not read.
  - **U1 reads "not the format":** 0.228 (21 of 92) of the 3B's in-view refusals become gold under U, 42% still refuse, and
    Steer-under-U − U is +0.152 (+0.065, +0.239).
- **Llama 3B S/U:** done, descriptive by rev 2. S0 is valid (55/66 = 0.833); S1 is not reliably distinguishable (half-width 0.167 > 0.15).
- **Rev 8, the repaired swap on Qwen 3B:** S0 FAILS on its lower bound (31/38 = 0.816 (0.684, 0.921)). The swap
  manipulation did not reach validity on this prompt family in two designs; the swap line closes.
- **Rev 0 (clean → cut patching):** confirmed for both models. 3B: C3 undetermined (median d_c 0.750). 7B (GPU read):
  C1 undetermined, "both partly right" (median d_c 0.679 (0.643, 0.714)); C2 "carries little" (mean R at layer 13
  0.005).

Pre-registration `research/proposals/2026-10-29-exp44-patching.md`:
- rev 0 at 6e21476;
- rev 1 at cfa7712: the thesis question and conditions S and U;
- rev 2–4 at 5146f5b, 7acce03 and f9952b2: the swap gate and the shared-prefix rule;
- rev 5 at f0ec6c7: the 1-nat gap on both filters, with R(0) a flag rather than a stop;
- rev 6 at 31553f6: difference-matched controls;
- rev 7 at 6f457fd: the filters fixed.

Decisions: D-44 (the user's order) and D-45 (the thesis question). Reviews: `research/reviews/2026-10-29-exp44-build.md`,
`research/reviews/2026-10-29-exp44-swap-build.md` and `research/reviews/2026-10-30-exp44-swap-qwen3b-results.md` (6230ac3).
Code:
- `memctl/analysis/patching.py`, `scripts/exp44_patch.py` and `scripts/exp44_report.py` (d1902f5, ffca6b2, 63bb3d2);
- `scripts/exp44_swap_gate.py` (63882c1, 88b2a19, 0dcb155);
- `memctl/analysis/swap.py`, `scripts/exp44_swap.py` and `scripts/exp44_swap_report.py` (dbd813e, 8f8589c, f22c8f8).

**The question (D-45).** When a small model fails on a question whose evidence is in view, can a failure of evidence
use (the evidence does not reach the answer computation) be told apart experimentally from a failure of answer
production (the evidence-dependent information is at the answer position, but the response decision overrides it)?

**Rev 0: clean → evidence-cut patching (laptop; development folds 0–1).**
- Method: for each §34 row, the P1 residual of the clean run is written, layer by layer, into the evidence-cut run.
  R(l) is the recovery of log p(gold first token) at the final layer, normalised between the cut and clean values.
  The causal depth d_c is the first layer with R ≥ 0.5.
- Filter (rev 5 and rev 7): clean rank ≤ 10, cut rank > 3, and lp_clean − lp_cut ≥ 1 nat (relaxed for both models,
  since fewer than 40 dev rows passed the strict filter).
- R(L−1) = 1 is asserted. |R(0)| ≥ 0.05 is a flag that excludes the row from the claims.

| dev, folds 0–1 | Qwen 3B | Qwen 7B (bf16) |
|---|---|---|
| rows / patched | 126 / 126 (all) | 126 / 61 |
| strict + gap / relaxed + gap | 30 / 52 | 38 / 59 |
| excluded for gap < 1 nat inside the rank filter | 5 | 7 |
| R(0)-flagged | 0 | 3 |
| claim rows | 52 | 56 |
| median d_c | 0.736 (0.667, 0.750) (C3) | 0.679 (0.643, 0.714) (C1) |
| mean R at layer 13 (the §34 J-lens mid-stack depth) | – | 0.022 (0.001, 0.045) (C2) |

On the development folds, the 7B's mid-stack J-lens signal carries almost no causal weight at the answer position
(R ≈ 0.02 at layer 13), and the evidence-dependent recovery happens from about two-thirds of the depth. The
confirmation reads are pending; no claim is made from these rows.

**Rev 0 confirmation, Qwen 3B (folds 2–4, 204 rows; 2026-10-30).** C3 is undetermined. 88 rows are in the used rank
filter; 10 are excluded for a gap under 1 nat, leaving 78 claim rows. None is R(0)-flagged, and R(L−1) passes on 78 of
78. The median d_c is 0.750 (0.722, 0.778). The lens places the 3B's answer about 0.15 of the stack later than patching
finds it causally usable (per-row d_c − d_jlens median −0.139); the pre-registered under-read criterion (upper bound
< 0.75) is missed by 0.028, so this is reported as undetermined. Report `runs/exp44_report_confirm_qwen3b.md`
(b1b5c45; review 3d6ca32).

**Rev 0 confirmation, Qwen 7B (folds 2–4, 204 rows, read on one A40 GPU in bf16; 2026-10-30).**
- **C1 is undetermined ("both partly right"):** median d_c 0.679 (0.643, 0.714), n 83. The J-lens reading needed an
  upper bound below 0.65 and the logit reading a lower bound above 0.70.
- **C2: "carries little".** The mean R at layer 13, the §34 J-lens mid-stack depth, is 0.005 (−0.010, +0.020).
- Rows: 96 in the used rank filter, 10 excluded for a gap under 1 nat, 3 R(0)-flagged (3.5%), so 83 claim rows.
  R(L−1) passes on 86 of 86.
- Robustness, with the flagged rows included: C1 is unchanged; C2 is −0.027.
- Sensitivity, pre-declared as a hold rule: C1 is unchanged (a) with the 5 rows that flipped between devices excluded
  (0.679 (0.643, 0.714), n 82; C2 0.003), and (b) with the median shifted by ±1 layer (0.643 or 0.714, the same
  reading). So the verdict stands, and the laptop CPU read was stopped at 75 of 204 rows (rows kept).
- Report: `runs/exp44_report_confirm_qwen7b_gpu.md` and `_sensitivity.md` (50eab42).

So on the 7B, the mid-stack J-lens signal of §34 carries almost no causal weight at the answer position, and the
evidence-dependent information becomes usable from about two-thirds of the depth. Neither lens's depth is confirmed
as the causal one.

**Device check (laptop CPU bf16 against pod CUDA bf16, 7B; 73 common rows, 36 patched on both).** d_c is equal on 34
of 36 rows; the other two differ by −1 and +2 layers of 28. The maximum |R| difference is 0.142 (median over rows
0.042). There is one eligibility flip (91b15a6e), two gap-sign flips, one 1-nat-gap flip and one R(0)-flag flip, and
the largest |log p| difference is about 1.4 nat. This is bf16 drift between devices: small on the deciding statistic,
able to move a row or two across the filter. Report `runs/exp44_patch/device_check_qwen7b.md` (cbffc52).

**Rev 1: the evidence swap (S) and the no-"unknown" control (U), Qwen 3B, pooled over all 330 rows (rev 4).**

- **S0 (validity) FAILS.** Among the 3B's successes in the swap subset, 34 of 46 follow the evidence (0.739 (0.609,
  0.870), the same at the 0.5, 1 and 2 nat margins). The bar is ≥ 0.80 with a lower bound ≥ 0.70. The swap does not
  control the answer reliably enough on this prompt family. Per §C, **S1–S3 are not read.**
- **U1 (prompt artefact): "not the format".** U is the §35 frozen-note line with no "unknown" option, run on the 3B's
  in-view refusals: 97 in scope, of which 92 carry a note to freeze; 5 have no note.
  - Gold under U: 0.228 (0.152, 0.315), within the "not the format" bar (≤ 0.25, upper bound < 0.35).
  - Under U, 42% of these refusals still refuse.
  - The §38 push under U gets 0.380: Steer-under-U − U = +0.152 (+0.065, +0.239).
  - On the 18 swap-subset refusals, U is 0.389 (descriptive).

Swap subset and cost: 89 rows (the §A rules with rev 2's per-row substitutes and rev 3's shared-prefix rule), 43 of
them in-view failures. S took 2.7 h and U 1.4 h, on the laptop, $0.

**Descriptive notes (not claims, because S0 failed).**
- Classes at 1 nat (answer production / evidence use / indeterminate):
  - successes 34 / 4 / 8;
  - refusals 13 / 1 / 7;
  - wrong answers 16 / 2 / 4.
- The 3B's failures are mostly on rows where the swap does move the answer preference (29 of 43). That is the
  direction the thesis would predict, but the following test is not trusted on this subset.
- Evidence-specific transfer (forward d_T 0.889, reverse 0.889, n 63) completes later than clean → cut recovery on the
  same rows (d_c 0.778, 26 development rows, unfiltered); the logit lens gives 0.917 and the J-lens 0.889 on those rows.
- The rev 6 controls behaved. The donor difference patch and the difference-norm random perturbation stay at about 5%
  of the forward shift at d_T. S3 validity passed only on the pre-declared fallback (the donor's late-layer mean shift
  reaches 1.22 nat, above ±0.5 but below the 2-nat "still disruptive" flag).

**Post-hoc, exploratory: why S0 failed (diagnosed after the outcome; not a result).** The 12 non-following successes
split three ways. By surface type, following among successes is name 12/14, currency 2/2, duration 6/7, bare number
10/15, other 4/8.
1. **Type-incompatible substitutes (7 rows).** The "other" rule drew substitutes from the same question type, which
   produced answers the question cannot take ("every week" → "under my bed", "Spotify" → "Emma", a year "2014." →
   "3,000 miles"). A model that declines to copy an impossible answer is behaving correctly; these rows test the
   substitute, not evidence use.
2. **Generic first tokens (overlapping with 1).** The read token is the first divergent token: "2" of a year, "The",
   "Chapter". On 2bf43736 and 5809eb10, the gold's log p under the swap is ≈ 0, because the question, not the
   evidence, predicts that token.
3. **Small bare numbers (4 rows).** "five" → "four", "3" → "25" and the like; Δ moves by only 0.2–1.0 nat.

The one duration miss (19b5f2b3) has Δ_clean < 0, so it is uninformative. On name, currency and duration rows alone,
following is 20/23 = 0.87. **That figure is selected after the outcome and cannot rescue S0.** It says what a repaired
design would need: same-answer-type substitutes with a non-generic first content token. No redesign is made here;
whether a repaired rev runs is the user's decision.

**Rev 1 on Llama 3.2 3B (pooled over all 330 rows; descriptive by rev 2).** 90 swap rows, 24 of them in-view
failures; U on 45 in-view refusals, all with a note.
- **S0 is valid:** 55 of 66 successes follow the evidence (0.833 (0.742, 0.924), the same at every margin).
- Classes at 1 nat (answer production / evidence use / indeterminate):
  - successes 55 / 0 / 11;
  - refusals 10 / 1 / 1;
  - wrong answers 9 / 0 / 3.
- **S1: not reliably distinguishable** (half-width 0.167 > 0.15; (a) indeterminate 0.167 ≤ 0.20 and (b) a margin move
  of 0.042 both hold).
  - Point estimate: 19 of 24 failures are answer-production and 1 of 24 evidence-use. That is the opposite of rev 1's
    prediction for Llama (an evidence-use majority).
  - Caveat, in the same breath: the swap subset is the verbatim single-line end of the population, where §43's
    note-ceiling rows are under-represented. So the §43 Llama–Qwen contrast is not reproduced on this subset, rather
    than refuted.
- S2, descriptive: the push (layer 17, α 2) opens 3 of 10 answer-production refusals; there is 1 evidence-use refusal.
- S3, descriptive: forward d_T 0.768 (0.750, 0.786), n 74; reverse 0.786. Validity on the pre-declared fallback.
- **U1: "not the format".** Gold under U 0.089 (0.022, 0.178), n 45. 31 of the 45 still refuse. Steer-under-U − U
  is +0.133 (+0.044, +0.244).
- Post-hoc, exploratory: the 11 non-following successes fall in the same classes as Qwen's: 5 "other" type
  mismatches, 4 bare-number form mismatches and 2 generic "The". Following by type: date 4/4, name 20/22, currency
  3/3, duration 5/5, bare 14/18, other 9/14.

**Rev 8: the repaired swap on Qwen 3B (D-47; pod, fp32).** This is a repaired design tested on the same rows after
the fault was seen; nothing here is confirmatory.
- The repair, rules 1–5: "other" substitutes dropped; bare numbers matched in form, digit count and unit presence; the
  leading determiner matched; the shared-prefix read kept; punctuation-only read tokens rejected.
- 70 rows. Labels 38 / 14 / 18 (success / refusal / wrong).
- **S0: 31 of 38 = 0.816 (0.684, 0.921): not valid (lower bound below 0.70).**
- Honesty split: changed-substitute rows 7 of 10; unchanged rows 24 of 28 (already known from rev 1).
- The seven non-followers are now almost all small-number rows ("four" → "five", "five" → "four", "3" → "5",
  "2" → "6", "2014." → "1300"), plus "Spotify" → "Emma" (a brand swapped for a person's name under the name rule) and
  the duration row whose Δ_clean < 0. Small counts are plausibly answers the model computes or reads off elsewhere in
  the prompt, so swapping the number string does not control them.
- Descriptive only, since S0 failed: among the 32 failures, 26 are answer-production, 6 indeterminate and 0
  evidence-use at 1 nat.

**The swap manipulation did not reach validity on this prompt family in two designs; the evidence-use /
answer-production split is not established by §44.**

**Caveats.** One benchmark and prompt family; one forced position; first-token logit differences. The swap subset is
biased toward extractive single-hop rows (§F). The 7B was run in bf16 on CPU. The rev 0 numbers above are development
folds only.

**Spend.** Laptop runs $0; one A40 pod for the 7B confirmation and rev 8 (about 17 min, about $0.16). All
experiments ≈ $20.22 of $30.

## 45. What do the lenses show of the answer computation, layer by layer and token by token? J-lens maps on controlled tasks, Qwen 2.5 3B (2026-10-30, pre-registered; pod run 2026-10-31)

**Status: COMPLETE (descriptive). No §46.**
- **The pre-registered readability criterion is inflated by echo.** It counts positions where a lens can only be
  echoing the token just fed in, so its "recovers" verdicts (1.00 everywhere) and its depths (J-lens median 0.00) do
  not measure what §45 asked. They stay in the report, marked "inflated by echo".
- **Read with the two post hoc echo rules (labelled post hoc throughout),** the answer token becomes readable only in
  the last six layers (first readable layer 30–33 of 36) on every task, format and role; no earlier intermediate is
  readable with these lenses.
- **Matched failures separate from their matched successes only at layer ≥ 30** (or never).
- **Lens stability:** the 56-prompt lens is stable (agreement 0.998, Spearman 0.980); its 128-prompt refit scores the
  §34 gate identically (0.014 vs the logit lens's 0.039) and reads the maps the same way; neither J-lens adds anything
  over the logit lens on the 3B.
- **No §46:** there is no intermediate or transition readable before layer 30 to name as a causal target, so the D-45
  stopping rule applies (document, do not invent).

Pre-registration `research/proposals/2026-10-30-exp45-jlens-maps.md`: rev 0 at 78d5ad1; rev 1 at 2d7122f (space form
everywhere, bare form at the last prompt token only; counted positions = the last prompt token and the answer span; F2
and stage 3 generate to the end of the `Answer:` line; the held half decides). Decisions: D-49 (the user's order: "amend
D-45, draft §45 after the 7B lands") and D-51 (the pod: "Yes to the pod at 0.59 hard stop 2.50"). Reviews:
- `research/reviews/2026-10-30-exp45-build.md` (2d7122f, addendum 5f00b6c);
- `research/reviews/2026-10-30-exp45-results.md` (1a6e1e5): the recount, the echo artefact and the post hoc rules.

Code: `scripts/exp45_cases.py`, `memctl/analysis/maps.py`, `scripts/exp45_maps.py`, `scripts/exp45_render.py`,
`scripts/exp45_pod.sh` (7573325, aef32ae); `scripts/exp45_report.py` with the post hoc rules A and B (34c0472). Run
`runs/exp45_maps/j56-jrefit-logit/` (per-prompt json and reports a43386b; post hoc reports 34c0472; npz maps, 1.7 GB, and
the refit lens on disk only).

**The question (proposal §1).** Why does a small model abstain or answer wrongly with sufficient evidence in context?
§34–§44 read one token at one forced position. §45 asks what the lenses show of how the answer computation develops
across layers and tokens, on tasks whose intermediate steps are known by construction. The output is maps, not claims
about the model's computation: a miss is "not recovered by these lenses", never "not computed".

**Design (fixed before the run).**
- Model: Qwen 2.5 3B Instruct, bf16, one A40. Its own chat template and greedy generation: F1 up to 16 new tokens; F2
  and stage 3 up to 160, stopping after the `Answer:` line.
- Lenses: the §34 J-lens (56 prompts, sha `ddbdb32d…`); a refit J-lens (128 WikiText prompts, seed 1, the §34
  settings, 68.5 min of the 75-min box); the logit lens. At every layer and position, the top-10 and the rank of every
  pinned token: the first content token of every entity, in a space form and a bare form.
- Cases: three tasks × 12 base cases × 4 variants × 2 formats × 3 hardness levels = 864 prompts.
  - Direct extraction (DE): no intermediate.
  - Relational composition (RC): intermediate = the supervisor's name.
  - Temporal updating (TU): competitor = the old city.
  - Variants: base, substitution, question change, order change. Formats: F1 plain chat; F2 the §23 memory-note
    wrapper. Hardness: H0, plus 3 distractors (H1), plus 10 (H2).
  - Even base cases form the dev half, odd the held half. **The held half decides**; dev is shown first.
- Readable (fixed, rev 1): the space-form id in the lens top-10 at a counted position, or the bare-form id at the last
  prompt token. Criterion (a), the gold, is read before the final layer. A criterion "recovers" at a share ≥ 0.70 with
  a Wilson lower bound ≥ 0.55.
- Stage 2: every failure, paired with the solved prompt of the same base case, variant and format at the lowest
  hardness. The only sentence it can earn: "the gold is / is not readable before the final layer on ≥ 0.70 of the
  matched failures". Stage 3: the 70 §44 rev 8 rows (clean and swap), descriptive. No causal test.

**Run and coverage (pod rre3aw7agtsm42, A40 secure $0.59/h, 00:43:46–02:47:10 UTC 2026-10-31, ≈ 2 h 03 min).**
- Chain 7069 s of the 14400 s stop: refit 4111 s (128/128 prompts), stage 1 2381 s, stage 3 400 s, eval 88 s.
- Stage 1: 864/864 rows. Stage 3: 140/140.
- Outcomes, held: gold 246 / other 88 / abstain 78 / competitor 20 (dev 256 / 73 / 76 / 27).
- `Answer:` marker missing on 2 held F2 rows and 1 dev F2 row (F1 has no marker by design).
- Stage 2, held: 186 failures, 78 at H2 (≥ 10, so the stage is read), 150 paired. Dev: 176, 74, 137.

**The echo artefact (results review §2).** At layer 0 a J-lens decodes the token being fed in. F1 outputs restate the
facts, so the gold is itself an input token at later generated positions, and the fixed rule counts those positions.
- Recount (34c0472, all solved F1 prompts): 233 generated positions in 220 of the 228 solved F1 prompts have the gold's
  space-form id as their input token. At layer 0, the gold is in the top 10 there on 129 of 233 for j56 and 122 for
  jrefit (rank 1 on 54 and 41; median rank 7 and 9). For the logit lens its layer-0 rank is never below 8 (median 389).
  - The review's "rank ≤ 7 on 233/233 for both J-lenses" is not reproduced by this recount; "≥ 8 on 233/233 for the
    logit lens" is. This is open with the Advisor.
- The logit lens has no depth-0 artefact. Its fixed 1.00, however, also rests on positions after the gold has been
  produced, where the residual still carries the token at layers 30–33.
- Two post hoc readings remove the echo:
  - **Rule A (input-token exclusion), "readable anywhere in the answer span (post hoc, rule A)":** a hit on X at counted
    position p does not count when the token fed in at p is X (either form).
  - **Rule B (production positions), "readable before the answer is produced (post hoc, rule B)":** count only the
    counted positions up to and including the one that produces X's first occurrence. In F2 the note usually states X
    before `Answer:`, so for F2 rule B mostly measures whether the note already held X, not readability. It is read per
    format.
  - The bare form at the last prompt token is unchanged under both rules. Every rule-A/B number below is **post hoc**,
    with the pre-registered number beside it.

**Stage 1, held (decides): the gold, before the final layer.** Shares (Wilson lower bound) for j56 / jrefit / logit.
Depth = first readable layer / 36, as quartiles.

| task (n; F1, F2) | fixed, pre-registered (inflated by echo) | rule A, post hoc | rule A depth (j56; logit) | rule B all, post hoc | rule B F1 | rule B F2 | rule B depth (j56; logit) |
|---|---|---|---|---|---|---|---|
| DE (95; 49, 46) | 1.00 / 1.00 / 1.00; depth 0.00/0.00/0.86 (j56), 0.86/0.86/0.89 (logit) | 1.00 (0.96) all | 0.83/0.89/0.89; 0.86/0.86/0.89 | 0.62 / 0.62 / 0.60 (0.52 / 0.52 / 0.50) | 1.00 (0.93) | 0.22 / 0.22 / 0.17 | 0.83/0.86/0.89; 0.86/0.86/0.92 |
| RC (79; 37, 42) | 1.00 all; depth 0.00/0.00/0.86 (j56), 0.86/0.89/0.89 (logit) | 1.00 (0.95) all | 0.86/0.89/0.89; 0.86/0.89/0.89 | 0.61 / 0.61 / 0.56 (0.50 / 0.50 / 0.45) | 1.00 (0.91) | 0.26 / 0.26 / 0.17 | 0.83/0.89/0.90; 0.86/0.89/0.92 |
| TU (72; 25, 47) | 1.00 all; depth 0.00/0.00/0.25 (j56), 0.86/0.86/0.89 (logit) | 1.00 (0.95) all | 0.86/0.89/0.89; 0.86/0.86/0.89 | 0.35 (0.25) all | 1.00 (0.87) | 0.00 | 0.83/0.83/0.86; 0.86/0.86/0.86 |

- The review's recount gives rule B on DE as 0.64 / 0.64 / 0.62 (F2 0.26 / 0.26 / 0.22), two rows more. Its rule B
  matched only the space-form id when looking for the first occurrence. Here either form counts, as specified, and on
  4 DE F2 rows the note states the gold in its bare form at the start of a line (for example "David lives in
  York.\nAnswer: David"). Everything else in this table matches the review.
- Under rule B the note already held the gold on 0.74–1.00 of the F2 rows (1 − the F2 column).
- Last prompt token only (echo-free by construction; F1's first generated token is usually "Based"):
  - Both forms, the pre-registered beside variant: DE 0.24 / 0.24 / 0.22, RC 0.30 / 0.30 / 0.20, TU 0.00.
  - Space form only, the review's count: DE 0.13 / 0.13 / 0.14, RC 0.04 / 0.03 / 0.08, TU 0.00.
- Dev (presentation) shows the same pattern: fixed 1.00 everywhere; rule A 1.00 with depth 0.83–0.89; rule B DE
  0.66 / 0.66 / 0.63, RC 0.60 / 0.60 / 0.56, TU 0.32; F1 1.00; F2 DE 0.18 / 0.18 / 0.13, RC 0.29 / 0.29 / 0.20, TU 0.00.
  - The review's dev figures (DE 0.61, RC 0.45) come from a count with the space form only and no bare form at the
    last prompt token.

**The other pinned entities, held, before the final layer (post hoc rules; j56 / jrefit / logit).**
- **RC intermediate (the supervisor's name; n 50 = 23 F1 + 27 F2).**
  - Fixed, as pre-registered (any layer, inflated by echo): 0.94 / 0.94 / 0.88. Before the final layer: 0.84 / 0.84 /
    0.72.
  - Rule A, post hoc: 0.84 (0.71) / 0.84 / 0.72 (0.58); F1 1.00; F2 0.70 (0.52) / 0.70 / 0.48.
  - Rule B, post hoc: 0.84 / 0.84 / 0.70; F2 0.70 / 0.70 / 0.44 (0.28).
  - Depth 0.89/0.89/0.92 under both rules.
- **TU competitor (the old city; n 72 = 25 F1 + 47 F2).**
  - Fixed, as pre-registered (any layer, inflated by echo): 0.47 / 0.49 / 0.43.
  - Rule A, post hoc: 0.29 (0.20) / 0.29 / 0.26; F1 0.64 (0.45) / 0.68 / 0.52; F2 0.11 (0.05) / 0.09 / 0.13.
  - Rule B, post hoc: 0.22 / 0.24 / 0.18; **F2 0.00 for every lens**. Depth 0.83–0.92.
- Both match the review.

**Stage 2, held: matched failures (186; 150 paired).** The pre-registered sentence asks whether the gold is not
readable before the final layer on ≥ 0.70 of them.

| reading | not readable before the final layer: j56 / jrefit / logit | sentence earned? |
|---|---|---|
| fixed (pre-registered; inflated by echo) | 65/186 = 0.35 (0.28, 0.42) / 65 / 67 = 0.36 | no |
| rule A, post hoc | 65/186 = 0.35 (0.28, 0.42) / 65 / 67 | no |
| rule B, post hoc | 124/186 = 0.67 (0.60, 0.73) / 124 / 127 = 0.68 | no |

- On dev: fixed and A 67/176 = 0.38; B 125/176 = 0.71 (descriptive; held decides).
- The review prints the fixed Wilson lower bound as 0.29 (0.2846, a rounding difference).

Departure layer: the first layer where the paired success has the gold in the top 10 and the failure does not, on the
pooled 287 pairs.

| | 0 | 1–9 | 10–29 | 30+ | none |
|---|---|---|---|---|---|
| fixed j56 (**struck: echo artefact**) | 172 | 50 | 13 | 27 | 25 |
| fixed logit (struck) | 10 | 6 | 4 | 194 | 73 |
| rule A j56, post hoc | 2 | 0 | 1 | 189 | 95 |
| rule A jrefit, post hoc | 2 | 0 | 5 | 181 | 99 |
| rule A logit, post hoc | 0 | 1 | 4 | 205 | 77 |
| rule B j56 / jrefit / logit, post hoc | 2 / 2 / 0 | 0 | 1 / 5 / 4 | 129 / 120 / 126 | 155 / 160 / 157 |

These match the review (rule B departures are not in the review).

**Stage 3: the 70 §44 rev 8 rows (clean gold, swap substitute; the counted span is after `Answer:` in the model's own
note; before the final layer; descriptive, as pre-registered).**

| target | lens | fixed (inflated by echo) | rule A, post hoc | rule B, post hoc |
|---|---|---|---|---|
| gold, clean rows | j56 | 35/70 = 0.50 | 34/70 = 0.49 (0.37, 0.60) | 10/70 = 0.14 (0.08, 0.24) |
| gold, clean rows | jrefit / logit | 34 / 37 | 34 / 36 | 9 / 13 |
| substitute, swap rows | j56 / jrefit / logit | 26 / 26 / 29 | 26 / 26 / 29 = 0.37–0.41 | 8 / 8 / 11 = 0.11–0.16 |

- By §23 label (gold on clean rows, j56):
  - rule A: refusal 4/14 = 0.29, success 20/38, wrong 10/18;
  - rule B: refusal 1/14, success 5/38, wrong 4/18.
- By §44 class (1 nat):
  - rule A: answer_production 26/57, indeterminate 6/10, evidence_use 2/3;
  - rule B: answer_production 7/57, indeterminate 3/10, evidence_use 0/3.
- Substitute on swap rows, refusal label: 1/14 under every rule.
- Median first readable layer (j56): 32 under rule A, 33 under B.
- **Discrepancy with the review, open with the Advisor.**
  - The review gives rule B 31/70 (refusal 4/14, answer_production 23/56, substitute 23/70). Its stage-3 recount
    indexed `input_ids` by stored-position index. Stage 3 stores only a subset of prompt positions, so it compared
    against the wrong tokens. With that indexing this code reproduces 31/70.
  - Read correctly, rule B on stage 3 falls to 0.14 for the same reason as F2: the model's own note usually states the
    answer before `Answer:`. Digit first tokens add to this (the "2" of a year counts as an occurrence of "25").
  - So "about half of the clean rows" holds for rule A (anywhere in the answer span), not for rule B.
  - The §44 class source here (`runs/exp44_swap/qwen3b_rev8/S`, `cls["1"]`) gives 57 answer-production rows at every
    margin; the review has 56.
- Under rules A and B the J-lenses read a digit gold early (layers 9–13) on a few stage-3 rows: "120", "2014.",
  "$2,000", "10 minutes". These are within-number continuations (the position fed "1" predicts "2"), not an earlier
  intermediate. The logit lens has none before layer 29 (2 rows at 29).

**Lens stability (proposal §5).**
- j56 against jrefit, held solved prompts: readability-call agreement 0.998 over 939 calls (pooled 0.997 over 1910);
  trajectory Spearman median 0.980. That is "stable" under the pre-registered 0.20 rule.
- The refit lens scores the §34 gate exactly as the 56-prompt lens did, with different weights (different sha,
  n_prompts 128): mean pass@k in the 12–24 band 0.0140 against the logit lens's 0.0386; `pass_at_k_all` 0.1065 against
  0.1069.
- The review's sentence: "the 56-prompt lens is stable (agreement 0.998, Spearman 0.980); its 128-prompt refit scores
  the §34 gate identically (0.014 vs the logit lens's 0.039) and reads the maps the same way; neither J-lens adds
  anything over the logit lens on the 3B".

**What was found.** In plain words:
- On the solved controlled prompts the answer token shows up in the lens top 10 only in the last six layers (layer
  30–33 of 36). It shows up at the position that produces it (F1: every row, rule B 1.00), or at the answer position
  after a note that already says it (F2: rule A 1.00; by rule B the note already held it on 0.74–1.00 of F2 rows). No
  lens reads it earlier at any counted position. At the last prompt token it is readable on 0.00–0.30 of rows with
  both forms, 0.00–0.14 with the space form alone. (Post hoc rules; the pre-registered 1.00 is inflated by echo.)
- The relational intermediate (the supervisor's name) and the temporal competitor (the old city) are readable only in
  the same late band. At F2's answer position the competitor is not readable at all (0.00 by rule B, post hoc): the
  lens sees the new city there and nothing of the old one before the final layer.
- Matched failures: the gold is readable before the final layer at the production positions on about a third of them
  (rule B 0.33 held, post hoc). It is readable anywhere in the answer span on about two thirds (rule A 0.65, post hoc),
  because failing outputs restate the facts. Where a failure departs from its matched success, it does so at layer
  ≥ 30 (189 of 287 pairs for j56 under rule A; 95 pairs never separate). The lenses cannot see a failure coming before
  the last six layers.
- On the §44 rows the gold is readable in the model's own answer span on about half of the clean rows (rule A 0.49–0.51,
  post hoc) and on 0.29 of the refusals. It is readable before its first occurrence on only 0.13–0.19 (rule B, post hoc),
  because the note usually states it first. The substitute is readable on 0.37–0.41 of swap rows (rule A). Median first
  layer 32.

**Why it matters.**
- §45 adds to §34. The 3B's answer appears late in the stack in every format and task tried, and no earlier
  intermediate is readable with these tools.
- The three readouts are one readout: the two J-lenses agree with each other and with the logit lens, and a
  128-prompt refit changes nothing. So the §34 lens caveat is settled as "stable, no better than the logit lens".
- There is no specific intermediate or transition, readable before layer 30, from which a failure departs. So
  there is no causal target for a §46, and the D-45 stopping rule applies: document, do not invent.

**Caveats.**
- The rules A and B were defined after the data existed (post hoc), to remove a mechanism found in the data. The
  pre-registered columns are kept and marked "inflated by echo".
- Rule B in F2 and stage 3 measures mostly whether the note already held the target.
- "Not readable" means not recovered by these lenses (a top-10 statement about a linear readout), not "not computed".
- F2's memory lines carry dates that increase with line order, so in TU the timestamp order is a second update cue
  beside the wording (a format effect, which F2 is there to show). This also bears on the F2 competitor reading.
- Digit golds are ambiguous first tokens (stage 3).
- Toy prompts are not long-context memory; stage 3 is the bridge.
- Greedy decoding, bf16 on the pod, one model and one family: nothing here generalises.
- Two items are open with the Advisor: the stage-3 rule B discrepancy above, and the echo-rank figure.

**Spend.** One A40 pod, ≈ 2 h 03 min, ≈ **$1.21** (balance 9.387 → 8.179). The post hoc rules, the reports and this text
were $0 (laptop, no model load). All experiments ≈ **$21.43 of $30**.

## 46. Is the relevant intermediate present where the lenses cannot see it? A probe and an interchange intervention on the §45 cases, Qwen 2.5 3B (2026-10-30, pre-registered; pod run 2026-10-31)

**Status: COMPLETE. Computed §5 verdict: row 4, "route late at P_last, instrument blind at P_B". Its second clause
stands on the intervention arm; its first clause is not established, because the probe arm is uninformative by design
(post hoc, labelled; the verdict is not revised).**
- **Intervention (held, 55 unseen base cases):** patching the residual at the supervisor's name token from a prompt
  with another supervisor moves the answer to that supervisor's city through layer 15 of 36 (L\* = 15; Transfer 0.80
  at L15, 0.69 at L16, 0 from L33). Both content-matched controls stay at 0 (k1 0/55 at every layer; k2 0/47 except
  1/47 at L20). The §45 lenses read nothing at the evidence positions, P_B among them, at layers 2–29, so at P_B they are blind to a variable that
  moves the answer.
- **Probe (held):** no layer decodable for any read; accuracy 0.04–0.11 at P_last, at or below 1/12 at every layer.
  Dev confounds label with base case (each of the 12 labels lives in one of 6 dev bases), so the dev-fitted probe
  learned base identity; "not decodable at P_last before layer 30" is a property of its training data, not of the
  representation.
- **k3 substitution floor 52/55 = 0.945, below the fixed 0.95 bar by one pair.** The three misses go toward abstention,
  which can only lower Transfer, so L\* stands with the shortfall stated beside it.
- **S2 (the last prompt token) is 0 at every layer by construction**, not a negative.

Pre-registration `research/proposals/2026-10-30-exp46-instrument-check.md`:
- rev 0 at e70d03f;
- rev 1 at 6ff90c8: aligned P / P′ / P″ triples replace the natural pairs (0 exist); first-city readout; c2 by dev CV;
  the pod order and the $1.80 stop-and-ask;
- rev 2 at 34d9586: H2 B″ from a distractor name; the batched hook; k3 as an input-embedding substitution;
- rev 3 at 10505f3: dev-CV folds by (format × hardness) cell;
- rev 4 at 34d1b4d: the readout is the conclusion;
- rev 5 at 45bb089: "decodable" = held numbers only, the dev-CV clauses descriptive; the exact row-span solver.

Decisions: D-53 (the user opened §46: "go on 2", after the Codex measurement audit) and D-54 (the pod: "Yes to the pod at
0.59 hard stop 2.50", typed in both panes). Reviews:
- `research/reviews/2026-10-30-exp46-build.md` (34d1b4d; addendum a5e86a1; §8 e57577a: dev recount, the rev 5 gate,
  the design sha recomputed before the held read);
- `research/reviews/2026-10-31-exp46-results.md` (13804dc): held recount, verdict, the probe-arm limitation, rulings.

Code:
- `memctl/analysis/probe46.py`, `memctl/analysis/patch46.py`;
- `scripts/exp46_capture.py`, `exp46_probe.py`, `exp46_patch.py`, `exp46_report.py`;
- the pod chain `exp46_pod.sh`, `exp46_pod_run.py`, `exp46_bundle.sh`, `exp46_d30.sh`;
- `tests/test_exp46.py` (18 tests).

Commits: b9562de (build), 959f97c (rev 4 readout), f3d7a3e and 2b24f6d (D-30 fixes), 4de18ca (attribution table),
83d0546 (rev 5 code and the dev design, design sha `8b54cf5e…c67e`), cde2d08 (`design_committed.json`), 11d8f61 (the
held read). Data: `runs/exp46_probe/cases46.jsonl` (§46 addition, sha `174e3ce4…`); runs
`runs/exp46_probe/cuda-bfloat16/` (576 residual captures, 492 MB, on disk only; `design.json`, `probe_dev.json`,
`probe_held.json` committed) and `runs/exp46_patch/cuda-bfloat16/` (per-triple json and reports).

**The question (proposal §1).** §45 (F13) read the RC intermediate (the supervisor's name) and the TU cities only at
layers 30–33. Is that a blind spot of vocabulary-projection lenses or the model's route? Three outcomes were fixed in
advance:
- the variable is linearly present mid-stack and causally carried (instrument blind spot);
- it is not there before the late step (route late, lenses adequate);
- it is present but moving it does not move the answer (represented, not used).

Prior art predicts the first on toy tasks (Feng & Steinhardt 2023; Geva et al. 2023).

**Design (fixed before the held read).**
- Model: Qwen 2.5 3B Instruct, bf16, one A40 for the capture and the interventions; probe fits on the laptop CPU.
- **Measurement A, probe.** On the §45 RC / TU prompts (576; outcomes reused from §45 stage 1, chat sha checked), the
  residual (output of decoder layer l, layers 0–35) at:
  - P_last: the last prompt token;
  - RC P_q: A's name in the question;
  - RC P_B: the supervisor's name in "{A}'s supervisor is {B}";
  - TU: the first occurrence of each city.

  A multinomial L2 logistic probe per (read, layer) is fitted on dev and read once on held. Reads: RC supervisor at
  P_last (the §5 verdict read) and at P_q; TU current and superseded city at P_last. Controls:
  - c1: Hewitt–Liang, a random label per base case;
  - c2: presence, the first distractor's name;
  - c3: chance.

  Held prompts never share an (A, B) or (A, C1, C2) combination with dev (0 overlaps), and every held label occurs in
  dev. **Decodable (rev 5) = held accuracy ≥ 0.70 with Wilson lower bound ≥ 0.55.**
- **Measurement B, interchange intervention (RC).** For each RC v0 / v3 row × F1 / F2 × H0–H2 (144; 72 dev / 72 held),
  a triple from the same text (`cases46.jsonl`, marked "§46 addition"):
  - two extra facts, "{B′} works in {C1′}" and "{B″} works in {C1}", inserted in all three prompts at one seeded slot;
  - the supervisor sentence names B (P, gold C1), B′ (P′, gold C1′) or B″ (P″, gold C1, the same-answer donor);
  - 0 alignment drops (equal length, one token differs, at P_B);
  - at H2 only one pool name is unused, so B″ is a distractor name with no "works in" fact (48 triples flagged; 0 with
    a rival "lives in" city).

  A pair enters if P and P′ are solved, and k2 also needs P″. Passes:
  - S1: P's residual at (P_B, l) overwritten by P′'s, layers 0–35, greedy generation from P's own prefix cache;
  - S2: the same at P_last, layers 0, 3, …, 33.

  Controls:
  - k1: P's residual plus a random vector of norm ‖h_P′ − h_P‖ (the §44 rev 6 difference-matched control);
  - k2: the donor P″ (answer unchanged), share of C1′;
  - k3: the input embedding at P_B overwritten by B′'s (token substitution), which must reproduce P′'s outcome on
    ≥ 0.95 of pairs.

  **Transfer(l)** = share of pairs concluding C1′. L\* = the largest layer such that every l ≤ L\* has Transfer ≥ 0.70,
  lower bound ≥ 0.55, k1 ≤ 0.10 and k2 ≤ 0.10. Under 30 held pairs, the result counts as under-powered.
- **Readout (rev 4): the conclusion.** F2: the first pool city or "unknown" on the `Answer:` line (max 160 tokens). F1:
  the last pool city or "unknown" in the generation, to end of turn (max 96). Outcome C1 / C1′ / other / abstain. The
  same rule decides the solved filter, k3, Transfer and S2.
- **§5 decision rule (fixed in rev 0):**

  | probe first decodable at P_last | L\* at S1 | verdict |
  |---|---|---|
  | ≤ 29 | ≥ 10 | instrument blind spot |
  | none ≤ 29 | < 10 | route late, lenses adequate |
  | ≤ 29 | < 10 | represented, not used |
  | none ≤ 29 | ≥ 10 | route late at P_last, instrument blind at P_B |

**What changed between rev 0 and the held read, and why (each before any patched pass or held read).**
- Rev 0's natural pairs do not exist: v1 keeps the gold on 12 of 12 bases, and B′'s fact never appears in v0. Hence the
  triples (rev 1).
- "Greedy ≤ 8 tokens" cannot read the outcome: the gold first appears at generated token median 10, and within 8
  tokens on 2 of 105 §45 rows. Rev 1 used a first-city rule.
- The first-city rule misreads restatements. The model lists every fact, including the inserted C1′, before
  concluding. Laptop F1 dev baselines (36 triples, pre-rev-4 code at max 48): solved as intended 100 / 96 / 91 of 108
  under conclusion / first-city / §45 rule. Hence the conclusion readout (rev 4).
- Base-grouped dev folds hold out the label itself (each dev label lives in one base). Hence (format × hardness) folds
  (rev 3).
- With those folds c1 is a near-sufficient statistic for the target within dev, so the dev-CV selectivity clause is
  unsatisfiable by construction. Rev 5 made it descriptive.
- Solver (rev 5, implementation): the first dev fit ran 10–70 s per fit at d 2048 (about 2 days) and was stopped after
  8.5 h. The fit now projects the standardised features onto the span of the training rows (exact for L2: the optimum
  lies in that span). Check on a dev fold: same penalised objective (25.803490 both, C 1, layer 10); the full fit's
  weights lie in the row space to 1e-14; argmax identical on 12/12, 12/12 and 23/24 at C 0.01 / 0.1 / 1. The dev
  probe then ran in 85 s.
- **Disclosure (rev 5 §4):** the first equality check of the two solvers compared their argmax on held rows'
  features. No held label was read and no accuracy computed. It was redone on the dev fold (the figures above), and
  ruled not a selection on held, since exactness fixes the solver.

**Run and coverage (pod zac774vlxnyj5f, A40 secure $0.59/h, EU-SE-1, created 07:18:11 UTC 2026-10-31, deleted
≈ 07:42:40, ≈ 24.5 min).**
- Pre-launch checks:
  - 74, then 18 + neighbour suites, tests pass;
  - D-30 PASS on 2b24f6d: fresh venv from the bundle; stub chain all ok; probe and report dev rc 0; both `--held`
    refuse without the marker; the projection limit stops the chain (exit 7); the deadline cut is partial with no
    traceback.
- D-22 smoke on the pod (3 dev triples, passes forced): k3 3/3; about 0.056 s per pass at S1 (36 layers plus k3 in one
  batch). Projection $0.263 against the $1.80 stop.
- Chain 1276 s of the 14400 s stop: capture 32 s (576/576), baselines 405 s (432 prompts), validate, S1 dev + held
  231 s, controls 342 s, S2 131 s; exit 0.
- Dev readout check (216 dev baselines): conclusion vs §45 rule agree on 207, conclusion vs first-city on 160. Solved
  as intended: 174 / 170 / 165 under conclusion / first-city / §45.
- The dev design was frozen (`design.json`, sha `8b54cf5e…c67e`, `design_committed.json` at cde2d08). The Advisor
  recomputed the sha from the committed code before the go. The held read ran once (11d8f61).

**Measurement B, held (55 pairs entered; 47 with k2). Dev beside it (51; 47).**

| layer | 0 | 1–3 | 4–9 | 10–15 | 16 | 17–25 | 26 | 27–31 | 32 | 33–35 |
|---|---|---|---|---|---|---|---|---|---|---|
| Transfer held | 0.927 | 0.945 | 0.818–0.855 | 0.727 / 0.727 / 0.727 / 0.764 / 0.764 / 0.800 (LB 0.598–0.676) | 0.691 (0.560, 0.797) | 0.60–0.73 | 0.527 | 0.25–0.31 | 0.036 | 0 |
| Transfer dev | 1.000 | 0.980 | 0.82–0.90 | 0.765 / 0.765 / 0.745 / 0.725 / 0.706 / 0.686 (LB at 15: 0.550) | 0.706 | 0.39–0.65 | 0.392 | 0.14–0.20 | 0.020 | 0 |

- **L\* = 15 held** (dev 14: layer 15 misses on the share, 0.686 against 0.70 (lower bound 0.550)).
- **k1** 0/55 held (0/51 dev) at every layer. **k2** 0/47 held at every layer except L20 (1/47 = 0.021); 0/47 dev.
  k2 is 0 under both the unused-name and the reused-distractor B″ groups.
- **k3 52/55 = 0.945 (0.851, 0.981) held, FAIL by one pair** (dev 49/51 = 0.961, pass).
  - The three held misses (RC03_v0_F2_H0, RC07_v0_F2_H2, RC11_v3_F2_H2) are all F2, and all are substitution →
    abstain where P′'s own baseline concluded C1′. The generated note diverges in its first tokens.
  - Direction of bias: an abstention is not a transfer, so the artefact lowers Transfer and cannot raise k1 or k2. It
    biases against the L\* reported.
  - Rev 0's alternative ("or the capture path is wrong") is not excluded. The Executor's reading (bf16 numerics in the
    batched suffix pass, prefix cache split at P_B, against the batch-1 baseline) is plausible and unchecked.
  - An optional dev-only batch-1 diagnostic is left to the user; the held misses are not re-run.
- **S2** (P_last, layers 0, 3, …, 33): patch and k1 both 0/55 at every layer, held and dev. **This is zero by
  construction:** P and P′ share the last prompt token, and the prefix cache still holds P's facts, which the model
  restates before concluding. S2 says nothing about what P_last carries.
- **Attribution × outcome (descriptive):** the supervisor the generation names before its conclusion (regex on
  "{A}'s supervisor is X", "{A}'s supervisor, X,", "supervisor X", pool names only), against the outcome.

  | layer | held | dev |
  |---|---|---|
  | 14 | B′/C1′ 38, B′/abstain 7, no name 8 | B′/abstain 7 |
  | 20 | B′/C1′ 32, B′/abstain 15 | B′/abstain 10 |
  | 27 | B/C1 18, B′/C1′ 15, B′/abstain 10 | |
  | 33 | B/C1 44, none/C1 8 | |

  Abstentions rise with the patch layer (held 3–4 at L0–3, 14–17 at L17–29), and the original supervisor B takes over
  from L27.

**Measurement A, probe (n 144 per read and split).**
- Held, RC supervisor @ P_last, layers 0–35: 0.06 0.08 0.08 0.09 0.08 0.08 0.06 0.07 0.08 0.04 0.06 0.09 0.08 0.06 0.06
  0.08 0.06 0.07 0.06 0.06 0.04 0.05 0.07 0.08 0.08 0.07 0.08 0.10 0.11 0.08 0.08 0.08 0.05 0.06 0.06 0.06.
  - Lower bounds 0.02–0.07; mean 0.071.
  - 1/12 = 0.083; c3 chance among names present 0.264; held c1 0.04–0.14.
  - **No layer decodable.**
- Held, the other reads (no layer decodable for any):
  - RC @ P_q: 0.01–0.23 (max 0.229 at L3, LB 0.17);
  - TU current: 0.01–0.10;
  - TU superseded: 0.01–0.10, a redundant read (a bijection of TU current within each base case; identical at 34 of
    36 dev layers).
- Dev CV (descriptive, rev 5), probe / c1 / c2 for RC @ P_last: 0.896 / 0.931 / 0.688 at L0, 0.444 / 0.549 / 0.292 at
  L29, 0.972 / 0.958 / 0.667 at L35. Chance 0.264; c1 ≥ probe at 33 of 36 layers.
  - RC @ P_q: c1 1.000 at every layer.
  - TU current: 0.729 / 0.965 at L0, 0.750 / 0.993 at L35.
  - dev_ok (the former gate) holds at no layer for any read.

**The probe arm's limitation (post hoc, labelled; the verdict is not revised).**
- The dev probe fits base-case identity, not the supervisor's name. Dev has 6 RC base cases and 12 labels, each in one
  base only (B in v0 / v2 / v3, B′ in v1), so within dev the label is a function of (base, variant). No dev-fitted
  probe can separate "the supervisor's name" from "this base case". The held read is a genuine across-base test of a
  probe that had no across-base training signal.
- Two facts say the arm is uninformative rather than negative:
  - At L31–35 the probe is at 1/12, although the dev-only §45 audit put the supervisor's name at median rank 6 at the
    last prompt token at L33. A readable signal exists there that this probe cannot pick up.
  - The same design at P_q is also at chance, although the questioner's name is the token under it.
- So the first clause of row 4 ("route late at P_last") is **not established**. The second ("instrument blind at P_B")
  rests on the intervention arm alone and stands.
- Deciding the first clause would take many base cases per label, with names varied independently of everything else:
  a new case list, not a re-fit (results review §4).

**What was found.** In plain words:
- The 3B writes "who the supervisor is" into its internal state at the supervisor's name and uses it to pick the city
  through the first 15 of 36 layers. Swap in another supervisor's state there and the answer follows on 0.73–0.95 of
  unseen cases. A random push of the same size, or a donor with the same answer, does nothing.
- The §45 lenses saw nothing at that position before layer 30, so there they were blind to something that drives the
  answer.
- When the swap comes in the middle layers and fails, the model usually names the new supervisor and then says
  "unknown", rather than answering with the old city.
- Whether the same information can be read at the answer position before layer 30 is still open. The probe could not
  learn it from this case list.

**Why it matters.**
- §45's "late" picture is, at the entity position, a limit of the instrument, not the model's route.
- The intervention arm meets one of the two verdicts that open the §44 bridge (proposal §6): the same intervention on
  the §44 rows, asking whether moving the gold's binding changes refuse against answer. That is a separate amendment
  and the user's decision. The probe arm would need the new case list before it could carry weight there.
- D-45 / D-53: §46 is documented and closed. No rescue variant, no re-fit, no new probe.

**Caveats.**
- k3 below its bar by one pair (above); the capture path is not proven exact on held F2.
- The triples are a §46 addition, with two extra "works in" facts beyond the §45 prompts.
- Greedy decoding, bf16, one model, toy prompts.
- The attribution table is a regex description.
- S2 is uninformative by construction.
- "Not decodable" is a statement about this probe and this training set.
- No sentence here says "erased", "suppressed" or "overridden" (proposal §5).

**Spend.** One A40 pod, ≈ 24.5 min, ≈ **$0.24**. Laptop F1 dev baselines, probe fits, D-30 and the reports: $0. All
experiments ≈ **$21.67 of $30**.

## 47. Does the evidence's value reach the answer computation on in-view failures? The §46 interchange at the evidence span on the §44 rows, Qwen 2.5 3B (2026-10-31, pre-registered; pod run 2026-10-31)

**Status: COMPLETE (pooled read). B0 PASS, B1 yes, B2 "answer production on these rows" as computed, B3 / B4
descriptive. The §44 bridge closes under D-45 / D-56 (record change D-59 pending the user's Go).**
- **On the 26 rows whose forced readout follows the evidence (13 successes, 13 failures: 8 wrong, 5 refusal), the
  gold's value stays in the residual at its own evidence token(s) through layer 30 of 36.** The answer position reads
  it from there at layers 31–32, identically on failures and on successes.
  - Transfer 13/13 at L0–21 and 11/13 at L27–30 in both classes, 0 from L33.
  - The difference-matched random vector moves it on 0 of 936 reads.
  - Embedding substitution reproduces the swap prompt on 26/26.
- **Ruling 1 (Advisor):** the entry filter is §44's class rule, so every entered row is a §44 answer-production row.
  B2 confirms §44's classification layer by layer. It cannot find an evidence-use failure, because such a row never
  enters.
- **For Q18:** the §44 rev 8 sample contains no evidence-use failure. The distinction is documented on one side only
  and is not established as a contrast (D-45).

Pre-registration `research/proposals/2026-10-31-exp47-bridge.md`:
- rev 0 at be2b9e5;
- rev 1 at a9b70ff: sites = exactly the differing positions, all occurrences; thresholds 11/13; fork readout;
  Answer-line stop at 160; logits_to_keep / chunks; smoke rows; k1 and the two sanities as the controls.

Decisions: D-56 (the user opened the bridge: "go bridge skip"), D-57 (the build: "build"), D-58 (the pod: "Yes to the
pod at 0.59 hard stop 1.50", typed in both panes). Reviews:
- `research/reviews/2026-10-31-exp47-build.md` (ACCEPT at 1689372 after the report condition; D-30 e5ea27f);
- `research/reviews/2026-10-31-exp47-results.md` (e9cdef1): the independent recount, rulings 1–10.

Code:
- `memctl/analysis/bridge47.py`;
- `scripts/exp47_rows.py`, `exp47_patch.py`, `exp47_report.py`;
- the pod chain `exp47_pod.sh`, `exp47_pod_run.py`, `exp47_bundle.sh`, `exp47_d30.sh`;
- `tests/test_exp47.py` (8 tests).

Commits: ffc8aa1 (build), 1689372 (review condition), e5ea27f (D-30 log), 83876e5 (the run, read once). Rows
`runs/exp47_bridge/rows47.jsonl` (sha `d7fa444d…`). Run `runs/exp47_bridge/cuda-float32/`: 36 per-row json, report
sha256 `0b8fea91…87e5`. fp32 (the §44 rows were read in fp32). **Read once, pooled.** All 70 rows were read under §44
(logit Δ, classes, P1 patching) and §45 stage 3; there is no unread split. The thresholds are §46's, fixed before the
build, so no choice was made on these rows.

**The question (proposal §1).** When the 3B fails with the evidence in view, is the evidence's binding carried to and
used by the answer computation (a failure downstream, of answer production), or not (a failure of evidence use)? §44
asked this with logit differences at the answer position and a swap whose validity read failed. §47 applies the tool
that reached validity in §46 to the §44 rows, unchanged: interchange of the residual at the evidence token itself,
with a difference-matched control.

**Design (fixed before the run).**
- **Rows.** The 70 §44 rev 8 swap rows (clean prompt with the gold G in one evidence line; swap prompt with the
  substitute S in its place, every other byte identical). A row is position-aligned when clean and swap have equal
  token length: 36 of 70 (success 20 / refusal 6 / wrong 10). The §44 forced ids are rebuilt with §44's own code and
  sha-checked against the S files.
  - Sites S1 = exactly the differing token positions, every occurrence: 1 position on 18 rows, 2 on 8, 3 on 4, 4 on 3,
    8 on 2, 9 on 1.
  - Three rows carry the gold more than once in the evidence line (3e321797, 8b9d4367, c7cf7dfd; all success, all
    entered).
- **Entry.** A row enters the Transfer read if its forced-P1 preference follows the evidence at 1 nat (Δ_clean > 0 >
  Δ_swap and Δ_clean − Δ_swap ≥ 1): 26 = success 13 / refusal 5 / wrong 8. Entered rows' Δ_clean runs 0.33 … 26.57 and Δ_swap −26.51 …
  −2.62.
  - **This filter is §44's class rule at margin 1.** So all 26 entered rows carry the §44 class answer_production,
    and the 10 non-entered rows are indeterminate 7 and evidence_use 3. This is a property of the design, recorded in
    the results review (ruling 1). Rev 0 §7's sentence that the class labels are "never inputs to B1–B3" is corrected
    there; the proposal is not edited after the read.
- **Passes, per row, every layer 0–35.**
  - The clean prompt's residual (output of decoder layer l) at the sites is overwritten by the swap run's at the same
    positions. The prefix before the first site is cached; the suffix is recomputed (logits for the last position
    only, chunks of 19 variants).
  - **Readout (i):** Δ = logit(G) − logit(S) at P1 under the §44 forced `Answer:` tail, from a copy of the patched cache.
    T(l) = (Δ_patched − Δ_clean) / (Δ_swap − Δ_clean); "transferred" = T ≥ 0.5.
  - **Readout (ii):** greedy generation from the other copy to the end of the Answer line (max 160); G / S / unknown
    / other, D1-normalised.
- **Controls.**
  - k1: per site position, h_clean plus a random direction of norm ‖h_swap − h_clean‖.
  - k3: the swap tokens' input embeddings at the sites (the swap prompt itself), which must reproduce the swap run's
    forced top-1 and generated outcome on ≥ 0.95 of entered rows.
  - No same-answer donor (k2) can be built on natural rows (rev 1 ruling 7).
  - S2 = P1 of the forced sequence, layers 0, 3, …, 33, patch and k1, readout (i).
- **Sanity, per row:** |T(35)| < 0.05; |1 − T(k3)| < 0.05; |Δ − Δ_§44| < 0.05 on the unpatched clean and swap runs.
  More than 5 % failing exits.
- **Criterion (§46 §4 verbatim).** L*_class = the largest layer such that every l ≤ L* has Transfer ≥ 0.70, Wilson
  lower bound ≥ 0.55 and k1 ≤ 0.10. With 13 entered rows per class this needs 11/13 (LB 0.578; 10/13 has LB 0.497).
  L* ≥ 10 is mid-stack carriage.
- **Reads.**
  - B0: k3.
  - B1: L*_success (≥ 10: §46's carriage holds on natural rows).
  - B2: L*_failure against L*_success (≥ 10 on both: the failure is downstream of evidence use; < 10 on failures
    only: evidence-use failure).
  - B3: refuse ↔ answer (descriptive under 8 refusal rows).
  - B4: per-row tables and S2.

**Run (pod yhnu0jzbs8rl67, A40 secure $0.59/h, EU-SE-1, created 23:40:33 UTC 2026-10-31, deleted 00:14:24 UTC,
lifetime 33 min 51 s; $0.34 posted: GPU $0.333 + disk $0.007).**
- Pre-launch: 8 tests (26 with §46's); D-30 PASS (fresh venv from the bundle, stub chain, projection stop, deadline
  cut).
- Laptop measure on 5809eb10: CPU fp32. Peak RSS was set by the model load, so the GPU peak was measured on the pod.
- D-22 smoke on ec81a493 / 7401057b / 5809eb10:
  - per-row 26.5 / 83.9 / 61.7 s;
  - peak CUDA 19.5 GB, so batch 19 (the rule: 10 above 36 GB);
  - k3 3/3; sanities 0; |Δ − Δ_§44| 0.0000 on all six baselines.
- Projection $0.613: the slowest smoke row × 36 × 1.15 plus the seconds spent; above the Advisor's $0.30–0.50 quote
  (the rule projects the slowest smoke row over all rows); below the $1.00 exit, so the chain
  continued under the approved $1.50 stop.
- Rows 1653 s (45.9 s per row; the slowest, 0e4e4c46, 112 s); chain 1923 s; exit 0.
- Sanity across all 36 rows: 0 failures (max |T(35)| 0.0014, max |1 − T(k3)| 0.0007, max |Δ − Δ_§44| 8e-5).
- The Advisor recounted every number below from the 36 per-row files with an independent script: 0 mismatches.

**Results.**

**B0: k3 26/26 = 1.000 (LB 0.871), PASS;** 36/36 on all rows (forced top-1 and generated outcome both equal the swap
run's).

**B1 and B2: L\* = 30 for both classes.** Transfer (readout (i)), k of n entered rows with T ≥ 0.5; k1 in brackets.

| layers | success (13) | failure (13: 8 wrong, 5 refusal) | pooled (26) |
|---|---|---|---|
| 0–21 | 13 [0] | 13 [0] | 26 |
| 22–26 | 13 [0] | 12 [0] | 25 |
| 27–30 | 11 [0] | 11 [0] | 22 |
| 31 | 5 [0] | 9 [0] | 14 |
| 32 | 2 [0] | 2 [0] | 4 |
| 33–35 | 0 [0] | 0 [0] | 0 |

- The 11/13 bar holds at every layer through 30 in both classes. L31 fails on the share for success (0.385) and on
  the lower bound for failure (0.692, LB 0.424).
- k1 over the 936 control reads on entered rows: T max 0.344, min −0.241, median 0.006, 0 at or above 0.5.
- 71 of 936 patch reads fall in [0.3, 0.7): none at L0–15, one each at L16–19, three per layer at L20–26, six per
  layer at L27–29, then 10 / 12 / 6 at L30 / 31 / 32.
- Without the three multi-occurrence rows (descriptive): success n 10, L* 30.
- Per-row L* (last layer of the initial run with T ≥ 0.5), entered rows: 21 on 1 row (59524333, wrong), 26 on 3
  (c19f7a0b refusal; 3e321797, 95bcc1c8 success), 30 on 7, 31 on 11, 32 on 4. Medians: success 30 (n 13), refusal
  31 (n 5), wrong 31 (n 8).
- **B1 reads "yes"** (L* ≥ 10): the carriage holds on natural rows. **B2 reads, as computed, "the binding is carried
  and used on failures as on successes: the failure is downstream of evidence use (answer production), on these
  rows".**
- **Caveat on B2, in the result itself (ruling 1).**
  - Every entered failure is a §44 answer-production row by construction of the entry filter, so B2 could not have
    found an evidence-use failure: such a row never enters.
  - What B2 adds to §44 is layer resolution. On the 13 failures whose readout follows, the gold's value is carried at
    its own positions through layer 30 and read from there at layers 31–32, and the random-vector control never moves
    it, exactly as on the 13 successes.
  - The §44 classification of these rows is confirmed mechanistically. It is not extended to a class §44 did not find.
- **S2 at P1:** patching the answer position alone moves the readout only from layer 33 (share with T ≥ 0.5: 0.038
  at L21–24, 0.077 at L27–30, 0.962 at L33). This is identical to §44's forward patch on the same rows (0 disagreements
  in 312 layer-row reads, max |ΔT| 1e-5). It is a **pipeline validation** (two code paths agree), not a result.
- **Generated readout (ii):**
  - Baselines, entered rows (clean → swap):
    - success G 11 / unknown 1 / other 1 → S 9 / G 1 / unknown 2 / other 1;
    - refusal unknown 5 → unknown 3 / other 2;
    - wrong other 6 / G 2 → other 6 / unknown 2.
  - Consistency with the §23 labels: the refusal rows refuse under this readout on 6/6 (5/5 entered); the success
    rows give the gold on 15/20 (11/13 entered). The §23 label was scored on a different generation path.
  - Patched success rows give S on 8–9 of 13 at L0–18 (the swap baseline gives 9), 5–7 at L19–30, 4 at L31, 1 at L32,
    0 from L33; G returns from L31 and is back at the clean 11 at L33. k1 never lowers G below 10.
  - **The generated outcome tracks the forced Δ at L0–18 and lags it at L19–30**, where "unknown" rises. Readout (ii)
    is weaker than readout (i).

**B3 and the induced abstentions (descriptive; 5 and 13 rows).**
- **Refusal rows.** All 5 entered refusal rows refuse at baseline under this readout. The patch produces the
  substitute's answer S on 1 row at L0 (ad7109d1: a recap naming "132 points" that the §23 rule does not class as a
  refusal, scored S by the D1 rule), 0 at L1–16, 1 at L17, 2 at L18–19 and 1 at L20–31; k1 produces an answer on 0/5 at every layer
  except two single reads (G at L15–16 on one row). The two rows:
  - 01493427 (gold 25, substitute 33; knowledge-update) gives S at L17–19 and L22–31.
  - ed4ddc30 (gold 20, substitute 32; knowledge-update) gives S at L18–21.

  **The swap prompt itself gives "other" on 01493427 and "unknown" on ed4ddc30**, so neither the gold text nor the
  substitute text yields the answer there. The patched generations are confabulated multi-entry recaps ("12 from
  2023/11/30 + 33 from 2023/12/20"; "32 dozen eggs … on 2023/03/09 … on 2023/03/15"). B3's share minus k1: 0.20 at
  L0, 0 at L1–14, −0.20 at L15–16 (k1's two G reads), 0.20 at L17, 0.40 at L18–19, 0.20 at L20–31. Five rows; nothing is claimed. In §44–§47 this is the only content-bearing
  intervention that turned an in-view refusal into an answer, and the answer was the substitute's, the wrong one.
- **Induced abstention on success rows.** Patch "unknown" exceeds k1's by 0.08–0.15 at L0–12, 0.08–0.23 at L13–18,
  0.23–0.38 at L19–29, 0.15 at L30, and 0 from L31. Three rows abstain under the patch where neither the clean prompt,
  the swap prompt nor k1 does: b6019101 (L19–29), c7cf7dfd (L17–21), ec81a493 (L13–30). This is the same mixed-state
  pattern as §46's 14–17/55 abstentions at L17–29: the sites carry the substitute while every later position's state
  below layer l was computed from the gold.

**Non-entered rows (10; descriptive only, not read).**
- indeterminate 7 (success 4, wrong 2, refusal 1) and evidence_use 3 (all success: 26bdc477, 5e1b23de, 6aeb4375).
- The three evidence_use rows have |Δ_swap − Δ_clean| of 0.96, 0.02 and 0.17, so the forced readout barely
  distinguishes gold from substitute on them. On all three, clean and swap generate the same outcome. T is numerically
  meaningless there (5e1b23de's runs 0.5–4.7). These are rows on which this readout is insensitive, not rows on which
  the evidence fails to arrive.
- On the seven indeterminate rows with usable denominators (2–17 nats), the T profiles end like the entered rows' (0 from
  L33) but fall earlier on two of them (5809eb10 drops below 0.5 from L18, ccb36322 from L19); per-row L* 17–32.
- k3 reproduces the swap on all 10. The report's L* for these rows is driven by the two near-zero denominators and
  is not quoted.

**Comparison with §46 (ruling 3).**
- §46's L* was 15, at the supervisor-name token, with the readout at the generated city two hops away (name →
  workplace → city). Here the readout is the gold value itself, one hop (span → forced P1), and the value stays at
  its own positions until the final read.
- The two depths are not in tension: each measures the depth at which the reading position stops consulting the
  written position for that task.
- T ≈ 1 at L0–21 is expected whenever the readout copies the span's value: below the layer at which P1 reads the
  sites, a patched residual there is equivalent to the token substitution (k3 = 1.0). The non-trivial content is the
  reading depth (31–32) with the controls at 0.
- "§46's mid-stack carriage holds on natural rows" is true. The bridge does not replicate §46's L* 15; it measures a
  different hop.

**What was found.** In plain words:
- On the §44 rows whose answer follows the evidence, the model keeps the evidence's value at the evidence's own
  tokens until layer 30 of 36 and copies it into the answer at layers 31–32.
- It does this the same way when its final answer is wrong or a refusal as when it is right.
- A random push of the same size never moves the answer.
- On these failures the evidence's value is available to the answer computation layer by layer, and the failure
  happens after that.
- But §44's sample had no failure of the other kind, so §47 could not compare the two kinds.

**Reading for Q18 (ruling 2).**
- §44's classes at 1 nat were refusal 11 / 0 / 3 and wrong 15 / 0 / 3 (answer production / evidence use /
  indeterminate). The only evidence_use rows are three successes with a near-zero readout difference.
- On these extractive single-hop rows every classifiable failure is an answer-production failure, and §47 shows that
  the carriage behind that class is layer for layer the same as on successes.
- The distinction Q18 asks for is therefore **documented on one side only; not established as a contrast**. D-45
  applies: document, do not invent a population that would supply the missing class.

**Limits.**
- Rev 0 §7 as corrected: the entry filter is the §44 class rule (ruling 1).
- Only equal-length substitutes: 36 of 70 rows, biased toward short golds of the same surface type.
- Extractive single-hop rows only.
- One model, greedy decoding, fp32.
- Readout (i) is a forced first-token logit difference. Readout (ii) lags readout (i) at L19–30.
- No same-answer control on natural rows.
- 5 refusal rows make B3 descriptive.
- The rows were previously read for other measurements (pooled).
- No sentence here says "suppressed", "erased", "overridden" or "gate".

**Spend.** One A40 pod, **$0.34 posted** (GPU $0.333 + disk $0.007; lifetime 33 min 51 s at $0.59/h). Laptop build, tests,
D-30 and the report: $0. All experiments ≈ **$22.00 of $30** (D-58).

## 49. Does a target decoded from the failing model release correct answers on false abstentions, and what does it cost where the evidence is insufficient? Decoded-target repair against random, a content-free push and an instruction change, Qwen 2.5 3B (2026-11-01, pre-registered rev 0a/0b; pod run 2026-11-02)

**Status: COMPLETE (STATE F16, D-68). Q-P SHOWN as direction-specific recovery against random; NOT selective. Q-6 not
interpretable. Closes under D-45; B does not open on this result.**
- **Who the false abstentions are.** 152 of the 165 held false abstentions already name the gold city in the
  model's own note before it writes "Answer: unknown". Before the note (P_pre), the 12-way probe is below the
  in-prompt candidate floor on every subset. So these failures are "note, then refuse", not a hidden answer the
  model never writes down. "Legible" does not apply to them.
- **Q-P (primary).** On 165 held false abstentions (90 bases), decoded-target repair R1 releases the gold city
  0.903 of the time, against 0.067 for a matched-norm random vector R3. R1 − R3 = **+0.836 [+0.774, +0.894]**
  (paired, base-clustered bootstrap).
  - On the 144 genuine abstentions (unanswerable twins), R1 names a city on **144/144**. Unsupported R1 − R3 =
    +1.000 [+1.000, +1.000].
  - The recovery is direction-specific, and it is not selective.
  - R1 outputs the probe's pick on every trial (ANS 165/165, UNA 144/144, CF 51/51), so its gold rate equals the
    P_ans probe accuracy (149/165). The repair adds no information beyond the decode, and the decode at P_ans mostly
    reads the note.
  - By note flag, R1 releases gold on 146/152 when the note names the gold and on 3/13 when it does not.
- **R1 against the content-free push R2.** R2 is the §38/§40 layer-29 vector at α 2, here at P_ans only.
  - Gold on false abstentions: R1 0.903 against R2 0.927. R1 − R2 = −0.024 [−0.069, +0.013], so no difference is
    shown.
  - Unsupported on genuine abstentions: R2 0.188 against R1 1.000.
  - The decoded target adds nothing measurable over the content-free push, and it costs more where the evidence is
    absent.
- **R4, the instruction change.** Both "unknown" sentences are removed from the prompt.
  - Gold on false abstentions: 1.000.
  - Unsupported on genuine abstentions: 0.451 (degenerate 0.500).
  - Damage on held correct answers: 3/265.
  - R1 − R4 gold = −0.097 [−0.149, −0.052].
- **Gates (secondary; templates only).** On H1 abstentions (56 false, 144 genuine), a P_pre sufficiency probe
  (G-suff) separates the two classes almost perfectly (AUROC 0.999).
  - G-suff fires on 96.4% of false and 0.7% of genuine abstentions.
  - U(G-suff) − U(never) = +0.235 [+0.179, +0.283] at λ 1, and +0.215 at λ 2.
  - G-content gives +0.125 [+0.080, +0.169].
  - UNA twins are built by template (the city fact replaced by distractor lines, at H1 only), so surface cues may
    remain (Codex review point 1); this is not general recognition of evidence sufficiency.
- **Q-6 (abstention against wrong-answer repair): not interpretable.** The *Legible Failures* calibration did not
  reproduce on F1: LF − random gold on F1-wrong = +0.043 [−0.065, +0.152] (n 46, 24 of them no-city), and
  +0.091 [−0.136, +0.318] on city-wrong only (n 22). No contrast between abstentions and wrong answers is claimed.

Pre-registration `research/proposals/2026-11-01-exp49-legible-abstentions.md`: rev 0a; rev 0b amendments B1–B8 from
Codex's review, before any build.

Decisions:
- D-63 (the user opened §49 and set the $5 pod allowance);
- D-64 (the user's standing pod approval in the Executor pane);
- D-67 (price cap raised to $1.20/h, dollar limits unchanged, after no A40 / A6000 capacity).

Reviews:
- `research/reviews/2026-11-01-exp49-build.md` (PASS after 4 fixes, 88bae3e);
- `research/reviews/2026-11-02-exp49-results.md` (ACCEPT; independent recount from the per-trial files).

Code:
- `memctl/analysis/abst49.py`;
- `scripts/exp49_cases.py`, `exp49_run.py`, `exp49_report.py`;
- the pod chain `exp49_pod.sh`, `exp49_pod_run.py`, `exp49_bundle.sh`, `exp49_d30.sh` (D-30 PASS from a fresh venv:
  exits 0 / 7 / 9 / 11);
- `tests/test_exp49.py` (22 tests).

Commits:
- e349f22 (build), 88bae3e (review fixes), c5af1a3 (pod path);
- **952cc80, the design file**, committed before any held output existed and verified by the pod against its own
  shallow clone;
- 24b9272, the held read: read once, report sha256 `580ef394…`.

Run details:
- Cases `runs/exp49/cases49.jsonl` (sha `d11894ac…`; 1,584 rows, 792 per split, split by base parity, twins in
  their base's split).
- Run `runs/exp49/run/`, with per-trial `baselines/`, `arms_dev/` and `arms_held/`.
- Pod vgg9yi8dwzqe1v: one secure L40S, OC-AU-1, $1.09/h, fp32, single rows; 2,589 s of compute, about 46 min
  lifetime, $0.847 posted.
- A laptop dev run was started and abandoned at 18/792 for speed (archived, unused).

**Design (fixed before the held read; design.json in 952cc80).**
- **Population.** Independent bases with gold balanced over 12 cities per split × task. City questions only
  (DE / RC / TU). Each base yields:
  - ANS-F2 at H0 / H1 / H2;
  - a counterfactual twin CF-H1 (gold → another city);
  - an unanswerable twin UNA-H1 (the city fact(s) replaced by the same number of distractor lines; one nuisance
    fact in every prompt, B1);
  - ANS-F1 at H1 (24 bases per task).
- **Sites.** P_pre is the last chat-template token (the assistant-header newline), as in §45/§46 P_last. P_ans is the last token of the model's own teacher-forced prefix
  through "Answer:".
- **Dev parts.** Dev is split fit / select / gate-tune (dev index mod 3). Chosen values:
  - ℓ*: P_pre 35, F1 35, P_ans 33; sufficiency probe layer 24 (select AUROC on H1).
  - α (R1) 0.5 by the tie rule: the objective was flat at −0.105 for every α, because R1 releases every tune trial
    at α 0.5 already (gold 0.895 on 19 false, unsupported 1.000 on 47 genuine).
  - α_LF 0.5.
  - τ_c 0.983, τ_s 0.736.
  - All 12 centroids are estimable from F2.
- **Arms at P_ans** (step 0, greedy, 16 tokens): R1 = α·‖h‖·unit(μ_ĉ − μ_abst); R2 (one-position); R3 random at norm
  α·‖h‖; D-gold (diagnostic). R4 regenerates from the R4 prompt.
- **Scoring.** patch46 `first_city`, no judge. Degenerate scores 0 in U.
- **Sanity.** No-op replays: 20/20 dev and 20/20 held reproduce the baseline continuation token for token.

**Held baselines.**

| Set | Outcomes |
|---|---|
| ANS-F2 | gold 265, abstain 165, degenerate 2 (no Answer: marker; excluded from M2 / M3) |
| CF-F2 | gold 86, abstain 57, degenerate 1 |
| UNA-F2 | abstain 144 / 144 |
| ANS-F1 | gold 26, wrong 6, competitor 16, degenerate 24 |

**Q-3, tracking (H1).** On the 51 bases where both the ANS-H1 and the CF-H1 baselines abstain, R1 and R2 each
release on 51 and name the CF city on 50: tracking 0.980 [0.941, 1.000] each, with their own denominators and not
compared with each other. The held CF-H1 abstention rate (0.396) is not lower than ANS-F2-H1's (0.389).

**M1 decoding (diagnostic).** At P_ans the probe reads F2-abstain at 0.903: 0.961 when the note names the gold, 0.231
when it does not; the shuffled-label probe is at about 0.08. At P_pre it reads F2-abstain 0.127 and F1-wrong 0.136,
against candidate floors 0.42 and 0.29. So decoding at P_ans is mostly reading the note.

**Spend.** One L40S pod, **$0.847 posted** (GPU $0.840 + disk $0.006; lifetime about 46 min at $1.09/h). A
guard-deleted L40S (6ya4wnygo4efq7, about 1 s) posted $0.0003. The laptop build, D-30 and timing: $0. §49 total
**$0.847** of its $5 allowance (D-63 / D-64). All experiments ≈ **$22.85 of $30**.

**Limits.** Synthetic templates only. The repair acts at P_ans on the model's own prefix, so it shows an effect
there, not the source of the natural failure. The F1/F2 contrast is bundled. One held read; no rescue (D-45).

## 50. What separates a false "unknown" after an answer-bearing note from an answer? Elicited sufficiency and city-margin profiles, Qwen 2.5 3B (2026-11-02, pre-registered rev 1 + rev 1a; pod run 2026-11-02)

**Status: COMPLETE, stage 1 only (STATE F17, D-71). The elicited sufficiency measure S is invalid in every task (dev
gate failed), so no profile label is possible. The city margin Δ_M is "lower" on false abstentions in DE and RC; TU is
underpowered. Stage 2 did not open. Closed under D-45.**
- **The elicited sufficiency measure failed.** Asked "Does the memory give enough information to tell …? Reply yes
  or no.", the 3B says "no":
  - on most answered rows in RC (yes rate 0.01) and TU (0.29);
  - on every unanswerable row (UNA-rel yes rate 0.00 in every task × hardness cell);
  - and the reversed question agrees with it on only 0.18–0.44 of rows.
  - This is a "no"-leaning response, not a sufficiency judgement. So §50 cannot say whether false abstentions involve
    a sufficiency misjudgement.
- **The gold-versus-rival-city margin is lower on false abstentions than on answers** of the same task, hardness and
  city: DE Δ_M **−1.21 [−2.77, −0.05]**, RC **−3.13 [−3.87, −2.22]** logits ("lower" in both).
  - It is also lower within 47 content-matched flip pairs (rep a against rep b): −2.50 [−3.96, −1.28].
  - The margin stays large: the mean M_city on false abstentions is DE 13.2, RC 15.9, TU 9.8 logits. The model still
    clearly prefers the right city over the others when it refuses.
  - What it loses is the comparison against " unknown". Only about a quarter to a third of false abstentions are near
    ties (|z_gold − z_unknown| < 1): DE 0.23, RC 0.35, TU 0.29.
- **This is the pattern a "not sure enough" account predicts. It cannot confirm that account:**
  - M_city is an output-side margin, not a validated confidence measure.
  - The alternative (sufficiency) could not be measured.
  - DE's margin is partly read at the wrong token (post hoc caveat below).
- **No causal claim.** Stage 2 did not open (no task has a profile), and no arm was run.

Pre-registration `research/proposals/2026-11-02-exp50-why-false-abstention.md`:
- rev 1 at 032b1b1 (Codex rev 0 review resolved point by point);
- rev 1a at 4590dda (time budget: one pod session, timing-only scale-down, stage-2-only wall-clock stop, cap $3.00).

Decisions: D-70 (the user opened §50: "Go for it…"; pod spend "up to $3.00 total", typed in the Executor pane), D-71
(closed under D-45; the user's "Go, record §50 as closed"). Reviews:
- `research/reviews/2026-11-02-exp50-build.md` (PASS after 3 fixes, 43e45b0);
- `research/reviews/2026-11-02-exp50-results.md` (ACCEPT; independent recount from the per-row files; post hoc caveat).

Code:
- `memctl/analysis/why50.py`;
- `scripts/exp50_cases.py`, `exp50_run.py`, `exp50_report.py`;
- the pod chain `exp50_pod.sh`, `exp50_pod_run.py`, `exp50_bundle.sh`, `exp50_d30.sh` (D-30 PASS from a fresh venv:
  exits 0 / 7 / 9 / 11);
- `tests/test_exp50.py` (16 tests).
- The §49 engine, guard and shallow-clone mark are reused.

Commits:
- 3ed2258 (build), 43e45b0 (review fixes);
- 088af4a (answer-bearing rule fixed on the dev audit);
- **0a43669, the design file**, committed after the Advisor's design OK, before any held row existed, and verified by
  the pod against its own shallow clone;
- f482de6, the held stage-1 read (read once);
- d535a56 and 7d69805, the report additions after the read (per-task Q-4, post hoc sensitivity; pre-registered
  numbers asserted unchanged).

Run details:
- Cases `runs/exp50/cases50.jsonl` (sha `c6eb5ed7…`, 2,914 rows). Report `runs/exp50/run/stage1_report.json`; per-row
  `runs/exp50/run/rows/`.
- Pod yp1fhtaqavw57f: one secure L40S at $1.09/h (A40 and A6000 unavailable), fp32, single rows. Lifetime about
  72 min; 4,072 s of chain; $1.319 posted.

**Design (fixed before the held read).**
- **Cases.** New cases from the §49 generator at seed 50; 48 bases per task per split (the timing projected 1.9 h,
  so no scale-down). Sets per split:
  - ANS rep a (H0 / H1 / H2);
  - ANS rep b (every timestamp shifted +7 days, nothing else changed);
  - UNA-rel at every hardness;
  - UNA-swap at H1 (byte-identical memory, a question about a distractor with only a job line; 104 bases had no such
    distractor and contribute no row);
  - ANS-F1 at H1.
  - 6 exact duplicates of §49 prompts were dropped; 72 of 288 bases share a core tuple with §49 (reported).
  - Splits: dev-fit (base ≡ 0 mod 4), dev-check (≡ 2 mod 4), held (odd).
- **Measures.**
  - S = P_yes / (P_yes + P_no) over four token variants each, in a separate prompt. The yes+no mass was ≥ 0.5 on
    every dev F2 row.
  - M_city = z_gold − max of the other 11 cities, and M_commit = z_gold − z_" unknown", both at P_ans (the last token
    of the model's own teacher-forced output through "Answer:").
  - K_v: an elicited 0–9 rating, expected digit.
- **S validity gate (dev-check; all three conditions per task).** It **failed in all three tasks**:

| Task | Answered | AUROC vs UNA-rel | AUROC vs UNA-swap | Yes rate (answered / rel / swap) | Polarity agreement |
|---|---|---|---|---|---|
| DE | 47 | 0.997 [0.990, 1.000] | 1.000 | 0.70 / 0.00 / 0.00 | 0.34 |
| RC | 105 | 0.704 [0.655, 0.752] | 0.552 [0.412, 0.695] | 0.01 / 0.00 / 0.00 | 0.18 |
| TU | 133 | 0.996 [0.988, 1.000] | 1.000 | 0.29 / 0.00 / 0.00 | 0.44 |

  - The UNA-swap rows were few in dev-check (about 15–20 per task); the gate was not rescued.
- **Answer-bearing note rule, fixed on the dev audit before the design commit.** The Advisor hand-checked 40 dev-check
  notes; the rule agreed on 35. All 5 disagreements were TU "moved from X to {gold}", where "moved from" fell inside
  the 6-word window before a destination gold. The fix: "moved from" negates only the city right after it.
  - The dev flags were recomputed from the stored notes with no model calls (64 TU rows changed). Agreement after the
    fix: 40/40.
  - The 64 newly flagged dev TU rows have no K_v (it was elicited only on rows flagged at measure time). Held rows
    were measured under the fixed rule.
- **Stage-2 dev quantities (descriptive only).** They were computed on the pre-fix flags:
  - ℓ = 26: the first layer where the whole-residual patch releases gold on ≥ 50% of the 60 dev flip pairs (50 bases),
    at 0.58.
  - No-op replay 120/120.
  - u_S transfer: AUROC 0.971 [0.949, 0.987] (pass).
  - u_M transfer: Spearman 0.253 (fails the 0.30 bar, so P-M is untestable).
  - cos(u_S, u_M) = 0.16.

**Held stage 1** (joint population: ANS-F2 rows with an answer-bearing note, outcome gold or abstain, defined
M_city; pipeline intervals, 10k base resamples within task):

| Task | False abstentions (bases) | Answered (bases) | Cells | Δ_M (logits) | 95% CI | State | Profile |
|---|---|---|---|---|---|---|---|
| DE | 180 (48) | 94 (39) | fine; 126/180 covered | −1.209 | [−2.766, −0.052] | lower | Other (S invalid, M lower) |
| RC | 57 (30) | 223 (48) | fine; 57/57 covered | −3.129 | [−3.872, −2.216] | lower | Other (S invalid, M lower) |
| TU | 14 (9) | 258 (48) | fine; 14/14 covered | −4.43 | [−7.29, −1.71] | underpowered | none |

- AUROC_M: DE 0.55, RC 0.90, TU 0.89. Standardised Δ_M: DE −0.42, RC −1.86.
- **Stage 2 does not open:** no task has a B-, C- or Joint-compatible profile.

**Post hoc caveat: DE's margin is partly read at the wrong token (Advisor recount; it does not revise the state).**
- In DE, **49 of 94 answered rows begin their answer with a name**, for example "Answer: David lives in York" (TU 15
  of 258, RC 0 of 223). For those rows the token after "Answer:" is the name, so M at P_ans is not read at the step
  that commits to the city.
- With answered rows restricted to city-first answers (45 in DE):
  - under the §4.4 cell rule (fine cells cover only 76/180 = 42% < 50%, so task × hardness cells): DE Δ_M
    **−5.03 [−6.27, −3.90]**;
  - fine cells forced, beside it: −4.36 [−5.94, −3.08], covering 76 false abstentions.
  - RC is unchanged (no name-first rows). TU is −4.72 [−7.34, −2.20], still underpowered.
- The direction is unchanged and the gap larger. The pre-registered DE value (−1.209, "lower") stands in the record
  with this caveat.
- For any follow-up: read the margin at the position that produces the first city of the model's own output.

**Secondary quantities.**
- **Q-2** (selected, conditional): 47 flip pairs from 43 bases (DE 24, RC 15, TU 8). Paired M_city (abstained −
  answered) −2.500 [−3.960, −1.279]; paired s −0.011 [−0.033, +0.004] (S invalid, not read).
- **Q-4.** The pooled Spearman values among answered rows (s with M_city −0.56; K_v with M_city −0.54) are a
  composition effect across tasks: RC has the highest margins and s ≈ 0, K_v ≈ 0. Within task (added after the read,
  descriptive):

| Task | n | ρ(s, M_city) | ρ(K_v, M_city) | mean s | mean K_v |
|---|---|---|---|---|---|
| DE | 94 | +0.02 | −0.18 | 0.71 | 2.11 |
| RC | 223 | −0.11 | +0.22 | 0.004 | 0.05 |
| TU | 258 | −0.15 | +0.19 | 0.24 | 2.82 |

  K_v ≈ 0.05 on RC suggests the verbal-rating prompt fails there, as S does.
- **Q-5** (exploratory): Spearman of M_city at the note's producing position with M_city at P_ans = 0.11 (n 826).
  Descriptive only; no caching claim.
- **UNA.** Wrong-city answers on 14 of 1,008 held UNA rows.
- **F1 diagnostic.** M_city at the first-city position: gold +22.2 (n 34), wrong −20.7 (11), competitor −22.0 (9);
  18 degenerate rows have no city. This is output consistency, as expected.

**Spend.** One L40S pod, **$1.319 posted** (GPU $1.310 + disk $0.009; lifetime about 72 min at $1.09/h), inside the
§50 cap of $3.00. The laptop build, tests, D-30 and timing: $0. All experiments ≈ **$24.2 of $30** (STATE spend line).

**Limits.**
- One model, synthetic city templates with a finite pool, F2 only.
- S and K_v are elicited in separate prompts and failed here. M_city is an output-side margin, not validated as an
  internal construct, and in DE it is partly read at a name token.
- The answer-bearing rule is lexical plus an audit, and was fixed once on dev.
- Q-2 uses selected flip pairs.
- One held read; no rescue (D-45). A better sufficiency measure, or the margin read at the city-producing position,
  would be a new proposal under a separate decision.

## 51. Can an agent's weight edit repair note-then-unknown selectively, and does its account survive sealed cases? Agent-proposed weight edits, Qwen 2.5 3B (2026-11-02, pre-registered rev 1a; editor run 2026-11-03; pods 2026-11-06/07)

**Status: COMPLETE (H review ACCEPT). Branch R: Repair SHOWN on sealed H. Better trade-off than B2, B4 and B5;
against B1 (logit bias) inconclusive; B3 not established. Branch M: pathway located and a causal path through the
L33 output at P_ans (sufficient, not necessary). The editor's account is NOT supported. Closure decision (D-74)
pending the user's "Go".**
- **The repair holds on sealed cases.** Opus 5.5, acting as the weight editor, made a rank-1 edit to L35 `o_proj`
  (submission 2, dose 2). On H it gave the right city on **212 of 261 false abstentions (0.816; B0 0)**:
  - U(λ=1) − B0 was **+0.117 [0.096, 0.138]**, p 1e-4;
  - unsupported answers went from 20/576 (0.035) to 24/576 (0.042);
  - CF-follow was 0.917 (bar ≥ 0.90), and Deg 0.008.
  - All four frozen repair bars are met.
- **The edit has a visible cost.** It breaks **11 of 593 already-correct answers (Dam 0.030)**, within the frozen
  bar of +0.05. B1 and B4 break none.
- **Better than three of five comparators, not all** (Holm over 10 hypotheses, FWER 0.05):
  - better than **B2** (prompt instruction; O2, p_holm 0.005), **B5** (directional ablation; O1, 0.007) and **B4**
    (§38 steering; O2, 0.033).
  - Against **B1** (logit bias against " unknown") the result is **inconclusive**: neither better (O2 raw p 0.015,
    Holm 0.092) nor equivalent (TOST p 0.21 for Rel, 0.70 for Uns). The point estimates favour the edit on
    unsupported answers (0.042 vs 0.095) at similar release (0.816 vs 0.793). That difference does not survive Holm.
  - **B3** is not established: it releases more but adds about 64 points of unsupported answers.
- **Branch M located a sufficient pathway, not the mechanism the editor claimed.** Submission 1 (two heads, L32h2
  and L33h10, dose 1.5) was tested; its primary component is L33 `o_proj`.
  - M1: reverting L33 removes 0.293 [0.225, 0.368] of the release, and the edit releases 0.717 more than the mean of
    five random B6 edits (M1b). Pathway located.
  - M2: transplanting the edit's change to L33's output at P_ans into the unedited model releases 0.808. Clamping it
    away in the edited model removes only 0.193. A causal path; **sufficient, not necessary**. The L32 head appears
    to carry the rest (observed redundancy; S reverts 0.264 vs 0.078), which is not proven.
  - **M3, the editor's account: not supported.** Its variable f adds nothing to predicting false abstention: ΔAUROC
    −4.5e-5 (p 0.82), against a baseline already at AUROC 0.998, a ceiling. The interaction was 0.0005
    [−0.025, 0.027] (p 0.49). f behaves like a generic "stop saying unknown" push, not an answerability signal.
  - **The M candidate is non-selective at its dose** (S, total effect: Uns 0.96). Branch M says where a release can be
    produced. It says nothing about selectivity or about the R candidate's L35 edit, and the two are not merged.
- **What the editor claimed is labelled as such.** Each submission's computation, evidence and predicted flips are
  **editor-claimed**. Branch R tests the edit, not its verbal story; branch M tests one submission's account and it
  failed.

Headline (Advisor wording):

> On sealed held-out cases, an agent-proposed rank-1 weight edit repaired false abstention: it gave the right city in
> 82% of the cases where the model had wrongly said "unknown", with only a small rise in unsupported answers (3.5% to
> 4.2%) and 3% damage to already-correct answers. It beat a prompt instruction, activation steering and directional
> ablation on the pre-registered trade-off tests, but not a simple output bias against "unknown", where the comparison
> was inconclusive. A separate two-head edit located a pathway at layer 33 that is sufficient to release answers. The
> editor's own account of what that component computes was not supported.

Pre-registration `research/proposals/2026-11-02-exp51-agent-weight-editing.md`:
- rev 1 at d9e67b7;
- rev 1a at 1be9281 and 15ec3ee (the §13 build contract from Codex's rev 1 review; zero-SD equivalence rule).

Decisions:
- D-72 (e1b292f): §51 opened; the editor budget is a $50 Console limit with a harness stop at $45.
- D-73: RunPod cap raised to $36; two-pod stage-4 plan.
- D-74 (closure): proposed by the Advisor; awaits the user's "Go".

Reviews:
- `research/reviews/2026-11-02-exp51-stage1-build-review.md` (build, delta, B5, stage-4 and F0 reviews);
- `research/reviews/2026-11-06-exp51-s-review.md` (ACCEPT; independent S recount);
- `research/reviews/2026-11-07-exp51-h-review.md` (ACCEPT; independent H and T-city recount; Holm reproduced).

Freezes:
- F0 at d29263a;
- F1 at 1a37a72 (bounds, D2/D3, resume and lost-call rules, launch procedure, meter validation, cost estimate);
- **F2 at 02bb2f6**: `runs/exp51/gpu/design.json`, a byte copy of pod S's `f2_candidate.json` (sha256 aa8e447f…,
  design_sha d02d9f9c). It was committed after the S review and before any H or T-city row was read. The H+T pod
  verified it against its own shallow clone.

Code:
- `memctl/analysis/edit51.py`;
- `scripts/exp51_cases.py`, `exp51_common.py`, `exp51_engine.py`, `exp51_run.py`, `exp51_mech.py`,
  `exp51_harness.py`, `exp51_freeze.py`, `exp51_stage4.py`, `exp51_smoke_s.py`;
- the pod chains `exp51_pod_s.sh` and `exp51_pod_ht.sh`, with bundles `exp51_s_bundle.sh` and `exp51_ht_bundle.sh`;
- `tests/test_exp51.py` (53 tests).

Results:
- E: `runs/exp51/run/e_entry.json`.
- S: `runs/exp51/gpu/s_select.json` and `f2_candidate.json` (ba24885).
- H and T-city: `runs/exp51/gpu_podHT/h_report.json` (sha 51bbbd2e…, 321a4f1).
- Arm records stay local.

**Design (fixed at F0–F2).**
- **Cases.** `runs/exp51/cases51.jsonl` (sha 5eeb7c40…; 5,928 rows, 456 bases). Sizes in bases per task: E 48, S 32,
  H 48, T-city 24 (r = 1.0, no scale-down).
  - Populations: ANS (graded FA / C), CF, UNA-rel, UNA-swap, NM.
  - T-format was dropped before F0 at the user's "yes drop T-format" (stage-1 build review, ruling 2). It is not a
    deviation.
- **The editor.** Opus 5.5 via the API, with tool access to the unedited model on E only: teacher-forced reads,
  generation, `save_edit`, Python.
  - It submits up to three edits, each with a component and computation claim, predictions by row type, flips, a
    signed endpoint, a most important component and an f definition with activity predictions.
  - One run only; a resume after an infrastructure crash counts as the same run.
- **Entry on E (laptop CPU, fp32, batch 1).**
  - R: U1 − B0 lower bound > 0, with ≥ 20 FA bases in DE or RC.
  - M: the declared signed endpoint has |effect| ≥ 0.05, a bound beyond 0, and ≥ 20 bases.
- **Selection on S (pod, A40, batch 64, sorted by case id, strict fp32).**
  - R dose: max U1 over 0.5–2 (ties go lower). Candidate: the highest U1 among entrants, ties by (n_params, index).
  - M: oriented effect / max(SE, 0.01). The primary component is the one whose revert removes the most effect.
  - Comparator doses: max U1. B1 is matched to the f-intervention.
- **Confirmation on H, read once; T-city is descriptive.**
  - R repair bars: U − B0 lower bound > 0, Dam − B0 ≤ +0.05, Deg ≤ 0.02, CF-follow ≥ 0.90.
  - "Better trade-off than k": O1 (more release, Uns not worse) or O2 (fewer Uns, release not worse), Holm over 10
    hypotheses.
  - M: M1 (revert), M2 (transplant / clamp), M3 (activity, predictor, interaction); Holm over 3.
  - Intervals are 10k base resamples.

**The editor run** (2026-11-03, 14:06–23:49 PST):
- 73 turns, 16 passes, 63 Python runs and 6 saved edits; 3 submissions. The ledger records 2 failed requests. The
  run ended on "finish".
- **Cost $3.58** of the $45 harness stop (1.59 M input tokens, 12.5 k generated).
- **Time:** about **34,950 s** from start to the last submission. The ledger's summed pass time (58,978 s) is larger
  because passes overlapped the harness's polling. The ledger figure is not a wall-clock time.

The three submissions (editor-claimed computations):

| # | Kind | Component (editor-claimed) | Endpoint |
|---|---|---|---|
| 1 | repair | Heads L32h2 and L33h10 write a near-constant "unknown" vector at the answer ":"; the edit removes it (two `o_proj` column blocks) | FA → gold |
| 2 | repair | L35 attention (heads 12, 13) carries an "answer available in the note" gate; a rank-1 `o_proj` edit makes it commit | FA → gold |
| 3 | diagnostic | The converse of 1: doubling the two heads' write | C → abstention |

**E (laptop, §8.4.2 replay at dose 1, base-weighted):**

| Arm | Rel | Uns | Dam | CF-follow | U1 | R entry (U1 − B0) | M effect |
|---|---|---|---|---|---|---|---|
| B0 | 0 | 0.040 | 0 | 0.653 | 0.305 | — | — |
| Sub 1 | 0.862 | 0.076 | 0.028 | 0.940 | 0.427 | +0.122 [0.102, 0.141] enters | FA gold +0.817 enters |
| Sub 2 | 0.739 | 0.050 | 0.015 | 0.887 | 0.427 | +0.122 [0.104, 0.139] enters | FA gold +0.706 enters |
| Sub 3 | 0.009 | 0.030 | 0.381 | 0.451 | 0.198 | −0.107 does not enter | C abstention +0.360 enters |

- FA bases on E: DE 44, RC 35, TU 15.
- Sub 1 raised UNA-swap invented answers from 17 to 29.

**S (pod S, selection).** FA bases: DE 29, RC 23, TU 11. B0 U1 was 0.303.
- R:
  - Sub 2 at dose 2: U1 0.429, U − B0 +0.126 [0.097, 0.155]. **This is the candidate.** Its dose is at the top of the
    grid, U1 was still rising, and no larger dose was searched.
  - Sub 1 at dose 1: 0.402.
  - Sub 3 did not enter (−0.059).
- Comparators at max-U1 doses: B1 −8 0.411, B2 0.389, B3 0.136, B4 α 1 0.415, B5 β 1 0.383.
- For the candidate on S: CF-follow 0.910 (borderline against the H bar), Rel 0.816, Uns 0.062, Dam 0.015.
- M:
  - Sub 1 at dose 1.5: FA gold +0.903 (SE 0.033). **This is the candidate.**
  - Primary component L33 `o_proj`: its revert removes 0.264; L32's removes 0.078.
  - B1 matched value −1 (effect 0.278) against the f-intervention's 0.220.
  - The f-intervention releases at most 44/185 FA rows (0.22).

**H (sealed, read once), R branch:**

| Arm | Rel | Uns | Dam | CF-follow | U1 |
|---|---|---|---|---|---|
| B0 | 0/261 (0) | 20/576 (0.035) | 0 | 0.632 | 0.317 |
| **Candidate (sub 2, dose 2)** | 212/261 (0.816) | 24/576 (0.042) | 11/593 (0.030) | 0.917 | **0.435** |
| B1 logit bias −8 | 0.793 | 0.095 | 0 | 0.919 | 0.406 |
| B2 "do not reply unknown" | 0.721 | 0.083 | 0.046 | 0.882 | 0.389 |
| B3 no "unknown" option | 0.977 | 0.682 | 0.021 | 0.975 | 0.130 |
| B4 §38 steering α 1 | 0.786 | 0.078 | 0 | 0.921 | 0.418 |
| B5 directional ablation β 1 | 0.678 | 0.031 | 0.099 | 0.838 | 0.401 |

| Comparator | dRel (candidate − k) | dUns | Holm result |
|---|---|---|---|
| B1 | +0.023 [−0.043, +0.088] | −0.054 [−0.069, −0.038] | not established (O2 p_holm 0.092); not equivalent |
| B2 | +0.096 [+0.013, +0.175] | −0.042 [−0.061, −0.023] | **better** (O2, 0.005) |
| B3 | −0.160 [−0.222, −0.103] | −0.641 [−0.681, −0.599] | not established |
| B4 | +0.031 [−0.029, +0.092] | −0.036 [−0.052, −0.021] | **better** (O2, 0.033) |
| B5 | +0.139 [+0.049, +0.225] | +0.010 [−0.007, +0.028] | **better** (O1, 0.007) |

**H, M branch** (sub 1, dose 1.5; primary component L33 `o_proj`):

| Test | Estimate | Holm (3) | Reading |
|---|---|---|---|
| M1 pathway | revert L33 0.293 [0.225, 0.368]; E* − mean of 5 random edits (B6) 0.717 [0.664, 0.770] | p 1e-4, reject | pathway located |
| M2 causal path | transplant 0.808 [0.744, 0.870]; clamp 0.193 [0.138, 0.253] | p 1e-4, reject | sufficient, not necessary |
| M3 account | predictor ΔAUROC −4.5e-5 [−1.8e-4, 5.2e-5]; interaction 0.0005 [−0.025, 0.027] | p 0.82, not rejected | not supported |

M3 activity predictions (per base, S margins):
- evidence_removed: directional, −1.08 [−1.54, −0.62]. As predicted.
- city_changed: −0.16 [−0.67, 0.33]; inside the margin 6.6.
- distractors_added: +1.03 [0.52, 1.55]; inside the margin 5.2.
- unrelated_vs_absent: −0.04 [−0.45, 0.39]; inside the margin 5.7.
- The account fails on the predictor and interaction tests (intersection-union).

**T-city (descriptive; new cities, not a second confirmation):**
- Candidate: U1 0.422 vs B0 0.303; Rel 0.817; Uns 0.062 (B0 0.059); CF-follow 0.912; Dam 0.031.
- The ordering against B4 and B1 is the same as on H.
- **The S-to-H consistency is descriptive, not extra evidence:** U1 0.429 → 0.435, CF-follow 0.910 → 0.917.

**Deviations (tooling only; no scoring, selection or gate rule changed; each accepted by the Advisor):**
- **D2, D3**: recorded at F1.
- **D4** (b6e3d8e; found by D-30 before any pod):
  - both bundles now ship the F0-frozen §38 B4 file (`runs/exp38_dev/qwen3b/directions.npz`, sha be1e3a09);
  - pod S's D-22 smoke runs on dummy-shaped stand-ins (`exp51_smoke_s.py`), because the real 2048-wide edits cannot
    apply to the 64-wide dummy.
- **D5** (4bf49ad): pod S attempt 1 (gn952337raypzx) crashed in the §10 replay check (KeyError `prompt_ids`) and was
  deleted at +9.5 min, about **$0.09**, counted in S.
  - The fix takes prompt ids from the laptop B0 records, which are identical to `chat_ids(prompt)` on all 200 rows,
    and adds a regression test.
  - Attempt 2 passed the gate at 200/200 in all four comparisons.
- **D6** (c880239): pod H+T's D-22 smoke had the same dummy-versus-real fault, and gets the same fix as D4.
- **D6b** (bf22ac3; found by a negative D-30 check with a tampered `design.json`): under errexit and pipefail, a
  failed F2 guard ended the chain with exit 1 at the `guard | tee` pipe, unlogged. It now logs "F2 GUARD FAILED" and
  exits 4. The chain was fail-closed both before and after the fix.

**Spend.**
- Editor API: **$3.58** (Console limit $50, harness stop $45); dry runs about $0.71.
- RunPod:
  - pod S **≈ $4.38** (attempt 1 ≈ $0.09; attempt 2 billed 7.263 h × $0.59 = $4.29, 7.15 h of chain);
  - pod H+T **$2.57** (4.361 h × $0.59; 4.31 h of chain).
  - E, the replay, the builds, tests and D-30 ran on the laptop at $0.
- RunPod programme total ≈ **$31.16 of $36** (D-73).

**Limits.**
- One model, one editor run, synthetic city templates, and the F2 note format only.
- The R candidate's dose is at the top of its grid.
- CF-follow was borderline on S (0.910) and passed on H (0.917).
- The rank-1 form and the L35 site are the editor's choices. Branch R does not test the editor's "evidence gate"
  story for that edit.
- M2 shows sufficiency, not necessity. The L32 redundancy is observed, not tested.
- M3's predictor test sat at an AUROC ceiling (0.998), so it had almost no room to show an increment.
- The novelty claim is limited to selectivity under sealed, pre-registered confirmation: agent-proposed weight
  edits themselves are not new (literature pass 2026-11-05).
- One sealed read; no rescue (D-45).

## 52. Do readers below 3B say "unknown" with the evidence in view? A feasibility screen of six instruction-tuned readers (2026-11-04, proposed rev 0, D-74; laptop stage 0 2026-11-08)

**Status: stage 0 RUN, gate FAILS for all six readers. Nothing is queued for Q-A to Q-D. Laptop, CPU fp32, $0.**
- **What was planned.** The §43 stage-0 screen, repeated below 3B. The question was whether the in-view "unknown" habit
  (Qwen 2.5 3B 0.300, §43) follows the family or the size.
  - Readers: Qwen2.5-0.5B-Instruct, Qwen2.5-1.5B-Instruct, Llama-3.2-1B-Instruct, gemma-3-1b-it, gemma-3-270m-it
    (the floor) and SmolLM2-1.7B-Instruct, at pinned HF revisions.
  - A reader is queued only if it passes all three bars: marker rate ≥ 0.70, reading floor ≥ 0.15, and in-view
    abstention ≥ 0.15 on at least 10 rows.
- **Setting (frozen at 0c88a93, before any model output).**
  - Rows: the 126 §43 dev rows (folds 0–1; answerable, all evidence in view, at least one evidence line). The qids
    and prompt bytes are identical to the Llama, Gemma and Phi §24 records.
  - Also run, for description only: 74 further fold-0–1 rows (62 answerable but not all in view, 12 unanswerable).
  - Prompt: the frozen reader prompt (7b5fc30), `fixed8`, with the §24 contexts. Prompts sha 7d3b5609. No tuning.
  - Each model's own chat template, one user turn. Qwen and SmolLM2 insert their own default system text; the Llama
    date is pinned to 09 Oct 2026.
  - Decoding: greedy, 256 new tokens, fp32 on CPU with 8 threads, transformers 5.17.0.
  - Family configs: `memctl/analysis/families.py` `SMALL52`, kept out of `FAMILIES`, so §43 paths are byte-identical
    and its tests pass.
- **Measures.**
  - The answer is the §23 graded answer: the phrase after the last `Answer:`, or the whole output when there is no
    marker.
  - Marker rate is over the 126 rows.
  - In-view abstention is the share of marked rows whose answer starts with "unknown".
  - Reading floor is `answer_selection.d1(answer, gold)` over the 126 rows.
- **Timing** (one median row of about 1,060 tokens per reader, measured before any duration was quoted).
  - Prefill takes 0.7 s (270M) to 8.1 s (SmolLM2). Decoding takes 0.07 to 0.48 s per token.
  - Measured mean seconds per row over the 126 rows: 2.3 (270M), 4.8 (Qwen 0.5B), 61.9 (Llama 1B), 7.3 (Gemma 1B),
    10.7 (Qwen 1.5B) and 21.6 (SmolLM2).
  - Wall time 09:26–15:34. Peak RSS ≤ 10.1 GB.
- **Result (126 in-view rows per reader).**

  | Reader | Marker | In-view abstention (marked) | Floor | Floor, marked only | §24 phrases | Gate |
  |---|---|---|---|---|---|---|
  | gemma-3-270m | 0.175 (22) | 0.045 (1/22) | 0.103 (13) | 0.048 (6) | 0 | FAIL: marker, floor, abstention |
  | Qwen 2.5 0.5B | 0.048 (6) | 0.167 (1/6) | 0.238 (30) | 0.000 (0) | 0 | FAIL: marker, abstention |
  | Llama 3.2 1B | 0.318 (40) | 0.050 (2/40) | 0.238 (30) | 0.032 (4) | 3 | FAIL: marker, abstention |
  | gemma-3-1b | 0.103 (13) | 0.000 (0/13) | 0.175 (22) | 0.040 (5) | 0 | FAIL: marker, abstention |
  | Qwen 2.5 1.5B | 0.016 (2) | 0.000 (0/2) | 0.016 (2) | 0.008 (1) | 0 | FAIL: marker, floor, abstention |
  | SmolLM2 1.7B | 0.802 (101) | **0.000 (0/101)** | 0.349 (44) | 0.278 (35) | 1 | FAIL: abstention |

  Note-then-refuse, the §49 signature (the gold already in the note), is 0 of the 4 in-view abstentions in all
  readers combined.
- **Reading, per the frozen gate.**
  - **SmolLM2 1.7B: the habit is absent.** It is the only reader that passes the marker and floor bars, and it never
    answers "unknown" with the evidence in view (0 of 101 marked rows). It never answers "unknown" on a marked row:
    0 of 48 marked rows answerable but not all in view, and 0 of 11 unanswerable rows. Because it does not abstain
    even when no answer is available, the absence reflects no abstention at all, not abstention that responds to
    the evidence.
  - **The other five do not follow the format, so in-view abstention is not measurable here.**
    - Qwen 0.5B and 1.5B, Gemma 1B and gemma-270m mostly answer at once without a note (1–5 new tokens).
    - Llama 1B writes a long note but seldom reaches the marker. It hit the 256-token limit on 70 of 126 rows.
  - Read with the confound stated in the proposal: one prompt, tuned on Qwen 7B, serves every reader.
- **Floor caveat.** On an unmarked row the graded answer is the whole output, so a long note that contains the gold
  counts toward the floor. Llama 1B's 0.238 falls to 0.032 when only marked rows count.
- **Post hoc, descriptive only (no gate, no claim).**

  | Reader | Bare-or-marked "unknown", 126 in-view rows | The same on 62 answerable rows not all in view | Marked "unknown", not all in view |
  |---|---|---|---|
  | gemma-3-270m | 0.024 (3) | 0.081 (5) | 3/8 |
  | Qwen 2.5 0.5B | 0.135 (17) | 0.177 (11) | 1/6 |
  | Llama 3.2 1B | 0.016 (2) | 0.032 (2) | 2/20 |
  | gemma-3-1b | 0.206 (26) | 0.355 (22) | 1/4 |
  | Qwen 2.5 1.5B | **0.960 (121)** | **0.935 (58)** | 0/3 |
  | SmolLM2 1.7B | 0.008 (1) | 0.032 (2) | 0/48 |

  - **Qwen 1.5B** outputs a bare "unknown", with no note and no marker, on 121 of 126 in-view rows and on 58 of 62
    rows where the evidence is not all in view. Its refusal does not respond to the evidence, and its floor is 2 of
    126. So it looks like a blanket refusal, not the 3B's evidence-use failure.
  - **Gemma 1B** says "unknown" more often when the evidence is not all in view (0.355) than when it is (0.206).
  - **Pipeline check** (requested at review; `scripts/exp52_pipecheck.py`): on all 126 rows, Qwen 0.5B and 1.5B got
    the same chat string and token ids, and they equal the Qwen 2.5 3B §23 prompt and ids for the same qids (126 of
    126). The vocabularies and chat templates are identical across the three. The 1.5B's pattern is the model's, not
    a configuration error.
- **What this leaves.**
  - Below 3B, with this prompt, the in-view "unknown" habit is not measurable in five readers and absent in the one
    that follows the format.
  - No reader goes to Q-A to Q-D. No prompt-variation arm was run. That would be a new proposal and the user's
    decision.

Results: `runs/exp52/freeze.json` (sha256 92268a0a…), `rows52.jsonl`, `probe.json` at 0c88a93. The rest, at
0f25a1c: `gen_{reader}.jsonl` (200 rows each, the cache for any later stage), `timing_*.json`, `report.json`,
`pipecheck.json` and the chain logs. Scripts: `scripts/exp52_probe.py`, `exp52_freeze.py`, `exp52_run.py`,
`exp52_report.py` and `exp52_pipecheck.py`.

## 53. Are the two §51 edits specific, and do they carry over to real questions? Random-edit controls, activity predictions and a LongMemEval transfer, Qwen 2.5 3B (2026-11-05, pre-registered rev 1b; pod 2026-11-07)

**Status: COMPLETE (results review ACCEPT). Each of the four pre-registered claims passed its own test (C1, C2, C4,
C5). No family-wise claim is made. On N a plain logit bias matches both edits on the utility, so the edits are
selective relative to random edits of the same kind, not better than the simplest baseline.**
- **What was tested.** §51 left two edits: submission 1 (zero the `o_proj` columns of L32h2 and L33h10,
  `c64ea090781450cf`) and submission 2 (rank-1 L35 `o_proj`, `8364bf135ac59a83`, dose 2). §53 asks whether each edit
  beats random edits of the same form. It also asks whether the editor's activity predictions for submission 2 hold
  on fresh bases, and whether either edit carries over to LongMemEval. N is a new controlled split (seed 54, 1,950
  rows), disjoint from E and S keys; it shares 867 of 1,350 keyed rows with H and none with T-city. N-dev (20% of
  bases) was used only for calibration and checks. N-held (80%) was read once. The sample was 150 FA, 150 C and
  150 UNA rows (seed 5300, drawn from the N-held B0 labels before any edited run; 228 FA were available).
- **C1 SHOWN: the two-head ablation's release is specific.** A-target released **0.751** of FA rows. The 20 random
  L32×L33 head pairs released at most 0.464 (median 0.163). Rank 1 of 21 (p = 1/21), and target − median = 0.589
  ≥ 0.30. z = 5.2 (descriptive).
- **C3 (descriptive): so is its cost.** A-target's unsupported-answer rate on UNA is **0.075**, the highest of all 21
  (random-pair max 0.071; B0 0.031).
- **C2 SHOWN, narrowly: the amplified pair creates abstention specifically.** The §51 diagnostic (the two heads × 2)
  turned **0.381** of correct answers into abstentions. The best of the 20 amplified random pairs reached 0.345
  (median 0.094). Rank 1 of 21; z = 2.9 (descriptive). Of the four claims, this one passed by the smallest gap.
- **C4 SHOWN: the L35 edit's read direction matters, relative to random reads.** D-target's U(λ=1) was **0.403**.
  The 20 D-randread edits used the same write vector and the same RMS gate on N-dev FA rows, with a random read
  vector, sign-aligned (rev 1b). Their best U1 was 0.300 (median −0.069). Rank 1 of 21. D-target's unsupported-answer
  rate (0.058) is below the D-randread median (0.228). z = 1.4 (descriptive).
  - The random reads varied widely: Rel 0.00–0.94, Uns 0.00–0.88, Dam 0.00–1.00 (median 0.31). They failed in two
    ways:
    - some released FA rows but also answered UNA rows (draw 07: Rel 0.94, Uns 0.88);
    - others broke generation outright (draws 12 and 19: degenerate on 0.91 and 1.00 of rows, Dam 0.99 and 1.00),
      even though each was sign-aligned on N-dev FA rows.
    - The best random draw was draw 13 (U1 0.300: Rel 0.79, Uns 0.12, Dam 0.08). D-target had Rel 0.691, Uns 0.058 and
      Dam 0.023.
  - **Beside C4, the simplest baseline: B1-ref.** §51's frozen logit bias against " unknown" (−8, never tuned on N)
    gave **U1 0.406**, against D-target 0.403 and A-target 0.414 (Rel 0.698, Uns 0.075, Dam 0). On N neither edit
    improves on a plain output bias. This is descriptive and consistent with §51's "B1 inconclusive". C4 says the gate
    `w` carries the selectivity relative to random read vectors. It does not say the edit beats an output bias.
- **C5 SHOWN: the editor's four activity predictions for submission 2 hold on fresh bases** (IUT p ≈ 1e-4). f is the
  L35 `o_proj` output at the answer ':' projected on `w/|w|`, in the **unedited** model. It is per base, on N-held
  (119–120 bases per manipulation; 7 B0 rows with no answer line have no read site). Mean Δf, task-stratified
  bootstrap:
  - evidence removed: **−3.77** [−3.89, −3.65] (predicted decrease; one-sided sign test);
  - distractors added: **−1.12** [−1.38, −0.86] (predicted decrease);
  - unrelated vs absent: **+0.22** [0.13, 0.32] (predicted increase);
  - city changed: **−0.03**, 90% CI [−0.13, 0.08]. This is equivalent within the frozen tolerance of ±1.123 (2 SD on
    N-dev), not invariance.
  - The directional predictions are sign-only tests. The 2-SD margin applies to the equivalence prediction alone.
  - C5 tests the editor's description of the L35 signal on the unedited model, not the edit's effect, and only its
    activity predictions: no predictor or interaction test was run for submission 2. It is the first test of
    submission 2's account; it does not establish that account, and it says nothing about §51's M3, which tested
    submission 1's account.
- **Descriptive arms.**
  - A-single: L32h2 alone released 0.486 and L33h10 alone 0.606 (the pair together 0.751).
  - A-late: 10 random pairs from L30–L35 released at most 0.306 (median 0.171).
  - D-randwrite: 10 edits with the gate `w` and a random write vector released at most 0.307 (median 0.227). Their
    U1 was at most 0.148.
  - B-DLA, direct logit attribution onto " unknown" − gold first token at P_ans, unedited, mean over FA rows: L32h2
    +7.58, L33h10 +10.52, L35h12 +1.40, L35h13 +0.52. Submission 2's L35 write at dose 2 gave −27.6 on FA, −34.8 on C
    and −0.52 on UNA. This is attribution only, not mechanism.
  - **B-opt: not estimable.** No FA row abstained under the option-removed prompt (B0 gold 145 of 150), so the
    edited arms had n = 0.
- **Arm C, real questions (LongMemEval dev folds 0–1, 138 rows; descriptive, small).**
  - B0 (fresh, on the pod) refused 28 of the 126 population rows, 26 of them with the answer in view. 35 contained
    the gold.
  - A-target released 20 of the 26 in-view refusals, but only **8 contain the gold**. Gold rose from 35 to 45; 2 of
    35 correct rows were damaged.
  - D-target released 19 (**7 with the gold**); gold 45; 3 damaged.
  - A release is not a correct answer: most releases did not contain the gold.
  - On the 12 abstention questions, both edits answered fewer than B0 (4 against 6).
  - Reproduction against the §23 cache is imperfect (refusal flag 121/138, first word 97/138, marked 119/138), so the
    baseline is the fresh B0, not §23's numbers.
- **Multiplicity.** C1, C2, C4 and C5 were judged each at its own pre-registered bar (rank 1 of 21 plus its
  co-requirement; IUT at 0.05 for C5), with no family-wise claim. If each target were exchangeable with its random
  draws, the chance that at least one claim passes is at most about 0.19. A rank test with 20 draws cannot give a p
  below 1/21.

Headline (Advisor-reviewed wording):

> Both §51 weight edits beat random edits of the same form. Zeroing the two chosen heads released 75% of
> the model's false "unknown" answers, against at most 46% for 20 random head pairs. The L35 edit's read direction
> scored better on the pre-registered trade-off than all 20 random read directions of the same size, with fewer
> unsupported answers than the median random read (0.058 against 0.228). The editor's four predictions about how its signal responds to the prompt held on fresh cases.
> But a plain logit bias against "unknown" matched both edits on the same trade-off. On real LongMemEval questions,
> most of the answers the edits released were not correct.

Pre-registration `research/proposals/2026-11-05-exp53-head-controls.md`:
- rev 0 (D-75); rev 1 at b70e7c5 (D-76; both edits);
- rev 1a at 3a01186 (per-claim rank bars replace Holm, which no 20-draw rank test could pass; RMS calibration; N key
  rule; frozen sample-drawing rule; arm C B0 fresh);
- rev 1b at e3b3018 / ea88023 (sign-align D-randread read vectors; proposal sha 97edca99).

Freeze: manifest `runs/exp53/manifest.json`, sha256 b2b14bac…9f93, 86 arms, committed at a177536 by the Advisor before
any N-held output. The pod verified it against its own shallow clone (guard; a tampered copy exits 4 in D-30).

Deviations and build notes (all before any N-held output):
- **D-randread sign (rev 1b).** In the first manifest (ca4e40d8), 13 of the 20 random read vectors had a negative
  median gate on N-dev FA rows (draws 0, 1, 2, 3, 5, 6, 8, 10, 12, 13, 17, 18, 19), so they pushed toward "unknown".
  Found at the Advisor's manifest review. These were negated; RMS scales are unchanged (`--stage signfix`). The
  post-flip medians are 0.078–1.063 (target 0.961).
- **Laptop smoke cut** to 6 labelled N-dev rows (2 FA, 2 C, 2 UNA) and 2 LongMemEval rows, with B0 records copied from
  the N-dev B0 (`--case-ids`, `--lme-limit`). The full 39-row smoke would have taken about 1.6 h on CPU. The pod's own
  GPU smoke ran 39 N-dev rows over all smoke arms before the held run.
- **Schedule.** The laptop N-dev B0 swapped (RSS 15–19 GB of 23 GB) and took 2 h 25 min. The Executor then lost
  1.5 h before starting calibration, after its waiter was stopped under memory pressure. Neither affects any result.

Spend: pod p27yys25r4uos2, A40 secure at $0.59/h, about 15,779 s billed ≈ **$2.59** (quote $2.55, hard stop $3.10). The
chain exited 0 at +15,645 s; no arm was cut. RunPod programme total ≈ **$33.75 of $36** (D-73).

Review: `research/reviews/2026-11-07-exp53-results-review.md` (ACCEPT; independent recount of all 87 arms, claims and
C5 estimates; margin use confirmed).

Code: `scripts/exp53_cases.py`, `exp53_lme_rows.py`, `exp53_run.py`, `exp53_pod.sh`, `exp53_bundle.sh`;
`tests/test_exp53.py` (7 tests). Build at 9ff3d61; sign rule at 1dce499.

Results:
- `runs/exp53/gpu_pod/runs/exp53/pod/report.json` (sha256 f1ba6dad…), plus timing, sample, logs and pod.done, at
  7b9e271. The 26,400 arm records and 1,560 B0 records stay local (`runs/exp53/gpu_pod/pod_out.tar.gz`).
- Calibration: `runs/exp53/calibration.json`. Laptop smoke: `runs/exp53/smoke_laptop/report.json`.

## 54. Is there a native switch behind false "unknown"? Twin-pair patching, Qwen 2.5 3B (2026-11-07, pre-registered rev 0a; laptop stage 1 2026-11-08)

**Status: STOPPED at the stage-1 feasibility gate (§6 / D-45). 21 primary pairs against a bar of 40. No component was
patched.**
- **What was planned.** §54 compares twin questions whose memories differ only in their timestamps, where the model
  answers one twin and says "unknown" on the other.
  - Primary pairs give both twins the answering twin's own note, token for token. Copying component outputs at the
    answer ':' between the twins was meant to find a set of at most 5 heads or MLPs (layers 20–35) that passes five
    pre-set tests.
  - Before any patching, stage 1 had to yield at least 40 primary pairs, from at least 20 bases, covering at least 2
    of the 3 tasks.
- **Data.** A new split W from the §51 F2 generator (via `exp53_cases.build`), seed 55: 150 bases, 1,950 rows.
  - E and S keys are excluded, and no prompt repeats any §49–§53 row.
  - W shares DE/TU keys with N (930 of 1,350 keyed rows) and with H (882), because those cells are allocated
    deterministically. No output of H, T-city, N-held or §50 held was read.
  - Seed 56 (stage 5) was not generated.
- **B0** on the 900 ANS rows, own notes, §51 procedure: 18.1 s per row on the first 64 rows, 5.2 h in all, peak RSS
  19.9 GB.
- **Gate (FAIL):**

  | Requirement | Bar | Result | |
  |---|---|---|---|
  | Primary pairs | ≥ 40 | **21** | fail |
  | Bases with a primary pair | ≥ 20 | 20 | pass |
  | Tasks covered | ≥ 2 | 3 | pass |

- **Counts.**
  - Of 450 twin pairs (rep a vs rep b, same base and hardness): 268 were both gold, 64 were gold/"unknown" flips,
    and 118 were other combinations.
  - The 64 flips contain 64 of the 288 false-abstention rows (**22%**).
  - Flips by task: DE 23, TU 21, RC 20. By hardness: H2 32, H1 20, H0 12.
  - Under the answering twin's note, **43 of the 64 flips (67%) are note-driven**: the abstaining twin then answers
    with the gold. **21 remain "unknown" (primary)**, and every one of the 21 satisfies D_A < 0 < D_F (median
    D_F − D_A = 2.2).
  - Primary by task: DE 8, RC 8, TU 5. By hardness: H0 1, H1 9, H2 11. By half: dev-A 10 pairs from 10 bases, dev-B
    11 pairs from 10 bases.
  - Note-driven by task: TU 16, DE 15, RC 12. By hardness: H2 21, H0 11, H1 11.
  - Median D (" unknown" minus A's first answer token) on the abstaining twin's own note: 4.9 for note-driven pairs,
    1.0 for primary pairs.
  - Stable pairs (both gold; the b twin still gold under rep a's note): 257 of 268 candidates, dev-A 126, dev-B 131.
    The other 11 turn "unknown" under rep a's note.
- **Reading (scoped).**
  - Among twin flips, 43 of 64 disappear when both twins share one note. Near-threshold "unknown" is mostly set by
    the note the model wrote, not by a late decision at the answer step.
  - This covers only the 22% of false-abstention rows that form twin flips. It is not evidence that no native switch
    exists.
- **Grading bug, found at the first gate read and fixed.**
  - The first count showed 0 primary pairs: every forced-note answer was graded "no_answer_line". The stage-1 code had
    graded only the generated tail, while §51 grades the note and marker ("head") plus the tail.
  - The code was fixed to head + tail, and the stored generations were regraded with the tokenizer only. No model was
    rerun and no D value changed.
  - The pre-regrade file is kept (`runs/exp54/pairs_pre_regrade.json`). The gate fails under either grading.
- **Stop.** Under §6 (stage 1, fewer than 40 primary pairs) and D-45:
  - no stage 2, no timing step and no enlargement of W;
  - the secondary (own-note) pairs are not analysed.

Pre-registration `research/proposals/2026-11-07-exp54-native-switch.md`: rev 0 (D-78, 7eb3c10); rev 0a at 20a46fe
(freeze review: criterion 1 referenced to stable twin pairs; criterion 4 floor; full-grid rule).

Freeze: `runs/exp54/freeze.json`, sha256 2ef769bc…54ce, committed at 4a3703f before any model call (amends 66fcafe).

Spend: $0 (laptop). RunPod programme total unchanged at ≈ $33.75 of $36.

Review: `research/reviews/2026-11-08-exp54-stage1-gate-review.md` (3db285b; gate FAIL and regrade accepted).

Code: `scripts/exp54_cases.py`, `exp54_run.py`, `exp54_patch.py` (the patching core, checked on the dummy model only;
never run on the real model). Build at 0aeef23.

Results: `runs/exp54/cases54.jsonl` (sha256 1f0d4133…), `cases54_counts.json`, `pairs.json`,
`pairs_pre_regrade.json` and the stage-1 logs, at 0aeef23 and 71b9ce7. The 900 B0 records stay local
(`runs/exp54/run/b0/W/`).

## 55. How much false abstention survives a clean, consistent abstention prompt? Frozen vs consistent vs no-option prompt, Qwen 2.5 3B (2026-11-08, pre-registered rev 0 + 0a + 0b; pod 2026-11-09)

**Status: RUN. R1: "the inconsistency is not the main cause". The consistent wording raises false abstention. R2:
"the false abstention is largely the price of having an abstention option". Removing the option is not a fix.**
- **What was planned.** The frozen reader prompt (7b5fc30) gives two conditions for "unknown":
  - the task line: "If the memory does not contain the answer, reply: unknown";
  - the note sentence: "Reply unknown only if the memory has nothing at all on the topic."

  Three arms, built by exact string replacement. Each arm's only difference from P0 is the two sentences
  (asserted per row):
  - **P0:** the frozen prompt, byte-identical.
  - **P1:** one consistent rule. The task line reads "...does not state the answer, reply: unknown" and the note
    sentence reads "Reply unknown only if the memory does not state the answer."
  - **P2:** both sentences removed. This is the §51 `prompt_b3` text, equal on all 750 W rows.
- **Data (development only; seed 56 sealed).**
  - **W:** the §54 split (seed 55), bases with even `base_index`: 75 bases, 25 per task. Rows: ANS 450, UNA-rel 225,
    UNA-swap 75. No CF rows.
  - **LongMemEval:** the 200 §52 rows (folds 0–1): 126 answerable with all evidence in view, 62 answerable but not
    all in view, 12 unanswerable. The prompts equal the Qwen 3B §23 records (200/200), P0 concat sha 7d3b5609 (= §52).
- **Freeze 3bfb13f, before any output.** Rulings, before any output:
  - **rev 0a (45eaee6):**
    - W runs at 256 new tokens.
    - LongMemEval P0 is rerun on the same engine as P1 and P2.
    - LongMemEval is graded with the §52 rules, descriptive only:
      - FA = the graded answer starts with "unknown";
      - gold proxy = `d1`;
      - the whole output is graded when there is no marker.
  - **rev 0b (bc8e145):** the user moved the run to a pod.
    - All arms are generated on the pod, so the §54 B0 reuse was dropped.
    - The 20 W check rows became a device check against the laptop §54 B0.
- **Engine.** The §51 HF engine: Qwen 2.5 3B Instruct fp32, tf32 off, greedy.
  - W free generation is batched at 64 in case-id order.
  - The W controlled prefix uses `chat_ids(prompt_Pk)` + P0's own note through the marker, 16 new tokens, graded
    head + tail. P2's prefix equals `CM.b3_prefix`; P0's prefix arm is P0's free generation itself.
  - LongMemEval runs at batch 1 with 256 new tokens.
  - W is graded with `grade51`. Rates are base-weighted, with 95% intervals from the paired base-cluster bootstrap
    stratified by task (stats51, B 10,000, seed 5102). LongMemEval uses a paired question bootstrap.
- **Run.** Pod x75t0k719w43yh, one A40 secure at $0.59/h. Created 01:32:32 and deleted 02:30:50 UTC on 2026-11-09;
  0 pods before and after.
  - Lifetime 3,498 s; **billing posted $0.587** (3,498,414 ms billed, disk included; quote $1.00, hard stop $1.50 / 150 min).
  - All rows complete and nothing dropped: W 750 × 3 free, 750 × 2 prefix; LongMemEval 200 × 3.
  - D-22 smoke: 8/8 graded classes and 8/8 tokens equal to the laptop §54 B0.
  - Device check: 20/20 tokens and classes equal.
  - Stage times: W free 1,225 s for 2,250 rows; prefix 427 s; LongMemEval 1,649 s for 600 rows. The quote assumed
    about 5 s per LongMemEval row; the measured rate is 2.1 s.
- **W, free generation** (base-weighted; counts in brackets):

  | Arm | FA (ANS) | Gold (ANS) | Uns (UNA) | Uns, UNA-rel | Uns, UNA-swap | U(λ = 1) |
  |---|---|---|---|---|---|---|
  | P0 | 0.311 (140/450) | 0.678 (305) | 0.030 (9/300) | 2/225 | 7/75 | 0.315 |
  | P1 | 0.380 (171/450) | 0.613 (276) | 0.033 (10/300) | 5/225 | 5/75 | 0.282 |
  | P2 | 0.000 (0/450) | 0.987 (444) | 0.653 (196/300) | 145/225 | 51/75 | 0.143 |

  - P1 − P0: FA **+0.069 [0.022, 0.118]**, Gold −0.064 [−0.111, −0.018], Uns +0.003 [−0.013, 0.020], U −0.033
    [−0.058, −0.008].
  - P2 − P1: FA −0.380 [−0.429, −0.333], Gold +0.373 [0.324, 0.427], Uns **+0.620 [0.563, 0.677]**, U **−0.138
    [−0.181, −0.096]**.
- **Readings, as frozen.**
  - **R1:** FA(P1) / FA(P0) = 1.22, which is at least ⅔. Reading: **"the inconsistency is not the main cause."** In
    direction, the consistent wording raises FA from 0.311 to 0.380. P0's wording, which differs from P1 in both
    sentences, including its stricter note sentence ("nothing at all on the topic"), gives lower false abstention than
    the consistent rule; the inconsistency was not inflating it. Which sentence is responsible was not tested.
  - **R2:** FA(P2) = 0 is at most ⅓ FA(P1), and the lower bound of Uns(P2) − Uns(P1) is 0.563 > 0. Reading: **"the
    false abstention is largely the price of having an abstention option."** Removing the option is **not a fix**:
    Uns rises by 0.62, far beyond the 0.05 allowance, and U(λ = 1) falls by 0.138.
- **Controlled prefix** (P0's own note held fixed; 5 rows per arm have no P_ans and are NM):

  | Arm | FA | Gold | Uns | Dam (P0-gold rows that become wrong) |
  |---|---|---|---|---|
  | P0 (= free) | 0.311 (140/450) | 0.678 | 0.030 | — |
  | P1 | 0.471 (209/447) | 0.527 | 0.027 (8/298) | 68/305 |
  | P2 | 0.000 (0/447) | 1.000 (447/447) | 0.587 (175/298) | 0/305 |

  - With the note fixed, the P1 instruction alone moves FA by **+0.160 [0.120, 0.202]**.
  - 163 of 171 P1 abstentions in free generation have the gold in their own note (`d1`).
  - Reading: the instruction acts at the answer step. No mechanism is claimed.
- **LongMemEval (descriptive; 12 unanswerable rows are too few for a claim).**

  | Arm | "unknown", in view (126) | Gold proxy, in view | Marker, in view | "unknown", not all in view (62) | Unanswerable answered (12) |
  |---|---|---|---|---|---|
  | P0 | 28 | 35 | 95 | 37 | 6 |
  | P1 | 35 | 34 | 89 | 36 | 4 |
  | P2 | 0 | 46 | 113 | 1 | 11 |

  - P1 − P0, in-view "unknown": +0.056 [0.000, 0.111].
  - P2 turns 28 in-view "unknown"s into 0 but raises the gold proxy by only 11 (35 → 46, +0.087 [0.016, 0.159]), and
    it answers 11 of 12 unanswerable rows.
  - Pod P0 matches the §23 vLLM bf16 cache on the "unknown" flag on only 173 of 200 rows. The engine matters for
    real-chat numbers.
- **Disclosures.**
  - The first laptop run was stopped at the user's request, before any P1, P2 or UNA output. Its partials (a
    20-row P0 rerun, 20/20 byte-identical to §54 B0, and 450 copied §54 B0 records) are unused and uncommitted
    (`runs/exp55/run/UNUSED.txt`).
  - The laptop stub of the pod smoke stage was stopped after its 8 W rows, because the dummy LME rows were slow. LME
    generation was covered by the LME stub. The stub tested resume (448 → 512 rows, finished files untouched), the
    deadline exit 9, and the prefix path on 20 real P0 notes.
  - D-30: a fresh venv (torch 2.8.0) passed the install line.
- **What this changes (decided in the proposal).**
  - False abstention on these controlled tasks is framed as abstention inflation, a calibration trade-off, with
    arXiv 2507.16199 as the closest prior work.
  - A cross-model table from cached outputs is not built on P0 alone.

Results: `runs/exp55/freeze.json` (sha256 24ec9eb6…), `rows_W.jsonl` and `rows_LME.jsonl` at 3bfb13f. Runner 43917de
/ 244a586. Report `runs/exp55/report.json`, pod logs and launch scripts at f07fbfd. The raw pod records stay local
(`runs/exp55/pod/run/`). Scripts: `scripts/exp55_prompts.py`, `exp55_freeze.py`, `exp55_run.py`, `exp55_pod.sh`
and `exp55_report.py`.

## 56. Which MLP neurons carry false "unknown" at the answer position? Direct-push ranking, ablation and the logit-bias dial, Qwen 2.5 3B (2026-11-10, pre-registered rev 0 + 0a; laptop stage 1 2026-11-10)

**Status: STOPPED at the stage-1 gate (proposal §7 / D-45). R1 = DISTRIBUTED: K50 = 14,966 neurons (95% CI 14,899–15,130),
against a bar of 500. Nothing was ablated.**
- **Question.** Is the push toward "unknown" at P_ans (the ':' of the answer marker) carried by a small set of SwiGLU
  neurons? Is it the same set for false and true abstentions? Does switching it off release false abstentions more
  selectively than the logit bias B1?
- **Design.**
  - Rows come from the §54 W split (seed 55). Dev-A (even `base_index`) chooses the candidate set K; dev-B (odd
    `base_index`) was to test it.
  - Classes at P_ans: FA (ANS row, "unknown"), C (ANS row, gold), TA (UNA row, "unknown"), UA (UNA row, other). NM
    (no usable P_ans) and W (ANS row, wrong answer) are excluded.
  - ANS records come from §54 B0. Dev-A UNA records come from the §55 pod P0 run (HF fp32, no TF32; 20 of 20 matched
    the laptop byte for byte at §55 check20).
  - The measure is direct logit attribution (DLA) per neuron at P_ans, through the final RMSNorm scale.
    - v = g ⊙ (u_unknown − u_t*) / σ, where g is the final-norm weight, u are unembedding rows, and σ is the RMS of
      the final-norm input.
    - Neuron i of layer l contributes a_i · ⟨W_down[:, i], v⟩, where a_i is the down_proj input.
    - D = logit[" unknown"] − logit[t*].
  - Rev 0a (df5d028): t* is the highest-logit first token among the row's 12 city names (all single tokens) on every
    class. All 36 layers are stored. The rev-0 rule (top token outside the B1 "unknown" set) is kept as a
    descriptive only. On TA rows it returned "未知" ("unknown" in Chinese) on 142 of 289.
  - Readings R1–R4 and the stopping rules are fixed in §6–§7 of the proposal. Seed 56 stays unused for data; it
    seeds the bootstraps only.
- **Stage 0 (freeze).**
  - Dev-A counts: FA 140, C 305, TA 289, UA 9. Out: NM 5, W 2.
  - Dev-B counts: ANS FA 148, C 298, NM 4. The 300 dev-B UNA rows were listed but never generated.
  - Two-row layer-sum check: PASS (max error 8.3e-6 after rev 0a).
  - Timing: read 6.7 s per row; ablated prefix generation 8.3 s; peak RSS 17.7 GB.
- **Stage 1 run.** 743 dev-A reads, 14:01–15:57 PT (22:01–23:57 UTC), 9.3 s per row, peak RSS 17.7 GB, no errors.
  Checks on all rows:
  - per-layer neuron DLA sums to the layer's MLP DLA (max error 1.6e-5);
  - embedding + attention + MLP DLA equals the linear D (max error 5.2e-6);
  - |D − linear D| ≤ 1.3e-5.
- **R1 (concentration; the reading that decides).**
  - Mean DLA over the 140 FA rows, 396,288 neurons; 191,604 have a positive mean.
  - K50 (smallest top-k holding half the positive push) = **14,966**. 95% CI [14,899, 15,130]: 2,000 draws, clustered
    by base over the 52 bases with an FA row. K50 > 500 → **DISTRIBUTED**.
  - Share of the positive push held by the top-k neurons:

    | k | 10 | 50 | 100 | 500 | 1,000 | 5,000 |
    |---|---|---|---|---|---|---|
    | share | 1.2% | 3.1% | 4.4% | 9.8% | 13.7% | 29.7% |

- **Layer profile (required by the stop rule).**
  - The positive-push share rises gently with depth: at most 1.3% per layer in layers 0–5, about 2.5% in 11–21,
    3.4–4.3% in 22–29, and 4.6–5.8% in 30–35. No layer holds more than 5.8%.
  - The top-K50 neurons sit in all 36 layers, from 4 in layer 1 to 970 in layer 31.
  - Net mean MLP DLA on FA rows by layer is small (|net| < 2) in every layer except layer 35, which is −12.3.
- **At neuron resolution the push is distributed.** Under §7 and D-45 the experiment stops:
  - no stage 2, 3 or 4;
  - no dev-B UNA generation;
  - K is frozen but unused.
- **Descriptives after the stop (no reading).**
  - Frozen K = the top 50 by mean FA DLA, with dev-A C-row mean activations as ablation values (`runs/exp56/K.json`,
    sha256 78a258be…). Layers: 22:1, 25:1, 27:1, 28:5, 29:2, 30:4, 31:3, 32:2, 33:10, 34:7, 35:14.
  - Σ over K of mean DLA by class: FA 8.46, TA 10.06, C 7.78, UA 9.96.
    - TA/FA = 1.189, CI [1.123, 1.268] (all 75 bases, K held fixed). R2 would read SHARED CHANNEL.
    - C/FA = 0.920: the top FA neurons push nearly as hard on rows the model answers correctly.
  - K50 on the other classes is just as spread: TA 13,931, C 14,743, UA 14,402. Top-50 overlap: FA∩TA 29, FA∩C 37.
  - D by class (mean / median / fraction > 0):

    | Class | Mean | Median | Fraction > 0 |
    |---|---|---|---|
    | FA | +6.93 | +4.08 | 1.00 |
    | TA | +27.5 | +30.4 | 1.00 |
    | C | −2.74 | −3.25 | 0.10 |
    | UA | −2.29 | −1.79 | 0.00 |

  - Direct-path split by component (summed over layers). Attention DLA: FA +14.50, C +3.81, TA +32.28. MLP DLA: FA
    −7.68, C −6.65, TA −4.87.
    - Layer-35 MLP pushes against "unknown" in every class (FA −12.3, TA −12.7, C −9.9).
    - This is a direct-path measure. Attention's share includes whatever upstream MLPs write that attention then
      moves to P_ans. It does not show that the decision is made in attention.
  - Write vectors of K (down_proj column × the sign of the mean FA activation), mean |cosine|:
    - with the §51 rank-1 edit U (8364bf13): 0.087 against 0.017 for random same-layer neurons. Ranks 15 and 16 reach
      −0.51 and −0.48; the sign is negative, as expected, because U is anti-"unknown";
    - with the §38 v_29: 0.036 against 0.017.
  - City t* equals the gold first token on FA 129 of 140 and on C 305 of 305.
- **Report bug, found before any verdict was written and fixed.**
  - The first report run unpacked the class index arrays in `X.CLASSES` order (FA, C, TA, UA) into variables named
    `fa, ta, cc, ua`. TA and C were therefore swapped in the R2 point estimate, the ablation values and the per-class
    labels.
  - The R2 point estimate printed 0.920 (actually C/FA), outside its own interval. The bootstrap selected rows by
    class name, so its interval was right.
  - Fix: unpack by name. R1 and the choice of K used only the FA rows, which were always correct.
  - The stop rule's layer profile was added in the same rerun. The Advisor recomputed all headline numbers
    independently from the stored reads.
- **Numerical noise.** The stage-0 FA check row gave D = 0.482948 in one process and 0.482868 in another (about 8e-5).
  Repeats within one process were identical. This is far below every threshold.

Pre-registration `research/proposals/2026-11-10-exp56-answer-neurons.md`: rev 0 (D-83, 6e6b075); rev 0a at df5d028
(stage-0 gate: city t* on every class, all 36 layers stored, stage 1 first, batched-arm stage 2 gated on a batch-1
equality test).

Freeze: `runs/exp56/freeze.json`, sha256 55d3d87b…44c2, committed at 3690b0f before any neuron was ranked (amends
98d96e7). Rows: `runs/exp56/rows.jsonl`, sha256 d833da43….

Spend: $0 (laptop). RunPod programme total unchanged.

Review: Advisor stage-1 ruling ACCEPT, 2026-11-10 (stop rule fires; numbers recomputed independently).

Code: `scripts/exp56_common.py`, `exp56_stage0.py`, `exp56_stage1.py`, `exp56_stage1_report.py`;
`tests/test_exp56.py` (dummy model). The stage-2 engine `scripts/exp56_arms.py` (one prefill per row, arms batched)
passes a batch-1 equality test on the dummy model only; it was never run on the 3B.

Results: `runs/exp56/stage1_report.json` (sha256 03802d97…), `K.json`, `c_mean_act.npy`, `rows_counts.json`,
`dla_test.json`, `timing.json` and the logs, at 98d96e7, 3690b0f and 7d78ee0. The 743 per-row reads (about 3.1 MB
each) stay local (`runs/exp56/run/stage1/`).

## 57. Which neurons differ on false "unknown", and does resetting them fix it? Total-effect ranking at the answer and over the note, random sets and the logit-bias dial, Qwen 2.5 3B (2026-11-10/11, pre-registered rev 0 + 0a + 0b; laptop stages 0–3, $0)

**Status: CLOSED after stage 3 (dev only). Selected neurons beat random sets by a wide margin, but no neuron set beats
a plain logit bias. The answer-position set K_P is WORSE than the bias at matched release (more answered-unanswerable
rows, plus damage to correct rows). The note/marker set K_N ties it (NO DIFFERENCE SHOWN), and its release almost
vanishes when the reset stays in distribution at the marker (R3′ NOT SHOWN), so it is read as an off-distribution
push. No causal sentence is written about K_N.**
- **Question.** §56 found the direct push toward "unknown" spread over ~15,000 MLP neurons. Does a total-effect
  ranking (attribution patching) find a small set whose reset to correct-row values releases false abstentions,
  beyond random sets of the same shape, and more selectively than the logit bias B1?
- **Design.**
  - Rows, classes and D as in §56 (W split, seed 55; t* = the best city first token of the row's pool). Dev-A chooses
    everything; dev-B tests. No held-out split was read; seed 57 was not generated.
  - Two regions: P = P_ans alone; N = the forced note and the "Answer" marker before P_ans.
  - Reset values: the dev-A C-row mean at P_ans (P, `c_mean_act.npy`) and the token-weighted dev-A C-row mean over
    all note positions (N, `c_mean_note.npy`, 305 rows, 7,768 positions).
  - Predicted effect Δ̂_i = Σ_{t∈R} (ā^C_i − a_i(t)) · ∂D/∂a_i(t), from one forward and one backward pass. Score
    S^R_i = mean over dev-A FA rows of −Δ̂_i. K_P and K_N = the top 50 by S^P and S^N.
  - Stage 3 (dev-B): each set reset at its region; 20 random per-layer-matched sets per region (seed 5701); B1 =
    −b on " unknown" at P_ans. Grading: §51 grader, head + tail.
  - Readings R1–R4 fixed in proposal §6; rev 0a (476a580) made R4 a matched-release comparison on dev-B over a frozen
    B1 grid; rev 0b (02c7cab, at the stage-2 gate) added a secondary arm K_N@marker.
- **Stage 0 (3501a72): both gates PASS.**
  - Memory: forward + backward at all 36 layers fits (peak RSS 17.3 GB on a 721-token row, 25.7 s).
  - Validity (Δ̂ vs real single-neuron patches, dev-A): Spearman P 0.998, N 0.950. Set-level sign 10/10; for 50-neuron
    sets the real effect was 0.51–0.87 of Δ̂.
  - Equality: the batched arm engine matched batch 1 on 440/440 row-arms (max |ΔD| 1.2e-4), 10.5× faster.
- **Stage 1 (descriptive, stored §56 reads).** 6,240 neurons differ between FA and C at |d| ≥ 1.0, all in layers
  21–35; strongly firing neurons are not enriched for the push (neuron-level p97 holds 7.4% of it).
- **Stage 2 (b284ce2; 743 dev-A rows, 19.1 s/row).**
  - **R1 = DISTRIBUTED in both regions.** P: K50 13,640 [13,065, 14,516]; N: K50 14,912 [14,903, 15,424]. Top-50
    share P 2.7%, N 1.9%. The percentile CI leans upward: on random halves of the FA bases K50 rose above the full
    value in 50/50 halves at N and 35/50 at P (6584388). R1 never stops stage 2 (§7).
  - **R2 (TA/FA over K):** P 1.136 [1.02, 1.26] SHARED; N 0.330 [0.27, 0.40] FA-SPECIFIC. C/FA (P 0.102, N 0.563) is
    not comparable across regions: the P reset is the C mean at P_ans, so C rows have Δ̂ ≈ 0 there by construction.
    At P, R2 ≈ 1 means K_P is "unknown"-output machinery shared by true and false abstentions, consistent with §56;
    it is not evidence about why FA happens.
  - K_P sits in layers 21–35 (mostly 26–31); K_N in layers 6 and 17–27 (mostly 19–22). K_P ∩ K_N = 0; K_N shares no
    neuron with §56's K or with the stage-1 top-1,000 by |d|.
  - **Where K_N acts:** the "\nAnswer" marker carries 0.90 of the K_N push on FA rows (3.7% of note positions; top
    token on 140/140 rows). The gold-city mention carries −0.002 and hedge words ~0.
- **Rev 0b check (250f548, dev-A).** At the marker, FA rows already sit at the C distribution for K_N (median |z|
  0.19, none above 2, in C-row SDs). The frozen note-mean reset sits 5.7 SD from C (median) and above 2 SD for all
  50 neurons. The primary K_N arm therefore pushes the marker far out of distribution.
- **Stage 3 (dev-B; FA 148, TA 291, C 298; 75 bases; 05:37–13:31 PT, 29 s/row; equality re-run with all 74 arms
  PASS 162/162, 1e147dd).**
  - Dev-B UNA generation (5800ebd): 300 rows under P0 → TA 291, UA 6, NM 3.
  - b_cal (1c1ad9b): the dev-A ladder rises monotonically but saturates at Rel 0.85 (b ≥ 13), below K_N's dev-A Rel
    0.871, so b_cal = 16 is **unmatched**.
  - B1 grid on dev-B (Rel / Uns): b 1 0.236/0.003; 2 0.466/0.007; 3 0.561/0.021; 4 0.649/0.024; 5 0.716/0.027;
    6 0.750/0.041; 8 0.804/0.076; 10 0.831/0.089; 12 0.858/0.127.

    | Arm | Rel (FA → gold) | Random sets: max / median | R3 | Uns (TA answered) | Dam (C rows) | R4 matched (Uns − B1 at equal Rel) | R2′ = Uns/Rel (B1 at nearest grid point) |
    |---|---|---|---|---|---|---|---|
    | K_P | 0.581 | 0.014 / 0.000 | **SHOWN** (1/21) | 0.052 | 7/298 → "unknown" | +0.030 [0.012, 0.057] **WORSE** | 0.089 (B1 b=3: 0.037) |
    | K_N | 0.831 | 0.034 / 0.007 | **SHOWN** (1/21) | 0.093 | 0 | +0.003 [−0.031, 0.025] **NO DIFFERENCE SHOWN** (5.8% draws dropped) | 0.112 (B1 b=10: 0.108) |
    | K_N@m (secondary) | 0.088 | 0.007 / 0.000 | **NOT SHOWN** (1/21 but Rel < 0.20) | 0.000 | 3/298 → "unknown" | not read (R3 NOT SHOWN) | — |
    | B1 (any b) | grid above | — | — | grid above | 0 by construction (step-0 bias against "unknown" cannot flip a gold answer) | — | — |

  - **R4 dev-A-calibrated: NOT READ for every set.** b_cal is unmatched, so the comparison is at unequal release
    (Rel − Rel(B1_cal) is negative for every arm), and lower Uns then follows from less release. The raw verdict is
    kept in the report as `verdict_raw` and is not a result.
  - **Rev 0b reading (fixed before the run).** The primary K_N arm is SHOWN and K_N@marker is not, so K_N's release
    is described as an off-distribution push at the marker. Δ̂ agrees: on dev-B FA rows the predicted ΔD tracks the
    real one for K_P (Spearman 0.96; mean −7.96 predicted vs −5.64 real) but not for K_N (Spearman 0.23; mean −16.7
    vs −7.9).
  - U(λ = 1), descriptive: base 0.332, K_P 0.396, K_N 0.422, K_N@m 0.344, B1 at b = 16 0.394. No U was computed for B1
    at the matched grid points (grid arms ran on FA and TA rows only).
- **Reading.** Resetting a frozen 50-neuron set releases far more false abstentions than random sets of the same
  shape, in both regions. Neither set does so more selectively than lowering the "unknown" logit: the answer-position
  set answers more unanswerable rows at equal release and turns 7 correct answers into "unknown" (it is shared "unknown"
  output machinery, R2 1.136, so releasing it also releases true abstentions); the note set ties
  the bias, and its effect depends on pushing the marker state 5.7 SD outside where correct rows sit. Kept in
  distribution, the same neurons release 0.088.
- **Not done.** Stage 4 (what the survivors are) is not proposed: K_P loses to B1 and K_N's effect is not
  in-distribution. No stage 5 (seed 57). No attention-head analysis (the §7 next unit if R3 had failed; it is still
  the open thread from §56's direct-path split).
- **Spend.** $0 (laptop). RunPod programme total unchanged.
