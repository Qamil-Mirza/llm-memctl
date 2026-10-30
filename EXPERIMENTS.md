# Experiments

Every experiment run while the framework was built, with the command that
reproduces it, the result, and what it does and does not show. Experiments
1–7 date from 2026-09-30; later sections carry their own dates. To get full
tables and figures, generate them with `python -m memctl.analysis.report
runs/<name>` (writes `runs/<name>/report.md` and `runs/<name>/plots/`; the
`runs/` folder is not in git; every table below can be regenerated with the
commands given).

**Contents** (sections are not in numeric order): 1, 1b, 1c, 1d, 2, 3, 4,
4b, D1, GRPO, D2 (D2b), D5 (D5b, D5c), 8, 9, 10, 11, 5, 5b, 6, 6c, 7 (7a, 7b).

**How to read the numbers**

- *Task success* is the mean over episodes of the share of queries answered
  correctly (recall task) or jobs completed (workflow task).
- Every controller in a table ran the same seeded episodes (seeds 0 to 99
  unless stated), so differences are paired.
- Budgets given in % are a share of the episode's uncompressed history.
- The task model is the scripted reader unless stated: it answers correctly
  exactly when the needed text is in ACTIVE memory. So task success here
  measures memory management only.
- **Provenance:** Experiments 1–6 were first run from an uncommitted working
  tree (`dirty: true`). On 2026-10-05 every training and sweep was rerun from
  clean commit `ef57fc4` (642 run folders, all `dirty: false`; Experiment 7
  needs a remote model and was not rerun). **Every task-success number
  reproduced exactly except 9 cells of Experiment 4b**: the two archive-trained
  RL policies (8 cells, moved by up to 0.095, within training-seed spread) and
  `salience_archive_retrieve` at 2% (0.737 → 0.744). The table below shows the
  clean values; the conclusions are unchanged. The
  per-cell detail in the 4b discussion (retrieval counts, failure counts) is
  from the original run. Experiments D1–D5c were run from clean commits as
  stated in their sections.

| # | Question | Status |
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

---

## 1. Do different policies differ at the same budget?

The first milestone: *at the same active-memory budget, do different
memory-management policies produce measurably different long-horizon task
performance?*

```bash
python -m memctl.sweep --config configs/sweeps/exp1_delete_only.yaml
python -m memctl.analysis.report runs/exp1_delete_only
```

Synthetic recall task, action set KEEP / EVICT / NO_OP, 11 controllers × 6
budgets × 3 horizons × 100 episodes (198 runs, about 15 minutes on 13 cores).

**Horizon 500** (the other horizons are in the report):

| controller | 2% | 5% | 10% | 25% | 50% | 100% |
|---|---|---|---|---|---|---|
| no_controller (FIFO by forced fallback) | 0.036 | 0.233 | 0.437 | 0.725 | 0.901 | 1.000 |
| fifo | 0.036 | 0.233 | 0.437 | 0.725 | 0.901 | 1.000 |
| lru | 0.037 | 0.240 | 0.459 | 0.755 | 0.921 | 1.000 |
| lfu | 0.041 | 0.219 | 0.451 | 0.770 | 0.929 | 1.000 |
| age_decay | 0.039 | 0.245 | 0.460 | 0.745 | 0.913 | 1.000 |
| random | 0.069 | 0.195 | 0.358 | 0.608 | 0.840 | 1.000 |
| similarity | 0.029 | 0.091 | 0.168 | 0.377 | 0.704 | 1.000 |
| salience | 0.402 | 0.580 | 0.762 | 0.962 | 1.000 | 1.000 |
| oracle_approx | 0.768 | 0.984 | 1.000 | 1.000 | 1.000 | 1.000 |
| oracle_exact | 0.871 | 0.992 | 1.000 | 1.000 | 1.000 | 1.000 |
| full_context (unlimited) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

Paired differences from the baseline at horizon 500, 10% budget (100 episodes):

| controller | difference | 95% interval | minimum detectable effect |
|---|---|---|---|
| lru | +0.023 | [+0.018, +0.027] | 0.007 |
| age_decay | +0.023 | [+0.018, +0.029] | 0.008 |
| lfu | +0.015 | [+0.001, +0.028] | 0.019 |
| random | −0.079 | [−0.094, −0.063] | 0.022 |
| similarity | −0.269 | [−0.286, −0.251] | 0.025 |
| salience | +0.326 | [+0.303, +0.347] | 0.031 |
| oracle_exact | +0.563 | [+0.548, +0.579] | 0.022 |

**The four pass criteria fixed before the run:**

1. *At 100% budget every controller matches unlimited context.* Yes: 1.000 in all 33 cells.
2. *With no reader noise every failure is an eviction.* Yes: 380,231 failed
   queries, all labelled `evicted`; 46,782 of them with cause `harness` (all
   from `no_controller`) and the rest with cause `controller`.
3. *The oracle-to-FIFO gap at 25% budget and below exceeds the minimum
   detectable effect.* Yes: 0.23 to 0.88 against 0.01 to 0.04.
4. *At least one pair of non-oracle controllers differs by more than that
   effect.* Yes. Of the 15 cells below 100% budget, the interval for the
   difference from FIFO excludes zero in 15 for salience, 14 for random, 14 for
   similarity and 12 for LRU.

**What it shows**

- Policies differ, by a lot. At horizon 500 and 10% budget the range between
  non-oracle controllers is 0.17 to 0.76.
- `no_controller` and `fifo` are the same policy reached two ways, and the
  numbers agree exactly. The fallback is reported, not hidden:
  `no_controller` has forced evictions on most steps and `fifo` has none.
- Recency-style rules (FIFO, LRU, LFU, age decay) are within a few points of
  each other. A content-aware rule (salience) is far ahead. Keeping what
  resembles the current observation (similarity) is worse than random.
- The exact oracle is never below the approximate one and separates from it at
  2% and 5%.

**What it does not show**

- That salience is a good policy in general. Its hand-set weights (user >
  observation > tool output, and "has identifiers") match how the generator
  decides what gets asked about. Experiment 5 shows it failing on another task.
- Anything about horizon. The budget is a fraction, so a longer episode has a
  larger budget in tokens. That is why success *rises* with horizon here.
  Experiment 1c removes the confound.

## 1b. Does attribution recover a failure rate we inject?

```bash
python -m memctl.sweep --config configs/sweeps/exp1b_attribution_check.yaml
```

The scripted reader fails 20% of the queries whose evidence it has. Horizon
500, 100 episodes, 5,265 queries per cell. Recovered rate = reasoning failures
/ (correct + reasoning failures).

| controller | budget | task success | evicted | task_model_reasoning | recovered rate |
|---|---|---|---|---|---|
| fifo | 10% | 0.349 | 2,982 | 454 | 0.199 |
| fifo | 25% | 0.574 | 1,452 | 793 | 0.208 |
| fifo | 100% | 0.792 | 0 | 1,093 | 0.208 |
| salience | 10% | 0.605 | 1,269 | 819 | 0.205 |
| salience | 25% | 0.762 | 203 | 1,050 | 0.207 |
| oracle_exact | 10% | 0.792 | 0 | 1,093 | 0.208 |

The injected rate on these seeds was 1,093 / 5,265 = 0.208. Attribution
recovers it at every budget and never blames the controller for it. The exact
oracle's 0.792 is the task model's ceiling, not the controller's.

## 1c. Horizon at a fixed token budget

```bash
python -m memctl.sweep --config configs/sweeps/exp1c_horizon_fixed_budget.yaml
```

Budget fixed at 800 tokens; horizon 200 to 2,000 steps.

| controller | 200 | 500 | 1000 | 2000 |
|---|---|---|---|---|
| fifo | 0.282 | 0.204 | 0.155 | 0.126 |
| lru | 0.291 | 0.209 | 0.160 | 0.130 |
| random | 0.264 | 0.173 | 0.133 | 0.111 |
| salience | 0.765 | 0.555 | 0.448 | 0.312 |
| oracle_approx | 1.000 | 0.972 | 0.882 | 0.772 |
| oracle_exact | 1.000 | 0.985 | 0.947 | 0.887 |

Everything degrades with horizon, the oracle least. The gap between the best
heuristic and the exact oracle grows from 0.24 to 0.58, so the room a learned
policy could win grows with horizon. The approximate oracle falls behind the
exact one by up to 0.12. An episode with the exact solve took 0.3 s at horizon
500, 0.8 s at 1,000 and 7.5 s at 2,000.

Confound: in this generator the distance between a fact and its query grows
with the horizon (log-uniform up to the horizon), so "longer horizon" and
"longer dependencies" move together here. Experiment 1d varies only the latter.

## 1d. Dependency distance

```bash
python -m memctl.sweep --config configs/sweeps/exp1d_dependency_gap.yaml
```

Horizon 500, budget 10%; only the minimum distance from a fact to its query changes.

| controller | min gap 5 | 25 | 100 | 250 |
|---|---|---|---|---|
| fifo | 0.504 | 0.265 | 0.035 | 0.051 |
| lru | 0.525 | 0.267 | 0.035 | 0.050 |
| random | 0.422 | 0.231 | 0.075 | 0.037 |
| salience | 0.769 | 0.740 | 0.687 | 0.573 |
| oracle_exact | 1.000 | 1.000 | 1.000 | 1.000 |

Recency rules collapse once dependencies are longer than the window the budget
can hold (0.50 to 0.04); a content-aware rule loses far less; the oracle is
unaffected. This is the clearest support for H3. Queries per episode fall from
56 to 18 as the gap grows (fewer queries fit before the episode ends), so the
estimates at gap 250 are noisier.

## 2. Archive and retrieval against deletion

```bash
python -m memctl.sweep --config configs/sweeps/exp2_archive_retrieval.yaml
```

Horizon 500. MOVE_TO_ARCHIVE and RETRIEVE_FROM_ARCHIVE are now allowed. The
archive is unlimited. Retrieval is BM25 over the archive, top 3, run by the
controller when a query arrives.

| controller | 2% | 5% | 10% | 25% |
|---|---|---|---|---|
| fifo_delete | 0.036 | 0.233 | 0.437 | 0.725 |
| fifo_archive_no_retrieval | 0.036 | 0.233 | 0.437 | 0.725 |
| fifo_archive_retrieve | 0.673 | 0.734 | 0.808 | 0.908 |
| lru_archive_retrieve | 0.671 | 0.729 | 0.811 | 0.914 |
| salience_delete | 0.402 | 0.580 | 0.762 | 0.962 |
| salience_archive_retrieve | 0.719 | 0.794 | 0.883 | 0.989 |
| salience_archive_retrieve_top1 | 0.605 | 0.722 | 0.842 | 0.986 |
| salience_archive_retrieve_embedding (hashing) | 0.629 | 0.749 | 0.861 | 0.980 |
| rag_keep_last4 | 0.672 | 0.672 | 0.672 | 0.672 |
| oracle_delete_only_exact | 0.871 | 0.992 | 1.000 | 1.000 |
| oracle_approx (archives, retrieves just in time) | 1.000 | 1.000 | 1.000 | 1.000 |

At 5% budget, BM25 top-3 retrieval has recall 0.56 to 0.69 and precision 0.10
to 0.26 (top-1: recall 0.37; hashing embedder: recall 0.44). Every failure of
an archiving controller is labelled `archived_not_retrieved`.

**What it shows**

- Archiving without retrieving is deletion under another name: identical
  success, with the failures relabelled from `evicted` to
  `archived_not_retrieved`. The value is in the retrieval.
- Archive + retrieval is worth far more than a better eviction rule: FIFO goes
  from 0.233 to 0.734 at 5%.
- Once retrieval exists, what stays in context matters less: FIFO and salience
  are 0.35 apart when deleting and 0.06 apart when archiving. This is the
  Phase 2 finding again, now attributable: the remaining failures are all
  retrieval failures.
- With a perfect retriever the oracle is at 1.000 with a 2% budget, above the
  exact delete-only optimum (0.871). H4 holds here.

**Confounds**

- The archive is free and unlimited (17,000 tokens at the peak). "Same memory
  budget" holds for ACTIVE only. Use `memory.archive_budget` to price it.
- Retrieving controllers take about 1.5 ms per step against 0.06 ms; the
  comparison is at equal ACTIVE budget, not equal compute.
- "Embedding" retrieval here is the hashing embedder, which is word overlap
  without BM25's weighting. It says nothing about dense retrieval (see 6c).

## 3. Compaction and consolidation

```bash
python -m memctl.sweep --config configs/sweeps/exp3_compaction_consolidation.yaml
```

Horizon 500, a more verbose and more redundant stream (`verbose_prob 0.5`,
`restate_prob 0.4`), all eight operations allowed. Two rewriters: `extractive`
keeps the most specific sentences; `truncate` keeps the first tokens.

| controller | extractive 5% | 10% | 25% | truncate 5% | 10% | 25% |
|---|---|---|---|---|---|---|
| fifo_delete | 0.238 | 0.439 | 0.726 | 0.238 | 0.439 | 0.726 |
| fifo_compact | 0.396 | 0.608 | 0.888 | 0.260 | 0.418 | 0.668 |
| fifo_consolidate | 0.238 | 0.447 | 0.743 | 0.238 | 0.447 | 0.743 |
| fifo_consolidate_tight | 0.241 | 0.455 | 0.760 | 0.240 | 0.452 | 0.752 |
| fifo_compact_archive_retrieve | 0.871 | 0.917 | 0.972 | 0.751 | 0.804 | 0.893 |
| salience_delete | 0.535 | 0.725 | 0.942 | 0.535 | 0.725 | 0.942 |
| salience_compact | 0.546 | 0.725 | 0.967 | 0.473 | 0.618 | 0.775 |
| salience_compact_consolidate | 0.429 | 0.610 | 0.928 | 0.394 | 0.535 | 0.775 |
| salience_compact_consolidate_tight | 0.503 | 0.717 | 0.993 | 0.446 | 0.605 | 0.779 |
| oracle_delete_only_exact | 0.987 | 1.000 | 1.000 | 0.987 | 1.000 | 1.000 |

**What it shows**

- Compaction helps when the rewriter keeps the detail and hurts when it does
  not. With `truncate`, 1,368 to 1,443 failures per cell at 10% are labelled
  `compression_lost_detail`, which is the attribution working as intended.
- It helps a weak policy more than a good one (FIFO +0.17 at 10%, salience +0.00).
- Consolidation of restated facts gives FIFO one or two points. Naive
  consolidation *hurts* salience (0.725 to 0.610): the merged item is long,
  cannot be compacted again, and is evicted whole, taking with it a fact the
  policy would have kept as a short item. Shortening the merged item
  (`_tight`) removes most of the loss.
- Keeping a short stub and archiving the original is the best non-oracle
  configuration (0.917 at 10%).

**Confound:** the `extractive` rewriter scores sentences by the same notion of
"specific" (identifiers, numbers) that marks facts in this generator, so it is
close to lossless by construction. H5 is supported only conditionally: the
result is a statement about the rewriter, not about compaction in general.

## 4. Learned controllers against heuristics and the oracle

```bash
for c in rl_bc rl_bc_mlp rl_ppo rl_ppo_seed1 rl_ppo_seed2 rl_ppo_gamma0 rl_ppo_gamma0_seed1 rl_ppo_gamma0_seed2 rl_bc_ppo; do
  python -m memctl.rl.train --config configs/rl/$c.yaml; done
python -m memctl.sweep --config configs/sweeps/exp4_rl_eval.yaml
```

Delete-only action set. Every policy is the same small network (27,526
parameters, DeepSets over per-item features; 19,334 for the plain MLP of `rl_bc_mlp`),
trained at horizon 200 on four budgets (5, 10, 25, 50%), on seeds 100000+,
with `policy_best.pt` chosen on validation seeds 50000+. Evaluation here uses
seeds 0–99 like every other experiment; horizon 500 is a transfer test.

| training | episodes | what it is |
|---|---|---|
| `rl_bc`, `rl_bc_mlp` | 240 | DAgger imitation of the hindsight expert |
| `rl_bc_ppo` | 240 + 960 | the same, then PPO on task reward |
| `rl_ppo` (3 seeds) | 2,400 | PPO from scratch, discount 0.995 |
| `rl_ppo_gamma0` (3 seeds) | 2,400 | PPO from scratch, discount 0 (immediate reward only) |

**Task success** (`rl_ppo_last` is `rl_ppo`'s final policy rather than its best-validated one):

| controller | 200: 5% | 10% | 25% | 50% | 500: 5% | 10% | 25% | 50% |
|---|---|---|---|---|---|---|---|---|
| fifo | 0.058 | 0.238 | 0.618 | 0.877 | 0.233 | 0.437 | 0.725 | 0.901 |
| lru | 0.058 | 0.245 | 0.645 | 0.905 | 0.240 | 0.459 | 0.755 | 0.921 |
| salience | 0.546 | 0.738 | 0.962 | 1.000 | 0.580 | 0.762 | 0.962 | 1.000 |
| rl_bc | 0.512 | 0.725 | 0.960 | 1.000 | 0.531 | 0.755 | 0.962 | 1.000 |
| rl_bc_mlp | 0.455 | 0.702 | 0.970 | 1.000 | 0.516 | 0.758 | 0.977 | 1.000 |
| rl_bc_ppo | 0.525 | 0.726 | 0.973 | 1.000 | 0.558 | 0.766 | 0.981 | 1.000 |
| rl_ppo | 0.485 | 0.621 | 0.899 | 0.981 | 0.361 | 0.590 | 0.900 | 0.987 |
| rl_ppo_seed1 | 0.358 | 0.543 | 0.889 | 0.997 | 0.345 | 0.613 | 0.924 | 0.997 |
| rl_ppo_seed2 | 0.469 | 0.621 | 0.874 | 1.000 | 0.383 | 0.634 | 0.881 | 1.000 |
| rl_ppo_last | 0.460 | 0.615 | 0.874 | 0.990 | 0.340 | 0.572 | 0.840 | 0.993 |
| rl_ppo_gamma0 | 0.308 | 0.581 | 0.914 | 0.995 | 0.421 | 0.687 | 0.925 | 0.999 |
| rl_ppo_gamma0_seed1 | 0.375 | 0.617 | 0.915 | 0.996 | 0.368 | 0.658 | 0.928 | 0.999 |
| rl_ppo_gamma0_seed2 | 0.461 | 0.639 | 0.891 | 0.988 | 0.427 | 0.637 | 0.894 | 0.988 |
| oracle_exact | 0.938 | 0.998 | 1.000 | 1.000 | 0.992 | 1.000 | 1.000 | 1.000 |

**Paired difference from salience** (the best heuristic), horizon 500:

| controller | 5% | 10% | 25% |
|---|---|---|---|
| rl_bc | −0.049 [−0.069, −0.030] | −0.007 [−0.020, +0.005] | −0.000 [−0.005, +0.005] |
| rl_bc_mlp | −0.064 [−0.077, −0.051] | −0.004 [−0.012, +0.004] | +0.015 [+0.010, +0.019] |
| rl_bc_ppo | −0.022 [−0.037, −0.008] | +0.004 [−0.006, +0.013] | +0.018 [+0.014, +0.023] |
| rl_ppo (3 seeds) | −0.20 to −0.24 | −0.13 to −0.17 | −0.04 to −0.08 |
| rl_ppo_gamma0 (3 seeds) | −0.15 to −0.21 | −0.08 to −0.13 | −0.04 to −0.07 |

**Other measurements**

- *Oracle gap closed* (relative to FIFO): imitation policies 0.39–0.64 at 5–10%
  and 0.86–0.93 at 25%; PPO policies 0.14–0.53 and 0.57–0.78; salience 0.46–0.66
  and 0.86–0.90.
- *Shadow agreement* (overlap between the items a policy removes and the
  items a shadow heuristic would remove in the same state): every learned
  policy overlaps FIFO and LRU by 0.00–0.04 and salience by 0.2–0.7. The learned
  policies are content-based, not recency-based.
- *Latency:* learned policies take 1–7 ms per step (PyTorch on one CPU thread,
  on a loaded machine); salience 0.2–0.3 ms; FIFO 0.02–0.14 ms.

**What it shows**

- **H1, as stated, is not supported on this task.** Learned controllers beat
  every recency heuristic by a wide margin (+0.24 to +0.49 over FIFO, all
  intervals far from zero) but do not beat the best hand-written heuristic.
  Imitation matches salience within the minimum detectable effect at 10% and
  25% and is a few points behind at 5%; imitation followed by PPO edges ahead
  at 25% (+0.018 [+0.014, +0.023] at horizon 500) and behind at 5%.
- The likely reason is a ceiling, not a training failure: which facts get
  asked about is drawn at random given the source type, so no policy without
  hindsight can do much better than "keep the specific, user-sourced items",
  which is what salience encodes and what the policies learn (shadow
  agreement 0.3–0.7 with salience, near zero with FIFO/LRU). The remaining
  oracle gap (0.34–0.46 at 5%) is mostly information the policy cannot have.
- Imitation of the hindsight expert is far more sample-efficient than PPO:
  240 episodes reach what 2,400 PPO episodes do not.
- **H2 (long-horizon reward beats immediate reward) is not supported here.**
  Over three seeds each, discount 0 and discount 0.995 are within seed noise
  (mean of three at horizon 500, 10%: 0.661 against 0.612; at horizon 200,
  5%: 0.381 against 0.437). In this task the immediate signal — "do not evict
  what the current query needs" — already identifies the kind of item worth
  keeping, so delayed credit adds little. The workflow task (5b) is the
  sharper test, and there both variants were unstable.
- PPO is noisy across seeds (spread of 0.13 at horizon 200, 5%) and drifts
  after its best point (`rl_ppo_last` below `rl_ppo`). Any PPO claim needs
  several seeds.
- Transfer to horizon 500 (2.5× the training horizon): imitation policies
  hold their position relative to salience; PPO policies lose 0.1 at 5%.
- The DeepSets context did not matter: the plain MLP is within a few points
  either way.

## 4b. Learned controllers with archive and retrieval

```bash
python -m memctl.rl.train --config configs/rl/rl_bc_archive.yaml
python -m memctl.rl.train --config configs/rl/rl_bc_ppo_archive.yaml
python -m memctl.sweep --config configs/sweeps/exp4b_rl_archive_eval.yaml
```

Horizon 200, MOVE_TO_ARCHIVE and RETRIEVE_FROM_ARCHIVE allowed. The policy's
retrieval head scores the 8 best BM25 candidates from the archive when a query
arrives. `rl_bc_delete_only_policy` is the Experiment 4 imitation policy run
here unchanged: its archive and retrieval outputs were never trained.

| controller | 2% | 5% | 10% | 25% |
|---|---|---|---|---|
| fifo_delete | 0.007 | 0.058 | 0.238 | 0.618 |
| salience_delete | 0.365 | 0.546 | 0.738 | 0.962 |
| salience_archive_retrieve | 0.744 | 0.833 | 0.898 | 0.984 |
| rl_bc_archive (imitation of the hindsight expert) | 0.340 | 0.528 | 0.731 | 0.969 |
| rl_bc_ppo_archive (imitation, then 1,344 PPO episodes) | 0.357 | 0.511 | 0.701 | 0.967 |
| rl_bc_delete_only_policy (untrained retrieval head) | 0.591 | 0.849 | 0.939 | 0.982 |
| oracle_approx | 0.966 | 1.000 | 1.000 | 1.000 |

At 5%: `rl_bc_archive` retrieves 1 item per episode (precision 0.74) and
fails 924 times by `evicted`; `rl_bc_ppo_archive` retrieves 5 (precision 0.54)
and fails 772 times by `evicted`; the untrained head retrieves 125 items per
episode (precision 0.11, recall 0.80) and fails only by
`archived_not_retrieved`; `salience_archive_retrieve` retrieves 54 (precision
0.15, recall 0.70).

**What it shows**

- **Imitating a hindsight expert is the wrong target when the expert knows
  what the learner cannot.** The expert deletes what it knows is never needed
  and archives the rest. The learner cannot tell the two apart, imitates the
  majority action (delete), and loses evidence it could have archived for
  free. Its failures are deletions, not retrieval misses. PPO on task reward
  did not undo the habit in 1,344 episodes: essentially no gain (clean rerun,
  `rl_bc_archive` → `rl_bc_ppo_archive`: 0.340 → 0.357 at 2%, 0.528 → 0.511
  at 5%).
- A policy that archives freely and retrieves liberally does better here
  than one that decides carefully, because the archive costs nothing and the
  retriever is decent: the untrained retrieval head, which retrieves most of
  its shortlist, is the best learned configuration and matches salience with
  retrieval. This is H4 restated: under uncertainty, archive.
- The right expert for this action set is not the hindsight-optimal one but a
  cost-aware one (archive when unsure; retrieve generously when the archive
  is free), or the learner needs the archive's asymmetry expressed in the
  reward. Both were tried later: the regret expert (D1) and a priced archive
  (D5).

## D1. The regret teacher (2026-10-05)

```bash
python -m memctl.rl.train --config configs/rl/d1/rl_bc_archive_{oracle,regret}_s{0,1,2}.yaml
python -m memctl.sweep --config configs/sweeps/d1_regret_expert_eval.yaml
python -m memctl.analysis.seeds runs/d1_regret_expert_eval
python -m memctl.rl.probe runs/rl_bc_archive_regret_s0/checkpoints/policy.pt
```

Same learner, data budget and DAgger schedule as Experiment 4b (`rl_bc_archive`);
only the expert differs. The **oracle** expert labels never-needed items
EVICT and needed ones MOVE_TO_ARCHIVE. The **regret** expert charges every
(item, operation) pair its hindsight regret with the oracle playing on (EVICT:
1 if the item is needed again, else 0; ARCHIVE: 0) and accepts every pair of
least regret, so for a never-needed item both EVICT and ARCHIVE are accepted
(`memctl/rl/expert.py`). Three training seeds each; evaluation on seeds 0–99
as in 4b. **Provenance: commit `5fd6a3a`, all 40 cells `dirty: false`.**

Task success, mean over the 3 training seeds [min, max]:

| controller | 2% | 5% | 10% | 25% |
|---|---|---|---|---|
| fifo_delete | 0.007 | 0.058 | 0.238 | 0.618 |
| fifo_archive_retrieve | 0.759 | 0.777 | 0.808 | 0.911 |
| salience_archive_retrieve | 0.744 | 0.833 | 0.898 | 0.984 |
| imitation, oracle expert | 0.335 [0.287, 0.376] | 0.512 [0.467, 0.541] | 0.722 [0.708, 0.731] | 0.958 [0.943, 0.969] |
| **imitation, regret expert** | **0.851** [0.843, 0.858] | **0.887** [0.870, 0.901] | **0.930** [0.915, 0.948] | **0.987** [0.983, 0.991] |
| oracle_approx | 0.966 | 1.000 | 1.000 | 1.000 |

Paired over episodes (seed-mean of the regret policies minus the baseline,
95% bootstrap interval): against the best heuristic at each budget, +0.092
[0.076, 0.107] at 2% (FIFO-archive), +0.053 [0.037, 0.070] at 5%, +0.031
[0.018, 0.045] at 10% (salience-archive), +0.003 [−0.003, 0.010] at 25%. The
gap to the oracle is −0.115, −0.113, −0.070 and −0.013.

**Mechanism, measured** (`memctl.rl.probe`, 10 evaluation episodes, 2%):

| policy | needed items deleted | never-needed items deleted | retrievals / episode | retrieval precision |
|---|---|---|---|---|
| Exp. 4b `rl_bc_archive` (oracle expert) | 92% | 99% | 0.8 | 0.88 |
| regret expert, seed 1 | 0% | 0% | 12.7 | 1.00 |

**What it shows**

- **The 4b failure is the teacher, not the learner.** The oracle-expert
  learner orders removals well (only 6% of what it removes is needed later)
  but chooses the *operation* as the expert's majority action, the same for
  needed and never-needed items: Weihs et al.'s policy averaging, measured.
  Accepting every least-regret action removes the privileged tie-break, and
  the same learner learns to archive.
- **Imitation now beats every fixed rule** at 2–10% and ties at 25%, with a
  quarter of the heuristics' retrievals (14 against 57 per episode).
- The Experiment 4 policy, whose archive outputs were never trained, scored
  0.591 at 2% in 4b, above the oracle-expert learner (0.340): imitating the
  oracle made the operation choice *worse than untrained*.

**What it does not show**

- The archive is free here, so "archive everything" is the right
  operation choice and the regret teacher says so. Whether a learner tracks
  a *priced* archive is D5 (complete).
- The remaining gap to the oracle (0.11 at 2%) is retrieval: the shortlist is
  BM25 top-8, and the learner retrieves 91% of the needed items it is shown.

## GRPO (2026-10-05)

```bash
python -m memctl.rl.train --config configs/rl/grpo/<name>_s{0,1,2}.yaml     # 18 configs
python -m memctl.sweep --config configs/sweeps/grpo_{recall,workflow}_eval.yaml
```

GRPO (`algorithm: grpo` in `memctl/rl/algorithms.py`): each training episode is
played 8 times with different action samples; an episode's advantage is its
return standardised within those 8, applied to all of its decisions with PPO's
clipped objective and no value network. Every training uses 1,584 episodes, the
budget of `rl_bc_ppo_archive`; 3 seeds; `policy_best.pt` (chosen on validation
seeds) evaluated on seeds 0–99. **Provenance: commit `16c6f5c`, 88 evaluation
cells `dirty: false`.**

Recall task with archive (task success, mean [min, max] over 3 seeds; needed
items deleted per episode at 2% in the last column):

| training | 2% | 5% | 10% | 25% | deleted needed, 2% |
|---|---|---|---|---|---|
| PPO from scratch | 0.717 [0.702, 0.736] | 0.899 | 0.922 | 0.940 [0.850, 0.988] | 0.0 |
| GRPO from scratch | 0.704 [0.616, 0.789] | 0.866 | 0.919 | 0.978 | 1.8 |
| imitate the **oracle** expert, then GRPO | 0.451 [0.347, 0.559] | 0.572 | 0.737 | 0.961 | **10.7** |
| imitate the **regret** expert, then GRPO | **0.859** [0.850, 0.864] | **0.906** | **0.944** | **0.992** | 0.0 |
| (D1) regret imitation alone | 0.851 | 0.887 | 0.930 | 0.987 | 0.0 |
| salience_archive_retrieve | 0.744 | 0.833 | 0.898 | 0.984 | – |

Workflow task, delete-only (validation success at 5% every 144 episodes, per seed):

| training | test 5% | test 10% | learning curve at 5% (validation) |
|---|---|---|---|
| GRPO from scratch | **0.999** [0.998, 1.000] | 1.000 | solved by 432 / 432 / 288 episodes in the three seeds |
| PPO from scratch | 0.741 [**0.227**, 1.000] | 0.999 | solved at ~1,000 episodes in two seeds; never in the third |
| fifo / oracle_approx_lazy | 0.028 / 0.896 | 0.221 / 1.000 | – |

**What it shows**

- **A warm start from the oracle poisons RL.** After imitating the oracle,
  1,440 episodes of GRPO leave the policy at 0.451 at 2%, against 0.70–0.72
  for RL from scratch with the same total budget; it still deletes 10.7 needed
  items per episode. The same holds for PPO: Experiment 4b's oracle-imitation
  + PPO policy scores 0.357 (clean rerun), PPO from scratch 0.717. This is
  Weihs et al.'s "IL warm start is strictly worse" (Poisoned Doors) measured in
  memory control, and it applies to the hindsight-SFT-then-GRPO recipe of
  Mem-T and ForesightKV.
- **The regret expert is the right warm start,** but RL adds little on top of
  it here (0.859 against 0.851): with a free archive, imitation already finds
  the policy RL would.
- **GRPO is the more reliable optimiser on the sequential task**: all three
  seeds solve it, three times faster than the PPO seeds that do, and none
  fails, where one PPO seed stays at 0.04 throughout. This is the instability
  seen in Experiment 5b (single PPO seeds at 0.992 with discount 0 and 0.404
  with discount 0.995), removed by comparing samples of the same episode
  instead of learning a value baseline.
- **On the recall task GRPO is no better than PPO** (0.704 against 0.717 at 2%)
  and its seeds spread more. Group-relative advantages help most where the
  episode's own luck is large (the workflow task's restarts), less where it is
  small.
- One GRPO workflow seed fell from 1.00 to 0.29 in its last validation;
  `policy_best.pt` avoids it, but late-training collapse is a known GRPO
  failure and worth a KL or learning-rate schedule if GRPO is used further.

## D2. One controller for both tasks (2026-10-05)

```bash
python -m memctl.rl.train --config configs/rl/d2/joint_{bc,cost}_regret_s{0,1,2}.yaml
python -m memctl.sweep --config configs/sweeps/d2_joint_recall_eval.yaml
python -m memctl.sweep --config configs/sweeps/d2_joint_workflow_eval.yaml
```

One policy, trained by DAgger on the regret expert with episodes alternating
between the recall task (archive and retrieval allowed) and the workflow task
(delete-only), `training.tasks` in `memctl/rl/train.py`. Twice the iterations
of a specialist, so each task gets the same number of training episodes as
its specialist. Specialists: the D1 regret policies (recall) and
`rl_bc_workflow` from the clean rerun (workflow). Three seeds each.
**Provenance: trainings at commit `9acfdb9`; all 100 evaluation cells
`dirty: false`.**

Recall task (task success, mean of 3 seeds [min, max]):

| controller | 2% | 5% | 10% | 25% |
|---|---|---|---|---|
| fifo_archive_retrieve | 0.759 | 0.777 | 0.808 | 0.911 |
| salience_archive_retrieve | 0.744 | 0.833 | 0.898 | 0.984 |
| recall specialist | 0.851 [0.843, 0.858] | 0.887 [0.870, 0.901] | 0.930 [0.915, 0.948] | 0.987 [0.983, 0.991] |
| **joint, `bc`** | **0.852** [0.845, 0.857] | **0.895** [0.887, 0.902] | **0.945** [0.936, 0.950] | **0.990** [0.988, 0.993] |
| joint, `cost` | 0.848 [0.839, 0.863] | 0.895 [0.882, 0.903] | 0.934 [0.930, 0.940] | 0.982 [0.975, 0.986] |
| oracle_approx | 0.966 | 1.000 | 1.000 | 1.000 |

Workflow task:

| controller | 5% | 10% | 20% | 40% |
|---|---|---|---|---|
| fifo | 0.028 | 0.221 | 0.891 | 1.000 |
| salience | 0.001 | 0.051 | 0.399 | 1.000 |
| recall specialist (never saw this task) | 0.000 | 0.001 | 0.008 | 0.100 |
| workflow specialist | 1.000 | 1.000 | 1.000 | 1.000 |
| **joint, `bc` and `cost`** (all 6 policies) | **1.000** | **1.000** | **1.000** | **1.000** |
| oracle_approx_lazy | 0.896 | 1.000 | 1.000 | 1.000 |

**What it shows**

- **One small controller (27,526 parameters) matches each specialist on its
  own task**, with no loss from sharing: on recall it is level with or
  slightly above the specialist; on the workflow task all six joint policies
  complete every job at every budget.
- **Every fixed rule is wrong on one of the two tasks**: salience is the best
  content rule on recall and the worst rule on the workflow task (0.001 at
  5%); FIFO is the best rule on the workflow task among heuristics and fails
  on recall when it deletes (0.007 at 2%). The recall specialist does not
  transfer (0.000 at 5%). The joint controller is the only controller,
  besides the oracle, that is not wrong on either; on the workflow task it
  also beats the oracle, whose hindsight goes blind after the first
  divergence (Experiment 5b).
- This is the thesis claim of `docs/archive/ONE_PAGE_RESEARCH_SUMMARY.md`
  ("unlike any fixed rule, is not wrong on both"), now measured.

**What it does not show**

- Both tasks are synthetic and the task model is scripted.

### D2b. The same action set on both tasks

In D2 the workflow task was delete-only, so the action set alone told the
policy which task it was in. D2b trains the joint policy with archive and
retrieval allowed on *both* tasks (`configs/rl/d2/joint_same_ops_s*.yaml`,
commit `103b4c0`, 88 evaluation cells `dirty: false`):

| controller | recall 2% | recall 5% | workflow 5% | workflow 10% | workflow 20% |
|---|---|---|---|---|---|
| **joint, same operations** | **0.859** [0.848, 0.866] | **0.896** | **1.000** (all seeds) | **1.000** | **1.000** |
| recall specialist | 0.851 | 0.887 | 0.226 [0.054, 0.438] | 0.592 | 0.999 |
| workflow specialist (delete-only) | – | – | 1.000 | 1.000 | 1.000 |
| fifo_archive_retrieve | 0.759 | 0.777 | 0.023 | 0.170 | 0.890 |
| salience_archive_retrieve | 0.744 | 0.833 | 0.009 | 0.084 | 0.577 |
| oracle_approx (archives) | 0.966 | 1.000 | 1.000 | 1.000 | 1.000 |

The result holds with the confound removed: one policy, one action set,
level with each specialist on its own task. Archiving does not rescue the
fixed rules on the workflow task (lexical retrieval does not bring back the
right token in time), while the recall specialist, now able to archive,
partly transfers (0.226 at 5%, 0.999 at 20%) where in D2 it scored 0.000.

## D5. A priced archive (2026-10-05)

```bash
python -m memctl.rl.train --config configs/rl/d5/<config>.yaml     # 14 configs
python -m memctl.sweep --config configs/sweeps/d5_priced_archive_eval.yaml
python -m memctl.analysis.priced runs/d5_priced_archive_eval --price <p>
```

Each item written to the archive costs *p* queries: an episode scores
(correct − p × archive writes) / queries (`memctl.analysis.priced`). The regret
expert carries the price (ARCHIVE costs p; EVICT costs 1 if the item is needed
again). Learners: accepted-set imitation (`bc`, as in D1) and cost-sensitive
imitation (`cost`: the policy's expected regret along the expert's removal
sequence, `memctl/rl/algorithms.py`), each trained at its own price, 2 seeds.
**Provenance: commit `5fd6a3a`, all 63 cells `dirty: false`.** Priced score at
2% budget, mean of 2 seeds (5% and 10% show the same pattern):

| price | `bc` (regret expert) | `cost` (regret expert) | `bc` (oracle expert) | salience archive | salience delete | oracle |
|---|---|---|---|---|---|---|
| 0 | **0.854** | 0.849 | 0.358 | 0.744 | 0.365 | 0.966 |
| 0.01 | 0.357 | **0.744** | 0.357 | 0.611 | 0.365 | 0.960 |
| 0.05 | 0.352 | 0.286 | 0.352 | 0.078 | **0.365** | 0.939 |
| 0.2 | 0.333 | −0.562 | 0.333 | −1.921 | **0.365** | 0.857 |

Archive writes per episode at 2%: `bc` with a price, 2; `cost`, 210 at 0.01
and 0.05, 121 at 0.2; heuristics, 247; oracle, 11.

**What it shows**

- **With any positive price the regret expert *is* the oracle expert**, and
  accepted-set imitation of it gives bit-identical policies (same numbers to
  three decimals in every cell). The oracle's labels are the regret labels
  of an infinitesimally priced archive; imitation cannot tell "cheap" from
  "free", and collapses to deleting everything (probe at 5%: 99% of needed
  items deleted).
- **Cost-sensitive imitation moves in the right direction** and wins clearly
  at a low price (0.744 against 0.611 for the best heuristic and 0.357 for
  `bc`). At 0.2 it starts to discriminate (probe at 5%: deletes 96% of
  never-needed items, 67% of needed ones).
- **It is miscalibrated at higher prices.** It keeps archiving about 200
  items per episode at 0.05, where deleting is right (it loses to
  delete-all), and archives too much at 0.2. A softmax expected-cost loss is
  minimised by a choice, not by a calibrated probability, and "archive
  anything" spreads its mass over many equally cheap pairs.

### D5b. The plug-in Bayes rule

```bash
python -m memctl.rl.needed --config configs/rl/d5/needed_s{0,1}.yaml
python -m memctl.sweep --config configs/sweeps/d5b_plugin_eval.yaml
```

The D1 regret policy (trained with a free archive) picks *which* item to
remove; a separate model gives P(needed again | features) for that item, and
the controller archives iff it exceeds the price (`archive_price`,
`needed_model`). The model is a 32-unit MLP fitted with log loss on 200 of the
policy's own episodes (about 37,000 removed items, base rate 4.5%) and
checked on 50 held-out episodes: Brier 0.034 against 0.043 for the base rate
(seed 0; 0.039 against 0.048 for seed 1), and calibrated per bin (predicted
0.065 / observed 0.061; 0.167 / 0.176; 0.41 / 0.36). 70% of removed items get
P < 0.01, and none of those was needed. **Provenance: commit `4d8e4d1`, all
18 cells and both models `dirty: false`;** cells pair with D5's on seeds 0–99.

Priced score, mean of 2 seeds:

| price | budget | plug-in | cost-sensitive | `bc` | salience archive | salience delete | oracle |
|---|---|---|---|---|---|---|---|
| 0.01 | 2% | **0.812** | 0.744 | 0.357 | 0.611 | 0.365 | 0.960 |
| 0.01 | 5% | **0.856** | 0.777 | 0.534 | 0.707 | 0.546 | 0.999 |
| 0.01 | 10% | **0.907** | 0.826 | 0.729 | 0.778 | 0.738 | 1.000 |
| 0.05 | 2% | **0.648** | 0.286 | 0.352 | 0.078 | 0.365 | 0.939 |
| 0.05 | 5% | **0.710** | 0.347 | 0.528 | 0.199 | 0.546 | 0.994 |
| 0.05 | 10% | **0.790** | 0.424 | 0.727 | 0.295 | 0.738 | 1.000 |
| 0.2 | 2% | 0.311 | −0.562 | 0.333 | −1.921 | **0.365** | 0.857 |
| 0.2 | 5% | 0.473 | −0.384 | 0.508 | −1.705 | **0.546** | 0.976 |
| 0.2 | 10% | 0.675 | −0.225 | 0.719 | −1.513 | **0.738** | 0.999 |

Paired against the best other non-oracle controller (95% bootstrap): at 0.01,
+0.068, +0.080, +0.081 (all intervals above +0.059); at 0.05, +0.284,
+0.163, +0.051 (all above +0.030); at 0.2, −0.053, −0.073, −0.063 (all below
−0.030).

**What it shows**

- **The information to price the archive is in the features; imitation
  losses do not extract it, a calibrated estimate does.** Separating "which
  item" (imitation) from "which operation" (a Bayes decision on a calibrated
  probability) wins by large margins where the trade-off is real (0.05).
- **At 0.2 it archives too eagerly.** The rule values a needed item at one
  query; its real value is lower (retrieval misses it ~9% of the time, and
  restatements make some items redundant). From D1/D5, archiving instead of
  deleting at 2% gains about 10.7 queries per episode for about 14.5 needed
  items removed, so about 0.74 query each. The fix is to estimate that value
  (or regress the counterfactual gain directly) instead of assuming 1.

**D5c: valuing a needed item at its measured worth does not fix 0.2.**
`need_value` (queries gained per needed item removed, archiving all against
deleting all, on 50 training seeds) came out at 0.67 and 0.74 for the two
seeds. With it (`configs/rl/d5/needed_v_s*.yaml`, sweep generated into
`runs/d5c_valued_plugin_eval.yaml`, commit `5b981b5`, 18 clean cells) the rule
is unchanged at 0.01 and 0.05 (differences within ±0.005, except −0.024 at
0.05/10%) and still 0.062–0.064 below salience-delete at 0.2. Ordering the
removals by expected cost min(P × value, price) instead of by the policy did
not help either (20-episode check, discarded). The remaining gap at 0.2 is
most likely the item *order*: the regret policy was trained where deletion
never happened, and its order is a worse deletion order than salience's
(the same policy deleting everything scores 0.333 at 2% against 0.365). At
high prices, where the archive is barely used, a deletion-trained order
should be used; a policy trained across prices would learn both.

**For the thesis**: the hindsight-teacher result now has three parts. The
oracle's argmin labels are the wrong target once an archive exists (D1);
least-regret *sets* fix a free archive but are indistinguishable from the
oracle at any positive price (D5); and the right object to learn from
hindsight is a calibrated expected regret, used in a decision rule (D5b),
which is what AggreVaTe-style cost-sensitive learning aims at but a softmax
expected-cost loss does not deliver.

## 8. A real language model as the task model (2026-10-05)

```bash
# Ollama on the local 4 GB GPU, served with OLLAMA_CONTEXT_LENGTH=16384 on port 11435
python -m memctl.sweep --config configs/sweeps/exp8a_llm_task_model.yaml --workers 1   # full context
python -m memctl.sweep --config configs/sweeps/exp8b_llm_task_model.yaml --workers 1   # under a budget
```

The scripted reader replaced by `qwen2.5:3b` (4-bit) through the `openai`
backend, on the recall task with archive and retrieval; 30 episodes (seeds
0–29); learned policies are seed 0 of D1. Generations cached in
`cache/generations_qwen3b_ctx16k`. The default Ollama context (4,096 tokens)
silently truncated full-context prompts in a first try (success 0.09); every
number here is with a 16,384-token context. **Provenance: commit `5fd6a3a`,
25 cells `dirty: false`.**

Task success with the LLM (evidence availability, i.e. what the scripted
reader would score, in the second table):

| controller | 2% | 5% | 10% | 25% |
|---|---|---|---|---|
| fifo_delete | 0.005 | 0.045 | 0.220 | 0.545 |
| fifo_archive_retrieve | 0.550 | 0.619 | 0.649 | 0.785 |
| salience_archive_retrieve | 0.551 | 0.605 | 0.599 | 0.801 |
| imitation, oracle expert | 0.277 | 0.445 | 0.520 | 0.799 |
| **imitation, regret expert** | **0.733** | **0.713** | **0.743** | **0.812** |
| oracle_approx | 0.886 | 0.864 | 0.836 | 0.839 |
| full context (no budget, all evidence present) | 0.723 | | | |

| evidence in memory at the question | 2% | 5% | 10% | 25% |
|---|---|---|---|---|
| salience_archive_retrieve | 0.756 | 0.832 | 0.915 | 0.984 |
| imitation, oracle expert | 0.331 | 0.538 | 0.718 | 0.968 |
| imitation, regret expert | 0.860 | 0.902 | 0.948 | 0.987 |
| oracle_approx | 0.966 | 1.000 | 1.000 | 1.000 |
| full context | 1.000 | | | |

Paired against full context (95% bootstrap over the 30 episodes): oracle at
2% +0.163 [+0.117, +0.212], at 10% +0.112 [+0.049, +0.174]; regret policy at
25% +0.089 [+0.045, +0.132], at 2% +0.010 [−0.047, +0.070]; salience-archive
at 25% +0.078 [+0.037, +0.121]; FIFO-archive at 25% +0.062 [+0.016, +0.109].

**What it shows**

- **For a real model, less context can be better context.** Full context has
  every answer in it and scores 0.723; the oracle's 2% memory scores 0.886,
  and every archiving controller at 25% beats full context. The scripted
  reader cannot show this: for it, full context is a perfect 1.000. Memory
  control here *raises* accuracy, it does not only save tokens.
- **The D1 result carries over.** With the LLM the regret-expert policy beats
  the oracle-expert one by 0.456 at 2% (0.733 against 0.277); the oracle-expert
  policy's failures are 423 `evicted`, the regret policy's none.
- **Accuracy given the evidence differs between controllers**, which the
  scripted reader hides: at 2% the LLM answers 92% of the questions whose
  evidence the oracle kept, 85% for the regret policy (its LLM failures
  include 52 `retrieved_but_ignored`: evidence fetched back from the archive
  that the model did not use). Which memory is easiest for a model to read is
  part of what a controller should optimise, and only an LLM in the loop
  measures it.
- With 30 episodes and one training seed per learned policy, these are
  first numbers; Experiment 9 repeats them at 100 episodes with a 7B model.

## 9. Experiment 8 with a 7B task model at 100 episodes (2026-10-06)

```bash
# vLLM v0.8.5 serving Qwen/Qwen2.5-7B-Instruct (bf16, 16,384-token context) on a rented
# RunPod A40; set base_url in the configs to your own server to rerun
python -m memctl.sweep --config configs/sweeps/exp9a_qwen7b_full_context.yaml --workers 8
python -m memctl.sweep --config configs/sweeps/exp9b_qwen7b_controllers.yaml --workers 16
```

Experiment 8 repeated with a stronger reader and more statistical power:
Qwen2.5-7B-Instruct (greedy) instead of `qwen2.5:3b`, 100 episodes (seeds
0–99) instead of 30, and all three training seeds of each D1 policy, plus the
D2 joint controller and two GRPO policies (seed 0). Same recall task, same
instructions, archive and retrieval allowed. Generations cached in
`cache/generations_qwen7b_vllm` (~106,000 calls). Compute: one A40 for
2.76 hours at $0.49/hour, about $1.35. **Provenance: commit `3d441b3`,
52 of 53 cells `dirty: false`; `oracle-approx` at 10% failed once on an empty
proxy response and was rerun at `e1acaaf` (adds HTTP retries, nothing else),
also clean.**

Task success; in brackets the paired difference from full context (0.898) with
a 95% bootstrap interval over the 100 episodes. Learned rows are the mean of
three training seeds.

| controller | 2% | 5% | 10% | 25% |
|---|---|---|---|---|
| fifo_delete | 0.007 | 0.058 | 0.236 | 0.588 |
| fifo_archive_retrieve | 0.741 | 0.761 | 0.787 | 0.857 |
| salience_archive_retrieve | 0.728 | 0.807 | 0.850 | 0.898 (+0.001 [−0.012, +0.012]) |
| imitation, oracle expert (3 seeds) | 0.320 | 0.485 | 0.678 | 0.880 |
| **imitation, regret expert (3 seeds)** | **0.832** (−0.066 [−0.081, −0.050]) | **0.861** | **0.884** (−0.013 [−0.028, +0.001]) | **0.908** (+0.011 [+0.000, +0.022]) |
| joint controller (D2, seed 0) | 0.841 | 0.865 | 0.892 | 0.911 |
| regret BC → GRPO (seed 0) | 0.833 | 0.860 | 0.890 | 0.906 |
| GRPO from scratch (seed 0) | 0.768 | 0.864 | 0.877 | 0.903 |
| oracle_approx | 0.937 (+0.039 [+0.022, +0.056]) | 0.955 (+0.057) | 0.948 (+0.050) | 0.947 (+0.050 [+0.035, +0.065]) |
| full context (no budget) | 0.898 | | | |

Paired at 2% (95% bootstrap): regret expert − oracle expert +0.512
[+0.492, +0.532]; regret policy − salience-archive +0.104 [+0.086, +0.123];
joint − specialist (seed 0) +0.004 [−0.008, +0.016]; GRPO fine-tuning − its
BC start −0.004 [−0.016, +0.008]; oracle − regret policy +0.105
[+0.088, +0.122].

**What it shows**

- **Every D-series result survives a real 7B reader.** The regret teacher beats
  the oracle teacher by 0.51 at 2% across three training seeds (the
  oracle-taught policies' failures are ~1,300 `evicted` per seed, the regret
  policies' none); the learned policy beats the best fixed rule (FIFO-archive, 0.741) by 0.09 at 2%;
  one joint controller matches the recall specialist; GRPO fine-tuning neither
  helps nor hurts a good imitation start.
- **"Less context can be better" holds, but is smaller with a stronger
  reader.** The 7B model reads full context far better than the 3B one (0.898
  against 0.723), so the oracle's edge over full context shrinks from +0.16 to
  +0.04–0.06, still with intervals that exclude 0. The learned policy at 25% of
  the history matches full context (+0.011 [+0.000, +0.022]) while the model
  reads 40% of the tokens (40,565 against 102,705 per episode); at 2% it gives
  up 0.066 for 4.6% of the tokens.
- **The remaining gap to the oracle is retrieval.** At 2% the regret policy's
  failures are mostly `archived_not_retrieved` (~300 per seed against 72 for
  the oracle): the evidence was kept in the archive but the retrieval step
  (the policy picks from a BM25 top-8 shortlist searched with the question
  text) did not bring it back. Better retrieval, not better eviction, is
  where the next 0.10 is.
- **Most of that gap is the second hop of two-hop questions** (breakdown of
  `failures.jsonl`, added 2026-10-06). Misses where the hop-1 fact was active
  and only the hop-2 fact was archived: regret BC 186 / 194 / 188 per seed
  (≈0.097 of 1,952 queries), regret BC → GRPO 209, joint 215,
  salience-archive 166, oracle-approx 9. The hop-2 fact names a bridge
  entity ("the priority of clerk-172") that the question ("the priority of
  the courier of sensor-171") never mentions, so the question-text shortlist
  rarely contains it, and the regret expert labels retrievals only inside
  that shortlist. GRPO retrieved more (17.5 against 14.3 items per episode)
  at lower precision, cutting both-missing cases but not hop-2-only ones.
  Next steps: `docs/archive/next_directions.html`.

## 10. Diagnostics and the follow-the-clue search (2026-10-06)

```bash
# diagnostic 1: where the needed evidence sits when the controller decides (scripted reader, no LLM)
python -m memctl.rl.shortlist --config configs/sweeps/exp9b_qwen7b_controllers.yaml --label bc_regret_s0 --out runs/diag1_shortlist
# diagnostics 2-3: GRPO rerun with per-group logging, and best-of-8 filtered imitation
python -m memctl.rl.train --config configs/rl/diag/diag2_grpo_audit_s0.yaml      # and _s1, _s2
python -m memctl.rl.train --config configs/rl/diag/diag3_bestof8_s0.yaml         # and _s1, _s2
python -m memctl.sweep --config configs/sweeps/diag_grpo_eval.yaml
# follow-the-clue search: regret imitation with bridge_search on
python -m memctl.rl.train --config configs/rl/bridge/rl_bc_bridge_regret_s0.yaml # and _s1, _s2
python -m memctl.sweep --config configs/sweeps/bridge_eval.yaml
```

Follows the plan in `docs/archive/next_directions.html`. Everything here uses
the scripted reader on the recall task (seeds 0–99), with archive and
retrieval allowed. **Provenance: diagnostics 2–3 at `c76b5cd`, the
follow-the-clue runs at `5b4ceea`, diagnostic 1 at `831be0f`; every run
`dirty: false`.**

**Diagnostic 1: the second hop is half out of reach, half refused.** At every
query, the rank of each needed fact in a lexical search of the whole archive
with the question text. Misses where hop 1 was in view and hop 2 was archived
and not retrieved, regret imitation at 2% (3 seeds):

| rank of the hop-2 fact | 1–3 | 4–8 | 9–20 | >20 |
|---|---|---|---|---|
| misses (s0 / s1 / s2) | 22 / 24 / 23 | 80 / 84 / 80 | 54 / 55 / 56 | 39 / 37 / 36 |

About 54% of these hop-2 facts are on the 8-item shortlist and the policy
declines them (2 of 114 shortlisted hop-2 facts retrieved); 46% are beyond
it. Hop-1 and one-hop facts on the shortlist are retrieved 97–99% of the
time. The oracle's archive holds 4 items on average at 2% (it deletes what is
never needed); the regret policy's holds 116, because a free archive makes
archiving everything least-regret.

**Diagnostic 2: GRPO's signal was real.** Over 540 GRPO groups (3 seeds × 180),
84% differ in which questions were answered, 1% only in forced-fallback
penalties, and 15% are fully tied (mostly at the 25% budget). The median
untied group spans 0.09 task success. The std-scaling noise hypothesis of
the next-directions report is rejected for this reward.

**Diagnostic 3: GRPO adds no more than best-of-8 filtered imitation.** Task
success, mean of 3 training seeds; in brackets the paired difference from
regret imitation (D1):

| controller | 2% | 5% | 10% | 25% |
|---|---|---|---|---|
| regret imitation (D1) | 0.851 | 0.887 | 0.930 | 0.987 |
| imitation then GRPO (rerun, `diag2`) | 0.859 (+0.008 [+0.001, +0.015]) | 0.906 (+0.019) | 0.944 (+0.014) | 0.992 (+0.005) |
| imitation then best-of-8 (`diag3`) | 0.861 (+0.010 [+0.004, +0.017]) | 0.896 (+0.009) | 0.944 (+0.014) | 0.988 (+0.001) |
| oracle_approx | 0.966 (+0.115) | 1.000 | 1.000 | 1.000 |

GRPO − best-of-8: −0.002 [−0.008, +0.003] at 2%, +0.010 [+0.003, +0.017] at
5%. Both add about one point over imitation.

**The follow-the-clue search.** `bridge_search` (`memctl/retrieval.py`) takes
the best matches to the question that contain its rarest word (what the
question names), and searches the archive again with the question's leftover
words plus each seed's three rarest new words (the bridge entity). Up to 4
items it finds join the 8-item shortlist, marked by a `bridge_score` feature
(feature version 2; version 1 checkpoints load unchanged). It knows nothing of
the task's sentence forms. Of the regret policy's 195 hop-2 misses (s0), the
question search has 102 in its top 8; the bridge search has 184 in its top 4.

| controller | 2% | 5% | 10% | 25% |
|---|---|---|---|---|
| fifo_archive_retrieve | 0.759 | 0.777 | 0.808 | 0.911 |
| regret imitation (D1) | 0.851 | 0.887 | 0.930 | 0.987 |
| **regret imitation + follow-the-clue** | **0.917** (+0.067 [+0.054, +0.079]) | **0.958** (+0.071 [+0.060, +0.083]) | **0.970** (+0.040 [+0.032, +0.048]) | **0.992** (+0.005 [+0.002, +0.009]) |
| oracle_approx | 0.966 | 1.000 | 1.000 | 1.000 |

Mean of 3 training seeds each; paired difference from D1 in brackets. The gap
to the oracle at 2% falls from 0.115 to 0.048 [0.041, 0.057]. Hop-2 misses
fall from ~195 to 23 / 22 / 54 per seed; retrieval recall rises from 0.83 to
0.90 at the same precision (0.91).

**GRPO on top of the follow-the-clue policy** (`configs/rl/grpo2/`, commit
`831be0f`, all `dirty: false`). Four variants, each starting from the three
follow-the-clue imitation policies, at matched experience (1,440 GRPO
episodes): plain GRPO (group std, 24 episodes × 60 iterations); batch-level
scaling computed per budget (`advantage: batch, stratify: true`, 96 × 15);
the same plus per-question credit (`question_weight: 1`: every sample of a
group meets the same question at the same step, so a decision also gets its
episode's reward at that step minus the group's mean); and the same plus the
regret expert's imitation loss (`imitation_weight: 0.5`).

| controller | 2% | 5% | 10% | 25% |
|---|---|---|---|---|
| follow-the-clue imitation | 0.917 | 0.958 | 0.970 | 0.992 |
| + plain GRPO | 0.928 (+0.011 [+0.004, +0.018]) | 0.976 (+0.017) | 0.983 (+0.013) | 0.995 |
| + GRPO, batch scale per budget | 0.930 (+0.013 [+0.008, +0.018]) | 0.971 (+0.013) | 0.982 (+0.012) | 0.996 |
| + per-question credit | 0.932 (+0.014 [+0.009, +0.020]) | 0.972 (+0.014) | 0.983 (+0.013) | 0.996 |
| + expert loss | 0.929 (+0.012 [+0.007, +0.016]) | 0.969 (+0.011) | 0.982 (+0.012) | 0.996 |
| oracle_approx | 0.966 | 1.000 | 1.000 | 1.000 |

Against plain GRPO, no variant differs at 2% (largest +0.003 [−0.001,
+0.008], per-question credit) and all are 0.004–0.007 lower at 5%. GRPO now
adds 1.1–1.4 points at 2% where it added 0.8 on the old shortlist, and the
gap to the oracle at 2% is 0.034–0.038. The GRPO changes recommended by the
next-directions report (batch scaling, budget strata, per-question credit,
an imitation term) make no measurable difference at this scale.

**What it shows**

- The remaining gap of Experiment 9 was a retrieval problem, as the failure
  breakdown said, and a generic multi-hop search closes 58% of it at 2%.
- On the old shortlist RL had nothing to find: GRPO's groups differed in real
  answers, but the decisive fact was rarely among the choices. RL and
  best-of-8 imitation each added about a point. With the fact in reach GRPO
  adds slightly more (1.1–1.4 points at 2%), and how its advantages are
  computed does not matter.

**What it does not show**

- These are scripted-reader numbers. Whether the gain holds with Qwen2.5-7B
  needs a rented GPU (Experiment 9's setup).
- Remaining failures at 2% (~165 per seed against the oracle's 72) are half
  one-hop. Facts ranked beyond 8 by the question search account for only
  ~27 of them, so the hindsight reranker of the next-directions plan has a
  ceiling of about 0.014 and is deferred.

## 11. The follow-the-clue policy with a 7B reader, and LoCoMo and LongMemEval question answering (2026-10-06)

```bash
# vLLM v0.8.5, Qwen2.5-7B-Instruct bf16 served as qwen2.5-7b-instruct, max len 32,768, RunPod A40
python -m memctl.sweep --config configs/sweeps/exp11a_qwen7b_bridge_recall.yaml --workers 8
python -m memctl.sweep --config configs/sweeps/exp11b_locomo_qa.yaml --workers 11
python -m memctl.sweep --config configs/sweeps/exp11c_longmemeval_qa.yaml --workers 5
python -m memctl.rl.train --config configs/rl/lme/lme_bc_regret_s0.yaml        # and the other five
python -m memctl.sweep --config configs/sweeps/exp11d_longmemeval_trained.yaml --workers 6
# same server restarted with --max-model-len 65536 and YaRN (factor 2), separate cache
python -m memctl.sweep --config configs/sweeps/exp11b_locomo_full_context.yaml --workers 1
```

One A40 for 6.07 hours at $0.49/hour, about $2.97. **Provenance: 11a–11c at
`2d8fbf4` (which makes the generation cache's writes atomic; a first launch
at `4549fdc` failed on half-written cache entries and was discarded),
LongMemEval training at `a5f90b4`, 11d and the LoCoMo full-context run at
later commits that change only documents; every cell `dirty: false`.** QA
answers are judged by the same Qwen2.5-7B (LongMemEval's official metric is
an LLM judge); a model judging its own answers may be lenient, so token F1
is reported too. LoCoMo's adversarial category is counted (the right answer
is a refusal).

**11a: the follow-the-clue gain holds with a 7B reader.** Recall task,
Experiment 9's seeds and prompt; cells shared with Experiment 9 reproduce it
from the cache. Mean of 3 training seeds; paired difference from regret
imitation in brackets.

| controller | 2% | 5% | 10% | 25% |
|---|---|---|---|---|
| fifo_archive_retrieve | 0.741 | 0.761 | 0.787 | 0.857 |
| regret imitation (D1) | 0.832 | 0.861 | 0.884 | 0.908 |
| **+ follow-the-clue search** | **0.889** (+0.057 [+0.045, +0.069]) | **0.913** (+0.051) | **0.917** (+0.033) | 0.910 (+0.002) |
| oracle_approx | 0.938 | 0.955 | 0.949 | 0.948 |
| full context (Experiment 9) | 0.898 | | | |

The gap to the oracle at 2% falls from 0.106 to 0.049, as with the scripted
reader (0.115 to 0.048).

**11b–11d: on real conversations the learned policies lose to simple
rules.** Judge accuracy / token F1, query-weighted. LoCoMo: all 10
conversations (1,986 questions, *including* the 446 adversarial ones; see
11e for the conventional scoring). LongMemEval-S: questions 0–99, about 115k
history tokens each. **These are single-session-user and multi-session
questions only** (70 + 30, 6 abstention): the file is sorted by question
type, so questions 0–99 hold none of the temporal-reasoning,
knowledge-update, single-session-assistant or preference questions. The
LongMemEval columns below describe the easiest fifth of the benchmark, not
LongMemEval as a whole; 11e gives the split that fixes this.

| controller | LoCoMo 10% | LoCoMo 25% | LongMemEval q0–99 5% | LongMemEval q0–99 10% |
|---|---|---|---|---|
| full context (64k YaRN server) | 0.510 / 0.312 | | not run (too long) | |
| fifo_archive_retrieve (top 5) | 0.382 / 0.397 | 0.400 / 0.382 | 0.470 / 0.550 | 0.420 / 0.410 |
| salience_archive_retrieve (top 5) | 0.375 / 0.388 | 0.410 / 0.395 | 0.360 / 0.398 | 0.290 / 0.294 |
| keep last 4 + retrieve top 5 | 0.339 / 0.343 | 0.339 / 0.343 | 0.530 / 0.595 | 0.530 / 0.595 |
| regret imitation, synthetic-trained (3 seeds) | 0.285 / 0.294 | 0.331 / 0.336 | 0.240 / 0.255 | 0.240 / 0.259 |
| + follow-the-clue, synthetic-trained (3 seeds) | 0.289 / 0.301 | 0.326 / 0.328 | 0.300 / 0.336 | 0.273 / 0.297 |
| regret imitation, trained on LongMemEval 100–499 (3 seeds) | | | 0.183 / 0.204 | 0.173 / 0.191 |
| + follow-the-clue, trained on LongMemEval 100–499 (3 seeds) | | | 0.227 / 0.247 | 0.270 / 0.269 |
| oracle_approx | 0.505 / 0.520 | 0.501 / 0.512 | 0.650 / 0.654 | 0.650 / 0.654 |

**Why** (retrieval statistics, 10% budget). The heuristics retrieve their
top 5 at every question (LongMemEval recall of needed items 0.65–0.66,
LoCoMo 0.35). The synthetic-trained policies retrieve almost nothing on real
text: 0–1 items per LongMemEval question and 2–70 per LoCoMo conversation of
~199 questions, recall 0.00–0.03; their failures are mostly
`archived_not_retrieved`. Their retrieval head learned when to fire on
templated facts, and conversational turns score differently. The
LongMemEval-trained policies never retrieve (0 per question). The cause is
label *starvation*, not imbalance: the shortlist is built only at a question
step, so a LongMemEval episode has exactly one retrieval decision (at most 8
labels) against about 210 removal decisions (8,438 decisions in 40 episodes
of `lme_bc_regret_s0`). The head barely trains, and at evaluation the rule
is deterministic (retrieve iff logit > 0), so a head whose logits sit below
0 never fires. The removal labels carry no signal either: with one question
nearly every turn is never needed, and the regret teacher gives EVICT and
ARCHIVE the same regret (0) for those, so expert agreement is 0.99999 and
the loss 7e-5 from the first iteration while the EVICT/ARCHIVE choice is a
coin flip. Two of the three seeds land on deleting, and fail mostly by
`evicted`. (Found by the peer review of 2026-10-06.)

**What it shows**

- The follow-the-clue result (Experiment 10) holds with a 7B reader.
- On LoCoMo and LongMemEval, memory management matters: the hindsight
  oracle beats the best simple rule by 0.10–0.12. The learned controller
  does not capture it. Trained on synthetic recall it does not transfer;
  trained on LongMemEval as set up here it learns not to retrieve.
- On LoCoMo, full context (0.510) matches the oracle (0.505) by the judge,
  but its F1 is far lower (0.312 against 0.520): with the whole
  conversation in view the model answers at length, which the judge
  accepts and F1 does not.

**What it does not show**

- Whether a learned controller can help on these benchmarks. The obvious
  next tests: let a fixed top-k retrieval run beside the learned removal
  policy; reweight the retrieval labels (or train on LoCoMo-style episodes
  with many questions); price deletion of an unseen-future item instead of
  treating it as free when only one question follows.
- Full context on LongMemEval-S (about 115k tokens per question) was not
  run; published results report it.

### 11e. Re-scoring Experiment 11 by the field's conventions (2026-10-06)

```bash
python -m memctl.analysis.qa_tables runs/exp11b_locomo_qa runs/exp11b_locomo_full_context \
  --exclude full_context__fraction0.1 full_context__fraction0.25 --baseline fifo_archive_retrieve --per-category
python -m memctl.analysis.qa_tables runs/exp11c_longmemeval_qa runs/exp11d_longmemeval_trained \
  --baseline fifo_archive_retrieve --per-category
```

No new runs: verdicts are rebuilt per question from `failures.jsonl`
(every episode). Headline accuracy leaves out the refusal-scored questions
(LoCoMo category 5, adversarial; LongMemEval `_abs`), which Mem0, Zep and
Memory-R1 also leave out: they are scored by `is_refusal`, so a controller
that keeps less and refuses more gains on them. For example, in conversation
0 the learned `bc_bridge_s0` scores 0.957 on adversarial questions and
0.00–0.10 on every other category. Intervals resample clusters
(conversations for LoCoMo, questions for LongMemEval); the Δ column is
paired with FIFO on the same questions, seeds averaged within a cluster.
F1 is only available for the episode logged in detail (conversation 0 /
question 0), so it is not a headline here; future sweeps should log every
episode in detail. With only 10 LoCoMo conversations the cluster bootstrap
undercovers; the tool also prints a question-level paired interval, which
ignores within-conversation dependence. Here both exclude 0 wherever one
does. Prompt tokens (reader prompt per question, `tokens_processed /
task_model_calls`, every episode) count the speaker, date and id label on
each memory line, about 1.9 times the memory's own tokens on LoCoMo. At 10%:
oracle 2,196, FIFO 3,388, learned about 3,050, keep-last-4 272, full context
32,506. On LongMemEval at 5%: oracle 236, keep-last-4 2,326, FIFO 5,305.
Keep-last-4 beats FIFO there with less than half the prompt.

LoCoMo, categories 1–4 (1,540 questions):

| controller | 10% | Δ vs FIFO | 25% | Δ vs FIFO | category 5 (refusal-scored), 10% |
|---|---|---|---|---|---|
| full context (64k YaRN) | 0.462 (0.425–0.495) | | | | 0.675 |
| oracle_approx | 0.384 (0.346–0.414) | +0.160 (+0.132, +0.182) | 0.378 | +0.121 | 0.922 |
| fifo_archive_retrieve | 0.224 (0.198–0.251) | – | 0.257 | – | 0.926 |
| salience_archive_retrieve | 0.218 | −0.006 (−0.025, +0.009) | 0.281 | +0.023 (+0.001, +0.045) | 0.917 |
| keep last 4 + retrieve top 5 | 0.151 | −0.073 | 0.151 | −0.106 | 0.989 |
| regret imitation (3 seeds) | 0.086 | −0.138 (−0.165, −0.111) | 0.161 | −0.096 | 0.972 |
| + follow-the-clue (3 seeds) | 0.094 | −0.130 (−0.157, −0.105) | 0.163 | −0.095 | 0.962 |

Per category at 10% (multi-hop / open-domain / single-hop / temporal):
full context 0.305 / 0.062 / 0.674 / 0.162; oracle 0.280 / 0.052 / 0.572 /
0.081; FIFO 0.078 / 0.021 / 0.360 / 0.056; learned 0.07–0.08 / 0.02 /
0.12–0.13 / 0.03. Temporal accuracy is low for every controller, full context
included, although each memory line carries its session date: the reader
answers in relative terms ("yesterday") where the gold answer is a date. That
is a reader-prompt issue, not a memory one.

LongMemEval-S questions 0–99 without abstention (94 questions; two types
only, see above):

| controller | 5% | Δ vs FIFO | 10% | Δ vs FIFO |
|---|---|---|---|---|
| oracle_approx | 0.628 (0.521–0.723) | +0.191 (+0.106, +0.277) | 0.628 | +0.245 |
| keep last 4 + retrieve top 5 | 0.500 | +0.064 (+0.021, +0.117) | 0.500 | +0.117 |
| fifo_archive_retrieve | 0.436 (0.340–0.532) | – | 0.383 | – |
| salience_archive_retrieve | 0.319 | −0.117 | 0.245 | −0.138 |
| + follow-the-clue, synthetic (3 seeds) | 0.255 | −0.181 (−0.270, −0.096) | 0.227 | −0.156 |
| regret imitation, synthetic (3 seeds) | 0.191 | −0.245 | 0.191 | −0.191 |
| + follow-the-clue, LongMemEval 100–499 (3 seeds) | 0.177 | −0.259 | 0.223 | −0.160 |
| regret imitation, LongMemEval 100–499 (3 seeds) | 0.131 | −0.305 | 0.121 | −0.262 |

Multi-session questions are near 0 for every controller (0.00–0.07), the
oracle included; single-session-user carries all the differences.

**What it shows**

- The conventional scoring makes every LoCoMo number lower and the gaps
  larger. Counting category 5 had flattered the controllers that keep the
  least: the learned ones and keep-last-4 score 0.96–0.99 there.
- The conclusions of 11b–11d stand, with intervals: on both benchmarks the
  learned controllers trail FIFO by 0.10–0.30, and the paired intervals
  exclude 0.
- On LoCoMo the oracle now trails full context (0.384 against 0.462). Three
  explanations, not yet separated: keeping only annotated evidence misses
  context the reader uses; the evidence annotations are incomplete (an audit
  found 6.4% of LoCoMo answers wrong, and evidence lists were not audited);
  and at 10% the oracle cannot hold all evidence, which is 31% of tokens.
- The reader prompt caps two LongMemEval types for every controller. 28 of
  the oracle's 30 multi-session failures at 5% are `task_model_reasoning`
  (evidence in view, answer wrong), mostly "unknown" where the gold is a
  count or a duration. Temporal questions get relative answers
  ("yesterday"). A short-phrase prompt with 32 output tokens leaves no room
  to count or do date arithmetic. `agent.reasoning: true` (a brief note,
  then `Answer: ...`, memory in arrival order) is the fix to verify on the
  oracle cell before controllers are re-run; it uses a fresh cache, so the
  rows above stay reproducible.
- LongMemEval evidence labels: 41 of 500 instances mark `has_answer` turns
  in only some of their answer sessions, and 21 mark none. The environment
  now adds one requirement per unmarked answer session (any of its turns),
  where before it fell back to the answer sessions only when nothing was
  marked. The 11c/11d numbers were scored under the old rule.
- The judge is strict, which brings false rejects as well as false accepts.
  In one oracle failure the gold is "I have worked on or bought five model
  kits..." and the answer "5" was judged wrong. The planned second-judge
  check should measure both directions.

**Next (evaluation protocol).** `env.folds` (stratified 5-fold over all 500
LongMemEval questions, by question type with abstention apart) replaces the
contiguous `subset`; all LongMemEval numbers above should be re-run on it.

## 12. Retrieval headroom on LoCoMo and LongMemEval (2026-10-06)

```bash
python -m memctl.rl.headroom --dataset locomo --dense --out runs/t2_headroom/locomo.json      # ~2 min CPU
python -m memctl.rl.headroom --dataset longmemeval --out runs/t2_headroom/longmemeval.json    # ~3 min CPU
```

Clean run at commit 1d8a93e (`runs/t2_headroom/*.json`). No controller and
no reader. Every turn of the history is a candidate, the question is the
query, and a requirement (one evidence turn, or for an unmarked LongMemEval
answer session any of its turns) counts as found when one of its items is in
the top k. Two rules per question: the *share* of its requirements found,
and whether *all* of them are found. The all-found rule bounds a controller
on questions that need several pieces of evidence (LoCoMo multi-hop,
LongMemEval multi-session). The share found bounds what a retrieval floor of k, or
a pick-k-of-N head over a shortlist of N, can reach. Questions with evidence
labels only, refusal-scored ones left out: LoCoMo 1,535, LongMemEval-S 470.
`bm25_label` prefixes "speaker (date):" to each turn; `bm25_bridge` adds the
follow-the-clue search's 4 items to the top k; `dense` is bge-small-en-v1.5
on labelled text; `fusion` is reciprocal rank fusion (k = 60) of
`bm25_label` and `dense`.

Requirements found in the top k, all questions:

| search | LoCoMo @5 | @8 | @12 | @16 | @20 | LongMemEval @5 | @8 | @12 | @16 | @20 |
|---|---|---|---|---|---|---|---|---|---|---|
| bm25 | 0.468 | 0.513 | 0.552 | 0.584 | 0.608 | 0.644 | 0.720 | 0.774 | 0.806 | 0.822 |
| bm25_label | 0.519 | 0.564 | 0.609 | 0.637 | 0.654 | 0.643 | 0.713 | 0.761 | 0.788 | 0.805 |
| bm25_bridge (k + 4 items) | 0.482 | 0.525 | 0.563 | 0.593 | 0.617 | 0.655 | 0.724 | 0.776 | 0.807 | 0.823 |
| dense (bge-small) | 0.448 | 0.518 | 0.576 | 0.613 | 0.640 | | | | | |
| **fusion** | **0.530** | **0.610** | **0.674** | **0.707** | **0.739** | | | | | |

All requirements found (BM25 / fusion on LoCoMo, BM25 on LongMemEval):

| | LoCoMo @5 | @16 | LongMemEval @5 | @16 |
|---|---|---|---|---|
| all questions | 0.432 / 0.481 | 0.533 / 0.646 | 0.517 | 0.706 |
| LoCoMo multi-hop | 0.036 / 0.085 | 0.114 / 0.209 | | |
| LongMemEval multi-session | | | 0.182 | 0.446 |
| LongMemEval temporal-reasoning | | | 0.496 | 0.709 |

*Footnote (2026-10-07, §19a).* The fusion figures above were computed by a copy of reciprocal rank fusion inside
`headroom.py` that cut BM25 at the largest k tabulated (20), while the shared `FusionRetriever` ranks both lists to
depth 50. They are inflated by about 0.015–0.026. With the shared retriever (1e0a6a4), LoCoMo fusion all-found is
0.465 @5 and 0.620 @16 (share found 0.514 and 0.681). Every controller run (§13 onward) used the shared retriever
and is unaffected. The BM25 and dense rows are unchanged.

Share found by type, BM25 @5 → @16: LoCoMo multi-hop 0.142 → 0.289 (fusion 0.253 →
0.428), open-domain 0.224 → 0.276, single-hop 0.562 → 0.679, temporal 0.578
→ 0.682. LongMemEval knowledge-update 0.787 → 0.926, multi-session 0.435 →
0.659, single-session-assistant 0.875 → 0.929, preference 0.339 → 0.561,
single-session-user 0.836 → 0.945, temporal 0.635 → 0.810.

**What it shows**

- There is room above a top-5 floor on both benchmarks. Recall@16 minus
  recall@5 is 0.116 on LoCoMo (BM25; 0.177 with fusion) and 0.162 on
  LongMemEval. A pick-5-of-16 head trained on evidence labels has up to that
  much to gain, so the review's gate for building it (≥ 0.10) passes on both.
- On LoCoMo, how turns are indexed matters. Speaker and date in the indexed
  text add 0.051 at @5, and fusing with bge-small adds 0.062 at @5 and 0.097
  at @8. Dense search alone is worse than BM25 at @5, which matches §6c,
  where bge replaced BM25. The gate for better units (≥ 0.05 at @5) passes on
  LoCoMo. On LongMemEval labels add nothing (the question date already
  carries the time), and BM25 is near its ceiling for single-session types.
- Follow-the-clue adds little on real text (+0.01): its rarest-word anchor
  was built for templated identifiers.
- Multi-hop (LoCoMo) and multi-session and preference (LongMemEval) are the
  hard cases for search itself: under half their evidence is in the top 5.
  By the all-found rule it is worse: a LoCoMo multi-hop question has all its
  evidence in the BM25 top 16 4% → 11% of the time, and a LongMemEval
  multi-session question 18% (top 5) → 45% (top 16).
- On LongMemEval, speaker and date labels slightly *lower* BM25 recall
  (0.806 → 0.788 at @16), so the shared LongMemEval search stays plain BM25;
  on LoCoMo it is labelled BM25 fused with bge-small. Dense search was not
  measured on LongMemEval (about 250k turn embeddings on CPU).

## 13. LongMemEval and LoCoMo under the reviewed protocol (2026-10-07)

Sweeps: `configs/sweeps/exp13/` (generated by `make.py`). Reader frozen at
7b5fc30 for the whole experiment: Qwen2.5-7B-Instruct, reasoning note then
`Answer:`, compact labels, memory in arrival order, 256 output tokens,
LongMemEval's official judge prompts (`judge.style: longmemeval`), LoCoMo
categories 1–4. Shared search per benchmark: plain BM25 on LongMemEval,
labelled BM25 fused with bge-small on LoCoMo (Experiment 12). No EVICT for any
controller. Stratified 5-fold LongMemEval, every controller on all 500
questions (learned ones on the test part of the fold they were not trained on).

**Pre-registered evidence gate (written before any learned cell is run).**
Learned controllers go to the GPU only through this gate. With no reader
(`agent: null`, `exp13_gate_f*`, free), each candidate (single-question
trained `learned_floor5`, composed-episode trained `compose_floor5`,
synthetic-trained `synthetic_floor5`) and `fifo_top5` run on every test fold
at 1/2/5% and at targets of 2k/3k/4k tokens (2% and 5% only). The measure is
evidence in view at the question (`needed_hit_rate`), paired with FIFO over
the 500 questions. A (candidate, budget, target) cell is a **go** when the
pooled paired 95% bootstrap interval of (candidate − FIFO) lies entirely
above 0. Go cells are run with the reader; no-go cells are not, and are
reported only in the evidence table. A controller that keeps no more
evidence than FIFO cannot be expected to answer better, so this saves GPU
time without choosing on the test answers.
The gate table also gives mean active memory (tokens) for each pair; the pairs
are matched by construction (same budget or target), and a pair more than 10%
apart is flagged but still decided by the rule.

**Pre-registered accuracy criterion for the learned rows (written before they
run).** A learned controller *wins* in a (budget, target) cell when its judge
accuracy (categories 1–4 on LoCoMo, abstention left out on LongMemEval) minus
FIFO's at the same budget and target, paired over the same questions and
pooled over the five folds, has a 95% bootstrap interval entirely above 0.
Per-type differences are secondary and reported without a decision.

**Gate result (LongMemEval, 2026-10-07; agent null, 500 questions pooled over
the five test folds; gate code 8e1b1c5, checkpoints 2567c3e chosen by
validation evidence).** Evidence in view at the question, candidate − FIFO
with the same floor at the same budget or target, paired 95% interval.

| candidate | budget, target | FIFO | candidate | difference | memory FIFO / candidate | |
|---|---|---|---|---|---|---|
| composed-episode trained | 5%, fill | 0.638 | 0.771 | +0.133 (+0.102, +0.163) | 4,784 / 4,863 | go |
| composed-episode trained | 5%, 4k | 0.636 | 0.713 | +0.076 (+0.045, +0.106) | 3,699 / 3,798 | go |
| composed-episode trained | 5%, 3k | 0.633 | 0.670 | +0.037 (+0.009, +0.064) | 2,733 / 2,850 | go |
| single-question trained | 5%, fill | 0.638 | 0.732 | +0.094 (+0.064, +0.124) | 4,784 / 4,908 | go |
| single-question trained | 5%, 4k | 0.636 | 0.690 | +0.054 (+0.024, +0.082) | 3,699 / 3,837 | go |
| composed-episode trained | 1–2%, any | 0.59–0.63 | 0.59–0.63 | −0.003 to +0.005 | | no |
| single-question trained | 1–2%, any; 5% 2k | 0.59–0.64 | 0.58–0.64 | −0.030 to +0.005 | | no |
| synthetic-trained | any | 0.59–0.64 | 0.52–0.64 | −0.09 to +0.001 | | no |

The full table (36 rows) is `runs/exp13_gate_table.md`. Targets at or above the
budget equal the fill row. The one memory flag (>10% apart) is on every 1%
row, where the learned floor holds about 975 tokens to FIFO's 809; all are
no-go.

- At 1–2% the retrieval floor does all the work: neither trained policy keeps
  more evidence than FIFO with the same floor.
- The synthetic-trained policy loses 0.05–0.09 evidence almost everywhere:
  its eviction features do not transfer to real conversations even with the
  floor (RQ4).
- Training on composed four-question episodes beats single-question training
  in every 5% row (+0.133 against +0.094 at fill): more questions per episode
  give the imitation teacher signal it lacked (Experiment 11, T4).

**Written before the paid learned cells ran.** The go rows sit at 2.7–4.9k
prompt tokens, where this reader refuses more as the prompt grows (see the
unknown-rate table). The evidence gain may therefore not turn into accuracy;
the 3k row is where it most plausibly does. Whatever the accuracy criterion
gives on these rows is the answer to RQ4; no rows are re-selected afterwards.

**Two comparisons per learned row (stated before the learned rows landed).**
(1) The pre-registered paired difference against FIFO with the same floor at
the same budget and target answers *does learning beat recency at the same
fill?* (RQ1). (2) The row's place on the accuracy-against-prompt-tokens
frontier, against the best rule-based row at equal or fewer prompt tokens,
answers *is a learned controller worth it over a simple rule?* (RQ4). On
LongMemEval the rule frontier is keep-last-0 + top 5 (0.466 at 1,554 tokens)
and FIFO + floor at 1% (0.440 at 1,148). A win on (1) alone means learning
beats recency at the same fill; a win on both means the learned controller is
worth having.

**Gate result (LoCoMo; agent null, categories 1–4, the shared fusion floor,
fill only; `runs/exp13_gate_locomo_table.md`).** The LongMemEval-trained
policies (fold-0 checkpoints) are a cross-benchmark transfer test. The gate
pairs per episode, and a LoCoMo episode is a conversation, so n = 10 and the
interval resamples conversations; with 10 clusters it undercovers. The rule
was not changed. Both go intervals sit far from 0 (lower bounds +0.074 and
+0.054), so undercoverage is unlikely to flip them.

| candidate | 5% | 10% | 25% |
|---|---|---|---|
| composed-episode trained (transfer) | −0.029 | −0.014 | **+0.096 (+0.074, +0.119) go** |
| single-question trained (transfer) | −0.080 | −0.097 | −0.060 |
| synthetic-trained | −0.027 | −0.032 | **+0.082 (+0.054, +0.107) go** |

(Evidence in view, candidate − FIFO; FIFO itself 0.318 / 0.399 / 0.510.)

*Hypothesis (both benchmarks): nothing at tight budgets, gains at the
loosest.* At tight budgets the floor and the question fill the budget and
nothing is left to choose; at loose budgets the policy has a real choice over
what else to keep. The LongMemEval target rows (3k, 4k) test this: if the
learned gain tracks the room left after the floor, the reading holds.

*Hypothesis (transfer): turn length.* Single-question LongMemEval training
transfers negatively to LoCoMo at every budget, composed training transfers
at 25%, and the synthetic policy passes on LoCoMo at 25% though it failed
everywhere on LongMemEval. LoCoMo turns are short (mean 26 words, median 22),
like the synthetic facts (about 11 words), while LongMemEval turns are long
(mean 161, median 75; every tenth instance sampled). A policy's size and
recency features may mean different things at these lengths. Not tested.

**Two comparisons for the paid LoCoMo learned rows (stated before they
landed):** paired accuracy against `fifo_top5_fusion` at 25%, and position on
the accuracy-against-prompt-tokens frontier against the best LoCoMo rule at
equal or fewer tokens (from the keep grid). Full context is a further
reference on LoCoMo, where conversations fit in context: 0.462 on categories
1–4 (§11e), with the earlier short-phrase reader, so not directly comparable.

**FIFO with the same floor, LongMemEval, all five folds (the learned rows'
partners).** Evidence in view stays at 0.62–0.63 while accuracy falls as the
prompt grows: 0.438 at 3,157 tokens (5%, 3k target), 0.423 at 4,197 (4k),
0.372 at 5,415 (filled to 5%). This is the refusal effect inside a single
controller. The oracle row (0.672, 394 tokens) is the same at 1, 2 and 5%:
it drops everything never needed, so the budget never binds.

### 13a. LongMemEval learned rows: result (2026-10-07)

The 25 gated cells (5 settings × 5 folds) against their FIFO partners with the
same floor, 470 non-abstention questions pooled, paired by question
(`runs/_pipelines/learned_report.py`, `mechanism_report.py`).

| learned row (5%) | learned | FIFO | difference (95% CI) | prompt tokens learned / FIFO |
|---|---|---|---|---|
| composed-episode, fill | 0.419 | 0.372 | **+0.047 (+0.004, +0.089)** | 5,894 / 5,413 |
| composed-episode, 4k | 0.426 | 0.423 | +0.002 (−0.043, +0.040) | 4,595 / 4,194 |
| composed-episode, 3k | 0.396 | 0.438 | **−0.043 (−0.085, −0.004)** | 3,465 / 3,158 |
| single-question, fill | 0.400 | 0.372 | +0.028 (−0.013, +0.066) | 5,853 / 5,413 |
| single-question, 4k | 0.417 | 0.423 | −0.006 (−0.049, +0.034) | 4,508 / 4,194 |

**Verdict under the pre-registered criteria.**

- **Criterion 1 (paired against FIFO at the same budget and target).**
  Learning beats recency in one of five settings: composed-episode training,
  filled to 5%. It loses in one: the same policy at a 3k target. The other
  three show no difference.
- **Holm correction (computed afterwards, not pre-registered, so it does not
  change the verdict).** With a Holm correction over the five tests, neither
  the gain (p = 0.032, adjusted 0.16) nor the loss (p = 0.042, adjusted 0.17)
  survives.
- **Criterion 2 (frontier).** No learned row reaches the rule frontier:
  keep-last-0 + top 5 scores 0.466 at 1,554 prompt tokens, and the best
  learned row scores 0.426 at 4,595.
- **Per-type differences** are descriptive. Preference is n = 30, and every
  per-type interval includes 0.

**Where the evidence went.** Questions are split by whether all their needed
items were in view.

| row (5%) | P(all in view) L / F | P(correct \| in view) L / F | P(correct \| not) L / F | memory lines L / F | content tokens per line L / F | label tokens L / F |
|---|---|---|---|---|---|---|
| composed, fill | 0.640 / 0.500 | 0.565 / 0.591 | 0.160 / 0.153 | 47.2 / 24.2 | 107 / 203 | 849 / 507 |
| composed, 4k | 0.549 / 0.504 | 0.605 / 0.658 | 0.208 / 0.185 | 34.4 / 18.9 | 114 / 198 | 678 / 434 |
| composed, 3k | 0.506 / 0.500 | 0.597 / 0.677 | 0.190 / 0.200 | 23.9 / 14.7 | 122 / 190 | 541 / 374 |
| single-question, fill | 0.596 / 0.500 | 0.564 / 0.591 | 0.158 / 0.153 | 43.9 / 24.2 | 115 / 203 | 822 / 507 |
| single-question, 4k | 0.555 / 0.504 | 0.602 / 0.658 | 0.187 / 0.185 | 30.3 / 18.9 | 128 / 198 | 633 / 434 |

Answers judged correct without all their evidence in view (0.15–0.21) mostly
had part of it in view: 17 of 27 for the composed policy filled to 5%, 30 of
36 for FIFO. The rest are the reader's own knowledge, or the judge accepting a
near answer.

The learned policies have all the needed evidence in view more often (up to
0.64 against 0.50). But **when the evidence is in view they answer correctly
less often than FIFO in every row** (0.597 against 0.677 at 3k). They keep 1.6
to 2 times as many memory lines, made of shorter turns, and each line carries
a "speaker, date" label of about 18–25 tokens that the budget does not count.

**Context rot.** The reader gets worse as the prompt grows, even when the
evidence is present ("context rot"). Here the symptom is refusal: with all
the evidence in view, "unknown" rises from 0.079 at 399 prompt tokens (the
oracle) to 0.296 at 6,223 (the evidence-only ceiling). The onset is about 3k
prompt tokens for this reader and prompt. The effect is known:
- Liu et al. 2023, "Lost in the Middle" (https://arxiv.org/abs/2307.03172).
- Chroma's 2025 report on context rot (https://research.trychroma.com/context-rot).
- LongMemEval's own finding that an 8B reader drops past about 3k retrieved
  tokens (https://arxiv.org/abs/2410.10813).

The controller's job is therefore to keep the prompt small and right, not to
fill the budget. The 3k onset is measured for one reader and one prompt;
checking another reader is the next paid step (N4 in
docs/archive/SUGGESTED_NEXT_STEPS.html).

**Mechanism and limitation.**

- *Limitation.* Matching on content tokens does not match prompt size, and the
  gap grows with the number of lines. The pre-registered results stand as
  run; nothing was re-matched.
- *Mechanism.* The hindsight teacher rewards evidence per content token, so
  short needed turns are cheap to keep. The reader's cost is per line, and for
  this reader every extra line raises the refusal rate. The controller
  optimises what the teacher asks for (+0.13 evidence at fill), and the reader
  does not convert it.

For RQ2 on real conversations: knowing what will be needed is not enough; the
teacher must also price what is kept.

**Next (not done).** One change, not two: count the per-line labels in the
budget, so the budget is the prompt the reader actually pays for and the budget
axis and the frontier axis are the same. Then give the regret teacher a cost
per prompt token, calibrated from the refusal curve: FIFO accuracy against
tokens at constant evidence, 0.438 at 3.2k to 0.372 at 5.4k. That is 0.066
accuracy per 2.26k tokens, about 3e-5 per prompt token, added to the regret of
keeping an item, so the teacher is indifferent between keeping a needed item
and the accuracy its extra tokens cost. The value is set in advance, not tuned. Labels are kept,
because dates are needed for temporal questions and speakers on LoCoMo.
Retrain on composed episodes, apply the evidence gate and a prompt-token gate
with no reader, and pay for an evaluation only if the agent-null proxy places
the policy ahead of keep-last-0 + top 5 on the frontier.

### 13b. LoCoMo: result (2026-10-07)

Categories 1–4 (1,540 questions), the shared fusion search, paired by
conversation with FIFO at the same budget.

| row | 5% | 10% | 25% | prompt tokens at 10% |
|---|---|---|---|---|
| oracle (hindsight) | 0.468 | 0.438 | 0.421 | 1,664 |
| FIFO + floor (fusion) | 0.237 | **0.299** | 0.273 | 2,917 |
| FIFO + floor (BM25, links to 11e) | 0.236 | 0.288 | 0.279 | 2,926 |
| salience + floor (fusion) | 0.198 | 0.228 | 0.288 | 2,845 |
| keep last 0/4/16 + fusion top 5 (budget-free) | | 0.155 / 0.170 / 0.184 | | 306 / 359 / 577 |
| composed-episode policy (LongMemEval transfer) | | | 0.290 (+0.017, −0.017 to +0.049) | 6,810 at 25% |
| synthetic-trained policy | | | 0.284 (+0.011, −0.022 to +0.041) | 6,879 at 25% |

**Verdict.** Neither learned row wins on either criterion. Both are level with
FIFO at 25%, and both lie below FIFO at 10% (0.299 at 2,917 tokens), the
best rule at equal or fewer tokens. The composed-episode policy is a
cross-benchmark transfer test; it was not trained on LoCoMo.

- **The same turning point on both benchmarks (context rot, §13a).** FIFO
  peaks near 3k prompt tokens: on LoCoMo 0.299 at 2.9k, then 0.273 at 7.4k; on LongMemEval 0.438 at
  3.2k, then 0.372 at 5.4k. The LoCoMo oracle also falls as its prompt grows:
  0.468 at 1.0k to 0.421 at 3.4k. With about 154 questions per conversation,
  more items stay needed at larger budgets. See
  `docs/research/figures/exp13_frontier.png`, produced by
  `memctl/analysis/frontier_plot.py`.
- **The two benchmarks fail differently.** On LongMemEval the five retrieved
  turns usually hold the evidence (0.62–0.64 in view), and the problem is
  selection. On LoCoMo they rarely do: keep-last + top 5 has 0.18–0.21 in
  view. The evidence is spread over several short turns, and multi-hop
  questions need all of them, so the problem is search. Fusion raised
  evidence by 0.03–0.05 but accuracy by about 0.01. LoCoMo gains therefore
  have to come from retrieval units (fact keys, session-level retrieval), not
  from eviction, as the all-found curves of §12 predicted.
- **Full context on LoCoMo** (0.462 on categories 1–4, §11e) was run only
  with the earlier short-phrase reader. It was not re-run under the frozen
  reader, so it is a reference under a different prompt.

### 13c. Provenance and cost

- **Reader frozen at 7b5fc30.** The check (both versions) and the ceiling ran
  from 7b5fc30.
- **Cells relaunched after the OOM incident.** At 21:58 PDT on 2026-10-06,
  running 12 trainings and a 16-worker sweep on the laptop, each process
  parsing the whole LongMemEval file, exhausted memory. Every job and the
  sessions running them were killed. Everything after that ran from 2567c3e
  (sharded loading, verified identical by a test), with jobs under a memory
  guard.
- **Cells holding episodes from both commits.** The first 16 `exp13_lme_keep`
  cells hold episodes from both commits. Their metadata records a
  reconstructed `git_history`, with the logs it rests on. The 5 ceiling cells
  hold only 7b5fc30 episodes.
- **Checkpoints.** All checkpoints were trained at 2567c3e (configs from
  69fd0f5 and f045f01). The gates ran at 8e1b1c5 and df3c7b0.
- **Cost.**
  - GPU: one A40 pod, 6.64 h (04:29–05:05 and 05:10–11:12 UTC on 2026-10-07)
    at $0.49/h, about $3.25. The user approved $3.50–5.
  - Laptop: 12 training runs (62–71 min each, run together, under 1 GB each)
    and 252 gate cells (about 35 min for LongMemEval, about 3 h for LoCoMo
    with bge on CPU).

## 14. A token price for kept memory (N1; 2026-10-07, pre-registered, free stage)

The change Experiment 13 called for: the budget counts what the reader pays,
and the controller is charged for what it keeps.

- `memory.count_labels: true` (b45ed8e): each item's tokens include its
  "speaker, date:" label, so the budget is the prompt.
- The token price (dccff18): a plug-in Bayes rule over a learned P(needed
  again | kept item). At every decision the controller archives each kept item
  whose P(needed) × need_value is below token_price × its tokens.
  - token_price = 3e-5 per token, fixed in advance from Experiment 13's FIFO
    refusal curve.
  - need_value = 0.44, also fixed in advance:
    P(correct | evidence in view) − P(correct | not), FIFO filled to 5%, §13a.
  - Prices of 1e-4 and 3e-4 are a declared sensitivity check, not tuning.
  - The floor still fetches the top 5 at every question, so archived evidence
    can come back.
  - Unlike the reviewer's N1, the eviction policy is not retrained. The price
    acts through the plug-in rule, so the composed-episode policy of
    Experiment 13 is reused.
- The needed model is trained per fold on the train part
  (`configs/rl/lme_n1/needed_f*.yaml`): active items sampled at every
  decision, hindsight labels. Base rate is about 0.6% needed.

**Pre-registered gate (written before it ran).** No reader, LongMemEval test
folds, labels counted (`configs/sweeps/exp14/`). A candidate cell (composed
policy with price 3e-5, 1e-4, 3e-4; without price as a control) goes to the
reader only if two things hold against keep-last-0 + top 5, the rule frontier
of Experiment 13:
1. its evidence in view, paired over questions, minus the rule's has a 95%
   interval entirely above 0; and
2. its mean memory tokens at the question are at most 1.1 times the rule's.

Analysis: `python -m memctl.analysis.gate runs/exp14_gate_f* --frontier
keep_last0_top5`. Paid cells, if any, go to the user with a cost first.

### 14a. Result: no-go (2026-10-07)

The arm is a **priced archive (plug-in)**: a learned needed model plus a fixed
decision rule. It is not a learned eviction policy, so it supports no RQ1
claim about learned eviction.
- **Order at each decision:** floor retrieval, then the composed-episode
  policy's own retrieval and its removals for any overflow, then the price
  rule over the items the policy kept.
- **Pricing makes a decision every step**, so the rule can keep memory below
  the budget.
- **Budgets are not comparable to §13 at the same fraction.** With labels
  counted, the history is about 7.5% longer. Mean budgets here are 2,192
  tokens at 2% and 5,481 at 5%.

**Gate** (`runs/exp14_gate_table.md`; 500 questions, no reader). Against
keep-last-0 + top 5 (evidence 0.630–0.637 at 1,218–1,333 tokens), no cell
passes the pre-registered rule. The cells that keep more evidence do so only
with 2.5–4 times the tokens:

| row | evidence | Δ vs keep-last-0 + top 5 (95% CI) | Δ vs unpriced policy, same budget | memory tokens / lines at the question |
|---|---|---|---|---|
| priced 3e-5, 5% | 0.767 | +0.130 (+0.099, +0.161) | −0.001 | 5,282 / 51.1 |
| priced 1e-4, 5% | 0.683 | +0.046 (+0.020, +0.072) | −0.085 | 3,367 / 40.5 |
| priced 3e-4, 5% | 0.639 | +0.002 (−0.012, +0.015) | −0.130 | 1,892 / 17.8 |
| priced 3e-4, 2% | 0.616 | −0.021 (−0.034, −0.009) | −0.020 | 1,336 / 7.5 |
| unpriced policy, 5% | 0.769 | +0.131 | — | 5,397 / 46.1 |
| keep-last-0 + top 5 | 0.630–0.637 | — | | 1,218–1,333 / 4.8–5.0 |

(The Δ against keep-last-0 + top 5 is taken against its 5% cell for 5% rows
and its 2% cell for 2% rows. The memory tokens count labels, so they are the
prompt less the fixed instructions.)

- **The pre-set price (3e-5) does almost nothing.** It archives 0.7% of kept
  items per step.
- **Higher prices trade evidence for tokens.** At 3e-4 the rule lands on the
  frontier rule's evidence, with 1.4 times its tokens. It never gets ahead.

**Why.**
- **Calibration** (`runs/exp14_calibration_f*.json`, test parts, about 340k
  sampled items). The needed models are calibrated but rank weakly: AUC
  0.62–0.74, base rate 1.1–1.4%. Almost no item is predicted above 3%, so at
  the higher prices nearly every item falls below the keep threshold
  P* = price × tokens / 0.44, and the rule archives by length alone. At 1e-4
  the cutoff is roughly 90 tokens.
- **Pricing tokens is the wrong target.** A per-token price archives long
  turns and keeps the short ones: at 1e-4, tokens fall 38% but lines only 12%.
  The reader's cost in Experiment 13 tracked lines (and the labels and
  distractors that come with them).

**What it settles.** The needed models' AUC (0.62–0.74 on about 340k
held-out item-steps, base rate 1.1–1.4%, nothing predicted above 0.045)
measures how well a turn's later need can be predicted *before the question
arrives*, from what the controller sees. Hindsight has that information, and
nothing in the stream carries it.
- On these benchmarks, query-blind eviction cannot beat "archive everything,
  retrieve at the question", whatever the price or teacher.
- This is the causal ceiling of the synthetic task (D1), now measured on real
  text. It matches KV Policy's finding that query-aware heuristics close most
  of the gap to learned query-blind policies.
- With a free, searchable archive, eviction to the archive is nearly free. The
  only decisions that carry value are made at the question: what to retrieve,
  how much to show, and so how large the prompt is.

**Scope for the thesis.** The original target was long-running tasks where
questions interleave with the stream and no free archive exists. That is where
eviction carries information, and where the synthetic results stand. LoCoMo
and LongMemEval ask their questions after the stream, with a searchable
archive, so they test retrieval and prompt size, and the eviction arm has
nothing to decide. The composed-episode environment (questions asked
mid-stream) is the closest real-text approximation of interleaving. The
archive still makes eviction free there.

**Next (re-ranked by the review).** First, a query-aware head that chooses
which 5 of the 16 shortlisted items fill the floor (N2). It keeps the same
lines in the prompt and has measured headroom: all-found recall@16 minus @5
is 0.16 on LongMemEval. A variant also chooses k (3, 5 or 8) per question.
The per-line price with a stop head is demoted, since it can only approach
keep-last-0 + top k. A better needed signal is dropped for QA after the
stream and kept for interleaved tasks. The two candidates considered before
the review:
1. A per-line price with a learned stop head: a policy that ends removal
   below the budget, trained against a priced teacher.
2. A better needed signal before any price is applied. The needed model
   sees only the policy's item features; an AUC of about 0.65 cannot
   separate the 1% of items that matter.

No paid cells were proposed.

## 15. A query-aware floor head (N2; 2026-10-07, pre-registered, free stage)

The only decision with information on these benchmarks is made at the
question (§14a), so this learns it. Memory keeps nothing but the current turn
(`keep_none`). At each question a head picks 5 of the 16 BM25 candidates
(`floor_head`, a top-k by its logit), so the prompt has the same 5 retrieved
lines as keep-last-0 + top 5. The only difference is *which* 5.

- **Training.** Imitation on composed 4-question episodes from each fold's
  train part, labels counted (`configs/rl/lme_n2/head_<arm>_f*.yaml`). The
  shortlist is 16 BM25 candidates, with no follow-the-clue items. **Three
  arms, declared before any gate result was read:**
  - `listwise`: minus the log of the softmax mass on the gold set. This is
    the chance that the top item is some gold item.
  - `listsum`: each gold item's cross-entropy against all 16, summed. It
    pushes every gold item up, as the all-found gate needs.
  - `residual`: `listsum` with logit + α × BM25 score on the retrieve column.
    α is learned from 20, which reproduces the BM25 order at the start.

  The first run was stopped before any gate ran, when the review pointed out
  that `listwise` does not match an all-found gate.
- **Headroom.** §12: all-found recall@16 minus @5 is about 0.16 on
  LongMemEval.

**Pre-registered gate (written before it ran).** No reader, the five test folds
(`configs/sweeps/exp15/`), 500 questions. The head goes to the reader only if
its *all-found* evidence (`evidence_complete_rate`: every needed item in view)
minus keep-last-0 + top 5's, paired over questions, has a 95% interval
entirely above 0. Lines in the prompt are equal by construction (5 retrieved);
mean memory tokens at the question are reported beside it. Analysis:
`python -m memctl.analysis.gate runs/exp15_gate_f* --frontier keep_last0_top5
--candidates head_top5 --measure evidence_complete_rate`; the token condition
of that tool (≤ 1.1×) also applies. The rule is the same for each of the
three arms; whichever passes, passes.

The same gate is also run on each fold's *train* part
(`exp15_gate_f*_train`), to show overfitting: about 27k parameters against
about 1,000 training lists. The adaptive k (3, 5 or 8) waits for the next
round.

### 15a. Result: all three arms pass the free gate (2026-10-07)

All runs are at 07b2166, not dirty; `runs/exp15_gate_table.md`. The baseline
is keep-last-0 + top 5 (BM25). 500 test questions, paired. The head and the
rule both retrieve exactly 5 turns per question.

| arm | all-found evidence | Δ vs rule (95% CI) | share found | Δ | memory tokens at question |
|---|---|---|---|---|---|
| keep-last-0 + top 5 | 0.506 | — | 0.637 | — | 1,333 |
| head, `listwise` | 0.592 | **+0.086 (+0.054, +0.120)** | 0.715 | +0.077 | 666 |
| head, `listsum` | 0.590 | **+0.084 (+0.052, +0.118)** | 0.716 | +0.078 | 561 |
| head, `residual` | 0.586 | **+0.080 (+0.048, +0.114)** | 0.708 | +0.070 | 613 |

- **All three pass the pre-registered rule.** Each keeps all of a question's
  evidence in view about 8 points more often, at under half the memory
  tokens: the head picks shorter turns. The three arms do not differ from one
  another.
- **No sign of overfitting.** On the train parts the gains are similar
  (+0.060 to +0.070 all-found).

**All-found evidence by question type:**

| arm | knowledge-update | multi-session | single-session-assistant | preference | single-session-user | temporal | abstention |
|---|---|---|---|---|---|---|---|
| keep-last-0 + top 5 | 0.625 | 0.182 | 0.875 | 0.300 | 0.828 | 0.472 | 0.500 |
| head, `listsum` | 0.722 | 0.314 | 0.768 | 0.333 | 0.875 | 0.622 | 0.567 |
| head, `listwise` | 0.736 | 0.314 | 0.786 | 0.267 | 0.891 | 0.614 | 0.600 |

The gains come where evidence is spread or dated: multi-session (+0.13),
temporal (+0.15) and knowledge-update (+0.10). The head loses on
single-session-assistant (−0.09 to −0.11), where the evidence sits in the
assistant's own, usually long, turns. The `listsum` loss did not separate from
`listwise` on multi-session.

**Controls (review; free; `exp15_control_f*`).** Two rules on the same 16
BM25 candidates test whether the head only learned "evidence turns are short
user turns".

| rule | all-found | head (`listsum`) minus rule | memory tokens |
|---|---|---|---|
| shortest 5 of 16 | 0.382 | +0.208 (+0.166, +0.250) | 357 |
| user turns only, top 5 of 16 | 0.530 | +0.060 (+0.028, +0.094) | 566 |

- **Neither rule comes within the head's interval**, so the head learned
  something beyond length and speaker.
- **The user-only rule matches the head's tokens.** The token saving comes
  largely from preferring user turns; the evidence gain does not.
- **Lengths of the chosen turns** (tokens, labels counted):

  | | median | interquartile range |
  |---|---|---|
  | gold evidence | 79 | 67–93 |
  | head's 5 | 78 | 60–101 |
  | BM25's top 5 | 148 | 76–435 |

In one sentence: the user-only rule explains the head's token saving, but not
its evidence gain.

**Paid reader test (approved by the user on 2026-10-07, about $0.30–0.40).**
`configs/sweeps/exp15/exp15_paid_f*.yaml`: the three arms and
keep-last-0 + top 5 on all 500 questions, with the reader frozen at 7b5fc30 and
the official judge. Criteria, as in §13:
1. paired accuracy against keep-last-0 + top 5, 95% interval above 0, with
   Holm over the three arms reported;
2. place on the accuracy-against-tokens frontier;
3. the §13a decomposition (P(all in view), P(correct | in view), the unknown
   rate, prompt tokens), to separate the gain from more evidence and the gain
   from fewer tokens.

**Caveat.** The gate measures the same evidence labels the head was trained to
find: `has_answer` turns, plus any turn of an unmarked answer session. Whether
more labelled evidence, at half the tokens, turns into more correct answers is
the reader's test, which is a paid step.

### 15b. Reader test: the learned floor head beats the rule (2026-10-07)

Pod gci9yw2757b6oa ran 18:12–18:22 UTC, about 0.16 h, about $0.08; the
estimate was $0.30–0.40. Sweeps `exp15_paid_f*` at 3d7689b, reader frozen as
in §13. The analysis is `runs/_pipelines/exp15_report.py`; its output is
`runs/exp15_paid_report.md`. 470 non-abstention questions, paired by question.

| row | accuracy | Δ vs keep-last-0 + top 5 (95% CI) | P(all in view) | P(correct \| in view) | P(correct \| not) | "unknown" | prompt tokens |
|---|---|---|---|---|---|---|---|
| keep-last-0 + top 5 | 0.466 | — | 0.506 | 0.706 | 0.220 | 0.283 | 1,554 |
| head, `listsum` | **0.513** | **+0.047 (+0.009, +0.087)** | 0.591 | 0.698 | 0.245 | 0.211 | 745 |
| head, `listwise` | **0.506** | **+0.040 (+0.002, +0.079)** | 0.591 | 0.712 | 0.208 | 0.211 | 846 |
| head, `residual` | 0.491 | +0.026 (−0.013, +0.064) | 0.585 | 0.691 | 0.210 | 0.236 | 794 |

**Verdict under the pre-registered criteria.**

- **Criterion 1 (paired accuracy).** Two of three arms win: `listsum` and
  `listwise`. The `residual` arm does not.
- **Holm correction (computed afterwards, so it does not change the
  verdict).** Over the three arms the adjusted p-values are 0.058 (`listsum`),
  0.089 (`listwise`) and 0.19 (`residual`), so no arm survives at 0.05.
- **Criterion 2 (frontier).** The winning arms are ahead of the rule on both
  axes: higher accuracy at under half the prompt tokens (0.513 at 745 against
  0.466 at 1,554). They are the first learned rows on real conversations to
  lie beyond the rule frontier. They remain far below the hindsight oracle
  (0.672 at 394).

**Where the gain comes from.** In one sentence: of the +0.047, about +0.041
comes from having the evidence in view more often and about +0.006 from the
conditional accuracies. P(correct | in view) is unchanged at half the tokens
because both 745 and 1,554 tokens are below the context-rot onset. So the gain
is selection, and there is unused room below the onset. That is the case for
an adaptive k: show 8 turns instead of 5 when the head is unsure.

In detail, accuracy is the mixture
P(in view) × P(correct | in view) + (1 − P(in view)) × P(correct | not).

- Keeping the rule's conditional accuracies and using `listsum`'s P(all in
  view) gives 0.507. So about +0.041 of `listsum`'s +0.047 comes from having
  all the evidence in view more often, and about +0.006 from the conditional
  accuracies.
- The shorter prompt did not raise P(correct | in view): 0.698 against 0.706.
  It did lower the unknown rate (0.211 against 0.283), mostly on questions
  whose evidence is now in view.
- At these prompt sizes (under 1.6k tokens), below the ~3k onset of context
  rot, fewer tokens buy little. More evidence is what pays.

**By type (accuracy, rule → `listsum`):**

| type | rule | `listsum` |
|---|---|---|
| multi-session | 0.256 | 0.322 |
| temporal | 0.323 | 0.370 |
| single-session-user | 0.734 | 0.844 |
| knowledge-update | 0.694 | 0.722 |
| preference (n = 30) | 0.067 | 0.167 |
| single-session-assistant | 0.857 | 0.786 |

The rows are descriptive. The one loss, single-session-assistant, matches the
evidence loss in §15a.

**LoCoMo transfer (free gate; `runs/exp15_gate_locomo_table.md`).** The fold-0
LongMemEval heads rank the shared fusion search's top 16 on LoCoMo categories
1–4, against fusion's own top 5, with keep-none and no reader.

- All-found evidence rises from 0.206 to 0.33–0.34: +0.13, interval about
  +0.11 to +0.15 over 10 conversations. Share-found rises by +0.11 to +0.12.
- But the head's five turns take 144–146 memory tokens against the rule's 94.
  That breaks the pre-registered 1.1× token condition, so the result is
  *no-go* by the rule, though the evidence gain transfers.
- The absolute difference is small (about 50 tokens; LoCoMo turns are short).
  No LoCoMo reader cell is proposed under this rule.

### 15c. Robustness: the gain replicates across training seeds and on a second host (2026-10-07)

Pod ji4gpj6dpi4tm0: an A40 on a CUDA 13.0 host, because A40s on 12.8 hosts
were out of stock. The user approved this check. All cells ran on this one host
with a fresh cache (`exp15_host_f*`, c6a9752): keep-last-0 + top 5 and the
`listsum` head from training seeds 0, 1 and 2.

**Host check, measured before the run.**
- The judge's verdict was identical in every comparison (50/50).
- 50 cached prompts from the first host, regenerated here with no cache, gave
  the same parsed answer 44/50 times.
- The same 50 prompts generated twice on this host gave the same answer
  49/50 times.
- So about 1 answer in 50 changes from vLLM's batch nondeterminism alone, and
  about 5 in 50 when the host changes. The wording moves; correctness barely
  does. This is also why accuracy is reported here with a bootstrap interval,
  not as a bare three-decimal number.

| row (host 2) | accuracy | Δ vs keep-last-0 + top 5 (95% CI) |
|---|---|---|
| keep-last-0 + top 5 | 0.457 | — |
| head `listsum`, seed 0 | 0.504 | +0.047 (+0.009, +0.087) |
| head `listsum`, seed 1 | 0.500 | +0.043 (+0.004, +0.081) |
| head `listsum`, seed 2 | 0.500 | +0.043 (+0.004, +0.083) |
| mean over the three seeds | | **+0.044 (+0.008, +0.082)**, seed SD 0.002 |

**The gain replicates.** Every training seed beats the rule on its own, on a
second host, by the same amount. The seed-averaged paired interval excludes 0.
"Borderline after Holm" in §15b was a statement about three loss arms
tested once. `listsum` was chosen for this replication *after* it scored best
among them. It was one of three declared arms, not a single arm specified in
advance, so the replication does not remove that selection step. What it
shows is that the selected arm's gain is not seed or host luck: three
independent training seeds, each excluding 0, on a second host. If a fully confirmatory test is wanted, the route is a pre-registered
test of `listsum` alone on LongMemEval-M. It has the same 500 questions with
far longer histories, so it is a harder setting, not new data. The partial cells of the first attempt on this
host, made before the host check, are in `runs/_aborted/` and are not reported.

**For the thesis.** On real conversations with a searchable archive, the
learned decision that pays is query-aware: which retrieved turns fill the few
lines the reader sees. A small head (about 27k parameters) trained by
imitation on gold evidence turns a gain in evidence into a gain in accuracy at
half the prompt. This is RQ4's positive answer, with the caveats above: one
benchmark, one reader, and a result that sits at the edge after a correction
for three arms. LoCoMo would be a cross-benchmark transfer test. The adaptive
number of turns (3, 5 or 8) is the declared next arm.

## 16. Plan (not built): set-valued GRPO on the selection head, with the reader's verdict as reward

Pre-registered plan, written 2026-10-07 and revised after review, before any
code. It is the RQ3 experiment on real conversations: one question, one
decision, one judged answer, so credit assignment is exact. Every choice below
is declared once and not swept.

- **Policy.** The Experiment 15 selection head (5 of 16 BM25 candidates, at
  the question, keep-none), started from each fold's `listsum` checkpoint.
  One RL seed per fold; the two extra `listsum` seeds already trained are the
  variance reference.
- **Action and probability.**
  - The action is an unordered 5-subset.
  - G = 8 subsets are sampled per question without replacement by
    Gumbel-top-k.
  - Their log-probability is the unordered-set probability of Kool et al.
    2020 (https://arxiv.org/abs/2002.06043), computed exactly over the 5!
    orders.
- **Reward.** The frozen reader and official judge of Experiment 13 (7b5fc30)
  give 1 or 0. To this is added a dense auxiliary term so that near-ties still
  carry signal: 0.2 × token F1 between the answer and the gold.
- **Advantage.** Group-mean baseline (reward minus the mean of its 8), with no
  division by the group's standard deviation. A binary reward makes that
  division explode on near-ties; Experiment 10 found the forms equivalent in
  effect.
- **Ties.**
  - Before training, one reader call per train question with empty memory;
    questions it answers correctly without memory are dropped (MemAgent's
    filter).
  - The share of groups with all-equal rewards is logged and reported every
    iteration.
- **Loss.** PPO-clipped policy gradient on the set log-probability, plus an
  imitation anchor of λ = 0.1 × the `listsum` loss. It is small on purpose:
  it keeps the head off degenerate subsets without tying it to the evidence
  labels, which RL is meant to go beyond (the reader may prefer a short
  unlabelled turn to a long labelled one). Learning rate 3e-4, 4 epochs per
  batch.
- **Cost (an upper bound).** Per fold:
  - at most 1,000 train questions × 8 subsets × 2 calls (answer and judge):
    about 16k;
  - the empty-memory filter: about 400 calls;
  - gate evaluations on the held-out 20% (about 200 questions × 2 calls × 5
    evaluations): about 2,000.

  That is about 18k calls per fold and about 90k for all five. The generation
  cache makes a repeated (subset, question) pair free, and repeats grow as the
  policy sharpens. Training runs fewer requests in parallel than Experiment
  15's sweep, so allow 30–40 min per fold: about 3 h of A40 with startup,
  **about $1.50, at most $2.50**.
- **Gate (paid, needs the reader).** On a held-out 20% of each fold's train
  part: the greedy top-5 accuracy of the GRPO head minus its `listsum`
  start's, paired, pooled over folds, with a 95% interval above 0.
- **Test (only if the gate passes).** The §15b protocol on the test parts.
  - GRPO head against its `listsum` start: paired; the RQ3 question.
  - GRPO head against keep-last-0 + top 5: paired; the RQ4 question.
  - Holm over the comparisons that reach the test, the §13a decomposition,
    and per-type rows.
- **Notes added at code review (before any paid call).**
  - The anchor contributes nothing for a shortlist with no gold item: an
    empty sum.
  - Abstention questions drop out at the empty-memory filter: empty memory
    gives "unknown", which is graded correct for them.
  - The held-out slice is evaluated at the start and after iterations 2, 4
    and 6, to give a learning curve.
  - The gate's power: about 80 held-out questions per fold, 400 pooled, so
    the paired interval is about ±0.04. A null means "no effect above about
    0.04", not "no effect".
  - The reward has a noise floor: on one host, about 1 answer in 50 changes
    between identical repeated calls (vLLM batch nondeterminism, §15b).
  - It runs on the robustness check's host and cache, so its starting
    accuracy is comparable with that check.
- **Standing rule from §16 on (added after two failed launches, 2026-10-07).**
  Before any paid launch, run one full-size iteration with a stub backend under
  the memory guard and record its peak RSS. Both failures would have been
  caught this way at no cost:
  - a cache-write race between threads, fixed in b2088bf;
  - an autograd graph that ran out of memory, fixed in 2eacf0d with a
    vectorised set log-probability that a test checks against the
    order-by-order loop in value and gradient.

  The full-size stub iteration peaks at 0.44 GB.
- **Added 2026-10-09 (§24):** before launch, delete the stub output and confirm at least 15 GB of free disk. A
  full-size stub rehearsal of many readers can itself fill the disk (§24's was 20 GB) and break a paid stage.
- **What a result means, stated in advance.**
  - A pass on RQ3 means RL on the reader's own correctness beats imitation of
    evidence labels. The per-type rows that move then show what the labels
    were missing; single-session-assistant, where the head loses, is the
    obvious candidate.
  - A fail means the evidence labels were already a sufficient target for a
    head this small.

### 16a. Result: the GRPO gate fails (2026-10-07)

Pod ji4gpj6dpi4tm0 (A40, CUDA 13.0 host) ran 18:45–19:30 UTC, about 0.74 h,
about $0.36, for the §15c robustness check and this run; the user approved
about $1.70, at most $2.70. The run is at 2eacf0d, not dirty. Peak RSS was
663 MB per fold. `runs/lme_grpo_f*`.
- **Cache integrity** after the cache-write race (b2088bf): 4,058 entries,
  0 written under the wrong key, 1 in flight and read as a miss.
- **Two failed launches** are in `runs/_aborted/` and are not reported.

**Held-out gate** (80 questions per fold, 400 pooled; greedy top 5 of the
GRPO head minus its `listsum` start, paired): **+0.0125, 95% interval
−0.0050 to +0.0325. The interval includes 0, so the gate fails and no
test-fold run follows**, as pre-registered.

| fold | start | after GRPO | held-out curve (iterations 0 / 2 / 4 / 6) | groups with all rewards equal | kept after filter |
|---|---|---|---|---|---|
| 0 | 0.550 | 0.538 | 0.550 / 0.537 / 0.550 / 0.537 | 45–52% | 303 |
| 1 | 0.625 | 0.613 | 0.625 / 0.625 / 0.637 / 0.613 | 44–49% | 303 |
| 2 | 0.588 | 0.600 | 0.588 / 0.588 / 0.588 / 0.600 | 49–55% | 302 |
| 3 | 0.538 | 0.538 | flat | 51–56% | 302 |
| 4 | 0.463 | 0.538 | 0.463 / 0.487 / 0.500 / 0.537 | 52–61% | 301 |

Spend on Experiments 15–16: about $0.44 ($0.08 for the Experiment 15 reader
test and $0.36 for this pod), against the $1.70 the user approved.

**What it means, as stated in advance.**
- The result is consistent with a gain of about one point and inconsistent
  with a gain above about three. It is not evidence of no effect. With this
  budget (6 iterations × 128 questions × 8 subsets), RL on the reader's own
  verdict did not improve the selection beyond imitation of the evidence
  labels by more than about 0.03, the gate's resolution. One fold
  (4) moved by +0.075; the other four did not move.
- Per the pre-registration, the evidence labels were already a sufficient
  target for a head this small.

**Why little moved.**
- About half of the groups had all 8 rewards equal and carried no signal.
- The reward's noise floor (1 answer in 50 from batch nondeterminism) is
  comparable with the per-subset differences being learned.
- The training-set reward rose by 8–10 points in some folds (correct rate
  from 0.44 to 0.52–0.54 by iteration 3 or 4) while the held-out slice did
  not move. With about 27k parameters and about 300 training questions per
  fold, the likely reading is that the head fitted the training questions
  through their features rather than learning a transferable preference.
  This is also why a longer run is not expected to help.

**For RQ3.** GRPO's credit assignment was exact here (one question, one
decision), so this is not a credit-assignment failure. On this task the
binding constraint is the reward's information per sample, not the optimiser.
This matches Experiment 10's finding on the synthetic task, where GRPO added
about one point over imitation. A longer run is not proposed: the curve is
flat in four folds of five, and the pre-registered gate decides.

**RQ3 across both tasks.** Credit assignment across time was never the
binding constraint:
- on the synthetic task, the needed fact was out of reach of the shortlist;
- on real conversations, the reward carries too little information per sample
  (44–61% of groups tied, a noise floor of about 1 answer in 50).

The one credit-assignment question still open is *within* the chosen set:
which of the five turns earned the reward.

**A possible §17 (declared, not planned; the user decides).** A leave-one-out
set reward would give each of the five turns its own advantage:
- the reader answers once with each 4-of-5 subset, plus the full set;
- each turn's advantage is the drop in reward when it is left out, a
  Shapley-style estimate.

This makes the per-item signal dense. It costs about 5 times the reader calls
of §16, roughly $2–3 for all folds. It is the only GRPO variant still judged
promising for this problem, and it would be a methodological contribution if
it worked. The alternative is to close RQ3 on the evidence above.

**Decision (the user, 2026-10-07): RQ3 is closed on the evidence above.**
§17 is not run. The thesis answers RQ3 as follows. With exact credit, GRPO
adds at most about one point over imitation of hindsight labels, on the
synthetic task and on real conversations. The limit is the information each
sample carries, not credit assignment across time.

## 18. Second-family judge check, and an exploratory LoCoMo reader test (2026-10-07, pre-registered)

The user approved both, about $0.65–1.15 in all.

**Judge check.**
- **Question.** Every Experiment 13–16 cell was judged by Qwen2.5-7B on its own answers (official LongMemEval
  prompts). Does a judge from another family agree, and does the ranking of controllers hold?
- **Judge.** `ibm-granite/granite-3.1-8b-instruct` with the same official prompts, served by vLLM on its own
  pod. Llama-3.1-8B, named in the plan, needs a gated licence the pod cannot accept; Granite is openly licensed
  and from a different family.
- **Sample.** 300 LongMemEval answers already generated: 75 each from Experiment 15's keep-last-0 + top 5 and
  `head_listsum`, and from Experiment 13's FIFO + floor at 5% and the oracle at 2%. They are stratified by
  question type, abstention excluded, seed 0.
- **Four outputs, pre-registered.**
  1. Agreement with the Qwen verdicts, as a share and as Cohen's kappa.
  2. False-accept rate: each question graded against a planted wrong answer, another question's gold of the same
     type.
  3. False-reject rate: each question graded against its own gold, given as the answer.
  4. Whether the controller ranking holds: the four controllers' accuracies under Granite in the same order as
     under Qwen, with the paired difference of `head_listsum` minus keep-last-0 + top 5 and its interval.

- **Added before the judge stage ran.**
  - The false-accept and false-reject tests also run on the Qwen judge, over the same items (the sampling is
    seeded). Both judges are reported side by side: agreement alone cannot say which judge is wrong.
  - **Pre-registered reading.** The result that matters is output 4: does the head − rule paired difference keep
    its sign, with its interval excluding 0, under Granite?
    - If it does, the controller ranking is judge-independent.
    - If the absolute numbers move but the ranking holds, that is the outcome the literature expects (Anatomy of
      Agentic Memory), and the thesis says so.
  - **Amended before the Granite stage ran.** The Qwen-judge run showed that 74 questions per controller have too
    little power for output 4's paired interval (head − rule +0.068, CI −0.027 to +0.162, under the original
    judge). Output 4 is therefore computed on all 470 non-abstention questions: the head's and the rule's answers
    are all re-judged by Granite (about 940 short calls). Outputs 1–3 stay on the 296-answer sample.
  - **How the Granite rates are read (written before they landed).** Qwen's own rates on the same 296 items are
    the reference: 1.7% false-accept and 2.7% false-reject, with agreement 1.000 against its stored verdicts
    (cache hits, a sanity check).
    - A second judge with comparable error rates that agrees with Qwen confirms the verdicts.
    - One with much higher error rates that disagrees indicts itself, not Qwen.

**Exploratory LoCoMo reader test.** This is outside the pre-registered claims and labelled exploratory in every
table.
- **What runs.** The fold-0 LongMemEval `listsum` head ranks the shared fusion search's top 16 on LoCoMo
  categories 1–4 (keep-none, 5 shown), against keep-last-0 + fusion top 5, both under the frozen reader.
- **Why.** Its evidence gain (+0.13 all-found, §15b) failed only the token rule, by 50 tokens.
- **What is reported.** The paired accuracy difference, conversation-clustered, with the evidence decomposition.

### 18a. Results (2026-10-07)

Pod l46lez5jcbqt4x ran 19:41–20:00 UTC, about 0.32 h, about $0.16; the user approved about $0.65–1.15. It served
Qwen for LoCoMo, then Granite-3.1-8B, loaded by changing the vLLM arguments. Run from 9a5c34e (judge stage) and
e9b6568 (LoCoMo). Files: `runs/exp18_judge_{qwen,granite}.json` and `runs/exp18_locomo`.

**Judge check (pre-registered outputs).**

| output | Qwen2.5-7B (original judge) | Granite-3.1-8B (second family) |
|---|---|---|
| 1. agreement with the Qwen verdicts (296 answers) | 1.000 (cache, sanity) | **0.973, κ = 0.946** |
| 2. false-accept rate (planted wrong answer) | 1.7% | 3.0% |
| 3. false-reject rate (the gold as the answer) | 2.7% | 4.1% |
| 4. head − rule, all 470 questions | +0.047 (+0.009, +0.085) | **+0.040 (+0.002, +0.079)** |

- **Accuracy on the 296-answer sample** (Qwen → Granite):

  | controller | Qwen | Granite |
  |---|---|---|
  | oracle at 2% | 0.689 | 0.730 |
  | `head_listsum` | 0.527 | 0.541 |
  | keep-last-0 + top 5 | 0.459 | 0.486 |
  | FIFO + floor at 5% | 0.338 | 0.365 |

  The order is the same under both judges.
- **Reading, as pre-registered.**
  - Granite's error rates are comparable to Qwen's (a percentage point or so higher), and the two agree on 97% of
    verdicts (97.8% over the 940 head and rule answers). So the second judge confirms the verdicts.
  - Granite is about 2–4 points more lenient across the board, which moves the absolute numbers and not the
    ranking.
  - Output 4 keeps its sign and its interval still excludes 0, so **the controller ranking and the head's gain are
    judge-independent**.

**Exploratory LoCoMo reader test (post hoc, outside the pre-registered claims).** The fold-0 LongMemEval `listsum`
head ranks the fusion top 16 on LoCoMo categories 1–4, against keep-last-0 + fusion top 5, with keep-none for both.

| row (exploratory) | accuracy | all-found evidence | prompt tokens |
|---|---|---|---|
| keep-last-0 + fusion top 5 | 0.155 | 0.181 | 306 |
| LongMemEval-trained head | **0.242** | 0.298 | 365 |

- The difference is +0.087, with a conversation-clustered interval of (+0.063, +0.110) and a question-level
  interval of (+0.068, +0.108). It holds in every category: multi-hop 0.057 → 0.121, single-hop 0.203 → 0.325,
  temporal 0.146 → 0.181, open-domain 0.042 → 0.073.
- The evidence gain transfers to LoCoMo and turns into answers.
- Both rows stay far below FIFO + floor at 10% (0.299 at 2,917 tokens). On LoCoMo, five turns are too few: it is a
  search problem (§13b). That is why the head's row is not a frontier win there.
- **The pre-registered gate's verdict (no-go, §15b) stands.** This row is evidence for transfer, not a claim.

## 19. Adaptive k: the head chooses how many turns to show (N4; 2026-10-07, pre-registered, free stage)

**Question.** The §15 head always shows 5 turns. On LoCoMo that is too few (§18a: 0.242 at 365 prompt tokens,
against FIFO + floor at 10%, 0.299 at 2,917), and on both benchmarks 5 turns sit far below the ~3k onset of
context rot. Does a head that chooses k from {5, 8, 16} per question keep as much evidence in view as always
showing 16, at clearly fewer tokens?

**Build (declared before any training run; amended after review, before any training).**
- **Wider shortlist.** 32 candidates instead of 16: plain BM25 on LongMemEval, the shared fusion search on
  LoCoMo, as before. Before training, the free headroom check of §12 is extended to @32 and reported (all-found
  @16 against @32).
- **The head, cross-fitted.** Each fold's train part is split in two halves, A and B, by question. A head is
  trained on each half as in §15 (`listsum`, composed 4-question episodes, seed 0), with
  `retrieve_candidates: 32`. Head A's ranking is used at test for every arm; head B is run as a replicate and
  reported beside it. (Using half the training data is part of the "wider shortlist + retrain" step below.)
- **The k-chooser, trained out of sample.** A small network (one hidden layer of 32) on a head's view of the
  shortlist: the top 16 sorted logits, the gaps at ranks 5/6, 8/9 and 16/17, the softmax entropy, and the
  question's length. Three outputs, P(every gold item is in the head's top k) for k = 5, 8 and 16, trained by
  binary cross-entropy on head A's outputs on half B and head B's outputs on half A. The chooser never sees a
  head's outputs on that head's own training lists, because there its top 5 holds the gold far more often than at
  test and the chooser would learn to return k = 5 everywhere.
- **The decision rule, fixed now.** Show the smallest k whose predicted all-found probability is within
  δ = 0.05 of the prediction at k = 16: "pay for more turns only for at least 5 points of predicted evidence". δ is
  not tuned on test data; a sweep over δ ∈ {0.02, 0.05, 0.10} is reported on the train parts only, as
  description. One δ serves both steps, although 5 → 8 costs 3 turns and 8 → 16 costs 8; this is a declared
  simplification.

**Arms.** All use head A's ranking of the same 32 candidates.
- `adaptive`: k chosen per question as above.
- `fixed5`, `fixed8`, `fixed16`: k fixed.
- `oracle-k`: the smallest k in {5, 8, 16} with all the gold in the head's top k, else 16. It is the ceiling
  for any chooser. It shows, before the chooser is trained, how many tokens adaptivity could save at no loss of
  evidence.
- References: the §15 five-of-16 head, keep-last-0 + top 5, and FIFO + floor (LongMemEval 5% with the 3k target,
  0.438 at 3,158 prompt tokens, §13a; LoCoMo 10%, 2,917, §13b).

**Measures.** All-found evidence (`evidence_complete_rate`), mean prompt tokens (the same axis as §13, §15 and
§18; memory tokens at the question beside them), mean k and its distribution by question type, and the
chooser's calibration on the test folds: the reliability of P(all-found@k) in 5 bins for each k. No reader.

**Where.** LongMemEval, the five test folds, 500 questions, paired by question: the primary test. LoCoMo
categories 1–4 with the fold-0 heads and chooser: transfer, intervals clustered by conversation (10
conversations, so they undercover). The train-part gate is still reported, but it is inflated for the chooser too;
the test-fold calibration is the real check against overfitting.

**Pre-registered gate (per benchmark).**
1. **More evidence.** The arm minus the §15 five-turn head, all-found, paired 95% interval entirely above 0.
2. **Tokens.** The arm's mean prompt tokens are at most FIFO + floor's at the comparison budget (3,158 on
   LongMemEval, 2,917 on LoCoMo) and at most 2,000, the explicit guard below the onset. The frontier plot, not the
   gate, carries any "how much less" claim.
3. **Ceiling first.** If `oracle-k` saves less than 25% of `fixed16`'s prompt tokens on a benchmark, adaptivity is
   not worth a reader cell there; that reading is declared now.
4. **Adaptivity earns its place** if `adaptive` passes 1 and 2, and against `fixed16`:
   (a) `fixed16` minus `adaptive`, all-found, paired 95% upper bound at most 0.02 (no material loss of evidence);
   (b) `adaptive`'s mean prompt tokens at most 0.75 × `fixed16`'s.
   Then `adaptive` goes to the reader. Otherwise the smallest fixed k that passes 1 and 2 goes to the reader, and
   "adaptivity not needed" is reported. A comparison of `adaptive` against "the best fixed arm" could never let
   `adaptive` through: its shown set is always a subset of `fixed16`'s, so its all-found can never exceed
   `fixed16`'s.

**Attribution of the gate-1 gain, in three declared steps.**
- `fixed5` minus the §15 head: the wider shortlist and the retrain (on half the data).
- `fixed16` minus `fixed5`: more turns.
- `adaptive` against `fixed16`, in evidence and tokens: adaptivity.

**Expectations, written before the run.** From §15 and §18a, a shown turn costs about 149 tokens on LongMemEval
and about 73 on LoCoMo.
- **LoCoMo.** There is room in evidence: fusion all-found is 0.481 @5 and 0.646 @16 (§12; corrected after the run
  to 0.465 and 0.620, see §19a and the §12 footnote). But `fixed16` costs
  about 1.2k tokens, well under the onset and under the 2,000 guard. So adaptivity is probably not needed there,
  and the likely outcome is `fixed16` (or `fixed8`) to the reader.
- **LongMemEval.** `fixed16` costs about 2.4k tokens, past the 2,000 guard and near the onset. That is where
  adaptivity has its case. The five-turn head already reaches 0.590 against 0.706 for BM25 @16, so the room is
  smaller, and a no-go is plausible.

**Caveat (as in §15).** The gate measures the same evidence labels the head is trained to find. Whether more
labelled evidence, at more tokens, turns into more correct answers is the reader's test.

**Paid step (not approved; it needs the user's approval before it runs).** For each benchmark where an arm
passes, a reader cell like §15b/§18: that arm, the §15 head and FIFO + floor, with the reader frozen at 7b5fc30
and the official judge. Criteria: paired accuracy against the five-turn head, interval above 0; place on the
accuracy-against-tokens frontier against FIFO + floor; the §13a decomposition. A full-size stub run under
`guard.sh` precedes any launch. Rough cost, from §15b and §18: about $0.10–0.30.

### 19a. Result: more turns pay; the chooser does not (2026-10-07)

> **Record note (2026-10-16 audit, `research/audits/2026-10-16-thesis-direction-audit.md`, re-verified by sha256):** the ten `runs/lme_n4_head_f*_{a,b}/checkpoints/policy_best.pt` files are five distinct networks: head A f1 = f2, f3 = f4; head B f0 = f1, f2 = f3; head A f0 = head B f4. Per-fold results are therefore not ten independent trainings. Paired bootstraps are unaffected.

All runs are at 4b44dfd, not dirty. The heads were trained at 325dc7f. Report:
`runs/_pipelines/exp19_report.py` → `runs/exp19_gate_report.md`. There is no reader: a stub model answers, so the
real prompt is built and counted. All-found is paired against the §15 head with a 95% bootstrap interval.

**LongMemEval** (500 test questions; head A, with head B as a replicate within about 0.015):

| arm | all-found | Δ vs §15 head (95% CI) | prompt tokens | mean k |
|---|---|---|---|---|
| keep-last-0 + top 5 | 0.506 | −0.084 (−0.118, −0.052) | 1,551 | 5 |
| §15 head (5 of 16) | 0.590 | — | 742 | 5 |
| `fixed5` | 0.596 | +0.006 (−0.016, +0.026) | 715 | 5 |
| `fixed8` | **0.700** | **+0.110 (+0.084, +0.138)** | 1,050 | 8 |
| `fixed16` | 0.744 | +0.154 (+0.122, +0.186) | 2,566 | 16 |
| `oracle-k` (ceiling) | 0.744 | +0.154 (+0.122, +0.186) | 1,265 | 8.6 |
| `adaptive` | 0.674 | +0.084 (+0.056, +0.112) | 1,248 | 8.6 |

**LoCoMo** (categories 1–4, fold-0 heads, 10 conversations, clustered by conversation):

| arm | all-found | Δ vs §15 head (95% CI) | prompt tokens | mean k |
|---|---|---|---|---|
| keep-last-0 + top 5 | 0.204 | −0.132 (−0.146, −0.117) | 306 | 5 |
| §15 head (5 of 16) | 0.336 | — | 365 | 5 |
| `fixed5` | 0.372 | +0.036 (+0.020, +0.050) | 407 | 5 |
| `fixed8` | 0.424 | +0.088 (+0.070, +0.103) | 511 | 8 |
| `fixed16` | 0.465 | +0.129 (+0.108, +0.147) | 733 | 16 |
| `oracle-k` (ceiling) | 0.469 | +0.133 (+0.115, +0.148) | 601 | 11.5 |
| `adaptive` | 0.382 | +0.045 (+0.030, +0.059) | 430 | 5.7 |

**Verdicts under the pre-registered gate.**
- **LongMemEval: `fixed8` goes to the reader; adaptivity not needed.**
  - Gate 3 (ceiling) passes: `oracle-k` keeps all of `fixed16`'s evidence at 49% of its tokens.
  - Gate 4 fails: `fixed16` − `adaptive` = +0.070 (+0.048, +0.094), against the 0.02 bound.
  - `fixed16` fails gate 2 (2,566 tokens, past the 2,000 guard).
- **LoCoMo: `fixed5` goes to the reader by rule 4's fallback; adaptivity is a no-go by gate 3** (`oracle-k` saves
  18% of `fixed16`'s tokens, under the 25% line).
  - The fallback ("the smallest fixed k passing 1 and 2") was meant as Occam among arms of equal evidence. As
    written it rewards the fewest tokens under the guard, so it picks `fixed5` although `fixed16` adds +0.093
    all-found at 733 tokens.
  - The rule is not amended after the fact. `fixed16` goes into the same reader cell as a labelled exploratory
    arm. §19b corrects the fallback for future gates.
- **Attribution (LongMemEval, head A).**
  - Wider shortlist and retrain: +0.006 (−0.016, +0.026), neutral with half the training data. As the review
    noted, this is conservative for gate 1.
  - More turns (`fixed16` − `fixed5`): +0.148.
  - Adaptivity (`adaptive` − `fixed16`): −0.070, at 49% of the tokens.

**Why the chooser fails: calibration on the test folds** (head A's `oracle-k` cells; predicted / observed
all-found, n, per bin):

| k | 0–0.2 | 0.2–0.4 | 0.4–0.6 | 0.6–0.8 | 0.8–1 |
|---|---|---|---|---|---|
| 5 | 0.08 / 0.43, 94 | 0.29 / 0.41, 51 | 0.48 / 0.43, 63 | 0.71 / 0.58, 48 | 0.96 / 0.75, 244 |
| 8 | 0.09 / 0.44, 61 | 0.29 / 0.53, 49 | 0.50 / 0.66, 47 | 0.69 / 0.69, 62 | 0.96 / 0.79, 281 |
| 16 | 0.11 / 0.49, 45 | 0.31 / 0.64, 39 | 0.49 / 0.60, 40 | 0.71 / 0.69, 78 | 0.96 / 0.83, 298 |

- The chooser is overconfident at both ends.
- On LoCoMo it predicts about 1.0 for 1,449 of 1,535 questions (observed 0.37–0.47), so it collapses toward
  k = 5.
- Cross-fitting removed the in-sample bias, but a 21-feature network fit full-batch to 400 rows still overfits
  (train loss 0.15–0.18).

**LoCoMo arms are not exactly nested.** `oracle-k` is 0.004 above `fixed16` on LoCoMo, though equal by
construction on LongMemEval.
- The head's features include each turn's retrieval history: access count, time since access, and whether it was
  retrieved before.
- A LoCoMo conversation asks about 154 questions, so what an arm showed earlier changes its later rankings. At the
  first question of each conversation, `oracle-k`'s shown set is a subset of `fixed16`'s in all 10 conversations.
- Every arm answers the same 1,540 questions, so the pairing holds.

**Headroom @32** (free; `runs/exp19_headroom/`). All-found, BM25 on LongMemEval: @16 0.706 → @32 0.779.
Fusion on LoCoMo, with the shared retriever: @5 0.465, @8 0.518, @16 0.620, @32 0.713.

**Correction to the §19 expectation line.** It quoted §12's fusion figures (0.481 @5, 0.646 @16). Those came from
a copy of the fusion inside `headroom.py` that cut BM25 at the largest k tabulated. The tool now calls the shared
retriever (1e0a6a4), and the figures of record are 0.465 @5 and 0.620 @16 (see the §12 footnote).

**Train-part gate (overfitting check; `runs/exp19_gate_train_report.md`).**
- The fixed arms match the test folds (head A: `fixed8` +0.096 and `fixed16` +0.146 over the §15 head, against
  +0.110 and +0.154 on test), so the heads do not overfit.
- The §19 chooser does better on the train parts (0.712) than on test (0.674), as expected for an overfit chooser.

## 19b. Adaptive k with a one-feature chooser (N4b; 2026-10-07, pre-registered, free stage)

**Why.** The §19a ceiling is real on LongMemEval: `oracle-k` keeps `fixed16`'s evidence at 49% of its tokens
(1,265 against 2,566), +0.044 over `fixed8` for about 215 more tokens. The §19 chooser failed on calibration, not
for lack of room. This is one more attempt, with a chooser that cannot overfit 400 rows, and it is time-boxed.

**The chooser (declared before any run).**
- **Input:** one feature per k, the head's softmax mass on its top k over the 32 candidates.
- **Model:** for each k, Platt scaling on the logit of that mass (clamped to [1e-4, 1 − 1e-4]): P(all-found@k) =
  sigmoid(a_k × logit(mass_k) + b_k). That is two parameters per k, fit by binary cross-entropy on the cross-fit
  rows (head A on half B, head B on half A), as in §19. There is no network and there are no other features.
- **Data:** the cross-fit runs are repeated at the §19b commit so that the mass is logged. The heads are §19's,
  unchanged.

**Calibration gate, before the δ rule is applied.** The expected calibration error, in 5 equal-width bins
weighted by count, must be at most 0.10 for each k on the cross-fit rows of every fold. If it fails, §19b stops
and the ceiling is the finding.

**Arms and gate.**
- The same arms as §19 (`fixed5/8/16`, `oracle-k`, and `adaptive` with the new chooser), the same gates 1–4, and
  the same δ = 0.05.
- LongMemEval only. LoCoMo is a declared no-go for adaptivity (§19a, an 18% ceiling) and stays closed.
- Every arm is rerun at the §19b commit, so that all arms share one commit.
- Test-fold calibration in 5 bins is reported as in §19a.

**Correction of §19's fallback rule** (for this and later gates). When `adaptive` fails gate 4, the arm that goes
to the reader is the fixed k that passes gates 1 and 2 with the **highest all-found**, its tokens under the guard.
This is the Occam intended in §19; the §19 wording picked the fewest tokens instead.

**Time box.** If §19b's `adaptive` does not pass gate 4 on the test folds, adaptivity stops. The finding is then
the ceiling: `oracle-k` reaches `fixed16`'s evidence at 1,265 tokens, but no chooser tried here can find it.

**Paid step (not approved).** One reader session after §19b, proposed to the user with its cost:
- LongMemEval: `fixed8` (pre-registered by §19), §19b `adaptive` if it passes gate 4, the §15 head, and FIFO +
  floor at 5%.
- LoCoMo: `fixed5` (pre-registered), `fixed16` (exploratory, labelled), the §15 head, and FIFO + floor at 10%.

### 19b-a. Result: calibrated, closer, still short; adaptivity stops (2026-10-07)

All runs are at ac3bd67, not dirty; the heads are §19's. Report: `runs/exp19b_gate_report.md`. The fixed arms,
`oracle-k` and the references reproduce §19a exactly at the new commit.

- **Calibration gate passes.** The expected calibration error on the cross-fit rows is 0.013–0.060 per k and
  fold, against the 0.10 limit. The fitted slopes are small (a ≈ 0.7–0.9 at k = 5, about 0.3 at k = 16), so the
  chooser separates easy from hard questions only weakly. On the test folds it is calibrated (top bin at k = 5:
  0.87 predicted, 0.87 observed).
- **Gate 4 fails on both heads**, narrowly:

  | head | `adaptive` all-found | tokens | mean k | `fixed16` − `adaptive` (95% CI) | 0.75 × `fixed16` tokens |
  |---|---|---|---|---|---|
  | A (primary) | 0.718 | 1,437 | 10.3 | +0.026 (+0.012, +0.042) | 1,925 |
  | B (replicate) | 0.720 | 1,703 | 11.5 | +0.020 (+0.008, +0.034) | 1,959 |

  The token condition holds; the evidence condition (upper bound ≤ 0.02) does not.
- **Verdict.** By the corrected fallback (the fixed k passing gates 1 and 2 with the highest all-found), `fixed8`
  goes to the reader on LongMemEval, as in §19a; `fixed16` fails the guard.
- **Time box reached: adaptivity stops.** The finding is the ceiling. `oracle-k` keeps `fixed16`'s evidence at
  1,265 tokens (49%). The best chooser tried here, one feature with Platt scaling, gets within 0.026 at 1,437
  tokens (56%), and the first, a 21-feature network, falls 0.070 short.
- **Against `fixed8`, descriptively.** `adaptive` is +0.018 all-found at +387 tokens. It is not sent to the
  reader, since it did not pass its gate.

**Note for the write-up (review).** Head B's `adaptive` (0.720 at 1,703 tokens) sits on the frontier between
`fixed8` (0.700 at 1,050) and `fixed16` (0.744 at 2,566). The chooser works in the right direction; the gap to the
ceiling is discrimination, not calibration. The LoCoMo non-nesting (§19a) also goes into the thesis limitations:
within a conversation, the head's ranking depends on the controller's own past picks.

## 19c. Reader test of the §19 picks (2026-10-07; approved by the user, about $0.25–0.50; criteria written before launch)

**Cells** (`configs/sweeps/exp19_paid/`; one pod session; every row on one host with a fresh cache, so all rows
are paired; reader frozen at 7b5fc30; official judge):
- **LongMemEval**, 500 questions (470 non-abstention scored, as §15b): `fixed8` (head A, 32 candidates), the §15
  head, FIFO + floor at 5% with the 3k target, and `fixed16` (EXPLORATORY; the user approved it as an extra arm,
  about $0.05). `fixed16` against `fixed8` (+0.044 evidence at 2.4× the tokens, near the onset) is a direct test of
  context rot. Like LoCoMo's `fixed16`, it gets no pass or fail verdict, only its paired difference against
  `fixed8` and a frontier point.
- **LoCoMo**, categories 1–4, fold-0 heads, fusion search: `fixed5` (head A), `fixed16` (head A, EXPLORATORY), the
  §15 head, and FIFO + floor at 10%.

**Criteria (pre-registered).**
1. **LongMemEval, primary:** `fixed8` minus the §15 head, accuracy, paired by question, 95% interval above 0.
2. **LoCoMo, primary:** `fixed5` minus the §15 head, accuracy, clustered by conversation, 95% interval above 0.
3. **LoCoMo, exploratory:** `fixed16` gets no pass or fail verdict, only a point on the accuracy-against-tokens
   frontier beside FIFO + floor at 10%.
4. **For every arm:** the §13a decomposition (P(all in view), P(correct | in view), P(correct | not), the unknown
   rate), with prompt tokens beside every accuracy.

**Expectation, written before launch:** positive on both primaries, size unknown.

**Safeguards.**
- A full-size stub run under `guard.sh`, with peak RSS recorded, precedes the launch. Done at 316c35e: all 60
  cells ran with no failures. Peak RSS per worker is 0.27 GB on LongMemEval and 0.84 GB on LoCoMo (the bge model).
  Workers: LoCoMo 10 (≈ 8.4 GB) beside LongMemEval 4 (≈ 1.1 GB), under the review's 10 GB line.
- The §15 head and FIFO + floor rows are new generations on the new host. They are reported beside §15b/§18a as a
  same-host replicate and do not overwrite the earlier figures.
- Hard stop at 1 hour of pod time (about $0.49). Estimate from the measured §18a and §15b throughput: about
  $0.30.
- The pod is verified terminated with list-pods afterwards.

### 19c-a. Result: eight turns win on LongMemEval; sixteen lose (context rot); on LoCoMo the primary misses, sixteen beat FIFO (2026-10-07)

Pod v7ddalhf74kxrh (A40, $0.49/h): created 21:51:40 UTC, reader ready 21:56:43, all 60 cells done 22:35:03,
terminated about 22:36, verified with list-pods. About 0.74 h, about $0.36. Sweeps `exp19_paid_*` at 316c35e; reader
frozen at 7b5fc30; official judge. Report: `runs/_pipelines/exp19_paid_report.py` → `runs/exp19_paid_report.md`.
Figure: `docs/research/figures/exp19c_frontier.png`.

**LongMemEval** (470 non-abstention questions, paired by question):

| row | accuracy | Δ vs §15 head (95% CI) | P(all in view) | P(correct \| in view) | P(correct \| not) | unknown | prompt tokens |
|---|---|---|---|---|---|---|---|
| §15 head (5 of 16) | 0.500 | — | 0.591 | 0.683 | 0.234 | 0.217 | 745 |
| **`fixed8` (primary)** | **0.545** | **+0.045 (+0.011, +0.079)** | 0.702 | 0.676 | 0.236 | 0.151 | 1,049 |
| `fixed16` (exploratory) | 0.487 | −0.013 (−0.053, +0.026) | 0.747 | 0.575 | 0.227 | 0.219 | 2,569 |
| FIFO + floor, 5%, 3k target | 0.432 | −0.068 (−0.111, −0.028) | 0.498 | 0.684 | 0.182 | 0.338 | 2,964 |

**LoCoMo** (categories 1–4, 1,540 questions per arm, clustered by conversation):

| row | accuracy | Δ vs §15 head (95% CI) | P(all in view) | P(correct \| in view) | P(correct \| not) | unknown | prompt tokens |
|---|---|---|---|---|---|---|---|
| §15 head (5 of 16) | 0.244 | — | 0.334 | 0.592 | 0.068 | 0.490 | 365 |
| **`fixed5` (primary)** | 0.267 | +0.023 (−0.001, +0.048) | 0.372 | 0.586 | 0.078 | 0.431 | 407 |
| `fixed16` (exploratory) | **0.344** | +0.101 (+0.077, +0.122) | 0.466 | 0.595 | 0.125 | 0.305 | 734 |
| FIFO + floor, 10% | 0.295 | +0.052 (+0.023, +0.081) | 0.427 | 0.538 | 0.115 | 0.277 | 3,087 |

**Verdicts under the pre-registered criteria.**
- **LongMemEval primary: pass.** `fixed8` beats the §15 head by +0.045 (+0.011, +0.079) at 1,049 tokens.
  - It also beats FIFO + floor by +0.113 (+0.072, +0.155) at about a third of the tokens.
  - The gain is evidence: P(all in view) rises from 0.591 to 0.702, and P(correct | in view) stays level (0.683 →
    0.676), because 1,049 tokens is still below the onset.
- **LoCoMo primary: no pass.** `fixed5` minus the §15 head is +0.023 (−0.001, +0.048); the interval touches 0. Its
  extra evidence (+0.038 in view) is too small to show in accuracy.

**Exploratory rows (no verdict; post hoc in the sense of §19a).**
- **Context rot, measured directly (LongMemEval `fixed16` against `fixed8`): −0.057 (−0.096, −0.019).**
  - `fixed16` has more evidence in view (0.747 against 0.702).
  - But at 2,569 tokens, P(correct | in view) falls from 0.676 to 0.575.
  - Same head, same ranking, the same question set. The only change is 8 more turns, and the reader answers worse
    with the evidence in front of it. This is the mechanism the thesis argues (§13a), now on a controlled pair.
- **LoCoMo `fixed16`: 0.344 at 734 tokens, against FIFO + floor at 10%, 0.295 at 3,087.** The difference is +0.049
  (+0.034, +0.062).
  - This is the first row on LoCoMo above the FIFO frontier, at about a quarter of its tokens.
  - On LoCoMo, 16 short turns are still far below the onset, so more evidence converts: P(correct | in view) stays
    at 0.595.
  - It is exploratory. It was not the arm the §19 rule picked, so it is evidence, not a claim.

**Same-host replicate** (new generations on the new host; the earlier figures stand):

| row | earlier | §19c |
|---|---|---|
| LongMemEval §15 head | 0.513 at 745 (§15b) | 0.500 at 745 |
| LongMemEval FIFO + floor 5%, 3k target | 0.438 at 3,158 (§13a, all 500 questions with abstention) | 0.432 at 2,964 (470 questions) |
| LoCoMo §15 head (LongMemEval-trained, fold 0) | 0.242 at 365 (§18a) | 0.244 at 365 |
| LoCoMo FIFO + floor 10% | 0.299 at 2,917 (§13b) | 0.295 at 3,087 |

The differences are within the run-to-run variation of §15c. (The FIFO prompt tokens differ a little because §13
counted all questions, including refusal-scored ones.)

## 19d. Confirming the context-rot pair (2026-10-07; approved by the user, about $0.15; written before launch)

> **Record note (2026-10-16 audit, `research/audits/2026-10-16-thesis-direction-audit.md`, re-verified by sha256):** the ten `runs/lme_n4_head_f*_{a,b}/checkpoints/policy_best.pt` files are five distinct networks: head A f1 = f2, f3 = f4; head B f0 = f1, f2 = f3; head A f0 = head B f4. So the "second head on a second host" on fold 4 was the same network as head A on fold 0, not an independent one. Paired bootstraps are unaffected.

**Question.** §19c found, as an exploratory row, that the same head showing 16 turns instead of 8 puts more
evidence in view but answers worse: −0.057 (−0.096, −0.019), with P(correct | in view) falling from 0.676 to
0.575. That was one head (A), one run, one host. Does the effect hold with the other cross-fitted head (B) on
another pod?

**Cells** (`configs/sweeps/exp19d/`): LongMemEval, the five test folds, head B (`lme_n4_head_f*_b`, 32 BM25
candidates), `fixed8` and `fixed16`. Both run in one new pod session with a fresh cache
(`cache/generations_qwen7b_vllm_v2_exp19d`), so every answer is a new generation. Reader frozen at 7b5fc30; official
judge. The pod's id, data center and CUDA version are recorded.

**Criterion (pre-registered).** On 470 non-abstention questions, paired by question:
- **Confirmed** if head B's `fixed16` minus `fixed8` accuracy has a 95% bootstrap interval entirely below 0.
- **Not confirmed** otherwise, and the §19c pair stays exploratory in the thesis, reported with this result
  beside it.
- Reported for both arms: the §13a decomposition (P(all in view), P(correct | in view), the unknown rate) and
  prompt tokens. The mechanism reading needs P(all in view) higher for `fixed16`, and P(correct | in view) lower.

**Expectation, written before launch.** Negative, smaller than §19c's −0.057 (regression to the mean), and the
interval may touch 0.

**Safeguards.** A full-size stub run under `guard.sh` first, with its peak RSS recorded. Hard stop at 45 minutes of
pod time (about $0.37). The pod is verified terminated with list-pods.

### 19d-a. Result: not confirmed; the mechanism points the same way (2026-10-07)

Pod kzblkszpawlunn: A40 at $0.49/h, CA-MTL-1, CUDA 12.8. §19c's pod was CUDA 13.0 in the same data centre, so per
the review's rule this is a "second host". Created 23:22:42 UTC, reader ready 23:26:55, all 10 cells done
23:33:51, terminated about 23:34, verified with list-pods. About 0.2 h, about $0.10. Sweeps `exp19d_f*` at 0ecceb0;
report `runs/_pipelines/exp19d_report.py` → `runs/exp19d_report.md`.

| arm (head B) | accuracy | P(all in view) | P(correct \| in view) | P(correct \| not) | unknown | prompt tokens |
|---|---|---|---|---|---|---|
| `fixed8` | 0.551 | 0.685 | 0.708 | 0.209 | 0.145 | 983 |
| `fixed16` | 0.526 | 0.745 | 0.620 | 0.250 | 0.196 | 2,617 |

- **Pre-registered criterion: NOT CONFIRMED.** `fixed16` minus `fixed8` is −0.026 (−0.066, +0.015) over 470
  questions; the interval includes 0.
- **The mechanism points the same way as §19c.**
  - More evidence in view: +0.060 here, +0.045 in §19c.
  - Lower accuracy when the evidence is in view: 0.708 → 0.620 (−0.088), against 0.676 → 0.575 (−0.101) in §19c.
  - The net accuracy effect is smaller, consistent with regression to the mean, as the expectation said.
- **Reading for the thesis.** The §19c pair stays exploratory, reported with this result beside it. Two heads on two
  hosts agree in sign on accuracy (−0.057 and −0.026), and in both the decomposition shows P(correct | in view)
  falling by about 9–10 points at about 2.6k tokens. Only §19c's own interval excludes 0. A pooled estimate was not
  pre-registered and is not claimed.

## 20. A write action: the reader summarises the turns it cannot afford to show (N5; 2026-10-08, DRAFT pre-registration)

**Status.** The text was agreed with the review at 1f1bdae. The user approved the gate stage (about $0.20–0.30)
on 2026-10-08. The reader stage needs a separate approval.

**Build and dry run.**
- Code at fe7051c: CONSOLIDATE gains a `write_note` mode (the note is added and its sources are untouched), the
  floor head gains rank-group and session notes, and the reader can log its prompt.
- Full-size stub run (stub writer, stub reader): 90 of 90 cells ran, with no failures. Peak RSS per worker is 0.27
  GB on LongMemEval and 0.86 GB on LoCoMo. Workers: LoCoMo 10 beside LongMemEval 4 (about 9.7 GB).
- The report (`runs/_pipelines/exp20_report.py`, written before the real notes) ran end to end on the stub output.
  - On the raw arms, which are real even in the stub run, strict containment and all-found agree only loosely:
    P(contained | all-found) = 0.53, against 0.30 without. Raw `fixed16` minus `fixed8` is +0.060 (+0.038, +0.083),
    so the +0.05 bar is about what eight more raw turns buy.
  - The gate stays as pre-registered. That calibration figure is reported beside the verdict, and a near miss is
    read with the content-word recall and the by-type table.
- In the gate stage the reader does not answer. A stub builds and logs the prompt; only the note writer calls the
  model. The host and the note cache (`cache/notes_qwen7b_exp20`, keyed by prompt) are recorded, so a later full
  write of every session can be checked against a sample.

**Question.** §19 showed that on LongMemEval eight turns is about the most the reader uses well. Showing 16
added evidence but no accuracy (§19c, §19d). The head ranks 32 candidates, so 24 of them are never seen. Can a
*written* memory item, a short summary of the turns the reader cannot afford, add the evidence without the cost
of the raw turns?

**The action (declared now).**
- **Writer:** the frozen reader (Qwen2.5-7B-Instruct, served as in §19c, temperature 0), with the existing
  `LLMConsolidator` prompt (`memctl/memory/compress.py`), unchanged: "Merge the notes below into one note of at
  most {limit} words. State each fact once. Keep every name, number and identifier. Reply with the merged note
  only."
- **Input:** the chosen turns as labelled text ("speaker (date): content"), so dates survive.
- **Chunking and length cap:** ranks 9–32 are summarised in three groups of 8 by rank (9–16, 17–24, 25–32), one
  note of at most 100 tokens each (the limit in the prompt, with the output truncated to 100), so about 300 tokens in
  all. One note for 24 turns (about 3,500 input tokens) would lose most facts to the cap and fail the gate for the
  cap, not the idea. Note lengths are reported.
- **Input limit:** writer inputs are cut at 12k tokens. One LongMemEval session in 19,195 is longer.
- **Question-blind.** The writer does not see the question, so a summary is not a second reading of the
  question. The *set* summarised in `fixed8+sum` does depend on the question, since it comes from the head's
  ranking: that arm tests **compression at retrieval time**. Its notes are written at the question and never reused.
  That is not yet a memory write in the Memory-R1 / MEM1 sense (written during the stream, independent of the
  question, reused across questions). The `fixed8+sess` arm below is that write. A question-aware writer is a
  possible later arm.
- **Placement:** one item shown in place of the turns it summarises, marked as a note in the prompt.

**Arms** (all on the §19 head A with 32 BM25 candidates; LongMemEval five test folds; tokens estimated from §19c):

| arm | shown | expected prompt tokens |
|---|---|---|
| `fixed8` (reference, the current best) | raw ranks 1–8 | 1,049 (measured) |
| `fixed8+sum` | raw ranks 1–8 + three notes (≤ 100 tokens each) for ranks 9–16, 17–24, 25–32 | ≤ about 1,370 |
| `fixed8+sess` | raw ranks 1–8 + stored session notes (below) | ≤ about 1,370 |
| `sum16` | two notes (≤ 100 tokens each) for ranks 1–8 and 9–16, no raw turns | ≤ about 390 |
| `fixed16` (reference) | raw ranks 1–16 | 2,569 (measured) |
| FIFO + floor, 5%, 3k target (reference) | — | 2,964 (measured) |

**The stream-time write: `fixed8+sess`.**
- Each session gets one note (≤ 100 tokens, the same writer prompt, question-blind), written once and stored.
- At a question, the raw top 8 are shown with the stored notes of the sessions in which ranks 9–32 fall. At most 3
  notes are shown, chosen by the head's best rank within each session.
- A note depends only on its session (question-blind, temperature 0, cached by session id). So writing only the
  notes some test question uses gives exactly the notes a full stream-time pass would write. On LongMemEval that
  is at most 1,500 notes (3 per question) instead of all 19,195 sessions, which would cost about $0.50 or more. The
  saving changes cost, not content. LoCoMo has 272 sessions in all, and every one is written.
- Sessions average about 2,200 tokens on LongMemEval (95th percentile about 3,900), so its input is longer than a
  group of 8 turns.
- **A limitation of this first version.** Notes are evicted at the next step and never become search candidates. A
  session note is "stored" and reused only through the head's mapping from rank to session (and the writer's cache),
  not by being found through search.

**Gate-stage spend and incidents (2026-10-08), recorded before any gate number is read.**

| step | pod | time (UTC) | cost | what happened |
|---|---|---|---|---|
| approved | — | — | $0.20–0.30 | the estimate treated note writing like judge calls |
| pod 1 | phuj1d8j8set0t, CUDA 13.2 | 00:13:53–00:56 | ≈ $0.34 | stopped at the local deadline about half done; note writing is sequential per question (2–3 notes of about 94 tokens), and a sweep runs cells, not episodes, in parallel |
| pod 2 (+$0.20 approved) | 82brrqh9fmttid, CUDA 13.0 | 01:33:42–01:58:15 | ≈ $0.20 | cells started on pod 1 refused to resume, because the pod URL was part of the saved config identity; only folds 2–4 and new LoCoMo cells advanced |
| pod 3 (≤ $0.20 approved) | f8t5ck7gw771e9, CUDA 13.0 | 02:03:16–02:25 | ≈ $0.18 | unfinished cells were quarantined (`runs/_aborted_exp20`) and rerun from scratch with cached notes; stopped at the local deadline |
| **total** | | | **≈ $0.72** | gate budgeted at $0.30 |

- **Coverage at stop.**
  - LongMemEval `fixed8_sum`: 422 of 500 questions (fold 1 has 46 of 100; folds 2–4 have 92 each).
  - `fixed8_sess`: 341 of 500.
  - `sum16`: 456 of 500.
  - All raw arms are complete (the raw cells of folds 2–4 and LoCoMo ran on CPU, for free).
  - LoCoMo `sum16`: 7 of 10 conversations; every other LoCoMo arm is complete.
- **Note-host agreement** (the review's condition). 50 notes written on pod 1 were rewritten on pod 2: 30 of 50
  match exactly, and the mean word overlap (Jaccard) is 0.898 (`runs/exp20_note_agreement.json`). Pod 1 wrote 6,063
  notes; the rest came from pods 2 and 3. Every arm of a question uses the same cached note, so this does not bias
  arm comparisons.
- **Fix.** `memctl/runlog.py` now leaves serving endpoints (`base_url`, `timeout_s`) out of the config identity used
  for resuming, with a test: a cell resumes against a new URL but not against a new experiment. Both lessons
  (estimate from the measured per-call latency of the call type; test a resume against a changed endpoint) are in
  the pre-launch checklist.
- **The gate is incomplete.** The pre-registered verdict needs all 470 questions. Whether to finish (a further
  spend) or to stop is the user's decision. No gate number has been computed.

**The free-gate measure (a reader-free proxy, declared now).** The gold evidence labels are turn ids, and a summary
is not a turn, so all-found cannot score summary arms. The proxy is **gold-answer containment**: the normalised
gold answer (`memctl.metrics.normalize_answer`, as in the F1 code) appears as a contiguous word sequence in the
normalised prompt text.
- It is reported beside all-found for the raw arms, so the two measures can be calibrated where both exist.
- Abstention questions are excluded (470 questions).
- It is conservative for summary arms: a paraphrase ("four hours" for "4 hours") is not counted.
- Long or descriptive golds (for example preference questions) are rarely contained in any arm, so containment is
  also reported by question type, with the preference and other descriptive types shown separately.
- **Descriptive only:** content-word recall of the gold answer (normalised tokens minus stopwords; recall ≥ 0.8
  counts). It shows how much paraphrase loss there is rather than guessing it. It is reported beside strict
  containment and all-found on every arm, by type, and does not gate.

**Gate (pre-registered, LongMemEval; the same rule for `fixed8+sum` and `fixed8+sess`, each against `fixed8`).**
An arm goes to the reader only if both hold:
1. Containment of `fixed8+sum` minus `fixed8`, paired by question, is at least +0.05, with a 95% interval
   entirely above 0.
2. Its mean prompt tokens are at most 1.5 × `fixed8`'s (about 1,574).

If both pass, both go to the reader. No Holm correction is applied at the gate, because the reader stage is
where the claim is made.

`sum16` is reported against `fixed8` and `fixed16` as description; it does not gate.

**LoCoMo** (categories 1–4, fold-0 heads, fusion search): the same arms, exploratory, as in §19c. Containment and
tokens are reported without a verdict.

**Reader stage (if the gate passes; criteria declared now).**
- **Primary:** each passing arm minus `fixed8`, accuracy on LongMemEval, paired, 95% interval above 0. Holm is
  applied over the arms that reach the reader.
- Also reported: the §13a decomposition for the raw arms, the unknown rate, and prompt tokens for every arm.
- LoCoMo stays exploratory.

**Caveat.** Memory-R1's written facts came from GPT-4o-mini; here the writer is the 7B reader itself, so smaller
gains are expected. **Expectation:** positive on containment, size unknown; accuracy is the reader's test.

**Cost (NEEDS SPEND; estimated from §19c's measured throughput).**
- **Retrieval-time notes** (`fixed8+sum` three per question, `sum16` two per question):
  - LongMemEval: 500 × 5 = 2,500 calls, inputs about 1–1.5k tokens.
  - LoCoMo: 1,540 × 5 = 7,700 calls, short inputs.
- **Session notes:** at most 1,500 on LongMemEval (inputs about 2.2k tokens) and 272 on LoCoMo.
- **Gate stage:** about 12,000 short generations (≤ 100 tokens) plus start-up, about 0.4–0.6 h, **about
  $0.20–0.30**. The session arm adds about $0.03–0.05 of that, under the $0.30 line, so it runs on both benchmarks.
- **Reader stage (if the gate passes):** summaries are cached, so it is answers and judging only, at about §19c's
  size. **About $0.30–0.45.**
- Before either: a full-size stub-backend run of the whole pipeline (stub summaries) under `guard.sh`, with peak
  RSS recorded.

### 20a. Result: both note arms fail the gate; stored session notes come closest (2026-10-08)

All gate cells are at fe7051c. The notes were written by four pods, the last of which only filled the cache-only
pass's 1,229 missing notes (`memctl/fill_cache.py`, pod sr6a8clhsm3b3f, 02:30:00–02:41 UTC, about $0.09).
Unfinished cells were quarantined (`runs/_aborted_exp20`, `runs/_aborted_exp20_b`) and rerun from scratch from the
cache on CPU. 14,292 note actions were all applied, and no prompt holds a placeholder. **Gate-stage spend: about
$0.81** (pods 1–3, about $0.72, plus the fill, about $0.09), against $0.20–0.30 approved at first; every extension
was approved by the user. Report: `runs/_pipelines/exp20_report.py` (written before the notes) →
`runs/exp20_gate_report.md`.

**LongMemEval** (470 questions; the gate):

| arm | containment | Δ vs `fixed8` (95% CI) | content recall ≥ 0.8 | all-found | prompt tokens | note tokens (mean) |
|---|---|---|---|---|---|---|
| `fixed8` | 0.423 | — | 0.530 | 0.702 | 1,049 | — |
| `fixed8+sum` | 0.451 | +0.028 (+0.015, +0.043) | 0.566 | n/a | 1,350 | 51 |
| `fixed8+sess` | 0.466 | +0.043 (+0.026, +0.062) | 0.574 | n/a | 1,348 | 42 |
| `sum16` | 0.185 | −0.238 (−0.281, −0.196) | 0.211 | n/a | 414 | 62 |
| `fixed16` (reference) | 0.483 | +0.060 (+0.038, +0.083) | 0.626 | 0.747 | 2,569 | — |
| FIFO + floor (reference) | 0.440 | +0.017 (−0.017, +0.051) | 0.577 | 0.498 | 2,964 | — |

**Verdict under the pre-registered gate: both note arms FAIL.**
- `fixed8+sum`: +0.028, below +0.05.
- `fixed8+sess`: +0.043, below +0.05.
- Both intervals sit above 0, and both arms are within the token limit (1,574).
- No reader stage is run.

**Reading** (with the calibration figure beside the verdict, as agreed). On the raw arms, P(contained | all-found)
= 0.53 against 0.30 without, so containment is a loose proxy.
- **The notes add evidence, but less than the bar.** The bar was set at about what 8 more raw turns buy (+0.060).
- **Stored session notes come closest.** They are question-blind and written once per session. Per token they
  are far more efficient than raw turns: +0.043 for about 300 extra tokens, against +0.060 for about 1,520.
- **The gains are where evidence is spread.** Multi-session goes from 0.248 to 0.339 with session notes (raw
  `fixed16`: 0.364), and temporal from 0.181 to 0.228. Single-session types barely move, and preference golds are
  never contained in any arm.
- **Notes cannot replace raw turns.** `sum16`, notes only, loses most of the evidence (−0.238).
- **Content-word recall moves like containment**, so paraphrase is not hiding a larger gain.

**LoCoMo (exploratory, no verdict):**
- `fixed8+sum` +0.042 (+0.032, +0.053) at 753 tokens, and `fixed8+sess` +0.032 (+0.023, +0.041) at 656. Both are
  above raw `fixed16` (+0.029 at 733).
- FIFO + floor at 10% is +0.045 at 3,087 tokens.
- On LoCoMo's short turns, notes match or beat 8 more raw turns at the same or fewer tokens.

**Summary for the write-up.**
- **Both halves.** The stream-time session note gave 70% of raw `fixed16`'s containment gain (+0.043 against
  +0.060) for a fifth of its extra tokens (about 300 against 1,520). On LoCoMo, both note forms beat raw `fixed16`
  at equal or fewer tokens. The write action is the most token-efficient evidence measured here, and it fell short
  of a bar set at "what eight raw turns buy".
- **Proxy caveat.** P(contained | all-found) is 0.53 on the raw arms, so containment under-reads evidence in view
  for every arm, the note arms included.
- **Declared limitation.** In this version notes never become search candidates.
- **Process result.**
  - Spend: $0.81 over four pods, for a $0.30 budget. There were three causes:
    1. The estimate used judge-call throughput for sequential note writing.
    2. Resumes were refused when the pod URL changed.
    3. Cells run their questions in sequence, so the slowest cell sets the pace.
  - Fixes: estimate from the measured latency of the call type; resumes ignore the serving endpoint (78b76f7);
    and a cache-only pass lists the exact missing generations, which `fill_cache` then writes in parallel (22fdbc4).
    The last pod took 11 minutes, about $0.09.

**What this means for the plan.** A first, untrained write action helps in the right places but does not clear
the pre-registered bar with the 7B writer. The stream-time session note is the better form. That fits the next
step: the sequential task, where a controller learns *when* to write.

## 21. A sequential task on real conversations: questions during the stream, and questions that depend on the model's own answers (N6; 2026-10-08, DRAFT pre-registration)

**Status: draft, under review.** Nothing is built until the text is agreed. The first gate is free (scripted reader,
no model).

**Why.** Every real-text result so far (§13–§20) asks all questions after the history has ended. Then the only
decision that matters is at the question, which is why query-blind control failed (§14a) and GRPO had nothing to
assign credit over (§16a). The plan's next step is a task where an early memory decision changes a later outcome,
on real text, following MEM1's recipe: questions asked during the stream, plus questions whose answer depends on
the model's own earlier answer.

**Amended after review (before any build).** As first drafted, with an unlimited searchable archive, every
early decision except deletion could be repaired at the question, and B quoted A, so the gate would very likely
read "dead". The amendment makes memory **lossy**, as a declared modelling choice: the setting of a bounded-memory
agent (MEM1's constant-size state; MemoPilot's updated memory).

**The task (LoCoMo categories 1–4 primary; LongMemEval composed 4-question episodes as corroboration).**
1. **Questions during the stream.** Each question is asked d sessions after its last evidence session, at most at
   the end. **Amended after the smoke test (one conversation, before any gate run):** with d = 1 every
   single-evidence question was asked inside the w = 2 window, so only 11 of 81 questions depended on notes. d is
   now 1–6, fixed per question by a hash of its id. On conversation 1 that makes 64 of 81 questions old, with ages
   1–2: 30, 3–5: 33, and 6+: 18.
2. **The lossy rule.**
   - Raw turns older than w = 2 sessions leave memory. They are gone, not archived or searchable.
   - What survives past w is only what the controller wrote: **one note per session**, written when the session
     ends, within a note budget of 100 tokens. The stored answer items also survive.
   - The controller's stream-time decision is therefore *which* of the session's content goes into its note. That
     decision is question-blind, and its consequences fall only on questions asked w or more sessions later. So it
     is an early decision with a delayed, unrepairable consequence by construction, and credit assignment is
     defined on this task.
3. **Notes are extractive for the free gate.** A note is a set of the session's turns, chosen within 100 tokens and
   stored as one searchable item. The scripted reader can then score evidence exactly by turn id. Abstractive notes
   (the §20 writer) need the model, so they belong to the paid stage.
4. **At a question**, the reader sees the top 8 by fusion search over everything that survives: the raw turns of
   the last w sessions, all notes and the answer items. A retrieved note brings all its turns into view.
5. **Recall-of-own-answer questions (B), now secondary.** d = 3 sessions after each in-stream A: "Earlier you were
   asked: '<A>'. What did you answer?" B is scored against A's *actual* answer (pure retention), with scoring
   against A's gold reported as description.

**The free oracle gate (decides whether any training happens).**
- **Reader:** scripted. A question is correct iff all of its evidence is in view. No model, no cost.
- **Note policies** (each picks turns question-blind within 100 tokens per session, except the oracle):
  - **oracle:** the session's gold evidence turns of future questions whose evidence lies in that session, most
    questions served first, within budget (hindsight);
  - **rules:** salience (density of numbers and capitalised words), the first turns of the session, the shortest
    turns, and random (seeded). The best rule at each budget is the comparison.
- **Learnable-headroom row (added at review, declared before the run): `learned`.**
  - A cross-fitted logistic model, trained on the other nine conversations, scores each turn's chance of being
    needed by a question asked w or more sessions later. Its features are question-blind: log length, salience,
    digit and capitalised-word counts, position in the session, whether the turn asks a question, speaker, and the
    turn's bge-small embedding. The session's top-scoring turns fill the note.
  - Its AUC and the base rate (the share of turns that are evidence for a later question, per conversation) are
    reported. §11's base rate was 1.2%; with about 150 questions per conversation this one should be far higher.
  - **Reading, declared now.** The oracle gap is the headroom. The learned row shows how much of it a supervised
    predictor already reaches. A GRPO stage (§22) is justified only by what lies between the learned row and the
    oracle. If the learned row closes most of the gap, the right §22 is supervised, not RL.
- **Primary measure:** accuracy on questions whose oldest evidence is older than w sessions at the time they are
  asked (only these depend on the notes). Also reported:
  - accuracy for all questions;
  - evidence survival (in memory at all, before search);
  - everything by evidence age in sessions (1–2, 3–5, 6+), the central axis under the lossy rule.
- **Control row:** the unlimited archive (no loss, every raw turn searchable). The oracle-minus-rule gap should
  vanish there, consistent with §13–§20; that is a finding in itself.
- **Gate, pre-registered:** on LoCoMo, oracle minus the best rule, on questions with evidence older than w, is at
  least 0.10 with a conversation-clustered 95% interval above 0. Otherwise credit assignment is dead on this task
  too, and §21 says so before any training. LongMemEval composed episodes are corroboration only.
- **Expectation, written before the run:** a large gap, since 100 tokens holds about 3–5 of a session's ~20 turns,
  so which ones matters, and rules cannot know which.
- w = 2 is fixed. A second w is descriptive, only if free.

**Looking ahead (§22, not part of this gate).** Hindsight credit is directly available here. A session's note is in
view or not at every later question, so its write decision can be credited from the outcomes of exactly the
questions it served. That per-write hindsight advantage is the natural GRPO variant, and cheap: the controller is
small and the reader frozen.

**Paid steps (not proposed yet).** A reader version of the gate (the real 7B answering A and B), and any training,
come only after the free gate passes, each with its own pre-registration and cost.

### 21a. Result: the oracle gate passes on LoCoMo; a supervised predictor closes about a fifth of the gap (2026-10-08)

Free and scripted (no model). `memctl/analysis/lossy_gate.py` at 93f4fe6 → `runs/exp21_gate.json`; table
`runs/_pipelines/exp21_report.py` → `runs/exp21_gate_report.md`. Settings: w = 2, notes of 100 tokens, top 8 by
fusion search, 10 conversations, 1,535 questions, 1,249 of them "old" (oldest evidence at least w sessions back).

| note policy | accuracy, old questions (lossy) | evidence survives (lossy) | accuracy (unlimited archive, control) |
|---|---|---|---|
| oracle (hindsight) | **0.286** | 0.348 | 0.552 |
| learned (cross-fitted) | 0.147 | 0.167 | 0.542 |
| first turns (best rule) | 0.109 | 0.131 | 0.536 |
| random | 0.051 | 0.060 | 0.535 |
| salience | 0.039 | 0.044 | 0.534 |
| shortest | 0.015 | 0.016 | 0.536 |

- **Gate: PASS.** Oracle minus the best rule (first turns) is **+0.177 (+0.126, +0.224)**, conversation-clustered,
  against the pre-registered +0.10. Credit assignment is defined and has headroom on this task.
- **Learnable headroom.** `learned` minus first turns is +0.038 (+0.010, +0.068); oracle minus `learned` is +0.139
  (+0.098, +0.179).
  - The supervised predictor reaches about a fifth of the gap, although it separates needed turns well (AUC
    0.74–0.84 by conversation; §11 was about 0.65).
  - The base rate is 16–31% of turns, against §11's 1.2%.
  - By the reading declared before the run, **the space between the learned row and the oracle is what a GRPO
    stage (§22) would have to earn**, and it is large.
- **The control behaves as predicted.** With an unlimited archive, the gap almost vanishes: oracle minus first
  turns is +0.016 (+0.005, +0.027). This agrees with §13–§20: when the archive is perfect, early decisions can be
  repaired at the question.
- **By evidence age** (lossy accuracy; oracle / learned / first turns):

  | age | n | oracle | learned | first turns |
  |---|---|---|---|---|
  | 1–2 | 534 | 0.543 | 0.474 | 0.451 |
  | 3–5 | 577 | 0.355 | 0.191 | 0.144 |
  | 6+ | 424 | 0.179 | 0.083 | 0.050 |

  The gap grows with age, as the lossy rule implies.
- **Even the oracle loses most of the evidence:** only 0.348 survives. 100 tokens hold about 3–5 turns, and a
  session often serves more questions than that. The note budget, not foresight, limits the oracle.
- **B (retention of the stored answer, d = 3): 1.000.** Search finds the answer item from B's quoted question, as
  the draft feared. B is uninformative here, as the review expected when it was made secondary.
- **Added at review, after the gate (declared before it ran): `learned-knapsack`.** The same cross-fitted model,
  but the note is filled by predicted probability per token, as the oracle fills by questions served per token. If
  it closes much of the +0.139, the space claimed for §22 shrinks.
  - **Result (`runs/exp21_gate_knapsack_report.md`; every other row reproduces exactly):** 0.146 on old-evidence
    questions (`learned` 0.147). Minus first turns: +0.037 (−0.002, +0.076). Oracle minus it: +0.140 (+0.101, +0.177).
  - By age it is 0.459 / 0.196 / 0.085, against `learned`'s 0.474 / 0.191 / 0.083.
  - Packing by probability per token closes none of the gap, so the space above the supervised row is real.
- **Not yet run:** the LongMemEval composed-episode corroboration row (free; to follow).

**What this means for §22.** The pre-registered condition for an RL stage is met. The oracle-minus-learned gap is
+0.139 on old-evidence questions, where each session's write decision is credited by exactly the later questions
it served. That is the per-write hindsight credit the review proposed. The reader stage (the real 7B answering, and
the §20 abstractive writer as a comparison) is paid and comes with its own pre-registration.

## 22. Credit assignment for the note-writer: uniform against per-write hindsight credit (N7; 2026-10-08, DRAFT pre-registration)

**Status: draft, under review.** The whole stage runs in the §21 simulator, scripted and free on CPU. Only the final
policies would ever meet the reader (item 6, paid, not proposed).

**Question.** §21a left +0.140 between the best supervised note-writer (`learned-knapsack`) and the hindsight
oracle on old-evidence questions. Each session's note decision has a delayed, unrepairable consequence there. Does
crediting each write by the later questions it served (per-write hindsight credit) learn a better writer than
GRPO's uniform episode-level credit, with the same rollouts, compute and initialisation?

**1. Policy.**
- A small extractive writer: a per-turn logit from the §21 `learned` features plus the turn's token count, through
  one hidden layer of 32.
- **Sampling:** a Plackett–Luce order over the session's turns; the note is filled in that order, skipping turns
  that do not fit the 100 tokens. The log-probability is that of the sampled order, up to its last picked turn,
  so a truncated prefix; a turn skipped for lack of room still counts in the order.
- **Initialisation:** fitted by imitation of the cross-fitted supervised scores (the `learned` model), so every
  variant starts from the same writer.

**2. Reward.** The scripted outcome of §21: on each old-evidence question, all evidence in view among the top 8 by
fusion search over the survivors. No evidence label enters the policy or its features. Whether a variant uses
labels *at training time* is stated per variant below (corrected at review). The outcome itself is computed from
the labels, as in any simulator.

**3. Variants** (pre-registered; the same rollouts, compute and initialisation; 3 seeds each; G = 8 rollouts per
conversation per update):
- **(i) Episode-level GRPO (uniform credit).** One reward per conversation rollout: its old-evidence accuracy. The
  advantage is normalised over the G rollouts of the same conversation, and every write in the rollout gets the
  same advantage.
- **(ii) Per-write hindsight credit.**
  - Session s's note is rewarded by the outcomes of exactly the questions asked w or more sessions later whose
    evidence lies in s (correct or not).
  - The advantage is normalised per session across the G rollouts. The stream is fixed, so every rollout visits
    the same sessions: this is GiGPO's anchor grouping with exact anchors and no state matching.
  - A question with evidence in several sessions credits each of them with its outcome.
  - **(ii) uses the evidence labels at training time** (to link questions to sessions), and none at test time.
- **(iii) Counterfactual credit (upper reference).**
  - A note's marginal contribution: the outcome with the note, minus the outcome re-simulated without it, summed
    over *all* questions asked w or more sessions after s. It is normalised as in (ii).
  - It is label-free in the policy's sense: no question is linked to a session by its evidence.
  - It needs one extra simulation per note. With a real reader that would mean extra reader calls, so it is an
    upper reference, not a deployable method.
  - It uses the evidence labels only through the outcome.
- **(iv) Time-forward credit (label-free control, added at review).**
  - Session s's note is rewarded by the outcomes of *every* question asked w or more sessions after s, normalised
    per session across the G rollouts exactly as (ii).
  - **Reading, declared now:** (ii) minus (iv) is what the hindsight linking buys over mere per-session grouping.
    If it is about 0, the deployable label-free variant (iv) is the result, which is better news, not worse.

**Advantage and update (the same for every variant, declared before any run).**
- Advantages are normalised by subtracting the group mean only (as in §16), with no division by the standard
  deviation, which is unstable for per-session groups of 8 with sparse rewards.
- The update is on-policy: one gradient step per batch of rollouts, with no importance ratio and no clipping.

**Training simulator (declared before any run).** Training uses a fast surrogate of the §21 retrieval:
- turn and question embeddings and per-turn BM25 scores are computed once per conversation;
- a note is ranked by the best of its turns on each list;
- the two lists are fused by reciprocal rank as in `FusionRetriever`.
Every test number comes from the exact §21 simulator. The surrogate's agreement with the exact one is reported on
the §21 policies before training.

**4. Folds.** Cross-fitted over conversations: 5 folds of 2 held-out conversations (leave-two-out). Each variant is
trained on 8 conversations and tested on the 2 held out. The supervised initialisation is also fitted without the
held-out conversations. All 10 conversations are tested once per seed. Intervals are clustered by conversation, and
seeds are averaged per question.

**Training budget.** Adam with learning rate 1e-3, no KL term, entropy bonus 0.01, G = 8, the same for every
variant. **The number of updates is provisional (300)** until one surrogate rollout has been timed.
- 300 updates × 8 conversations × G = 8, over 5 folds, 3 seeds and 4 variants, is about 1.15M conversation rollouts.
- If the measured total exceeds about 12 hours on the laptop under `guard.sh`, updates or G are cut, and which one
  is written here before training.
- **Measured before training (2026-10-08).**
  - Surrogate against exact simulator, on the §21 policies for old-evidence questions: oracle 0.290 against 0.286,
    first turns 0.108 against 0.109, salience 0.040 against 0.039, random 0.054 against 0.051.
  - Timing, from 3 updates on fold 0 including about 6 s of start-up: about 2.3–2.9 s per update for (i), (ii)
    and (iv), and about 42 s for (iii), which re-simulates once per note.
  - 300 updates is about 12–15 min per run for (i), (ii) and (iv), and about 3.5 h for (iii). That is about 62
    core-hours in all, about 5–6 h on 12 parallel workers, under the 12 h line. **The budget stays 300 updates,
    G = 8.**
- Training learning curves are reported, with no early stopping on test. The test policy is greedy as declared;
  the stochastic policy's mean is descriptive. The test policy is greedy (top-scoring order, filled by the same
budget rule).

**5. Gate (pre-registered; held-out old-evidence accuracy, conversation-clustered 95% intervals).**
- **Per-write credit pays** if (ii) minus (i) has an interval entirely above 0 **and** (ii) minus `learned-knapsack`
  has an interval entirely above 0. The same comparisons are reported for (iv), the label-free version.
- If (ii) does not beat (i): "uniform credit suffices" is the finding.
- If neither beats `learned-knapsack`: "RL adds nothing over the supervised writer here" is the finding, and
  §21a's +0.140 is recorded as headroom that neither credit scheme reaches.
- (iii) is reported as the reference for how much better credit could do.
- Also reported: everything by evidence age, and the share of the oracle gap closed:
  (variant − learned-knapsack) / (oracle − learned-knapsack).

**Expectation, written before any run.** (ii) beats (i): per-write credit gives each session a direct signal, while
uniform credit spreads one noisy number over about 20 writes. Both should beat `learned-knapsack` modestly, by a
third of the gap or less. (iii) is at least as good as (ii).

**6. Paid stage (NEEDS SPEND; not proposed until the gate is read).**
- The final policies of (i), (ii), `learned-knapsack` and the oracle, run with the frozen reader on held-out
  conversations, with the §20 abstractive notes compared against extractive notes.
- One pod, an estimate from measured latency, and a stub run with a resume rehearsal first.

### 22a. Result: per-write credit beats uniform credit (clause 1 passes); RL does not beat the supervised writer (clause 2 fails) (2026-10-08)

**Status: closed.** All 60 runs (4 variants × 5 folds × 3 seeds) finished by 20:19 UTC on 2026-10-08. The final
table is `runs/exp22_report_final.md` (`runs/exp22_eval_final.json`).
Everything is free, at c7316f0. The test is the exact §21 simulator on each fold's held-out conversations, greedy
(`runs/_pipelines/exp22_eval.py` → `runs/exp22_eval_main.json`). Each question is averaged over seeds, and the
intervals are clustered by conversation.

**Held-out old-evidence accuracy (1,249 questions, 10 conversations):**

| policy | accuracy | age 2 | age 3–5 | age 6+ |
|---|---|---|---|---|
| oracle (§21) | 0.286 | 0.306 | 0.355 | 0.179 |
| `learned-knapsack`: the cross-fitted logistic writer, packed per token (the named gate row; refit per §22 fold) | **0.146** | | | |
| (ii) per-write hindsight credit | 0.131 | 0.156 | 0.163 | 0.074 |
| each run's MLP init, packed per token (descriptive) | 0.117 | 0.114 | 0.153 | 0.069 |
| (i) episode GRPO (uniform credit) | 0.116 | 0.132 | 0.150 | 0.061 |
| first turns (§21) | 0.109 | 0.129 | 0.144 | 0.050 |

The first age column holds only old questions of age 2.

**Verdict under the pre-registered gate** (the review's ruling, which corrects my first reading). The gate names
`learned-knapsack`, the §21a row: the cross-fitted logistic writer packed per token. Each run's MLP init was a
different, weaker row that I introduced at evaluation time.
- **Clause 1, PASS.** (ii) minus (i) is **+0.015 (+0.007, +0.024)**. Per-write hindsight credit beats uniform
  credit, and uniform credit learns nothing: (i) minus its init is −0.001 (−0.011, +0.009). That is a real
  credit-assignment result.
- **Clause 2, FAIL.** (ii) minus `learned-knapsack` is **−0.014 (−0.038, +0.008)**. Per the §22 text: "RL adds
  nothing over the supervised writer here".
- **Descriptive.** (ii) minus its own MLP init is +0.015 (+0.0003, +0.029).
  - The logistic writer refit per fold (`runs/_pipelines/exp22_logistic_folds.py`) scores 0.146 packed per token
    and 0.143 by top probability, matching §21a.
- **Stated cause.** The MLP policy overfits its 8 training conversations: surrogate training accuracy rises from
  0.282 to 0.350 for (ii) (0.247 to 0.289 for (i)), against 0.131 held out.
  - The RL gain is smaller than the gap between the MLP init and the logistic writer, so the policy class, not the
    credit signal, limits this result. §22b tests that with a policy that cannot overfit.
- **(iv) time-forward credit (label-free; `runs/exp22_eval_fwd.json`, table `runs/exp22_report_iii_pending.md`):**
  0.119.
  - (ii) minus (iv): **+0.012 (+0.002, +0.023)**.
  - (iv) minus (i): +0.003 (−0.005, +0.010).
  - (iv) minus `learned-knapsack`: −0.027 (−0.048, −0.007).
  - **Attribution.** Per-session grouping alone (iv) is about the same as uniform credit (i), and the whole
    per-write gain comes from the hindsight linking: (ii) minus (iv) is +0.012 (+0.002, +0.023).
  - **Consequence for deployment.** The gain needs evidence labels at training time, to link questions to the
    sessions holding their evidence. The supervised writer uses the same labels, so neither method is label-free;
    the label-free variant (iv) does not carry the effect.
  - **Why (ii) and (iii) share a training curve** (0.282 → 0.350 on average). Under the lossy rule, a question whose
    evidence lies wholly in session s can be answered past w only if s's note holds that evidence, so its outcome
    without the note is 0. Its outcome, which is (ii)'s credit, then equals the note's marginal effect, which is
    (iii)'s credit.
    - The two differ only on multi-session questions, and through displacement: a note that wins a top-8 slot can
      push another item out.
    - So the signals nearly coincide by construction, not by accident. The run files differ (fold 0, seed 0: 0.365
      against 0.367 at the end; the writers differ), so it is not a duplicated run.
- **(iii) counterfactual credit (the upper reference; all 15 runs):** 0.141 held-out.
  - (iii) minus (ii): +0.009 (+0.004, +0.015).
  - (iii) minus (i): +0.025 (+0.015, +0.035).
  - (iii) minus `learned-knapsack`: −0.005 (−0.025, +0.013).
  - The strongest credit signal is the best RL variant, but it does not beat the supervised writer either.
- **Measured total compute:** 03:11 to 20:19 UTC, about **17 h** wall-clock on 12 workers, against an estimate of
  5–6 h. The last three (iii) runs alone ran about 6.5 h at the tail. The lessons (time jobs under the real
  parallel load; never put more long jobs in the queue than there are slots) are in the pre-launch checklist.
- **Compute slip.** Under the 12-job load a cheap run took about 30–40 min, not the measured 12–15 min. The
  laptop has 8 physical cores, and 15 slow (iii) jobs left a tail. The batch ran well past the 5–6 h estimate; the
  measured total is added when the batch ends.
- **Init deviation.** §22 declared "imitation of the supervised scores"; the code fits the BCE labels directly.
  That is the same supervised model, one step shorter.

## 22b. The same experiment with a policy that cannot overfit (N7b; 2026-10-08, pre-registered before any run)

**Why.** §22a's clause 2 failed because the MLP policy overfits 8 conversations, and its init (0.117) starts below
the supervised writer (0.146). Here the policy is the logistic writer itself, so the init *is* `learned-knapsack`
and the comparison is clean.

**Policy.** Linear in the §21 features plus the token count: the `learned` model's form, with one logit per turn.
It is initialised at the cross-fitted logistic fit for each fold (8 training conversations, C = 0.1, as in §21),
so every variant starts at the gate row. The sampling and log-probability are §22's (a Plackett–Luce ordered
prefix), and the test-time packing for all is the knapsack rule (probability per token, filled greedily).

**Variants.** (i) episode GRPO, (ii) per-write hindsight credit and (iv) time-forward credit. (iii) runs only if
time allows, as it is the slow one, and is reported as a reference if it does.

**Held fixed from §22.** The budget (300 updates, G = 8), 3 seeds, the 5 leave-two-out folds, mean-only
advantages, the on-policy single-step update with no ratio or clipping, the surrogate for training and the exact
simulator for testing.

**Gate (the same as §22's, against the same named row).**
- (ii) minus (i): interval entirely above 0.
- (ii) minus `learned-knapsack`: interval entirely above 0.

**Expectation, written before any run.** (ii) beats (i) again. Whether (ii) beats `learned-knapsack` is open. If it
does not, the finding is that the credit signal is real but the supervised writer already captures what these
features can express: the limit is the features, not the credit.

**Diagnostic (descriptive).** Held-out accuracy on the surrogate is recorded every 50 updates, so overfitting is
seen rather than inferred. There is no early stopping on it.

**Compute.** A linear policy is cheaper than the MLP. It is timed before launch, under the real parallel load and
with the longest jobs first, and the measured total is written here before training.
- **Measured (2026-10-08, 16:30 UTC, beside the last three §22 (iii) jobs):** about 4.6 s per update, including
  the held-out diagnostic. That is about 23 min per run alone, and about 32 min under a 12-job load.
- 45 runs ((i), (ii) and (iv) × 5 folds × 3 seeds), 9 at a time: **about 2.5–3 h**.
- (iii) is not run in this batch, as allowed by the text above.

### 22b-a. Result: RL from the supervised writer makes it worse; per-write credit least (2026-10-08)

All 45 runs ((i), (ii) and (iv) × 5 folds × 3 seeds), from 1e78524, finished at 20:18 UTC (about 3.8 h; the
estimate was 2.5–3 h).
- **Compute incident.** §22b took 16:30–20:18 UTC, about 3.8 h against 2.5–3 h. §22a took about 17 h against 5–6 h.
- **Causes, measured.**
  - Per-update cost rose under load: §22b's 4.6 s per update was timed beside 3 jobs, but it ran beside 12 on 8
    physical cores (16 threads).
  - §22b adds a surrogate held-out evaluation every 50 updates.
  - §22a queued 15 slow (iii) jobs into 12 slots, leaving a 6.5 h tail of three.
- **Rule for the next simulator estimate:** use the per-update time measured under the real parallel load, and
  never queue more long jobs than there are slots. Measured from §22b: 45 runs × 300 updates in about 3.8 h on 9
  slots, so about 0.76 h per run, or **about 9 s per update under load**. Evaluation is the exact simulator on held-out conversations, with knapsack packing for every
policy, as declared (`runs/_pipelines/exp22b_eval.py` → `runs/exp22b_eval.json`; table `runs/exp22b_report.md`).
The init row is each run's own start, which is the cross-fitted logistic writer.

| policy | held-out old-evidence accuracy | age 2 | age 3–5 | age 6+ |
|---|---|---|---|---|
| oracle (§21) | 0.286 | 0.306 | 0.355 | 0.179 |
| `learned-knapsack` (the init) | **0.145** | 0.145 | 0.192 | 0.080 |
| (ii) per-write hindsight | 0.117 | 0.129 | 0.143 | 0.075 |
| (iv) time-forward | 0.096 | 0.105 | 0.124 | 0.053 |
| (i) episode GRPO | 0.081 | 0.097 | 0.101 | 0.046 |

**Verdict under the pre-registered gate.**
- **Clause 1, PASS.** (ii) minus (i) is **+0.036 (+0.021, +0.054)**.
- **Clause 2, FAIL**, and in the wrong direction: (ii) minus `learned-knapsack` is **−0.027 (−0.049, −0.003)**.
- (ii) minus (iv) is +0.021 (+0.008, +0.037), and (iv) minus (i) is +0.015 (+0.007, +0.024). The hindsight linking
  again carries most of the per-write advantage.

**The held-out diagnostic** (surrogate, every 50 updates, mean over runs) shows the decline directly, not
inferred. It was computed on held-out conversations during training and used for interpretation only, not for any
choice in §22b. §22c nearly reused it to fix its stopping point, and that rule was withdrawn.

| update | 0 | 50 | 100 | 150 | 200 | 250 | 300 |
|---|---|---|---|---|---|---|---|
| (ii) hindsight | 0.145 | 0.145 | 0.142 | 0.137 | 0.127 | 0.117 | 0.117 |
| (iv) time-forward | 0.141 | 0.113 | 0.102 | 0.096 | 0.097 | 0.097 | 0.097 |
| (i) episode | 0.144 | 0.107 | 0.096 | 0.087 | 0.085 | 0.086 | 0.081 |

The training curves (surrogate, the first and last 20 updates) separate two failures:
- **(i) and (iv) get worse even on their own training conversations** (0.143 → 0.090 and 0.145 → 0.116). With
  uniform or time-forward credit, the gradient is mostly noise, and it moves a good init away from where it was.
- **(ii) improves on training (0.153 → 0.220) while declining held out (0.145 → 0.117)**, which is overfitting.
  The "linear" writer still has 392 inputs (384 of them embedding dimensions) and learns from 8 conversations. The
  supervised fit was regularised (C = 0.1), but the RL stage has no pull back to the init: §22 declared no KL
  term.

**Interpretation (labelled as such; the verdict lines above stand).** Clause 1 must not be read as "hindsight
credit generalises better". The curves show two different failures. Surrogate old-evidence accuracy on the
training conversations, from the first update to the last (the means of the first and last 20 updates in
brackets):
- **(i) episode** goes from 0.146 to 0.091 (0.143 → 0.090), and **(iv) time-forward** from 0.146 to 0.116
  (0.145 → 0.116). Both get *worse on their own training set* over 300 updates. At this group size (G = 8) and
  learning rate, their gradient carries no usable signal, and they drift.
- **(ii) hindsight** is the only variant that improves on training: 0.146 to 0.220 (0.153 → 0.220). It then
  overfits 8 conversations with no KL pull to the init: held out it goes 0.145 → 0.117, flat for the first 100
  updates.
- So clause 1 says that **hindsight credit is the only variant with a usable training signal in this regime**.
  Clause 2's failure is about regularisation and data, not the credit signal.

**Reading.** The expectation written for this case ("the credit signal is real but the supervised writer already
captures what the features can express") holds only in part:
- The credit signal is real: per-write credit beats uniform credit in both §22 and §22b, and it is the only
  variant that improves on its training data.
- But RL from the supervised writer does not just fail to add; it subtracts.
- Two declared choices are the likely causes: no KL term (or other pull back to the init), and the embedding
  features (high-dimensional against 8 training conversations).
- One more is possible: training samples notes by logit (a Plackett–Luce order), while the test packs by
  probability per token.
- A follow-up (a KL term to the init, or the 8 handcrafted features without the embedding, and training sampled as
  tested) would be a new pre-registration. None is proposed here.

## 22c. Per-write credit with a pull to the supervised writer: the close of RQ3 (N7c; DRAFT, not run; the user decides)

**Status:** the design was amended at the user's objection to any use of held-out information (see the stopping
rule). The re-estimated laptop time needs the user's go-ahead before relaunch. Whatever it shows, it closes RQ3 on
this task.

**Question.** §22b showed that hindsight credit is the only variant with a usable training signal, but that it
overfits 8 conversations with nothing pulling it back to the supervised init. With such a pull, and stopping
where the held-out diagnostic was still flat, does per-write credit at least not damage the supervised writer, and
perhaps improve it?

**Fixed now.**
- **Policy and init:** the §22b linear writer, initialised at the cross-fitted logistic fit (`learned-knapsack`).
- **Variants:** (ii) per-write hindsight credit, and (i) episode GRPO as the control. Both carry the same pull.
- **The pull:** a KL penalty to the init, β × KL(π_θ ‖ π_init), on each session's first-pick distribution (the
  softmax over that session's turns), summed over sessions, with **β = 0.1**. β is fixed now and not tuned.
  - **Scope of the pull.** The KL is on the first-pick distribution, not on the full ordered-prefix Plackett–Luce
    distribution, so later picks are pulled only through the shared weights.
- **Stopping rule (amended at the user's objection; it sees training conversations only).** No design choice
  uses held-out information.
  - Within each fold, the 8 training conversations are split 6/2, with the split fixed by the seed (a permutation
    with `numpy.random.default_rng(1000 + seed)`; the last 2 are the inner validation pair).
  - A writer is trained on the 6 from an init fitted on those 6 only (the logistic writer, which is also the KL
    anchor), with a checkpoint every 25 updates up to 300. Fitting the init on all 8 would leak the inner pair. The checkpoint with
    the best surrogate accuracy (greedy, knapsack packing) on the inner 2 is chosen; ties go to fewer updates.
  - A new writer is then trained on all 8, from the logistic fit on the 8 (the `learned-knapsack` row and the KL
    anchor), for exactly that many updates, and evaluated **once**, on
    the fold's held-out conversations, with the exact simulator.
  - Nothing on the held-out side is computed or written before that update count is fixed. The same rule applies
    to (i). The chosen update counts are reported as description.
  - Inner validation pairs (fold: seed 0 / seed 1 / seed 2):

    | fold | seed 0 | seed 1 | seed 2 |
    |---|---|---|---|
    | 0 | 7, 9 | 2, 7 | 3, 9 |
    | 1 | 7, 9 | 0, 7 | 1, 9 |
    | 2 | 7, 9 | 0, 7 | 1, 9 |
    | 3 | 5, 9 | 0, 5 | 1, 9 |
    | 4 | 5, 7 | 0, 5 | 1, 7 |
- **The first rule, withdrawn.** It fixed 100 updates from §22b's held-out diagnostic. A launch under that rule
  (20:39 UTC, 2026-10-08) was stopped at about 20:41 UTC at the user's objection, before any run finished. No
  held-out accuracy from it was computed or read. Its partial folders are quarantined in
  `runs/exp22c_peek_quarantine/`.
- **Held fixed from §22b:** G = 8; mean-only advantages; on-policy, one step per batch; Adam with learning rate
  1e-3; entropy bonus 0.01; the surrogate for training and the exact simulator for testing; knapsack packing at
  test; 5 leave-two-out folds × 3 seeds. That is 15 runs per variant, 30 in all.
- **No held-out diagnostic during training.** Test accuracy is computed once per run, at the end.

**Single clause (pre-registered).** (ii)+KL minus `learned-knapsack`, on held-out old-evidence accuracy
(conversation-clustered, seeds averaged), has a 95% lower bound above −0.01.
- If it holds: per-write credit with a pull to the init does no material harm to the supervised writer. Any gain
  is reported with its interval.
- If not: RL does not help the supervised note-writer even when regularised.
- Either way, **RQ3 is closed on this task**: credit assignment matters (per-write over uniform), and imitation
  is the stronger learner here.
- (ii)+KL minus (i)+KL is reported as description.

**Expectation, written down.** The lower bound clears −0.01, with a gain near 0.

**Laptop time, from measured throughput (§22b: about 9 s per update under a 12-job load, with 8 training
conversations).**
- Inner selection: 300 updates on 6 conversations, about 6.8 s each, about 34 min per run, plus 12 inner
  evaluations (small).
- Retrain on 8 conversations for the chosen count: up to 300 updates, about 9 s each, at most 45 min.
- Per run: about 40–80 min. 30 runs on 12 slots is 3 waves (12, 12, 6), about **2–4 h** wall-clock (about 3 h if
  the chosen counts sit near the middle), under `guard.sh`.
- No more long jobs than slots.

### 22c-a. Result: with a pull to the init, RL stays level with the supervised writer; the clause fails narrowly; RQ3 closed (2026-10-08)

All 30 runs ((ii) and (i), 5 folds × 3 seeds), from ea95030 under the amended rule, ran 20:43–22:40 UTC (about
2 h, within the 2–4 h estimate). The update counts were chosen on the inner pairs by the **surrogate** simulator.
Each writer was evaluated **once**, on its fold's held-out conversations, by the exact simulator, with knapsack
packing (`runs/_pipelines/exp22b_eval.py` → `runs/exp22c_eval.json`; table `runs/exp22c_report.md`, copied to `docs/research/exp22c_report.md`). The init row
is the logistic fit on all 8 training conversations (`learned-knapsack`).

**Chosen update counts** (folds 0–4 × seeds 0–2, in order; descriptive):
- (ii): 0, 0, 50, 50, 0, 0, 150, 200, 125, 25, 125, 50, 0, 150, 0. That is 0 (no RL at all) in 6 of 15 runs.
- (i): 250, 225, 225, 150, 75, 225, 0, 225, 25, 250, 200, 25, 175, 175, 0.

| policy | held-out old-evidence accuracy | age 2 | age 3–5 | age 6+ |
|---|---|---|---|---|
| oracle (§21) | 0.286 | 0.306 | 0.355 | 0.179 |
| (i) episode GRPO + KL | 0.147 | 0.140 | 0.195 | 0.087 |
| `learned-knapsack` (the init) | 0.145 | 0.145 | 0.192 | 0.080 |
| (ii) per-write hindsight + KL | 0.144 | 0.141 | 0.189 | 0.083 |

**Verdict under the pre-registered clause.**
- (ii)+KL minus `learned-knapsack` is **−0.001 (−0.012, +0.011)**. The lower bound, −0.012, is below −0.01, so
  the clause **FAILS, narrowly**.
- Descriptive: (ii)+KL minus (i)+KL is −0.004 (−0.014, +0.008); (i)+KL minus `learned-knapsack` is +0.002
  (−0.001, +0.006).

**Reading.**
- The pull did what it was for. Unlike §22b, neither variant degrades the supervised writer: both stay within
  about a point of it.
- Neither improves on it either. The inner selection often chose no RL at all for (ii) (0 updates in 6 of 15
  runs), so the training conversations themselves showed no update count that helped.
- The clause's failure is a width failure: the point estimate is −0.001, and the interval reaches just past the
  −0.01 margin.
- **RQ3, closed on this task.** Credit assignment matters (per-write over uniform in §22 and §22b, wherever RL
  learns anything). But RL does not beat imitation for the note-writer, with or without regularisation.
- Supervised learning on the same evidence labels is the stronger and cheaper learner here, consistent with §10
  and §16a.

## 23. Does a memory controller lift a small reader to the next size class? (N8; 2026-10-08, pre-registered)

**Status:** run 2026-10-08 on pod l29poll3lrjuat, about $0.83; the result is in §23a. The pre-registration text
below is unchanged.

**Why it is usable as a drop-in.** The §19 head is reader-free: it was trained on evidence labels, never on any
reader's answers. So the same controller is used, unchanged, for all five readers.

**The user's hypothesis, as pre-registered claims** (LongMemEval, 470 non-abstention questions, paired by question
across readers):
1. **Claim 1.** The §19 head with 8 turns (`fixed8`), read by a 3B, is at least as accurate as FIFO + floor read by
   a 7B of the same family (Qwen2.5-Instruct).
2. **Claim 2.** `fixed8` read by a 7B is at least as accurate as FIFO + floor read by a 14B.
- **"At least as accurate" (non-inferiority; margin fixed at review).** The paired difference (small reader with
  `fixed8`, minus large reader with FIFO + floor) has a 95% interval whose lower bound is above −0.03. "Better" is
  reported if the lower bound is above 0.
- **The number the claim will be remembered by (descriptive): the share of the size gap closed.**
  - share = (`fixed8` at the small reader − FIFO at the small reader) / (FIFO at the large reader − FIFO at the
    small reader), with a bootstrap interval.
  - It is reported for Qwen 3B→7B, Qwen 7B→14B and Granite 2B→8B.
  - Above 1 means the controller more than bridged the size step.
- **Secondary, same family.** Granite-3.1 2B with `fixed8`, against Granite-3.1 8B with FIFO + floor.
- LoCoMo is exploratory, and only if the cost is small.

**Readers (five).**
- Qwen2.5-Instruct 3B, 7B and 14B: the size ladder. The 14B runs in bf16 on the A40 with `--max-model-len 16384`;
  if the KV cache does not fit, AWQ is used, declared at launch.
- Granite-3.1 instruct 2B and 8B: a second family. It already served on our pod (§18); Llama is gated.

**Arms per reader** (on the same 500 test questions, five folds, as §19c):
- FIFO + floor at 5% with the 3k target: the baseline.
- `fixed8`: the §19 head A, 32 BM25 candidates.
- `fixed16`: the context-rot probe.
- **The 7B rows are rerun in this session** for pairing. No §19c row is reused in the primary table.

**Prompt and judge.**
- **One prompt for every reader:** the frozen 7b5fc30 reader prompt, unchanged. There is no per-model prompt
  tuning; that is a stated limitation. The "unknown" rate is reported beside every accuracy as the format-failure
  check.
- **One judge for every reader:** Qwen2.5-7B with the official LongMemEval prompts, so all readers sit on one scale
  (§18 found the controller ranking judge-independent).
- A vLLM server holds one model, so **answers are generated first** with a stub judge, then **all judged in one
  separate pass** with the Qwen 7B judge, with the same official prompts (the §18 `memctl.judge_check` route). The
  judge cache is keyed by question and answer text, and **the judge never sees which reader wrote the answer.**

**Measures.**
- Accuracy and prompt tokens.
- The §13a decomposition. P(all in view) is the same for every reader on the same arm (the controller does not
  depend on the reader), so **P(correct | in view) is the reader-only number**, and its chart against reader size
  is the main figure.
- By question type.
- Descriptive, per reader: the accuracy-against-tokens frontier and where `fixed16` falls below `fixed8` (the
  degradation point).

**Expectation, written before any run.**
- Claim 1 is likely: §19c's `fixed8` at 7B beat FIFO + floor by +0.113, a margin a smaller reader may keep.
- Claim 2 is uncertain: a 14B is less fragile with long prompts, so FIFO + floor gains more from it.
- A 3B or 2B reader may show a high "unknown" rate under the frozen prompt. That would count against claim 1 and
  is reported as a format failure, not hidden.

**Cost (NEEDS SPEND; from measured throughput; a figure for the user before any launch).**
- Per reader: download and load (§19c: reader ready about 5 min after create; Granite in §18 loaded by changing
  the vLLM arguments), plus 3 arms × 500 questions = 1,500 reader calls.
- At §19c's measured rate (2,000 LongMemEval questions with judging in about 29 min on 4 workers; about 16 workers
  planned here), that is about 10–15 min per reader, and about 20–30 min for the 14B.
- Then one judge pass of 7,500 short calls with Qwen 7B: about 10–15 min.
- **One A40 session, models switched in turn:** about 2–2.5 h, so **about $1.00–1.25**, with a hard stop at 3 h
  (about $1.50). This is the figure accepted at review, for the user's approval.
- Before launch: a full-size stub run with a resume rehearsal against a changed endpoint, and peak RSS recorded.
  - **Done at c9bd9e7.** Reader `qwen3b` (stub), 5 folds × 3 arms, was interrupted after 25 s (cells at 11–21 of 100
    questions), then resumed against a different `base_url`. All 15 cells resumed (`resumed: true`, none refused)
    and finished with 100 questions each. Peak RSS is 0.29 GB per worker.
  - The judge script (`runs/_pipelines/exp23_judge.py`) ran on the stub output: 470 answers per arm. Its stub
    entries (486 files) were deleted from `cache/judge_exp23`, so the real pass starts from an empty cache.
- **Launch (recorded before any result).**
  - Pod l29poll3lrjuat, created 18:14:24 UTC, in EU-SE-1 at **$0.59/h** (not the $0.49 estimated). The hard stop
    is moved to 2.5 h → **20:44 UTC, about $1.48**, under the user's ceiling of about $1.50.
  - Stage order: Qwen 3B, Qwen 14B, Granite 2B, Granite 8B, then Qwen 7B (answers and the blind judge pass, one
    load).
  - **Triage rule, pre-declared:** if Qwen 14B has not finished by 19:30 UTC, the Granite pair (secondary) is
    dropped before anything else, so that the Qwen 7B rows the primary claim needs still run.
- **Grading, declared before the judge pass** (after the Qwen 3B format check below).
  - Every reader-and-arm cell reports accuracy on the answerable questions as three disjoint shares:
    answered-correct, answered-wrong and unknown.
  - The headline accuracy is answered-correct, as pre-registered, and the claim tests are unchanged.
  - Beside them, one descriptive line per reader: how much of the `fixed8` minus FIFO gap is "fewer unknowns"
    against "more correct among answered" (P(correct | answered) compared across arms). The controller can help
    because evidence is in view, or because a small reader stops refusing. Both are legitimate but different
    claims, and the write-up names which one holds at each size.
  - A cost column, **"false answers on unanswerable"**, counts the abstention questions answered with something
    other than "unknown". n = 30, so no interval is quoted.
- **Qwen 3B format check** (descriptive; same host; before any judging; `runs/_pipelines/exp23_unknown.py`):

  | arm | unknown on the 470 answerable questions | false answers on the 30 unanswerable |
  |---|---|---|
  | FIFO + floor | 0.709 | 1 of 30 |
  | `fixed8` | 0.372 | 7 of 30 |
  | `fixed16` | 0.570 | 2 of 30 |

  - At 3B the controller roughly halves the reader's refusals, so it changes the reader's willingness to answer,
    and it also answers more of the unanswerable questions.
  - `fixed16` sits between the two: a longer prompt makes the small reader refuse more. That is a second
    observation of the §19c/§19d pattern, with no verdict.
- LoCoMo (exploratory) would add about 6,000 calls per reader. It is not included unless the user wants it.

### 23a. Result: the controller lifts a 7B to the 14B's baseline, but not a 3B to the 7B's (2026-10-08)

Pod l29poll3lrjuat (A40, EU-SE-1, $0.59/h): 18:14:24 to about 19:38:40 UTC, about 1.40 h, **about $0.83** (the
approval was about $1.00–1.25). Terminated, and list-pods was empty afterwards. RunPod's billing had $0.25 on record
at termination (it lags), to be rechecked. Answers were generated at c9bd9e7; the one blind Qwen 7B judge pass
judged 7,050 answers. Every cached verdict is a clean "yes" or "no", with no anomalies. Report:
`runs/_pipelines/exp23_report.py` → `runs/exp23_report.md`; verdicts in `runs/exp23_verdicts.json`.

**Accuracy on the 470 answerable LongMemEval questions** (95% interval by question; prompt tokens per arm:
FIFO + floor 2,964, `fixed8` 1,049, `fixed16` 2,569; P(all in view) per arm: 0.498, 0.702, 0.747, the same for
every reader):

| reader | FIFO + floor | `fixed8` | `fixed16` | `fixed8` minus FIFO |
|---|---|---|---|---|
| Qwen 3B | 0.162 (0.130, 0.196) | **0.315** (0.274, 0.357) | 0.202 | +0.153 (+0.109, +0.200) |
| Qwen 7B | 0.436 (0.391, 0.481) | **0.538** (0.494, 0.583) | 0.487 | +0.102 (+0.060, +0.145) |
| Qwen 14B | 0.504 (0.460, 0.551) | **0.606** (0.564, 0.651) | 0.581 | +0.102 (+0.060, +0.145) |
| Granite 2B | 0.368 (0.326, 0.413) | 0.364 (0.321, 0.409) | 0.353 | −0.004 (−0.047, +0.038) |
| Granite 8B | 0.421 (0.377, 0.466) | 0.470 (0.426, 0.515) | 0.489 | +0.049 (+0.002, +0.096) |

**The drop-in lift, `fixed8` minus FIFO + floor per reader (same controller, unchanged):** Qwen 3B +0.153
(+0.109, +0.200), Qwen 7B +0.102 (+0.060, +0.145), Qwen 14B +0.102 (+0.060, +0.145), Granite 2B −0.004
(−0.047, +0.038), Granite 8B +0.049 (+0.002, +0.096). Report copy: `docs/research/exp23_reader_ladder_report.md`;
figure: `docs/research/figures/exp23_ladder.png`.

The Qwen 7B and 14B rows give the same difference by coincidence. Their answers differ on 227 of 470 questions,
and their win/loss counts are 78/30 and 76/28: the same net of +48.

**Pre-registered claims** (non-inferiority margin −0.03; "better" if the lower bound is above 0):
- **Claim 1, NOT SHOWN.** `fixed8` at Qwen 3B minus FIFO + floor at Qwen 7B is **−0.121 (−0.170, −0.070)**. The
  controller closes **56% (41–72%)** of the 3B-to-7B gap, but not all of it.
- **Claim 2, NON-INFERIOR.** `fixed8` at Qwen 7B minus FIFO + floor at Qwen 14B is **+0.034 (−0.013, +0.079)**.
  The lower bound, −0.013, is inside the −0.03 margin. The share of the size gap closed is **1.50 (0.83, 2.85)**:
  the controller more than bridges the 7B-to-14B step, at about a third of the prompt.
- **Secondary (Granite 2B → 8B), NOT SHOWN.** −0.057 (−0.102, −0.013). The controller adds nothing to Granite 2B
  (−0.004).

**Where the gain comes from** (the declared split; unknown share on answerable questions, and P(correct | answered),
FIFO → `fixed8`):

| reader | unknown | P(correct \| answered) | reading |
|---|---|---|---|
| Qwen 3B | 0.709 → 0.372 | 0.555 → 0.502 | **fewer refusals**; precision falls a little |
| Qwen 7B | 0.328 → 0.151 | 0.649 → 0.634 | mostly fewer refusals |
| Qwen 14B | 0.270 → 0.183 | 0.691 → **0.742** | **more correct among answered**, and fewer refusals |
| Granite 2B | 0.081 → 0.181 | 0.400 → 0.444 | more refusals, more precise; no net gain |
| Granite 8B | 0.262 → 0.160 | 0.571 → 0.559 | fewer refusals |

- The controller helps small Qwen readers mainly by making them answer, and the 14B by making its answers right.
  The write-up names the mechanism per size, as agreed.
- **The reader-only number, P(correct | all in view) under `fixed8`,** rises with size: 0.397 at 3B, 0.667 at
  7B, 0.773 at 14B. Granite: 0.452 at 2B, 0.573 at 8B.
- **Cost: false answers on the 30 unanswerable questions** (correct abstentions shown as 30 minus this):

  | reader | FIFO + floor | `fixed8` | `fixed16` |
  |---|---|---|---|
  | Qwen 3B | 1 | 7 | 2 |
  | Qwen 7B | 6 | 7 | 6 |
  | Qwen 14B | 5 | 3 | 2 |
  | Granite 2B | 25 | 15 | 16 |
  | Granite 8B | 9 | 14 | 12 |

**Descriptive.**
- **Refusal (not accuracy) by family.** Qwen: the share answering "unknown" under `fixed16` against `fixed8` goes
  from more (3B, 0.570 against 0.372) to fewer (14B, 0.140 against 0.183). Long-prompt fragility shrinks with size
  *in the Qwen family*. Granite does not follow one pattern, so this is a Qwen-family observation.
- **Context-rot probe, `fixed16` minus `fixed8`:**
  - Qwen 3B −0.113 (−0.160, −0.068); 7B −0.051 (−0.089, −0.015); 14B −0.026 (−0.062, +0.009);
  - Granite 2B −0.011; 8B +0.019 (both intervals include 0).
  - The cost of a longer prompt shrinks with Qwen size. The 7B row repeats §19c's sign on a third host (§19d's
    failed to reach significance).
- **7B pairing check against §19c, same questions:**
  - `fixed8`: 0.538 here against 0.545, with per-question agreement 0.968;
  - FIFO + floor: 0.436 against 0.432, agreement 0.979.
  - The rerun reproduces.

**Limitations.** One frozen prompt for all readers (the 3B's refusal rate is partly a prompt-format effect); one
benchmark; one judge (Qwen 7B, whose ranking was judge-independent in §18).

## 24. Does the drop-in lift hold in other model families? (N9; 2026-10-08, pre-registered, run 2026-10-09)

**Status:** run 2026-10-09 on pod 7d7fnvmk5actq0, about $1.44; the result is in §24a. The pre-registration text
below is unchanged apart from the launch and deviation notes added before the judge pass.

**Why.** In §23 the controller lifted every Qwen size, but Granite 2B not at all. One family plus one exception is
not enough to say whether the lift is a property of the controller or of Qwen. §24 repeats §23 unchanged with
three more families, two sizes each.

**Readers (six; each fits one A40 in bf16).**

| family | small | mid | gated on Hugging Face |
|---|---|---|---|
| Llama | Llama-3.2-3B-Instruct | Llama-3.1-8B-Instruct | yes |
| Gemma 3 | gemma-3-4b-it (text only) | gemma-3-12b-it (text only) | yes |
| Phi-4 | Phi-4-mini-instruct (3.8B) | phi-4 (14B) | no |

- **Fallback**, only if a gate or vLLM support blocks a family: Mistral-7B-Instruct-v0.3 and
  Ministral-8B-Instruct-2410. Both repositories also ask the user to accept terms, so the fallback does not remove
  the token prerequisite. The two sizes are close (7B, 8B), so claim (b) for that pair says little; it would be
  reported as descriptive only.
- **Prerequisite (the user's step):** accept the Llama, Gemma (and, for the fallback, Mistral) licences on Hugging
  Face and put `HF_TOKEN` in `.env`. Today `.env` has no `HF_TOKEN`. `runs/_pipelines/exp24_stage.sh` stops at once
  (exit 2) for a gated reader when the token is missing; tested.
- Served with the §23 image (vLLM v0.8.5) and `--max-model-len 16384`. Gemma 3 is supported from vLLM 0.8.0; it is
  served text-only (`--limit-mm-per-prompt image=0`). If a family does not load, that is declared at launch and the
  fallback pair takes its place.

**Nothing changes on the test side.**
- The same 470 answerable and 30 abstention LongMemEval questions (five folds, 100 each), the same three arms
  (FIFO + floor at 5% with the 3k target, `fixed8`, `fixed16`), the same per-fold head checkpoints, the same frozen
  reader prompt (7b5fc30).
- The configs are the §23 configs with only the reader name and the cache directory changed (857bb5b).
- The same blind Qwen 7B judge, official LongMemEval prompts: answers first with a stub judge, then one judge pass
  (`runs/_pipelines/exp24_judge.py`), which never sees which reader wrote an answer.
- The same report columns: three disjoint shares (answered-correct, answered-wrong, unknown), P(correct |
  answered), P(correct | all in view), prompt tokens, and false answers on the 30 unanswerable questions.
- **How "unknown" is counted (interpretation only).** An answer counts as unknown only if it literally starts
  with "unknown", the reply the reader prompt asks for (`memctl/envs/qa.py:44`). A new family may refuse in other
  words, and those answers land in answered-wrong. So the report also prints, as a descriptive column only, the
  count of judge-wrong answers that contain one of these refusal phrases (fixed now; case-insensitive, a curly apostrophe read as straight): "don't
  know", "do not know", "not mention", "no information", "cannot determine", "unable to". No claim
  depends on it: claims (a) and (b) use only the judge's correct or incorrect.
- **Token counts, stated plainly.** The memory budget, the 3k target and the prompt-token column are counted with
  memctl's model-free counter (words plus punctuation marks, `memctl/memory/items.py:42`), as in §23, not with each
  reader's tokenizer. So every reader is shown exactly the same text in each arm; only the number of model tokens
  that text costs differs by tokenizer.

**Pre-declared claims, per family** (margin −0.03, as in §23; "better" if the lower bound is above 0):
- **(a) Lift.** `fixed8` minus FIFO + floor, at each size, has a 95% interval above 0.
- **(b) Size step.** `fixed8` at the small size minus FIFO + floor at the mid size has a lower bound above −0.03
  (non-inferior).
- Reported beside (b): the share of the size gap closed, with its bootstrap interval.

**Cross-family claim, stated before any result.** "The smallest size gains most" was a Qwen-only observation in
§23, and Granite contradicted it. §24 tests it in each new family: (lift at small) minus (lift at mid), paired by
question. It **holds** in a family if the lower bound is above 0, is **reversed** if the upper bound is below 0,
and is otherwise undetermined. No claim is adjusted after the results.

**Confound, stated before any result.** The one prompt was tuned on Qwen 7B (training slice). A low lift in a
family may reflect how well the prompt fits that family, not the controller. A family-specific prompt, if ever
tried, is a separate pre-registered arm, tuned on training conversations only (no held-out peeking).

**Stub check (done, free).**
- All 8 readers (6 and the 2 fallback) × 5 folds × 3 arms ran full size on the stub backend: 1,500 answers per
  reader, no errors. P(all in view) per arm is 0.498, 0.702, 0.747 and prompt tokens are 2,964, 1,049, 2,569:
  identical to §23, as they should be (the controller does not depend on the reader).
- **Resume on a new endpoint:** Llama 3B was stopped after 40 s (cells at 78–100 of 100) and resumed against a
  different `base_url`: all 15 cells resumed, none refused, each ended at 100.
- Peak RSS 0.58 GB per sweep process (three cells).
- The judge script judged the stub answers (11,280 with the fallback); the report script ran end to end. All stub
  generation and judge cache entries were deleted afterwards, so the real run starts from empty caches.

**Cost (NEEDS SPEND; from §23's measured stage times on the same A40, $0.59/h).**
- §23 measured: first model ready 4.4 min after create; model switches 3.1–6.2 min; answer stages Qwen 3B 6.0 min,
  Qwen 7B 10.3, Granite 8B 18.1, Qwen 14B 19.3; judge pass 3.1 min for 7,050 answers.
- §24: answer stages 66–80 min (3–4B readers 6–8 min each; Llama 8B 10–18; Gemma 12B and Phi-4 17–20); seven
  model loads (six readers and the Qwen 7B judge) 25–42 min; judge pass about 4 min for 8,460 answers.
- **About 1.6–2.1 h, so about $0.95–1.25.** Hard stop at 2.5 h after create (about $1.48). If the pod's price is
  not $0.59/h, the figure is re-quoted before the stages start.
- **Run order, one pod, judge last:** Llama 3B, Llama 8B, Phi-4-mini, Phi-4, Gemma 4B, Gemma 12B, then Qwen 7B
  (judge). Families run as pairs, so a cut drops a whole family, never half of one; Gemma goes last because it
  carries the vLLM-support risk.
- **Triage rule, pre-declared:** if Gemma 4B has not started by 1 h 50 min after create, the Gemma pair is dropped
  and the judge runs, so the families already answered are graded inside the hard stop.
- **Approved by the user (2026-10-08, about $0.95–1.25). Order changed at launch, before any result:** Meta had
  not yet granted access to Llama-3.2-3B (HTTP 403; Llama 8B, both Gemmas and Phi-4 are reachable), so Llama moves
  to the end: Phi-4-mini, Phi-4, Gemma 4B, Gemma 12B, Llama 8B, Llama 3B, then Qwen 7B (judge). The triage rule
  now applies to the Llama pair (the last family): if it has not started by 1 h 50 min after create, it is dropped.
  If Llama 3B is still refused when its turn comes, Llama 8B runs alone and only claim (a) at 8B is reported for
  Llama; (b) and the cross-family test are not tested for it. The token reaches the pod as a RunPod secret, so its
  value never passes through this session.
- **Deviations during the run (recorded before the judge pass; no grading result seen).**
  - Gemma 4B first loaded with vLLM's default dtype and produced only `<pad>` tokens (the known float16 overflow).
    The stage was stopped after 75 answers, all empty; they were deleted ungraded from the cache and the run
    folders. The model was reloaded with `--dtype bfloat16` (the pre-registered bf16), and Gemma 12B is served the
    same way.
  - The check after the reload was one outside question, "What is the capital of France? Answer briefly."
    (answer "Paris."), not a benchmark question; then the first 60 benchmark answers were checked for emptiness
    and `<pad>` only (0 of each), never for correctness.
  - Phi-4-mini's answers were checked the same way (emptiness and form only): 14 of 1,500 empty.
  - **Disk full during Llama 8B (about 04:20 UTC).** The laptop disk filled up (the §24 stub output was 20 GB), and
    the Llama 8B `fixed8`/`fixed16` cells stopped with ENOSPC at 54–69 of 100 questions; the FIFO cells were complete.
    The stub output was deleted, one truncated last line of `f0/fixed8/steps.jsonl` was removed, and 9 partial
    `.tmp` cache files were deleted (every cache entry parses). The stage resumed at 04:24 on the same endpoint,
    through the rehearsed resume path, and finished at 04:28:54.
  - **Checks after the resume (before the judge pass):** every Llama 8B cell has exactly 100 distinct questions,
    each with one scored answer; P(all in view) and prompt tokens per question equal Phi-4's in all 15 cells (the
    controller side is reader-independent, so the resume changed no state); none of the 1,500 cached Llama 8B
    answers is empty.
- **Degenerate answers, a descriptive column (declared before the judge pass).** Per reader and arm, the report
  counts raw reader outputs that are: **empty** (blank after stripping); in a **repetition loop** (one word repeated
  at least 9 times in a row, the regex `\b(\w+)( \1\b){8,}`; this is the rule behind the Phi-4-mini figure of
  about 6.5% quoted on the channel during the run); or containing **special-token text** (`<|` or `<pad>`). These
  answers are judged like any other; claims (a) and (b) are unchanged.

### 24a. Result: the controller lifts all six new readers; it bridges a size step only in Gemma 3 (2026-10-09)

Pod 7d7fnvmk5actq0 (A40, CA-MTL-1, $0.59/h): 02:57:37 to about 05:23:40 UTC, about 2.43 h, **about $1.44**
(approved about $0.95–1.25; hard stop $1.48). About 43 min of that ($0.42) was the pod idle after the judge
finished at 04:40, while a question to the user was pending; my error, and a rule now forbids it. Terminated;
list-pods was empty afterwards. The one blind Qwen 7B judge pass judged 8,460 answers. Report:
`runs/_pipelines/exp24_report.py` → `runs/exp24_report.md`, copied to
`docs/research/exp24_reader_families_report.md`; verdicts in `runs/exp24_verdicts.json`.

**Accuracy on the 470 answerable questions** (P(all in view) per arm 0.498, 0.702, 0.747 and prompt tokens 2,964,
1,049, 2,569, as in §23):

| reader | FIFO + floor | `fixed8` | `fixed16` | `fixed8` minus FIFO |
|---|---|---|---|---|
| Llama 3.2 3B | 0.266 | **0.428** | 0.306 | +0.162 (+0.115, +0.209) |
| Llama 3.1 8B | 0.479 | **0.540** | 0.534 | +0.062 (+0.015, +0.111) |
| Gemma 3 4B | 0.357 | **0.504** | 0.421 | +0.147 (+0.102, +0.194) |
| Gemma 3 12B | 0.489 | **0.596** | 0.574 | +0.106 (+0.062, +0.153) |
| Phi-4-mini (3.8B) | 0.253 | **0.385** | 0.362 | +0.132 (+0.079, +0.183) |
| Phi-4 (14B) | 0.502 | **0.630** | 0.591 | +0.128 (+0.085, +0.172) |

**Pre-registered claims.**
- **(a) Lift: all six PASS.** `fixed8` beats FIFO + floor at every size in every family (every lower bound above 0).
  With §23, the drop-in lift now holds in 10 of 11 readers across 5 families; Granite 2B is the one exception.
- **(b) Size step:**
  - **Gemma 3, NON-INFERIOR.** `fixed8` at 4B minus FIFO at 12B is +0.015 (−0.028, +0.057); share of the size gap
    closed 1.11 (0.81, 1.55).
  - **Llama, NOT SHOWN.** −0.051 (−0.100, −0.004); share 0.76 (0.56, 0.98). It falls just short of the margin.
  - **Phi-4, NOT SHOWN.** −0.117 (−0.168, −0.066); share 0.53 (0.34, 0.72). The 3.8B-to-14B step is large.
- **Cross-family "the smaller size gains most":** **HOLDS in Llama** (+0.100, CI +0.038..+0.164); **undetermined in
  Gemma 3** (+0.040, −0.017..+0.098) and **Phi-4** (+0.004, −0.060..+0.066). With §23 (Qwen holds; Granite
  contradicts), it is a pattern of some families, not a rule.

**Where the gain comes from** (unknown share and P(correct | answered), FIFO → `fixed8`):
- Every reader refuses less with `fixed8` (unknown falls in all six).
- Unlike the small Qwen readers in §23, the small readers here also get **more precise**: P(correct | answered)
  rises for Llama 3B (0.466 → 0.569), Gemma 4B (0.439 → 0.552) and Phi-4-mini (0.312 → 0.420). Only Llama 8B
  loses a little precision (0.650 → 0.623) while answering more.
- **Reader-only number, P(correct | all in view) under `fixed8`:** Llama 0.545 → 0.679, Gemma 0.627 → 0.733,
  Phi 0.482 → 0.803 (small → mid).

**Descriptive.**
- **Context-rot probe, `fixed16` minus `fixed8`:** Llama 3B −0.121 (−0.166, −0.077), Gemma 4B −0.083 (−0.126,
  −0.040); the other four include 0 (Llama 8B −0.006, Gemma 12B −0.021, Phi-4-mini −0.023, Phi-4 −0.038). As in
  Qwen, the small readers of Llama and Gemma pay most for a longer prompt.
- **Degenerate answers** (declared before the judge pass): only Phi-4-mini has any. FIFO 2 empty / 39 loop / 0
  special-token; `fixed8` 12 / 13 / 29; `fixed16` 0 / 40 / 1 (of 500). Its answered-wrong share (0.53–0.56) is
  partly these. The other five readers have none.
- **Refusals in other words:** judge-wrong answers with a listed refusal phrase are 0–6 per cell, so the literal
  "unknown" rule misses very little.

**Limitations.** The one prompt was tuned on Qwen 7B (stated in advance); one benchmark; one judge. The Gemma bf16
reload and the Llama 8B disk-full resume are recorded above, with their checks. The Llama order change was made
before any result.

## 25. StreamMemBench: does the §19 head transfer to a benchmark with feedback and follow-ups? (N10; 2026-10-08, pre-registered, closed 2026-10-09 without a paid run)

**Status:** closed without a paid run at the user's decision (2026-10-09); the result is in §25a. The
pre-registration text below is unchanged apart from the review amendments made before the decision.

**The benchmark, checked against the source** (arXiv 2606.14571; github.com/landian60/StreamMemBench at commit
b32965525da5f982599af9794954c616ddeb34ff, cloned to `external/StreamMemBench`, not in git).
- **Text only: confirmed.** Every observation is an EgoLife first-person narration line ("I put my phone away.",
  357,705 lines) or a dialogue line ("Jake: Okay.", 297,770 lines). No images or audio.
- **3,347 five-minute segments: confirmed.** 6 participants × 7 days, 42 day files.
- **15,520 queries: does not match the release.** The paper says 7,760 anchors and 15,520 queries. The released data
  (Chinese and English) has 2,854 evidence segments, **8,107 anchors and 16,214 tasks** (one initial and one
  follow-up per anchor), as its README and metadata say. §25 uses the release.
- **Size and licence.** The repository is 832 MB with the data inside (no separate download); the English release is
  347 MB, copied to `data/streammembench/` (gitignored). Code: MIT. Data: derived from EgoLife, "subject to the
  upstream EgoLife license and Hugging Face access terms". Used here for research only, not redistributed.
- **A task, per anchor** (their `runner.py`): the stream up to the evidence segment is ingested; the **initial
  request** is asked right after that segment; a **user simulator** (an LLM) compares the answer with a hidden
  `expected_behavior` and evidence statement and returns affirm or revise with 1–3 sentences of feedback; on
  revise, the system answers again with the feedback; the interaction is stored; then the **follow-up request** is
  asked.
- **The four metrics are judge-made, not rule-based.** Each is a 0/1 per anchor:
  - fidelity: an LLM auditor says the records saved at that segment contain the evidence statement;
  - initial evidence use: the simulator affirms the first answer;
  - feedback incorporation: the simulator affirms the revised answer (only anchors that were revised);
  - follow-up reuse: the simulator affirms the follow-up answer.
- **Their judge:** DeepSeek-V4-Pro, three votes per decision at temperature 0.5, majority vote (paper; their default
  config). Backbones in the paper: DeepSeek-V4-Flash, Gemini-3-Flash, Qwen3-14B. Their RAG-raw with Qwen3-14B:
  initial use 0.170, feedback incorporation 0.630, follow-up reuse 0.220 (paper's table).
- **Label quality.** Supporting observations carry a day-level `stream_index`. 20,644 of 25,671 point at the same
  text in the same segment; the rest point elsewhere. 1,498 of 8,107 anchors have no matching line.
  `source_span.raw_indices` is not usable as a pointer (out of range for most anchors).

**The environment** (`memctl/envs/streammembench.py`, registered as `streammembench`; tests in
`tests/test_streammembench.py`).
- One episode is one participant-day (mean 126k tokens, close to LongMemEval_s's 103k). The paper keeps one memory
  per participant over 7 days; a day is used so a stream has the scale the controllers were built for.
- Lines are grouped into chunks of about 150 tokens (between LongMemEval's median turn, 91, and mean, 210), never
  across a segment. A chunk is one memory item, labelled "Name, DAY1 11:10:00 - 11:15:00". The day is shown up to
  its last evaluated segment.
- After each evaluated segment, each anchor runs: initial request → (revise: the feedback, quoting the request and
  the first answer) → the stored interaction (their `store_interaction` text) → follow-up request. Feedback and the
  stored interaction enter memory like any other observation, so the controller decides whether they stay in view.
- The user simulator is **their prompt, vendored verbatim** (checked equal to their file), sent as one message.
  One vote at temperature 0, not three at 0.5. Unparsable output counts as revise, and is counted.
- Fidelity is not computed: memctl keeps every chunk, so it would be 1 for every arm (their RAG-raw also scores
  1.00). The label-based "evidence in view" (the §13a P(all in view)) replaces it as a diagnostic. For the
  follow-up, "in view" means the evidence chunks or the stored interaction.
- The budget fraction is taken from the day's stream tokens, without a reference pass: the stream depends on the
  answers, so a reference pass would call the reader (`memctl/harness/runner.py`, `history_tokens`).

**Split (no held-out peeking).**
- **Training side, reserved:** participants A1_JAKE and A2_ALICE. Any tuning (prompt, chunk size, head retraining)
  uses them only.
- **Evaluation:** participants A3–A6 (1,848 evidence segments). **200 segments are drawn with seed 25**
  (`random.Random(25).sample`, `sample_items`): 28 participant-days, 565 anchors. Days are dealt to 7 shards by
  anchor count (`shard_days`).
- **A3_TASHA DAY1 is excluded from every §25 evaluation pool, for good** (amended at review, before any run). It is
  dropped after sampling and sharding (`env.exclude_days` in all 7 configs), so the seed-25 sample and the other
  shards are unchanged.
  - Correction to the review note: the day **is** in the current shards, in shard 6 (`exp25_qwen7b_s6.yaml`), which
    now runs 3 days instead of 4. All 28 A3–A6 days are in the sample, so no shard could lack it.
  - The evaluated set is therefore **27 days, and 565 anchors minus that day's sampled anchors**. That count is not
    computed now (no computation on the A3–A6 sample before the decision); it is printed by the run or by the
    closing evaluation below.
- Nothing in §25 is trained on StreamMemBench. The design above came from LongMemEval and the benchmark's schema.
- **Disclosure:** one stub probe on one evaluation day (A3_TASHA DAY1) printed the label-based evidence-in-view per
  arm before the training-side check was set up. No reader output was seen and no setting was changed because of
  it; the numbers below are training-side only.
- **Order of decisions (from this session's history).** The seed (25) was fixed before that probe: it was
  hard-coded in the config generator, and the probe ran from configs it made. The 200-segment size was also the
  value in the probe's configs, but it was **not final** then: 150 and 200 were still being weighed, and 200 was
  chosen after the probe, from the stub's call counts (cost and power). The shard dealing by anchor count was also
  added after the probe, for cost. Nothing was committed until after the probe.
- **With the exclusion, the probe can no longer affect what is evaluated.**

**Arms** (the §23 arms, unchanged: 5% budget, `memory.count_labels`, lexical search):
- FIFO + floor (3k target, top 5 retrieved): the baseline.
- `fixed8`: the §19 head A (`runs/lme_n4_head_f0_a/checkpoints/policy_best.pt`, trained on half A of LongMemEval fold 0's
  training part only), 32 BM25 candidates, keep none.
- `fixed16`: the same head, 16 shown.
- **Their RAG-raw: not included.** It does not drop in: one record per segment (about 1,700 tokens), top 10, so
  about 17k-token prompts, a 32k context and about 6× the prefill per call, with their own answer prompt. The
  environment supports `unit: segment` if it is wanted later as its own arm.

**Reader and simulator.**
- Reader: Qwen2.5-7B-Instruct (vLLM, A40), `reasoning: false`, compact labels, up to 320 new tokens. Instructions:
  their RAG answer prompt, adapted ("You are {person}'s personal AI assistant … say you don't know …"). No tuning.
- Simulator: **the same Qwen 7B server.** The feedback is part of the stream, so it must run online; it cannot be a
  separate blind pass as in §23. It sees the request, the answer and the hidden labels, never the arm.
- Qwen3-14B (the paper's open backbone): **not included.** memctl's OpenAI backend cannot yet turn off Qwen3's
  thinking mode; that is code to add first. If added, about +1.6–2.3 h (+$0.95–1.35, from §23's 14B/7B time ratio).
- Their API judge (DeepSeek-V4-Pro, 3 votes, plus the fidelity audit): about 11 calls per anchor and arm, so about
  18,600 calls, 19M input and 3M output tokens for §25. It is a paid API with a key, and it would sit in the loop
  while the pod runs. **Not proposed.**

**Pre-registered claim (primary).** Zero-shot transfer: `fixed8` against FIFO + floor, Qwen 7B, the evaluated anchors
(27 days), paired by anchor.
- **Metrics:** follow-up reuse (all evaluated anchors) and feedback incorporation (anchors revised in both arms; their
  unconditional mean is also reported).
- **"Transfers" (non-inferior):** on both metrics, the 95% interval of `fixed8` minus FIFO has a lower bound above
  −0.03. **"Better":** lower bound above 0 on both. **"Worse":** upper bound below 0 on either. Otherwise
  undetermined.
- Intervals: bootstrap over anchors (paired), as §23; a bootstrap over days (anchors share a day's memory) is
  reported beside it as a check.
- **Direction expected, written before any run: `fixed8` below FIFO on both metrics**, so the claim is expected to
  fail. Reason: the free training-side check below.
- Secondary (descriptive): initial evidence use, `fixed16`, prompt tokens, evidence in view, the share of answers
  with a refusal phrase (the §24 list: "don't know", "do not know", "not mention", "no information", "cannot
  determine", "unable to"), simulator parse failures per arm (flagged if above 5% of calls).

**Free training-side check (A1 and A2, 100 segments with seed 25, 275 anchors, stub reader; evidence in view at
read time):**

| arm | initial | revised | follow-up (evidence or stored interaction) | reader prompt tokens |
|---|---|---|---|---|
| FIFO + floor | 0.978 | 0.960 | 1.000 | about 2,990 |
| `fixed8` | 0.433 | 0.291 | 0.233 | about 1,000 |
| `fixed16` | 0.516 | 0.382 | 0.338 | about 1,800 |

- The benchmark asks each request right after its evidence segment, and the feedback and stored interaction are the
  newest items. A recency window keeps all of them. The head archives everything and searches with the request,
  which is usually future-oriented ("what should I remind them to do tonight?") and shares few words with the
  evidence.
- So §25 mainly tests whether a head trained on LongMemEval's "old evidence" questions transfers to a benchmark
  where the evidence is recent. The check says it does not reach the evidence. A run would measure how much that
  costs the reader; it is unlikely to show a transfer.
- Not built, and a separate pre-registration if wanted: the head plus a recent window (keep the last ~3k tokens,
  head fills the rest), chosen on the training side.

**Confounds and limitations, stated before any run.**
- The head was trained on LongMemEval chat turns; this stream is first-person narration and multi-speaker dialogue.
  A loss may be the domain, not the head.
- The reader and the simulator are the same model, so the simulator may favour its own phrasing. It is the same for
  every arm, but not neutral; the paper's numbers (DeepSeek judge) are not comparable to ours.
- One vote at temperature 0, not three at 0.5; one day of memory, not seven; English release only.
- About 18% of anchors have no usable evidence label; for them the whole segment counts, so evidence in view is
  lenient there.

**Stub check (done, free; prelaunch rule).**
- Full size: 7 shards × 3 arms = 21 cells, 28 days (before the A3_TASHA DAY1 exclusion), 565 anchors per arm, stub reader and stub simulator. The stub
  simulator's output never parses, so every initial answer is revised: the call count is an upper bound. Per arm
  1,695 reader and 1,695 simulator calls; **10,170 calls in all**. Mean reader prompt: FIFO 2,952 tokens, `fixed8`
  999, `fixed16` 1,826; simulator prompt about 870 (more with real, longer answers). Largest cell: 261 reader calls.
- **Resume on a new endpoint:** all 7 sweeps were killed after 14 s (cells at 0–3 of 3–5 days), then rerun with a
  different `base_url` for the reader and the simulator. All 21 cells finished with every day once (no duplicates,
  none refused); the cells that had finished days resumed from them. An interrupted day restarts, and its calls
  come back from the generation cache.
- **Peak RSS 0.64 GB** per sweep process (3 cells), so 7 sweeps need about 4.5 GB on the launching machine.
- All stub cache entries were deleted afterwards (`cache/generations_exp25_qwen7b`, `cache/simulator_exp25`), so the
  real run starts empty.

**Cost (NEEDS SPEND; from §23's measured stage times on the same A40, $0.59/h).**
- §23 measured: first model ready 4.4 min after create; Qwen 7B answered 1,500 LongMemEval questions in 10.3 min
  (about 0.41 s per call across 15 cells; mean prompt about 2,200 tokens, up to 256 new tokens).
- §25: up to 10,170 calls (reader prompts mean about 1,930 tokens, up to 320 new tokens; simulator prompts about
  1,000 tokens, short JSON replies). If 60–80% of initial answers are revised (the paper's RAG-raw rates), about
  9,200–9,800 calls. At 0.3–0.41 s per call: **about 50–70 min of generation**.
- With pod start, model load and checks: **about 1.0–1.4 h, so about $0.60–0.85.** One model for everything (Qwen
  7B), one stage.
- **Hard stop at 2 h after create (about $1.18).** If the pod's price is not $0.59/h, the figure is re-quoted
  before the run starts.
- **Triage rule, pre-declared:** if fewer than half of the days are done 75 min after create, `fixed16`
  (descriptive) is stopped first, so the two arms of the claim finish.

**If §25 closes without a paid run (pre-declared result, free; amended at review).**
- One label-only evaluation: evidence in view at the initial, revised and follow-up requests, for FIFO + floor,
  `fixed8` and `fixed16`, on the evaluation anchors (the 565 of the seed-25 sample, less A3_TASHA DAY1's), with the
  stub reader and stub simulator, from the committed configs. Reported as §25's result, with 95% intervals by
  anchor.
- It is computed **exactly once**, and **only after the user decides there is no paid run**.
- If the user chooses the paid run instead, it is **not** computed beforehand: that would be a peek at the
  evaluation side. The paid run reports the same numbers as its diagnostic.
- Not executed as of this amendment.

### 25a. Result: closed without a paid run; the head does not transfer, because recency wins (2026-10-09)

**Decision.** At 05:23 UTC on 2026-10-09 the user closed §25 as a free result (the reviewer's recommendation), so
there is no paid run. The pre-declared closing evaluation was then computed **once**, at 05:27 UTC, from the
committed configs with only the reader and simulator set to stub (`runs/_pipelines/exp25_close_cfg.py`,
`exp25_close_eval.py`; per-anchor table in `runs/exp25_close_eval.json`). $0.

**Evaluated set:** 541 anchors on 27 participant-days (the seed-25 sample of 565, less A3_TASHA DAY1's 24); the
excluded day is absent from the output.

**Evidence in view** (share of requests with all their evidence lines in the reader's prompt; 95% interval by
anchor):

| request | FIFO + floor | `fixed8` | `fixed16` | `fixed8` minus FIFO | `fixed16` minus FIFO |
|---|---|---|---|---|---|
| initial | 0.930 (0.908, 0.950) | 0.318 (0.279, 0.355) | 0.401 (0.359, 0.440) | −0.612 (−0.656, −0.569) | −0.529 (−0.573, −0.486) |
| revised | 0.908 (0.882, 0.932) | 0.229 (0.194, 0.264) | 0.290 (0.253, 0.329) | −0.678 (−0.719, −0.638) | −0.617 (−0.662, −0.573) |
| follow-up | 0.996 (0.991, 1.000) | 0.100 (0.078, 0.126) | 0.181 (0.150, 0.214) | −0.896 (−0.921, −0.869) | −0.815 (−0.847, −0.782) |

Mean reader prompt: FIFO 2,949 tokens, `fixed8` 998, `fixed16` 1,824.

**What it means.**
- **The §19 head does not transfer zero-shot to StreamMemBench.** The pre-declared direction (`fixed8` below FIFO)
  holds on every request type, by a wide margin; it matches the training-side check (A1/A2).
- **Why:** this benchmark asks each request right after its evidence segment, and the follow-up right after the
  stored interaction, so the most recent turns are the evidence. FIFO keeps them; the head archives everything
  and searches with a forward-looking request that shares few words with the evidence. The worst gap is on the
  follow-up, the request that most needs the earlier exchange.
- The reader was not run, so this is a statement about evidence in view, not accuracy. §23 shows what a reader
  does when the evidence is not in view (P(correct | in view) against out of view), so a paid run would very
  likely only have confirmed the direction.
- The "revised" row covers every anchor, because the stub simulator asks for a revision every time; with a real
  simulator only some anchors are revised.

**Next, if wanted:** a head-plus-recent-window arm, designed on A1/A2 and the LongMemEval training folds only,
would be a separate pre-registration.

## 26. An open Jev stand-in as the memory controller (OpenJev; 2026-10-08, pre-registered, run 2026-10-09)

**Status:** pre-registered and accepted at review (94b3f72, 881c7c4). The review added arm B (OpenJev as selector),
the GPU plan and the step order below, written before any spend. The free parts (fork check, code, stub checks of
both arms) are done. **Run 2026-10-09: the smoke, arm B's fill and phase 2, about $0.64; arm A not run, by the
user's decision. The result is in §26a.**

**Why.** The JEV controller (`memctl/controllers/jev.py`) has never run on a real model: TypeSafe's Jev needs a paid
key. OpenJev is an open server with the same wire API that we can run on our own GPU. It gives the framework's
"lightweight decision model" arm its first real numbers.

**What OpenJev is, stated plainly.**
- It **approximates** Jev: same request and answer shapes, a different model (DiffusionGemma 26B-A4B, read as a
  diffusion canvas). It is not TypeSafe's model and is not affiliated with TypeSafe.
- **Its accuracy against real Jev is unpublished.** The repository reports only its own dev-set accuracy (76.7%
  on 2,501 questions, CHANGELOG 0.6.0) and no comparison with Jev. No result here is a result about Jev.
- A different project, `openjev/openjev` on Hugging Face (a Qwen3.5-based 27B, CC BY-NC 4.0, about 55 GB of
  weights on the hub), shares the name. We do not use it.

**Fork check (2026-10-08; code cloned to `external/openjev`, gitignored, read only; nothing of it was run).**
- **Pinned commit:** `75f22b6dad8c360fdba0e0ebd3dc0a1187628f60` (2026-10-06). Its server code is identical to the
  0.6.0 release commit 7a01c81; only README, CHANGELOG and a benchmark script differ.
- **Licence: confirmed.** `LICENSE` is Apache 2.0 and `pyproject.toml` says Apache-2.0.
- **Model: confirmed.** `nvidia/diffusiongemma-26B-A4B-it-NVFP4`, Apache-2.0 on Hugging Face, not gated, 18.86 GB,
  revision `ec4ff3df`. Its quantisation config asks for NVFP4 weights **and an FP8 KV cache** (this matters below).
- **Image: confirmed, with a trap.** `razorback16/openjev:0.6.0` exists (pushed 2026-10-06, 6.4 GB compressed,
  digest `sha256:07c2e9f5fd98…2b5beb`). **`latest` still points to 0.2.0**, so we pin the digest, never `latest`.
- **CUDA 13: confirmed.** Base `nvidia/cuda:13.0.1-base-ubuntu24.04`, torch 2.13 (cu130), vLLM main at `a3e0243b`.
  The pod's host driver must support CUDA 13; that is a filter at create time.
- **"24 GB or more": claimed, untested by us.** The README says so; the author tested only an RTX PRO 6000
  Blackwell (96 GB, sm_120). We would leave room: 32 GB or more, or `OPENJEV_MAX_MODEL_LEN=32768` on 24 GB.
- **API compatibility with what `jev.py` sends and parses: confirmed from the code** (`openjev/api.py`,
  `openjev/engine.py`). `POST /v1/systemone` takes `{model, state, questions}`; `state` may be a JSON object, as
  ours is; a `choice` takes `criteria: {name: description}`. The answer is `{model, answers: {id: {choice,
  probabilities, confidence}}, usage: {input_tokens, output_tokens}}`, keyed by our option names. It accepts our
  default `jev-latest` as an alias. Against TypeSafe's live API it could not be checked (no key, no spend).
- **Differences that matter here.**
  - Confidence is `1 − H(p)/ln K`; TypeSafe documents `(n·peak − 1)/(n − 1)`. `jev.py` uses the server's value. The
    repair step ranks by P(KEEP), not by confidence, so decisions do not depend on it; confidences are logged
    only.
  - The server reads about 12 questions at a time and **prefills the whole state once per group**, with that
    group's questions in the system prompt before the state. A 40-item request is several full prefills.
  - Uncertain answers trigger three automatic re-reads (prefix-cached, not billed).
  - A same request gets the same answer (the noise seed is a hash of the request).

**Does it run on an A40? Not as shipped (reasoned from the pinned vLLM source; not tried).**
- NVFP4 weights do load below Blackwell: vLLM `a3e0243b` sets ModelOpt NVFP4's minimum to SM75 and falls back to
  Marlin weight-only FP4 (bf16 compute) for both dense and MoE layers (`modelopt.py`, `kernels/linear/__init__.py`,
  `fused_moe/oracle/nvfp4.py`). The A40 is SM86, so it has no FP4 tensor cores but can run the weights.
- **But the image forces `--attention-backend TRITON_ATTN`, and the checkpoint's FP8 KV cache is refused by that
  backend below SM89** (`v1/attention/backends/triton_attn.py`: "native FP8 (fp8e4nv) requires SM89+"). So on an
  A40 the server should stop at start-up unless we add `OPENJEV_VLLM_ARGS="--kv-cache-dtype bfloat16"` (the
  entrypoint appends it last). That path is untested by the author.
- **GPU classes, by what they need:**
  - Ada, SM89 (L40S 48 GB; RTX 4090 or L4 24 GB): FP8 KV native, NVFP4 through Marlin; runs with no flag, but the
    author has not tested it. **This is the cheapest class that runs the image as shipped.**
  - Blackwell, SM120 (RTX 5090 32 GB; RTX PRO 6000 96 GB, the tested card): native NVFP4, the fastest. The RTX
    5090 is the cheapest card of the tested architecture.
  - Ampere, SM86 (A40): only with the KV flag above, and the slowest per token.
- Prices for these classes are in neither repository's docs. The reviewer read them live; they are in the GPU plan
  below.

**Arms** (the same 500 LongMemEval test questions, five folds, Qwen2.5-7B reader, frozen 7b5fc30 prompt, as §19
and §23):
- FIFO + floor at 5% with the 3k target (the baseline).
- `fixed8`: the §19 head A, 8 of 32 BM25 candidates. "The §19 head" in the brief is read as this arm: it is the
  head's deployed form, and adaptive k was stopped in §19b.
- **Arm A, "OpenJev as memory manager"** (`jev`, `mode: manage`, label `openjev_manage`, model `openjev-latest`),
  the adapter's defaults:
  - Called only when memory is over budget (`call: pressure`), one Choice question per unpinned active item (KEEP
    or MOVE_TO_ARCHIVE, the operations this benchmark allows), up to 40 items a request.
  - At the question, a choice per archived BM25 candidate (RETRIEVE or leave). It gets 32 candidates, the §19
    head's pool. Its BM25 is `jev.py`'s plain one, without labels.
  - `repair: true`: when its choices leave memory over budget, the adapter archives the kept items it was least
    sure about.
  - **Counted against it, as the `jev.py` docstring says:** repaired items and the harness's forced evictions or
    returned retrievals are part of the arm. They are reported per question, with the forced share of removals,
    and never excluded.
- **Arm B, "OpenJev as selector"** (`jev`, `mode: select`, `select_k: 8`, label `openjev_select`; added at review,
  2026-10-08, before any spend). Built so that B against `fixed8` differs only in who ranks:
  - Everything but the current input is archived on arrival, as the head's `keep_none`. No call until the question.
  - At the question, the same 32 BM25 candidates `fixed8` sees (`retrieve_candidates: 32`, the same plain BM25 over
    the same archive). **One request, one Choice question whose 32 options are the items** ("Which memory item does
    the assistant most need to answer the current input?"); the state is the question and the 32 items, described
    as in arm A.
  - **It shows the reader 8 items, as `fixed8`:** the 8 with the highest probability in that one answer
    distribution (ties keep the BM25 order). A single Choice gives a ranking over all 32, so no per-item KEEP
    question is needed; this is the listwise analogue of the head's top 8 by logit.
  - If the search returns 8 or fewer candidates, all are shown and no call is made, as the head does. (It never
    happened: all 50 training probes and all 500 stub questions had more.)
  - **`jev.py` could not do this before;** the smallest change was added (`mode: select`, off by default, `manage`
    unchanged), with a unit test on the local fake server.
- FIFO + floor and `fixed8` are rerun in the same session for pairing, as in §23. No §23 row is reused.
- Nothing in either arm was tuned. Every setting is a `jev.py` default or comes from the §19/§23 configs. No
  held-out question was looked at to choose anything.

**Two phases, one pod at a time (the design; no idle pod).**
- The JEV requests do not depend on the reader: in LongMemEval the question is the last step of every episode
  (checked on all 500 stub episodes), so every OpenJev call happens before the reader answers.
- **Phase 1, OpenJev's own pod (their image, pinned digest).** `configs/sweeps/exp26/exp26_fillB_f{0..4}.yaml`
  (arm B) and `exp26_fillA_f{0..4}.yaml` (arm A): one arm at a time, stub reader, five folds at once. Every answer
  is stored in `cache/jev_exp26` under a hash of the request, with its latency and the server's own time.
- **Phase 2, our reader pod (the §23 image, vLLM 0.8.5, A40).** `exp26_qwen7b_f{0..4}.yaml`: FIFO + floor,
  `fixed8` and arm B; `exp26_qwen7bA_f{0..4}.yaml`: arm A alone, so its question count can be cut to a prefix.
  Both OpenJev arms replay the stored answers only (`cache_only`); a request not stored is an error, so phase 2
  cannot call any server.
- **Why not one GPU for both.** The two servers are different images (CUDA 13 vLLM main against vLLM 0.8.5), and a
  RunPod pod runs one image. Serving Qwen 7B inside OpenJev's newer vLLM would change the reader that §23 pairs
  with. Running both pods at once would leave the reader pod idle for all of phase 1.

**Claims, pre-registered** (470 answerable questions, paired by question, 95% bootstrap intervals; margin −0.03 as
in §23). Each arm is tested on its own; there is no claim on A against B.
- **Arm A, OpenJev as memory manager** (written before review; unchanged):
  1. **Claim A1, against the baseline.** A minus FIFO + floor: **better** if the lower bound is above 0, **worse**
     if the upper bound is below 0, otherwise undetermined.
  2. **Claim A2, against the learned head.** A minus `fixed8`: **non-inferior** if the lower bound is above −0.03.
     Reported the other way too: `fixed8` is **better** if the upper bound of A minus `fixed8` is below 0.
  - Expectation: A2, `fixed8` better (trained on this benchmark's evidence labels; OpenJev has never seen the task).
    A1 undetermined: nothing caps what A keeps or retrieves except the budget, so its prompts may run near the 5%
    budget (about 5,500 tokens against FIFO's 2,964 and `fixed8`'s 1,049), and §19c/§23 showed that longer prompts
    cost this reader accuracy.
- **Arm B, OpenJev as selector** (written at review, before any run):
  1. **Claim B1, against the learned head (the main test of B).** B minus `fixed8`: **non-inferior** if the lower
     bound is above −0.03; `fixed8` **better** if the upper bound is below 0; B **better** if the lower bound is
     above 0. Same candidates, same k, same prompt length class: only the ranker differs.
  2. **Claim B2, against the baseline.** B minus FIFO + floor: **better** if the lower bound is above 0, **worse** if
     the upper bound is below 0.
  - Expectation: B2 better (with the same pool and k, `fixed8` beat FIFO + floor by +0.10 in §23, and B shows about
    the same number of tokens). B1 undetermined, with a lean to `fixed8`: the head was trained on this benchmark's
    evidence labels, but OpenJev reads every candidate's full text beside the question, which the head does not.
  - The reader-free part of B1 is P(all in view), B against `fixed8` on the same 32 candidates: it says how much of
    any accuracy gap is the ranking itself.
- **Cost axis (descriptive; its own pod, not an API price): GPU-seconds per 1,000 questions.**
  - Each OpenJev arm: its phase-1 pod seconds after the model is ready, divided by questions done, × 1,000 (one
    GPU). Load time is reported apart. The sum of the server's `Server-Timing` model time is printed beside it (it
    can exceed wall time, since reads run in parallel).
  - `fixed8` and FIFO + floor use no GPU: about 134 and 80 CPU-seconds per 1,000 questions on the laptop (stub run).
  - Reader prompt tokens per arm, beside accuracy, as always.
- **Reported beside the claims (descriptive):** the §13a decomposition (P(all in view), P(correct | in view)), the
  three disjoint shares (answered-correct, answered-wrong, unknown), P(correct | answered), OpenJev calls per
  question, items retrieved, repaired items and forced actions per question, and false answers on the 30
  unanswerable questions.
- **If the hard stop cuts arm A's phase 1** (the triage rule, pre-declared): each fold runs its questions in the
  fixed fold order, so a cut leaves the first m of every fold. Arm A's phase 2 (`exp26_qwen7bA`) then runs on
  exactly those m per fold, and A's claims are paired with the other arms on those questions. The claims are
  pre-registered on all 470; on fewer, A's differences are reported as descriptive and labelled underpowered, with
  n. Arm B is not affected: it runs in full first.
- Phase-1 output is checked only for calls, timings and errors. Nothing in an arm changes after its phase 1 starts.

**Code (free; 94b3f72, arm B in the follow-up commit).**
- `jev.py`: the endpoint is config `endpoint`, then config `base_url`, then `$JEV_BASE_URL`, else TypeSafe's URL
  (unchanged); the model is config `model`, then `$JEV_MODEL`, else `jev-latest`. A server other than TypeSafe's
  never receives the TypeSafe key (it gets `$OPENJEV_API_KEY` if set, otherwise no header) and costs $0 per token.
  Optional retries on 429/503/529, the server's model time, and the answer cache. All off by default.
- `mode: select` (arm B) with `select_k`; the default `mode: manage` is the code path as before. The fake answerer
  gives a selection question's options weights by similarity to the input.
- `tests/fake_jev_server.py`: a local HTTP stand-in in OpenJev's answer shape, answering with the hand-written fake
  heuristic. Tests (`tests/test_controllers_llm.py`, 23 pass): the default URL, model and price are unchanged; env
  and config overrides work, config wins; calls reach the local server with no Authorization header; a replay
  from the cache needs no server, and a miss is an error; select mode archives on arrival with no call, then makes
  one call with one 4-option question and retrieves the 2 most probable; the default mode is `manage` and an
  unknown mode is refused. The full suite: 216 pass; the 2 failures in `test_import_boundaries.py` are on main
  already and untouched here.

**Stub check (done, free; against the local fake server, no network).**
- **Full size, the paid layout (five folds at once, three workers each, `guard.sh`).**
  - Three arms × five folds (FIFO + floor, `fixed8`, arm A), the fake OpenJev live over HTTP: stopped by SIGKILL
    after 30 s (FIFO 63–68, `fixed8` 52–57, A 21–22 of 100 per cell), then resumed against a different JEV server
    and a different reader `base_url`. All 15 cells resumed (`resumed: true`), each ended at 100 distinct
    questions. Peak RSS 0.45 GB per sweep process.
  - P(all in view): FIFO + floor 0.500, `fixed8` 0.700 over all 500 (§23: 0.498 and 0.702 on the 470 answerable);
    prompt tokens 2,963 and 1,050 (§23: 2,964 and 1,049). The OpenJev numbers in the stub (A: 0.116, 18.6 forced
    actions per question) are the fake heuristic's and say nothing about OpenJev.
- **The two phases, arm A.**
  - Phase 1 (fill), stopped after 25 s (12–14 of 100 per fold) and resumed against a second fake server: all five
    cells ended at 100. 15,642 requests reached the servers (2,075 + 13,567), exactly the 15,642 stored answers:
    the resumed episodes replayed from the cache and no request was sent twice. Peak RSS 0.11 GB per process.
  - Phase 2 (replay), with both fake servers shut down and a dead JEV URL: all 15 cells finished; all 15,642
    OpenJev calls came from the cache; on every one of 247,250 steps the memory and the actions equal phase 1's.
    Peak RSS 0.27 GB per process.
- **The two phases, arm B (after review; the configs as committed).**
  - Phase 1 (`fillB`), stopped by SIGKILL after 15 s (64–65 of 100 per fold) and resumed against a second fake
    server: all five cells resumed (`resumed: true`) and ended at 100 distinct questions. 500 requests in all (321 +
    179), one per question, none sent twice. Peak RSS 0.29 GB per process.
  - Arm A's fill was rerun under its new label (15,642 requests). Then phase 2 (`exp26_qwen7b` with FIFO + floor,
    `fixed8` and B; `exp26_qwen7bA` with A), servers down, dead JEV URL: all 20 cells finished with no cache miss.
    Every one of the 500 episodes of each OpenJev arm equals its phase-1 row on all 37 compared fields (episode-
    level check; step logs were switched off this time to save disk). Peak RSS 0.26 GB per process.
  - All stub run folders and caches were deleted afterwards, so the real run starts from empty caches.
- **Request size, counted with DiffusionGemma's own tokenizer, on training conversations of fold 0 only**
  (OpenJev's layout replicated, 12 questions per read):
  - **Arm A** (313 requests from 10 conversations): the state is about 8,500 tokens (p90 9,600, max 15,600); about
    23 questions a request (max 40), so 2 reads; **about 23,300 prefill tokens a request** (p90 33,800). Calls per
    question are the fake heuristic's: **31.3**. A's real rate is unknown until the smoke: it depends on how much
    it keeps (the ceiling is about one call per step, about 510).
  - **Arm B** (50 conversations, one request each): the state is **about 13,700 tokens** (median 13,600, p90 15,800,
    max 21,600), larger than the 3–6k first guessed, because 32 LongMemEval turns are long; one question, one read;
    **about 14,300 prefill tokens per question**, 1 call per question, 500 calls.

**GPU plan (written before any smoke; live secure-cloud prices read by the reviewer on 2026-10-08).**
- Prices per hour: A40 $0.59, L4 $0.59, RTX 4090 $0.89, RTX 6000 Ada 48 GB $0.99, L40S 48 GB $1.09, RTX PRO 4500
  Blackwell 32 GB $0.72.
- **First try: A40 with `OPENJEV_VLLM_ARGS="--kv-cache-dtype bfloat16"`.** This is a declared deviation from the
  shipped config: the KV cache is bf16, not the checkpoint's FP8. Answers may differ slightly from the shipped
  server's; it is reported as such.
- **Cap: 10 minutes from pod create to "server ready"** (`GET /v1/models` answers). If the server refuses to start
  or is not ready in 10 minutes, the A40 is terminated.
- **Fallback: an RTX 6000 Ada ($0.99/h) or an L40S ($1.09/h), 48 GB, SM89, which run the image as shipped** (FP8 KV
  native). Whichever has stock; the price is re-quoted at that rate before phase 1.
- **No 24 GB cards** (L4, RTX 4090): with states up to about 22k tokens and five folds at once, the KV cache would
  be tight.
- The RTX PRO 4500 Blackwell ($0.72/h) is the tested architecture but has not been checked against this image's
  kernels; it is not in the plan.
- **Step 1 approved by the user (2026-10-09, 05:23 UTC, at most $0.92); arm B's fill, arm A and phase 2 are not
  approved.** Launch settings, fixed before the pod: image `razorback16/openjev@sha256:07c2e9f5fd98b9f6b525014bd6278a7
  37f1d9c0bfc671bb93296b03a8c2b5beb` (the 0.6.0 index digest, checked on Docker Hub), port 8080/http, 80 GB container
  disk, `OPENJEV_VLLM_ARGS="--kv-cache-dtype bfloat16"` (the declared A40 deviation) and
  `OPENJEV_MAX_MODEL_LEN=32768` (shipped default 65,536; a bf16 KV cache takes twice the FP8 memory, and the
  largest measured state is about 22k tokens, so no request is affected). Ready means `GET /health` answers
  within 10 minutes of create; the smoke then runs `exp26_smoke.yaml` through `runs/_pipelines/exp26_smoke.sh`.
- **Step 1 result (smoke, 2026-10-09).** Pod fhuj9gabzst39f (A40, CA-MTL-1, $0.59/h), 05:26:31 to about 05:45:15
  UTC, about 0.31 h, **about $0.18**; terminated, list-pods empty.
  - **The A40 works with the declared flag.** The image pulled in 3 min (digest confirmed in the pod log); vLLM
    accepted `kv_cache_dtype=bfloat16`, loaded the weights in 32 s (18.15 GiB; weight-only FP4 through Marlin, as
    predicted), and `/health` answered at 05:34:01, 7.5 min after create. No fallback card was needed.
  - **One bug, fixed before any answer was stored:** the first smoke call got HTTP 403, because RunPod's proxy
    refuses Python's default User-Agent. `jev.py` now sends `User-Agent: memctl`, as `memctl/llm.py` already did
    (1f98053; 23 jev tests pass). The rerun stored every answer.
  - About 7 min of the pod time was idle after the smoke finished (05:37:41), because my wait loop matched its
    own command line; disclosed.
  - **Measured, 2 training conversations of fold 0 (no test question), stub reader, both arms at once:**

    | | arm A (memory manager) | arm B (selector) |
    |---|---|---|
    | OpenJev calls per question | 21.5 | 1 |
    | wall seconds per call | 4.4 (median 4.0) | 6.4 |
    | OpenJev wall seconds per question | about 94 | about 6.4 |
    | input tokens per call (server `usage`) | about 21,100 | about 12,700 |
    | server model seconds per call | 12.3 | 8.7 |
    | forced evictions per question | 1 | 0 |

  - **Cost axis, from this smoke (single stream):** arm A about 94,000 OpenJev wall seconds per 1,000 questions,
    arm B about 6,400; `fixed8` uses no GPU (about 134 CPU seconds). Concurrency across folds will lower wall time;
    how much is measured in phase 1.
  - **Re-quote for the next steps (not approved; for the user):**
    - Arm B's fill on a new A40: 7.5 min to ready, then 500 calls at about 6.4 s each, five folds at once:
      about 15–55 min in all, **about $0.15–0.55, hard stop 60 min ($0.59)**.
    - Arm A's fill: 500 × 94 s is 13 h single-stream; even a threefold gain from concurrency is about 4.4 h
      (about $2.60), so under the $3.00 cap only a prefix of each fold is likely (the pre-declared cut rule).
    - Phase 2 (all arms on our vLLM 0.8.5 A40): about $0.25, hard stop 45 min.
- **Decision (the user, 2026-10-09, after the smoke): arm B's fill and phase 2 are approved (about $0.40–0.80);
  arm A is not run.** Arm A is reported from the smoke only, on the cost axis (calls, seconds and tokens per
  question): **cost measured in the smoke; accuracy not run, by decision.** Its pre-registered claims A1 and A2
  are marked **not tested**, not dropped. At about 94 OpenJev-seconds per question it is about 26 A40-hours (about
  $15) per 1,000 questions, against about $0.03 per 1,000 for the reader and no GPU for the head. Phase 2 runs FIFO + floor, `fixed8` and arm B
  (`exp26_qwen7b_f{0..4}.yaml`) and then the one blind Qwen 7B judge pass (`runs/_pipelines/exp26_judge.py`, the
  §23 judge with these arms and `cache/judge_exp26`); claims B1 and B2 are graded as pre-registered.

**Sequence and cost (NEEDS SPEND; measured inputs, one guess, marked).**
- **Measured:** §23 Qwen 7B answered 1,500 LongMemEval questions in 10.3 min; first model ready 4.4 min after
  create, model switches 3.1–6.2 min; judge pass 3.1 min for 7,050 answers. Here: arm A about 0.73M prefill tokens
  per question (31.3 calls × 23,300, if OpenJev calls as often as the fake); arm B about 14,300 per question.
- **From OpenJev's README (their measurement, RTX PRO 6000):** about 31K prompt tokens/s with 8K-token states at
  full load.
- **GUESS (until the smoke measures it): an A40 at 6–10K prompt tokens/s**, an Ada 48 GB card about the same or a
  little faster (weight-only FP4 through Marlin on both). For arm A that is 73–122 GPU-s per question, 10–17 h for
  500; for arm B, 1.4–2.4 s per question, **12–20 min for 500**.
1. **Step 0, the smoke, on their container** (razorback16/openjev pinned by digest; not our vLLM 0.8.5 image),
   before any experiment cell.
   - A40 attempt: at most 10 min to ready (about $0.10 if it fails). If ready: `GET /v1/models`, then
     `configs/sweeps/exp26/exp26_smoke.yaml` (arms A and B on 2 training conversations: about 60 A calls and 2 B
     calls, one at a time). Measured: A's seconds per call and calls per question (the cost axis), B's seconds per
     call, prefill tokens per second, and the server's `usage.input_tokens` against the counts above. **About 25
     min, about $0.25; hard stop 30 min on the A40 ($0.30).**
   - Fallback, only if the A40 fails: the same on an RTX 6000 Ada or L40S. Image pull and weights 15–25 min, smoke
     about 10 min. **Hard stop 45 min ($0.74 at $0.99, $0.82 at $1.09).**
   - **Smoke ceiling: about $0.30 if the A40 works; about $0.92 if it fails and the fallback runs** ($0.10 + $0.82).
2. **Step 1, arm B's full phase 1** (`exp26_fillB`), on the same pod, right after the smoke (no second load).
   - About 12–20 min by the guess; re-quoted from the smoke's measured rate before it starts.
   - **Hard stop 45 min of pod time ($0.44 on the A40, $0.82 at $1.09).**
3. **Step 2, arm A's phase 1** (`exp26_fillA`), on the same pod, **only if the smoke's measured rate puts all 500
   questions under the $3.00 cap** (500 × measured calls per question × measured seconds per call × the pod's
   price, plus 10% margin). Otherwise it runs until **the $3.00 hard stop** and is cut by the prefix rule above.
   The OpenJev pod is terminated after this step.
4. **Step 3, phase 2, all arms on our vLLM 0.8.5 A40** ($0.59/h): about 5 min to ready, about 10.3 min for FIFO +
   floor, `fixed8` and B (1,500 answers), up to 3.5 min more for A (up to 500), about 1 min to judge about 2,000
   answers. **About 25 min, about $0.25; hard stop 45 min ($0.44).**
- **Ceilings.** Without arm A: about $1.18 if the A40 works ($0.30 + $0.44 + $0.44 hard stops; expected about
  $0.65) and about $2.18 on the fallback ($0.92 + $0.82 + $0.44). With arm A at its cap: add $3.00, so **at most
  about $5.20**. Each step needs the user's approval; none is approved yet.
- Free disk is checked with `df -h /` before and during every local run of the pipeline (at least 15 GB free).
- The pods are verified terminated with list-pods after each phase.

### 26a. Result: OpenJev as a selector beats FIFO but not the head, at about twice the head's prompt (2026-10-09)

**Pods and cost** (all A40 at $0.59/h, each terminated, list-pods empty after each):
- Step 1, smoke: fhuj9gabzst39f, about 0.31 h, about $0.18 (above).
- Phase 1, arm B's fill: 5pv7jpxgjpfwnj (OpenJev image), 05:47:09 to about 06:17:20 UTC, about 0.50 h, about $0.30.
  Ready at 05:54:39; 500 answers (100 per fold), 0 errors, stored in `cache/jev_exp26`.
- Phase 2: ttrilgxqznp6v4 (vLLM 0.8.5, Qwen2.5-7B), 06:17:21 to about 06:33:50 UTC, about 0.27 h, about $0.16.
  Arm B replayed only stored answers (`cache_only`; its URL was `http://unused.invalid`). 1,500 answers, then the
  one blind Qwen 7B judge pass (1,410 answerable answers judged).
- **§26 in all: about $0.64.**
- **Served model string:** the server reports its model as `openjev-0.1` (OpenJev's own model version, which
  `openjev-latest` aliases), from the release image `razorback16/openjev:0.6.0` pinned by digest
  `sha256:07c2e9f5…2b5beb`. They are two version labels for one artefact, not a mismatch.

**Accuracy on the 470 answerable questions** (report: `runs/_pipelines/exp26_report.py` → `runs/exp26_report.md`;
verdicts in `runs/exp26_verdicts.json`):

| arm | accuracy (95% CI) | unknown | P(correct \| answered) | P(all in view) | P(correct \| in view) | prompt tokens |
|---|---|---|---|---|---|---|
| FIFO + floor | 0.445 (0.400, 0.489) | 0.323 | 0.657 | 0.498 | 0.705 | 2,964 |
| `fixed8` (the §19 head) | **0.526** (0.481, 0.570) | 0.157 | 0.624 | 0.702 | 0.652 | 1,049 |
| arm B, OpenJev as selector | 0.489 (0.445, 0.534) | 0.245 | 0.648 | 0.611 | 0.627 | 2,373 |

**Pre-registered claims.**
- **B1, arm B minus `fixed8`: −0.036 (−0.081, +0.009), NOT SHOWN.** The lower bound is below the −0.03 margin, and
  the upper bound is above 0, so neither "non-inferior" nor "`fixed8` better" is shown. This matches the
  expectation written before the run (undetermined, leaning to `fixed8`).
- **B2, arm B minus FIFO + floor: +0.045 (+0.004, +0.087), BETTER**, as expected.
- **A1, A2: not tested, by decision** (above).

**What it means.**
- Given the same 32 candidates and the same 8 slots, OpenJev picks evidence less well than the head: all evidence
  in view 0.611 against 0.702. The head was trained on this benchmark's evidence labels; OpenJev has never seen
  the task, and it still beats FIFO.
- **Its picks are longer:** 2,373 prompt tokens against `fixed8`'s 1,049 for the same 8 items, so it prefers long
  turns. It reaches below the head's accuracy at more than twice the head's prompt.
- **Cost axis:** arm B used about 2,700 OpenJev GPU-seconds per 1,000 questions with five folds at once (13.1 s per
  call, about 13,800 input tokens per call), about $0.44 per 1,000 questions at $0.59/h, against about 134 CPU
  seconds for the head. Arm A (smoke): about 94 s per question single-stream, about $15 per 1,000.
- **Pairing check:** this session's FIFO + floor and `fixed8` agree with §23's Qwen 7B verdicts on 98.3% and 96.6%
  of questions (0.445 against 0.436; 0.526 against 0.538), and `fixed8` minus FIFO is +0.081 (+0.040, +0.121)
  against §23's +0.102.

**Limitations.** OpenJev approximates Jev; nothing here is a result about TypeSafe's Jev. The A40 runs it with a
bf16 KV cache (declared deviation) and weight-only FP4. One benchmark, one reader, one judge. Arm A's accuracy was
not measured.

## 27. LRE, a published query-blind scorer, as a baseline controller (N11; 2026-10-09, pre-registered, run 2026-10-09)

**Status:** pre-registered before any held-out evaluation. The free parts (code check, controller, tests, per-fold
training, training-side label check, configs, stub check) are done (code at c2258e6). The reader pass needs spend and the user's
approval; it is meant to share one pod with §28 (query-rewrite arms), both pairing with this session's `fixed8`.

**Why.** LRE ("Learning What Not to Forget", arXiv 2606.20954) is a published, CPU-only, learned scorer for what to
keep in a long context. It is the nearest outside baseline to the §19 head: small, cheap and trained. It gives the
head a peer from the literature, not only FIFO.

**The key design difference, stated plainly.** LRE is **query-blind** at selection: its score of a turn never sees
the question. The §19 head **sees the question** (its BM25 candidates and features are built from it). So any gap
between them is partly "knowing the question", by design. That is the comparison the user asked for.

**Code check (2026-10-09; cloned to `external/LRE`, gitignored, read only; none of it was run).**
- **Pinned commit:** `1aa51d672d772b27a272cab1393cd9968c13d158` (2026-06-23, "Initial LRE code"; the only commit).
- **Licence: MISMATCH.** There is **no LICENSE file**. `pyproject.toml` says **MIT**; the paper is said to be CC BY
  4.0. We copy no file: `memctl/lre.py` re-implements the few formulas, with the source cited. Either licence allows
  that.
- **What was claimed, against the code:**

  | claim | in the code | verdict |
  |---|---|---|
  | L2 logistic regression | `LogisticRegression(max_iter=1000, class_weight="balanced")`, sklearn default L2, C = 1 | **checks out** |
  | CPU-only | sklearn only; no GPU path | **checks out** |
  | query-blind | the conversational scorer's inputs are the unit's text and position only | **checks out** |
  | kilobyte-scale | the agent scorer (10 features) is; the conversational one has up to 10,006 weights plus a 10,000-word vocabulary. Ours: **about 480 KB as JSON, 10,007 parameters** | **partly**: hundreds of KB |
  | "~10 causal features (trajectory + TF-IDF)" | the conversational scorer is **TF-IDF (up to 10,000 terms) + 6 trajectory features** (position, recency, log length, digits, has "?", capitalised words). The 10-feature vector (`lre/features.py`) belongs to the **agent** scorer, which has no TF-IDF | **mismatch** |
  | "causal" | position and recency use n, the number of units; offline, n is the final length. At question time this is causal | **checks out at question time** |
  | label: overlap with the gold answer, recall ≥ 0.4 | `answer_overlap_labels(threshold=0.4, metric="recall")`, the `lre_self_sup` condition. Tokens: lower case, a/an/the removed, punctuation deleted; **other stop words are kept** | **checks out**; the stop-word detail is new |
  | top-k by score under a budget fraction 0.20, verbatim | `run_downstream.topk`: k = ceil(0.20 × number of units), kept text verbatim. The fraction is of **units, not tokens** | **checks out**, with that detail |
  | unit = a turn | on LongMemEval their unit is a **whole session**; on LoCoMo a turn | **mismatch on LongMemEval** |
- **Other differences that matter here:**
  - Their default downstream run also has `lre_supervised`, trained on gold evidence sessions. We use the answer-
    overlap rule only, as the user decided.
  - Their LongMemEval split is a 5-fold grouped CV with a **dedup guard**: training units whose text also appears in
    the test fold are dropped. That reads held-out texts, so **we do not use it** (the §19 head was trained without
    it too). Shared filler turns can therefore be in both sides; their labels come from training answers only.
  - Their reader (OpenRouter, GPT-4o-mini) and judge are not used. Ours are §23's.

**Our implementation (`memctl/lre.py`, `memctl/lre_train.py`, `memctl/controllers/lre.py`).**
- **Label rule as implemented** (their rule, applied to a training question's own gold answer): a turn is positive
  when the share of the answer's tokens found in the turn is at least 0.4. Tokens as above. A numeric answer is
  compared as text. Abstention questions are included, as their loader includes them.
- **Rows:** every turn of every training question's history, in the order the controller sees them (sessions by
  date). About 197,300 rows per fold. **Only the four training folds are loaded** (`fold_indices(..., part="train")`).
- **Features:** TF-IDF (sklearn, English stop words, at most 10,000 terms, fitted on the training rows) and their 6
  trajectory features, standardised on the training rows. At a decision, idx is the turn's arrival order and n the
  number of turns seen so far. At the question this equals the training layout.
- **Model:** their settings unchanged (C = 1, balanced classes, max_iter 1000, random_state 0). Nothing was tuned.
- **Stored as JSON** (vocabulary, idf, weights, scaler), scored without sklearn. A test checks it gives sklearn's own
  logits to 1e-9.
- **Training (laptop CPU, one process):**

  | fold | rows | train s | load s | model size | iterations | sha256 (first 12) |
  |---|---|---|---|---|---|---|
  | 0 | 197,285 | 55.2 | 16.3 | 479,792 B | 270 | 81ba2adbb690 |
  | 1 | 197,321 | 46.5 | 20.0 | 480,055 B | 220 | f7a72c9feeaf |
  | 2 | 197,307 | 59.9 | 19.9 | 479,723 B | 305 | 074e5e33b8b7 |
  | 3 | 197,562 | 61.2 | 17.7 | 479,839 B | 307 | d43bfdffb615 |
  | 4 | 197,525 | 60.6 | 19.2 | 480,034 B | 286 | 86c3c97074b8 |

  - About 1 min to train and 1.2 min wall per fold; peak RSS 0.83 GB. 10,007 parameters.
  - The models are committed in `configs/sweeps/exp27/lre/lre_f{0..4}.json`, with `train_f{k}.json` beside them.
- **Tests** (`tests/test_lre.py`, 9 pass): the label rule (recall, not F1; articles and punctuation; empty and numeric
  answers); the six features; JSON model equals sklearn; unknown mode and missing model refused; slot mode archives
  on arrival, scores only the shortlist, never the question, and retrieves the k best by logit; a short shortlist is
  shown whole; native mode archives the lowest score, not the oldest; native retrieves exactly what FIFO + floor
  retrieves and never removes the question. Full suite: 233 pass, 2 skipped; the 2 failures in
  `test_import_boundaries.py` are on main already and untouched.

**Training side only: LRE's labels against our evidence labels** (descriptive; per fold, on that fold's training
questions; no held-out question). Our evidence turn: a turn marked `has_answer`, or any turn of an answer session
with no marked turn. Answerable questions:

| fold | LRE positive rate | evidence rate | agreement | precision (LRE label is evidence) | recall (evidence gets an LRE label) |
|---|---|---|---|---|---|
| 0 | 0.059 | 0.0057 | 0.939 | 0.030 | 0.312 |
| 1 | 0.065 | 0.0061 | 0.933 | 0.031 | 0.331 |
| 2 | 0.068 | 0.0061 | 0.930 | 0.029 | 0.327 |
| 3 | 0.061 | 0.0065 | 0.936 | 0.033 | 0.315 |
| 4 | 0.065 | 0.0062 | 0.933 | 0.030 | 0.315 |

- LRE's rule marks about 10 times as many turns as our evidence labels, and **about 97% of its positives are not
  evidence**. It finds about a third of the evidence turns.
- The high agreement (0.93) is only because almost every turn is negative under both.
- Why (fold 0's 376 answerable training questions): the median answer is **2 tokens** (61% have at most 2), so
  recall 0.4 needs **one** shared token, which long turns often have. The median LRE-positive turn is **331 words**,
  against 66 for negatives and **56 for evidence turns**. The rule favours long turns.
- Abstention questions (about 6% of rows): LRE marks 2% of turns; precision 0.19–0.20, recall 0.08–0.12.
- This does not change either arm. It is reported because it says what LRE learns: "turns that share words with
  some answer", not "the evidence".

**Arms** (the same 500 LongMemEval test questions, five folds, Qwen2.5-7B reader, frozen 7b5fc30 prompt, budget 5%,
as §23 and §26; `configs/sweeps/exp27/exp27_qwen7b_f{0..4}.yaml`):
- FIFO + floor at 5% with the 3k target, and `fixed8` (the §19 head, 8 of 32 BM25 candidates): the §23 settings,
  rerun in the same session for pairing. No §23 or §26 row is reused.
- **LRE-native** (`lre`, `mode: native`, label `lre_native`). FIFO + floor with LRE's score in place of recency:
  - Same removal (MOVE_TO_ARCHIVE), same 3k target, same 5% budget. When memory is over the target, the
    lowest-scoring turns are archived until it fits; ties go oldest-first.
  - **Archive and retrieve are FIFO + floor's own, unchanged:** at the question, the top 5 BM25 hits from the
    archive (fit to the room) are retrieved first and protected; then removal runs. The question is never removed.
  - So LRE-native and FIFO + floor differ only in the keep order. It is the code path of the `fifo` controller.
  - Their native budget is "20% of units"; ours is the §23 token budget. That is a declared deviation, so all arms
    share one budget.
- **LRE-slot** (`lre`, `mode: slot`, `select_k: 8`, label `lre_slot`). Like-for-like with `fixed8` and §26 arm B:
  - Everything but the current input is archived on arrival (as the head's `keep_none`).
  - At the question: the same 32 BM25 candidates `fixed8` sees (same plain BM25, same archive). LRE scores them,
    without the question, and the 8 highest are retrieved (ties keep the BM25 order). Up to 8 candidates: all shown.
  - B against `fixed8` differs only in who ranks the same 32.
  - Long picks can exceed the budget; then the harness's forced removals are counted against the arm, as in §26.
- Nothing in the LRE arms was tuned. Every setting is LRE's default or comes from the §23 configs.

**Claims, pre-registered** (470 answerable questions, paired by question, 95% bootstrap by question; margin −0.03 as
in §23; `configs/sweeps/exp27/exp27_report.py`):
1. **C1, the head against LRE as selector.** `fixed8` minus LRE-slot. **Pre-declared direction: the head better**
   (lower bound above 0). LRE-slot **non-inferior** if the upper bound is below +0.03; otherwise not shown.
2. **C2, LRE as selector against the baseline.** LRE-slot minus FIFO + floor: **better** if the lower bound is
   above 0, **worse** if the upper bound is below 0. Expected: better (the user's decision; same pool and k as
   `fixed8`, which beat FIFO by +0.08 to +0.10).
3. **C3, LRE's own use against the baseline.** LRE-native minus FIFO + floor: better / worse as C2; **non-
   inferior** if the lower bound is above −0.03. **Expectation: undetermined.**
- **Reader-free, per arm: P(all in view)**, from the evaluation's own episodes, computed **once**, by the report
  script, after the judge pass. It is not computed before: in the stub check below, the LRE arms' evidence and token
  columns were not printed (only FIFO + floor's and `fixed8`'s, which are known from §23).
- **Reported beside (descriptive):** the §13a decomposition, the three shares (answered-correct, answered-wrong,
  unknown), P(correct | answered), prompt tokens, forced removals per question, false answers on the 30
  unanswerable questions, and the pairing check of FIFO + floor and `fixed8` against §23 and §26.
- **A warning, written before the evaluation, from training-side data only.** A 20-question sanity run on fold 0's
  *training* questions (in-sample for LRE, stub reader): LRE-slot had all evidence in view on 0.10 against
  `fixed8`'s 0.95 and FIFO's 0.80, with 4,497 prompt tokens against 1,008. LRE-native had 0.70 at 2,902 tokens.
  This is consistent with the label check (LRE prefers long turns). **So C2 may well fail, against the expectation
  above.** The claims and expectations are left as decided; this note is the honest prior. It is in-sample and
  n = 20, and its output was deleted.

**Stub check (done, free; the paid layout: five folds at once, three workers each, `guard.sh`; stub reader and stub
judge; output in `runs/_stub27`, cache in `cache/stub27`).**
- Step logs were switched off in the stub (`detail_episodes: 0`; logging is not part of a cell's identity) to keep
  15 GB free: a first attempt with them on used 1.6 GB by 30% done (about 1.7 MB per question), and was deleted.
- **Two interruptions, three endpoints.** Pass A (`http://stub-a.invalid`) was stopped by SIGKILL after 30 s (FIFO
  67–75, `fixed8` 55–60, LRE-native 29–30, LRE-slot not started, of 100 per cell). Pass B (`stub-b`) resumed and was
  stopped by SIGKILL with LRE-slot at 19–28 of 100 (one `fixed8` cell at 99). Pass C (`stub-c`) finished.
  - **All 20 cells resumed (`resumed: true`), none refused, and each ended with 100 distinct questions.**
  - Peak RSS 0.23–0.24 GB per sweep process; pass C took 32–45 s per fold.
- **Known numbers reproduce** (over all 500): P(all in view) FIFO + floor 0.500, `fixed8` 0.700 (§26's stub: 0.500 and
  0.700); prompt tokens at read about 2,963 and 1,050.
- The judge script (`configs/sweeps/exp27/exp27_judge.py`, the §26 judge with these arms and `cache/judge_exp27`)
  ran with the stub judge on a 3-question run with step logs on: 12 answers, 4 arms.
- **All stub run folders and caches were deleted** (`runs/_stub27*`, `cache/stub27*`), so the real run starts from
  empty caches. Free disk after: 19 GB.

**Cost (NEEDS SPEND; from measured §23 numbers).**
- **Measured inputs:** §23's Qwen 7B answered 1,500 LongMemEval questions in 10.3 min (0.41 s each, five folds at
  once); model load 3–6 min; the judge did about 7,000 answers in 3.1 min; A40 $0.59/h.
- **§27's extra arms only** (the shared pod's load, FIFO + floor and `fixed8` are counted once, under §28 or
  whichever pass runs them):
  - LRE-native and LRE-slot: 1,000 answers, about 6.9 min at §23's rate. LRE-slot's prompts may be up to about 4.5k
    tokens (the training-side sanity run), against §23's mix of about 2.2k, so allow up to 1.5×: **7–10 min.**
  - Judge: 940 more answerable answers, under 1 min.
  - **About 8–11 min of pod time: about $0.08–0.11.**
- **If §27 runs on its own pod** (no §28): load 3–6 min, 2,000 answers 14–17 min, judge 1,880 answers about 1 min:
  **about 18–24 min, about $0.18–0.24.**
- **Hard stop:** §27's share at 20 min of pod time (**$0.20**); on its own pod, 40 min from create (**$0.39**).
  Triage, pre-declared: if the stop is reached, LRE-native is dropped before LRE-slot (C1 and C2 are the main tests).
- **Disk:** the real run writes step logs (the judge reads them): about 1.7 MB per question, so about 3.4 GB for
  §27's 2,000. Free disk is 19 GB now, so 15 GB would be left: **before launch confirm at least 19 GB free**, or free
  space first. LRE runs on the laptop CPU; no GPU for the controller.
- **Not approved. The user decides.**

### 27a. Result: LRE as a slot picker fails badly; LRE in place of recency is level with FIFO; the head leads (2026-10-09)

**Pod and cost.** Shared with §28: qpeirhl5gzw7dj (A40, EU-SE-1, $0.59/h, vLLM 0.8.5, Qwen2.5-7B), 07:14:04 to about
07:37:30 UTC, about 0.39 h, **about $0.23 for §27 and §28 together**; terminated, list-pods empty. Run under a one-day
budget approval from the user (since expired); the programme total after this pod was $12.58. §27's answers ran
07:16–07:31 (2,000 answers, 0 failures); the one blind Qwen 7B judge pass judged 1,880. Report:
`configs/sweeps/exp27/exp27_report.py` → `runs/exp27_report.md`; verdicts `runs/exp27_verdicts.json`.

| arm | accuracy (95% CI) | unknown | P(correct \| answered) | P(all in view) | P(correct \| in view) | prompt tokens |
|---|---|---|---|---|---|---|
| FIFO + floor | 0.436 (0.391, 0.481) | 0.330 | 0.651 | 0.498 | 0.701 | 2,964 |
| `fixed8` (the §19 head) | **0.536** (0.491, 0.581) | 0.157 | 0.636 | 0.702 | 0.661 | 1,049 |
| LRE-native | 0.464 (0.419, 0.509) | 0.300 | 0.663 | 0.506 | 0.756 | 2,860 |
| LRE-slot | 0.109 (0.081, 0.138) | 0.766 | 0.464 | 0.028 | 0.769 | 4,305 |

**Pre-registered claims** (margin −0.03):
- **C1, `fixed8` minus LRE-slot: +0.428 (+0.377, +0.477), the head better**, as expected.
- **C2, LRE-slot minus FIFO + floor: −0.328 (−0.377, −0.277), WORSE.** The expectation was "better"; the
  training-side warning in §27 (all evidence in view 0.10 on 20 training questions) was right.
- **C3, LRE-native minus FIFO + floor: +0.028 (−0.002, +0.057), non-inferior** (pre-declared as undetermined).

**What it means.**
- **Query-blind scoring cannot pick the 8 turns a question needs.** With the same 32 candidates, LRE-slot has all
  evidence in view for 2.8% of questions against the head's 70.2%. It prefers long turns (4,305 prompt tokens for
  8 items, against the head's 1,049), as its loose label rule taught it: one shared word of a two-word answer
  makes a turn positive, so long turns are positive most often.
- **As a keep-order in place of recency it does no harm and may help a little** (+0.028, interval just touching 0),
  at the same prompt size as FIFO. The evidence-in-view rate barely moves (0.506 against 0.498).
- **Pairing check:** `fixed8` minus FIFO is +0.100 (+0.060, +0.143), against §23's +0.102.

**Limitations.** LRE is re-implemented from their formulas (no LICENSE file in the repository); our unit is a turn,
theirs a session; their held-out-text de-duplication was not used. One benchmark, reader and judge.

## 28. Query rewriting for the head's candidate pool (2026-10-09, pre-registered, run 2026-10-09)

**Status:** pre-registered before any recall was computed (64735e2). The free part is in §28a: **arm (b), the rule
filter, fails its gate on all five folds and stops.** Arm (a) is built and rehearsed; its gate and the reader
NEED SPEND and are not approved.

**Why.**
- The §19 head picks 8 of the 32 BM25 candidates. It cannot show a turn that is not in the 32. So a better pool
  could lift `fixed8` with no change to the head.
- **The LongMemEval paper's claim, not ours** (Wu et al., arXiv 2410.10813, memory-design section): time-aware
  query expansion gives +7–11% recall on temporal questions; fact-augmented keys give +4% retrieval and +5%
  accuracy.
- **Caveat from the same paper:** Llama 3.1 8B struggled to extract time ranges; GPT-4o did not. Our rewriter
  would be Qwen2.5-7B, so the same risk applies.

**Where the query is formed.** `memctl/controllers/rl.py`, `decide()`: the BM25 query is the question
observation's text, "(asked on DATE) question". The new `query_rewrite` config (`memctl/query_rewrite.py`) changes
only the search. The head and the reader still see the question unchanged.

**Two variants, against `fixed8` (raw question).**
- **(a) LLM rewrite.** Qwen2.5-7B gets the question and the question date, and writes two lines: `QUERY:` extra
  search words, and `RANGE:` a date range or "none". One call per question, cached (`cache/rewrite_exp28`), so the
  reader run replays it (`cache_only`; a miss is an error).
  - BM25 query = the question text plus the `QUERY` words (expansion; nothing of the question is removed).
  - The range is widened by 3 days each side and applied by `demote` (below). A malformed answer gives the raw
    question. Both fixed before any rewrite.
- **(b) Rule-based time filter.** No model. A regex parser (`rule_range`) reads time phrases ("N days/weeks/months
  ago", "last week/month/year/weekend", "the past N weeks", "yesterday", "last Saturday", "in March", "May 3rd",
  "the 5th of April", "this month", YYYY/MM/DD) and turns them into a date range, counted back from the question
  date. Several phrases give their union. The end is never after the question date.
  - **No new dependency.** `dateparser` is not installed; the parser uses only the standard library.
  - **Written blind.** The parser was written from general English phrasing, and checked only on made-up
    sentences, before any LongMemEval question was read for this experiment. No question text is read to change it.
- **Two ways to apply a range** (both rank the whole archive once, so BM25 statistics do not change):
  - `drop`: only candidates whose session date is in the range (the pool may be smaller than 32);
  - `demote`: in-range candidates first, then the rest, each in BM25 order, up to 32.

**No held-out peeking: cross-fitting, as the §19 head.**
- The head has one checkpoint per fold, trained on that fold's training part. Every §28 choice is made the same
  way: for fold f, on fold f's training part only (about 376 answerable questions), then applied to fold f's
  test part.
- With five folds over all 500 questions, every question is in some fold's training part. So "train first, test
  later" cannot mean "rewrite these questions later". What is held out is **the labels**: fold f's test-part
  evidence is read by nothing until the evaluation.
- Enforced in code (`memctl/rewrite_gate.py`): the gate reads labels only through each fold's training part;
  `evaluate`, the only code that reads test-part labels, refuses to run before the arm (a) gate is final, and runs
  once (it refuses if its output exists).
- The rewrites themselves read no label, so all 500 are written in one pass (a question's rewrite does not depend
  on which fold it is tested in).

**The metric for every choice: BM25 recall@32 of evidence turns.** Per question, the share of its evidence
requirements in the 32 (a marked turn; for an answer session with no marked turn, any turn of it), as the harness
counts them. Abstention questions are left out. "All in the 32" is reported beside it.

**The gate (one rule for both variants; fixed now).**
- On fold f's training part, a variant **passes** if mean recall@32 rises by at least **+0.01** over the raw
  question, **or** rises by at least **+0.02 on the temporal-reasoning questions while not falling overall**.
- A variant goes to the reader only if it passes on **all five** folds. If neither passes, §28 stops before the pod,
  as §25 did.

**Arm (b)'s settings, chosen per fold on the training part.** Action `drop` or `demote`, and slack 0.5, 1, 2 or 4
(the window around a point in time: ±2 days, ±4 for weeks, ±15 for months, ±120 for years, times the slack). The
choice is the highest overall training recall@32; ties go to `demote`, then the smaller slack. Then the gate.

**Arm (a)'s prompt: the first version is frozen now** (`configs/sweeps/exp28/prompts/v1.txt`; `{asked}` is the
question date with its weekday, `{question}` the question):

```
You help search a user's past chats with an AI assistant. A keyword search (BM25) will look for the chat messages that answer the user's question below.

The question was asked on {asked}.
Question: {question}

Write exactly two lines and nothing else:
QUERY: the words a message that answers the question would likely contain: the key nouns, names, places, activities and verbs of the question, plus a few close synonyms. Do not repeat question words such as "how many" or "when", and do not write dates here.
RANGE: if the question refers to a time, such as "two weeks ago", "last Saturday", "in March", "yesterday" or "last month", the dates the chat most likely took place, as YYYY/MM/DD - YYYY/MM/DD, worked out from the date the question was asked. If the question does not refer to a time, write: RANGE: none
```

- It is not tuned now: a small CPU model is not a stand-in for Qwen 7B's behaviour. Its training recall@32 is the
  **first step on the pod**, before any reader call.
- **Up to 3 revisions on the pod (the reviewer's refinement, 2026-10-09).** After v1, up to three more versions
  (v2–v4) may be written, each scored once by the gate on the training parts of the folds still open.
  - **A fold freezes the first version that passes on its training part.** Folds already frozen are not
    rescored.
  - If after v4 any fold has no passing version, **arm (a) is dropped before any reader call**, and the reason is
    recorded. Nothing is carried forward as "best of".
  - **Superseded by the ruling below (design (i)):** a revision is written from LoCoMo dev outputs only, never
    from LongMemEval.
  - Every version's text, hash and per-fold training recall is logged (`runs/exp28_gate/state.json`) and copied
    into §28.
  - The script enforces the order: versions in turn, each scored once, a scored prompt file may not change, at
    most four, and the reader stage refuses to start until `final` has been written.

**Prompt design on a disjoint dev set: design (i) (the reviewer's ruling, 2026-10-09; recorded before any pod).**
- **The ruling.** "Label-free" is not the test. Every LongMemEval question is a test question in one fold, so no
  person or agent may read LongMemEval rewrite outputs, or LongMemEval question text, to design v2–v4. My first
  version let `inspect` show format stats of LongMemEval outputs; that is withdrawn.
- **The LoCoMo check (done first, free).** LoCoMo (`data/locomo/locomo10.json`) has what RANGE needs:
  - session dates on all 288 sessions, all parsed ("1:56 pm on 8 May, 2023");
  - no question date, so the conversation's **last session date** is used (LoCoMo's questions come after the whole
    conversation);
  - time phrases: the rule parser fires on 201 of the 1,540 non-adversarial questions (42 of the 321 temporal ones;
    an aggregate count). LoCoMo's temporal questions mostly ask "when ...?" rather than give a relative anchor, so
    RANGE is exercised, but less than on LongMemEval. That is a stated limit of the dev set.
  - So design (i) is used, not (ii) (a per-fold prompt lineage).
- **The dev set:** every fifth non-adversarial LoCoMo question with evidence, in file order: **307 questions** (74
  temporal, 167 single-hop, 48 multi-hop, 18 open-domain). Its dates are put in LongMemEval's format, and each
  question is shaped as the LongMemEval question step ("Question: (asked on DATE) ..."), so the same prompt applies.
  LoCoMo is disjoint from LongMemEval; no §28 claim is about LoCoMo.
- **The flow per version (enforced in `memctl/rewrite_gate.py`):**
  - `rewrite` writes v*N* for the 307 dev questions and the 500 LongMemEval questions, and prints only the counts.
  - `inspect` reads **LoCoMo dev outputs only**: format stats, dev recall@32 (raw against rewrite, from LoCoMo's own
    evidence labels), and a few dev questions with their outputs. Asked for any id that is not a dev id (a
    LongMemEval number or question id), it **raises PermissionError before reading anything** (tested).
  - `gate` reads the LongMemEval rewrites with training-part labels, per fold, and prints **aggregates only**:
    recall per fold, overall and temporal, the gains, pass or fail. No question text, no rewrite output.
  - `final` closes the gate. LongMemEval rewrites are never printed or shown, before or after.
  - A revision v*N*+1 is written from `inspect`'s dev output only, and committed before it is scored.

**Arms on the shared pod (later; Qwen2.5-7B reader, frozen 7b5fc30 prompt, five test folds, §19 head A per fold).**
- `fixed8` (raw question). The §27 branch reruns it for pairing; this branch's configs use the same reader cache,
  `cache/generations_exp27_qwen7b`, so both branches' `fixed8` rows are the same generations.
- `fixed8_rewrite` (arm (a)), only if its gate is GO.
- `fixed8_timefilter` (arm (b)), only if its gate is GO.

**Pre-registered claims** (470 answerable questions, paired by question, 95% bootstrap by question as §23; margin
−0.03).
1. **Each variant minus `fixed8`, on all 470.** Non-inferior if the lower bound is above −0.03; better if above 0.
   Expected ≥ 0.
2. **The same difference on the temporal-reasoning subset, the primary subgroup** (n is given in §28a from the
   `question_type` field). Same thresholds.
- Reported once, at evaluation (`rewrite_gate evaluate`): reader-free P(all in view) and BM25 recall@32 per arm on
  the test folds.
- Abstention questions: false answers on the 30 unanswerable questions, reported as usual.

### 28a. Free part: the rule filter fails its gate on every fold; arm (a) is built and rehearsed (2026-10-09)

**Arm (b), the rule time filter: NO GO.** `python -m memctl.rewrite_gate rule-tune` (training parts only; 23 s on
the laptop, peak RSS 0.05 GB; `runs/exp28_gate/rule.json`).
- Per fold, the best setting by training recall was **`demote` with slack 4** (the widest window) in all five
  folds. Every `drop` setting was worse still (best `drop` 0.829–0.837 against raw 0.847–0.861).
- Recall@32 on each fold's training part (376 answerable questions; temporal-reasoning n = 101–102):

  | fold | raw | rule filter | gain | temporal raw | temporal rule | temporal gain | gate |
  |---|---|---|---|---|---|---|---|
  | 0 | 0.851 | 0.844 | −0.007 | 0.855 | 0.847 | −0.008 | fails |
  | 1 | 0.853 | 0.849 | −0.004 | 0.849 | 0.840 | −0.009 | fails |
  | 2 | 0.861 | 0.853 | −0.008 | 0.879 | 0.867 | −0.012 | fails |
  | 3 | 0.855 | 0.847 | −0.008 | 0.858 | 0.846 | −0.012 | fails |
  | 4 | 0.847 | 0.843 | −0.005 | 0.850 | 0.842 | −0.007 | fails |

- **Plainly: the rule filter lowers recall, overall and on temporal questions, in every fold.** The best setting
  is the one that changes the pool least. Arm (b) does not go to the reader.
- The parser fires on about 15% of questions (56–60 of 376; 22–24 of the 101–102 temporal ones).
- **Why (descriptive; fold 0's training part; no choice depends on it).** On the 56 questions where it fires,
  only 56% of the evidence lies in the range at slack 1, and 84% at slack 4. All of a question's evidence lies in
  the range for 25 of 56 questions at slack 1, and 43 of 56 at slack 4. Recall on those 56: raw 0.712, `demote`
  slack 4 0.667, `drop` slack 1 0.386.
  - A time phrase in the question often does not point at the session the evidence is in. Example of the kind:
    the user mentions an event in a later chat, or the evidence spans several sessions.
  - This also bears on arm (a): an LLM's range, even a correct reading of the phrase, faces the same mismatch.
- The total number of answerable temporal-reasoning questions (the primary subgroup at evaluation) is **n = 127**
  (from the `question_type` field).
- **One fix before the numbers above, disclosed:** the first pass gave the search the question without the
  "Question: " prefix the environment puts on it (`memctl/envs/qa.py`). That was corrected so the gate's query is
  the controller's. The verdict did not change (every fold failed both times). The gate's 32 candidates were
  checked equal to the controller's shortlist on 4 training questions of fold 0 (4 of 4).

**Arm (a), the LLM rewrite: built, not measured (it needs Qwen 7B).**
- Prompt v1 is frozen (§28 above; committed at 64735e2, before any recall was computed).
- The pod step, in order (`configs/sweeps/exp28/stage.sh`):
  1. `stage.sh gate 1 URL`: rewrite the 307 LoCoMo dev and the 500 LongMemEval questions; print the dev output
     (`inspect`); score v1 on each fold's training part (aggregate output only).
  2. If a fold is still open: write v2 from the LoCoMo dev output only, commit it, `stage.sh gate 2 URL`; up to v4.
  3. The gate closes when every fold has a version, or after v4 (`runs/exp28_gate/selected.json`).
  4. `stage.sh reader URL`: only if GO. `fixed8` and `fixed8_rewrite`, five folds, then the blind judge
     (`configs/sweeps/exp28/judge_report.py`). The report gives the two pre-registered differences.
  5. `python -m memctl.rewrite_gate evaluate`: test-fold recall@32 and all-in-32 per arm, once.
- **The scripts enforce the order** (checked in the stub run): a version scored before the one before it, or
  twice, is refused; a scored prompt file that changes is refused; `final` refuses while a fold is open and fewer
  than four versions are scored; after `final` no more rewriting; the reader stage refuses unless the gate is GO;
  `evaluate` refuses before the gate is final, and refuses to run twice.

**Stub check (done, free; stub reader, stub rewriter, stub judge, separate stub caches).**
- **Gate loop, full size:** v1–v4 each rewrote 500 questions and scored the five training parts, 30–56 s per
  version, peak RSS 0.08 GB. The stub's answers are all malformed (it answers "unknown"), so every gain is 0, no
  fold passed, and `final` wrote **DROP**, as the rule says. The reader stage then refused to start.
- **Reader stage, full size (gate file forced to GO for the rehearsal only):** both arms × five folds, three workers
  each, as on the pod. Stopped by SIGKILL after 40 s (60–100 of 100 per cell), then resumed against a different
  `base_url`. All 10 cells resumed (`resumed: true`) and ended with 100 distinct questions; the resume took 24 s.
  The stub judge graded 940 answerable answers (60 abstention answers kept).
- Peak RSS **0.60 GB** per sweep process. Disk: about 315 MB per fold (1.6 GB in all), so the run fits (20 GB free;
  the rule is at least 15 GB).
- `fixed8_rewrite` replayed every rewrite from the cache with no miss. With malformed stub rewrites it equals the
  raw question, so both arms gave P(all in view) 0.700 (§26's stub: 0.700).
- All stub output and stub caches were deleted afterwards.
- **Redone for design (i) (gate loop only; the reader stage is unchanged):** v1–v4 each rewrote the 307 dev and 500
  LongMemEval questions, printed the dev output and scored the training parts with aggregate output: 42–45 s per
  version, peak RSS 0.53 GB (loading LoCoMo). `final` wrote DROP (stub answers are malformed); the reader stage
  refused; `inspect --ids 0 17` was refused with PermissionError. The log held no LongMemEval question text. Stub
  output deleted.

**Price (NEEDS SPEND; only §28's extra work on the shared §27 pod, Qwen2.5-7B already loaded; A40 $0.59/h).**
- From §23's measured rate (1,500 LongMemEval answers in 10.3 min; judge about 3 min per 7,000 answers):
  - one rewrite version: 807 short calls (307 LoCoMo dev + 500 LongMemEval; about 230 prompt tokens, at most 96
    out), estimated at most 2.4 min, plus about 0.7 min of laptop scoring: **about 3 min per version** (an
    estimate; short calls are not measured); at most 4 versions, 3,228 calls;
  - writing a revision: the pod waits; **capped at 10 min per revision**, else the arm is dropped;
  - `fixed8_rewrite` answers: 500, about 3.4 min; judge: 470 more answers, under 1 min.
- **Best case (v1 passes): about 8 min, about $0.08.** **Worst case (three revisions, GO at v4): about 47 min, about
  $0.46.** Dropped after v4: about 42 min, about $0.41. (Design (i) adds about 1 min per version for the dev pass.)
- **Hard stop: raised from 45 to 50 min of §28-only pod time (about $0.49),** because the dev pass makes the worst
  case about 47 min. If it is reached, arm (a) stops where it is and is reported as not run.
- If §27's pod is not running, a pod of its own adds about 5 min to ready and 3.4 min for `fixed8` (about $0.09).
- Nothing is approved. No pod was started for §28.

**Recommendation.**
- Arm (b): stopped by its pre-declared gate.
- Arm (a): the reviewer's rule stops §28 only if neither variant improves training recall, and (a) cannot be
  measured without the GPU. Run its gate **only as a ride-along on the §27 pod** (about $0.03 for v1's gate
  alone; at most $0.49 with revisions and the reader). Expect little: the pool already holds 0.85 of the evidence,
  and the rule filter shows that time ranges miss evidence sessions often. If the user prefers, stop here at $0.

### 28b. Result: the LLM rewrite passes its training gate at v3; on the test folds it is level with the raw question (2026-10-09)

**Pod and cost.** Shared with §27 (pod qpeirhl5gzw7dj, 07:14:04 to about 07:37:30 UTC, about $0.23 for both; the
approval is in §27a). §28's gate ran 07:16–07:34 while §27's reader ran; its reader and judge ran
07:34–07:37.

**The gate, aggregate output only.** Every revision was written from the LoCoMo dev output alone and committed
before it was scored; no LongMemEval question text or rewrite was shown.

| version | LoCoMo dev: malformed / with a range (of 307) | training recall@32 gain, overall (folds 0–4) | temporal gain | gate |
|---|---|---|---|---|
| v1 (frozen before the pod) | 299 / 7 | about +0.001 in every fold | 0.000 | fail in all folds |
| v2 (28c4929) | 1 / 306, range = the asked date | −0.168 to −0.193 | −0.006 to −0.017 | fail in all folds |
| v3 (171faef) | 0 / 3 (median width 7 days) | +0.024, +0.020, +0.011, +0.016, +0.020 | −0.001 to −0.009 | **pass in all folds** |

- v1 failed on format: the model wrote `KEYWORDS:` or bare words, never `QUERY:`, so the parser fell back to the
  raw question. v2 fixed the format but set RANGE to the date the question was asked for almost every question,
  which pushed the latest sessions up. v3 makes RANGE "none" by default and never the asked date.
- **v3's two in-prompt examples:** the first ("charity race run marathon fundraiser Melanie") is the model's own
  QUERY line for LoCoMo dev question conv-26_q5, from v2's dev output; the second ("dinner restaurant friends ate",
  asked 2023/05/20, "two weeks ago") was invented. Neither comes from a LongMemEval question; no LongMemEval
  question text was read in writing any revision.
- `final`: GO, v3 for every fold. The gate passed on the overall criterion; training recall on temporal questions
  fell slightly in every fold, reported as pre-declared.

**Test folds** (report: `configs/sweeps/exp28/judge_report.py report` → `runs/exp28_report.md`; 1,000 answers, 0
failures; one blind Qwen 7B judge pass):

| arm | accuracy (95% CI) | temporal-reasoning accuracy (n = 127) | unknown | P(all in view) | prompt tokens | false answers on the 30 unanswerable |
|---|---|---|---|---|---|---|
| `fixed8` (raw question) | 0.534 (0.489, 0.579) | 0.402 | 0.157 | 0.702 | 1,049 | 6 |
| `fixed8` + LLM rewrite (v3) | 0.540 (0.496, 0.585) | 0.425 | 0.134 | 0.717 | 1,064 | 7 |

**Pre-registered claims** (margin −0.03):
- **All 470 answerable: +0.006 (−0.017, +0.030), non-inferior.**
- **Temporal-reasoning subgroup (primary), n = 127: +0.024 (−0.016, +0.063), non-inferior;** not shown better.
- The rule filter (arm b) was stopped by its free gate (§28a) and not run.

**What it means.** A Qwen 7B query rewrite is safe but adds little: the 32-candidate pool already holds about 0.85
of the evidence, and the rewrite lifts all-in-view only from 0.702 to 0.717. Its first two prompts show the
LongMemEval paper's caveat in practice: a 7B model invents date ranges unless told plainly not to.

**Pairing.** `fixed8` here used the same cached generations as §27's (470 of 470 answers identical); the two
separate judge passes agree on 469 of 470 verdicts (0.534 here, 0.536 in §27).

## 29. A reader-utility training signal for the §19 head (2026-10-09, pre-registered; free part done, pod NEEDS SPEND)

**Status:** pre-registered before any utility row exists. The free part (rows, code, tests, full-size stub check) is
done; see "Stub check" below. The pod steps need spend and the user's per-run approval. Nothing has been launched.

**Question.** The §19 head learned from evidence labels: "this turn holds the answer". Does it do better if it
learns from what the reader actually gets out of each turn?

**What stays the same as §19.** The head architecture (`listsum`-trained deepsets head, hidden 64, **27,526
parameters**), the pool (32 plain-BM25 candidates, `keep_none`, top 8 shown), the five folds, the §23 reader
settings. Only the training signal changes.

**The signal.**
- Per answerable training question q and per candidate turn c:
  **utility(q, c) = log P(gold | question + turn c) − log P(gold | question alone)**, from Qwen2.5-7B.
- **Prompt:** the frozen 7b5fc30 reader prompt (compact labels, the QA goal, "## Memory", "## Current input"), with
  `reasoning: false`. So the note instruction is left out and the prompt ends with "Answer:". The memory holds the
  one turn, or "(empty)" for the baseline. A capture-time check asserts that this equals the reader's own
  `LLMAgent.build_prompt`, turn by turn.
- **Chat form:** Qwen2.5's chat template (its default system line, the user message), then the gold answer as the
  assistant reply, with no end token.
- **Token span:** the gold's tokens are the last n prompt tokens. n is the server tokenizer's count for the gold
  alone. This holds because the gold follows "assistant\n", and Qwen2's pre-tokenizer never merges a newline into
  the next word. Each record also checks that those n decoded tokens spell the gold; failures are counted.
- **Endpoint (vLLM 0.8.5, checked in its source):** `POST /v1/completions` with `echo: true, logprobs: 1,
  max_tokens: 1, temperature: 0`. In 0.8.5, echo with logprobs turns on prompt logprobs and returns the prompt's
  tokens and log-probabilities (the first is null) before the one generated token. `usage.prompt_tokens` marks the
  end of the prompt. n comes from `POST /tokenize` (`add_special_tokens: false`).
- **Abstention questions are left out** (their "gold" explains why there is no answer). Every §29 head trains on
  the answerable training questions only.
- **One pod pass, cached** per (model, prompt text) in `cache/utility_exp29`, so it is replayable and resumes against
  any `base_url`. The rows: `runs/exp29/utility.jsonl.gz`.

**Exact call count.** 470 answerable questions. Their pools hold **15,039 candidates** (one question's pool has 31,
the rest 32), plus **470 baselines: 15,509 logprob calls**, plus 470 tokenize calls. (The ≈ 4 × 376 × 32 of the
brief counts each question once per training fold. The rows do not depend on the fold, so each question is scored
once.) Prompt size, from the rows: mean 434 tokens (memctl count), median 362, max 5,465; **6.60 M tokens in all**.

**The fold guard.**
- The utility rows are fold-invariant: one reader, and each row comes from its own question's gold answer. So one
  pass serves all five folds.
- But a head for fold k must never train on rows of fold k's test questions.
- In code (`memctl/reader_utility.py`): `training_rows(fold, part)` reads only the questions of fold k's training
  part (or its inner split). Other lines of the rows and utility files are skipped **unparsed**. It also raises if any
  test question of fold k reaches it.
- Tested: in a rows file whose lines for every other question are the text POISON, fold 2's training rows load. Fold
  3's do not, because fold 3's training part holds fold 2's test questions.
- The test folds are touched once, by the reader pass at the pre-declared evaluation.

**The decision rows (free, done).** `python -m memctl.reader_utility rows` replays every question to its question
step with the §19 controller settings (untrained policy) and records the head's inputs: 32 candidates, their 24
features, the global features, the evidence labels and the reader lines.
- With `keep_none` and one question per episode, the inputs do not depend on the policy's weights.
- The budget is 5%, as at evaluation. (§19 trained at 1.25%; that changes only two global features.)
- 500 questions in 20 s on the laptop, peak RSS 0.27 GB. `runs/exp29/rows.jsonl.gz`, 9.0 MB.

**Training (CPU, from the cached rows; fixed now, nothing tuned but w).**
- Per fold, on the 376 answerable training questions: Adam, lr 0.003, batch 64 questions, 40 epochs, gradient clip
  1.0, seed 0, the last epoch kept. Same network and initialisation as §19.
- **Utility head:** cross-entropy against softmax(utility / τ) over the 32, τ = 1 nat.
- **Evidence-rows head:** §19's listsum loss on the evidence labels (questions with no evidence in the 32 skipped).
- **Blend head:** (1 − w) × listsum + w × the utility loss.
- This trainer is not §19's (no DAgger, no 4-question composition, the whole training part instead of half). So the
  evidence-rows head is run beside the others as a **control arm, with no claim**: utility minus evidence-rows is the
  signal alone, same trainer; evidence-rows minus `fixed8` is the trainer alone.

**The blend weight: pre-declared grid, inner 6/2 split of training conversations.**
- On LongMemEval a conversation is one question's own history. Fold k's training questions are dealt stratum by
  stratum (type, abstention apart); places 0–5 of every 8 go to inner-train, 6–7 to inner-val. That gives 281–282
  and 94–95 answerable questions.
- **Grid: w ∈ {0.1, 0.25, 0.5, 0.75, 0.9}** (strictly between the two pure heads, which are their own arms).
- **Criterion:** a blend head per w trained on inner-train; on inner-val, the reader's mean log P(gold | question + the
  head's top 8 in arrival order), same prompt shape as the signal. Highest mean wins; ties go to the w closest to
  0.5, then the smaller.
- **Note (review): the criterion leans toward utility.** Mean log P(gold) under the head's top 8 is close to the
  quantity the utility term trains toward, so it may favour the larger w. It stays as the criterion: it is a
  reader-side proxy computed on training questions only, with no generation and no judge. The reader pass on the test
  folds is what decides C2.
- At most 5 × 472 = **2,360 calls**, fewer when two weights pick the same 8.
- Reported (training side, descriptive): each w's inner-val score and its pool-relative all-found@8.
- Then the final blend head is trained with fold k's chosen w on all 376.

**Arms** (the same 500 LongMemEval test questions, five folds, Qwen2.5-7B reader, frozen 7b5fc30 prompt, budget 5%,
as §23 and §27; `configs/sweeps/exp29/exp29_qwen7b_f{0..4}.yaml`; new cache `cache/generations_exp29_qwen7b`):
- FIFO + floor at 5% with the 3k target, and `fixed8` (the §19 head A): rerun in the same session for pairing.
- `utility_head`, `blend_head`: 8 of the same 32, ranked by the new heads.
- `evidence_rows`: the control above. **It carries no claim, now or after the run**: it is reported only as the two
  descriptive differences below, and no verdict is drawn from it.

**Claims, pre-registered** (470 answerable questions, paired by question, 95% bootstrap by question, margin −0.03,
as §23; `configs/sweeps/exp29/judge_report.py report`):
1. **C1, utility head − `fixed8`.** Better if the lower bound is above 0; worse if the upper bound is below 0;
   non-inferior if the lower bound is above −0.03. **Direction undetermined.**
2. **C2, blend head − `fixed8`.** Same thresholds. **Expected: at least non-inferior.**
3. **C3, utility head − FIFO + floor; C4, blend head − FIFO + floor.** Better / worse as above. **Expected: better.**
- **Reader-free columns, per arm, as usual:** P(all in view) and prompt tokens. They are read once, by the report,
  after the judge pass.
- **Beside them (descriptive):** the §13a decomposition, unknown rate, P(correct | answered), false answers on the
  30 unanswerable questions, the two control differences above, and the `fixed8` − FIFO pairing check (§23 +0.102,
  §27 +0.100).

**Optional second signal: declared, costed, not run in this pass.**
- Definition: leave-one-out utility within the evidence head's top 8, log P(gold | question + the 8) − log P(gold |
  question + the 8 without turn c).
- **Note (review): the main signal is pointwise.** It scores one turn at a time against the question alone. Turns
  that help only together (a multi-session count, or a date in one turn and the event in another) each get little
  credit, so the utility head may under-rank them. Leave-one-out credit within a set is the declared way to address
  that; it is not part of this run.
- Why not now: `fixed8` shows all 8, in arrival order, so credit inside the 8 cannot change what is shown. It would
  matter only as a demotion rule (a top-8 turn with negative credit swapped for the 9th), which is a new design, not
  declared here. Also, the evidence head's top 8 differ by fold, so the rows are not fold-invariant: 4 folds × 470 ×
  (1 + 8) = 16,920 calls of about 1.1k tokens each.

**Stub check (free, done; stub scorer, stub reader and stub judge through a fake vLLM 0.8.5 server,
`configs/sweeps/exp29/fake_vllm.py`; separate stub folders and caches).**
- **Signal pass, full size, two endpoints.** Pass A (`127.0.0.1:18101`) was stopped by SIGKILL after 25 s, with
  4,954 cache entries written. Pass B (`:18102`) resumed and finished in 31 s: 15,509 calls planned, 12,502 new,
  the rest from the cache; **0 span-check failures**; peak RSS 0.35 GB. The cache then held 15,867 entries: **15,483
  distinct prompts** (26 candidate turns repeat within their question) and 384 distinct gold answers.
- **Tune, full size** (third endpoint): all five folds in parallel, 65 s wall, peak RSS 0.38 GB; every fold wrote
  its weight. (A first attempt ran torch with many threads and was much slower; training now uses one thread per
  process, and its per-fold file makes a stopped tune resumable.)
- **Final training:** 15 heads in 119 s with the controller check, peak RSS 0.38 GB; each head 27,526 parameters,
  about 23–31 s. **Controller check passed:** in all five folds the trained heads, loaded by the real `rl` controller,
  show exactly the 8 that the rows predict (3 training questions × 3 heads per fold).
- **Reader pass, full size** (five folds × five arms × 100, three workers per fold, through the fake server's chat
  route): stopped by SIGKILL after 40 s (endpoint A), resumed and stopped again after 25 s (endpoint B), then
  finished on endpoint C in 54 s. **Every cell ended with 100 distinct questions; every cell that had started
  resumed (`resumed: true`), none refused**; peak RSS 0.27 GB. Step logs were off (`detail_episodes: 0`) to save
  disk.
- **Judge and report:** a 3-question-per-fold run with step logs on (75 answers): the judge graded 75, and the report
  ran and printed the four claim lines. Its numbers were not read. Step logs: about 26 MB per fold, so 1.7 MB per
  answer, as in §27.
- `fixed8` and FIFO + floor reproduce their known P(all in view), 0.700 and 0.500 (§27's stub: 0.700 and 0.500).
- **Disclosed: one test-fold number was seen.** The check script printed P(all in view) for every stub arm on the
  test folds. For the utility and blend arms these are stub heads (fake utility) and mean nothing. But the
  evidence-rows head does not depend on the utility, so its stub value, **0.698**, is the real head's. Nothing was
  chosen or changed after it: the arms, claims, grid, loss and training settings above were all written before the
  stub run, and the evidence-rows arm carries no claim. It is reported here so the reader knows.
  - **Ruling (review):** it stays disclosed as a leak; it does not taint C1–C4, because no claim involves the
    evidence-rows arm and nothing was decided after it.
  - **Fixed so it cannot happen again.** The leak came from an ad hoc script that read the stub reader cells, which
    played the test folds. Now (a) the rehearsal's reader cells play each fold's **training** part (`stage.sh` rewrites
    `part: test` to `part: train` under `STUB=1`), and (b) every stub metric goes through
    `memctl.reader_utility.cell_questions`, which raises PermissionError for a cell not on a training part, or for any
    episode whose question is in its fold's test part, before any metric is read. The stub judge and report use the
    same check. The same pattern as the rows guard: refuse, never filter. Tested
    (`test_stub_metrics_refuse_test_fold_cells`). No metric was recomputed for this fix.
- **Disk.** Free space was 13.6–14.4 GB throughout (other runs hold the rest), never under the 12 GB floor. The stub
  output's peak footprint was 244 MB. **All stub output, stub caches, stub configs and stub logs were deleted.** The
  free rows file (`runs/exp29/rows.jsonl.gz`, sha256 2a82c8ac87e3…) is kept for the pod day.
- Tests: `tests/test_reader_utility.py`, 7 pass with the stub-metric guard test added later (prompt equality, losses, the fold guard and the POISON file,
  deterministic training and a checkpoint the controller loads, the scorer against the fake server with resume on a
  second URL). Full suite: 254 pass; the 2 failures in `test_import_boundaries.py` are on main already.

**Price (NEEDS SPEND; per-run approval by the user; nothing approved).**
- **Measured inputs:** §23's Qwen 7B answered 1,500 LongMemEval questions in 10.3 min on one A40 at $0.59/h; model
  load 3–6 min; the judge did about 7,000 answers in 3.1 min.
- **How the scoring calls are estimated.** They are prompt-only (one output token). §23 pushed about 3.3 M prompt
  tokens (mean 2,194 over its three arms) through in 10.3 min, besides all its generated text. Charging all 10.3 min
  to the prompts gives a floor of about **5,300 prompt tokens/s**. It is also about the A40's compute limit for a 7B
  model (≈ 14 GFLOP per token at about half of its bf16 peak), so it is not optimistic.
  - Signal: 6.60 M tokens → **at most about 21 min**.
  - Tune: at most 2,360 calls × at most 2,383 tokens (BM25's top 8; the head's picks are shorter) → **at most about
    18 min**, plus about 1 min of laptop training (measured: 65 s for all five folds in parallel).
  - Final training and the controller check: **about 2 min** (measured), the pod waiting.
- **Reader:** 2,500 answers at §23's 0.41 s each → **about 17 min**; up to 26 min if the new heads pick long turns
  (×1.5). **Judge:** 2,350 answers, about 1 min.
- **Total, one pod, one model load:** about 45–75 min (the upper end adds every upper bound) → **about
  $0.45–0.74**.
- **Hard stop: 90 min from create (about $0.89).** Triage, pre-declared:
  - if the signal pass is not done by 40 min from create, stop: nothing downstream can run;
  - if the reader has not started by 60 min, drop the `evidence_rows` control arm first;
  - the reader then runs the four arms the claims need; anything not finished by the stop is reported as not run.
- **Disk:** the reader writes step logs (about 1.7 MB per answer, §27): about 4.3 GB for 2,500. **Before launch,
  confirm at least 17 GB free** (4.3 GB plus the 12 GB floor), or free space first. The signal cache is about 65 MB.
- The pod is verified terminated with list-pods afterwards.
- **Not approved. The user decides.**

**Pod order** (`configs/sweeps/exp29/stage.sh`): `signal URL` → `tune URL` → `train` → `reader URL` → `report`.
`rows` ran before the pod. The heads are written to `configs/sweeps/exp29/heads/f{k}/` and committed after the run.

### 29a. Pod run, partial: signal, tune and heads done; the reader did not run (2026-10-09)

**Approval and spend.** The user approved the quote ($0.45–0.74, stop $0.89) on the executor's channel. Pod
`puynnuax9qx54u` (A40, CA-MTL-1, $0.59/h, vLLM 0.8.5, Qwen 7B) was created 21:48:01 UTC and deleted 23:16:25 UTC
(list-pods empty): 1.47 h ≈ **$0.87**, inside the stop. **No reader answer was graded and no claim was tested.**

**What finished.** Signal pass 22:02:49–22:20:29 (15,509 logprob calls + 470 tokenize; about 14.5 calls/s). Tune
22:20:29–22:25:18 (chosen blend weight per fold: 0.5, 0.25, 0.25, 0.25, 0.9, from the inner 282/94 split of each
fold's training questions). Final heads and the controller check by 22:27:44 (3 training questions × 3 heads per fold:
controller picks equal row picks). The heads are in `configs/sweeps/exp29/heads/f{k}/`.

**Deviation 1: vLLM out of memory, server flags changed.** At 21:56 the first signal calls crashed vLLM: the
prompt-logprob gather (`gather_logprobs`) asked for 2.32 GiB with 1.45 GiB free at `--gpu-memory-utilization 0.9`.
The stub's fake server could not show this. At 22:00 the pod was relaunched with `--gpu-memory-utilization 0.70
--max-num-batched-tokens 2048` (same image and model); 32 parallel ~5k-token logprob calls then passed. **Mixed
configs:** 474 cache entries (459 logprob, 15 tokenize) were written 21:56:16–21:56:47 under 0.9; every entry from
22:02:50 is under 0.70/2048; none in between. They are not recomputed: the utility is a difference of two logprobs
from one server, and the batching flag changes only chunked-prefill grouping (float-noise drift). Failed calls raise
before the cache write, so no error was cached. About 12 min of pod time was lost.

**Deviation 2: the laptop lost internet, the pod idled.** From about 22:27 to 23:15 UTC the laptop had no internet
(journal: NetworkManager CONNECTED_SITE, DNS failures). `stage.sh reader` sat in `wait_model`, which had no deadline,
for about 48 min while the pod billed. The pre-registered 60-min triage fired on time (22:48:03: `evidence_rows`
dropped from the reader configs; restored afterwards in git). The reader started at 23:15:46 and the pod was deleted
40 s later, before the stop. About 100 answers were cached; the cells were marked failed at deletion. No metric was
read from them.

**Fixes (free, before any rerun).** `wait_model` now gives up after `WAIT_MAX` seconds (default 900). Logprob runs
launch at 0.70/2048. Checklist: one real-shape logprob call at full concurrency in the pod's first minute. Proposed: a
self-stop on the pod itself, so a laptop outage cannot leave it billing.

**Shared-pod driver (built, dry-run passed, not run).** `configs/sweeps/exp29/exp29_32_pod.sh POD CREATE_EPOCH` runs
the §29 reader rerun and then the §32 reader on one pod, from any cwd (it changes into each branch's worktree). It
has a readiness gate at create + 15 min, a 32-call long-prompt pre-check, §29 triage at + 12 min, and group-kill
deadlines at + 45 min (§29) and + 85 min (§32). Stage markers are now written only on success: `memctl.sweep` exits 0
when cells fail, so the stages read their logs too. A STUB=1 dry run against the fake server with the real stage
scripts finished both stages (all stub output deleted).

**What is left (NEEDS SPEND).** Only the reader and the judge, on the saved heads: about 5 min load + 17–26 min reader
(four claim arms; `evidence_rows` adds about a quarter) + 1 min judge ≈ $0.25–0.33, hard stop 40 min ($0.39). Not
approved.

### 29b. Result: the reader-utility signal does not beat the §19 head (2026-10-10)

**Run.** The user approved the shared-pod quote ($0.57–0.74, stop $0.89; relayed 03:39 UTC). Pod `009mqsuixjvjyn`
(A40, CA-MTL-1, $0.59/h, vLLM 0.8.5 at 0.70/2048), created 03:39:31 and deleted about 04:09:31 UTC (list-pods
empty): 0.50 h ≈ **$0.30**, shared with §32. Driver `configs/sweeps/exp29/exp29_32_pod.sh` (fa99fcd). Ready 03:45:11
(pre-check 32 of 32); §29 done 03:56:13: 25 of 25 cells, the judge graded 2,350 answers (150 abstention answers kept
unjudged). No triage, no deadline. The report (`runs/exp29_report.md`) was run once.

| arm | accuracy (95% CI) | unknown | P(all in view) | prompt tokens |
|---|---|---|---|---|
| FIFO + floor | 0.438 (0.394, 0.483) | 0.328 | 0.498 | 2,964 |
| `fixed8` | 0.543 (0.498, 0.587) | 0.153 | 0.702 | 1,049 |
| utility head | 0.530 (0.485, 0.574) | 0.211 | 0.666 | 1,881 |
| blend head | 0.540 (0.496, 0.585) | 0.183 | 0.691 | 1,383 |
| evidence-rows head (control) | 0.540 (0.496, 0.587) | 0.157 | 0.700 | 1,047 |

- C1, utility head − `fixed8`: −0.013 (−0.051, +0.026) → **UNDETERMINED**.
- C2, blend head − `fixed8` (expected ≥): −0.002 (−0.036, +0.032) → **UNDETERMINED**.
- C3, utility head − FIFO + floor: +0.091 (+0.049, +0.134) → **BETTER**. C4, blend − FIFO + floor: +0.102 (+0.060,
  +0.145) → **BETTER**.
- Descriptive: evidence-rows − `fixed8` −0.002; utility − evidence-rows −0.011 (−0.047, +0.026); pairing check
  `fixed8` − FIFO + floor +0.104 (§23 +0.102, §27 +0.100).
- **Reader noise floor.** `fixed8` scores 0.543 here and 0.536 in §32c on the same picks (prompt tokens equal on 470 of 470). vLLM at temperature 0 is not batch-deterministic: 35 of 470 answer texts differ between the two passes. The 7 verdict flips split into 6 from different answers and 1 from the judge on an identical answer. So two passes over the same picks differ by about ±0.007 in accuracy.

**Said plainly.** Training the head on what the reader finds useful (the drop in the gold answer's log-probability)
instead of on which turns hold the evidence did not help: the utility head is level with the head at best and picks
longer turns (1,881 vs 1,049 tokens, fewer "all in view"). Mixing the two signals gives back the evidence head. Every
learned head still beats FIFO + floor by about 0.10. The reader-side signal is closed as a negative; the evidence
labels were already a sufficient signal at this n.

## 32. The §19 head against standard rerankers of the same pool (N12; 2026-10-09, pre-registered, free stage)

**Status:** the free part (code, rows, training-side selection, reader-free pass, split audit, tests, stub rehearsal)
is done on branch `reranker-baselines` (§32a, §32b). The reader pass NEEDS SPEND and is not approved.

**Why.** As deployed (`fixed8`), the §19 head is a top-k reranker: memory keeps nothing but the current turn
(`keep_none`), and at a question the head picks 8 of the 32 BM25 candidates by its logit
(`memctl/controllers/rl.py`, the shortlist search and the floor-head top-k). The fair outside comparison is
therefore not a memory controller (§26, §27) but a standard supervised reranker of the same 32. If one of those
matches the head, the head is "a reranker", and the thesis should say so.

**The question set is LongMemEval** (500 questions: 470 answerable + 30 abstention, the five stratified folds of
`memctl/splits.py`), as in §19, §23 and §27. (The brief said LoCoMo; the 470 + 30 set and the folds are
LongMemEval's.)

**Held fixed across every arm.** The 32 BM25 candidates of the head (plain `LexicalRetriever`, the question text as
the query, the archive under `keep_none`), k = 8, budget 5% of the history, the same 500 questions and five folds,
the same harness (picks that do not fit are returned by `enforce_budget` and counted as forced removals). No arm sees
gold evidence at inference. One new controller does it: `rerank` (`memctl/controllers/rerank.py`) is the `rl`
controller with only the top-k scorer swapped (`memctl/rerank.py`); a test checks it sees the head's pool, in the
head's order, with the head's own feature rows.

**Arms.**
1. `bm25_top8`: the BM25 order, top 8. No learning.
2. `rrf_top8`: reciprocal rank fusion (constant 60) of the BM25 rank and the bge-small-en-v1.5 dense rank (cosine of
   the labelled turn text and the question, as `DenseRetriever`), **both ranks counted within the 32-candidate
   pool**. `FusionRetriever` as written fuses whole-archive ranks of a *labelled* BM25 and would change the pool,
   so it is not used; within the pool the dense order is the same as the whole-archive one, only the RRF rank
   numbers differ. Ties keep the BM25 order (all arms).
3. `lr_pointwise`: logistic regression on exactly the head's inputs: 24 item features + 6 global features
   (`memctl/features.py`), one row per candidate, standardised. Grid: C ∈ {0.001, 0.01, 0.1, 1, 10, 100} ×
   class_weight ∈ {none, balanced}.
4. `gbdt_pointwise`: LightGBM is **not installed**, so scikit-learn's `HistGradientBoostingClassifier` on the same
   rows. Grid: max_depth ∈ {3, 6, none} × max_leaf_nodes ∈ {7, 15, 31}, learning rate 0.05, rounds ∈ {25, 50,
   100, 200, 400} (read off one 400-round fit), no internal early stopping. **No lambdarank row** (it needs LightGBM).
5. `cross_encoder_zero`: cross-encoder/ms-marco-MiniLM-L-6-v2 (22.7M parameters) on (question text, turn text) for
   the 32 candidates, max length 512, zero-shot; top 8. Scores are cached on disk, so the reader pass replays them.
6. ~~`cross_encoder_tuned`~~: **dropped before any test-fold run** (see §32a). The brief allowed it only if CPU
   fine-tuning takes minutes; measured, it does not.
7. `fixed8`: the §19 head A as deployed, per fold (`runs/lme_n4_head_f{k}_a/checkpoints/policy_best.pt`), not
   retrained, run by the unchanged `rl` controller.

**Training rows (same rows as the head, DAgger-free; training part only).** `memctl/rerank_data.py`.
- **Fitting rows, half A** of each fold's training part (the head A's own half): the head's training setup
  (`configs/rl/lme_n4/head_f{k}_a.yaml`): composed 4-question episodes, budget fraction 0.0125, the regret expert
  (retrieval_risk 0.36), the head's own seeds 100000–100249 (10 iterations × 25 episodes). DAgger-free: the expert
  drives every episode (the head's iteration 0), so the rows depend on no learner. About 1,000 question decisions and
  32,000 rows per fold.
- **Label:** the head's: a candidate is positive if it is the **designated carrier** of a requirement needed at
  this question (`Hindsight.designated`: of a requirement's carriers, the shortest, the earliest among equals). For a
  marked (`has_answer`) turn that is the turn itself. For an answer session with no marked turn, it is one turn of
  that session only (a test pins this; see "Findings" below). Pointwise models use every row, including lists with
  no positive (the head's list loss skips those).
- **Inner validation, half B** of each fold's training part (the head A never trained on it): every half-B question
  once, played as the test folds are (one question per episode, 5% budget). About 200 questions per fold.

**Selection (no held-out peeking; fixed in code before any fit: the pointwise grids at dcdce6c
(`memctl/rerank_train.py`), the cross-encoder tuning at 25ba106 (`memctl/rerank_ce.py`)).**
- The selection measure is **inner all-found@8** on half B (every requirement has one of its carriers among the 8;
  computed from the candidates' requirement sets, no budget), ties broken by requirement recall@8, then the simpler
  setting.
- Each grid setting is fitted on half A and scored on half B; the best setting's half-A model is the fold's model
  (not refitted).
- Cross-encoder fine-tuning, fixed before any run: every positive and 7 negatives per question drawn uniformly from
  its other candidates (seed 0); binary cross-entropy; AdamW, lr 2e-5, batch 16, 10% linear warm-up then linear
  decay, max length 512, 2 epochs; the epoch with the higher half-B all-found@8 is kept.
- **The best classical reranker** (the primary claim's comparator) is `lr_pointwise` or `gbdt_pointwise`, whichever
  has the higher half-B all-found@8 pooled over the five folds (ties: recall, then `lr`). It is recorded below before
  any test-fold number is computed.
- The test folds are read once, by the reader-free pass, after every choice above is committed.

**Reader-free measures (all arms; per fold and pooled; each arm minus `fixed8`, paired 95% bootstrap by question,
10,000 resamples).** P(all requirements in view) (`evidence_complete_rate`), requirement recall
(`needed_hit_rate`), precision (`retrieval_precision`: the share of the shown turns that carry some requirement),
prompt tokens (stub reader, the real 7b5fc30 prompt), forced removals, selection latency per question (one process,
one thread, fresh score caches, BM25 search excluded), and parameter count. Sensitivity row: the same on the test
questions that share no answer session with any training question.

**Pre-registered reader claims** (Qwen2.5-7B reader with the frozen 7b5fc30 prompt, one blind Qwen 7B judge pass;
470 answerable questions, paired by question, 95% bootstrap; margin −0.03; `configs/sweeps/exp32/report.py`).
1. **Primary: D1 = `fixed8` minus the best classical reranker, accuracy.** Pre-declared direction: **the head
   better** (D1 > 0). Head **better** if the lower bound is above 0; head **non-inferior** if the lower bound is
   above −0.03; head **worse** if the upper bound is below 0; otherwise not shown. Read the other way
   (descriptive): the baseline is "within the margin of the head" if the upper bound is below +0.03.
2. **Secondary: D2 = `fixed8` minus `cross_encoder_zero`**, the same thresholds and wording.
- Descriptive: every other arm minus `fixed8`; the three disjoint answer shares (an "unknown" answer judged correct
  is flagged, `memctl/analysis/shares.py`); P(correct | answered); prompt tokens; false answers on the 30
  unanswerable questions; the pairing check of `fixed8` against §27 (0.536) and §28 (0.534).

### 32a. Training-side choices, recorded before any test-fold number (2026-10-09)

**Rows** (`memctl/rerank_data.py`, code at dcdce6c; laptop CPU, peak RSS 0.35 GB per process): half A, 1,000
question decisions and about 32,000 rows per fold (1,332–1,395 positives); half B, 200 questions per fold.

**Pointwise selection on half B** (`python -m memctl.rerank_train`; `runs/exp32_rerankers/inner_selection.json`, a
copy in `configs/sweeps/exp32/models/`; models `lr_f{k}.json`, `gbdt_f{k}.pkl` committed beside it):

| fold | `lr` setting | all-found@8 | recall | `gbdt` setting (depth / leaves / rounds) | all-found@8 | recall |
|---|---|---|---|---|---|---|
| 0 | C 0.001 | 0.700 | 0.800 | none / 15 / 100 | 0.685 | 0.788 |
| 1 | C 0.001 | 0.695 | 0.798 | none / 15 / 25 | 0.680 | 0.783 |
| 2 | C 0.001 | 0.695 | 0.810 | 3 / 15 / 50 | 0.680 | 0.805 |
| 3 | C 0.01 | 0.705 | 0.812 | 3 / 15 / 25 | 0.695 | 0.816 |
| 4 | C 0.01, balanced | 0.680 | 0.787 | 3 / 7 / 25 | 0.685 | 0.796 |
| pooled | | **0.695** | 0.801 | | 0.685 | 0.798 |

- **The best classical reranker is `lr_pointwise`** (pooled half-B all-found@8 0.695 against 0.685), by the rule
  fixed in §32. It is the primary claim's comparator. The margin is small (2 questions in 1,000); the rule decides.
- `lr` chose the strongest regularisation (C = 0.001, the grid's edge) in 3 of 5 folds: the features carry little
  that a looser fit uses. Not widened: the grid was fixed in advance.
- Sizes: `lr` 31 parameters; `gbdt` 325–2,900 tree nodes.
- One code fix during this step, before any result: `truncate_gbdt` set the read-only `n_iter_` and crashed on
  fold 0 before writing anything (fixed at 7e4fca7; nothing else changed).

**Cross-encoder fine-tuning dropped (decided before any test-fold run).** The run with the pre-fixed settings
(`memctl/rerank_ce.py tune`, 25ba106) on five folds at once, 3 threads each: two folds were killed by the kernel
for memory (about 6 GB resident each at max length 512), and the other three had not finished their **first**
epoch after 73 minutes. The brief admits the arm only if fine-tuning takes minutes, so `cross_encoder_tuned` is
not run, and the reader stage has six arms. No tuned model or score was produced or read.

**Found while building (not fixed; reported):**
- **The §19 head A checkpoints of folds 1 and 2 are byte-identical, and so are those of folds 3 and 4**
  (`runs/lme_n4_head_f{k}_a/checkpoints/policy_best.pt`, sha256 2825193… and 849e524…). Cause: `memctl/splits.py`
  deals half A and half B alternately from each fold's training list, and with k = 5 removing place 1 or place 2
  (or 3 or 4) from the dealing order leaves the same alternate half. So half A is the same 200 questions for folds
  1 and 2 (and for 3 and 4), and the same seeds give the same head. **This is not a leak**: half A never meets its
  fold's test part (checked: 0 of 200 in every fold). But `fixed8` is three distinct heads, not five, and the §32
  pointwise models of folds 1 and 2 are identical too (`lr_f1.json` = `lr_f2.json`). The half-B sets differ, so the
  inner checks are not duplicated. Consequences: `fixed8`'s per-fold spread (0.650–0.750) comes from three heads, not
  five, so it understates the variance of head training; the paired-by-question bootstrap is unaffected. The same
  artefact applies to head B and to every `fixed8` number in §19–§28, which all use these checkpoints.
- Under `keep_none` the turn just before the question is archived at the question step, after the search, so it is
  never a candidate (all arms alike). On LongMemEval it carries a requirement for 1 question of 500 and is never the
  only carrier: harmless.
- A turn longer than the whole budget is deleted on arrival by `enforce_budget` (EVICT), not archived, so it can
  never be retrieved (all arms alike): 6 of 500 `fixed8` episodes in §27 (one 7,737-token turn). Pinned by a test.

### 32b. Reader-free result, split audit and the reader stage, built and rehearsed (2026-10-09/10)

**Runs** (stub reader; `runs/exp32_free_f{k}`, every cell clean: at 565d1d8, the `rrf_top8` cells at 63bd7ae;
report `configs/sweeps/exp32/free_report.py` → `runs/exp32_rerankers/free_report.md`). `fixed8` reproduces §19a
(P(all in view) 0.700, 1,050 prompt tokens). The `rrf_top8` cells were rerun once to write their dense-similarity
cache for the reader stage; on the 277 questions the first, interrupted run had finished, the picks were identical.

**Pooled, 500 test questions** (Δ = arm minus `fixed8`, paired 95% bootstrap by question):

| arm | P(all in view) | Δ | requirement recall | Δ | precision of the 8 | prompt tokens | forced removals / q | parameters |
|---|---|---|---|---|---|---|---|---|
| `fixed8` (§19 head) | **0.700** | — | 0.801 | — | 0.197 | 1,050 | 0.012 | 27,526 |
| `bm25_top8` | 0.584 | −0.116 (−0.150, −0.084) | 0.713 | −0.089 (−0.113, −0.065) | 0.175 | 2,467 | 0.044 | 0 |
| `rrf_top8` | 0.666 | −0.034 (−0.060, −0.010) | 0.780 | −0.022 (−0.041, −0.002) | 0.198 | 1,974 | 0.020 | 33.4M (bge-small) |
| `lr_pointwise` | 0.684 | **−0.016 (−0.036, +0.002)** | 0.790 | −0.011 (−0.027, +0.004) | 0.195 | **948** | 0.012 | 31 |
| `gbdt_pointwise` | 0.666 | −0.034 (−0.058, −0.012) | 0.780 | −0.021 (−0.040, −0.003) | 0.191 | 985 | 0.012 | 325–2,900 nodes |
| `cross_encoder_zero` | 0.666 | −0.034 (−0.064, −0.004) | 0.786 | −0.015 (−0.036, +0.006) | 0.203 | 1,993 | 0.018 | 22.7M |

- Precision is within ±0.006 of `fixed8` for every arm but BM25 (−0.022). Prompt tokens against `fixed8`: BM25
  +1,417, RRF +924, cross-encoder +943, `lr` −102, `gbdt` −65 (all intervals exclude 0).
- **Said plainly: the logistic regression on the head's own 30 inputs comes within 0.016 of the head reader-free,
  and its interval includes 0,** at 31 parameters and 10% fewer prompt tokens. The head is not shown better than
  it on evidence. Every other baseline is below the head (intervals exclude 0), and BM25, RRF and the zero-shot
  cross-encoder also need about twice the head's prompt (they prefer long turns).
- **Pre-registered reading of D1, reader-free:** the head is not shown better than a 31-parameter logistic regression
  on its own 30 inputs. This is evidence only; the reader claim (fixed8 minus `lr_pointwise` on accuracy, margin
  −0.03) has not been run.
- So, reader-free, most of the head's gain over BM25 (+0.116) is available to a linear model on the same features
  (+0.100); the network adds about +0.016 on top, not significant at this n.

**Per fold, P(all in view):**

| arm | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 |
|---|---|---|---|---|---|
| `fixed8` | 0.650 | 0.680 | 0.700 | 0.720 | 0.750 |
| `bm25_top8` | 0.510 | 0.580 | 0.580 | 0.620 | 0.630 |
| `rrf_top8` | 0.610 | 0.640 | 0.670 | 0.690 | 0.720 |
| `lr_pointwise` | 0.660 | 0.670 | 0.670 | 0.690 | 0.730 |
| `gbdt_pointwise` | 0.630 | 0.630 | 0.650 | 0.700 | 0.720 |
| `cross_encoder_zero` | 0.640 | 0.650 | 0.710 | 0.650 | 0.680 |

**Inner split, half B (training side; descriptive except the lr/gbdt choice in §32a; `memctl/rerank_ce.py inner`):**
all-found@8 pooled over the five folds: `fixed8` 0.704, `lr` 0.695, `gbdt` 0.685, `rrf` 0.683, cross-encoder
0.674, BM25 0.607, the same order as on the test folds. (Half B is the same set for folds 0 and 1 and for folds 2
and 3: the dealing artefact in §32a.)

**Selection latency** (fold 0's test part, one process, one thread, fresh caches, first question dropped; the BM25
search, about 25 ms, is common to all arms and excluded; `runs/exp32_latency`):

| arm | median ms / question | p90 ms |
|---|---|---|
| `bm25_top8` | 0.2 | 0.2 |
| `lr_pointwise` (features + 31 weights) | 4.8 | 5.9 |
| `fixed8` (features + head) | 5.3 | 6.5 |
| `gbdt_pointwise` | 5.8 | 6.9 |
| `cross_encoder_zero` | 5,334 | 5,529 |
| `rrf_top8` (bge-small embeddings of 32 turns) | 10,926 | 11,240 |

- Building the features is most of the head's 5.3 ms; the pointwise models cost the same order. `fixed8` was timed
  through the `rerank` controller with the head's own logit; its picks equal the deployed `rl` controller's on all
  100 questions.

**Split audit (free; `configs/sweeps/exp32/split_audit.py` → `runs/exp32_rerankers/split_audit.json`).**
- (a) Share of a test question's filler sessions (by session id) that appear in some training question's haystack:
  **0.317 pooled** (folds 0.328, 0.314, 0.317, 0.321, 0.304).
- (b) Test questions sharing an answer session with a training question's answer sessions: **8 of 500** (per fold
  1, 3, 1, 2, 1). The same count per fold has its answer session anywhere in a training haystack.
- **Sensitivity row (the 492 questions with no shared answer session), P(all in view):** `fixed8` 0.699; `lr` −0.018
  (−0.037, +0.000); `gbdt` −0.035 (−0.059, −0.012); RRF −0.030 (−0.055, −0.006); cross-encoder −0.037 (−0.067,
  −0.008); BM25 −0.112 (−0.144, −0.081). Order and gaps do not change: the shared sessions do not drive the result.

**Tests** (`tests/test_rerank.py`, 9 pass; full suite 257 pass, the 2 known `test_import_boundaries` failures
unchanged):
- question dependence: the same turn's `retrieval_score` and `observation_similarity` change with the question,
  its question-free features do not;
- a reranker sees the head's candidates, in the head's order, with the head's own feature rows, and retrieves its
  k best; ties keep the BM25 order; the RRF arithmetic; the JSON logistic model equals sklearn;
- top-k picks that do not fit are returned to the archive by `enforce_budget` (never deleted, never the question)
  and counted as forced removals; at a budget smaller than one turn the harness deletes that turn on arrival;
- the designated-carrier label of a fallback session (below);
- the three answer shares are disjoint and an "unknown" judged correct is flagged (checked on §23–§28's verdict
  files too: 0 of 19,800 rows);
- score caches: a `cache_only` scorer replays and refuses a miss.

**Finding from the label test (not fixed: it is the head's label, kept for like-for-like):** for an answer session
with no marked turn, the evaluation accepts any of its turns, but the training label marks only the designated
carrier (the shortest turn). The session's other turns are negatives, and when the shortest turn is not in the 32,
that requirement has no positive at all although a carrier is in the pool. It affects the head and the pointwise
baselines alike (instances with such sessions: 41 partly marked + 21 unmarked, of 500).

**The reader stage (built, rehearsed, NOT run).** `configs/sweeps/exp32/stage.sh URL`: the §27 Qwen2.5-7B reader
config with the frozen 7b5fc30 prompt, six arms × five folds (`configs/sweeps/exp32/exp32_qwen7b_f{k}.yaml`, three
workers per fold), then one blind Qwen 7B judge pass (`configs/sweeps/exp32/judge.py`, cache `cache/judge_exp32`),
then `configs/sweeps/exp32/report.py` (the pre-registered claims).
- The cross-encoder and RRF scores are replayed from the free pass's caches (`cache/exp32_ce/{zero,dense}_f{k}.jsonl`,
  `cache_only`; a miss is an error, and the stage refuses to start without them), so no model runs on the laptop
  during pod time.
- Resume: run the same command again; a new pod URL is fine. Marks `runs/_logs/exp32_{READY,ANSWERS_DONE,DONE}`;
  it waits on its own children, no `pgrep -f`.
- Hard stop: `DEADLINE=<unix s>` kills every step's whole process group (sweep parents and their pool workers). A
  first version used `timeout`; the rehearsal showed that killing the script then left the sweeps running in
  `timeout`'s own process groups, so it was replaced before any paid use.
- **Stub rehearsal (full size, free; stub reader and judge, separate stub caches):** pass A (`http://stub-a.invalid`)
  SIGKILLed after about 50 s (cells at 1–100 of 100); pass B (`stub-b`) stopped by a 20 s `DEADLINE` (no process left
  behind); pass C (`stub-c`) resumed to the end and the stub judge graded 2,820 answerable answers (180 abstention
  answers kept aside). **All 30 cells resumed, each ended with 100 distinct questions, and P(all in view) and prompt
  tokens equal the free pass on every question.** Peak RSS **0.60 GB** per sweep process; disk about 0.94 GB per
  fold (4.7 GB in all; 114 GB free). The report script ran on the stub verdicts. All stub output and stub caches
  were deleted.

**Price (NEEDS SPEND; from §23's measured rates on an A40 at $0.59/h).**
- §23: 1,500 Qwen 7B answers in 10.3 min with five folds at once (0.41 s per answer); model load 3–6 min; judge
  about 7,000 answers in 3.1 min.
- Here: 3,000 answers (6 arms × 500) ≈ 20.6 min. The mean prompt over the six arms is about 1,570 tokens (stub count),
  below §23's mix (about 2,200), so no slow-down is expected; allow 1.3× → about 27 min. Judge 2,820 answers ≈ 1.3
  min. Load 3–6 min.
- **About 30–35 min of pod time: about $0.30–0.35.** (The brief guessed $0.50–0.90; the dropped tuned arm and the
  replayed scores make it smaller.)
- **Hard stop: 60 min from pod create (about $0.59)**; run the stage with `DEADLINE` = create + 55 min. If it is
  reached, the cells resume on a later approved pod; nothing is reported from a partial pass.
- **Not approved. The user decides.**

**Disclosure.** No test-fold number was computed before 565d1d8 (every choice committed before it). The split
audit reads only session ids of the data file. The reader-free pass is the only read of test-fold evidence, as
pre-registered.

### 32c. Reader result: the head is non-inferior to every baseline, and better than none of the learned ones (2026-10-10)

**Run.** Shared pod with §29 (`009mqsuixjvjyn`, 03:39:31–about 04:09:31 UTC, 0.50 h ≈ $0.30 in all; approved quote
$0.57–0.74, stop $0.89). §32 ran 03:56:13–04:09:21: 30 of 30 cells, the judge graded 2,820 answers (180 abstention
answers kept aside). No deadline fired. The report (`configs/sweeps/exp32/report.py` → `runs/exp32_report.md`) was
run once.

| arm | accuracy (95% CI) | unknown | P(correct \| answered) | prompt tokens |
|---|---|---|---|---|
| `fixed8` (the head) | 0.536 (0.491, 0.581) | 0.153 | 0.633 | 1,049 |
| `lr_pointwise` (31 parameters) | 0.523 (0.479, 0.568) | 0.155 | 0.620 | 952 |
| `cross_encoder_zero` | 0.523 (0.479, 0.568) | 0.217 | 0.668 | 2,001 |
| `gbdt_pointwise` | 0.515 (0.470, 0.560) | 0.162 | 0.614 | 984 |
| `rrf_top8` | 0.509 (0.464, 0.555) | 0.223 | 0.655 | 1,978 |
| `bm25_top8` | 0.468 (0.423, 0.513) | 0.296 | 0.665 | 2,468 |

- **Primary, `fixed8` − `lr_pointwise`:** +0.013 (−0.017, +0.043), n = 470 → **head NON-INFERIOR**, not better.
- **Secondary, `fixed8` − `cross_encoder_zero`:** +0.013 (−0.030, +0.053) → **head NON-INFERIOR**, not better.
- Descriptive: − BM25 +0.068 (+0.026, +0.113, better); − RRF +0.028 (−0.011, +0.068); − GBDT +0.021 (−0.015, +0.055).
- **Reader noise floor.** `fixed8` scores 0.536 here and 0.543 in §29b on the same picks (prompt tokens equal on 470 of 470). vLLM at temperature 0 is not batch-deterministic: 35 of 470 answer texts differ between the two passes. The 7 verdict flips split into 6 from different answers and 1 from the judge on an identical answer. So two passes over the same picks differ by about ±0.007 in accuracy.

**Said plainly.** On reader accuracy the §19 head is at least as good as each standard reranker (non-inferior at the
−0.03 margin), but it is not shown better than a 31-parameter logistic regression on its own 30 inputs, nor than an
off-the-shelf cross-encoder. Only plain BM25 order is clearly worse. The learned part that matters is the feature set
and the evidence labels, not the network; the cross-encoder gets the same accuracy at about twice the prompt. The
three-heads caveat (§32a) applies to `fixed8`.

**Spot check (LLM second grader, human adjudication of disagreements).** 50 judged answers stratified by arm (8–9 each) and spread over question
types, seed 32 (`configs/sweeps/exp32/spot_sample.py`): `runs/exp32_spotcheck/sample.csv` (arm and verdict hidden)
and `key.csv`.
- **Spot check done (2026-10-10).** The second grader was an LLM (Claude), not a person: it graded all 50 blind,
  before reading the key (`runs/exp32_spotcheck/grader.csv`, sha256 7f4d99024041e030…, with a one-line reason each). It agreed
  with the Qwen judge on 45 of 50. All 5 disagreements are the judge marking wrong an answer the grader accepts
  (samples 10, 12, 26, 36, 39: arms RRF, lr ×2, GBDT, `fixed8`). The user ruled on those 5 only and called all 5
  correct (`runs/exp32_spotcheck/adjudication.csv`, sha256 6c20be40ebd53f21…). Judge false positives were checked by the LLM
  grader alone; it found none in the 28 judged correct.
- **Where the strictness falls: question type, not arm.** 3 of the 5 are single-session-preference questions (30 of
  470, rubric-style gold answers that one-line reader answers only partly match; 12 and 36 are borderline), 1 is
  multi-session arithmetic with an accepted alternative (10), 1 an exact-match paraphrase (39). Every arm answers the
  same 470 questions, so the paired comparisons are unaffected; the per-type preference accuracies (0.067–0.233) are
  the least trustworthy numbers here.
- **Magnitude.** 28 → 33 of 50 is +0.10 absolute (+18% relative). For the thesis: with a strict 7B judge, absolute
  accuracies are understated by roughly 0.1; the paired comparisons are not affected, because the strictness falls on
  question types, which every arm shares.

## 30. UtilMem: does the §19 head help when the evidence is spread over many sessions among look-alike distractors? (N12; 2026-10-09, pre-registered; closed, not run)

**Status: CLOSED without a pod run, at the user's decision (2026-10-10).** Nothing was spent. The free part below
(adapter, checks, stub run, price) is kept as the record; the original status line follows.

**Status:** the free part is done (adapter, checks, stub run, price). The paid run **NEEDS SPEND and is not
approved**. Nothing below was computed on the evaluation side.

**The benchmark, checked against the source** (arXiv 2608.30508, Qing, Shi and Vosoughi; README says EMNLP 2026
main).
- **Code: none yet.** github.com/peijunallin/UtilMem at commit b7ebd4a1ba31379a67540f2b1c95ec48351f6d79 holds one
  README. "Evaluation and benchmark-construction code — coming soon". No LICENSE file. Cloned to
  `external/UtilMem` (not in git).
- **Data:** Hugging Face `KrisQ/utilmem` at revision 15a774f0d630498e17ad4ba3eedaf06d08c5796e, one file,
  `multi_domain_eval_strong.json`, 54 MB (sha256 2816fb72…). Ungated.
- **Licence: none stated** for the code or the data (no dataset card). The paper is CC BY 4.0, and its ethics
  statement says the benchmark is "intended for research evaluation". The mental-wellness domain is rewritten Reddit
  posts. **So the file was not copied to `data/`.** It sits at `external/UtilMem-hf/` (gitignored, not
  redistributed). Copying it to `data/utilmem/` is the user's call.
- **1,717 instances: confirmed.** 100 bundles; each bundle has 3–4 questions per domain: studychat 352, finance 350,
  mental_health 319, fitness 350, edgar_10k 346.
- **Five domains: confirmed** (learning support, finance, mental wellness, fitness, 10-K filings).
- **History length.** A bundle is one user's 65–85 sessions (mean 74), 228–358 turns, in date order. By our token
  count (words and punctuation) a history is 77k–149k tokens, median about 94k. The paper says a mean of 120K with
  its tokenizer.
- **"Separates evidence recovered from evidence used": only partly checkable.** The release gives, per question
  domain, the evidence session ids (2,372 in all), and nothing else: **no gold answer, no reference answer, no
  turn-level label.** The paper separates the two with an analysis (its §5.2: judge score against session recall
  for NaiveRAG). That code is not released.
- **The scorer is an LLM rubric judge, not a rule.**
  - The same QA model first answers from the evidence sessions alone. That answer is the reference, pinned at
    score 10. Then it answers from the memory system's retrieval.
  - A judge sees the evidence sessions, the question, the reference and the answer. It rates five sub-dimensions,
    then gives one integer score, 1–10 (prompt: their Appendix F).
  - Two judges, GPT-5.2 and GPT-5.4 (paid API), averaged. QA model: Qwen3-235B-A22B, temperature 0.
  - Metrics: RS (mean score); NR, "RS rescaled to 0–100", their headline; DR, the share scored 6 or lower.
  - Their results: NaiveRAG NR 58.9–59.3, DR 49.2–49.6; A-MEM 56.0 / 54.4; Mem0 17.3 / 98.3.
  - Even with most evidence retrieved (recall at least 0.8), the mean score stays at 6.4–7.9.
- **Two leaks in the release, closed in the adapter.**
  - Distractor session ids start with `noise_`. The adapter renames every session `s000`, `s001`, … in date order,
    so no controller or reader can see the label.
  - Evidence turns are longer than distractor turns. Example (development side, median tokens): finance assistant
    turns 1,150 against 494; studychat assistant turns 529 against 139; 10-K user turns at the 90th percentile 8,161
    against 513 (pasted filing text). A length-aware rule could find evidence by size. It cannot be closed without
    changing the data, so it is stated as a confound: it may favour BM25 and the head over FIFO.

**Where each question sits relative to its evidence** (development side, 349 questions; `configs/sweeps/exp30/exp30_structure.py`):
- **The question is not right after its evidence.** It is asked after the whole history (the paper gives each
  question the full history). The newest evidence session ends a median 12.7k tokens before the question (0 to
  73k). The oldest starts a median 81k tokens back.
- In no question (0 of 349) are all the evidence sessions inside the last 6k tokens. In 17.5% does any evidence
  session end inside the last 3k.
- 3–7 evidence sessions per question (median 5). Evidence text per question: median 8.3k tokens; 10-K questions
  median 25k, up to 53k.
- So this is **not** the §25 situation: recency should not win here by construction. The free check below
  measures it.

**The environment** (`memctl/envs/utilmem.py`, registered as `utilmem`; tests in `tests/test_utilmem.py`).
- One episode is one question: the bundle's turns in date order (one turn is one memory item, as in
  LongMemEval), then the question.
- **Labels** (evaluation only): one requirement per evidence session of the question's domain, met when any of its
  turns is in view. So "all in view" means every evidence session is touched: session recall 1, the paper's §5.2
  recall. It does not mean every evidence turn is in view.
- **Turns are cut at 1,000 tokens** (`max_turn_tokens`; chosen on the development side: 4.0% of turns are longer;
  the longest is 37,897 tokens). Without the cut, one pasted filing could exceed the whole 5% budget, and 10-K
  reference prompts reach 53k tokens, more than the reader's context. The cut applies to every arm, the reference
  and the judge's context alike. With it, the median history is about 75k tokens, the 5% budget about 3.8k, and
  the longest reference prompt about 17.6k tokens.
- No gold answer, so the environment records the answer and logs every question's evidence status; the score
  comes afterwards from the judge pass.

**Split (no held-out peeking).**
- **Development side:** 20 of the 100 bundles, drawn with seed 30 (`split_bundles`): bundles 0, 3, 6, 9, 10, 17,
  20, 26, 31, 32, 37, 48, 50, 59, 66, 69, 78, 79, 82, 83 (349 questions). Every check and every choice in §30 uses
  only these. The split was committed (bd02deb) before any evidence position was computed.
- **Evaluation side:** the other 80 bundles (1,368 questions). Nothing is computed on them before the
  pre-declared evaluation.
- If a reference answer is itself a refusal, the question is kept (the paper pins every reference at 10), and the
  count is reported.
- **Disclosure:** while reading the file format, the first five session headers of bundle 0 were printed (dates,
  role and domain). Bundle 0 fell on the development side. A 2-question timing run (development side, stub
  reader) printed `fixed8`'s summary for those 2 questions before the check below was declared; nothing was changed
  because of it.
- **Pre-registered evaluation sample:** 100 questions per domain, drawn with seed 30 from the evaluation side
  (`question_pool(..., "eval", {"seed": 30, "per_domain": 100})`): **500 questions**, dealt to 5 shards.
- Nothing is trained on UtilMem. The head and the reader are used as they are.

**Arms** (the §23 arms, unchanged: 5% budget, `memory.count_labels`, lexical search):
- FIFO + floor (3k target, top 5 retrieved): the baseline.
- `fixed8`: the §19 head A (`runs/lme_n4_head_f0_a/checkpoints/policy_best.pt`, trained on LongMemEval only),
  32 BM25 candidates, keep none. Zero-shot.
- `fixed16`: the same head, 16 shown.
- **Reference ("oracle"):** the question domain's evidence sessions only, unlimited budget (`env.oracle: true`,
  `full_context`). This is the paper's reference answer, rewritten by our reader, since theirs is not released.

**Reader and judge.**
- Reader: Qwen2.5-7B-Instruct, vLLM 0.8.5 on an A40, `--max-model-len 24576`, temperature 0, `reasoning: false`,
  compact labels, **up to 512 new tokens** (fixed now; the paper sets no length). Instructions: a long-form version
  of the §23 prompt (`utilmem.GOAL`): ground every detail in the memory, beware of look-alike conversations, reply
  "unknown" if nothing is relevant. No tuning.
- Judge: **the paper's Appendix F prompt, vendored** (`utilmem.JUDGE_PROMPT`; list markers rebuilt from the HTML).
  The context is the reference's evidence sessions, cut the same way. **One judge, Qwen2.5-7B, on the same pod**,
  in one blind pass after all answers, as §23: it never sees the arm. Unparsable verdicts are counted and dropped
  (flagged if above 5% per arm).
- **Their GPT-5.2 and GPT-5.4 judges are not proposed.** About 1,500 judge calls of about 9k–10k tokens each for
  the 500 questions, on a paid API, plus the reference answers that only our reader can write. Not comparable to
  their numbers anyway, since the reference model differs.
- Their Qwen3-235B QA model is not proposed (it needs several 80 GB GPUs).

**Measures** (per arm, on the 500 questions).
- **Their metrics:** RS, NR = (RS − 1) / 9 on a 0–1 scale, DR (score 6 or lower).
- **Our three shares**, for the §23 tables: unknown = the answer starts with "unknown" or has a §24 refusal phrase
  in its first 200 characters; answered-correct = not unknown and score 7 or higher (not degraded, by the paper's
  own threshold); answered-wrong = not unknown and score 6 or lower.
- **P(correct | all in view):** answered-correct among questions with every evidence session touched, beside
  P(correct | not all in view).
- Evidence in view: session recall, all in view, any in view, and the share of in-view turns that are same-domain
  distractors. Prompt tokens.

**Pre-registered claim (primary).** `fixed8` minus FIFO + floor on NR, Qwen 7B, the 500 questions, paired by
question.
- **"Non-inferior":** the 95% paired bootstrap interval's lower bound is above −0.03 (NR on its 0–1 scale, so 3
  NR points). **"Better":** lower bound above 0. **"Worse":** upper bound below 0. Otherwise undetermined.
- A bootstrap over bundles (questions share a history) is reported beside it as a check.
- **Direction expected, written before any check: `fixed8` above FIFO + floor (better).** Reason: the evidence is
  old and spread out, so FIFO's recent window rarely holds it, and the head picks 8 turns where FIFO + floor
  retrieves 5. Against it: the distractors are written to share words with the evidence, and the paper finds
  that a few distractors in view cost much of the score.
- Secondary (descriptive): DR, RS, the three shares, `fixed16`, by domain, P(correct | all in view).

**Free development-side check (pre-declared before it ran).** The three arms on the 349 development questions
with a stub reader (`configs/sweeps/exp30/exp30_inview.py`): session recall, all in view, any in view, evidence-turn
share, the share of in-view turns that are same-domain distractors, and reader prompt tokens, with 95% intervals
by question.
- If FIFO + floor already holds nearly all the evidence (as in §25), §30 stops before any pod.
- If every arm's all-in-view is near 0, the benchmark measures a regime the head cannot reach, and that is said
  before any pod.


**Free development-side check: result** (349 development questions, stub reader, run once after the declaration
above; 95% interval by question; `runs/exp30/dev_inview.json`):

| arm | session recall | all in view | any in view | evidence turns in view | in-view turns that are evidence | in-view turns that are same-domain distractors | reader prompt tokens |
|---|---|---|---|---|---|---|---|
| FIFO + floor | 0.536 (0.516, 0.555) | 0.049 (0.029, 0.072) | 0.997 | 0.206 | 0.560 | 0.174 | 2,896 |
| `fixed8` | 0.581 (0.559, 0.602) | 0.092 (0.063, 0.123) | 0.994 | 0.215 | 0.513 | 0.366 | 1,689 |
| `fixed16` | 0.683 (0.660, 0.705) | 0.180 (0.140, 0.224) | 0.997 | 0.326 | 0.468 | 0.385 | 3,296 |

- `fixed8` minus FIFO: session recall +0.045 (+0.022, +0.067); all in view +0.043 (+0.009, +0.077); same-domain
  distractor share +0.192 (+0.170, +0.214); 1,207 fewer prompt tokens.
- `fixed16` minus FIFO: session recall +0.147 (+0.122, +0.173); all in view +0.132 (+0.089, +0.178).
- By domain, all in view (FIFO / `fixed8` / `fixed16`): 10-K 0.01 / 0.01 / 0.01; studychat 0.00 / 0.00 / 0.05;
  fitness 0.00 / 0.03 / 0.14; finance 0.08 / 0.28 / 0.17; mental health 0.16 / 0.14 / 0.57.

**What the check says.**
- **Not the §25 case.** FIFO + floor sees about half of the evidence sessions, not all. Recency does not win.
- **Neither stop rule fires.** All in view is low (0.05–0.18) but not near 0; every arm touches about half the
  evidence sessions.
- **The budget is smaller than the evidence.** A 5% budget is about 3.8k tokens; the evidence alone is a median
  8.3k. So every arm sees at most a fifth to a third of the evidence turns, and the reference answer saw all of
  them. Expect low scores and a high DR for every arm; the arm differences will be squeezed.
- **The head finds a little more evidence, and twice the look-alikes.** `fixed8` touches slightly more evidence
  sessions with 42% fewer prompt tokens, but 37% of what it shows is same-domain distractors against FIFO's 17%.
  The paper finds a few distractors cost much of the score. So the sign of `fixed8` minus FIFO on NR is genuinely
  open. The pre-registered direction (better) stays as written; this check weakens it.

**Stub check (done, free; prelaunch rule).**
- Full size: 5 arm sweeps (3 cells each) and 5 reference sweeps, 20 cells, 500 questions per arm, all at once,
  stub reader (`runs/exp30/stub.sh`).
- **Resume on a new endpoint:** all 10 sweeps were killed after 15 s (arm cells at 6–21 of 100 questions; the
  reference cells had already finished, in 5–10 s), then rerun with a different `base_url`. All 20 cells finished
  with 100 distinct questions each, `resumed: true`, none refused.
- **Peak RSS 0.26 GB** per arm sweep (3 cells), 0.05 GB per reference sweep. About 1.5 min per arm sweep on the
  laptop (no model).
- **Calls and prompt sizes** (controller side only; no evidence number was read on the evaluation side): 1,500
  reader calls (mean prompt: FIFO 2,916 tokens, `fixed8` 1,660, `fixed16` 3,141; largest 4,379); 500 reference
  calls (mean 6,487, largest 18,316); 1,500 judge calls (mean 7,239 with stub answers, about 7,900 with real ones;
  largest about 19,900). All fit `--max-model-len 24576` with 512 new tokens.
- The judge script, the in-view script and the report ran end to end on 8 development questions with made-up
  scores (pipeline check only).
- All stub output and stub caches were deleted (`cache/generations_exp30_eval_qwen7b`, `cache/judge_exp30_stub`).
- **Disk: 14 GB free**, below the 15 GB floor (not from this work). At least 1 GB must be freed before any launch.

**Cost (NEEDS SPEND; from §23's measured calls on the same A40, $0.59/h).**
- §23's 1,500 Qwen 7B calls (mean prompt 2,193 tokens, mean output 27) took 10.3 min with about 12 calls in
  flight. A fit of their per-call times gives about 1.23 s + 0.28 s per 1,000 prompt tokens + 0.109 s per output
  token. **Long answers make output the cost driver here.** The output slope comes from short answers, so it is
  an extrapolation.
- §30: 3,500 calls. Answers assumed 250–450 tokens (cap 512), judge verdicts 150–300 (cap 400). That gives
  **about 2.1–3.7 h of generation** (central about 2.8 h), plus about 5 min to load and 10 min of checks.
- **About $1.30–2.30 (central about $1.75). Hard stop at 4 h after create (about $2.36).** If the pod is not
  $0.59/h, the figure is quoted again before the run.
- **Stage order, one model (Qwen 7B), one pod:** the 15 arm cells and the 5 reference cells at once, then the
  judge pass, FIFO and `fixed8` first, `fixed16` last (`--arms`).
- **Triage rule, declared now:** if fewer than half of the arm-cell questions are done 1.5 h after create, the
  `fixed16` cells are stopped and `fixed16` is not judged, so that the claim's two arms finish.
- Their GPT-5.2 and GPT-5.4 judges are not priced in: about 1,500 calls of about 8k tokens each per judge, on a
  paid API that needs a key. Not proposed.

**If §30 closes without a paid run (pre-declared, free).**
- One label-only evaluation on the 500 evaluation questions: session recall, all in view and the distractor share
  for the three arms, from the committed configs with only the reader set to stub, with 95% intervals by question.
- Computed **exactly once**, and **only after the user decides there is no paid run**. If the user chooses the
  paid run, the run reports the same numbers as its diagnostic instead.
- Not executed.

**Limitations, stated before any run.**
- One judge, Qwen 7B, also the reader that wrote the reference and the answers: it may favour its own phrasing. It
  is the same for every arm, but not neutral, and its reliability on 8k-token rubric prompts is untested. Our
  numbers are not comparable with the paper's.
- Turns cut at 1,000 tokens (4% of turns; mostly pasted filings and assignments).
- The length confound above (evidence turns are longer than distractors).
- No licence for the data; one benchmark; one budget (5%), below the size of the evidence.

**Recommendation (to the user): no pod for now.** The check closes the §25 failure mode, but the run would most
likely end "undetermined" or "non-inferior" on a small difference, for about $1.75. The judge is an untested 7B
self-judge, and the data has no licence. A pod is worth it only if a UtilMem row is wanted for the write-up.

## 31. Test-time training on the 3B reader: the last route to §23's claim 1 without LoRA (2026-10-09, pre-registered; closed, not run)

**Status: CLOSED without a pod run, at the user's decision (2026-10-10).** The free part below is kept as the record.
No pod was started and nothing was spent. Reasons given at the decision: the third-party re-creation found almost no
gain at small scale (+0.2 F1), and our prompts (about 1,150 tokens) are far shorter than the contexts qTTT was built for.

**Why.**
- §23's claim 1 was NOT SHOWN: Qwen 3B with `fixed8` (0.315) against Qwen 7B with FIFO + floor (0.436), −0.121
  (−0.170, −0.070), margin −0.03.
- The controller already puts the evidence in view. What the 3B lacks is reading: P(correct | all in view) under
  `fixed8` is 0.397 at 3B against 0.667 at 7B, and the 3B answers "unknown" on 0.372 of answerable questions.
- Test-time training (TTT) changes the reader for one question at a time, using only that question's prompt. It is
  the only reader change left that is not LoRA training.

**The two methods, and their code.**
- **qTTT** (Bansal et al., "Let's (not) just put things in Context: Test-Time Training for Long-Context LLMs",
  ICLR 2026, arXiv 2512.13898).
  - **No code released.** No repository is linked from the paper, and none was found on GitHub on 2026-10-09.
    The paper is CC BY 4.0.
  - **What is adapted (Algorithm 1, Appendix C):** W_Q, the query projection of every attention layer. Nothing else.
  - One prefill caches the prompt's keys and values; they are never recomputed.
  - Then N steps of AdamW, each on the next-token loss of one random k-token span of the same prompt, with the span
    attending to the frozen cached prefix. Then the answer is decoded from the frozen cache with the adapted W_Q.
  - **Defaults:** N = 32, k = 128, AdamW with weight decay 0.01, gradient clipping 1.0, bf16, batch 1.
  - **Learning rate: no single default.** The paper picks it per dataset on held-out validation data, from
    {3e−4, 3e−5, 1e−5, 3e−6, 1e−6, 3e−7}. Its Table 1 says results are flat between 1e−5 and 1e−6.
  - Models: Qwen3 1.7B–8B (and 32B), contexts of 8k–128k tokens on the benchmarks.
- **EASE-TTT** (Yuan et al., arXiv 2606.06906).
  - **No code released.** The authors' repository (github.com/xia0pengyuan/EASE-TTT, created 2026-05-26) is empty:
    no commits, no licence. A third-party MIT re-creation exists (andrewtjin/STEEZ-TTT, commit e8968d5, Qwen3-0.6B
    only). It was read, not run.
  - **What is adapted:** LoRA (rank 8, scale 16, dropout 0.05) on the query projections only.
  - **Loss:** KL from a target attention distribution to the layer-14 attention of the last token. The target puts
    mass 0.6 on the top 4 "evidence" chunks (512-token chunks cut at loss spikes), ranked by how much each lowers the
    loss on the question. 15 steps, AdamW at 1e−4, weight decay 0.01.
  - Models: Qwen3-0.6B, Qwen3-1.7B, Llama-3.2-1B. Reported cost: 9.1 s per example against qTTT's 6.7 s
    (Qwen3-1.7B, contexts up to 32k).
- **The third-party re-creation's finding (not ours):** at Qwen3-0.6B, qTTT with the paper's objective changed
  266 of 550 answers but gained only +0.2 F1 overall; the paper's qTTT gain was not reproduced at that size.

**Decision: qTTT only. EASE-TTT is not run** (decided before any run, for these reasons):
- At our prompt lengths it has nothing to select. A `fixed8` prompt is about 1,150 Qwen tokens: two or three
  512-token chunks. "Top 4 chunks" is then the whole prompt, so the target is uniform and carries no signal.
- Using per-turn chunks instead would be a new method, not EASE-TTT.
- Its layer (14 of 28 in Qwen3) has no counterpart in a 36-layer Qwen2.5-3B. Choosing one would need a training-fold
  sweep and a GPU.

**Hyperparameters, all fixed now, before any run (no held-out peeking).**
- N = 32 steps, k = 128 tokens, AdamW with weight decay 0.01, clipping 1.0, W_Q of all 36 layers (the bias stays
  frozen): the paper's defaults.
- **Learning rate 3e−6**, chosen from the paper alone: the geometric middle of its recommended range [1e−6, 1e−5].
  In its Table 1, on the four rows nearest our prompt lengths (512 to 2,536 tokens), 3e−6 and 1e−5 are tied (sum
  128.5 against 127.1). Our data was not used.
- **Nothing is learned across questions.** Each question starts from the original weights; W_Q is restored bit for
  bit and the optimizer is dropped after each answer (tested). The random spans are seeded by the prompt's hash, so
  an answer does not depend on the order of questions, and a resumed run gives the same answer.
- The training signal is the question's own prompt (its memory lines and the question text), with no labels.

**A precision trap, found in the free part.** At lr 3e−6 one AdamW step moves a weight by about 3e−6. A bf16
weight of size 0.02 has a rounding step of about 1e−4, so in plain bf16 the update rounds away and **nothing
adapts**. The server keeps W_Q (and computes the query projection) in float32; the rest of the model is bf16
(tested: a bf16 model's W_Q moves at 3e−6).

**Serving: vLLM cannot do this, so a small HF server.** `memctl/ttt_server.py` (torch, transformers and the standard
library only; the pod fetches this one file at a pinned commit).
- An OpenAI-compatible `/v1/chat/completions`. One loaded model answers under two names: `qwen2.5-3b-instruct-hf`
  (no adaptation) and `qwen2.5-3b-instruct-hf-qttt` (32 steps). memctl's `openai` backend calls it unchanged, so
  `memctl.sweep` writes the same generation cache and replays and resumes as with vLLM.
- The prompt is the frozen 7b5fc30 reader prompt (`memctl/agents/llm.py`, reasoning on, compact labels, 256 new
  tokens), as one user turn through the model's chat template, as vLLM does.
- Greedy decoding with the Qwen2.5 `generation_config` repetition penalty 1.05, which we believe vLLM 0.8.5 applied
  in §23 (not verified). The paired rerun makes the TTT effect immune to this, and the A-against-§23 check measures it.
- Several model replicas (processes) share one GPU behind one port. `/stats` gives running mean timings.
- A wrong model name is refused (404), so no answer can be cached under the wrong arm.
- Tested on transformers 5.17 (the venv) and 4.51.3 (vLLM 0.8.5 needs at least 4.51.1; the image's exact version is
  printed on the pod before the run).

**Arms (pre-registered; LongMemEval, the same 470 answerable + 30 abstention questions, 5 folds, §23's controllers
unchanged):**

| arm | controller | reader |
|---|---|---|
| A | `fixed8` | 3B, HF server, no adaptation |
| B | `fixed8` | 3B, HF server, qTTT |
| C | FIFO + floor | 3B, HF server, no adaptation |
| D | FIFO + floor | 3B, HF server, qTTT |
| 7B | FIFO + floor | §23's Qwen 7B answers (vLLM), judged again in the same pass |

- **The no-adaptation 3B rows are rerun in the same session, not taken from §23.** HF and vLLM differ in kernels
  and rounding, so §23's vLLM rows would mix "TTT" with "a different serving stack". A and C go through the same
  server and code path as B and D, so B − A is TTT alone. §23's 3B `fixed8` answers are judged again as a check of
  the serving change.
- **The 7B row is reused from §23** (answers only; they are judged again in the same blind pass). Those answers
  came from the same vLLM 0.8.5 image and model at temperature 0, and §23's 7B rerun matched §19c on 0.979 of
  questions, so a new run would repeat them. Claim 1 compares two readers on two serving stacks either way, as §23
  did.
- **Claim 1 again:** B − 7B FIFO + floor, paired by question; non-inferior if the 95% lower bound is above −0.03,
  better if above 0.
- **TTT effect (paired):** B − A (primary), and D − C as the control, so TTT's gain is separated from the
  controller's. (B − A) − (D − C) is descriptive.
- **Judge:** one blind Qwen 7B pass with the official LongMemEval prompts, as §23; it sees question, gold and answer
  text only. Answers are made first with a stub judge, then judged together.
- **Reported beside accuracy:** the unknown share, P(correct | answered), false answers on the 30 unanswerable
  questions, and the mean span loss before and after adaptation from the server log (a mechanical check that
  adaptation happened).
- **Expectation, written before any run:** claim 1 NOT SHOWN. qTTT targets "score dilution" in long contexts
  (8k–128k tokens in the paper); our prompts are 1–3k tokens. The 3B's problem is refusals and reading, and the
  re-creation found almost no qTTT gain at 0.6B. A gain at FIFO + floor (longer prompts) larger than at `fixed8`
  would fit the dilution story.

**Cost per query: measured on CPU, extrapolated to the A40.**
- **Method.** The real Qwen2.5-3B layer shapes (hidden 2,048, 16 query and 2 key-value heads, MLP 11,008, vocabulary
  151,936; config and tokenizer only, about 12 MB; no weights downloaded), random weights, float32, 8 CPU threads
  (i7-11800H), with 1 and then 2 of the 36 layers. A straight line through the two gives the cost per layer and the
  fixed part (embedding and output head); 36 layers are extrapolated from it. `python -m memctl.ttt_server bench`.
  Prompt lengths are the arms' means in Qwen tokens: `fixed8` about 1,150, FIFO + floor about 3,250 (1.095 Qwen
  tokens per memctl token, measured on §23's 3B prompts).
- **Measured on CPU (seconds):**

  | | 1 layer, 1,150 / 3,250 tokens | 2 layers, 1,150 / 3,250 | 36 layers (extrapolated), 1,150 / 3,250 |
  |---|---|---|---|
  | prefill | 0.52 / 1.55 | 1.10 / 3.18 | about 21 / 59 |
  | 32 qTTT steps | 25.6 / 27.0 | 31.8 / 33.6 | about 242 / 258 |
  | decode, 32 tokens | 3.3 / 3.3 | 4.0 / 4.0 | about 28 |

  - Peak RSS 3.4 GB (2 layers). The span loss fell in every run (for example 12.32 → 12.24), so the steps do update.
- **FLOPs per question (counted from the shapes; 3.09B parameters with the tied output head):** 32 steps on 128-token
  spans are about 55 TFLOP (`fixed8`) and 60 TFLOP (FIFO + floor), about 1.7–1.9 TFLOP a step (forward, backward to
  the activations, W_Q gradients, attention to the cache). The prefill is about 6.6 and 19.6 TFLOP.
- **CPU check of the count:** the CPU ran the 32 steps at 0.72 of the efficiency of the prefill (0.23 against 0.31
  TFLOP/s), so the steps have no hidden cost beyond their FLOPs and small-matrix overhead.
- **A40 extrapolation. Assumptions:** bf16 tensor peak 149.7 TFLOP/s; the prefill at 40% of peak (60 TFLOP/s);
  the steps at 0.72 of that (as on CPU), with a pessimistic case at 15% of peak; 0.5 s of kernel-launch overhead for
  32 steps (about 1,600 kernels a step at about 10 µs); decoding by HF at 30 ms a token (the memory-bandwidth floor
  is 9 ms: 6.2 GB of weights at 696 GB/s; HF's Python loop adds the rest); 32 output tokens on average (the mean of
  §23's 3B answers in Qwen tokens; p90 85, maximum 256; used for cost only).

  | per question, one replica | `fixed8` | FIFO + floor |
  |---|---|---|
  | prefill | 0.1 s | 0.3 s |
  | 32 qTTT steps (expected / pessimistic) | 1.8 s / 2.9 s | 1.9 s / 3.1 s |
  | decode (32 tokens) | 1.0 s | 1.0 s |
  | **total with qTTT** | **about 2.9 s (pessimistic 4.0 s)** | **about 3.2 s (pessimistic 4.4 s)** |
  | total without (arms A, C) | about 1.1 s | about 1.3 s |

  - **Feasible: well under 10 s a question.** The worst single question (6,357-token prompt, 256 tokens out) is
    about 12 s.
  - **A warning from the paper itself.** Its Table 11 lists 13.7 s for 32 qTTT steps on Qwen3-4B on an A100 at every
    context length. Scaled to 3.09B parameters and to the A40's half of the A100's bf16 peak, that is about 22 s a
    question, over the 10 s line. That table looks derived from FLOPs rather than timed (its 8k-token prefill of a
    1.7B model takes 8.7 s, about 1% of the A100's peak), and our FLOP count gives 1–3 s. The disagreement is the
    reason for the time gate below.
- **GPU memory per replica:** bf16 weights 6.2 GB; float32 W_Q, its saved copy, gradient and AdamW moments 3.0 GB;
  cache, span activations and logits under 1 GB. About 10 GB, so 4 replicas fit the A40's 48 GB (3 if not).

**Full-size stub run** (the pre-launch rule; `configs/sweeps/exp31/exp31_stub.sh`).
- **Pipeline at full size:** 5 folds × 4 cells × 100 questions = 2,000 answers through the HF server (1,000 with
  32 qTTT steps), with the real Qwen2.5 tokenizer and chat template, a random Qwen2-architecture model, 4 replicas
  and 2 sweep workers a fold. 2,000 of 2,000 answers, 0 server errors after the fix below; all 20 cells finished
  with 100 episodes.
- **Resume against a changed endpoint:** interrupted with the plain cells at 25–31 questions, resumed on port 8132,
  interrupted again, resumed on port 8134. Every cell resumed (`resumed: true`, none refused) and finished.
- **The stub found a bug before any spend:** in the first pass every qTTT request failed (HTTP 500). With fewer layers
  than the checkpoint, the per-layer type list kept 36 entries and the cache crop failed on the empty ones. Fixed
  (`_trim_layer_types`); the real 36-layer model does not hit it.
- **Peak RSS:** 0.82 GB for the stub server (the largest process), 0.34 GB per sweep process.
- **Stub model:** the first pass used a 42M-parameter random model; it was too slow on the laptop CPU (about 37 s
  a qTTT answer under load), so the resumed passes used a 10M one (hidden 64, 1 layer). Random weights write
  256 tokens every time; the mean server time was 2.7 s without and 13.1 s with adaptation (10.5 s of it the 32
  steps, dominated by the 151,936-word output head). These are CPU stub numbers, not a GPU estimate.
- **Judge and report scripts** (`configs/sweeps/exp31/exp31_judge.py`, `exp31_report.py`) ran on the stub output
  with the stub judge: they read §23's 3B `fixed8` and 7B FIFO rows correctly (unknown 0.372 and 0.328, false
  answers 7 and 6 of 30, as §23). `--prune` keeps only scored step rows, because full step logs are about 1.7 MB a
  question (about 3.4 GB for this run); the stub resume passes ran with step logs off for the same reason.
- **Stub output deleted** (runs and caches, about 0.5 GB at the peak). Free disk stayed at 14 GB throughout.

**Price (NEEDS SPEND; per-run user approval).**
- **Plan, one A40 (EU-SE-1, $0.59/h as §23), one pod, two phases:**
  1. The §23 image (vLLM 0.8.5, which has torch and transformers ≥ 4.51.1) with the start command replaced: fetch
     `memctl/ttt_server.py` from GitHub at a pinned commit (ae1a5e8, or the branch head at launch, recorded), then
     `serve --model Qwen/Qwen2.5-3B-Instruct
     --revision aa8e72537993ba99e69dfaafa59ed015b17504d1 --served-name qwen2.5-3b-instruct-hf --replicas 4`.
     Locally: `configs/sweeps/exp31/exp31_stage.sh POD_URL` (5 folds × 4 cells, 4 workers a fold).
  2. The pod switched to `vllm serve Qwen/Qwen2.5-7B-Instruct` (as §23); then `exp31_judge.py --prune` (2,820 judge
     calls: 4 cells and 2 §23 rows of 470) and `exp31_report.py`.
- **Time gate (pre-registered, automatic in `exp31_stage.sh`):** after 100 qTTT answers, if the mean server time per
  qTTT answer divided by the 4 replicas is above 10 s, all sweeps stop and the pod is terminated. A failed gate costs
  about 15 min, about $0.15.
- **Estimate:** pod ready about 7 min (3B download, 4 loads); reader 2,000 answers about 30–45 min (1,000 at about
  3–4.4 s and 1,000 at about 1.2 s, with the replicas overlapping decoding about 2×); 7B load about 7 min; judge about
  5–10 min. **About 50–70 min, about $0.50–0.70.**
- **Hard stop: 1.5 h after create, about $0.89.** If the reader stage is not done by 1 h 05 min, the FIFO + floor
  qTTT cell (arm D, the control) is dropped first, so that arms A, B and the judge pass still fit.
- Not yet measured, and only measurable on the pod: GPU seconds per step. The gate is the check.

**Recommendation.** **Pod, if the user wants this route closed; otherwise stop.** It is feasible (about 3–4 s a question
expected, under the 10 s line) and cheap (under $0.90 with the hard stop), and it is the last reader change short of
LoRA. But the expected result is NOT SHOWN: the paper's gains come at 8k–128k tokens, our prompts are 1–3k, and an
independent re-creation found almost no qTTT gain at 0.6B. To pass the margin, B would need about 0.46 (the 7B's
0.436 plus about +0.02, given an interval about ±0.05 wide): B − A of about +0.14.

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

## 44. Is the answer at the answer position put there by the evidence? Activation patching and an evidence swap (2026-10-29, pre-registered; in progress)

**Status: IN PROGRESS.**
- **Rev 1, conditions S and U on Qwen 2.5 3B:** done.
  - **S0 FAILS** (34/46 = 0.739): the swap is not valid, and S1–S3 are not read.
  - **U1 reads "not the format":** 0.228 (21 of 92) of the 3B's in-view refusals become gold under U, 42% still refuse, and
    Steer-under-U − U is +0.152 (+0.065, +0.239).
- **Llama 3B S/U:** done, descriptive by rev 2. S0 is valid (55/66 = 0.833); S1 is not reliably distinguishable (half-width 0.167 > 0.15).
- **Rev 0 (clean → cut patching):** development done for both models. The confirmation reads (folds 2–4) are pending.

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

No repaired rev is ordered. The user decides between a rev 8 (Qwen 3B only, with "other" substitutes dropped, bare
numbers form-matched and the first content token read) and closing the swap line.

**Caveats.** One benchmark and prompt family; one forced position; first-token logit differences. The swap subset is
biased toward extractive single-hop rows (§F). The 7B was run in bf16 on CPU. The rev 0 numbers above are development
folds only.

**Spend.** $0 (laptop). All experiments ≈ $20.06 of $30.

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

## 5. The sequential task

```bash
python -m memctl.sweep --config configs/sweeps/exp5_workflow.yaml
```

Workflow task: 16 interleaved jobs of 4 stages, 400-step budget; a stage needs
the token its job's previous stage returned, and an agent that has lost it must
restart the job. Archive and retrieval allowed. Task success is the share of
jobs completed.

| controller | 5% | 10% | 20% | 40% |
|---|---|---|---|---|
| no_controller / fifo | 0.028 | 0.221 | 0.891 | 1.000 |
| lru | 0.028 | 0.209 | 0.879 | 1.000 |
| age_decay | 0.018 | 0.139 | 0.800 | 1.000 |
| random | 0.026 | 0.129 | 0.536 | 0.987 |
| lfu | 0.004 | 0.052 | 0.283 | 0.996 |
| salience | 0.001 | 0.051 | 0.399 | 1.000 |
| fifo_archive_retrieve | 0.023 | 0.170 | 0.890 | 1.000 |
| salience_archive_retrieve | 0.009 | 0.084 | 0.577 | 1.000 |
| oracle_approx (archives, retrieves) | 1.000 | 1.000 | 1.000 | 1.000 |
| full_context | 1.000 | 1.000 | 1.000 | 1.000 |

**What it shows**

- **The ranking flips.** Salience, the best heuristic on the recall task, is
  the worst here; FIFO and LRU are the best. Used tokens look exactly like
  needed tokens to a content rule, and it fills memory with them.
- Memory failures compound. At 10% FIFO restarts a job 51 times per episode
  and the episode runs to the 400-step limit; with full context it finishes
  in 192 steps with no restart.
- Lexical retrieval does not rescue the archive here: recall 0.01 to 0.10.
  The instruction that needs a token shares more words with other
  instructions than with the item holding the token. All those failures are
  labelled `archived_not_retrieved`, so the blame lands on the retriever.

This is the experiment that motivates a learned controller: no fixed heuristic
is good on both tasks.

## 5b. Learned controllers on the sequential task

```bash
python -m memctl.rl.train --config configs/rl/rl_bc_workflow.yaml
python -m memctl.rl.train --config configs/rl/rl_ppo_workflow.yaml
python -m memctl.rl.train --config configs/rl/rl_ppo_workflow_gamma0.yaml
python -m memctl.sweep --config configs/sweeps/exp5b_workflow_rl.yaml
```

Delete-only. `rl_bc_transfer` is the recall-task imitation policy from
Experiment 4, run here without retraining. `oracle_approx_lazy` is the
approximate oracle that evicts only under pressure.

| controller | 5% | 10% | 20% | 40% | restarts / episode at 5% | steps at 5% |
|---|---|---|---|---|---|---|
| fifo | 0.028 | 0.221 | 0.891 | 1.000 | 64.2 | 400 |
| lru | 0.028 | 0.209 | 0.879 | 1.000 | 64.3 | 400 |
| salience | 0.001 | 0.051 | 0.399 | 1.000 | 73.5 | 400 |
| rl_bc_transfer (trained on the recall task) | 0.042 | 0.247 | 0.959 | 1.000 | 67.8 | 400 |
| rl_bc_workflow (imitation, 240 episodes) | 1.000 | 1.000 | 1.000 | 1.000 | 1.2 | 199 |
| rl_ppo_workflow (PPO, discount 0.995) | 0.404 | 0.983 | 1.000 | 1.000 | 43.1 | 399 |
| rl_ppo_workflow_gamma0 (PPO, discount 0) | 0.992 | 1.000 | 1.000 | 1.000 | 16.5 | 309 |
| oracle_approx | 0.808 | 1.000 | 1.000 | 1.000 | 25.0 | 297 |
| oracle_approx_lazy | 0.896 | 1.000 | 1.000 | 1.000 | 12.1 | 260 |

**What it shows**

- **A policy trained on this task solves it.** At a 5% budget (190 tokens,
  less than the 16 jobs' live tokens can need at once) the imitation policy
  completes every job with 1.2 restarts per episode, where FIFO completes 3%
  with 64 restarts. It keeps tokens whose stage has not run yet and drops
  used ones and noise: features it can see (`access_count`, source type,
  specificity).
- **It beats the oracle, and the reason is instructive.** In the 49 episodes
  where live tokens exceed the budget a restart is unavoidable for everyone,
  including the oracle. After the first restart the episode no longer matches
  the reference pass, so the oracle's hindsight describes a different
  episode and it evicts blindly: 25 restarts per episode. The learned policy
  never relied on hindsight and recovers. The oracle is a valid reference
  only up to the first divergence on this task; the harness now records that
  step (`hindsight_diverged_at`).
- **The recall-trained policy does not transfer** (0.042 at 5%). It behaves
  like FIFO here, not like the salience rule it resembles on its own task
  (removal overlap with FIFO 0.00, with salience 0.24). Weak support for H7
  across tasks: it does not collapse the way salience does, and it does not
  solve the task.
- **H2 is not supported here either.** The immediate-reward PPO run ends at
  0.992 and the long-horizon one at 0.404 at 5%; both learning curves were
  unstable (the long-horizon run peaked at 0.42 on validation after 360
  episodes and drifted), and there is one seed of each. This is the task
  where the two rewards should disagree, and the comparison needs more seeds
  and a stabler optimiser before it says anything.

## 6. LoCoMo and LongMemEval: evidence retention

```bash
python -m memctl.sweep --config configs/sweeps/exp6_locomo_retention.yaml
python -m memctl.sweep --config configs/sweeps/exp6_longmemeval_retention.yaml --workers 4
```

No task model (`agent: null`), so **task success is 0 by construction and is
not a result**. The measure is `evidence_complete_rate`: the share of questions
whose evidence was all in ACTIVE when the question was asked, which is the
accuracy a perfect reader would reach.

**LoCoMo** (10 conversations, 1,986 questions):

| controller | 10% | 25% | 50% |
|---|---|---|---|
| fifo_delete | 0.016 | 0.173 | 0.445 |
| random_delete | 0.041 | 0.169 | 0.428 |
| similarity_delete | 0.051 | 0.181 | 0.463 |
| salience_delete | 0.152 | 0.329 | 0.584 |
| rag_keep_last4 | 0.280 | 0.280 | 0.280 |
| fifo_archive_retrieve_embedding (hashing) | 0.151 | 0.305 | 0.538 |
| fifo_archive_retrieve_lexical | 0.417 | 0.525 | 0.672 |
| oracle_delete_only_exact | 0.614 | 0.953 | 1.000 |
| oracle_approx (archives, retrieves) | 1.000 | 1.000 | 1.000 |

BM25 retrieval recall is 0.35 to 0.38 and precision 0.06 to 0.09. The exact
delete-only oracle reaches 1.000 only at 50%, consistent with the earlier
measurement that evidence is about 31% of LoCoMo's tokens.

**LongMemEval-S** (first 100 questions, about 102,000 tokens of history each):

| controller | 5% | 10% | 25% |
|---|---|---|---|
| fifo_delete | 0.050 | 0.060 | 0.210 |
| random_delete | 0.020 | 0.080 | 0.190 |
| salience_delete | 0.490 | 0.670 | 0.820 |
| fifo_archive_retrieve_embedding (hashing) | 0.270 | 0.280 | 0.380 |
| fifo_archive_retrieve_lexical | 0.590 | 0.600 | 0.670 |
| oracle_approx | 1.000 | 1.000 | 1.000 |

**What it shows:** the adapters, the evidence tracking and the retrieval
metrics run on real benchmark data, and the ordering seen on the synthetic task
(content-aware > recency, retrieval > deletion) appears here too.

**What it does not show**

- QA accuracy. That needs a task model and was not run; the GPU block has since
  been lifted by renting one (RunPod, vLLM; Experiment 9). The numbers above
  are an upper bound for a perfect reader.
- Anything about long-horizon control. Every question comes after the history.
- That salience is good on LongMemEval for a meaningful reason. It prefers
  user turns to assistant turns, and in this dataset the evidence is mostly in
  short user turns while assistant turns are long. That is a property of the
  dataset's construction.
- LoCoMo has 10 episodes; differences between close controllers are not resolved.

## 6c. Dense against lexical retrieval on LoCoMo

```bash
python -m memctl.sweep --config configs/sweeps/exp6c_locomo_dense_retrieval.yaml --workers 2
```

LoCoMo, budget 10%, no task model, FIFO that archives everything it removes
and retrieves the top 5 when a question arrives. Two retrievers (BM25 over the
words; cosine over an embedding) and two embedders (word-hashing;
`BAAI/bge-small-en-v1.5` on CPU).

| retriever | embedder | evidence complete | retrieval recall | retrieval precision | controller time / episode |
|---|---|---|---|---|---|
| BM25 | (none needed) | 0.417 | 0.347 | 0.094 | 3–6 s |
| embedding cosine | hashing | 0.151 | 0.115 | 0.031 | 0.9 s |
| embedding cosine | bge-small | 0.240 | 0.215 | 0.058 | 0.6 s + embedding |

The dense embedder roughly doubles the hashing embedder's recall but stays
well below BM25 on this data. Questions here name specific things ("the LGBTQ
support group") that exact word overlap catches and a small sentence
embedding blurs; the earlier work found dense retrieval better when the
embedded text carried the speaker's name, which this run did not add. Ten
episodes; treat the ordering as indicative. The point for the framework is
that swapping the retriever is a config change and every retrieval is
measured.

## 7. A real language model in the loop (smoke tests)

Both tests use `llama3.1:8b` (4-bit) served by Ollama on this machine's CPU,
through the `openai` backend; generations are cached in `cache/generations`.
They are pipeline checks, run at a size the machine allows, not results.

### 7a. As the task model

```bash
python -m memctl.sweep --config configs/sweeps/exp7_llm_smoke.yaml --workers 1
```

Synthetic recall task, horizon 60, budget 25%, 3 episodes, 11 queries in all.

| controller | task success | evidence in ACTIVE at the question (`needed_hit_rate`) | failures | model time per episode |
|---|---|---|---|---|
| fifo | 0.23 | 0.23 | 9 × `evicted` | 166 s |
| salience | 0.92 | 0.92 | 1 × `evicted` | 86 s |
| oracle_exact | 1.00 | 1.00 | – | 26 s |
| full_context | 1.00 | 1.00 | – | 130 s |

The model answered every question whose evidence was in its context and none
whose evidence was not, so task success equals evidence availability and
every failure is attributed to memory. The agent was swapped in by config; the
harness, controllers and metrics did not change. With 11 queries this says
nothing about the model or the controllers beyond "the pipeline works".

### 7b. As the memory controller

```bash
python -m memctl.sweep --config configs/sweeps/exp7b_prompted_controller_smoke.yaml --workers 1
```

Synthetic recall task, horizon 40, budget 40%, 2 episodes, 5 queries in all,
scripted reader as the task model. The model is asked, whenever memory is over
budget, for a JSON list of actions.

| controller | task success | failures | model calls / episode | parsable replies | forced evictions / episode | controller time / episode |
|---|---|---|---|---|---|---|
| fifo | 0.00 | 5 × `evicted` | – | – | 0 | 0 s |
| salience | 0.88 | 1 × `evicted` | – | – | 0 | 0 s |
| prompted_llama3.1_8b | 0.50 | 4 × `evicted` | 9 | 3 of 9 | 15.5 | 824 s |
| oracle_exact | 1.00 | – | – | – | 0 | 0 s |

Six of the nine replies were not usable JSON (the model kept listing every
item, hit the 120-token reply limit and truncated the list), so on those steps
the controller did nothing and the fallback evicted oldest-first — which the
metrics show as 15.5 forced evictions per episode against 0 for every other
controller. When the reply parsed, its evictions were applied and counted as
its own. The pipeline works and the cost of an LLM controller is visible: 824
s of model time per 40-step episode on this CPU. Five queries say nothing
about how good such a controller could be.
