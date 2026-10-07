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
- This is the thesis claim of `docs/research/ONE_PAGE_RESEARCH_SUMMARY.md`
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
  Next steps: `docs/research/next_directions.html`.

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

Follows the plan in `docs/research/next_directions.html`. Everything here uses
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

## 13. LongMemEval and LoCoMo under the reviewed protocol (2026-10-07, in progress)

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
