# Experiments

Every experiment run while the framework was built, with the command that
reproduces it, the result, and what it does and does not show. Dates are
2026-09-30. Full tables and figures are in each `runs/<name>/report.md` and
`runs/<name>/plots/` (the `runs/` folder is not in git; every table below can be
regenerated with the two commands given).

**How to read the numbers**

- *Task success* is the mean over episodes of the share of queries answered
  correctly (recall task) or jobs completed (workflow task).
- Every controller in a table ran the same seeded episodes (seeds 0 to 99
  unless stated), so differences are paired.
- Budgets given in % are a share of the episode's uncompressed history.
- The task model is the scripted reader unless stated: it answers correctly
  exactly when the needed text is in ACTIVE memory. So task success here
  measures memory management only.
- **Provenance:** all runs were made from an uncommitted working tree
  (`metadata.json` records commit `d215c8f` plus `dirty: true`), and the code
  changed between the early and late experiments. See
  [SCIENTIFIC_VALIDITY_REPORT.md](SCIENTIFIC_VALIDITY_REPORT.md) §7. Rerun after
  committing before quoting a number in the thesis.

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
| D1 | Does an archive-aware regret teacher fix 4b? | **Yes**: 0.34 → 0.85 at 2%, above every heuristic at every budget. Clean commit, 3 seeds. PASSED |
| 5 | Does the ranking hold on a sequential task? | **No, it flips.** PASSED |
| 5b | Learned controllers on the sequential task | Task-trained imitation reaches 1.0 where the blind oracle does not; the recall-trained policy does not transfer. PASSED |
| 6 | LoCoMo and LongMemEval: evidence retention under a budget | Pipeline PASSED; QA accuracy BLOCKED |
| 6c | Dense against lexical retrieval on LoCoMo | See §6c. PASSED |
| 7 | A real language model as task model and as controller | Smoke tests only. PARTIAL |

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

Delete-only action set. Every policy is the same small network (about 12,000
parameters; DeepSets over per-item features, or a plain MLP for `rl_bc_mlp`),
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
| salience_archive_retrieve | 0.737 | 0.833 | 0.898 | 0.984 |
| rl_bc_archive (imitation of the hindsight expert) | 0.383 | 0.521 | 0.724 | 0.965 |
| rl_bc_ppo_archive (imitation, then 1,344 PPO episodes) | 0.452 | 0.578 | 0.710 | 0.937 |
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
  moved it a little (+0.06 at 2–5%) in 1,344 episodes but did not undo the
  habit.
- A policy that archives freely and retrieves liberally does better here
  than one that decides carefully, because the archive costs nothing and the
  retriever is decent: the untrained retrieval head, which retrieves most of
  its shortlist, is the best learned configuration and matches salience with
  retrieval. This is H4 restated: under uncertainty, archive.
- The right expert for this action set is not the hindsight-optimal one but a
  cost-aware one (archive when unsure; retrieve generously when the archive
  is free), or the learner needs the archive's asymmetry expressed in the
  reward. Neither was tried.

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
  0.591 at 2% in 4b, above the oracle-expert learner (0.383): imitating the
  oracle made the operation choice *worse than untrained*.

**What it does not show**

- The archive is free here, so "archive everything" is the right
  operation choice and the regret teacher says so. Whether a learner tracks
  a *priced* archive is D5 (running).
- The remaining gap to the oracle (0.11 at 2%) is retrieval: the shortlist is
  BM25 top-8, and the learner retrieves 91% of the needed items it is shown.

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

- QA accuracy. That needs a task model, which is BLOCKED on this machine (see
  INFRA_REPORT.md). The numbers above are an upper bound for a perfect reader.
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
