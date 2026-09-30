# Scientific validity report

What the results in [EXPERIMENTS.md](EXPERIMENTS.md) can and cannot support,
hypothesis by hypothesis, and the confounds that anyone quoting them must
carry along. Written 2026-09-30, against the runs of that day.

## 1. The stage gate

The build plan said: do not proceed to the next stage if the current
experiment cannot distinguish controller quality from retrieval or task-model
failures. The check was run at each stage:

| Stage | Check | Result |
|---|---|---|
| 1 (delete-only) | With a perfect reader, every failure must be an eviction; with a known injected reasoning-failure rate, attribution must recover it | 380,231 failures, all `evicted`; injected 20.8% recovered as 19.9–20.8% in every cell (Exp. 1, 1b) |
| 2 (archive) | Archiving without retrieval must equal deletion, with failures relabelled | Identical success to three decimals; all failures `archived_not_retrieved` (Exp. 2) |
| 2 (rewrites) | A lossy rewriter must be blamed as such | 1,368–1,443 `compression_lost_detail` per cell with `truncate`, none with `extractive` (Exp. 3) |
| 4 (RL) | The learned policy's advantage must not come from ground truth | Controllers receive no dependencies (test); training, validation and test seeds are disjoint (100000+, 50000+, 0–99) |

The framework separates the three failure sources on every task it was run on.
That is the precondition for the claims below, not one of them.

## 2. Hypotheses

**H1 — a learned controller can beat static heuristics at equal budget.**
Partly. On the recall task every learned controller beats every recency
heuristic by a wide, paired-significant margin (+0.24 to +0.49 over FIFO), but
none beats the best hand-written rule: imitation matches salience within the
minimum detectable effect at 10–25% and trails it by 0.02–0.06 at 5%;
imitation plus PPO is ahead by +0.018 [+0.014, +0.023] at 25% and behind at
5% (Exp. 4). On the sequential task a learned controller reaches 1.0 at every
budget where FIFO reaches 0.03–0.89 and salience 0.00–0.40 (Exp. 5b), and a
policy trained on the other task does not transfer that. The honest statement
is: *learned controllers reach the best fixed heuristic on the task they were
trained on and are not fooled by the task where that heuristic fails; they did
not beat the best heuristic on its own task.* See §3.

**H2 — long-horizon reward beats immediate reward.** Not supported by this
evidence. With three PPO seeds each, discount 0.995 and discount 0 are within
seed noise on the recall task (Exp. 4). On the workflow task, where a used
token looks like a needed one, the two variants were both unstable and the
immediate-reward run ended far higher (0.992 against 0.404 at 5%, one seed
each; Exp. 5b). The recall task is a
weak test of H2: the immediate signal already identifies the kind of item
worth keeping. See §3.

**H3 — the advantage of good memory management grows with horizon, tighter
budgets and longer dependencies.** Supported, for the *room* rather than for
any particular method: at a fixed 800-token budget the gap between the best
heuristic and the exact oracle grows from 0.24 to 0.58 as the horizon goes
from 200 to 2,000 (Exp. 1c); at a fixed budget recency rules fall from 0.50 to
0.04 as the minimum dependency distance grows from 5 to 250 while the oracle
stays at 1.0 (Exp. 1d); and every controller's gap to the oracle grows as the
budget shrinks (Exp. 1).

**H4 — archive + retrieval beats deletion under uncertainty.** Supported on the
recall task with a strong qualifier: FIFO goes from 0.23 to 0.73 at 5% budget
once it may archive and retrieve (Exp. 2), and with perfect retrieval the
oracle reaches 1.0 at 2% where the exact delete-only optimum is 0.87. The
qualifier is that the value is entirely in the retriever: on the workflow task
the same lexical retriever has recall 0.01–0.10 and archiving gives nothing
(Exp. 5), and on LoCoMo its recall is 0.35–0.38 (Exp. 6). The archive was also
unlimited and free.

**H5 — compaction and consolidation improve the success/memory frontier when
memories are redundant.** Conditionally. Compaction with a rewriter that keeps
the detail helps a weak policy a lot (FIFO +0.17 at 10%) and a good one
little (salience +0.00); a rewriter that loses the detail hurts both (Exp. 3).
Consolidation of restated facts gives one or two points, and a naive version
hurts a good policy. The extractive rewriter's notion of "specific" coincides
with the generator's notion of "fact", so the positive result is partly built
in; the negative result (lossy rewriting is attributable and harmful) stands.

**H6 — a lightweight JEV policy trades decision quality for latency.**
Not tested: no credentials. The adapter is ready (INFRA_REPORT.md).

**H7 — policies transfer across budgets, base models and tasks.** Partly
tested. Across budgets: yes, within the training range (the policies were
trained on four budgets and evaluated on the same four; Exp. 4). Across
horizons: the 200-step policies were evaluated at 500 steps (Exp. 4). Across
tasks: the recall-trained policy on the workflow task behaves like FIFO, not
like the salience rule it resembles on its own task (Exp. 5b). Across base
models: not tested; only the scripted reader was used.

## 3. Learned controllers (Experiments 4, 4b, 5b)

What can be said about the RL arm, and why it is less than the hypotheses ask for.

1. **The result is a ceiling, not a failure to learn.** The learned policies
   converge to the same behaviour as the salience rule (removal-set overlap
   0.3–0.7 with salience, 0.00–0.04 with FIFO/LRU) because that is what the
   observable features support: which facts get queried is a random draw given
   the source type. The remaining gap to the oracle at 5% (0.34–0.46) is
   mostly information no causal policy has. A "Bayes-informed" reference
   policy that knows the generator's query probabilities would show how much
   of that gap is closable; it was not built, so "fraction of oracle gap
   closed" has an unknown maximum.
2. **Imitation of a privileged expert is the wrong target once the action
   set has cheap insurance.** With an archive available, the hindsight expert
   deletes what it knows is useless; the learner cannot know it, copies the
   deletion, and loses evidence it could have archived (Exp. 4b: its failures
   are `evicted`, not `archived_not_retrieved`). An untrained retrieval head
   that archives and retrieves indiscriminately does better. Any future
   imitation work should use a cost-aware expert, and any RL work should
   price deletion against archiving in the reward.
3. **PPO numbers are single-run noise unless seeds are pooled.** Spread
   across three seeds is up to 0.13 at horizon 200, 5%; policies drift after
   their best validation point; the best-validated checkpoint is 0.02–0.06
   above the final one. Three seeds is the minimum used here and still small.
4. **The learned policy that beat the oracle (Exp. 5b) beat a blind
   oracle.** On the workflow task the reference-pass oracle cannot see the
   episode after the first restart; the learned policy recovers from
   restarts because it never relied on hindsight. That is a statement about
   robustness to divergence, not about optimality.
5. **Sample efficiency favours imitation by an order of magnitude** (240
   episodes against 2,400) on a task where the expert's labels are
   informative. This will not hold where the expert is privileged in a way
   that matters (point 2).
6. **Cost.** The learned controllers spend 1–7 ms per step against 0.2 ms for
   salience; at equal latency salience wins outright. The RL arm's advantage,
   where it exists, is generality across tasks, not speed.

## 4. Confounds and threats, with what was done about them

Ordered by how much they limit what can be claimed.

1. **The synthetic generator decides the heuristic ranking.** Salience's
   weights (user > observation > tool output; "has identifiers") were set by
   hand before any run, but they match how the generator chooses what gets
   asked about. Experiment 5 shows the same rule failing on a different task.
   *Never* quote a heuristic's rank on the recall task as a property of the
   heuristic.
2. **The oracle gap is not fully closable by any policy without hindsight.**
   Which facts get queried is partly random (a Bernoulli draw per fact), so a
   causal policy cannot know it. "Fraction of the oracle gap closed" therefore
   has an unknown ceiling below 1. A "Bayes-optimal informed prior" reference
   was not built; it would settle what share of the gap is information and
   what share is policy quality.
3. **The task model is perfect.** With the scripted reader, task success is a
   pure function of memory contents. A language model would add reasoning
   failures (attributable, per Exp. 1b) and would also change *which* memory
   states lead to success. Experiment 7 is a smoke test, not evidence.
4. **A fraction budget confounds horizon with absolute budget** (Exp. 1); the
   fixed-token variant (1c) is the one to cite for horizon effects, and it
   still confounds horizon with dependency distance, which 1d isolates.
5. **The oracle is a reference, not always a bound.** `oracle_exact` is
   optimal only for delete-only action sets under a perfect reader.
   `oracle_approx` (furthest next use) is not optimal for unequal item sizes
   (behind exact by up to 0.12 in Exp. 1c). Neither compacts or consolidates.
   On the workflow task, whose stream depends on the agent, the oracle is
   blind after the first divergence and is beaten by a learned policy
   (Exp. 5b) — a fact about the reference, not about optimality.
6. **Access counts are exact only with scripted agents.** LRU and LFU read
   `used_item_ids`, which a scripted agent reports exactly and a language
   model cannot (the `llm` agent approximates it by string containment).
7. **Retrieval is lexical or hashing-based in every experiment but 6c.** The
   hashing embedder is word overlap without BM25's weighting and was worse
   than BM25 everywhere. A dense embedder (`bge-small`) was checked only on
   LoCoMo (6c), where it also trailed BM25 (recall 0.22 against 0.35). The
   retrieval-failure findings (Exp. 2, 5, 6) are therefore findings about
   these retrievers; a stronger retriever would move the archive results.
8. **The archive is free.** No experiment priced archive tokens or retrieval
   compute; `archive_budget` and the `retrieval_cost` reward term exist for
   that.
9. **Statistics.** Episodes are the unit and comparisons are paired, but no
   correction was made for the number of comparisons (hundreds of cells).
   Treat a difference as real only if it is large relative to its minimum
   detectable effect and consistent across budgets and horizons. LoCoMo has
   10 episodes; its close differences are unresolved.
10. **Model-free tokens.** Budgets count words and punctuation, not model
    tokens; the ratio varies by text.
11. **Latency numbers were taken on a saturated machine** and exclude the
    exact oracle's solve time (recorded separately).
12. **Public benchmarks.** LoCoMo and LongMemEval are public; a language
    model may have seen them. This does not affect the scripted evaluations
    reported here and would affect any future LLM-based run.
13. **LongMemEval's salience result is a dataset artefact**: evidence sits in
    short user turns, assistant turns are long, and salience prefers user turns.
14. **Single training seed for most RL configurations.** Three seeds for the
    two PPO variants on the recall task, one for everything else.

## 4b. Defects found by an independent code review, and what they changed

A reviewer with no stake in the results read the harness after the first
experiments had run. Four defects were confirmed and fixed the same day; the
affected cells were rerun (the tables in EXPERIMENTS.md are from the reruns).

| Defect | Runs affected | Fix |
|---|---|---|
| The forced fallback deleted items a controller had *retrieved at that same step* when the retrieval did not fit, so archived evidence was lost permanently. Only the archive-mode oracle and the RL controller triggered it; every heuristic budgets its retrievals. | `oracle_approx` in Exp. 2, 5, 6; RL cells of Exp. 4b | An unaffordable retrieval is returned to the archive before anything is evicted (design decision E16) |
| Action counts mixed units: one per action for controllers, one per item for the fallback, so `forced_share_of_removals` and the actions table were not comparable across controllers | tables only | Everything counts items |
| On the workflow task, hindsight-derived metrics (`requirements_destroyed`, `unnecessary_token_share`, regret rows, imitation labels) kept being computed after the episode had diverged from the reference pass, describing another episode | Exp. 5, 5b columns; `rl_bc_workflow` labels | Divergence is detected by content hash; those metrics are `null` from then on; imitation labels after divergence are dropped; `rl_bc_workflow` was retrained |
| `invalid_action` was assigned whenever *any* controller action at the step had been rejected, related to the lost item or not | no recorded run (no rejected actions occurred) | Requires a rejected action targeting the lost item, or a step on which every controller action was rejected |

Two smaller points were also fixed: `retrieved_but_ignored` now means
retrieved at that step, and the query observation no longer counts as an
"unnecessary" token. The reviewer confirmed, by brute force on 40 small
instances, that the exact integer program is optimal for the delete-only
setting; that the RL log-probabilities are reproduced exactly; that seeds are
disjoint; and that no controller can reach ground truth.

## 5. What would change the conclusions

- A task model that is not perfect: rerun Experiments 1–5 with the `llm`
  agent on a GPU host. Attribution will say how much of the change is the model.
- A generator whose query distribution is not predictable from source type:
  set `sources.*.query_prob` equal across sources and rerun Experiment 1;
  salience's advantage should shrink to its "has identifiers" component.
- A priced archive: set `memory.archive_budget` and a non-zero
  `retrieval_cost` weight and rerun Experiment 2.
- More PPO seeds and a hyper-parameter sweep before concluding anything about
  RL versus imitation.

## 6. Reporting rules used

Every number quoted in EXPERIMENTS.md comes from `summary.json` or
`report.json` of the run it names; the tables were generated from those files,
not typed. Differences are paired by seed; intervals are 95% bootstrap over
episodes; the minimum detectable effect is reported next to every difference.
No hidden model reasoning was logged or used anywhere.

## 7. Provenance caveat

All runs were made from an uncommitted working tree on branch
`framework-rebuild`; `metadata.json` records commit `d215c8f` with
`dirty: true`. The code changed while experiments ran. Changes that touched
behaviour after a run had used it:

| Change | Runs made before it | Effect on those runs |
|---|---|---|
| Synthetic env accepts an answer that *contains* the value (for language models) | Exp. 1, 1b | none: the scripted reader answers with the bare value or `unknown` |
| `evidence_complete_rate` metric added | Exp. 1, 1b | the field is absent from those episode rows |
| `oracle_solve_s` recorded | Exp. 1–1d, 2, 3 (partly), 5 | the field is absent from those rows |
| Consolidation `ratio` option | Exp. 3's first 48 cells | none: the option is off unless configured |
| Trainer: validation seeds 50000+ and `policy_best.pt` | `rl_bc*` trainings | those trainings validated on seeds 0–19, which overlap the test seeds; they selected nothing on it (final policy used), so no selection bias, but the logged learning-curve numbers for `rl_bc*` are not held-out |

Before quoting any number in the thesis: commit, then rerun the sweeps with
one command each (they take about two hours in total on this machine). The
configs and seeds are unchanged, so the numbers should reproduce.
