# memctl: research guide

How to use the framework to run the thesis experiments. Results are in
[EXPERIMENTS.md](EXPERIMENTS.md), what can and cannot run here is in
[INFRA_REPORT.md](INFRA_REPORT.md), and what the results do and do not show is
in [SCIENTIFIC_VALIDITY_REPORT.md](SCIENTIFIC_VALIDITY_REPORT.md). The design
record is [docs/superpowers/specs/2026-09-30-memctl-framework-design.md](docs/superpowers/specs/2026-09-30-memctl-framework-design.md).

## The question

At the same active-memory budget and compute budget, do learned
memory-management policies outperform fixed heuristics on long-running tasks?

The controller manages memory and nothing else. A frozen task model does the
task and only ever reads memory.

```
environment ──observation──▶ memory state ◀──actions── memory controller
     ▲                           │                        (heuristic | JEV | RL |
     │                           ▼                         prompted LLM | oracle)
     └────task action──── frozen task model
```

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -c constraints.txt -e ".[dev]"
.venv/bin/pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu   # RL controller; CPU is enough
.venv/bin/python -m pytest -q          # 153 tests, about 80 seconds
.venv/bin/python -m memctl.doctor      # what can run on this machine
```

The Docker image (`docker compose run --rm memctl`) still works and carries the
GPU stack for local Hugging Face models.

## Running things

| To do this | Run |
|---|---|
| One experiment | `python -m memctl.run --config configs/stage1_smoke.yaml` |
| Resume an interrupted one | `python -m memctl.run --resume runs/<experiment_id>` |
| A grid of experiments | `python -m memctl.sweep --config configs/sweeps/exp1_delete_only.yaml` |
| Tables and figures for a grid | `python -m memctl.analysis.report runs/exp1_delete_only` |
| Train the RL controller | `python -m memctl.rl.train --config configs/rl/rl_bc.yaml` |
| Learning curve | `python -m memctl.analysis.plots runs/rl_bc` |
| Environment check | `python -m memctl.doctor` |

Running a sweep again resumes it: cells have stable folder names and finished
episodes are skipped.

## One step of an episode

1. The observation is added to ACTIVE. Memory may now be over budget.
2. The controller decides. The engine applies each action or rejects it.
3. If ACTIVE is still over budget, the harness first returns any item
   retrieved at this step to the archive (an unaffordable retrieval is undone,
   not turned into a deletion), then evicts oldest-first. Both are logged with
   source `harness`, penalised in the reward and reported as
   `forced_evictions`, `forced_fallback_rate` and `forced_share_of_removals`.
4. If the observation needs a response, the agent reads ACTIVE and answers.
5. The environment scores the answer and gives the next observation.

The harness never retrieves, compacts or consolidates. If it is not in step 3,
a controller did it.

## The memory model

- **Tiers:** `ACTIVE` (the agent reads it; limited by the budget), `ARCHIVE`
  (kept, unreadable until retrieved), `DELETED` (gone for agent and controller,
  kept in `items.jsonl` for evaluation).
- **Items are immutable.** Compaction and consolidation create new items linked
  by `derived_from_ids`; the sources get `superseded_by`.
- **Tokens** are counted model-free (words plus punctuation).
- **Budget:** `memory.budget: {fraction: 0.25}` is a share of the episode's
  uncompressed history (measured in a reference pass); `{tokens: 800}` is absolute.

## Operations

| Operation | Effect | Status |
|---|---|---|
| `KEEP`, `NO_OP` | nothing | implemented |
| `EVICT` | ACTIVE → DELETED | implemented |
| `MOVE_TO_ARCHIVE` | ACTIVE → ARCHIVE | implemented |
| `RETRIEVE_FROM_ARCHIVE` | ARCHIVE → ACTIVE, by id | implemented |
| `COMPACT` | replace an item with a shorter one; the source is deleted | implemented |
| `COMPACT_AND_ARCHIVE` | short version in ACTIVE, full item to ARCHIVE | implemented |
| `CONSOLIDATE` | merge two or more items into one | implemented |
| `PROMOTE`, `DEMOTE`, `PIN`, `UNPIN`, `UPDATE`, `SUPERSEDE` | — | in the schema; rejected as "no handler yet" |

`memory.allowed_operations` sets what an experiment allows. **Implemented is not
the same as allowed**: an operation outside the list is rejected and counted as
an invalid action, which is how delete-only, archive-only and no-compression
ablations are run on one code path.

To add an operation: one function in `memctl/memory/engine.py` decorated with
`@handler(Operation.X)`. Nothing else changes.

## Controllers

All implement `MemoryController` (`memctl/controllers/base.py`):
`reset`, `observe`, `decide(memory, task) -> list[MemoryAction]`, `update`,
`save`, `load`, `decision_info`.

| Config name | What it is |
|---|---|
| `no_controller` | never acts; the forced fallback makes it FIFO truncation. **The no-memory-controller arm.** |
| `full_context` | never acts, unlimited budget. The infinite-context reference. |
| `fifo`, `lru`, `lfu`, `random`, `age_decay`, `similarity`, `salience` | priority heuristics: remove the lowest-priority items when over budget |
| `archive_everything` | RAG-style: only the newest items stay in ACTIVE |
| `oracle` | hindsight; `method: approx` or `exact`. Analysis only. |
| `prompted_llm` | a language model that replies with a JSON list of actions |
| `jev` | TypeSafe's Jev model behind an adapter; needs `JEV_API_KEY` |
| `rl` | a learned item scorer; `checkpoint:` loads a trained policy |

Every priority heuristic takes the same options, which is how the delete-only,
archive-only, compression-only and retrieval variants are built:

```yaml
controller:
  name: lru
  label: lru_archive_retrieve            # the name used in reports
  removal: [COMPACT, MOVE_TO_ARCHIVE]    # tried in order; default [EVICT]
  retrieve: {top_k: 3, method: lexical}  # lexical | embedding; omit for no retrieval
  consolidate: {}                        # omit for no consolidation
```

To add a controller: one class, one line in `memctl/controllers/__init__.py`.

## Environments

All implement `TaskEnvironment` (`memctl/envs/base.py`): `reset`, `step`,
`get_observation`, `is_done`, `get_reward`, `task_success`, and optionally
`get_ground_truth_dependencies`, `snapshot`, `restore`.

| Config name | Phase | What it tests |
|---|---|---|
| `synthetic_recall` | 0 | Facts, distractors and queries with exact ground truth. Knobs for horizon, dependency distance, reuse, updates, restatement, verbosity. |
| `locomo`, `longmemeval` | 1 | Retention, archival, retrieval and evidence preservation under a budget. **Not long-horizon control**: all questions come after the history. |
| `workflow` | 2 | Interleaved multi-stage jobs. Forgetting a token forces a restart, which changes the rest of the episode. |

## Task models

| Config name | What it is |
|---|---|
| `scripted_reader`, `scripted_tool_agent` | succeed exactly when the needed text is in ACTIVE; `noise` injects reasoning failures at a known rate |
| `llm` | a language model (`model: {backend: hf | openai | stub, ...}`) |
| `null` | never answers; for measuring evidence retention without a task model |

## Hindsight

`memctl/hindsight/` is used by the oracle, regret, attribution and imitation,
and by nothing else. Ordinary controllers never import it (a test enforces this).

| Module | What it gives |
|---|---|
| `collect.py` | `Hindsight` from a reference pass: item sizes, arrival steps, when each item is needed |
| `evidence.py` | where each needed fact is now, traced through compaction and consolidation |
| `exact.py` | the exact delete-only plan (integer program) |
| `trace.py` | evidence without benchmark labels: string tracing and tool-value tracing |
| `ablation.py` | counterfactual ablation: replay without one item |
| `regret.py` | requirements destroyed per decision; oracle-continuation regret by replay |

## What a run writes

```
runs/<experiment_id>/
  config.yaml  metadata.json  summary.json
  episodes.jsonl          one row per finished episode (the resume marker)
  failures.jsonl          one row per failed query, with its cause
  regret.jsonl            one row per evidence requirement a decision destroyed
  steps.jsonl  memory_actions.jsonl  rewards.jsonl  retrieval_events.jsonl  items.jsonl
                          per-step detail, for the first logging.detail_episodes episodes
  checkpoints/  plots/
```

`metadata.json` holds the git commit (and whether the tree was dirty), CPU, GPU,
every package version and the model identifiers. No hidden reasoning is logged:
only observable model outputs, structured decisions and metadata.

## Metric definitions worth knowing

- `retrieval_precision`: share of retrieved items that carried evidence for the
  query answered *at that step*. A retrieval made for a later query counts as a
  miss. `retrieval_recall`: share of evidence that was only in the archive and
  was retrieved in time.
- `needed_hit_rate`: share of evidence requirements met in ACTIVE when the
  agent read; `evidence_complete_rate`: share of queries with *all* their
  evidence in ACTIVE (a perfect reader's accuracy).
- `action_counts` count items, not actions: one action with five targets
  counts five.
- `requirements_destroyed` and `unnecessary_token_share` come from hindsight
  and are `null` once an episode has diverged from its reference pass
  (`hindsight_diverged_at`), which happens on the workflow task after the first
  restart. The oracle is blind from that step on, too.
- Controller latency excludes the exact oracle's plan computation, which is
  recorded separately as `oracle_solve_s`.

## Failure attribution

Every failed query gets one label: `evicted` (with cause `controller` or
`harness`), `compression_lost_detail`, `consolidation_incorrect`,
`archived_not_retrieved`, `retrieved_but_ignored`, `task_model_reasoning`,
`invalid_action`, `unknown`. Read this before concluding that a controller is
bad: the failure may belong to the retriever or to the task model.

## Reward

Terms are computed and logged every step whatever their weight, so a run can be
re-scored without being re-run:

```yaml
reward:
  weights: {task_reward: 1.0, forced_fallback: -0.1, invalid_action: -0.1, active_memory_cost: -0.01}
```

Available terms: `task_reward`, `active_memory_cost`, `retrieval_cost`,
`controller_compute_cost`, `invalid_action`, `forced_fallback`,
`memory_hit_rate`, `retrieval_success`, `hindsight_eviction_regret`. Add one
with `@term("name")` in `memctl/rewards.py`.

## Rules for running an experiment

1. **Never compare at a single budget.** Sweep budgets and read the frontier.
2. **Same seeds for every controller.** Sweeps do this; comparisons are paired.
3. **Read the minimum detectable effect** in the report before reading a difference.
4. **Read the failure attribution** before blaming a controller.
5. **Match action sets.** A controller allowed to archive is not comparable with
   one that may only delete, unless that is the comparison.
6. **The oracle is a reference, not always a bound.** `oracle_exact` is optimal
   for delete-only action sets. `oracle_approx` is not optimal when item sizes
   differ, and neither compacts or consolidates.
