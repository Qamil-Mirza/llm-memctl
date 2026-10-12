# memctl: research guide

How to use the framework to run the thesis experiments. Results are in
[EXPERIMENTS.md](EXPERIMENTS.md), what can and cannot run here is in
[INFRA_REPORT.md](INFRA_REPORT.md), and what the results do and do not show is
in [SCIENTIFIC_VALIDITY_REPORT.md](docs/archive/SCIENTIFIC_VALIDITY_REPORT.md). The design
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
.venv/bin/python -m pytest -q          # 165 tests
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
| …and probe an LLM server | `python -m memctl.doctor --base-url http://localhost:8000/v1` |
| A grid in parallel, or rerun named cells | `python -m memctl.sweep --config <sweep>.yaml --workers N --only <label>,<label>` |
| Pool training seeds (`_s<k>` labels) | `python -m memctl.analysis.seeds runs/<sweep> [--metric task_success]` |
| Score a sweep under a priced archive | `python -m memctl.analysis.priced runs/<sweep> --price 0.05 [--json]` |
| What a policy does with needed / never-needed items | `python -m memctl.rl.probe runs/<training>/checkpoints/policy.pt [--episodes 20] [--fractions 0.02 0.05]` |
| Fit P(needed again) for the plug-in archive rule | `python -m memctl.rl.needed --config configs/rl/d5/needed_s0.yaml` |

Running a sweep again resumes it: cells have stable folder names and finished
episodes are skipped.

## Training the RL controller

`python -m memctl.rl.train --config <x>.yaml` takes an ordinary experiment
config with `controller: {name: rl, ...}` plus a `training` section (full
reference in the docstring of `memctl/rl/train.py`):

```yaml
training:
  seed_offset: 100000                 # training seeds; validation uses eval.seed_offset; experiments use 0, 1, ...
  budget_fractions: [0.02, 0.05]      # optional: a budget drawn per episode
  phases:                             # run in order on one policy
    - {algorithm: bc, iterations: 8, episodes: 16, epochs: 4, lr: 0.003, dagger: true}
    - {algorithm: grpo, iterations: 40, episodes: 24, group: 8, epochs: 4, lr: 0.0003}
  eval: {every: 4, episodes: 20, seed_offset: 50000}
  expert: {kind: regret}              # oracle (default) | regret; archive_cost: p prices the archive
  tasks:                              # optional: one policy on several tasks, one episode each in turn
    - {label: recall, env: {name: synthetic_recall}}
    - {label: workflow, env: {name: workflow}, agent: {name: scripted_tool_agent}}
```

Algorithms: `bc` (imitation of the expert, DAgger by default), `cost`
(cost-sensitive imitation: expected regret along the expert's removals),
`grpo` (group-relative advantages over `group` samples of one episode, no value
network), `ppo`, `reinforce`. The expert (`memctl/rl/expert.py`): `oracle`
labels the hindsight argmin; `regret` accepts every (item, operation) pair of
least hindsight regret. The output folder holds `checkpoints/policy.pt` (last)
and `checkpoints/policy_best.pt` (best validation success).

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
| `rl` | a learned item scorer; `checkpoint:` loads a trained policy; with `needed_model:` (from `memctl.rl.needed`) and `archive_price:` it archives an item iff P(needed) × `need_value` (default 1) > `archive_price` |

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

The `openai` backend talks to any OpenAI-compatible chat-completions server
(vLLM, Ollama `/v1`, llama.cpp), greedy decoding:

```yaml
agent:
  name: llm
  model: {backend: openai, name: qwen2.5-7b-instruct, base_url: http://localhost:8000/v1,
          api_key_env: MY_KEY, timeout_s: 120, cache_dir: cache/generations_x}
```

`api_key_env` names an environment variable (optional); `timeout_s` defaults
to 300; `cache_dir` saves every generation to disk so reruns cost nothing.
Retries on transient failures (5xx, timeouts, empty bodies; 5 retries with
backoff) and a `User-Agent` header (some proxies reject urllib's) are built in.
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

## §33 Jacobian lens (how to run)

Prompts come from the §23 cache; no reader is called. Local development on the 0.5B:

    .venv/bin/python scripts/jacobian_lens.py --reader qwen05b --records-from qwen3b --folds 0 --limit 10 --device cpu --dtype float32 --ablate
    .venv/bin/python scripts/exp33_report.py --small qwen05b --large qwen05b --records-from qwen3b --folds 0

Paid run (one A40, user-approved): `bash scripts/exp33_pod.sh`, then copy `runs/exp33_jacobian/` back.
What to copy to the pod (next to a checkout of this branch; `runpodctl send` or `rsync` both work), all
checked by the script's preflight, which exits 3 naming the first missing path before installing anything:

- `data/longmemeval/longmemeval_s_cleaned.json`
- `runs/exp23_verdicts.json`
- `runs/exp23_<reader>_f<fold>/fixed8__fraction0.05/items.jsonl.gz`, `steps.jsonl.gz` and `episodes.jsonl`
  for reader `qwen3b` and `qwen7b` and fold 0 to 4 (30 files)
- `cache/generations_exp23_qwen3b/` and `cache/generations_exp23_qwen7b/` (whole folders)

The script pins `transformers==5.17.0` (the local `.venv` version) and prints the image's torch version.
Report on development folds only until the pre-registration is committed:

    .venv/bin/python scripts/exp33_report.py --small qwen3b --large qwen7b --folds 0,1 --variant l2
    .venv/bin/python scripts/exp33_report.py --small qwen3b --large qwen7b --folds 2,3,4 --variant l2 --confirm

Local 0.5B CPU reference: 18.2 s/question, peak RSS 6,681 MB (10-question stub, fold 0, float32, with `--ablate`, `logits_to_keep`; `runs/exp33_jacobian/_stub2_qwen05b/_timing.json`; before `logits_to_keep`: 22.6 s and 9,364 MB in `_stub_qwen05b`).

## §34 J-lens on small readers (how to run)

Pin: `external/jacobian-lens@581d398 (jlens 0.1.0, Apache-2.0)`. Locally it is installed into `.venv`; on
the pod the runbook runs `pip install -e external/jacobian-lens`. The fit corpus is
`data/jlens_corpus/wikitext-103-raw-v1-test.parquet`. Prompts come from the §23 cache; no reader generates.

Local development chain on the 0.5B (CPU, float32; a 4-prompt dev lens, then 10 questions of fold 0):

    .venv/bin/python scripts/jlens_fit.py --reader qwen05b --n-prompts 4 --min-prompts 2 --max-minutes 30 --chunk 2 --device cpu --dtype float32
    .venv/bin/python scripts/jlens_read.py --reader qwen05b --records-from qwen3b --lens runs/exp34_jlens/lenses/qwen05b.pt --folds 0 --limit 10 --device cpu --dtype float32
    .venv/bin/python scripts/jlens_eval.py --reader qwen05b --lens runs/exp34_jlens/lenses/qwen05b.pt --sets multihop,association --limit 20 --device cpu --dtype float32
    .venv/bin/python scripts/exp34_report.py --small qwen3b --large qwen7b --records-from qwen3b --folds 0 --root runs/exp34_jlens/local_report_root

The report expects `<root>/<reader>/`, `<root>/eval/<reader>.json` and `<root>/lenses/<reader>.json` for both
readers, so the local root points both names at the 0.5B outputs. To recreate it:

    mkdir -p runs/exp34_jlens/local_report_root/eval runs/exp34_jlens/local_report_root/lenses
    cd runs/exp34_jlens/local_report_root
    ln -s ../qwen05b qwen3b && ln -s ../qwen05b qwen7b
    ln -s ../../eval/qwen05b.json eval/qwen3b.json && ln -s ../../eval/qwen05b.json eval/qwen7b.json
    ln -s ../../lenses/qwen05b.json lenses/qwen3b.json && ln -s ../../lenses/qwen05b.json lenses/qwen7b.json

Paid run (one GPU, user-approved quote first): `bash scripts/exp34_pod.sh [max_minutes_7b=90] [max_minutes_3b=40] [n_prompts=128]`.
It runs, in order: install (pins `transformers==5.17.0`, the local `.venv` version, plus `-e external/jacobian-lens`,
and prints torch, transformers and the jlens commit); fit `qwen3b` then `qwen7b` (`--min-prompts 32 --dim-batch 32`);
read both readers on folds 0 to 4; lens-quality eval for both; the community 7B cross-check; then
`bash scripts/exp33_pod.sh` (the §33 secondary). Each stage prints its elapsed time (also in
`runs/exp34_jlens/pod.log`). Rerun the same command to continue: a fit resumes from its checkpoint (a complete
lens is not refitted), reads skip finished questions, an eval newer than its lens is skipped, and the community
lens is downloaded once. Exit codes: 3 = preflight found a missing input (named) before installing anything;
5 = a fitted lens has `below_minimum: true` in its sidecar, so nothing is read with it (rerun with a longer box
to resume the fit); 6 (from `jlens_fit.py`) = the settings or corpus differ from an existing checkpoint.

The community cross-check downloads `andyx10/jacobian-lens-qwen2.5-7b-instruct` / `qwen7b_lens.pt` with
`huggingface_hub.hf_hub_download`, copies it (never moves it out of the HF cache) to
`runs/exp34_jlens/lenses/community_qwen7b.pt`, and reads the development folds 0,1 only into
`runs/exp34_jlens/community_qwen7b/qwen7b/`, so the report takes `--community-root runs/exp34_jlens/community_qwen7b`.

What to copy to the pod (next to a checkout of this branch), all checked by the preflight:

- the §33 list above: `data/longmemeval/longmemeval_s_cleaned.json`, `runs/exp23_verdicts.json`, the 30
  `runs/exp23_<reader>_f<fold>/fixed8__fraction0.05/` files, and `cache/generations_exp23_qwen3b/` and
  `cache/generations_exp23_qwen7b/` (whole folders)
- `data/jlens_corpus/wikitext-103-raw-v1-test.parquet`
- `external/jacobian-lens/` (whole folder; the preflight checks `external/jacobian-lens/pyproject.toml`)

Copy back `runs/exp34_jlens/` and `runs/exp33_jacobian/`; check each sidecar's `n_prompts` and `below_minimum`
and any `_failed.json`.

Fold rule: folds 0,1 are development (lens gate, k, the band and thresholds are fixed there, then the
pre-registration is committed); folds 2,3,4 are confirmation, read only by the report with `--confirm` after
that commit:

    .venv/bin/python scripts/exp34_report.py --small qwen3b --large qwen7b --folds 0,1 --community-root runs/exp34_jlens/community_qwen7b
    .venv/bin/python scripts/exp34_report.py --small qwen3b --large qwen7b --folds 2,3,4 --confirm
