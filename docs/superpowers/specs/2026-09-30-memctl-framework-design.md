# memctl framework design

Date: 2026-09-30. Approved in conversation the same day, with the five decisions recorded
in section 2. This is the design record; `README_RESEARCH.md` is the user guide and
`EXPERIMENTS.md` holds results. Later additions (regret expert, GRPO, the `cost`
algorithm, joint `training.tasks`, the plug-in archive rule, the `openai` backend) are
documented in `README_RESEARCH.md` and `EXPERIMENTS.md`, not here.

## 1. Purpose

A research framework for the question: **do learned memory-management policies beat fixed
heuristics at the same memory and compute budget?** The controller manages memory only; a
frozen task model does the task. Controllers (heuristic, JEV, RL, prompted LLM, oracle) are
interchangeable behind one interface, and so are task environments and task models.

First milestone: *at the same active-memory budget, do different memory-management policies
produce measurably different long-horizon task performance?*

## 2. Decisions taken by the user

1. **Stage 1 task model is a scripted reader** with a configurable noise knob. The `Agent`
   interface stays modular so an LLM can replace it without touching the harness.
2. **The full initial action set is implemented** (KEEP, EVICT, MOVE_TO_ARCHIVE,
   RETRIEVE_FROM_ARCHIVE, COMPACT, COMPACT_AND_ARCHIVE, CONSOLIDATE, NO_OP); PROMOTE,
   DEMOTE, PIN, UNPIN, UPDATE, SUPERSEDE exist in the enum without handlers. **Which
   operations an experiment allows is a config field.** The first experiment allows only
   KEEP, EVICT, NO_OP.
3. **Over budget, the harness evicts oldest-first.** Every forced eviction is logged,
   separated from controller actions, penalised in the reward and reported as a metric.
4. **The hindsight oracle is part of Stage 1**: an approximate one (never-needed first, then
   furthest next use) and an exact one where solvable, plus oracle-gap metrics.
5. **No automatic retrieval anywhere in the harness.** Archive search is a controller
   component; every retrieval is a logged controller action.

## 3. Engineering defaults chosen (not decided by the user)

| # | Default | Reason |
|---|---|---|
| E1 | `MemoryItem` is frozen; the content of an id never changes. COMPACT and CONSOLIDATE create new items with `derived_from_ids`; sources get `superseded_by`. | Stable ids, full lineage in the logs, views are safe to hand to controllers. |
| E2 | The environment owns observation ids; a memory item ingested from an observation keeps that id. | Ground-truth dependencies can name items without the env knowing about memory. |
| E3 | Step order: ingest observation → `controller.decide` → engine applies actions → harness enforces budget → agent reads ACTIVE and acts → env steps. | The controller may act on the new item; the budget holds whenever the agent reads. |
| E4 | Actions are validated and applied one at a time. An invalid action is rejected, logged and leaves the state unchanged. | A learning policy must survive its own mistakes. |
| E5 | The controller runs every step. | Proactive archiving and consolidation need it; its compute is counted. |
| E6 | RETRIEVE_FROM_ARCHIVE moves named items ARCHIVE → ACTIVE, where they stay until removed. | One mechanism; retrieved items pay the budget like any other. |
| E7 | COMPACT replaces the source with a shorter item (source DELETED). COMPACT_AND_ARCHIVE leaves the short item in ACTIVE and moves the full source to ARCHIVE. | The second is "keep a stub, keep the original recoverable". |
| E8 | CONSOLIDATE merges two or more ACTIVE items into one; sources are deleted, or archived with `parameters.archive_sources`. | |
| E9 | Evidence is a list of requirements; each names alternative source items and a `needle` string. A requirement is met by a *carrier*: a source item, or an item derived from one whose content still contains the needle. | Makes "compression lost the detail" and "consolidation was wrong" decidable by string tracing. |
| E10 | A budget given as a fraction is a share of the tokens of the *reference pass*: the same episode run with unlimited budget and no controller. | "100% of uncompressed history" for any environment. |
| E11 | Hindsight (item sizes, arrival steps, need times) is collected from the reference pass. Exact only when the observation stream does not depend on memory decisions; labelled approximate otherwise. | Works for every environment without a privileged preview API. |
| E12 | Tokens are counted model-free (words plus punctuation), as in Phase 2. | Budgets mean the same thing for every model. |
| E13 | Per-step logs are written in full for the first `logging.detail_episodes` episodes of a run; every episode gets an episode row, failure rows and action counts. | A full grid would otherwise write millions of step rows. |
| E14 | The task instruction is outside the memory budget; the current observation is handed to the agent directly as well as being ingested. | The agent can always read the question it is asked. |
| E15 | "Access" means the task model reported using the item. | Exact for scripted agents; an approximation must be chosen for LLM agents. |
| E16 | The fallback undoes this step's unaffordable retrievals (back to ARCHIVE) before evicting oldest-first. | Added after review: otherwise a retrieval that did not fit was turned into a permanent deletion of archived evidence. |
| E17 | Hindsight-derived metrics are withdrawn from the step at which an episode's observation stream stops matching the reference pass. | Added after review: on the workflow task the reference ids denote different observations after a restart. |

## 4. Modules and dependency rule

```
memory/items, memory/actions                     (no imports from memctl)
memory/state, memory/compress, memory/engine
embed, retrieval, features                       (read memory only)
controllers/base   envs/base   agents/base   rewards   attribution
controllers/*      envs/*      agents/*      hindsight/*
harness/episode  →  harness/runner  ←  config, runlog, sysinfo
analysis/*                                        (reads run folders only)
```

- Controllers import `memory`, `embed`, `retrieval`, `features` and `controllers.base`.
  They never import `envs`, `agents`, `harness` or `analysis`.
- Environments never import controllers or the harness.
- `analysis` reads files and imports nothing from `harness`, `controllers`, `envs`, `agents`.
- `tests/test_import_boundaries.py` enforces these.

## 5. Interfaces

```python
class MemoryController:
    def reset(self, episode: EpisodeInfo) -> None
    def observe(self, event: StepEvent) -> None
    def decide(self, memory: MemoryView, task: TaskState) -> list[MemoryAction]
    def update(self, feedback: Feedback) -> None
    def save(self, path) / load(self, path)
    def decision_info(self) -> dict        # scores, probabilities, latency, token counts

class TaskEnvironment:
    def reset(self, seed) -> Observation
    def step(self, agent_action) -> StepResult
    def get_observation(self) / is_done(self) / get_reward(self)
    def get_ground_truth_dependencies(self) -> list[Dependency]   # optional
    def snapshot(self) / restore(self, snapshot)                  # optional

class Agent:
    def reset(self, seed) -> None
    def act(self, memory: MemoryView, observation, task: TaskState) -> AgentStep
```

Controllers never receive dependencies. The oracle receives a `Hindsight` object through
`receive_hindsight`, which the runner calls only for controllers with `uses_hindsight`.

## 6. Failure attribution

One label per failed query, most severe first:

`evicted` · `compression_lost_detail` · `consolidation_incorrect` ·
`archived_not_retrieved` · `retrieved_but_ignored` · `task_model_reasoning` ·
`invalid_action` · `unknown`.

`evicted` carries a cause: `controller` or `harness`. `invalid_action` is used when
the needed item was removed by the fallback on a step where the controller emitted a
rejected action.

## 7. Build order

1. Memory core, synthetic environment, scripted reader, heuristics, oracle, harness,
   logging. Experiment 1 (delete-only) and its confound check.
2. Archive and retrieval, compaction and consolidation, regret. Experiments 2 and 3.
3. JEV adapter and prompted-LLM controller.
4. RL controller: imitation from the oracle, then policy gradient. Experiment 4.
5. LoCoMo, LongMemEval and the sequential workflow environment. Experiment 5.
6. Analysis, plots, `memctl.doctor`, and the four reports.

Each component ends with a validation status: PASSED, PARTIAL, BLOCKED or FAILED, recorded
in `INFRA_REPORT.md`.
