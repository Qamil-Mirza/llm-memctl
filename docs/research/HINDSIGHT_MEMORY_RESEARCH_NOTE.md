# Hindsight-Supervised Memory Management for Long-Running Agents: a Research Note on `memctl`

*Technical memo, 2026-09-30. Reconstructed from the repository at commit `4fe0f7c` (`main`), its run artifacts under `runs/`, and the four project reports (`README_RESEARCH.md`, `EXPERIMENTS.md`, `INFRA_REPORT.md`, `SCIENTIFIC_VALIDITY_REPORT.md`). Every quantitative claim names the run folder or file it was read from. Literature context was gathered by a scholarly search on the same day and is confined to §16–18. Status labels used throughout: **implemented**, **experimentally validated**, **partially supported**, **unsupported**, **blocked by infrastructure**.*

---

## 1. Executive summary

**Problem.** A long-running agent accumulates observations, tool outputs and its own actions; its task model reads a bounded active context. Something must decide, before the need arises, what to keep in context, what to park in an archive that a retriever may or may not find later, what to destroy, and what to rewrite or merge. Because the cost of each decision arrives later and the decisions compete for one budget, this is a sequential decision problem with a credit-assignment problem attached, not a retrieval problem.

**Hypothesis.** The project's stated question is whether learned memory-management policies beat fixed heuristics at the same active-memory and compute budget. Seven sub-hypotheses (H1–H7, §11) cover learned vs heuristic, delayed vs immediate reward, growth of the advantage with horizon and dependency length, archive+retrieval vs deletion, compaction/consolidation, a lightweight JEV policy, and transfer.

**System built (implemented, 153 passing tests).** `memctl` is a framework with three tiers (ACTIVE, ARCHIVE, DELETED), an eight-operation engine behind a configurable allowed set, one controller interface behind which fourteen controllers run (no-controller, FIFO, LRU, LFU, random, age-decay, similarity, salience, archive-everything, approximate and exact-ILP hindsight oracles, a prompted LLM, a JEV adapter, and a small learned policy trained by DAgger imitation of the oracle and by PPO), four environments (a synthetic recall task with exact ground truth, a sequential workflow task where forgetting forces job restarts, LoCoMo and LongMemEval adapters), scripted and LLM task models, a per-failure attribution scheme with seven labels, hindsight regret tracking, paired-seed sweeps with bootstrap intervals and minimum detectable effects, and run folders stamped with git state.

**Main findings (experimentally validated on scripted task models).** Memory policy changes task success by up to 0.56 (paired oracle-over-FIFO difference at a 10 % budget, horizon 500; `runs/exp1_delete_only/report.json`), and by 0.59 between non-oracle controllers in the same cell. Attribution recovers an injected 20.8 % reasoning-failure rate as 19.9–20.8 % (`runs/exp1b_attribution_check`). Archive plus BM25 retrieval lifts FIFO from 0.233 to 0.734 at a 5 % budget and every remaining failure is a retrieval miss (`runs/exp2_archive_retrieval`). The best heuristic on the recall task (salience) is the worst on the sequential task and FIFO the best (`runs/exp5_workflow`). Learned controllers beat every recency rule by +0.24 to +0.49, match the best content rule within the minimum detectable effect at 10–25 % budgets, and do not beat it beyond +0.018 at one budget (`runs/exp4_rl_eval`). On the sequential task an imitation policy trained on that task completes every job at every budget, where FIFO completes 3 % at 5 % and the hindsight oracle 81 % (`runs/exp5b_workflow_rl`).

**Strongest positive result.** `rl_bc_workflow`: task success 1.000 at a 5 % budget with 1.2 restarts per episode, against 0.028 for FIFO (64 restarts) and 0.808 for the approximate oracle (25 restarts), from 240 imitation episodes.

**Strongest negative result.** Imitating the hindsight expert with archiving allowed is the wrong target: `rl_bc_archive` scores 0.521 at 5 % and fails 924 times by `evicted`, while the delete-only policy run with an *untrained* retrieval head scores 0.849 by archiving everything and retrieving liberally (`runs/exp4b_rl_archive_eval`). H2 (delayed reward helps) is not supported by any cell.

**Most important confound.** The synthetic generator decides the heuristic ranking: the salience rule's hand-set weights match the generator's per-source query probabilities, and the extractive rewriter's notion of "specific" matches the generator's facts. A second confound of the same rank: every reported number uses a perfect scripted task model, a lexical retriever and a free, unlimited archive. And every run is stamped `dirty: true` at commit `d215c8f`; nothing here is a final thesis number until rerun from a clean commit.

**Current best interpretation.** Under a budget, memory policy matters materially; archive plus retrieval matters more than fine eviction differences and then the retriever decides; no fixed rule is good across task structures, which motivates learning; hindsight imitation reaches the best rule cheaply but cannot exceed it where future use is unpredictable from observable features (an instance of the imitation gap, §16), and it teaches the wrong action once cheap insurance (archiving) exists; optimal eviction is not optimal memory management.

**Most promising next direction.** Replace hindsight deletion labels with archive-aware counterfactual regret labels (AggreVaTe/LOLS-style, §17 D1), train one controller across task generators against a Bayes-informed ceiling (D2), and move the sequential claim onto a language-model task model and a published trajectory-dependent benchmark (D8). The literature search (§16) finds every component of this project published somewhere in 2026 (LRE, MemCon, EMBER, MemAudit, Shen 2026, Parrot/ForesightKV) and no system that combines them; the defensible contribution is the conjunction plus two results no neighbour reports: the wrong-teacher effect under archival, and oracle blindness after trajectory divergence.

## 2. Research motivation

### 2.1 The setting

A long-running agent receives a stream of observations: user messages, tool outputs, environment events, its own actions and intermediate results. Whatever it will need later has to be somewhere it can reach when the need arrives. The task model reads a bounded *active context*; everything else is either in an archive it cannot read until something brings it back, or gone. In `memctl` these three places are the tiers `ACTIVE`, `ARCHIVE` and `DELETED` (`memctl/memory/items.py`), and the agent only ever reads `ACTIVE` (`memctl/agents/base.py`).

Six distinct operations act on this state, and the repository implements each as a separate engine handler (`memctl/memory/engine.py`):

| Concept | Operation(s) | What changes | Reversible? |
|---|---|---|---|
| Keep | `KEEP`, `NO_OP` | nothing | – |
| Permanent eviction | `EVICT` | ACTIVE → DELETED | no |
| Archival | `MOVE_TO_ARCHIVE` | ACTIVE → ARCHIVE; agent can no longer read it | yes, by retrieval |
| Retrieval | `RETRIEVE_FROM_ARCHIVE` | ARCHIVE → ACTIVE; costs budget again | – |
| Compression | `COMPACT`, `COMPACT_AND_ARCHIVE` | a shorter derived item replaces the source; the source is deleted, or archived in full | partly: detail may be lost |
| Consolidation | `CONSOLIDATE` | two or more items merge into one derived item; sources deleted or archived | partly |

The distinction between *active memory* (bounded, readable), *archival memory* (unbounded in every experiment here, unreadable until retrieved) and *permanent eviction* (unrecoverable) is what makes this a richer problem than either context truncation or retrieval-augmented generation. Truncation only has eviction. RAG only has archive plus retrieval. An agent memory controller has all of them, plus lossy rewrites.

### 2.2 Why this is sequential decision-making, not retrieval

Retrieval answers "given a query now, which stored items match it?" The controller's question is different: *before any query exists*, which items should occupy the scarce active budget, which should be parked where a retriever might or might not find them later, and which should be destroyed to make room? Three properties make that a control problem:

1. **The consequence of a decision arrives later.** An item evicted at step *t* costs nothing until the step at which it is needed, which in the synthetic recall environment is drawn log-uniformly between 10 steps and the whole horizon (`memctl/envs/synthetic.py`, `gap`). The per-decision regret measured in this project (§4.5) is exactly that delayed cost.
2. **Decisions interact through the budget.** Keeping one item is evicting another. The exact hindsight solution is a packing problem over time (§4.3), not a per-item score.
3. **In genuinely sequential tasks, decisions change the future stream.** In the workflow environment, forgetting a stage token forces the job to restart, which produces new observations and new tokens and pushes other jobs' tokens out in turn (`memctl/envs/workflow.py`). The trajectory itself depends on the memory policy; this is the property that separates a long-horizon control benchmark from a long-context QA benchmark (§7).

### 2.3 Credit assignment

Task reward arrives at query steps (recall task) or at job completions (workflow task). A memory decision made hundreds of steps earlier contributed to it, along with every other decision in between, and along with the retriever and the task model. This is the credit-assignment problem in its standard form, made harder because two other actors (retrieval, reasoning) can also cause the failure. The framework's response was to build attribution into the harness: every failed query receives exactly one label naming which of the three actors lost the evidence (§9.5), and a hindsight tracker records at which step a future requirement became unrecoverable (§4.5).

### 2.4 Why a fixed heuristic may not suffice

A heuristic encodes an assumption about which items will be needed. Recency assumes need decays with age; frequency assumes past use predicts future use; a "salience" rule assumes specific, user-sourced content is what gets asked about. Whether the assumption holds is a property of the task. The clearest empirical fact in this repository is that the assumption flips between the two synthetic tasks: the salience rule is the best heuristic on the recall task and the worst on the workflow task, where a token that was just used looks identical to one that is still needed (Experiment 5, §10). If no fixed rule is good on both, a policy that learns which rule applies has a reason to exist. That is the motivation for the learned arm, and it is a motivation, not yet a demonstrated advantage (§11, H1).

## 3. Formal problem statement

The definitions below follow the code. Where the implementation is narrower than the general formulation, the narrower version is stated.

### 3.1 Episode, observations, items

An episode is generated by a seeded environment and runs for at most *T* steps (`horizon`). At step *t* the environment emits one observation *o_t* with an id, text content, a source type (user, observation, tool_output, action, retrieved_document, generated_summary, consolidated_memory) and a flag `requires_response`. The observation is ingested as a memory item *m* with the same id (`MemoryState.ingest`), size

  s(m) = count_tokens(content) = number of `\w+` words plus punctuation marks (model-free; `memctl/memory/items.py`),

arrival step *c(m)*, and mutable bookkeeping (tier, access count, last access step, retrieval count, `derived_from_ids`, `superseded_by`). Items are immutable in content: compaction and consolidation create *new* items with lineage links (design decision E1).

The memory state at step *t* is the tuple of tiers

  M_t = (A_t, R_t, D_t),  A_t ∩ R_t ∩ D_t = ∅,

with active token load tok(A_t) = Σ_{m∈A_t} s(m) and archive load tok(R_t). The budget *B* is either an absolute token count or a fraction *f* of the *reference-pass history* H (the total tokens of the same seeded episode run with unlimited budget and no controller): B = max(1, ⌊f·H⌋) (`Experiment.budget_for`). The archive budget B_R was `None` (unlimited) in every experiment.

### 3.2 Controller state and action

The controller sees a read-only `MemoryView` and a `TaskState` (`memctl/memory/state.py`, `memctl/task.py`):

  s_t = (t, B, B_R, A_t, R_t, tok(A_t), tok(R_t), recent actions, recent retrievals; goal, o_t, T).

It never receives ground-truth dependencies (`tests/test_import_boundaries.py` enforces that controllers cannot import `hindsight`; the runner passes hindsight only to controllers with `uses_hindsight = True`).

An action is a *list* of structured `MemoryAction`s applied in order:

  a_t = [(op_k, targets_k, destination_k, parameters_k, confidence_k)]_k,  op_k ∈ Ω,

where the implemented operation set is

  Ω_impl = {KEEP, EVICT, MOVE_TO_ARCHIVE, RETRIEVE_FROM_ARCHIVE, COMPACT, COMPACT_AND_ARCHIVE, CONSOLIDATE, NO_OP}

and six more (PROMOTE, DEMOTE, PIN, UNPIN, UPDATE, SUPERSEDE) exist in the enum with no handler and are rejected as invalid. Each experiment fixes an allowed subset Ω ⊆ Ω_impl (`memory.allowed_operations`); an action outside Ω is rejected, logged, penalised and leaves the state unchanged (E4). Targets are item ids; `parameters` carry the compaction ratio or `max_tokens`, `archive_sources` for consolidation, and the retrieval query, method and scores.

### 3.3 Transition

One step follows the fixed order E3 (`memctl/harness/episode.py`):

1. Ingest: A ← A ∪ {m_t}. Memory may now exceed *B*.
2. Controller: a_t = π_θ(s_t). Engine applies each action, accepting or rejecting it independently: M ← F_Ω(M, a_t).
3. Fallback Φ (harness, not controller): while tok(A) > B, first return any item retrieved *at this step* to ARCHIVE (oldest first; E16), then evict the oldest unpinned item to DELETED. Every forced action is logged with source `harness`, penalised through the `forced_fallback` reward term and reported as `forced_evictions`, `forced_fallback_rate`, `forced_share_of_removals`.
4. Agent: if `requires_response`, u_t = π_task(A_t, o_t, goal). The task model is frozen and reads only ACTIVE.
5. Environment: (o_{t+1}, r_t, done, info) = E(u_t).

So M_{t+1} = ingest(Φ(F_Ω(ingest(M_t, o_t), a_t)), o_{t+1}) and the controller acts on the new item before the agent reads. The harness never retrieves, compacts or consolidates on its own (decision 5 of the design record); every such change is a logged controller action.

### 3.4 Reward

The reward is a weighted sum of per-step terms, every term computed and logged whatever its weight (`memctl/rewards.py`):

  R_t = Σ_k w_k · φ_k(ctx_t).

The terms implemented, with ctx_t the step context:

| term | φ_k |
|---|---|
| `task_reward` | environment reward r_t (1 for a correct answer / completed job, else 0) |
| `active_memory_cost` | tok(A_t)/B at the moment the agent reads |
| `retrieval_cost` | tokens retrieved this step / B |
| `controller_compute_cost` | controller wall-clock seconds this step |
| `invalid_action` | number of rejected controller actions |
| `forced_fallback` | number of items the harness had to evict |
| `memory_hit_rate` | requirements of the current query present in ACTIVE / requirements |
| `retrieval_success` | retrieved items that the current query needed |
| `hindsight_eviction_regret` | (privileged) future requirements made unrecoverable this step |

**Every experiment and every RL training in the repository used the default weights** (`memctl/config.py`, and no sweep or RL config overrides `reward`):

  R_t = r_t − 0.1 · forced_evictions_t − 0.1 · invalid_actions_t.

No run priced active tokens, retrieval or controller compute, and no run used the hindsight regret term for shaping. For PPO the return-to-go of a decision at step *t* is G_t = Σ_{k≥t} γ^{k−t} R_k with γ = 0.995 or γ = 0 (`memctl/rl/train.py:fill_returns`). The episode-level reward logged as `reward_total` is Σ_t R_t.

### 3.5 Task model

The task model π_task is frozen throughout. In every reported experiment except the smoke tests of Experiment 7 it is scripted: `scripted_reader` answers a recall query correctly *iff* the latest statement of the asked attribute is in ACTIVE (with an optional injected failure rate `noise`), and `scripted_tool_agent` runs a workflow instruction *iff* the required token is in ACTIVE and otherwise restarts the job (`memctl/agents/scripted.py`). With `noise = 0`, task success is a deterministic function of memory contents, which is the property the framework relies on to attribute failures.

## 4. Hindsight and oracle formulation

### 4.1 Needed items and evidence requirements

Ground truth is a list of `Dependency` objects, one per query: (query id, step *q*, requirements, gold, category). Each `EvidenceRequirement` is a set of alternative source item ids plus a *needle* string, the exact sentence that states the fact (`memctl/task.py`). A requirement is *met* at step *q* by any **carrier** in ACTIVE, where a carrier is a source item, or an item derived from a source (through compaction or consolidation) whose content still contains the needle (E9; `memctl/hindsight/evidence.py`). This string trace is what makes "compression lost the detail" and "consolidation was wrong" decidable without a model.

How future use is known per environment:

| environment | source of dependencies | exact? |
|---|---|---|
| synthetic_recall | generator records which fact each query asks about, including two-hop link facts | yes |
| workflow | environment records, for each instruction at stage *k* > 0, the tool-output item that stated the job's stage-(*k*−1) token | only until the first restart |
| LoCoMo | dataset's `evidence` dialogue ids per QA pair | yes (labels are the dataset's) |
| LongMemEval | turns marked `has_answer`; if none, any turn of the answer sessions | yes (coarse fallback) |

Two label-free tracers also exist (`memctl/hindsight/trace.py`): `string_dependencies` (evidence = earlier items containing the gold answer) and `tool_value_dependencies` (evidence = earlier items containing a specific value the agent's action used). Both are unit-tested and neither was used as the evidence source of a reported experiment.

### 4.2 The reference pass and the `Hindsight` object

Hindsight is collected from a **reference pass**: the same seed run with unlimited budget and `no_controller` (`Experiment.hindsight`). It yields, for every item, its size, arrival step, source type and a content hash; the full dependency list; the total history tokens *H* (the 100 % budget); and the reference success. From this,

  needs(i) = { q : item *i* is the *designated* carrier of a requirement of the query at step *q* },

where the designated carrier among alternatives is the smallest item (earliest among equals), the one a frugal policy would keep; `all_needs(i)` counts any alternative. The oracle's key primitive is

  next_need(i, t) = min { q ∈ needs(i) : q ≥ t }, or NEVER.

Hindsight is **exact** when the observation stream does not depend on the agent (`stream_depends_on_agent = False`) and marked approximate otherwise (E11). Divergence is detected per step by comparing the content hash of the arriving observation with the reference item of the same id (E17); the episode row records `hindsight_diverged_at`, and from that step on every hindsight-derived metric is `null` and imitation labels are dropped.

### 4.3 The approximate oracle

`OracleController(method="approx")` (`memctl/controllers/oracle.py`) runs every step:

1. Retrieve every archived item *i* with next_need(*i*, *t*) = *t* (if archiving and retrieval are allowed).
2. Drop every ACTIVE item with next_need = NEVER (eager, default) or only as many as the budget requires (`eager: false`, the `oracle_approx_lazy` variant). "Drop" is EVICT when allowed, else MOVE_TO_ARCHIVE.
3. If still over budget, remove the item with the **furthest next need** first; archive it when archiving is allowed, else evict.

Step 3 is Belady's rule for equal-size pages. With unequal item sizes it is not optimal, and Experiment 1c measures the gap: the exact plan is ahead of it by up to 0.12 (0.887 vs 0.772 at horizon 2,000). With an unlimited archive and perfect just-in-time retrieval, the just-in-time oracle is optimal whenever each query's evidence fits the budget, and it reaches 1.000 in every archive-enabled cell of Experiments 2, 5 and 6 (except 0.966 at a 2 % budget in 4b, where a query's evidence did not always fit).

### 4.4 The exact delete-only oracle (integer program)

`solve_delete_only` (`memctl/hindsight/exact.py`) computes the optimal keep schedule for the delete-only action set. Variables: for every needed item *i* and its *k*-th need time q_{i,k}, x_{i,k} ∈ {0,1} means "still in ACTIVE at q_{i,k}, kept continuously since arrival"; for every query *q*, y_q ∈ {0,1} means "all evidence present".

  maximise  Σ_q y_q − ε · Σ_{i,k} s_i x_{i,k}
  subject to  x_{i,k+1} ≤ x_{i,k}                       (once evicted, never back)
              y_q ≤ Σ_{i ∈ carriers(req)} x_{i, pos(i,q)}  for every requirement of *q*
              Σ_{i present at τ} s_i · x_{i, pos(i,τ)} ≤ B   for every check step τ

where the check steps are the arrival steps of needed items and all need times (occupancy only rises at arrivals and falls after needs), and ε = 1/(10·Σ s_i·|needs(i)|) breaks ties toward less memory. Solved with CBC through PuLP; 60 s time limit; at most 50,000 variables; on failure the controller falls back to the approximate rule and records `oracle_method: approx` with the reason. It is exact under three conditions that hold in the delete-only experiments: deleted items never return, items are not rewritten, and the reader answers correctly whenever evidence is in ACTIVE. The independent reviewer confirmed optimality by brute force on 40 small instances (SCIENTIFIC_VALIDITY_REPORT.md §4b). No exact solver exists for other action sets (`SOLVERS` has one entry).

Wall time is the practical limit: the sweep log records the `oracle-exact` cell of 100 episodes at 800 tokens taking 28 s at horizon 500, 83 s at 1,000 and 755 s at 2,000 (`runs/exp1c_horizon_fixed_budget.log`), i.e. roughly 0.3, 0.8 and 7.5 s per episode. (The per-episode `oracle_solve_s` field was added after Experiments 1–1d ran, so those rows do not carry it.)

### 4.5 Regret

Three regret notions are implemented (`memctl/hindsight/regret.py`, `ablation.py`, `evidence.py`):

**(a) Requirements destroyed** (cheap; logged for every episode as `regret.jsonl`). The `EvidenceTracker` watches every future requirement. When an applied removal (EVICT, MOVE_TO_ARCHIVE, COMPACT, COMPACT_AND_ARCHIVE, CONSOLIDATE) leaves a future requirement with no carrier in ACTIVE or ARCHIVE and no source still to arrive, one row is written: step, query, steps until needed, operation, source (controller / harness / intervention), targets, loss kind. Aggregated as `requirements_destroyed` per episode and as histograms of `steps_until_needed`.

**(b) Continuation regret by replay.** With S(·) the task success of a full episode,

  regret(t, a) = S(τ | π runs to *t*, oracle from *t*) − S(τ | π runs to *t*, *a* forced at *t*, oracle from *t*),

implemented by a `SwitchController` that hands over to the approximate oracle at the switch step, and the harness's `interventions` hook, which applies forced actions with source `intervention` and outside the allowed set. Two episodes per decision; exact to the extent the oracle is. This matches the C*-style definition "best achievable future utility without the forced action minus with it", with the oracle standing in for the optimum. Unit-tested; not used at scale.

**(c) Counterfactual ablation.** value(i) = S(baseline) − S(item *i* evicted at its arrival step), under the experiment's own controller and budget. One replay per item. Unit-tested; not used at scale.

### 4.6 Imitation targets, DAgger, RL fine-tuning

The imitation expert (`memctl/rl/expert.py`) is the approximate oracle's rule expressed on the RL controller's decision structure. For each ACTIVE candidate *i* at step *t*:

  label(i) = EVICT with rank 0                                        if next_need(i,t) = NEVER (ARCHIVE if EVICT is disallowed)
  label(i) = ARCHIVE (if allowed, else EVICT) with rank = position of next_need(i,t) among distinct needs, furthest = 1   otherwise

and for each shortlisted archive candidate *j*: retrieve(j) = 1[next_need(j,t) = t]. Rank −1 marks "not this way". The imitation loss follows the expert's removal sequence: at each step the accepted set is every available (item, operation) pair with the lowest rank, and the loss is −log of the probability mass the policy puts on that set, plus binary cross-entropy on the retrieval labels (`memctl/rl/algorithms.py:imitation_loss`). DAgger: in iteration 0 the expert drives the episode; from iteration 1 the learner drives and the expert labels every visited state; the dataset aggregates (cap 40,000 decisions). PPO fine-tuning then optimises the task reward (§3.4) from the imitation initialisation.

### 4.7 When the hindsight expert is the wrong teacher

Optimal *eviction* and optimal *memory management* coincide only when eviction is the only removal. Formally, with delete-only actions the value of keeping item *i* is its future evidence contribution minus the budget it occupies, and the oracle's information (next_need) is exactly what determines it. Add an archive with retrieval probability ρ and zero cost, and the value of the three options for an item with unknown future need *p* becomes

  V(keep) ≈ p,   V(archive) ≈ ρ·p,   V(evict) = 0,

so for any *p* > 0, archiving dominates evicting. The hindsight expert has *p* ∈ {0, 1}: it evicts the *p* = 0 items and archives the rest, and is correct *for itself*. A learner that cannot observe *p* sees the expert evict the majority of items (most items are never needed) and learns "evict"; every mistaken eviction it makes then costs *p* instead of (1−ρ)·p. This is the mechanism behind Experiment 4b (§10.7): the imitation policy's failures are `evicted`, not `archived_not_retrieved`, and an *untrained* retrieval head that archives everything and retrieves liberally does better. The general statement is

  argmax over eviction schedules of S ≠ argmax over (eviction, archival, retrieval) schedules of S,

and, more to the point for learning, the optimal *policy under the learner's information* differs from the hindsight policy exactly when the action set contains a cheap insurance action. The cure is either an expert defined under the learner's information (a cost-aware expert that archives when unsure) or a reward that prices deletion against archiving; neither was tried.

## 5. System architecture

### 5.1 Components

| Component | Where | Role |
|---|---|---|
| `MemoryItem`, `Tier`, `Fidelity`, `count_tokens` | `memctl/memory/items.py` | the unit of memory; frozen dataclass |
| `MemoryState`, `MemoryView` | `memctl/memory/state.py` | tiers, token accounting, event trail; the read-only view controllers get |
| `MemoryAction`, `Operation`, `ActionResult`, `ActionSource` | `memctl/memory/actions.py` | the action schema; results say applied/rejected and who acted (controller / harness / intervention) |
| `MemoryEngine` + `@handler` registry | `memctl/memory/engine.py` | validates and applies actions; enforces the allowed set; the forced-fallback `enforce_budget` |
| Compressors, consolidators | `memctl/memory/compress.py` | `truncate`, `extractive`, `llm`; `dedup`, `llm` |
| `MemoryController` | `memctl/controllers/base.py` | `reset / observe / decide / update / save / load / decision_info`; registry of 14 names |
| `TaskEnvironment` | `memctl/envs/base.py` | `reset / step / get_observation / is_done / get_reward / task_success / get_ground_truth_dependencies / snapshot / restore` |
| `Agent` | `memctl/agents/base.py` | frozen task model: `act(memory_view, observation, task) → AgentStep(action, used_item_ids, info)` |
| `RewardFunction` + `@term` registry | `memctl/rewards.py` | §3.4 |
| `EvidenceTracker`, `failure_label` | `memctl/hindsight/evidence.py`, `memctl/attribution.py` | carrier tracing, regret rows, one label per failed query |
| `run_episode` | `memctl/harness/episode.py` | the only place environment, agent, controller and memory meet |
| `Experiment`, `run_experiment`, `RunLogger` | `memctl/harness/runner.py`, `memctl/runlog.py` | reference pass, budgets, run folders, resume, metadata (git commit + dirty flag, CPU, GPU, packages, model ids) |
| `memctl.sweep` | `memctl/sweep.py` | grids with stable cell names, paired seeds, process pool |
| `memctl.analysis.{load,summarize,stats,report,plots}` | | per-run `summary.json`; per-sweep `report.md` / `report.json` with paired bootstrap, MDE, oracle gap; nine figures plus learning curves |
| `Featurizer`, `ItemPolicy`, `RLController`, trainer | `memctl/features.py`, `memctl/rl/policy.py`, `memctl/controllers/rl.py`, `memctl/rl/train.py` | the learned arm |
| Retrievers, embedders | `memctl/retrieval.py`, `memctl/embed.py` | BM25; cosine over hashing or sentence-transformer embeddings |

### 5.2 Data flow

```
                 seed
                  │
   ┌──────────────▼──────────────┐      reference pass (unlimited budget, no controller)
   │      TaskEnvironment        │──────────────────────────────────────────────┐
   └──────────────┬──────────────┘                                              ▼
                  │ o_t                                                  Hindsight (sizes, arrivals,
                  ▼                                                      needs) ── only to oracle,
   ┌─────────────────────────────┐                                       regret, attribution,
   │  MemoryState.ingest(o_t)    │  ACTIVE may now exceed B               imitation labels
   └──────────────┬──────────────┘
                  │ MemoryView + TaskState (no ground truth)
                  ▼
   ┌─────────────────────────────┐   heuristic │ oracle │ prompted LLM │ JEV │ RL (imitation / PPO)
   │      MemoryController       │
   └──────────────┬──────────────┘
                  │ [MemoryAction …]
                  ▼
   ┌─────────────────────────────┐   accept / reject each; log with source=controller
   │        MemoryEngine         │
   └──────────────┬──────────────┘
                  │
                  ▼
   ┌─────────────────────────────┐   undo unaffordable retrievals, then evict oldest-first;
   │  harness fallback Φ         │   log with source=harness; penalise
   └──────────────┬──────────────┘
                  │ ACTIVE ≤ B
                  ▼
   ┌─────────────────────────────┐   scripted reader / scripted tool agent / LLM
   │   frozen task model reads   │
   │   ACTIVE, emits u_t         │
   └──────────────┬──────────────┘
                  │ u_t
                  ▼
   ┌─────────────────────────────┐   r_t, correct?, next o_{t+1}
   │      TaskEnvironment        │
   └──────────────┬──────────────┘
                  │
                  ▼
   EvidenceTracker → failure label, regret rows;  RewardFunction → R_t, terms;  Feedback → controller.update
```

Shadow controllers (`shadow_controllers` in the config) are asked the same question at every step and never applied; the overlap between the items the real controller removes and the items a shadow would remove is logged as `shadow_agreement`.

### 5.3 Swappability and why it matters

Environment, agent, controller, compressor, consolidator, embedder and retriever are each one config key. The same sweep runs `fifo`, `salience`, `oracle`, `prompted_llm`, `jev` and `rl` cells over identical seeded episodes, so any difference between two cells is a difference in the controller, not in the episodes. The delete-only, archive, and full action sets are one code path with a different `allowed_operations` list, so an ablation never changes the harness. The reference pass, the tracker and the attribution run identically for every controller, so "the oracle" and "the failure labels" mean the same thing in every table. This is a precondition for the stage-gate check in SCIENTIFIC_VALIDITY_REPORT.md §1, which is the framework's first result: with a perfect reader every failure is an eviction (380,231 of 380,231 failures in Experiment 1, verified from `failures.jsonl`), and an injected 20.8 % reasoning-failure rate is recovered as 19.9–20.8 % in every cell of Experiment 1b.

Dependency boundaries are enforced by tests (`tests/test_import_boundaries.py`): controllers never import environments, agents, harness, analysis or hindsight.

## 6. Controllers implemented

All fourteen registered names (`memctl/controllers/__init__.py`), with what each sees and does. "Priority heuristics" share one decision procedure (`PriorityController.decide`): compute a retrieval action if configured and a query is arriving, a consolidation action if configured, any proactive actions, then, if the projected load exceeds *B*, remove lowest-priority items with the configured removal operations (`removal: [EVICT]` by default; `[MOVE_TO_ARCHIVE]`, `[COMPACT, EVICT]`, `[COMPACT_AND_ARCHIVE, MOVE_TO_ARCHIVE]` in the ablations) until it fits. Retrieval, when configured, is BM25 (`lexical`) or cosine (`embedding`) over the archive with `top_k` hits when `requires_response` is set.

| name | sees | rule | trainable | cost per step (measured) | role |
|---|---|---|---|---|---|
| `no_controller` | nothing used | never acts; the harness fallback makes it FIFO truncation | no | 0 | the **no-memory-controller arm** |
| `full_context` | – | never acts, unlimited budget | no | 0 | infinite-context reference |
| `fifo` | item origin step | keep newest | no | 0.02–0.14 ms | recency baseline; equals `no_controller` exactly |
| `lru` | `last_accessed_at` | keep most recently used by the task model | no | ~0.04 ms | access-based recency |
| `lfu` | `access_count` | keep most used | no | ~0.04 ms | frequency |
| `random` | seeded RNG | random priority | no | – | floor |
| `age_decay` | age, access count | (1+uses)·exp(−age/τ), τ = 50 | no | – | decayed recency |
| `similarity` | item, current observation and goal embeddings | keep what resembles the current observation or goal (cosine; hashing embedder in every run) | no | – | content-similarity heuristic |
| `salience` | source type, specificity | w(source)·(0.2 + min(1, specific/3)) + specific/tokens, with fixed weights user 1.0 > consolidated 0.9 > summary 0.8 > observation 0.6 > action 0.5 > retrieved 0.4 > tool_output 0.3; "specific" counts capitalised or digit-bearing words | no | 0.2–0.3 ms | the content-aware heuristic |
| `archive_everything` (`rag_keep_last4`) | – | archive everything but the newest *k* = 4; retrieve top-*k* on queries | no | ~1.5–2 ms | RAG-style reference |
| `oracle` (`approx` / `exact`) | **hindsight** | §4.3 / §4.4 | no | 0.05 ms per step + ILP solve | reference; analysis only |
| `prompted_llm` | item excerpts (40 tokens), budget, allowed ops, BM25 shortlist | asks a language model for a JSON action list when over budget or a query could use the archive; unparsable replies → no action | no (prompted) | 824 s per 40-step episode with llama3.1:8b on CPU (Exp. 7b) | LLM-controller arm (smoke test only) |
| `jev` | JSON state with per-item features | one TypeSafe Choice question per item; repair step removes least-confident KEEPs | no (external model) | not measured (no key) | JEV arm; **blocked** |
| `rl` | 24 per-item features + 6 global features (§6.1) | §6.1 | yes: DAgger imitation, PPO, REINFORCE | 1.9–5.2 ms (h500, 5 %, loaded machine) | the learned arm |

Notes. `archive_everything` in Experiment 2 is also insensitive to the budget by construction (0.672 at every budget), which confirms it is a pure retrieval baseline. The `similarity` controller was worse than random at every budget in Experiment 1 (0.168 vs 0.358 at 10 %, horizon 500). REINFORCE is implemented as PPO with one epoch and no clipping, unit-tested, and used in no experiment.

### 6.1 The learned controller

**Features** (`memctl/features.py`, `FEATURE_VERSION = 1`). Per item: normalised log age, tokens/budget, log tokens, one-hot source type (7), archived flag, one-hot fidelity (3), log access count, log time since access, goal similarity, observation similarity, max similarity to another active item, retrieved-before flag, is-current-observation flag, specific-token count (capped) and specific density, retrieval score (for archive candidates). Global: load/budget, excess/budget, step/horizon, active count/100, archive/budget, `requires_response`. No raw text and no embedding vector enters the policy (`use_embeddings: false` in every config); the similarity features use the hashing embedder.

**Policy** (`memctl/rl/policy.py`). `ItemPolicy` encodes each item with a two-layer MLP (hidden 64), optionally concatenates a DeepSets context (mean and max over candidates) and the global features, and emits one logit per (item, column) for the five columns EVICT, MOVE_TO_ARCHIVE, COMPACT, COMPACT_AND_ARCHIVE, RETRIEVE, plus a scalar value from the pooled state. Parameter counts recorded in the training metadata: 27,526 (DeepSets) and 19,334 (MLP). (EXPERIMENTS.md says "about 12,000"; the run artifacts say 27,526, and the artifacts are taken as ground truth here.)

**Decision procedure** (`memctl/controllers/rl.py`). When a query arrives and the archive is non-empty, a BM25 shortlist of 8 archive candidates gets a Bernoulli retrieve decision each. When memory is over budget (including the tokens of accepted retrievals), the policy picks (item, operation) pairs one at a time without replacement from a softmax over all remaining allowed pairs, until the projected excess is ≤ 0; disallowed operations and non-compactable items are masked. Greedy at evaluation, sampled during training; the log-probability of the whole decision is the sum over picks plus the retrieval Bernoullis, so PPO ratios are exact (confirmed by the reviewer).

**Training** (`memctl/rl/train.py`). Phases run in order on the same policy: `bc` (DAgger, §4.6) and/or `ppo` (§3.4; clip 0.2, value coefficient 0.5, entropy 0.01 or 0.003, 4 epochs, batch 256, gradient clip 1.0, Adam). Training seeds start at 100,000 + config seed; validation seeds at 50,000 (20 episodes per budget fraction, every few iterations); evaluation seeds 0–99 as in every sweep. `policy_best.pt` is the best validation checkpoint, `policy.pt` the final one. Budget fractions are drawn per training episode from the config list.

Provenance caveat: the `rl_bc*` trainings ran before validation seeds were moved to 50,000+, so their logged validation curves used seeds 0–19, which overlap the test seeds; they used the final policy, not a validation-selected one, so no selection on the test seeds occurred (SCIENTIFIC_VALIDITY_REPORT.md §7).

## 7. Environments

| | `synthetic_recall` | `workflow` | `locomo` | `longmemeval` |
|---|---|---|---|---|
| task | answer queries about facts stated earlier | run 16 interleaved jobs of 4 stages; each stage needs the token the previous stage returned | answer QA pairs after a two-person conversation | answer one question after a multi-session chat history |
| success | share of queries answered with the current value | share of jobs completed within the 400-step budget | share of questions judged correct (token-F1 ≥ 0.5 or contains gold; refusal for adversarial); LLM judge optional | same, one question per episode |
| memory dependency | each query needs the latest carrier of one fact (two for two-hop queries); updates make earlier carriers stale; restatements add alternative carriers | each stage-*k* instruction needs exactly the tool-output item that stated the stage-(*k*−1) token | dataset evidence turns per question | `has_answer` turns, or any turn of the answer sessions |
| future use known | exactly, from the generator | exactly until the first restart; approximate afterwards | from dataset labels | from dataset labels |
| do actions alter the future? | no | **yes**: a lost token forces a restart, new tokens, new observations, and the 400-step budget runs out | no | no |
| QA-style or sequential | QA with interleaved queries (queries arrive during the stream, not only after it) | genuinely sequential | QA after history | QA after history |
| scale used | horizon 200 / 500 / 1,000 / 2,000; ~7.2k / 18.0k history tokens at 200 / 500; 19.5 / 52.6 queries per episode | 400-step limit; 3.8k history tokens; 16 jobs | 10 conversations, 1,986 questions (198.6 per episode), ~22.0k tokens per conversation; 787 steps per episode | first 100 instances of LongMemEval-S, ~102k tokens history, 1 question, ~499 steps |
| good for | controlled ablations: horizon, dependency distance (`gap.min`), reuse, updates, restatement, verbosity | whether memory failures compound; ranking flips; oracle blindness | retention and retrieval on real dialogue | very long histories; long-turn assistant text |
| cannot establish | anything about real text or real models; heuristic ranks are generator properties | generality: one task family | long-horizon control; QA accuracy without a task model (blocked here) | same; official judge not run |

**Synthetic recall** (`memctl/envs/synthetic.py`). Defaults: 35 % of free steps state a new fact, the rest are distractors (25 % of them long); sources user / tool_output / observation with shares 0.3 / 0.4 / 0.3 and *query probabilities* 0.7 / 0.15 / 0.35; gap from fact to query log-uniform in [10, horizon]; re-query 0.35; update 0.15; restate 0.15 (as a long tool_output item); two-hop 0.15; verbose wrapping 0.3. Facts read "As of step *n*, the *attribute* of *entity* is *value*" so a reader holding two statements can tell which is current. The per-source query probabilities are the confound behind the salience result (§14).

**Workflow** (`memctl/envs/workflow.py`). Instructions ("run the review stage of job-104 using its intake token") are user items; tool results ("As of step *n*, the intake token of job-104 is K93Q") are tool_output items; half of free steps are log noise. A wrong or missing token resets the job to stage 0 and discards its token. With full context an episode completes in 192 steps with no restart; at 5 % budget (190 tokens) the 16 jobs' live tokens can exceed the budget, so in 49 of 100 episodes a restart is unavoidable for every controller including the oracle (`runs/exp5b_workflow_rl/*/episodes.jsonl`).

**LoCoMo / LongMemEval adapters** (`memctl/envs/qa.py`, `locomo.py`, `longmemeval.py`). Turns are shown in order as items (LoCoMo turns as `user` with speaker metadata; LongMemEval user turns as `user`, assistant turns as `observation`), then the questions. Seeds index conversations (`seed mod 10`) or instances. Scoring is local token-F1 unless an LLM judge is configured (none was).

**Long-term memory benchmark vs long-horizon control benchmark.** LoCoMo and LongMemEval ask every question after the history ends, so nothing the agent forgets changes what it sees next; they measure retention, archival, retrieval and evidence preservation under a budget (Experiment 6 measures exactly that, with no task model). They cannot show that a controller helps an agent *complete a task* whose later steps depend on earlier memory. The workflow environment has that property and is the only one here that does; it is synthetic and small. Neither kind substitutes for the other.
## 8. Experimental methodology

### 8.1 Protocol shared by every sweep

- **Paired seeded episodes.** Episode *i* of every cell uses seed `base_seed + i` (base 0), so every controller in a sweep sees identical episodes; a test checks that a repeated seed reproduces an episode exactly. Comparisons are paired by seed.
- **Budgets are swept**, never compared at a single value. Fraction budgets are shares of the reference-pass history; Experiment 1c uses an absolute 800-token budget to remove the horizon–budget confound.
- **Task model:** `scripted_reader` (noise 0) or `scripted_tool_agent` (noise 0) except 1b (noise 0.2), 6 (`null`, no task model) and 7 (llama3.1:8b).
- **Action sets:** delete-only {KEEP, EVICT, NO_OP} in 1, 1b–1d, 4, 5b, 7; archive set {+ MOVE_TO_ARCHIVE, RETRIEVE_FROM_ARCHIVE} in 2, 4b, 5, 6, 6c; full set (+ COMPACT, COMPACT_AND_ARCHIVE, CONSOLIDATE) in 3.
- **Archive:** unlimited and free in every run (`archive_budget: None`, `retrieval_cost` weight 0).
- **Retrieval:** BM25 over content words with document frequencies from the archive, `top_k` 3 (synthetic) or 5 (LoCoMo, LongMemEval); `embedding` variants use cosine over the 256-bucket hashing embedder except 6c, which also uses `BAAI/bge-small-en-v1.5` on CPU.
- **Compression/consolidation (Exp. 3):** `compact_ratio` 0.5, `min_compact_tokens` 30; rewriters `extractive` (keep most specific sentences) and `truncate`; consolidator `dedup` (join, drop repeated sentences), grouping the newest item with up to 3 older items sharing a specific sentence; `_tight` variants shorten the merged item to 40 % of its sources.
- **Reward:** default weights (§3.4) everywhere.
- **Logging:** full per-step rows for the first 1–3 episodes of each cell; every episode has a row in `episodes.jsonl`, failure rows and regret rows.
- **Hardware:** 16-core laptop CPU, 23 GB RAM, RTX 3050 4 GB (unused by any scripted experiment); 10–13 sweep workers plus up to 10 trainings shared the cores, so latencies are order-of-magnitude only.

### 8.2 Experiment matrix

| ID | config | environment | controllers | budgets | horizon / size | episodes | status |
|---|---|---|---|---|---|---|---|
| 1 | `configs/sweeps/exp1_delete_only.yaml` | synthetic | 11 (no_controller, fifo, lru, lfu, random, age_decay, similarity, salience, oracle_approx, oracle_exact, full_context) | 2, 5, 10, 25, 50, 100 % | 200 / 500 / 1,000 | 100 per cell, 198 cells | full run |
| 1b | `exp1b_attribution_check.yaml` | synthetic, reader noise 0.2 | fifo, salience, oracle_exact | 10, 25, 100 % | 500 | 100 | full run |
| 1c | `exp1c_horizon_fixed_budget.yaml` | synthetic | fifo, lru, random, salience, oracle_approx, oracle_exact | 800 tokens | 200 / 500 / 1,000 / 2,000 | 100 | full run |
| 1d | `exp1d_dependency_gap.yaml` | synthetic, `gap.min` ∈ {5, 25, 100, 250} | fifo, lru, random, salience, oracle_exact | 10 % | 500 | 100 | full run |
| 2 | `exp2_archive_retrieval.yaml` | synthetic | 12 variants (delete / archive-no-retrieval / archive+retrieve / top-1 / hashing embedding / rag_keep_last4 / two oracles) | 2, 5, 10, 25 % | 500 | 100 | full run (rerun after fix E16) |
| 3 | `exp3_compaction_consolidation.yaml` | synthetic, `verbose_prob` 0.5, `restate_prob` 0.4 | 10 variants × 2 rewriters | 5, 10, 25 % | 500 | 100 | full run |
| 4 | `exp4_rl_eval.yaml` + 9 trainings in `configs/rl/` | synthetic, delete-only | fifo, lru, salience, rl_bc, rl_bc_mlp, rl_bc_ppo, rl_ppo ×3 seeds (+ `_last`), rl_ppo_gamma0 ×3 seeds, oracle_exact | 5, 10, 25, 50 % | 200 (train) / 500 (transfer) | 100 | full run |
| 4b | `exp4b_rl_archive_eval.yaml` + 2 trainings | synthetic, archive set | fifo_delete, salience_delete, salience_archive_retrieve, rl_bc_delete_only_policy, rl_bc_archive, rl_bc_ppo_archive, oracle_approx | 2, 5, 10, 25 % | 200 | 100 | full run (rerun after E16) |
| 5 | `exp5_workflow.yaml` | workflow, archive set | 11 (as Exp. 1 minus similarity/lfu variants, plus fifo/salience archive+retrieve) | 5, 10, 20, 40 % | 400-step limit | 100 | full run (rerun after E16; pre-fix cells kept in `runs/_superseded/`) |
| 5b | `exp5b_workflow_rl.yaml` + 3 trainings | workflow, delete-only | fifo, lru, salience, rl_bc_transfer, rl_bc_workflow, rl_ppo_workflow, rl_ppo_workflow_gamma0, oracle_approx, oracle_approx_lazy | 5, 10, 20, 40 % | 400-step limit | 100 | full run; **only the `rl_bc_workflow` cells were rerun after fix E17** (see §12.3) |
| 6a | `exp6_locomo_retention.yaml` | LoCoMo, `agent: null` | 9 variants | 10, 25, 50 % | 10 conversations | 10 | full run of the pipeline; **QA accuracy blocked** |
| 6b | `exp6_longmemeval_retention.yaml` | LongMemEval-S, `agent: null` | 6 variants | 5, 10, 25 % | 100 instances | 100 | as 6a |
| 6c | `exp6c_locomo_dense_retrieval.yaml` | LoCoMo, `agent: null` | fifo archive+retrieve × {BM25, cosine} × {hashing, bge-small} | 10 % | 10 conversations | 10 | full run (rerun once after a duplicated-rows incident) |
| 7a | `exp7_llm_smoke.yaml` | synthetic, horizon 60, llama3.1:8b as task model via Ollama on CPU | fifo, salience, oracle_exact, full_context | 25 % | 60 | 3 (11 queries in all) | **smoke test** |
| 7b | `exp7b_prompted_controller_smoke.yaml` | synthetic, horizon 40, llama3.1:8b as controller | fifo, salience, prompted_llm, oracle_exact | 40 % | 40 | 2 (5 queries) | **smoke test** |
| H6 | – | – | `jev` | – | – | – | **blocked**: no `JEV_API_KEY` |
| P2 | `runs/locomo_4b_4bit/` (branch `phase2-locomo-baseline`) | LoCoMo, Qwen3.5-4B 4-bit as answerer | keep_newest | 10 % | 3 conversations, 180 questions | – | earlier pipeline validation; not comparable with 16-bit |

### 8.3 RL trainings

| training | algorithm | γ | episodes | budgets | env | wall time | best validation |
|---|---|---|---|---|---|---|---|
| `rl_bc`, `rl_bc_mlp` | DAgger BC, 10 it × 24 ep, 3 epochs, lr 3e-3 | – | 240 | 5/10/25/50 % | recall h200 | 10 / 17 min | final policy used |
| `rl_bc_ppo` | BC 10×24 then PPO 40×24, lr 1e-4, entropy 0.003 | 0.995 | 1,200 | same | recall h200 | 43 min | 0.828 at 480 ep |
| `rl_ppo`, `_seed1`, `_seed2` | PPO 100×24, lr 3e-4, entropy 0.01 | 0.995 | 2,400 | same | recall h200 | 46–53 min | 0.779 / 0.718 / 0.764 |
| `rl_ppo_gamma0`, `_seed1`, `_seed2` | same | 0 | 2,400 | same | recall h200 | 43–49 min | 0.738 / 0.745 / 0.759 |
| `rl_bc_archive` | DAgger BC 10×24 | – | 240 | 2/5/10/25 % | recall h200, archive set | 23 min | final policy used |
| `rl_bc_ppo_archive` | BC 6×24 then PPO 60×24, lr 3e-4 | 0.995 | 1,584 | same | recall h200, archive set | 36 min | 0.693 at 1,584 ep |
| `rl_bc_workflow` | DAgger BC 10×24 (labels after divergence dropped) | – | 240 | 5/10/20/40 % | workflow | 33 min | 1.0 at 48 ep |
| `rl_ppo_workflow`, `_gamma0` | PPO 60×24 | 0.995 / 0 | 1,440 | same | workflow | 42 min each | 0.854 at 360 ep / 1.0 at 360 ep |

Source: `runs/rl_*/summary.json`, `train_log.jsonl`. EXPERIMENTS.md rounds the PPO episode counts to "1,200–2,400" and the archive PPO run to "1,344 PPO episodes" (1,584 − 240 imitation episodes = 1,344; consistent).

## 9. Evaluation metrics

All formulas are taken from `memctl/harness/episode.py`, `memctl/analysis/stats.py` and `memctl/analysis/summarize.py`. The **episode** is the statistical unit throughout; per-cell numbers are means over 100 (or 10) episodes.

**Task success.** Recall: correct / scored queries in the episode, where correct means the gold value appears as a token of the answer (so a language model may wrap it in a sentence). Workflow: completed jobs / 16. QA benchmarks: judged-correct / questions. `query_accuracy` pools queries across episodes and is reported alongside but not used for tests. Interpretation: the fraction of the task the agent got right under this memory policy. Limitation: with a perfect reader it is a pure function of memory contents; with a real model it mixes in reasoning failures, which attribution separates.

**Memory budget usage.** `active_tokens_mean` = (1/steps)·Σ_t tok(A_t) at the read; `active_tokens_peak` = max_t tok(A_t); the retained share of the full history is `active_tokens_mean / history_tokens`. `unnecessary_token_share` (hindsight) = Σ_t tokens in ACTIVE held by items no query at or after *t* needs / Σ_t tok(A_t), computed only while hindsight is valid.

**Archive usage.** `archive_tokens_final`, `archive_tokens_peak`; `retrievals` (RETRIEVE actions), `retrieved_items` (items, so one action with three targets counts three).

**Hit rate / evidence availability.** `needed_hit_rate` = Σ_queries (requirements met in ACTIVE at the read) / Σ requirements. `evidence_complete_rate` = queries with *all* requirements met / queries with labels; this is the accuracy a perfect reader would reach and is the measure used in Experiment 6, where there is no task model.

**Retrieval precision and recall.**
  precision = retrieved items that carried evidence for the query answered *at that step* / retrieved items (a retrieval made for a later query counts as a miss);
  recall = requirements whose evidence was only in the archive and was retrieved in time / requirements whose evidence was only in the archive (retrieved or not).
Both are `null` when the denominator is zero.

**Failure attribution** (`memctl/attribution.py`), one label per failed query, most severe first:

| label | meaning | who is blamed |
|---|---|---|
| `evicted` (cause `controller` / `harness`) | a needed carrier was deleted outright | controller, or the fallback |
| `compression_lost_detail` | the only carriers are compacted items that no longer contain the needle | the rewriter / the controller that compacted |
| `consolidation_incorrect` | same, for consolidated items | consolidator |
| `archived_not_retrieved` | a carrier is in the archive and was not retrieved | retriever / retrieval policy |
| `retrieved_but_ignored` | all evidence was in ACTIVE and some of it was retrieved this step; the model still failed | task model |
| `task_model_reasoning` | all evidence was in ACTIVE; the model failed | task model |
| `invalid_action` | the fallback deleted the carrier on a step where the controller's action targeting it (or all its actions) was rejected | controller |
| `unknown` | no evidence label | – |

`catastrophic_forgetting_events` = evicted + compression_lost_detail + consolidation_incorrect + invalid_action.

**Oracle gap closed** (`stats.oracle_gap_closed`):

  closed = (S_method − S_baseline) / (S_oracle − S_baseline),

with all three means over the *same* seeds, baseline `no_controller` or `fifo`, oracle the exact one when present else the approximate one. 0 = no better than FIFO; 1 = matches the oracle; negative = worse than FIFO (random and similarity in Experiment 1); > 1 is possible only when a method beats the oracle, which happened on the workflow task where the oracle is blind (§10.9) and is a warning sign rather than an achievement. Undefined when the oracle leaves no gap. Two caveats: the oracle is not always a bound (§4.3), and part of the gap is information no causal policy has (§14), so the achievable maximum is unknown and below 1.

**Regret.** Per decision: `regret.jsonl` rows (§4.5a), summarised as `requirements_destroyed` per episode, by operation, by source, and as a histogram of steps-until-needed. Per episode relative to the oracle: `regret_vs_oracle` = mean over paired seeds of (S_oracle − S_method).

**Latency.** `controller_latency_s` = Σ_t wall time of `decide`; reported per step. Excludes the exact oracle's ILP solve (recorded separately as `oracle_solve_s` where present). `agent_latency_s` = Σ_t wall time of `act`. Retrieval latency is inside controller latency.

**Compute / token cost.** `tokens_processed` = agent prompt tokens + controller input tokens; `task_model_calls`, `controller_model_calls`; `estimated_cost_usd` from `pricing` (zero in every run) plus any cost the controller reports (JEV would).

**Fallback frequency.** `forced_evictions` (items), `forced_fallback_rate` = steps with a forced action / steps, `forced_share_of_removals` = forced evictions / all removals. A controller with a high forced share is not managing the budget itself: `no_controller` has forced evictions on most steps (450 items per 500-step episode at 10 %), every priority heuristic has zero, and the prompted LLM in 7b had 15.5 per 40-step episode because six of nine replies were unparsable.

**Paired comparison, bootstrap, MDE** (`stats.py`). For two cells on the same seeds, d_i = S_a(i) − S_b(i). The mean difference is reported with a 95 % percentile bootstrap interval from 5,000 resamples of the episodes and a two-sided bootstrap p-value. The minimum detectable effect is

  MDE = (z_{0.975} + z_{0.80}) · sd(d) / √n ≈ 2.80 · sd(d) / √n,

the smallest true paired difference that *n* episodes would detect at α = 0.05 with 80 % power. Why episodes and not queries: queries inside one episode share a memory state and are correlated, so treating 5,265 queries as independent would overstate precision by roughly the factor √(queries per episode). Why paired: the same seeds remove between-episode variance from the difference, which is why MDEs of 0.007–0.03 are reachable with 100 episodes. Per-cell means also carry a 95 % bootstrap interval (2,000 resamples). No multiple-comparison correction was applied anywhere (§14).
## 10. Experiments and results

Every number below was read from `runs/<sweep>/<cell>/summary.json`, `report.json` or `episodes.jsonl` during the preparation of this note, not copied from EXPERIMENTS.md; the few places where EXPERIMENTS.md differs from the artifacts are flagged. All runs carry `metadata.json` with commit `d215c8f` and `dirty: true` (654 of 654 metadata files), so **none of these numbers is final** (§14.14).

### 10.1 Summary table

| ID | Question | Env | Controllers | Budget | Seeds | Metric | Result | Interpretation | Status |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Do policies differ at equal budget? | recall | 11 | 2–100 % | 0–99 | task success | at h500, 10 %: non-oracle range 0.168 (similarity) to 0.762 (salience); oracle_exact 1.000; FIFO 0.437 | large policy effects; four pre-registered criteria met | full |
| 1b | Does attribution recover injected reasoning noise? | recall, noise 0.2 | fifo, salience, oracle | 10/25/100 % | 0–99 | recovered rate | 0.199–0.208 vs injected 0.208 | attribution separates model from memory | full |
| 1c | Does headroom grow with horizon at fixed tokens? | recall | 6 | 800 tok | 0–99 | success | best-heuristic-to-exact-oracle gap 0.235 → 0.575 from h200 to h2000 | room for a better policy grows | full |
| 1d | …with dependency distance? | recall | 5 | 10 % | 0–99 | success | FIFO 0.504 → 0.051, salience 0.769 → 0.573, oracle 1.000 throughout as `gap.min` 5 → 250 | recency collapses; oracle unaffected | full |
| 2 | Archive + retrieval vs deletion | recall | 12 | 2–25 % | 0–99 | success, recall/precision | FIFO 0.233 → 0.734 at 5 %; archive-no-retrieval = delete exactly; BM25 top-3 recall 0.56–0.69 | retrieval is where the value is; retriever becomes the bottleneck | full |
| 3 | Compaction and consolidation | recall, verbose | 10 × 2 rewriters | 5–25 % | 0–99 | success, attribution | extractive compaction: FIFO +0.169 at 10 %, salience +0.000; truncate hurts both; naive consolidation hurts salience 0.725 → 0.610 | conditional on the rewriter; lossy rewrites are attributable | full |
| 4 | Learned vs heuristics vs oracle (delete-only) | recall | 3 heuristics, 9 learned, oracle | 5–50 % | 0–99 | success, paired diff | imitation within MDE of salience at 10/25 %, −0.05 at 5 % (h500); BC+PPO +0.018 [+0.014, +0.023] at 25 %; PPO-only −0.13 to −0.24 below salience | learned ≫ recency, ≈ best content rule, not > it | full |
| 4b | …with archive and retrieval | recall | 7 | 2–25 % | 0–99 | success, failures | rl_bc_archive 0.521 at 5 % (924 `evicted` failures) vs untrained-retrieval-head policy 0.849 vs salience_archive_retrieve 0.833 | imitating a hindsight deleter is the wrong target once archiving exists | full |
| 5 | Does the ranking hold on a sequential task? | workflow | 11 | 5–40 % | 0–99 | success | salience 0.051 vs FIFO 0.221 at 10 %; retrieval recall 0.01–0.10; archive+retrieve gives nothing | ranking flips; lexical retrieval fails | full |
| 5b | Learned controllers on the sequential task | workflow, delete-only | 9 | 5–40 % | 0–99 | success, restarts | rl_bc_workflow 1.000 at every budget (1.2 restarts at 5 %); oracle 0.808 (25 restarts); recall-trained policy 0.042 | task-trained imitation solves it; oracle blind after divergence; no cross-task transfer | full (partial rerun, §12.3) |
| 6a | Evidence retention on LoCoMo | LoCoMo, no task model | 9 | 10–50 % | 10 conv. | evidence_complete_rate | fifo_delete 0.016 / salience 0.152 / fifo_archive_retrieve_lexical 0.417 / exact delete oracle 0.614 at 10 %; BM25 recall 0.35–0.38 | same ordering as synthetic; QA accuracy not measured | pipeline full; accuracy blocked |
| 6b | …on LongMemEval-S | LME-S, no task model | 6 | 5–25 % | 100 inst. | evidence_complete_rate | salience 0.49 / fifo_archive_retrieve_lexical 0.59 / fifo_delete 0.05 at 5 % | salience result is a dataset artefact | as 6a |
| 6c | Dense vs lexical retrieval on LoCoMo | LoCoMo | 2 × 2 | 10 % | 10 | retrieval recall | BM25 0.347; bge-small 0.215; hashing 0.115 | dense < lexical on this data, with this embedder | full, 10 episodes |
| 7a | Real LLM as task model | recall h60 | 4 | 25 % | 3 | success = needed_hit_rate | FIFO 0.233, salience 0.917, oracle 1.0; model answered iff evidence present | pipeline works | smoke |
| 7b | Real LLM as controller | recall h40 | 4 | 40 % | 2 | success, parse rate | 0.50; 6 of 18 replies parsable; 15.5 forced evictions per episode; 824 s per episode | pipeline works; nothing about quality | smoke |
| H6 | JEV controller | – | jev | – | – | – | not run | – | blocked |

### 10.2 Experiment 1: policies differ at equal budget

*What happened.* At horizon 500 and 10 % budget the eleven controllers span 0.168 (similarity) to 0.762 (salience) among non-oracles, with the recency family clustered at 0.437–0.460 and the exact oracle at 1.000; the paired difference of salience over FIFO is +0.326 [+0.303, +0.347] against an MDE of 0.031, and the exact oracle's is +0.563 [+0.548, +0.579]. The four criteria written before the run all passed: every controller reaches 1.000 at 100 % budget (33 cells); every one of the 380,231 failed queries is labelled `evicted` (46,782 with cause `harness`, all from `no_controller`); the oracle-to-FIFO gap at ≤ 25 % is 0.275–0.835 at horizon 500 (0.23–0.88 over all three horizons) against MDEs of 0.015–0.022; and the salience-vs-FIFO interval excludes zero in all 15 sub-100 % cells. `no_controller` and `fifo` agree to three decimals in every cell, with the former showing 99–980 forced evictions per episode (across horizons and sub-100 % budgets) and the latter none.

*Why.* The generator asks about user-sourced facts with probability 0.7 and tool outputs with 0.15, and facts are the only items containing capitalised identifiers and digits; the salience rule encodes both. Recency rules can only keep items whose query gap falls inside the window the budget affords, and the gap is log-uniform up to the horizon. Similarity keeps what resembles the *current* observation, which is usually a distractor or an unrelated fact.

*What it supports.* Memory policy is a first-order determinant of task success under a budget (the "up to ~0.6" figure the project quotes is the 0.563 oracle-over-FIFO paired difference at h500/10 %, and the range between non-oracle controllers is 0.59 at that cell). The measurement apparatus works: attribution, forced-fallback accounting and the oracle references behave as designed.

*What it does not support.* Any claim about salience as a policy: its weights were set by hand before the run but match the generator's query distribution (§14.1). Any claim about horizon: the fractional budget grows with the horizon, so success *rises* with horizon here.

*Follow-up.* Set every source's `query_prob` equal and rerun; salience's advantage should shrink to its "has identifiers" component. That isolates how much of the effect is source-type prediction.

### 10.3 Experiment 1b: attribution recovers injected noise

With the reader failing 20 % of queries whose evidence it has, the recovered reasoning-failure rate (reasoning failures / (correct + reasoning failures)) is 0.199–0.208 in every cell against an injected rate of 1,093/5,265 = 0.208 on these seeds, and the exact oracle's success is 0.792 at every budget, the reader's ceiling. Attribution never blames the controller for a model failure. This is the precondition for reading any later failure table, not a result about memory.

### 10.4 Experiments 1c and 1d: horizon and dependency distance

At a fixed 800-token budget everything degrades with horizon, the oracle least: FIFO 0.282 → 0.126, salience 0.765 → 0.312, exact oracle 1.000 → 0.887 from 200 to 2,000 steps; the best-heuristic-to-exact-oracle gap grows from 0.235 to 0.575; the approximate oracle trails the exact one by up to 0.115. With horizon and budget fixed and only the minimum fact-to-query distance varied (1d), FIFO and LRU fall from ~0.50 to ~0.04 between gap 5 and gap 100 while salience falls from 0.769 to 0.573 and the exact oracle stays at 1.000; queries per episode fall from 56 to 18, so the gap-250 cells are noisier. Supports H3 as a statement about *room*: the value of knowing what will be needed grows with horizon and dependency length. Does not show that any non-oracle policy captures that room. Confound in 1c: in this generator, dependency distance grows with horizon (log-uniform up to the horizon), which 1d isolates.

### 10.5 Experiment 2: archive and retrieval

*What happened.* `fifo_archive_no_retrieval` equals `fifo_delete` to three decimals at every budget, with every failure relabelled from `evicted` to `archived_not_retrieved`. With BM25 top-3 retrieval FIFO goes from 0.233 to 0.734 at 5 % (0.036 → 0.673 at 2 %), LRU likewise, and salience from 0.580 to 0.794. The FIFO–salience gap shrinks from 0.347 (deleting) to 0.060 (archiving). Retrieval recall is 0.56–0.69 at 5 % (top-1: 0.37; hashing embedder: 0.44), precision 0.10–0.26, and every remaining failure is `archived_not_retrieved`. The approximate oracle, which archives and retrieves just in time, reaches 1.000 at 2 %, above the exact delete-only optimum of 0.871. `rag_keep_last4` scores 0.672 regardless of budget.

*Why.* Archiving is deletion until something retrieves; once something does, the question "what stays in context" matters less than "can the retriever find it". The archive is unlimited (17,600 tokens at peak against a 2 % budget of ~360) and free.

*What it supports.* H4 on this task: archive + retrieval beats any eviction rule, and by more than the difference between eviction rules. Once an archive exists the retriever is the bottleneck, and the framework says so through the labels.

*What it does not support.* That archiving helps in general: on the workflow task the same retriever has recall 0.01–0.10 and archiving gives nothing (§10.8). That the comparison is fair on compute: retrieving controllers spend ~1.4–1.8 ms per step against 0.07 ms, and the archive is unpriced.

*Follow-up.* Set `memory.archive_budget` and a non-zero `retrieval_cost` weight; the archive advantage should shrink and a placement–retrieval trade-off should appear. Separately, sweep the retriever (top-*k*, dense models) on the workflow task to see whether the archive result is a retriever result.

### 10.6 Experiment 3: compaction and consolidation

*What happened.* With the extractive rewriter, compaction lifts FIFO from 0.439 to 0.608 at 10 % and salience from 0.725 to 0.725; with the truncating rewriter it lowers FIFO to 0.418 and salience to 0.618, and 1,368–1,443 failures per 10 % cell are labelled `compression_lost_detail` (none with extractive). Consolidation of restated facts gives FIFO one or two points (0.439 → 0.447; tight 0.455). Naive consolidation hurts salience (0.725 → 0.610); the tight variant recovers most of it (0.717) and at 25 % is the best non-oracle cell (0.993). Keeping a stub and archiving the original (`fifo_compact_archive_retrieve`, extractive) reaches 0.917 at 10 %.

*Why.* The merged item is long, cannot be compacted again (fidelity is no longer FULL), and is evicted whole, taking a fact the policy would have kept as a short item. The extractive rewriter scores sentences by the same "specific" notion (identifiers, numbers) that marks facts in this generator, so it is nearly lossless by construction.

*What it supports.* Lossy rewriting is harmful and the attribution labels it correctly (the negative result stands on its own). Compaction helps a weak policy much more than a good one.

*What it does not support.* H5 in general: the positive result is a statement about a rewriter that is matched to the generator.

*Follow-up.* Use the `llm` compressor and consolidator (implemented, unit-tested with a scripted model) on a GPU host, and count `compression_lost_detail` and `consolidation_incorrect`; a verified compressor would be one whose output is checked against the needle before the source is deleted.

### 10.7 Experiments 4 and 4b: learned controllers

*What happened (4, delete-only).* Every learned policy beats every recency rule by a wide, paired-significant margin (+0.24 to +0.49 over FIFO at 5–25 %). Against salience at horizon 500, recomputed from `episodes.jsonl`: imitation (`rl_bc`) −0.049 [−0.069, −0.030] at 5 %, −0.007 [−0.020, +0.005] at 10 %, −0.000 [−0.005, +0.005] at 25 %; imitation then PPO (`rl_bc_ppo`) −0.022 [−0.037, −0.008], +0.004 [−0.006, +0.013], +0.018 [+0.014, +0.023]; PPO from scratch −0.20 to −0.24 at 5 %, −0.13 to −0.17 at 10 %, −0.04 to −0.08 at 25 % across three seeds; γ = 0 PPO −0.15 to −0.21, −0.08 to −0.13, −0.04 to −0.07. Oracle gap closed at h500: imitation 0.39–0.43 (5 %), 0.56–0.58 (10 %), 0.86–0.93 (25 %); salience 0.46, 0.58, 0.86; PPO 0.15–0.25, 0.27–0.44, 0.57–0.74. Shadow agreement of every learned policy with FIFO and LRU is 0.00–0.01 and with salience 0.36–0.68. Latency 1.9–5.2 ms per step against 0.19 ms for salience. The MLP without DeepSets context is within a few points either way. PPO seed spread is up to 0.13 at h200/5 %, and the final PPO policy is below its best-validation checkpoint (`rl_ppo_last` 0.572 vs 0.590 at h500/10 %).

*What happened (4b, archive set).* Imitation of the hindsight expert with archiving allowed (`rl_bc_archive`) scores 0.521 at 5 % with 924 `evicted` failures and 1.25 retrieved items per episode (precision 0.74); after 1,344 PPO episodes 0.578 with 772 `evicted` failures. The Experiment-4 delete-only policy run *unchanged* with an untrained retrieval head scores 0.849, retrieving 125 items per episode (precision 0.11, recall 0.80) and failing only by `archived_not_retrieved`; salience with archive + retrieval scores 0.833 (54 items, precision 0.15, recall 0.70).

*Why.* On the recall task, which items get queried is a Bernoulli draw given the source type, so the observable features support exactly the rule salience encodes and no more; the remaining gap to the oracle is information no causal policy has. On 4b, the mechanism of §4.7: the expert evicts what it knows is useless, the learner copies the majority action and loses evidence it could have archived for free.

*What it supports.* Learned controllers can reach the best fixed rule from 240 imitation episodes and are content-based rather than recency-based. Imitation is an order of magnitude more sample-efficient than PPO here. Hindsight imitation is the wrong target when the action set contains cheap insurance.

*What it does not support.* H1 as stated: no learned policy beats the best heuristic on this task beyond +0.018 at one budget. H2 (§10.10). Anything about generality across tasks from Experiment 4 alone.

*Follow-up.* (i) A "Bayes-informed" reference that knows the generator's query probabilities, to measure how much of the oracle gap is closable. (ii) A cost-aware expert (archive when unsure; retrieve when free) or a reward that prices deletion, for 4b. (iii) More PPO seeds and a learning-rate sweep before any PPO-vs-imitation claim.

### 10.8 Experiment 5: the sequential task flips the ranking

*What happened.* At 10 % budget FIFO completes 0.221 of jobs and salience 0.051; at 20 % 0.891 vs 0.399; LRU tracks FIFO; LFU is worse than random. Archiving with BM25 retrieval gives nothing (FIFO 0.170 vs 0.221 at 10 %; retrieval recall 0.014–0.057 for FIFO, 0.044–0.60 for salience), and every failure of an archiving controller is `archived_not_retrieved`. With full context an episode finishes in 192 steps with zero restarts; at 10 % FIFO restarts 51 jobs per episode and every episode hits the 400-step limit. The oracle with archive and retrieval completes everything at every budget with no restarts.

*Why.* A stage token that has just been used and one that is still needed are the same kind of item (tool output, one identifier) to any content rule; salience fills memory with used tokens. FIFO wins because the next stage of a job usually comes soon. The instruction that needs a token shares more words with other instructions than with the item holding the token, so BM25 ranks the wrong items.

*What it supports.* No fixed heuristic examined here is good on both tasks (salience: best on recall, worst on workflow; FIFO: near-worst on recall, best on workflow). Memory failures compound when the task is sequential. The archive result of Experiment 2 is retriever-dependent.

*What it does not support.* That salience is bad in general, or that retrieval cannot work here (a retriever keyed on the job id would).

*Follow-up.* Give the retriever the job identifier (structured query) and rerun; if recall rises to that of Experiment 2, the archive result transfers and the placement question reopens on this task.

### 10.9 Experiment 5b: learned controllers on the sequential task

*What happened.* `rl_bc_workflow` (imitation on this task, 240 episodes) completes every job at every budget: 1.000 at 5 % with 1.2 restarts per episode and 199 steps, where FIFO completes 0.028 with 64 restarts. Its validation success was already 1.0 after 48 episodes. `rl_ppo_workflow_gamma0` reaches 0.992 (16.5 restarts); `rl_ppo_workflow` (γ = 0.995) 0.404 (43 restarts), having peaked at 0.416 on validation after 360 episodes and drifted. The recall-trained policy (`rl_bc_transfer`) scores 0.042 at 5 % and 0.247 at 10 %, i.e. FIFO-like success with shadow overlap 0.00 with FIFO and 0.24 with salience. The approximate oracle scores 0.808 at 5 % with 25 restarts (lazy variant 0.896, 12 restarts).

*Why the learner beats the oracle.* In the 49 episodes (of 100 at 5 %) where the 16 jobs' live tokens exceed 190 tokens, a restart is unavoidable for every controller; both the oracle and `rl_bc_workflow` restart in exactly those 49 episodes. After the first restart the stream no longer matches the reference pass, so the oracle's `next_need` describes another episode and it evicts blindly. The learned policy uses features it can see (`access_count`, source type, specificity): a token whose stage has run has been accessed; one still needed has not.

*What it supports.* A learned policy trained on the task solves the task where no fixed rule and no hindsight reference does, and is robust to trajectory divergence because it never depended on hindsight. This is the strongest positive result for the learned arm, with the caveat that "beats the oracle" is a statement about the reference's blindness, not about optimality.

*What it does not support.* Cross-task transfer (H7): the recall-trained policy does not collapse the way salience does but does not solve the task either. H2 (§10.10). Robustness of PPO: one seed each, both unstable.

*Provenance.* Only the four `rl_bc_workflow` cells were rerun after fix E17 (their rows carry `hindsight_diverged_at`; the other 32 cells' rows do not and were resumed from the earlier run, `runs/exp5b_workflow_rl.rerun.log`). E17 changes only hindsight-derived metrics and imitation labels, so the task-success numbers of the other cells are unaffected, but their `requirements_destroyed` values are post-divergence-contaminated and should not be quoted.

*Follow-up.* Three or more seeds for each PPO variant; a tracked-hindsight oracle that re-plans from the diverged stream (snapshot/restore exist on the environment interface) to restore a valid reference after restarts.

### 10.10 H2: discount factor

On the recall task, means over three seeds at horizon 500 are 0.612 (γ = 0.995) vs 0.661 (γ = 0) at 10 % and 0.363 vs 0.405 at 5 %; at horizon 200, 0.595 vs 0.612 (10 %) and 0.437 vs 0.381 (5 %). The sign changes across cells and the differences are within the seed spread (up to 0.13). On the workflow task, one seed each, the immediate-reward run ended far higher (0.992 vs 0.404 at 5 %) and both learning curves were unstable. H2 is not supported by this evidence and the recall task is a weak test of it: the immediate signal ("do not evict what the current query needs") already identifies the item class worth keeping. The workflow task is where the two rewards *should* disagree, and there the comparison has one seed per arm.

### 10.11 Experiment 6: real benchmark data, no task model

On LoCoMo at 10 %, `evidence_complete_rate` is 0.016 (FIFO delete), 0.152 (salience), 0.417 (FIFO archive + BM25 top-5), 0.614 (exact delete-only oracle), 1.000 (just-in-time oracle); BM25 recall 0.35–0.38, precision 0.06–0.09. The exact delete-only oracle reaches 1.000 only at 50 %, consistent with evidence being ~31 % of LoCoMo tokens. On LongMemEval-S at 5 %, salience 0.49 and FIFO archive + BM25 0.59 against FIFO delete 0.05; BM25 recall 0.65. On 6c the dense embedder roughly doubles the hashing embedder's recall (0.215 vs 0.115) but stays below BM25 (0.347). The adapters, evidence tracking and retrieval metrics run on real data and the synthetic ordering (content > recency, retrieval > deletion) reappears. Not shown: QA accuracy (no task model fits the 4 GB card), anything about long-horizon control, and any meaningful reason for salience on LongMemEval (evidence sits in short user turns; assistant turns are long; salience prefers user turns). Ten LoCoMo episodes cannot resolve close differences.

### 10.12 Experiment 7: a real language model in the loop (smoke tests)

7a: llama3.1:8b (4-bit, Ollama, CPU) as task model on 3 episodes / 11 queries answered every question whose evidence was in context and none whose evidence was not, so task success equals `needed_hit_rate` for every controller and every failure is `evicted`; model time 26–166 s per 60-step episode. 7b: the same model as controller on 2 episodes / 5 queries: 6 of 18 replies parsed (the model listed every item and hit the 120-token limit), the fallback evicted oldest-first on the rest (15.5 forced evictions per episode against 0 for every other controller), task success 0.50, 824 s per 40-step episode. Both are pipeline checks and say nothing about the models or the controllers.

## 11. Hypothesis status

| Hypothesis | Status | Evidence | Caveats |
|---|---|---|---|
| **H1** A learned controller beats static heuristics at equal budget | **PARTIALLY SUPPORTED** | Beats every recency rule by +0.24–0.49 (Exp. 4); matches salience within MDE at 10–25 %, trails it by 0.02–0.06 at 5 %; BC+PPO +0.018 [+0.014, +0.023] at 25 %/h500 only; solves the workflow task (1.000) where every heuristic fails (Exp. 5b) | "Best heuristic" is task-specific; no single learned policy was shown best on both tasks; recall ceiling is information-limited; workflow success is one task and one seed per PPO arm |
| **H2** Long-horizon reward beats immediate reward | **NOT SUPPORTED** | γ 0.995 vs γ 0 within seed noise on recall (3 seeds each); γ 0 far ahead on workflow (1 seed each), both unstable | Recall is a weak test; workflow needs seeds and a stabler optimiser; no negative result either |
| **H3** Advantage of good management grows with horizon, tighter budget, longer dependencies | **SUPPORTED (for headroom)** | Gap best-heuristic-to-oracle 0.235 → 0.575 as horizon 200 → 2,000 at 800 tokens (1c); recency 0.50 → 0.04 vs oracle 1.0 as `gap.min` 5 → 250 (1d); all gaps grow as budget shrinks (1) | Shows room, not that a method captures it; synthetic generator only |
| **H4** Archive + retrieval beats deletion under uncertainty | **SUPPORTED, with a retrieval qualifier** | FIFO 0.233 → 0.734 at 5 % (Exp. 2); oracle 1.000 at 2 % vs delete-only optimum 0.871; ordering holds on LoCoMo / LongMemEval evidence metrics | Value is entirely in the retriever: recall 0.01–0.10 on workflow (Exp. 5) and 0.35 on LoCoMo; archive unpriced and unlimited |
| **H5** Compaction / consolidation improve the frontier when memories are redundant | **PARTIALLY SUPPORTED (conditional)** | Extractive compaction: FIFO +0.17 at 10 %, salience +0.00; truncate hurts both and is attributed; consolidation ±0.01–0.02; naive consolidation hurts salience −0.115, tight recovers | Extractive rewriter matched to the generator; no LLM rewriter run; negative results are the robust part |
| **H6** A lightweight JEV policy trades decision quality for latency | **BLOCKED** | Adapter, batching, repair, cost accounting implemented and tested with a labelled FAKE client; real client never run | Needs `JEV_API_KEY`; response parsing unverified |
| **H7** Policies transfer across budgets, horizons, base models, tasks | **PARTIALLY SUPPORTED / INCONCLUSIVE** | Across budgets: yes within the training range; across horizons: imitation holds at 2.5× training horizon, PPO loses ~0.1 at 5 %; across tasks: recall-trained policy gives 0.042 on workflow (FIFO-like, not salience-like); across base models: not tested | Budget transfer was inside the trained range; only scripted task models |

## 12. Failure analysis

### 12.1 Observed failure modes

| mode | where observed | evidence |
|---|---|---|
| Bad eviction | every delete-only cell | all 380,231 Exp. 1 failures are `evicted`; recency rules at long gaps (1d) |
| Failed retrieval | Exp. 2 (remaining failures), Exp. 5 (all archive failures), Exp. 6 | `archived_not_retrieved`; BM25 recall 0.56–0.69 (recall task), 0.01–0.10 (workflow), 0.35 (LoCoMo) |
| Archive–retrieval bottleneck | Exp. 2, and Phase 2 before it | once archiving exists, FIFO and salience are 0.06 apart; 7/8 of evidence reaching the model in Phase 2 got there by search |
| Task-model reasoning failure | Exp. 1b (injected), 7a (none observed) | `task_model_reasoning` recovered at 0.199–0.208 |
| Naive consolidation losing information | Exp. 3 | salience 0.725 → 0.610; merged item evicted whole; `consolidation_incorrect` 2–51 per cell with truncate |
| Lossy compaction | Exp. 3 | 1,368–1,443 `compression_lost_detail` per 10 % cell with truncate |
| Imitation copying wrong action semantics | Exp. 4b | `rl_bc_archive` fails by `evicted` (924) not by retrieval; untrained retrieval head does better |
| Fallback behaviour | `no_controller`, 7b | 450 forced evictions per 500-step episode at 10 %; 15.5 per 40-step episode for the prompted LLM |
| Oracle blindness under changed trajectories | Exp. 5b | oracle 0.808 with 25 restarts per episode at 5 %; beaten by `rl_bc_workflow` |
| PPO instability / drift | Exp. 4, 5b | seed spread 0.13; final < best-validation by 0.02–0.06; workflow γ 0.995 run drifted from 0.416 to 0.088 validation |
| Model / hardware | INFRA_REPORT.md | Qwen3.5-4B needs ~12 GB; 4 GB card runs only ≤ 1B at 16-bit or 4-bit ≤ 3,900-token prompts |
| LLM smoke-test limits | Exp. 7 | 5–11 queries; 824 s per episode on CPU; unparsable JSON |

### 12.2 The four defects found in code review

An independent reviewer read the harness after the first experiments had run. Four defects were confirmed and fixed the same day (SCIENTIFIC_VALIDITY_REPORT.md §4b; the fixes are in the working tree that became commit `4fe0f7c`).

| # | defect | experiments affected | fix | rerun |
|---|---|---|---|---|
| 1 | The forced fallback **deleted items retrieved at the same step** when the retrieval did not fit, turning an unaffordable retrieval into a permanent loss of archived evidence. Only the archive-mode oracle and the RL controller triggered it (heuristics budget their retrievals). | `oracle_approx` cells of Exp. 2, 5, 6; RL cells of Exp. 4b | Fallback first returns this step's retrievals to the archive, oldest first, then evicts (E16; `MemoryEngine.enforce_budget`) | Exp. 2, 4b, 5, 6a, 6b, 6c rerun (`runs/*.rerun.log`); pre-fix Exp. 5 kept in `runs/_superseded/exp5_workflow_before_fallback_fix` |
| 2 | **Action counts mixed units**: one per action for controllers, one per item for the fallback, so `forced_share_of_removals` and the actions table were not comparable | tables only | everything counts items | tables regenerated |
| 3 | On the workflow task, **hindsight-derived metrics kept being computed after divergence** from the reference pass (`requirements_destroyed`, `unnecessary_token_share`, regret rows, imitation labels), describing a different episode | Exp. 5, 5b columns; `rl_bc_workflow` labels | divergence detected by content hash; metrics `null` from then on; labels after divergence dropped (E17) | `rl_bc_workflow` retrained (`runs/_superseded/rl_bc_workflow_prefix_labels` is the old one) and its Exp. 5b cells rerun; **other Exp. 5b cells were not** (§10.9) |
| 4 | `invalid_action` was assigned whenever *any* controller action at the step was rejected, related to the lost item or not | no recorded run (no rejected actions occurred) | requires a rejected action targeting the lost item, or a step on which every controller action was rejected | none needed |

Two smaller corrections: `retrieved_but_ignored` now means retrieved at that step, and the query observation no longer counts as an "unnecessary" token. The reviewer also confirmed the ILP's optimality by brute force on 40 instances, that recorded RL log-probabilities are reproduced exactly, that seed sets are disjoint, and that no controller can reach ground truth.

### 12.3 Provenance gaps found while preparing this note

- Every run's `metadata.json` records commit `d215c8f` (the scaffold commit) with `dirty: true`. The code that produced them is the tree committed later as `4fe0f7c`, but nothing in the run folders proves which intermediate state of that tree produced which cell.
- 32 of 36 Exp. 5b cells hold episode rows from before fix E17 (their rows lack `hindsight_diverged_at`); task success is unaffected, hindsight-derived columns are not.
- `oracle_solve_s` is absent from Exp. 1–1d, 2, 3 (partly) and 5 rows; the per-episode solve times quoted for 1c come from cell wall times in the sweep log.
- The policy parameter count quoted in EXPERIMENTS.md (~12,000) disagrees with the training metadata (27,526 DeepSets / 19,334 MLP).
- Earlier attempts at `rl_ppo`, `rl_ppo_gamma0` and `rl_bc_ppo` exist in `runs/_superseded/*_first_attempt`; the reported ones are the second attempts.

## 13. What we have actually learned

Ordered from most to least robust.

1. **Memory policy materially affects task success under a bounded budget.** Robust: 0.17–0.76 across non-oracle controllers at one budget; paired intervals far from zero; holds at every sub-100 % budget and on three horizons, and the ordering reappears on real LoCoMo and LongMemEval data. Generator-specific in its *magnitudes* only.
2. **The three failure sources can be separated, and must be, before a controller is blamed.** Robust as a property of the framework: 100 % `evicted` with a perfect reader; injected noise recovered; relabelling to `archived_not_retrieved` when archiving is enabled; `compression_lost_detail` under a lossy rewriter.
3. **Archive plus retrieval can matter more than fine differences between eviction heuristics, and then the retriever is the bottleneck.** Robust on the recall task and on the QA benchmarks' evidence metrics; *retriever-dependent*: it vanished on the workflow task with the same BM25 retriever.
4. **No fixed heuristic examined is best across task structures.** Robust as a two-task existence proof (salience and FIFO swap places between Experiments 1 and 5). Whether this generalises to real tasks is a hypothesis.
5. **Learned policies have a plausible motivation and a demonstrated ceiling.** Robust: imitation matches the best rule on the recall task from 240 episodes and solves the workflow task where no rule does. Not shown: one policy good on both; any advantage over the best task-specific rule on its own task beyond +0.018.
6. **Hindsight supervision is useful and can encode the wrong action semantics.** Robust: DAgger from the hindsight expert is ~10× more sample-efficient than PPO on the recall task; with an archive available the same expert teaches deletion and the learner loses evidence it could have archived. The remedy (cost-aware expert or priced deletion) is untested.
7. **Optimal eviction is not optimal memory management.** Robust as an argument (§4.7) and as an observation (4b); not yet a quantified gap.
8. **Sequential tasks expose what QA benchmarks cannot.** Robust on one synthetic task: compounding restarts, ranking flips, oracle blindness after divergence. Hypothesis that real long-horizon tasks behave alike.
9. **Long-horizon versus immediate reward made no measurable difference** on the recall task and an unreplicated large difference in the wrong direction on the workflow task. Inconclusive.
10. **Everything above was measured with a scripted task model, a lexical retriever, an unpriced archive and hand-set generators.** Each is a stated confound, not a footnote.

## 14. Statistical and scientific validity

1. **Synthetic-generator dependence.** Salience's weights match the generator's `query_prob` per source; the extractive rewriter's "specific" matches the generator's facts. Heuristic rankings on the recall task are properties of the generator. Experiment 5 shows the same rules failing elsewhere. Never quote a heuristic's rank on the recall task as a property of the heuristic.
2. **Unknown achievable ceiling.** Which facts get queried is a Bernoulli draw per fact, so a causal policy cannot reach the oracle; "fraction of oracle gap closed" has an unknown maximum below 1. A Bayes-informed reference policy was not built.
3. **Perfect task model.** Task success is a function of memory contents. A language model adds reasoning failures (attributable) and changes *which* memory states succeed (not attributable in advance). Experiment 7 is a smoke test.
4. **Fraction budgets confound horizon with absolute budget** (Exp. 1); 1c fixes tokens but confounds horizon with dependency distance; 1d isolates distance.
5. **The oracle is a reference, not always a bound.** Exact only for delete-only; approximate is behind exact by up to 0.115 with unequal sizes; neither compacts or consolidates; blind after divergence on the workflow task.
6. **Access counts are exact only with scripted agents** (LRU, LFU, `access_count` feature); the LLM agent approximates them by string containment.
7. **Retrieval was lexical or hashing-based everywhere except 6c**, where a small dense model trailed BM25. Retrieval findings are findings about these retrievers.
8. **The archive is free and unlimited.** No run priced archive tokens, retrievals or controller compute; comparisons are at equal ACTIVE budget, not equal cost.
9. **Statistics.** Episodes are the unit; comparisons are paired; 95 % bootstrap intervals and MDEs are reported per cell. No correction for hundreds of comparisons. LoCoMo has 10 episodes. Treat a difference as real only if it is large relative to its MDE and consistent across budgets and horizons.
10. **PPO seeds.** Three for two variants on the recall task; one for everything else. Seed spread (0.13) exceeds most of the differences of interest.
11. **Model-free tokens.** Budgets count words and punctuation; the ratio to model tokens varies with the text.
12. **Latency** was measured on a saturated machine and excludes the ILP solve.
13. **Action-space mismatch** is controlled by `allowed_operations` but the *expert* was not matched to the action space (4b), and no exact oracle exists beyond delete-only.
14. **Reward and hindsight leakage.** Controllers cannot import hindsight (test); the `hindsight_eviction_regret` term has weight 0 everywhere; seeds are disjoint. The `rl_bc*` validation curves used test-overlapping seeds but selected nothing.
15. **Public benchmarks** may have been seen by any language model used later; irrelevant to scripted runs.
16. **Hardware-driven model choice.** The task and controller models in Experiment 7 were chosen because they run on a CPU, not for any scientific reason.
17. **Dirty-git runs.** Every run in `runs/` is stamped `d215c8f` + `dirty: true`, and the code changed while experiments ran (SCIENTIFIC_VALIDITY_REPORT.md §7 lists the behaviour-affecting changes and argues none altered the numbers of runs made before them). **Runs stamped dirty must not be treated as final thesis numbers until rerun from a clean commit.** The configs and seeds are unchanged and the full rerun takes about two hours of CPU.

## 15. Infrastructure limitations

| limitation | fact | how it constrains or biases the results |
|---|---|---|
| 4 GB GPU | RTX 3050 Laptop; Qwen3.5-4B needs 8.4 GB at 16-bit, 3.4 GB in 4-bit and then only ≤ 3,900-token prompts; nothing above ~1B runs at 16-bit | every language-model experiment beyond a smoke test is blocked; all reported results use scripted task models; H1–H5 are untested under reasoning noise |
| Local models on CPU | llama3.1:8b via Ollama, ~5 s per short prompt idle, far slower under load; 824 s per 40-step episode as controller | Experiment 7 cannot be scaled; LLM-controller quality is unknown; the prompted controller's JSON failures may be a prompt/size artefact |
| API-gated judge | LongMemEval's official metric is an LLM judge; none configured | QA accuracy on LongMemEval would be local token-F1, which under-credits paraphrases |
| JEV key absent | `JEV_API_KEY` not set; client written from documentation, never run | the JEV arm (one of the three thesis arms) has no data; response parsing unverified |
| Dense retrieval on CPU | ~20 ms per text idle; ~800 texts per LoCoMo episode | fine for 6c; too slow to put dense features in RL training on CPU |
| Exact ILP cost | 0.3 / 0.8 / 7.5 s per episode at horizons 500 / 1,000 / 2,000 and 800 tokens; 60 s limit never hit | longer horizons or tighter budgets will fall back to the approximate plan (recorded in `oracle_method`) |
| LongMemEval-S memory | each sweep worker parses a 277 MB file (~1.5 GB of objects) | 4 workers max on this machine |
| Contention | 10–13 sweep workers plus up to 10 trainings on 16 cores | latencies are order-of-magnitude; PPO wall times not comparable across runs |
| Process-pool leak | killing a sweep can leave workers writing into cell folders | happened once (6c); cells deleted and rerun |
| Root-owned `__pycache__` from Docker | harmless | – |
| Dependencies | `transformers` not in `.venv` (`hf` backend blocked there); Docker image carries the CUDA stack and passed 148 tests before the review fixes | the GPU path is ready for a rented card but was not exercised after the fixes |
## 16. Related work

The literature search was run on 2026-09-30 over arXiv, the ACL Anthology, OpenReview, PMLR/NeurIPS/ICML/ICLR proceedings, USENIX/ACM/IEEE and Semantic Scholar, by four parallel surveys (agent memory architectures; RL-trained memory managers; KV-cache eviction and learned cache replacement; benchmarks and privileged-expert imitation), then synthesised. Every citation below was checked against a primary page during that search; where a venue could not be confirmed it is marked "(venue unverified)". The full synthesis, with about 190 sources, is kept alongside this note as `docs/research/RELATED_WORK_LITERATURE_REPORT.md`; this section is the condensed version. Items the search could not locate (AgentSM, RLMem, LTM-Bench, a paper titled "Learning to Forget" for agents) are not cited here.

**The one-line finding of the search:** every component of this project exists somewhere in the literature, but no published system combines them. The novelty risk is combinational, not component-level. The closest neighbours are all from 2026.

### 16.1 Per-paper records

Fields: problem and method · memory action space · training signal · benchmarks · what it solves / leaves open · how memctl differs.

**Architectural and prompted systems**

| Paper | Problem and method | Action space | Signal | Benchmarks | Solves / leaves open | memctl differs |
|---|---|---|---|---|---|---|
| **MemGPT**, Packer et al., arXiv 2310.08560, 2023 ([link](https://arxiv.org/abs/2310.08560)) | fixed context window; OS-style paging between main context and recall/archival storage, LLM-callable functions | core append/replace, archival insert/search, recall search; eviction is an *automatic* FIFO flush with recursive summary at a token threshold | none (prompted) | DMR, conversation opener, multi-doc QA, nested KV (all retrieval over fixed history) | solves the appearance of unbounded context; leaves the eviction *decision* to a fixed rule; no oracle, no sequential task | eviction is a learned per-item choice by a separate controller; FIFO+summary is our baseline |
| **A-MEM**, Xu et al., NeurIPS 2025 (per arXiv), 2502.12110 ([link](https://arxiv.org/abs/2502.12110)) | rigid memory ops; Zettelkasten notes with links and "evolution" | note, link, evolve, retrieve; no deletion, no budget | none | conversational QA | learned structure; leaves retention untouched | we measure the retention decision it never makes |
| **Mem0** (2504.19413), **Zep** (2501.13956), **HippoRAG** (NeurIPS 2024, 2405.14831), **ECHO** (2608.21755) | extract/consolidate/retrieve from an unbounded store | write / update / invalidate / retrieve; nothing evicted | none | LoCoMo, LongMemEval, DMR; ECHO's own audit finds Mem0 ahead on a matched sample and leakage in its query-expansion rules | retrieval quality; no active-budget constraint | archive+retrieve is one of our actions, competing with evict under a hard budget |
| **Generative Agents** (Park et al., 2304.03442, UIST 2023 venue unverified), **MemoryBank** (Zhong et al., 2305.10250, AAAI 2024 unverified) | recency/importance/relevance retrieval; Ebbinghaus forgetting | append, reflect, retrieve; MemoryBank forgets by decay | none | believability sandbox; companion chatbot | source of our age-decay, similarity and salience heuristics | we treat these scores as baselines and show their ranking flips between tasks |
| **ACE**, Zhang et al., ICLR 2026 (per arXiv), 2510.04618 ([link](https://arxiv.org/abs/2510.04618)) | names *brevity bias* and *context collapse* under iterative rewriting; incremental delta updates by generator/reflector/curator | append/edit playbook bullets; no budget, no archive | none (execution feedback) | AppWorld, FiNER, Formula | avoids lossy rewriting | we make lossy rewriting an action and attribute its failures (`compression_lost_detail`) |
| **LLMLingua** (EMNLP 2023, 2310.05736), **Sculptor** (2508.04664), **sleep-time compute** (2504.13171) | prompt compression under a budget; summary/hide/restore tools; offline consolidation | compact; hide/restore (an archive primitive) | small compressor LM / none | GSM8K, BBH; stateful GSM | compaction and archive primitives exist; no learned choice among them | COMPACT, COMPACT_AND_ARCHIVE, CONSOLIDATE are learned alternatives to EVICT |
| Surveys: Du 2603.07670; Hu et al. 2512.13564; Jiang et al. "Anatomy of Agentic Memory" 2602.19320; Zhang et al. 2404.13501 | — | — | — | — | list "estimating importance without future information", "learning to forget", fragile rule-based consolidation, under-scaled benchmarks and overlooked costs as open problems | these are the problems the framework is built to measure |

**RL that trains the task model (2025–2026)**

| Paper | Problem and method | Action space | Signal | Benchmarks | Solves / leaves open | memctl differs |
|---|---|---|---|---|---|---|
| **Memory-R1**, Yan et al., ACL 2026, 2508.19828 ([link](https://arxiv.org/abs/2508.19828)) | heuristic pipelines lack a learned write policy; Memory Manager LLM + Answer Agent, both fine-tuned (7–8B) | ADD / UPDATE / DELETE / NOOP on a bank; answer agent distils 60 candidates | PPO / GRPO, outcome EM, 152 QA pairs | LoCoMo, MSC, LongMemEval | learned writes from few labels; no budget, no oracle, no sequential task | 27k-parameter separate controller; EVICT is measured for what it destroys; hard budget; ILP oracle |
| **MemAct**, Zhang et al., 2510.12635 (venue unverified) ([link](https://arxiv.org/abs/2510.12635)) | context curation as part of the agent's action space; Prune&Write; DCPO segments trajectories at edits | delete-set + write; no archive | GRPO, terminal +1, −0.1 over 20K tokens or 40 steps | multi-hop QA, BrowseComp-Plus | shows fixed pruning underperforms learned pruning; own limits: "information deletion becoming relevant later", lossy summaries, future "tiered caching" | MemAct's future work is our present design; we measure that risk as `evicted` vs `archived_not_retrieved` |
| **AgeMem**, Yu et al., ACL 2026, 2601.01885 ([link](https://arxiv.org/abs/2601.01885)) | unified LTM+STM control as tools of the task LLM; argues *against* "auxiliary controllers" | LTM Add/Update/Delete; STM Retrieve/Summary/Filter (largest published set) | step-wise GRPO, terminal reward broadcast | ALFWorld, SciWorld, PDDL, BabyAI, HotpotQA | sequential eval; soft budget; no oracle, no attribution | closest action-set overlap; its motivation is the argument our design must answer empirically |
| **Memex(RL)**, Wang et al., 2603.04257 ([link](https://arxiv.org/html/2603.04257v1)) | "keep the active state small but do not throw evidence away"; indexed archive + exact dereference; 30B task model | CompressExperience (archive+summary), ReadExperience; no plain evict | GRPO, R_task − context − redundancy − format penalties, 8K soft threshold | modified ALFWorld where IDs must be recovered from memory: 24.2 → 85.6 % | nearest published analogue of our workflow task; own limits: one env, unvalidated sufficiency assumption | controller is the task model; no heuristic sweep, no oracle, no attribution; our "retriever becomes the bottleneck" matches its rising read calls |
| **MEM1** (ICLR 2026, 2506.15841), **MemAgent** (ByteDance, ICLR 2026 oral, 2507.02259), **Mem-α** (2509.25911, ICLR 2026 unverified), **ReSum** (2509.13313), **Context-Folding** (ICML 2026, 2510.11967) and successors | overwrite-state, fixed 1,024-token memory, tiered insert/update/delete, rule-triggered summarisation, branch-and-fold | coarse or implicit eviction; no per-item archive | PPO / DAPO / GRPO, outcome or shaped | multi-hop QA, WebShop, RULER, MemoryAgentBench, SWE-Bench | budget by construction or penalty; task model trained | items stay addressable; per-item choice; frozen task model |
| **HiMPO** (2606.16285), **Mem-T** (2601.23014), **Memory-R2** (2605.21768), **UMA** (EMNLP 2026, 2602.18493) | hindsight / counterfactual *credit* for memory writes inside RL of an LLM writer; Memory-R2 names "memory turns the agent's past actions into part of its future environment" and uses local re-rollouts | write / ADD-UPDATE-DELETE / CRUD | GRPO variants with hindsight-gated or re-rollout credit; Mem-T also supervised on hindsight-scored demos | LoCoMo, search QA, Ledger-QA | dense credit; Mem-T needs gold evidence | we use a hindsight *policy* as a DAgger teacher for a non-LLM controller; Memory-R2's observation is the published form of our oracle-blindness finding and its re-rollouts are the remedy we have not applied |

**Separate lightweight controllers over frozen task models (2026)**

| Paper | Problem and method | Action space | Signal | Benchmarks | Solves / leaves open | memctl differs |
|---|---|---|---|---|---|---|
| **LRE**, Lia & Mazumder, 2606.20954 ([link](https://arxiv.org/abs/2606.20954)) | "an eviction policy runs at write time and must decide what to keep before the query that needs it exists"; kilobyte logistic-regression scorer over causal features; verbatim keep within a strict 2048-token budget | keep / evict only | supervised, or hindsight self-labels (identifier reused ≥ 3 later steps; ≥ 40 % overlap with the answer); self-labels give 95 % of supervised | AppWorld (41.1 vs 44.0 keep-everything, 14 tasks no other policy solves), LoCoMo/LongMemEval-S | tiny budgeted eviction; its case study of abstractive compression breaking identifiers is our `compression_lost_detail`; no archive, keep-everything as the only reference | **closest overall**; we add archive/retrieve/compact/consolidate, a set-level DeepSets policy, DAgger against an oracle, PPO, an exact ILP reference and attribution; "small learned scorer" is LRE's, not ours |
| **MemCon**, Jiang et al., 2607.13591 ([link](https://arxiv.org/abs/2607.13591)) | memory ops as an MDP; tabular UCB bandit over a discretised state, "table lookup, not a second LLM invocation", frozen task LLMs | Retrieve variants, PlanInject, Re-Retrieve, Consolidate, Forget, NoOp | episode-end reward 1.0·success + 0.3·(1−T/T_max) − 0.5·failure, reverse-discounted | ALFWorld 67.9 vs 59.7, PDDL, ScienceWorld, GAIA | separate non-LLM controller on sequential tasks; store-side ops, no hard active budget; oracle absent in the fuller extraction (conflicting) | per-item retention under a hard budget; 24-feature set state; archive/retrieve pair the controller fills; oracle; attribution |
| **EMBER**, Li, Banerjee, Che, 2606.05894 ([link](https://arxiv.org/html/2606.05894)) | budgeted evidence retention; separate 7–14B writer, frozen retriever and reader; B_ret ∈ {512…8192} | insert / merge / overwrite / skip | GRPO, answer-gated evidence chain − budget penalty | LongMemEval-RR, RULER-HotpotQA, MultiQ; baselines Random/Recency/salience + **oracle retention** bound; Retain-Recall vs Read-Recall | budget sweep with oracle bound and a placement/retrieval split; QA only; own limit: stale evidence needs deletion controls | closest *protocol* to ours; LLM-scale writer; oracle is gold packing, not furthest-next-use ILP; no archive, no sequential task, no imitation |
| **TRACER**, Lin et al., 2608.29363 ([link](https://arxiv.org/abs/2608.29363)) | "compression–consequence gap"; lightweight REINFORCE policy sets per-tool retention ratios; learned counterfactual outcome model vs full retention | continuous retention ratio per tool output | success + tokens + re-invocations | enterprise sessions, LOCA-bench; 29–46 % token cut | separate RL policy with consequence-aware reward on sequential sessions; no archive, soft token objective | discrete typed actions incl. reversible archive; hard budget; oracle |
| **OSL-MR**, Kang et al., 2606.10616 ([link](https://arxiv.org/abs/2606.10616)) | retention as constrained optimisation (NP-hard); "online-observable features, offline-available supervision" | retain / drop | offline privileged supervision, online features | LoCoMo, LongMemEval | the same privileged-teacher / observable-student structure as our DAgger arm | we add archive and sequential tasks and measure the teacher's failure |
| **SOLAR**, Sun, Cao, Lam, 2607.00394 ([link](https://arxiv.org/abs/2607.00394)) | learning-augmented eviction for semantic buffers; competitive ratio ≤ 3, regret O(√(KT log T)) | evict | online, regret bound | MemoryBench: LRU/LFU "consistently underperform FIFO on semantic workloads" | the only paper reporting our heuristic-ranking instability, with a theoretical reason | our flip is task-structure-driven; we should cite SOLAR as corroboration, not claim discovery |
| **AdaCoM** (2605.30785), **MemSifter** (2603.03379), **Hindsight Memory-PRM** (2608.29605), **provider-routing MemAgent** (2609.32521), **ACON** (ICML 2026, 2510.00615), **TraceRetain** (2606.29178, ICML 2026 per note), **MeClear** (2609.09115) | 4B–8B LLM managers/retrievers; delete-and-re-answer interventional credit; hindsight-best routing with oracle 89.6 vs 71.0; contrastive compression guidelines; CEM-weighted eviction score; Shapley suppression | rewrite/merge/delete; select sessions; Write/Merge/Noop; route; compress; evict; reversible suppress | GRPO, marginal-utility RL, reward-filtered SFT, CEM, attribution | BrowseComp-Plus, LoCoMo, GAIA, AppWorld, ALFWorld | AdaCoM shows the optimal policy depends on the frozen task model; Memory-PRM's intervention is a one-action-deviation roll-out; TraceRetain's null on clean streams warns that policy spreads must be reported across budgets and noise | LLM-scale or single-purpose; none has our seven-way action set, hindsight-policy teacher, ILP oracle and seven-label attribution together |

**Oracle diagnostics and attribution (2026)**

| Paper | Contribution | Relation to memctl |
|---|---|---|
| **MemAudit**, Bhargava & Sobral Barrento, 2605.02199 ([link](https://arxiv.org/html/2605.02199)) | exact MILP package oracle for budgeted memory writing; package ratio vs OPT; "a certified upper bound for diagnostic purposes, not a prescriptive deployment algorithm" | same construction as our exact ILP oracle (§4.4); we additionally use the oracle as a DAgger teacher and find it is the wrong target once archiving exists, which MemAudit's remark anticipates without testing |
| **Restore-counterfactual audit**, Shen, 2609.08279 ([link](https://arxiv.org/abs/2609.08279)) | reinstate needed evidence at read time; split failures into irreversible (evicted) / recoverable (retrieval miss) / residual; irreversibility 1.00 for every policy at 8k tokens on LongMemEval-S | prior three-way attribution; ours is a seven-way refinement on a learned controller and a sequential task; its 8k result predicts our "at tight budgets only an archive helps" |
| **MemTrace**, Deng et al., 2605.28732 (ID conflicts with 2606.17328 in one note; verify) | operation-level error attribution (information loss vs retrieval misalignment) across Mem0/RAG/long-context | may overlap `task_model_reasoning`; undetermined from the abstract |
| **Epistemics of Agent Memory**, 2609.33013 | keep/compress/abstract/forget under a token budget; ConsolidationBench, oracle-by-construction | closest work on learned CONSOLIDATE with an oracle benchmark |
| Rate–distortion view (2607.08032); "Remember the Decision" (2605.10870); "What Does Context Compression Cost an Agent?" (2608.16370); ForgetEval (2606.15903) | irreversible compaction discards "before the query is known and with no way to undo it"; error grows super-linearly with compaction events while reversible retrieval-backed memory stays flat; compression induces re-retrieval loops invisible to task metrics | the theoretical case for our ARCHIVE-vs-EVICT distinction; our restarts are the sequential form of the hidden re-retrieval cost |

**Benchmarks**

| Benchmark | Shape | Evidence labels | Notes | Relation |
|---|---|---|---|---|
| **LoCoMo**, Maharana et al., ACL 2024 ([link](https://aclanthology.org/2024.acl-long.747/)) | QA after a multi-session conversation | turn ids per QA | paper reports 50 conversations / 7,512 questions; released `locomo10` has 10 conversations; an independent audit finds 6.4 % of the answer key wrong and the judge accepting 62.8 % of wrong-but-adjacent answers ([audit](https://dev.to/penfieldlabs/we-audited-locomo-64-of-the-answer-key-is-wrong-and-the-judge-accepts-up-to-63-of-intentionally-33lg)); Zep–Mem0 dispute over category 5 | our adapter; evidence-retention only (Exp. 6); the ~7-point noise floor bounds any future judged comparison |
| **LongMemEval**, Wu et al., ICLR 2025 ([link](https://iclr.cc/virtual/2025/poster/28290)) | one question after ~40 sessions (S) or ~500 (M) | `answer_session_ids`, per-turn `has_answer` | GPT-4o judge; MemDelta (2606.29914) shows rankings flip on the embedding model alone | our adapter (S split, first 100) |
| MemoryAgentBench (ICLR 2026), MemBench, BEAM, EverMemBench, LoCoMo-Plus (ACL 2026), LongMemEval-V2, AMA-Bench, LongBench/RULER/NoLiMa/BABILong/HELMET | ingest-then-probe | varies | forgetting only lowers a score | scope boundary: none tests control |
| **MemoryArena** (ICML 2026, 2602.16313), **Mem2ActBench** (ACL 2026, 2601.19935), **MERIT** (2609.05441), **MemGym** (2605.20833), **PM-Bench** (COLM 2026, 2607.12385) | forgetting changes later actions: interdependent subtasks; memory grounds tool parameters; marginal utility on dependent tasks; a gym with memory-isolated scores and a reward model; prospective memory | partly | MemoryArena: LoCoMo-saturated agents "perform poorly"; MERIT: swapping memory moved success by up to 60 points and agents acted on correctly retrieved values only 55 % of the time | the comparison class for our workflow task; none has been run here; MERIT's 55 % is the published form of `retrieved_but_ignored` and its 60-point swing the analogue of our 0.6 spread |

**Caching, Belady, KV eviction, imitation theory**

| Paper | Contribution | Relation |
|---|---|---|
| **Belady 1966** (IBM Syst. J. 5(2)); **Hawkeye** (Jain & Lin, ISCA 2016); **Glider** (Shi et al., MICRO 2019); **Mockingjay** (Shah et al., HPCA 2022, venue inferred) | MIN evicts the furthest next use; needs ~8× cache lookahead; binary friendly/averse labels → reuse-distance ranking labels because binary labels flip on small errors | our approximate oracle *is* MIN; the binary-to-ranking progression is a warning that oracle labels should be rankings or acceptable sets (our imitation loss uses acceptable sets) |
| **Parrot**, Liu et al., ICML 2020 ([link](http://proceedings.mlr.press/v119/liu20f/liu20f.pdf)) | DAgger imitation of Belady at miss states; on-policy collection +9.8 % on average but "highly program-dependent"; ranking loss beats log-likelihood on the argmax by 3.5 %; possible only because the access stream "is independent of the action" | the direct methodological ancestor of our DAgger arm; its program-dependence predicts our task-dependent imitation result; its exogenous-stream assumption is exactly what our workflow task violates |
| **LRB**, Song et al., NSDI 2020 | exact MIN imitation impractical; relaxed "Belady boundary", delayed labels, good-decision ratio; 9–13 % more misses than MIN | an online-available labelling scheme we could adopt |
| **LeCaR** (HotStorage 2018), **Cacheus** (FAST 2021) | regret-minimising mixture of LRU/LFU experts; LeCaR underperforms ARC/LIRS on many workloads; workload primitives decide the ranking | our heuristic flip is the workload-primitive finding; an online mixture over our own heuristics is an unrun baseline |
| **Demand-MIN**, Jain & Lin, ISCA 2018 ([link](https://dl.acm.org/doi/10.1109/ISCA.2018.00020)); "Beyond Belady" (2212.13671) | with a prefetch channel MIN no longer minimises demand misses; with variable sizes an online policy can beat MIN's byte miss ratio | the published precedent for "imitating the deletion oracle is wrong once a fetch channel exists"; our finding should be stated as its memory-control instance |
| **ForesightKV** (ICML 2026, 2602.03203), **KVP** (ICML 2026, 2602.10238), **KVpop**, **DistillCache** (MLP policy, REINFORCE on KL), **MemDecay** (half-lives 148–189 steps for system tokens vs 14–16 for scratchpad) | future-attention "Golden Eviction" labels → ranking SL → GRPO; per-head RL on future utility; feature-based MLP eviction policy; heterogeneous item lifetimes | ForesightKV is the token-level twin of our imitation-then-PPO recipe; DistillCache the closest architectural analogue; MemDecay supports per-item type features; none has an archive |
| H2O, Scissorhands, StreamingLLM, FastGen, SnapKV, PyramidKV, Keyformer, TOVA (2023–24; several venues unverified); Quest, InfLLM | past-attention heuristics; "pivotal tokens stay pivotal" assumption; Quest/InfLLM never delete, they offload and re-fetch | KV-level analogues of our heuristics and of archive+retrieve |
| **Weihs et al.**, NeurIPS 2021 (ADVISOR) ([link](https://arxiv.org/abs/2007.12173)); **Cai et al.**, NeurIPS 2024; **Swamy et al.** 2022 (venue unverified); Vuorio et al. 2024 | Proposition 1: π^IL(o) = E[π^teach(S) \| f(S) = o], the privileged information is averaged out; IL warm-start can be strictly worse than none (Poisoned Doors); distillation is sound iff the privileged variable is a deterministic function of the student's history; off-policy IL of privileged experts "latches", on-policy recovers; the expert never demonstrates information-gathering | the formal explanation of "imitation matches but does not beat the best heuristic": where future use is unpredictable from features, DAgger learns the average of the oracle's action, which a tuned heuristic already approximates. The mapping to memory is our inference, not prior work |
| **DAgger** (AISTATS 2011); **AggreVaTe** (2014); **LOLS** (ICML 2015) ([link](https://proceedings.mlr.press/v37/changb15.html)); "Revisiting DAgger in the Era of LLM-Agents" (2605.12913) | on-policy aggregation; cost-to-go rather than 0–1 mismatch tolerates suboptimal experts; roll-in/roll-out one-step-deviation regret "can improve upon the reference policy" | AggreVaTe/LOLS-style regret labels are what would let us keep the oracle as teacher while learning to archive (§17, D1) |
| **Bhattacharjee & Mahajan**, ALT 2022 ([link](https://arxiv.org/abs/2201.03806)); Kara & Yüksel JMLR 2022; Toro Icarte et al. 2020 | online learning of what to remember against memory-constrained experts; finite-window near-optimality; explicit write actions over external memory in RL | the abstract form of our recall task; no archive tier, no LLM |
| Agentic credit assignment: Wei et al. 2505.11821, GiGPO 2505.10978 (NeurIPS 2025 per PDF), HCAPO, RAGEN; dissent "Coverage, Not Targeting" 2609.02417 | dense per-turn reward usually beats terminal; one paper finds uniform dense beats targeted | no paper ablates immediate vs delayed reward or γ for memory decisions; our H2 test is open ground and currently underpowered |

### 16.2 Comparison table

| System | Year / venue | Separate controller? | Action space | Hard active budget? | Archive + retrieval? | Training signal | Oracle / hindsight? | Failure attribution? | Sequential eval? | Closest overlap |
|---|---|---|---|---|---|---|---|---|---|---|
| **memctl (this project)** | 2026 | S (27k params, no text) | KEEP/EVICT/ARCHIVE/RETRIEVE/COMPACT/COMPACT+ARCHIVE/CONSOLIDATE | yes | yes (unlimited) | DAgger on Belady-style oracle; PPO on task reward | exact ILP + approximate | 7 labels | yes (synthetic workflow); QA retention on LoCoMo/LME | — |
| MemGPT | 2023 | no | append/replace/insert/search; auto FIFO flush | yes (thresholds) | yes | none | no | no | no | tiers + archive |
| Generative Agents / MemoryBank | 2023 | heuristic score | append/reflect/retrieve; decay forgetting | no | no | none | no | no | sim only | our heuristics |
| ACE | ICLR 2026 | prompted roles | edit playbook | no | no | none | no | names collapse modes | AppWorld | compaction failure modes |
| Memory-R1 | ACL 2026 | T (8B) | ADD/UPDATE/DELETE/NOOP | no | store | PPO/GRPO outcome | no | no | no | PPO; DELETE |
| MemAct | 2025 (unverified) | T | delete-set + write | penalty (20K) | no | GRPO terminal | no | no | partly | hard limit; deletion risk |
| AgeMem | ACL 2026 | T | Add/Update/Delete; Retrieve/Summary/Filter | soft | LTM store | step-wise GRPO | no | no | yes | largest action overlap |
| Memex(RL) | 2026 | T (30B) | archive+summary; read-exact | soft (8K) | yes | GRPO shaped | proposition only | no | yes (mod. ALFWorld) | archive-not-delete |
| MEM1 / MemAgent (ByteDance) / Mem-α / Context-Folding | ICLR/ICML 2026 | T | overwrite / fixed memory / tiered CRUD / fold | by construction or soft | no | PPO/DAPO/GRPO | no | no | WebShop, SWE-Bench | PPO arm precedents |
| HiMPO / Mem-T / Memory-R2 / UMA | 2026 | T | writes / CRUD | UMA 16k | store | hindsight or re-rollout credit | credit, not policy | interventions | no | hindsight signal |
| LRE | 2026 | S (kilobyte LR) | keep/evict | yes (2048) | no | hindsight self-labels | keep-all only | case study | AppWorld + LoCoMo | **closest overall** |
| MemCon | 2026 | S (tabular bandit) | retrieve variants, consolidate, forget, noop | no | store | episode-end shaped | likely no | no | yes | small controller, frozen LLM |
| EMBER | 2026 | S (7–14B) | insert/merge/overwrite/skip | yes (512–8192) | no | GRPO evidence-chain | oracle retention | retain- vs read-recall | no | budget sweep + oracle bound |
| TRACER | 2026 | S (REINFORCE) | retention ratio per tool | soft | no | success + tokens | counterfactual outcome model | per-tool | yes | separate RL policy |
| OSL-MR | 2026 | S | retain/drop | yes | no | offline privileged supervision | offline | no | no | teacher/student split |
| SOLAR | 2026 | S (online) | evict | yes | no | regret bound | vs OPT | no | no | heuristic flip |
| MemAudit | 2026 | evaluation | write-side packing | yes | no | — | exact MILP | write layer | no | exact ILP oracle |
| Restore-counterfactual audit | 2026 | evaluation | evict policies | yes (8k/80k) | top-k | — | restoration | 3 classes | no | prior attribution |
| Parrot | ICML 2020 | S | evict one line | yes | refetch on miss | DAgger on Belady | Belady | no | no (fixed trace) | ancestor of DAgger arm |
| LRB | NSDI 2020 | S (GBM) | evict | yes | origin refetch | relaxed Belady | boundary | good-decision ratio | no | online labels |
| LeCaR / Cacheus | 2018 / 2021 | S (mixture) | LRU/LFU mix | yes | refetch | regret | no | no | no | mixture baseline |
| Demand-MIN | ISCA 2018 | oracle | evict + prefetch | yes | prefetch | — | corrected oracle | no | no | oracle changes with fetch channel |
| ForesightKV / KVP / DistillCache | ICML 2026 / 2026 | S (scorer) | KV evict | yes | no | SL on future labels + GRPO / RL / REINFORCE | future attention | no | no | IL-then-RL twin at token level |
| Bhattacharjee & Mahajan | ALT 2022 | theory | retain bounded facts | yes (slots) | no | online regret | hindsight expert | no | no | abstract recall task |

### 16.3 Overlaps, distinctions, novelty risks

**Likely overlaps (do not claim as new).** A tiny language-model-free retention scorer under a hard budget trained from hindsight labels (LRE). A separate non-LLM controller with consolidate/forget over a frozen LLM on ALFWorld-style tasks (MemCon). An exact ILP oracle for budgeted memory as a certified bound (MemAudit). Evicted-vs-retrieval-miss attribution by restoration (Shen 2026). Budget sweeps against random/recency/salience with an oracle bound and a retain/read split (EMBER). Archive-with-index instead of deletion on a memory-dependent ALFWorld (Memex(RL)). DAgger on a Belady oracle followed by RL (Parrot, ForesightKV). Heuristic-ranking instability across workloads (SOLAR, Cacheus). "Retrieved but not acted on" (MERIT, 55 %). The endogenous-environment problem for memory credit (Memory-R2).

**Likely distinctions (stated as "no paper found", not "none exists").** No paper trains one separate policy over keep, evict, archive, retrieve, compact and consolidate jointly under one hard active budget with a frozen task model. No paper trains a sub-million-parameter *neural* controller with PPO for memory control. No paper reports an attribution finer than three classes. No paper states or tests that imitating a hindsight deletion oracle is the wrong target once archival exists (the pieces are in Demand-MIN, MemAudit, Memex(RL), MemAct). No paper measures the hindsight oracle's undefinedness after trajectory divergence (every caching paper assumes an exogenous stream; Memory-R2 names it). No paper runs both an approximate and an exact delete-only oracle and DAgger against them on the same tasks.

**Unresolved novelty risks.** LRE (revised 2026-09-03) and MemCon may add archive actions or an oracle in later revisions. MemCon's body may contain an oracle (conflicting extractions). "Beyond Memory Leaderboards: Budgeted Context Restoration" (2607.16848) and MemTrace could overlap the attribution scheme; neither was fully extracted. The Epistemics paper's ConsolidationBench may anticipate an oracle framing for CONSOLIDATE. TRACER's per-tool counterfactual and Memory-PRM's delete-and-re-answer are one-action-deviation constructions a reviewer may equate with our attribution. The sequential result rests on a scripted environment while MemoryArena, Mem2ActBench, MERIT and MemGym exist; Memex(RL) solved the nearest analogue by training the task model, so "separate controller vs task-model training" is an unrun comparison. Unfetched 2026 titles that could overlap: "Selective Forgetting" (2608.28978), "Learning What to Remember: Multi-Factor Value Model" (2606.12945), "What Training Data Teaches RL Memory Agents" (2605.23067), MemOps, DynamicMem, StreamMemBench, TrustMem, MemCoRe. The imitation-gap mapping to memory is an inference; present it as explanation, not prior art.

## 17. Suggested next research directions

Ranked by a joint reading of scientific value (V), novelty potential (N), feasibility on this machine (F), compute (C, lower is cheaper) and thesis suitability (T); each on high / medium / low.

| # | Direction | V | N | F | C | T |
|---|---|---|---|---|---|---|
| D1 | Replace hindsight deletion labels with archive-aware counterfactual regret (AggreVaTe/LOLS-style) | H | H | H | low | H |
| D2 | One controller across task generators, with a Bayes-informed ceiling reference | H | M | H | low | H |
| D3 | Learn a joint placement–retrieval policy with a structured or dense retriever | H | H | M | med | H |
| D4 | Re-planning oracle after divergence; regret labels valid on sequential tasks | M | H | H | low | M |
| D5 | Price the archive and retrieval; measure the placement–retrieval frontier | M | M | H | low | M |
| D6 | Immediate vs delayed reward, powered: ≥ 5 seeds × γ ∈ {0, 0.9, 0.995} on the workflow task, plus dense regret shaping | M | M | H | med | M |
| D7 | Verified compression / consolidation operator checked against future evidence needs | M | M | M | med (GPU) | M |
| D8 | LLM task model and a published trajectory-dependent benchmark (MemGym / MemoryArena / Mem2ActBench) | H | M | L here | high (rented GPU) | H (required for a thesis claim) |
| D9 | Regret prediction as a transferable signal across budgets, horizons and task models | M | M | M | med | L–M |
| D10 | JEV arm with an action-matched control | M | L | blocked | low ($) | M (one of the three arms) |

**D1. Replace hindsight deletion labels with archive-aware counterfactual Q-values.** *Motivation:* Experiment 4b (imitation of the deleter scores 0.521 at 5 % and fails by `evicted`; an untrained retrieval head scores 0.849). *Papers:* Demand-MIN shows the oracle changes once a fetch channel exists; AggreVaTe and LOLS supply cost-to-go and one-step-deviation regret labels that tolerate a suboptimal reference; Memory-R2 and Hindsight Memory-PRM compute exactly such labels for memory writes; Weihs et al. explain why argmax imitation of a privileged teacher averages the information out. *Experiment:* for each (item, operation) pair at a decision, label with continuation regret (§4.5b) under the experiment's own retriever, so ARCHIVE is credited with ρ·p rather than treated as wrong; train `rl_bc_archive` on these labels (the `SwitchController` and interventions hook already exist); evaluate on the 4b grid. *Falsifier:* the new policy does not exceed the untrained-retrieval-head policy at 2–10 % within its MDE, or its failures remain `evicted`.

**D2. Train one controller across multiple task generators.** *Motivation:* H1 is only half-answered: imitation matches the best heuristic on each task separately, and the recall-trained policy does not transfer (0.042 on workflow). *Papers:* Cacheus and SOLAR (workload-dependent rankings), LRE and MemCon (single-task separate controllers), AgeMem's argument against auxiliary controllers. *Experiment:* DAgger and PPO on a mixture of recall and workflow episodes (plus a third generator with intermediate structure, e.g. recall with `requery_prob` high), evaluate on all; add a "Bayes-informed" reference that knows the generator's `query_prob` per source to bound the closable gap; add a LeCaR-style online mixture of the existing heuristics as the strongest fixed baseline. *Falsifier:* the joint policy trails the per-task best heuristic on either task by more than its MDE, or the online heuristic mixture matches it.

**D3. Learn a joint placement–retrieval policy.** *Motivation:* once an archive exists the retriever decides the outcome (Exp. 2: FIFO–salience gap 0.35 → 0.06; Exp. 5: BM25 recall 0.01–0.10 makes the archive worthless). *Papers:* MemSifter (learned retrieval only), EMBER (retain-vs-read split), Memex(RL) (read calls rise after archiving), joint caching-and-prefetching (Yuan et al. 2025, DEAP). *Experiment:* keep the Bernoulli retrieval head but feed it a shortlist from a structured query (job id on the workflow task) or a dense model, train placement and retrieval jointly with the priced archive of D5, and report `retrieval_recall` and `needed_hit_rate` separately. *Falsifier:* joint training does not raise workflow retrieval recall above the fixed BM25 top-*k*, or placement quality no longer matters once retrieval is learned (which would itself be a finding).

**D4. A re-planning oracle after divergence.** *Motivation:* the oracle is blind after the first restart (0.808 with 25 restarts at 5 %) and no regret label is valid past that point. *Papers:* Memory-R2 (local re-rollouts from checkpoints), Parrot's exogenous-stream assumption. *Experiment:* use `snapshot`/`restore` to recompute hindsight from the diverged step (a fresh reference pass from the current environment state), giving a "tracked oracle" and valid post-divergence labels; retrain `rl_bc_workflow` with them; compare oracle_approx, oracle_approx_lazy and tracked oracle at 5 %. *Falsifier:* the tracked oracle does not beat the lazy oracle's 0.896, or labels past divergence do not change the learned policy.

**D5. Price the archive and retrieval.** *Motivation:* H4 was measured with a free, unlimited archive (17,600 tokens against a 360-token budget). *Papers:* EMBER's retained-evidence budget; MERIT's cost-aware evaluation; the rate–distortion view. *Experiment:* sweep `memory.archive_budget` and the `retrieval_cost` weight; plot success against total tokens (active + archive + retrieved) and find where deletion becomes rational. *Falsifier:* the archive ordering is unchanged at every price, meaning the free-archive result was not an artefact.

**D6. Immediate versus delayed reward, properly powered.** *Motivation:* H2 is inconclusive (3 seeds within noise on recall; 1 seed each, 0.992 vs 0.404, on workflow). *Papers:* Wei et al. 2505.11821 and "Coverage, Not Targeting" disagree on dense credit; no memory-specific ablation exists. *Experiment:* γ ∈ {0, 0.9, 0.995} × 5 seeds on the workflow task, with and without the `hindsight_eviction_regret` term as shaping during training only, learning-rate sweep, best-validation checkpoints. *Falsifier:* no monotone or consistent effect of γ across seeds.

**D7. A verified compression operator.** *Motivation:* extractive compaction helped only because it matched the generator; truncation produced 1,368–1,443 `compression_lost_detail` failures per cell; naive consolidation hurt salience by 0.115. *Papers:* ACE (context collapse), LRE's identifier-paraphrase failure, the Epistemics paper's ConsolidationBench. *Experiment:* run the `llm` compressor and consolidator on a GPU host and add a verification step that keeps the source (archives it) whenever the rewrite drops a needle recoverable by the tool-value tracer; compare lost-detail counts. *Falsifier:* verification does not reduce `compression_lost_detail` relative to the unverified LLM rewriter at equal memory.

**D8. A language-model task model and a published trajectory-dependent benchmark.** *Motivation:* every reported number uses a scripted reader; the sequential result rests on one synthetic task. *Papers:* MemGym (built for controller training, memory-isolated scores), MemoryArena, Mem2ActBench, MERIT; Memex(RL) as the task-model-trained comparator. *Experiment:* rent a ≥ 12 GB card, rerun Experiments 1–5 with the `llm` agent (attribution will report the reasoning share), then port the controller interface to MemGym. *Falsifier:* controller differences vanish under a real model (failures dominated by `task_model_reasoning`), or the learned controller's workflow advantage does not reappear on a published benchmark.

**D9. Regret prediction as a transferable signal.** *Motivation:* `requirements_destroyed` is logged for every episode and could supervise a value head directly. *Experiment:* train a regret predictor on recall at horizon 200 and test on horizon 500, on the workflow task and, later, on LLM-agent runs. *Falsifier:* predicted regret does not correlate with realised regret out of distribution.

**D10. The JEV arm.** Unblocked by a key; run `configs/stage1_smoke.yaml` with `controller: {name: jev}` to verify parsing, then the Experiment 1 grid with an action-matched control and JEV's API cost counted.

## 18. Candidate thesis formulations

**A. Learned memory management under a bounded context (framework and comparison thesis).**
*Claim:* under an explicit active-context budget, a small learned controller separate from a frozen task model reaches the best task-specific heuristic on each of two task structures and, unlike any fixed heuristic, is not wrong on both. *Question:* do learned memory policies beat fixed heuristics at equal budget and compute, and when? *Contribution:* the framework (seven-action engine, harness-level attribution, exact and approximate oracles, paired sweeps) plus the two-task evidence. *Required experiments:* D2 (joint policy and Bayes ceiling), D5 (priced archive), clean-commit reruns, D8 with an LLM task model on at least one published benchmark. *Risk:* medium. The overlaps with LRE, MemCon and EMBER are real; the defence is the conjunction and the attribution apparatus, and the thesis must show a *single* policy winning across tasks, which is not yet shown.

**B. Hindsight and regret-based training for memory policies (supervision thesis).**
*Claim:* hindsight oracles are valuable teachers for memory controllers only when their action semantics match the learner's action set and information; with an archive available, deletion labels must be replaced by archive-aware counterfactual regret, and after trajectory divergence the oracle must be recomputed. *Question:* what is the right teacher for a memory policy? *Contribution:* the 4b mechanism (§4.7) generalised and fixed (D1), the tracked oracle (D4), and a comparison of DAgger, AggreVaTe/LOLS-style regret labels and PPO under matched budgets. *Required experiments:* D1, D4, D6, plus a formal statement in the language of Weihs et al. and Demand-MIN. *Risk:* medium-low. The negative result already exists and no paper states it for memory; the risk is that the fix is straightforward and a reviewer sees it as an application of AggreVaTe.

**C. Joint placement and retrieval for long-horizon agents (systems thesis).**
*Claim:* once archival memory exists, placement quality matters only through retrieval, and a policy that learns both under a priced archive dominates a placement policy with a fixed retriever. *Question:* should retrieval be learned jointly with placement? *Contribution:* the joint policy (D3), the priced-archive frontier (D5), and the retriever-dependence result across tasks. *Required experiments:* D3, D5, D8 on a trajectory-dependent benchmark. *Risk:* medium-high. Retrieval learning is a crowded field (MemSifter, EMBER) and the workflow retrieval failure may be fixable by a structured query alone, which would make the learned component unnecessary.

The evidence does not yet single out one of these. B is the best supported by results already in hand; A is what the original three-arm brief asked for; C is where the largest empirical effects live (0.23 → 0.73). A defensible plan is B as the core with A's framework as the vehicle, and C's joint policy as the final experiment if compute allows.

## 19. Immediate next steps

**A. Next 48 hours.**
1. Commit the working tree that produced the runs (it is `4fe0f7c`, already on `main`), then rerun every sweep and training from that commit so `metadata.json` reads clean (about two hours of CPU for the sweeps; the RL trainings add several hours). Until then no number in this note is final.
2. Rerun the 32 Experiment 5b cells that were resumed from pre-E17 rows (delete the cells and rerun the sweep), so hindsight-derived columns are consistent.
3. Fix the parameter-count statement in EXPERIMENTS.md (27,526 / 19,334, from the training metadata).
4. Add `oracle_solve_s` to the report tables and rerun Experiments 1–3 so the field is populated.

**B. Next two weeks.**
1. D1: implement the archive-aware regret expert using `continuation_regret` and retrain `rl_bc_archive`.
2. D2: joint recall+workflow training and the Bayes-informed ceiling reference; add a LeCaR-style heuristic mixture baseline.
3. D6: the powered γ sweep on the workflow task (5 seeds × 3 values, ~10 trainings of 40 min each).
4. D5: the priced-archive sweep of Experiment 2 (one config change).
5. Run `stage1_smoke` with the `jev` controller as soon as a key exists.

**C. Before showing this as a thesis proposal.**
1. All quoted numbers from clean-commit runs, with the report tables regenerated.
2. One experiment with a language-model task model beyond a smoke test (rented GPU), with attribution reported.
3. A written positioning against LRE, MemCon, EMBER, MemAudit, Shen 2026, Parrot and Memex(RL) (this section is the draft).
4. A decision on the thesis formulation (§18), with the required experiments scheduled.

**D. What must be rerun from a clean commit.** Everything in `runs/`: all 14 sweeps (`configs/sweeps/*.yaml`) and all 14 trainings (`configs/rl/*.yaml`), in the order trainings → evaluation sweeps. Priority order if time is short: Exp. 1, 2, 4, 4b, 5, 5b (they carry every headline number), then 1b–1d, 3, 6, 6c.

**E. Questions for Joey, Dacheng and Asim.**
1. Is the three-arm brief (no controller / JEV / RL) still the target, or is the supervision thesis (B) an acceptable reframing given the JEV block?
2. Is a scripted task model acceptable for the core claims if a single LLM replication is provided, or must every experiment run with a language model?
3. Which published trajectory-dependent benchmark (MemGym, MemoryArena, Mem2ActBench) do they consider credible, and is compute for it available?
4. Given LRE and MemCon, is the combinational novelty (seven actions, frozen task model, oracle + attribution) enough for the thesis, or should the contribution be narrowed to the hindsight-teacher result?
5. Should the archive be priced from the start, and in what unit (tokens, retrieval calls, dollars)?
6. Is a JEV key obtainable, and if not, should the JEV arm be dropped from the thesis?

## Bibliography

Repository documents (ground truth for every result): `README_RESEARCH.md`, `EXPERIMENTS.md`, `INFRA_REPORT.md`, `SCIENTIFIC_VALIDITY_REPORT.md`, `blockers.md`, `docs/restart.md`, `docs/superpowers/specs/2026-09-30-memctl-framework-design.md`; run artifacts under `runs/` (`summary.json`, `report.json`, `episodes.jsonl`, `failures.jsonl`, `metadata.json`, `train_log.jsonl`, sweep `.log` files).

Literature (venue as verified on 2026-09-30; "(venue unverified)" where the primary page did not confirm it):

- Packer et al. MemGPT: Towards LLMs as Operating Systems. arXiv 2310.08560, 2023. https://arxiv.org/abs/2310.08560
- Xu et al. A-MEM: Agentic Memory for LLM Agents. NeurIPS 2025 (per arXiv). https://arxiv.org/abs/2502.12110
- Chhikara et al. Mem0. arXiv 2504.19413, 2025. https://arxiv.org/abs/2504.19413
- Rasmussen et al. Zep: A Temporal Knowledge Graph Architecture for Agent Memory. arXiv 2501.13956, 2025. https://arxiv.org/abs/2501.13956
- Jiménez Gutiérrez et al. HippoRAG. NeurIPS 2024. https://arxiv.org/abs/2405.14831
- Qian et al. ECHO: A Cognitively Inspired, Auditable Memory Plane for Long-Horizon Agents. arXiv 2608.21755, 2026. https://arxiv.org/abs/2608.21755
- Park et al. Generative Agents. arXiv 2304.03442, 2023 (UIST 2023 venue unverified). https://arxiv.org/abs/2304.03442
- Zhong et al. MemoryBank. arXiv 2305.10250, 2023 (AAAI 2024 venue unverified). https://arxiv.org/abs/2305.10250
- Shinn et al. Reflexion. arXiv 2303.11366, 2023 (NeurIPS 2023 venue unverified). https://arxiv.org/abs/2303.11366
- Zhang et al. Agentic Context Engineering (ACE). ICLR 2026 (per arXiv). https://arxiv.org/abs/2510.04618
- Jiang et al. LLMLingua. EMNLP 2023. https://arxiv.org/abs/2310.05736
- Li et al. Sculptor. arXiv 2508.04664, 2025. https://arxiv.org/abs/2508.04664
- Lin et al. Sleep-time Compute. arXiv 2504.13171, 2025. https://arxiv.org/abs/2504.13171
- Zhang et al. A Survey on the Memory Mechanism of LLM-based Agents. arXiv 2404.13501, 2024. https://arxiv.org/abs/2404.13501
- Hu et al. Memory in the Age of AI Agents: A Survey. arXiv 2512.13564. https://arxiv.org/abs/2512.13564
- Du. Memory for Autonomous LLM Agents: Mechanisms, Evaluation, and Emerging Frontiers. arXiv 2603.07670, 2026. https://arxiv.org/abs/2603.07670
- Jiang et al. Anatomy of Agentic Memory. arXiv 2602.19320, 2026. https://arxiv.org/abs/2602.19320
- Yan et al. Memory-R1. ACL 2026. https://arxiv.org/abs/2508.19828 ; https://aclanthology.org/2026.acl-long.583/
- Zhang et al. Memory as Action (MemAct). arXiv 2510.12635, 2025 (venue unverified). https://arxiv.org/abs/2510.12635
- Yu et al. Agentic Memory (AgeMem). ACL 2026. https://arxiv.org/abs/2601.01885
- Wang et al. Memex(RL). arXiv 2603.04257, 2026. https://arxiv.org/html/2603.04257v1
- Zhou et al. MEM1. ICLR 2026. https://arxiv.org/abs/2506.15841
- Yu et al. MemAgent: Reshaping Long-Context LLM with Multi-Conv RL-based Memory Agent. ICLR 2026 oral. https://arxiv.org/abs/2507.02259
- Wu et al. ReSum. arXiv 2509.13313, 2025. https://arxiv.org/abs/2509.13313
- Wang et al. Mem-α. arXiv 2509.25911, 2025 (ICLR 2026 venue unverified). https://arxiv.org/abs/2509.25911
- Sun et al. Context-Folding. ICML 2026. https://arxiv.org/abs/2510.11967
- Yan et al. HiMPO. arXiv 2606.16285, 2026. https://arxiv.org/html/2606.16285
- Yue et al. Mem-T. arXiv 2601.23014, 2026. https://arxiv.org/abs/2601.23014
- Yan et al. Memory-R2. arXiv 2605.21768, 2026. https://arxiv.org/abs/2605.21768
- Zhang et al. UMA: Learning to Remember. EMNLP 2026. https://arxiv.org/abs/2602.18493
- Lia, Mazumder. Learning What Not to Forget (LRE). arXiv 2606.20954, 2026. https://arxiv.org/abs/2606.20954
- Jiang et al. Memory as a Controlled Process (MemCon). arXiv 2607.13591, 2026. https://arxiv.org/abs/2607.13591
- Li, Banerjee, Che. EMBER. arXiv 2606.05894, 2026. https://arxiv.org/html/2606.05894
- Yi et al. AdaCoM. arXiv 2605.30785, 2026. https://arxiv.org/html/2605.30785
- Tan et al. MemSifter. arXiv 2603.03379, 2026. https://arxiv.org/html/2603.03379
- Jia et al. Hindsight Memory-PRM. arXiv 2608.29605, 2026. https://arxiv.org/html/2608.29605
- Wei et al. MemAgent: Learning to Manage Heterogeneous Memory Providers. arXiv 2609.32521, 2026. https://arxiv.org/html/2609.32521v1
- Reddy. Selective Memory Retention (TraceRetain). arXiv 2606.29178, 2026 (ICML 2026 unverified). https://arxiv.org/abs/2606.29178
- Sun, Cao, Lam. SOLAR. arXiv 2607.00394, 2026. https://arxiv.org/abs/2607.00394
- Kang et al. ACON. ICML 2026. https://arxiv.org/abs/2510.00615
- Kang et al. OSL-MR. arXiv 2606.10616, 2026. https://arxiv.org/abs/2606.10616
- Lin et al. TRACER. arXiv 2608.29363, 2026. https://arxiv.org/abs/2608.29363
- Yang et al. MeClear. arXiv 2609.09115, 2026. https://arxiv.org/abs/2609.09115
- Bhargava, Sobral Barrento. MemAudit. arXiv 2605.02199, 2026. https://arxiv.org/html/2605.02199
- Shen. What Eviction Destroys: A Restore-Counterfactual Audit. arXiv 2609.08279, 2026. https://arxiv.org/abs/2609.08279
- Deng et al. MemTrace. arXiv 2605.28732, 2026 (ID discrepancy with 2606.17328; verify). https://arxiv.org/abs/2605.28732
- Annapureddy, Thamatani. The Epistemics of Agent Memory. arXiv 2609.33013, 2026. https://arxiv.org/abs/2609.33013
- Colaco, Lahjouji. What to Keep, What to Forget: A Rate–Distortion View. arXiv 2607.08032, 2026. https://arxiv.org/abs/2607.08032
- Zou et al. Remember the Decision, Not the Description. arXiv 2605.10870, 2026. https://arxiv.org/abs/2605.10870
- Liu. What Does Context Compression Cost an Agent? arXiv 2608.16370, 2026. https://arxiv.org/pdf/2608.16370
- Yang. ForgetEval. arXiv 2606.15903, 2026. https://arxiv.org/abs/2606.15903
- Maharana et al. Evaluating Very Long-Term Conversational Memory of LLM Agents (LoCoMo). ACL 2024. https://aclanthology.org/2024.acl-long.747/ ; https://github.com/snap-research/locomo ; audit: https://github.com/snap-research/locomo/issues/27
- Wu et al. LongMemEval. ICLR 2025. https://iclr.cc/virtual/2025/poster/28290 ; https://github.com/xiaowu0162/longmemeval
- MemDelta. arXiv 2606.29914, 2026. https://arxiv.org/html/2606.29914v1
- Hu, Wang, McAuley. MemoryAgentBench. ICLR 2026. https://arxiv.org/abs/2507.05257
- Li et al. LoCoMo-Plus. ACL 2026. https://arxiv.org/abs/2602.10715
- He et al. MemoryArena. ICML 2026. https://arxiv.org/abs/2602.16313
- Mem2ActBench. ACL 2026. https://arxiv.org/abs/2601.19935
- Mishra, Mishra. When Does Memory Help? (MERIT). arXiv 2609.05441, 2026. https://arxiv.org/abs/2609.05441
- Xu et al. MemGym. arXiv 2605.20833, 2026. https://arxiv.org/abs/2605.20833
- Liu, Gabriel. PM-Bench. COLM 2026. https://arxiv.org/abs/2607.12385
- Belady. A Study of Replacement Algorithms for a Virtual-Storage Computer. IBM Systems Journal 5(2), 1966.
- Jain, Lin. Back to the Future: Leveraging Belady's Algorithm (Hawkeye). ISCA 2016. https://dl.acm.org/doi/10.1109/ISCA.2016.17
- Shi et al. Applying Deep Learning to the Cache Replacement Problem (Glider). MICRO 2019. https://www.cs.utexas.edu/~lin/papers/micro19c.pdf
- Shah, Jain, Lin. Effective Mimicry of Belady's MIN Policy (Mockingjay). HPCA 2022 (venue inferred). https://www.cs.utexas.edu/~lin/papers/hpca22.pdf
- Liu et al. An Imitation Learning Approach for Cache Replacement (Parrot). ICML 2020. http://proceedings.mlr.press/v119/liu20f/liu20f.pdf
- Song et al. Learning Relaxed Belady (LRB). NSDI 2020. https://www.usenix.org/system/files/nsdi20-paper-song.pdf
- Vietri et al. LeCaR. HotStorage 2018. https://www.usenix.org/system/files/conference/hotstorage18/hotstorage18-paper-vietri.pdf
- Rodriguez et al. Cacheus. FAST 2021. https://www.usenix.org/system/files/fast21-rodriguez.pdf
- Jain, Lin. Rethinking Belady's Algorithm to Accommodate Prefetching (Demand-MIN). ISCA 2018. https://dl.acm.org/doi/10.1109/ISCA.2018.00020
- Beyond Belady (LRU-BaSE). arXiv 2212.13671 (venue unverified). https://arxiv.org/html/2212.13671v1
- Yuan et al. A Joint Learning Approach to Hardware Caching and Prefetching. NeurIPS 2025 ML-for-Systems workshop. https://arxiv.org/abs/2510.10862
- Lykouris, Vassilvitskii. Competitive Caching with Machine Learned Advice. ICML 2018 / J. ACM 2021. https://arxiv.org/pdf/1802.05399
- Bhattacharjee, Mahajan. Learning what to remember. ALT 2022. https://arxiv.org/abs/2201.03806
- Kara, Yüksel. Near Optimality of Finite Memory Feedback Policies in POMDPs. JMLR 2022. https://www.jmlr.org/papers/volume23/20-1152/20-1152.pdf
- Zhang et al. H2O. arXiv 2306.14048, 2023 (NeurIPS 2023 venue unverified). https://arxiv.org/abs/2306.14048
- Liu et al. Scissorhands. arXiv 2305.17118, 2023 (venue unverified). https://arxiv.org/abs/2305.17118
- Xiao et al. StreamingLLM. ICLR 2024. https://arxiv.org/abs/2309.17453
- Ge et al. Model Tells You What to Discard (FastGen). ICLR 2024. https://arxiv.org/abs/2310.01801
- Li et al. SnapKV. arXiv 2404.14469, 2024 (venue unverified). https://arxiv.org/abs/2404.14469
- Tang et al. Quest. ICML 2024. https://arxiv.org/abs/2406.10774
- Xiao et al. InfLLM. arXiv 2402.04617 (venue unverified). https://arxiv.org/abs/2402.04617
- Nawrot et al. Dynamic Memory Compression. ICML 2024. https://arxiv.org/abs/2403.09636
- Dong et al. ForesightKV. ICML 2026. https://arxiv.org/abs/2602.03203
- Moschella, Manduchi, Sener. Learning to Evict from Key-Value Cache (KVP). ICML 2026. https://arxiv.org/abs/2602.10238
- Hauzenberger et al. KVpop. arXiv 2607.05061, 2026. https://arxiv.org/abs/2607.05061
- Althoubi. DistillCache. arXiv 2608.08878, 2026. https://arxiv.org/abs/2608.08878
- Matam, Kim. MemDecay. arXiv 2607.10582, 2026. https://arxiv.org/abs/2607.10582
- Weihs et al. Bridging the Imitation Gap by Adaptive Insubordination (ADVISOR). NeurIPS 2021. https://arxiv.org/abs/2007.12173
- Cai et al. Provable Partially Observable RL with Privileged Information. NeurIPS 2024. https://proceedings.neurips.cc/paper_files/paper/2024/file/74d188c51d97fcfbc0269f584d6a53b7-Paper-Conference.pdf
- Swamy et al. Sequence Model Imitation Learning with Unobserved Contexts. arXiv 2208.02225, 2022 (venue unverified). https://arxiv.org/abs/2208.02225
- Vuorio et al. A Bayesian Solution To The Imitation Gap. arXiv 2407.00495, 2024. https://arxiv.org/abs/2407.00495
- Chen et al. Learning by Cheating. CoRL 2019. https://proceedings.mlr.press/v100/chen20a.html
- Pinto et al. Asymmetric Actor Critic for Image-Based Robot Learning. arXiv 1710.06542, 2017. https://arxiv.org/abs/1710.06542
- Warrington et al. Robust Asymmetric Learning in POMDPs (A2D). ICML 2021. https://proceedings.mlr.press/v139/warrington21a.html
- Nguyen et al. COSIL. CoRL 2022. https://arxiv.org/abs/2211.01991
- Kim et al. Distilling Realizable Students from Unrealizable Teachers. arXiv 2505.09546, 2025. https://arxiv.org/abs/2505.09546
- Ross, Gordon, Bagnell. A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning (DAgger). AISTATS 2011. https://arxiv.org/abs/1011.0686
- Ross, Bagnell. Reinforcement and Imitation Learning via Interactive No-Regret Learning (AggreVaTe). arXiv 1406.5979, 2014. https://arxiv.org/pdf/1406.5979
- Sun et al. Deeply AggreVaTeD. ICML 2017. https://arxiv.org/abs/1703.01030
- Chang et al. Learning to Search Better than Your Teacher (LOLS). ICML 2015. https://proceedings.mlr.press/v37/changb15.html
- Li et al. Revisiting DAgger in the Era of LLM-Agents. arXiv 2605.12913, 2026. https://arxiv.org/abs/2605.12913
- Harutyunyan et al. Hindsight Credit Assignment. NeurIPS 2019. https://arxiv.org/abs/1912.02503
- Mesnard et al. Counterfactual Credit Assignment in Model-Free RL. ICML 2021. https://arxiv.org/abs/2011.09464
- Arjona-Medina et al. RUDDER. arXiv 1806.07857. https://arxiv.org/abs/1806.07857
- Andrychowicz et al. Hindsight Experience Replay. NeurIPS 2017. https://arxiv.org/abs/1707.01495
- Wei et al. Reinforcing Multi-Turn Reasoning via Fine-Grained Reward Structure. arXiv 2505.11821 (NeurIPS 2025 unverified). https://arxiv.org/abs/2505.11821
- Feng et al. GiGPO. arXiv 2505.10978 (NeurIPS 2025 unverified). https://arxiv.org/abs/2505.10978
- Wang et al. RAGEN. arXiv 2504.20073, 2025. https://arxiv.org/abs/2504.20073
- Zhou et al. Coverage, Not Targeting. arXiv 2609.02417, 2026. https://arxiv.org/abs/2609.02417

## Experiment provenance

| Experiment | Config | Run folder | Key artifacts | Commit stamped | Dirty | Rerun after review fixes? |
|---|---|---|---|---|---|---|
| 1 | `configs/sweeps/exp1_delete_only.yaml` | `runs/exp1_delete_only/` (198 cells) | `report.json`, `*/summary.json`, `*/failures.jsonl` | d215c8f | yes | not needed (delete-only) |
| 1b | `exp1b_attribution_check.yaml` | `runs/exp1b_attribution_check/` | `*/summary.json` | d215c8f | yes | not needed |
| 1c | `exp1c_horizon_fixed_budget.yaml` | `runs/exp1c_horizon_fixed_budget/` + `.log` | `*/summary.json`; solve times from the sweep log | d215c8f | yes | not needed |
| 1d | `exp1d_dependency_gap.yaml` | `runs/exp1d_dependency_gap/` | `*/summary.json` | d215c8f | yes | not needed |
| 2 | `exp2_archive_retrieval.yaml` | `runs/exp2_archive_retrieval/` + `.rerun.log` | `*/summary.json` | d215c8f | yes | yes (E16) |
| 3 | `exp3_compaction_consolidation.yaml` | `runs/exp3_compaction_consolidation/` | `*/summary.json` | d215c8f | yes | not needed |
| 4 | `exp4_rl_eval.yaml`; `configs/rl/rl_bc*.yaml`, `rl_ppo*.yaml` | `runs/exp4_rl_eval/`, `runs/rl_*/` | `report.json`, `*/episodes.jsonl`, `rl_*/summary.json`, `train_log.jsonl`, `checkpoints/policy*.pt` | d215c8f | yes | second attempts (`runs/_superseded/*_first_attempt`) |
| 4b | `exp4b_rl_archive_eval.yaml`; `rl_bc_archive.yaml`, `rl_bc_ppo_archive.yaml` | `runs/exp4b_rl_archive_eval/` + `.rerun.log` | as above | d215c8f | yes | yes (E16) |
| 5 | `exp5_workflow.yaml` | `runs/exp5_workflow/` + `.rerun.log`; pre-fix in `runs/_superseded/exp5_workflow_before_fallback_fix/` | `*/summary.json`, `*/episodes.jsonl` (`env_stats.restarts`) | d215c8f | yes | yes (E16) |
| 5b | `exp5b_workflow_rl.yaml`; `rl_bc_workflow.yaml`, `rl_ppo_workflow*.yaml` | `runs/exp5b_workflow_rl/` + `.rerun.log`; old labels in `runs/_superseded/rl_bc_workflow_prefix_labels/` | as above | d215c8f | yes | **partial**: only `rl_bc_workflow` cells (E17) |
| 6a/6b | `exp6_locomo_retention.yaml`, `exp6_longmemeval_retention.yaml` | `runs/exp6_locomo_retention/`, `runs/exp6_longmemeval_retention/` + `.rerun.log` | `*/summary.json` | d215c8f | yes | yes (E16) |
| 6c | `exp6c_locomo_dense_retrieval.yaml` | `runs/exp6c_locomo_dense_retrieval/` + `.rerun.log` | `*/summary.json` | d215c8f | yes | yes (duplicated rows incident) |
| 7a/7b | `exp7_llm_smoke.yaml`, `exp7b_prompted_controller_smoke.yaml` | `runs/exp7_llm_smoke/`, `runs/exp7b_prompted_controller_smoke/` | `*/summary.json`, `*/steps.jsonl`, `cache/generations/` | d215c8f | yes | smoke only |
| Phase 2 | branch `phase2-locomo-baseline`, tag `phase2-complete` | `runs/locomo_4b_4bit/comparison.md` | – | ba06b3a | – | superseded design |
| Tests | `tests/` | – | 153 passed in 16 s on 2026-09-30 (this session) | 4fe0f7c / main | clean | – |

## Unresolved questions

| # | Question | Why it matters | What would settle it |
|---|---|---|---|
| U1 | How much of the oracle gap on the recall task is closable by any causal policy? | "gap closed" has an unknown ceiling; H1 cannot be scored without it | a Bayes-informed reference policy (D2) |
| U2 | Does a single learned policy beat the best task-specific heuristic on both tasks? | the core of H1 | joint training (D2) |
| U3 | Is the 4b failure fixed by an archive-aware regret expert? | decides the supervision thesis | D1 |
| U4 | Does long-horizon reward matter for memory decisions? | H2 inconclusive; one seed per arm on the task that should show it | D6 |
| U5 | Is the archive result a retriever result? | H4's qualifier | structured/dense retrieval on the workflow task (D3), priced archive (D5) |
| U6 | Do the findings survive a language-model task model? | every number uses a scripted reader | D8 |
| U7 | Does the workflow result transfer to a published trajectory-dependent benchmark? | sequential claim rests on one synthetic task | MemGym / MemoryArena run (D8) |
| U8 | What does JEV do with the same action set? | one of the three arms is empty | key + Experiment 1 grid (D10) |
| U9 | Do the numbers reproduce from a clean commit? | every run is stamped dirty | full rerun (§19 D) |
| U10 | Does MemCon or "Beyond Memory Leaderboards" contain an oracle or attribution that anticipates ours? | novelty risk | read the full texts |
| U11 | Is the imitation-gap explanation (Weihs et al.) the actual mechanism on the recall task? | turns a heuristic explanation into a claim | test the deterministic-filter condition: make future use a deterministic function of features and check that imitation then beats salience |
| U12 | How should archive tokens and retrieval calls be priced relative to active tokens? | the frontier depends on it | D5, and the advisors' answer to §19 E5 |
