# Infrastructure report

What was built, whether it runs, and what stops it from running here. Written
2026-09-30 on the machine described below; `python -m memctl.doctor` reproduces
the environment part of this report at any time.

## Machine

| | |
|---|---|
| CPU | 11th Gen Intel Core i7-11800H, 16 logical cores, 23 GB RAM |
| GPU | NVIDIA GeForce RTX 3050 Laptop, 4 GB |
| Python | 3.13.9 in `.venv` (host); 3.12 in the Docker image |
| PyTorch | 2.14.0, CPU build, in `.venv`; CUDA build in the Docker image |
| Local model server | Ollama at `http://localhost:11434/v1` serving `llama3.1:8b` (Q4, running on CPU). It belongs to another project on this machine and was used only for the two smoke tests in Experiment 7. |
| Credentials | `JEV_API_KEY` not set |
| Data | `data/locomo/locomo10.json` (2.8 MB); `data/longmemeval/longmemeval_oracle.json` (15 MB) and `longmemeval_s_cleaned.json` (277 MB), downloaded this session from the Hugging Face dataset `xiaowu0162/longmemeval-cleaned` |

## Status of every component

Statuses: **PASSED** implemented, unit-tested and run end to end in an
experiment here; **PARTIAL** implemented and tested, but validated only with a
stand-in, at smoke-test scale, or on part of its scope; **BLOCKED** implemented
but not runnable here for an external reason, given with what unblocks it;
**FAILED** ran and did not work. Tests: 153 pass on the host (about 80 s); the Docker image ran the suite as it
stood before the review fixes (148 passed, 86 s).

### Core

| Component | Status | Evidence / what blocks it |
|---|---|---|
| Memory state, tiers, token accounting | PASSED | 26 unit tests; every experiment |
| Action schema (14 operations) | PASSED | all 14 in the enum; 8 with handlers; the other 6 are rejected as "no handler yet" and counted as invalid actions |
| Engine: KEEP, EVICT, NO_OP | PASSED | Experiments 1–6 |
| Engine: MOVE_TO_ARCHIVE, RETRIEVE_FROM_ARCHIVE | PASSED | Experiments 2, 4b, 5, 6 |
| Engine: COMPACT, COMPACT_AND_ARCHIVE, CONSOLIDATE | PASSED | Experiment 3 (extractive and truncating rewriters, dedup consolidator) |
| Configurable allowed-operation set | PASSED | delete-only, archive and full action sets used across experiments; a disallowed operation is rejected and counted (integration test) |
| Forced-fallback eviction, logged, penalised, reported | PASSED | `no_controller` arm; `forced_evictions`, `forced_fallback_rate`, `forced_share_of_removals` in every episode row |
| Controller interface and registry | PASSED | 14 registered controllers |
| Heuristic controllers (FIFO, LRU, LFU, random, age decay, similarity, salience, archive-everything, no-controller, full-context) | PASSED | Experiments 1–6 |
| Hindsight oracle, approximate | PASSED | all experiments; marked approximate; not a bound when items differ in size, when compaction is allowed, or after the stream diverges (workflow task) |
| Hindsight oracle, exact (integer program, delete-only) | PASSED | PuLP/CBC; 0.3 s per episode at horizon 500, 7.5 s at 2,000 (Experiment 1c); falls back to approximate with a logged reason when the solver fails |
| Exact oracle for other action sets | not implemented | `hindsight/exact.py` has a `SOLVERS` registry with one entry; with an unlimited archive and perfect retrieval the just-in-time oracle is already optimal whenever each query's evidence fits |
| Synthetic recall environment (Phase 0) | PASSED | exact ground truth; knobs for horizon, dependency distance, reuse, updates, restatement, verbosity |
| Workflow environment (Phase 2, sequential) | PASSED | Experiments 5, 5b; forgetting changes the rest of the episode |
| LoCoMo adapter (Phase 1) | PARTIAL | pipeline and evidence metrics run on all 10 conversations (Experiment 6); QA accuracy needs a task model, see below |
| LongMemEval adapter (Phase 1) | PARTIAL | oracle and S splits parsed; evidence metrics on 100 questions of S; QA accuracy and the official LLM judge not run |
| Scripted task models (reader, tool agent) with a noise knob | PASSED | Experiment 1b recovers the injected rate |
| Language-model task model (`agents/llm.py`) | PARTIAL | unit-tested with a scripted model; run with a real model only in the Experiment 7 smoke test; the Hugging Face backend is BLOCKED in `.venv` (no `transformers`) and by the 4 GB GPU for any model above about 1B parameters at 16-bit |
| Archive search (BM25, embedding cosine) | PASSED | Experiments 2, 5, 6 |
| Dense embeddings (sentence-transformers, `BAAI/bge-small-en-v1.5`) | PARTIAL | installed and checked on CPU this session; Experiment 6c runs it on LoCoMo (see EXPERIMENTS.md); slow on CPU (about 20 ms per text without load) |
| Compressors: extractive, truncate; LLM compressor and consolidator | PASSED / PARTIAL | the first two in Experiment 3; the LLM ones unit-tested with a scripted model only |
| Prompted-LLM controller | PARTIAL | unit-tested with scripted replies (JSON, garbage, invalid operations); real-model smoke test in Experiment 7b |
| JEV controller | **BLOCKED** | adapter, batching, repair and logging are implemented and tested with a labelled FAKE client; the real HTTP client is written from the documented request and response shapes but has never been run, because `JEV_API_KEY` is absent. **To unblock:** put the key in `.env`; the first real call verifies the response parsing in `TypeSafeJevClient.parse_response`. |
| RL controller: item policy (MLP / DeepSets), decision procedure, checkpoints | PASSED | Experiments 4, 4b, 5b |
| RL: behaviour cloning / DAgger from the hindsight expert | PASSED | 5 trainings (`rl_bc*`) |
| RL: PPO and REINFORCE | PASSED | 10 PPO trainings; REINFORCE is PPO with one epoch and no clipping, unit-tested, not used in an experiment |
| RL: other algorithms (DQN-style, GRPO, contextual bandit) | not implemented | `rl/algorithms.py` has the `ALGORITHMS` registry; a new algorithm is one class with `update` |
| Hindsight: evidence tracking through rewrites, requirements-destroyed regret | PASSED | `regret.jsonl` in every run; used by attribution in Experiment 3 |
| Hindsight: continuation regret by replay, counterfactual ablation | PASSED (tests) | unit-tested on the synthetic task; not used at scale in an experiment |
| Hindsight: string and tool-value dependency tracing | PASSED (tests) | recovers labelled evidence on the synthetic and workflow tasks; not used as the evidence source of an experiment |
| Failure attribution (8 labels) | PASSED | Experiments 1, 1b, 2, 3, 5 exercise `evicted` (both causes), `compression_lost_detail`, `archived_not_retrieved`, `task_model_reasoning`, `invalid_action`; `consolidation_incorrect` and `retrieved_but_ignored` only in unit tests |
| Reward terms, configurable weights, per-term logging | PASSED | `rewards.jsonl`; PPO trainings |
| Run folders, resume, metadata (git, CPU, GPU, packages, model ids) | PASSED | resume test with an injected crash; every run |
| Sweeps (grids, paired seeds, stable cell names, parallel) | PASSED | 14 sweeps this session |
| Analysis: paired bootstrap, minimum detectable effect, oracle gap, report | PASSED | `report.md` per sweep |
| Plots (9 figures + learning curve) | PASSED | palette validated with the colour-vision script; figures inspected |
| `memctl.doctor` | PASSED | this report's machine section |
| Docker image | PASSED | the existing image runs the full test suite (148 passed, Python 3.12) with the project mounted; it was not rebuilt this session |

### Reproducibility checklist

| Requirement | State |
|---|---|
| Deterministic seeds | Yes: episode seed = base seed + index; same seeds across sweep cells; RL sampling uses a seeded generator. A test checks that a repeated seed reproduces an episode exactly. |
| Pinned dependencies | `constraints.txt` pins every package; torch is the CPU wheel from the PyTorch index |
| Versioned configs | `schema_version: 1`; the resolved config is saved with every run |
| Model identifiers and revisions | in `metadata.json` (`models`) and each episode row (`agent_model`, `controller_model`); the Hugging Face backend records `name@revision` |
| Git commit with every run | Yes, with a `dirty` flag. **Every run this session is marked dirty** because the code was never committed; see the validity report |
| Environment metadata, CPU/GPU | Yes (`metadata.json`) |
| Resume failed experiments | Yes, at episode granularity, with the same config checked |
| Checkpoint controllers | Yes: `checkpoints/controller` per run, `policy.pt` and `policy_best.pt` per training |
| Unit tests for memory actions; integration test for a full toy episode | Yes: `tests/test_memory_actions.py`, `tests/test_episode_integration.py` |

## Constraints found, and recommendations

1. **GPU memory (4 GB).** Nothing above about 1B parameters runs at 16-bit;
   the earlier work measured that Qwen3.5-4B needs 8.4 GB (3.4 GB in 4-bit, and
   then only for prompts up to about 3,900 tokens). Every language-model
   experiment beyond a smoke test is blocked on this. *Recommendation:* rent a
   card with at least 12 GB (24 GB runs every budget). The Docker image and the
   `hf` backend are ready for it. The scripted-agent experiments do not need
   it and should stay on CPU.
2. **JEV credentials.** The JEV arm cannot be validated at all without
   `JEV_API_KEY`. *Recommendation:* obtain the key, run
   `configs/stage1_smoke.yaml` with `controller: {name: jev}` once to verify
   the client, then run the Experiment 1 grid with `jev` added (about 200
   requests per episode at horizon 200; the docs list $0.042 per million input
   tokens).
3. **Real-model latency here.** `llama3.1:8b` on CPU answers a short prompt in
   about 5 s when the machine is idle and far slower under load. Experiment 7
   ran with one worker. *Recommendation:* do not draw conclusions from it;
   repeat on a GPU host.
4. **Exact oracle cost grows with horizon and tightness.** 7.5 s per episode
   at horizon 2,000 and 800 tokens; the 60 s solver limit was never hit.
   Longer horizons will hit it, and the controller then falls back to the
   approximate plan and says so in `oracle_method`.
5. **LongMemEval-S memory.** Each sweep worker parses the 277 MB file (about
   1.5 GB of Python objects). Use 4 workers on this machine.
6. **Latency measurements are noisy.** Controller latency is measured in
   Python while 10 to 13 sweep workers and up to 10 trainings share 16 cores.
   Compare orders of magnitude, not milliseconds. The exact oracle's solve time
   is recorded separately (`oracle_solve_s`) and is not part of its per-step
   latency.
7. **Dense embeddings on CPU** cost about 20 ms per text when the machine is
   idle; a LoCoMo episode embeds about 800 texts. Fine for Experiment 6c,
   too slow for RL training with dense features on CPU.
8. **Killing a sweep can leave its pool workers running.** The sweep uses a
   `spawn` process pool; if the parent is killed with SIGTERM the workers keep
   writing episodes into their cell folders, and a sweep started afterwards on
   the same folder will produce duplicated episode rows (this happened once to
   Experiment 6c; the cells were deleted and rerun). Check with
   `ps -eo pid,cmd | grep multiprocessing.spawn` before restarting a sweep.
9. **Root-owned `__pycache__` folders** left by earlier Docker runs
   (`memctl/benchmarks/__pycache__`, `memctl/controllers/__pycache__`) cannot
   be removed without `sudo`. They are harmless (Python skips writing there).

## Compute used this session

| Work | Wall time (parallel) |
|---|---|
| Experiment 1 grid: 198 runs × 100 episodes | 15 min on 13 workers |
| Experiments 1b–1d, 2, 3, 5, 6: about 300 runs | about 70 min on 10 workers |
| RL: 5 imitation trainings (240 episodes each) | 5–15 min each |
| RL: 10 PPO trainings (1,200–2,400 episodes each) | 36–53 min each, under contention |
| Experiment 7 smoke tests (real model on CPU) | see EXPERIMENTS.md §7 |
