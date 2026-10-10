# Experiment index

Where every experiment lives: its section, configs, run folders, caches, commit or PR, cost and verdict.
The numbers and caveats are in [EXPERIMENTS.md](../EXPERIMENTS.md); this file only points to them.
`f*` means folds 0–4 and `s*` means seeds. Since 2026-10-10, the per-step traces in `runs/`
(`items`, `steps`, `rewards`, `memory_actions`) are stored as `.jsonl.gz`.
`memctl.runlog.open_trace` reads them transparently.

## Shared reproducibility facts

- **Data:** `data/longmemeval` (LongMemEval-S, 500 questions, 470 answerable; 5 stratified folds from
  `memctl/splits.py`), `data/locomo` (10 conversations, 1,986 questions; 1,535 in categories 1–4),
  `data/streammembench`.
- **Reader protocol (§13 onward):** Qwen2.5-7B-Instruct on vLLM 0.8.5, A40, prompt frozen at `7b5fc30`
  (reasoning note, then `Answer:`, 256 output tokens). Official LongMemEval judge prompts, Qwen-7B judge.
- **Caches:** `cache/generations_*` and `cache/judge_*` hold every paid LLM output. Never delete them:
  regenerating costs money and is not bit-reproducible (STATE.md caveat 2).
- **External code (gitignored, pinned):** `external/LRE@1aa51d6`, `external/openjev@75f22b6`,
  `external/StreamMemBench@b329655`, `external/vllm-a3e0243b`, `external/UtilMem@b7ebd4a`,
  `external/UtilMem-hf` (HF KrisQ/utilmem rev `15a774f…`, no licence), `external/UtilMem-paper`,
  `external/STEEZ-TTT@e8968d5`, `external/STEEZ-TTT-hf`. The last five moved here from agent worktrees
  on 2026-10-10.
- **Phase 2 (before the rebuild):** branch `phase2-locomo-baseline`, tag `phase2-complete`.

## Synthetic tasks, scripted reader (clean rerun at `ef57fc4`, 2026-10-05)

| § | Question | Configs | Runs | Verdict |
|---|---|---|---|---|
| 1 | Do policies differ at the same budget? | `configs/sweeps/exp1_delete_only.yaml` | `runs/exp1_delete_only` | PASSED; salience +0.326 vs FIFO; oracle +0.563 |
| 1b | Attribution recovers injected failure | `exp1b_attribution_check.yaml` | `runs/exp1b_attribution_check` | PASSED (0.199–0.208 vs 0.208) |
| 1c | Horizon at a fixed token budget | `exp1c_horizon_fixed_budget.yaml` | `runs/exp1c_horizon_fixed_budget` | PASSED; oracle gap 0.24 → 0.58 |
| 1d | Dependency distance | `exp1d_dependency_gap.yaml` | `runs/exp1d_dependency_gap` | PASSED |
| 2 | Archive + retrieval vs deletion | `exp2_archive_retrieval.yaml` | `runs/exp2_archive_retrieval` | PASSED; FIFO 0.233 → 0.734 |
| 3 | Compaction / consolidation | `exp3_compaction_consolidation.yaml` | `runs/exp3_compaction_consolidation` | PASSED (conditional; rewriter near-lossless) |
| 4 | Learned vs heuristics vs oracle | `configs/rl/rl_{bc,bc_mlp,ppo*,bc_ppo}.yaml`, `exp4_rl_eval.yaml` | `runs/exp4_rl_eval`, `runs/rl_*` | H1, H2 not supported; imitation ≫ PPO |
| 4b | Learned + archive | `configs/rl/rl_bc{,_ppo}_archive.yaml`, `exp4b_rl_archive_eval.yaml` | `runs/exp4b_rl_archive_eval` | oracle imitation deletes; fixed by D1. 9 cells changed in rerun |
| 5, 5b | Sequential workflow task | `exp5_workflow.yaml`, `exp5b_workflow_rl.yaml`, `configs/rl/rl_*workflow*.yaml` | `runs/exp5_workflow`, `runs/exp5b_workflow_rl`, `runs/rl_*workflow*` | ranking flips; BC 1.000 |
| 6, 6c | Evidence retention on LoCoMo / LME; dense vs lexical | `exp6_*_retention.yaml`, `exp6c_locomo_dense_retrieval.yaml` | `runs/exp6_*`, `runs/exp6c_*` | pipeline PASSED; no QA |
| 7 | LLM smoke tests (dirty, never rerun) | `exp7_llm_smoke.yaml`, `exp7b_*` | reports only, in `runs/_archive_dirty_2026-09-30/exp7*` | pipeline checks, not results |
| D1 | Regret teacher | `configs/rl/d1/*`, `d1_regret_expert_eval.yaml` | `runs/d1_regret_expert_eval`, `runs/rl_bc_archive_{oracle,regret}_s*` | 0.851 at 2% (+0.092 vs best heuristic); `5fd6a3a` |
| GRPO | GRPO vs PPO; oracle warm start | `configs/rl/grpo/*`, `grpo_{recall,workflow}_eval.yaml` | `runs/grpo_*_eval` | oracle-BC → GRPO 0.451 (poisoned); `16c6f5c` |
| D2, D2b | One controller for both tasks | `configs/rl/d2/*` | `runs/d2_joint_*`, `runs/d2b_same_ops_*` | joint = specialists; `9acfdb9`, `103b4c0` |
| D5, D5b, D5c | Priced archive; plug-in Bayes | `configs/rl/d5/*` | `runs/d5_priced_archive_eval`, `runs/d5b_plugin_eval`, `runs/d5c_valued_plugin_eval` | PARTIAL; fails at price 0.2 |

## Real reader

| § | Question | Configs | Runs (+ caches) | Commit / PR | Cost | Verdict |
|---|---|---|---|---|---|---|
| 8 | qwen2.5:3b as task model | `exp8{a,b}_llm_task_model.yaml` | `runs/exp8{a,b}_*` (`cache/generations_qwen3b_ctx16k`) | `5fd6a3a` / #2 | local | less context can be better |
| 9 | 7B task model, 100 episodes | `exp9{a,b}_*.yaml` | `runs/exp9{a,b}_*` (`cache/generations_qwen7b_vllm`) | `3d441b3` / #3 | ≈ $1.35 | regret BC +0.011 at 25% with 40% of tokens |
| 10 | Diagnostics; follow-the-clue | `configs/rl/{diag,bridge,grpo2}/*`, `diag_grpo_eval.yaml`, `bridge_eval.yaml` | `runs/diag*`, `runs/bridge_eval`, `runs/grpo2_eval` | `c76b5cd`, `5b4ceea`, `831be0f` / #5 | free | +0.067 at 2%; GRPO ≈ best-of-8 |
| 11 | Follow-the-clue at 7B; LoCoMo / LME QA | `exp11{a,b,c,d}_*.yaml`, `configs/rl/lme/*` | `runs/exp11*`, `runs/lme_bc_*` (`cache/generations_qwen7b_vllm`, `_yarn64k`) | `2d8fbf4` / #5 | ≈ $2.97 | learned loses to rules on benchmarks |
| 12 | Retrieval headroom | `python -m memctl.rl.headroom` | `runs/t2_headroom` | `1d8a93e` / #6 | free | gates passed (fusion figures corrected in §19a) |
| 13 | LME + LoCoMo, reviewed protocol | `configs/sweeps/exp13/` | `runs/exp13_*`, `runs/lme_floor_*`, `runs/lme_compose4_*` (`cache/generations_qwen7b_vllm_v2`) | `2567c3e` / #6 | ≈ $3.25 | NEGATIVE; context-rot onset about 3k tokens |
| 14 | Token price for kept memory | `configs/rl/lme_n1/`, `configs/sweeps/exp14/` | `runs/exp14_*`, `runs/lme_needed_f*` | `dccff18` / #6 | free | NEGATIVE; AUC 0.62–0.74 |
| 15 | Query-aware floor head (5 of 16) | `configs/rl/lme_n2/`, `configs/sweeps/exp15/` | `runs/exp15_*`, `runs/lme_n2_*` (`cache/generations_qwen7b_vllm_v2_host2`) | #6 | ≈ $0.44 (with §16) | +0.047 (+0.009, +0.087); replicated on a second host |
| 16 | Set-valued GRPO with reader reward | `runs/_configs_exp16` | `runs/lme_grpo_f*` | `2eacf0d` / #6 | ≈ $0.36 | FAIL +0.0125; RQ3 closed |
| 18 | Second-family judge; LoCoMo exploratory | `configs/sweeps/exp18/` | `runs/exp18_judge_{qwen,granite}.json`, `runs/exp18_locomo` (`cache/judge_granite`, `cache/generations_qwen7b_vllm_v2_exp18`) | `9a5c34e`, `e9b6568` / #7 | ≈ $0.16 | judge-independent; κ 0.946 |
| 19, 19b | Adaptive k (5/8/16 of 32) | `configs/rl/lme_n4/`, `configs/sweeps/exp19{,b}/` | `runs/lme_n4_head_f*_{a,b}` (**the `fixed8` heads; f1 = f2, f3 = f4**), `runs/exp19_*`, `runs/exp19b_*` | `4b44dfd`, `ac3bd67` / #8 | free | adaptivity stopped; `fixed8` carried |
| 19c, 19d | Reader test of §19 picks; context-rot pair | `configs/sweeps/exp19_paid/`, `exp19d/` | `runs/exp19_paid_*`, `runs/exp19d_*` (`cache/generations_qwen7b_vllm_v2_exp19`, `_exp19d`) | `316c35e`, `0ecceb0` / #8 | ≈ $0.46 | `fixed8` +0.045 vs §15 head; rot pair not confirmed |
| 20 | Write action (notes) | `configs/sweeps/exp20/` | `runs/exp20_*` (`cache/notes_qwen7b_exp20`) | `fe7051c` / #9 | ≈ $0.81 | gate FAIL |
| 21 | Sequential lossy-memory gate | `memctl/analysis/lossy_gate.py` | `runs/exp21_gate*.{json,md}` | `93f4fe6` / #10 | free | oracle gate PASS +0.177 |
| 22, 22b, 22c | Credit assignment for the note-writer | `runs/_pipelines/exp22*_eval.py` | `runs/exp22/`, `runs/exp22b/`, `runs/exp22c/`, `runs/_cache_exp22`, `runs/exp22*_{eval,report}*` | `c7316f0`, `1e78524`, `ea95030` / #10, #12, #14 | free | per-write > uniform; never > supervised; RQ3 closed |
| 23 | Reader size ladder | `configs/sweeps/exp23/` | `runs/exp23_*`, `runs/exp23_verdicts.json` (`cache/generations_exp23_*`, `cache/judge_exp23`) | `c9bd9e7` / #11 | ≈ $0.83 | claim 2 non-inferior; claim 1 not shown |
| 24 | Reader families | `configs/sweeps/exp24/` | `runs/exp24_*`, `runs/exp24_verdicts.json` (`cache/generations_exp24_*`, `cache/judge_exp24`) | `857bb5b` / #15 | ≈ $1.44 | lift in all six; size step only for Gemma |
| 25 | StreamMemBench transfer | `configs/sweeps/exp25/` | `runs/exp25_close_eval.json`, `runs/_pipelines/exp25_close_*.py` (`cache/simulator_exp25`, `cache/generations_exp25_qwen7b`) | #16 | $0 | does not transfer |
| 26 | OpenJev as controller | `configs/sweeps/exp26/` | `runs/exp26_*`, `runs/exp26_verdicts.json` (`cache/jev_exp26`, `cache/generations_exp26_qwen7b`, `cache/judge_exp26`) | #17 | ≈ $0.64 | B > FIFO; B vs `fixed8` not shown |
| 27 | LRE baseline | `configs/sweeps/exp27/` | `runs/exp27_*`, `runs/lre_exp27_f*` (`cache/generations_exp27_qwen7b`, `cache/judge_exp27`) | `c2258e6` / #19 | ≈ $0.23 (shared with §28) | LRE-slot far worse; LRE-native non-inferior to FIFO |
| 28 | Query rewriting | `configs/sweeps/exp28/` | `runs/exp28_*` (`cache/rewrite_exp28`, `cache/judge_exp28`) | `171faef` / #20 | (shared) | non-inferior, not better |
| 29 | Reader-utility signal | `configs/sweeps/exp29/` | `runs/exp29/`, `runs/exp29_qwen7b_f*`, `runs/exp29_{report.md,verdicts.json}` (`cache/utility_exp29`, `cache/generations_exp29_qwen7b`, `cache/judge_exp29`) | #22 | ≈ $1.02 | closed NEGATIVE |
| 30 | UtilMem | `configs/sweeps/exp30/` | `runs/exp30/` | #24 | $0 | closed without a run |
| 31 | Test-time training on 3B | `configs/sweeps/exp31/` | none | #24 | $0 | closed without a run |
| 32 | Head vs standard rerankers | `configs/sweeps/exp32/` | `runs/exp32_free_f*`, `runs/exp32_latency`, `runs/exp32_rerankers/`, `runs/exp32_qwen7b_f*`, `runs/exp32_{report.md,verdicts.json}`, **`runs/exp32_spotcheck`** (judge audit) (`cache/exp32_ce`, `cache/hf_exp32`, `cache/generations_exp32_qwen7b`, `cache/judge_exp32`) | #23 | ≈ $0.15 | non-inferior to LR and CE, not better |

## Deleted or moved on 2026-10-10

- **Moved from agent worktrees** (the only copies): `runs/exp29*`, `runs/exp32_*`, `runs/exp30`, the
  exp29 caches, and the external folders listed above. Per-worktree logs and spot-check copies are in
  `runs/_from_worktrees/<branch>/`; the three spot-check copies are byte-identical to
  `runs/exp32_spotcheck`. Checksums: `runs/_from_worktrees/MANIFEST.sha256`.
- **Deleted (never reported):** `runs/_aborted` (§15 first host, §16 failed launches), `runs/_aborted_exp20`,
  `runs/_aborted_exp20_b` (§20 unfinished cells), `runs/exp22c_peek_quarantine` (§22c stopped launch;
  no held-out number was computed). Log: `runs/_deleted_2026-10-10.txt`.
- **Trimmed:** `runs/_archive_dirty_2026-09-30` (2.5 GB → 16 MB). It keeps `*.md`, `summary.json`,
  `report.json`, `metadata.json`, `*.yaml`, `*.log` and `train_log.jsonl`. Episodes and traces of the
  dirty runs are gone; the clean rerun replaces them.
