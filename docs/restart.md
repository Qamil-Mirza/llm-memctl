# Restarting: what exists, what survives, and what the new design has to decide

Written 2026-09-29, when the thesis goal was restated and turned out to be wider than what
Phase 2 built. This file exists so the next design does not have to re-derive what has
already been measured, and does not re-inherit the assumptions that turned out to be too
narrow.

`main` has since been cleared back to a scaffold. The Phase 2 code is preserved in full on
branch `phase2-locomo-baseline` (tag `phase2-complete` marks its last code commit), and
**every file path below refers to that branch**: read one with
`git show phase2-locomo-baseline:<path>`, or bring it back with
`git checkout phase2-locomo-baseline -- <path>`.

## 1. The goal, as stated

A **three-arm comparison of how an agent manages memory on long-horizon tasks**:

1. **Baseline** — the agent on a long-horizon task with no memory controller.
2. **JEV** — JEV decides memory **eviction, consolidation, and fetch / send to a
   persistent store**, framed as *tool calls the policy makes*.
3. **RL** — an RL backbone decides those same tool calls.

## 2. Where Phase 2 does not reach

Four gaps, each confirmed in the code rather than inferred:

| Gap | Where it is fixed in the code |
|---|---|
| **Consolidation is impossible.** Nothing can summarise, merge or rewrite an item. | `Item` is `@dataclass(frozen=True)` (`items.py`); `Placement` carries only `item_id` and `place` (`controllers/base.py`), so a controller can only *move an item that already exists*. (`summary.py` is metrics, not summarisation.) |
| **Fetch is not a controller action.** | Store search is automatic inside the harness (`memory.search_store`); `recall(id)` is emitted by the *frozen agent* in its output text (`agent.py`). The controller has no fetch action. |
| **The controller is not a tool-calling policy.** It is an eviction policy invoked on overflow / arrival / session-end triggers. | `episode.run_conversation` calls the controller on those triggers only; `candidates()` is the in-context set plus the arriving item, so a controller cannot reach into `STORE` or `ARCHIVE` to reorganise. `docs/architecture.md`: "The agent only reads memory. It never changes where items live." |
| **LoCoMo is long-*context* QA, not a long-*horizon task*.** It measures recall. | `benchmarks/locomo.py` appends every question *after* every item, so all questions are asked at the end. No tool use, no task completion, no decision whose consequence feeds a later step. |

**On "no memory controller" as a baseline:** with a token budget there is no such thing —
something must decide what goes. The honest baseline is one of two things already here:
`full_context` (unlimited budget, never evicts: the infinite-context reference) or
`keep_newest`, which is FIFO truncation, i.e. what a chat application does by default.
`keep_newest` *is* the no-controller arm; the name disguises it.

## 3. Measured facts worth keeping

These cost real time to establish. None of them depends on the Phase 2 framing.

### The machine and the models

- GPU here is an **RTX 3050 Laptop, 4 GB** (3.95 GB usable).
- `Qwen3.5-4B` loads as `Qwen3_5ForCausalLM` (text-only; the 0.67 GB vision tower is never
  read). **8.41 GB at 16-bit**, so it does not load on this card. 4-bit NF4 is **3.41 GB**
  and works only up to ~3,900-token prompts; B = 25% and above run out of memory, in 4-bit
  and with CPU offload alike. **~12 GB is the real minimum; 24 GB runs everything.**
- These are hybrid-attention models: only 8 of the 4B model's 32 layers use full attention,
  so the KV cache is about **32 KB per token**. The prompt is rarely the problem; the
  weights are.
- 4-bit changes the generated text, so 4-bit and 16-bit results are not comparable. It is
  in the generation cache key for that reason.

### Porting defects found (all fixed, all in git)

1. Docker 29 needs `nvidia-ctk cdi generate` as well as `runtime configure`, and it is a
   root step. 2. `python:3.12-slim` has no C compiler, and Qwen3.5's linear-attention layers
   are Triton kernels compiled on first use. 3. The shared-prefix prefill materialised a
   logit per position per vocabulary entry (4096 × 248,320 ≈ 2 GB) and discarded it.
4. The embedder and its CUDA context hold ~0.3 GB, which on a small card is decisive.
5. The container wrote root-owned files into `runs/`, `cache/` and `data/`.

Fixes 2–5 are correct on any card. All are documented in `docs/porting.md` and
`docs/decisions.md` entries 37–47.

### LoCoMo, if it is ever used again

- 10 conversations, **5,882 items**, **1,986 questions** (1,540 non-adversarial, 446
  adversarial). Mean history 26,923 word-tokens.
- **Evidence is 24.3% of items and about 31% of tokens.** So B = 50% is the first budget at
  which a perfect policy could hold all the evidence in context.
- Only 5 questions have no evidence label.

### The retrieval bottleneck (the most important finding)

At B = 10% with `keep_newest`, of the questions that had all their evidence available:

| Source | Share |
|---|---|
| placement kept it in context | **0.073** |
| a store search rescued it | **0.472** |

Store evidence recall is **0.481**, and the largest failure label was
`evidence_in_store_not_retrieved` (40 of 64 wrong answers) against
`evidence_present_model_wrong` (22). **About seven eighths of the evidence reaching the
model got there because the retriever found it, not because the controller kept it.**

Two structural reasons, both of which a new design should decide deliberately:

- **Eviction is non-destructive.** No rule controller ever drops, so a bad placement does
  not lose information — it moves the item into a ~48%-recall lottery. That compresses the
  measurable difference between controllers toward zero.
- **The oracle is the only controller that drops**, so its store is small and nearly pure
  evidence and its search rarely misses. Part of its apparent advantage is a cleaner store,
  not better placement. `compare.py` now separates the two.

### Statistical power

Smallest oracle-minus-baseline gap McNemar's exact test can call solid at p < 0.05, at 25%
discordance:

| Questions | Detectable gap |
|---|---|
| 138 | **~10 accuracy points** |
| 1,986 | **~2.3 points** |

The gap previously observed was 4.3 points. **The earlier experiment could not have
detected its own effect**; "not statistically solid" was a property of the sample size, not
a finding. Any new design should state its minimum detectable effect before running.

### Timing

- 4-bit, B = 10%, end to end including judge, controller and oracle annotation:
  **2.56 s per question** (180 questions in 7m44s).
- Oracle ILP: 25 s for all 10 plans at B = 10%, under a second at larger budgets.
- Pipeline validation run (4-bit, B = 10%, 3 conversations, 180 questions): accuracy 0.644,
  F1 0.485, BLEU-1 0.440, 40 memory and 22 reasoning failures of 64 wrong. Kept in
  `runs/locomo_4b_4bit/`. Not comparable with 16-bit numbers.

## 4. What survives the redesign

Reusable more or less as is:

| Component | Why it survives |
|---|---|
| `memory.py` — `MemoryState`, budget accounting, the all-or-nothing `apply` | The invariant "placements are checked before anything moves" is worth keeping whatever the action space becomes |
| `items.py` — `Place`, model-free token counting | Budgets that mean the same thing for every model |
| `attribution.py` — one failure label per wrong answer | The memory-versus-reasoning split is the measurement that makes any of this a result rather than a leaderboard |
| `controllers/oracle*.py` | Two roles beyond being a ceiling: it is **dense per-decision supervision** (`oracle_place` is logged next to every decision), and it bounds what any policy could achieve |
| `features.py` | The observation for a learned policy — but see the RL note below |
| `llm.py`, `run.py`, `report.py`, `compare.py`, `plots.py` | Model backends and caching, run folders stamped with a git hash, the paired test, the power analysis, the placement-versus-retrieval split |
| `Dockerfile`, `docker-compose.yml`, `scripts/` | The GPU porting work is independent of the research question |

Must change: the agent / controller boundary, the action space, the item model (for
consolidation), and the benchmark.

## 5. What the new design has to decide

Roughly in dependency order — each answer constrains the next.

1. **What is the long-horizon task?** It needs interleaved work, later steps that depend on
   earlier results, and a success signal. Without "forgetting the wrong thing makes a later
   step fail", consolidation and fetch have nothing to bite on and the memory arm cannot be
   distinguished from a retrieval benchmark.
2. **Are memory operations tools the agent calls, or a policy invoked on triggers?** This is
   the agent / controller boundary and it determines everything downstream, including the
   `step()` signature of any RL environment.
3. **What is the action space?** Per-item placement (currently 4^N per overflow), or
   operations with arguments (`evict(id)`, `consolidate(ids)`, `fetch(query)`,
   `store(id)`)? The second matches the stated goal; the first is what exists.
4. **How does consolidation work?** Who writes the summary — the agent, a separate model,
   the controller? Does it replace its sources or coexist with them? How are its tokens
   charged, and is information allowed to be lost? This breaks two current invariants
   (items are immutable; one item is in exactly one place) and is the most invasive change.
5. **Is fetch an action with a cost?** Who pays, how many per step, and does automatic
   top-k search survive alongside it? If search stays automatic, expect it to dominate
   again as it does now.
6. **What is the cost axis when the controller itself spends?** Phase 2 used average prompt
   tokens, which measures only what the *agent* read. JEV's dollars, RL inference and
   consolidation generations all sit outside it. A controller that spends heavily currently
   looks free.
7. **What exactly is the baseline arm?** Unlimited context, truncation, or both.
8. **For RL:** episode boundary, observation, reward, and the sample budget. See below.
9. **Does the token budget stay the binding resource,** or is it wall-clock, dollars, or
   tool calls?

## 6. A note on the RL arm

The reward function is not the binding lever. In rough order of how much they constrain the
result:

1. **Sample cost.** One episode is one conversation: ~200 questions at ~2.5 s each, so
   **~8 minutes per episode**. A thousand episodes is days of GPU. The generation cache does
   not help, because changing the policy changes the placements, which changes the prompts,
   which misses the cache every time.
2. **The observation has no text semantics.** `ItemFeatures.as_vector()` is nine numbers
   (age, tokens, times referenced, steps since last use, similarity to the current query,
   is_incoming, and a 3-way one-hot on kind). A policy reading only that can learn recency
   and size heuristics; it cannot learn "this fact will be asked about". Embeddings have to
   enter the observation.
3. **Action-space shape**, per item 3 above.
4. **Credit assignment.** Task success arrives once, at the end, after thousands of
   decisions.
5. **The reward definition** itself.

**The shortcut already paid for:** `oracle_place` sits next to every decision in
`decisions.jsonl`, which is dense per-decision supervision. Behaviour cloning or DAgger
against the oracle is far more tractable than RL from sparse reward, and the failure
labels give per-decision blame usable for shaping. A supervised policy first, then RL to
improve on it, with the oracle bounding both.

## 7. Open threats to validity, for whatever comes next

- **JEV can drop and archive; no rule baseline can.** "JEV beats the baseline" would
  conflate *JEV is smart* with *being allowed to drop helps*. An action-space-matched
  control is needed (`random` already accepts a `destinations` list).
- **Oracle agreement is 88% dominated by decisions where the oracle drops** (51,095 of
  58,018 in the validation run), which rule controllers can never match. The headline
  agreement figure is near-meaningless across controllers; read it per choice.
- **LoCoMo is public**, so an LLM-based controller may have memorised which turns get asked
  about. Nothing inside the harness can detect this. Running the same controller on
  synthetic data it cannot have seen is the available check.
- **An LLM controller is not deterministic** and JEV responses are not cached, so reruns
  cost money and may not reproduce. More than one seed is needed to separate a real
  improvement from one sample of a noisy controller.
- **`ARCHIVE` and `recall()` were never exercised** by any Phase 2 run: every rule
  controller evicts to `STORE`. No Phase 2 result says anything about the archive half of
  the memory model.
