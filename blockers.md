# Blockers

Issues hit while porting memctl to the GPU machine and preparing the full LoCoMo
experiment (2026-09-29). Each is marked **fixed**, **open**, or **design**. Code paths refer
to branch `phase2-locomo-baseline`; `main` is now a scaffold. Fuller detail, with every
measurement, is in `docs/restart.md`.

## 1. Hardware — open

**The GPU has 4 GB; the answering model needs about 12.**

- Machine: RTX 3050 Laptop GPU, 4 GB (3.95 GB usable).
- `Qwen3.5-4B` is 8.41 GB at 16-bit, so it does not load at all.
- 4-bit NF4 fits (3.41 GB) but only up to B = 10% prompts (~3,900 tokens). B = 25%, B = 50%
  and `full_context` all run out of memory, in 4-bit and with CPU offload alike. CPU
  offload was also 17× slower (5.0 s per question against 0.30 s).
- 4-bit changes the generated text, so 4-bit results cannot be compared with the Mac's
  16-bit ones. The Mac baseline (accuracy 0.661 at B = 25%) was therefore never reproduced.
- **Needed:** a card of at least 12 GB; 24 GB runs everything. Rented at about
  $0.35/hour, the full grid was estimated at 10–14 hours.

## 2. Environment and porting — fixed

None of these showed on the Mac. All are correct on any card.

| Problem | Symptom | Fix |
|---|---|---|
| Docker could not see the GPU | `failed to discover GPU vendor from CDI` | NVIDIA Container Toolkit, plus `nvidia-ctk runtime configure` and `nvidia-ctk cdi generate` (root) |
| No C compiler in the image | `Failed to find C compiler` on every generation — Qwen3.5's linear-attention layers are Triton kernels compiled on first use | `build-essential` in the image; Triton cache kept in the models volume |
| Prefill wasted ~2 GB | Out of memory: logits computed for every position (4096 × 248,320) and thrown away | Ask the model for one position's logits (output unchanged) |
| Embedder on the GPU | Out of memory even at B = 10%: `bge-small` held 0.3 GB | `memory.embedder_device: cpu` option |
| Container wrote root-owned files | `runs/`, `cache/`, `data/` needed root to manage | Container runs as the host user (`MEMCTL_UID` / `MEMCTL_GID`) |
| No-Docker path in the run script | `COMPOSE=""` still ran Docker (`:-` treats empty as unset) | `NO_DOCKER=1` flag, tested |
| Docker restart during a build | Build died silently; `tail` hid the exit code | Rebuilt; noted in the porting docs |

## 3. Measurement — design

Found in the data, not in the code. They limit what the experiment could conclude.

- **The retriever, not the controller, decided most outcomes.** At B = 10%, placement put
  all the evidence in context for 7% of questions; store search rescued another 47%. Store
  search found only 48% of the evidence it held. The largest failure label was "evidence in
  store, not retrieved" (40 of 64 wrong answers).
- **Eviction never lost anything.** Every rule controller sends evicted items to the store
  and none drops, so a poor placement costs little and controllers look alike.
- **The oracle had an extra advantage.** It is the only controller that drops items, so
  its store holds almost only evidence and search rarely misses. Part of its lead is a
  cleaner store, not better placement.
- **The earlier sample was too small to see the effect.** At 138 questions McNemar's test
  can only detect a gap of about 10 accuracy points; the observed gap was 4.3. The full
  1,986-question run would detect about 2.3.
- **The archive was never tested.** No controller archives, so `recall()` and the archive
  index were never used.
- **Oracle agreement was misleading.** 88% of graded decisions are ones where the oracle
  drops, which rule controllers can never match.

## 4. Research design — design (why work stopped)

Phase 2 measured something narrower than the thesis. The goal is a three-arm comparison
(no controller / JEV / RL) of agent memory management on **long-horizon tasks**, where
memory operations are **tool calls**. Phase 2 cannot express that:

- **No consolidation.** Items are immutable and a controller can only move an existing
  item, never summarise or merge.
- **Fetch is not a controller action.** Search is automatic; only the answering agent can
  recall.
- **The controller is an eviction policy** run when the budget overflows, not a policy
  choosing tool calls.
- **LoCoMo is long-context question answering, not a long-horizon task.** Every question
  comes after the conversation ends; nothing the agent forgets breaks a later step.
  **LongMemEval has the same shape**, so it would inherit this blocker.
- **"No memory controller" needs defining.** With a budget something must evict; the
  candidates are unlimited context or plain truncation (`keep_newest`).
- **JEV:** no API key yet. When it arrives, compare it with a control that has the same
  four actions, and count its own API cost.
- **RL:** the binding constraint is sample cost (one episode is about 8 minutes of GPU and
  the cache cannot help), then an observation with no text in it. The reward function
  matters less than either.
