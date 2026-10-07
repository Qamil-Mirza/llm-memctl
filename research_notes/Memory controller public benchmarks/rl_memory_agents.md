# RL- and imitation-trained memory managers for LLM agents (LoCoMo / LongMemEval focus)

Note: all numbers below are self-reported by the paper's authors unless marked otherwise. No independent reproductions were found within the search budget. Notes gathered 2026-10-06 (about 18 tool calls). Several systems on the list (MemGen, ReasoningBank, Memento, SUPO's numbers, ReSum's numbers, MEM1's per-benchmark tables) were only partly covered; see Gaps.

## Q1. Memory-R1: exact setup, reward, data, scores, ablations

### Takeaway
Memory-R1 trains two separate LLM policies: a Memory Manager (ADD/UPDATE/DELETE/NOOP) and an Answer Agent that "distills" 60 retrieved memories. Both use only an outcome exact-match (EM) reward and 152 LoCoMo QA pairs. It reports large gains over Mem0 on LoCoMo and zero-shot transfer to LongMemEval. Retrieval is always done (RAG top-60), and RL only shapes what gets written and which memories the reader uses. That is the "hybrid" design the thesis asks about.

### Cited Findings
- arXiv 2508.19828; v1 dated 2025-08-27, v5 (current) 2026-01-14 — [arXiv abs](https://arxiv.org/abs/2508.19828)
- Architecture: a Memory Manager that "learns structured operations, including ADD, UPDATE, DELETE, and NOOP" and an Answer Agent that "pre-selects and reasons over relevant entries"; trained "with outcome-driven RL (PPO and GRPO)"; model scales 3B–14B; benchmarks LoCoMo, MSC, LongMemEval — [arXiv abs](https://arxiv.org/abs/2508.19828)
- Manager reward: operations are "judged by their effect on downstream QA", scored by exact match from a **frozen** Answer Agent. Answer Agent reward: EM between predicted and gold answer. GRPO advantage A_i = (r_i − mean(r))/std(r) with KL to a reference policy — [arXiv HTML](https://arxiv.org/html/2508.19828)
- The manager and the answer agent are trained separately "to ensure stability under sparse rewards" — [arXiv HTML](https://arxiv.org/html/2508.19828)
- Data: LoCoMo with a 1:1:8 train/val/test split, i.e. **152 / 81 / 1307 questions**. Memory banks are built from the preceding dialogue turns. **60 candidate memories retrieved per question** by similarity-based RAG — [arXiv HTML](https://arxiv.org/html/2508.19828)
- Compute: 4× H100 80GB (8 for Qwen-2.5-14B), total batch 128, micro-batch 2 per GPU — [arXiv HTML](https://arxiv.org/html/2508.19828)
- LoCoMo, LLaMA-3.1-8B (F1 / BLEU-1 / LLM-Judge): Mem0 30.41 / 22.22 / 45.68; Memory-R1-PPO 41.05 / 32.91 / 57.54; Memory-R1-GRPO **45.02 / 37.51 / 62.74**. The authors report GRPO is +28% F1, +34% BLEU-1, and +30% Judge (relative) over the MemoryOS baseline — [arXiv HTML](https://arxiv.org/html/2508.19828)
- LoCoMo, Qwen-2.5-7B: PPO 41.72 / 33.70 / 59.53; GRPO 43.14 / 36.44 / 61.51 — [arXiv HTML](https://arxiv.org/html/2508.19828)
- LongMemEval (zero-shot, trained only on LoCoMo), Table 5: LLaMA-3.1-8B PPO 43.60 / 39.50 / 55.20, GRPO 45.20 / 39.30 / 55.40; Qwen-2.5-7B PPO 40.30 / 35.50 / 47.40, GRPO 46.70 / 41.10 / 57.80 — [arXiv HTML](https://arxiv.org/html/2508.19828)
- MSC: the paper says it evaluates zero-shot on MSC, but the fetch tool found no separate MSC table with baselines in the HTML — [arXiv HTML](https://arxiv.org/html/2508.19828)
- Ablations (PPO, LoCoMo F1): full 41.0; without Memory-Manager RL 34.5; without Answer-Agent RL 32.5; without memory distillation 39.3. Distillation also takes GRPO from 41.0 to 45.0 — [arXiv HTML](https://arxiv.org/html/2508.19828)
- No explicit reward shaping or class rebalancing of NOOP against the other operations was found in the text — [arXiv HTML](https://arxiv.org/html/2508.19828)

### Inferences
- Memory-R1 never asks the learned policy to decide *whether to retrieve*. Retrieval of 60 candidates is fixed, and learning acts on writes and on filtering at read time. That avoids the "retrieval almost never fires" failure our DeepSets controller shows.
- Its manager reward comes from a question *about the memory state the manager just produced*. Each training example is (dialogue up to a point, question), so every operation gets a QA-based signal. Our LongMemEval training had one question per episode and labels that were almost all 0.
- 152 QA pairs is small, but the trained models are 3B–14B LLMs with strong priors. That is not evidence that a 27k-parameter policy could learn from that little data.
- The answer agent's RL (−8.5 F1 when removed) matters about as much as the manager's (−6.5 F1). Some of the gain is in reading, not memory management. That is a confound if we compare our controller with Memory-R1 head to head.

### Gaps
- Exact MSC numbers, the LongMemEval baseline rows, the embedding model used for RAG, the number of epochs, and the GRPO group size were not located in the fetched text.
- No independent reproduction of Memory-R1 was found.

## Q2. Other systems: action space, reward, credit assignment, model size, benchmarks

### Takeaway
Almost every 2025–2026 system trains the LLM itself (3B–14B) with GRPO or a variant, using an outcome QA reward. The terminal advantage is copied to every step or context in the trajectory: MemAgent, MemSearcher, ReSum-GRPO, and AgeMem all do this. Mem-α and AgeMem add shaped auxiliary rewards (format, compression, memory quality). Only Mem-α, MemSkill, and Memory-R1 report LongMemEval or LoCoMo numbers. Of those, MemSkill is the only one with a small separate learned controller.

### Cited Findings
**Mem-α (arXiv 2509.25911; Anuttacon / UCSD / Stanford)**
- Memory has three parts. Core memory is a summary of at most 512 tokens and supports update only. Semantic and episodic memory support insert, update, and delete. Every action is a structured function call — [arXiv HTML](https://arxiv.org/html/2509.25911v1)
- Reward r_t = r1 + r2,t + β·r3 + γ·r4,t:
  - r1: QA accuracy through a RAG pipeline over the *final* memory state.
  - r2: fraction of tool calls that execute.
  - r3: compression, 1 − mem_len/chunk_len.
  - r4: fraction of valid operations, judged by a Qwen3-32B validator.
  - Weights β = 0.05, γ = 1.0.

  Training uses GRPO without a KL term — [arXiv HTML](https://arxiv.org/html/2509.25911v1)
- Backbone Qwen3-4B. The authors say Qwen3-8B had instruction-following failures. Compute: 32 H100, 3 days, 205 steps, learning rate 1e-6, batch 32, 8 rollouts — [arXiv HTML](https://arxiv.org/html/2509.25911v1)
- Data: 562 instances, **stratified down from 4,139 "due to class imbalance"**. Sources: SQuAD, HotpotQA, PerLTQA, LME-Train, NLU, TREC-C, PubMed-RCT, BookSum. Maximum length 30k tokens. **Each instance carries 4–100 evaluation questions** — [arXiv HTML](https://arxiv.org/html/2509.25911v1)
- MemoryAgentBench (accurate retrieval / test-time learning / long-range understanding / average):

  | System | AR | TTL | LRU | Avg |
  |---|---|---|---|---|
  | Mem-α (4B) | 0.647 | 0.607 | 0.129 | 0.592 |
  | BM25 RAG-Top2 | 0.574 | 0.550 | 0.065 | 0.502 |
  | Long-context Qwen3-32B | — | — | — | 0.461 |
  | MemAgent | — | — | — | 0.198 |
  | MEM1 | — | — | — | 0.071 |

  The authors claim it generalizes from 30k-token training to more than 400k tokens — [arXiv HTML](https://arxiv.org/html/2509.25911v1)

**MemAgent (arXiv 2507.02259; ByteDance Seed / Tsinghua; ICLR 2026 oral)**
- Overwrite memory of 1,024 tokens, 5,000-token chunks, 8K total context — [arXiv HTML](https://arxiv.org/html/2507.02259v1)
- Training: Multi-Conv DAPO with a rule-based binary verifier. The final-answer reward is "uniformly applied across all conversations originating from the same sample". Models Qwen2.5-7B/14B-Instruct. Data: 32,768 HotpotQA-synthetic samples (filtered from 80k by removing questions answerable without context), 32K-token training contexts — [arXiv HTML](https://arxiv.org/html/2507.02259v1); [ICLR 2026](https://iclr.cc/virtual/2026/oral/10007826)
- Reported to extrapolate from 8K training to 3.5M-token QA with less than 10% loss, and above 95% on 512K NIAH — [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/4264ee4376776907c0b87ed70b959585-Abstract-Conference.html)

**MEM1 (arXiv 2506.15841; NeurIPS 2025)**
- At each turn the agent rewrites one compact internal state and discards the earlier context, so memory stays near constant.
- Training environments are built by composing existing QA datasets into multi-objective task sequences.
- MEM1-7B reports 3.5× better performance and 3.7× less memory than Qwen2.5-14B-Instruct on 16-objective multi-hop QA — [arXiv HTML](https://arxiv.org/html/2506.15841v1)

**AgeMem / Agentic Memory (arXiv 2601.01885; Jan 2026, revised Jul 2026; ACL'26 SAC Highlight per abstract page)**
- Six tools: LTM Add / Update / Delete and STM Retrieve / Summary / Filter — [arXiv HTML](https://arxiv.org/html/2601.01885)
- Three-stage progressive RL: (1) build LTM from casual conversation; (2) reset context while LTM persists, then manage STM under deliberately inserted distractor messages; (3) answer a formal query using both — [arXiv HTML](https://arxiv.org/html/2601.01885)
- Step-wise GRPO: the terminal advantage (r_T − μ)/(σ + ε) is "broadcast to all preceding steps of the same trajectory" — [arXiv HTML](https://arxiv.org/html/2601.01885)
- Reward = wᵀ[R_task (LLM judge), R_context (compression, preventive actions, information preservation), R_memory (storage quality, maintenance, relevance)] + penalties (overflow, excess tool use, interaction limits) — [arXiv HTML](https://arxiv.org/html/2601.01885)
- Models Qwen2.5-7B-Instruct and Qwen3-4B-Instruct. Benchmarks are ALFWorld, SciWorld, PDDL, BabyAI, HotpotQA (no LoCoMo or LongMemEval):
  - Average AgeMem 41.96 vs Mem0 37.14 (Qwen2.5).
  - Average AgeMem 54.31 vs A-Mem 45.74 (Qwen3).
  - RL adds +8.53 / +8.72 points over the non-RL variant.

  — [arXiv HTML](https://arxiv.org/html/2601.01885); code at GitHub y1y5/AgeMem per [arXiv abs](https://arxiv.org/abs/2601.01885)

**MemSearcher (arXiv 2511.02805)**
- Keeps a compact memory in place of the full history.
- Multi-context GRPO "propagates trajectory-level advantages across all conversations".
- Qwen2.5-3B/7B-Instruct, trained on the Search-R1 data. Reports +11% / +12% average over seven benchmarks; the 3B model beats 7B baselines — [arXiv abs](https://arxiv.org/abs/2511.02805v1)

**ReSum (arXiv 2509.13313)**
- Web agent with periodic summarization; ReSumTool-30B is a fine-tuned Qwen3-30B.
- ReSum-GRPO cuts long trajectories into segments and broadcasts the trajectory-level advantage to every segment — [arXiv HTML](https://arxiv.org/html/2509.13313v3)

**SUPO**
- Summarization-augmented policy optimization. It derives a policy gradient that trains tool use and summarization end-to-end beyond a fixed context limit. Described only via search snippet; numbers not retrieved — [search result pointing to arXiv/ACL 2026](https://aclanthology.org/2026.acl-long.966.pdf)

**MemRL (arXiv 2601.03192)**
- Non-parametric "runtime RL": the LLM is frozen, and episodic-memory entries carry learned Q-values (utilities) updated from environment feedback.
- Two-phase retrieval: filter by semantic relevance, then select by Q-value.
- Benchmarks HLE, BigCodeBench, ALFWorld, Lifelong Agent Bench (no LoCoMo) — [arXiv HTML v2](https://arxiv.org/html/2601.03192v2); [arXiv abs](https://arxiv.org/abs/2601.03192v1)

**MemSkill (arXiv 2602.02474)**
- Covered in Q4.

**LightMem (arXiv 2510.18866)**
- Not an RL system. Pipeline: sensory compression, topic-grouped STM, offline "sleep-time" LTM consolidation.
- Reports up to +7.7% / +29.3% accuracy on LongMemEval / LoCoMo with GPT and Qwen backbones.
- Token reduction up to 38× / 20.9×; API-call reduction up to 30× / 55.5× — [arXiv abs](https://arxiv.org/abs/2510.18866)

**ContextPilot (arXiv 2608.28476)**
- Appeared in search as "Teaching Agents for Proactive Context Management via Fine-grained RL". Not read — [arXiv PDF](https://arxiv.org/pdf/2608.28476)

### Inferences
- The dominant recipe is: LLM policy + GRPO + binary or LLM-judge outcome reward + the same advantage for every step. Nobody in this set reports learned per-step credit (a critic or a hindsight teacher) for memory operations. Our DAgger-from-regret-teacher stage is unusual. That makes it a possible contribution, but it is also less validated.
- Mem-α is the only system with a strong BM25 RAG baseline in its main table. It beats BM25-top2 by about 0.07 average on MemoryAgentBench, so its margin over a simple retrieval heuristic is modest. That matches our finding that top-k BM25 is hard to beat.
- MemAgent and MEM1 do poorly on Mem-α's benchmark (0.198, 0.071). That suggests RL memory agents trained on synthetic multi-hop QA do not transfer well to other memory task types. This is the third party's comparison, not an independent one.

### Gaps
- MemGen, ReasoningBank, and Memento were not researched within the budget. As far as I recall, ReasoningBank and Memento are case- or strategy-memory approaches for agent tasks rather than conversational QA, but this was not verified here.
- Exact SUPO and ReSum benchmark numbers, MEM1 per-task tables, and Mem-α's LongMemEval-only sub-score were not retrieved.

## Q3. How do they handle delayed, sparse reward for memory operations?

### Takeaway
There are five main tricks:
1. **Many questions per episode**, so every memory state is probed by several questions: Mem-α (4–100 questions per instance), MemSkill (~200 queries per LoCoMo conversation), and Memory-R1 (one QA reward per pre-question memory state).
2. **Copying the trajectory-level advantage to every step or context**: MemAgent, MemSearcher, ReSum, AgeMem.
3. **Shaped dense auxiliary rewards**: Mem-α's per-step format and validity rewards plus a compression reward; AgeMem's context and memory-quality terms plus penalties.
4. **Staged curriculum**: AgeMem's three stages with distractors.
5. **Data rebalancing**: Mem-α stratified 4,139 instances down to 562 because of class imbalance.

No paper found reweights a rare "retrieve" action class directly.

### Cited Findings
- Mem-α: QA reward over the final memory state, using 4–100 questions per instance. Dense per-step terms: tool-format success, and operation validity judged by Qwen3-32B. A compression term with β = 0.05 — [arXiv HTML](https://arxiv.org/html/2509.25911v1)
- Mem-α subsampled its training set "due to class imbalance" (4,139 to 562 instances, stratified) — [arXiv HTML](https://arxiv.org/html/2509.25911v1)
- AgeMem: a three-stage curriculum with distractor messages, plus step-wise GRPO with the terminal advantage broadcast to every earlier step. It is explicitly designed for the "sparse and discontinuous rewards induced by memory operations" — [arXiv HTML](https://arxiv.org/html/2601.01885); [arXiv abs](https://arxiv.org/abs/2601.01885)
- AgeMem penalizes excessive tool use and context overflow. Its context-management reward includes "information preservation" — [arXiv HTML](https://arxiv.org/html/2601.01885)
- Memory-R1 trains the manager and the answer agent separately (the answer agent is frozen during manager training) "to ensure stability under sparse rewards" — [arXiv HTML](https://arxiv.org/html/2508.19828)
- MemAgent applies the final reward uniformly to every conversation of a sample. It filters out questions answerable without context, which removes about 50% of the data and so ensures the reward depends on memory — [arXiv HTML](https://arxiv.org/html/2507.02259v1)
- MemSkill: reward is task performance (F1 or success rate) on the downstream training queries after memory construction finishes. LoCoMo gives ~200 queries per conversation — [arXiv HTML](https://arxiv.org/html/2602.02474)

### Inferences
- Our LongMemEval training failure (one question per episode, labels almost all 0) is the exact situation these papers design around.
  - Concatenating several LongMemEval questions over a shared or merged haystack (as Mem-α and MemSkill do with multi-query instances) would densify the signal.
  - Training on LoCoMo, which has about 200 questions per conversation, gives far more labels per episode.
- MemAgent's filtering idea carries over directly: drop training questions where the regret label does not depend on memory.
- AgeMem's penalty on "information preservation" and Mem-α's validity reward both make careless deletion costly. Our teacher treats deletion as free. A deletion cost, or a reward for retained evidence, is the obvious fix with precedent.

### Gaps
- No paper found uses explicit class weighting or focal loss on a retrieve/no-retrieve action, or reports how often its policy retrieves. I found no source on this.

## Q4. Does anyone train a small separate policy for memory? Evidence of synthetic-to-real transfer?

### Takeaway
MemSkill is the closest match. It trains a lightweight MLP controller with PPO that picks Top-K memory "skills" executed by a frozen 70B–80B LLM. It is trained on 6 LoCoMo conversations and transfers to LongMemEval and to a different executor LLM. It is trained on *real* LoCoMo dialogue, though, not synthetic data. MemRL learns utilities without training any network weights. No paper was found showing a small policy trained on synthetic data transferring to real dialogue memory QA. The synthetic-trained LLM agents (MemAgent, MEM1) score poorly on another group's mixed benchmark.

### Cited Findings
- MemSkill controller: a state encoder (text span + retrieved memories), a skill encoder, and a scorer. All are "lightweight MLPs". It scores state–skill pairs so that the skill bank can grow — [arXiv HTML](https://arxiv.org/html/2602.02474)
- MemSkill actions: ordered Top-K skill selection via Gumbel-Top-K; K = 3 in training, K = 7 at evaluation on LoCoMo/LongMemEval. Trained with PPO using the without-replacement joint probability — [arXiv HTML](https://arxiv.org/html/2602.02474)
- MemSkill data: 10 LoCoMo conversations with ~200 queries each, split 6/2/2 train/val/test; ~100 stratified LongMemEval samples used for transfer only. Executor: LLaMA-3.3-70B-Instruct or Qwen3-Next-80B-A3B-Instruct — [arXiv HTML](https://arxiv.org/html/2602.02474)
- MemSkill results (LLaMA, LLM-judge):

  | System | LoCoMo | LongMemEval | ALFWorld success rate |
  |---|---|---|---|
  | MemSkill | 53.82 | 60.89 | 80.36 |
  | MemoryOS | 48.64 | 39.83 | 61.77 |
  | A-Mem | 49.71 | 38.04 | 66.51 |
  | Mem0 | 34.58 | 46.81 | 77.82 |

  Moving to a Qwen executor without retraining gives LoCoMo 54.14 — [arXiv HTML](https://arxiv.org/html/2602.02474)
- MemSkill processes 512-token spans: 215 LLM calls vs 1,288 for MemoryOS on LoCoMo — [arXiv HTML](https://arxiv.org/html/2602.02474)
- MemRL: frozen LLM, with learned Q-values over memory entries for value-aware retrieval (non-parametric) — [arXiv abs](https://arxiv.org/abs/2601.03192v1)
- PPRO (arXiv 2607.00017): trains a query rewriter with GRPO, using evidence-retrieval quality and answer quality as reward. Memory bank and answer model are frozen. It reports gains on LoCoMo and LongMemEval-S over training-free and training-based baselines; numbers and model size are not in the abstract — [arXiv abs](https://arxiv.org/abs/2607.00017)
- A Hugging Face model "knowme-memory-gate-model-grpo" (a GRPO-trained memory gate) exists, but no paper or numbers were found — [HF](https://huggingface.co/Longlong418/knowme-memory-gate-model-grpo)
- Mem-α (trained partly on synthetic and SQuAD/HotpotQA data at ≤30k tokens) is reported to generalize to more than 400k tokens. Its training mix includes LME-Train, which is LongMemEval-derived real-ish data — [arXiv HTML](https://arxiv.org/html/2509.25911v1)
- MemAgent and MEM1 were trained on synthetic or composed QA. On MemoryAgentBench, run by the Mem-α authors, they score 0.198 and 0.071 average vs BM25-top2 at 0.502 — [arXiv HTML](https://arxiv.org/html/2509.25911v1)

### Inferences
- MemSkill is the strongest precedent for the thesis's design: a small, separately trained policy plus a frozen large LLM, PPO, an outcome F1 reward, LoCoMo, and transfer to LongMemEval. Its key differences:
  - It trains on real LoCoMo with dense multi-query rewards.
  - Its actions choose *how to write* (skills), not whether to retrieve.
  - Its executor is 70B+.
  It would be the natural comparison point in the thesis.
- Evidence that synthetic-only training transfers to real dialogue is weak or negative across the papers found. That is consistent with our result.

### Gaps
- No paper reports the parameter count of MemSkill's MLPs in the fetched text.
- No study was found that explicitly measures synthetic-to-real transfer for a small memory policy.

## Q5. Hybrid designs: always retrieve, learn write/evict/consolidate

### Takeaway
The top LoCoMo/LongMemEval systems mostly always retrieve and put learning (or engineering) into the write side or the read-side filter. Examples: Memory-R1 (fixed top-60 RAG plus learned writes and distillation), MemSkill (learned write skills), and LightMem / EDU-graph baselines (heuristic writes and consolidation with fixed retrieval). None found makes "retrieve or not" a learned, sparse decision.

### Cited Findings
- Memory-R1 always retrieves 60 memories by similarity and learns ADD/UPDATE/DELETE/NOOP plus answer-time selection — [arXiv HTML](https://arxiv.org/html/2508.19828)
- Mem-α computes its reward through a fixed RAG pipeline over the memory the agent writes. The agent only learns writes — [arXiv HTML](https://arxiv.org/html/2509.25911v1)
- LightMem: fixed retrieval with offline "sleep-time" consolidation; large token and API savings plus accuracy gains on LongMemEval/LoCoMo — [arXiv abs](https://arxiv.org/abs/2510.18866)
- "A Simple Yet Strong Baseline" (arXiv 2511.17208): decomposes sessions into enriched elementary discourse units in a graph. Uses dense retrieval plus LLM filtering, with no RL or fine-tuning. Claims competitive LoCoMo/LongMemEval_S performance with shorter contexts by being "non-compressive" — [arXiv abs](https://arxiv.org/abs/2511.17208)
- PPRO keeps retrieval always on and learns to rewrite the query — [arXiv abs](https://arxiv.org/abs/2607.00017)

### Inferences
- The literature supports moving our controller to "always retrieve BM25 top-k (or follow-the-clue) into a reserved slice of the budget, and learn keep / archive / delete / consolidate". That removes the rare-retrieve-action problem, and it matches what the strongest learned systems do.
- Strong non-learned baselines (LightMem, the EDU graph, BM25) are competitive. A thesis result that "learned controller ≈ heuristics on real text" is consistent with the field. The honest framing would be: learning helps the write/evict side under tight budgets, not the retrieval trigger.

### Gaps
- Exact LoCoMo/LongMemEval scores for LightMem and the EDU-graph baseline were not retrieved (only relative improvements for LightMem).
