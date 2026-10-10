# RL-trained and learned memory management for LLM agents (2025–2026): what's new and what to borrow

Context: a small (~27k-param MLP) separate controller for a frozen reader. Actions are keep / evict / archive / retrieve / compact / consolidate under a hard token budget. Trained with DAgger from a cost-sensitive regret expert (GRPO optional). Known failure: evidence gets archived but is never retrieved (lexical top-3). Already covered elsewhere, so mentioned only in passing: LRE 2606.20954, MemCon 2607.13591 (a UCB controller over memory ops with a frozen LLM; ALFWorld 59.7 -> 64.9), Memex(RL) 2603.04257, EMBER 2606.05894, restore-counterfactual audit 2609.08279, Parrot, ForesightKV.

Verification legend. **[V]** means I fetched the arXiv abstract page in this session. **[S]** means the claim comes only from a search-engine snippet. **[M]** means it comes from my prior knowledge (cutoff June 2026). For [M] items the arXiv ID is believed correct but the numbers were not re-checked this session, so verify them before citing in the thesis.

---

## Q1. The canonical RL memory agents (MEM1 ... MemOS) and 2026 successors: action spaces, rewards, algorithms, benchmarks, results

### Takeaway
Almost every RL memory agent from 2025 to 2026 trains **the LLM itself**, usually a Qwen 3B–14B, with a GRPO variant. The memory operations are text actions: overwrite a memory, CRUD, or fold/branch. The reward is outcome-only QA correctness or task success. The newer work (2026) mostly fixes credit assignment: tree- or branch-based dense rewards, and grouping memory rollouts separately from QA rollouts. It does not change the action space. The newest work also starts to ask *when consolidation beats retention as a function of budget*, and how to estimate memory utility for items that are never retrieved. Those two questions are exactly the thesis's open problems.

### Cited Findings

**2025 first wave: the LLM is the memory policy**
- **MEM1** (2506.15841) [M]. At each turn the agent rewrites one compact "internal state" and discards the previous context, so memory stays constant. Trained with PPO-style RL on multi-objective multi-hop QA and WebShop. The authors report Qwen2.5-7B MEM1 beating Qwen2.5-14B-Instruct on a 16-objective task, with about 3.5x better performance and 3.7x less memory. — [arXiv 2506.15841](https://arxiv.org/abs/2506.15841)
- **MemAgent** (2507.02259) [M]. Reads the document in chunks and keeps a fixed-length memory that the LLM *overwrites* after each chunk. Trained with "Multi-Conv DAPO", a GRPO/DAPO extension over independent conversations. Trained at 8K context, it extrapolates to about 3.5M tokens on RULER-HotpotQA with a small loss. — [arXiv 2507.02259](https://arxiv.org/abs/2507.02259)
- **Memory-R1** (2508.19828) [S+M]. Two RL-tuned agents: a Memory Manager with ADD / UPDATE / DELETE / NOOP, and an Answer Agent that "memory-distills" the retrieved entries. Trained with PPO and GRPO on outcome reward from very little data (about 152 QA pairs [M]); evaluated on LoCoMo. — [search snippet / arXiv 2508.19828](https://arxiv.org/pdf/2508.19828v3)
- **ReSum** (2509.13313) [S+M]. A ReAct web agent that periodically summarises its history into a reasoning state and restarts from it. "ReSum-GRPO" adapts GRPO to these segmented trajectories. Reported gains are about +4.5% over ReAct, and about +8% more after ReSum-GRPO [M]. — [arXiv 2509.13313](https://arxiv.org/abs/2509.13313)
- **MemGen** (2509.24704) [M]. A "memory trigger" (an RL-trained decider of *when* to invoke memory) plus a "memory weaver" that produces latent memory tokens and inserts them into the reasoning. Here the memory is latent, not text. — [arXiv 2509.24704](https://arxiv.org/abs/2509.24704)
- **Mem-α** (2509.25911) [S]. RL trains an agent to build a three-part memory (core / episodic / semantic) with insert, update and delete tools. The reward is downstream QA accuracy on the built memory, plus format and compression terms [M]. It was trained on about 30k-token sequences and generalises to more than 400k tokens, over 13x the training length. — [HF papers 2509.25911](https://huggingface.co/papers/2509.25911); [arXiv html](https://arxiv.org/html/2509.25911v1)
- **Context-Folding / FoldGRPO** (2510.11967) [S]. The agent *branches* into a sub-trajectory for a subtask, then *folds* it into a summary. FoldGRPO adds process rewards for decomposition and context management. It matches or beats ReAct with about 10x smaller active context. — [review page](https://liner.com/review/scaling-longhorizon-llm-agent-via-contextfolding); [emergentmind FoldGRPO](https://www.emergentmind.com/topics/foldgrpo)
- **AgentFold** (2510.24699) [M]. Alibaba/Tongyi web agent with "granular" (one step) and "deep" (multi-step) folding of history into multi-scale summaries. Trained with SFT, not RL. Reported about 36% on BrowseComp with a 30B-A3B model [M]. — [arXiv 2510.24699](https://arxiv.org/abs/2510.24699)
- **MemAct / Memory-as-Action** (2510.12635; ACL 2026 Findings) [S]. Context curation as in-place delete/insert edit actions. **DCPO** (Dynamic Context Policy Optimization) splits trajectories into segments at memory-edit points, so RL stays consistent when the prefix changes. MemAct-RL-14B matches models 16x larger and cuts average context length by 51%. The authors say the learned strategies "adapt to model capabilities". — [arXiv 2510.12635](https://arxiv.org/pdf/2510.12635); [ACL Anthology](https://preview.aclanthology.org/ingest-acl/2026.findings-acl.956/)
- **FoldAct** (2512.22733) [S]. Fixes instability and inefficiency of RL with context folding for search agents. — [arXiv 2512.22733](https://arxiv.org/html/2512.22733v1)

**Non-RL memory systems that are the usual baselines**
- **A-MEM** (2502.12110) [M]. Zettelkasten-style notes. The LLM links and "evolves" notes when it writes. — [arXiv 2502.12110](https://arxiv.org/abs/2502.12110)
- **Mem0** (2504.19413) [M]. An LLM extracts facts and then chooses ADD / UPDATE / DELETE / NOOP against similar memories; there is also a graph variant. On LoCoMo it reports about 26% relative gain over OpenAI memory (LLM-judge), and large latency and token savings versus full context. — [arXiv 2504.19413](https://arxiv.org/abs/2504.19413)
- **MemoryOS** (2506.06326) [M]. A three-tier OS-style hierarchy (short / mid / long-term persona). Mid-term pages are promoted or evicted by a "heat" score built from visit count, recency and length. This is a hand-tuned forgetting curve. — [arXiv 2506.06326](https://arxiv.org/abs/2506.06326)
- **MemOS** (2507.03724) [M]. A "MemCube" abstraction that unifies plaintext, activation (KV) and parametric memory, with lifecycle scheduling. — [arXiv 2507.03724](https://arxiv.org/abs/2507.03724)
- **LightMem** (2510.18866) [M]. Atkinson–Shiffrin-inspired: sensory pre-compression, topic-segmented short-term memory, and *offline "sleep-time"* consolidation of long-term memory. Reports accuracy gains on LongMemEval with order-of-magnitude token and API-call savings. — [arXiv 2510.18866](https://arxiv.org/abs/2510.18866)

**2026 successors (mostly verified this session)**
- **Mem-T** (2601.23014) [V]. Densifies rewards with **MoT-GRPO**: a memory-operation *tree* where sparse terminal reward is back-propagated through the tree, with "hindsight credit assignment". It jointly optimises memory construction and multi-turn retrieval over a hierarchical DB. Up to +14.92% over A-Mem and Mem0, and about 24.45% fewer inference tokens than GAM. Model size is not stated in the abstract. — [arXiv 2601.23014](https://arxiv.org/abs/2601.23014)
- **UMA, "Learning to Remember"** (2602.18493) [V]. A single policy does CRUD on a structured Memory Bank. The memory-state reward is the **mean reward of QA trajectories branching from each sampled memory state**. **Task-Stratified GRPO** normalises the memory group and the per-question QA groups separately. It adds a new **Ledger-QA** diagnostic for long-horizon state tracking over accumulated updates. Best average at a 16k budget. — [arXiv 2602.18493](https://arxiv.org/abs/2602.18493)
- **MemRL** (2601.03192) [V]. *Non-parametric* RL with a frozen LLM. Episodic memories carry learned **Q-values** (utility). Retrieval has two phases: filter by semantic similarity, then rank by Q-value. Evaluated on HLE, BigCodeBench, ALFWorld and Lifelong Agent Bench. Code is at MemTensor/MemRL. The abstract gives no specific numbers. — [arXiv 2601.03192](https://arxiv.org/abs/2601.03192)
- **MemChain** (2607.24097) [V+S]. Explicit memory actions build an *ordered evidence trace* (retrieved memories arranged by semantic role and dependency). Training is SFT on traces, then **TMPO** (Trace-Guided Memory Policy Optimization), an RL step rewarded by downstream answer quality. It reports state of the art on LoCoMo and LongMemEval-S while passing less context to the answer model. Per the search snippet, the **Qwen3-4B policy transfers across frozen Qwen3-1.7B / 8B / 14B readers**, plus closed-source readers. — [arXiv 2607.24097](https://arxiv.org/abs/2607.24097)
- **MemoPilot, "From Player to Master"** (2606.08656) [S]. RL over memory improves test-time learning of a *frozen* player. Elo 1762 (Limit Hold'em) and 1590 (RPS), ranked first. — [arXiv 2606.08656](https://arxiv.org/abs/2606.08656)
- **Adaptive memory structures** (2602.14038) [S]. Learns to pick a memory structure from interaction features, with *offline* supervision from downstream response quality, a three-level hierarchy, and a Beta-Mixture gate for fusion. — [arXiv 2602.14038](https://arxiv.org/html/2602.14038v1)
- **REALM** (2609.16053) [V]. Not RL. The memory graph is *reconsolidated using retrieval feedback*: items retrieved together are reorganised into local structures. LoCoMo 75.97% (+7.17) and LongMemEval 65.11% (+1.31). — [arXiv 2609.16053](https://arxiv.org/abs/2609.16053)
- **TraceRetain** (2606.29178) [V]. Bounded memory for a frozen agent. Entries are scored by interpretable features (success, age, access frequency, redundancy, specificity, similarity, downstream utility) and the lowest are evicted. A CEM-tuned variant keeps Precision@5 at 16.6% versus 3.8% for FIFO under 75% distractor writes, with 97/100 ALFWorld success. — [arXiv 2606.29178](https://arxiv.org/abs/2606.29178)
- **Hindsight** (2512.12818) [S]. A retain / recall / reflect memory architecture. It appears only in the benchmark-critique search, so details are not verified. — [arXiv 2512.12818](https://arxiv.org/pdf/2512.12818)

### Inferences, with "what to borrow"
- **Borrow: grouped/stratified advantages (UMA's Task-Stratified GRPO).** If the thesis's optional GRPO pools advantages across episodes of very different difficulty or budget, normalise *within* (budget, task-type) strata instead. This is a cheap fix for PPO/GRPO variance, the known binding constraint on scripted tasks.
- **Borrow: the branching value of a memory state (UMA).** Score a controller decision by the mean reader success over several *downstream queries* branched from the resulting memory state. This is a lower-variance target than one terminal outcome. It fits the regret expert as an extra DAgger label source.
- **Borrow: operation-tree hindsight credit (Mem-T MoT-GRPO).** The thesis already uses a hindsight regret expert, so the "tree" idea is mostly already covered. The new part is applying it to *retrieval* operations, not only writes.
- **Borrow: Q-value-reranked retrieval (MemRL).** This targets the "archived but not retrieved" failure directly. Keep lexical top-k as phase 1, then rerank candidates with a learned per-item utility, where the MLP itself can be the scorer. The cost is small, and the reader stays frozen.
- **Borrow: segment-at-edit-points for RL (MemAct DCPO).** If GRPO trajectories are ever rolled out with a changing context prefix, split them at memory edits. This is only relevant if the reader becomes trainable, which it is not here.
- **Borrow: retrieval-feedback reconsolidation (REALM).** When two archived items are retrieved together and help, merge them into one archive unit. This raises future recall under top-3 lexical retrieval without changing the retriever.

### Gaps
- I did not fetch or verify exact numbers for MEM1, MemAgent, ReSum, MemGen, AgentFold, A-MEM, Mem0, MemoryOS, MemOS or LightMem this session [M]. Their IDs are listed, but numbers should be checked before citing.
- The abstracts of Mem-T, UMA and MemChain do not give policy model sizes.
- No 2026 paper found here uses a *non-LLM* tiny controller with a GRPO-style algorithm. The closest are MemCon (UCB), TraceRetain (CEM over features) and OAS (Q2/Q3).

---

## Q2. Who trains a small separate controller vs. the LLM itself? Evidence on transfer across readers

### Takeaway
The field splits into three camps. (a) The LLM-as-memory-policy camp is dominant: MEM1, MemAgent, Mem-α, MemAct, Context-Folding, UMA, Mem-T. (b) A *separate LLM policy* camp feeds a frozen reader: Memory-R1's manager, and MemChain's 4B policy, which **transfers across 1.7B–14B frozen readers**. (c) A small group uses *tiny, non-LLM controllers* over a frozen LLM: MemCon UCB, TraceRetain, OAS, MemRL Q-values. The thesis sits in camp (c). In 2026 that camp is growing but is still mostly bandit, heuristic or feature-scored, with no imitation-trained MLP. That gap is the thesis's novelty claim.

### Cited Findings
- MemChain: a Qwen3-4B memory policy transfers across frozen Qwen3-1.7B, 8B and 14B answer models, and also helps closed-source readers. — [search snippet](https://arxiv.org/pdf/2607.24097); [arXiv abs](https://arxiv.org/abs/2607.24097)
- MemAct: the learned strategies "adapt to model capabilities". This is evidence that the *best* policy depends on the reader, so a policy tuned on one reader may be suboptimal on another. — [arXiv 2510.12635](https://arxiv.org/pdf/2510.12635)
- VISTA (2606.30005) [V]. A *training-free* context layer. Working memory becomes typed, addressable blocks, and archived payloads stay recoverable at full fidelity. A dashboard shows per-block token use, recency, archive status and remaining budget. The authors argue frontier LLMs have "proprioceptive blindness" about their own context budget. Results: Gemini-3-Flash goes from 22.7% to 50.7% on LOCA-Bench, and 58.0% on BrowseComp-Plus. The gains grow with context pressure. — [arXiv 2606.30005](https://arxiv.org/abs/2606.30005)
- Scale-conditioned evaluation (2605.07313) [V]. Whether a memory interface is reliable depends on the *backbone*. LiCoMemory with Qwen3-8B goes over the interaction budget, while Qwen3-32B and 235B stay within it. HippoRAG loses 16–20 points as irrelevant sessions are added. — [arXiv 2605.07313](https://arxiv.org/abs/2605.07313)
- MemRL, MemCon and TraceRetain all keep the LLM frozen and learn only the memory layer. — [2601.03192](https://arxiv.org/abs/2601.03192); [2607.13591](https://arxiv.org/html/2607.13591); [2606.29178](https://arxiv.org/abs/2606.29178)
- Memory-R1 trains a separate manager and answer agent, but both are LLMs. — [arXiv 2508.19828](https://arxiv.org/pdf/2508.19828v3)

### Inferences, with "what to borrow"
- **Borrow: a reader-transfer experiment in MemChain's style.** Train the controller on one reader, then evaluate it frozen on smaller and larger readers (for example Qwen2.5-1.5B/3B/14B). MemChain gives a citable precedent and a format for the table. The 2605.07313 result predicts the reverse effect for weak readers, so expect the hardest transfer to go *down* in reader size.
- **Borrow: VISTA-style state features.** Per-block token use, recency, access history and budget remaining are close to what a 27k MLP should see. VISTA's result shows these features alone help even a prompted frontier model. That makes "prompted LLM + VISTA dashboard" a strong baseline the thesis could add to its prompted-controller arm.
- **Positioning:** the thesis's controller is orders of magnitude smaller than MemChain's 4B policy. Showing transfer at ~27k params would be a stronger claim than MemChain's.

### Gaps
- Per-reader numbers for MemChain's transfer are not in the abstract. The full PDF was not read.
- I found no paper that transfers a *tiny non-LLM* memory controller across reader sizes. This looks like an open gap, but it is not proven absent.

---

## Q3. Cost-priced memory writes, forgetting curves, learned consolidation/summarisation

### Takeaway
The most directly relevant new result is **"Retain or Consolidate?" (2607.17545)**. It shows a budget-dependent *crossover*: consolidation (merge or abstract) wins by up to 48% absolute under tight budgets, and raw retention wins under loose budgets. A *lightweight offline learner* (OAS) picks the operator from pre-generation features using held-out harm calibration. The second is **Causal Memory Policy (2610.02070, October 2026)**. It shows memory utility is *unidentifiable* for memories that are never retrieved, and fixes this by reserving context slots for propensity-sampled memories. This is the formal version of the thesis's "archived but never retrieved" failure.

### Cited Findings
- **Retain or Consolidate?** (2607.17545) [V]. Consolidation operators are Merge, Abstract and Rewrite. "Consolidation improves absolute accuracy by up to 48% under tight budgets, whereas retention is preferable under loose budgets." LoCoMo shows the crossover at a smaller budget because its evidence is shorter. Cross-note abstraction and merging beat local rewriting. The OAS learner estimates action utilities from pre-generation features with held-out harm calibration. — [arXiv 2607.17545](https://arxiv.org/abs/2607.17545)
- **Causal Memory Policy (CMP)** (2610.02070) [V]. Utility cannot be estimated for memories that are never retrieved. Identification fails for 54% of required memories on LongMemEval and 67% on LoCoMo, and this persists in deployed systems. CMP reserves a fixed number of context slots for memories sampled with known propensities, then uses self-normalised IPW. The paper proves unbiasedness, gives the exact variance, and derives an optimal rule under *irreversible* operations. AUC for separating required from non-required memories rises from 0.54 to 0.66. Caveat: per-query utility reaches 0.78 AUC, yet "no aggregation available to a retention policy predicts a memory's value on unseen queries." — [arXiv 2610.02070](https://arxiv.org/abs/2610.02070)
- **TraceRetain** uses age and access-frequency features (a forgetting-curve proxy) plus downstream utility to choose evictions. — [arXiv 2606.29178](https://arxiv.org/abs/2606.29178)
- **MemoryOS** uses a "heat" score for promotion and eviction [M]. — [arXiv 2506.06326](https://arxiv.org/abs/2506.06326)
- **LightMem** does offline "sleep-time" consolidation, separating consolidation cost from online latency [M]. — [arXiv 2510.18866](https://arxiv.org/abs/2510.18866)
- **Eviction-policy study for LLM semantic caches** (2608.20280) [V]. Over 18 settings, LFU is never beaten by more than 0.041 points. FIFO trails by up to 8.67 points at tight capacity. Only 2–4% of semantic hits are answer-substitutable. No learned policies were tested. — [arXiv 2608.20280](https://arxiv.org/abs/2608.20280)
- **"Agentic Context Management: Solving Agent Memory and Cost"** (2607.21503) [S]. Treats memory plus cost as a lifecycle and architecture problem. Not fetched. — [arXiv 2607.21503](https://arxiv.org/pdf/2607.21503)
- **Mem-α** includes a compression term in its reward [M]. **MemAct** jointly optimises retention and task performance, cutting context by 51%. — [2509.25911](https://arxiv.org/html/2509.25911v1); [2510.12635](https://arxiv.org/pdf/2510.12635)

### Inferences, with "what to borrow"
- **Borrow (high priority): CMP's exploration slots.** Reserve 1 of the reader's retrieval slots, alongside the lexical top-3, for an archived item sampled with a known propensity, and log the outcome. This gives unbiased utility estimates for archived items that lexical retrieval never surfaces. That is exactly the data the regret expert and DAgger need to learn *what to retrieve*. CMP's finding that utility does not aggregate across queries is a warning: features must be query-conditioned.
- **Borrow: budget-conditioned operator choice (2607.17545).** The thesis already has compact and consolidate. Check whether the learned policy reproduces the crossover (consolidate at 2% budget, retain at 25%). If it does, that is external validation. If not, add budget fraction as an explicit input feature and/or add OAS-style harm calibration (a held-out estimate of when consolidation hurts) to the expert's cost.
- **Borrow: an LFU / access-count feature.** The cache study suggests access frequency is a hard-to-beat eviction signal. Make sure the MLP sees per-item access counts, and report LFU as a heuristic baseline.
- **Borrow: sleep-time consolidation (LightMem).** Run consolidation offline, between episodes or during idle steps, so its token cost does not count against the in-episode budget. Report both accountings.

### Gaps
- No paper found that prices *archive storage* explicitly, as a per-token storage cost separate from context cost, inside an RL reward. The thesis's cost-sensitive regret expert may be novel here, but this is unconfirmed.
- No learned forgetting-curve (decay-rate) model for agent text memory was found in this search. TRIM-KV-style learned decaying retention gates exist at the KV-cache level (ICLR 2026 snippet), but the paper's ID and title were not verified. — [ICLR 2026 proceedings PDF](https://proceedings.iclr.cc/paper_files/paper/2026/file/c3887741ee4e655494770be95303383f-Paper-Conference.pdf)

---

## Q4. Benchmarks for long-horizon agent memory in 2026 and their weaknesses

### Takeaway
LoCoMo and LongMemEval remain the standard but are increasingly criticised. LoCoMo is short (about 26k tokens), has only 10 conversations, and has answer-key errors. Both benchmarks mix memory with reasoning. Below some scale, full context beats retrieval. Newer evaluations instead test scale (scale-conditioned protocols, BEAM), lifecycle operations (MemOps), risk (MemRiskBench), and state tracking (Ledger-QA). The thesis's own finding, that full context beats a 2%-budget policy only modestly, matches the "full context wins at small scale" critique.

### Cited Findings
- **LoCoMo** (2402.17753) [M for ID]. About 26k tokens per conversation, so it fits in modern context windows. Only 10 public conversations. An independent audit reports a 6.4% answer-key error rate (99 of 1,540 errors that corrupt scores), and an LLM judge accepted 62.81% of deliberately wrong answers. Knowledge updates are not scored explicitly. — [Mem0 benchmark guide](https://mem0.ai/blog/ai-memory-benchmarks-in-2026); [Cognee guide](https://www.cognee.ai/ai-memory-benchmarks). These are vendor blogs, so the audit numbers should be traced to the original audit, which I did not locate.
- **LongMemEval** (2410.10813) [M for ID]. 500 questions. Synthetic conversations with limited topical diversity. — [Mem0 guide](https://mem0.ai/blog/ai-memory-benchmarks-in-2026)
- Both benchmarks mix memory evaluation with reasoning and synthesis. High LoCoMo scores can coexist with low task completion when memory has to support actions. — [Cognee guide](https://www.cognee.ai/ai-memory-benchmarks) [vendor source]
- "Below ~150 conversations, full-context answering beats retrieval memory." This is a "tenure crossover" in architecture rankings. — [Ground Truth First, arXiv 2607.21962](https://arxiv.org/pdf/2607.21962) [S]
- **MemoryAgentBench** (2507.05257). Incremental multi-turn evaluation of four competencies: accurate retrieval, test-time learning, long-range understanding, and conflict resolution [M for the list]. — [arXiv 2507.05257](https://arxiv.org/pdf/2507.05257)
- **Scale-conditioned evaluation** (2605.07313) [V]. Holds evidence fixed while adding irrelevant sessions, and reports budget-compliant reliability, memory-call burden, failure regimes, and a "usable-scale boundary". — [arXiv 2605.07313](https://arxiv.org/abs/2605.07313)
- **MemOps** (2607.12893) [S]. Benchmarks lifecycle memory operations in long-horizon conversations. — [arXiv 2607.12893](https://arxiv.org/pdf/2607.12893)
- **MemRiskBench** (2609.14976) [S]. Trace-aware evaluation of whether memory preserves risk-relevant information. — [arXiv 2609.14976](https://arxiv.org/pdf/2609.14976)
- **Ledger-QA** (in UMA, 2602.18493) [V]. State tracking over accumulated updates. — [arXiv 2602.18493](https://arxiv.org/abs/2602.18493)
- **LOCA-Bench, BrowseComp-Plus, GAIA** are used for context management in agentic tool use (VISTA). — [arXiv 2606.30005](https://arxiv.org/abs/2606.30005)
- Agentic RL memory papers mainly use multi-hop QA (HotpotQA-style, RULER-HotpotQA), WebShop, ALFWorld and BrowseComp. — MemAgent, MEM1, AgentFold [M]; TraceRetain and MemCon on ALFWorld [V]

### Inferences, with "what to borrow"
- **Borrow: a scale-conditioned protocol.** Hold the evidence fixed and add distractor turns to plot the thesis's success vs. budget curve against horizon length. This directly answers the "full context wins at short horizons" critique. Each environment's "usable-scale boundary" makes a clean headline.
- **Borrow: report CMP's identification-failure metric** (the share of required items never retrieved) as a diagnostic for the archive-retrieval failure.
- If the thesis adds an external benchmark, LongMemEval-S is the most defensible choice, and MemChain and CMP give directly comparable numbers. Mention LoCoMo's answer-key problems if LoCoMo is used.

### Gaps
- I could not find the original LoCoMo answer-key audit paper. The 6.4% and 62.81% figures come from vendor blogs.
- BEAM (named in the Mem0 guide title) was not investigated.
- "Ground Truth First" (2607.21962) is cited from a search snippet only.
