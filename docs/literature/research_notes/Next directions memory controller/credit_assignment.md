# Step/turn-level credit assignment for long-horizon agent RL (2024–2026), applied to a per-decision memory controller

Scope note: sources were checked on 2026-10-06 via arXiv abstract/HTML pages and search snippets. Items marked **[unverified]** were not re-checked in this session (recalled IDs or claims); treat them as leads, not citations.

Project framing used for relevance judgements: ~27k-param MLP controller, ~200 decisions/episode (keep / evict / archive / retrieve), hard token budget, sparse reward = fraction of ~20 questions answered, frozen LLM reader (Qwen2.5-7B), deterministic resettable env per seed, hindsight oracle (Belady furthest-next-use) available, current GRPO uses one episode-level advantage for all ~200 steps, GRPO on top of imitation gave no gain, residual failure = "archived but not retrieved".

## Q1. What do the main step/turn-level credit-assignment methods do, what is the evidence, what is the cost?

### Takeaway
Methods cluster into four families: (a) critic-free group baselines at the step level (GiGPO, turn-level GRPO, TreeRPO, Tree-GRPO, SPO), (b) Monte-Carlo values from branching off shared prefixes (VinePPO, SPO, AgentPRM targets), (c) learned critics / PRMs, including privileged "hindsight" critics (ArCHer, SWEET-RL, AgentPRM, IGPO's intrinsic reward), and (d) classic return-redistribution / hindsight methods (RUDDER, HCA, Shapley/SCAR). Reported gains over trajectory-level GRPO are consistent (roughly +3 to +12 points on agent benchmarks, larger on some QA), and the cheapest ones (GiGPO) add almost no compute; MC-branching methods buy accuracy with extra rollouts but often win on wall-clock anyway.

### Cited Findings

**GiGPO (Group-in-Group Policy Optimization)** — Feng, Xue, Liu, An; NeurIPS 2025; arXiv:2505.10978; code in verl-agent.
- Two-level advantage: episode-level group advantage A_E (as GRPO) plus a step-level advantage A_S computed by "anchor state grouping": after rollouts, identical environment states that recur across trajectories in the group are collected, and the actions taken from each anchor state are compared using their discounted return-to-go Σ_{k≥t} γ^{k−t} r_k, normalised within the anchor group. Final advantage A = A_E + ω·A_S, with ω = 1 untuned in most experiments — [arXiv HTML](https://arxiv.org/html/2505.10978)
- When states never repeat, it "naturally degrades to GRPO"; for complex observation spaces they fall back to similarity grouping (longest matching subsequence, threshold 0.9) — [arXiv HTML](https://arxiv.org/html/2505.10978)
- In their tasks, singleton step groups were <35% throughout training (i.e. >65% of states recur); large groups (≥10) fell from ~20% early to ~3.1% by iteration 140 as the policy sharpened — [arXiv HTML](https://arxiv.org/html/2505.10978)
- Cost: "same GPU memory overhead, identical LLM rollout, and … little to no additional time cost"; no critic, no extra rollouts. Gains vs GRPO: >12% ALFWorld, >9% WebShop; search-QA 42.1% (3B) / 47.2% (7B) — [arXiv abs](https://arxiv.org/abs/2505.10978)

**Turn-level credit assignment for multi-turn tool use (MT-GRPO)** — Zeng, Wei, Brown, Frunza, Nevmyvaka, Hong; arXiv:2505.11821 (v2 retitled "…via Turn-Level Reward Design").
- Replaces trajectory-level ("bandit") advantage with turn-level advantages that combine turn-level rewards (e.g. tool executed correctly) with outcome reward; pluggable into GRPO/PPO. Reported 100% tool-execution success and 50% exact-match vs 20–30% for baselines that fail to call tools — [HF papers](https://huggingface.co/papers/2505.11821); code [GitHub](https://github.com/SiliangZeng/Multi-Turn-RL-Agent)

**Tree-GRPO ("Tree Search for LLM Agent Reinforcement Learning")** — arXiv:2509.21240; ICLR 2026 poster.
- Tree nodes are whole agent steps (thought, action, observation), not tokens. Sharing prefixes yields more rollouts per token/tool-call budget; tree structure gives step-wise process signals from outcome reward alone; advantages are group-relative both intra-tree and inter-tree. Reported +16% to +69% relative on multi-hop QA and better results at one-quarter the rollout budget vs chain-based RL — [arXiv](https://arxiv.org/abs/2509.21240), [ICLR 2026](https://iclr.cc/virtual/2026/poster/10008772)

**TreeRPO** — arXiv:2506.05183.
- Tree sampling to estimate expected reward at intermediate reasoning steps; GRPO-style normalisation within step-level sibling groups. Qwen2.5-Math Pass@1 19.0% → 35.5%; +2.9% over GRPO with 18.1% shorter responses — [arXiv](https://arxiv.org/abs/2506.05183)

**TreeRL / EPTree** — ACL 2025; arXiv:2506.11902.
- On-policy tree search where branches are forked at the highest-entropy intermediate tokens; ~2 iterations to build trees; intermediate supervision from the tree without a separate PRM (avoids PRM distribution shift / hacking). Beats i.i.d. multi-chain sampling and MCTS at equal inference budget on math/code — [arXiv](https://arxiv.org/abs/2506.11902), [ACL Anthology](https://aclanthology.org/2025.acl-long.604)

**ARPO (Agentic Reinforced Policy Optimization)** — arXiv:2507.19849; ICLR 2026.
- Observed token-entropy spikes right after tool responses; adaptively branches extra partial rollouts at those high-entropy steps (mixing global trajectory sampling with step-level sampling) and uses "advantage attribution estimation" so shared-prefix vs branched tokens get appropriate advantages. Better than trajectory-level RL on 13 benchmarks with ~half the tool-call budget — [arXiv](https://arxiv.org/abs/2507.19849), [ICLR 2026](https://www.iclr.cc/virtual/2026/poster/10009306)

**SPO (Segment Policy Optimization)** — NeurIPS 2025; arXiv:2505.23564.
- Middle granularity: partitions the trajectory into segments (cutpoint-based in SPO-chain; tree-based variant for long CoT), estimates each segment's advantage by Monte-Carlo from the policy (no critic), and uses a probability-mask so only key tokens get the segment advantage. +6–12 points over PPO/GRPO on GSM8K (short CoT) — [arXiv HTML](https://arxiv.org/html/2505.23564v2), [NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/a6536243037d1e32c20de85137d478da-Abstract-Conference.html), [code](https://github.com/hgjungg/SPO)

**VinePPO** — arXiv:2410.01679.
- Replaces PPO's value network with unbiased MC value estimates obtained by resetting to intermediate states (re-feeding the prefix) and sampling continuations. Finds learned value nets rank alternative steps barely better than random on reasoning tasks. Slower per iteration but reaches baselines' peak with fewer gradient steps and up to 3.0× less wall-clock on MATH/GSM8K — [arXiv](https://arxiv.org/abs/2410.01679), [HTML v2](https://arxiv.org/html/2410.01679v2)

**ArCHer** — arXiv:2402.19446.
- Hierarchical actor-critic: off-policy, utterance-level (turn-level) critic with TD learning at the high level; token-level policy gradient at the low level using the turn-level value. Reports ~100× sample-efficiency over prior on-policy methods on agent tasks — [arXiv](https://arxiv.org/abs/2402.19446)

**SWEET-RL** — Zhou, Jiang, Tian, Weston, Levine, Sukhbaatar, Li (Meta FAIR / Berkeley); arXiv:2503.15478.
- Trains a turn-level critic that sees *training-time-only (privileged) information* (e.g. the reference solution) that the actor never sees; critic outputs step-level rewards for the policy. +6% absolute success/win-rate on ColBench vs other multi-turn RL; Llama-3.1-8B matches/exceeds GPT-4o there — [arXiv](https://arxiv.org/abs/2503.15478)

**Process reward models for agents**
- AgentPRM (Choudhury, Cornell; arXiv:2502.10325): lightweight actor-critic where PRM targets are computed by Monte-Carlo rollouts; InversePRM learns process rewards from demonstrations without outcome labels; 3B models beat GPT-4o baselines on ALFWorld; paper explicitly analyses reward hacking — [arXiv](https://www.arxiv.org/abs/2502.10325)
- A different paper with the same name, "AgentPRM: Process Reward Models for LLM Agents via Step-Wise Promise and Progress" (arXiv:2511.08325) — naming collision; not reviewed in detail — [arXiv](https://arxiv.org/pdf/2511.08325)
- IGPO (ICLR 2026; arXiv:2510.14967): turn reward = marginal increase in the policy's own probability of producing the ground-truth answer after that turn (information gain); combined with outcome reward; avoids external PRM and MC cost; improves accuracy and sample efficiency on search-agent benchmarks — [HF papers](https://huggingface.co/papers/2510.14967), [ICLR 2026](https://iclr.cc/virtual/2026/poster/10007215)

**Shapley / counterfactual credit**
- SCAR (arXiv:2505.20417): distributes the sequence-level reward over tokens/spans via Shapley values (marginal contribution of each unit across coalitions), used as dense reward at each unit's completion; faster convergence and higher final reward than standard RLHF and attention-based dense-reward baselines (sentiment, summarisation, instruction tuning) — [arXiv](https://arxiv.org/abs/2505.20417v1)

**Classic return redistribution / hindsight**
- RUDDER (arXiv:1806.07857; NeurIPS 2019): learns a return predictor (LSTM) and redistributes the delayed return to steps via contribution analysis, producing a "return-equivalent" MDP with near-immediate rewards; strong gains on long-delay Atari games (Bowling, Venture, etc.) on top of PPO — [arXiv](https://arxiv.org/abs/1806.07857)
- Hindsight Credit Assignment, HCA (Harutyunyan et al., NeurIPS 2019; arXiv:1912.02503): rewrites value functions in terms of the hindsight probability that an action was taken given a later state or return, h(a | s, outcome)/π(a | s); credit goes to actions that made the observed outcome more likely — [arXiv](https://arxiv.org/abs/1912.02503), [NeurIPS](https://proceedings.neurips.cc/paper/2019/hash/195f15384c2a79cedf293e4a847ce85c-Abstract.html)

### Inferences
- Rough cost ladder (cheapest first): GiGPO anchor grouping (free, needs state recurrence) < turn-level reward design / IGPO-style intrinsic reward (cheap, needs a scorable proxy) < learned privileged critic (SWEET-RL; one extra small model) < tree/segment branching (Tree-GRPO, TreeRPO, SPO, ARPO; extra partial rollouts but shared prefix) < full VinePPO MC per step (K rollouts per evaluated state).
- For this project, the cost driver is LLM-reader calls, not controller compute. The controller itself is tiny, so any learned critic is essentially free; the binding cost is how many extra episodes/questions the 7B reader must answer.
- A common thread across 2025 papers: the gains come less from a cleverer optimiser and more from *giving different steps different advantages via a same-state comparison*. That is exactly what the current setup lacks (one advantage for 200 steps).

### Gaps
- No head-to-head comparison found across GiGPO, Tree-GRPO, ARPO, and VinePPO on a common benchmark with matched compute.
- None of the cited papers has horizons near ~200 decisions with a tiny non-LLM policy; most agent tasks have 5–50 turns. Transfer to 200-step horizons is untested.
- The [unverified] PRIME implicit PRM (arXiv:2502.01456) and EMPG entropy-modulated policy gradients (arXiv:2509.09265) are relevant leads but were not checked this session.

## Q2. Methods that branch rollouts from a shared prefix (same state, different action) — how applicable to a deterministic, resettable environment?

### Takeaway
Branching from a shared prefix is the most directly applicable family: VinePPO's key move (reset to an intermediate state and sample continuations) is trivial in a deterministic per-seed simulator, and Tree-GRPO / TreeRPO / ARPO show that step-level groups of siblings give dense, critic-free advantages from outcome reward alone. GiGPO gets some of this for free, but only where states recur, which will be rare after the first divergent memory decision.

### Cited Findings
- VinePPO relies on being able to "reset directly to any intermediate state simply by re-feeding the partial context" and uses MC rollouts from there for unbiased values — [arXiv](https://arxiv.org/abs/2410.01679)
- Tree-GRPO: prefix sharing increases rollouts per budget and creates intra-tree sibling groups for step-level relative advantages; 1/4 budget matched chain-based methods — [arXiv](https://arxiv.org/abs/2509.21240)
- TreeRPO: step-level groups from tree sampling, GRPO-style normalisation per group — [arXiv](https://arxiv.org/abs/2506.05183)
- TreeRL/EPTree and ARPO both choose *where* to branch by uncertainty (token entropy) instead of uniformly, saving budget — [TreeRL](https://arxiv.org/abs/2506.11902), [ARPO](https://arxiv.org/abs/2507.19849)
- SPO: fewer estimation points (segments rather than tokens) make MC estimation affordable and unbiased — [arXiv HTML](https://arxiv.org/html/2505.23564v2)
- GiGPO degrades to GRPO when states don't repeat; its own recurrence statistic (>65% non-singleton groups) comes from ALFWorld/WebShop-style environments with small discrete state spaces — [arXiv HTML](https://arxiv.org/html/2505.10978)

### Inferences (application to the memory controller)
- **Why GiGPO alone will help little here:** 8 rollouts of the same seed share an identical state only until the first differing memory action; after that, memory contents differ and exact anchor matches vanish. Anchor grouping would mainly credit the first few divergent decisions. A similarity key (e.g. same step index + same Jaccard-similar memory set, or same "item under decision + same question-phase") could restore groups, but that is our own heuristic, not something the paper tested.
- **Recommended branching design ("vine" over decisions):** pick a decision step t in a reference rollout; reset the simulator to the exact state s_t (deterministic seed); for each candidate action a ∈ {keep, evict, archive, retrieve-X} take a, then continue with the current policy (or a fixed policy, e.g. the imitation policy / oracle) to a *truncated horizon* — until the affected item is next needed or the next k questions are answered — rather than to episode end. Advantage A(s_t, a) = R̂(a) − mean_a' R̂(a'). This is TreeRPO/Tree-GRPO sibling-group advantage with VinePPO-style resets.
- **Budget control:** branch only at "contested" decisions, chosen by (i) policy entropy (EPTree/ARPO analogue), (ii) disagreement with the Belady oracle, or (iii) decisions that touch an item which a later question needs. Truncating horizons to the next relevant question reduces reader calls from ~20 questions per branch to typically 1–3.
- **Cheaper still:** because questions are graded by a frozen reader, cache reader answers keyed on (question, exact context). Many branches will produce identical contexts at question time and need no new reader call.
- **For the "archived but not retrieved" failure specifically:** branch at each step preceding a question whose evidence is in the archive: compare "retrieve the needed item now" vs the policy's action. This yields a direct, low-variance advantage for retrieve decisions, which episode-level GRPO dilutes over ~200 steps.

### Gaps
- No published work found applying shared-prefix branching to a non-LLM controller managing an LLM's context; this combination appears to be open.
- Bias from continuing with a non-current policy after the branch (off-policy continuation) is not analysed in the cited papers; VinePPO uses the current policy.

## Q3. How do agent RL frameworks (verl-agent, RAGEN/StarPO, SkyRL, Agent-R1, AgentGym-RL) handle long horizons and multi-turn credit?

### Takeaway
Most frameworks still default to trajectory-level advantages; long-horizon handling is mostly about *context management and training stability* (per-step context rather than full history, trajectory filtering, horizon curricula). verl-agent is the exception, shipping GiGPO step-level credit. Memory-management RL papers (Memory-R1) use plain outcome-driven PPO/GRPO.

### Cited Findings
- **verl-agent / GiGPO:** "step-wise multi-turn interaction paradigm that avoids concatenating full histories", i.e. per-step input with a customisable memory, to control context growth over long horizons; GiGPO provides step-level credit — [GiGPO arXiv HTML](https://arxiv.org/html/2505.10978)
- **RAGEN / StarPO** (arXiv:2504.20073): StarPO is a *trajectory-level* agent RL framework. Identifies the "Echo Trap" (reward-variance cliff + gradient spikes, collapse into repetitive behaviour); StarPO-S stabilises it with trajectory filtering (keep high-variance prompts), adding a critic, and gradient stabilisation. Also finds that without fine-grained reasoning-aware reward signals, reasoning hardly emerges in multi-turn RL — [arXiv](https://arxiv.org/abs/2504.20073), [code](https://github.com/RAGEN-AI/RAGEN)
- **AgentGym-RL** (arXiv:2509.08755; ICLR 2026): modular framework; ScalingInter-RL restricts the interaction horizon early (exploitation) and gradually lengthens it (exploration), reducing collapse under long horizons; matches/surpasses commercial models on 27 tasks — [arXiv](https://arxiv.org/abs/2509.08755), [ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/1571ce1ff735be4551db50887043726e-Abstract-Conference.html)
- **SkyRL-Agent** (arXiv:2511.16108): focus on efficient asynchronous dispatching and backend interoperability (SkyRL-train, verl, Tinker); SA-SWE-32B 24.4% → 39.4% Pass@1 on SWE-Bench Verified with >2× cost reduction. Emphasis is systems efficiency, not a new credit-assignment method — [arXiv](https://arxiv.org/abs/2511.16108)
- **Agent-R1** (arXiv:2511.14460): modular framework; formalises LLM agents as an extended MDP; validated on multi-hop QA — [arXiv](https://arxiv.org/abs/2511.14460v1)
- **Memory-R1** (arXiv:2508.19828; ACL 2026): Memory Manager (ADD/UPDATE/DELETE/NOOP) + Answer Agent, both trained with outcome-driven PPO and GRPO; with only 152 QA training pairs beats baselines across LoCoMo, MSC, LongMemEval, 3B–14B — [arXiv](https://arxiv.org/abs/2508.19828v4)
- [unverified] MEM1 (arXiv:2506.15841) trains an agent to consolidate an internal state under constant memory with outcome RL — recalled, not re-checked.

### Inferences
- The closest prior work on learned memory operations (Memory-R1) succeeds with outcome-only RL, but its episodes have far fewer memory operations per reward than our ~200 decisions per ~20 questions; credit dilution grows with decisions-per-reward.
- AgentGym-RL's horizon curriculum maps naturally to this project: train first on short episodes (few items / few questions), then lengthen. It is cheap to try and attacks the same dilution problem from another angle.
- RAGEN's trajectory filtering (drop low-variance groups) is directly relevant: groups of 8 same-seed rollouts that all score the same give zero advantage and only add noise; filtering them is standard practice (also DAPO-style dynamic sampling).

### Gaps
- Could not confirm whether RAGEN, SkyRL or Agent-R1 ship step-level advantage estimators beyond optional critics (would require reading repos).

## Q4. Reward shaping from dense hindsight signals vs purely outcome reward — benefits and reward-hacking risks

### Takeaway
Dense signals reliably speed learning when they come from a *same-state comparison* (MC branching) or a *privileged critic trained toward the true outcome* (SWEET-RL, AgentPRM); hand-written or learned process rewards that are *summed* into returns are prone to hacking. Mitigations with evidence: min-form credit (PURE), keeping a verifiable outcome term, and potential-based / return-equivalent redistribution (RUDDER) that leaves optimal policies unchanged.

### Cited Findings
- PURE (NeurIPS 2025; arXiv:2504.15275): PRM reward hacking in RL fine-tuning is mainly caused by summation-form credit (value = discounted sum of step rewards), which lets the policy farm high-reward steps; replacing with min-form (value = minimum of future step rewards) largely removes hacking and reaches verifiable-reward performance in ~30% of steps; adding 10% verifiable rewards further helps — [arXiv](https://arxiv.org/abs/2504.15275)
- AgentPRM paper analyses reward hacking of MC-trained process rewards in agent settings — [arXiv](https://www.arxiv.org/abs/2502.10325)
- TreeRL motivates tree-derived supervision partly because separately trained PRMs suffer distribution mismatch and reward hacking — [arXiv](https://arxiv.org/abs/2506.11902)
- SCAR: earlier dense-reward methods that score partial sequences are inaccurate because partials are out-of-distribution for the reward model; Shapley redistribution keeps the total equal to the sequence reward — [arXiv](https://arxiv.org/abs/2505.20417v1)
- RUDDER: redistributed rewards form a return-equivalent MDP, so the optimal policy is unchanged while delays shrink — [arXiv](https://arxiv.org/abs/1806.07857)
- SWEET-RL: asymmetric (privileged) critic gives step-level rewards trained against the real objective; +6% over other multi-turn RL — [arXiv](https://arxiv.org/abs/2503.15478)
- RAGEN: without fine-grained reward signals, desired behaviour (reasoning) hardly emerges under outcome-only multi-turn RL — [arXiv](https://arxiv.org/abs/2504.20073)
- IGPO combines intrinsic turn rewards with outcome supervision rather than replacing it — [HF papers](https://huggingface.co/papers/2510.14967)

### Inferences (concrete options for per-decision advantages in this project)
1. **Hindsight "needed-again" label as a shaping term (cheapest, highest hacking risk).** For each evict/archive of item i at step t, penalty if i is needed by a question before it is retrieved; for each retrieve, bonus if the retrieved item is used by a question before it is evicted. Risk: the policy can over-retrieve or thrash; the budget prevents "keep everything" but not churn. Mitigate with small weight, keep the outcome term, and/or use the PURE min-form idea over decisions.
2. **Oracle-advantage (HCA/SWEET-RL flavour).** Use the Belady oracle as a privileged critic: A(s_t,a) = Q_oracle-continuation(s_t,a) − V. Concretely: from s_t take a, then run the oracle to the truncated horizon; compare against the policy's action. This credits the decision under the best follow-up rather than the current policy (biased toward "decisions that are good if later play is good"), but is deterministic and needs no learned critic. Must be noted as biased vs on-policy MC.
3. **Return-equivalent redistribution (RUDDER / SCAR flavour).** Each question's 0/1 reward is credited only to decisions that touched that question's evidence items within its window (an "item-causal" redistribution). Sum of redistributed rewards = episode reward, so the objective is unchanged; variance falls because ~200 irrelevant decisions get no credit for a given question. This is the most natural fit given the per-item hindsight data and needs zero extra reader calls.
4. **MC branching (Q2) for the contested decisions** — unbiased, most expensive; use to audit options 1–3.
5. Keep a GRPO episode term (GiGPO's A_E + ω·A_S form) so step terms refine rather than replace the outcome signal.
- The "archived but not retrieved" failure is a missed *retrieve* action at the right moment: under episode-level advantage, a successful retrieve shares credit with ~199 other decisions. Options 2–4 give that single decision its own signal.

### Gaps
- Potential-based shaping theory (Ng, Harada & Russell, ICML 1999) guarantees policy invariance for shaping of the form γΦ(s')−Φ(s); not re-fetched this session **[unverified citation]**. Our hindsight label (option 1) is *not* potential-based, so it can change the optimal policy.
- No source found that measures reward hacking specifically for memory/context-management agents.
- No empirical evidence found on whether oracle-continuation advantages (option 2) outperform on-policy MC for policy improvement; this is a design hypothesis for the thesis to test.
