# GRPO-family and critic-free policy-gradient variants (2024–2026): what each fixes, and what applies to a small learned memory controller

Setting assumed throughout (from the thesis): ~27k-parameter MLP controller over a frozen LLM agent; per-step discrete actions (keep / evict / archive / retrieve) for ~200 steps; sparse episode reward = fraction of ~20 questions answered (so values come in steps of 0.05) minus small costs; GRPO with G=8 rollouts per episode seed, group-standardised advantage, PPO clip, no value net; strong DAgger imitation warm start from a cost-sensitive regret expert; GRPO fine-tuning of that start gave no gain (-0.004 at 2% budget); remaining gap to the hindsight oracle (~0.10 at 2%) is mostly "archived but never retrieved"; LLM-in-the-loop rollouts cost ~20 s/episode.

Mapping note: LLM papers talk about "tokens" and "responses". For us a token is one controller action at one step, and a response is one 200-step episode. Fixes that work on "token-level vs sequence-level" map onto "step-level vs episode-level".

Verification note: papers marked [fetched] had their arXiv abstract or HTML fetched, or were confirmed through search-result snippets, in this session. Papers marked [from memory] are well-known, and their IDs match my training knowledge, but I did not re-open them in this session. See the Gaps sections.

---

## Q1. What does each main variant change, and what evidence supports it?

### Takeaway
Most 2025 "GRPO fixes" correct one of four things: (a) normalisation biases in the advantage or loss (Dr. GRPO, Lite PPO, REINFORCE++, RLOO); (b) importance-ratio and clipping instability (DAPO clip-higher, GSPO, GMPO, M2PO); (c) entropy collapse (DAPO clip-higher, Clip-Cov/KL-Cov); (d) wasted compute on groups with no signal (DAPO dynamic sampling, prompt replay, RL-ZVP). Almost all the evidence comes from billion-parameter LLMs doing maths or code, where responses vary in length and ratios are taken per token. Only a few of these problems carry over to a 27k-parameter MLP with fixed-length, 200-step episodes.

### Cited Findings

**Baseline: GRPO (DeepSeekMath, 2024)** [from memory]
- GRPO drops PPO's value network. Its baseline is the mean reward of G samples for the same prompt, and its advantage is divided by the group std. It adds a KL penalty to a reference model directly in the loss, not in the reward. — [DeepSeekMath, arXiv:2402.03300](https://arxiv.org/abs/2402.03300)
- DeepSeek-R1-Zero used pure GRPO with rule-based rewards and no SFT. — [DeepSeek-R1, arXiv:2501.12948](https://arxiv.org/abs/2501.12948)

**Dr. GRPO ("Understanding R1-Zero-Like Training", 2025)** [fetched]
- It identifies "an optimization bias in GRPO, which artificially increases response length (especially for incorrect outputs)". — [arXiv:2503.20783](https://arxiv.org/abs/2503.20783)
- Response-level length bias comes from the 1/|o_i| term. For positive advantages it gives "greater gradient updates for shorter responses". For negative ones, "longer responses are penalized less". — [arXiv:2503.20783 (HTML)](https://arxiv.org/html/2503.20783)
- Question-level difficulty bias comes from dividing by std: "Questions with lower standard deviations (e.g., those that are too easy or too hard, with the outcome rewards being almost all 1 or 0) are given higher weights during policy updates." — [arXiv:2503.20783 (HTML)](https://arxiv.org/html/2503.20783)
- The fix removes both the 1/|o_i| term and the std term. This recovers PPO with an unbiased Monte Carlo baseline. Reported result: 43.3% AIME 2024 with a 7B base model. — [arXiv:2503.20783](https://arxiv.org/abs/2503.20783)

**DAPO (ByteDance Seed / Tsinghua, 2025)** [from memory]
- It adds four techniques to GRPO:
  - Clip-Higher: separate clip bounds, ε_low=0.2 and a larger ε_high≈0.28, so low-probability actions can grow. This targets entropy collapse.
  - Dynamic Sampling: over-sample, then drop groups where accuracy is all 0 or all 1, so every batch has non-zero advantages.
  - Token-level policy-gradient loss: average over all tokens in the batch, not per sample then per batch. This removes the long-sequence under-weighting.
  - Overlong reward shaping: soft penalty for truncated responses.
- Reported: 50 points on AIME 2024 with Qwen2.5-32B in about half the steps of DeepSeek-R1-Zero-Qwen-32B. The KL term is dropped. — [arXiv:2503.14476](https://arxiv.org/abs/2503.14476); code at github.com/BytedTsinghua-SIA/DAPO

**Lite PPO ("Tricks or Traps?", ICLR 2026)** [fetched via search snippets]
- A systematic ablation of RL tricks across difficulty levels, model sizes and alignment status. Finding: two tricks are enough, (i) group-mean baseline with batch-level std normalisation and (ii) token-level loss aggregation. This "Lite PPO" beats GRPO and DAPO in their setting. The authors stress that each trick's effectiveness depends on model scale, alignment status and data difficulty. — [arXiv:2508.08221](https://arxiv.org/abs/2508.08221); [ICLR 2026 listing](https://mlanthology.org/iclr/2026/liu2026iclr-tricks/)

**RLOO ("Back to Basics", 2024)** [from memory]
- REINFORCE with a leave-one-out baseline: each sample's baseline is the mean of the other k−1 samples. No std division and no PPO clipping is needed in the RLHF setting. It is competitive with PPO, DPO and RAFT, with lower cost. — [arXiv:2402.14740](https://arxiv.org/abs/2402.14740)

**ReMax (2023/ICML 2024)** [from memory]
- REINFORCE whose baseline is the reward of the greedy (argmax) decoded response for the same prompt. It needs only one extra rollout and no value net. Reported savings: about 46% GPU memory versus PPO, with faster training. — [arXiv:2310.10505](https://arxiv.org/abs/2310.10505)

**REINFORCE++ (2025)** [fetched via search snippets]
- A critic-free REINFORCE with PPO-style tricks (clipping, token-level KL penalty). It uses global (batch-level) advantage normalisation instead of per-prompt group normalisation. The claimed benefit is robustness to prompt and reward-model overfitting, which per-prompt normalisation can encourage. — [arXiv:2501.03262](https://arxiv.org/abs/2501.03262)

**GSPO (Qwen, 2025)** [fetched via search snippets]
- The importance ratio is defined on the whole-sequence likelihood (length-normalised), and clipping is done at the sequence level, not per token. Claimed benefits: better efficiency than GRPO, and stability for Mixture-of-Experts RL, where token-level ratios are noisy. Tested on a Qwen3-30B-A3B cold-start model. — [arXiv:2507.18071](https://arxiv.org/abs/2507.18071)

**GMPO (Microsoft Research et al., ICLR 2026)** [fetched via search snippets]
- Replaces GRPO's arithmetic mean of token-level importance-weighted rewards with a geometric mean, which is less sensitive to outlier ratios. Reported: up to +4.1% Pass@1 over GRPO for 7B on maths. — [arXiv:2507.20673](https://arxiv.org/abs/2507.20673); [code](https://github.com/callsys/GMPO); [ICLR 2026 proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/44a1f18afd6d5cc34d7e5c3d8a80f63b-Abstract-Conference.html)

**VAPO (ByteDance Seed, 2025)** [from memory]
- Value-based: it brings back a critic, with value pre-training, decoupled GAE and length-adaptive GAE, plus DAPO-style tricks. Reported: 60.4 on AIME 2024 with Qwen2.5-32B. The paper argues that a well-trained critic beats critic-free methods on long chain-of-thought reasoning. — [arXiv:2504.05118](https://arxiv.org/abs/2504.05118)

**Entropy-collapse fixes** [fetched]
- Entropy collapses early in RL. Empirically, performance R and entropy H follow R = −a·e^H + b, so performance is capped once entropy reaches zero. The change in entropy is driven by the covariance between an action's probability and its logit update, which is roughly its advantage. This covariance stays mostly positive. **Clip-Cov** clips gradient updates for high-covariance tokens. **KL-Cov** puts a KL penalty on them. Both keep exploration going and lift plateaus. — [arXiv:2505.22617](https://arxiv.org/abs/2505.22617)
- Related, not re-fetched: training on only the ~20% highest-entropy "forking" tokens matches or beats full-gradient RLVR — [arXiv:2506.01939](https://arxiv.org/abs/2506.01939) [from memory].

**KL-free variants** [from memory]
- DAPO and Open-Reasoner-Zero drop the KL-to-reference term entirely, and report that it is unnecessary for reasoning RL from a base model — [DAPO arXiv:2503.14476](https://arxiv.org/abs/2503.14476); [Open-Reasoner-Zero arXiv:2503.24290](https://arxiv.org/abs/2503.24290).
- ProRL takes the other side. It keeps KL and periodically resets the reference policy for prolonged-RL stability — [arXiv:2505.24864](https://arxiv.org/abs/2505.24864).

**Off-policy and stale-data stability** [fetched via search snippets]
- M2PO constrains the second moment of the importance weights. It reports stable training on data at least 256 model updates stale while matching on-policy performance. The same paper describes a "prosperity-before-collapse" pattern when the trust region is removed. — [arXiv:2510.01161](https://arxiv.org/abs/2510.01161)

### Inferences
- **Length-normalisation fixes (Dr. GRPO's 1/|o| removal, DAPO token-level loss, overlong shaping) are mostly irrelevant to us.** Our episodes have fixed length (~200 steps), so there is no length bias to correct. The one exception: if the loss averages per-step terms within an episode and then across episodes, that is equivalent to token-level averaging when lengths are equal, so nothing changes.
- **Std-normalisation fixes (Dr. GRPO, Lite PPO batch std, REINFORCE++ global norm) are highly relevant.** See Q5. This is the most likely cheap win, or at least the cheapest bug to rule out.
- **GSPO is risky for us.** It replaces per-step ratios with a sequence ratio. For 200 decisions, even a length-normalised sequence ratio blends 200 independent decisions, so it gives up per-step credit. Its motivation (MoE routing noise in token ratios) does not exist for a small MLP. GMPO's outlier damping is cheap and harmless, but its motivation, outlier token ratios, is weak in a 4-action space.
- **Clip-higher and Clip-Cov target entropy collapse.** For a policy warm-started from a near-deterministic imitation policy, entropy may already be low. That could explain why GRPO could not improve it. Measuring per-step policy entropy at the start of fine-tuning is a cheap diagnostic. If it is near zero, the 8 rollouts are near-identical, groups tie, and there is no signal.
- **VAPO's argument that a critic helps long horizons is worth noting.** It conflicts with the thesis finding that PPO lost to GRPO from scratch. A per-step value net is cheap at 27k parameters, so the critic-free choice is not forced by compute here, only by the finding about variance.

### Gaps
- I did not re-open DAPO, VAPO, RLOO, ReMax, DeepSeekMath, Open-Reasoner-Zero, ProRL or arXiv:2506.01939 in this session. Their numbers above (ε_high=0.28, 50 and 60.4 AIME, 46% memory) are from memory and should be spot-checked before citing in the thesis.
- None of these papers evaluates small non-LLM policies or fixed-horizon control. The transfer to an MLP controller is my inference, not shown evidence.

---

## Q2. Which variants address zero-variance groups (all rollouts tie, so the advantage is 0)?

### Takeaway
There are three families: (1) filter or resample (DAPO dynamic sampling, plus cheaper predictive skipping and prompt replay); (2) extract signal from tied groups anyway (RL-ZVP, which rewards or penalises the absolute outcome); (3) inject an external signal on failed groups (teacher or expert distillation such as RSTG, ReLIFT, LUFFY). For us, (1) wastes expensive rollouts unless the filtering is predictive. (3) maps naturally onto "fall back to regret-expert labels on tied seeds", which is essentially continued DAgger.

### Cited Findings
- DAPO Dynamic Sampling over-samples and discards groups where all rewards are equal, until the batch is full of non-zero-advantage groups — [arXiv:2503.14476](https://arxiv.org/abs/2503.14476) [from memory].
- RL-ZVP extracts signal from zero-variance prompts by "directly rewarding correctness and penalizing errors even without contrasting responses", scaled by token-level features (entropy). Reported gains over GRPO: up to +8.61 accuracy points and +7.77 pass-rate points. — [arXiv:2509.21880](https://arxiv.org/html/2509.21880v1) [fetched via search snippets]
- Prompt Replay (Mar 2026) reuses prompts only, not trajectories, so training stays on-policy. It prioritises prompts with pass rate near 0.5. Results: fewer zero-variance prompts, higher mean |advantage|, and faster early gains (Llama-3.2-3B, Qwen3-8B). Gains plateau and converge to the baseline under aggressive settings. The authors say it is most useful when rollout compute is the bottleneck. — [arXiv:2603.21177](https://arxiv.org/abs/2603.21177) [fetched]
- RSTG ("Distill Where You Fail", Aug 2026) applies on-policy teacher distillation only on negative zero-variance groups (all fail). Samples are weighted by teacher confidence; distillation targets tokens with high student uncertainty or large teacher-student divergence; SFT on correct teacher trajectories adds positive signal. Reported: +4.02% maths and +3.05% code over naive GRPO+on-policy-distillation. — [arXiv:2608.00782](https://arxiv.org/abs/2608.00782) [fetched]
- Difficulty-targeted online data selection with rollout replay is another data-efficiency method aimed at prompts of moderate difficulty — [arXiv:2506.05316](https://arxiv.org/pdf/2506.05316) [title seen in search only].
- Related, not re-fetched: GRESO predicts zero-variance prompts from their history and skips them before rolling out — [arXiv:2506.02177](https://arxiv.org/abs/2506.02177) [from memory]. PODS down-samples rollouts to the most informative (max-variance) subset — [arXiv:2504.13818](https://arxiv.org/abs/2504.13818) [from memory].

### Inferences
- **The core tie problem for us.** Reward is a fraction of ~20 questions, so episodes from a strong imitation start likely differ by 0 or 1 questions. Sometimes all 8 tie on the question term and differ only by tiny cost terms. With std-normalisation, the cost noise becomes a full-size advantage (see Q5). This is worse than a true zero.
- **Cheapest fixes, in order:**
  1. Log the fraction of groups with zero or near-zero spread at fine-tune start.
  2. Seed-level prompt replay or prioritisation: keep a per-seed history of within-group spread and resample seeds near the "frontier", for example seeds where the imitation policy sometimes misses a retrieval.
  3. On tied-failure seeds, apply a DAgger/regret-expert cross-entropy loss on the visited states. This is RSTG translated to our setting, and the expert is free because it is computed from hindsight.
- **RL-ZVP-style absolute-reward terms** amount to adding a REINFORCE term with a fixed baseline, such as the imitation policy's mean score. That is equivalent to a batch-level or global baseline, which is what Lite PPO and REINFORCE++ do.
- **Step-level grouping avoids episode ties.** If episodes tie overall, step-level comparisons can still differ. See GiGPO and Tree-GRPO in Q4.

### Gaps
- RL-ZVP's exact formula for the advantage on tied groups was not read. Only the summary was verified.
- No paper found studies zero-variance groups when the reward is a fraction such as k/20 rather than binary.

---

## Q3. Fine-tuning from a strong imitation/SFT start without degrading it

### Takeaway
LLM work has converged on single-stage hybrids that keep an expert or SFT signal inside the RL loss (LUFFY, SRFT, CHORD, ReLIFT), rather than pure RL after SFT. Robotics work (WSRL) shows that naive online RL from an offline or imitation start causes early "unlearning". For a controller whose expert is cheap (the regret expert is computed in hindsight), the strongest fit is: GRPO loss + expert cross-entropy on visited states (DAgger kept on) + optional KL to the frozen BC policy. The expert term should be weighted up when the policy is uncertain or the group is tied (CHORD, SRFT, RSTG).

### Cited Findings
- LUFFY adds off-policy expert reasoning traces to GRPO groups ("Mixed-Policy GRPO") and shapes the policy with regularised importance sampling, to avoid "superficial and rigid imitation". Reported: +6.4 average on six maths benchmarks and +6.2 out-of-distribution. It trains weak models where on-policy RLVR "completely fails". — [arXiv:2504.14945](https://arxiv.org/abs/2504.14945) [fetched via search snippets]
- SRFT is a single stage. Its loss is an entropy-weighted sum of SFT on demonstrations, RL on demonstrations, RL on positive rollouts (entropy-weighted) and RL on negative rollouts. Reported: 59.1 average on five benchmarks, +9.0 over the best RL baseline and +3.4 over SFT→RL. — [arXiv:2506.19767](https://arxiv.org/pdf/2506.19767) [fetched via search snippets]
- CHORD (ICLR 2026) treats SFT as a dynamically weighted auxiliary term inside RL. A token-level weight raises SFT only where the model is uncertain. — [arXiv:2508.11408](https://alphaxiv.org/abs/2508.11408) [fetched via search snippets]
- ReLIFT alternates RL with SFT steps on expert demonstrations for the "hardest questions", meaning those where recent rollouts all failed. — described in [SFT/RL survey results](https://arxiv.org/pdf/2509.06948) [fetched via search snippets; ReLIFT's own ID, believed to be arXiv:2506.07527, is from memory].
- Other hybrids found but not read: AMFT meta-learns the imitation-exploration balance — [arXiv:2508.06944](https://arxiv.org/pdf/2508.06944); "Beyond Two-Stage Training: Cooperative SFT and RL" — [arXiv:2509.06948](https://arxiv.org/pdf/2509.06948).
- RAFT / rejection-sampling fine-tuning (SFT on own high-reward samples) is a strong minimal baseline. "A Minimalist Approach to LLM Reasoning" reports that RAFT++ is competitive with GRPO early, and that GRPO's edge comes mainly from discarding all-wrong prompts (Reinforce-Rej). — [arXiv:2504.11343](https://arxiv.org/abs/2504.11343) [from memory]
- WSRL (ICLR 2025): when moving from offline to online RL, retaining offline data mostly prevents early divergence caused by distribution shift. A short warm-up of rollouts from the pre-trained policy avoids "unlearning and forgetting" without keeping the offline data. — [arXiv:2412.07762](https://arxiv.org/abs/2412.07762) [fetched via search snippets]

### Inferences
- The thesis result (GRPO fine-tune gives -0.004, roughly no change) matches the LLM picture: RL from a strong SFT start mostly sharpens and does not discover new behaviour. The missing behaviour here, retrieving from the archive at the right time, may be rare under the imitation policy.
- Because our expert is free, the LUFFY idea transfers directly: put 1–2 regret-expert (or hindsight-oracle) rollouts into each group of 8 as off-policy members. The group then always has spread at the expert's level. The policy gets positive advantage toward retrieval behaviours it never samples, which targets "archived but not retrieved". It needs importance weighting or LUFFY's policy shaping, because the expert is off-policy.
- An oracle member in the group is not the same as oracle imitation. The thesis found that imitating the oracle gave a poor start. Using the oracle only as a contrastive group member keeps the student's own distribution as the main training data. This is an untested hypothesis.
- A KL to the frozen BC policy (rather than to an "init" reference) is the standard guard against degradation. ProRL's reference resets show how to loosen it over time.
- RAFT/RFT is a near-free ablation: keep the best-of-8 episodes per seed and behaviour-clone them. If it matches GRPO, the RL machinery is not buying anything.

### Gaps
- No paper found tests these hybrids on non-language, long-horizon control with a cheap hindsight expert. Robotics residual-RL and BC+RL papers (e.g. IBRL, policy decorator) exist but were not checked in this session.
- ReLIFT's arXiv ID was not confirmed.

---

## Q4. Sample efficiency for expensive rollouts: off-policy/replay, partial rollouts, prefix sharing, tree rollouts, step-level credit

### Takeaway
Two agent-specific methods fit our structure very well, because our environment can be replayed exactly from a seed. GiGPO adds step-level groups built from repeated states across the group's trajectories. Tree-GRPO branches rollouts from shared prefixes, so several episodes share early LLM calls and step-level advantages come almost free. Off-policy reuse (M2PO, replay) is a secondary lever, useful because the 27k-parameter policy itself is cheap and the LLM reader is the cost.

### Cited Findings
- GiGPO (NeurIPS 2025) keeps the episode-level group advantage. It adds step-level advantages by "retroactively identifying repeated environment states (anchor states) across trajectories" and grouping actions taken from the same state. Reported: >12% over GRPO on ALFWorld and >9% on WebShop, at "<0.002%" extra time cost, with no critic. — [arXiv:2505.10978](https://arxiv.org/abs/2505.10978v3); [NeurIPS 2025 poster](https://neurips.cc/virtual/2025/poster/118123) [fetched via search snippets]
- Tree-GRPO (ICLR 2026): each tree node is a full agent interaction step. Sharing prefixes raises the number of rollouts per fixed budget of tokens or tool calls. Intra-tree and inter-tree group advantages give step-level process signals from outcome reward alone. Intra-tree GRPO is shown equivalent to step-level preference learning. Better than chain-based RL across 11 QA datasets. — [arXiv:2509.21240](https://arxiv.org/abs/2509.21240) [fetched via search snippets]
- TRACE (Jun 2026) is a unified rollout-budget allocation framework for agentic RL — [arXiv:2606.11119](https://arxiv.org/pdf/2606.11119) [title seen in search only, not read].
- Off-policy GRPO:
  - M2PO stays stable with data at least 256 updates stale — [arXiv:2510.01161](https://arxiv.org/abs/2510.01161).
  - "How Off-Policy Can GRPO Be? Mu-GRPO" — [arXiv:2605.17570](https://arxiv.org/pdf/2605.17570) [title seen only].
  - A rollout-level, advantage-prioritised replay buffer with age-based eviction, and RePO (replay of off-policy responses), were mentioned in search results but not opened — [arXiv:2602.20722](https://arxiv.org/html/2602.20722v2) [unverified which paper this is].
- Partial rollouts (resuming long, unfinished generations in later iterations) come from Kimi k1.5 — [arXiv:2501.12599](https://arxiv.org/abs/2501.12599) [from memory]. Fully asynchronous RL with staleness-aware PPO comes from AReaL — [arXiv:2505.24298](https://arxiv.org/abs/2505.24298) [from memory].

### Inferences
- **Tree branching is the strongest fit for our cost structure.** With a fixed seed the episode stream is deterministic, and LLM calls happen only at question times (~20 per episode). A tree that shares the first t steps and branches at a "decision point" reuses cached LLM answers for questions before t.
  - Good branch points: where the controller archives something, or could retrieve.
  - Branches at the same node form a step-level group. Their return difference is pure credit for the decision at t, which is exactly the "should I retrieve now?" decision the gap analysis points to.
  - Rough accounting: branching 4 ways at step 100 costs ~10 LLM calls per branch instead of 20.
- **GiGPO anchor states map onto our states, with a caveat.** Two rollouts of the same seed that reach the same (step index, memory-set) state form a step group. Exact memory-set matches may be rare after divergence. Grouping by step index plus a coarse state hash, or Tree-GRPO's explicit branching, avoids relying on chance overlap.
- **Partial rollouts and async infrastructure solve an LLM-generation-length problem we do not have.** Our policy forward pass is negligible.
- **Replay plus a PPO clip lets each expensive episode be used for several updates.** With 27k parameters, many epochs over a buffer are cheap. The risk is staleness. M2PO-style second-moment ratio control is a principled guard.
- **The cheapest efficiency trick may be the reader, not RL.** Caching the reader's answer per (question, retrieved-context) pair removes repeated LLM calls across rollouts whenever memory states coincide. This is my suggestion, not taken from any paper.

### Gaps
- Mu-GRPO, TRACE, arXiv:2602.20722 and arXiv:2606.04560 were seen only as search-result titles. Their claims are not verified.
- The Kimi k1.5 and AReaL details are from memory.

---

## Q5. Pitfalls of std-normalisation with binary or few-valued rewards

### Takeaway
Dividing by the per-group std up-weights groups that are nearly tied: too easy, too hard, or for us, "all 8 answer the same questions and differ only by tiny cost terms". This turns noise into full-size gradients. The fixes are to remove std (Dr. GRPO) or use batch- or global-level std (Lite PPO, REINFORCE++), possibly with an epsilon floor or a minimum-spread threshold.

### Cited Findings
- "Questions with lower standard deviations (e.g., those that are too easy or too hard, with the outcome rewards being almost all 1 or 0) are given higher weights during policy updates." Dr. GRPO removes the std term. — [arXiv:2503.20783](https://arxiv.org/html/2503.20783) [fetched]
- Lite PPO's recommended setting uses the group mean for the baseline but the batch std for scale, described as "transforming sparse rewards into robust signals". — [arXiv:2508.08221](https://arxiv.org/abs/2508.08221) [fetched via search snippets]
- REINFORCE++ uses global advantage normalisation instead of per-prompt normalisation, motivated by robustness against prompt overfitting. — [arXiv:2501.03262](https://arxiv.org/abs/2501.03262) [fetched via search snippets]
- RLOO uses a leave-one-out mean baseline with no std division — [arXiv:2402.14740](https://arxiv.org/abs/2402.14740) [from memory].
- Related, not re-fetched: Mroueh analyses GRPO's effective loss with binary rewards. It shows that std-normalised GRPO acts as an adaptive weighted contrastive loss whose weights depend on the group's success probability. — [arXiv:2503.06639](https://arxiv.org/abs/2503.06639) [from memory; verify].

### Inferences
- **Concrete risk in the current recipe.** Suppose 8 rollouts each answer 17/20 questions, and their costs differ by about 0.002. Group std ≈ 0.001, so advantages become ±1 driven entirely by cost. The update then pushes the policy toward fewer archive or retrieve actions, which is the opposite of what closes the "not retrieved" gap. This could explain why GRPO fine-tuning was flat or slightly negative. It is testable by logging, for each group, the std and the share of the advantage due to cost terms.
- **Recommended ablation:** (a) group-mean baseline with no std (Dr. GRPO); (b) group mean with batch std (Lite PPO); (c) (a) or (b) plus skipping groups whose spread on the question term is zero. Each is a one-line change.
- **Few-valued rewards make ties common.** The k/20 reward with G=8 makes exact ties on the question term frequent once the policy is good. Step-level advantages (Q4) or a larger G on frontier seeds are the structural fixes.

### Gaps
- No paper found quantifies the std pitfall for mixed reward (discrete accuracy plus small continuous cost). The cost-amplification mechanism above is my inference from the cited bias.
- Mroueh (2503.06639) was not re-opened.

---

## Summary judgement for the memory controller (ranked by expected value per effort)

1. **Fix advantage normalisation:** no std, or batch std (Dr. GRPO, Lite PPO). Separate or rescale the cost term so it cannot dominate near-tied groups. Cheap, and directly targets a plausible cause of the flat fine-tune.
2. **Keep the expert in the loop during RL:** DAgger/regret cross-entropy on visited states plus KL to the BC policy (CHORD, SRFT, ReLIFT, RSTG style). Up-weight it on tied or failed seeds. Protects the warm start.
3. **Put expert or oracle trajectories into GRPO groups** (LUFFY-style mixed-policy groups) to create contrast that points toward retrieval behaviour the student rarely samples.
4. **Use step-level credit through branching** from shared prefixes at archive and retrieve decision points (Tree-GRPO; GiGPO anchor states). This attacks both sparse reward and rollout cost, and fits the deterministic, seed-replayable environment.
5. **Prioritise seeds with spread** (prompt replay on seeds) rather than DAPO-style over-sample-and-discard, which wastes LLM calls.
6. **Monitor entropy.** Use clip-higher or Clip-Cov only if entropy is near zero at fine-tune start.
7. **Low priority or not applicable:** length-normalisation fixes (fixed horizon); overlong shaping; GSPO and GMPO (motivated by token-ratio and MoE noise in large LLMs); partial rollouts and async RL (generation length is not our bottleneck).
8. **Free ablation:** RAFT / best-of-8 behaviour cloning, to check whether the RL objective adds anything over filtered imitation.
