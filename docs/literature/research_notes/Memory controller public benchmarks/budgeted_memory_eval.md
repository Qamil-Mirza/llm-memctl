# Budget-constrained memory management for LLMs, and fair evaluation on LoCoMo / LongMemEval

Notes compiled 2026-10-06. "SR" = self-reported by the authors or vendor, not independently reproduced. Several numbers below came from WebFetch summaries of arXiv HTML pages; they should be spot-checked against the PDF tables before they go into the thesis.

## Q1. Learned vs heuristic eviction/compression: when do learned policies win, and with what training signal?

### Takeaway
Learned eviction/compression beats recency/salience heuristics mainly when (a) it is trained on data from the target distribution, (b) its training signal is the *future utility* of each item, computed in hindsight (the same idea as our oracle teacher), and (c) the heuristics it is compared with have no access to the query. When heuristics *can* see the query (SnapKV/TOVA in KV-cache work; top-k BM25 in our setting), they close most of the gap to learned policies. Learned policies also tend to specialize to their training distribution.

### Cited Findings
- **KV Policy (KVP, 2026)**: per-head RL agents (2-layer MLP, ~650K params each, 112 agents) trained offline on generation traces. The supervision is *future attention*: the cumulative attention a token actually receives later, computed in hindsight. Training uses Plackett-Luce ranking with REINFORCE leave-one-out. The reward is the negated AUC of future-attention loss across *all* budgets, so one policy works at any budget. — [Learning to Evict from KV Cache, arXiv 2602.10238](https://arxiv.org/html/2602.10238) (SR)
- KVP beats StreamingLLM (recency+sinks), H2O, KeyDiff, K-Norm, LagKV, and Random on RULER-4k and OASST2-4k. It extrapolates from 4K to RULER-128K. — [arXiv 2602.10238](https://arxiv.org/html/2602.10238) (SR)
- KVP caveats: on zero-shot tasks, "a specialization pattern emerges": the OASST2-trained agent does best on conversational tasks and the RULER-trained one on retrieval. LagKV sometimes matches KVP at the tightest budgets. Query-aware attention methods (SnapKV, TOVA) "achieve similar per-budget costs as KVP when given query information at inference time." Supervised learning-to-rank losses (regression, ranking, soft-sort) did *worse* than the RL ranking objective: they over-fit to dominant tokens or could not separate the top items under tight budgets. — [arXiv 2602.10238](https://arxiv.org/html/2602.10238) (SR)
- ForesightKV, also training-based and predicting each KV pair's long-term contribution, reports beating prior methods at half the cache budget on reasoning benchmarks. — [arXiv 2602.03203](https://arxiv.org/html/2602.03203v1) (SR, not fetched in detail)
- **Memory-R1 (2025)**: an LLM memory manager (ADD/UPDATE/DELETE/NOOP) plus an answer agent, trained with PPO/GRPO. The reward is exact-match correctness of a frozen answer agent after each memory operation. It was trained on **only 152 LoCoMo QA pairs** (1:1:8 train/val/test = 152/81/1307 questions), with the adversarial subset excluded. — [arXiv 2508.19828](https://arxiv.org/html/2508.19828) (SR)
- Memory-R1 LoCoMo results, LLaMA-3.1-8B: GRPO F1 35.73 / BLEU-1 27.70 / Judge 59.83, versus Mem0 27.29 / 18.63 / 43.93 and A-Mem 21.62 / 16.93 / 44.76. Qwen-2.5-7B: GRPO F1 43.14 / Judge 61.51, versus Mem0 30.61 / 53.30. The paper claims zero-shot gains on MSC and LongMemEval despite training only on LoCoMo. — [arXiv 2508.19828](https://arxiv.org/html/2508.19828) (SR; the judge model was not identified in the fetched text)
- **Mem-α (2025)**: Qwen3-4B trained with GRPO to write to core/episodic/semantic memory. Reward = QA correctness + tool-format success + 0.05 × compression ratio + memory-validity judge. It trained on 562 instances (SQuAD, HotpotQA, PerLTQA, LME-train, NLU, TREC-Coarse, PubMed-RCT, BookSum), each at most ~20-30K tokens. — [arXiv 2509.25911](https://arxiv.org/html/2509.25911v1) (SR)
- Mem-α on the MemoryAgentBench test set: avg 0.592 (memory 129K tokens), versus long-context 0.461 (33K) and BM25 RAG-top2 0.502 (207K). Validation: 0.642 (7.9K), versus long-context 0.588 (10.8K) and BM25 0.567 (11.3K). Test inputs exceed 400K tokens, about 13x the training length. — [arXiv 2509.25911](https://arxiv.org/html/2509.25911v1) (SR). Note: on the test set Mem-α uses ~4x more memory than the long-context baseline, so the comparison is **not budget-matched**.
- **ACON (2025)**: optimizes a natural-language "compression guideline" for an LLM compressor, without fine-tuning. It finds paired trajectories where full context succeeds and compressed context fails, has an LLM analyze why, and updates the guideline. Results: 26-54% lower peak tokens with task success maintained or improved on AppWorld, OfficeBench, and multi-objective QA. Distilled small compressors keep >95% of accuracy, and smaller agent LMs improve by up to 46%. — [arXiv 2510.00615](https://arxiv.org/abs/2510.00615); [MSR page](https://www.microsoft.com/en-us/research/?p=1173094) (SR)
- **AgentFold (2025)**: the agent learns a "folding" action that either condenses one step finely or consolidates whole sub-tasks. It is motivated by the observation that ReAct contexts saturate with noise, while "fixedly summariz[ing] the full history at each step risk[s] the irreversible loss of critical details." AgentFold-30B-A3B scores 36.2% on BrowseComp and 47.3% on BrowseComp-ZH. — [arXiv 2510.24699](https://arxiv.org/pdf/2510.24699) (SR)
- **RECOMP (2023)**: extractive and abstractive compressors trained with the *end-task LM signal*. They compress retrieved documents to as little as 6% with minimal loss, can return an empty string (selective augmentation), and transfer across LMs. — [arXiv 2310.04408](https://arxiv.org/abs/2310.04408) (SR)
- **LongLLMLingua (2023)**: *question-aware* prompt compression. Up to +21.4% on NaturalQuestions with ~4x fewer tokens (GPT-3.5-Turbo), 94% cost reduction on LooGLE, and 1.4-2.6x latency speed-up. Its premise is that performance depends on the "density and position of key information." — [arXiv 2310.06839](https://arxiv.org/abs/2310.06839) (SR)
- **Neural Paging (2026)**: frames the context window as a semantic cache with a learned Page Controller trained by PPO. It defines a "Semantic Belady" oracle that evicts the block with the longest time until next use. On synthetic Zipf traces (8 cache blocks) the fault rates are Belady 0.121, LRU 0.226 (competitive ratio ~1.86), and FIFO/LFU/Random worse. It reports **no** learned-policy results and **no** real-LLM evaluation, and says "transfer across domains remains an open problem." — [arXiv 2603.02228](https://arxiv.org/html/2603.02228v1)

### Inferences
- The closest published analogue to our controller (KVP: small policy, hindsight future-utility signal, multi-budget objective) succeeds *in distribution* and transfers partly, specializing to its training domain. That matches our pattern: strong on synthetic data, weak on LoCoMo/LongMemEval.
- KVP's finding that query-aware heuristics match a learned query-blind policy is directly relevant. Top-5 BM25 sees the question; our controller decides keep/archive/delete *before* the question arrives. Part of our gap may be structural, not a training failure. Two fair comparisons follow: (i) give the controller a retrieve step at question time, or (ii) compare against query-blind heuristics only, reporting BM25 separately as a "query-aware" reference.
- KVP's ablation (pure ranking/regression losses < RL with a budget-weighted cost) suggests our imitation loss may be weighting the wrong errors. A multi-budget, cost-weighted objective is worth trying.
- Memory-R1 shows that a small in-domain training split (152 QAs, 1:1:8 split of LoCoMo) is an accepted protocol. Fine-tuning or imitating on a LoCoMo train split and testing on the rest is defensible, if the split is disclosed.
- Mem-α, ACON, and Memory-R1 all reward downstream QA/task success through the actual reader LM. Our teacher's "which items will future questions need" is an upstream proxy. On real dialogue, evidence labels are noisy, so a reader-in-the-loop reward (as in GRPO) is likely more robust than the oracle labels.

### Gaps
- Gist tokens, xRAG, MemTool, H2O/SnapKV/StreamingLLM primary papers: not fetched. Their numbers are not included here.
- I found no paper that compares a tiny (<100K-param) item-level learned controller against FIFO/BM25 on LoCoMo or LongMemEval under a matched token budget. This appears to be an open niche.

## Q2. Summarize vs delete: hierarchical/consolidation approaches on LoCoMo / LongMemEval

### Takeaway
On both benchmarks, storing evicted content as **atomic facts/observations or fine-grained rounds** helps more than storing **session summaries**. Session summaries performed about the same as raw dialog on LoCoMo. On LongMemEval, compressing too far (into facts as the stored values) lost information, except on multi-session questions.

### Cited Findings
- LoCoMo RAG with GPT-3.5-turbo-16K, by retrieval unit (overall F1). Dialog top-5 31.7, top-25 35.8. **Observation** top-5 41.4, top-25 38.0. **Session summary** top-5 32.5, top-10 31.5. The paper concludes observations beat raw dialog and summaries, and more retrieved items hurt (signal-to-noise). — [LoCoMo, arXiv 2402.17753](https://arxiv.org/html/2402.17753)
- LoCoMo baselines: full 16K context with GPT-3.5-turbo-16K reaches 37.8 F1 (temporal 20.3, adversarial 2.1). Humans score 87.9 F1. — [arXiv 2402.17753](https://arxiv.org/html/2402.17753)
- LongMemEval: splitting sessions into rounds "significantly enhances reading performance." Compressing into facts caused information loss, except for multi-session queries. Fact-augmented multi-key indexing (facts used as *keys*, raw rounds as *values*) improved Recall@5 by 9.4% and accuracy by 5.4%. Chain-of-Note with JSON formatting added up to 10 points. — [LongMemEval, arXiv 2410.10813](https://arxiv.org/html/2410.10813)
- AgentFold motivates multi-scale folding because fixed full-history summarization "risk[s] the irreversible loss of critical details." — [arXiv 2510.24699](https://arxiv.org/pdf/2510.24699)
- BEAM's LIGHT combines episodic long-term memory, short-term working memory, and a salient-facts scratchpad. It reports +3.5% to +12.69% over the strongest baselines. — [BEAM, arXiv 2510.27246](https://arxiv.org/html/2510.27246v1) (SR)

### Inferences
- For our "archive" action, the evidence favors archiving into a *retrievable* store, indexed by extracted facts and returning raw rounds. A lossy summary that replaces the content is the weaker choice. A summary node can help multi-hop/multi-session questions, but should not be the only copy.
- LoCoMo's "more retrieved = worse" result means budget is not monotone. A 25% budget can score below a 5% one if the extra content is noise. Our budget sweeps should expect non-monotone curves for both heuristics and learned controllers.

### Gaps
- MemGPT recursive summaries, the MemoryBank forgetting curve, and RAPTOR: not fetched. I found no controlled LoCoMo/LongMemEval ablation of "summarize evicted" vs "delete evicted" at a fixed token budget.

## Q3. Distribution shift and synthetic/real training data

### Takeaway
There is little evidence on synthetic-to-real transfer for memory policies. The successful learned systems either train on real benchmark-style data (Memory-R1 on LoCoMo; Mem-α on a mix including PerLTQA and LongMemEval-train) or show only domain-specialized transfer (KVP). Neural Paging names cross-domain transfer as an open problem. Several generators and benchmarks could serve as more realistic training data.

### Cited Findings
- KVP: the conversation-trained agent is best on conversational tasks and the retrieval-trained agent on retrieval. — [arXiv 2602.10238](https://arxiv.org/html/2602.10238)
- Neural Paging: synthetic traces only; "transfer across domains remains an open problem." — [arXiv 2603.02228](https://arxiv.org/html/2603.02228v1)
- Mem-α's training mix includes **PerLTQA** and **LME-train**, a LongMemEval-style training split, alongside SQuAD/HotpotQA/BookSum/classification sets. It generalizes from ≤30K to >400K tokens. — [arXiv 2509.25911](https://arxiv.org/html/2509.25911v1) (SR)
- LongMemEval's construction recipe embeds task "needle" sessions in filler sessions drawn from ShareGPT/UltraChat. Construction took ~400 annotator hours. — [arXiv 2410.10813](https://arxiv.org/html/2410.10813)
- **MemoryAgentBench** converts long-context datasets into incremental multi-turn chunks. It tests four competencies: accurate retrieval, test-time learning, long-range understanding, and *selective forgetting*. No current method masters all four. — [arXiv 2507.05257](https://arxiv.org/html/2507.05257v2) (ICLR 2026)
- **BEAM** is an automatic generator of coherent, diverse conversations up to 10M tokens: 100 conversations and 2,000 validated questions. LLMs with 1M-token context windows (with or without RAG) still struggle as dialogues lengthen. Data is at [HF Mohammadta/BEAM](https://huggingface.co/datasets/Mohammadta/BEAM) and [BEAM-10M](https://huggingface.co/datasets/Mohammadta/BEAM-10M). — [arXiv 2510.27246](https://arxiv.org/html/2510.27246v1)
- Memory-R1 reports zero-shot transfer from LoCoMo to MSC and LongMemEval. — [arXiv 2508.19828](https://arxiv.org/html/2508.19828) (SR)

### Inferences
- The cheapest way to reduce the shift: train the controller (imitation + GRPO) on LongMemEval-style generated histories (needles + ShareGPT/UltraChat filler) or BEAM conversations. Evaluate on LoCoMo and a held-out LongMemEval split. Avoid testing on the exact generator used for training.
- MemoryAgentBench's "selective forgetting" competency maps directly onto our delete action. It is a natural extra benchmark where a learned deletion policy could plausibly beat FIFO.

### Gaps
- PerLTQA, MSC, and MemBench construction details were not fetched. I found no published study that trains a memory policy on purely synthetic task data and measures the transfer gap on LoCoMo.

## Q4. Evaluation pitfalls and reporting conventions on LoCoMo / LongMemEval

### Takeaway
LoCoMo has a ~6.4% wrong answer key, an unusable adversarial category (conventionally excluded), and a widely copied judge prompt that is lenient. It is also short enough (~9K-26K tokens) that full context is a very strong baseline. LongMemEval_S (~115K tokens) and _M (~1.5M tokens) are cleaner. The official GPT-4o judge reached >97% human agreement, and the dataset has 30 abstention questions (`_abs`). Report F1 and judge, per category, with a full-context baseline and an oracle/evidence-only reader upper bound.

### Cited Findings
**LoCoMo data and annotation**
- The original paper describes 50 conversations: avg 304.9 turns, 19.3 sessions, 9,209.2 tokens. Five QA categories: single-hop, multi-hop, temporal, open-domain, adversarial (expected answer: unanswerable). Metric: F1 after answer normalization. — [arXiv 2402.17753](https://arxiv.org/html/2402.17753)
- The released and commonly used version is 10 conversations, ~600 dialogues and ~26K tokens each (Mem0). Zep says 16K-26K tokens. Both differ from the paper's 9.2K average, so state which release you use. — [Mem0, arXiv 2504.19413](https://arxiv.org/html/2504.19413); [Zep blog](https://www.getzep.com/blog/lies-damn-lies-statistics-is-mem0-really-sota-in-agent-memory/)
- Audit: 99 of 1,540 non-adversarial answers are wrong (6.4%). Error types are hallucinated facts, wrong temporal reasoning, and 24 speaker-attribution errors. The best possible score is ~93.6%. — [Penfield Labs LoCoMo audit](https://penfieldlabs.substack.com/p/we-audited-locomo-64-of-the-answer); corrected key at [dial481/locomo-audit](https://github.com/dial481/locomo-audit)

**LoCoMo category 5 and judge leniency**
- Category 5 (adversarial) is excluded by convention. Mem0's reason: "ground truth answers were unavailable." Zep: "unusable due to missing ground truth answers." Memory-R1 also excludes it. — [arXiv 2504.19413](https://arxiv.org/html/2504.19413); [Zep blog](https://www.getzep.com/blog/lies-damn-lies-statistics-is-mem0-really-sota-in-agent-memory/); [arXiv 2508.19828](https://arxiv.org/html/2508.19828)
- Judge leniency: Mem0's judge prompt says "be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT" and accepts relative dates. — [arXiv 2504.19413](https://arxiv.org/html/2504.19413)
- The gpt-4o-mini judge accepted **62.81%** of intentionally wrong but topically adjacent answers, while catching specific factual errors ~89% of the time. — [Penfield audit](https://penfieldlabs.substack.com/p/we-audited-locomo-64-of-the-answer)

**Full-context baselines and reproducibility**
- Full context is strong on LoCoMo. Mem0 reports full context ~73% J (26,031 tokens), best RAG (k=2, 256-token chunks) ~61%, Mem0 66.88 ± 0.15, and Mem0g 68.44 ± 0.17. Each is the mean ± SD of 10 runs. — [arXiv 2504.19413](https://arxiv.org/html/2504.19413) (SR). Zep reports itself at 75.14 ± 0.17 against a 72.90 full-context baseline. — [Zep blog](https://www.getzep.com/blog/lies-damn-lies-statistics-is-mem0-really-sota-in-agent-memory/) (vendor SR, disputed between vendors)
- No standard pipeline: systems differ in ingestion, answer prompt, and models. The audit cites reproducibility issues (EverMemOS #73, Mem0 #3944, a Zep score discrepancy). — [Penfield audit](https://penfieldlabs.substack.com/p/we-audited-locomo-64-of-the-answer)

**F1 vs judge, judge robustness, and context saturation**
- F1 vs judge: F1 penalizes abstractive answers. In one comparison A-Mem ranked 4th on the judge but 5th on F1 (0.116), and SimpleMem got higher F1 (0.268) despite weak synthesis. Across three judge rubrics (MAGMA, Nemori, SimpleMem), absolute scores moved but the **ranking of systems stayed consistent**. — [Anatomy of Agentic Memory, arXiv 2602.19320](https://arxiv.org/html/2602.19320)
- Saturation: LoCoMo (~20K) and MemBench (~100K) fit in a 128K window. Only LongMemEval_M (>1M) structurally requires external memory. The paper proposes a "Context Saturation Gap" (memory system minus full context) as a diagnostic. — [arXiv 2602.19320](https://arxiv.org/html/2602.19320)

**LongMemEval splits and judging**
- Splits: 500 questions. `longmemeval_s` is ~115K tokens / ~40 sessions (Llama-3 tokenizer). `longmemeval_m` is ~500 sessions. `longmemeval_oracle` contains evidence sessions only. Cleaned `_s_cleaned` / `_m_cleaned` files were released 2025-09. Abstention questions have `question_id` ending in `_abs` (30 instances). Official eval: `evaluate_qa.py gpt-4o`. — [LongMemEval repo](https://github.com/xiaowu0162/LongMemEval)
- _M is ~1.5M tokens. The GPT-4o judge had >97% agreement with human experts. Oracle vs _S accuracy: GPT-4o ~87% vs ~60%, Llama-3.1-70B 74% vs 33%, Llama-3.1-8B 71% vs 45%. — [arXiv 2410.10813](https://arxiv.org/html/2410.10813)
- The audit argues LongMemEval_S (~115K/question) now fits in 200K-1M context windows, making it "a context-window test." — [Penfield audit](https://penfieldlabs.substack.com/p/we-audited-locomo-64-of-the-answer) (opinion)

**Self-judging**
- LLM judges favor their own outputs. GPT-4 shows significant self-preference, and the mechanism is a preference for low-perplexity text, whatever its source. — [Self-Preference Bias in LLM-as-a-Judge, arXiv 2410.21819](https://arxiv.org/html/2410.21819v1)

### Inferences
- Our setup uses the same Qwen2.5-7B as reader and judge, which risks self-preference. However, self-preference favors the reader's style equally across memory policies, so *rankings* between policies are probably less affected than absolute numbers (consistent with 2602.19320's ranking stability). The risk is larger when policies change answer *style*, e.g. verbose answers under thin context. Re-judge a sample with a different-family judge (GPT-4o per LongMemEval convention, or another open model) and report agreement.
- Validate our judge prompt the way Penfield did: feed it deliberately wrong but topically adjacent answers and report the false-accept rate. If we used a Mem0-style "generous" prompt, our judge scores are inflated and less discriminative between policies.
- Recommended reporting set:
  - full context (no budget), with the saturation gap relative to it;
  - an evidence-only oracle reader (LongMemEval _oracle, or LoCoMo evidence turn IDs);
  - our hindsight oracle controller at the same budget;
  - FIFO / keep-last-k / salience;
  - BM25 top-k at a **matched token budget**, flagged as query-aware;
  - the learned controller.
- Further reporting conventions:
  - Per-category results; LoCoMo cat 1-4 only (cat 5 reported separately if at all); LongMemEval abstention reported separately.
  - Use the cleaned LongMemEval_S, and the corrected LoCoMo key as a robustness check.
  - Mean ± SD over seeds/runs (Mem0 used 10 runs).
- With a 2-25% budget on LoCoMo (~26K tokens), the controller holds ~500-6,500 tokens. Top-5 BM25 over observations scored best in LoCoMo's own RAG table, so beating it under a tight budget is a high bar.

### Gaps
- I did not find the LoCoMo-Plus or BEAM judge protocols in detail, nor a published false-accept rate for Qwen2.5-7B as a judge.

## Q5. Oracle / hindsight upper bounds for memory

### Takeaway
Hindsight oracles are common as *upper bounds* and as *training signals*. Examples: Belady (evict the item whose next use is furthest away), future-attention ranking (KVP), and evidence-only context (LongMemEval_oracle). Our oracle/regret teacher fits this lineage.

### Cited Findings
- Semantic Belady: offline policy that evicts the block with the largest time-to-next-use. It assumes access is independent of the policy, plus a "bounded-sensitivity" relaxation. On synthetic traces, LRU is ~1.86x the oracle fault rate. — [arXiv 2603.02228](https://arxiv.org/html/2603.02228v1)
- KVP uses realized future attention from generation traces as its hindsight ranking target. — [arXiv 2602.10238](https://arxiv.org/html/2602.10238)
- LongMemEval_oracle (evidence sessions only) is the standard reader upper bound (GPT-4o ~87%). — [LongMemEval repo](https://github.com/xiaowu0162/LongMemEval); [arXiv 2410.10813](https://arxiv.org/html/2410.10813)
- LoCoMo annotates the evidence turn IDs for each QA, which allows an evidence-only upper bound. — [arXiv 2402.17753](https://arxiv.org/html/2402.17753)
- H2O frames its heavy-hitter eviction as an "oracle"-style policy for KV caches. Follow-up work notes that Belady is optimal for standard caches "but not necessarily for KV cache." — [H2O, arXiv 2306.14048](https://arxiv.org/html/2306.14048v3) (search-snippet level only)

### Inferences
- Report two ceilings: (1) the evidence-only reader, which measures reader capability, and (2) the hindsight budgeted oracle at each budget, which measures the controller ceiling. Then the gap can be split into "controller vs oracle" and "oracle vs evidence-only". If our hindsight oracle under budget is itself not far above FIFO/BM25 on LoCoMo, the headroom for any controller is small. Measure this before more tuning.
- Belady-style oracles assume access is independent of the policy. Our oracle has the same assumption when it marks items "needed by future questions." It breaks if the reader's answer depends on co-retained context (multi-hop). That is a plausible reason the oracle-imitation signal transfers worse on LoCoMo multi-hop/temporal questions.

### Gaps
- I found no paper that defines a budgeted hindsight oracle specifically for LoCoMo or LongMemEval and reports its score. This appears to be unreported.
