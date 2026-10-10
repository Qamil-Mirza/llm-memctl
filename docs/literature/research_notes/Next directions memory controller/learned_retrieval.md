# Learned retrieval and "when/what to retrieve" methods (2024-2026) for closing the "archived but not retrieved" gap

Verification legend: items marked **[V]** were checked against a search result or fetched page in this session. Items marked **[PK]** come from prior knowledge of the paper (arXiv ID believed correct) but the specific number or detail was NOT re-verified in this session; treat as needing a check before citing in the thesis.

Project framing used to judge relevance: the gap is ~300 "in archive, not retrieved" failures per 100 episodes (oracle: 72) plus ~20 "retrieved but ignored". Retrieval is lexical hashing top-3. Hindsight labels of the answering archive item exist. LLM calls cost ~1 s, so training signals that avoid the LLM are preferred.

## Q1. RL-trained retrieval/search agents: what is learned, reward, algorithm

### Takeaway
Almost all 2025 search-agent RL work fine-tunes a 3B-7B LLM to emit queries and decide when to search, using an outcome reward (exact match/F1) and PPO or GRPO. Two results transfer directly to a small, separate controller: s3 (train only the searcher, frozen generator, reward = gain over naive RAG, ~2.4k samples) and DeepRetrieval (reward = retrieval recall itself, no generator in the loop). StepSearch shows that step-level information-gain rewards beat sparse outcome rewards, which is the analogue of using hindsight labels as dense reward.

### Cited Findings
- **s3** decouples the searcher from the generator and trains only the searcher with a "Gain Beyond RAG" reward, the improvement in generator accuracy over naive RAG; the generator can be frozen or proprietary. **[V]** — [arXiv:2505.14146](https://arxiv.org/abs/2505.14146)
- s3 uses 2.4k training samples and beats baselines trained on >70x more data across six general and five medical QA benchmarks; EMNLP 2025; code at github.com/pat-jj/s3. **[V]** — [arXiv:2505.14146](https://arxiv.org/html/2505.14146v2)
- s3 critique of prior work: search-only metrics (e.g., NDCG) ignore downstream utility, while joint fine-tuning entangles retrieval with generation and is incompatible with frozen models. **[V]** — [arXiv:2505.14146](https://arxiv.org/abs/2505.14146)
- **DeepRetrieval**: MDP where state = original query, action = rewritten/augmented query, reward = retrieval metric (recall, NDCG, or SQL execution accuracy); no supervised data. 65.07% recall on publication search (prior SOTA 24.68%) and 63.18% on trial search (prior 32.11%) with a 3B model. **[V]** — [arXiv:2503.00223](https://arxiv.org/abs/2503.00223v3)
- **StepSearch**: step-wise PPO ("StePPO") with token-level process rewards based on information gain of each retrieval step plus redundancy penalties; motivated by sparse global rewards underperforming on multi-hop QA. +11.2 and +4.2 absolute points for 3B and 7B over RL search baselines with 19k training examples; EMNLP 2025. **[V]** — [arXiv:2505.15107](https://arxiv.org/abs/2505.15107)
- Survey characterisation: Search-R1 uses PPO and GRPO with rule-based outcome rewards; R1-Searcher is two-stage cold-start RL to learn when to invoke search and how to use it; R1-Searcher++ adds SFT, an internal-knowledge reward to avoid redundant search, and a dynamic memory; ReSearch is end-to-end RL without supervised tool-use traces; ReZero rewards retrying a search. **[V]** — [Survey arXiv:2510.16724](https://arxiv.org/abs/2510.16724v2); list of papers at [Awesome-RL-based-Agentic-Search-Papers](https://github.com/ventr1c/Awesome-RL-based-Agentic-Search-Papers)
- **Search-R1** details: interleaves `<search>` calls in reasoning, masks retrieved tokens from the policy-gradient loss, outcome EM reward, Qwen2.5-3B/7B. **[PK]** — [arXiv:2503.09516](https://arxiv.org/abs/2503.09516). Reported improvement figures differ between arXiv versions (v1 vs later); do not quote a single number without checking the version. **[PK]**
- **R1-Searcher**: stage 1 rewards issuing retrieval calls (format/retrieve reward), stage 2 rewards answer correctness. Algorithm reported as REINFORCE++ in the original paper per prior knowledge, while the survey above calls it PPO — conflicting; check the paper. **[PK]** — [arXiv:2503.05592](https://arxiv.org/abs/2503.05592); R1-Searcher++ [arXiv:2505.17005](https://arxiv.org/abs/2505.17005) **[PK ID]**
- **ReSearch**: GRPO, reasoning with search as part of the chain, outcome reward only. **[PK]** — [arXiv:2503.19470](https://arxiv.org/abs/2503.19470)
- **ZeroSearch**: replaces the real search engine with an LLM that simulates search results, with a curriculum that progressively degrades document quality; avoids API cost during RL. **[PK]** — [arXiv:2505.04588](https://arxiv.org/abs/2505.04588)
- **Memory-R1** (memory-specific successor): two RL-fine-tuned agents, a Memory Manager choosing {ADD, UPDATE, DELETE, NOOP} and an Answer Agent; on LoCoMo (152 train / 81 val / 1,307 test questions) Memory-R1-GRPO improves F1 by 68.9% over Mem0; ACL 2026. **[V]** — [arXiv:2508.19828](https://arxiv.org/pdf/2508.19828v3). The Answer Agent filters ("distils") the retrieved memories before answering. **[PK]**
- **Iterative retrieval via policy optimization**: the retriever is an MDP trained from LLM feedback, deciding which item to append next; only ~4M extra parameters for a state encoder on top of an off-the-shelf dense retriever; generalises to inference LLMs other than the training one; EMNLP 2024 outstanding paper. **[V]** — [arXiv:2406.14739](https://arxiv.org/abs/2406.14739)

### Inferences
- The most transferable design is s3/DeepRetrieval-style: a small searcher trained against a frozen reader. For this thesis the cheapest version is DeepRetrieval's: reward = "did the top-k contain the hindsight-labelled item" (recall@3), which needs no LLM call. s3's "gain over naive RAG" is the LLM-in-the-loop version and could be used sparingly for fine-tuning.
- StepSearch's success with dense information-gain rewards suggests replacing the sparse end-of-episode reward with a per-retrieval reward from hindsight labels; this attacks the PPO-variance problem noted in project memory.
- The 2406.14739 result (4M-parameter state encoder over a frozen retriever, trained from LLM feedback, transfers across readers) is the closest published analogue to a ~27k-parameter controller choosing archive items.
- None of these methods are needed in their full LLM-fine-tuning form; the controller is not an LLM and does not need to generate queries, only to score/select archive items or rewrite the query from structured fields (entity, attribute).

### Gaps
- No 2026 search-agent paper beyond Memory-R1 (ACL 2026) and the Oct 2025 survey was verified in this session; there are likely many 2026 GRPO variants (e.g., ParallelSearch, AutoRefine, IKEA) that I did not check.
- Exact algorithm for R1-Searcher (PPO vs REINFORCE++) is conflicting between my prior knowledge and the survey.

## Q2. Adaptive retrieval: when to retrieve, how much

### Takeaway
A large ACL 2025 comparison found that simple uncertainty estimates match complex adaptive-retrieval pipelines (Self-RAG, FLARE, DRAGIN, etc.) on QA and are more efficient. For this project the "when" decision is mostly settled (a question arrives, so always retrieve); the open lever is "how many / which", where learned k and a recency-aware score matter more.

### Cited Findings
- Comparison of 35 adaptive-retrieval methods (8 recent pipelines + 27 uncertainty-estimation techniques) on 6 datasets with 10 metrics: uncertainty-estimation methods often beat complex pipelines on efficiency and self-knowledge while keeping comparable QA accuracy. ACL 2025. **[V]** — [arXiv:2501.12835](https://arxiv.org/abs/2501.12835v2)
- Follow-up "LLM-Independent Adaptive RAG" uses external (non-LLM) features to decide when to retrieve. **[V title only]** — [arXiv:2505.04253](https://arxiv.org/html/2505.04253v1)
- **Self-RAG**: LM trained to emit reflection tokens (retrieve / is-relevant / is-supported / is-useful). **[PK]** — [arXiv:2310.11511](https://arxiv.org/abs/2310.11511)
- **FLARE**: retrieve when the next generated sentence contains low-probability tokens, using the tentative sentence as the query. **[PK]** — [arXiv:2305.06983](https://arxiv.org/abs/2305.06983)
- **Adaptive-RAG**: a small classifier (T5-size) predicts query complexity and routes to no-retrieval / single-step / multi-step. **[PK]** — [arXiv:2403.14403](https://arxiv.org/abs/2403.14403)
- **DRAGIN**: triggers retrieval from token uncertainty + attention-based importance and builds queries from attended tokens. **[PK]** — [arXiv:2403.10081](https://arxiv.org/abs/2403.10081)
- **SeaKR**: uses self-aware uncertainty from internal LLM states to decide retrieval and to rerank retrieved snippets by how much they reduce uncertainty. **[PK]** — [arXiv:2406.19215](https://arxiv.org/abs/2406.19215)
- LongMemEval: Llama 3.1 8B Instruct's accuracy drops sharply beyond ~3k retrieved tokens, while GPT-4o keeps improving past 20k retrieved tokens — i.e., the useful k depends on the reader. **[V]** — [arXiv:2410.10813](https://arxiv.org/html/2410.10813v2)

### Inferences
- Adaptive-RAG's small-classifier router is the right scale for this thesis: the controller could output a k (e.g., 1/3/5/10) as a discrete action, rewarded by hindsight recall minus a token cost.
- Since a 7B reader degrades with large contexts (LongMemEval's Llama-8B finding), raising k from 3 is not free; a learned k per question is preferable to a global increase. A cheap ablation: recall@k curve of the current lexical retriever for k=1..20 against hindsight labels — if the labelled item is usually at rank 4-10, a reranker or bigger k fixes most of the 300; if it is absent from top-20, the problem is the scoring function (e.g., versioning), not k.

### Gaps
- No verified paper on learning k specifically with a small policy and a token-cost penalty; this is a plausible but unsourced design.

## Q3. Training a retriever/reranker from reader feedback or hindsight labels

### Takeaway
The standard recipes are (a) REPLUG LSR: make the retriever's distribution over top-k match the reader's likelihood of the gold answer given each doc; (b) contrastive (InfoNCE) training with the gold passage as positive and in-batch/hard negatives; (c) distillation from a cross-encoder or reader. With hindsight labels already in hand, (b) needs zero LLM calls and is the obvious first step; (a) needs one forward pass per candidate and is the upgrade for "retrieved but ignored".

### Cited Findings
- **REPLUG / LSR**: treats the LM as a black box; retriever trained by minimising KL between the retrieval distribution over top-k docs and the LM's (softmaxed) likelihood of the ground-truth continuation given each doc. **[PK]** — [arXiv:2301.12652](https://arxiv.org/abs/2301.12652)
- **ARL2**: uses an LLM to annotate relevance labels and trains a retriever aligned to the LLM's preferences, with self-guided adaptive learning. **[PK]** — [arXiv:2402.13542](https://arxiv.org/abs/2402.13542)
- **Atlas** compared retriever-training losses from reader signals (attention distillation, EMDR², perplexity distillation, LOOP). **[PK]** — [arXiv:2208.03299](https://arxiv.org/abs/2208.03299)
- **LLM-Embedder**: one embedder trained from LLM feedback (reward = how much a candidate improves the LLM's output) for several retrieval needs including memory. **[PK]** — [arXiv:2310.07554](https://arxiv.org/abs/2310.07554)
- s3's reward is itself a reader-feedback signal ("gain beyond RAG"). **[V]** — [arXiv:2505.14146](https://arxiv.org/abs/2505.14146)
- Iterative retriever trained from LLM feedback with only ~4M added parameters; transfers across inference LLMs. **[V]** — [arXiv:2406.14739](https://arxiv.org/abs/2406.14739)
- Ettin rerankers were trained by pointwise-MSE distillation of a larger reranker's (mxbai-rerank-large-v2) scores — a standard cheap distillation recipe. **[V]** — [HF blog: Ettin reranker family](https://huggingface.co/blog/ettin-reranker)

### Inferences
- Recommended order for this thesis: (1) InfoNCE/listwise softmax over archive candidates with the hindsight item as positive and other versions of the same fact as hard negatives (no LLM calls); (2) if "retrieved but ignored" matters, a REPLUG-LSR-style target computed from Qwen's answer log-likelihood for a small set of candidates, cached once offline; (3) RL only for the selection/k decision.
- Hard negatives matter here: older versions of the same (X, Y) fact are lexically almost identical to the current one, which is exactly what a hashing embedder cannot separate. Training with "stale version" as hard negative directly targets the versioned-fact failure.
- Because the reward from hindsight labels is computed without the LLM, training can use millions of samples on CPU, sidestepping the ~1 s/call cost.

### Gaps
- REPLUG/ARL2/Atlas/LLM-Embedder details were not re-fetched in this session.

## Q4. Temporal / versioned-fact retrieval

### Takeaway
For "latest statement wins" data, the simplest strong baseline is a fused score: semantic/lexical similarity plus a recency term, or a hard filter keeping only the newest item per (entity, attribute) key. A 2025 paper shows a plain recency prior solves synthetic freshness retrieval perfectly. LongMemEval treats knowledge-update as a core category and shows indexing/reading-format choices move accuracy by several points.

### Cited Findings
- Recency prior: score(q,d,t) = α·cos(q,d) + (1−α)·0.5^(age_days/h), defaults α=0.7, h=14 days; achieves Latest@10 = 1.00 and as-of correctness = 1.00 on synthetic data and Latest-Set@10 = 1.00 on a real dataset (CERT); its heuristic trend detection fails (F1 0.08). **[V]** — [arXiv:2509.19376](https://arxiv.org/html/2509.19376v1)
- **VersionRAG** models document evolution as a hierarchical version graph; 90% accuracy on version-sensitive queries vs 58% standard RAG and 64% GraphRAG. **[V via search snippet]** — [arXiv:2510.08109](https://arxiv.org/html/2510.08109v1)
- Re3 ("Relevance Recency Retrieval") uses a time-aware dual relevance encoder and a "Conflict-Aware Recency Filter" that suppresses obsolete versions of a fact. **[V via search snippet only; arXiv ID not confirmed — the search returned arXiv:2510.13590 nearby but I did not confirm it is Re3]** — [arXiv:2510.13590 (unconfirmed match)](https://arxiv.org/abs/2510.13590v1)
- Temporal QA survey covering TimeRAG, MRAG, etc. **[V]** — [arXiv:2505.20243](https://arxiv.org/pdf/2505.20243)
- **LongMemEval** (ICLR 2025): five abilities including knowledge updates; memory design split into indexing, retrieval, reading; "round" granularity beats "session"; fact-augmented key expansion improves recall@k by 9.4% and QA accuracy by 5.4%; time-aware query expansion improves temporal-reasoning recall by 6.8-11.3% (with a strong LLM doing the expansion); long-context and commercial assistants drop ~30% accuracy over sustained interactions. **[V]** — [arXiv:2410.10813](https://arxiv.org/abs/2410.10813)
- LongMemEval does not report a separate knowledge-update breakdown in the parts I read (first ~100k characters). **[V]** — [arXiv:2410.10813](https://arxiv.org/html/2410.10813v2)
- Facts with internal (parametric) conflicts — often outdated/evolving facts — are harder to update with context. **[V via snippet]** — [DynamicQA, arXiv:2407.17023](https://arxiv.org/abs/2407.17023)

### Inferences
- The project's task has a structured key (entity Y, attribute X, step N). A deterministic "latest version per key" filter, or adding step N as a feature in the retrieval score, should eliminate the share of the 300 failures caused by retrieving a stale version instead of the current one. This is a cheap confound check before any learned retriever: if a key+recency retriever closes most of the gap to the oracle, the controller's learned part is not what's missing.
- Analogue of fact-augmented key expansion: index each archived item under its parsed (Y, X) key in addition to its text.

### Gaps
- No verified result on how often 7B readers pick the older version when both versions are in context; this would directly explain part of the "retrieved but ignored" bucket. Measuring it on the project data is cheap.

## Q5. Context ordering / "lost in the middle" and "retrieved but ignored"

### Takeaway
LLMs have a U-shaped positional attention bias, so evidence in the middle of the context is used less. Fixes include placing the most relevant item at the start or end, calibrating attention, and structured reading formats (JSON + Chain-of-Note), each worth up to ~10-15 points in published settings.

### Cited Findings
- U-shaped attention bias: tokens at the beginning and end get more attention regardless of relevance; calibration ("found-in-the-middle") improves RAG by up to 15 points; ACL Findings 2024. **[V]** — [arXiv:2406.16008](https://arxiv.org/abs/2406.16008v2)
- Original "Lost in the Middle": accuracy highest when the relevant document is at the beginning or end of the context and drops when it is in the middle. **[PK]** — [arXiv:2307.03172](https://arxiv.org/abs/2307.03172)
- LongMemEval: Chain-of-Note + structured JSON formatting of retrieved memories improves QA by up to 10 absolute points across three LLMs, including with oracle retrieval. **[V]** — [arXiv:2410.10813](https://arxiv.org/html/2410.10813v2)
- Other position fixes: attention sorting (re-order by attention, then regenerate) **[PK]** [arXiv:2310.01427](https://arxiv.org/abs/2310.01427); Ms-PoE multi-scale positional encoding **[PK]** [arXiv:2403.04797](https://arxiv.org/abs/2403.04797); LongLLMLingua document reordering + compression **[PK]** [arXiv:2310.06839](https://arxiv.org/abs/2310.06839).

### Inferences
- For ~20 ignored cases with top-3 retrieval, positional bias is probably minor (only 3 items). More likely causes: the item is placed far from the question, or a stale version also appears and the reader picks it. Cheap fixes: put retrieved items immediately before the question, sort retrieved versions by step ascending so the newest is last (nearest the question), and label it explicitly (e.g., "[LATEST]" / JSON with step field).

### Gaps
- No verified study of position effects with very few (≤3) retrieved items.

## Q6. Small/cheap rerankers and embedders for CPU

### Takeaway
Viable CPU options span 17M-300M parameters: Ettin cross-encoder rerankers (17M/32M/68M...), bge-small-en (33M), and EmbeddingGemma-300M (best open multilingual model <500M on MTEB, <200MB RAM quantized, Matryoshka dims down to 128).

### Cited Findings
- **Ettin reranker family**: six Sentence-Transformers CrossEncoder rerankers at 17M, 32M, 68M, 150M, 400M, 1B, built on Ettin ModernBERT encoders; distilled from mxbai-rerank-large-v2 scores; HF ids like `cross-encoder/ettin-reranker-17m-v1`. **[V]** — [HF blog](https://huggingface.co/blog/ettin-reranker)
- Reranker scaling/training-strategy studies (2026). **[V titles only]** — [Scaling Laws for Cross-Encoder Reranking, arXiv:2603.04816](https://arxiv.org/pdf/2603.04816); [Encoder-only cross-encoder training strategies, arXiv:2603.03010](https://arxiv.org/pdf/2603.03010)
- **EmbeddingGemma**: 308M parameters, highest-ranking open multilingual embedder under 500M on MTEB, <200MB RAM quantized, Matryoshka output (768/512/256/128). **[V]** — [Google Developers Blog](https://developers.googleblog.com/introducing-embeddinggemma/); [HF blog](https://huggingface.co/blog/embeddinggemma). Paper: [arXiv:2509.20354](https://arxiv.org/abs/2509.20354) **[PK ID]**
- **bge-small-en**: 33.4M parameters. **[V, secondary source]** — [PromptLayer model page](https://www.promptlayer.com/models/bge-small-en)
- Other small options from prior knowledge: all-MiniLM-L6-v2 (22M), cross-encoder/ms-marco-MiniLM-L-6-v2, Qwen3-Embedding-0.6B [arXiv:2506.05176](https://arxiv.org/abs/2506.05176), Model2Vec static embeddings (no transformer at inference). **[PK]**

### Inferences
- For synthetic templated sentences, a pretrained embedder may not beat lexical matching on the entity/attribute tokens; the real problem is version discrimination, which none of these models know about. A fine-tuned 17M-33M model with stale-version hard negatives, or a tiny learned scorer over [lexical score, key match, recency rank] features, is likely enough and keeps the "small controller" story intact.
- Keep the controller at ~27k parameters and treat the reranker as a fixed tool; or train a small feature-based reranker (logistic/MLP over a handful of features) from hindsight labels — still tiny and CPU-trivial.

### Gaps
- No verified CPU latency numbers per model were found in this session.
