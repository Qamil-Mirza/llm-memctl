# Retrieval methods and learned retrieve/no-retrieve decisions for a budgeted conversational memory controller

Research date: 2026-10-06. Scope: candidate generation (what gets onto the shortlist) and gating (whether/which shortlist items get pulled into the active context), for a small policy that picks from a BM25 top-8 shortlist with independent Bernoulli decisions and, on LoCoMo/LongMemEval, almost never retrieves (retrieval recall 0.00-0.03), losing to an "always fetch BM25 top-5" heuristic.

## Q1. How do BM25, dense, hybrid and rerankers compare on conversational memory, and which embedders are practical on CPU?

### Takeaway
On LongMemEval, a dense retriever beats BM25 by roughly 9 points of session Recall@5, and indexing each turn together with extracted facts helps more than swapping retrievers. On LoCoMo, BM25 + dense score fusion beats either alone by about 11 points Hit@1, but on LongMemEval-S BM25 is already near the ceiling, and in one study a cross-encoder reranker *hurt*. The candidate generator is probably not why the policy's recall is near zero: BM25 top-k already finds most evidence, so the failure is on the gating side.

### Cited Findings
- LongMemEval (Wu et al., ICLR 2025) tests BM25, Contriever and Stella V5 1.5B over chat history. Dense beats sparse. Session-granularity Recall@5 is 0.720 for Stella vs 0.634 for BM25. — [LongMemEval, arXiv 2410.10813, App. E.2](https://arxiv.org/html/2410.10813)
- LongMemEval key expansion: index each round under "value + extracted user fact". At round granularity with Stella, Recall@5 goes 0.582 -> 0.644 and Recall@10 0.692 -> 0.784. Averaged across readers, this gives +9.4% recall@k and +5.4% final QA accuracy. — [LongMemEval §5, Table 3](https://arxiv.org/html/2410.10813)
- LongMemEval key variants: fact keys give a consistent gain (Stella, session values: Recall@5 0.732 vs 0.720). Summary keys helped Contriever (Recall@5 0.732, session values). Keyphrase keys sometimes hurt. — [LongMemEval App. E.2](https://arxiv.org/html/2410.10813)
- LongMemEval value granularity: splitting sessions into rounds improves GPT-4o reading, and fact decomposition helps multi-session questions. The authors skipped embedders larger than about 1.5B because of latency. — [LongMemEval](https://arxiv.org/html/2410.10813)
- LoCoMo original paper (Maharana et al., ACL 2024), RAG with a DRAGON retriever and a gpt-3.5-turbo-16k reader. The retrieval unit matters more than k:
  - Retrieving "observations" (LLM-extracted assertions about each speaker) gives overall F1 41.4 at top-5, falling to 37.8 at top-50.
  - Retrieving raw dialog turns gives overall F1 31.7 at top-5 and 34.8 at top-50.
  - Session summaries give 29.9-32.5.
  - Temporal F1 is 41.9 with observations vs 21.3-26.2 with dialog turns.
  - Multi-hop F1 with dialog turns rises 19.4 -> 37.2 as k grows from 5 to 50.
  — [LoCoMo, arXiv 2402.17753, Table 3](https://arxiv.org/html/2402.17753)
- "Training-Free Lexical-Dense Fusion for Conversational-Memory Retrieval" (arXiv 2606.04194):
  - Score-level fusion of BM25 with a dense late-interaction score (one weight, tuned leave-one-conversation-out) adds +8.8 to +17.2 points of LoCoMo Hit@1 over the dense score alone, across six encoders (p<1e-4).
  - Best result: Hit@1 0.752 / NDCG@5 0.829 with e5-large-v2, which is +11.2 pp over BM25.
  - On LongMemEval-S, BM25 "saturates" and the fusion gain over BM25 is small and not significant.
  - Adding a cross-encoder reranker *degraded* Hit@1 by 6.9 pp.
  - The whole pipeline runs CPU-only.
  — [arXiv 2606.04194](https://arxiv.org/abs/2606.04194)
- A search-engine summary of the same work says dense late interaction helps most on multi-hop and temporal LoCoMo questions and trails BM25 on adversarial ones. I could not check this against the paper text. — [EmergentMind summary of 2606.04194](https://www.emergentmind.com/papers/2606.04194)
- Another LoCoMo study (search snippet only, not read in full) reports hybrid BM25+dense vs vector-only evidence recall of about 90.2% vs 90.3%, a statistical tie. — [arXiv 2606.04194 search result / LoCoMo topic page](https://www.emergentmind.com/topics/locomo)
- A search snippet attributes to some LoCoMo analysis that raw retrieval recall was 98.6%, but only 22.5% of gold evidence survived truncation to the token budget without good ranking. I could not trace it to a primary source. The MEMTIER abstract does not contain it. — [search result listing incl. MEMTIER, arXiv 2605.03675](https://arxiv.org/abs/2605.03675)
- MTEB scores for small embedders, from the BGE model card (Average / Retrieval / dim):

  | Model | Average | Retrieval | Dim | Params |
  |---|---|---|---|---|
  | bge-small-en-v1.5 | 62.17 | 51.68 | 384 | 33.4M |
  | bge-base-en-v1.5 | 63.55 | 53.25 | 768 | |
  | bge-large-en-v1.5 | 64.23 | 54.29 | 1024 | |
  | gte-small | 61.36 | 49.46 | 384 | |
  | gte-base | 62.39 | 51.14 | 768 | |
  | e5-small-v2 | 59.93 | 49.04 | 384 | |
  | e5-base-v2 | 61.50 | 50.29 | 768 | |
  | e5-large-v2 | 62.25 | 50.56 | 1024 | |
  | all-mpnet-base-v2 | 57.78 | 43.81 | 768 | |

  The same card lists bge-reranker-base/large cross-encoders (reranking average about 67.3/67.6) as "more accurate but less efficient". — [BAAI/bge-small-en-v1.5 model card](https://huggingface.co/BAAI/bge-small-en-v1.5)
- One secondary source (not primary) gives BGE-small CPU throughput of roughly 5-50 ms per 256-token text, or 50-500 embeddings/s on a modern CPU. — [search summary, generalcompute blog](https://www.generalcompute.com/blog/embedding-model-benchmarks-speed-vs-quality-for-rag-retrieval?agent=true) (low confidence)

### Inferences
- For our controller the bottleneck is almost certainly gating, not candidate generation. LongMemEval's BM25 already gets 0.63 session Recall@5, and the fusion paper calls BM25 "saturated" on LongMemEval-S. So a policy with retrieval recall of 0.00-0.03 is throwing away a shortlist that usually contains the evidence.
- The cheapest candidate-side improvements, in rough order:
  1. **Index unit / key expansion.** Index turns with an extracted fact or observation line. On LoCoMo, observation RAG beat dialog RAG by about 10 F1 points at top-5. In LongMemEval, fact keys gave +9.4% recall. This can be done once, offline, with the rented 7B model.
  2. **BM25 + bge-small (33M params, 384-d) score fusion or RRF.** CPU-feasible. Worth about 10 Hit@1 on LoCoMo, roughly nothing on LongMemEval-S.
  3. **Time-aware filtering** for temporal questions.
- Skip cross-encoder reranking as a first move. The one directly relevant study found it hurt (-6.9 Hit@1), and it costs the most CPU time.
- LoCoMo F1 *falls* with observation k (41.4 at top-5 -> 37.8 at top-50) but *rises* with dialog k for multi-hop questions. This supports a fixed small k of high-precision units, which fits a budgeted context.

### Gaps
- I did not find a single table of recall@k on LoCoMo covering Contriever / bge / e5 / gte / Stella side by side. The fusion paper uses six encoders, but I only saw its abstract-level numbers.
- I found no primary CPU-latency benchmark for bge-small, e5-small or gte-small. The figures above come from secondary sources.
- The "98.6% recall but 22.5% survives truncation" claim could not be traced to a primary source.
- There are hybrid-RRF numbers specifically on LongMemEval beyond "fusion gain not significant", but I did not find them.

## Q2. Query rewriting/expansion and iterative multi-hop retrieval

### Takeaway
Time-aware query expansion is the one memory-specific rewrite with published gains: about +45% relative Recall@10 on LongMemEval temporal questions. General methods (HyDE, IRCoT) give large gains in open-domain settings, but each needs an LLM call per query or hop. That fits the A40 reader, not the laptop policy loop. Our existing two-hop "follow-the-clue" BM25 is a cheap IRCoT-style approximation.

### Cited Findings
- LongMemEval time-aware query expansion: an LLM extracts a time range from the question, and candidate rounds are filtered by timestamp. On temporal-reasoning questions with GPT-4o as extractor, Recall@5 goes 0.421 -> 0.451 and Recall@10 0.499 -> 0.722. The average recall gain is +11.3% with round values. — [LongMemEval Table 4](https://arxiv.org/html/2410.10813)
- IRCoT (Trivedi et al., ACL 2023) alternates chain-of-thought steps with retrieval, using each CoT sentence as the next query. The abstract reports retrieval gains of up to 21 points and QA gains of up to 15 points on HotpotQA, 2Wiki, MuSiQue and IIRC. — [IRCoT, arXiv 2212.10509](https://arxiv.org/abs/2212.10509v2)
- A secondary summary reports IRCoT on HotpotQA (GPT-3) at +11.3 recall over one-step retrieval and QA F1 60.7 vs 53.6. — [beancount.io research log (secondary)](https://beancount.io/bean-labs/research-logs/2026/05/19/ircot-interleaving-retrieval-chain-of-thought-multi-step-qa)
- HyDE (Gao et al., 2022/ACL 2023) embeds an LLM-written hypothetical answer document instead of the query. With unsupervised Contriever on TREC DL19 it reaches nDCG@10 61.3 and mAP 41.8, about matching fine-tuned Contriever (62.1 / 41.7), with better Recall@1k (88.0 vs 83.6). — [HyDE, arXiv 2212.10496](https://arxiv.org/pdf/2212.10496)
- Moskvoretskii et al. (ACL 2025): on multi-hop QA, iterative or multi-call retrieval (DRAGIN) beats single always-retrieve. 2Wiki In-Accuracy is 0.456 for DRAGIN (2.92 retrieval calls per question) vs 0.374 for always-retrieve once. HotpotQA is 0.430 (2.56 calls) vs 0.410. — [arXiv 2501.12835](https://arxiv.org/html/2501.12835)

### Inferences
- Multi-hop gains in the literature come from *more* retrieval rounds, not fewer. That is the opposite of what our policy learned on real conversations.
- Time-aware filtering suits a controller well because it can run without an LLM: parse dates with rules, then filter candidates by session timestamp. This is an inference, untested.
- HyDE-style rewriting needs a generator. With the 7B reader on the A40 it could make the shortlist query once per question. It is unclear whether it beats plain BM25 on conversational text, since no conversational-memory HyDE numbers were found.

### Gaps
- I found no published HyDE or LLM-decomposition results on LoCoMo or LongMemEval specifically.
- I did not get IRCoT's per-dataset recall table from the primary source. Only the abstract claim is verified.

## Q3. Adaptive retrieval / gating: do learned gates beat always-retrieve?

### Takeaway
In the largest comparison (35 methods, 6 QA datasets), adaptive gates roughly *match* always-retrieve on accuracy and win mainly on cost. On multi-hop data the best gates retrieve on about 99% of questions anyway. Always-retrieve wins or ties whenever retrieval is cheap and the needed knowledge is not in the model's weights. That is exactly the conversational-memory case: the answer is never in the weights.

### Cited Findings
- Moskvoretskii et al., "Adaptive Retrieval Without Self-Knowledge? Bringing Uncertainty Back Home" (ACL 2025). They compare 35 methods: 8 adaptive-RAG pipelines (e.g., FLARE, DRAGIN, Rowen, SeaKR, Adaptive-RAG, IRCoT) and 27 uncertainty estimators, on 6 datasets with 10 metrics. Conclusion: simple uncertainty estimators often beat complex pipelines on efficiency and self-knowledge "while maintaining comparable QA performance". — [arXiv 2501.12835](https://arxiv.org/abs/2501.12835)
- Selected In-Accuracy (retrieval calls per question) from that paper:

  | Dataset | Always-retrieve | Never-retrieve | Best uncertainty gate | Pipeline |
  |---|---|---|---|---|
  | NQ | 0.496 (1.00) | 0.446 | Lex-Similarity 0.512 (0.58) | DRAGIN 0.480 (2.24) |
  | SQuAD | 0.312 | 0.176 | Lex-Similarity 0.318 (0.96) | Adaptive-RAG 0.286 (0.97) |
  | TriviaQA | 0.610 | 0.636 | Lex-Similarity 0.646 (0.22) | DRAGIN 0.666 (2.06) |
  | HotpotQA | 0.410 | | Max Entropy 0.414 (0.99) | |
  | 2Wiki | 0.374 | | 0.384 (0.98) | |
  | MuSiQue | 0.100 | | 0.104 (0.99) | |

  Never-retrieve beats always-retrieve only on TriviaQA (0.636 vs 0.610), which is popular knowledge already in the model's weights. On SQuAD, gating gives no savings: the gate keeps 96% of retrievals. — [arXiv 2501.12835](https://arxiv.org/html/2501.12835)
- RetrievalQA (2024) is a 1,271-question benchmark built so that the needed knowledge is *absent* from the LLM. Its authors found calibration-based gates depend heavily on threshold tuning, and vanilla prompting does not reliably guide retrieval decisions. — [RetrievalQA, arXiv 2402.16457](https://arxiv.org/html/2402.16457v1)
- A 2026 study of adaptive retrieval for reasoning found that on GSM8K retrieval was triggered on only 7% of problems and rarely helped when CoT was already right (helped 5.3%, hurt 1.5%). On MATH-500, retrieval was used 38.8% of the time and helped and hurt equally (25 vs 25). — [arXiv 2602.07213](https://arxiv.org/abs/2602.07213)
- Primary references for the named gating methods, which I did not re-read for numbers in this pass:
  - Self-RAG: reflection tokens decide whether to retrieve. [arXiv 2310.11511](https://arxiv.org/abs/2310.11511)
  - FLARE: retrieve when generated tokens are low-confidence. [arXiv 2305.06983](https://arxiv.org/abs/2305.06983)
  - DRAGIN: entropy/attention-triggered retrieval. [arXiv 2403.10081](https://arxiv.org/abs/2403.10081)
  - Adaptive-RAG: a small classifier routes by query complexity. [arXiv 2403.14403](https://arxiv.org/abs/2403.14403)
  - SKR: self-knowledge-guided retrieval. [arXiv 2310.05002](https://arxiv.org/abs/2310.05002)
- Memory-specific evidence that a learned filter can help *after* always retrieving a large pool: Memory-R1 (ACL 2026) always retrieves 60 candidate memories per question, then an RL-trained Answer Agent "distills" (filters) them. Without distillation, LoCoMo scores were F1 34.37 / BLEU-1 40.95 / Judge 60.14; with distillation, 37.51 / 45.02 / 62.74. The ablation text appears to swap F1 and BLEU-1 relative to the paper's main table (main: F1 45.02, BLEU-1 37.51), so treat the metric labels with care. Both agents were trained with PPO/GRPO on only 152 LoCoMo QA pairs, with an exact-match reward, on LLaMA-3.1-8B and Qwen-2.5-3B/7B/14B. Memory-R1-GRPO (LLaMA-3.1-8B) vs Mem0 on LoCoMo: F1 45.02 vs 30.41, Judge 62.74 vs 45.68. — [Memory-R1, arXiv 2508.19828v4](https://arxiv.org/html/2508.19828v4)

### Inferences
- In conversational memory the knowledge is never in the weights, so the "never retrieve" branch that gating exploits (TriviaQA-like cases) mostly does not exist. The literature predicts always-retrieve top-k should be a hard baseline to beat, which matches what we see.
- The successful learned design in this space (Memory-R1) is **retrieve-wide, then filter/select**, not "decide whether to retrieve at all". Recasting our policy as "always bring in top-k, learn which k (or which to drop)" matches the evidence better than a Bernoulli gate starting from "retrieve nothing".
- Under a tight budget, the gate's real job is choosing *which* items fill limited slots, not *whether* to retrieve. Uncertainty gates only save calls, and retrieval calls cost us almost nothing (BM25 over a small archive).

### Gaps
- I found no published "when to retrieve" gate evaluated on LoCoMo or LongMemEval against always-retrieve. Gating results are all from open-domain QA.
- I did not extract Self-RAG, FLARE or SKR headline numbers from their primary papers in this pass.

## Q4. Learning-to-rank / retrievers trained from downstream reward or reader distillation

### Takeaway
REPLUG LSR is the cheapest pattern: use the reader's likelihood of the gold answer given each candidate as a soft target, and train the selector with KL. It gave +5-6% downstream in its paper. For us this would replace the sparse QA reward with a dense per-candidate label computed by the 7B reader on the A40.

### Cited Findings
- REPLUG (Shi et al., NAACL 2024). REPLUG LSR trains an off-the-shelf retriever using the black-box LM's output scores as supervision: it minimises KL between the retrieval distribution over top-k documents and the LM's softmax over per-document answer likelihoods. The tuned retriever improves GPT-3 (175B) language modelling by 6.3% and Codex five-shot MMLU by 5.1%. — [REPLUG, arXiv 2301.12652](https://arxiv.org/pdf/2301.12652)
- Memory-R1 trains its memory-filtering agent end-to-end with RL from an exact-match reward and reaches large LoCoMo gains with 152 training questions (numbers in Q3). This is evidence that answer-level reward can teach selection in conversational memory, at 3B-14B LLM scale. — [Memory-R1](https://arxiv.org/html/2508.19828v4)

### Inferences
- REPLUG-style labels suit our setup. For each question, score each of the 8 shortlist items by log p(gold answer | item, question) under the Qwen2.5-7B reader. That is 8 forward passes per question, with no generation, batchable on the A40. Then train the small policy with a KL or listwise loss to that distribution. This turns a near-zero-signal RL problem into supervised distillation. Offline labelling is a one-time cost.
- A cheaper proxy that needs no reader: use the gold evidence turn IDs that LoCoMo and LongMemEval ship as labels. LongMemEval's own recall metrics use them, so they are a supervised listwise target with no GPU.

### Gaps
- I did not verify ATLAS attention-distillation numbers, or RL-trained reranker results on conversational memory, in this pass.
- I found no cost or gain numbers for REPLUG-LSR-style distillation on conversational memory benchmarks.

## Q5. Extreme label imbalance and set selection instead of independent Bernoulli

### Takeaway
With independent Bernoulli outputs, a rare positive class and a cost on retrieval, the policy can collapse to "never retrieve". Standard fixes: (a) make the action "pick exactly k" (Gumbel-top-k / Plackett-Luce sampling without replacement), so the policy cannot output nothing; (b) use listwise or softmax losses over the shortlist, which are normalized across candidates and therefore unaffected by the positive/negative ratio; (c) if Bernoulli is kept, reweight positives or use focal loss and initialise the bias at the always-retrieve prior.

### Cited Findings
- Gumbel-top-k: perturb logits with Gumbel noise and take the top k. This gives an exact sample of k items without replacement from the softmax (Plackett-Luce), and the sample's log-probability is available for policy gradients. — [Kool, van Hoof & Welling, "Stochastic Beams and Where to Find Them", ICML 2019, arXiv 1903.06059](https://arxiv.org/abs/1903.06059); tutorial: [UvA DL notebooks, "Sampling subsets with Gumbel-Top-k relaxations"](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/DL2/sampling/subsets.html)
- A continuous relaxation of this subset sampling (a sequence of softmaxes) makes k-subset selection differentiable. — [Xie & Ermon, "Reparameterizable Subset Sampling via Continuous Relaxations", IJCAI 2019, arXiv 1901.10517](https://arxiv.org/abs/1901.10517)
- Lower-variance REINFORCE for samples drawn without replacement: the unordered set estimator with a built-in baseline. — [Kool et al., ICLR 2020, arXiv 2002.06043](https://arxiv.org/abs/2002.06043)
- Hard top-k "kills the gradient", so training usually falls back on straight-through tricks or noisy REINFORCE. Gumbel-top-k relaxations avoid this. — [T. Ahle, "Differentiable top-k" blog](https://thomasahle.com/blog/differentiable_topk.html) (secondary)
- Focal loss down-weights easy, mostly-negative examples by (1-p)^gamma, with an alpha-balance term for the rare class. The paper also recommends initialising the final bias to the positive prior (pi=0.01) for stability under extreme imbalance. — [Lin et al., "Focal Loss for Dense Object Detection", arXiv 1708.02002](https://arxiv.org/abs/1708.02002) (2017, foundational)
- Efficient policy-gradient training of Plackett-Luce ranking policies (PL-Rank). — [Oosterhuis, SIGIR 2021, arXiv 2105.00855](https://arxiv.org/abs/2105.00855)

### Inferences
- The most direct fix for the "never retrieves" failure: change the retrieval action from 8 independent Bernoullis to "sample k of 8 via Plackett-Luce / Gumbel-top-k", with k fixed by the budget or picked from a small set {0, 2, 5}. The always-top-5 heuristic is then just one point in the policy space (the BM25-rank ordering), and the policy can only fail by choosing a *worse* 5, not by retrieving nothing.
- Pair this with a listwise warm-start. First do supervised training with softmax cross-entropy against gold evidence IDs (or REPLUG-style reader scores, Q4). Then RL-fine-tune with the without-replacement REINFORCE estimator (Kool 2020) to cut PPO variance, which our notes identify as the binding constraint on scripted tasks.
- If the Bernoulli head is kept, it should at least (i) initialise logits at BM25-rank-based priors so the starting policy equals "top-5", and (ii) use positive reweighting or focal loss on any supervised auxiliary loss.

### Gaps
- I found no papers that apply these set-selection estimators specifically to memory-controller retrieval in LLM agents. The recommendation is an inference from general ranking/RL results.
- I have no empirical comparison of focal loss vs listwise loss for retrieval gating under the ~1-3% positive rate we see.
