# Top memory systems on LoCoMo and LongMemEval: scores and the design choices behind them

Conventions: "J" = LLM-as-judge accuracy (%), F1 = token F1. "SR" = self-reported by the system's authors; "3P" = run by a third party (a competitor or an independent paper). Unless noted, LoCoMo numbers exclude the adversarial category (category 5) and cover about 1,540 questions. LongMemEval numbers are on LongMemEval_S (500 questions, about 115k tokens of history per question) unless noted.

## Q1. Which systems report results, and what do they score?

### Takeaway
Scores fall into three tiers. (a) The 2025 "memory layer" generation (Mem0, Zep, LangMem, A-Mem, MemoryOS) used a GPT-4o-mini or GPT-4o reader and scored about 58–75 J on LoCoMo and 64–71 on LongMemEval. (b) Late-2025 retrieval-heavy systems (EmergenceMem, Hindsight, Supermemory, LightMem) scored 82–91 on LongMemEval and 83–90 on LoCoMo. (c) 2026 vendor claims (Mastra, Mem0-2026, ByteRover, Agent Zero) score 92–96. Almost every number in tiers (b) and (c) is self-reported, and the judge model changes from paper to paper.

### Cited Findings

**LoCoMo (J, adversarial excluded)**
- Mem0 paper (GPT-4o-mini reader and judge; all baselines run by Mem0, so they are 3P for the other systems). Overall J: full-context 72.90; Mem0g 68.44; Mem0 66.88; Zep 65.99; best RAG 60.97 (256-token chunks, k=2); OpenAI memory 52.90; LangMem 58.10; A-Mem 48.38 — [Mem0 paper](https://arxiv.org/html/2504.19413)
- Mem0 paper, per-category J (single-hop / multi-hop / open-domain / temporal): Mem0 67.13/51.15/72.93/55.51; Mem0g 65.71/47.19/75.71/58.13; Zep 61.70/41.35/76.60/49.31; LangMem 62.23/47.92/71.12/23.43; OpenAI 63.79/42.92/62.29/21.71; A-Mem* 39.79/18.85/54.05/49.91 — [Mem0 paper](https://arxiv.org/html/2504.19413)
- Zep re-ran its own system: 75.14 ± 0.17 J (SR rebuttal), against the 65.99 Mem0 reported for Zep — [Zep blog](https://www.getzep.com/blog/lies-damn-lies-statistics-is-mem0-really-sota-in-agent-memory/)
- Letta's filesystem agent (GPT-4o-mini; tools grep / search_files / open) scored 74.0 (SR), against Mem0's 68.5 — [Letta blog](https://www.letta.com/blog/benchmarking-ai-agent-memory)
- Hindsight (SR; judge GPT-OSS-120B): OSS-20B 83.18, OSS-120B 85.67, Gemini-3 89.61 (single-hop 86.17, multi-hop 70.83, open-domain 95.12, temporal 83.80). Prior systems as listed by Hindsight: Memobase 75.78, Zep 75.14, Backboard 90.00 (claimed) — [Hindsight paper](https://arxiv.org/html/2512.12818v1)
- LightMem (ICLR 2026) re-ran all baselines under the same retrieval setup (3P for the baselines; judge GPT-4o-mini). GPT-4o-mini backbone: FullText 71.83, NaiveRAG 63.64, A-Mem 64.16, Mem0 61.69, MemoryOS 58.25, LangMem 57.20, LightMem 70.26–72.99. Qwen3-30B-A3B backbone: FullText 74.87, NaiveRAG 66.95, LangMem 60.53, MemoryOS 61.04, A-Mem 56.10, Mem0 43.31, LightMem 71.36–72.60 — [LightMem](https://arxiv.org/abs/2510.18866)
- 2026 claims (all SR): Agent Zero 93.60; Mem0 (2026 algorithm) 92.50; ByteRover 2.0 92.20; Hindsight 89.60; Memobase 75.80; Zep 75.10 — [Agent Zero Memory](https://arxiv.org/abs/2608.29606). Mem0's own 2026 leaderboard lists ZeroMemory 96.1 (not verified), Zep 94.7 (SR; third-party tests give 75.1), Mem0 92.5, Dakera 88.2 — [Mem0 2026 benchmark guide](https://mem0.ai/blog/ai-memory-benchmarks-in-2026)

**LongMemEval_S (accuracy, GPT-4o judge unless noted)**
- Paper baselines, long-context readers (oracle sessions → full ~115k history): GPT-4o 0.870 → 0.606; Llama-3.1-70B 0.744 → 0.334; Llama-3.1-8B 0.710 → 0.454; Phi-3-128k 0.702 → 0.380; Phi-3.5-mini 0.660 → 0.342 — [LongMemEval paper](https://arxiv.org/html/2410.10813)
- Commercial assistants on a simplified setting: ChatGPT (GPT-4o) 0.577 and Coze 0.330, against 0.918 offline reading — [LongMemEval paper](https://arxiv.org/html/2410.10813)
- Zep (SR): GPT-4o-mini 63.8 against 55.4 full context; GPT-4o 71.2 against 60.2. Context drops from 115k to 1.6k tokens. Per type (4o-mini, full context → Zep): preference 30.0 → 53.3; single-session-user 81.4 → 92.9; temporal 36.5 → 54.1; multi-session 40.6 → 47.4; single-session-assistant 81.8 → 75.0; knowledge-update 76.9 → 74.4 — [Zep paper](https://arxiv.org/html/2501.13956)
- EmergenceMem (SR; GPT-4o reader): Simple 82.4 (7.12 s/item), Simple Fast 79.0, Internal 86.0. Baselines they report: full-context GPT-4o 63.8, naive RAG 52.0, oracle GPT-4o 82.4 — [Emergence blog](https://www.emergence.ai/blog/sota-on-longmemeval-with-rag)
- Hindsight (SR; judge GPT-OSS-120B): OSS-20B 83.6, OSS-120B 89.0, Gemini-3 91.4. Full context with OSS-20B scores 39.0. Supermemory scores GPT-4o 81.6, GPT-5 84.6, Gemini-3 85.2. Hindsight OSS-120B per type: single-session-user 100, single-session-assistant 98.2, preference 86.7, knowledge-update 92.3, temporal 85.7, multi-session 81.2 — [Hindsight paper](https://arxiv.org/html/2512.12818v1)
- LightMem (3P baselines; judge GPT-4o-mini). GPT-4o-mini backbone: FullText 56.80, NaiveRAG 61.00, A-Mem 62.60, Mem0 53.61, MemoryOS 44.80, LangMem 37.20, LightMem 64.29–68.64. Qwen3-30B-A3B backbone: FullText 54.80, NaiveRAG 60.80, A-Mem 65.20, Mem0 39.51, LightMem up to 70.20. GLM-4.6 backbone: FullText 36.71, NaiveRAG 73.20, LightMem 73.20 — [LightMem](https://arxiv.org/abs/2510.18866)
- 2026 claims (SR): Agent Zero 95.60 (gpt-5.5; eight backbones all fall between 92.2 and 95.6), Mastra observational memory 94.87 (GPT-5-mini), Hindsight 91.40, EmergenceMem 86.00, Supermemory 85.20, Zep 71.20 — [Agent Zero Memory](https://arxiv.org/abs/2608.29606). Mem0 2026: 94.4 (SR); BEAM-1M 64.1, BEAM-10M 48.6 — [Mem0 2026 guide](https://mem0.ai/blog/ai-memory-benchmarks-in-2026)

### Inferences
- Only one 2025 comparison was run under a single harness for many systems: the LightMem table. In it, plain NaiveRAG beats Mem0, MemoryOS and LangMem on both benchmarks and with every backbone. Most "memory layer" results on the vendor leaderboards cannot be compared with each other.
- From the 2026 claims onward, the reader models are frontier LLMs (GPT-5.x, Gemini-3). They are not comparable to our Qwen2.5-7B setting.

### Gaps
- MemGPT/Letta (except the filesystem blog post), MemOS, MIRIX, Nemori, Memobase and Supermemory: I did not fetch their primary papers or pages within the budget. Their numbers above come only from secondary tables (Hindsight, Agent Zero).
- I could not access the Agent Zero PDF's design section in detail. I only read its tables and its ablation.

## Q2. What design choices does each system use?

### Takeaway
Every top system retrieves on every query, with no gate on whether to retrieve. They index fine-grained units (rounds or extracted facts) but often return larger context (whole sessions or narrative facts). They use hybrid retrieval (dense + BM25, often + graph or temporal channels), fuse the results (RRF), rerank with a cross-encoder, and attach timestamps to each memory.

### Cited Findings
- **Mem0**: an LLM extracts facts per message pair, using the 10 previous messages as context. Each new fact is compared with the 10 most similar memories, and an LLM tool call chooses ADD / UPDATE / DELETE / NOOP. Memory is about 7k tokens per conversation (Mem0g about 14k; Zep over 600k; the raw conversation about 26k). Search p50 is 0.148 s — [Mem0 paper](https://arxiv.org/html/2504.19413)
- **Mem0 2026 algorithm**: extraction is single-pass and ADD-only (the UPDATE/DELETE path was dropped). Facts generated by the agent are stored too. Entities are linked. Retrieval fuses semantic, BM25 and entity signals. About 6.7–7.0k tokens per retrieval — [Mem0 2026 guide](https://mem0.ai/blog/ai-memory-benchmarks-in-2026)
- **Zep/Graphiti**: three subgraphs (episodes = raw messages; semantic entities and edges; communities). Edges are bi-temporal (event time and transaction time). Search is hybrid: BM25 + cosine + breadth-first graph traversal. Rerankers available: RRF, MMR, episode-mention frequency, node distance, cross-encoder. It retrieves the top-10 edges and the top-10 entity nodes — [Zep paper](https://arxiv.org/html/2501.13956)
- **EmergenceMem Simple**: matches the query against individual turns, reranks those turns with a cross-encoder, scores each session by the NDCG of its turns, and returns whole sessions with chain-of-thought. The Fast variant drops the reranker and uses one LLM call to extract facts from the retrieved turns before answering (79.0 against 82.4) — [Emergence blog](https://www.emergence.ai/blog/sota-on-longmemeval-with-rag)
- **Hindsight**: memory units are self-contained "narrative" facts that span several turns. Each has an occurrence interval and a mention timestamp. Four retrieval channels: dense, BM25, graph spreading activation, and temporal (date parsing). These are fused with RRF, reranked with the cross-encoder ms-marco-MiniLM-L-6-v2, and greedily packed up to a token budget — [Hindsight paper](https://arxiv.org/html/2512.12818v1)
- **LightMem**: compresses the input with LLMLingua-2 (keeping a ratio r of 0.4–0.8), segments it into topics using attention and similarity, keeps a short-term buffer of 256–1024 tokens, and does offline "sleep-time" consolidation. All methods use the same retriever and the same number of retrieved entries — [LightMem](https://arxiv.org/abs/2510.18866)
- **Letta filesystem agent**: the conversation is stored as files. The agent decides when to grep, search or open them, iterating over several steps — [Letta blog](https://www.letta.com/blog/benchmarking-ai-agent-memory)
- **Memory-R1** (an RL-trained memory manager; the closest work to our thesis): the manager chooses ADD / UPDATE / DELETE / NOOP and is trained with PPO/GRPO on only 152 LoCoMo QA pairs. The answer agent retrieves 60 candidate memories by similarity, then filters them ("memory distillation") — [Memory-R1](https://arxiv.org/html/2508.19828)
- **LongMemEval reference pipeline**: the dense retriever Stella V5 (1.5B) is the default; values are rounds, keys are the value plus extracted user facts, and the reader uses Chain-of-Note with JSON-formatted memory — [LongMemEval paper](https://arxiv.org/html/2410.10813)

### Inferences
- None of the top systems learns *whether* to retrieve. Every one fetches top-k (10–60 items, or several thousand tokens) on every query. The only learned component in Memory-R1 is the write/update policy, and even there retrieval is fixed at 60 items. Our controller's near-zero retrieval rate is therefore an outlier against the whole field. The simplest change to copy is "always retrieve top-k, then let a learned policy filter or rerank" instead of "learn a retrieve action".

### Gaps
- Exact top-k and embedding models are not given for Hindsight, Supermemory or Mastra in the sources I read.

## Q3. Which ablations show which components matter most?

### Takeaway
The largest measured effects come from four places: (1) how you read the evidence (Chain-of-Note/JSON, up to +10 points); (2) indexing rounds rather than sessions, plus fact-augmented keys (+9.4% recall, +5.4% accuracy); (3) time-aware query expansion on temporal questions (+6.8 to 11.3% recall, but only with a strong LLM); and (4) whether retrieval happens at all (removing Memory-R1's memory manager costs 6.5 F1). Hybrid versus single-channel retrieval is worth only 1–2 points at the top end.

### Cited Findings
- LongMemEval, value granularity: splitting sessions into rounds "significantly enhances" GPT-4o and performs about the same with Llama-3.1-8B. Compressing to summaries or facts hurts QA overall through information loss, except on multi-session questions. GPT-4o keeps improving past 20k retrieved tokens, while **Llama-3.1-8B "drops sharply beyond 3k retrieved tokens"** — [LongMemEval paper](https://arxiv.org/html/2410.10813)
- LongMemEval, key expansion (K = V + fact): +9.4% recall@k and +5.4% final accuracy on average. With round values, Recall@5 rises from 0.582 to 0.644 and NDCG@5 from 0.481 to 0.498. With session values, Recall@5 rises from 0.706 to 0.732 — [LongMemEval paper](https://arxiv.org/html/2410.10813)
- LongMemEval, time-aware query expansion (GPT-4o extracts the time range): with session values, Recall@10 rises from 0.654 to 0.707; with round values, Recall@5 rises from 0.451 to 0.526. The paper reports a 6.8–11.3% gain overall. Llama-3.1-8B "often hallucinat[es]" the time ranges — [LongMemEval paper](https://arxiv.org/html/2410.10813)
- LongMemEval, reading: without Chain-of-Note and JSON formatting, accuracy is up to 10 points lower even with oracle retrieval — [LongMemEval paper](https://arxiv.org/html/2410.10813)
- LongMemEval, end-to-end (round values + fact keys, top-10): QA accuracy is GPT-4o 0.682, Llama-3.1-70B 0.584, Llama-3.1-8B 0.572. These come from an appendix table as summarised by the fetch tool; check them against the paper's Table 10 — [LongMemEval paper](https://arxiv.org/html/2410.10813)
- Mem0, RAG chunking ablation (J, GPT-4o-mini). With k=1: 128-token chunks 47.77, 256 50.15, 512 46.05, 1024 40.74, 2048 37.93, 4096 36.84, 8192 44.53. With k=2: 59.56 / 60.97 / 58.19 / 50.68 / 48.57 / 51.79 / 60.53. Going from k=1 to k=2 is worth about 8–13 points — [Mem0 paper](https://arxiv.org/html/2504.19413)
- Memory-R1 ablations (Llama-3.1-8B, F1): removing the memory manager drops F1 from 41.0 to 34.5. Removing the answer agent drops it to 32.5. Memory distillation adds about 1.7 F1 — [Memory-R1](https://arxiv.org/html/2508.19828)
- LightMem: removing topic segmentation costs 6.3 accuracy points (GPT-4o-mini) and 5.4 (Qwen) — [LightMem](https://arxiv.org/abs/2510.18866)
- Agent Zero, retrieval channel ablation on LongMemEval (gpt-5.6-sol): hybrid 95.20; embedding only 94.00 (−1.2); exact-substring grep 93.60 (−1.6); BM25/lexical only 93.40 (−1.8) — [Agent Zero Memory](https://arxiv.org/abs/2608.29606)
- EmergenceMem: dropping the cross-encoder reranker (and adding an extract-then-answer step) costs 3.4 points (82.4 → 79.0) — [Emergence blog](https://www.emergence.ai/blog/sota-on-longmemeval-with-rag)
- Fact extraction loses assistant-side content. On LongMemEval single-session-assistant questions (GPT-4o-mini), NaiveRAG scores 98.21, while LightMem scores 32.14 and Mem0 41.07. In the other direction, on temporal questions full text scores 31.58 and LightMem 67.18 — [LightMem](https://arxiv.org/abs/2510.18866)

### Inferences
- For a 7B reader at a budget of a few thousand tokens, the evidence points to: index at round level; add timestamps to keys; retrieve every turn; keep the retrieved set small (about 3k tokens or less) and precise; format the context as structured JSON with dates. Time-aware expansion probably needs a rule-based date parser rather than the 7B model itself.

### Gaps
- Mem0 did not report a component ablation (for example, removing the UPDATE operations) in the sections I read. Hindsight reports no ablations in the part I read.

## Q4. How do simple baselines compare, and what are the controversies?

### Takeaway
On LoCoMo, full context (about 26k tokens) beats almost every 2025 memory system (72.9 against 66.9 for Mem0). On LongMemEval_S, full context is weak (60.2 for GPT-4o, 45.4 for Llama-3.1-8B, 39.0 for OSS-20B), so retrieval genuinely helps there. Plain RAG is a strong baseline and often beats the vendor systems. LoCoMo has about 6.4% wrong labels, which caps scores near 93.6%, and its GPT-4o-mini judge is lenient.

### Cited Findings
- On LoCoMo, full context scores 72.90 J against 68.44 for the best Mem0 variant — [Mem0 paper](https://arxiv.org/html/2504.19413). LightMem's re-run gives full text 71.83 (GPT-4o-mini) and 74.87 (Qwen3-30B), above all baselines and roughly equal to LightMem itself — [LightMem](https://arxiv.org/abs/2510.18866)
- The Zep–Mem0 dispute. Zep says Mem0's run of Zep: (1) gave both speakers the user role; (2) put timestamps inside the message text instead of the `created_at` field; (3) ran searches one after another, which inflated latency. Corrected, Zep scores 75.14 against the 65.99 Mem0 reported — [Zep blog](https://www.getzep.com/blog/lies-damn-lies-statistics-is-mem0-really-sota-in-agent-memory/)
- Zep's critique of LoCoMo: conversations of only 16–26k tokens; category 5 has no ground truth; some questions refer to images missing from the BLIP captions; some answers attribute statements to the wrong speaker; some questions are ambiguous — [Zep blog](https://www.getzep.com/blog/lies-damn-lies-statistics-is-mem0-really-sota-in-agent-memory/)
- Penfield audit: 99 of 1,540 LoCoMo answers are wrong (6.4%): hallucinated facts, wrong date arithmetic, 24 speaker misattributions. Theoretical maximum about 93.6%. A GPT-4o-mini judge accepted 62.81% of answers that were deliberately wrong but on-topic. The audit also lists reproduction failures (EverMemOS #73, Mem0 #3944, a Zep scoring discrepancy) — [Penfield audit](https://dev.to/penfieldlabs/we-audited-locomo-64-of-the-answer-key-is-wrong-and-the-judge-accepts-up-to-63-of-intentionally-33lg)
- Mem0's own 2026 guide warns that the judge model, the answer model, reranking, prompts and data splits all shift scores. ByteRover's tests put Zep at 75.1 against Zep's claimed 94.7 — [Mem0 2026 guide](https://mem0.ai/blog/ai-memory-benchmarks-in-2026)
- Mixing judges: Hindsight scored itself with GPT-OSS-120B but compared against baselines judged by GPT-4o — [Hindsight paper](https://arxiv.org/html/2512.12818v1)
- Adversarial category: Mem0 and Memory-R1 both exclude it explicitly — [Mem0 paper](https://arxiv.org/html/2504.19413); [Memory-R1](https://arxiv.org/html/2508.19828)
- Naive RAG on LongMemEval: 52.0 according to Emergence — [Emergence blog](https://www.emergence.ai/blog/sota-on-longmemeval-with-rag); 61.0 (GPT-4o-mini) and 73.2 (GLM-4.6) in LightMem's re-run — [LightMem](https://arxiv.org/abs/2510.18866). Retriever choice, session granularity: BM25 R@5 0.634 / R@10 0.710; Contriever 0.723 / 0.823; Stella 0.720 / 0.794 — [LongMemEval paper](https://arxiv.org/html/2410.10813)

### Inferences
- Several 2026 LoCoMo claims (Agent Zero 93.60, Mem0 92.5, ByteRover 92.2) sit at or near the estimated 93.6% label ceiling. At that level, differences mostly reflect judge leniency and noise.
- In our own LoCoMo results, full context (0.51) equals the oracle (0.51). That fits the field: LoCoMo histories are short enough that full context is a ceiling. LongMemEval is the more informative benchmark for a budgeted controller.

### Gaps
- No single independent study re-runs all of the 2026 top systems under one harness. I did not find one within the budget.

## Q5. Which results use small (≤8B) open readers?

### Takeaway
Few. Memory-R1 is the main one. With Qwen-2.5-7B on LoCoMo, Mem0 scores 53.3 J, A-Mem 40.8, MemoryOS 51.3, and Memory-R1 61.5. The LongMemEval paper itself gives Llama-3.1-8B: 0.454 with full context, 0.710 with oracle sessions, and about 0.57 with the best retrieval pipeline.

### Cited Findings
- Memory-R1, LoCoMo J with Qwen-2.5-7B-Instruct: LoCoMo-RAG 12.17, A-Mem 40.78, MemoryOS 51.26, Mem0 53.30, Memory-SFT 61.13, Memory-R1-PPO 59.53, Memory-R1-GRPO 61.51 (F1 43.14) — [Memory-R1](https://arxiv.org/html/2508.19828)
- Memory-R1, LoCoMo J with Llama-3.1-8B-Instruct: LoCoMo-RAG 13.62, A-Mem 44.76, Mem0 45.68, MemoryOS 48.20, Memory-SFT 58.76, PPO 57.54, GRPO 62.74 (F1 45.02). Per category F1, GRPO against MemoryOS: single-hop 35.73 vs 31.89, multi-hop 35.65 vs 13.80, open-domain 47.42 vs 40.74, temporal 49.86 vs 28.74 — [Memory-R1](https://arxiv.org/html/2508.19828)
- Memory-R1 transferred zero-shot to LongMemEval with overall F1 45.2 (Llama) and 46.7 (Qwen-2.5-7B) — [Memory-R1](https://arxiv.org/html/2508.19828)
- LongMemEval, Llama-3.1-8B: oracle 0.710, full context on _S 0.454, best round+fact retrieval about 0.572. Accuracy drops sharply when more than 3k tokens are retrieved. The 8B model cannot do time-aware query expansion reliably — [LongMemEval paper](https://arxiv.org/html/2410.10813)
- Phi-3.5-mini (3.8B): 0.660 with oracle sessions, 0.342 with full context — [LongMemEval paper](https://arxiv.org/html/2410.10813)
- A-Mem's original paper also evaluates small open models on LoCoMo (F1). The search snippet says it reports F1 45.85 against MemGPT 25.52 with GPT-4o-mini, but I did not read the paper's small-model tables — [A-Mem](https://arxiv.org/pdf/2502.12110)

### Inferences
- Our numbers (LoCoMo judge: learned 0.29–0.33, FIFO 0.38–0.40, oracle 0.51; LongMemEval: keep-last-4 0.53, oracle 0.65) are broadly in the same range as the 7B results above: Mem0 with Qwen-2.5-7B scores 53.3 on LoCoMo; Llama-8B with oracle sessions on LongMemEval scores 0.71. Still, judges and protocols differ, so only rough comparison is possible. Memory-R1 shows that an RL-trained memory policy plus "retrieve 60, then filter" lifts a 7B reader by about 8 J over Mem0, using only 152 training QAs. That makes it the most directly relevant point of comparison for our learned controller.

### Gaps
- I did not verify the A-Mem small-model tables, MemoryOS's Qwen numbers, or Memory-R1's judge model.
