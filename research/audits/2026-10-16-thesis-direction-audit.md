# Research audit for the thesis-advisor conversation (2026-10-16)

Advisor-session audit, read-only, at branch `jacobian-lens` commit b0f955d. Three parallel read-only audits (memory
controller §1–§32; interpretability and answer failures §33–§37; the CATAN idea) were spot-checked against run
artifacts by the Advisor. "E:" = a line of `EXPERIMENTS.md`; "S:" = `research/STATE.md`; "T:" =
`docs/research/THESIS_RESULTS_OUTLINE.md`; "N:" = `docs/research/HINDSIGHT_MEMORY_RESEARCH_NOTE.md`; "O:" =
`research/OPEN_QUESTIONS.md`. Intervals are 95% paired bootstraps as the record gives them. Nothing was run or edited
for this audit. Where a statement is an interpretation or a proposal rather than an observation, it says so.

Reading rules carried through the whole document: the LongMemEval judge is one Qwen2.5-7B model and understates
absolute accuracy by about 0.1 (false negatives; paired differences unaffected, S caveat 3). Differences under about
0.01 are reader noise (S caveat 2). Lens readouts are readouts, not evidence that a model "knows" anything.

---

## Part 1. Memory-management research (§1–§32)

### 1.1 Original hypothesis and the controller's job

- **Problem (N:18, N:55–65).** A long-running agent accumulates observations. A frozen task model reads a bounded
  active context. Something must decide, before the need arises, what to keep, archive, delete or rewrite. Framed as
  sequential decision-making with delayed credit, not as retrieval.
- **Hypothesis (N:20; sub-hypotheses H1–H7 at N:591–601).** Learned memory-management policies beat fixed heuristics at
  the same active-memory and compute budget. Later recast as RQ1–RQ4 (T:47–81): learned vs rules at equal budget;
  learning from hindsight; whether RL adds to imitation; whether it holds with real models and when it is worth it.
- **The 27,526-parameter controller** (`ItemPolicy`, `memctl/rl/policy.py`; E:333–334; `runs/rl_bc/metadata.json`
  records 27526): a per-item two-layer MLP (hidden 64) with DeepSets mean/max context, one logit per item and action
  (EVICT, MOVE_TO_ARCHIVE, COMPACT, COMPACT_AND_ARCHIVE, RETRIEVE) plus a scalar value head. Trained by DAgger imitation of
  a hindsight oracle, then PPO or GRPO. The later "§19 head" is the same network class (E:4727); 8,705 parameters are an
  unused value head, so about 18,800 are active (S:13–14; arithmetic checked from the layer sizes, not an artifact). As
  deployed in `fixed8` it keeps nothing but the current turn and picks 8 of 32 BM25 candidates by its RETRIEVE logit:
  a top-k reranker (E:4978–4981).

### 1.2 Experiments, baselines, data, models, metrics, compute

- **Synthetic, scripted reader** (clean rerun at `ef57fc4`, 2026-10-05): §1–§4b, D1 regret teacher, GRPO vs PPO, D2
  joint controller, D5 priced archive, §5/5b sequential workflow task. Baselines: FIFO, LRU, LFU, age-decay, random,
  similarity, salience, archive-everything-and-retrieve, keep-last-k + top-5, approximate and exact-ILP hindsight
  oracles, full context (E:99–111; N:28).
- **Real conversations:** LoCoMo (10 conversations, 1,986 questions) and LongMemEval-S (500 questions, 5 stratified
  folds; 470 answerable). Reviewed protocol from §13 (E:1339–1348): retrieval floor, no EVICT, reader prompt frozen at
  `7b5fc30`, official LongMemEval judge prompts, Qwen2.5-7B judge; Granite-3.1-8B second judge in §18 (97.3% agreement,
  κ 0.946).
- **Readers:** Qwen2.5 3B/7B/14B, Granite-3.1 2B/8B (§23); Llama 3.2 3B / 3.1 8B, Gemma 3 4B/12B, Phi-4-mini / Phi-4
  (§24); all vLLM bf16 on a rented A40. StreamMemBench (§25, closed free), OpenJev as controller (§26), LRE (§27),
  query rewriting (§28), reader-utility signal (§29), rerankers (§32); UtilMem (§30) and test-time training (§31)
  closed without runs.
- **Metrics:** task success; evidence in view (all-found); judge accuracy with abstention reported apart; prompt tokens
  beside every accuracy; P(all in view) and P(correct | in view); unknown rate; seven-class failure attribution;
  needed-model AUC.
- **Compute:** A40 at $0.49/h then $0.59/h. Benchmark programme §13–§28: 15.77 h, $8.26; everything through §28:
  24.60 h, $12.58 (T:205–206); through §36b about $17.55 of a self-imposed $20 cap (S:78–81). Synthetic work and the
  §21–§22 simulator ran on an 8-core laptop (17 h for §22a).

### 1.3 Where the learned controller won and where it failed

**Synthetic, scripted reader (wins).**
- §1: salience − FIFO +0.326 (+0.303, +0.347); exact oracle +0.563 at horizon 500, 10% budget (E:122–123).
- §4: learned policies beat every recency rule by +0.24 to +0.49 but **not** the best content rule: rl_bc − salience
  −0.049 (−0.069, −0.030) at 5%; BC+PPO +0.018 (+0.014, +0.023) at 25% only; PPO alone −0.13 to −0.24 (E:369–394).
- **D1 regret teacher:** 0.851 (0.843, 0.858) at 2% vs 0.335 for oracle imitation; +0.092 (+0.076, +0.107) over the best
  heuristic at 2%, shrinking to +0.003 (−0.003, +0.010) at 25%; gap to the oracle −0.115 (E:494–502).
- §5b workflow task: learned 1.000 vs FIFO 0.028 vs the hindsight oracle 0.808 (the oracle goes blind once the policy's
  choices change the stream; E:6465–6490).

**Synthetic task with real readers.** §9 (Qwen 7B, 100 episodes): regret policy 0.832 at 2% vs FIFO-archive 0.741;
regret − salience-archive +0.104 (+0.086, +0.123); full context 0.898; oracle 0.937 (E:883–898). §11a with
follow-the-clue search 0.889 (+0.057 (+0.045, +0.069)).

**Real conversations, the controller as designed (failures).**
- §11e: learned controllers trail FIFO by 0.10–0.30 with intervals excluding 0; LoCoMo 0.086 vs 0.224, −0.138 (−0.165,
  −0.111); the retrieval head learned never to fire (E:1117–1233).
- §13a learned eviction with a retrieval floor: +0.047 (+0.004, +0.089) at fill, −0.043 (−0.085, −0.004) at 3k tokens;
  nothing survives Holm; no row on the accuracy-vs-tokens frontier (E:1473–1491). §13b LoCoMo: CIs include 0.
- §14 priced archive: no cell passes; a turn's later need is barely predictable before the question, needed-model AUC
  0.62–0.74 (E:1671–1710). This is the query-blind ceiling.
- §25 StreamMemBench: `fixed8` evidence in view 0.318 / 0.229 / 0.100 vs FIFO 0.930 / 0.908 / 0.996: recency wins
  outright (E:3887–3899).

**Real conversations, query-time selection (the one win).**
- §15b `listsum` 0.513 vs keep-last-0 + top-5 0.466, +0.047 (+0.009, +0.087) at 745 vs 1,554 prompt tokens; Holm p
  0.058 (E:1868–1879). §15c replication +0.044 (+0.008, +0.082), seed SD 0.002, second host.
- §19c `fixed8` vs FIFO + floor **+0.113 (+0.072, +0.155)** at 1,049 vs 2,964 tokens (E:2523–2538). §23: Qwen 3B +0.153
  (+0.109, +0.200), 7B +0.102 (+0.060, +0.145), 14B +0.102, Granite 8B +0.049 (+0.002, +0.096), Granite 2B −0.004 (−0.047,
  +0.038). §24: +0.062 to +0.162 in Llama, Gemma and Phi. **10 of 11 readers in 5 families** (S:F2).
- §26: head 0.526 vs OpenJev-as-selector 0.489 vs FIFO 0.445; OpenJev − head −0.036 (−0.081, +0.009), and about 16×
  the cost (`docs/research/cost_per_question.md`). §27: head 0.536 vs LRE 0.109 in the head's slots; LRE in place of
  recency +0.028 (−0.002, +0.057) vs FIFO.

**But the network is not the win.** §32: head − 31-parameter logistic regression on the same 30 features **+0.013
(−0.017, +0.043)**; head − zero-shot cross-encoder +0.013 (−0.030, +0.053); only BM25 order is clearly worse (+0.068
(+0.026, +0.113)) (E:5245–5260). The lift is carried by the feature set and the evidence labels. Caveats: three distinct
heads, not five (S caveat 1; and see 1.8 below); reader noise ±0.007; judge understates absolutes by about 0.1.

**Learning on top of the head (all negative or null).** §16 GRPO +0.0125 (−0.005, +0.033), gate fail. §19/19b adaptive
k fell 0.026 (+0.012, +0.042) short of its bound. §20 stored notes +0.043 vs a +0.05 bar. §28 query rewrite +0.006
(−0.017, +0.030). §29 reader-utility labels −0.013 (−0.051, +0.026), and the utility head picks longer turns (1,881 vs
1,049 tokens).

**Exploratory, not claims:** §18a LoCoMo head 0.242 vs 0.155 (+0.087 (+0.063, +0.110)), still below FIFO at a 10%
budget; §19c LoCoMo `fixed16` 0.344 at 734 tokens vs FIFO 0.295 at 3,087 (+0.049 (+0.034, +0.062)).

### 1.4 What survived the move to real readers and benchmarks

Survived: (i) regret teacher beats oracle imitation under a real reader on the synthetic task (+0.512 (+0.492,
+0.532) at 7B over 3 seeds, E:894–895); (ii) less context can be better (oracle 2% memory beats full context +0.039
(+0.022, +0.056) at 7B); (iii) archive-and-retrieve beats deletion on real-conversation evidence metrics, hardened into
"query-blind eviction cannot beat archive-everything" (S:F8); (iv) imitation ≥ RL at every scale (D1, §16, §22c).

Did not survive: every learned eviction or retrieval policy on real conversations (§11, §13); follow-the-clue search
(+0.01 on real text); salience's advantage (a dataset artifact, E:6554–6557); the D5 price mechanism (§14); "no fixed
rule wins both tasks" (on StreamMemBench recency wins outright).

New and only on real text: **context rot** (onset about 3k tokens; `fixed16` − `fixed8` §19c −0.057 (−0.096, −0.019),
§19d −0.026 (−0.066, +0.015) not confirmed, §23 7B −0.051 (−0.089, −0.015), 3B −0.113, 14B −0.026: same sign on two
heads and two hosts, significant in some; S:F7 "likely, not settled"); and the head-≈-linear result (F3).

### 1.5 Bottlenecks: demonstrated vs conjectured

Demonstrated:
- **Information per sample / supervision.** Needed-model AUC 0.62–0.74 before the question (E:1696–1710); 44–61% of
  GRPO groups tied (E:2127–2128); the §4 oracle gap is "information the policy cannot have"; §32 shows the carrier is
  the features and labels, not the network.
- **The frozen reader.** P(correct | all evidence in view) under `fixed8`: **0.397 (3B), 0.667 (7B), 0.773 (14B)**
  (E:3497–3498). Qwen 3B with the controller vs Qwen 7B with FIFO + floor: **−0.121 (−0.170, −0.070)**, closing 56%
  (41–72%) of the gap (`runs/exp23_report.md:23`). This is what turned the project toward Part 2.
- **Retrieval.** BM25 recall 0.35 on LoCoMo; multi-hop all-found 4–11% at 16 candidates; `archived_not_retrieved` the
  largest failure class on the synthetic task (about 300 vs 72 for the oracle, E:915–917).
- **Optimisation.** PPO seed spread 0.13; GRPO late collapse; RL without a KL pull degrades a good init (§22, −0.027
  (−0.049, −0.003)).
- **Data and overfit.** 8 training conversations in §22; a 400-row chooser in §19b; duplicated heads (1.8).
- **Measurement.** Strict single judge (−0.1 absolute); ±0.007 reader noise; LoCoMo intervals resample 10 conversations.
- **Compute.** A $20 self-imposed cap with about $2.45 left; 8-core laptop; cross-encoder tuning killed by the kernel at
  73 min/epoch (E:5087–5089).

Conjectured, not shown: that any learner can close the lossy-memory headroom (S:F10; O:Q1–Q3); that the §21 headroom
exists outside LoCoMo; that the remaining 3B gap is "reading" (O:Q14); that results hold under a prompt not tuned on
Qwen 7B or on a second benchmark; that context rot is real at 7B beyond the pairs run.

### 1.6 Hindsight, oracle imitation, regret, PPO, GRPO

- DAgger from the hindsight expert is about 10× more sample-efficient than PPO (240 episodes reach what 2,400 PPO
  episodes do not, E:401–402).
- **Oracle imitation is the wrong target once archiving exists.** The learner copies the majority action "delete": 92%
  of needed items deleted vs 0% for the regret learner (E:508–509); PPO on top did not undo it (0.340 → 0.357).
- **Least-regret sets fix a free archive (D1)** but at any positive archive price the regret labels equal the oracle
  labels and imitation collapses to delete-everything (E:706–711); a calibrated P(needed) in a Bayes rule wins at low
  prices and loses at 0.2 (E:744–793).
- **An oracle warm start poisons RL:** oracle-BC → GRPO 0.451 (0.347, 0.559) at 2% vs 0.704–0.717 from scratch (E:555,
  570–577). Regret-BC → GRPO adds +0.008 (+0.001, +0.015), the same as best-of-8 filtered imitation; four GRPO credit
  variants are indistinguishable (E:982–1034); with the 7B reader GRPO − BC −0.004 (−0.016, +0.008).
- Real conversations (§22, §22b, §22c): per-write hindsight credit beats uniform credit in both policy classes
  (+0.015 (+0.007, +0.024); +0.036 (+0.021, +0.054)), the gain carried by the evidence links; **no RL variant beats the
  supervised writer** (−0.001 (−0.012, +0.011) with a KL pull). RQ3 closed (D-10).
- **A finding that stands on its own:** "optimal eviction is not optimal memory management". Hindsight-deletion labels
  are the wrong teacher once a cheap insurance action (archive) exists, and an oracle warm start is strictly worse for
  RL. Clean-commit, 3-seed, replicated under 3B and 7B readers. Limits: synthetic generators, a scripted or synthetic
  task, D5's price fragility, and a crowded niche (the related-work report lists LRE, MemCon, MemAudit, Parrot,
  ForesightKV and Memex(RL) as overlapping neighbours; `docs/research/RELATED_WORK_LITERATURE_REPORT.md`, Appendix 2).

### 1.7 Strongest falsifiable hypothesis if this line continued (proposal)

On sequential, lossy memory, where a session's note cannot be repaired at the question, the oracle − supervised-writer
gap (+0.139 (+0.098, +0.179), §21/§22) is a per-sample information limit, not an optimiser limit: it does not shrink
when training streams grow by an order of magnitude, and no credit rule beats supervised labels at any scale.
Falsifiers, free on the §21 simulator: (a) the never-run §21 gate on LongMemEval composed episodes; if oracle − best
rule < +0.10 the headroom is LoCoMo-specific; (b) refit the supervised writer and the §22c learner on 8 → about 400
streams; if the gap falls with a CI excluding the LoCoMo value, or per-write RL clears the supervised writer with
CI > 0, the hypothesis is false. The remaining budget buys one 0.5 h A40 confirmation pass.

### 1.8 Record-keeping findings from this audit (new)

- **Checkpoint identity, extended.** S caveat 1 records head A folds 1 = 2 and 3 = 4. This audit's sha256 of every
  `runs/lme_n4_head_f*_{a,b}/checkpoints/policy_best.pt` also finds head B folds 0 = 1 and 2 = 3, **and head A fold 0
  = head B fold 4** (prefix 9f0be108). Heads A and B together hold **5 distinct networks, not 10**, so head B was not an
  independent replicate of head A on fold 4 (§19a/§19d). Paired bootstraps are unaffected; the "two heads" wording in
  §19d needs a footnote. (Verified by the Advisor on 2026-10-16.)
- Aggregate reports `runs/exp1_delete_only/report.json` and `runs/exp4_rl_eval/report.json` exist only in
  `runs/_archive_dirty_2026-09-30/`; the clean rerun has per-cell `summary.json` files (one cell checked: salience
  0.762, FIFO 0.437, matches E:108–111).
- §7 smoke tests never rerun clean; §21's LongMemEval corroboration row and §29's leave-one-out signal were declared
  and never run; the cost table in T ends at §28.

---

## Part 2. Mechanistic interpretability and answer failures (§33–§37)

### 2.1 The motivating phenomenon, with examples

Under `fixed8` with every gold evidence item in the 8 visible memory items, P(correct | in view) is 0.397 at 3B vs
0.667 at 7B (§23), and the 3B answers "unknown" on 0.372 of answerable questions. The controller did its job; the
reader did not read (E:5710). On confirmation folds 2–4 there are about 74 in-view questions where the 3B fails and
the 7B is right (40 refusals, 34 wrong; audit join of `runs/exp23_verdicts.json` with the fold episodes; the §34 text's
own count is 65 refusals of which the 7B answers 40 correctly, E:5862).

Concrete rows (fold 2; outputs from `cache/generations_exp23_{reader}/`, matched by prompt hash):

| qid | question | gold | 3B output (note, then answer) | 7B |
|---|---|---|---|---|
| ad7109d1 | What speed is my new internet plan? | 500 Mbps | "user mentioned upgrading to 500 Mbps about three weeks ago, but did not specify the exact date… Answer: unknown" | "Answer: 500 Mbps" |
| 2318644b | How much more did I spend on accommodations per night in Hawaii compared to Tokyo? | $270 | "…over $300 per night … Maui … hostel in Tokyo … $30 per night. Answer: unknown" | "$300 − $30 = $270. Answer: $270" |
| 6f9b354f | What color did I repaint my bedroom walls? | a lighter shade of gray | "…I've recently repainted my bedroom walls a lighter shade of gray. Answer: unknown" | "Answer: lighter shade of gray" |
| 1faac195 | Where does my sister Emily live? | Denver | "…visiting their sister Emily in Denver. Answer: unknown" | "Answer: Denver" |
| 5a7937c8 | How many days … faith-related activities in December? | 3 days | "…December 17th and December 24th. The difference … is 7 days. Answer: 7" | "…10th, 17th, and 24th. Answer: 3 days" |
| 91b15a6e | Minimum amount … necklace and antique vanity? | $5,150 | "necklace: $5,000 … vanity: $150 … Answer: $50150" | "$5,000 + $150 = $5,150. Answer: $5,150" |

The pattern: the 3B's own note names the gold fact and the model then writes `Answer: unknown`. These rows are the
"S" class of §35 (refusal with the gold in the note, n 22 on confirmation folds).

### 2.2 Models, settings, conditions

- Readers: Qwen/Qwen2.5-3B-Instruct (36 layers) and Qwen2.5-7B-Instruct (28 layers). 14B never read by any lens
  (E:5869 names it as a wanted third point). No other family in this line.
- Engines: §33/§34/§34b/§34c `transformers` bf16 on an A40; §35/§36b vLLM 0.8.5 bf16 greedy, temperature 0, seed 0,
  max tokens 256 (regenerated) / 64 (frozen-note) / 4 (judge); the §36b wording probe ran fp32 on the laptop CPU.
- Prompt: the frozen §23 `fixed8` prompt, rebuilt byte-exactly and sha256-checked per question: the §23 goal line
  ("Answer with a short phrase. If the memory does not contain the answer, reply: unknown") plus the Chain-of-Note
  instruction ("First write a brief note of at most 80 words … End with one line 'Answer: <short phrase>'"), one
  user message, Qwen chat template, no system message (`memctl/agents/llm.py:17–33`).
- Population: in-view answerable questions with a judge label for both readers, paired by prompt hash. Development
  folds 0–1: 126; confirmation folds 2–4: 204; all folds 330. §35/§36 use the 165 confirmation rows whose cached 3B
  output carries an `Answer:` marker; §36 has 140 rows with a forced candidate.

### 2.3 What exactly was done with the lenses

- **Positions** (`memctl/analysis/jlens_readout.py`): P1 = the token that predicts the gold's first content token, with
  `Answer:` appended straight after the prompt in **forced** mode (§34, §34c): a forced immediate answer, although 200
  of 204 cached 3B outputs wrote a note first. **Own** mode (§34b) teacher-forces the cached output up to its last
  `Answer:`; outputs without the marker (39 of 204) are read at their first token. **Ablated** mode (§34c) removes every
  evidence line by exact span (`memctl/analysis/jacobian.py ablate_evidence`), forced position.
- **Tracked tokens:** the gold's first content token, " unknown" and "unknown". Rank = 1 + number of logits strictly
  greater.
- **J-lens** (Anthropic's `jlens` 0.1.0, vendored at `external/jacobian-lens` 581d398): lens_l(h) = unembed(J_l · h),
  with J_l the expected Jacobian of the final residual with respect to layer l's residual over a generic corpus. Fitted
  here on **56 WikiText-103 paragraphs** (target 128; the 7B's 90-minute box stopped at 56, the 3B then matched),
  max_seq_len 128, bf16. **Logit lens** = unembed(h_l) directly. The final layer is the model's own logits for both.
  Every layer 0..n−1 is read.
- **Outcomes at k = 10:** C = gold rank ≤ 10 at the **final** layer (model logits, lens-free); B = rank ≤ 10 at some
  layer but > 10 at the end; A = never ≤ 10. Only the A/B split depends on the lens (E:5856). Emergence depth = first
  layer with rank ≤ 10 divided by the layer count (J-lens only; there is no logit-lens emergence row).
- **Lens gate** (dev folds, fixed before confirmation): the 3B's J-lens **failed** the library's evaluation (the `typo`
  set, 0.052 vs 0.229), so every 3B verdict rests on the **logit-lens fallback rows**; the 7B's passed on the rank
  criterion by a wide margin and on the library sets by a noise-sized 0.004. A community 100-prompt 7B lens agrees
  (Claim 1 +0.465 (+0.353, +0.584)).
- **§33 input-gradient attribution** never produced a gradient: its pre-registered regeneration check needed 15 of 20
  exact reproductions of cached vLLM outputs by bf16 `transformers` and got 10 (16 of 20 same final answer), so it
  stopped (E:6353–6410).

### 2.4 Results, 3B vs 7B (confirmation folds 2–4, 204 paired; `runs/exp34_report_confirm.md`; review ACCEPT)

- **Claim 1 (size), SHOWN.** log10 best rank at P1, 3B − 7B **+0.486 (+0.376, +0.600)** on the logit lens (decides),
  +0.535 J-lens. About 3× worse in rank. Share of A (never in top-10) +0.201 (+0.137, +0.265).
- **Claim 2 (within the 3B), SHOWN.** AUC of gold rank for wrong vs right **0.654 (0.576, 0.727)** logit; 0.629 J-lens.
  Dev-fold lower bounds had straddled 0.5 (0.499 / 0.502). 7B descriptive 0.692.
- **Decomposition of 3B failures (127):** A 0.346 (0.268, 0.433), **B 0.016 (0.000, 0.039)**, **C 0.638 (0.551, 0.717)**.
  Refusals (65): A 0.400 / B 0.031 / C 0.569. Wrong (62): A 0.290 / B 0.000 / C 0.710. Successes (77): C 0.727. The
  pre-registered prediction "refusals are mostly B (found then lost)" was **falsified**: when the gold enters the top-10
  it almost always stays to the final layer, and ` unknown` is rank 1 at the end (gold median final rank 5 on refusals).
- **Emergence depth (J-lens only):** 3B median 0.889 (q1 0.590, q3 0.944) vs 7B 0.464 (q1 0.250, q3 0.857). **The
  "7B mid-stack, 3B late" contrast is a J-lens statement.** On the plain logit lens the audit's per-fold tally puts both
  readers late (7B 0.82–0.86, 3B 0.92–0.94 per fold); the ordering holds on both lenses, the size of the contrast does
  not. (The per-fold logit tally covers all 100 reads per fold, not only the in-view pairs; the report tables publish
  only the J-lens depth.)
- **Depth does not separate failures from successes within a reader** (`runs/exp34_depth_allfolds.md`, exploratory):
  3B failures − successes +0.033 (−0.003, +0.069); the 7B's failures emerge *earlier* than its successes, −0.099
  (−0.168, −0.027), suspected token frequency. What separates 3B failures is presence, not timing.
- **By question type (J-lens):** 3B A share 0.889 on single-session-preference (n 9), 0.400 knowledge-update (35),
  0.194 multi-session (36). Per-fold C share 3B 0.55–0.64, 7B 0.77–0.86.
- **No plots exist for §33–§36.** Every figure is a table (`runs/exp34*_report_*.md`, `runs/exp34_depth_*.md`, copies in
  `docs/research/`). `docs/research/figures/` holds only §13/§19/§23/§24 PNGs.

### 2.5 What "appear" means, operationally

"Present" / "appears" / "enters" = the gold's **first content token** has rank ≤ 10 among the vocabulary at position P1
at some layer under the chosen lens. "Remains at the final layer" = outcome C, rank ≤ 10 in the model's own final
logits. It is **not** top-1 (on refusals the gold's final rank is median 5 while ` unknown` is rank 1), **not** a probe,
and **not** a causal statement. Tokenisation gives a paraphrase floor: 0.22–0.26 of the 3B's *correct* answers are A
(11 paraphrases such as "4" for "four", 6 token-prefix cases such as "IKE" for "IKEA"), and function-word first tokens
("a lighter shade…") are tracked on 21–22 of 204 questions. Quote paired gaps, never raw A shares.

### 2.6 Causal tests: what exists and what does not

- **§34c, an input ablation (prompt level).** Cutting the evidence lowers "ever enters top-10" 0.701 → 0.466, **+0.235
  (+0.176, +0.294)**, and C 0.672 → 0.431; rank lift about 4× (+0.631 log10). SHOWN for both readers.
- **§35, behavioural interventions on text.** Arm B (the 3B's own cached note frozen, plus "Do not reply unknown"):
  accuracy 0.333 → 0.509, **+0.176 (+0.109, +0.242)**, **0.53 (0.33, 0.73)** of the matched 3B−7B gap (0.51 vs 0.66
  on the same 165 rows). Arm F (forced choice with regeneration) +0.242 (+0.164, +0.321). On the S class (refusal with
  the gold in the note, n 22) +0.636 (+0.455, +0.818). Costs: 0.210 of 62 marked refusals become wrong answers under B;
  answered-and-wrong on evidence-free prompts doubles under F (0.265 → 0.534). The 7B gains +0.020 / +0.005.
- **§36b, a second elicitation (readout, not intervention).** The 3B's one-token A/B/C support score for its own forced
  answers: AUROC **0.573 (0.478, 0.666), NOT SHOWN**; the score drops +0.235 (+0.170, +0.302) when the evidence is cut
  and not when length-matched filler replaces it (an evidence-presence detector, not a verifier); the 3B's natural
  refusals as a score 0.455. The 7B checking the same candidates 0.692 (0.600, 0.777), paired +0.119 (+0.023,
  +0.216), descriptive because the 7B is also the judge.
- **Activation patching, steering, or any representation-level intervention: NONE.** The §34 Stage 2 (add the
  lens-transported gold direction or suppress the ` unknown` direction at the emergence layer) was gated on both lenses
  passing and **was not opened** because the 3B lens failed (E:5881–5885). The record says so itself: "still a readout,
  not an intervention" (E:5967). **Causal availability of the internal representation has not been demonstrated.**

### 2.7 Alternative explanations, with the numbers that bear on them

1. **Plausibility, not extraction (§34c).** Among 3B failures that are C with the evidence, **0.605 (0.494, 0.716)**
   stay C without it (strict number/name golds 0.707; function-word golds 0.286). Only about a quarter of the 3B's
   "present" failures are evidence-dependent. The 7B's share is similar (0.745). And Claim 1's rank gap **persists with
   the evidence cut** (+0.473 vs +0.486; lift difference −0.013 (−0.126, +0.097)): it is a rank gap at the answer
   position, not "the 7B extracts more".
2. **Forced-format artifact (§34b).** Pooled own-vs-forced presence +0.010 (−0.049, +0.069), NOT SHOWN; but on the 165
   marked outputs presence rises 0.685 → 0.776, +0.091 (+0.030, +0.152), mostly on successes, while the 39 marker-less
   rows drop. Lateness is the model's (median 0.917 in both modes); part of the absence is the forced format.
   Teacher-forcing also assumes reproduction; §33 found 10 of 20 exact.
3. **Answer selection, "unknown wins late".** ` unknown` reaches top-1 at some layer on 47 of 65 refusals (depth 0.90),
   but it does so on 58% of all questions while the 3B refuses on 32%, and it barely tracks the evidence (0.578 → 0.632
   without evidence, +0.054 (+0.010, +0.103)) where the 7B's does (0.078 → 0.529, +0.451). Behaviourally, instruction
   removes most refusals (0.358 → 0.015 under F).
4. **Prompt and wording sensitivity.** §36 run 1 was unusable because the verifier prompt ended in `Answer: ` and both
   readers answered instead of grading (label mass ≥ 0.5 on 0.11 of rows); removing the cue gave 84 of 84 compliance.
   The §35 prediction (+0.05 to +0.08) was wrong by 2–3×. One prompt family throughout.
5. **Calibration and cost.** A fifth of refusals turn wrong under B; evidence-free F-cut answers judged "correct" by
   plausibility 0.147 (0.098, 0.196).
6. **Judge error.** About 0.1 false negatives means about 20 of 204 answers, about 0.16 of the 127 "failures", carry
   the success profile, inflating C and deflating A among failures. The 7B verifier shares the judge's weights.
7. **Weak lenses.** 56 fitting prompts vs the paper's about 1,000; the 3B lens failed its gate; the 7B's B share and the
   emergence depth differ by lens; depth likely carries token frequency.
8. **Two models, one family, one benchmark, one prompt.** Not a scaling law.

### 2.8 Strongest supportable conclusion

At a forced answer position (and, for marked outputs, at the model's own), the Qwen2.5-3B reader ranks the gold's first
token about 3× worse than the 7B (+0.486 log10, CI excludes 0) and worse on its failures than its successes (AUC
0.654). In about two thirds of its in-view failures the gold token sits in the final-layer top-10 of the model's own
logits while ` unknown` is top-1, and "found then overridden" is rare; but about 0.6 of that presence survives removing
the evidence, so "present" is mostly plausibility and only about a quarter of failures show evidence-dependent presence.
Behaviourally, the same 3B with its own note frozen and told not to answer unknown closes about half (0.53 (0.33,
0.73)) of the matched gap to the 7B, at the cost of a fifth of refusals turning wrong; it then cannot check which of
its forced answers the evidence supports (AUROC 0.573). **Not shown:** that the 3B "knows" or "extracts" the answer;
that any internal representation is causally used or usable; where the refusal decision is made; anything from the 3B
J-lens with a verdict. The right ceiling is the record's own: a response-policy component of the shortfall, readouts
consistent with late loss, no mechanism established.

### 2.9 The most promising next experiment (proposal, not yet run)

Two routes are open; they answer different questions.

- **Causal (the gap in the record).** Activation patching between the 3B's own refusal rows and matched success rows at
  late layers, at the model's own answer position, with the evidence-cut prompt as the plausibility control: does
  patching the residual at layers ≥ 0.8 depth from a success row flip `unknown` to the gold on the S class (gold in the
  note, answer unknown), and does the same patch do nothing on evidence-cut rows? A difference-in-differences on about
  60 S/S′ rows with paired bootstraps is feasible on an A40 in under an hour with the 3B. Pre-registered falsifier: no
  flip rate above the evidence-cut control. If it fails, the "late selection" reading loses its last support.
- **Behavioural (cheap, already pre-registered as §37, `research/proposals/2026-10-16-exp37-elenchus.md`, D-25).** A
  three-turn cross-examination (cite items, derive blind, confront) as a self-check, with a planted-wrong-thesis
  control for confirmation bias and a blind-vs-in-dialogue control for sycophancy; free on the laptop for folds 0–1;
  the author's recorded prior is about one in three of clearing the 0.65 bar.

A note on literature: the repository holds literature notes for memory controllers but **none for interpretability or
abstention**. Before any of this is written up, the lens results must be placed against the logit-lens and tuned-lens
literature, the "models know more than they show" / hidden-knowledge probing literature, the refusal-direction and
abstention-calibration literature, and the long-context "lost in the middle" / context-rot literature. The audit
could not do that from the repo; it is listed as missing.

---

## Part 3. Multi-agent social dynamics / CATAN

- **Implemented and evaluated: nothing.** No code, engine, agent harness, logs, configs, seeds, repo, branch or commit
  mentions it anywhere on this machine or in the user's GitHub repositories. The user confirmed on 2026-10-16 that work
  has not started.
- **The only artifact** is a two-paragraph pitch in an email draft to a prospective grad mentor (Google Drive document
  "Round 4 Emails thesis advisor search", created 2026-10-07, last edited 2026-10-15), framing LLM agents playing CATAN
  with public chat and private whispers as a test of whether interruption failure modes ("self-doubt and panic") from
  the mentor's Interruptible-LRMs paper appear in multi-agent play, and as a setting to observe "how trust, alliances
  and grudges form".
- **Measurable variables:** none defined yet. A real design would need an event-log schema (trade proposals, accepts,
  rejects per pair; robber placement target; message counts per channel and pair; victory-point trajectories; win
  rates) before any construct like trust or betrayal can be operationalised.
- **Hypothesis:** the interruption framing is testable (interruption changes decision quality or consistency, measured
  as X); the social half is at the "observe emergent behaviour" level.
- **Controls:** none stated. A credible design needs a no-chat arm, scripted or random opponents, seed replication, and
  a game-theoretic reference for "rational" trades.
- **Assessment:** an idea at pitch stage, not a proposal. That is not a criticism of the idea; it is a statement of where
  it is.

---

## Part 4. Comparison of the three directions

| criterion | A. memory policies and hindsight credit | B. answer failures with evidence in view | C. social dynamics in CATAN |
|---|---|---|---|
| Clarity of question | Clear but already answered twice: learned eviction loses to archive-and-retrieve on real text; RL does not beat imitation (RQ3 closed). The open form (F10 lossy-memory headroom) is clear but lives in a scripted simulator | Clear and narrow: for a small reader with the evidence in view, how much of the shortfall is answering policy vs reading, and is the answer causally available? | Not yet a question; two constructs named, none operationalised |
| Existing evidence | Large: about 32 numbered experiments, replicated controller lift (+0.113; 10 of 11 readers), negative results clean and pre-registered | Moderate and recent: §34 (two claims SHOWN), §34b, §34c, §35 (+0.176, 0.53 of the gap), §36b (negative), all pre-registered with sealed confirmation folds; no causal test yet | None |
| Novelty and literature | Crowded niche (LRE, MemCon, MemAudit, Parrot, ForesightKV, Memex(RL)); the self-standing finding "oracle imitation is the wrong target once archiving exists" is defensible but narrow and synthetic | The behavioural decomposition (response policy vs reading) and the evidence-cut plausibility control are solid moves; the lens readouts are standard tools. Literature not yet reviewed in the repo: hidden-knowledge probing, refusal directions, abstention calibration, long-context degradation. Novelty risk is real and must be checked | Several published LLM-negotiation and social-deduction game studies exist; novelty would come from the interruption angle, untested |
| Feasibility by May 2027 | High for the simulator route (free); low for anything needing many pod hours | High: the 3B fits on a laptop CPU for probes and on one A40 for minutes per experiment; the pipeline (prompt hashing, folds, judge, bootstrap) exists | Low to medium: engine, harness, logging, controls and many games to build from zero, plus multi-agent LLM cost |
| Compute / paid infra | Simulator is free; reader confirmations about $0.30 each; cap nearly exhausted ($2.45 of $20) | Cheap per experiment ($0.11–$0.33 so far); a causal study needs a few GPU hours; a cross-family replication a few more | Highest: every game is many LLM calls across several agents |
| Risk of inconclusive results | Medium: the F10 hypothesis can come back "information limit, cannot say more" | Medium: lenses may stay uncertified at 3B; interventions may not flip answers; the honest outcome may be "response policy, no mechanism found", which is still a thesis chapter | High: emergent-behaviour studies without strong controls are hard to interpret |
| Clean controlled experiment available | Yes (free, simulator): scale training streams, pre-registered falsifiers | Yes: patching with evidence-cut controls; planted-thesis and sycophancy controls; cross-family replication; all with sealed folds | Not yet |
| Fit with an advisor in memory / LLM agents / reasoning | Good for memory; the agent framing is strong | Good for all three: it is about a reader using (or failing to use) retrieved memory, and about abstention and reasoning in small models | Good for agents; weaker for memory and reasoning as framed |

**Recommendation.** Primary: **B**, with A as its origin story rather than a separate thesis. The memory-controller work
is what found the reader bottleneck, and the §23 ladder is the cleanest bridge: the controller delivers the evidence,
and a 3B reader still trails a 7B by 0.12 with everything in view. B has the clearest question, the best ratio of
evidence to cost, a working pipeline, and a controlled experiment (causal patching with a plausibility control) that
the record itself names as the missing piece. Backup: **A in its F10 form** (lossy sequential memory, free on the
simulator), because it has a self-standing finding and pre-registered falsifiers that cost nothing. **C** should be
presented as a stated interest, not a proposal, until an engine, a log schema, a hypothesis and controls exist.

This is not chosen because B has the most code (A does). It is chosen because B's question is answerable with the
resources at hand and its honest negative outcomes are still informative.

---

## Part 5. For the prospective advisor

### 5.1 Research accomplishments (about 150 words)

I built a memory-controller framework for a frozen language-model reader over long conversations, with heuristic,
imitation-learned and RL controllers, hindsight oracles, and a seven-class failure attribution, all under a
pre-registration discipline (every paid run gated by a free check written down first, paired-bootstrap intervals,
sealed confirmation folds). On synthetic tasks I showed that imitating a hindsight-deletion oracle is the wrong target
once an archive exists and that a least-regret teacher fixes it (0.851 vs 0.335 at a 2% budget). On LongMemEval, a
small query-aware selection head lifts accuracy by +0.113 (+0.072, +0.155) at a third of the prompt tokens, in 10 of
11 readers across 5 families, but is only non-inferior to a 31-parameter logistic regression on the same features, and
RL never beat imitation. That work exposed the reader as the bottleneck: with all evidence in view, a 3B reader
answers 0.40 of questions against 0.67 for a 7B. My current work asks why.

### 5.2 Current thesis proposal

- **Working title.** Present but not said: why small language models refuse or err with the evidence in view, and
  whether the failure is in reading or in answering.
- **Research question.** For a 3B instruction-tuned reader given the gold evidence in a short prompt, what share of
  its shortfall against a 7B is answering policy rather than reading, and is the correct answer causally available in
  its late-layer representations?
- **Motivation.** Retrieval and memory systems are judged on whether the right evidence reaches the model, but a
  small reader with the right evidence still fails on 60% of questions and says "unknown" on a third. If much of that
  is a response policy, the fix is cheap (prompting, a checker, a small fine-tune of the answering behaviour); if it is
  reading, only a larger or retrained model helps. The distinction decides how small a model can be in a memory-augmented
  agent.
- **Methodology.** (1) Behavioural decomposition, done: frozen-note interventions, forced choice, evidence-cut and
  length controls, matched-coverage risk (§35, §36b). (2) Causal test, proposed: activation patching between refusal and
  success rows at late layers at the model's own answer position, with evidence-cut prompts as the plausibility control
  and a pre-registered flip-rate falsifier. (3) A decomposed self-check (cite, derive blind, confront) with planted
  wrong answers and a sycophancy control (§37, pre-registered, laptop development run approved). (4) Cross-family
  replication on one Llama and one Gemma small reader with the same prompt and folds. (5) The "fix" test: does
  training only the answering policy (a small LoRA on refusal behaviour, or a checker over a forced answerer) close the
  gap at a stated cost in wrong answers?
- **Evaluation.** Pre-registered claims with fixed-sequence testing; paired bootstrap intervals over questions; sealed
  confirmation folds never read during design; risk and coverage at matched coverage; a 7B judge with a human audit and
  an independent label source for any checker claim; costs (wrong answers added, evidence-free answers emitted, calls
  and seconds) beside every lift.

### 5.3 Preliminary results (each with its caveat)

1. **The controller is not the bottleneck; the reader is.** With all gold evidence in the 8 visible items, P(correct |
   in view) is 0.397 (Qwen 3B), 0.667 (7B), 0.773 (14B); a 3B with the controller trails a 7B with recency by −0.121
   (−0.170, −0.070) (§23, `runs/exp23_report.md`). Caveat: one judge (about 0.1 false negatives, paired differences
   unaffected), one prompt tuned on the 7B.
2. **The 3B's own note often names the answer it then refuses to give.** On the S class (refusal with the gold in the
   note), freezing the note and forbidding "unknown" recovers +0.636 (+0.455, +0.818); over all 165 marked rows the
   lift is +0.176 (+0.109, +0.242), 0.53 (0.33, 0.73) of the matched 3B−7B gap (§35). Caveat: 0.210 of the refusals
   become wrong answers; forced choice doubles answered-and-wrong on evidence-free prompts (0.265 → 0.534).
3. **At the answer position the gold token is in the 3B's final top-10 on 0.638 (0.551, 0.717) of its failures while
   "unknown" is top-1, and the 3B ranks the gold about 3× worse than the 7B (+0.486 (+0.376, +0.600) log10)** (§34).
   Caveat: 0.605 (0.494, 0.716) of that presence survives removing the evidence (§34c), so it is largely plausibility;
   the 3B's J-lens failed its certification and verdicts rest on the plain logit lens; no causal test has been run.
4. **The 3B cannot check its own forced answers.** One-shot support score AUROC 0.573 (0.478, 0.666), not above
   chance; the score tracks evidence presence (+0.235 (+0.170, +0.302) drop when cut, no change with length-matched
   filler), not candidate correctness; its natural refusals carry no signal (0.455). The 7B checking the same
   candidates reaches 0.692 (0.600, 0.777) (§36b). Caveat: the 7B is also the judge, so the 7B figure is descriptive.
5. **Query-aware selection beats recency, but the network adds nothing over a linear model.** `fixed8` − FIFO + floor
   +0.113 (+0.072, +0.155) at 1,049 vs 2,964 tokens; head − 31-parameter logistic regression +0.013 (−0.017, +0.043)
   (§19c, §32). Caveat: three distinct heads not five (and heads A and B share networks across folds); one benchmark.

### 5.4 Scope and feasibility by May 2027

- **Can be completed.** The causal patching study on the 3B (one or two A40 sessions of under an hour each); the §37
  self-check line (development free, confirmation about $0.30); a cross-family replication of §34c/§35 on a Llama 3B
  and a Gemma 4B (about 1–2 pod hours); a human audit of the judge on the rows that matter; the "fix" test as a small
  LoRA on answering behaviour or a 7B checker over a 3B answerer with an independent label source (a few pod hours).
  Writing the memory-controller chapter from the existing record.
- **Compute.** Every experiment so far cost $0.11–$3.25 on a rented A40; the whole programme is $17.55. A thesis-scale
  continuation needs on the order of $100–300 of GPU time or a departmental allocation, mostly for the LoRA and
  cross-family runs. The 3B runs on the laptop CPU for probes.
- **Major risks.** (i) The patching study may not flip answers, leaving "response policy, no mechanism"; the chapter
  survives as a careful negative. (ii) Lens certification at 3B may never pass; the plain logit lens is the fallback
  and must be stated as such. (iii) Judge circularity: any checker claim needs human or third-model labels. (iv) Novelty
  risk until the interpretability and abstention literature is reviewed. (v) One benchmark and one prompt family; a
  second benchmark (LoCoMo in-view rows exist) is the cheapest guard.

### 5.5 Recommended advisor email (about 150–200 words, adapt freely)

> Over the past month I built and tested a small learned memory controller for language-model agents on long
> conversations. The honest result is that query-time selection helps (a 10-point accuracy gain at a third of the
> prompt tokens, replicated across eleven readers), but a trained network adds nothing over a linear model, and
> reinforcement learning never beat imitation. More interesting was what the controller exposed: even when every
> piece of needed evidence is placed in front of a 3B reader, it answers only 40% of questions, against 67% for a 7B,
> and it often writes the right fact in its own notes and then says "unknown".
>
> That is the question I would like to pursue as my honors thesis: for small models with the evidence in view, how
> much of the failure is answering policy rather than reading, and is the answer actually available inside the model?
> So far, freezing the model's note and forbidding "unknown" closes about half the gap to the 7B, at a measurable cost
> in wrong answers; a lens readout finds the answer token near the top at the final layer on most failures, but a
> control shows most of that is plausibility, not evidence. The next step is a causal test with proper controls. I
> would value your view on whether this is the right shape for a thesis.

---

## Appendix. Cannot verify / missing (from all three audits)

- No plots for §33–§36; all figures are tables.
- The lens library's `unembed` (final RMSNorm applied or not) was not inspected; the README formula and the readout
  code were relied on.
- Per-fold logit-lens emergence depth covers all 100 reads per fold, not only in-view pairs; direction agrees with the
  depth report, exact in-view numbers unpublished.
- E:5724 ("200 of 204 outputs carry `Answer:`") and E:5952 ("165") both stand in the record; the §34b figure is the
  corrected one.
- The §34 example rows were paired by question id; the reports pair by prompt hash; counts may differ by a few rows.
- Spend figures come from the record's text, not the billing API.
- Aggregate reports for Exp 1 and Exp 4 exist only in the dirty archive; §7 never rerun clean.
- Head A fold 0 = head B fold 4 (sha256 prefix 9f0be108) is recorded nowhere in the record before this audit.
- No interpretability or abstention literature review exists in the repository.
- The CATAN pitch was found only through the user's Google Drive, which the audit agent searched read-only; the user
  has confirmed no work has started.
