# Scientific state (as of 2026-10-10)

The current, defensible state of the memctl project. Every claim points to a section of
[EXPERIMENTS.md](../EXPERIMENTS.md), which is the record. If this file and EXPERIMENTS.md disagree,
EXPERIMENTS.md wins, and this file should be fixed. Only the Advisor edits this file
(see [RESET_HANDOFF.md](../RESET_HANDOFF.md)).

## In one paragraph

The project asked whether a learned memory controller helps a language model answer questions over long
histories. On synthetic tasks a learned controller can beat fixed rules. On real conversations
(LongMemEval, LoCoMo), the only thing that clearly helps is **query-aware selection at question time**.
That means a small head that picks 8 of the 32 BM25 candidates for each question (`fixed8`, §19). It
beats FIFO + floor by about +0.10 accuracy at a third of the prompt tokens, and the lift holds in 10 of
11 readers from 5 families. But the head is **not shown better than a 31-parameter logistic regression on
its own inputs** (§32). What carries the result is the feature set and the evidence labels, not the
network. RL never beat imitation (RQ3 closed). A reader-side utility signal did not help (§29). The
frozen reader is the ceiling: with perfect evidence, a 3B reader still trails a 7B reader.

## Strongest supported findings

| # | Finding | Evidence | Strength |
|---|---|---|---|
| F1 | Query-aware selection beats recency. `fixed8` − FIFO + floor ≈ +0.10 to +0.11 accuracy on LongMemEval (470 answerable), at about 1,049 vs about 2,960 prompt tokens | §19c +0.113 (+0.072, +0.155); §23 +0.102; §26, §27, §29, §32 pairing checks +0.081 to +0.104 | Replicated on 4+ pods and 2 hosts |
| F2 | The lift holds across reader families | §23–§24: CI > 0 in 10 of 11 readers (Qwen 3/7/14B, Llama 3/8B, Gemma 4/12B, Phi-4-mini/14B, Granite 8B); Granite 2B is the exception | Pre-registered |
| F3 | The learned network adds nothing measurable over a linear model on the same 30 features | §32: head − LR +0.013 (−0.017, +0.043) with a reader; −0.016 gap (−0.036, +0.002) without one; head − zero-shot cross-encoder +0.013 (−0.030, +0.053). Only BM25 order is clearly worse (+0.068) | Pre-registered; NON-INFERIOR, not better |
| F4 | Training on what the reader finds useful does not beat evidence labels | §29: utility − `fixed8` −0.013 (−0.051, +0.026); blend −0.002; the utility head picks longer turns (1,881 vs 1,049 tokens) | Closed negative |
| F5 | RL does not beat imitation | §16 GRPO +0.0125 (−0.005, +0.033), gate FAIL; §22/22b/22c per-write credit beats uniform credit (+0.015, +0.036), but never beats the supervised writer (−0.014, −0.027, −0.001) | RQ3 closed (user, 2026-10-07; again after §22c) |
| F6 | The reader is the ceiling | §23: `fixed8` at 3B vs FIFO at 7B −0.121, closing only 56% of the gap. P(correct given all evidence in view) rises with reader size. A controller bridges one size step only for Qwen 7B→14B and Gemma 4B→12B | Pre-registered |
| F7 | Context rot: more evidence hurts beyond about 3k tokens | §13a onset; `fixed16` − `fixed8` §19c −0.057 (sig.), §19d −0.026 (not confirmed), §23 7B −0.051 (sig.) | Mixed; treat as likely, not settled |
| F8 | Query-blind eviction cannot beat "archive everything, retrieve at the question" on these benchmarks | §13, §14 (needed-item AUC 0.62–0.74) | Closed negative |
| F9 | Synthetic tasks: least-regret imitation fixes archive-aware control; an oracle warm start poisons RL; no fixed rule wins on both tasks | D1 (0.34 → 0.85 at 2%), GRPO section, §5, D2 | Clean rerun at `ef57fc4` |
| F10 | Lossy, sequential memory has real headroom no learner closed | §21 oracle − best rule +0.177 (+0.126, +0.224); oracle − supervised +0.139; §22c oracle − (ii) +0.142 | This is the opening for the pivot |

## Caveats that travel with every number above

1. **Checkpoint duplication (§32, applies to §19–§32).** The §19 head-A checkpoints for folds 1 and 2
   are byte-identical, and so are folds 3 and 4 (sha256 `2825193…`, `849e524…`). The cause is the half-A
   dealing in `memctl/splits.py`. It is not a leak (0 of 200 overlap with the test part), but `fixed8`
   is **3 distinct heads, not 5**. Its per-fold spread (0.650–0.750) understates training variance.
   Paired bootstraps are unaffected. The same holds for head B and for `lr_f1 = lr_f2`.
2. **Reader noise floor (§29b, §32c).** vLLM at temperature 0 is not batch-deterministic. Two passes
   over the same picks give different answer text on 35 of 470 questions, and 7 verdicts flip (6 from
   changed answers, 1 from the judge on an identical answer). Same-pick accuracy moves by about ±0.007.
   Across hosts it is about 5 in 50 answers (§15c). **Differences under about 0.01 are noise.**
3. **Judge (§18, §32 spot-check).** One Qwen-7B judge for §26–§32. The §18 Granite-8B check agrees on
   97.3% (κ 0.946) and keeps the ranking. In the §32 spot-check, a blind LLM grader agreed on 45 of 50.
   All 5 disagreements were judge false negatives, which the user ruled correct (samples 10, 12, 26,
   36, 39). False positives: 0 in 28. **Absolute accuracies are understated by about 0.1; paired
   comparisons are unaffected**. Per-type preference accuracies are the least trustworthy numbers.
   No frontier or second-family judge has scored §26–§32.
4. **One benchmark, one prompt, one reader for most claims.** The positive results are on LongMemEval,
   with a prompt frozen at `7b5fc30` and tuned on Qwen 7B. On LoCoMo, intervals resample 10
   conversations, so the clustered bootstrap undercovers. The head transfers to LoCoMo only as an
   exploratory result (§18, §19c).
5. **Selection before replication.** `listsum` (§15) and `fixed8` (§19) were picked after scoring
   several arms. Later replications do not remove that selection step.
6. **Label quirks (§32).** In an answer session with no marked turn, only the shortest turn is a
   positive (41 partly marked + 21 unmarked of 500). 6 of 500 episodes contain a turn longer than the
   budget.
7. **Provenance.** Experiments 1–6 were first run dirty (`d215c8f`) and replaced by a clean rerun at
   `ef57fc4` on 2026-10-05. All numbers reproduced except 9 cells of Exp 4b. Exp 7 was never rerun;
   only its reports survive in `runs/_archive_dirty_2026-09-30/`. The §12 fusion figures were inflated
   by 0.015–0.026 (corrected in §19a).

## What is closed and what is open

- **Closed:** RQ3 (RL vs imitation); adaptive k (§19b); write action (§20, gate fail); query rewriting
  (§28, non-inferior only); utility signal (§29); StreamMemBench transfer (§25, does not transfer:
  recency wins); UtilMem (§30) and test-time training (§31), both closed without a run; LRE as a
  baseline (§27, worse than FIFO when it fills the head's slots).
- **Open:** see [OPEN_QUESTIONS.md](OPEN_QUESTIONS.md). The candidate pivot (sequential memory
  management and delayed credit assignment) builds on F10 and F5. No new architecture has been chosen.

## Spend

About $13.75 for all experiments to date. The last recorded total is $12.58 after §27–§28. Adding §29
(about $1.02) and §32 (about $0.15) gives $13.75 by arithmetic; no table yet records that figure. Every
paid run needs the user's advance approval (2026-10-09 rule: $20 programme cap, launch window and
conditions in EXPERIMENTS.md Methods). No RunPod resources existed at the reset (0 pods, endpoints or
volumes).
