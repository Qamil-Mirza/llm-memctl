# Cost per question: a draft for deciding whether cost is an evaluation axis

Draft of 2026-10-08. It uses existing run data only; nothing new was run or paid for. Every number carries its
source. **[measured]** is from our logs, **[derived]** is arithmetic on measured numbers, and **[assumption]** is a
stated price not measured by us.

## LongMemEval, per question (470 answerable test questions)

| arm | reader prompt tokens | controller compute per question | accuracy at Qwen 7B | A40, $ per 1,000 questions | API, $ per 1,000 questions |
|---|---|---|---|---|---|
| FIFO + floor (5%, 3k target) | 2,964 [measured, §23] | BM25 search: 102 ms CPU [measured] | 0.436 [measured, §23] | 0.091 [derived] | 0.59 [derived] |
| `fixed8` (§19 head, 8 of 32) | 1,049 [measured, §23] | BM25 + one forward pass of a 27k-parameter head: 122 ms CPU [measured] | **0.538** [measured, §23] | **0.032** [derived] | **0.21** [derived] |
| `fixed16` (§19 head, 16 of 32) | 2,569 [measured, §23] | as `fixed8`: 125 ms CPU [measured] | 0.487 [measured, §23] | 0.079 [derived] | 0.51 [derived] |
| `fixed8` + stored session notes (§20) | 1,348 [measured, §20] | as `fixed8`, plus 3 writer calls of about 2.2k input and 42 output tokens each [measured: calls and output; input from the mean LongMemEval session, 2,159 tokens] | not measured: the §20 gate failed, so no reader run | reader 0.042 + writer about 0.27 if every note is written fresh [derived: 3 × 2.2k input tokens] | reader 0.27 + writer about 1.32 if fresh [derived: 3 × 2.2k input tokens] |
| prompted LLM controller (§7b) | — | 9 calls per 40-step episode, 824 s CPU per episode (Llama-3.1-8B, synthetic recall task) [measured, §7b] | not measured on LongMemEval | — | — |
| OpenJev as selector, arm B (§26) | 2,373 [measured, §26] | 1 OpenJev call of about 13,800 input tokens; 13.1 s per call with 5 calls in flight, about 2.7 A40-seconds per question [measured, §26 phase 1] | 0.489 [measured, §26; `fixed8` 0.526 and FIFO 0.445 in the same session] | reader 0.074 + OpenJev about 0.44 = **about 0.52** [derived] | reader 0.47 + about 0.58 if it were TypeSafe's Jev at $0.042/M [derived, hypothetical] |
| OpenJev as memory manager, arm A (§26) | not measured | about 21.5 OpenJev calls of about 21,100 input tokens each; about 94 A40-seconds per question, one stream [measured, §26 smoke, 2 training conversations] | not run, by decision | OpenJev about **15.4** [derived, one stream] | about 19 if it were TypeSafe's Jev at $0.042/M [derived, hypothetical] |

**How the dollar columns were derived.**
- **A40.** In §23's Qwen 7B stage, the pod produced 1,500 answers in 618 s of wall-clock time at 15 concurrent
  requests (19:24:49–19:35:07 UTC) [measured]. That is $0.101 at $0.59/h for about 3.29M prompt tokens, or about
  **$0.031 per million prompt tokens, decoding included** [derived]. Each row is its prompt tokens × that rate.
- **Note writer on the A40.** `fill_cache` wrote 1,229 notes (mean input 1,714 tokens) in 531 s at 32 concurrent
  requests [measured, §20]. That is $0.087 for 2.1M input tokens, or **about $0.041 per million writer input
  tokens** [derived].
- **API [assumption].** **$0.20 per million input tokens** for a hosted 7–8B instruct model. That is a round
  figure in the range of public list prices for this class, not a quote from one provider. Answer tokens (up to
  about 100 per question) add about $0.02 per 1,000 questions at that rate, which is ignored.
- **Head on CPU.** The controller's extra cost over FIFO + floor is about 20 ms of laptop CPU per question
  [measured difference: 122 ms against 102 ms, both including BM25 search over about 500 turns]. At any cloud CPU
  price that is well under $0.001 per 1,000 questions [derived].

**Session notes in deployment.** A session note is question-blind and written once per session (§20). In a
deployed system its cost is paid once per session, not per question, so the per-question writer figure above is an
upper bound for a stream that asks many questions per session.

## Reading for the axis question

- **Accuracy per dollar.** `fixed8` is the most accurate arm at 7B and also the cheapest, at about a third of FIFO
  + floor's reader cost. The controller's own cost (a CPU forward pass) is negligible against the reader's. On this
  benchmark, adding cost as an axis strengthens the result rather than trading against it.
- **The axis matters for LLM-based controllers.** A prompted controller (§7b) or a note writer (§20) spends model
  calls of its own. The writer's per-question cost, if notes are written fresh, is several times the reader's.
- **Proposal.** Report cost as reader prompt tokens plus controller model calls and tokens, as here. Dollars are a
  derived column, at stated prices.
- **A decision-model controller is measured now (§26, OpenJev on an A40).** As a selector it costs about 14 times
  the head's whole arm (about $0.52 against $0.032 per 1,000 questions) and is less accurate (0.489 against 0.526).
  As a full memory manager it costs about $15 per 1,000 questions before the reader runs. The cost axis separates
  the controller families more sharply than accuracy does.

## JEV arm on LongMemEval: estimated API cost (draft; H6 is not unblocked)

**Setup assumed:** 500 test questions × 32 candidates, one Choice question per item, one batched request per
question (memctl/controllers/jev.py).
- **Price [assumption, from the code]:** **$0.042 per million input tokens, output free.** This is recorded in
  `memctl/controllers/jev.py` from docs.typesafe.ai/models, read 2026-09-28, and **re-read on 2026-10-08:
  unchanged** ("Price (per Btok / per Mtok) | $42 / $0.042", output tokens free; model listed as `jev-1.13.0`). The
  range below therefore stands.
- **Tokens per request [derived].** Each item sends its text (about 150 tokens; LongMemEval turns average about
  150 in our prompts), about 40 tokens of metadata fields, and about 70 tokens of question and criteria: about 260
  tokens per item. Over 32 items, with the state wrapper, that is **about 8.4k tokens per question**.
- **Total:** 500 × 8.4k is about **4.2M input tokens, about $0.18**. Allowing ×0.7–1.5 for JEV's own token
  counting: **about $0.12–0.27**.
- **Caveat.** The JEV HTTP client has never been run against the live API (stated in jev.py), and `JEV_API_KEY`
  is still owed. A first real call would also check the token accounting.
