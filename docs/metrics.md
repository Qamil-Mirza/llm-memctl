# Metrics

Every number in `metrics.json` is computed from `answers.jsonl` and `decisions.jsonl`
by `memctl/summary.py`, so each one can be traced to single questions and decisions.

## Answer quality

| Metric | Meaning | Code |
|---|---|---|
| Token F1 | Word overlap between the answer and the gold answer (lowercased, no punctuation or articles). As in LoCoMo, A-Mem and Memory-R1. | `metrics.token_f1` |
| BLEU-1 | Share of answer words found in the gold answer, with a penalty for answers shorter than the gold answer. | `metrics.bleu1` |
| Judge | A model reads the question, gold answer and given answer and replies CORRECT or WRONG. Set in the config under `judge:`. Outputs are cached. | `judge.py` |
| Accuracy | Share of answers that are right. "Right" means the judge said CORRECT, or F1 of at least 0.5 when no judge is configured. | `episode.ask` |

All four are reported overall and per category: single-hop, multi-hop, temporal,
open-domain, adversarial.

**Adversarial questions** ask about something that never happened in the conversation.
As in the official LoCoMo scoring, the answer is right only if the model says the
information is not there ("not mentioned" or "no information"). F1, BLEU-1 and the
judge are all set to 1 or 0 by this rule.

**The judge is a small local model** (`Qwen/Qwen3.5-2B` by default) and makes mistakes.
Read accuracy together with F1 and BLEU-1. A stronger judge can be set in the config.

## Failure attribution

Each wrong answer gets exactly one label (`attribution.py`):

| Label | Kind | Meaning |
|---|---|---|
| `evidence_dropped` | memory | A needed item had been deleted. |
| `evidence_archived_not_recalled` | memory | A needed item was in the archive and the agent did not recall it. |
| `evidence_in_store_not_retrieved` | memory | A needed item was in the store but not in the top-k search results. |
| `evidence_present_model_wrong` | reasoning | All needed items were in the prompt. |
| `no_evidence_label` | unknown | The benchmark lists no evidence for the question. |

If several evidence items were missing for different reasons, the most severe reason
wins: dropped, then archived, then not retrieved.

## Evidence availability

For every evidence item of every question: where it was when the question was asked
(in context, in the store and retrieved or not, in the archive and recalled or not,
dropped). Also the share of questions that had all their evidence in the prompt.

## Gap to oracle

`closed_gap = (score_method - score_keep_newest) / (score_oracle - score_keep_newest)`

0 means no better than `keep_newest`; 1 means as good as the oracle. Computed in the
report from runs with the same benchmark, questions, model, seed and budget. It is
"n/a" when the oracle does not beat `keep_newest`.

## Agreement with the oracle

After a controller decides, the oracle's choice for the same item at the same step is
written next to it in `decisions.jsonl` (`oracle_place`, `agrees_with_oracle`):

- `DROPPED`: no later question needs the item (`STORE` if dropping is switched off).
- `CONTEXT`: a later question needs it and the exact plan keeps it in context.
- `STORE`: a later question needs it but it does not fit in the exact plan.

Leaving an item in context is graded too: each controller call writes one `keep` row
listing the items it kept and which of them the oracle would have moved. Agreement is
reported overall, for moves, for keeps, and per oracle choice.

Rule controllers never drop anything, so they cannot agree on items the oracle drops.
That is expected; the per-choice breakdown in the report shows it.

The oracle follows the same move rules as every controller: a filed item never returns
to context, and dropped items are gone.

## Cost

**The cost axis on every chart is average prompt tokens**, not the budget `B`. `B` limits
only the context; retrieved and recalled items are added on top. Prompt tokens count
everything the model read.

Average and peak budgeted context tokens, average and peak prompt tokens (including
retrieved and recalled items), recalls and their token cost, store searches, controller
calls, Jev calls, API cost in dollars and wall-clock time. Tokens are counted as words
plus punctuation marks (see `docs/decisions.md`, entry 1).
