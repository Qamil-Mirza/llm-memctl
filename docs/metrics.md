# Metrics

Phase 1 implements the rows marked "now". The rest arrive in Phase 2.

| Metric | Meaning | State |
|---|---|---|
| Token F1 | Word overlap between the answer and the gold answer (lowercased, no punctuation or articles) | now |
| BLEU-1 | Unigram precision with brevity penalty, as in Memory-R1 and A-Mem | Phase 2 |
| LLM judge | A configurable, cached judge model | Phase 2 |
| Evidence availability | Share of needed evidence items that were in `CONTEXT`, retrieved from `STORE`, or recallable from `ARCHIVE` when the question was asked | now |
| Context tokens (average, peak) | Budgeted tokens after each event | now |
| Prompt tokens (average, peak) | Size of the full prompt per question, including retrieved and recalled items | now |
| Recalls, store searches, controller calls, Jev calls, Jev cost | Counts and dollars | now |
| Failure attribution | One label per wrong answer | Phase 2 |
| Gap to oracle | `(score_method - score_keep_newest) / (score_oracle - score_keep_newest)` | Phase 2 |
