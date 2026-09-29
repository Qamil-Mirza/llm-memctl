# Decisions made where the brief was ambiguous

Each entry: what was unclear, what was chosen, and why. Newest last.

| # | Question | Choice | Why |
|---|---|---|---|
| 1 | How are tokens counted? | Words plus punctuation marks (`items.count_tokens`), not a model tokenizer. | Budgets then mean the same for every model, and tests need no download. Real prompt sizes in model tokens can be reported in Phase 2. |
| 2 | Do archive index lines use the budget? | Yes (`memory.charge_archive_index: true`). | Otherwise archiving everything would be free and would beat every method trivially. An index line is always cheaper than its item. |
| 3 | Do retrieved and recalled items use the budget `B`? | No. `B` limits `CONTEXT` plus the index. Retrieval is limited by `store_top_k`. Full prompt tokens are reported as a cost. | Simplest reading of "added to the prompt for that turn only". |
| 4 | What does `recall(id)` do to the item? | It is shown for that question only and stays in `ARCHIVE`. Cost per recall: `recall_cost_tokens` plus the item's tokens. | Mirrors how `STORE` retrieval works. |
| 5 | Can a controller move a filed item back into `CONTEXT`? | No. Only search and recall bring items back. | Keeps the memory model small. |
| 6 | What budget does `full_context` get? | Unlimited. Its real token use is reported. | It is the upper reference and never evicts. |
| 7 | `file_everything` must file items even when there is room. | Controllers may set `runs_on_every_arrival = True`. | Otherwise it would act like `full_context` until the first overflow. |
| 8 | What counts as an item being "used" (for LRU and features)? | It was retrieved, recalled, or new text was similar to it (cosine at least 0.3). | Dialogue has no explicit reads, so similarity stands in for a reference. |
| 9 | Which embedder? | Built-in hashing bag-of-words by default; any sentence-transformers model by config. | No download for tests. Real runs should use a dense model. |
| 10 | What is logged in `decisions.jsonl`? | Placements the controller returned. Items it left alone are counted in `candidates_considered`. | Logging every untouched item at every overflow would make files very large. |
| 11 | Files beyond the brief's layout | `embed.py`, `episode.py`, `metrics.py`, `benchmarks/types.py`, `benchmarks/synthetic.py`. | Each holds one idea and keeps the listed files short. |
| 12 | Jev key variable | Read `JEV_API_KEY` as the brief says (the SDK's own default is `TYPESAFE_API_KEY`). | Follow the brief; pass the key to the SDK explicitly. |
| 13 | "Tiny model" for the smoke test | A model-free `stub` backend that answers with the best-matching memory line. | Runs in milliseconds with no download. A real tiny model is added with the Hugging Face backend in Phase 2. |
| 14 | Same-day reruns of the same config | They overwrite the run folder. | Runs are deterministic and cached, so the content is the same. |
| 15 | Answerer model id | `Qwen/Qwen3.5-0.8B`, confirmed on the Hugging Face API on 2026-09-28. | The brief asked to check it. |
