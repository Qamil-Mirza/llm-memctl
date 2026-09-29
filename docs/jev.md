# Jev: what we found and how we use it

Source: the official TypeSafe skill (`typesafe-ai/skills`, plugin version 0.5.7,
installed locally) and the live docs at <https://docs.typesafe.ai>, read on
2026-09-28. Nothing here is guessed.

## Facts from the docs

| Topic | What the docs say | Page |
|---|---|---|
| Endpoint | `POST https://api.typesafe.ai/v1/systemone`, header `Authorization: Bearer <API_KEY>` | `/api` |
| Request | `state` (string, object or array), `model`, `questions` (map of id to question) | `/api` |
| Choice question | `{"type": "choice", "instructions": ..., "criteria": {option: description}}`, up to 255 options | `/primitives/choice` |
| Choice answer | `choice`, `probabilities` (sum to 1), `confidence` (0 to 1) | `/api` |
| Confidence | For a Choice: `(n * peak - 1) / (n - 1)`, where `n` is the number of options | `/confidence` |
| Model | `jev-latest` is an alias for `jev-1.13.0`; the response reports the exact version | `/models` |
| Price | $0.042 per million input tokens; output tokens are free | `/models` |
| Limits | 64k tokens per request; 32k for `state` plus the longest question; 1,200 requests per minute | `/models` |
| Python SDK | `pip install typesafe-sdk`; `TypeSafeClient().system_one(state=..., questions=...)` | `/sdk/python` |
| SDK key variable | The SDK reads `TYPESAFE_API_KEY`. We read `JEV_API_KEY` and pass it in explicitly. | `/sdk/python/api/constants` |
| Errors | 401 bad key, 422 bad request, 429 rate limit, 529 overloaded (SDK retries with backoff) | `/api` |

## How the placement is framed

Following the skill's guidance (named JSON fields in `state`, one narrow judgment
per question, options described so they are distinct, all questions about the
same state in one request):

- **One request per decision.** `state.memory_items` holds every candidate item
  with its text and features. Jev reads the state once.
- **One Choice question per item**, pointing at it with a backticked path:
  "Where should the memory item `` `memory_items.item_3` `` be kept so that the
  assistant can still answer later questions about this conversation?"
- **Options** are the places: `CONTEXT`, `STORE`, `ARCHIVE`, `DROPPED`, each with a
  description of what it costs and what it is for. `DROPPED` is left out when
  `allow_drop` is false.
- We store Jev's `choice` and `confidence` on each `Placement`.

Jev answers each question independently, so its choices can exceed the budget.
When that happens the controller evicts the items with the lowest
`P(CONTEXT)` first, each to its next most probable place. These decisions are
marked in their `reason`.

## Status

- `JEV_API_KEY` was **not set** when Phase 1 was built, so only `FakeJevClient` has run.
- `FakeJevClient` is a hand-written heuristic. It is not Jev and says nothing about
  Jev's quality. Runs need `--allow-fake-jev`; the controller name becomes
  `jev-FAKE`, the run folder and `metrics.json` say so, and every reason starts
  with "FAKE Jev".
- The real client is Phase 3. Until then a set key produces a clear error instead
  of silently using the fake.

## Logging

Each request is written to `jev_calls.jsonl` with the request body, response,
latency and cost. The API key is sent only in a header and is never logged.

## Open questions for Phase 3

1. A decision with many candidates may exceed the 32k `state` limit at large
   budgets. Plan: split candidates across several requests.
2. The docs note accuracy shifts as the state grows (`/model-jaggedness/jev-1.13`).
   Worth measuring against the number of candidates per request.
