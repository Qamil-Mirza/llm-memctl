# §48 (side track): one learned memory controller with the full menu, under a storage cap

**Status: FROZEN at rev 0c (user, 2026-11-01, D-62): not built; to be picked up later from the GitHub issue.**

**Status before freezing:** DRAFT rev 0c (Advisor, 2026-11-01; rev 0a: the controller allocates the context window every turn and may retrieve on any turn; rev 0b: prompt cap 3,000 tokens from the measured context-rot onset; rev 0c: the user's decisions on §9 and the Advisor's self-review fixes, D-61).**

**Status line of rev 0:** DRAFT rev 0 (Advisor, 2026-11-01). Not pre-registered; nothing is built until the user and the review agree
the text.** Side track under D-60: it does not change the thesis scope (D-45) or any thesis claim.

## 1. Why

No controller in this programme was ever trained with the whole menu of memory operations at once:

| work | data | what a learned controller could do |
|---|---|---|
| Exp 4 | synthetic | keep / evict |
| Exp 4b, D1, D2 | synthetic | evict, archive, retrieve |
| Exp 3 | synthetic | all eight operations, **hand-written rules only** (no learning) |
| §14 | LongMemEval | a learned "needed later?" model deciding what moves to the archive |
| §15–§19 (the selector) | LongMemEval | retrieve only; "archive every turn" fixed |
| §20–§22 | LoCoMo (§21 lossy task) | which turns go into one 100-token note per session |

On LongMemEval and LoCoMo as usually posed (free unlimited archive, questions after the stream), the extra operations
have nothing to decide: query-blind eviction cannot beat "archive everything, retrieve at the question" (§14a, F8).
A full-menu controller trained there would most likely relearn the selector. The question only has content where
memory is bounded and questions arrive during the stream. §21a showed that such a task on real conversations has
headroom (oracle − best rule +0.177 (+0.126, +0.224)) that no learner closed (F10).

**Question.** Under a fixed storage cap on real conversations with questions during the stream, does a single learned
controller that may keep, evict, archive, retrieve, compact, compact-and-archive and consolidate beat the same
controller restricted to archive and retrieve, at the same cap?

## 2. What already exists (no new infrastructure claimed)

- Operations: `memctl/memory/engine.py` implements KEEP, EVICT, MOVE_TO_ARCHIVE, RETRIEVE_FROM_ARCHIVE, COMPACT,
  COMPACT_AND_ARCHIVE, CONSOLIDATE, NO_OP (PROMOTE/DEMOTE/PIN/UNPIN/UPDATE/SUPERSEDE have no handler; out of scope).
- Compression: `memctl/memory/compress.py` (Truncate, Extractive, LLM compressor; Dedup and LLM consolidator).
- Search ("vector store"): `memctl/retrieval.py` lexical (BM25), dense (sentence-transformers, bge-small), fusion.
  The archive is searched in memory; no external vector database is needed at LoCoMo scale.
- Task: the §21 lossy LoCoMo stream (`memctl/analysis/lossy_gate.py`), questions asked d = 1–6 sessions after their
  last evidence, scripted reader (correct iff the evidence is in view).
- Training: DAgger imitation and PPO/GRPO loops in `memctl/rl/`; per-write hindsight credit from §22.

## 2b. Mechanics: the actions, input, output, training and evaluation (requested by the user before any build)

**Where an item can be.** ACTIVE (in the prompt: the reader sees it), ARCHIVE (stored, searchable, unseen until
retrieved), DELETED (gone). Caps: ACTIVE ≤ A tokens, ARCHIVE ≤ S tokens.

**The discrete actions** (one per active item per step, except consolidate and retrieve):

| action | moves | effect | cost / risk |
|---|---|---|---|
| KEEP | stays ACTIVE | nothing changes (the engine only checks the item exists) | spends prompt tokens every step |
| EVICT | ACTIVE → DELETED | gone for good | cannot be undone |
| ARCHIVE | ACTIVE → ARCHIVE | out of the prompt, still searchable | spends storage; refused if storage is full |
| COMPACT | ACTIVE → shorter ACTIVE copy; original DELETED | the item is rewritten shorter (extractive compressor in the free stages) | the detail a later question needs may be cut |
| COMPACT_AND_ARCHIVE | short copy stays ACTIVE; original → ARCHIVE | a stub in view, full text retrievable | spends both tiers |
| CONSOLIDATE (set action) | ≥ 2 ACTIVE items → one merged ACTIVE item; sources DELETED (or ARCHIVED) | merges restated or related items | merged item can be long or lose a fact (Exp 3) |
| RETRIEVE (any turn) | ARCHIVE → ACTIVE | brings stored items back into view, up to k = 8 per turn, within the ACTIVE cap | prompt tokens; a wrong pick wastes the slot |
| **DROP_FROM_ARCHIVE (new, build item)** | ARCHIVE → DELETED | frees storage | the engine's EVICT acts on ACTIVE only today, and archiving is refused when storage is full, so a storage cap needs this |

**The ACTIVE tier is the reader's context window.** It is never emptied by rule (unlike the selector's `keep_none`): on every turn the controller receives the full list of what is in the window and decides, item by item, what stays. What survives, plus anything retrieved, plus the new turn, is exactly what the reader sees on that turn.

**Input to the controller at each step** (a new turn has just arrived and is always placed in view):
- per active item: the existing item features (`memctl/features.py`, 24 plus `bridge_score` in version 2: age, tokens, source, fidelity, access counts,
  similarity to the current turn and to other items, specific-token density, retrieval score, bridge score) plus new
  ones: is-compacted, is-consolidated, times retrieved;
- global: active and archive fill against their caps, step / horizon, whether a question is pending;
- every turn: a shortlist of up to 32 archive candidates from fusion search, queried with the new turn (and, at a question, with the question text), with their features.

**Output:**
- at an ordinary step: one action per active item from {KEEP, EVICT, ARCHIVE, COMPACT, COMPACT_AND_ARCHIVE}, plus at most
  one CONSOLIDATE over candidate pairs (pairs proposed by similarity above a threshold, so the choice stays small),
  plus DROP_FROM_ARCHIVE when storage would overflow;
- every turn: which ≤ 8 of the shortlisted candidates to RETRIEVE into the window (possibly none), the selector's head extended to every turn; at a question this is the last decision before the reader answers;
- if the chosen actions break a cap, the harness's fallback removes oldest-first and counts each as a forced removal.

**Network:** the same small DeepSets-style scorer as Exp 4/§19 (per-item MLP with pooled context), with a 5-way action
head per item, a pairwise consolidate score, and the retrieval logit. R1 uses the same network with the non-archive
outputs masked.

**Training:**
1. Hindsight labels from the stream: for each item, which later questions it serves and how many tokens it costs.
2. Teacher: greedy hindsight plan (questions served per stored token, D1 least-regret recipe), for the full menu and
   for the restricted menu. The teacher may retrieve ahead of a question (an item is worth bringing into view on the
   turns before a question that needs it); the controller can learn this only from cues visible at the time (for
   example the topic returning), and the share of its retrievals made before a question is reported.
3. Imitation (DAgger): roll out the current policy, ask the teacher what it would do at the states the policy
   actually visits, add those labels, retrain; a few rounds. Loss: cross-entropy on each item's action, listwise loss
   on the retrieval picks.
4. Optional RL stage only if declared before stage 1 (F5: RL never beat imitation here).
5. Cross-fitted: leave-one-conversation-out within the 6 development conversations.

**Evaluation:**
- Scripted reader (free): a question is correct iff its evidence is in view at the question (compaction caveat §6).
- Primary metric: accuracy on old-evidence questions (evidence ≥ w sessions back), conversation-clustered 95% CI.
- Secondary (descriptive): tokens in view at the question, storage used, forced removals, share of each action,
  accuracy by evidence age.
- Read once on the 4 held conversations after all choices are committed; stage 2 adds a real reader (paid).

## 3. The task (changes from §21 in bold)

- LoCoMo categories 1–4, 10 conversations, questions during the stream as in §21 (d by hash of question id).
- **Two tiers with caps instead of §21's fixed rule.** Active memory (what the reader sees besides retrieved items) has
  a cap A tokens; the archive has a cap S tokens, much smaller than the history. Raw turns arrive into active memory;
  nothing leaves except by a controller action or by the harness's budget fallback (counted as forced removals).
- **Every turn the controller may apply any operation in §2 to any active item and retrieve up to k = 8 archive items
  (fusion-search shortlist queried with the new turn). A question is a turn whose answer is scored; retrieval is not
  reserved for it.**
- Caps are declared before any gate read (rev 0b, user 2026-11-01: a prompt cap below the window, because context
  rot is already measured here).
  - **Prompt cap A = 3,000 tokens (primary).** Source: §13a / F7, measured before this proposal: the 7B reader's
    accuracy peaks near 3k prompt tokens and falls beyond it on both benchmarks (LoCoMo 0.299 at 2.9k → 0.273 at
    7.4k; LongMemEval 0.438 at 3.2k → 0.372 at 5.4k; "unknown" with all evidence in view 0.079 at 399 tokens →
    0.296 at 6,223), matching LongMemEval's own report of an 8B reader dropping past about 3k retrieved tokens. The
    reader's window (32,768 tokens, Qwen2.5 config) is the hard physical limit, not the operating cap: a whole LoCoMo
    conversation is about 11k–22k tokens (words × 1.35, measured 2026-11-01), so a window-sized cap would never bind
    and the scripted reader would score "keep everything" as perfect. The scripted reader cannot see context rot;
    the cap carries that knowledge into the free stages.
  - **Secondary caps A = 1,500 and 6,000 tokens** (free stages only, descriptive), to show whether any advantage of
    the full menu grows or shrinks with memory pressure. Not a tuning knob: the primary claim is read at 3,000 only.
  - The 3k onset was measured for one reader (Qwen2.5-7B-Instruct) and one prompt family; stage 2 uses the same
    reader, so the cap and the reader agree.
  - **Storage cap S:** chosen so that archive-everything overflows on every conversation (S ≈ 10% of the median
    conversation's tokens), the exact value fixed on development conversations from token counts only, before any
    accuracy is computed.

## 4. Arms

- **R0 archive-all (FIFO overflow):** archive every turn except the new one; when the archive is full, drop the
  oldest. Every turn, retrieve by the §19-style scorer from the shortlist up to the prompt cap. This is "the selector
  under a cap", with the same every-turn retrieval as the learned arms so the comparison isolates the menu.
- **R1 restricted learned:** the learned controller with only archive / evict-from-archive / retrieve.
- **F1 full learned:** the same network and training with the full menu.
- **H rules:** Exp 3's best full-menu rule (salience + compact-and-archive), §21's first-turns note rule adapted to caps.
- **O oracles:** a hindsight teacher for the full menu (O-full) and for the restricted menu (O-restricted). Exact
  optimisation is combinatorial; the teacher is greedy by questions-served per stored token, the D1 least-regret
  recipe, stated as a heuristic upper reference, not a true optimum.

## 5. Stages and gates (each read once on its declared data)

**Stage 0, free (CPU, scripted reader): is there headroom for the menu?**
- Read: O-full − O-restricted on old-evidence questions, conversation-clustered CI.
- Gate: ≥ +0.05 with lower bound > 0. Below it: the menu adds nothing even with hindsight at this cap; document and
  stop (D-45 stopping rule), with the cap stated as the scope.

**Stage 1, free: does the learned full menu beat the learned restricted menu?**
- Training: imitation of the matching teacher (DAgger), cross-fitted leave-one-conversation-out, as §21's `learned`.
  An RL stage is optional and not expected to help (F5: RL never beat imitation); if run, it is declared before.
- Claim C1: F1 − R1 > 0 (95% CI, conversation-clustered). Claim C2 (descriptive): which operations F1 uses, by share
  of decisions and by evidence age. Claim C3 (descriptive): F1 − R0.
- Not a claim: F1 against the oracles beyond reporting the gap closed.

**Stage 2, paid, separate pre-registration: real reader.** Qwen2.5-7B-Instruct (or the 3B) answers on the stage 1 arms,
the LLM compressor as a compaction variant. Quote, hard stop and the two-pane approval before any pod (D-40 cap $30;
spend ≈ $22.00).

## 6. The open measurement problem (must be settled before stage 0)

The scripted reader marks a question correct iff its evidence turns are in view. Compaction and consolidation keep
part of a turn, so "in view" is undefined for a compacted turn. Options for review:
- (a) **Answer-string survival:** a compacted item counts as carrying the evidence iff the normalised gold answer
  string survives in it. Measure first, on development conversations, how often the gold string appears in its own
  evidence turn at all (if rarely, (a) is unusable).
- (b) Extractive compaction only, by whole sentences, with sentence-level evidence derived from (a) on the raw turn.
- (c) Skip the scripted reader for compaction arms and run them only in stage 2 (paid).
Recommendation: (a) with the coverage check first; if coverage < 0.7, (b).

Exp 3's confound applies: an extractive compressor that keeps "specific" sentences is near-lossless on text built
from such sentences. LoCoMo is natural text, but the result is a statement about the compressor used.

## 7. Discipline

- Development: 6 conversations; held: 4 conversations read once after all choices (cap, features, teacher, arms) are
  committed. Leave-one-out cross-fitting stays within the development set until the held read.
- LongMemEval composed episodes (§21's planned corroboration row) as a second free check, declared before.
- Commit and hash this text before any gate run; a stub run of the full pipeline before the first real read.

## 8. What would change the plan

- Stage 0 fails: the side track closes with the documented cap; no stage 1.
- §6 coverage too low for (a) and (b): stage 1 runs without compaction arms and says so.
- C1 fails: report "under this cap the full menu adds nothing learnable over archive and retrieve".

## 9. Decisions for the user (before review)

1. **Decided (D-61).** Prompt cap A = 3,000 primary (1,500 / 6,000 descriptive). Storage cap S = 10% of the median
   development conversation's tokens, computed from token counts only before any accuracy is read, so archive-everything
   overflows on every conversation.
2. **Decided (D-61): option (a), answer-string survival,** with the coverage check first on development conversations;
   if coverage < 0.70, option (b).
3. **Decided (D-61): stage 2 is wanted.** It still needs its own pre-registration, a quote from measured timings, a
   hard stop, and the two-pane approval before any pod; nothing about the pod is pre-approved.

## 10. Advisor self-review (rev 0c, before the build hand-off)

- **Scoring with several evidence turns.** A question is correct only if *all* its evidence is in view (as §21);
  under answer-string survival, a compacted or merged item counts for an evidence turn only if the gold string survives
  in it. For multi-evidence questions the gold string may sit in one turn only; the coverage check reports this case
  separately and it is declared, not repaired after the read.
- **Every-turn retrieval cannot exceed the cap.** Retrieved items count against A on the turn they arrive; the
  controller's own picks are cut to fit before the harness fallback, and forced removals are reported per arm.
- **The teacher is a heuristic, not an optimum** (greedy by questions served per stored token). "Oracle" in tables
  means this teacher; stage 0's gate compares two heuristic teachers on the same footing, and that is the claim.
- **LongMemEval corroboration** uses the existing composed mid-stream episodes (§14–§15), same caps, declared before;
  it does not decide any claim.
- **Build items for the Executor (nothing else is new):** DROP_FROM_ARCHIVE handler; capped two-tier environment on
  the §21 stream with every-turn shortlist; per-item 5-way action head, pairwise consolidate score and every-turn
  retrieval head; full-menu and restricted teachers; DAgger loop with the masked restricted variant; coverage check for
  answer-string survival; stub run of the whole pipeline with timing of one fold (for the stage-2 quote and the ETA).
- **Stopping rule (D-45 spirit):** if stage 0 fails, document the cap as scope and stop; no rescue variant.
