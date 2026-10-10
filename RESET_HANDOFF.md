# Reset handoff (2026-10-10)

Read this first. It replaces every earlier Claude conversation. Nothing you need is only in a chat.

## The project in five lines

memctl studies **learned memory controllers** for LLMs answering questions over long histories. After
32 experiments, the one robust positive result is **query-aware selection at question time** (`fixed8`:
a small head picks 8 of the 32 BM25 turns). It gives about +0.10 accuracy over FIFO on LongMemEval and
lifts 10 of 11 readers. But it is **no better than a 31-parameter logistic regression on the same
features**, RL never beat imitation, and the frozen reader is the ceiling. The user is considering a
pivot toward **sequential memory management and delayed credit assignment**. No new architecture has
been chosen. Do not start one without the user.

## Read in this order

1. `research/STATE.md`: what is true now, with caveats (3 heads not 5; ±0.007 reader noise; judge
   understates accuracy by about 0.1).
2. `research/OPEN_QUESTIONS.md`: what is unresolved; Q1–Q5 are the pivot-relevant ones.
3. `research/DECISIONS.md`: what was decided and why. Do not re-litigate a decision without new evidence.
4. `research/EXPERIMENT_INDEX.md`: where every experiment's configs, runs and caches live.
5. `EXPERIMENTS.md`: the full record (about 5,900 lines); read only the sections you need.
6. `README_RESEARCH.md`: how to run things.

## Hard rules (from the user; they override any agent's judgment)

- **Spending:** ask the user before **any** credit spend (RunPod, APIs). Quote the price and a hard stop.
  Never leave a pod billing while you wait on a question. Before a paid launch: full-size stub run,
  peak RSS, cost from measured latency, and a resume test on a new endpoint. $20 programme cap
  (about $13.75 spent).
- **No held-out peeking:** design choices come from training-side data only. A labelled leak is still a
  leak. Pre-register criteria before any test-fold number exists.
- **Stay inside `~/Code/llm-memctl`.** Never alter anything outside it.
- **Never delete** `cache/`, `data/`, `external/` or anything under `runs/` without the user's approval.
  No force-push, no history rewrite, and no deleting remote branches.
- **Verify before claiming:** quote the commit hash and grep the new text before saying a change is in.
  Report failures with their output.
- **Never put claude.ai session links** in commits or PRs.
- **Reports to the user** are in plain, high-school language: what was found, why it matters, what next.
  Details go in EXPERIMENTS.md.

## Five panes, five owners

One writer per file. If you need a change in a file you do not own, message its owner (SendMessage) or
the Orchestrator. Never edit it yourself. Role prompts are in `research/roles/`.

| Pane | Owns (sole writer) | May read | Must not |
|---|---|---|---|
| **Advisor** | `research/STATE.md`, `DECISIONS.md` (only after the user decides), `OPEN_QUESTIONS.md`, draft pre-registrations in `research/proposals/` | everything | edit code, configs, runs or EXPERIMENTS.md; launch jobs; spend |
| **Executor** | `memctl/`, `tests/`, `configs/`, `scripts/`, new `runs/` and `cache/` entries, its own section of `EXPERIMENTS.md` (results only, in a pre-registered section) | everything | edit `research/*`; change pre-registered criteria after data exists; spend without a quoted user approval; work on `main` |
| **Reviewer** | `research/reviews/<date>-<topic>.md`, PR review comments | everything | edit code or conclusions; approve its own work. It signs off every pre-registration, every result section and every change to STATE.md |
| **Orchestrator** | `research/BOARD.md` (task queue, assignments, status), branch and PR bookkeeping | everything | edit science files; merge a PR without reviewer sign-off **and** user approval |
| **Research Monitor** | `research/MONITOR_LOG.md` | everything, plus read-only RunPod listing and billing | edit anything else; create or stop paid resources (it alerts the user and the Executor instead) |

**How a conclusion changes.** Executor result → Reviewer verdict in `research/reviews/` → user approval →
Advisor edits STATE.md and adds a DECISIONS.md row. No agent changes a conclusion on its own.

**Branches and worktrees.** One branch per task, named after the task, created from `origin/main`.
Worktrees go under `.worktrees/<branch>` (gitignored). **In every worktree, `runs/` and `cache/` must be
symlinks to the main tree's folders**, so results never exist only inside a worktree. This happened
before the reset: §29 and §32 results lived only in agent worktrees. Remove a worktree only after its
results are in the main tree.

## State of the tree at handoff

- Branch `reset-2026-10-10` (PR pending) holds this cleanup. `main` = `origin/main` at `62eed0d`, plus
  this PR once merged. PRs #1–#24 are merged and none are open.
- Local branches: `main`, `reset-2026-10-10` and `phase2-locomo-baseline` (no upstream; tag
  `phase2-complete`). All other branches exist on origin only.
- No worktrees. No RunPod pods, endpoints or volumes. No jobs running.
- `runs/` is 25 GB, with per-step traces gzipped (read transparently by `memctl.runlog.open_trace`).
  Rescued worktree data is checked against `runs/_from_worktrees/MANIFEST.sha256`. Deletions are logged
  in `runs/_deleted_2026-10-10.txt`.
- Tests: 279 pass and 2 fail. Both failures predate the reset: `tests/test_import_boundaries.py`, where
  `memory/engine.py` imports `memctl.retrieval`, plus an analysis-module boundary. These are for the
  Executor, as a first small task, if the user agrees.
- Some `__pycache__` folders are root-owned (from the Docker image). Removing them needs `sudo`;
  they are harmless.
- `.env` holds secrets. Never print it and never commit it.

## First moves after the reset (none started)

1. The Reviewer reads this PR and `research/STATE.md` against EXPERIMENTS.md, and flags any claim that
   overstates the record.
2. The Advisor drafts the pivot framing from Q1–Q4 as a **proposal**, not a build. The user decides.
3. The Orchestrator opens `research/BOARD.md` with only approved tasks.
