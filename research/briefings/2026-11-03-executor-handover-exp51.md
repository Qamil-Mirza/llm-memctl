# Executor handover: §51 (2026-11-03)

You are the new **Executor** for this repository. You replace the previous Executor session.

That session stopped because its replies kept being blocked by a safeguard false positive (`reasoning_extraction`).
The trigger seems to have been its own long history about the harness's transcript logging. Nothing was lost, and no
process is running.

**Where to start reading:**
- this note;
- `research/STATE.md`;
- `research/DECISIONS.md`, especially D-72 and D-73;
- the §51 proposal, `research/proposals/2026-11-02-exp51-agent-weight-editing.md` (rev 1a; §13 is the binding build
  contract);
- the Advisor's review log, `research/reviews/2026-11-02-exp51-stage1-build-review.md`, read from the bottom up.

## Roles and how to talk to the Advisor

- **Advisor and reviewer:** the Claude session named **`llm-memctl-6d`**. Reach it with `SendMessage` (use
  `ListAgents` to find it).
  - It owns `research/STATE.md`, `DECISIONS.md`, `OPEN_QUESTIONS.md`, `proposals/`, `reviews/`, `audits/` and
    `briefings/`.
  - It reviews your builds and commits the freeze manifests (F0, F1, F2).
- **You** own the code (`scripts/`, `memctl/`, `tests/`), `runs/`, and `EXPERIMENTS.md`. Commit your own files by
  explicit path.
- **Codex** (tmux pane `SLM-Utilization-0:0.2`) reviews proposals on request.

## Standing rules (all binding)

- **No spending without the user's own approval line typed in your pane.** It must give the price and a hard stop, and
  it covers RunPod and the Anthropic API alike. An approval relayed by the Advisor is not enough.
- **Programme cap: $36 for RunPod (D-73).** About $24.21 is spent. Never two pods at once. Check `list-pods` is empty
  before and after a pod. Every pod needs the D-22 smoke test and the D-30 fresh-venv bundle test.
- **Editor budget (separate):** a $50 limit on the Anthropic Console workspace; the harness stops itself at $45.
  About $0.71 has been used by the two dry runs (the Console agrees).
- **No held-out peeking.** H and T-city stay sealed until F2 is committed. They are read once, by the guarded H+T pod.
- **Stay inside `~/Code/llm-memctl`.**
- **Never stage or commit `writing/`.** It holds the user's untracked thesis work.
- **No claude.ai session links and no attribution trailers** in commits or PRs.
- **`.env` holds secrets** (HF_TOKEN, RUNPOD_API_KEY, ANTHROPIC_API_KEY). Never print or commit it.
- **Report to the user in plain language:** what was found, why it matters, what's next.
- **Measure before quoting a time or a cost.**
- **D-45 stopping rule:** document a negative result; don't invent a rescue.
- **One editor run only.** A resume after an infrastructure or harness crash is the same run.

## §51 state

**Frozen.** F0 was committed by the Advisor at **d29263a** (`runs/exp51/freeze/F0_draft.json`, sha 3ed50681…).

**Post-F0 deviations, accepted by the Advisor and to be recorded in F1:**
- **D2** (d2f1cee): a tooling import fix only.
- **D3** (475be08, f0a97b9, 645c6ad):
  - `thinking.display: "summarized"` in API_CONFIG;
  - resume from the transcript and ledger;
  - a $0 scripted end-to-end test;
  - the 36-layer dummy;
  - clearer refusal wording, and `/work/` path mapping in `save_edit`;
  - a dry-run turn cap of 15;
  - a lost-call placeholder on resume;
  - a `transcript.pre_resume.<n>` copy;
  - `harness_pid.json` and `exp51_stop.sh`.

**Done:**
- **The laptop B0 on E:** 1,872 rows. B5 is fitted on E: ℓ* 34, with 284 abstained and 572 gold-answered rows.
- **The stage-4 two-pod chain:** pod S, then F2, then the guarded pod H+T. It is built and reviewed.
- **Two API dry runs** ($0.33 and $0.37, both passed):
  - run 1 checked caching and the meter;
  - run 2 checked the summarized display and resume.

**Uncommitted work-in-progress from the previous Executor (yours to inspect and finish; don't discard it blindly):**
- `scripts/exp51_freeze.py`: about 160 lines added, building `--point F1`. It includes the cost model and
  `cost_estimate`.
- `scripts/exp51_harness.py`:
  - `CONTEXT_STOP_TOKENS = 900_000`;
  - `Meter.context_tokens`;
  - the worst-case estimate now uses 2.5 chars per token, against a measured 2.9.
- `tests/test_exp51.py`: 9 lines added.
- `runs/exp51/freeze/F1_draft.json`: an early draft.

Run the tests and read `git diff` before building on it.

(`research/reviews/…build-review.md` is also modified. It is the Advisor's file; leave it alone.)

## Your next task: the F1 draft and a cost estimate for the real editor run

**What F1 must contain** (Advisor's request):
- the final bound values, with the F0 reduction rule applied to the measured laptop timing;
- the split sizes;
- D2 and D3, each with every source sha (old → new);
- the resume rule: "a resume after an infra or harness crash is the same run";
- the lost-call placeholder rule;
- the launch and stop procedure (the harness records its own pid and pgid; stop with `exp51_stop.sh`);
- the meter-validation record: dry runs at $0.6965 on the meter against $0.71 on the Console. The gap is one in-flight
  call at the deliberate kill.

**The cost estimate** for the real run:
- build it from both dry runs' per-call costs, scaled to the F1 turn bound (≤ 120) and to the context growth expected
  from real E tool results;
- give an expected figure, a high figure, and the $45 harness stop.

**Open question the Advisor will rule on with F1: refusals.** The harness ends the run on a `refusal` stop reason, and a
run that ended on a refusal cannot be resumed. With one run only, propose how F1 should handle a refusal. It was never
seen in 22 dry-run calls. Options include ending and documenting it, or one retry of the same request.

**When you're done:** send the Advisor the F1 draft sha and the estimate. The Advisor reviews and commits F1. The real
editor run starts only after that, and only after the user's own approval line with the price and stop is in your
pane.

**After the editor run:**
1. The §8.4.2 replay of each submission on E, on the laptop CPU at batch 1, at $0.
2. Pod S: needs its own quote ($5.81, stop $6.68) and the user's line.
3. The Advisor's S review and the F2 commit.
4. Pod H+T: $3.92, stop $4.51, again with the user's line.
