# memctl

**Honors thesis repository. Current question: what inside a reasoning model makes it ignore a valid update that
arrives in the middle of its thinking?**

Wu, Miroyan et al. ("Are Large Reasoning Models Interruptible?", arXiv 2510.11713) showed that when a reasoning model
is given a corrected problem partway through its reasoning, it often finishes the old problem anyway, and called this
self-doubt. This repository holds the code, results and research record of a study that looks for the layers,
attention heads and neurons responsible, on Qwen3 1.7B and 8B with the authors' benchmark.

> **Start here:** [research/STATE.md](research/STATE.md) (what is supported, with strength and caveats), then
> [selfdoubt/PROPOSAL.md](selfdoubt/PROPOSAL.md) (the pre-registration and the results of stages 0–2),
> [EXPERIMENTS.md](EXPERIMENTS.md) (the record) and [REPRODUCE.md](REPRODUCE.md) (one command per reported number).

## How the project got here

1. **Memory controllers (§1–§32, closed 2026-11-08).** Does a learned memory controller help a model answer questions
   over long histories? The only clear win was query-aware selection at question time, and a 31-parameter logistic
   regression did as well as the learned head. The ceiling was the reader: with the evidence in view, a 3B reader
   still trailed a 7B one.
2. **False abstention (§33–§57, closed 2026-11-11).** Why does a small reader say "unknown" with the evidence in
   view? The answer is usually present inside the model, the refusal is a late decision, and offering the "unknown"
   option causes much of it. Steering, head ablation, weight edits and neuron resets each release it and each beats
   its random control, but none is more selective than a plain logit bias against the "unknown" token. The line was
   closed as groundwork, with no paper planned from it.
3. **Self-doubt (from 2026-11-08; the thesis since 2026-11-11).** The same shape of failure in reasoning models: the
   information is taken in and then not used. It inherits the localisation tools and two controls that every
   mechanism claim here must pass: random component sets of the same shape, and a logit bias at matched effect.

## Where things are

| Path | What |
|---|---|
| [research/STATE.md](research/STATE.md), [DECISIONS.md](research/DECISIONS.md), [OPEN_QUESTIONS.md](research/OPEN_QUESTIONS.md) | Findings F1–F24 with their strength, every decision by number, and the open questions |
| [EXPERIMENTS.md](EXPERIMENTS.md) | The record: §1–§57 condensed, live entries from §58 |
| [REPRODUCE.md](REPRODUCE.md) | Conventions for new work, and the command that recomputes each finding |
| `selfdoubt/` | Code, tests, the pre-registration (`PROPOSAL.md`), the spend log (`SPEND.md`), frozen inputs and result summaries |
| `runs/selfdoubt/` | Per-row outputs, not in git |
| `external/interrupt-lrm/` | The original authors' code, pinned, not in git |
| [archive/](archive/README.md) | Data of the two closed eras as checksummed tarballs, not in git |

## The closed eras

Their code, tests, proposals, reviews and full experiment text are at two git tags, and their data is in `archive/`:

| Era | Tag | Data |
|---|---|---|
| Memory controllers, §1–§32 | `memctl-era-final` | `archive/memctl-era/` |
| False abstention, §33–§57 | `false-abstention-era-final` | `archive/false-abstention/` |

```bash
git show false-abstention-era-final:EXPERIMENTS.md          # the full record of §33–§57
git worktree add --detach .worktrees/fa false-abstention-era-final   # the whole tree, runnable once data is untarred
```

[archive/README.md](archive/README.md) lists every tarball and the restore steps. The tags exist only on this laptop;
`archive/llm-memctl-all-2026-11-11.bundle` is a single-file copy of the whole history. The earlier Phase 2 work
(eviction policies on LoCoMo) is on branch `phase2-locomo-baseline`, tag `phase2-complete`.

## Running

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest
```

This gives 106 passed and 1 skipped module; with torch installed all 126 tests pass. The pinned Stage 2 environment
is `.[dev,models]` (torch 2.8.0, transformers 4.57.1), and pods add `.[pods]` (vLLM 0.11.0). GPU runs use rented
RunPod pods; [selfdoubt/SPEND.md](selfdoubt/SPEND.md) records each one.

## Working rules

- Design choices come from development data only. Held-out data is read once, after the design is frozen by commit.
- Every paid run needs the user's approval in advance, with a quote from a timed stub.
- A number is reported only if [REPRODUCE.md](REPRODUCE.md) has the command that recomputes it from kept files.
- Secrets (`HF_TOKEN`, `RUNPOD_API_KEY`, `RUNPOD_API_KEY_2`, `ANTHROPIC_API_KEY`) go in `.env`, copied from
  `.env.example`.
