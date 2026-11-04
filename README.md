# memctl

**Honors thesis: A Mechanistic Investigation of Evidence-Use Failures In Language Models.**

When a small language model has the evidence it needs in front of it, why does it still answer "unknown" or answer
wrong? This repository holds the code, runs and research record for that question. Most of the work is on Qwen2.5-3B,
with checks on Llama, Gemma and Phi.

> **Start here:** [research/STATE.md](research/STATE.md) (what is currently supported, with strength and caveats),
> then [EXPERIMENTS.md](EXPERIMENTS.md) (the record; it wins any disagreement) and
> [research/EXPERIMENT_INDEX.md](research/EXPERIMENT_INDEX.md) (where each experiment's configs, runs, commits and
> costs live).

## How the project got here

1. **Memory controllers (Experiments 1–32).** The project began by asking whether a learned memory controller helps
   a model answer questions over long histories (LongMemEval, LoCoMo). The only clear win was query-aware selection at
   question time, and a 31-parameter logistic regression did as well as the learned head (§32). RL never beat
   imitation. The ceiling was the reader itself: with perfect evidence in view, a 3B reader still trailed a 7B one.
2. **Why the reader fails (Experiments 34–50).** That ceiling became the thesis. The main findings so far:
   - The answer is usually present inside the model when it fails. On most failures the gold answer's first token is
     in the final layer's top 10, in all four families (§34, §43).
   - Many failures are a refusal, not a gap in knowledge. One instruction line ("do not reply unknown") lifts 3B
     accuracy by 0.176 (§35). A single content-free steering direction at layer 29 opens the refusal, and the evidence
     then decides whether the answer is right (§38–§40).
   - The fixes are not selective. Every push tried so far also makes the model answer questions it should refuse
     (§41, §42, §49).
   - On controlled tasks, most false "unknown"s are "note, then refuse": the model writes the right answer in its own
     note and then says "unknown" (§49). Asking the model whether the evidence is sufficient was not a usable measure
     (§50).
3. **Proposed: agent-guided weight editing (Experiment 51).** A Claude editor proposes small weight edits to the 3B,
   tested on a working split and confirmed on sealed splits under a frozen pre-registration. The design is in
   [research/proposals/2026-11-02-exp51-agent-weight-editing.md](research/proposals/2026-11-02-exp51-agent-weight-editing.md).
   No result is recorded yet.

## Research questions

- **RQ1 (behaviour):** can a push make the model answer, without making it guess, when the evidence is in view?
- **RQ2 (inside the model):** is a failure one of not using the evidence, or of using it and still not producing
  the answer?
- **RQ3 (exploratory):** if the evidence is used, what makes the model refuse anyway, and where would a fix go?

## Where things are

| Path | What |
|---|---|
| [research/](research/) | State, decisions, open questions, proposals, reviews, audits and briefings |
| [EXPERIMENTS.md](EXPERIMENTS.md) | Full record of every experiment, with numbers and caveats |
| `memctl/` | Library: memory, controllers, retrieval and reranking, LLM readers and judge, splits, run logging |
| `scripts/` | Experiment scripts (for example `exp51_*.py`) |
| `configs/`, `runs/`, `cache/` | Configs, run outputs, and cached paid LLM outputs (never delete the caches) |
| [docs/research/](docs/research/) | Per-experiment reports and the earlier research note |
| [README_RESEARCH.md](README_RESEARCH.md), [INFRA_REPORT.md](INFRA_REPORT.md) | How to run things, and infrastructure notes |
| [RESET_HANDOFF.md](RESET_HANDOFF.md) | Working roles and rules since the 2026-10-10 reset |

The earlier Phase 2 work (eviction policies on LoCoMo) is kept on branch `phase2-locomo-baseline`, tag
`phase2-complete`:

```bash
git show phase2-locomo-baseline:memctl/memory.py
git checkout phase2-locomo-baseline -- memctl/memory.py   # bring one file back
```

## Running

```bash
python3 -m venv .venv
.venv/bin/pip install -c constraints.txt -e ".[dev]"
.venv/bin/pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pytest -q
.venv/bin/python -m memctl.run --config configs/stage1_smoke.yaml
```

Secrets (`HF_TOKEN`, `RUNPOD_API_KEY`, `ANTHROPIC_API_KEY`) go in `.env`, copied from `.env.example`. GPU runs use
rented RunPod pods. Each experiment's proposal records its cost and hard stop.

The Docker setup is kept for running local Hugging Face models on a GPU:

```bash
cp .env.example .env
printf 'MEMCTL_UID=%s\nMEMCTL_GID=%s\n' "$(id -u)" "$(id -g)" >> .env
docker compose build
docker compose run --rm memctl          # runs the tests
```

On Linux the GPU needs the NVIDIA Container Toolkit, including
`nvidia-ctk runtime configure --runtime=docker` and
`nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml` (both root). Full notes, with the
measured model sizes, are in `docs/porting.md` on the `phase2-locomo-baseline` branch.
