# memctl

Honors thesis: how well an agent performs on long-horizon tasks under three kinds of
memory management —

1. **no memory controller** (baseline),
2. **JEV** deciding eviction, consolidation, and fetch / send to a persistent store,
3. **an RL policy** deciding those same operations.

> **Status: framework built and validated on synthetic tasks; every result reproduced
> from clean commits on 2026-10-05.** A real LLM has been evaluated as the task model:
> Experiment 8 (`qwen2.5:3b` via Ollama, local GPU) and Experiment 9 (Qwen2.5-7B-Instruct
> via vLLM on a rented RunPod GPU). The JEV arm is still blocked (no key). LLM runs use the
> `openai` backend against any OpenAI-compatible server, set by `model.base_url`.
> Start with [README_RESEARCH.md](README_RESEARCH.md) (how to run things), then
> [EXPERIMENTS.md](EXPERIMENTS.md) (results),
> [INFRA_REPORT.md](INFRA_REPORT.md) (what is blocked here and by what) and
> [SCIENTIFIC_VALIDITY_REPORT.md](SCIENTIFIC_VALIDITY_REPORT.md) (what the results do and do not show).
> Research write-ups are in [docs/research/](docs/research/):
> [one-page summary](docs/research/ONE_PAGE_RESEARCH_SUMMARY.md),
> [research note](docs/research/HINDSIGHT_MEMORY_RESEARCH_NOTE.md),
> [related work](docs/research/RELATED_WORK_LITERATURE_REPORT.md) and
> [next directions](docs/research/next_directions.html).

The three arms are three of the controllers behind one interface: `no_controller`,
`jev` and `rl`, alongside heuristic baselines, a prompted-LLM controller and a hindsight
oracle. [docs/restart.md](docs/restart.md) records what the earlier Phase 2 work measured and
why it was redesigned.

## The earlier work

Phase 2 (eviction policies on LoCoMo, GPU-ported, with a hindsight oracle, failure
attribution, a paired significance test and cloud run scripts) is kept intact on branch
`phase2-locomo-baseline`, with tag `phase2-complete` on its last code commit. To look at or
reuse a file from it:

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
