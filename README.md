# memctl

Honors thesis: how well an agent performs on long-horizon tasks under three kinds of
memory management —

1. **no memory controller** (baseline),
2. **JEV** deciding eviction, consolidation, and fetch / send to a persistent store,
3. **an RL policy** deciding those same operations.

> **Status: scaffold.** The design is being written. Start with
> [docs/restart.md](docs/restart.md): what the earlier Phase 2 work measured, which parts of
> it are worth bringing back, and the questions the new design has to answer.

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

The Docker setup is kept because it is independent of the research design.

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
