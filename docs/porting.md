# Moving memctl to another machine (for example a laptop with an NVIDIA GPU)

The project runs in a Docker container. The container holds the Python packages.
The project folder is shared with the container, so you can edit code and configs
without rebuilding, and results stay in ordinary folders that are easy to copy:

| Folder | What is in it | Copy it to the new machine? |
|---|---|---|
| `runs/` | Results of every run | Yes, if you want to compare old and new runs |
| `cache/` | Cached generations and judge verdicts | Optional. See "Reusing the cache" below |
| `data/` | Benchmark files | No. They are downloaded again when missing |
| model weights | Docker volume `models` | No. They are downloaded on first use |

## 1. One-time setup on the GPU machine

1. Install the NVIDIA driver and check that `nvidia-smi` lists the GPU.
2. Install Docker.
   - **Linux:** install Docker Engine, then the NVIDIA Container Toolkit
     (<https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html>).
   - **Windows:** install Docker Desktop with the WSL 2 backend. GPU support is built in.
     Run the commands below inside a WSL 2 terminal.
3. Check that containers can see the GPU:
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
   ```

## 2. Get the project

```bash
git clone <your repository URL> memctl && cd memctl
cp .env.example .env        # then put JEV_API_KEY in .env
docker compose build        # about 10 minutes the first time
```

## 3. Run

```bash
docker compose run --rm memctl                      # run the tests
docker compose run --rm memctl python -m memctl.run --config configs/locomo_small.yaml
docker compose run --rm memctl python -m memctl.run --config configs/locomo_small.yaml \
    --controller oracle --budget-fraction 0.10
```

Results appear in `runs/` on the host. On a machine without an NVIDIA GPU use
`docker compose run --rm cpu ...` instead of `memctl`.

To check that the model is on the GPU:

```bash
docker compose run --rm memctl python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 4. Does the model fit on the GPU?

The model is loaded in 16-bit precision. Rough memory needs, before the prompt itself:

| Model | Weights | Fits on |
|---|---|---|
| `Qwen/Qwen3.5-2B` | about 5 GB | 8 GB cards |
| `Qwen/Qwen3.5-4B` | about 10 GB | 12 GB cards and larger |

Long prompts need more memory on top. If the GPU runs out of memory, use a smaller
budget, a smaller model (`model.name` in the config), or ask for the 4-bit option to be added.

## Reusing the cache

Cache entries are keyed by the prompt and the model name, not by the machine. If
you copy `cache/` to the new machine, runs with the same prompts reuse the answers
that were generated on the old machine and the model is not called. A different
machine can produce slightly different answers for the same prompt, so for results
that must all come from one machine, do not copy `cache/`.

## Without Docker

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python -c constraints.txt -e ".[dev,models]"
.venv/bin/python -m pytest
```
