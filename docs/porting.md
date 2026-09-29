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
   - **Linux:** install Docker Engine, then the NVIDIA Container Toolkit. Installing the
     package is not enough: Docker also has to be told about it, and Docker 29 resolves
     `--gpus all` through CDI, so the spec has to be generated. All of this needs root.
     Tested on Ubuntu 24.04 with Docker 29.5.2:
     ```bash
     curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
       | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
     curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
       | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
       | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
     sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
     sudo nvidia-ctk runtime configure --runtime=docker
     sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
     sudo systemctl restart docker
     ```
     Without the last three lines `--gpus all` fails with
     `failed to discover GPU vendor from CDI: no known GPU vendor found`.
     Note that `systemctl restart docker` kills any build in progress.
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

Measured from the published weights rather than guessed. `AutoModelForCausalLM` loads
`Qwen3_5ForCausalLM`, the text-only class, so the vision tower on these checkpoints is
never read:

| Part | `Qwen3.5-4B` | `Qwen3.5-2B` |
|---|---|---|
| language-model layers | 7.14 GB | 2.75 GB |
| embeddings (tied, also used as the output head) | 1.27 GB | 1.02 GB |
| vision tower and MTP head, **not loaded** | 0.91 GB | 0.78 GB |
| **loaded at 16-bit** | **8.41 GB** | **3.77 GB** |

On top of that comes the key-value cache. These are hybrid-attention models: only 8 of
the 4B model's 32 layers use full attention, so the cache is about **32 KB per token**
(1 GB at 32,000 tokens). The shared prefix is held twice while a question is answered,
so budget for twice the prompt length.

**A 24 GB card runs everything**, including `full_context` at about 22,000 model tokens.
A 12 GB card runs every budget in `configs/`. Below that, see the next section.

## 4b. On a GPU too small for 16-bit weights

Set `load_in_4bit: true` under `model:` in the config. Weights become 4-bit NF4 and
everything is computed in float16. Install the extra first (it is already in the image):

```bash
uv pip install --python .venv/bin/python -c constraints.txt -e ".[dev,models,quantized]"
```

**This changes the answers.** 4-bit and 16-bit weights produce different text for the
same prompt, so 4-bit results are not comparable with 16-bit ones and are kept apart in
the generation cache. It is a way to run the pipeline on a small card, not a way to
reproduce a 16-bit result.

Measured on a 4 GB RTX 3050 Laptop GPU (3.95 GB usable), 4-bit `Qwen3.5-4B`:

| Prompt | Budget it corresponds to | Result |
|---|---|---|
| 3,850 model tokens | B = 10% | works, 0.30 s per question |
| 11,250 model tokens | B = 25% | out of memory |
| 22,350 model tokens | B = 50% and `full_context` | out of memory |

16-bit does not load at all on that card. `model.device_map: auto` with
`model.max_memory` spreads the weights over GPU and CPU and keeps 16-bit precision, but
it was measured at 5.0 s per question (17x slower) and still runs out of memory at
B = 25%, because the weights leave even less room for the cache. A card of 12 GB or more
is the real answer.

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
