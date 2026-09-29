#!/usr/bin/env bash
# Set up memctl on a rented GPU machine and check it can run the full experiment.
#
#   git clone <repo> memctl && cd memctl && scripts/provision_gpu_box.sh
#
# Works on both shapes of rented box:
#   * a virtual machine with its own Docker (Lambda, Hetzner, GCP, Vast "Docker" templates)
#     -> builds the image and runs everything through docker compose
#   * a pod that is itself already a container with the GPU visible (RunPod and similar)
#     -> Docker-in-Docker is usually unavailable, so it installs into a local venv instead
#
# It stops with a clear message if the GPU is too small, rather than starting a run that
# will die hours later. Nothing here downloads a model until the checks have passed.

set -euo pipefail
cd "$(dirname "$0")/.."

NEED_GB=${NEED_GB:-12}
# MODE=venv forces the venv branch even where Docker works, which is how the venv path
# gets tested without a rented box. MODE=docker forces the other. Leave unset to detect.
FORCE_MODE=${MODE:-}
step() { printf '\n=== %s\n' "$*"; }
die()  { printf '\nSTOP: %s\n' "$*" >&2; exit 1; }

step "1. The GPU"
command -v nvidia-smi > /dev/null || die "nvidia-smi not found. Install the NVIDIA driver first."
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
VRAM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
VRAM_GB=$((VRAM_MIB / 1024))
printf 'usable: about %s GiB\n' "$VRAM_GB"
if [ "$VRAM_GB" -lt "$NEED_GB" ]; then
    die "this GPU has about ${VRAM_GB} GiB. configs/locomo_full.yaml needs about ${NEED_GB} GiB:
     Qwen3.5-4B is 8.41 GB at 16-bit and full_context prompts reach about 22,000 tokens.
     Rent a 24 GB card, or use configs/locomo_small_4bit.yaml, whose numbers are NOT
     comparable with 16-bit ones (docs/decisions.md entries 39 and 41)."
fi

step "2. Docker, or a venv if there is no usable Docker"
MODE=venv
if [ -n "$FORCE_MODE" ]; then
    MODE=$FORCE_MODE
    echo "mode forced to $MODE"
elif command -v docker > /dev/null 2>&1 && docker info > /dev/null 2>&1; then
    if docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi > /dev/null 2>&1; then
        MODE=docker
        echo "Docker can reach the GPU."
    else
        echo "Docker is running but cannot reach the GPU."
        if [ "$(id -u)" = 0 ] || sudo -n true 2>/dev/null; then
            echo "Installing the NVIDIA Container Toolkit (see docs/porting.md section 1)."
            SUDO=""; [ "$(id -u)" = 0 ] || SUDO=sudo
            curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
                | $SUDO gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
            curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
                | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
                | $SUDO tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null
            $SUDO apt-get update -qq && $SUDO apt-get install -y -qq nvidia-container-toolkit
            $SUDO nvidia-ctk runtime configure --runtime=docker
            $SUDO nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
            $SUDO systemctl restart docker || $SUDO service docker restart
            sleep 5
            docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi > /dev/null 2>&1 \
                && MODE=docker || echo "still cannot reach the GPU from Docker; falling back to a venv."
        else
            echo "No root, so the toolkit cannot be installed. Falling back to a venv."
        fi
    fi
elif [ -z "$FORCE_MODE" ]; then
    echo "No usable Docker (normal on a pod that is already a container). Using a venv."
fi
echo "mode: $MODE"

step "3. Install"
if [ "$MODE" = docker ]; then
    docker compose build
    RUN="docker compose run --rm memctl python"
else
    # gcc is needed at run time: Qwen3.5's linear-attention layers are Triton kernels
    # compiled on first use (docs/decisions.md entry 37).
    command -v gcc > /dev/null 2>&1 || {
        echo "installing a C compiler"
        SUDO=""; [ "$(id -u)" = 0 ] || SUDO=sudo
        $SUDO apt-get update -qq && $SUDO apt-get install -y -qq build-essential
    }
    command -v uv > /dev/null 2>&1 || curl -fsSL https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    uv venv --python 3.12
    uv pip install --python .venv/bin/python -c constraints.txt -e ".[dev,models,quantized]"
    RUN=".venv/bin/python"
fi

step "4. The tests"
$RUN -m pytest -q

step "5. torch sees the GPU"
$RUN -c "
import torch
assert torch.cuda.is_available(), 'torch cannot see the GPU'
free, total = torch.cuda.mem_get_info()
print('%s, %.1f GiB free of %.1f GiB, torch %s' % (
    torch.cuda.get_device_name(0), free/2**30, total/2**30, torch.__version__))
"

step "6. The models (9.32 GB + 4.55 GB + 0.13 GB)"
$RUN -c "
from huggingface_hub import snapshot_download
for name in ['Qwen/Qwen3.5-4B', 'Qwen/Qwen3.5-2B', 'BAAI/bge-small-en-v1.5']:
    print('downloading', name, flush=True)
    snapshot_download(name)
print('models ready')
"

step "7. The benchmark (2.8 MB)"
$RUN -c "
from memctl.benchmarks.locomo import load_locomo
conversations = load_locomo()
questions = sum(len(c.questions()) for c in conversations)
print('%d conversations, %d questions' % (len(conversations), questions))
assert len(conversations) == 10 and questions == 1986, 'unexpected benchmark size'
"

step "Ready"
cat <<EOF
Run the grid in batches (read comparison.md after the first one):

$( [ "$MODE" = docker ] \
    && echo "  scripts/run_grid.sh first
  scripts/run_grid.sh rest
  scripts/run_grid.sh full_context" \
    || echo "  NO_DOCKER=1 scripts/run_grid.sh first
  NO_DOCKER=1 scripts/run_grid.sh rest
  NO_DOCKER=1 scripts/run_grid.sh full_context" )

Copy back runs/ when it is finished. Leave cache/ behind unless you want later runs on
another machine to reuse these generations (docs/porting.md, "Reusing the cache").
EOF
