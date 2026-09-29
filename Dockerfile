# memctl in a container. Works on a machine with an NVIDIA GPU (uses it automatically)
# and on a machine without one (falls back to CPU). See docs/porting.md.
FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models \
    MPLCONFIGDIR=/tmp/matplotlib

# git is used to stamp every run folder with the commit it was made from
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/* \
    && git config --system --add safe.directory /app

WORKDIR /app

# Install the dependencies first so that code changes do not reinstall them.
# On Linux x86-64, the default torch wheel includes NVIDIA CUDA support.
COPY pyproject.toml constraints.txt ./
RUN mkdir memctl && touch memctl/__init__.py \
    && pip install -c constraints.txt -e ".[dev,models]"

COPY . .

# docker-compose.yml mounts the project folder over /app, so code edits need no rebuild and
# results, caches and datasets stay on the host. Model weights live in the "models" volume.

CMD ["python", "-m", "pytest", "-q"]
