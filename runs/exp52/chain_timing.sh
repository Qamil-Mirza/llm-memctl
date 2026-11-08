#!/bin/bash
cd /home/qamil-mirza/Code/llm-memctl
export MALLOC_ARENA_MAX=2 HF_ENABLE_PARALLEL_LOADING=false
for k in gemma270m qwen05b llama1b gemma1b qwen15b smol17b; do
  echo "== $k $(date +%T)"
  .venv/bin/python scripts/exp52_run.py $k --timing 2>&1 | grep -v -i warn
done
echo "ALL DONE $(date +%T)"
