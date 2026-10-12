#!/usr/bin/env bash
# §33 on one A40: both Qwen readers, all folds, with the ablation check. Resumable; rerun to continue.
#   bash scripts/exp33_pod.sh [folds]      default folds 0,1,2,3,4
set -euo pipefail
cd "$(dirname "$0")/.."
FOLDS="${1:-0,1,2,3,4}"
# transformers pinned to the local .venv version the branch was tested with; torch comes with the image.
TRANSFORMERS_VERSION=5.17.0

# Preflight: every input must be on the pod before anything is installed or downloaded.
missing() { echo "preflight: missing input $1" >&2; exit 3; }
INPUTS=(data/longmemeval/longmemeval_s_cleaned.json runs/exp23_verdicts.json)
for READER in qwen3b qwen7b; do
  for FOLD in 0 1 2 3 4; do
    for FILE in items.jsonl.gz steps.jsonl.gz episodes.jsonl; do
      INPUTS+=("runs/exp23_${READER}_f${FOLD}/fixed8__fraction0.05/${FILE}")
    done
  done
done
for INPUT in "${INPUTS[@]}"; do test -e "$INPUT" || missing "$INPUT"; done
for CACHE in cache/generations_exp23_qwen3b cache/generations_exp23_qwen7b; do
  test -d "$CACHE" && test -n "$(ls -A "$CACHE")" || missing "$CACHE (directory, non-empty)"
done
echo "preflight ok: ${#INPUTS[@]} files and 2 cache folders present"

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
python -m pip install -q -e . "transformers==${TRANSFORMERS_VERSION}" scipy 2>&1 | tail -1
export TOKENIZERS_PARALLELISM=false
mkdir -p runs/exp33_jacobian
for READER in qwen3b qwen7b; do
  START=$(date +%s)
  python scripts/jacobian_lens.py --reader "$READER" --folds "$FOLDS" --targets gold,margin,own --ablate \
    --regen-check 20 --regen-min 15 --device cuda --dtype bfloat16 --out "runs/exp33_jacobian/$READER" \
    2>&1 | tee -a "runs/exp33_jacobian/${READER}_pod.log"
  echo "$READER done in $(( $(date +%s) - START )) s" | tee -a "runs/exp33_jacobian/${READER}_pod.log"
done
