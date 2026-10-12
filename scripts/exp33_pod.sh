#!/usr/bin/env bash
# §33 on one A40: both Qwen readers, all folds, with the ablation check. Resumable; rerun to continue.
#   bash scripts/exp33_pod.sh [folds]      default folds 0,1,2,3,4
set -euo pipefail
cd "$(dirname "$0")/.."
FOLDS="${1:-0,1,2,3,4}"
python -m pip install -q -e . transformers accelerate scipy 2>&1 | tail -1
export TOKENIZERS_PARALLELISM=false
mkdir -p runs/exp33_jacobian
for READER in qwen3b qwen7b; do
  START=$(date +%s)
  python scripts/jacobian_lens.py --reader "$READER" --folds "$FOLDS" --targets gold,margin,own --ablate \
    --regen-check 20 --regen-min 15 --device cuda --dtype bfloat16 --out "runs/exp33_jacobian/$READER" \
    2>&1 | tee -a "runs/exp33_jacobian/${READER}_pod.log"
  echo "$READER done in $(( $(date +%s) - START )) s" | tee -a "runs/exp33_jacobian/${READER}_pod.log"
done
