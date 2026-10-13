#!/usr/bin/env bash
# §34b on one GPU: re-read the §23 prompts at the reader's OWN answer position (cached output teacher-forced up to its
# last "Answer:") with the §34 lenses already on disk. Reads only; no fitting. Resumable; rerun to continue.
#   bash scripts/exp34b_pod.sh [folds=0,1,2,3,4]
# Exit 3: an input or lens is missing (preflight). Exit 7/8 (jlens_read.py): lens does not fit the model / every read failed.
set -euo pipefail
export PIP_BREAK_SYSTEM_PACKAGES=1  # RunPod PyTorch images mark system Python externally managed (PEP 668)
cd "$(dirname "$0")/.."
DONE_FILE=runs/exp34b_pod.done
rm -f "$DONE_FILE"
trap 'rc=$?; mkdir -p "$(dirname "$DONE_FILE")"; echo "exit=$rc at $(date -u +%FT%TZ)" > "$DONE_FILE"' EXIT
FOLDS="${1:-0,1,2,3,4}"
TRANSFORMERS_VERSION=5.17.0
LENSES=runs/exp34_jlens/lenses
OUT=runs/exp34_jlens_own

missing() { echo "preflight: missing input $1" >&2; exit 3; }
INPUTS=(data/longmemeval/longmemeval_s_cleaned.json runs/exp23_verdicts.json external/jacobian-lens/pyproject.toml
        "$LENSES/qwen3b.pt" "$LENSES/qwen3b.json" "$LENSES/qwen7b.pt" "$LENSES/qwen7b.json")
for READER in qwen3b qwen7b; do
  for FOLD in 0 1 2 3 4; do
    for FILE in items.jsonl.gz steps.jsonl.gz episodes.jsonl; do
      INPUTS+=("runs/exp23_${READER}_f${FOLD}/fixed8__fraction0.05/${FILE}")
    done
  done
  test -d "cache/generations_exp23_${READER}" || missing "cache/generations_exp23_${READER}/"
done
for f in "${INPUTS[@]}"; do test -e "$f" || missing "$f"; done
echo "preflight ok: ${#INPUTS[@]} files and 2 cache folders present"

T0=$(date +%s)
STAGE_START=$T0
mkdir -p "$OUT"
stage_done() {
  local now; now=$(date +%s)
  echo "stage $1: elapsed $(( now - STAGE_START )) s (total $(( now - T0 )) s)" | tee -a "$OUT/pod.log"
  STAGE_START=$now
}

python -m pip install -q -e . -e external/jacobian-lens pyarrow scipy "transformers==${TRANSFORMERS_VERSION}" 2>&1 | tail -1
python -c "import torch, transformers; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'transformers', transformers.__version__)"
export TOKENIZERS_PARALLELISM=false
stage_done install

for READER in qwen3b qwen7b; do
  python scripts/jlens_read.py --reader "$READER" --lens "$LENSES/$READER.pt" --folds "$FOLDS" --position-mode own \
    --device cuda --dtype bfloat16 --out "$OUT/$READER" 2>&1 | tee -a "$OUT/${READER}_read.log"
  stage_done "read own $READER"
done
echo "all stages done in $(( $(date +%s) - T0 )) s; copy $OUT/ back"
