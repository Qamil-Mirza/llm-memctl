#!/usr/bin/env bash
# §34 on one GPU: fit J-lenses for both Qwen readers (7B first, 3B capped at the 7B's prompt count), read the §23
# prompts, lens-quality eval, community 7B cross-check (non-fatal), then the §33 secondary. Resumable; rerun to
# continue (a below_minimum false lens is final; REFIT=1 resumes its fit from the checkpoint).
#   bash scripts/exp34_pod.sh [max_minutes_7b=90] [max_minutes_3b=40] [n_prompts=128]
# Exit 3: an input is missing (preflight, before anything is installed). Exit 5: a lens is below the minimum.
# Exit 6 (jlens_fit.py): checkpoint settings differ. Exit 7/8 (jlens_read.py): lens does not fit the model / every read failed.
set -euo pipefail
export PIP_BREAK_SYSTEM_PACKAGES=1  # RunPod PyTorch images mark system Python externally managed (PEP 668)
cd "$(dirname "$0")/.."
M7="${1:-90}"
M3="${2:-40}"
N="${3:-128}"
MIN_PROMPTS=32
# transformers pinned to the local .venv version the branch was tested with; torch comes with the image.
TRANSFORMERS_VERSION=5.17.0
OUT=runs/exp34_jlens
LENSES="$OUT/lenses"
COMMUNITY_LENS="$LENSES/community_qwen7b.pt"
COMMUNITY_OUT="$OUT/community_qwen7b/qwen7b"   # exp34_report.py --community-root $OUT/community_qwen7b

# Preflight: every input must be on the pod before anything is installed or downloaded (§33 list + §34 inputs).
missing() { echo "preflight: missing input $1" >&2; exit 3; }
INPUTS=(data/longmemeval/longmemeval_s_cleaned.json runs/exp23_verdicts.json
        data/jlens_corpus/wikitext-103-raw-v1-test.parquet external/jacobian-lens/pyproject.toml)
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

T0=$(date +%s)
STAGE_START=$T0
stage_done() {
  local now; now=$(date +%s)
  echo "stage $1: elapsed $(( now - STAGE_START )) s (total $(( now - T0 )) s)" | tee -a "$OUT/pod.log"
  STAGE_START=$now
}

python -m pip install -q -e . -e external/jacobian-lens pyarrow scipy "transformers==${TRANSFORMERS_VERSION}" 2>&1 | tail -1
python -c "import torch, transformers; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'transformers', transformers.__version__)"
echo "jlens commit $(git -C external/jacobian-lens rev-parse --short HEAD 2>/dev/null || echo unknown) (expected 581d398)"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$LENSES"
stage_done install

# Fit, 7B first. A lens whose sidecar says "below_minimum": false is final and is not refitted on a rerun (even if
# the time box stopped it short of its target); set REFIT=1 to resume such a fit from its checkpoint. The 3B is then
# fitted on the 7B's prompt count N7 (same seed, so the same prompt prefix), so both lenses average the same prompts.
n_prompts_of() { grep -Eo '"n_prompts":[[:space:]]*[0-9]+' "$1" 2>/dev/null | grep -Eo '[0-9]+$' || true; }
fit_reader() {
  local READER="$1" MINUTES="$2" NP="$3" SIDE="$LENSES/$1.json"
  if test -e "$LENSES/$READER.pt" && grep -Eq '"below_minimum":[[:space:]]*false' "$SIDE" 2>/dev/null \
     && test "${REFIT:-0}" != 1; then
    echo "$READER lens final (n_prompts $(n_prompts_of "$SIDE")), not refitted (REFIT=1 to refit)"
  else
    python scripts/jlens_fit.py --reader "$READER" --n-prompts "$NP" --min-prompts "$MIN_PROMPTS" --max-minutes "$MINUTES" \
      --dim-batch 32 --device cuda --dtype bfloat16 --out "$LENSES" 2>&1 | tee -a "$OUT/${READER}_fit.log"
  fi
  # Only an explicit "below_minimum": false passes; true, a missing key, a truncated or absent sidecar all stop.
  if ! grep -Eq '"below_minimum":[[:space:]]*false' "$SIDE" 2>/dev/null; then
    echo "fit: $READER sidecar $SIDE does not say below_minimum false" \
         "($(grep -E '"(n_prompts|below_minimum)"' "$SIDE" 2>/dev/null | tr -d ' ,\n' || true)); not reading with it." \
         "Rerun with a longer time box to resume the fit." | tee -a "$OUT/pod.log" >&2
    exit 5
  fi
  stage_done "fit $READER"
}
fit_reader qwen7b "$M7" "$N"
N7=$(n_prompts_of "$LENSES/qwen7b.json")
if test -z "$N7"; then
  echo "fit: no n_prompts in $LENSES/qwen7b.json; cannot cap the 3B fit" | tee -a "$OUT/pod.log" >&2
  exit 5
fi
echo "qwen7b lens averages $N7 prompts; fitting qwen3b on the same $N7" | tee -a "$OUT/pod.log"
fit_reader qwen3b "$M3" "$N7"
N3=$(n_prompts_of "$LENSES/qwen3b.json")
test "$N3" = "$N7" || echo "warning: qwen3b lens averages $N3 prompts, qwen7b $N7 (a final lens is kept; REFIT=1 to refit)" \
  | tee -a "$OUT/pod.log" >&2

# Read every fold (resumable per question).
for READER in qwen3b qwen7b; do
  python scripts/jlens_read.py --reader "$READER" --lens "$LENSES/$READER.pt" --folds 0,1,2,3,4 --device cuda \
    --dtype bfloat16 2>&1 | tee -a "$OUT/${READER}_read.log"
  stage_done "read $READER"
done

# Lens-quality eval (skipped when a result newer than the lens exists).
for READER in qwen3b qwen7b; do
  if test -e "$OUT/eval/$READER.json" && test "$OUT/eval/$READER.json" -nt "$LENSES/$READER.pt"; then
    echo "$READER eval done, skipped"
  else
    python scripts/jlens_eval.py --reader "$READER" --lens "$LENSES/$READER.pt" --device cuda --dtype bfloat16 \
      2>&1 | tee -a "$OUT/${READER}_eval.log"
  fi
  stage_done "eval $READER"
done

# Community 7B lens cross-check, development folds only. Copy out of the HF cache; never move or delete it.
# Non-fatal: a failed download or read is logged and the §33 secondary still runs.
community_check() {
  if ! test -e "$COMMUNITY_LENS"; then
    python -c "import shutil, huggingface_hub; shutil.copyfile(huggingface_hub.hf_hub_download('andyx10/jacobian-lens-qwen2.5-7b-instruct', 'qwen7b_lens.pt'), '$COMMUNITY_LENS.tmp')" \
      || return 1
    test -e "$COMMUNITY_LENS.tmp" || return 1
    mv "$COMMUNITY_LENS.tmp" "$COMMUNITY_LENS" || return 1
  fi
  python scripts/jlens_read.py --reader qwen7b --lens "$COMMUNITY_LENS" --out "$COMMUNITY_OUT" --folds 0,1 --device cuda \
    --dtype bfloat16 2>&1 | tee -a "$OUT/community_qwen7b_read.log" || return 1
}
if community_check; then
  stage_done "community cross-check"
else
  echo "community cross-check failed; continuing" | tee -a "$OUT/pod.log" >&2
  stage_done "community cross-check (failed)"
fi

# §33 secondary (its own preflight and pins).
bash scripts/exp33_pod.sh
stage_done "§33 secondary"
echo "all stages done in $(( $(date +%s) - T0 )) s; copy $OUT/ and runs/exp33_jacobian/ back"
