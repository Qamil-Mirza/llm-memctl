#!/usr/bin/env bash
# §35 on one A40: arms N, A, B, F, Fcut, Ncut for the 3B, then the 7B, then the 7B judge over both readers, folds 0-4.
# Pre-registration: research/proposals/2026-10-13-exp35-answer-selection.md at commit 12c9168 (round 2, conditional ACCEPT).
# Resumable (rows are appended per chunk, judge verdicts are cached by text); rerun to continue.
#   bash scripts/exp35_pod.sh [folds=0,1,2,3,4]
#   ENGINE=stub bash scripts/exp35_pod.sh       # CPU, canned outputs, no install, writes runs/exp35_stub
#
# Engine: vLLM 0.8.5 offline (`vllm.LLM`, bf16, max_model_len 8192, prefix caching), with transformers 4.51.3 (vLLM
# 0.8.5's own pin). 0.8.5 is the version of the vLLM server that produced every §23 answer and verdict, so the engine
# differs from §23 only in offline vs server batching. pip replaces the image's torch 2.8 by torch 2.6.0 (cu124 wheels;
# the A40 driver runs them); the resolution was dry-run checked for Python 3.12. If `import vllm` fails after the
# install, the script falls back to transformers 5.17.0 batched greedy generate (bf16, left padding). The engine is
# decided once, before any generation, and then runs every arm of both readers and the judge: one run never mixes
# engines (the generator refuses to resume rows from another engine; the report refuses a mixed run directory).
# Sampling: greedy, temperature 0, top_p 1, seed 0; max_new_tokens N/F/Fcut/Ncut 256 (as §23), A/B 64, judge 4. Raw
# input = ids(chat template with generation prompt) + ids(continuation); A and B continue the cached note (completion
# style, not a chat message).
# Expected time: install ~4-6 min, 3B ~4 min, 7B ~7 min (both including model download), judge ~5 min: ~20-25 min.
# Hard stop: HARD_STOP_S (default 5100 s = 85 min, under the 90 min pod stop). Generation must end JUDGE_RESERVE_S
# (default 900 s) before it, so the judge always runs on whatever was generated. Stage order is the pre-registered
# triage order read backwards (what is dropped first runs last): 3B N,A,B,F; 3B Fcut,Ncut; 7B all arms; judge. A
# generation stage cut by its deadline is logged and the chain moves on to the judge (rows are appended per chunk).
# Exit 3: an input is missing (preflight). Exit 124: the judge was cut by the hard stop. Per-question log lines:
# folds 0-1 only.
set -euo pipefail
export PIP_BREAK_SYSTEM_PACKAGES=1  # RunPod PyTorch images mark system Python externally managed (PEP 668)
cd "$(dirname "$0")/.."
FOLDS="${1:-0,1,2,3,4}"
ENGINE="${ENGINE:-vllm}"
VLLM_VERSION=0.8.5
VLLM_TRANSFORMERS=4.51.3
FALLBACK_TRANSFORMERS=5.17.0
HARD_STOP_S="${HARD_STOP_S:-5100}"
JUDGE_RESERVE_S="${JUDGE_RESERVE_S:-900}"
if [ "$ENGINE" = stub ]; then OUT=runs/exp35_stub; PY="${PYTHON:-.venv/bin/python}"; else OUT=runs/exp35; PY="${PYTHON:-python}"; fi
DONE_FILE="$OUT/pod.done"
mkdir -p "$OUT"
rm -f "$DONE_FILE"
trap 'rc=$?; echo "exit=$rc engine=$ENGINE at $(date -u +%FT%TZ)" > "$DONE_FILE"' EXIT

missing() { echo "preflight: missing input $1" >&2; exit 3; }
INPUTS=(data/longmemeval/longmemeval_s_cleaned.json runs/exp23_verdicts.json)
for READER in qwen3b qwen7b; do
  for FOLD in 0 1 2 3 4; do
    for FILE in items.jsonl.gz steps.jsonl.gz episodes.jsonl; do
      INPUTS+=("runs/exp23_${READER}_f${FOLD}/fixed8__fraction0.05/${FILE}")
    done
  done
  test -d "cache/generations_exp23_${READER}" || missing "cache/generations_exp23_${READER}/"
done
test -d cache/judge_exp23 || missing "cache/judge_exp23/"
for f in "${INPUTS[@]}"; do test -e "$f" || missing "$f"; done
if [ -z "${HF_TOKEN:-}" ] && [ -s /workspace/.hf_token ]; then export HF_TOKEN; HF_TOKEN=$(tr -d '[:space:]' < /workspace/.hf_token); fi
if [ "$ENGINE" != stub ] && [ -z "${HF_TOKEN:-}" ]; then
  echo "warning: HF_TOKEN not set and /workspace/.hf_token absent; Hugging Face may rate-limit the model download" >&2
fi
echo "preflight ok: ${#INPUTS[@]} files and 3 cache folders present; engine $ENGINE; folds $FOLDS; out $OUT" | tee -a "$OUT/pod.log"

T0=$(date +%s)
STAGE_START=$T0
stage_done() {
  local now; now=$(date +%s)
  echo "stage $1: elapsed $(( now - STAGE_START )) s (total $(( now - T0 )) s of $HARD_STOP_S)" | tee -a "$OUT/pod.log"
  STAGE_START=$now
}
left() { echo $(( HARD_STOP_S - ($(date +%s) - T0) - $1 )); }
run_stage() {  # run_stage NAME RESERVE_S SOFT CMD...: the stage gets the time left minus RESERVE_S; SOFT=1 goes on after a cut
  local name=$1 reserve=$2 soft=$3; shift 3
  local budget; budget=$(left "$reserve")
  if [ "$budget" -le 0 ]; then
    echo "deadline reached before stage $name (skipped)" | tee -a "$OUT/pod.log"
    if [ "$soft" = 1 ]; then return 0; fi
    exit 124
  fi
  set +e
  timeout "$budget" "$@" 2>&1 | tee -a "$OUT/${name}.log"
  local rc=${PIPESTATUS[0]}
  set -e
  if [ "$rc" -eq 124 ] && [ "$soft" = 1 ]; then echo "stage $name cut by its deadline; moving on" | tee -a "$OUT/pod.log"
  elif [ "$rc" -ne 0 ]; then echo "stage $name failed: exit $rc" | tee -a "$OUT/pod.log"; exit "$rc"; fi
  stage_done "$name"
}

if [ "$ENGINE" != stub ]; then
  $PY -m pip install -q -e . "vllm==${VLLM_VERSION}" "transformers==${VLLM_TRANSFORMERS}" scipy 2>&1 | tail -2
  if $PY -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.version.cuda)" 2>&1 | tee -a "$OUT/pod.log"; then
    ENGINE=vllm
  else
    echo "vLLM import failed: falling back to transformers ${FALLBACK_TRANSFORMERS} for every stage" | tee -a "$OUT/pod.log"
    $PY -m pip install -q "transformers==${FALLBACK_TRANSFORMERS}" accelerate 2>&1 | tail -1
    ENGINE=hf
  fi
  $PY -c "import transformers; print('transformers', transformers.__version__)" | tee -a "$OUT/pod.log"
  export TOKENIZERS_PARALLELISM=false VLLM_LOGGING_LEVEL=WARNING
  stage_done install
else
  export HF_HUB_OFFLINE=1  # the stub tokenises with a locally cached Qwen2.5 tokenizer if there is one; never downloads
fi

GEN=($PY scripts/exp35_generate.py --folds "$FOLDS" --backend "$ENGINE" --out "$OUT")
run_stage generate_qwen3b_main "$JUDGE_RESERVE_S" 1 "${GEN[@]}" --reader qwen3b --arms N,A,B,F
run_stage generate_qwen3b_cut "$JUDGE_RESERVE_S" 1 "${GEN[@]}" --reader qwen3b --arms Fcut,Ncut
run_stage generate_qwen7b "$JUDGE_RESERVE_S" 1 "${GEN[@]}" --reader qwen7b --arms N,A,B,F,Fcut,Ncut
run_stage judge 0 0 $PY scripts/exp35_generate.py --judge --reader qwen3b,qwen7b --folds "$FOLDS" --backend "$ENGINE" --out "$OUT"
echo "all stages done in $(( $(date +%s) - T0 )) s; copy $OUT/ and cache/judge_exp35/ back" | tee -a "$OUT/pod.log"
