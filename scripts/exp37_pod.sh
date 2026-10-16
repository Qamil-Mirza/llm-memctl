#!/usr/bin/env bash
# §37 on one A40 (NEEDS the user's spend approval; quoted only if the laptop development gate passes): the three-turn
# elenchus for the 3B (V, CutB, CutF, Pnum, Pname, Vnum, Vdlg) then the 7B on the 3B's candidates (V, CutB, Pnum, Pname),
# then the §35 judge (Qwen2.5-7B, in the 7B's process) on the 3B's d and d', folds 0-4. Rows go to runs/exp37.
# Pre-registration: research/proposals/2026-10-16-exp37-elenchus.md at 714da06 (rev 2), §6. Generator:
# scripts/exp37_elenchus.py (texts and scores in memctl/analysis/elenchus.py). Resumable; rerun to continue.
#   RUNPOD_POD_ID=<id> bash scripts/exp37_pod.sh [folds=0,1,2,3,4]
#   ENGINE=stub bash scripts/exp37_pod.sh        # CPU, fake replies, no install, writes runs/exp37_stub
#
# Engine: vLLM 0.8.5 offline (V1, bf16, max_model_len 8192, prefix caching, gpu util 0.70, max_num_batched_tokens 2048),
# greedy; transformers fallback decided once before any generation (no mixing; rows record their engine).
# D-22 smoke in each process: 5 development V rows through E1-E3; abort unless >= 4 of 5 parse at E1 and >= 4 of 5 have
# E3 label mass >= 0.5. A failed 3B smoke aborts the chain (exit 5) before any row. A failed 7B smoke skips the 7B's
# items ("not compliant") but the judge still runs (judge-only process).
# Triage (§6), read backwards: the 3B's groups are V, CutB, CutF, Pnum (never dropped) | Pname | Vnum | Vdlg (dropped
# first); a group starts only if more than its reserve is left before the generation deadline (the reserve keeps time
# for the 7B's never-drop items and the judge). The judge pass on d is never dropped: the 7B's Pname keeps R_JUDGE.
# Deadlines: generation by HARD_STOP_S (3000 s) after start; the detached watchdog stops the pod at SELF_STOP_S
# (3600 s) with `runpodctl stop pod $RUNPOD_POD_ID` (exported from the launch command). No `pgrep -f`; every wait has a
# deadline. Exit 3: missing input; 4: no GPU; 5: 3B smoke failed. Sentinel $OUT/pod.done on any exit.
set -euo pipefail
export PIP_BREAK_SYSTEM_PACKAGES=1  # RunPod PyTorch images mark system Python externally managed (PEP 668)
cd "$(dirname "$0")/.."
FOLDS="${1:-0,1,2,3,4}"
ENGINE="${ENGINE:-vllm}"
DEFAULT_OUT=runs/exp37
export RUNPOD_POD_ID="${RUNPOD_POD_ID:-}"
if [ "$ENGINE" != stub ] && [ -z "$RUNPOD_POD_ID" ]; then
  echo "warning: RUNPOD_POD_ID is empty; the in-pod self-stop cannot fire. Pass RUNPOD_POD_ID=<id> in the launch command." >&2
fi
VLLM_VERSION=0.8.5
VLLM_TRANSFORMERS=4.51.3
FALLBACK_TRANSFORMERS=5.17.0
HARD_STOP_S="${HARD_STOP_S:-3000}"
SELF_STOP_S="${SELF_STOP_S:-3600}"
R7_CORE="${R7_CORE:-600}"     # the 7B's never-drop group (load, smoke, V, CutB, Pnum), seconds
R_JUDGE="${R_JUDGE:-240}"     # the judge pass on d and d' (never dropped)
R_VNUM="${R_VNUM:-120}"       # the 3B's V-num
R_VDLG="${R_VDLG:-300}"       # the 3B's in-dialogue E2 (dropped first)
if [ "$ENGINE" = stub ]; then OUT="${OUT:-runs/exp37_stub}"; PY="${PYTHON:-.venv/bin/python}"; else OUT="${OUT:-$DEFAULT_OUT}"; PY="${PYTHON:-python}"; fi
DONE_FILE="$OUT/pod.done"
mkdir -p "$OUT"
rm -f "$DONE_FILE"
T0=$(date +%s)
DEADLINE=$(( T0 + HARD_STOP_S ))
trap 'rc=$?; echo "exit=$rc engine=$ENGINE at $(date -u +%FT%TZ)" > "$DONE_FILE"' EXIT
echo "§37; RUNPOD_POD_ID ${RUNPOD_POD_ID:+set}${RUNPOD_POD_ID:-EMPTY}" | tee -a "$OUT/pod.log"

# The self-stop watchdog: detached, so it outlives this script; its pid is kept in a file (never searched by name).
if [ "$ENGINE" != stub ]; then
  nohup setsid bash -c "sleep $SELF_STOP_S; echo \"self-stop at \$(date -u +%FT%TZ)\" >> '$OUT/pod.log'; \
    if command -v runpodctl >/dev/null && [ -n \"\${RUNPOD_POD_ID:-}\" ]; then runpodctl stop pod \"\$RUNPOD_POD_ID\"; \
    else echo 'self-stop: runpodctl or RUNPOD_POD_ID missing; STOP THE POD BY HAND' >> '$OUT/pod.log'; fi" \
    > "$OUT/watchdog.log" 2>&1 < /dev/null &
  echo $! > "$OUT/watchdog.pid"
  echo "watchdog pid $(cat "$OUT/watchdog.pid"): pod stops itself at +${SELF_STOP_S} s" | tee -a "$OUT/pod.log"
fi

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
EXP35="${EXP35:-runs/exp35}"
for FOLD in ${FOLDS//,/ }; do
  INPUTS+=("$EXP35/qwen3b/B/fold${FOLD}.jsonl" "$EXP35/qwen3b/Fcut/fold${FOLD}.jsonl")
done
for f in "${INPUTS[@]}"; do test -e "$f" || missing "$f"; done
if [ -z "${HF_TOKEN:-}" ] && [ -s /workspace/.hf_token ]; then export HF_TOKEN; HF_TOKEN=$(tr -d '[:space:]' < /workspace/.hf_token); fi
if [ "$ENGINE" != stub ] && [ -z "${HF_TOKEN:-}" ]; then
  echo "warning: HF_TOKEN not set and /workspace/.hf_token absent; Hugging Face may rate-limit the model download" >&2
fi
echo "preflight ok: ${#INPUTS[@]} files and 2 cache folders present; engine $ENGINE; folds $FOLDS; out $OUT" | tee -a "$OUT/pod.log"

STAGE_START=$T0
stage_done() {
  local now; now=$(date +%s)
  echo "stage $1: elapsed $(( now - STAGE_START )) s (total $(( now - T0 )) s of $HARD_STOP_S)" | tee -a "$OUT/pod.log"
  STAGE_START=$now
}
STAGE_RC=0
run_stage() {  # run_stage NAME CMD...: killed at the generation deadline; a cut is logged and the chain goes on.
  # Exit 5 (smoke failed) is returned in STAGE_RC for the caller to decide; any other failure ends the chain.
  local name=$1; shift
  STAGE_RC=0
  local budget=$(( DEADLINE - $(date +%s) ))
  if [ "$budget" -le 0 ]; then echo "deadline reached before stage $name (skipped)" | tee -a "$OUT/pod.log"; return 0; fi
  set +e
  timeout "$budget" "$@" 2>&1 | tee -a "$OUT/${name}.log"
  local rc=${PIPESTATUS[0]}
  set -e
  STAGE_RC=$rc
  if [ "$rc" -eq 124 ]; then echo "stage $name cut by the deadline" | tee -a "$OUT/pod.log"
  elif [ "$rc" -eq 5 ]; then echo "stage $name: SMOKE FAILED (readout not compliant)" | tee -a "$OUT/pod.log"
  elif [ "$rc" -ne 0 ]; then echo "stage $name failed: exit $rc" | tee -a "$OUT/pod.log"; exit "$rc"; fi
  stage_done "$name"
}

if [ "$ENGINE" != stub ]; then
  GPU_DEADLINE=$(( $(date +%s) + 120 ))
  until nvidia-smi > /dev/null 2>&1; do
    if [ "$(date +%s)" -ge "$GPU_DEADLINE" ]; then echo "GPU not visible after 120 s" | tee -a "$OUT/pod.log"; exit 4; fi
    sleep 5
  done
  timeout 900 $PY -m pip install -q -e . "vllm==${VLLM_VERSION}" "transformers==${VLLM_TRANSFORMERS}" scipy 2>&1 | tail -2
  if $PY -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.version.cuda)" 2>&1 | tee -a "$OUT/pod.log"; then
    ENGINE=vllm
  else
    echo "vLLM import failed: falling back to transformers ${FALLBACK_TRANSFORMERS} for every stage" | tee -a "$OUT/pod.log"
    timeout 600 $PY -m pip install -q "transformers==${FALLBACK_TRANSFORMERS}" accelerate 2>&1 | tail -1
    ENGINE=hf
  fi
  $PY -c "import transformers; print('transformers', transformers.__version__)" | tee -a "$OUT/pod.log"
  export TOKENIZERS_PARALLELISM=false VLLM_LOGGING_LEVEL=WARNING VLLM_USE_V1=1
  stage_done install
else
  export HF_HUB_OFFLINE=1  # the stub tokenises with a locally cached Qwen2.5 tokenizer if there is one; never downloads
fi

R3=$(( R7_CORE + R_JUDGE )); R2=$(( R3 + R_VNUM )); R1=$(( R2 + R_VDLG ))
# Amendment 1 (9639186): the E3 fallback wording is the E3 wording for every call.
GEN=($PY scripts/exp37_elenchus.py --folds "$FOLDS" --backend "$ENGINE" --out "$OUT" --exp35 "$EXP35" --deadline "$DEADLINE" --e3-fallback)
run_stage elenchus_qwen3b "${GEN[@]}" --reader qwen3b --smoke --reserves "R1=$R1,R2=$R2,R3=$R3" \
  --plan "V,CutB,CutF,Pnum|Pname@R3|Vnum@R2|Vdlg@R1"
if [ "$STAGE_RC" -eq 5 ]; then
  echo "3B smoke failed: chain aborted before any row (exit 5)" | tee -a "$OUT/pod.log"; exit 5
fi
run_stage elenchus_qwen7b "${GEN[@]}" --reader qwen7b --smoke --judge --reserves "RJ=$R_JUDGE" --plan "V,CutB,Pnum|Pname@RJ"
if [ "$STAGE_RC" -eq 5 ]; then
  echo "7B smoke failed: the 7B is NOT COMPLIANT; its items are skipped; the judge still runs" | tee -a "$OUT/pod.log"
  run_stage judge_qwen7b "${GEN[@]}" --reader qwen7b --judge-only
fi
echo "all stages done in $(( $(date +%s) - T0 )) s; copy $OUT/ and cache/judge_exp37/ back before the self-stop at +${SELF_STOP_S} s" | tee -a "$OUT/pod.log"
