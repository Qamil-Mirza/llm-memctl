#!/usr/bin/env bash
# §36b rerun (research/proposals/2026-10-15-exp36b-verifier-wording.md): FRAME = the adopted probe wording, default
# "a" (the dev probe's pick: label mass >= 0.5 on 84/84 dev rows for V and for CutB; runs/exp36_probe/summary.md); rows go to runs/exp36b; P2/P3 become
# P2p/P3p (the frame with the run-1 P2/P3 label texts). Each reader's process first runs a real-model smoke: 5 dev rows
# (folds 0-1) of item V through the frame, gate label mass >= 0.5 on >= 4 of 5 (wording e: a letter in the reply).
# A failed 3B smoke aborts the chain (exit 5) before any row is written; a failed 7B smoke ends the chain with the 3B
# rows only and the 7B logged as "not compliant" (proposal §2). RUNPOD_POD_ID must be passed in by the launch command
# (e.g. `RUNPOD_POD_ID=<id> FRAME=c bash scripts/exp36_pod.sh` over ssh): it is exported so the self-stop can fire.
#
# Run 1 (the frozen wording, runs/exp36) is reproduced with FRAME=run1.
#
# §36 on one A40: the 3B then the 7B verify the 3B's §35 candidates (single-token A/B/C support labels) plus C4 prompt
# logprobs, folds 0-4. Pre-registration: research/proposals/2026-10-15-exp36-verification.md (frozen at d2a67f2 + a3a3f9c).
# No judge pass (the labels are §35's). Resumable (rows are appended per chunk); rerun to continue.
#   bash scripts/exp36_pod.sh [folds=0,1,2,3,4]
#   ENGINE=stub bash scripts/exp36_pod.sh       # CPU, fake logprobs, no install, writes runs/exp36_stub
#
# Engine: vLLM 0.8.5 offline (`vllm.LLM`, V1 engine, bf16, max_model_len 8192, prefix caching, gpu_memory_utilization
# 0.70, max_num_batched_tokens 2048), transformers 4.51.3 (vLLM 0.8.5's own pin), torch 2.6.0 from vLLM's wheel deps.
# Same pins as §35. If `import vllm` fails after the install the script falls back to transformers 5.17.0 forward
# passes for every stage (decided once, before any verification; the generator refuses to resume rows of another
# engine and the report refuses a mixed run directory). Greedy, max_tokens 1, top-20 logprobs; C4: prompt_logprobs.
#
# Order = proposal §6 triage read backwards (what is dropped first runs last). Each reader runs once (one model load)
# with item groups in priority order; a group starts only if more than its reserve is left before the generation
# deadline, the reserve being the time the other reader's higher-priority groups still need:
#   3B: V, CutB, CutF, C4, Abs, CF | Vn, LC (reserve R7_CORE) | P2, P3 (reserve R7_CORE + R7_TIER2 + R7_OWN)
#   7B: V, CutB, CutF, Abs, CF     | Vn, LC                     | V_own, C4 (own candidates) | P2, P3
# Never dropped: the 3B's V, Cut-B, Cut-F and C4, the 7B's V on 3B candidates (they run first in each process).
# Each process starts with a probe: the longest label input and the longest C4 input of its run, timed (the
# first-minute real-shape call; a failure there stops the chain before any row is written).
# Expected: install ~4-6 min, 3B ~5 min, 7B ~8 min including downloads: ~20 min.
# Deadlines: generation must end by HARD_STOP_S (default 3000 s = 50 min) after this script starts; a watchdog started
# here stops the pod itself at SELF_STOP_S (default 3600 s = 60 min) with `runpodctl stop pod $RUNPOD_POD_ID`, whatever
# state the chain is in (copy the results back before then). Every wait loop has a deadline. No `pgrep -f` anywhere.
# Exit 3: an input is missing (preflight). Exit 4: the GPU did not come up. Exit 124: a never-drop stage was cut.
# The sentinel $OUT/pod.done is written on any exit. Per-question log lines: folds 0-1 only.
set -euo pipefail
export PIP_BREAK_SYSTEM_PACKAGES=1  # RunPod PyTorch images mark system Python externally managed (PEP 668)
cd "$(dirname "$0")/.."
FOLDS="${1:-0,1,2,3,4}"
ENGINE="${ENGINE:-vllm}"
FRAME="${FRAME:-a}"   # the §36b dev probe's pick (runs/exp36_probe/summary.md): wording a, 84/84 rows compliant on V and CutB
case "$FRAME" in
  a|b|c|d|e|f|g|h) FRAME_ARGS=(--frame "$FRAME" --smoke); P23="P2p,P3p"; DEFAULT_OUT=runs/exp36b ;;
  run1) FRAME_ARGS=(); P23="P2,P3"; DEFAULT_OUT=runs/exp36 ;;
  *) echo "FRAME must be set to the adopted wording (a b c d e f g h) or run1; got '${FRAME}'" >&2; exit 2 ;;
esac
export RUNPOD_POD_ID="${RUNPOD_POD_ID:-}"
if [ "$ENGINE" != stub ] && [ -z "$RUNPOD_POD_ID" ]; then
  echo "warning: RUNPOD_POD_ID is empty; the in-pod self-stop cannot fire. Pass RUNPOD_POD_ID=<id> in the launch command." >&2
fi
VLLM_VERSION=0.8.5
VLLM_TRANSFORMERS=4.51.3
FALLBACK_TRANSFORMERS=5.17.0
HARD_STOP_S="${HARD_STOP_S:-3000}"
SELF_STOP_S="${SELF_STOP_S:-3600}"
R7_CORE="${R7_CORE:-600}"     # what the 7B's never-drop group needs (load + V, CutB, CutF, Abs, CF), seconds
R7_TIER2="${R7_TIER2:-240}"   # the 7B's Vn + LC
R7_OWN="${R7_OWN:-300}"       # the 7B's own-candidate pass (V_own + C4)
if [ "$ENGINE" = stub ]; then OUT="${OUT:-runs/exp36_stub}"; PY="${PYTHON:-.venv/bin/python}"; else OUT="${OUT:-$DEFAULT_OUT}"; PY="${PYTHON:-python}"; fi
DONE_FILE="$OUT/pod.done"
mkdir -p "$OUT"
rm -f "$DONE_FILE"
T0=$(date +%s)
DEADLINE=$(( T0 + HARD_STOP_S ))
trap 'rc=$?; echo "exit=$rc engine=$ENGINE frame=$FRAME at $(date -u +%FT%TZ)" > "$DONE_FILE"' EXIT
echo "frame $FRAME; RUNPOD_POD_ID ${RUNPOD_POD_ID:+set}${RUNPOD_POD_ID:-EMPTY}" | tee -a "$OUT/pod.log"

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
  INPUTS+=("$EXP35/qwen3b/B/fold${FOLD}.jsonl" "$EXP35/qwen3b/Fcut/fold${FOLD}.jsonl" "$EXP35/qwen7b/B/fold${FOLD}.jsonl")
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

GEN=($PY scripts/exp36_verify.py --folds "$FOLDS" --backend "$ENGINE" --out "$OUT" --exp35 "$EXP35" --deadline "$DEADLINE" "${FRAME_ARGS[@]}")
run_stage verify_qwen3b "${GEN[@]}" --reader qwen3b \
  --plan "V,CutB,CutF,C4,Abs,CF|Vn,LC@${R7_CORE}|${P23}@$(( R7_CORE + R7_TIER2 + R7_OWN ))"
if [ "$STAGE_RC" -eq 5 ]; then
  echo "3B smoke failed: chain aborted before any row (exit 5)" | tee -a "$OUT/pod.log"; exit 5
fi
# The 3B's never-drop items must be complete before the 7B starts.
$PY - "$OUT" "$FOLDS" <<'EOF' | tee -a "$OUT/pod.log"
import json, sys
from pathlib import Path
out, folds = Path(sys.argv[1]), sys.argv[2]
m = json.loads((out / "qwen3b" / "_manifest.json").read_text()).get("counts", {})
bad = [i for i in ("V", "CutB", "CutF", "C4") if not m.get(i) or m[i].get("cut") or m[i].get("skipped_group")]
print("3B never-drop items complete" if not bad else f"3B never-drop items INCOMPLETE: {bad}")
sys.exit(124 if bad else 0)
EOF
run_stage verify_qwen7b "${GEN[@]}" --reader qwen7b --plan "V,CutB,CutF,Abs,CF|Vn,LC|V_own,C4|${P23}"
if [ "$STAGE_RC" -eq 5 ]; then
  echo "7B smoke failed: the 7B is NOT COMPLIANT under frame $FRAME; the chain ends with the 3B rows only" | tee -a "$OUT/pod.log"
fi
echo "all stages done in $(( $(date +%s) - T0 )) s; copy $OUT/ back before the self-stop at +${SELF_STOP_S} s" | tee -a "$OUT/pod.log"
