#!/usr/bin/env bash
# §47 pod stage (research/proposals/2026-10-31-exp47-bridge.md rev 0 + rev 1; D-57): one A40, Qwen2.5-3B in float32
# (the §44 rows were read in float32; the Δ assert needs the same arithmetic), CUDA, transformers 5.17.0. Never two
# pods. Image: runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404 (its torch kept).
#
#   RUNPOD_POD_ID=<id> bash scripts/exp47_pod.sh                     # on the pod, from the unpacked bundle
#   ENGINE=stub bash scripts/exp47_pod.sh                            # CPU, a tiny random Qwen2 (36 layers)
#   ENGINE=stub INSTALL=1 PYTHON=<venv>/bin/python bash scripts/exp47_pod.sh   # D-30: the SAME install line
#   ENGINE=stub PROJECT_LIMIT=0 bash scripts/exp47_pod.sh            # the projection must stop the chain (exit 7)
#
# Order: preflight -> watchdog -> GPU check + install -> population (36 rows, 13 / 13 entered, the rows sha from
# bundle_info; exit 3) -> stub check on the pod's device (the 3 smoke rows on the stub, Δ assert off; a second run
# redoes nothing; exit 5) -> download the 3B -> D-22 smoke (the 3 smoke rows ec81a493 / 7401057b / 5809eb10 on the 3B:
# sanity and the Δ assert must hold on every row; exit 5) -> projection (the slowest smoke row x 36 x 1.15 + the
# seconds spent; above PROJECT_LIMIT = $1.00 at $0.59/h the chain stops, exit 7) -> the 36 rows -> report -> status.
# Deadlines: HARD_STOP_S (default 5400) after start all model work ends (the run stops between rows, exit 9,
# resumable). The watchdog stops the pod at SELF_STOP_S (default 6000) with `runpodctl stop pod $RUNPOD_POD_ID`; a
# laptop DELETE follows the copy-back. No outcome is printed. Exit 3: input. 4: GPU / install / download. 5: stub
# check or smoke failed. 7: projection above the limit. 8: sanity / Δ assert above 5 %.
# Copy back: $OUT, runs/exp47_bridge/cuda-float32.
set -euo pipefail
export PIP_BREAK_SYSTEM_PACKAGES=1
cd "$(dirname "$0")/.."
ENGINE="${ENGINE:-hf}"
INSTALL="${INSTALL:-}"
TRANSFORMERS_VERSION=5.17.0
HARD_STOP_S="${HARD_STOP_S:-5400}"
SELF_STOP_S="${SELF_STOP_S:-6000}"
PROJECT_LIMIT="${PROJECT_LIMIT:-$([ "${ENGINE:-hf}" = stub ] && echo 1000 || echo 1.00)}"
RATE="${RATE:-0.59}"
export RUNPOD_POD_ID="${RUNPOD_POD_ID:-}"
if [ "$ENGINE" = stub ]; then
  OUT="${OUT:-runs/exp47_pod_stub}"; PY="${PYTHON:-.venv/bin/python}"; DEV=cpu; DT=float32
  RES="$OUT/full"; LIMIT="--only ${STUB_ROWS:-ec81a493 7401057b}"; NOASSERT="--no-assert44"
else
  OUT="${OUT:-runs/exp47_pod}"; PY="${PYTHON:-python}"; DEV=cuda; DT=float32
  RES=runs/exp47_bridge/cuda-float32; LIMIT=""; NOASSERT=""
fi
if [ "$ENGINE" != stub ] && [ -z "$RUNPOD_POD_ID" ]; then
  echo "warning: RUNPOD_POD_ID is empty; the in-pod self-stop cannot fire. Pass RUNPOD_POD_ID=<id> in the launch command." >&2
fi
STUBS="$OUT/stub_models"
DONE_FILE="$OUT/pod.done"
TIMES="$OUT/stage_times.json"
mkdir -p "$OUT"
rm -f "$DONE_FILE" "$TIMES"
T0=$(date +%s)
DEADLINE=$(( T0 + HARD_STOP_S ))
trap 'rc=$?; echo "exit=$rc engine=$ENGINE at $(date -u +%FT%TZ) total_s=$(( $(date +%s) - T0 ))" > "$DONE_FILE"' EXIT
echo "exp47 pod chain; engine $ENGINE; RUNPOD_POD_ID ${RUNPOD_POD_ID:+set}${RUNPOD_POD_ID:-EMPTY}; rows -> $RES" | tee -a "$OUT/pod.log"
echo "deadlines: HARD_STOP_S=$HARD_STOP_S s, SELF_STOP_S=$SELF_STOP_S s; projection limit \$$PROJECT_LIMIT at \$$RATE/h" | tee -a "$OUT/pod.log"

if [ "$ENGINE" != stub ]; then
  nohup setsid bash -c "sleep $SELF_STOP_S; echo \"self-stop at \$(date -u +%FT%TZ)\" >> '$OUT/pod.log'; \
    if command -v runpodctl >/dev/null && [ -n \"\${RUNPOD_POD_ID:-}\" ]; then runpodctl stop pod \"\$RUNPOD_POD_ID\"; \
    else echo 'self-stop: runpodctl or RUNPOD_POD_ID missing; STOP THE POD BY HAND' >> '$OUT/pod.log'; fi" \
    > "$OUT/watchdog.log" 2>&1 < /dev/null &
  echo $! > "$OUT/watchdog.pid"
  echo "watchdog pid $(cat "$OUT/watchdog.pid"): pod stops itself at +${SELF_STOP_S} s" | tee -a "$OUT/pod.log"
fi

missing() { echo "preflight: missing input $1" >&2; exit 3; }
INPUTS=(pyproject.toml runs/exp47_bridge/rows47.jsonl runs/exp47_bridge/rows47_counts.json
        scripts/exp47_patch.py scripts/exp47_report.py scripts/exp47_pod_run.py
        memctl/analysis/bridge47.py memctl/analysis/maps.py memctl/analysis/steering.py memctl/analysis/answer_selection.py)
for f in "${INPUTS[@]}"; do test -e "$f" || missing "$f"; done
EXPECT=""
if [ -e bundle_info.json ]; then
  EXPECT=$($PY -c "import json; print(json.load(open('bundle_info.json')).get('rows47_sha256',''))" 2>/dev/null || true)
fi
echo "preflight ok: ${#INPUTS[@]} files present; out $OUT" | tee -a "$OUT/pod.log"

STAGE_START=$(date +%s)
stage_done() {   # stage_done <name> <status> <rc> [note]
  local now; now=$(date +%s)
  echo "stage $1 ($2, rc $3): elapsed $(( now - STAGE_START )) s (total $(( now - T0 )) s of $HARD_STOP_S)" | tee -a "$OUT/pod.log"
  $PY scripts/exp47_pod_run.py stage-time --file "$TIMES" --stage "$1" --status "$2" --rc "$3" \
    --elapsed $(( now - STAGE_START )) --t-offset $(( STAGE_START - T0 )) --note "${4:-}" \
    --hard-stop-s "$HARD_STOP_S" --self-stop-s "$SELF_STOP_S" --engine "$ENGINE" \
    || echo "stage-time: could not write $TIMES" | tee -a "$OUT/pod.log"
  STAGE_START=$now
}
left() { echo $(( DEADLINE - $(date +%s) )); }
status_of() { [ "$1" -eq 0 ] && echo ok || { { [ "$1" -eq 124 ] || [ "$1" -eq 9 ]; } && echo partial || echo failed; }; }

install_line() {
  TORCH_VERSION=$($PY -c "import torch; print(torch.__version__)")
  echo "torch==${TORCH_VERSION}" > "$OUT/constraints.txt"
  timeout 900 $PY -m pip install -q -c "$OUT/constraints.txt" -e . "transformers==${TRANSFORMERS_VERSION}" accelerate \
    huggingface_hub safetensors 2>&1 | tail -2
  $PY -c "import torch, transformers, numpy, huggingface_hub, safetensors, accelerate; \
import memctl.analysis.maps, memctl.analysis.bridge47, memctl.analysis.steering, memctl.analysis.answer_selection, \
scripts.exp47_patch, scripts.exp47_report, scripts.exp47_pod_run; \
print('torch', torch.__version__, 'transformers', transformers.__version__); \
assert transformers.__version__ == '${TRANSFORMERS_VERSION}'" 2>&1 | tee -a "$OUT/pod.log" \
    || { echo "install check failed" | tee -a "$OUT/pod.log"; exit 4; }
}

if [ "$ENGINE" != stub ]; then
  GPU_DEADLINE=$(( $(date +%s) + 120 ))
  until nvidia-smi > /dev/null 2>&1; do
    if [ "$(date +%s)" -ge "$GPU_DEADLINE" ]; then echo "GPU not visible after 120 s" | tee -a "$OUT/pod.log"; exit 4; fi
    sleep 5
  done
  install_line
  $PY -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; p = torch.cuda.get_device_properties(0); \
print('gpu', p.name, round(p.total_memory / 2**30, 1), 'GiB')" 2>&1 | tee -a "$OUT/pod.log" \
    || { echo "CUDA check failed" | tee -a "$OUT/pod.log"; exit 4; }
  export TOKENIZERS_PARALLELISM=false
  stage_done install ok 0
else
  export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=""
  if [ -n "$INSTALL" ]; then install_line; stage_done install ok 0; fi
fi

set +e
$PY scripts/exp47_pod_run.py population ${EXPECT:+--expect-rows-sha "$EXPECT"} 2>&1 | tee -a "$OUT/pod.log"
RC=${PIPESTATUS[0]}
set -e
[ "$RC" -eq 0 ] || { echo "population check failed (exit $RC)" | tee -a "$OUT/pod.log"; stage_done population failed "$RC"; exit 3; }
stage_done population ok 0

run_step() {   # run_step <log name> <timeout s> <command...>
  local name="$1" t="$2"; shift 2
  if [ "$t" -le 30 ]; then echo "SKIP $name: only $t s left" | tee -a "$OUT/pod.log"; return 124; fi
  set +e
  timeout "$t" "$@" 2>&1 | grep -v -E "unauthenticated|Writing model shards|Loading weights" | tee -a "$OUT/${name}.log"
  local rc=${PIPESTATUS[0]}
  set -e
  return "$rc"
}

# Stub check on this device: a tiny random Qwen2 (36 layers) with the real tokenizer; the Δ assert is off (a random
# model cannot reproduce the §44 deltas), the sanity holds by construction.
$PY scripts/exp47_pod_run.py make-stub --out "$STUBS" 2>&1 | { grep -v -E "unauthenticated|Writing model shards" || true; } | tee -a "$OUT/pod.log"
SW="--weights $STUBS/qwen3b"
SC=0
run_step stubcheck "$(left)" $PY scripts/exp47_patch.py --only ec81a493 $SW --device "$DEV" --dtype float32 --no-assert44 --out "$OUT/stubcheck" || SC=$?
if [ "$SC" -eq 0 ]; then
  run_step stubcheck "$(left)" $PY scripts/exp47_patch.py --only ec81a493 $SW --device "$DEV" --dtype float32 --no-assert44 --out "$OUT/stubcheck" || SC=$?
  REDONE=$($PY -c "import json; print(len(json.load(open('$OUT/stubcheck/_timing.json'))))")
  echo "stub resume: rows redone $REDONE (want 0)" | tee -a "$OUT/pod.log"
  [ "$REDONE" = 0 ] || SC=6
fi
if [ "$SC" -ne 0 ]; then echo "STUB CHECK FAILED (rc $SC): chain aborted before any download" | tee -a "$OUT/pod.log"; stage_done stub_check failed "$SC"; exit 5; fi
stage_done stub_check ok 0

if [ "$ENGINE" = stub ]; then
  W="$SW"
else
  set +e
  timeout $(( $(left) > 1800 ? 1800 : $(left) )) $PY - <<'PYEOF' 2>&1 | tee -a "$OUT/pod.log"
import time, huggingface_hub
t = time.time()
path = huggingface_hub.snapshot_download("Qwen/Qwen2.5-3B-Instruct", allow_patterns=["*.json", "*.safetensors", "*.txt", "tokenizer*"])
print(f"download Qwen/Qwen2.5-3B-Instruct: {time.time() - t:.0f} s -> {path}", flush=True)
PYEOF
  RC=${PIPESTATUS[0]}
  set -e
  [ "$RC" -eq 0 ] || { stage_done download failed "$RC"; echo "download failed (exit $RC)" | tee -a "$OUT/pod.log"; exit 4; }
  stage_done download ok 0
  W=""
fi

# D-22 smoke: the 3 smoke rows on the model; sanity and the Δ assert must hold on every row.
SM=0
run_step smoke "$(left)" $PY scripts/exp47_patch.py --smoke $W --device "$DEV" --dtype "$DT" $NOASSERT --out "$OUT/smoke" || SM=$?
if [ "$SM" -ne 0 ]; then echo "SMOKE FAILED (rc $SM): chain aborted before the full run" | tee -a "$OUT/pod.log"; stage_done smoke failed "$SM"; exit 5; fi
stage_done smoke ok 0

set +e
$PY scripts/exp47_pod_run.py project --smoke "$OUT/smoke/smoke.json" --elapsed $(( $(date +%s) - T0 )) --rate "$RATE" \
  --limit "$PROJECT_LIMIT" 2>&1 | tee -a "$OUT/pod.log"
RC=${PIPESTATUS[0]}
set -e
if [ "$RC" -ne 0 ]; then echo "PROJECTION above \$$PROJECT_LIMIT (rc $RC): chain stopped; ask before continuing" | tee -a "$OUT/pod.log"; stage_done projection stop "$RC"; exit 7; fi
stage_done projection ok 0

RR=0; run_step rows "$(left)" $PY scripts/exp47_patch.py $W --device "$DEV" --dtype "$DT" $NOASSERT --out "$RES" --deadline-s $(( $(left) - 30 )) $LIMIT || RR=$?
stage_done rows "$(status_of "$RR")" "$RR"
RP=0; run_step report "$(left)" $PY scripts/exp47_report.py --root "$RES" || RP=$?
stage_done report "$(status_of "$RP")" "$RP"
$PY scripts/exp47_pod_run.py status --out "$OUT" --root "$RES" 2>&1 | tee -a "$OUT/pod.log"
echo "all stages done in $(( $(date +%s) - T0 )) s (rows $RR, report $RP); copy back before the self-stop at +${SELF_STOP_S} s" | tee -a "$OUT/pod.log"
[ "$RR" -eq 0 ] && [ "$RP" -eq 0 ]
