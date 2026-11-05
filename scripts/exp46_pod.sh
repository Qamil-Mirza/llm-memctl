#!/usr/bin/env bash
# §46 pod stage (research/proposals/2026-10-30-exp46-instrument-check.md, revs 0-3; D-54): one A40, Qwen2.5-3B bf16,
# CUDA, transformers 5.17.0. Never two pods. Image: runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404 (its torch kept).
#
#   RUNPOD_POD_ID=<id> bash scripts/exp46_pod.sh                     # on the pod, from the unpacked bundle
#   ENGINE=stub bash scripts/exp46_pod.sh                            # CPU, a tiny random Qwen2 (36 layers)
#   ENGINE=stub INSTALL=1 PYTHON=<venv>/bin/python bash scripts/exp46_pod.sh   # D-30: the SAME install line
#   ENGINE=stub PROJECT_LIMIT=0 bash scripts/exp46_pod.sh            # the projection must stop the chain (exit 7)
#
# Order: preflight -> watchdog -> GPU check + install -> population (576 RC/TU cases with their §45 stage-1 records,
# 144 triples, shas from bundle_info; exit 3) -> stub check on the pod's device (capture 3 prompts, patch smoke on 1
# triple, capture resume redoes nothing; exit 5) -> download the 3B -> D-22 smoke (capture 3 prompts; patch smoke on 3
# dev triples with the passes forced: k3 must reproduce P''s baseline outcome on every row; exit 5) -> projection
# (capture and patch rates scaled to the full run plus the seconds spent; above PROJECT_LIMIT dollars = $1.80 at
# $0.59/h the chain stops, exit 7, and the Executor asks) -> capture (576 prompts) -> (i) baselines (432 prompts) ->
# validate (dev baselines only) -> (ii) S1 dev, S1 held -> (iii) controls k1 / k2 -> (iv) S2 -> status -> pod.done.
# Deadlines: HARD_STOP_S (default 14400) after start all model work ends (each step stops between triples, exit 9,
# resumable). The watchdog stops the pod at SELF_STOP_S (default 15000) with `runpodctl stop pod $RUNPOD_POD_ID`; a
# laptop DELETE follows the copy-back. No outcome is printed (counts and seconds only). Exit 3: input. 4: GPU /
# install / download. 5: stub check or smoke failed. 7: projection above the limit.
# Copy back: $OUT, runs/exp46_probe/cuda-bfloat16, runs/exp46_patch/cuda-bfloat16.
set -euo pipefail
export PIP_BREAK_SYSTEM_PACKAGES=1
cd "$(dirname "$0")/.."
ENGINE="${ENGINE:-hf}"
INSTALL="${INSTALL:-}"
TRANSFORMERS_VERSION=5.17.0
HARD_STOP_S="${HARD_STOP_S:-14400}"
SELF_STOP_S="${SELF_STOP_S:-15000}"
PROJECT_LIMIT="${PROJECT_LIMIT:-$([ "${ENGINE:-hf}" = stub ] && echo 1000 || echo 1.80)}"
RATE="${RATE:-0.59}"
export RUNPOD_POD_ID="${RUNPOD_POD_ID:-}"
if [ "$ENGINE" = stub ]; then
  OUT="${OUT:-runs/exp46_pod_stub}"; PY="${PYTHON:-.venv/bin/python}"; DEV=cpu; DT=float32
  CAP="$OUT/full/capture"; PAT="$OUT/full/patch"; LIMIT="--every ${STUB_EVERY:-24}"
else
  OUT="${OUT:-runs/exp46_pod}"; PY="${PYTHON:-python}"; DEV=cuda; DT=bfloat16
  CAP=runs/exp46_probe/cuda-bfloat16; PAT=runs/exp46_patch/cuda-bfloat16; LIMIT=""
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
echo "exp46 pod chain; engine $ENGINE; RUNPOD_POD_ID ${RUNPOD_POD_ID:+set}${RUNPOD_POD_ID:-EMPTY}; capture -> $CAP; patch -> $PAT" | tee -a "$OUT/pod.log"
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
INPUTS=(pyproject.toml runs/exp45_maps/cases.jsonl runs/exp46_probe/cases46.jsonl runs/exp46_probe/cases46_counts.json
        scripts/exp46_capture.py scripts/exp46_patch.py scripts/exp46_pod_run.py scripts/exp45_cases.py
        memctl/analysis/probe46.py memctl/analysis/patch46.py memctl/analysis/maps.py memctl/analysis/jacobian.py
        runs/exp45_maps/j56-jrefit-logit/stage1/RC00_v0_F1_H0.json)
for f in "${INPUTS[@]}"; do test -e "$f" || missing "$f"; done
EXPECT=""; EXPECT46=""
if [ -e bundle_info.json ]; then
  EXPECT=$($PY -c "import json; print(json.load(open('bundle_info.json')).get('cases_sha256',''))" 2>/dev/null || true)
  EXPECT46=$($PY -c "import json; print(json.load(open('bundle_info.json')).get('cases46_sha256',''))" 2>/dev/null || true)
fi
echo "preflight ok: ${#INPUTS[@]} files present; out $OUT" | tee -a "$OUT/pod.log"

STAGE_START=$(date +%s)
stage_done() {   # stage_done <name> <status> <rc> [note]
  local now; now=$(date +%s)
  echo "stage $1 ($2, rc $3): elapsed $(( now - STAGE_START )) s (total $(( now - T0 )) s of $HARD_STOP_S)" | tee -a "$OUT/pod.log"
  $PY scripts/exp46_pod_run.py stage-time --file "$TIMES" --stage "$1" --status "$2" --rc "$3" \
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
    huggingface_hub safetensors scikit-learn 2>&1 | tail -2
  $PY -c "import torch, transformers, numpy, sklearn, huggingface_hub, safetensors, accelerate; \
import memctl.analysis.maps, memctl.analysis.jacobian, memctl.analysis.probe46, memctl.analysis.patch46, memctl.agents.llm, \
scripts.exp46_capture, scripts.exp46_patch, scripts.exp46_pod_run, scripts.exp46_probe, scripts.exp46_report, scripts.exp45_cases; \
print('torch', torch.__version__, 'transformers', transformers.__version__, 'sklearn', sklearn.__version__); \
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
$PY scripts/exp46_pod_run.py population ${EXPECT:+--expect-cases-sha "$EXPECT"} ${EXPECT46:+--expect-cases46-sha "$EXPECT46"} 2>&1 | tee -a "$OUT/pod.log"
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

# Stub check on this device: a tiny random Qwen2 (36 layers) with the real tokenizer.
$PY scripts/exp46_pod_run.py make-stub --out "$STUBS" 2>&1 | { grep -v -E "unauthenticated|Writing model shards" || true; } | tee -a "$OUT/pod.log"
SW="--weights $STUBS/qwen3b"
SC=0
run_step stubcheck "$(left)" $PY scripts/exp46_capture.py --limit 3 $SW --device "$DEV" --dtype float32 --out "$OUT/stubcheck/capture" || SC=$?
[ "$SC" -eq 0 ] && { run_step stubcheck "$(left)" $PY scripts/exp46_patch.py --smoke 1 $SW --device "$DEV" --dtype float32 --out "$OUT/stubcheck/patch" || SC=$?; }
if [ "$SC" -eq 0 ]; then
  run_step stubcheck "$(left)" $PY scripts/exp46_capture.py --limit 3 $SW --device "$DEV" --dtype float32 --out "$OUT/stubcheck/capture" || SC=$?
  REDONE=$($PY -c "import json; print(len(json.load(open('$OUT/stubcheck/capture/_timing.json'))))")
  echo "stub resume: prompts redone $REDONE (want 0)" | tee -a "$OUT/pod.log"
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

# D-22 smoke: capture 3 prompts, patch every stage on 3 dev triples (passes forced); abort on any failure.
SM=0
run_step smoke "$(left)" $PY scripts/exp46_capture.py --limit 3 $W --device "$DEV" --dtype "$DT" --out "$OUT/smoke/capture" || SM=$?
[ "$SM" -eq 0 ] && { run_step smoke "$(left)" $PY scripts/exp46_patch.py --smoke 3 $W --device "$DEV" --dtype "$DT" --out "$OUT/smoke/patch" || SM=$?; }
if [ "$SM" -ne 0 ]; then echo "SMOKE FAILED (rc $SM): chain aborted before the full runs" | tee -a "$OUT/pod.log"; stage_done smoke failed "$SM"; exit 5; fi
stage_done smoke ok 0

set +e
$PY scripts/exp46_pod_run.py project --capture-timing "$OUT/smoke/capture/_timing.json" --smoke "$OUT/smoke/patch/smoke.json" \
  --elapsed $(( $(date +%s) - T0 )) --rate "$RATE" --limit "$PROJECT_LIMIT" 2>&1 | tee -a "$OUT/pod.log"
RC=${PIPESTATUS[0]}
set -e
if [ "$RC" -ne 0 ]; then echo "PROJECTION above \$$PROJECT_LIMIT (rc $RC): chain stopped; ask before continuing" | tee -a "$OUT/pod.log"; stage_done projection stop "$RC"; exit 7; fi
stage_done projection ok 0

dl() { echo $(( $(left) - 20 )); }   # each Python step stops between items before the chain's deadline
RA=0; run_step capture "$(left)" $PY scripts/exp46_capture.py $W --device "$DEV" --dtype "$DT" --out "$CAP" --deadline-s "$(dl)" $LIMIT || RA=$?
stage_done capture "$(status_of "$RA")" "$RA"
STUB_TRIPLES=""; [ "$ENGINE" = stub ] && STUB_TRIPLES="--limit-triples ${STUB_TRIPLES_N:-2}"
RB=0; run_step baselines "$(left)" $PY scripts/exp46_patch.py --stage baselines $W --device "$DEV" --dtype "$DT" --out "$PAT" --deadline-s "$(dl)" $STUB_TRIPLES || RB=$?
stage_done baselines "$(status_of "$RB")" "$RB"
RV=0; run_step validate "$(left)" $PY scripts/exp46_patch.py --stage validate --out "$PAT" || RV=$?
stage_done validate "$(status_of "$RV")" "$RV"
R1=0
for SPLIT in dev held; do
  run_step s1 "$(left)" $PY scripts/exp46_patch.py --stage s1 --split "$SPLIT" $W --device "$DEV" --dtype "$DT" --out "$PAT" --deadline-s "$(dl)" $STUB_TRIPLES || R1=$?
done
stage_done s1 "$(status_of "$R1")" "$R1"
RK=0; run_step controls "$(left)" $PY scripts/exp46_patch.py --stage controls $W --device "$DEV" --dtype "$DT" --out "$PAT" --deadline-s "$(dl)" $STUB_TRIPLES || RK=$?
stage_done controls "$(status_of "$RK")" "$RK"
R2=0; run_step s2 "$(left)" $PY scripts/exp46_patch.py --stage s2 $W --device "$DEV" --dtype "$DT" --out "$PAT" --deadline-s "$(dl)" $STUB_TRIPLES || R2=$?
stage_done s2 "$(status_of "$R2")" "$R2"

$PY scripts/exp46_pod_run.py status --out "$OUT" --capture "$CAP" --patch "$PAT" 2>&1 | tee -a "$OUT/pod.log"
echo "all stages done in $(( $(date +%s) - T0 )) s (capture $RA, baselines $RB, validate $RV, s1 $R1, controls $RK, s2 $R2); copy back before the self-stop at +${SELF_STOP_S} s" | tee -a "$OUT/pod.log"
[ "$RA" -eq 0 ] && [ "$RB" -eq 0 ] && [ "$R1" -eq 0 ] && [ "$RK" -eq 0 ] && [ "$R2" -eq 0 ]
