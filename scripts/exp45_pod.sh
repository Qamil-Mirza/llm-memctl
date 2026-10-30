#!/usr/bin/env bash
# §45 pod stage (research/proposals/2026-10-30-exp45-jlens-maps.md §6, rev 0, 78d5ad1; D-49): one A40, Qwen2.5-3B
# bf16, CUDA, transformers 5.17.0, the vendored jacobian-lens. Runs ONLY after the §44 pod has terminated (never two
# pods). Image: runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404 (its torch is kept); container disk >= 30 GB.
#
#   RUNPOD_POD_ID=<id> bash scripts/exp45_pod.sh                     # on the pod, from the unpacked bundle
#   ENGINE=stub bash scripts/exp45_pod.sh                            # CPU, a tiny random Qwen2, everything under OUT
#   ENGINE=stub INSTALL=1 PYTHON=<venv>/bin/python bash scripts/exp45_pod.sh   # D-30: the SAME install line
#   ENGINE=stub STUB_BREAK=bos bash scripts/exp45_pod.sh             # the smoke must fail (exit 5), no full run written
#
# Order: preflight -> watchdog -> GPU check + install -> population (864 cases and their sha, 140 stage 3 prompts
# re-checked against the §44 gate's shas, the §34 lens sha; exit 3) -> pre-launch stub check ON THE POD'S DEVICE (tiny
# random Qwen2 with the real tokenizer: a 2-prompt stub lens, 3 smoke cases, 3 stage 1 + 2 stage 3 prompts, then
# stage 1 again: the resume must redo nothing; exit 5) -> download the 3B -> D-22 smoke (3 dev cases through the full
# map path with j56 + logit: lens shape, finite ranks, BOS count, the final-layer top-1 at the last prompt token
# equal to the laptop fp32 read in runs/exp45_maps/smoke_ref; exit 5 before any full run) -> stage A refit (a second
# 3B J-lens, 128 WikiText prompts, minimum 96, seed 1, §34 settings: dim batch 32, max_seq_len 128, skip_first 16,
# chunk 4; time box min(75 min, what is left after MAPS_RESERVE_S)) -> stage B maps: stage 1 (864 prompts, dev half
# first) with j56 + jrefit + logit when the refit sidecar says below_minimum false, else j56 + logit -> stage 3 (the
# 140 §44 rev 8 prompts) -> the §34 lens eval on the refit lens (reported, not a gate) -> status -> pod.done.
# Deadlines: HARD_STOP_S (default 14400 s) after start all model work ends; the maps runs stop between prompts before
# it (exit 9, prompts so far kept, resumable). A watchdog stops the pod itself at SELF_STOP_S (default 15000 s) with
# `runpodctl stop pod $RUNPOD_POD_ID`; a laptop DELETE follows the copy-back. $OUT/pod.done on any exit;
# $OUT/stage_times.json seconds per stage. No outcome is printed (counts and seconds only; the 3 smoke cases' top-1 ids).
# Exit 3: input / population. 4: GPU / install / download. 5: stub check or smoke failed. Copy back: $OUT,
# runs/exp45_maps/{j56-jrefit-logit,j56-logit,lenses/qwen3b.json,eval}.
set -euo pipefail
export PIP_BREAK_SYSTEM_PACKAGES=1
cd "$(dirname "$0")/.."
ENGINE="${ENGINE:-hf}"
INSTALL="${INSTALL:-}"
TRANSFORMERS_VERSION=5.17.0
HARD_STOP_S="${HARD_STOP_S:-14400}"
SELF_STOP_S="${SELF_STOP_S:-15000}"
FIT_MAX_MIN="${FIT_MAX_MIN:-75}"
MAPS_RESERVE_S="${MAPS_RESERVE_S:-4200}"     # stage 1 (~25-35 min) + stage 3 (~10 min) + eval (~5 min), with margin
STUB_BREAK="${STUB_BREAK:-}"
export RUNPOD_POD_ID="${RUNPOD_POD_ID:-}"
if [ "$ENGINE" = stub ]; then
  OUT="${OUT:-runs/exp45_pod_stub}"; PY="${PYTHON:-.venv/bin/python}"; DEV=cpu
  MAPS_ROOT="$OUT/full"; N_FIT=4; MIN_FIT=3; STUB_LIMIT="--limit ${STUB_LIMIT:-6}"; STUB_LIMIT3="--limit ${STUB_LIMIT3:-2}"
  EVAL_LIMIT="--limit 2"; FIT_DIM=8; FIT_SEQ=64; FIT_SKIP=4
else
  OUT="${OUT:-runs/exp45_pod}"; PY="${PYTHON:-python}"; DEV=cuda
  MAPS_ROOT=runs/exp45_maps; N_FIT=128; MIN_FIT=96; STUB_LIMIT=""; STUB_LIMIT3=""; EVAL_LIMIT=""; FIT_DIM=32; FIT_SEQ=128; FIT_SKIP=16
fi
LENSES="$MAPS_ROOT/lenses"
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
echo "exp45 pod chain; engine $ENGINE; RUNPOD_POD_ID ${RUNPOD_POD_ID:+set}${RUNPOD_POD_ID:-EMPTY}; maps -> $MAPS_ROOT" | tee -a "$OUT/pod.log"
echo "deadlines: HARD_STOP_S=$HARD_STOP_S s (model work ends), SELF_STOP_S=$SELF_STOP_S s (pod stops itself), refit box <= $FIT_MAX_MIN min, maps reserve $MAPS_RESERVE_S s" | tee -a "$OUT/pod.log"

if [ "$ENGINE" != stub ]; then
  nohup setsid bash -c "sleep $SELF_STOP_S; echo \"self-stop at \$(date -u +%FT%TZ)\" >> '$OUT/pod.log'; \
    if command -v runpodctl >/dev/null && [ -n \"\${RUNPOD_POD_ID:-}\" ]; then runpodctl stop pod \"\$RUNPOD_POD_ID\"; \
    else echo 'self-stop: runpodctl or RUNPOD_POD_ID missing; STOP THE POD BY HAND' >> '$OUT/pod.log'; fi" \
    > "$OUT/watchdog.log" 2>&1 < /dev/null &
  echo $! > "$OUT/watchdog.pid"
  echo "watchdog pid $(cat "$OUT/watchdog.pid"): pod stops itself at +${SELF_STOP_S} s" | tee -a "$OUT/pod.log"
fi

# Preflight: every input the chain reads (scripts/exp45_bundle.sh packs exactly these and the code).
missing() { echo "preflight: missing input $1" >&2; exit 3; }
INPUTS=(pyproject.toml runs/exp45_maps/cases.jsonl runs/exp45_maps/cases_counts.json runs/exp45_maps/rows44.jsonl
        runs/exp34_jlens/lenses/qwen3b.pt runs/exp34_jlens/lenses/qwen3b.json
        data/jlens_corpus/wikitext-103-raw-v1-test.parquet external/jacobian-lens/pyproject.toml
        scripts/exp45_maps.py scripts/exp45_pod_run.py scripts/jlens_fit.py scripts/jlens_eval.py
        memctl/analysis/maps.py memctl/analysis/jlens_readout.py memctl/analysis/jacobian.py)
for SET in association multihop multilingual order-ops poetry typo; do
  INPUTS+=("external/jacobian-lens/data/evaluations/lens-eval-${SET}.json")
done
[ "$ENGINE" != stub ] && INPUTS+=(runs/exp45_maps/smoke_ref/reference.json)
for f in "${INPUTS[@]}"; do test -e "$f" || missing "$f"; done
EXPECT_SHA=""
if [ -e bundle_info.json ]; then
  EXPECT_SHA=$($PY -c "import json; print(json.load(open('bundle_info.json')).get('cases_sha256',''))" 2>/dev/null || true)
fi
echo "preflight ok: ${#INPUTS[@]} files present; out $OUT" | tee -a "$OUT/pod.log"

STAGE_START=$(date +%s)
stage_done() {   # stage_done <name> <status> <rc> [note]
  local now; now=$(date +%s)
  echo "stage $1 ($2, rc $3): elapsed $(( now - STAGE_START )) s (total $(( now - T0 )) s of $HARD_STOP_S)" | tee -a "$OUT/pod.log"
  $PY scripts/exp45_pod_run.py stage-time --file "$TIMES" --stage "$1" --status "$2" --rc "$3" \
    --elapsed $(( now - STAGE_START )) --t-offset $(( STAGE_START - T0 )) --note "${4:-}" \
    --hard-stop-s "$HARD_STOP_S" --self-stop-s "$SELF_STOP_S" --engine "$ENGINE" \
    || echo "stage-time: could not write $TIMES" | tee -a "$OUT/pod.log"
  STAGE_START=$now
}
left() { echo $(( DEADLINE - $(date +%s) )); }
status_of() { [ "$1" -eq 0 ] && echo ok || { { [ "$1" -eq 124 ] || [ "$1" -eq 9 ]; } && echo partial || echo failed; }; }

# The install line (the pod's and the D-30 fresh-venv test's). Every import the chain makes is checked after it.
install_line() {
  TORCH_VERSION=$($PY -c "import torch; print(torch.__version__)")
  echo "torch==${TORCH_VERSION}" > "$OUT/constraints.txt"   # keep the installed torch; never let pip replace it
  timeout 900 $PY -m pip install -q -c "$OUT/constraints.txt" -e . -e external/jacobian-lens \
    "transformers==${TRANSFORMERS_VERSION}" accelerate huggingface_hub safetensors pyarrow 2>&1 | tail -2
  $PY -c "import torch, transformers, jlens, jlens.hooks, numpy, pyarrow, pyarrow.parquet, huggingface_hub, safetensors, \
accelerate, yaml, matplotlib; import memctl.analysis.maps, memctl.analysis.jlens_readout, memctl.analysis.jacobian, \
memctl.agents.llm, memctl.metrics, scripts.exp45_maps, scripts.exp45_pod_run, scripts.jlens_fit, scripts.jlens_eval, \
scripts.exp45_render, scripts.exp45_report; print('torch', torch.__version__, 'transformers', transformers.__version__, \
'numpy', numpy.__version__); assert transformers.__version__ == '${TRANSFORMERS_VERSION}'" 2>&1 | tee -a "$OUT/pod.log" \
    || { echo "install check failed" | tee -a "$OUT/pod.log"; exit 4; }
  echo "jlens commit $(git -C external/jacobian-lens rev-parse --short HEAD 2>/dev/null || cat external/jacobian-lens/COMMIT 2>/dev/null || echo unknown)" | tee -a "$OUT/pod.log"
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
  df -h . 2>/dev/null | tail -1 | tee -a "$OUT/pod.log" || true
  export TOKENIZERS_PARALLELISM=false
  stage_done install ok 0
else
  export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=""
  if [ -n "$INSTALL" ]; then install_line; stage_done install ok 0; fi
fi

set +e
$PY scripts/exp45_pod_run.py population ${EXPECT_SHA:+--expect-cases-sha "$EXPECT_SHA"} 2>&1 | tee -a "$OUT/pod.log"
RC=${PIPESTATUS[0]}
set -e
[ "$RC" -eq 0 ] || { echo "population check failed (exit $RC)" | tee -a "$OUT/pod.log"; stage_done population failed "$RC"; exit 3; }
stage_done population ok 0

# run_step <log name> <timeout s> <command...>: one model process, logged; returns its exit code.
run_step() {
  local name="$1" t="$2"; shift 2
  if [ "$t" -le 30 ]; then echo "SKIP $name: only $t s left" | tee -a "$OUT/pod.log"; return 124; fi
  set +e
  timeout "$t" "$@" 2>&1 | grep -v -E "unauthenticated|Writing model shards" | tee -a "$OUT/${name}.log"
  local rc=${PIPESTATUS[0]}
  set -e
  return "$rc"
}

# Pre-launch stub check: a tiny random Qwen2 with the real tokenizer, on this machine's device.
$PY scripts/exp45_pod_run.py make-stub --out "$STUBS" 2>&1 | { grep -v -E "unauthenticated|Writing model shards" || true; } | tee -a "$OUT/pod.log"
SW="--weights $STUBS/qwen3b"
SC=0
run_step stubcheck "$(left)" $PY scripts/jlens_fit.py --reader qwen3b $SW --n-prompts 2 --min-prompts 2 --max-minutes 0 \
  --dim-batch 8 --max-seq-len 64 --skip-first 4 --device "$DEV" --dtype float32 --out "$OUT/stubcheck/lenses" || SC=$?
STUB_LENS="$OUT/stubcheck/lenses/qwen3b.pt"
[ "$SC" -eq 0 ] && { run_step stubcheck "$(left)" $PY scripts/exp45_maps.py --smoke 3 --lens56 "$STUB_LENS" --allow-lens-sha \
  $SW --device "$DEV" --out "$OUT/stubcheck/smoke" ${STUB_BREAK:+--break-bos} || SC=$?; }
[ "$SC" -eq 0 ] && { run_step stubcheck "$(left)" $PY scripts/exp45_maps.py --stage 1 --limit 3 --lens56 "$STUB_LENS" \
  --allow-lens-sha $SW --device "$DEV" --root "$OUT/stubcheck/maps" || SC=$?; }
[ "$SC" -eq 0 ] && { run_step stubcheck "$(left)" $PY scripts/exp45_maps.py --stage 3 --limit 2 --lens56 "$STUB_LENS" \
  --allow-lens-sha $SW --device "$DEV" --root "$OUT/stubcheck/maps" || SC=$?; }
if [ "$SC" -eq 0 ]; then   # resume: a second stage 1 run over the same prompts must redo nothing
  run_step stubcheck "$(left)" $PY scripts/exp45_maps.py --stage 1 --limit 3 --lens56 "$STUB_LENS" --allow-lens-sha $SW \
    --device "$DEV" --root "$OUT/stubcheck/maps" || SC=$?
  REDONE=$($PY -c "import json; print(json.load(open('$OUT/stubcheck/maps/j56-logit/stage1/_timing.json'))['runs'][-1]['prompts'])")
  echo "stub resume: prompts redone $REDONE (want 0)" | tee -a "$OUT/pod.log"
  [ "$REDONE" = 0 ] || SC=6
fi
if [ "$SC" -ne 0 ]; then echo "STUB CHECK FAILED (rc $SC): chain aborted before any download" | tee -a "$OUT/pod.log"; stage_done stub_check failed "$SC"; exit 5; fi
stage_done stub_check ok 0

# Download (real) or the stub (stub: the tiny model stands in for the 3B; a stub j56 lens stands in for the §34 lens).
if [ "$ENGINE" = stub ]; then
  W="$SW"; L56="$STUB_LENS"; SHA_FLAG="--allow-lens-sha"; REF=""; W_PATH="$STUBS/qwen3b"
else
  set +e
  timeout $(( $(left) > 1800 ? 1800 : $(left) )) $PY - <<'EOF' 2>&1 | tee -a "$OUT/pod.log"
import time, huggingface_hub
t = time.time()
path = huggingface_hub.snapshot_download("Qwen/Qwen2.5-3B-Instruct", allow_patterns=["*.json", "*.safetensors", "*.txt", "tokenizer*"])
print(f"download Qwen/Qwen2.5-3B-Instruct: {time.time() - t:.0f} s -> {path}", flush=True)
EOF
  RC=${PIPESTATUS[0]}
  set -e
  [ "$RC" -eq 0 ] || { stage_done download failed "$RC"; echo "download failed (exit $RC)" | tee -a "$OUT/pod.log"; exit 4; }
  stage_done download ok 0
  W=""; L56=runs/exp34_jlens/lenses/qwen3b.pt; SHA_FLAG=""; REF="--reference runs/exp45_maps/smoke_ref"; W_PATH=""
fi

# D-22 smoke: 3 dev cases through the full map path with j56 + logit; abort on any failure.
SM=0
run_step smoke "$(left)" $PY scripts/exp45_maps.py --smoke 3 --lens56 "$L56" $SHA_FLAG $W $REF --device "$DEV" \
  --out "$OUT/smoke" ${STUB_BREAK:+--break-bos} || SM=$?
if [ "$SM" -ne 0 ]; then echo "SMOKE FAILED (rc $SM): chain aborted before the full runs" | tee -a "$OUT/pod.log"; stage_done smoke failed "$SM"; exit 5; fi
stage_done smoke ok 0

# Stage A: the refit lens, time box min(FIT_MAX_MIN, what is left after the maps reserve and one chunk).
BOX=$($PY scripts/exp45_pod_run.py refit-box --left "$(left)" --max-min "$FIT_MAX_MIN" --reserve "$MAPS_RESERVE_S")
[ "$ENGINE" = stub ] && BOX=0   # stub: one chunk (the box is checked between chunks)
RF=0
if [ "$ENGINE" != stub ] && [ "$BOX" -eq 0 ]; then
  echo "SKIP refit: $(left) s left, maps reserve $MAPS_RESERVE_S s" | tee -a "$OUT/pod.log"; stage_done refit skipped -1 "box 0"
else
  echo "refit qwen3b: $N_FIT prompts (min $MIN_FIT), seed 1, box $BOX min" | tee -a "$OUT/pod.log"
  run_step refit $(( $(left) - MAPS_RESERVE_S / 2 )) $PY scripts/jlens_fit.py --reader qwen3b ${W_PATH:+--weights "$W_PATH"} \
    --n-prompts "$N_FIT" --min-prompts "$MIN_FIT" --max-minutes "$BOX" --dim-batch "$FIT_DIM" --max-seq-len "$FIT_SEQ" \
    --skip-first "$FIT_SKIP" --seed 1 --device "$DEV" --dtype "$([ "$ENGINE" = stub ] && echo float32 || echo bfloat16)" \
    --out "$LENSES" || RF=$?
  stage_done refit "$(status_of "$RF")" "$RF" "$(grep -Eo '"n_prompts":[[:space:]]*[0-9]+|"below_minimum":[[:space:]]*(true|false)' "$LENSES/qwen3b.json" 2>/dev/null | tr '\n' ' ')"
fi

# Stage B: maps, stage 1 then stage 3 (exp45_maps uses the refit lens only if its sidecar says below_minimum false).
REFIT_ARG=""; [ -e "$LENSES/qwen3b.pt" ] && REFIT_ARG="--lens-refit $LENSES/qwen3b.pt"
# reserves kept after stage 1 for stage 3 + eval, and after stage 3 for the eval (stub: none, so a short HARD_STOP_S
# cuts stage 1 itself, which the D-30 deadline test needs)
EVAL_RESERVE=$([ "$ENGINE" = stub ] && echo 0 || echo 300)
S3_RESERVE=$([ "$ENGINE" = stub ] && echo 0 || echo 900)
R1=0
run_step maps_stage1 $(( $(left) - S3_RESERVE - EVAL_RESERVE )) $PY scripts/exp45_maps.py --stage 1 --lens56 "$L56" $SHA_FLAG $REFIT_ARG $W \
  --device "$DEV" --root "$MAPS_ROOT" --deadline $(( DEADLINE - S3_RESERVE - EVAL_RESERVE )) $STUB_LIMIT || R1=$?
stage_done maps_stage1 "$(status_of "$R1")" "$R1"
R3=0
run_step maps_stage3 $(( $(left) - EVAL_RESERVE )) $PY scripts/exp45_maps.py --stage 3 --lens56 "$L56" $SHA_FLAG $REFIT_ARG $W \
  --device "$DEV" --root "$MAPS_ROOT" --deadline $(( DEADLINE - EVAL_RESERVE )) $STUB_LIMIT3 || R3=$?
stage_done maps_stage3 "$(status_of "$R3")" "$R3"

# The §34 lens eval on the refit lens (reported, not a gate).
RE=0
if [ -e "$LENSES/qwen3b.pt" ]; then
  # jlens_eval ignores --weights for Qwen readers: the stub goes in through --model-id (weights and tokenizer)
  run_step eval "$(left)" $PY scripts/jlens_eval.py --reader qwen3b ${W_PATH:+--model-id "$W_PATH"} --lens "$LENSES/qwen3b.pt" \
    --device "$DEV" --dtype "$([ "$ENGINE" = stub ] && echo float32 || echo bfloat16)" --out "$MAPS_ROOT/eval" $EVAL_LIMIT || RE=$?
  stage_done eval "$(status_of "$RE")" "$RE"
else
  stage_done eval skipped -1 "no refit lens"
fi

$PY scripts/exp45_pod_run.py status --out "$OUT" --maps-root "$MAPS_ROOT" --lenses "$LENSES" 2>&1 | tee -a "$OUT/pod.log"
echo "all stages done in $(( $(date +%s) - T0 )) s (refit rc $RF, stage 1 rc $R1, stage 3 rc $R3, eval rc $RE); copy $OUT and $MAPS_ROOT back before the self-stop at +${SELF_STOP_S} s" | tee -a "$OUT/pod.log"
[ "$R1" -eq 0 ] && [ "$R3" -eq 0 ]
