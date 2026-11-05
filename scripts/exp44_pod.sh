#!/usr/bin/env bash
# §44 rev 8 pod stage (research/proposals/2026-10-29-exp44-patching.md "Rev 8", 83fe60a, D-47): one A40 runs
#   (A) the rev 0 7B confirmation, all 204 rows of folds 2-4, bf16 CUDA, eligible rows patched exactly as the laptop
#       run (scripts/exp44_patch.py --model qwen7b --folds 2,3,4 --run --confirm) -> runs/exp44_patch/qwen7b_gpu/;
#   (B) rev 8 S on Qwen 3B, fp32 CUDA (scripts/exp44_swap.py --model qwen3b --gate-rev 8 --pooled; S only)
#       -> runs/exp44_swap/qwen3b_rev8/S/.
# Order A then B: A is the shorter run by the code-path estimate (204 rows x 2 prefills + 28 single-token passes on
# about half the rows, ~5-10 min) and the critical path (the laptop 7B fallback is stopped once the GPU read is
# accepted); B (70 rows x 3 fp32 prefills + 147 single-token passes on the following rows, ~10-17 min) has B_RESERVE_S
# kept for it, so a slow A cannot starve it. Image: runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404 (its torch is
# kept); container disk >= 40 GB (7B bf16 15.2 GB + 3B 6.2 GB + venv).
#
#   RUNPOD_POD_ID=<id> bash scripts/exp44_pod.sh                  # on the pod, from the unpacked bundle
#   ENGINE=stub bash scripts/exp44_pod.sh                         # CPU, tiny random models, everything under OUT
#   ENGINE=stub INSTALL=1 PYTHON=<venv>/bin/python bash scripts/exp44_pod.sh   # D-30: the SAME install line
#   ENGINE=stub STUB_BREAK=layers bash scripts/exp44_pod.sh       # the smoke must fail (exit 5), no full run written
#
# Order: preflight -> install -> GPU check -> watchdog -> population (204 / 330 / 70 rows, the bundle's fingerprint,
# every rev 8 swap prompt re-derived against its sha; exit 3) -> pre-launch stub check (tiny random Qwen2 with the real
# tokenizer, ON THE POD'S DEVICE: 3 rows of A with --patch-all and 3 rows of B with every row forced to follow, so the
# patch passes and controls run; then A again to check the resume redoes nothing; exit 5) -> downloads (7B into
# cache/exp44_hf, where exp44_patch.resolve_model looks; 3B into the HF cache) -> D-22 smoke, 3 rows of EACH real run
# (A with --patch-all so the timing is the patched-row cost; B as configured), measured seconds per row printed; any
# failure exits 5 before either full run -> full A -> full B -> status -> pod.done.
# Deadlines: HARD_STOP_S (default 4500 s) after start all model work ends (each run gets `timeout` of what is left; A
# gets what is left minus B_RESERVE_S, default 1500 s); both runs are resumable, rows written so far are kept. A
# watchdog stops the pod itself at SELF_STOP_S (default 5400 s) with `runpodctl stop pod $RUNPOD_POD_ID`.
# $OUT/pod.done on any exit; $OUT/stage_times.json seconds per stage. No outcome is printed (counts and seconds only).
# Exit 3: input / population. 4: GPU / install / download. 5: stub check or smoke failed. Copy back: $OUT,
# runs/exp44_patch/qwen7b_gpu, runs/exp44_swap/qwen3b_rev8.
set -euo pipefail
export PIP_BREAK_SYSTEM_PACKAGES=1
cd "$(dirname "$0")/.."
ENGINE="${ENGINE:-hf}"
INSTALL="${INSTALL:-}"
TRANSFORMERS_VERSION=5.17.0
HARD_STOP_S="${HARD_STOP_S:-4500}"
SELF_STOP_S="${SELF_STOP_S:-5400}"
B_RESERVE_S="${B_RESERVE_S:-1500}"
STUB_BREAK="${STUB_BREAK:-}"
export RUNPOD_POD_ID="${RUNPOD_POD_ID:-}"
if [ "$ENGINE" = stub ]; then
  OUT="${OUT:-runs/exp44_pod_stub}"; PY="${PYTHON:-.venv/bin/python}"; DEV=cpu
  A_OUT="$OUT/full/qwen7b_gpu"; B_OUT="$OUT/full/qwen3b_rev8"
else
  OUT="${OUT:-runs/exp44_pod}"; PY="${PYTHON:-python}"; DEV=cuda
  A_OUT=runs/exp44_patch/qwen7b_gpu; B_OUT=runs/exp44_swap/qwen3b_rev8
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
echo "exp44 rev 8 pod chain; engine $ENGINE; RUNPOD_POD_ID ${RUNPOD_POD_ID:+set}${RUNPOD_POD_ID:-EMPTY}; A -> $A_OUT; B -> $B_OUT" | tee -a "$OUT/pod.log"
echo "deadlines: HARD_STOP_S=$HARD_STOP_S s (model work ends), B_RESERVE_S=$B_RESERVE_S s, SELF_STOP_S=$SELF_STOP_S s (pod stops itself)" | tee -a "$OUT/pod.log"

if [ "$ENGINE" != stub ]; then
  nohup setsid bash -c "sleep $SELF_STOP_S; echo \"self-stop at \$(date -u +%FT%TZ)\" >> '$OUT/pod.log'; \
    if command -v runpodctl >/dev/null && [ -n \"\${RUNPOD_POD_ID:-}\" ]; then runpodctl stop pod \"\$RUNPOD_POD_ID\"; \
    else echo 'self-stop: runpodctl or RUNPOD_POD_ID missing; STOP THE POD BY HAND' >> '$OUT/pod.log'; fi" \
    > "$OUT/watchdog.log" 2>&1 < /dev/null &
  echo $! > "$OUT/watchdog.pid"
  echo "watchdog pid $(cat "$OUT/watchdog.pid"): pod stops itself at +${SELF_STOP_S} s" | tee -a "$OUT/pod.log"
fi

# Preflight: every input the chain reads (scripts/exp44_bundle.sh packs exactly these and the code).
missing() { echo "preflight: missing input $1" >&2; exit 3; }
INPUTS=(pyproject.toml data/longmemeval/longmemeval_s_cleaned.json runs/exp23_verdicts.json
        runs/exp44_swapgate/gate.json runs/exp44_swapgate/rev8/gate.json runs/exp44_swapgate/rev8/eligible_qwen3b.jsonl
        scripts/exp44_patch.py scripts/exp44_swap.py scripts/exp44_swap_gate.py scripts/exp44_pod_run.py
        memctl/analysis/patching.py memctl/analysis/swap.py memctl/analysis/steering.py memctl/jacobian_data.py)
for READER in qwen3b qwen7b; do
  for FOLD in 0 1 2 3 4; do
    for FILE in items.jsonl.gz steps.jsonl.gz episodes.jsonl; do
      INPUTS+=("runs/exp23_${READER}_f${FOLD}/fixed8__fraction0.05/${FILE}")
    done
  done
done
for f in "${INPUTS[@]}"; do test -e "$f" || missing "$f"; done
for D in cache/generations_exp23_qwen3b cache/generations_exp23_qwen7b runs/exp34_jlens/qwen7b; do
  test -d "$D" && test -n "$(ls -A "$D")" || missing "$D (directory, non-empty)"
done
EXPECT_FP=""
if [ -e bundle_info.json ]; then
  EXPECT_FP=$($PY -c "import json; print(json.load(open('bundle_info.json')).get('population_fingerprint',''))" 2>/dev/null || true)
fi
echo "preflight ok: ${#INPUTS[@]} files and 3 folders present; out $OUT" | tee -a "$OUT/pod.log"

STAGE_START=$(date +%s)
stage_done() {   # stage_done <name> <status> <rc> [note]
  local now; now=$(date +%s)
  echo "stage $1 ($2, rc $3): elapsed $(( now - STAGE_START )) s (total $(( now - T0 )) s of $HARD_STOP_S)" | tee -a "$OUT/pod.log"
  $PY scripts/exp44_pod_run.py stage-time --file "$TIMES" --stage "$1" --status "$2" --rc "$3" \
    --elapsed $(( now - STAGE_START )) --t-offset $(( STAGE_START - T0 )) --note "${4:-}" \
    --hard-stop-s "$HARD_STOP_S" --self-stop-s "$SELF_STOP_S" --engine "$ENGINE" \
    || echo "stage-time: could not write $TIMES" | tee -a "$OUT/pod.log"
  STAGE_START=$now
}
left() { echo $(( DEADLINE - $(date +%s) )); }

# The install line (the pod's and the D-30 fresh-venv test's). Every import the chain makes is checked after it.
install_line() {
  TORCH_VERSION=$($PY -c "import torch; print(torch.__version__)")
  echo "torch==${TORCH_VERSION}" > "$OUT/constraints.txt"   # keep the installed torch; never let pip replace it
  timeout 900 $PY -m pip install -q -c "$OUT/constraints.txt" -e . \
    "transformers==${TRANSFORMERS_VERSION}" accelerate huggingface_hub safetensors 2>&1 | tail -2
  $PY -c "import torch, transformers, numpy, yaml, huggingface_hub, safetensors, accelerate; \
import memctl.analysis.steering, memctl.analysis.swap, memctl.analysis.patching, memctl.analysis.jlens_readout, \
memctl.analysis.jacobian, memctl.analysis.families, memctl.analysis.answer_selection, memctl.jacobian_data, \
memctl.judge, memctl.metrics, memctl.envs.longmemeval, memctl.memory.items, \
scripts.exp44_patch, scripts.exp44_swap, scripts.exp44_swap_gate, scripts.exp44_pod_run; \
print('torch', torch.__version__, 'transformers', transformers.__version__, 'numpy', numpy.__version__); \
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
  df -h . 2>/dev/null | tail -1 | tee -a "$OUT/pod.log" || true
  export TOKENIZERS_PARALLELISM=false
  stage_done install ok 0
else
  export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=""
  if [ -n "$INSTALL" ]; then install_line; stage_done install ok 0; fi
fi

set +e
$PY scripts/exp44_pod_run.py population ${EXPECT_FP:+--expect-fingerprint "$EXPECT_FP"} 2>&1 | grep -v "unauthenticated" | tee -a "$OUT/pod.log"
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

# Pre-launch stub check: tiny random models with the real tokenizer, on this machine's device.
STUB_LAYERS_A=3; STUB_LAYERS_B=4
[ "$STUB_BREAK" = layers ] && STUB_LAYERS_B=5   # D-30 negative test: the smoke check must refuse the layer count
$PY scripts/exp44_pod_run.py make-stub --out "$STUBS" 2>&1 | { grep -v -E "unauthenticated|Writing model shards" || true; } | tee -a "$OUT/pod.log"
SC=0
run_step stubcheck "$(left)" $PY scripts/exp44_patch.py --model qwen7b --folds 2,3,4 --confirm --run --limit 3 --patch-all \
  --weights "$STUBS/qwen7b" --out "$OUT/stubcheck/A" || SC=$?
[ "$SC" -eq 0 ] && { $PY scripts/exp44_pod_run.py smoke-check --kind patch --dir "$OUT/stubcheck/A" --n 3 \
  --engine "$DEV-bfloat16" --n-layers $STUB_LAYERS_A 2>&1 | tee -a "$OUT/pod.log"; SC=${PIPESTATUS[0]}; }
[ "$SC" -eq 0 ] && { run_step stubcheck "$(left)" $PY scripts/exp44_swap.py --model qwen3b --gate-rev 8 --pooled --limit 3 \
  --weights "$STUBS/qwen3b" --stub-force-follow --out "$OUT/stubcheck/B" || SC=$?; }
[ "$SC" -eq 0 ] && { $PY scripts/exp44_pod_run.py smoke-check --kind swap --dir "$OUT/stubcheck/B" --n 3 \
  --engine "$DEV-float32" --n-layers $STUB_LAYERS_B 2>&1 | tee -a "$OUT/pod.log"; SC=${PIPESTATUS[0]}; }
if [ "$SC" -eq 0 ]; then   # resume: a second A run over the same rows must redo nothing
  run_step stubcheck "$(left)" $PY scripts/exp44_patch.py --model qwen7b --folds 2,3,4 --confirm --run --limit 3 --patch-all \
    --weights "$STUBS/qwen7b" --out "$OUT/stubcheck/A" || SC=$?
  REDONE=$($PY -c "import json; print(json.load(open('$OUT/stubcheck/A/_timing.json'))['runs'][-1]['rows'])")
  echo "stub resume: rows redone $REDONE (want 0)" | tee -a "$OUT/pod.log"
  [ "$REDONE" = 0 ] || SC=6
fi
if [ "$SC" -ne 0 ]; then echo "STUB CHECK FAILED (rc $SC): chain aborted before any download" | tee -a "$OUT/pod.log"; stage_done stub_check failed "$SC"; exit 5; fi
stage_done stub_check ok 0

# Downloads (real) or the stub models (stub).
if [ "$ENGINE" = stub ]; then
  W_A="--weights $STUBS/qwen7b"; W_B="--weights $STUBS/qwen3b --stub-force-follow"; L_A=$STUB_LAYERS_A; L_B=$STUB_LAYERS_B
else
  set +e
  timeout $(( $(left) > 1800 ? 1800 : $(left) )) $PY - <<'EOF' 2>&1 | tee -a "$OUT/pod.log"
import time, huggingface_hub
keep = ["*.json", "*.safetensors", "*.txt", "*.model", "tokenizer*"]
for repo, cache in (("Qwen/Qwen2.5-7B-Instruct", "cache/exp44_hf"), ("Qwen/Qwen2.5-3B-Instruct", None)):
    t = time.time()
    path = huggingface_hub.snapshot_download(repo, cache_dir=cache, allow_patterns=keep)
    print(f"download {repo}: {time.time() - t:.0f} s -> {path}", flush=True)
EOF
  RC=${PIPESTATUS[0]}
  set -e
  [ "$RC" -eq 0 ] || { stage_done download failed "$RC"; echo "download failed (exit $RC)" | tee -a "$OUT/pod.log"; exit 4; }
  stage_done download ok 0
  W_A=""; W_B=""; L_A=28; L_B=36
fi

# D-22 smoke: 3 rows of each real run (A with --patch-all: the timing is the patched-row cost); abort on any failure.
SM=0
run_step smoke "$(left)" $PY scripts/exp44_patch.py --model qwen7b --folds 2,3,4 --confirm --run --limit 3 --patch-all \
  $W_A --out "$OUT/smoke/qwen7b" || SM=$?
[ "$SM" -eq 0 ] && { $PY scripts/exp44_pod_run.py smoke-check --kind patch --dir "$OUT/smoke/qwen7b" --n 3 \
  --engine "$DEV-bfloat16" --n-layers $L_A 2>&1 | tee -a "$OUT/pod.log"; SM=${PIPESTATUS[0]}; }
[ "$SM" -eq 0 ] && { run_step smoke "$(left)" $PY scripts/exp44_swap.py --model qwen3b --gate-rev 8 --pooled --limit 3 \
  $W_B --out "$OUT/smoke/qwen3b_rev8" || SM=$?; }
[ "$SM" -eq 0 ] && { $PY scripts/exp44_pod_run.py smoke-check --kind swap --dir "$OUT/smoke/qwen3b_rev8" --n 3 \
  --engine "$DEV-float32" --n-layers $L_B 2>&1 | tee -a "$OUT/pod.log"; SM=${PIPESTATUS[0]}; }
if [ "$SM" -ne 0 ]; then echo "SMOKE FAILED (rc $SM): chain aborted before the full runs" | tee -a "$OUT/pod.log"; stage_done smoke failed "$SM"; exit 5; fi
stage_done smoke ok 0

# (A) the 7B confirmation, all 204 rows, as the laptop run (eligible rows patched); keeps B_RESERVE_S for B.
RA=0
run_step full_A $(( $(left) - B_RESERVE_S )) $PY scripts/exp44_patch.py --model qwen7b --folds 2,3,4 --confirm --run \
  $W_A --out "$A_OUT" || RA=$?
stage_done full_A "$([ "$RA" -eq 0 ] && echo ok || { [ "$RA" -eq 124 ] && echo partial || echo failed; })" "$RA"
# (B) rev 8 S on Qwen 3B, fp32, S only.
RB=0
run_step full_B "$(left)" $PY scripts/exp44_swap.py --model qwen3b --gate-rev 8 --pooled $W_B --out "$B_OUT" || RB=$?
stage_done full_B "$([ "$RB" -eq 0 ] && echo ok || { [ "$RB" -eq 124 ] && echo partial || echo failed; })" "$RB"

$PY scripts/exp44_pod_run.py status --out "$OUT" --patch-dir "$A_OUT" --swap-dir "$B_OUT" 2>&1 | tee -a "$OUT/pod.log"
echo "all stages done in $(( $(date +%s) - T0 )) s (A rc $RA, B rc $RB); copy $OUT, $A_OUT, $B_OUT back before the self-stop at +${SELF_STOP_S} s" | tee -a "$OUT/pod.log"
[ "$RA" -eq 0 ] && [ "$RB" -eq 0 ]
