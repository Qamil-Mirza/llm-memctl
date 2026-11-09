#!/usr/bin/env bash
# selfdoubt Stage 0 FULL pod: the user's "1.7B-first" plan of 2026-11-09 (PROPOSAL.md, Budget, "Stage 0 plan chosen
# by the user"). Held-out split only; generation only, no scoring (readings need the frozen scorer at 1baf905 and
# --heldout-confirmed, and go to the Advisor first). The Executor needs the user's own line in its pane before launch.
# Steps in priority order, each until done or the run deadline (graceful: finished requests kept, streamed to disk):
#   1. Qwen3-1.7B GSM8K, all 250 held-out
#   2. Qwen3-1.7B MATH-500, the first 100 held-out in seed-56 shuffle order (common.shuffled_heldout)
#   3. Qwen3-8B GSM8K, all 250 held-out
# Order: preflight -> self-stop watchdog -> GPU -> hub + background downloads -> vLLM install -> split sha -> steps.
# Copy back runs/selfdoubt/full (per-model gen.jsonl, phases.jsonl, run_meta.json, measure.json, logs, env.txt).
#   RUNPOD_POD_ID=<id> RUN_S=<s> SELF_STOP_S=<s> bash selfdoubt/stage0_full_pod.sh
#   ENGINE=stub PYTHON=<venv>/bin/python bash selfdoubt/stage0_full_pod.sh       # files + imports + split only
# Laptop rehearsal (dev split, never held-out): SPLIT=dev N1=3 N2=2 N3=3 M8=M17=Qwen/Qwen3-0.6B R8=R17=<rev>
#   EXTRA="--max-model-len 3072 ..." in a scratch copy.
# Knobs: MAXTOK (16384), USD_H (0.60), CHUNK (4096 = one generate call per phase), FULLFRAC (0.3 of a step's time
# for its FULL phase, so a step cut by the deadline still has continuations), SPLIT (heldout), N1/N2/N3 (items per
# step: all / 100 / all).
set -euo pipefail
export PIP_BREAK_SYSTEM_PACKAGES=1 HF_HUB_DISABLE_PROGRESS_BARS=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 TOKENIZERS_PARALLELISM=false
cd "$(dirname "$0")/.."
ENGINE="${ENGINE:-vllm}"; RUN_S="${RUN_S:-1500}"; MAXTOK="${MAXTOK:-16384}"; SELF_STOP_S="${SELF_STOP_S:-}"
USD_H="${USD_H:-0.60}"; CHUNK="${CHUNK:-4096}"; FULLFRAC="${FULLFRAC:-0.3}"
SPLIT="${SPLIT:-heldout}"; N1="${N1:-}"; N2="${N2:-100}"; N3="${N3:-}"
OUT=runs/selfdoubt/full; PY="${PYTHON:-python}"
mkdir -p "$OUT"; rm -f "$OUT/pod.done"
T0=$(date +%s); DEADLINE=$(( T0 + RUN_S ))
trap 'rc=$?; echo "exit=$rc engine=$ENGINE at $(date -u +%FT%TZ) total_s=$(( $(date +%s) - T0 ))" > "$OUT/pod.done"' EXIT
log() { echo "$(date -u +%T) $*" | tee -a "$OUT/pod.log"; }
left() { echo $(( DEADLINE - $(date +%s) )); }
log "selfdoubt stage0 full pod; split $SPLIT; engine $ENGINE; run budget $RUN_S s; self-stop ${SELF_STOP_S:-none} s; max_tokens $MAXTOK"

# Pins (2026-11-08). vLLM 0.11.0 is the newest release whose PyPI metadata pins torch==2.8.0 (0.10.2 also does;
# 0.11.1+ pin torch 2.9.0; the authors' 0.10.1 pins torch 2.7.1), so the image's torch 2.8.0+cu128 is kept.
# transformers 4.57.1: vLLM 0.11.0 has no upper bound and 5.x is untested with it. Tested on a laptop RTX 3050 with
# this exact line (Qwen3-0.6B, 2026-11-08).
VLLM_PIN="vllm==0.11.0"; TF_PIN="transformers==4.57.1"; HUB_PIN="huggingface_hub[hf_xet]<1.0"
M8="${M8:-Qwen/Qwen3-8B}";    R8="${R8:-b968826d9c46dd6066d109eabc6255188de91218}"
M17="${M17:-Qwen/Qwen3-1.7B}"; R17="${R17:-70d244cc86ccca08cf5af4e1e306ecf908b1ad5e}"
EXTRA="${EXTRA:-}"   # extra stage0_run.py args (laptop rehearsal only: --max-model-len/--gpu-memory-utilization/...)
SPLIT_SHA=c9e99ab227e45d10298d91d4f0a17e150cfefa3a24d872555a2330c7d9392703
UPSTREAM=b87d61bd2e5e998f34878fb0ee5fb2f5550f98c6

for f in selfdoubt/__init__.py selfdoubt/common.py selfdoubt/engines.py selfdoubt/scoring.py selfdoubt/stage0_run.py \
         selfdoubt/stage0_report.py selfdoubt/stage0_full_pod.sh selfdoubt/vendor/interrupt_lrm/__init__.py \
         selfdoubt/vendor/interrupt_lrm/prompt_utils_qwen.py selfdoubt/vendor/interrupt_lrm/math_utils.py \
         selfdoubt/vendor/interrupt_lrm/math_parsing_util.py selfdoubt/vendor/interrupt_lrm/LICENSE \
         selfdoubt/vendor/interrupt_lrm/prompts/intervene/wo_prompt_guidance.json \
         selfdoubt/vendor/interrupt_lrm/prompts/intervene/w_prompt_guidance.json runs/selfdoubt/split.json; do
  test -e "$f" || { log "missing $f"; exit 3; }; done
sha() { sha256sum "$1" | cut -d' ' -f1; }
[ "$(sha runs/selfdoubt/split.json)" = "$SPLIT_SHA" ] || { log "split.json sha mismatch"; exit 3; }
[ "$(sha selfdoubt/vendor/interrupt_lrm/prompts/intervene/wo_prompt_guidance.json)" = \
  c8c2eff691410da2e3cd8216186107465d921bf6d7c7e06185b9eb232fe7ca42 ] || { log "wo wrapper sha mismatch"; exit 3; }
[ "$(sha selfdoubt/vendor/interrupt_lrm/prompts/intervene/w_prompt_guidance.json)" = \
  5108777179a4385cef92bbd1227c10f5529d2ac20d726f9d1f61831864ce15f8 ] || { log "w wrapper sha mismatch"; exit 3; }
grep -q "$UPSTREAM" selfdoubt/vendor/interrupt_lrm/__init__.py || { log "vendored commit mismatch"; exit 3; }

import_check() {
  $PY -c "import sys; sys.path.insert(0, '.'); from selfdoubt import common, engines, scoring, stage0_run, stage0_report; \
import importlib.util as u; print('selfdoubt imports ok; vllm', 'present' if u.find_spec('vllm') else 'absent')" \
    | tee -a "$OUT/pod.log" || { log "import check failed"; exit 4; }
}
split_check() {   # recomputes the split from the pinned dataset revision and compares bytes with the shipped file
  $PY -c "import sys; sys.path.insert(0, '.'); from selfdoubt import common as C; s, sha = C.ensure_split(C.load_items()); \
assert sha == '$SPLIT_SHA', sha; print('split sha ok', sha)" | tee -a "$OUT/pod.log" || { log "split sha mismatch"; exit 3; }
}

if [ "$ENGINE" = stub ]; then
  export CUDA_VISIBLE_DEVICES=""; import_check; split_check
  log "stub: preflight only (files, wrapper sha, imports, split sha); no GPU work"; exit 0
fi

# Self-stop: the pod removes itself if the laptop vanishes. An ssh shell lacks the container's RUNPOD_* env, so read
# it from PID 1 (RUNPOD_POD_ID is also exported by the launch command). The laptop's DELETE timer stays the backstop.
if [ -n "$SELF_STOP_S" ]; then
  nohup setsid bash -c "sleep $SELF_STOP_S; echo \"\$(date -u +%T) self-stop\" >> '$OUT/pod.log'; \
    for kv in \$(tr '\\0' '\\n' < /proc/1/environ | grep -E '^RUNPOD_(API_KEY|POD_ID)='); do export \"\$kv\"; done; \
    if command -v runpodctl >/dev/null && [ -n \"\${RUNPOD_POD_ID:-}\" ]; then runpodctl remove pod \"\$RUNPOD_POD_ID\" \
    || runpodctl stop pod \"\$RUNPOD_POD_ID\"; else echo 'STOP BY HAND' >> '$OUT/pod.log'; fi" \
    > "$OUT/watchdog.log" 2>&1 < /dev/null &
  log "self-stop watchdog armed at +$SELF_STOP_S s (runpodctl $(command -v runpodctl || echo MISSING))"
fi

until nvidia-smi >/dev/null 2>&1; do [ $(( $(date +%s) - T0 )) -lt 120 ] || { log "no GPU"; exit 4; }; sleep 5; done
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | tee -a "$OUT/pod.log"
TORCH_VERSION=$($PY -c "import torch; print(torch.__version__.split('+')[0])"); echo "torch==${TORCH_VERSION}" > "$OUT/constraints.txt"
[ "$TORCH_VERSION" = 2.8.0 ] || { log "image torch $TORCH_VERSION != 2.8.0 (vLLM 0.11.0 pins 2.8.0)"; exit 4; }

# Hub first (seconds), then both model downloads in the background while vLLM installs.
timeout 300 $PY -m pip install -q -c "$OUT/constraints.txt" "$HUB_PIN" 2>&1 | tail -2
snap() { $PY -c "import huggingface_hub as h; print(h.snapshot_download('$1', revision='$2', \
allow_patterns=['*.json','*.safetensors','*.txt','tokenizer*','merges.txt','vocab.json']))"; }
( timeout 900 bash -c "$(declare -f snap); PY=$PY; snap $M17 $R17" > "$OUT/snap17.path" 2> "$OUT/snap17.err"; echo $? > "$OUT/snap17.rc" ) &
( timeout 900 bash -c "$(declare -f snap); PY=$PY; snap $M8 $R8" > "$OUT/snap8.path" 2> "$OUT/snap8.err"; echo $? > "$OUT/snap8.rc" ) &
log "downloads started; installing $VLLM_PIN $TF_PIN"
timeout 900 $PY -m pip install -q -c "$OUT/constraints.txt" "$VLLM_PIN" "$TF_PIN" "$HUB_PIN" \
  latex2sympy2==1.9.1 word2number==1.1 pyarrow numpy 2>&1 | tail -3
$PY -c "import vllm, transformers, torch; assert vllm.__version__ == '0.11.0', vllm.__version__; \
assert transformers.__version__ == '4.57.1'; assert torch.__version__.startswith('2.8.0'); print(vllm.__version__, transformers.__version__, torch.__version__)" \
  | tee -a "$OUT/pod.log" || { log "install check failed"; exit 4; }
$PY -m pip freeze 2>/dev/null | grep -iE "^(vllm|torch|transformers|xformers|flashinfer|tokenizers|huggingface|hf.xet|latex2sympy2|antlr4|sympy)" > "$OUT/env.txt" || true
log "install done at +$(( $(date +%s) - T0 )) s"
import_check; split_check
wait_snap() {  # wait_snap <tag>: wait for the background download (bounded by its own timeout 900)
  until [ -s "$OUT/snap$1.rc" ]; do [ "$(left)" -gt 60 ] || { log "download $1 not done in time"; exit 6; }; sleep 3; done
  [ "$(cat "$OUT/snap$1.rc")" = 0 ] || { log "snapshot $1 failed: $(tail -2 "$OUT/snap$1.err")"; exit 6; }
  tail -1 "$OUT/snap$1.path"
}
HO=""; [ "$SPLIT" = heldout ] && HO="--allow-heldout"
step() {  # step <name> <model> <snapshot> <source> <n or empty>
  local name="$1" model="$2" snapdir="$3" source="$4" n="$5" rc=0
  local guard=$(( $(left) - 30 ))
  [ "$guard" -gt 300 ] || { log "skipping $name: only $guard s left"; return 8; }
  set +e; timeout "$guard" $PY -u selfdoubt/stage0_run.py --engine vllm --mode full --split-name "$SPLIT" $HO \
    --model "$model" --snapshot "$snapdir" --source "$source" ${n:+--limit "$n"} --max-tokens "$MAXTOK" \
    --usd-per-hour "$USD_H" --chunk "$CHUNK" --full-frac "$FULLFRAC" --deadline-s "$(( guard - 60 ))" $EXTRA \
    2>&1 | tee -a "$OUT/$name.log" | grep -vE "^(INFO|DEBUG|WARNING|\(EngineCore)"; rc=${PIPESTATUS[0]}; set -e
  log "stage $name rc $rc at +$(( $(date +%s) - T0 )) s"; return $rc
}
S17=$(wait_snap 17)
log "snapshot 1.7B $S17 at +$(( $(date +%s) - T0 )) s"
RC=0
set +e; step s1_1p7b_gsm8k "$M17" "$S17" gsm8k "$N1"; rc=$?; set -e; [ $rc -eq 0 ] || RC=$rc
set +e; step s2_1p7b_math500 "$M17" "$S17" math500 "$N2"; rc=$?; set -e; [ $rc -eq 0 ] || RC=$rc
S8=$(wait_snap 8)
log "snapshot 8B $S8 at +$(( $(date +%s) - T0 )) s"
set +e; step s3_8b_gsm8k "$M8" "$S8" gsm8k "$N3"; rc=$?; set -e; [ $rc -eq 0 ] || RC=$rc
log "pod done at +$(( $(date +%s) - T0 )) s (rc $RC): copy back $OUT"
exit $RC
