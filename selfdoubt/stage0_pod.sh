#!/usr/bin/env bash
# selfdoubt Stage 0 STUB pod (selfdoubt/PROPOSAL.md "Budget" item 1). NOT APPROVED: the stub needs the user's own
# approval (hard cap $0.25) before any pod exists. Written and dry-checked only.
# Order: preflight (files, split sha) -> GPU check -> install (pinned vLLM line) -> snapshot both models at pinned
# revisions -> stub Qwen3-8B (time guard) -> stub Qwen3-1.7B (time guard) -> CPU report -> pod.done.
# Each model stage stops between chunks at its guard (exit 9, resumable: rerun skips done keys). Copy back
# runs/selfdoubt/stub (gen.jsonl, phases.jsonl, run_meta.json, measure.json, report.json, logs).
#   RUN_S=<seconds of run budget> bash selfdoubt/stage0_pod.sh
#   ENGINE=stub PYTHON=<venv>/bin/python [INSTALL=1] bash selfdoubt/stage0_pod.sh   # files + imports only, no GPU
# Knobs: MAXTOK (16384), USD_H (0.60, for the projection), PCT8 (65 = % of the post-setup time for 8B), CHUNK (64
# requests per generate call; a guard kill loses at most the chunk in flight).
set -euo pipefail
export PIP_BREAK_SYSTEM_PACKAGES=1 HF_HUB_DISABLE_PROGRESS_BARS=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 TOKENIZERS_PARALLELISM=false
cd "$(dirname "$0")/.."
ENGINE="${ENGINE:-vllm}"; INSTALL="${INSTALL:-}"; RUN_S="${RUN_S:-1500}"; MAXTOK="${MAXTOK:-16384}"
USD_H="${USD_H:-0.60}"; PCT8="${PCT8:-65}"; CHUNK="${CHUNK:-64}"
OUT=runs/selfdoubt/stub; PY="${PYTHON:-python}"
mkdir -p "$OUT"; rm -f "$OUT/pod.done"
T0=$(date +%s); DEADLINE=$(( T0 + RUN_S ))
trap 'rc=$?; echo "exit=$rc engine=$ENGINE at $(date -u +%FT%TZ) total_s=$(( $(date +%s) - T0 ))" > "$OUT/pod.done"' EXIT
log() { echo "$(date -u +%T) $*" | tee -a "$OUT/pod.log"; }
left() { echo $(( DEADLINE - $(date +%s) )); }
log "selfdoubt stage0 stub pod; engine $ENGINE; run budget $RUN_S s; max_tokens $MAXTOK"

# Pins (2026-11-08). vLLM 0.11.0 is the newest release whose PyPI metadata pins torch==2.8.0 (0.10.2 also does;
# 0.11.1+ pin torch 2.9.0; the authors' 0.10.1 pins torch 2.7.1), so the image's torch 2.8.0+cu128 is kept.
# transformers 4.57.1: vLLM 0.11.0 has no upper bound and 5.x is untested with it.
VLLM_PIN="vllm==0.11.0"; TF_PIN="transformers==4.57.1"
M8=Qwen/Qwen3-8B;   R8=b968826d9c46dd6066d109eabc6255188de91218
M17=Qwen/Qwen3-1.7B; R17=70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
SPLIT_SHA=c9e99ab227e45d10298d91d4f0a17e150cfefa3a24d872555a2330c7d9392703
UPSTREAM=b87d61bd2e5e998f34878fb0ee5fb2f5550f98c6

for f in selfdoubt/__init__.py selfdoubt/common.py selfdoubt/engines.py selfdoubt/scoring.py selfdoubt/stage0_run.py \
         selfdoubt/stage0_report.py selfdoubt/vendor/interrupt_lrm/__init__.py \
         selfdoubt/vendor/interrupt_lrm/prompt_utils_qwen.py selfdoubt/vendor/interrupt_lrm/math_utils.py \
         selfdoubt/vendor/interrupt_lrm/math_parsing_util.py selfdoubt/vendor/interrupt_lrm/LICENSE \
         selfdoubt/vendor/interrupt_lrm/prompts/intervene/wo_prompt_guidance.json \
         selfdoubt/vendor/interrupt_lrm/prompts/intervene/w_prompt_guidance.json; do
  test -e "$f" || { log "missing $f"; exit 3; }; done
sha() { sha256sum "$1" | cut -d' ' -f1; }
[ "$(sha selfdoubt/vendor/interrupt_lrm/prompts/intervene/wo_prompt_guidance.json)" = \
  c8c2eff691410da2e3cd8216186107465d921bf6d7c7e06185b9eb232fe7ca42 ] || { log "wo wrapper sha mismatch"; exit 3; }
[ "$(sha selfdoubt/vendor/interrupt_lrm/prompts/intervene/w_prompt_guidance.json)" = \
  5108777179a4385cef92bbd1227c10f5529d2ac20d726f9d1f61831864ce15f8 ] || { log "w wrapper sha mismatch"; exit 3; }
grep -q "$UPSTREAM" selfdoubt/vendor/interrupt_lrm/__init__.py || { log "vendored commit mismatch"; exit 3; }

install_line() {
  TORCH_VERSION=$($PY -c "import torch; print(torch.__version__.split('+')[0])"); echo "torch==${TORCH_VERSION}" > "$OUT/constraints.txt"
  [ "$TORCH_VERSION" = 2.8.0 ] || { log "image torch $TORCH_VERSION != 2.8.0 (vLLM 0.11.0 pins 2.8.0)"; exit 4; }
  timeout 1200 $PY -m pip install -q -c "$OUT/constraints.txt" "$VLLM_PIN" "$TF_PIN" "huggingface_hub<1.0" \
    latex2sympy2==1.9.1 word2number==1.1 pyarrow numpy 2>&1 | tail -3
  $PY -c "import vllm, transformers, torch; assert vllm.__version__ == '0.11.0', vllm.__version__; \
assert transformers.__version__ == '4.57.1'; assert torch.__version__.startswith('2.8.0'); print(vllm.__version__, transformers.__version__, torch.__version__)" \
    | tee -a "$OUT/pod.log" || { log "install check failed"; exit 4; }
  $PY -m pip freeze 2>/dev/null | grep -iE "^(vllm|torch|transformers|xformers|flashinfer|tokenizers|latex2sympy2|antlr4|sympy)" > "$OUT/env.txt" || true
}
import_check() {
  $PY -c "import sys; sys.path.insert(0, '.'); from selfdoubt import common, engines, scoring, stage0_run, stage0_report; \
import importlib.util as u; print('selfdoubt imports ok; vllm', 'present' if u.find_spec('vllm') else 'absent')" \
    | tee -a "$OUT/pod.log" || { log "import check failed"; exit 4; }
}
split_check() {
  $PY -c "import sys; sys.path.insert(0, '.'); from selfdoubt import common as C; s, sha = C.ensure_split(C.load_items()); \
assert sha == '$SPLIT_SHA', sha; print('split sha ok', sha)" | tee -a "$OUT/pod.log" || { log "split sha mismatch"; exit 3; }
}

if [ "$ENGINE" = stub ]; then
  export CUDA_VISIBLE_DEVICES=""; [ -n "$INSTALL" ] && install_line; import_check; split_check
  log "stub: preflight only (files, wrapper sha, imports, split sha); no GPU work"; exit 0
fi

until nvidia-smi >/dev/null 2>&1; do [ $(( $(date +%s) - T0 )) -lt 120 ] || { log "no GPU"; exit 4; }; sleep 5; done
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | tee -a "$OUT/pod.log"
install_line; import_check; split_check
snap() { $PY -c "import huggingface_hub as h; print(h.snapshot_download('$1', revision='$2', \
allow_patterns=['*.json','*.safetensors','*.txt','tokenizer*','merges.txt','vocab.json']))"; }
S8=$(timeout 900 bash -c "$(declare -f snap); PY=$PY; snap $M8 $R8") || { log "snapshot 8B failed"; exit 6; }
S17=$(timeout 600 bash -c "$(declare -f snap); PY=$PY; snap $M17 $R17") || { log "snapshot 1.7B failed"; exit 6; }
log "snapshots $S8 $S17 at +$(( $(date +%s) - T0 )) s"

step() {  # step <name> <guard seconds> <args...>
  local name="$1" guard="$2"; shift 2; local rc=0
  set +e; timeout "$guard" $PY -u selfdoubt/stage0_run.py --engine vllm --mode stub --max-tokens "$MAXTOK" \
    --usd-per-hour "$USD_H" --chunk "$CHUNK" --deadline-s "$(( guard - 90 ))" "$@" 2>&1 | tee -a "$OUT/$name.log"; rc=${PIPESTATUS[0]}; set -e
  log "stage $name rc $rc at +$(( $(date +%s) - T0 )) s"; return $rc
}
REST=$(( $(left) - 120 ))                       # keep 2 min for the 1.7B load margin and the report
[ "$REST" -gt 300 ] || { log "not enough time left after setup ($REST s)"; exit 7; }
G8=$(( REST * PCT8 / 100 ))
RC=0
set +e; step stub_8b "$G8" --model "$M8" --snapshot "$S8"; rc=$?; set -e; [ $rc -eq 0 ] || RC=$rc
G17=$(( $(left) - 120 ))
if [ "$G17" -gt 180 ]; then
  set +e; step stub_1p7b "$G17" --model "$M17" --snapshot "$S17"; rc=$?; set -e; [ $rc -eq 0 ] || RC=$rc
else
  log "skipping 1.7B: only $G17 s left"; [ $RC -ne 0 ] || RC=8
fi
set +e; timeout 300 $PY -u selfdoubt/stage0_report.py --mode stub 2>&1 | tail -60 | tee -a "$OUT/report.log"; set -e
log "pod done at +$(( $(date +%s) - T0 )) s (rc $RC): copy back $OUT"
exit $RC
