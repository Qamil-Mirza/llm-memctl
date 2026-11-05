#!/usr/bin/env bash
# §51 D-30: the measurement chain from the EXTRACTED BUNDLE in a FRESH venv built with the chain's own install line
# (INSTALL=1), CPU torch standing in for the image's CUDA torch, the dummy model (no download; the tokenizer comes from
# the local HF cache, offline). Passes when the chain exits 0, pod.done says exit=0 and the smoke wrote a complete
# measure.json, and the bundle holds no .env / .venv / tests / .git / arrays / writing/.
#   nice -n 19 bash scripts/exp51_d30.sh
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=$(pwd)
TMPROOT="${TMPROOT:-$REPO/runs/_tmp}"; mkdir -p "$TMPROOT"
T=$(mktemp -d "$TMPROOT/exp51_d30.XXXXXX")
[ -n "${KEEP:-}" ] || trap 'rm -rf "$T"' EXIT
export PIP_CACHE_DIR="$T/pipcache" PIP_DISABLE_PIP_VERSION_CHECK=1
fail() { echo "D-30: $1"; echo "D-30 FAILED (exp51)"; exit 1; }
bash scripts/exp51_bundle.sh "$T/tree.tar.gz"
mkdir -p "$T/tree"; tar xzf "$T/tree.tar.gz" -C "$T/tree"
"${PYBIN:-python3}" -m venv "$T/venv"; V="$T/venv/bin/python"
"$V" -m pip install -q --upgrade pip
"$V" -m pip install -q torch --index-url https://download.pytorch.org/whl/cpu
cd "$T/tree"
for x in .venv .env tests .git writing; do test ! -e "$x" || fail "the bundle contains $x"; done
! find . -name '*.npy' -o -name '*.npz' -o -name '*.safetensors' | grep -q . || fail "the bundle contains arrays"
set +e
ENGINE=stub INSTALL=1 PYTHON="$V" OUT=runs/exp51/gpu_measure/pod_stub bash scripts/exp51_pod_measure.sh > "$T/chain.log" 2>&1
RC=$?
set -e
tail -5 "$T/chain.log"
[ "$RC" -eq 0 ] || fail "chain exit $RC"
grep -q '^exit=0' runs/exp51/gpu_measure/pod_stub/pod.done || fail "pod.done not exit=0"
grep -q '"done": true' runs/exp51/gpu_measure/pod_stub/smoke/measure.json || fail "smoke incomplete"
grep -q Traceback "$T/chain.log" && fail "traceback in the chain log"
echo "D-30 PASSED (exp51)"
