#!/usr/bin/env bash
# §40 D-30 pre-launch test: the stub chain end to end FROM THE EXTRACTED BUNDLE in a FRESH virtual environment built
# with the bundle's own install line (scripts/exp40_pod.sh `install_line`, run by INSTALL=1), so a package missing
# from that line fails here, not on the pod. CPU torch stands in for the image's CUDA torch (the line constrains pip to
# whatever torch is installed, as on the pod). Everything (bundle, tree, venv, pip cache) lives in one temp dir.
#
#   nice -n 19 bash scripts/exp40_d30.sh                 # TMPROOT defaults to /home/qamil-mirza/.claude/jobs/c8c40ae9/tmp
#   KEEP=1 bash scripts/exp40_d30.sh                     # keep the temp dir for inspection
#   EXP=exp42 nice -n 19 bash scripts/exp40_d30.sh       # §42: the exp42 bundle and the exp42 stub chain
#
# Passes when: the normal stub chain exits 0 with a report; STUB_JUDGE=silent exits 5 (smoke); STUB_JUDGE=disagree
# exits 7 (agreement). Network is needed once, for pip.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=$(pwd)
EXP="${EXP:-exp40}"
export EXP
TMPROOT="${TMPROOT:-/home/qamil-mirza/.claude/jobs/c8c40ae9/tmp}"
mkdir -p "$TMPROOT"
T=$(mktemp -d "$TMPROOT/${EXP}_d30.XXXXXX")
if [ -z "${KEEP:-}" ]; then trap 'rm -rf "$T"' EXIT; fi
export PIP_CACHE_DIR="$T/pipcache" PIP_DISABLE_PIP_VERSION_CHECK=1
PYBIN="${PYBIN:-python3}"
echo "D-30 ($EXP): temp dir $T"
bash scripts/exp40_bundle.sh "$T/${EXP}_tree.tar.gz"
mkdir -p "$T/tree"
tar xzf "$T/${EXP}_tree.tar.gz" -C "$T/tree"
"$PYBIN" -m venv "$T/venv"
V="$T/venv/bin/python"
"$V" -m pip install -q --upgrade pip
"$V" -m pip install -q torch --index-url https://download.pytorch.org/whl/cpu
cd "$T/tree"
test ! -e .venv || { echo "D-30: the bundle contains a .venv" >&2; exit 3; }
fail=0
ENGINE=stub INSTALL=1 PYTHON="$V" OUT="runs/${EXP}_stub" bash scripts/exp40_pod.sh > "$T/chain.log" 2>&1 || fail=$?
echo "D-30 normal chain exit $fail; $(cat "runs/${EXP}_stub/pod.done")"
if [ "$fail" -ne 0 ] || [ ! -s "runs/${EXP}_stub/${EXP}_report_stub.md" ]; then tail -30 "$T/chain.log"; echo "D-30 FAILED"; exit 1; fi
if [ "$EXP" = exp42 ]; then
  grep -q "§42: selective steering" "runs/${EXP}_stub/${EXP}_report_stub.md" || { echo "D-30: no §42 block in the report"; echo "D-30 FAILED"; exit 1; }
  test -s "runs/${EXP}_stub/qwen3b/cond42.jsonl" || { echo "D-30: no cond42.jsonl"; echo "D-30 FAILED"; exit 1; }
  grep -E "§42 fresh set|cond42 fires" "runs/${EXP}_stub/run.log" | head -3
fi
rc=0; ENGINE=stub STUB_JUDGE=silent PYTHON="$V" OUT="runs/${EXP}_stub_silent" bash scripts/exp40_pod.sh > "$T/silent.log" 2>&1 || rc=$?
echo "D-30 silent judge exit $rc (want 5)"
[ "$rc" -eq 5 ] || { tail -20 "$T/silent.log"; echo "D-30 FAILED"; exit 1; }
rc=0; ENGINE=stub STUB_JUDGE=disagree PYTHON="$V" OUT="runs/${EXP}_stub_disagree" bash scripts/exp40_pod.sh > "$T/disagree.log" 2>&1 || rc=$?
echo "D-30 disagreeing judge exit $rc (want 7)"
[ "$rc" -eq 7 ] || { tail -20 "$T/disagree.log"; echo "D-30 FAILED"; exit 1; }
grep -E "^torch|install|stage" "runs/${EXP}_stub/pod.log" | head -12
echo "D-30 PASS ($EXP; bundle $(du -h "$T/${EXP}_tree.tar.gz" | cut -f1), sha256 $(sha256sum "$T/${EXP}_tree.tar.gz" | cut -c1-16), fresh venv $("$V" --version), repo $REPO untouched)"
