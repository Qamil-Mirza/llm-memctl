#!/usr/bin/env bash
# §40 D-30 pre-launch test: the stub chain end to end FROM THE EXTRACTED BUNDLE in a FRESH virtual environment built
# with the bundle's own install line (scripts/exp40_pod.sh `install_line`, run by INSTALL=1), so a package missing
# from that line fails here, not on the pod. CPU torch stands in for the image's CUDA torch (the line constrains pip to
# whatever torch is installed, as on the pod). Everything (bundle, tree, venv, pip cache) lives in one temp dir.
#
#   nice -n 19 bash scripts/exp40_d30.sh                 # TMPROOT defaults to /home/qamil-mirza/.claude/jobs/c8c40ae9/tmp
#   KEEP=1 bash scripts/exp40_d30.sh                     # keep the temp dir for inspection
#
# Passes when: the normal stub chain exits 0 with a report; STUB_JUDGE=silent exits 5 (smoke); STUB_JUDGE=disagree
# exits 7 (agreement). Network is needed once, for pip.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=$(pwd)
TMPROOT="${TMPROOT:-/home/qamil-mirza/.claude/jobs/c8c40ae9/tmp}"
mkdir -p "$TMPROOT"
T=$(mktemp -d "$TMPROOT/exp40_d30.XXXXXX")
if [ -z "${KEEP:-}" ]; then trap 'rm -rf "$T"' EXIT; fi
export PIP_CACHE_DIR="$T/pipcache" PIP_DISABLE_PIP_VERSION_CHECK=1
PYBIN="${PYBIN:-python3}"
echo "D-30: temp dir $T"
bash scripts/exp40_bundle.sh "$T/exp40_tree.tar.gz"
mkdir -p "$T/tree"
tar xzf "$T/exp40_tree.tar.gz" -C "$T/tree"
"$PYBIN" -m venv "$T/venv"
V="$T/venv/bin/python"
"$V" -m pip install -q --upgrade pip
"$V" -m pip install -q torch --index-url https://download.pytorch.org/whl/cpu
cd "$T/tree"
test ! -e .venv || { echo "D-30: the bundle contains a .venv" >&2; exit 3; }
fail=0
ENGINE=stub INSTALL=1 PYTHON="$V" OUT=runs/exp40_stub bash scripts/exp40_pod.sh > "$T/chain.log" 2>&1 || fail=$?
echo "D-30 normal chain exit $fail; $(cat runs/exp40_stub/pod.done)"
if [ "$fail" -ne 0 ] || [ ! -s runs/exp40_stub/exp40_report_stub.md ]; then tail -30 "$T/chain.log"; echo "D-30 FAILED"; exit 1; fi
rc=0; ENGINE=stub STUB_JUDGE=silent PYTHON="$V" OUT=runs/exp40_stub_silent bash scripts/exp40_pod.sh > "$T/silent.log" 2>&1 || rc=$?
echo "D-30 silent judge exit $rc (want 5)"
[ "$rc" -eq 5 ] || { tail -20 "$T/silent.log"; echo "D-30 FAILED"; exit 1; }
rc=0; ENGINE=stub STUB_JUDGE=disagree PYTHON="$V" OUT=runs/exp40_stub_disagree bash scripts/exp40_pod.sh > "$T/disagree.log" 2>&1 || rc=$?
echo "D-30 disagreeing judge exit $rc (want 7)"
[ "$rc" -eq 7 ] || { tail -20 "$T/disagree.log"; echo "D-30 FAILED"; exit 1; }
grep -E "^torch|install|stage" runs/exp40_stub/pod.log | head -12
echo "D-30 PASS (bundle $(du -h "$T/exp40_tree.tar.gz" | cut -f1), fresh venv $("$V" --version), repo $REPO untouched)"
