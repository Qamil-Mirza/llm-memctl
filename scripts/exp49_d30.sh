#!/usr/bin/env bash
# §49 D-30 pre-launch test: the chain end to end FROM THE EXTRACTED BUNDLE in a FRESH virtual environment built with the
# bundle's own install line (scripts/exp49_pod.sh `install_line`, run by INSTALL=1), so a package missing from that
# line fails here, not on the pod. CPU torch stands in for the image's CUDA torch. The stub check loads a tiny random
# Qwen2 (36 layers) with the real tokenizer (local cache, offline) through the real HF engine; the smoke and the main
# chain use the canned FakeEngine (no model). This script plays the laptop at the design mark: it commits design.json
# in a git repo made in the extracted tree and writes the marker, as the real launch does with a shallow clone.
# Everything lives in one temp dir under TMPROOT (inside the repo), removed at the end (KEEP=1 keeps it).
#
#   nice -n 19 bash scripts/exp49_d30.sh
#
# Passes when: (1) the normal stub chain exits 0, pod.done says exit=0, every stage is ok, status.json counts 1,584
# baselines and a report, the smoke passes, and the held stages ran only after a git-verified mark; (2) PROJECT_LIMIT=0
# stops at the projection (exit 7) with no run written; (3) a HARD_STOP_S ending inside dev baselines cuts it (rc 9 or
# 124), still writes pod.done and stage_times.json, and no traceback appears; (4) a marker naming a wrong commit makes
# wait-mark refuse (exit 11). Network is needed once, for pip.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=$(pwd)
TMPROOT="${TMPROOT:-$REPO/runs/_tmp}"
mkdir -p "$TMPROOT"
T=$(mktemp -d "$TMPROOT/exp49_d30.XXXXXX")
if [ -z "${KEEP:-}" ]; then trap 'rm -rf "$T"' EXIT; fi
export PIP_CACHE_DIR="$T/pipcache" PIP_DISABLE_PIP_VERSION_CHECK=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=""
PYBIN="${PYBIN:-python3}"
fail() { echo "D-30: $1"; echo "D-30 FAILED (exp49)"; exit 1; }
echo "D-30 (exp49): temp dir $T"
bash scripts/exp49_bundle.sh "$T/tree.tar.gz"
mkdir -p "$T/tree"
tar xzf "$T/tree.tar.gz" -C "$T/tree"
"$PYBIN" -m venv "$T/venv"
V="$T/venv/bin/python"
"$V" -m pip install -q --upgrade pip
"$V" -m pip install -q torch --index-url https://download.pytorch.org/whl/cpu
cd "$T/tree"
test ! -e .venv || fail "the bundle contains a .venv"
test ! -e .env || fail "the bundle contains a .env"
test ! -e tests || fail "the bundle contains tests"
test ! -e .git || fail "the bundle contains .git"
G="git -c user.name=d30 -c user.email=d30@localhost"
mark() {   # mark <run dir> [commit override]: commit <run>/design.json in a repo made in the tree, write the marker
  local run="$1"
  [ -d .git ] || git init -q .
  $G add -f "$run/design.json" && $G commit -q -m "d30 design" && local c; c=$(git rev-parse HEAD)
  "$V" -c "import sys; sys.path.insert(0, 'scripts'); sys.path.insert(0, '.'); import exp49_run as R; from pathlib import Path; R.mark_design(Path('$run'), '${2:-}' or '$c')"
}
wait_ready() {   # wait_ready <run dir> <pid>
  for i in $(seq 1 900); do
    [ -e "$1/design.ready" ] && return 0
    kill -0 "$2" 2>/dev/null || return 1
    sleep 1
  done
  return 1
}
O="runs/exp49_pod_stub"
ENGINE=stub INSTALL=1 PYTHON="$V" OUT="$O" MARK_WAIT_S=600 bash scripts/exp49_pod.sh > "$T/chain.log" 2>&1 &
P=$!
wait_ready "$O/run" "$P" || { tail -40 "$T/chain.log"; fail "no design.ready"; }
test ! -e "$O/run/baselines/$(head -c 0 /dev/null)RC01_ANS_F2_H1.json" || fail "a held baseline exists before the mark"
mark "$O/run"
rc=0; wait "$P" || rc=$?
echo "D-30 normal chain exit $rc; $(cat "$O/pod.done" 2>/dev/null || echo 'no pod.done')"
[ "$rc" -eq 0 ] || { tail -40 "$T/chain.log"; fail "normal chain exit $rc"; }
grep -q '^exit=0 ' "$O/pod.done" || fail "pod.done does not say exit=0"
! grep -q "Traceback" "$T/chain.log" || { grep -n -A5 Traceback "$T/chain.log" | head -30; fail "a traceback in the normal chain"; }
"$V" - "$O" <<'PY' || fail "status / stage_times check"
import json, sys
o = sys.argv[1]
s = json.load(open(f"{o}/status.json"))
assert s["baselines"] == 1584 and s["report"] is True and s["design"] is True, s
t = json.load(open(f"{o}/stage_times.json"))
names = [x["stage"] for x in t["stages"]]
want = ["install", "population", "stub_check", "smoke", "projection", "baselines_dev", "probe_dev", "arms_dev", "design",
        "wait_mark", "baselines_held", "arms_held", "report"]
assert names == want, names
assert t["hard_stop_s"] == 23400 and t["self_stop_s"] == 25200, t
assert all(x["status"] == "ok" for x in t["stages"]), t["stages"]
assert json.load(open(f"{o}/smoke/run/smoke.json"))["pass"]
print("D-30 status", s, "stages", [(x["stage"], x["elapsed_s"]) for x in t["stages"]])
open(f"{o}/_dev_offset", "w").write(str(next(x["t_offset_s"] for x in t["stages"] if x["stage"] == "baselines_dev")
                                       - t["stages"][0]["elapsed_s"]))
PY
rc=0; ENGINE=stub PROJECT_LIMIT=0 PYTHON="$V" OUT="${O}_proj" bash scripts/exp49_pod.sh > "$T/proj.log" 2>&1 || rc=$?
echo "D-30 projection limit 0 exit $rc (want 7)"
[ "$rc" -eq 7 ] || { tail -20 "$T/proj.log"; fail "the projection did not stop the chain"; }
[ ! -d "${O}_proj/run/baselines" ] || fail "a run was written after the projection stop"
HS=$(( $(cat "$O/_dev_offset") + 33 ))
rc=0; ENGINE=stub FAKE_DELAY_S=0.2 HARD_STOP_S=$HS PYTHON="$V" OUT="${O}_deadline" bash scripts/exp49_pod.sh > "$T/deadline.log" 2>&1 || rc=$?
echo "D-30 deadline chain (HARD_STOP_S $HS) exit $rc; $(cat "${O}_deadline/pod.done" 2>/dev/null || echo 'no pod.done')"
test -e "${O}_deadline/pod.done" || fail "no pod.done under the deadline"
! grep -q "Traceback" "$T/deadline.log" || { grep -n -A5 Traceback "$T/deadline.log" | head -30; fail "a traceback under the deadline"; }
"$V" -c "import json; t = json.load(open('${O}_deadline/stage_times.json')); st = {x['stage']: (x['status'], x['rc']) for x in t['stages']}; print('D-30 deadline stages', st); assert st['baselines_dev'][0] == 'partial' and st['baselines_dev'][1] in (9, 124), st; assert 'probe_dev' not in st" \
  || { tail -20 "$T/deadline.log"; fail "deadline stage record"; }
rc=0; ENGINE=stub PYTHON="$V" OUT="${O}_badmark" MARK_WAIT_S=600 bash scripts/exp49_pod.sh > "$T/badmark.log" 2>&1 &
P=$!
wait_ready "${O}_badmark/run" "$P" || { tail -20 "$T/badmark.log"; fail "no design.ready (bad-mark run)"; }
mark "${O}_badmark/run" 0000000000000000000000000000000000000000
wait "$P" || rc=$?
echo "D-30 wrong-commit marker exit $rc (want 11)"
[ "$rc" -eq 11 ] || { tail -20 "$T/badmark.log"; fail "a wrong commit was not refused"; }
[ ! -d "${O}_badmark/run/arms_held" ] || fail "held arms ran under a refused mark"
echo "D-30 PASS (exp49; bundle $(du -h "$T/tree.tar.gz" | cut -f1), sha256 $(sha256sum "$T/tree.tar.gz" | cut -c1-16), fresh venv $("$V" --version), repo $REPO untouched)"
