#!/usr/bin/env bash
# §46 D-30 pre-launch test: the stub chain end to end FROM THE EXTRACTED BUNDLE in a FRESH virtual environment built
# with the bundle's own install line (scripts/exp46_pod.sh `install_line`, run by INSTALL=1), so a package missing from
# that line fails here, not on the pod. CPU torch stands in for the image's CUDA torch; the stub model is a tiny random
# Qwen2 (36 layers) with the real Qwen2.5 tokenizer (local cache, offline); NO real model is loaded. Everything lives in
# one temp dir under TMPROOT (inside the repo), removed at the end (KEEP=1 keeps it).
#
#   nice -n 19 bash scripts/exp46_d30.sh          # TMPROOT defaults to runs/_tmp (git-ignored)
#
# Passes when: (1) the normal stub chain exits 0, pod.done says exit=0, status.json counts 24 captures (every 24th case) and 2 baselines
# and a readout rule, stage_times.json carries every stage (ok) with the default deadlines 14400 / 15000; (2) the
# probe (dev) and the report (dev) run from the tree on the stub outputs and exit 0, and the probe's --held and the
# report's --held refuse (exit 4) without design_committed.json; (3) PROJECT_LIMIT=0 stops the chain at the projection
# (exit 7) with no full run written; (4) a HARD_STOP_S that ends inside the capture cuts it (partial, rc 9 or 124),
# still writes pod.done and stage_times.json, and no Python traceback appears. Network is needed once, for pip.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=$(pwd)
TMPROOT="${TMPROOT:-$REPO/runs/_tmp}"
mkdir -p "$TMPROOT"
T=$(mktemp -d "$TMPROOT/exp46_d30.XXXXXX")
if [ -z "${KEEP:-}" ]; then trap 'rm -rf "$T"' EXIT; fi
export PIP_CACHE_DIR="$T/pipcache" PIP_DISABLE_PIP_VERSION_CHECK=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=""
PYBIN="${PYBIN:-python3}"
fail() { echo "D-30: $1"; echo "D-30 FAILED (exp46)"; exit 1; }
echo "D-30 (exp46): temp dir $T"
bash scripts/exp46_bundle.sh "$T/tree.tar.gz"
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
O="runs/exp46_pod_stub"
rc=0; ENGINE=stub INSTALL=1 PYTHON="$V" OUT="$O" bash scripts/exp46_pod.sh > "$T/chain.log" 2>&1 || rc=$?
echo "D-30 normal chain exit $rc; $(cat "$O/pod.done" 2>/dev/null || echo 'no pod.done')"
[ "$rc" -eq 0 ] || { tail -40 "$T/chain.log"; fail "normal chain exit $rc"; }
grep -q '^exit=0 ' "$O/pod.done" || fail "pod.done does not say exit=0"
! grep -q "Traceback" "$T/chain.log" || { grep -n -A5 Traceback "$T/chain.log" | head -30; fail "a traceback in the normal chain"; }
"$V" - "$O" <<'PY' || fail "status / stage_times check"
import json, sys
o = sys.argv[1]
s = json.load(open(f"{o}/status.json"))
assert s["capture"] == 24 and s["baselines"] == 2 and s["readout"] == "conclusion", s
t = json.load(open(f"{o}/stage_times.json"))
names = [x["stage"] for x in t["stages"]]
assert names == ["install", "population", "stub_check", "smoke", "projection", "capture", "baselines", "validate",
                 "s1", "controls", "s2"], names
assert t["hard_stop_s"] == 14400 and t["self_stop_s"] == 15000, t
assert all(x["status"] == "ok" for x in t["stages"]), t["stages"]
assert json.load(open(f"{o}/smoke/patch/smoke.json"))["pass"]
print("D-30 status", s, "stages", [(x["stage"], x["elapsed_s"]) for x in t["stages"]])
offset = next(x["t_offset_s"] for x in t["stages"] if x["stage"] == "capture") - t["stages"][0]["elapsed_s"]
open(f"{o}/_capture_offset", "w").write(str(offset))
PY
rc=0; "$V" scripts/exp46_probe.py --capture "$O/full/capture" > "$T/probe.log" 2>&1 || rc=$?
echo "D-30 probe dev (from the tree, stub capture) exit $rc"
[ "$rc" -eq 0 ] && test -e "$O/full/capture/probe_dev.json" || { tail -20 "$T/probe.log"; fail "probe dev"; }
rc=0; "$V" scripts/exp46_probe.py --capture "$O/full/capture" --held > "$T/probe_held.log" 2>&1 || rc=$?
[ "$rc" -eq 4 ] && test ! -e "$O/full/capture/probe_held.json" || { tail -5 "$T/probe_held.log"; fail "probe --held did not refuse (rc $rc)"; }
rc=0; "$V" scripts/exp46_report.py --probe "$O/full/capture" --patch "$O/full/patch" > "$T/report.log" 2>&1 || rc=$?
echo "D-30 report dev exit $rc"
[ "$rc" -eq 0 ] && test -e "$O/full/patch/report_dev.md" || { tail -20 "$T/report.log"; fail "report dev"; }
rc=0; "$V" scripts/exp46_report.py --probe "$O/full/capture" --patch "$O/full/patch" --held > /dev/null 2>&1 || rc=$?
[ "$rc" -eq 4 ] || fail "report --held did not refuse (rc $rc)"
rc=0; ENGINE=stub PROJECT_LIMIT=0 PYTHON="$V" OUT="${O}_proj" bash scripts/exp46_pod.sh > "$T/proj.log" 2>&1 || rc=$?
echo "D-30 projection limit 0 exit $rc (want 7)"
[ "$rc" -eq 7 ] || { tail -20 "$T/proj.log"; fail "the projection did not stop the chain"; }
[ ! -d "${O}_proj/full" ] || fail "a full run was written after the projection stop"
HS=$(( $(cat "$O/_capture_offset") + 40 ))
rc=0; ENGINE=stub HARD_STOP_S=$HS STUB_EVERY=1 PYTHON="$V" OUT="${O}_deadline" bash scripts/exp46_pod.sh > "$T/deadline.log" 2>&1 || rc=$?
echo "D-30 deadline chain (HARD_STOP_S $HS) exit $rc; $(cat "${O}_deadline/pod.done" 2>/dev/null || echo 'no pod.done')"
test -e "${O}_deadline/pod.done" || fail "no pod.done under the deadline"
! grep -q "Traceback" "$T/deadline.log" || { grep -n -A5 Traceback "$T/deadline.log" | head -30; fail "a traceback under the deadline"; }
"$V" -c "import json; t = json.load(open('${O}_deadline/stage_times.json')); st = {x['stage']: (x['status'], x['rc']) for x in t['stages']}; print('D-30 deadline stages', st); assert st['capture'][0] == 'partial' and st['capture'][1] in (9, 124), st" \
  || { tail -20 "$T/deadline.log"; fail "deadline stage record"; }
echo "D-30 PASS (exp46; bundle $(du -h "$T/tree.tar.gz" | cut -f1), sha256 $(sha256sum "$T/tree.tar.gz" | cut -c1-16), fresh venv $("$V" --version), repo $REPO untouched)"
