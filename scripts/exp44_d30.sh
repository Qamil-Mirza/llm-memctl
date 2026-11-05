#!/usr/bin/env bash
# §44 rev 8 D-30 pre-launch test: the stub chain end to end FROM THE EXTRACTED BUNDLE in a FRESH virtual environment
# built with the bundle's own install line (scripts/exp44_pod.sh `install_line`, run by INSTALL=1), so a package
# missing from that line fails here, not on the pod. CPU torch stands in for the image's CUDA torch. The stub models are
# tiny random Qwen2 networks with the real Qwen2.5 tokenizer (read from the local Hugging Face cache, offline); NO real
# model is loaded. Everything (bundle, tree, venv, pip cache) lives in one temp dir under TMPROOT, removed at the end
# (KEEP=1 keeps it).
#
#   nice -n 19 bash scripts/exp44_d30.sh      # TMPROOT defaults to /home/qamil-mirza/.claude/jobs/c8c40ae9/tmp
#
# Passes when: (1) the normal stub chain exits 0, pod.done says exit=0, status.json counts 204 patch rows and 70 swap
# rows, stage_times.json carries every stage with the default deadlines; (2) the rev 8 report and the device check run
# from the tree and exit 0; (3) STUB_BREAK=layers exits 5 at the stub check with no full run written; (4) with
# B_RESERVE_S larger than HARD_STOP_S, run A is skipped (logged, stage partial rc 124), B still runs, pod.done is
# written, no crash. Network is needed once, for pip.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=$(pwd)
TMPROOT="${TMPROOT:-/home/qamil-mirza/.claude/jobs/c8c40ae9/tmp}"
mkdir -p "$TMPROOT"
T=$(mktemp -d "$TMPROOT/exp44_d30.XXXXXX")
if [ -z "${KEEP:-}" ]; then trap 'rm -rf "$T"' EXIT; fi
export PIP_CACHE_DIR="$T/pipcache" PIP_DISABLE_PIP_VERSION_CHECK=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=""
PYBIN="${PYBIN:-python3}"
fail() { echo "D-30: $1"; echo "D-30 FAILED (exp44 rev 8)"; exit 1; }
echo "D-30 (exp44 rev 8): temp dir $T"
bash scripts/exp44_bundle.sh "$T/tree.tar.gz"
mkdir -p "$T/tree"
tar xzf "$T/tree.tar.gz" -C "$T/tree"
"$PYBIN" -m venv "$T/venv"
V="$T/venv/bin/python"
"$V" -m pip install -q --upgrade pip
"$V" -m pip install -q torch --index-url https://download.pytorch.org/whl/cpu
cd "$T/tree"
test ! -e .venv || fail "the bundle contains a .venv"
test ! -e .env || fail "the bundle contains a .env"
test ! -e runs/exp44_patch || fail "the bundle contains exp44_patch rows"
O="runs/exp44_pod_stub"
rc=0; ENGINE=stub INSTALL=1 PYTHON="$V" OUT="$O" bash scripts/exp44_pod.sh > "$T/chain.log" 2>&1 || rc=$?
echo "D-30 normal chain exit $rc; $(cat "$O/pod.done" 2>/dev/null || echo 'no pod.done')"
[ "$rc" -eq 0 ] || { tail -30 "$T/chain.log"; fail "normal chain exit $rc"; }
grep -q '^exit=0 ' "$O/pod.done" || fail "pod.done does not say exit=0"
"$V" - "$O" <<'PY' || fail "status / stage_times check"
import json, sys
o = sys.argv[1]
s = json.load(open(f"{o}/status.json"))
assert (s["patch_rows"], s["swap_S_rows"], s["patch_failed"]) == (204, 70, {}), s
t = json.load(open(f"{o}/stage_times.json"))
names = [x["stage"] for x in t["stages"]]
assert names == ["install", "population", "stub_check", "smoke", "full_A", "full_B"], names
assert t["hard_stop_s"] == 4500 and t["self_stop_s"] == 5400, t
assert all(x["status"] == "ok" for x in t["stages"]), t["stages"]
for k in ("stubcheck/A", "stubcheck/B", "smoke/qwen7b", "smoke/qwen3b_rev8"):
    assert json.load(open(f"{o}/{k}/_smoke.json"))["pass"], k
print("D-30 status", s, "stages", [(x["stage"], x["elapsed_s"]) for x in t["stages"]])
PY
grep -E "^torch|^population|^smoke|stub resume" "$O/pod.log" | cut -c1-220 | head -8
rc=0; "$V" scripts/exp44_swap_report.py --model qwen3b --gate-rev 8 --pooled --root "$O/full" --samples 300 > "$T/report.log" 2>&1 || rc=$?
echo "D-30 rev 8 report (from the tree, stub rows) exit $rc"
[ "$rc" -eq 0 ] && grep -q "S0 on the changed-substitute rows" "$O/full/qwen3b_rev8/report_pooled.md" || { tail -20 "$T/report.log"; fail "report"; }
rc=0; "$V" scripts/exp44_device_check.py --laptop "$O/smoke/qwen7b" --gpu "$O/full/qwen7b_gpu" --out "$O/device_check" > "$T/device.log" 2>&1 || rc=$?
echo "D-30 device check (stub smoke vs stub full) exit $rc: $(grep -m1 'Laptop rows' "$O/device_check.md" | cut -c1-160)"
[ "$rc" -eq 0 ] || { tail -20 "$T/device.log"; fail "device check"; }
rc=0; ENGINE=stub STUB_BREAK=layers PYTHON="$V" OUT="${O}_break" bash scripts/exp44_pod.sh > "$T/break.log" 2>&1 || rc=$?
echo "D-30 broken stub check exit $rc (want 5)"
[ "$rc" -eq 5 ] || { tail -20 "$T/break.log"; fail "a failed check did not stop the chain"; }
[ ! -d "${O}_break/full" ] || fail "a full run was written after a failed check"
rc=0; ENGINE=stub B_RESERVE_S=100000 PYTHON="$V" OUT="${O}_deadline" bash scripts/exp44_pod.sh > "$T/deadline.log" 2>&1 || rc=$?
echo "D-30 deadline chain (A starved) exit $rc; $(cat "${O}_deadline/pod.done" 2>/dev/null || echo 'no pod.done')"
grep -q "SKIP full_A" "${O}_deadline/pod.log" || { tail -20 "$T/deadline.log"; fail "run A was not skipped under the deadline"; }
"$V" -c "import json; t = json.load(open('${O}_deadline/stage_times.json')); st = {x['stage']: (x['status'], x['rc']) for x in t['stages']}; print('D-30 deadline stages', st); assert st['full_A'] == ('partial', 124) and st['full_B'][0] == 'ok', st" \
  || fail "deadline stage record"
echo "D-30 PASS (exp44 rev 8; bundle $(du -h "$T/tree.tar.gz" | cut -f1), sha256 $(sha256sum "$T/tree.tar.gz" | cut -c1-16), fresh venv $("$V" --version), repo $REPO untouched)"
