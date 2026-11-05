#!/usr/bin/env bash
# §43 Part B D-30 pre-launch test, one family: the stub chain end to end FROM THE EXTRACTED BUNDLE in a FRESH virtual
# environment built with the bundle's own install line (scripts/exp43_lens_pod.sh `install_line`, run by INSTALL=1), so
# a package missing from that line fails here, not on the pod. CPU torch stands in for the image's CUDA torch. The stub
# models are tiny random networks with the real tokenizers (read from the local Hugging Face cache, offline). Everything
# (bundle, tree, venv, pip cache) lives in one temp dir under TMPROOT, removed at the end (KEEP=1 keeps it).
#
#   FAMILY=llama nice -n 19 bash scripts/exp43_lens_d30.sh      # TMPROOT defaults to /home/qamil-mirza/.claude/jobs/c8c40ae9/tmp
#
# Passes when: (1) the normal stub chain exits 0, pod.done says exit=0, status.json counts 330 / 330 / 330 rows and an
# eval, stage_times.json carries every stage with the deadlines; (2) the laptop report runs from the tree (dev folds)
# and exits 0; (3) STUB_BREAK=bos (a doubled BOS, or one where the family has none) exits 5 at the smoke with no full
# read written; (4) HARD_STOP_S=45 skips or cuts stages and logs them, without a crash. Network is needed once, for pip.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=$(pwd)
if [ "${1:-}" = --family ]; then FAMILY="$2"; shift 2; fi
FAMILY="${FAMILY:-}"
case "$FAMILY" in llama|gemma|phi) ;; *) echo "D-30: FAMILY must be llama, gemma or phi" >&2; exit 2;; esac
export FAMILY
TMPROOT="${TMPROOT:-/home/qamil-mirza/.claude/jobs/c8c40ae9/tmp}"
mkdir -p "$TMPROOT"
T=$(mktemp -d "$TMPROOT/exp43_lens_${FAMILY}_d30.XXXXXX")
if [ -z "${KEEP:-}" ]; then trap 'rm -rf "$T"' EXIT; fi
export PIP_CACHE_DIR="$T/pipcache" PIP_DISABLE_PIP_VERSION_CHECK=1
PYBIN="${PYBIN:-python3}"
fail() { echo "D-30: $1"; echo "D-30 FAILED ($FAMILY)"; exit 1; }
echo "D-30 (exp43 lens, $FAMILY): temp dir $T"
bash scripts/exp43_lens_bundle.sh --family "$FAMILY" "$T/tree.tar.gz"
mkdir -p "$T/tree"
tar xzf "$T/tree.tar.gz" -C "$T/tree"
"$PYBIN" -m venv "$T/venv"
V="$T/venv/bin/python"
"$V" -m pip install -q --upgrade pip
"$V" -m pip install -q torch --index-url https://download.pytorch.org/whl/cpu
cd "$T/tree"
test ! -e .venv || fail "the bundle contains a .venv"
test ! -e .env || fail "the bundle contains a .env"
O="runs/exp43_lens_stub_${FAMILY}"
rc=0; ENGINE=stub INSTALL=1 PYTHON="$V" OUT="$O" bash scripts/exp43_lens_pod.sh > "$T/chain.log" 2>&1 || rc=$?
echo "D-30 normal chain exit $rc; $(cat "$O/pod.done" 2>/dev/null || echo 'no pod.done')"
[ "$rc" -eq 0 ] || { tail -30 "$T/chain.log"; fail "normal chain exit $rc"; }
grep -q '^exit=0 ' "$O/pod.done" || fail "pod.done does not say exit=0"
"$V" - "$O" <<'EOF' || fail "status / stage_times check"
import json, sys
o = sys.argv[1]
s = json.load(open(f"{o}/status.json"))
assert (s["logit_small"], s["logit_large"], s["jlens_small"], s["eval"]) == (330, 330, 330, True), s
t = json.load(open(f"{o}/stage_times.json"))
names = [x["stage"] for x in t["stages"]]
want = ["install", "population", "stub_models", "smoke_small", "smoke_large", "logit_small", "logit_large", "fit", "jread", "eval"]
assert names == want, names
assert t["hard_stop_s"] == 9900 and t["self_stop_s"] == 10500, t
assert all(x["status"] == "ok" for x in t["stages"]), t["stages"]
print("D-30 status", s, "stages", [(x["stage"], x["elapsed_s"]) for x in t["stages"]])
EOF
grep -E "^torch|jlens commit|population|smoke ok" "$O/pod.log" | head -6
rc=0; "$V" scripts/exp43_lens_report.py --family "$FAMILY" --root "$O" --samples 300 > "$T/report.log" 2>&1 || rc=$?
echo "D-30 report (dev folds, from the tree) exit $rc"
[ "$rc" -eq 0 ] && grep -q "Claim B1" "$O/report_dev.md" || { tail -20 "$T/report.log"; fail "report"; }
# (--confirm is never run here: it would open the §24 judge labels of folds 2-4)
rc=0; ENGINE=stub STUB_BREAK=bos PYTHON="$V" OUT="${O}_bos" bash scripts/exp43_lens_pod.sh > "$T/bos.log" 2>&1 || rc=$?
echo "D-30 broken-BOS smoke exit $rc (want 5)"
[ "$rc" -eq 5 ] || { tail -20 "$T/bos.log"; fail "broken BOS did not stop the chain at the smoke"; }
[ ! -d "${O}_bos/logit" ] || fail "a full read was written after a failed smoke"
rc=0; ENGINE=stub HARD_STOP_S=45 PYTHON="$V" OUT="${O}_deadline" bash scripts/exp43_lens_pod.sh > "$T/deadline.log" 2>&1 || rc=$?
echo "D-30 HARD_STOP_S=45 chain exit $rc; $(grep -cE 'SKIP|cut by the deadline|deadline: stopped' "${O}_deadline/pod.log" "${O}_deadline"/*.log 2>/dev/null | awk -F: '{s+=$2} END {print s}') skip/cut lines"
grep -qE 'SKIP|cut by the deadline' "${O}_deadline/pod.log" || { tail -20 "$T/deadline.log"; fail "no stage was skipped or cut under a 45 s stop"; }
"$V" -c "import json; t = json.load(open('${O}_deadline/stage_times.json')); print('D-30 deadline stages', [(x['stage'], x['status']) for x in t['stages']])"
echo "D-30 PASS ($FAMILY; bundle $(du -h "$T/tree.tar.gz" | cut -f1), sha256 $(sha256sum "$T/tree.tar.gz" | cut -c1-16), fresh venv $("$V" --version), repo $REPO untouched)"
