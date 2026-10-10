#!/bin/bash
# One shared Qwen 7B pod (vLLM 0.8.5, --gpu-memory-utilization 0.70 --max-num-batched-tokens 2048): the §29 reader rerun
# (five arms + judge, cached heads), then the §32 reader (six arms + judge). Usage: exp29_32_pod.sh POD_ID CREATE_EPOCH
#   Ready gate: the server must answer by create + 15 min, then the real-shape pre-check must pass, or exit 3.
#   §29: DEADLINE create + 45 min (whole process group killed); triage at create + 12 min: if its reader has not
#        started, evidence_rows is dropped from the five fold configs (restored from git afterwards).
#   §32: DEADLINE create + 85 min via its own stage (whole process groups). The pod's hard stop is create + 90 min.
# Any cwd: the driver cds into the §29 worktree (W29) for that stage, its triage and the yaml restore, then into the
# §32 worktree (W32); each stage resolves its relative paths there. STUB=1 is passed through to both stages.
# URL, MIN, S29 and S32 may be overridden for a dry run of the timing logic. Lines starting "EV " are progress events.
set -u
ROOT=/home/qamil-mirza/Code/llm-memctl
W29=$ROOT/.claude/worktrees/agent-a0e62e719012a7c34
W32=$ROOT/.claude/worktrees/agent-a6a97dd6399cd3da7
POD=${1:?pod id}; C=${2:?create epoch}
URL=${URL:-https://$POD-8000.proxy.runpod.net}
M=${MIN:-60}  # seconds per "minute" (a dry run shrinks it)
READY=$((C+15*M)); T29=$((C+12*M)); D29=$((C+45*M)); D32=$((C+85*M))
S29=${S29:-bash configs/sweeps/exp29/stage.sh reader}
S32=${S32:-bash configs/sweeps/exp32/stage.sh}
LOGS=$ROOT/runs/_logs; mkdir -p $LOGS
T=${STUB:+stub_}  # the stages tag their markers stub_ under STUB=1
now() { date +%s; }
until curl -sf -m 10 -A memctl $URL/v1/models | grep -q '"qwen2.5-7b-instruct"'; do
  [ $(now) -ge $READY ] && { echo "EV READY_FAIL $(date -u +%T)"; exit 3; }; sleep 5; done
# Real-shape pre-check: 32 concurrent reader-shaped calls with a long prompt (about 6k tokens, above the readers'
# longest) must all return 200 within 180 s. Neither stage makes logprob calls; the 0.70/2048 flags cover those.
python3 - $URL <<'PY' || { echo "EV PRECHECK_FAIL $(date -u +%T)"; exit 3; }
import json, sys, urllib.request, concurrent.futures as cf
url = sys.argv[1] + "/v1/chat/completions"
def call(i):
    text = f"Session {i}. " + "Yesterday I met Caroline at the support group and we talked about painting. " * 400
    body = json.dumps({"model": "qwen2.5-7b-instruct", "max_tokens": 64, "temperature": 0,
                       "messages": [{"role": "user", "content": text + "\nWhat did they talk about?"}]}).encode()
    req = urllib.request.Request(url, body, {"Content-Type": "application/json", "User-Agent": "memctl"})
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            return r.status
    except Exception as e:
        return repr(e)[:80]
with cf.ThreadPoolExecutor(32) as ex:
    codes = list(ex.map(call, range(32)))
bad = [c for c in codes if c != 200]
print("precheck:", len(codes) - len(bad), "of", len(codes), "ok", bad[:3], flush=True)
sys.exit(1 if bad else 0)
PY
echo "EV READY $(date -u +%T)"
# §29
cd $W29
rm -f runs/_logs/${T}exp29_reader_START runs/_logs/${T}exp29_reader_ANSWERS_DONE runs/_logs/${T}exp29_reader_DONE
( until [ $(now) -ge $T29 ] || [ -f runs/_logs/${T}exp29_reader_START ]; do sleep 1; done
  if [ ! -f runs/_logs/${T}exp29_reader_START ]; then
    for k in 0 1 2 3 4; do python3 - configs/sweeps/exp29/exp29_qwen7b_f$k.yaml <<'PY'
import re, sys
p = sys.argv[1]; t = open(p).read()
t2 = re.sub(r"  - name: rl\n    label: evidence_rows\n(?:    .*\n)+?(?=  [a-z]|\Z)", "", t)
assert t2 != t and "evidence_rows" not in t2
open(p, "w").write(t2)
PY
    done; echo "EV TRIAGE29 evidence_rows dropped $(date -u +%T)"
  fi ) &
TRI=$!
WAIT_MAX=$(( READY - $(now) + 60 )) setsid $S29 $URL > $LOGS/exp29_32_s29.out 2>&1 &
P29=$!
( sleep $(( D29 - $(now) )); echo "EV DEADLINE29 $(date -u +%T)"; kill -KILL -- -$P29 2>/dev/null ) &
WD=$!
wait $P29; s=$?
kill $WD 2>/dev/null; wait $TRI 2>/dev/null; kill $TRI 2>/dev/null
git checkout -q -- configs/sweeps/exp29/exp29_qwen7b_f0.yaml configs/sweeps/exp29/exp29_qwen7b_f1.yaml configs/sweeps/exp29/exp29_qwen7b_f2.yaml configs/sweeps/exp29/exp29_qwen7b_f3.yaml configs/sweeps/exp29/exp29_qwen7b_f4.yaml
[ -f runs/_logs/${T}exp29_reader_DONE ] && echo "EV S29_DONE $(date -u +%T)" || echo "EV S29_INCOMPLETE status=$s $(date -u +%T)"
# §32
cd $W32
rm -f runs/_logs/${T}exp32_READY runs/_logs/${T}exp32_ANSWERS_DONE runs/_logs/${T}exp32_DONE
DEADLINE=$D32 $S32 $URL > $LOGS/exp29_32_s32.out 2>&1; s=$?
[ -f runs/_logs/${T}exp32_DONE ] && echo "EV S32_DONE $(date -u +%T)" || echo "EV S32_INCOMPLETE status=$s $(date -u +%T)"
echo "EV ALL_DONE $(date -u +%T)"
