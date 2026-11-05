#!/usr/bin/env bash
# Watch a pod chain until its sentinel file appears, without ever searching for a process name.
#   bash scripts/pod_watch.sh HOST PORT DONE_FILE [LOG_FILE] [INTERVAL_S=300] [MAX_POLLS=72]
# DONE_FILE is the path the pod script writes on exit (scripts/exp34_pod.sh: /workspace/llm-memctl/runs/exp34_pod.done;
# scripts/exp33_pod.sh: .../runs/exp33_pod.done). LOG_FILE, if given, has its "stage ..." lines echoed on each poll.
# Exit 0: sentinel found (its content, e.g. "exit=0 at <time>", is printed). Exit 3: SSH failed on 3 polls in a row
# (pod gone or unreachable). Exit 4: MAX_POLLS reached. POD_SSH overrides the ssh command (tests use "bash -c").
set -uo pipefail
HOST="${1:?host}"; PORT="${2:?port}"; DONE="${3:?done file}"; LOG="${4:-}"; INTERVAL="${5:-300}"; MAX="${6:-72}"
SSH="${POD_SSH:-ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15 -p $PORT root@$HOST}"
fails=0
for i in $(seq 1 "$MAX"); do
  if out=$($SSH "if test -e '$DONE'; then echo DONE_SENTINEL; cat '$DONE'; fi; if test -n '$LOG' && test -e '$LOG'; then grep '^stage ' '$LOG' | tail -3; fi" 2>&1); then
    fails=0
    echo "=== poll $i $(date -u +%T)"; echo "$out" | grep -v DONE_SENTINEL
    if echo "$out" | grep -q DONE_SENTINEL; then echo "CHAIN_DONE"; exit 0; fi
  else
    fails=$((fails + 1)); echo "=== poll $i $(date -u +%T): ssh failed ($fails/3)"
    if [ "$fails" -ge 3 ]; then echo "POD_UNREACHABLE"; exit 3; fi
  fi
  sleep "$INTERVAL"
done
echo "WATCH_TIMEOUT after $MAX polls"; exit 4
