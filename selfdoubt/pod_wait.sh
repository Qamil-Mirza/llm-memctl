#!/usr/bin/env bash
# Wait for $OUTREL/pod.done on the pod (OUTREL from $L/plan.json) or the copy-back deadline, copy back $OUTREL into the
# worktree and the main checkout, DELETE the pod, kill the hard-stop timer, confirm 0 pods. Adapted from
# runs/exp55/pod_wait.sh. Run in the background after selfdoubt/pod_launch.sh, with the same L (and COPY_AT for long runs).
set -uo pipefail
WT="${WT:-/home/qamil-mirza/Code/llm-memctl/.worktrees/self-doubt}"; MAIN="${MAIN:-/home/qamil-mirza/Code/llm-memctl}"; cd "$WT"
PLAN="${PLAN:-}"; if [ -n "$PLAN" ]; then set -a; . "$PLAN"; set +a; fi   # L and COPY_AT from the launcher's plan
L="${L:-runs/selfdoubt/stub_launch}"; COPY_AT="${COPY_AT:-1920}"   # s after create: copy whatever exists by then
OUTREL=$(python3 -c "import json; print(json.load(open('$L/plan.json'))['outrel'])" 2>/dev/null || echo runs/selfdoubt/stub)
# same account as the launcher: KEY_VAR from pod.json (older pod.json without it = RUNPOD_API_KEY), same header file
KEY_VAR=$(python3 -c "import json; print(json.load(open('$L/pod.json')).get('key_var', 'RUNPOD_API_KEY'))")
. selfdoubt/pod_key.sh
HDR="$WT/$L/auth.hdr"; [ -s "$HDR" ] || rp_write_hdr "$HDR"
POD=$(python3 -c "import json; print(json.load(open('$L/pod.json'))['pod'])")
CREATED=$(python3 -c "import json; print(json.load(open('$L/pod.json'))['created'])")
read IP PORT < "$L/ssh.txt"
S="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15 -o BatchMode=yes -p $PORT root@$IP"
until $S "test -e /root/sd/$OUTREL/pod.done" 2>/dev/null; do
  [ $(( $(date +%s) - CREATED )) -lt "$COPY_AT" ] || { echo "copy-back deadline, copying what exists" >> "$L/wait.log"; break; }
  sleep 20
done
$S "cat /root/sd/$OUTREL/pod.done" >> "$L/wait.log" 2>&1
$S "cd /root/sd && tar -czf /root/stub_out.tgz $OUTREL" && \
  scp -q -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -P $PORT root@$IP:/root/stub_out.tgz "$L/stub_out.tgz" && \
  tar -xzf "$L/stub_out.tgz" -C "$WT" && echo "copied back $(date -u +%T)" >> "$L/wait.log"
curl -s -X DELETE -H @"$HDR" "$API/pods/$POD" >> "$L/wait.log" 2>&1
echo "deleted $POD at $(date -u +%T) (lifetime $(( $(date +%s) - CREATED )) s)" >> "$L/wait.log"
T=$(cat $L/hardstop.pid); { kill -- -"$T" 2>/dev/null || kill "$T" 2>/dev/null; } && echo "hard-stop timer killed" >> "$L/wait.log"
mkdir -p "$MAIN/$OUTREL" && cp -a "$OUTREL/." "$MAIN/$OUTREL/" && echo "copied to main checkout" >> "$L/wait.log"
n=$(curl -s -H @"$HDR" "$API/pods" | python3 -c "import sys,json; print(len(json.load(sys.stdin)))")
rm -f "$HDR"
echo "pods after: $n ($KEY_VAR)" >> "$L/wait.log"; echo "WAIT DONE pods_after=$n"
