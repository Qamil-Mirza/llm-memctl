#!/usr/bin/env bash
# Wait for runs/selfdoubt/stub/pod.done on the pod (or the copy-back deadline), copy back runs/selfdoubt/stub into the
# worktree and the main checkout, DELETE the pod, kill the hard-stop timer, confirm 0 pods. Adapted from
# runs/exp55/pod_wait.sh. Run in the background after selfdoubt/stub_launch.sh.
set -uo pipefail
WT=/home/qamil-mirza/Code/llm-memctl/.worktrees/self-doubt; MAIN=/home/qamil-mirza/Code/llm-memctl; cd "$WT"
L=runs/selfdoubt/stub_launch; COPY_AT="${COPY_AT:-1920}"   # s after create: copy whatever exists by then
KEY=$(grep -E '^(export )?RUNPOD_API_KEY=' $MAIN/.env | head -1 | sed -E 's/^(export )?RUNPOD_API_KEY=//; s/^"//; s/"$//; s/^'"'"'//; s/'"'"'$//')
API=https://rest.runpod.io/v1
POD=$(python3 -c "import json; print(json.load(open('$L/pod.json'))['pod'])")
CREATED=$(python3 -c "import json; print(json.load(open('$L/pod.json'))['created'])")
read IP PORT < "$L/ssh.txt"
S="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15 -o BatchMode=yes -p $PORT root@$IP"
until $S "test -e /root/sd/runs/selfdoubt/stub/pod.done" 2>/dev/null; do
  [ $(( $(date +%s) - CREATED )) -lt "$COPY_AT" ] || { echo "copy-back deadline, copying what exists" >> "$L/wait.log"; break; }
  sleep 20
done
$S "cat /root/sd/runs/selfdoubt/stub/pod.done" >> "$L/wait.log" 2>&1
$S "cd /root/sd && tar -czf /root/stub_out.tgz runs/selfdoubt/stub" && \
  scp -q -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -P $PORT root@$IP:/root/stub_out.tgz "$L/stub_out.tgz" && \
  tar -xzf "$L/stub_out.tgz" -C "$WT" && echo "copied back $(date -u +%T)" >> "$L/wait.log"
curl -s -X DELETE -H "Authorization: Bearer $KEY" "$API/pods/$POD" >> "$L/wait.log" 2>&1
echo "deleted $POD at $(date -u +%T) (lifetime $(( $(date +%s) - CREATED )) s)" >> "$L/wait.log"
kill "$(cat $L/hardstop.pid)" 2>/dev/null && echo "hard-stop timer killed" >> "$L/wait.log" && rm -f "$L/auth.hdr"
mkdir -p "$MAIN/runs/selfdoubt/stub" && cp -a runs/selfdoubt/stub/. "$MAIN/runs/selfdoubt/stub/" && echo "copied to main checkout" >> "$L/wait.log"
n=$(curl -s -H "Authorization: Bearer $KEY" "$API/pods" | python3 -c "import sys,json; print(len(json.load(sys.stdin)))")
echo "pods after: $n" >> "$L/wait.log"; echo "WAIT DONE pods_after=$n"
