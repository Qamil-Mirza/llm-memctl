#!/usr/bin/env bash
# Wait for runs/exp55/pod/pod.done on the pod, copy back runs/exp55/pod, DELETE the pod, kill the hard-stop timer,
# confirm 0 pods. Run in the background after pod_launch.sh.
set -uo pipefail
cd /home/qamil-mirza/Code/llm-memctl
L=runs/exp55/pod_launch_out
KEY=$(grep -E '^(export )?RUNPOD_API_KEY=' .env | head -1 | sed -E 's/^(export )?RUNPOD_API_KEY=//; s/^"//; s/"$//; s/^'"'"'//; s/'"'"'$//')
API=https://rest.runpod.io/v1
POD=$(python3 -c "import json; print(json.load(open('$L/pod.json'))['pod'])")
read IP PORT < "$L/ssh.txt"
S="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15 -p $PORT root@$IP"
CREATED=$(python3 -c "import json; print(json.load(open('$L/pod.json'))['created'])")
until $S "test -e /root/w/runs/exp55/pod/pod.done" 2>/dev/null; do
  [ $(( $(date +%s) - CREATED )) -lt 8820 ] || { echo "near hard stop, copying what exists" >> "$L/wait.log"; break; }
  sleep 30
done
$S "cat /root/w/runs/exp55/pod/pod.done" >> "$L/wait.log" 2>&1
mkdir -p runs/exp55/pod
$S "cd /root/w && tar -czf /root/pod_out.tgz runs/exp55/pod chain.out" && \
  scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -P $PORT root@$IP:/root/pod_out.tgz runs/exp55/pod_out.tgz && \
  tar -xzf runs/exp55/pod_out.tgz -C . && echo "copied back $(date -u +%T)" >> "$L/wait.log"
curl -s -X DELETE -H "Authorization: Bearer $KEY" "$API/pods/$POD" >> "$L/wait.log" 2>&1
echo "deleted $POD at $(date -u +%T) (lifetime $(( $(date +%s) - CREATED )) s)" >> "$L/wait.log"
kill "$(cat $L/hardstop.pid)" 2>/dev/null && echo "hard-stop timer killed" >> "$L/wait.log"
n=$(curl -s -H "Authorization: Bearer $KEY" "$API/pods" | python3 -c "import sys,json; print(len(json.load(sys.stdin)))")
echo "pods after: $n" >> "$L/wait.log"; echo "WAIT DONE pods_after=$n"
