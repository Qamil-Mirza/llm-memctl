#!/usr/bin/env bash
# §55 pod launcher (laptop side). User approval in the Executor pane: "approved: §55 on one A40, about $1.00, hard stop
# $1.50 / 150 min". One A40 secure, image-based (no template, no volume). Local DELETE timer at 150 min after create.
# Steps: list-pods must be 0 -> create -> hard-stop timer -> wait ssh -> upload bundle -> start chain (RUN_S) -> exit.
# The waiter (pod_wait.sh) copies back and deletes. RUNPOD_API_KEY from .env, never printed.
set -euo pipefail
cd /home/qamil-mirza/Code/llm-memctl
L=runs/exp55/pod_launch_out; mkdir -p "$L"
KEY=$(grep -E '^(export )?RUNPOD_API_KEY=' .env | head -1 | sed -E 's/^(export )?RUNPOD_API_KEY=//; s/^"//; s/"$//; s/^'"'"'//; s/'"'"'$//')
API=https://rest.runpod.io/v1
n=$(curl -s -H "Authorization: Bearer $KEY" "$API/pods" | python3 -c "import sys,json; print(len(json.load(sys.stdin)))")
echo "pods before: $n" | tee "$L/launch.log"; [ "$n" = 0 ] || { echo "ABORT: pods exist"; exit 2; }
PUB=$(cat ~/.ssh/id_ed25519.pub)
BODY=$(python3 - "$PUB" <<'PY'
import json, sys
print(json.dumps({"name": "exp55", "imageName": "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404",
  "gpuTypeIds": ["NVIDIA A40"], "gpuCount": 1, "cloudType": "SECURE", "containerDiskInGb": 80,
  "ports": ["22/tcp"], "env": {"PUBLIC_KEY": sys.argv[1]}}))
PY
)
curl -s -X POST -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" -d "$BODY" "$API/pods" > "$L/create.json"
POD=$(python3 -c "import json; d=json.load(open('$L/create.json')); print(d.get('id',''))")
[ -n "$POD" ] || { echo "CREATE FAILED: $(python3 -c "import json;d=json.load(open('$L/create.json'));print({k:d.get(k) for k in ('error','message','status')})")"; exit 3; }
PRICE=$(python3 -c "import json; d=json.load(open('$L/create.json')); print(d.get('costPerHr') or d.get('adjustedCostPerHr'))")
echo "{\"pod\": \"$POD\", \"price\": \"$PRICE\", \"created\": $(date +%s), \"delete_after_s\": 9000}" > "$L/pod.json"
echo "created $POD at $(date -u +%T) price $PRICE/h" | tee -a "$L/launch.log"
nohup setsid bash -c "sleep 9000; echo HARDSTOP \$(date -u +%T) >> '$L/hardstop.log'; \
  curl -s -X DELETE -H 'Authorization: Bearer $KEY' '$API/pods/$POD' >> '$L/hardstop.log' 2>&1" > /dev/null 2>&1 < /dev/null &
echo $! > "$L/hardstop.pid"; echo "hard-stop timer pid $(cat $L/hardstop.pid) (150 min)" | tee -a "$L/launch.log"
for i in $(seq 1 90); do
  J=$(curl -s -H "Authorization: Bearer $KEY" "$API/pods/$POD")
  SSH=$(echo "$J" | python3 -c "import sys,json; d=json.load(sys.stdin); m=d.get('portMappings') or {}; ip=d.get('publicIp'); print(f\"{ip} {m.get('22')}\" if ip and m.get('22') else '')")
  [ -n "$SSH" ] && break; sleep 10
done
[ -n "$SSH" ] || { echo "NO SSH after 15 min; pod left for the hard stop / manual delete" | tee -a "$L/launch.log"; exit 4; }
echo "$SSH" > "$L/ssh.txt"; set -- $SSH; IP=$1; PORT=$2
S="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15 -p $PORT root@$IP"
for i in $(seq 1 30); do $S true 2>/dev/null && break; sleep 10; done
$S "mkdir -p /root/w" && scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -P $PORT runs/exp55/bundle.tgz root@$IP:/root/w/ \
  && $S "cd /root/w && tar -xzf bundle.tgz && rm -rf .git && git init -q . && \
     RUNPOD_POD_ID=$POD RUN_S=7500 nohup setsid bash scripts/exp55_pod.sh > /root/w/chain.out 2>&1 < /dev/null & echo started" \
  | tee -a "$L/launch.log"
echo "launched at $(date -u +%T)" | tee -a "$L/launch.log"
