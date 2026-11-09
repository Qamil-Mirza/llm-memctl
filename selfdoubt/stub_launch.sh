#!/usr/bin/env bash
# selfdoubt stub pod launcher (laptop side), adapted from runs/exp55/pod_launch.sh. Needs the user's own approval line
# in the Executor pane. One A40 SECURE, image-based (no template, no volume), disk 80, ports 22/tcp, PUBLIC_KEY.
# Steps: 0 pods -> create -> local DELETE timer at HARD_S after create -> wait ssh (no ssh by NOSSH_S: DELETE now) ->
# upload bundle (selfdoubt/ + split.json) -> start stage0_pod.sh with RUNPOD_POD_ID, RUN_S and SELF_STOP_S fitted to
# the time left -> exit. selfdoubt/stub_wait.sh copies back and deletes. RUNPOD_API_KEY from .env, never printed.
set -euo pipefail
WT=/home/qamil-mirza/Code/llm-memctl/.worktrees/self-doubt; cd "$WT"
HARD_S="${HARD_S:-2100}"; NOSSH_S="${NOSSH_S:-480}"; COPY_S="${COPY_S:-240}"
L=runs/selfdoubt/stub_launch; mkdir -p "$L"
KEY=$(grep -E '^(export )?RUNPOD_API_KEY=' /home/qamil-mirza/Code/llm-memctl/.env | head -1 | sed -E 's/^(export )?RUNPOD_API_KEY=//; s/^"//; s/"$//; s/^'"'"'//; s/'"'"'$//')
API=https://rest.runpod.io/v1
[ ! -e runs/selfdoubt/stub/pod.done ] && [ ! -e runs/selfdoubt/stub/measure.json ] || { echo "ABORT: clear runs/selfdoubt/stub first"; exit 2; }
n=$(curl -s -H "Authorization: Bearer $KEY" "$API/pods" | python3 -c "import sys,json; print(len(json.load(sys.stdin)))")
echo "pods before: $n" | tee "$L/launch.log"; [ "$n" = 0 ] || { echo "ABORT: pods exist"; exit 2; }
tar -czf "$L/bundle.tgz" --exclude=__pycache__ --exclude=selfdoubt/tests selfdoubt runs/selfdoubt/split.json runs/selfdoubt/split.sha256
PUB=$(cat ~/.ssh/id_ed25519.pub)
BODY=$(python3 - "$PUB" <<'PY'
import json, sys
print(json.dumps({"name": "selfdoubt-stub", "imageName": "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404",
  "gpuTypeIds": ["NVIDIA A40"], "gpuCount": 1, "cloudType": "SECURE", "containerDiskInGb": 80,
  "ports": ["22/tcp"], "env": {"PUBLIC_KEY": sys.argv[1]}}))
PY
)
curl -s -X POST -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" -d "$BODY" "$API/pods" > "$L/create.json"
CREATED=$(date +%s)
POD=$(python3 -c "import json; d=json.load(open('$L/create.json')); print(d.get('id','') if isinstance(d, dict) else '')")
[ -n "$POD" ] || { echo "CREATE FAILED: $(head -c 400 "$L/create.json")"; exit 3; }
PRICE=$(python3 -c "import json; d=json.load(open('$L/create.json')); print(d.get('costPerHr') or d.get('adjustedCostPerHr'))")
echo "{\"pod\": \"$POD\", \"price\": \"$PRICE\", \"created\": $CREATED, \"delete_after_s\": $HARD_S}" > "$L/pod.json"
echo "created $POD at $(date -u +%T) price $PRICE/h" | tee -a "$L/launch.log"
# the timer reads the key from a 600-mode header file, so it never appears on a process command line (ps)
( umask 077; printf 'Authorization: Bearer %s\n' "$KEY" > "$WT/$L/auth.hdr" )
nohup setsid bash -c "sleep $HARD_S; echo HARDSTOP \$(date -u +%T) >> '$WT/$L/hardstop.log'; \
  curl -s -X DELETE -H @'$WT/$L/auth.hdr' '$API/pods/$POD' >> '$WT/$L/hardstop.log' 2>&1" > /dev/null 2>&1 < /dev/null &
echo $! > "$L/hardstop.pid"; echo "hard-stop timer pid $(cat $L/hardstop.pid) ($HARD_S s after create)" | tee -a "$L/launch.log"
del_now() { echo "$1: deleting $POD now" | tee -a "$L/launch.log"; curl -s -X DELETE -H "Authorization: Bearer $KEY" "$API/pods/$POD" >> "$L/launch.log" 2>&1
  kill "$(cat $L/hardstop.pid)" 2>/dev/null || true; exit 4; }
SSH=""
while [ $(( $(date +%s) - CREATED )) -lt "$NOSSH_S" ]; do
  J=$(curl -s -H "Authorization: Bearer $KEY" "$API/pods/$POD")
  SSH=$(echo "$J" | python3 -c "import sys,json; d=json.load(sys.stdin); m=d.get('portMappings') or {}; ip=d.get('publicIp'); print(f\"{ip} {m.get('22')}\" if ip and m.get('22') else '')" 2>/dev/null || true)
  [ -n "$SSH" ] && break; sleep 10
done
[ -n "$SSH" ] || del_now "NO SSH MAPPING after $NOSSH_S s"
echo "$SSH" > "$L/ssh.txt"; set -- $SSH; IP=$1; PORT=$2
S="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15 -o BatchMode=yes -p $PORT root@$IP"
until $S true 2>/dev/null; do [ $(( $(date +%s) - CREATED )) -lt "$NOSSH_S" ] || del_now "SSH NOT UP after $NOSSH_S s"; sleep 10; done
echo "ssh up at +$(( $(date +%s) - CREATED )) s" | tee -a "$L/launch.log"
RUN_S=$(( HARD_S - ($(date +%s) - CREATED) - COPY_S )); SELF_STOP_S=$(( RUN_S + 150 ))
$S "mkdir -p /root/sd" && scp -q -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -P $PORT "$L/bundle.tgz" root@$IP:/root/sd/ \
  && $S "cd /root/sd && tar -xzf bundle.tgz && mkdir -p runs/selfdoubt/stub && \
     RUNPOD_POD_ID=$POD RUN_S=$RUN_S SELF_STOP_S=$SELF_STOP_S nohup setsid bash selfdoubt/stage0_pod.sh > runs/selfdoubt/stub/launch.out 2>&1 < /dev/null & echo started" \
  | tee -a "$L/launch.log"
echo "launched at $(date -u +%T): RUN_S $RUN_S, self-stop +$SELF_STOP_S s, hard stop at create+$HARD_S s" | tee -a "$L/launch.log"
