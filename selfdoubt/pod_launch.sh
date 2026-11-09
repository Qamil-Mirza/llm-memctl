#!/usr/bin/env bash
# selfdoubt pod launcher (laptop side), adapted from runs/exp55/pod_launch.sh. Stub (default) or Stage 0 full:
#   POD_SCRIPT=selfdoubt/stage0_full_pod.sh OUTREL=runs/selfdoubt/full L=runs/selfdoubt/full_launch HARD_S=8100 \
#   NAME=selfdoubt-stage0 bash selfdoubt/pod_launch.sh Needs the user's own approval line
# in the Executor pane. One A40 SECURE, image-based (no template, no volume), disk 80, ports 22/tcp, PUBLIC_KEY.
# Steps: 0 pods -> create -> local DELETE timer at HARD_S after create -> wait ssh (no ssh by NOSSH_S: DELETE now) ->
# upload bundle (selfdoubt/ + split.json) -> start stage0_pod.sh with RUNPOD_POD_ID, RUN_S and SELF_STOP_S fitted to
# the time left -> exit. selfdoubt/pod_wait.sh copies back and deletes.
# Account: KEY_VAR (default RUNPOD_API_KEY; RUNPOD_API_KEY_2 = gmail) from .env, never printed (selfdoubt/pod_key.sh).
# Pre-flight before create: EXPECT_EMAIL must match, balance >= HARD_S x PRICE_H, 0 pods on that account.
set -euo pipefail
WT="${WT:-/home/qamil-mirza/Code/llm-memctl/.worktrees/self-doubt}"; cd "$WT"
HARD_S="${HARD_S:-2100}"; NOSSH_S="${NOSSH_S:-480}"; COPY_S="${COPY_S:-240}"
POD_SCRIPT="${POD_SCRIPT:-selfdoubt/stage0_pod.sh}"; OUTREL="${OUTREL:-runs/selfdoubt/stub}"; NAME="${NAME:-selfdoubt-stub}"
L="${L:-runs/selfdoubt/stub_launch}"; mkdir -p "$L"
echo "{\"pod_script\": \"$POD_SCRIPT\", \"outrel\": \"$OUTREL\"}" > "$L/plan.json"
[ ! -e "$OUTREL/pod.done" ] && [ ! -e "$OUTREL/measure.json" ] || { echo "ABORT: clear $OUTREL first"; exit 2; }
. selfdoubt/pod_key.sh
PRICE_H="${PRICE_H:-0.59}"; CAP_USD=$(python3 -c "print(round($HARD_S * $PRICE_H / 3600, 3))")
# one header file (600) for every call of this run: launcher, DELETE timer, waiter
HDR="$WT/$L/auth.hdr"; rp_write_hdr "$HDR" || exit 2
if ! rp_preflight "$HDR" "$CAP_USD" | tee "$L/launch.log"; then rm -f "$HDR"; echo "ABORT: pre-flight failed"; exit 2; fi
tar -czf "$L/bundle.tgz" --exclude=__pycache__ --exclude=selfdoubt/tests selfdoubt runs/selfdoubt/split.json runs/selfdoubt/split.sha256
PUB=$(cat ~/.ssh/id_ed25519.pub)
BODY=$(python3 - "$PUB" "$NAME" <<'PY'
import json, sys
print(json.dumps({"name": sys.argv[2], "imageName": "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404",
  "gpuTypeIds": ["NVIDIA A40"], "gpuCount": 1, "cloudType": "SECURE", "containerDiskInGb": 80,
  "ports": ["22/tcp"], "env": {"PUBLIC_KEY": sys.argv[1]}}))
PY
)
curl -s -X POST -H @"$HDR" -H "Content-Type: application/json" -d "$BODY" "$API/pods" > "$L/create.json"
CREATED=$(date +%s)
POD=$(python3 -c "import json; d=json.load(open('$L/create.json')); print(d.get('id','') if isinstance(d, dict) else '')")
[ -n "$POD" ] || { echo "CREATE FAILED: $(head -c 400 "$L/create.json")"; rm -f "$HDR"; exit 3; }
PRICE=$(python3 -c "import json; d=json.load(open('$L/create.json')); print(d.get('costPerHr') or d.get('adjustedCostPerHr'))")
echo "{\"pod\": \"$POD\", \"price\": \"$PRICE\", \"created\": $CREATED, \"delete_after_s\": $HARD_S, \"key_var\": \"$KEY_VAR\", \"email\": \"$EXPECT_EMAIL\"}" > "$L/pod.json"
echo "created $POD at $(date -u +%T) price $PRICE/h" | tee -a "$L/launch.log"
# the timer reads the key from the same 600-mode header file, so it never appears on a command line (ps)
nohup setsid bash -c "sleep $HARD_S; echo HARDSTOP \$(date -u +%T) >> '$WT/$L/hardstop.log'; \
  curl -s -X DELETE -H @'$HDR' '$API/pods/$POD' >> '$WT/$L/hardstop.log' 2>&1" > /dev/null 2>&1 < /dev/null &
echo $! > "$L/hardstop.pid"; echo "hard-stop timer pid $(cat $L/hardstop.pid) ($HARD_S s after create)" | tee -a "$L/launch.log"
del_now() { echo "$1: deleting $POD now" | tee -a "$L/launch.log"; curl -s -X DELETE -H @"$HDR" "$API/pods/$POD" >> "$L/launch.log" 2>&1
  T=$(cat $L/hardstop.pid); kill -- -"$T" 2>/dev/null || kill "$T" 2>/dev/null || true; exit 4; }
SSH=""
while [ $(( $(date +%s) - CREATED )) -lt "$NOSSH_S" ]; do
  J=$(curl -s -H @"$HDR" "$API/pods/$POD")
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
  && $S "cd /root/sd && tar -xzf bundle.tgz && mkdir -p $OUTREL && \
     RUNPOD_POD_ID=$POD RUN_S=$RUN_S SELF_STOP_S=$SELF_STOP_S nohup setsid bash $POD_SCRIPT > $OUTREL/launch.out 2>&1 < /dev/null & echo started" \
  | tee -a "$L/launch.log"
echo "launched at $(date -u +%T): RUN_S $RUN_S, self-stop +$SELF_STOP_S s, hard stop at create+$HARD_S s" | tee -a "$L/launch.log"
