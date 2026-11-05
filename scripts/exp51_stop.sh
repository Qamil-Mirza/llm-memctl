#!/usr/bin/env bash
# §51: stop a running harness by the process group it recorded (never by $! or pgrep).
#   scripts/exp51_stop.sh <run dir>
set -euo pipefail
f="$1/harness_pid.json"
read -r pid pgid own < <(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['pid'], d['pgid'], d['own_group'])" "$f")
[ "$own" = "True" ] || { echo "refused: harness pid $pid is not its own process group ($pgid); launch it with setsid" >&2; exit 2; }
alive() { [ -r "/proc/$1/stat" ] && ! awk '{exit !($3 == "Z")}' "/proc/$1/stat"; }   # a zombie is dead
alive "$pid" || { echo "harness $pid is not running"; exit 0; }
kill -9 -- "-$pgid"
for _ in 1 2 3 4 5; do alive "$pid" || break; sleep 1; done
if alive "$pid"; then echo "STILL ALIVE: $pid" >&2; exit 3; fi
left=$(ps -eo pgid= | awk -v g="$pgid" '$1 == g' | wc -l)
[ "$left" -eq 0 ] || { echo "process group $pgid still has $left live or zombie members" >&2; }
echo "stopped process group $pgid"
