#!/bin/bash
# §57 stage 3 chain: waits for the dev-B UNA run, then marker -> commit -> equality (gate) -> calib -> commit -> run -> dhat -> report
set -e
cd /home/qamil-mirza/Code/llm-memctl
export HF_HUB_OFFLINE=1
L=runs/exp57/logs
U=$(cat $L/devb_una.pid)
while kill -0 $U 2>/dev/null; do sleep 30; done
grep -q DEVB_UNA_DONE $L/devb_una.log || { echo "devb-una did not finish" >> $L/chain3.out; exit 1; }
git add -f runs/exp57/devb_una_counts.json $L/devb_una.log $L/devb_una.pid
git commit -q -m "§57 dev-B UNA generation: 300 rows, §55 P0 settings, class counts (records stay local in runs/exp56/run/devb_una)"
.venv/bin/python scripts/exp57_stage3.py marker > $L/stage3_marker.log 2>&1
git add -f runs/exp57/c_mean_marker.npz runs/exp57/marker.json $L/stage3_marker.log $L/chain3.sh
git commit -q -m "§57 rev 0b: marker C means (dev-A C rows, position-matched by token) frozen before any dev-B arm; K_N-at-marker descriptive"
.venv/bin/python scripts/exp57_stage3.py equality > $L/stage3_equality.log 2>&1
.venv/bin/python -c "import json,sys; sys.exit(0 if json.load(open('runs/exp57/stage3_equality.json'))['PASS'] else 1)" \
  || { echo "EQUALITY FAIL" >> $L/chain3.out; exit 1; }
git add -f runs/exp57/stage3_equality.json runs/exp57/stage3_equality_rows.json $L/stage3_equality.log
git commit -q -m "§57 stage-3 equality (dev-A, all stage-3 arms incl. K_N@marker and the B1 grid): PASS; measured per-arm rate and ETA"
.venv/bin/python scripts/exp57_stage3.py calib > $L/stage3_calib.log 2>&1
git add -f runs/exp57/b_cal.json $L/stage3_calib.log
git commit -q -m "§57 stage 3: b_cal frozen from dev-A FA calibration (before any dev-B arm)"
.venv/bin/python scripts/exp57_stage3.py run > $L/stage3_run.log 2>&1
.venv/bin/python scripts/exp57_stage3.py dhat > $L/stage3_dhat.log 2>&1
.venv/bin/python scripts/exp57_stage3.py report > $L/stage3_report.log 2>&1
echo CHAIN3_DONE >> $L/chain3.out
