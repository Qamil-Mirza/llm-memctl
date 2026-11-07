#!/bin/bash
# after calibration (PID $1): laptop smoke on 6 labelled N-dev rows, smoke report, manifest
cd /home/qamil-mirza/Code/llm-memctl
while kill -0 "$1" 2>/dev/null; do sleep 10; done
echo "[$(date +%T)] calibration exited"
if [ ! runs/exp53/calibration.json -nt runs/exp53/calibrate.pid ]; then echo "CHAIN FAIL: no fresh calibration.json"; exit 2; fi
export HF_HUB_OFFLINE=1
.venv/bin/python scripts/exp53_run.py --stage pod --smoke --root runs/exp53/smoke_laptop \
  --case-ids "$(cat runs/exp53/smoke_laptop_ids.txt)" --lme-limit 2 > runs/exp53/smoke_laptop.log 2>&1 || { echo "CHAIN FAIL: smoke rc=$?"; exit 3; }
echo "[$(date +%T)] smoke done"
.venv/bin/python scripts/exp53_run.py --stage report --smoke --root runs/exp53/smoke_laptop >> runs/exp53/smoke_laptop.log 2>&1 || { echo "CHAIN FAIL: report rc=$?"; exit 4; }
echo "[$(date +%T)] smoke report done"
.venv/bin/python scripts/exp53_run.py --stage manifest > runs/exp53/manifest.log 2>&1 || { echo "CHAIN FAIL: manifest rc=$?"; exit 5; }
echo "[$(date +%T)] CHAIN OK $(cat runs/exp53/manifest.log)"
