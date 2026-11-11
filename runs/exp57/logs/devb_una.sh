#!/bin/bash
set -e
cd /home/qamil-mirza/Code/llm-memctl
export HF_HUB_OFFLINE=1
.venv/bin/python scripts/exp57_stage2.py devb-una > runs/exp57/logs/devb_una.log 2>&1
echo DEVB_UNA_DONE >> runs/exp57/logs/devb_una.log
