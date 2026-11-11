#!/bin/bash
set -e
cd /home/qamil-mirza/Code/llm-memctl
export HF_HUB_OFFLINE=1
.venv/bin/python scripts/exp57_stage2.py read > runs/exp57/logs/stage2_read.log 2>&1
.venv/bin/python scripts/exp57_stage2.py report > runs/exp57/logs/stage2_report.log 2>&1
.venv/bin/python scripts/exp57_stage2.py tokens > runs/exp57/logs/stage2_tokens.log 2>&1
echo CHAIN2_DONE >> runs/exp57/logs/stage2_tokens.log
