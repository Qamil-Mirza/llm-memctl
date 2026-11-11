#!/bin/bash
set -e
cd /home/qamil-mirza/Code/llm-memctl
export HF_HUB_OFFLINE=1
.venv/bin/python scripts/exp57_stage0.py cmean-note > runs/exp57/logs/cmean_note.log 2>&1
.venv/bin/python scripts/exp57_stage0.py freeze > runs/exp57/logs/freeze.log 2>&1
.venv/bin/python scripts/exp57_stage0.py validity > runs/exp57/logs/validity.log 2>&1
.venv/bin/python scripts/exp57_stage0.py equality > runs/exp57/logs/equality.log 2>&1
echo CHAIN_DONE >> runs/exp57/logs/freeze.log
