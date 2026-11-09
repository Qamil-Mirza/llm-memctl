#!/usr/bin/env bash
# Read-only account check ($0, creates nothing):
#   KEY_VAR=RUNPOD_API_KEY_2 EXPECT_EMAIL=qamilmirza@gmail.com CAP_USD=0 bash selfdoubt/pod_preflight.sh
set -euo pipefail
cd "$(dirname "$0")/.."
. selfdoubt/pod_key.sh
HDR=$(mktemp); trap 'rm -f "$HDR"' EXIT
rp_write_hdr "$HDR"
rp_preflight "$HDR" "${CAP_USD:-0}"
