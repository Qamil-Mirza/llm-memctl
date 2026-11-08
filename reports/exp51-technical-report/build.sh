#!/usr/bin/env bash
# Build the report PDF with Tectonic (https://tectonic-typesetting.github.io). Usage: ./build.sh [path-to-tectonic]
# Figure 3 (trade-off) is regenerated separately: figures/tradeoff_data.py (repo .venv) then figures/tradeoff_plot.py.
set -euo pipefail
cd "$(dirname "$0")"
TECTONIC="${1:-tectonic}"
"$TECTONIC" --keep-logs main.tex
mv main.pdf exp51-technical-report.pdf
rm -f main.log main.blg
echo "wrote exp51-technical-report.pdf"
