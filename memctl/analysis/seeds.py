"""Pool training seeds in a sweep:  python -m memctl.analysis.seeds runs/<sweep> [--metric task_success]

Cells whose controller label ends in `_s<k>` are treated as training seeds of one
method. For each method and budget it prints the mean over seeds, the spread
(min to max over seeds), and the mean of a few diagnostics. Heuristics and the
oracle have one cell each and are shown as they are.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

CELL = re.compile(r"^(?P<label>.+?)(?:_s(?P<seed>\d+))?__fraction(?P<fraction>[\d.]+)$")
DIAGNOSTICS = ("requirements_destroyed", "retrieved_items", "retrieval_precision", "archive_tokens_peak")


def pooled(root: str | Path, metric: str = "task_success") -> dict:
    groups: dict[tuple[str, float], list[dict]] = defaultdict(list)
    for cell in sorted(Path(root).iterdir()):
        match = CELL.match(cell.name)
        if not match or not (cell / "summary.json").exists():
            continue
        groups[(match["label"], float(match["fraction"]))].append(json.loads((cell / "summary.json").read_text()))
    table: dict = defaultdict(dict)
    for (label, fraction), summaries in groups.items():
        values = [s[metric] for s in summaries]
        row = {"mean": sum(values) / len(values), "min": min(values), "max": max(values), "seeds": len(values)}
        for name in DIAGNOSTICS:
            known = [s[name] for s in summaries if s.get(name) is not None]
            row[name] = sum(known) / len(known) if known else None
        table[label][fraction] = row
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sweep")
    parser.add_argument("--metric", default="task_success")
    args = parser.parse_args()
    table = pooled(args.sweep, args.metric)
    fractions = sorted({f for rows in table.values() for f in rows})
    print(f"{'method':32s}" + "".join(f"{f'{x:g}':>22s}" for x in fractions))
    for label in sorted(table):
        cells = []
        for fraction in fractions:
            row = table[label].get(fraction)
            if row is None:
                cells.append(f"{'-':>22s}")
            elif row["seeds"] > 1:
                cells.append(f"{row['mean']:7.3f} [{row['min']:.3f},{row['max']:.3f}]".rjust(22))
            else:
                cells.append(f"{row['mean']:7.3f}".rjust(22))
        print(f"{label:32s}" + "".join(cells))


if __name__ == "__main__":
    main()
