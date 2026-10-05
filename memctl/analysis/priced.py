"""Score a sweep under a priced archive:  python -m memctl.analysis.priced runs/<sweep> --price 0.05

Each episode scores (correct - price * items written to the archive) / queries:
task success, minus the archive's cost in the same unit (queries), so that a
controller that archives everything pays for it. Price 0 gives task success.
Prints one row per cell: mean priced score with a 95% bootstrap interval over
episodes, task success, and archive writes per episode.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from memctl.analysis.load import read_jsonl

ARCHIVE_WRITES = ("MOVE_TO_ARCHIVE", "COMPACT_AND_ARCHIVE")


def priced_score(episode: dict, price: float) -> float:
    writes = sum(episode["action_counts"]["controller"].get(op, 0) for op in ARCHIVE_WRITES)
    return (episode["correct"] - price * writes) / episode["queries"] if episode["queries"] else 0.0


def bootstrap(values: list[float], samples: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(values, k=len(values))) / len(values) for _ in range(samples))
    return means[int(0.025 * samples)], means[int(0.975 * samples) - 1]


def score_sweep(root: str | Path, price: float) -> list[dict]:
    rows = []
    for cell in sorted(Path(root).iterdir()):
        path = cell / "episodes.jsonl"
        if not path.exists():
            continue
        episodes = read_jsonl(path)
        scores = [priced_score(e, price) for e in episodes]
        low, high = bootstrap(scores)
        writes = [sum(e["action_counts"]["controller"].get(op, 0) for op in ARCHIVE_WRITES) for e in episodes]
        rows.append({
            "cell": cell.name, "episodes": len(episodes), "price": price,
            "priced": sum(scores) / len(scores), "ci95": [low, high],
            "success": sum(e["task_success"] for e in episodes) / len(episodes),
            "archive_writes": sum(writes) / len(writes),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sweep")
    parser.add_argument("--price", type=float, default=0.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    rows = score_sweep(args.sweep, args.price)
    if args.json:
        print(json.dumps(rows, indent=2))
        return
    print(f"{'cell':55s} {'priced':>7s} {'95% CI':>15s} {'success':>8s} {'writes':>7s}")
    for row in rows:
        low, high = row["ci95"]
        print(f"{row['cell']:55s} {row['priced']:7.3f} [{low:6.3f},{high:6.3f}] {row['success']:8.3f} {row['archive_writes']:7.1f}")


if __name__ == "__main__":
    main()
