#!/usr/bin/env python
"""§50 pod helpers (scripts/exp50_pod.sh): the §49 helpers (make-stub, stage-time, wait-mark) reused; population and
projection with §50's counts. Nothing here reads or prints an outcome.

    python scripts/exp50_pod_run.py population [--expect-cases-sha S]
    python scripts/exp50_pod_run.py project --smoke S --elapsed 900 [--rate 0.59 --limit 3.00]
    python scripts/exp50_pod_run.py make-stub | stage-time | wait-mark ...   (as exp49_pod_run)

`project`: the D-22 smoke's seconds per measured row x the case count, plus its seconds per patched generation x
N_RELEASES (dev W sweep 35 x about 60 pairs + no-ops + held arms about 60 pairs x 2 x 21, rounded up), x 1.30, plus the
seconds spent and a 2,400 s allowance for the gate read, audit and design commit; exit 7 above the limit in dollars.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import exp49_pod_run as P49  # noqa: E402

CASES = Path("runs/exp50/cases50.jsonl")
N_RELEASES, FACTOR, MARK_ALLOWANCE_S = 5000, 1.30, 2400


def population(expect: str | None) -> int:
    data = CASES.read_bytes()
    rows = [json.loads(l) for l in data.decode().splitlines() if l.strip()]
    c = Counter(r["split"] for r in rows)
    problems = []
    if not c.get("dev") or not c.get("held"):
        problems.append(f"split counts {dict(c)}")
    if expect and hashlib.sha256(data).hexdigest() != expect:
        problems.append("cases50.jsonl sha differs from bundle_info")
    print(json.dumps({"rows": len(rows), "by_split": dict(c), "problems": problems}), flush=True)
    return P49.EXIT_INPUT if problems else 0


def project(smoke: Path, elapsed: float, rate: float, limit: float) -> int:
    sm = json.loads(Path(smoke).read_text())
    n_rows = sum(1 for l in CASES.read_text().splitlines() if l.strip())
    run_s = (sm["per_row_measure_s"] * n_rows + sm["per_release_s"] * N_RELEASES) * FACTOR
    total = elapsed + run_s + MARK_ALLOWANCE_S
    dollars = total / 3600 * rate
    payload = {"elapsed_s": elapsed, "n_rows": n_rows, "per_row_measure_s": sm["per_row_measure_s"],
               "per_release_s": sm["per_release_s"], "run_projection_s": run_s, "total_s": total,
               "dollars": round(dollars, 3), "limit": limit, "rate": rate, "stop_and_ask": dollars > limit}
    Path(smoke).with_name("projection.json").write_text(json.dumps(payload, indent=1))
    print(json.dumps(payload), flush=True)
    return P49.EXIT_PROJECTION if dollars > limit else 0


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] in ("population", "project"):
        import argparse
        ap = argparse.ArgumentParser()
        sub = ap.add_subparsers(dest="cmd", required=True)
        m = sub.add_parser("population"); m.add_argument("--expect-cases-sha", default=None)
        m = sub.add_parser("project"); m.add_argument("--smoke", required=True)
        m.add_argument("--elapsed", type=float, required=True); m.add_argument("--rate", type=float, default=0.59)
        m.add_argument("--limit", type=float, default=3.00)
        a = ap.parse_args()
        if a.cmd == "population":
            return population(a.expect_cases_sha)
        return project(Path(a.smoke), a.elapsed, a.rate, a.limit)
    return P49.main()


if __name__ == "__main__":
    sys.exit(main())
