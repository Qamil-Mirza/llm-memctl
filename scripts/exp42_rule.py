"""§42: write the frozen selective-steering rule, rule42.json, from §40's committed outputs (proposal
research/proposals/2026-10-25-exp42-selective-steering.md, rev 1 at 8fd2613).

    python scripts/exp42_rule.py                 # writes runs/exp42_inputs/rule42.json
    python scripts/exp42_rule.py --check         # recomputes and compares with the file on disk (exit 1 if different)

tau = numpy.percentile(s_proj over §40's 120 R rows, 70, method='linear'), asserted equal to -18.896762090918696; the
input sha (runs/exp40_pod/qwen3b/apass.jsonl) is asserted and pinned in the file. Reads only §40's apass.jsonl and
selection.json; writes nothing under runs/exp40_pod.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from memctl.analysis import exp40 as ex  # noqa: E402


def build(apass: str = ex.APASS40, selection: str = ex.SELECTION40) -> dict:
    sha = hashlib.sha256(Path(apass).read_bytes()).hexdigest()
    if sha != ex.APASS40_SHA256:
        raise SystemExit(f"REFUSED: {apass} sha {sha}, expected {ex.APASS40_SHA256}")
    rows = [json.loads(x) for x in Path(apass).read_text().splitlines() if x.strip()]
    R = json.loads(Path(selection).read_text())["R"]
    rule = ex.compute_rule42(rows, R, sha)
    if rule["tau"] != ex.TAU42:
        raise SystemExit(f"REFUSED: tau {rule['tau']!r} != the pinned {ex.TAU42!r}")
    ex.check_rule42(rule)
    return rule


def text_of(rule: dict) -> str:
    return json.dumps(rule, indent=1, ensure_ascii=False) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=ex.RULE42_PATH)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    rule = build()
    out = Path(args.out)
    if args.check:
        same = out.exists() and out.read_text() == text_of(rule)
        print(f"rule42 {'matches' if same else 'DIFFERS FROM'} the recomputation: tau {rule['tau']!r}")
        return 0 if same else 1
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text_of(rule))
    print(f"wrote {out}: tau {rule['tau']!r} ({rule['tau_method']}), fires on {rule['fires_on_40_R']} of "
          f"{rule['n_R']} §40 R rows; input sha {rule['input_sha256']}; file sha "
          f"{hashlib.sha256(out.read_bytes()).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
