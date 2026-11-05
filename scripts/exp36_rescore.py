"""§36: rescore stored rows from their top-20 logprobs with the current scorer (no regeneration).

    .venv/bin/python scripts/exp36_rescore.py --folds 0,1 [--root runs/exp36] [--out runs/exp36_rescored]

The original row files are never touched: rescored rows go to <out>/<reader>/<item>/fold<k>.jsonl (C4 rows copied
unchanged) and <out>/<reader>/_manifest.json carries the original manifest plus `scorer_version`. Folds 2-4 are
refused without --confirm (they stay sealed until the coordinator asks). The report reads the result with
`--root runs/exp36_rescored`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from memctl.analysis import verification as ver  # noqa: E402
from scripts.exp35_generate import READERS, read_jsonl, write_jsonl  # noqa: E402

CONFIRM_FOLDS = {2, 3, 4}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", default="0,1")
    parser.add_argument("--root", default="runs/exp36")
    parser.add_argument("--out", default="runs/exp36_rescored")
    parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args(argv)
    folds = sorted({int(f) for f in args.folds.split(",")})
    if set(folds) & CONFIRM_FOLDS and not args.confirm:
        print(f"REFUSED: folds {sorted(set(folds) & CONFIRM_FOLDS)} are sealed; pass --confirm only when asked")
        return 2
    root, out = Path(args.root), Path(args.out)
    if root.resolve() == out.resolve():
        raise SystemExit("REFUSED: --out must differ from --root (originals are never overwritten)")
    counts = {}
    for reader in READERS:
        manifest_path = root / reader / "_manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            manifest.update({"scorer_version": ver.SCORER_VERSION, "rescored_from": str(root), "rescored_folds": folds})
            (out / reader).mkdir(parents=True, exist_ok=True)
            (out / reader / "_manifest.json").write_text(json.dumps(manifest, indent=1))
        for item in ver.ITEMS:
            for fold in folds:
                src = root / reader / item / f"fold{fold}.jsonl"
                if not src.exists():
                    continue
                rows = [ver.rescore_row(r) for r in read_jsonl(src)]
                write_jsonl(out / reader / item / f"fold{fold}.jsonl", rows)
                counts[f"{reader}/{item}"] = counts.get(f"{reader}/{item}", 0) + len(rows)
    print(f"rescored with {ver.SCORER_VERSION} into {out} (folds {folds}): {json.dumps(counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
