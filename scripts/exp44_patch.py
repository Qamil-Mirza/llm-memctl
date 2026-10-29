#!/usr/bin/env python
"""§44: single-position activation patching (clean -> evidence-cut at P1), one row per question, resumable.

    GUARD_VM_GB=22 runs/_pipelines/guard.sh .venv/bin/python scripts/exp44_patch.py --model qwen3b --folds 0,1 --limit 5 --run --patch-all
    GUARD_VM_GB=22 runs/_pipelines/guard.sh .venv/bin/python scripts/exp44_patch.py --model qwen7b --folds 0 --limit 2 --timing-only

Spec: research/proposals/2026-10-29-exp44-patching.md (rev 0). Population: the §34 330 paired rows (answerable,
all evidence in view, at least one evidence line, for both Qwen sizes; identical prompts; asserted). Folds 0-1 only:
a confirmation fold (2-4) is refused (exit 2) — the confirmation read is not yet authorised (spec §6 step 4: after the
dev report is committed and the Advisor has recorded the fixed filter; `CONFIRM_AUTHORISED` is flipped then).

Outputs runs/exp44_patch/<model>/<qid>.json (sha-stamped: method, engine, model, clean and cut input ids; a row is
done only if all match, otherwise it is redone), _timing.json (appended per run: s/row, peak RSS), _failed.json.
--timing-only: writes under runs/exp44_stage0/<model>/ instead and prints the projection for 330 rows.

Rev 5 (filter version `pt.FILTER_VERSION`): which rows get patched (without --patch-all) is the relaxed filter WITH
the 1-nat gap (`pt.patch_eligible`). The resume stamp is the passes' stamp only (method, engine, model, prompts, ids);
the filter version is not in it, so a complete row written under rev 0 is never redone. Its stored derived fields
(eligible, sanity_asserted, ...) are left as written: the report re-derives them from the recorded lp / rank / R
(`pt.derive`), which is the one source of truth. A complete row is redone only if its stamp differs, or it is
unpatched and the current rule (or --patch-all) wants it patched.
Exit 2: confirmation folds asked for. Exit 3: |R(L-1) - 1| >= 0.01 on an eligible patched row (the row is written
first, with its sanity block, then the run stops). |R(0)| >= 0.05 does not stop the run (rev 5): the row is written
with r0_flag = True and the run goes on. Exit 4: the §34 prompt hash on disk differs from the record's.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memctl.analysis import patching as pt  # noqa: E402

MODELS = {"qwen3b": {"id": "Qwen/Qwen2.5-3B-Instruct", "dtype": "float32", "n_layers": 36},
          "qwen7b": {"id": "Qwen/Qwen2.5-7B-Instruct", "dtype": "bfloat16", "n_layers": 28}}
LOCAL_7B_CACHE = "cache/exp44_hf"            # the 7B weights live inside the repo (not ~/.cache)
CONFIRM_AUTHORISED = False
N_POPULATION = 330
REFUSED, SANITY, PROMPT_MISMATCH = 2, 3, 4


def write_json(path: Path, payload) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, path)


def check_folds(folds, confirm: bool) -> None:
    bad = sorted(set(folds) - set(pt.DEV_FOLDS))
    if bad and not (confirm and CONFIRM_AUTHORISED):
        why = "--confirm not given" if not confirm else "the confirmation read is not yet authorised (spec §6.4)"
        print(f"exp44_patch: folds {bad} are confirmation folds; {why}", file=sys.stderr, flush=True)
        raise SystemExit(REFUSED)


def population(model: str, folds, runs_dir: str = "runs"):
    """The §34 population rows of `folds` for `model`, paired with the other Qwen size (same rule, same prompt)."""
    from memctl.jacobian_data import load_records

    other = "qwen7b" if model == "qwen3b" else "qwen3b"
    rows = []
    for fold in folds:
        mine = {r.question_id: r for r in load_records(model, fold, runs_dir)[0]}
        theirs = {r.question_id: r for r in load_records(other, fold, runs_dir)[0]}
        inview = lambda r: r.answerable and r.all_in_view and bool(r.evidence_lines)  # noqa: E731
        for q in sorted(set(mine) & set(theirs)):
            a, b = mine[q], theirs[q]
            if not (inview(a) and inview(b)):
                continue
            if a.prompt != b.prompt or a.evidence_lines != b.evidence_lines:
                raise ValueError(f"{q}: the 3B and 7B §34 records differ (prompt or evidence)")
            rows.append(a)
    return rows


def check_exp34_prompt(record, model: str, root: Path = Path("runs/exp34_jlens")) -> None:
    path = root / model / f"{record.question_id}.json"
    if not path.exists():
        return
    on_disk = json.loads(path.read_text()).get("prompt_sha256")
    if on_disk != pt.sha256_text(record.prompt):
        print(f"exp44_patch: {record.question_id}: §34 prompt sha differs from the record's", file=sys.stderr)
        raise SystemExit(PROMPT_MISMATCH)


def stamp(engine_name: str, model_id: str, x: pt.RowInputs) -> dict:
    return {"method_sha256": pt.method_sha256(), "engine": engine_name, "model_id": model_id,
            "clean_prompt_sha256": pt.sha256_text(x.clean_prompt), "cut_prompt_sha256": pt.sha256_text(x.cut_prompt),
            "clean_ids_sha256": pt.sha256_ids(x.clean_ids), "cut_ids_sha256": pt.sha256_ids(x.cut_ids)}


def is_done(path: Path, want: dict, patch_all: bool) -> bool:
    try:
        row = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    if row.get("stamp") != want:
        return False
    if row.get("patched"):
        return True
    # a row recorded unpatched is redone under --patch-all, or if the current rule makes it eligible
    wants_patch = patch_all or pt.patch_eligible(row["rank_clean"], row["rank_corrupt"], row["lp_clean"],
                                                 row["lp_corrupt"])
    return not wants_patch


def resolve_model(model: str) -> str:
    """The 3B from the HF cache by id; the 7B from the in-repo snapshot (never downloaded here)."""
    if model != "qwen7b":
        return MODELS[model]["id"]
    snaps = sorted(Path(LOCAL_7B_CACHE).glob("models--Qwen--Qwen2.5-7B-Instruct/snapshots/*"))
    if not snaps:
        raise SystemExit(f"exp44_patch: no 7B snapshot under {LOCAL_7B_CACHE}")
    return str(snaps[-1])


def run(engine, records, out: Path, model: str, model_id: str, patch_all: bool, log=print) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    failed_path = out / "_failed.json"
    failed = json.loads(failed_path.read_text()) if failed_path.exists() else {}
    seconds, sanity_rows, flagged = {}, [], []
    for number, record in enumerate(records):
        try:
            x = pt.build_inputs(engine.tokenizer, record)
        except ValueError as error:
            failed[record.question_id] = repr(error)
            write_json(failed_path, failed)
            log(f"[{number + 1}/{len(records)}] {record.question_id} FAILED {error!r}")
            continue
        want = stamp(engine.name, model_id, x)
        path = out / f"{record.question_id}.json"
        if is_done(path, want, patch_all):
            continue
        row = pt.patch_row(engine, x, patch=True if patch_all else None)
        row.update(pt.derive(row))
        row.update({"model": model, "stamp": want, "correct": record.correct, "unknown": record.unknown})
        write_json(path, row)
        failed.pop(record.question_id, None)
        write_json(failed_path, failed)
        seconds[record.question_id] = row["timing"]["total_s"]
        line = (f"[{number + 1}/{len(records)}] {record.question_id} f{record.fold} {row['timing']['total_s']:.1f}s "
                f"tok={row['n_tokens_clean']}/{row['n_tokens_cut']} rank clean={row['rank_clean']} "
                f"corrupt={row['rank_corrupt']} lp {row['lp_clean']:.2f}/{row['lp_corrupt']:.2f}"
                f"{' eligible' if row['eligible'] else ''}{' GAP<1 (excluded)' if row['gap_excluded_relaxed'] else ''}")
        if row["patched"]:
            s = row["sanity"]
            line += (f" R(0)={s['r_zero']} R(L-1)={s['r_final']} d_c={row['d_c']} "
                     f"R(L-1) {'PASS' if s['final_ok'] else 'FAIL'}{'' if row['sanity_asserted'] else ' (not asserted)'}"
                     f"{' R0_FLAG (excluded from claims)' if row['r0_flag'] else ''}")
            sanity_rows.append({"question_id": record.question_id, **s, "asserted": row["sanity_asserted"],
                                "r0_flag": row["r0_flag"]})
            if row["r0_flag"]:
                flagged.append(record.question_id)
        log(line)
        try:
            pt.check_sanity(row)
        except pt.SanityError as error:
            log(f"exp44_patch: {error}")
            raise SystemExit(SANITY)
    return {"seconds": seconds, "sanity": sanity_rows, "r0_flagged": flagged}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=sorted(MODELS))
    p.add_argument("--folds", default="0,1")
    p.add_argument("--confirm", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="first N population rows (fold, qid order)")
    p.add_argument("--qids", default=None, help="comma-separated question ids (within --folds)")
    p.add_argument("--timing-only", action="store_true", help="stage 0: write under runs/exp44_stage0/<model>/")
    p.add_argument("--run", action="store_true", help="write under runs/exp44_patch/<model>/")
    p.add_argument("--patch-all", action="store_true", help="patch every row, not only the filter-eligible ones")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)
    folds = [int(f) for f in args.folds.split(",")]
    check_folds(folds, args.confirm)
    if args.timing_only == args.run:
        p.error("give exactly one of --timing-only and --run")
    records = population(args.model, folds)
    if args.qids:
        keep = set(args.qids.split(","))
        records = [r for r in records if r.question_id in keep]
    records = records[: args.limit] if args.limit else records
    for record in records:
        check_exp34_prompt(record, args.model)
    out = Path(args.out or (f"runs/exp44_stage0/{args.model}" if args.timing_only else f"runs/exp44_patch/{args.model}"))
    from memctl.analysis.steering import HFEngine

    t0 = time.perf_counter()
    engine = HFEngine(resolve_model(args.model), threads=args.threads, dtype=MODELS[args.model]["dtype"])
    load_s = time.perf_counter() - t0
    rss_after_load = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"loaded {args.model} ({engine.name}) in {load_s:.0f}s, peak RSS {rss_after_load:.0f} MB, "
          f"{len(records)} rows", flush=True)
    started = time.perf_counter()
    result = run(engine, records, out, args.model, MODELS[args.model]["id"], args.patch_all,
                 log=lambda s: print(s, flush=True))
    secs = result["seconds"]
    per_row = sum(secs.values()) / len(secs) if secs else None
    summary = {"model": args.model, "model_id": MODELS[args.model]["id"], "engine": engine.name,
               "threads": args.threads, "folds": folds, "rows": len(secs), "patch_all": args.patch_all,
               "seconds_per_row": per_row, "seconds": secs, "load_s": load_s, "wall_s": time.perf_counter() - started,
               "peak_rss_mb_after_load": rss_after_load,
               "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
               "projected_h_330_rows_all_patched": per_row * N_POPULATION / 3600 if per_row else None,
               "sanity": result["sanity"], "r0_flagged": result["r0_flagged"], "method_sha256": pt.method_sha256(),
               "filter_version": pt.FILTER_VERSION}
    timing_path = out / "_timing.json"
    existing = json.loads(timing_path.read_text()) if timing_path.exists() else {"runs": []}
    existing["runs"].append(summary)
    write_json(timing_path, existing)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("seconds", "sanity")}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
