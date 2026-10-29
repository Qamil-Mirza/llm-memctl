#!/usr/bin/env python
"""§44 rev 1 amendment §B: conditions S (clean / swap / cut, following, classes, bidirectional patching, controls) and U
(no "unknown" option, with and without the §38 Steer setting), one resumable sha-stamped row per question.

    GUARD_VM_GB=22 runs/_pipelines/guard.sh .venv/bin/python scripts/exp44_swap.py --model qwen3b --folds 0,1 --limit 10 --smoke
    (same prefix) .venv/bin/python scripts/exp44_swap.py --model qwen3b --pooled          # all 330 (refused until authorised)
    (Llama: prefix MALLOC_ARENA_MAX=2 HF_ENABLE_PARALLEL_LOADING=false as §43)

Spec: research/proposals/2026-10-29-exp44-patching.md, rev 1 amendment §B (D-45) + rev 2 (5146f5b). Rows: the swap
subset of runs/exp44_swapgate/eligible_<model>.jsonl (rev 2 rules; the swap prompt is rebuilt from the gate's own
functions and must match the gate's swap_prompt_sha256). Population and §23 labels: the gate's qwen_population /
llama_population (330 rows per family).

Fold guard: folds 0-1 run freely (the 10-row S0 smoke is the only dev-fold use and decides nothing). Folds 2-4 need
--pooled and POOL_AUTHORISED = True, flipped only after the Advisor's re-read of the rev 2 recount (rev 3): exit 2.
Stages (each resumable): S base (every row in scope), S patch (rows that follow at 1 nat; the mismatched donor's swap
residual must be on disk, so a row whose donor is not done yet is patched in a final pass), U (in-view refusal rows of
the full population in scope: U and U_steer; swap-subset refusal rows add A and A_steer for S2). --smoke: S only.
Outputs runs/exp44_swap/<model>/{S/<qid>.json, resid/<qid>.npz, U/<qid>.json, _timing.json, _failed.json}.
Exit 3: a sanity bound failed on a patched row (the row is written first). Exit 5: the gate's swap prompt sha differs.
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

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from memctl.analysis import families as F  # noqa: E402
from memctl.analysis import patching as pt  # noqa: E402
from memctl.analysis import swap as sw  # noqa: E402

MODELS = {"qwen3b": {"id": "Qwen/Qwen2.5-3B-Instruct", "family": None, "refusal": F.QWEN3B},
          "llama3b": {"id": F.LLAMA3B.model_id, "family": F.LLAMA3B, "refusal": F.LLAMA3B}}
GATE_DIR = Path("runs/exp44_swapgate")
OUT = Path("runs/exp44_swap")
DEV_FOLDS = (0, 1)
POOL_AUTHORISED = True    # rev 4 (f9952b2): POOL on all 330 rows; the swap set is fixed from here
REFUSED, SANITY, GATE_MISMATCH = 2, 3, 5


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, path)


def check_folds(folds, pooled: bool) -> None:
    bad = sorted(set(folds) - set(DEV_FOLDS))
    if bad and not (pooled and POOL_AUTHORISED):
        why = "--pooled not given" if not pooled else "the pooled read is not yet authorised (rev 2 recount flagged)"
        print(f"exp44_swap: folds {bad} are confirmation folds; {why}", file=sys.stderr, flush=True)
        raise SystemExit(REFUSED)


def load_population(model: str):
    import exp44_swap_gate as G
    return G.qwen_population() if model == "qwen3b" else G.llama_population()


def load_eligible(model: str, gate_dir: Path = GATE_DIR) -> tuple[list[dict], str]:
    path = gate_dir / f"eligible_{model}.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    gate = json.loads((gate_dir / "gate.json").read_text())
    if gate.get("rev") != 3:
        raise SystemExit(f"exp44_swap: {gate_dir}/gate.json is not the rev 3 gate")
    return sorted(rows, key=lambda e: (e["fold"], e["question_id"])), hashlib.sha256(path.read_bytes()).hexdigest()


def rebuild_swap(record, e: dict):
    """The gate's swap prompt for one eligible row, re-derived and checked against its sha."""
    import exp44_swap_gate as G
    why, place = G.place_gold(record, G.Checker())
    if why is not None or place["line_index"] != e["evidence_line_index"]:
        raise SystemExit(f"exp44_swap: {record.question_id}: placement differs from the gate ({why})")
    prompt, old, new = G.swap_prompt(record, place, e["substitute"])
    if pt.sha256_text(prompt) != e["swap_prompt_sha256"] or pt.sha256_text(record.prompt) != e["clean_prompt_sha256"]:
        print(f"exp44_swap: {record.question_id}: swap / clean prompt sha differs from the gate's", file=sys.stderr)
        raise SystemExit(GATE_MISMATCH)
    return prompt, old, new


def label23(record) -> str:
    return "success" if record.correct else ("refusal" if record.unknown else "wrong")


def stamp(engine_name: str, model_id: str, x: sw.SwapInputs, e: dict, eligible_sha: str) -> dict:
    return {"method_sha256": sw.method_sha256(), "engine": engine_name, "model_id": model_id,
            "clean_ids_sha256": sw.sha_ids(x.clean_ids), "swap_ids_sha256": sw.sha_ids(x.swap_ids),
            "cut_ids_sha256": sw.sha_ids(x.cut_ids), "substitute": e["substitute"],
            "swap_prompt_sha256": e["swap_prompt_sha256"], "eligible_sha256": eligible_sha}


def read_row(path: Path, want: dict) -> dict | None:
    try:
        row = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return row if row.get("stamp") == want else None


class SRun:
    def __init__(self, engine, model: str, records: dict, order: list[dict], out: Path, eligible_sha: str, log=print):
        self.engine, self.model, self.records, self.order, self.out, self.log = engine, model, records, order, out, log
        self.eligible_sha = eligible_sha
        self.refusal_ids = MODELS[model]["refusal"].refusal_ids
        self.chat_fn = None if MODELS[model]["family"] is None else engine.chat_text
        self.inputs, self.seconds = {}, {}

    def x(self, i: int) -> sw.SwapInputs:
        e = self.order[i]
        if i not in self.inputs:
            rec = self.records[e["question_id"]]
            prompt, old, new = rebuild_swap(rec, e)
            self.inputs[i] = sw.build_swap_inputs(self.engine.tokenizer, rec, prompt, old, new, e["substitute"],
                                                  self.refusal_ids, self.chat_fn)
            x = self.inputs[i]
            if ((x.g_id, x.s_id, list(x.prefix_ids)) != (e["gold_read_id"], e["substitute_read_id"],
                                                         e["shared_prefix_ids"])):
                raise SystemExit(f"exp44_swap: {e['question_id']}: read ids / shared prefix differ from the gate's")
        return self.inputs[i]

    def path(self, i: int) -> Path:
        return self.out / "S" / f"{self.order[i]['question_id']}.json"

    def resid(self, i: int) -> Path:
        return self.out / "resid" / f"{self.order[i]['question_id']}.npz"

    def want(self, i: int) -> dict:
        return stamp(self.engine.name, MODELS[self.model]["id"], self.x(i), self.order[i], self.eligible_sha)

    def donor(self, i: int) -> int | None:
        return sw.choose_donor(self.order, i)

    def donor_resid(self, j: int | None) -> np.ndarray | None:
        if j is None:
            return None
        row = read_row(self.path(j), self.want(j))
        if row is None or not self.resid(j).exists():
            return None
        return np.load(self.resid(j))["swap"]

    def needs_patch(self, row: dict, i: int) -> bool:
        if not row["follows"]["1"]:
            return False
        j = self.donor(i)
        donor_q = None if j is None else self.order[j]["question_id"]
        return not row["patched"] or row.get("donor") != donor_q

    def finish(self, i: int, row: dict) -> None:
        rec = self.records[self.order[i]["question_id"]]
        row.update({"model": self.model, "label": label23(rec), "question_type": rec.question_type,
                    "surface_type": self.order[i]["surface_type"], "substitute": self.order[i]["substitute"],
                    "gold": rec.gold, "row_index": i, "stamp": self.want(i)})
        write_json(self.path(i), row)

    def patch(self, i: int, row: dict, state: dict) -> dict:
        j = self.donor(i)
        donor = self.donor_resid(j)
        if donor is None and j is not None:
            return row                                  # the donor is not done yet: the final pass patches it
        row = sw.patch_passes(self.engine, self.x(i), row, state, donor, i)
        row["donor"] = None if j is None else self.order[j]["question_id"]
        self.finish(i, row)
        s = row
        self.log(f"  patched {row['question_id']}: d_T fwd={s['d_T_fwd']} rev={s['d_T_rev']} "
                 f"T_fwd(0)={s['sanity_fwd']['t_zero']:.4f} T_fwd(L-1)={s['sanity_fwd']['t_final']:.4f} "
                 f"T_rev(0)={s['sanity_rev']['t_zero']:.4f} self={'ok' if s['control_self']['pass'] else 'FAIL'} "
                 f"donor={row['donor']} {row['timing']['patch_s']:.0f}s")
        try:
            sw.check_patch_sanity(row)
        except sw.SanityError as error:
            self.log(f"exp44_swap: {error}")
            raise SystemExit(SANITY)
        return row

    def state_for(self, i: int) -> dict:
        x = self.x(i)
        r = np.load(self.resid(i))
        return {"clean_prep": pt._prepare(self.engine, x.clean_ids), "swap_prep": pt._prepare(self.engine, x.swap_ids),
                "h_clean": r["clean"], "h_swap": r["swap"]}

    def run(self) -> None:
        deferred = []
        for i, e in enumerate(self.order):
            t0 = time.perf_counter()
            try:
                want = self.want(i)
            except pt.AlignmentError as error:
                failed = self.out / "_failed.json"
                f = json.loads(failed.read_text()) if failed.exists() else {}
                f[e["question_id"]] = repr(error)
                write_json(failed, f)
                self.log(f"[{i + 1}/{len(self.order)}] {e['question_id']} FAILED {error!r}")
                continue
            row = read_row(self.path(i), want)
            state = None
            if row is None or not self.resid(i).exists():
                row, state = sw.base_passes(self.engine, self.x(i))
                self.resid(i).parent.mkdir(parents=True, exist_ok=True)
                np.savez(self.resid(i), clean=state["h_clean"], swap=state["h_swap"])
                self.finish(i, row)
                self.log(f"[{i + 1}/{len(self.order)}] {e['question_id']} f{e['fold']} {row['label']:8s} "
                         f"D clean={row['clean']['delta']:+.2f} swap={row['swap']['delta']:+.2f} "
                         f"cut={row['cut']['delta']:+.2f} F={row['F']:+.2f} follows={row['follows']['1']} "
                         f"cls={row['cls']['1']} {row['timing']['base_s']:.0f}s")
            if self.needs_patch(row, i):
                if self.donor_resid(self.donor(i)) is None and self.donor(i) is not None:
                    deferred.append(i)
                else:
                    row = self.patch(i, row, state or self.state_for(i))
            self.seconds[e["question_id"]] = time.perf_counter() - t0
            del state
        for i in deferred:
            t0 = time.perf_counter()
            row = read_row(self.path(i), self.want(i))
            if self.needs_patch(row, i):
                self.patch(i, row, self.state_for(i))
            self.seconds[self.order[i]["question_id"]] += time.perf_counter() - t0


def u_rows(engine, model: str, records: list, swap_qids: set, out: Path, log=print) -> dict:
    """U / U_steer on every refusal row in scope; A / A_steer too on swap-subset refusal rows."""
    layer = sw.STEER[model][0]
    vec = sw.steer_vector(model)
    wsha = sw.u_wording_sha256(model)
    seconds = {}
    rows = [r for r in records if label23(r) == "refusal"]
    for n, rec in enumerate(rows):
        kinds = sw.U_KINDS if rec.question_id in swap_qids else sw.U_KINDS[:2]
        path = out / "U" / f"{rec.question_id}.json"
        want = {"wording_sha256": wsha, "engine": engine.name, "model_id": MODELS[model]["id"],
                "prompt_sha256": pt.sha256_text(rec.prompt), "output_sha256": pt.sha256_text(rec.output),
                "kinds": list(kinds), "method_sha256": sw.method_sha256()}
        if read_row(path, want) is not None:
            continue
        t0 = time.perf_counter()
        gens = sw.u_generate(engine, rec, kinds, vec, layer)
        row = {"question_id": rec.question_id, "fold": rec.fold, "model": model, "label": "refusal",
               "in_swap_subset": rec.question_id in swap_qids, "gold": rec.gold, **gens, "stamp": want}
        write_json(path, row)
        seconds[rec.question_id] = time.perf_counter() - t0
        log(f"[U {n + 1}/{len(rows)}] {rec.question_id} " + ("no marker" if gens.get("no_marker") else " ".join(
            f"{k}={gens[k]['cls']}/{'G' if gens[k]['gold'] else '-'}:{gens[k]['phrase']!r:.30}" for k in kinds))
            + f" {seconds[rec.question_id]:.0f}s")
    return seconds


def check_inputs(model: str, records: dict, order: list[dict]) -> int:
    """No forward pass: every swap row's inputs are rebuilt and aligned with the tokenizer alone; token counts and
    the U rows' marker presence are printed (the basis of the time estimate)."""
    from transformers import AutoTokenizer

    cfg = MODELS[model]
    tok = AutoTokenizer.from_pretrained(cfg["id"])
    fake = type("E", (), {"tokenizer": tok, "name": "tokenizer-only",
                          "chat_text": staticmethod(lambda p: F.chat_text(tok, cfg["family"], p))})()
    run = SRun(fake, model, records, order, Path("/nonexistent"), "check")
    if cfg["family"] is None:
        run.chat_fn = None
    lens, bad = [], {}
    for i, e in enumerate(order):
        try:
            x = run.x(i)
            lens.append((len(x.clean_ids), len(x.swap_ids), len(x.cut_ids)))
        except (pt.AlignmentError, ValueError) as error:
            bad[e["question_id"]] = repr(error)
    from memctl.analysis import answer_selection as sel
    refusals = [r for r in records.values() if label23(r) == "refusal"]
    marked = sum(sel.split_note(r.output)[0] is not None for r in refusals)
    swapq = {e["question_id"] for e in order}
    print(json.dumps({"model": model, "n_swap_rows": len(order), "built": len(lens), "failed": bad,
                      "mean_tokens_clean_swap_cut": [sum(c) / len(lens) for c in zip(*lens)] if lens else None,
                      "max_tokens_clean": max((l[0] for l in lens), default=None),
                      "labels": {lab: sum(label23(records[e["question_id"]]) == lab for e in order)
                                 for lab in ("success", "refusal", "wrong")},
                      "u_refusal_rows": len(refusals), "u_marked": marked,
                      "u_swap_subset_refusals": sum(r.question_id in swapq for r in refusals)}), flush=True)
    return 0 if not bad else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=sorted(MODELS))
    p.add_argument("--folds", default="0,1")
    p.add_argument("--pooled", action="store_true", help="all five folds (refused until POOL_AUTHORISED)")
    p.add_argument("--limit", type=int, default=None, help="first N swap rows in (fold, id) order")
    p.add_argument("--smoke", action="store_true", help="condition S only (the 10-row S0 smoke)")
    p.add_argument("--skip-u", action="store_true")
    p.add_argument("--only-u", action="store_true")
    p.add_argument("--check-inputs", action="store_true",
                   help="tokenizer only (no model): rebuild every swap row's clean / swap / cut inputs and report")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)
    folds = list(range(5)) if args.pooled else [int(f) for f in args.folds.split(",")]
    if not args.check_inputs:              # the input check runs no model and reads no outcome: any fold
        check_folds(folds, args.pooled)
    out = Path(args.out or (OUT / ("smoke" if args.smoke else "") / args.model))
    population = [r for r in load_population(args.model) if r.fold in folds]
    records = {r.question_id: r for r in population}
    eligible, eligible_sha = load_eligible(args.model)
    order = [e for e in eligible if e["fold"] in folds]
    order = order[: args.limit] if args.limit else order
    if args.check_inputs:
        return check_inputs(args.model, records, order)
    from memctl.analysis.steering import HFEngine

    t0 = time.perf_counter()
    cfg = MODELS[args.model]
    engine = HFEngine(cfg["id"], threads=args.threads, family=cfg["family"], dtype="float32")
    load_s = time.perf_counter() - t0
    print(f"loaded {args.model} ({engine.name}) in {load_s:.0f}s; {len(order)} swap rows, folds {folds}", flush=True)
    started = time.perf_counter()
    summary = {"model": args.model, "engine": engine.name, "threads": args.threads, "folds": folds,
               "n_swap_rows": len(order), "load_s": load_s, "method_sha256": sw.method_sha256()}
    if not args.only_u:
        run = SRun(engine, args.model, records, order, out, eligible_sha, log=lambda s: print(s, flush=True))
        run.run()
        summary["s_seconds"] = run.seconds
    if not (args.smoke or args.skip_u):
        summary["u_seconds"] = u_rows(engine, args.model, population, {e["question_id"] for e in order}, out,
                                      log=lambda s: print(s, flush=True))
    summary.update({"wall_s": time.perf_counter() - started,
                    "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024})
    timing = out / "_timing.json"
    existing = json.loads(timing.read_text()) if timing.exists() else {"runs": []}
    existing["runs"].append(summary)
    write_json(timing, existing)
    print(json.dumps({k: v for k, v in summary.items() if not k.endswith("_seconds")}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
