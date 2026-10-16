"""§37 laptop development run (CPU, transformers fp32, the 3B, folds 0-1 only).

    GUARD_VM_GB=22 runs/_pipelines/guard.sh .venv/bin/python scripts/exp37_dev.py [--out runs/exp37_dev]
    ... scripts/exp37_dev.py --run [--items V,CutB,...]     # only after the Advisor's build review

Pre-registration: research/proposals/2026-10-16-exp37-elenchus.md at 714da06 (rev 2), §5. Without --run the script
does exactly two things and stops:
1. the E3 calibration check (§2), before anything else: 20 development rows, identical pair -> A, another question's
   candidate -> C, pass >= 18/20 each, on the adopted E3 wording (amendment 1: the pre-fixed fallback; the primary
   failed 17/20, 18/20); a failure closes §37 on compliance (exit 6).
2. a 5-row timing: the first 5 development V rows through E1, E2 (blind), E3, E2 in dialogue and V-num, timed per
   turn (prefill and decode separately); it prints a projection for the whole laptop run from the measured per-token
   rates and every planned prompt's exact input length, writes runs/.../timing.json (timings only, no model output),
   and stops.
With --run it runs the items (resumable per question/item/turn, rows under <out>/qwen3b/<item>/<turn>.fold<k>.jsonl).
No AUROC or other correctness-linked number is computed here; the report is scripts/exp37_report.py.
Folds other than 0 and 1 are refused. HF_TOKEN is read from .env if the model is not cached, never printed.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from memctl.analysis import elenchus as el  # noqa: E402
from scripts import exp37_elenchus as gen  # noqa: E402

DEV = (0, 1)
LAPTOP_ITEMS = ("V", "CutB", "CutF", "Pnum", "Pname", "Vnum", "Vdlg")   # §5.4 plus Cut-F (amendment 1, 9639186)
TIMING_N = 5


def timing(src, engine, opts, log) -> dict:
    """Time E1, E2, E3, E2d, E3 (for d') and V-num on the first 5 development V rows; nothing is written."""
    tasks = [t for f in DEV for t in gen.build_tasks("V", f, src)][:TIMING_N]
    rec = {"E1": [], "E2": [], "E3": [], "E2d": [], "Vnum": []}
    for t in tasks:
        e1 = el.e1_messages(t.prompt, t.memory_lines, t.candidate, opts.e1_fallback)
        reply, st = engine.generate([e1], el.MAX_TOKENS["E1"])[0]
        rec["E1"].append(st)
        p = el.parse_e1(reply, len(t.memory_lines))
        cited = p["G"] or [1]   # timing only: a citation-less row is timed on item 1
        d, st = engine.generate([el.e2_blind_messages(t.prompt, t.memory_lines, cited)], el.MAX_TOKENS["E2"])[0]
        rec["E2"].append(st)
        _, st = engine.top20([el.e3_messages(t.prompt, t.question_id, t.candidate, el.first_line(d) or "x")])[0]
        rec["E3"].append(st)
        _, st = engine.generate([el.e2_dialogue_messages(e1, reply, cited)], el.MAX_TOKENS["E2d"])[0]
        rec["E2d"].append(st)
        _, st = engine.top20([el.vnum_messages(t.prompt, t.memory_lines, t.candidate)])[0]
        rec["Vnum"].append(st)
    prefill = [(s["prefill_s"], s["n_input"]) for v in rec.values() for s in v]
    decode = [(s["decode_s"], s["n_new"]) for k in ("E1", "E2", "E2d") for s in rec[k] if s["n_new"] > 1]
    rate_p = sum(a for a, _ in prefill) / sum(b for _, b in prefill)
    rate_d = sum(a for a, _ in decode) / max(sum(b - 1 for _, b in decode), 1)
    per_turn = {k: {"mean_s": statistics.mean(s["prefill_s"] + s.get("decode_s", 0) for s in v),
                    "mean_input_tokens": statistics.mean(s["n_input"] for s in v),
                    "mean_new_tokens": statistics.mean(s.get("n_new", 1) for s in v)} for k, v in rec.items()}
    return {"rows": len(tasks), "prefill_s_per_token": rate_p, "decode_s_per_token": rate_d, "per_turn": per_turn}


def projection(src, engine, t: dict) -> dict:
    """Hours per item on the laptop: exact E1 / V-num input lengths per planned row; E2, E3 and E2d at the timed means."""
    rp, rd, pt = t["prefill_s_per_token"], t["decode_s_per_token"], t["per_turn"]
    out = {}
    for item in LAPTOP_ITEMS:
        tasks = [x for f in DEV for x in gen.build_tasks(item, f, src)]
        if item == "Vnum":
            secs = sum(engine.n_tokens(engine.chat_text(el.vnum_messages(x.prompt, x.memory_lines, x.candidate))) * rp
                       for x in tasks)
        elif item == "Vdlg":
            secs = len(tasks) * (pt["E2d"]["mean_s"] + pt["E3"]["mean_s"])
        else:
            e1_tokens = sum(engine.n_tokens(engine.chat_text(el.e1_messages(x.prompt, x.memory_lines, x.candidate)))
                            for x in tasks)
            secs = (e1_tokens * rp + len(tasks) * pt["E1"]["mean_new_tokens"] * rd
                    + len(tasks) * (pt["E2"]["mean_s"] + pt["E3"]["mean_s"]))
        out[item] = {"rows": len(tasks), "hours": secs / 3600}
    out["total_laptop_hours"] = sum(v["hours"] for k, v in out.items() if k in LAPTOP_ITEMS)
    out["over_8h"] = out["total_laptop_hours"] > 8
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", default="0,1")
    parser.add_argument("--out", default="runs/exp37_dev")
    parser.add_argument("--exp35", default="runs/exp35")
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--run", action="store_true", help="run the items (only after the Advisor's build review)")
    parser.add_argument("--items", default=",".join(LAPTOP_ITEMS))
    parser.add_argument("--backend", default="hf", choices=("hf", "stub"))
    args = parser.parse_args(argv)
    folds = [int(f) for f in args.folds.split(",")]
    if not set(folds) <= set(DEV):
        print(f"REFUSED: folds {sorted(set(folds) - set(DEV))} are not development folds; the laptop run is folds 0-1 only")
        return 2
    out = Path(args.out)
    log = lambda s: print(s, flush=True)  # noqa: E731
    src = gen.Sources(folds, args.runs, Path(args.exp35))
    t0 = time.perf_counter()
    engine = gen.build_engine(args.backend, gen.MODELS["qwen3b"], args.threads)
    log(f"engine {engine.name}, model loaded in {time.perf_counter() - t0:.0f} s")
    # Amendment 1 (9639186): the primary E3 sentence failed calibration (17/20, 18/20); the pre-fixed fallback passed
    # (18/20, 19/20) and is the E3 wording for every call. The check is repeated on it at the start of every run; no
    # other E3 wording may be tried.
    opts = gen.Opts(e3_fallback=True)
    verdict = gen.calibrate("qwen3b", src, engine, out, opts, log)
    if not verdict["pass"]:
        log("E3 calibration failed with the adopted (fallback) wording: §37 closes on compliance (exit 6)")
        return gen.CALIBRATION_FAILED
    if not args.run:
        t = timing(src, engine, opts, log)
        proj = projection(src, engine, t)
        record = {"engine": engine.name, "threads": args.threads, "e3_fallback": opts.e3_fallback, "timing": t,
                  "projection": proj}
        (out / "qwen3b" / "timing.json").write_text(json.dumps(record, indent=1))
        log(json.dumps(record, indent=1))
        log(f"projection: {proj['total_laptop_hours']:.1f} h for {', '.join(LAPTOP_ITEMS)}; over 8 h: "
            f"{proj['over_8h']}. Stopping (no --run).")
        return 0
    items = [x for x in args.items.split(",") if x]
    for item in items:
        gen.run_item(item, "qwen3b", folds, src, engine, out, opts, log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
