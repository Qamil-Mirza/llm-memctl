#!/usr/bin/env python
"""§51 timed GPU measurement (proposed; needs its own quote and the user's approval line before any pod). Turns the
S + H + T-city quote from estimated into measured: per-row seconds on the pod for each pass type at batch 1 and at
fixed batch layouts, GPU memory, and agreement with the laptop's batch-1 B0 (a 64-row preview of the §10 CPU<->pod
replay check). E rows only (their laptop B0 records are shipped); nothing sealed is touched.

    python scripts/exp51_gpu_measure.py --inputs runs/exp51/gpu_measure/inputs.jsonl --out runs/exp51/gpu_measure/out \\
        [--backend hf|dummy] [--device cuda|cpu] [--snapshot DIR] [--n 64] [--batches 32,64] [--budget-s 900]

Writes <out>/measure.json after every step (a stopped pod keeps what it measured). Each step is skipped once the
wall-clock budget is spent.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import exp51_common as CM  # noqa: E402
import exp51_engine as EN  # noqa: E402


def gpu_mem_gb(e) -> float | None:
    if e.device == "cpu":
        return round(EN.peak_rss_gb(), 2)
    return round(e.torch.cuda.max_memory_allocated() / 2 ** 30, 2)


def make_inputs(n: int = 64, out: Path = ROOT / "runs/exp51/gpu_measure/inputs.jsonl") -> dict:
    """Laptop side: n mixed E rows with their batch-1 B0 records (prompt ids included), sorted by case id."""
    import exp51_run as RUN
    rws = RUN.mixed_pick(CM.rows("E"), CM.RUN, n)
    items = []
    for r in rws:
        rec = CM.b0("E", r["case_id"])
        items.append({"row": {k: r[k] for k in ("case_id", "base_key", "task", "set", "gold", "base_gold", "split",
                                                "prompt", "facts", "question")},
                      "b0": {k: rec[k] for k in ("prompt_ids", "gen_ids", "text", "cls", "j", "p_ans_ok", "p_ans",
                                                 "n_prompt", "first_trigger_step", "gen_s")}})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(x) + "\n" for x in items))
    return {"n": len(items), "path": str(out)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", default=str(ROOT / "runs/exp51/gpu_measure/inputs.jsonl"))
    p.add_argument("--out", required=True)
    p.add_argument("--backend", default="hf", choices=("hf", "dummy"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--snapshot", default=None)
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--batches", default="32,64")
    p.add_argument("--budget-s", type=float, default=900)
    p.add_argument("--make-inputs", action="store_true", help="laptop: write the inputs file and exit")
    a = p.parse_args(argv)
    if a.make_inputs:
        print(json.dumps(make_inputs(a.n, Path(a.inputs))))
        return 0
    t_start = time.time()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    items = [json.loads(l) for l in Path(a.inputs).read_text().splitlines() if l.strip()][:a.n]
    items.sort(key=lambda x: x["row"]["case_id"])
    res = {"n_rows": len(items), "steps": {}}

    def save():
        res["elapsed_s"] = round(time.time() - t_start, 1)
        (out / "measure.json").write_text(json.dumps(res, indent=1))

    def left() -> float:
        return a.budget_s - (time.time() - t_start)
    t0 = time.perf_counter()
    e = EN.load(a.backend, 8, edit_dir=out / "edits", device=a.device, snapshot=a.snapshot)
    res["engine"], res["load_s"], res["mem_after_load_gb"] = e.fingerprint(), round(time.perf_counter() - t0, 1), gpu_mem_gb(e)
    if a.device != "cpu":
        res["gpu"] = e.torch.cuda.get_device_name(0)
    save()
    prompts = [list(x["b0"]["prompt_ids"]) for x in items]
    laptop = [x["b0"]["gen_ids"] for x in items]

    def agree(gens):
        return {"token_equal_laptop": sum(g == l for g, l in zip(gens, laptop)),
                "class_equal_laptop": sum(CM.grade(x["row"], e.decode(g), [])["cls"] == x["b0"]["cls"]
                                          for g, x in zip(gens, items))}
    # 1. batch-1 total effect (16 rows): the replay preview and the batch-1 rate
    if left() > 120:
        k = min(16, len(items))
        t0 = time.perf_counter()
        g1 = [e.generate(prompts[i], mode="total").gen_ids for i in range(k)]
        wall = time.perf_counter() - t0
        res["steps"]["total_b1"] = {"rows": k, "s_per_row": round(wall / k, 3), **{
            kk: v for kk, v in zip(("token_equal_laptop", "class_equal_laptop"),
                                   (sum(a_ == b for a_, b in zip(g1, laptop[:k])),
                                    sum(CM.grade(x["row"], e.decode(g), [])["cls"] == x["b0"]["cls"]
                                        for g, x in zip(g1, items[:k]))))}, "mem_gb": gpu_mem_gb(e)}
        save()
    # 2. batched total effect at each fixed layout
    batched = {}
    for bs in [int(x) for x in a.batches.split(",") if x]:
        if left() < 120:
            res["steps"][f"total_b{bs}"] = {"skipped": "budget"}
            continue
        t0 = time.perf_counter()
        gens = []
        for i in range(0, len(items), bs):
            gens += [g.gen_ids for g in e.generate_batched(prompts[i:i + bs], mode="total")]
        wall = time.perf_counter() - t0
        batched[bs] = gens
        res["steps"][f"total_b{bs}"] = {"rows": len(gens), "s_per_row": round(wall / len(gens), 3), **agree(gens),
                                        "equal_gpu_b1_first16": sum(x == y for x, y in zip(gens, g1)) if "total_b1" in
                                        res["steps"] else None, "mem_gb": gpu_mem_gb(e)}
        save()
    # 3. controlled prefix and the B1 / B4 shortcut (resume), at the largest layout
    bs = max([int(x) for x in a.batches.split(",") if x] or [1])
    ok = [x for x in items if x["b0"]["p_ans_ok"]]
    if left() > 120 and ok:
        t0 = time.perf_counter()
        n = 0
        for i in range(0, len(ok), bs):
            chunk = ok[i:i + bs]
            n += len(e.generate_batched([CM.prefix_ids(x["b0"]) for x in chunk], mode="prefix"))
        res["steps"][f"prefix_b{bs}"] = {"rows": n, "s_per_row": round((time.perf_counter() - t0) / n, 3),
                                         "mem_gb": gpu_mem_gb(e)}
        save()
    trig = [x for x in items if x["b0"].get("first_trigger_step")]
    if left() > 120 and trig:
        b1 = CM.b1_token_set(e.tok) if a.backend == "hf" else [e.encode(" unknown")[0]]
        arm = EN.Arm(bias={"ids": b1, "value": -4.0, "mode": "online"})
        t0 = time.perf_counter()
        n = 0
        for i in range(0, len(trig), bs):
            chunk = trig[i:i + bs]
            t = [x["b0"]["first_trigger_step"] for x in chunk]
            n += len(e.generate_batched([list(x["b0"]["prompt_ids"]) + list(x["b0"]["gen_ids"][:k]) for x, k in
                                         zip(chunk, t)], arm, mode="resume",
                                        max_new=[EN.MAX_NEW_TOTAL - k for k in t],
                                        text_before=[e.decode(x["b0"]["gen_ids"][:k]) for x, k in zip(chunk, t)]))
        res["steps"][f"resume_b{bs}"] = {"rows": n, "s_per_row": round((time.perf_counter() - t0) / n, 3)}
        save()
    # 4. teacher-forced reads (batch 1): all 36 layers' residuals at P_ans
    if left() > 60 and ok:
        k = min(16, len(ok))
        t0 = time.perf_counter()
        for x in ok[:k]:
            ids = CM.prefix_ids(x["b0"])
            e.read(ids, [len(ids) - 1], layers=list(range(e.n_layers)), logit_positions=[len(ids) - 1])
        res["steps"]["read_b1"] = {"rows": k, "s_per_row": round((time.perf_counter() - t0) / k, 3),
                                   "mem_gb": gpu_mem_gb(e)}
        save()
    # the route gate fixed before measurement: the production layout (the largest batch) vs the laptop batch-1 B0
    if batched:
        bs_prod = max(batched)
        st = res["steps"][f"total_b{bs_prod}"]
        res["route_gate"] = {"rule": "graded-class agreement with CPU batch-1 >= 62/64 -> batched route usable; "
                                     "token mismatches disclosed (Advisor re-check, 2026-11-02)",
                             "layout": f"batch {bs_prod}, rows sorted by case id", "n": st["rows"],
                             "class_agree": st["class_equal_laptop"], "token_agree": st["token_equal_laptop"],
                             "threshold": 62 if st["rows"] == 64 else None,
                             "usable": bool(st["rows"] == 64 and st["class_equal_laptop"] >= 62)}
    res["done"] = True
    save()
    print(json.dumps({k: v for k, v in res.items() if k != "engine"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
