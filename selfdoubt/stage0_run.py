#!/usr/bin/env python
"""Stage 0 generation (selfdoubt/PROPOSAL.md, "Stage 0"). One model per invocation.

Per item: FULL (revised_problem, update-aware system prompt) -> cuts X in {0.3, 0.5, 0.9} of the reasoning trace
(the authors' token-level cut) with N / U0 / UG continuations -> X = 0 control (U0, UG right after "<think>").

    # local CPU smoke ($0): Qwen3-0.6B, 2 dev items, short decodes
    HF_HUB_OFFLINE=1 .venv/bin/python selfdoubt/stage0_run.py --engine hf --mode smoke --model Qwen/Qwen3-0.6B \
        --max-tokens 256
    # pod (see selfdoubt/stage0_pod.sh): vLLM, 10 dev items per model, writes runs/selfdoubt/stub/measure.json
    python selfdoubt/stage0_run.py --engine vllm --mode stub --model Qwen/Qwen3-8B --snapshot <path>

Records: runs/selfdoubt/<mode>/<model>/gen.jsonl keyed (model, item, arm, cut); a rerun skips done keys.
Phase timings: runs/selfdoubt/<mode>/<model>/phases.jsonl (one line per phase per invocation).
Exit 9 = stopped at --deadline-s between chunks (resumable).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from selfdoubt import common as C  # noqa: E402
from selfdoubt import engines as E  # noqa: E402


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


def run_phase(name, engine, store, reqs, max_tokens, chunk, deadline, out_dir, invocation):
    """Generate the pending requests in chunks; append records and one stats line per chunk (phases.jsonl), so a
    kill between or during chunks loses at most the chunk in flight and the timings of finished chunks survive."""
    pending = [r for r in reqs if not store.done(r.key)]
    log(f"phase {name}: {len(reqs)} requests, {len(reqs) - len(pending)} already done, {len(pending)} to run")
    stopped = False
    for i in range(0, len(pending), chunk):
        if deadline and time.time() > deadline:
            stopped = True
            log(f"deadline reached in phase {name}; {len(pending) - i} requests left (resumable)")
            break
        batch = pending[i:i + chunk]
        st = {"phase": name, "invocation": invocation, "chunk_index": i // chunk, "engine": engine.name,
              "model": engine.model_name, "n_requests": len(batch), "prompt_tokens": 0, "gen_tokens": 0,
              "n_hit_max": 0, "max_tokens": max_tokens, "chunk": chunk}
        with E.GpuMemPoller() as poll:
            tb = time.time()
            res = engine.generate(batch, max_tokens)
            st["wall_s"] = round(time.time() - tb, 2)
        recs = []
        for r, g in zip(batch, res):
            rec = {"key": r.key, "model": r.model, "revision": C.MODELS[r.model], "item_id": r.item_id,
                   "source": r.source, "arm": r.arm, "cut": r.cut, "seed": r.seed, "engine": engine.name,
                   "prompt_sha256": C.sha256_text(r.prompt), "text": g["text"],
                   "n_prompt_tokens": g["n_prompt_tokens"], "n_gen_tokens": g["n_gen_tokens"],
                   "finish_reason": g["finish_reason"], "hit_max": g["hit_max"], "max_tokens": g["max_tokens"],
                   "sampling": C.SAMPLING, "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **r.meta}
            if r.arm == "FULL":
                rec["prompt"] = r.prompt
                rec["total_reasoning_length"] = C.total_reasoning_length(engine.tok, r.model, g["text"])
            recs.append(rec)
            st["prompt_tokens"] += g["n_prompt_tokens"]
            st["gen_tokens"] += g["n_gen_tokens"]
            st["n_hit_max"] += int(g["hit_max"])
        store.append(recs)
        st["gen_tok_per_s"] = round(st["gen_tokens"] / st["wall_s"], 2) if st["wall_s"] else None
        st["nvidia_smi_max_used_mib"] = poll.max_mib
        st.update(engine.memory())
        with (out_dir / "phases.jsonl").open("a") as f:
            f.write(json.dumps(st) + "\n")
        log(f"  {name} chunk {i // chunk + 1}: {json.dumps(st)}")
    return stopped


def write_measure(mode_dir: Path, usd_per_hour: float, n_items_full: int, extra: dict | None = None) -> dict:
    """Aggregate every model dir under mode_dir into measure.json, with a projected cost for a full Stage 0."""
    out = {"usd_per_hour": usd_per_hour, "n_items_per_model_projected": n_items_full, "models": {},
           "provenance": C.provenance(), **(extra or {})}
    total_h = 0.0
    for md in sorted(p for p in mode_dir.iterdir() if p.is_dir() and (p / "gen.jsonl").exists()):
        store = C.Store(md / "gen.jsonl")
        recs = list(store.recs.values())
        phases = [json.loads(l) for l in (md / "phases.jsonl").read_text().splitlines()] \
            if (md / "phases.jsonl").exists() else []
        meta = json.loads((md / "run_meta.json").read_text()) if (md / "run_meta.json").exists() else {}
        items = sorted({r["item_id"] for r in recs})
        by_phase = {}
        for ph in ("FULL", "CONT"):
            rr = [r for r in recs if (r["arm"] == "FULL") == (ph == "FULL")]
            ps = [p for p in phases if p["phase"] == ph]
            wall = sum(p["wall_s"] for p in ps)
            gen = sum(p["gen_tokens"] for p in ps)
            by_phase[ph] = {
                "n_records": len(rr), "prompt_tokens": sum(r["n_prompt_tokens"] for r in rr),
                "gen_tokens": sum(r["n_gen_tokens"] for r in rr), "n_hit_max": sum(r["hit_max"] for r in rr),
                "max_gen_tokens": max((r["n_gen_tokens"] for r in rr), default=0),
                "wall_s_measured": round(wall, 1), "gen_tok_per_s_measured": round(gen / wall, 1) if wall else None,
                "nvidia_smi_max_used_mib": max((p.get("nvidia_smi_max_used_mib") or 0 for p in ps), default=None),
                "torch_max_memory_allocated_mib": max((p.get("torch_max_memory_allocated_mib") or 0 for p in ps),
                                                      default=None),
                "peak_rss_mib": max((p.get("peak_rss_mib") or 0 for p in ps), default=None)}
        n = max(len(items), 1)
        wall_items = sum(by_phase[p]["wall_s_measured"] for p in by_phase)
        load_s = meta.get("load_s")
        # Linear projection from the stub's own wall time per item. Conservative: the stub runs small batches, and a
        # full run batches 100s of requests, so its tokens/s should be higher. Model load is a one-off per model.
        proj_h = (wall_items / n * n_items_full + (load_s or 0)) / 3600.0
        total_h += proj_h
        out["models"][md.name] = {
            "n_items": len(items), "items": items, "n_records": len(recs), "phases": by_phase, "load_s": load_s,
            "gen_tokens_per_item": round(sum(by_phase[p]["gen_tokens"] for p in by_phase) / n, 1),
            "wall_s_per_item": round(wall_items / n, 2),
            "projected_hours_full": round(proj_h, 3), "projected_usd_full": round(proj_h * usd_per_hour, 3),
            "run_meta": meta}
    out["projected_hours_all_models"] = round(total_h, 3)
    out["projected_usd_all_models"] = round(total_h * usd_per_hour, 3)
    (mode_dir / "measure.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["hf", "vllm"], required=True)
    ap.add_argument("--mode", choices=["smoke", "stub", "full"], required=True)
    ap.add_argument("--model", required=True, choices=sorted(C.MODELS))
    ap.add_argument("--snapshot", default=None, help="local snapshot dir of the pinned revision (pod)")
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--max-model-len", type=int, default=40960)
    ap.add_argument("--chunk", type=int, default=256, help="requests per generate call (resume granularity)")
    ap.add_argument("--deadline-s", type=float, default=None, help="stop between chunks after this many seconds")
    ap.add_argument("--split-name", choices=["dev", "heldout"], default="dev")
    ap.add_argument("--allow-heldout", action="store_true", help="required to generate on held-out items")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--cut-unit", choices=["token", "char"], default="token", help="token = the authors' rule")
    ap.add_argument("--usd-per-hour", type=float, default=0.60)
    ap.add_argument("--n-items-full", type=int, default=1000, help="items per model for the cost projection")
    ap.add_argument("--out-root", default=str(C.RUNS))
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--device", default="cpu", help="hf engine only")
    a = ap.parse_args(argv)
    if a.split_name == "heldout" and not a.allow_heldout:
        raise SystemExit("held-out generation needs --allow-heldout (readings are taken on held-out only after the "
                         "dev choices are frozen)")
    if a.mode != "full" and a.split_name != "dev":
        raise SystemExit("smoke and stub use dev items only")

    t_start = time.time()
    deadline = t_start + a.deadline_s if a.deadline_s else None
    out_root = Path(a.out_root)
    items = C.load_items()
    split, split_sha = C.ensure_split(items, out_root / "split.json")
    log(f"split.json sha256 {split_sha}")
    sel = C.select_items(items, split, a.mode, a.split_name, a.limit)
    mode_dir = out_root / a.mode
    md = mode_dir / C.model_tag(a.model)
    md.mkdir(parents=True, exist_ok=True)
    store = C.Store(md / "gen.jsonl")
    log(f"{a.mode}: {len(sel)} items {[int(r['id']) for r in sel]}; store has {len(store.recs)} records")

    t0 = time.time()
    if a.engine == "hf":
        eng = E.HFEngine(a.model, a.snapshot, device=a.device)
    else:
        eng = E.VLLMEngine(a.model, a.snapshot, max_model_len=a.max_model_len,
                           gpu_memory_utilization=a.gpu_memory_utilization)
    load_s = round(time.time() - t0, 1)
    log(f"engine {a.engine} loaded {a.model} in {load_s} s")
    invocation = time.strftime("%Y%m%dT%H%M%S")
    meta_path = md / "run_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta.update({"model": a.model, "revision": C.MODELS[a.model], "engine": a.engine, "mode": a.mode,
                 "max_tokens": a.max_tokens, "max_model_len": a.max_model_len, "cut_unit": a.cut_unit,
                 "cuts": list(C.CUTS), "x0": C.X0, "sampling": C.SAMPLING, "split_sha256": split_sha,
                 "split_name": a.split_name, "item_ids": [int(r["id"]) for r in sel], "load_s": load_s,
                 "provenance": C.provenance(), "snapshot": a.snapshot})
    meta.setdefault("invocations", []).append(invocation)
    meta_path.write_text(json.dumps(meta, indent=1) + "\n")

    stopped = run_phase("FULL", eng, store, C.plan_full(a.model, eng.tok, sel), a.max_tokens, a.chunk, deadline,
                           md, invocation)
    if not stopped:
        conts = C.plan_continuations(a.model, eng.tok, sel, store.full_by_item(a.model), C.CUTS, a.cut_unit)
        stopped = run_phase("CONT", eng, store, conts, a.max_tokens, a.chunk, deadline, md, invocation)
    m = write_measure(mode_dir, a.usd_per_hour, a.n_items_full,
                      {"note": "projection = stub wall s/item x N + model load; conservative (small batches)"})
    log(f"measure.json written: projected ${m['projected_usd_all_models']} for {a.n_items_full} items/model "
        f"(models present: {list(m['models'])})")
    log(f"done in {time.time() - t_start:.1f} s; stopped_at_deadline={stopped}")
    return 9 if stopped else 0


if __name__ == "__main__":
    sys.exit(main())
