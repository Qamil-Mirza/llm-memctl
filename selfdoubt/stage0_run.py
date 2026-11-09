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
Exit 9 = stopped at --deadline-s (vLLM: mid-chunk, finished requests kept; resumable).
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
    """Generate the pending requests in chunks; append records and one stats line per chunk (phases.jsonl). The
    deadline is checked between chunks and passed to the engine, which (vLLM) stops mid-chunk and returns finished
    requests in full and unfinished ones as partial token counts: those are counted in the chunk's stats (throughput)
    but not stored, so a rerun regenerates them. A hard kill still loses the chunk in flight."""
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
              "n_hit_max": 0, "n_unfinished": 0, "gen_tokens_partial": 0, "max_tokens": max_tokens, "chunk": chunk}
        with E.GpuMemPoller() as poll:
            tb = time.time()
            res = engine.generate(batch, max_tokens, deadline=deadline)
            st["wall_s"] = round(time.time() - tb, 2)
        recs = []
        for r, g in zip(batch, res):
            if g.get("unfinished"):
                st["n_unfinished"] += 1
                st["gen_tokens_partial"] += g["n_gen_tokens"]
                continue
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
        all_tok = st["gen_tokens"] + st["gen_tokens_partial"]
        st["gen_tok_per_s"] = round(all_tok / st["wall_s"], 2) if st["wall_s"] else None
        st["nvidia_smi_max_used_mib"] = poll.max_mib
        st.update(engine.memory())
        st["trace"] = getattr(engine, "last_trace", None)
        with (out_dir / "phases.jsonl").open("a") as f:
            f.write(json.dumps(st) + "\n")
        log(f"  {name} chunk {i // chunk + 1}: {json.dumps({k: v for k, v in st.items() if k != 'trace'})}")
        if st["n_unfinished"]:
            stopped = True
            log(f"deadline reached inside phase {name}: {st['n_unfinished']} requests unfinished (resumable)")
            break
    return stopped


N_CONT = len(C.CUTS) * 3 + 2   # N/U0/UG at each cut + U0/UG at X = 0


def busy_rate(traces: list, n_requests: list) -> float | None:
    """Generated tokens/s over the trace windows in which at least half of the chunk's requests were still running:
    the steady rate a large run sees, without the tail where few long sequences decode alone."""
    tok = dt = 0.0
    for tr, n in zip(traces, n_requests):
        for (t0, u0, c0), (t1, _u1, c1) in zip(tr or [], (tr or [])[1:]):
            if u0 >= max(1, n / 2) and t1 > t0:
                tok += c1 - c0
                dt += t1 - t0
    return round(tok / dt, 1) if dt else None


def write_measure(mode_dir: Path, usd_per_hour: float, n_items_full: int, extra: dict | None = None,
                  source_counts: dict | None = None) -> dict:
    """Aggregate every model dir under mode_dir into measure.json, with projected costs for a full Stage 0.

    Two projections per model. "token" (the one to quote): generated tokens per item, per source, times the item
    count of that source, divided by the CONT phase's busy rate (tokens/s while >= half its requests ran), plus the
    model load. Its "_low" uses finished requests only; "_high" counts every request stopped by the deadline as
    reaching max_tokens. Prefill time is not modelled (prefix caching shares the FULL prompt + cut). "wall_linear"
    (stub wall s/item x N) is kept only for comparison: a 10-item batch is dominated by its longest sequence, so it
    overstates a full run."""
    counts = source_counts or {"all": n_items_full}
    out = {"usd_per_hour": usd_per_hour, "source_counts_projected": counts, "requests_per_item": 1 + N_CONT,
           "models": {}, "provenance": C.provenance(), **(extra or {})}
    tot = {"wall_linear": 0.0, "token_low": 0.0, "token_high": 0.0}
    for md in sorted(p for p in mode_dir.iterdir() if p.is_dir() and (p / "gen.jsonl").exists()):
        store = C.Store(md / "gen.jsonl")
        recs = list(store.recs.values())
        phases = [json.loads(l) for l in (md / "phases.jsonl").read_text().splitlines()] \
            if (md / "phases.jsonl").exists() else []
        meta = json.loads((md / "run_meta.json").read_text()) if (md / "run_meta.json").exists() else {}
        items = sorted({r["item_id"] for r in recs})
        n_full = sum(r["arm"] == "FULL" for r in recs)
        # requests still missing now (a resume may have filled earlier deadline stops): planned FULL items without a
        # record, and the 11 continuations of every FULL item that lack a record
        missing = {"FULL": max(0, len(meta.get("item_ids") or []) - n_full),
                   "CONT": max(0, N_CONT * n_full - (len(recs) - n_full))}
        by_phase = {}
        for ph in ("FULL", "CONT"):
            rr = [r for r in recs if (r["arm"] == "FULL") == (ph == "FULL")]
            ps = [p for p in phases if p["phase"] == ph]
            wall = sum(p["wall_s"] for p in ps)
            gen_all = sum(p["gen_tokens"] + p.get("gen_tokens_partial", 0) for p in ps)
            n_unf = missing[ph]
            mt = max((r["max_tokens"] for r in rr), default=None) or (ps[-1]["max_tokens"] if ps else None)
            fin_tok = sum(r["n_gen_tokens"] for r in rr)
            by_src = {}
            for src in sorted({r["source"] for r in rr}):
                sr = [r for r in rr if r["source"] == src]
                by_src[src] = {"n_records": len(sr), "n_items": len({r["item_id"] for r in sr}),
                               "gen_tokens_per_request": round(sum(r["n_gen_tokens"] for r in sr) / len(sr), 1),
                               "n_hit_max": sum(r["hit_max"] for r in sr),
                               "max_gen_tokens": max(r["n_gen_tokens"] for r in sr)}
            by_phase[ph] = {
                "n_records": len(rr), "n_missing_now": n_unf,
                "n_unfinished_at_deadline_all_invocations": sum(p.get("n_unfinished", 0) for p in ps),
                "gen_tokens_partial_unfinished": sum(p.get("gen_tokens_partial", 0) for p in ps),
                "prompt_tokens": sum(r["n_prompt_tokens"] for r in rr), "gen_tokens": fin_tok,
                "n_hit_max": sum(r["hit_max"] for r in rr), "max_tokens": mt,
                "max_gen_tokens": max((r["n_gen_tokens"] for r in rr), default=0),
                "gen_tokens_per_request_low": round(fin_tok / len(rr), 1) if rr else None,
                "gen_tokens_per_request_high": round((fin_tok + n_unf * (mt or 0)) / (len(rr) + n_unf), 1)
                if (rr or n_unf) else None,
                "by_source": by_src,
                "wall_s_measured": round(wall, 1),
                "gen_tok_per_s_measured": round(gen_all / wall, 1) if wall else None,
                "gen_tok_per_s_busy": busy_rate([p.get("trace") for p in ps], [p["n_requests"] for p in ps]),
                "nvidia_smi_max_used_mib": max((p.get("nvidia_smi_max_used_mib") or 0 for p in ps), default=None),
                "torch_max_memory_allocated_mib": max((p.get("torch_max_memory_allocated_mib") or 0 for p in ps),
                                                      default=None),
                "torch_max_memory_reserved_mib": max((p.get("torch_max_memory_reserved_mib") or 0 for p in ps),
                                                     default=None),
                "peak_rss_mib": max((p.get("peak_rss_mib") or 0 for p in ps), default=None)}
        n = max(len(items), 1)
        wall_items = sum(by_phase[p]["wall_s_measured"] for p in by_phase)
        load_s = meta.get("load_s") or 0
        n_all = sum(counts.values())
        proj = {"wall_linear_hours": (wall_items / n * n_all + load_s) / 3600.0}
        rate = by_phase["CONT"]["gen_tok_per_s_busy"] or by_phase["CONT"]["gen_tok_per_s_measured"] \
            or by_phase["FULL"]["gen_tok_per_s_measured"]
        per_src = {}
        if rate:
            f, c = by_phase["FULL"], by_phase["CONT"]
            def up(ph):   # phase-wide uplift of the high bound (missing requests counted at max_tokens)
                return (ph["gen_tokens_per_request_high"] or 0) - (ph["gen_tokens_per_request_low"] or 0)
            for bound in ("low", "high"):
                toks = 0.0
                for src, k in counts.items():
                    # the source's own mean per request when measured, else the phase mean; high adds the uplift
                    fs = f["by_source"].get(src, {}).get("gen_tokens_per_request", f["gen_tokens_per_request_low"])
                    cs = c["by_source"].get(src, {}).get("gen_tokens_per_request", c["gen_tokens_per_request_low"])
                    fs, cs = fs or 0, cs or 0
                    if bound == "high":
                        fs, cs = fs + up(f), cs + up(c)
                    item_tok = fs + N_CONT * cs
                    per_src.setdefault(src, {})[f"gen_tokens_per_item_{bound}"] = round(item_tok, 1)
                    per_src[src][f"hours_{bound}"] = round(k * item_tok / rate / 3600.0, 3)
                    per_src[src][f"usd_{bound}"] = round(k * item_tok / rate / 3600.0 * usd_per_hour, 3)
                    toks += k * item_tok
                proj[f"token_{bound}_hours"] = (toks / rate + load_s) / 3600.0
        for k2 in ("wall_linear", "token_low", "token_high"):
            if f"{k2}_hours" in proj:
                tot[k2] += proj[f"{k2}_hours"]
        out["models"][md.name] = {
            "n_items": len(items), "items": items, "n_records": len(recs), "phases": by_phase, "load_s": load_s,
            "gen_tokens_per_item_observed": round(sum(by_phase[p]["gen_tokens"] for p in by_phase) / n, 1),
            "wall_s_per_item": round(wall_items / n, 2), "rate_used_tok_per_s": rate,
            "projection_by_source": per_src,
            **{f"projected_{k}": round(v, 3) for k, v in proj.items()},
            **{f"projected_{k[:-6]}_usd": round(v * usd_per_hour, 3) for k, v in proj.items()},
            "run_meta": meta}
    for k2, v in tot.items():
        out[f"projected_{k2}_hours_all_models"] = round(v, 3)
        out[f"projected_{k2}_usd_all_models"] = round(v * usd_per_hour, 3)
    # back-compat key (tests and older readers): the token-based high bound
    out["projected_usd_all_models"] = out["projected_token_high_usd_all_models"]
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
    ap.add_argument("--deadline-s", type=float, default=None, help="graceful stop after this many seconds")
    ap.add_argument("--full-frac", type=float, default=None,
                    help="share of the post-load time given to the FULL phase (rest: CONT); default: no split")
    ap.add_argument("--split-name", choices=["dev", "heldout"], default="dev")
    ap.add_argument("--allow-heldout", action="store_true", help="required to generate on held-out items")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--cut-unit", choices=["token", "char"], default="token", help="token = the authors' rule")
    ap.add_argument("--usd-per-hour", type=float, default=0.60)
    ap.add_argument("--n-items-full", type=int, default=1000, help="items per model for the cost projection")
    ap.add_argument("--out-root", default=str(C.RUNS))
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--max-num-seqs", type=int, default=None, help="vLLM max_num_seqs (default: vLLM's own, 256)")
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
                           gpu_memory_utilization=a.gpu_memory_utilization, max_num_seqs=a.max_num_seqs)
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

    # --full-frac: the FULL phase gets its own earlier deadline, so the CONT phase (11 of every 12 decodes, the cost
    # driver) is always measured; FULL items stopped there get no continuations (a rerun regenerates them).
    t_gen = time.time()
    full_deadline = t_gen + a.full_frac * (deadline - t_gen) if (deadline and a.full_frac) else deadline
    stopped_full = run_phase("FULL", eng, store, C.plan_full(a.model, eng.tok, sel), a.max_tokens, a.chunk,
                             full_deadline, md, invocation)
    stopped_cont = True
    if not (deadline and time.time() > deadline):
        conts = C.plan_continuations(a.model, eng.tok, sel, store.full_by_item(a.model), C.CUTS, a.cut_unit)
        stopped_cont = run_phase("CONT", eng, store, conts, a.max_tokens, a.chunk, deadline, md, invocation)
    stopped = stopped_full or stopped_cont
    counts = {src: sum(1 for r in items if r["source"] == src) for src in C.SOURCES}
    m = write_measure(mode_dir, a.usd_per_hour, a.n_items_full, {"note": "see write_measure docstring"},
                      source_counts=counts)
    log(f"measure.json written: projected (token-based, all {sum(counts.values())} rows per model) "
        f"${m['projected_token_low_usd_all_models']}-${m['projected_token_high_usd_all_models']} "
        f"(models present: {list(m['models'])})")
    log(f"done in {time.time() - t_start:.1f} s; stopped_at_deadline={stopped}")
    return 9 if stopped else 0


if __name__ == "__main__":
    sys.exit(main())
