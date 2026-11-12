#!/usr/bin/env python
"""Stage 1 generation (selfdoubt/stage1.py). Same CLI shape as stage0_run.py, so stage0_full_pod.sh drives it with
RUNNER=selfdoubt/stage1_run.py and EXTRA="--cut X --payload P".

FULL traces: read from <out-root>/full/<Model>/gen.jsonl (Stage 0's store, shipped in the bundle); items without one
get a FULL generated there first, exactly as Stage 0 would (same prompt, same seed). Stage 1 continuations go to
<out-root>/stage1/<Model>/gen.jsonl (keys "S1|model|item|arm|cut"; a rerun skips done keys).

    # local ($0): fake engine or Qwen3-0.6B on CPU, dev items
    .venv/bin/python selfdoubt/stage1_run.py --engine hf --mode full --split-name dev --limit 2 --source gsm8k \
        --model Qwen/Qwen3-0.6B --max-tokens 256 --out-root runs/selfdoubt/s1_local/dev --cut 0.9
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
from selfdoubt import stage1 as S1  # noqa: E402
from selfdoubt.stage0_run import log, run_phase  # noqa: E402


def main(argv=None, engine=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["hf", "vllm", "fake"], required=True)
    ap.add_argument("--mode", choices=["full"], default="full")
    ap.add_argument("--model", required=True, choices=sorted(C.MODELS))
    ap.add_argument("--snapshot", default=None)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--max-model-len", type=int, default=40960)
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--deadline-s", type=float, default=None)
    ap.add_argument("--full-frac", type=float, default=None, help="accepted for CLI parity; FULL is rarely needed")
    ap.add_argument("--split-name", choices=["dev", "heldout"], default="dev")
    ap.add_argument("--allow-heldout", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--source", choices=list(C.SOURCES), default=None)
    ap.add_argument("--usd-per-hour", type=float, default=0.59)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--max-num-seqs", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--cut", type=float, required=True, help="dev-chosen cut (frozen before held-out)")
    ap.add_argument("--payload", default="plain,doc",
                    help="comma list; plain = PRIMARY (carries the reading), doc = SECONDARY (descriptive)")
    ap.add_argument("--arms", default=",".join(S1.ARMS))
    ap.add_argument("--sys", default="s1", help="comma list of system-prompt settings (s1, s1t, s0); one store each, "
                    "<out-root>/stage1[_<sys>]/<Model>/gen.jsonl")
    a = ap.parse_args(argv)
    if a.split_name == "heldout" and not a.allow_heldout:
        raise SystemExit("held-out generation needs --allow-heldout (only after the dev choices are frozen)")
    arms = tuple(x for x in a.arms.split(",") if x)
    payloads = tuple(x for x in a.payload.split(",") if x)
    assert payloads and set(payloads) <= set(S1.PAYLOADS), payloads
    assert set(arms) <= set(S1.ARMS), arms
    systems = tuple(x for x in a.sys.split(",") if x)
    assert systems and set(systems) <= set(S1.SYSTEMS) and len(set(systems)) == len(systems), systems
    t0 = time.time()
    deadline = t0 + a.deadline_s if a.deadline_s else None
    out_root = Path(a.out_root)
    items = C.load_items()
    split, sha = C.ensure_split(items, out_root / "split.json")
    sel = C.select_items(items, split, "full", a.split_name, a.limit, sources=(a.source,) if a.source else C.SOURCES)
    tag = C.model_tag(a.model)
    full_store = C.Store(out_root / "full" / tag / "gen.jsonl")
    log(f"stage1: {len(sel)} {a.split_name} items, cut {a.cut}, payloads {payloads}, arms {arms}, systems {systems}; "
        f"FULL records {len(full_store.full_by_item(a.model))}")
    if engine is None:
        if a.engine == "hf":
            engine = E.HFEngine(a.model, a.snapshot, device=a.device)
        else:
            engine = E.VLLMEngine(a.model, a.snapshot, max_model_len=a.max_model_len,
                                  gpu_memory_utilization=a.gpu_memory_utilization, max_num_seqs=a.max_num_seqs)
    inv = time.strftime("%Y%m%dT%H%M%S")
    need_full = [r for r in sel if int(r["id"]) not in full_store.full_by_item(a.model)]
    stopped = False
    if need_full:
        stopped = run_phase("FULL", engine, full_store, C.plan_full(a.model, engine.tok, need_full), a.max_tokens,
                            a.chunk, deadline, out_root / "full" / tag, inv)
    for sy in systems:                 # one store per system-prompt setting, in the order given
        md = out_root / S1.store_dir(sy) / tag
        md.mkdir(parents=True, exist_ok=True)
        store = C.Store(md / "gen.jsonl")
        meta = {"model": a.model, "revision": C.MODELS[a.model], "split_name": a.split_name, "split_sha256": sha,
                "cut": a.cut, "payloads": list(payloads), "arms": list(arms), "sys": sy,
                "system_prompt": S1.system_prompt_s1(sy), "item_ids": [int(r["id"]) for r in sel],
                "max_tokens": a.max_tokens, "sampling": C.SAMPLING, "provenance": C.provenance()}
        (md / "run_meta.json").write_text(json.dumps(meta, indent=1) + "\n")
        if deadline and time.time() > deadline:
            stopped = True
            break
        reqs, seen = [], set()
        for pl in payloads:            # V arms and N are payload-free: planned once
            for r in S1.plan(a.model, engine.tok, sel, full_store.full_by_item(a.model), a.cut, pl, arms, sy):
                if r.key not in seen:
                    seen.add(r.key)
                    reqs.append(r)
        log(f"stage1 {sy}: {len(reqs)} requests; store {md / 'gen.jsonl'} has {len(store.recs)}")
        stopped = run_phase(f"S1{'' if sy == 's1' else '_' + sy}", engine, store, reqs, a.max_tokens, a.chunk,
                            deadline, md, inv) or stopped
    log(f"stage1 done in {time.time() - t0:.1f} s; stopped_at_deadline={stopped}")
    return 9 if stopped else 0


if __name__ == "__main__":
    sys.exit(main())
