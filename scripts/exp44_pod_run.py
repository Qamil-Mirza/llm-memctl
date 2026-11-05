#!/usr/bin/env python
"""§44 rev 8 pod helpers (scripts/exp44_pod.sh): stub models, the population check, smoke checks, stage times, status.

    python scripts/exp44_pod_run.py make-stub --out runs/exp44_pod/stub_models      # tiny random Qwen2, real tokenizer
    python scripts/exp44_pod_run.py population [--expect-fingerprint FP]           # counts + fingerprint, no outcome
    python scripts/exp44_pod_run.py smoke-check --kind patch --dir D --n 3 --engine cuda-bfloat16 --n-layers 28
    python scripts/exp44_pod_run.py stage-time --file T --stage S --status ok --rc 0 --elapsed 12 --t-offset 40 ...
    python scripts/exp44_pod_run.py status --out runs/exp44_pod

Spec: research/proposals/2026-10-29-exp44-patching.md "Rev 8" (83fe60a, D-47): (A) the rev 0 7B confirmation, all
204 rows of folds 2-4, bf16 CUDA -> runs/exp44_patch/qwen7b_gpu/; (B) rev 8 S on Qwen 3B, fp32 CUDA ->
runs/exp44_swap/qwen3b_rev8/S/. Nothing here reads or prints an outcome (labels, lp, d_c, follows): the smoke check
asserts finiteness, counts, the construction bound and the engine, and prints seconds per row.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

TOKENIZER_ID = "Qwen/Qwen2.5-3B-Instruct"     # the 3B and 7B share the Qwen2.5 tokenizer (same vocabulary and ids)
STUB_LAYERS = {"qwen3b": 4, "qwen7b": 3}
N_CONFIRM_ROWS, N_POPULATION, N_REV8 = 204, 330, 70


def make_stub(out: Path, tokenizer_id: str = TOKENIZER_ID) -> None:
    import torch
    import transformers

    tok = transformers.AutoTokenizer.from_pretrained(tokenizer_id)
    for name, layers in STUB_LAYERS.items():
        cfg = transformers.Qwen2Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128,
                                       num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=2,
                                       max_position_embeddings=8192, tie_word_embeddings=False)
        torch.manual_seed(44)
        net = transformers.Qwen2ForCausalLM(cfg).eval()
        d = out / name
        net.save_pretrained(d)
        tok.save_pretrained(d)
        print(f"stub {name}: {layers} layers, width 64, vocab {len(tok)} -> {d}", flush=True)


def fingerprint_parts(runs_dir: str = "runs") -> dict:
    import exp44_patch as P
    import exp44_swap_gate as G

    conf = P.population("qwen7b", (2, 3, 4), runs_dir)
    pop = G.qwen_population(runs_dir)
    el = Path("runs/exp44_swapgate/rev8/eligible_qwen3b.jsonl")
    gate = json.loads(Path("runs/exp44_swapgate/rev8/gate.json").read_text())
    rows = [json.loads(l) for l in el.read_text().splitlines() if l.strip()]
    h = hashlib.sha256()
    for r in conf + pop:
        h.update(f"{r.question_id}|{r.fold}|{hashlib.sha256(r.prompt.encode()).hexdigest()}\n".encode())
    h.update(hashlib.sha256(el.read_bytes()).hexdigest().encode())
    return {"n_confirm_rows_7b": len(conf), "per_fold_7b": {f: sum(r.fold == f for r in conf) for f in (2, 3, 4)},
            "n_population_3b": len(pop), "n_rev8_rows": len(rows), "rev8_gate_rev": gate.get("rev"),
            "fingerprint": h.hexdigest()}


def population(expect: str | None) -> int:
    import exp44_swap as S

    info = fingerprint_parts()
    ok = (info["n_confirm_rows_7b"] == N_CONFIRM_ROWS and info["n_population_3b"] == N_POPULATION
          and info["n_rev8_rows"] == N_REV8 and info["rev8_gate_rev"] == 8)
    # every rev 8 swap prompt re-derived from the gate's own functions and checked against its sha (no tokenizer)
    records = {r.question_id: r for r in S.load_population("qwen3b")}
    order, _ = S.load_eligible("qwen3b", gate_rev=8)
    for e in order:
        S.rebuild_swap(records[e["question_id"]], e)
    info["rev8_swap_prompts_checked"] = len(order)
    if expect and info["fingerprint"] != expect:
        print(f"population: fingerprint {info['fingerprint'][:16]} != bundle's {expect[:16]}", file=sys.stderr)
        ok = False
    print("population", json.dumps(info), flush=True)
    return 0 if ok else 3


def _finite(*xs) -> bool:
    return all(x is not None and math.isfinite(x) for x in xs)


def smoke_check(kind: str, d: Path, n: int, engine: str, n_layers: int) -> int:
    rows = [json.loads(p.read_text()) for p in sorted((d / "S" if kind == "swap" else d).glob("*.json"))
            if not p.name.startswith("_")]
    problems = []
    if len(rows) != n:
        problems.append(f"{len(rows)} rows written, want {n}")
    for r in rows:
        q = r["question_id"]
        if engine not in r["stamp"]["engine"]:
            problems.append(f"{q}: engine {r['stamp']['engine']} lacks {engine}")
        if r["n_layers"] != n_layers:
            problems.append(f"{q}: {r['n_layers']} layers, want {n_layers}")
        if kind == "patch":
            if not _finite(r["lp_clean"], r["lp_corrupt"], r["lp_unknown_clean"], r["lp_unknown_corrupt"]):
                problems.append(f"{q}: non-finite log prob")
            if r["patched"] and (len(r["R"]) != n_layers or not all(_finite(v) for v in r["lp_patched"])):
                problems.append(f"{q}: patched row has {len(r['R'])} R values or a non-finite patched lp")
            if r["patched"] and r["R"][-1] is not None and abs(r["R"][-1] - 1) >= 0.01:
                problems.append(f"{q}: |R(L-1) - 1| >= 0.01 (construction)")
        else:
            if not _finite(r["clean"]["delta"], r["swap"]["delta"], r["cut"]["delta"]):
                problems.append(f"{q}: non-finite Delta")
            bound = lambda t: t is not None and abs(t - 1) < 0.01  # noqa: E731
            if r["patched"] and (r["T_fwd"] is None or r["T_rev"] is None or len(r["T_fwd"]) != n_layers
                                 or not bound(r["T_fwd"][-1])
                                 or not bound(r["T_rev"][-1]) or not r["control_self"]["pass"]):
                problems.append(f"{q}: patched row fails the construction bound or the self control")
    timing = json.loads((d / "_timing.json").read_text())["runs"][-1] if (d / "_timing.json").exists() else {}
    if kind == "patch":
        secs = {r["question_id"]: r["timing"]["total_s"] for r in rows}
        passes = {r["question_id"]: r["timing"]["n_patch_passes"] for r in rows}
    else:
        secs = timing.get("s_seconds", {})
        passes = {r["question_id"]: r["timing"].get("n_patch_passes", 0) for r in rows}
    per_row = sum(secs.values()) / len(secs) if secs else None
    out = {"kind": kind, "dir": str(d), "n_rows": len(rows), "seconds_per_row": per_row, "seconds": secs,
           "patch_passes": passes, "load_s": timing.get("load_s"), "peak_rss_mb": timing.get("peak_rss_mb"),
           "problems": problems, "pass": not problems}
    (d / "_smoke.json").write_text(json.dumps(out, indent=1))
    print(f"smoke {kind}: {len(rows)} rows, measured {per_row if per_row is None else round(per_row, 2)} s/row "
          f"(passes {list(passes.values())}), load {timing.get('load_s')}, "
          f"{'PASS' if not problems else 'FAIL ' + '; '.join(problems)}", flush=True)
    return 0 if not problems else 5


def stage_time(a) -> int:
    path = Path(a.file)
    t = json.loads(path.read_text()) if path.exists() else {"stages": [], "hard_stop_s": a.hard_stop_s,
                                                            "self_stop_s": a.self_stop_s, "engine": a.engine}
    t["stages"].append({"stage": a.stage, "status": a.status, "rc": a.rc, "elapsed_s": a.elapsed,
                        "t_offset_s": a.t_offset, "note": a.note})
    path.write_text(json.dumps(t, indent=1))
    return 0


def status(out: Path, patch_dir: Path, swap_dir: Path) -> int:
    count = lambda d: sum(1 for p in d.glob("*.json") if not p.name.startswith("_")) if d.exists() else 0  # noqa: E731
    s = {"patch_rows": count(patch_dir), "patch_failed": json.loads((patch_dir / "_failed.json").read_text())
         if (patch_dir / "_failed.json").exists() else {}, "swap_S_rows": count(swap_dir / "S"),
         "swap_resid": sum(1 for _ in (swap_dir / "resid").glob("*.npz")) if (swap_dir / "resid").exists() else 0}
    (out / "status.json").write_text(json.dumps(s, indent=1))
    print("status", json.dumps(s), flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make-stub")
    m.add_argument("--out", required=True)
    m.add_argument("--tokenizer", default=TOKENIZER_ID)
    p = sub.add_parser("population")
    p.add_argument("--expect-fingerprint", default=None)
    s = sub.add_parser("smoke-check")
    s.add_argument("--kind", choices=("patch", "swap"), required=True)
    s.add_argument("--dir", required=True)
    s.add_argument("--n", type=int, default=3)
    s.add_argument("--engine", required=True, help="substring the row's engine name must carry, e.g. cuda-bfloat16")
    s.add_argument("--n-layers", type=int, required=True)
    t = sub.add_parser("stage-time")
    for k in ("--file", "--stage", "--status", "--engine"):
        t.add_argument(k, required=True)
    for k in ("--rc", "--elapsed", "--t-offset", "--hard-stop-s", "--self-stop-s"):
        t.add_argument(k, type=int, required=True)
    t.add_argument("--note", default="")
    st = sub.add_parser("status")
    st.add_argument("--out", required=True)
    st.add_argument("--patch-dir", required=True)
    st.add_argument("--swap-dir", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "make-stub":
        make_stub(Path(a.out), a.tokenizer)
        return 0
    if a.cmd == "population":
        return population(a.expect_fingerprint)
    if a.cmd == "smoke-check":
        return smoke_check(a.kind, Path(a.dir), a.n, a.engine, a.n_layers)
    if a.cmd == "stage-time":
        return stage_time(a)
    return status(Path(a.out), Path(a.patch_dir), Path(a.swap_dir))


if __name__ == "__main__":
    sys.exit(main())
