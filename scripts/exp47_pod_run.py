#!/usr/bin/env python
"""§47 pod helpers (scripts/exp47_pod.sh). Nothing here reads or prints an outcome.

    python scripts/exp47_pod_run.py make-stub --out D        # tiny random Qwen2 (36 layers, width 32), the real tokenizer
    python scripts/exp47_pod_run.py population [--expect-rows-sha S]
    python scripts/exp47_pod_run.py project --smoke S --elapsed 900 [--rate 0.59 --limit 1.00]
    python scripts/exp47_pod_run.py stage-time --file T --stage S --status ok --rc 0 --elapsed 12 --t-offset 40 ...
    python scripts/exp47_pod_run.py status --out D --root R

`project` (rev 1 ruling 8): the D-22 smoke's measured seconds per row over the three smoke rows (the longest among
them), times the 36 rows, times 1.15, plus the seconds already spent; exit 7 above the limit in dollars.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TOKENIZER_ID = "Qwen/Qwen2.5-3B-Instruct"
STUB_LAYERS = 36
N_ROWS = 36
ROWS = Path("runs/exp47_bridge/rows47.jsonl")
EXIT_INPUT, EXIT_PROJECTION = 3, 7


def make_stub(out: Path) -> int:
    import torch
    import transformers
    target = out / "qwen3b"
    if (target / "config.json").exists():
        return 0
    tok = transformers.AutoTokenizer.from_pretrained(TOKENIZER_ID)
    cfg = transformers.Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64,
                                   num_hidden_layers=STUB_LAYERS, num_attention_heads=4, num_key_value_heads=2,
                                   max_position_embeddings=8192, tie_word_embeddings=True)
    torch.manual_seed(47)
    transformers.Qwen2ForCausalLM(cfg).eval().save_pretrained(target)
    tok.save_pretrained(target)
    print(f"stub qwen3b: {STUB_LAYERS} layers, width 32, vocab {len(tok)} -> {target}", flush=True)
    return 0


def population(expect: str | None) -> int:
    problems = []
    data = ROWS.read_bytes()
    rows = [json.loads(l) for l in data.decode().splitlines() if l.strip()]
    ent = [r for r in rows if r["entered"]]
    if len(rows) != N_ROWS:
        problems.append(f"{len(rows)} rows, want {N_ROWS}")
    if sum(r["label"] == "success" for r in ent) != 13 or sum(r["label"] != "success" for r in ent) != 13:
        problems.append("entered counts differ from 13 / 13")
    if expect and hashlib.sha256(data).hexdigest() != expect:
        problems.append("rows47.jsonl sha differs from bundle_info")
    print(json.dumps({"rows": len(rows), "entered": len(ent), "problems": problems}), flush=True)
    return EXIT_INPUT if problems else 0


def project(smoke: Path, elapsed: float, rate: float, limit: float) -> int:
    sm = json.loads(Path(smoke).read_text())
    per_row = max(x["total"] for x in sm["timing"]) if sm["timing"] else sm["per_row_s"]
    run_s = per_row * N_ROWS * 1.15
    total = elapsed + run_s
    dollars = total / 3600 * rate
    payload = {"elapsed_s": elapsed, "per_row_s_max": per_row, "run_projection_s": run_s, "total_s": total,
               "dollars": round(dollars, 3), "limit": limit, "rate": rate, "stop_and_ask": dollars > limit}
    Path(smoke).with_name("projection.json").write_text(json.dumps(payload, indent=1))
    print(json.dumps(payload), flush=True)
    return EXIT_PROJECTION if dollars > limit else 0


def stage_time(a) -> int:
    path = Path(a.file)
    t = json.loads(path.read_text()) if path.exists() else {"stages": [], "hard_stop_s": a.hard_stop_s,
                                                            "self_stop_s": a.self_stop_s, "engine": a.engine}
    t["stages"].append({"stage": a.stage, "status": a.status, "rc": a.rc, "elapsed_s": a.elapsed,
                        "t_offset_s": a.t_offset, "note": a.note})
    path.write_text(json.dumps(t, indent=1))
    return 0


def status(out: Path, root: Path) -> int:
    s = {"rows": len(list((root / "rows").glob("*.json"))), "report": (root / "report.json").exists()}
    (out / "status.json").write_text(json.dumps(s, indent=1))
    print(json.dumps(s), flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make-stub"); m.add_argument("--out", required=True)
    m = sub.add_parser("population"); m.add_argument("--expect-rows-sha", default=None)
    m = sub.add_parser("project"); m.add_argument("--smoke", required=True)
    m.add_argument("--elapsed", type=float, required=True); m.add_argument("--rate", type=float, default=0.59)
    m.add_argument("--limit", type=float, default=1.00)
    m = sub.add_parser("stage-time")
    for k in ("--file", "--stage", "--status", "--note", "--engine"):
        m.add_argument(k, default="")
    for k in ("--rc", "--elapsed", "--t-offset", "--hard-stop-s", "--self-stop-s"):
        m.add_argument(k, type=int, default=0)
    m = sub.add_parser("status"); m.add_argument("--out", required=True); m.add_argument("--root", required=True)
    a = ap.parse_args()
    if a.cmd == "make-stub":
        return make_stub(Path(a.out))
    if a.cmd == "population":
        return population(a.expect_rows_sha)
    if a.cmd == "project":
        return project(Path(a.smoke), a.elapsed, a.rate, a.limit)
    if a.cmd == "stage-time":
        return stage_time(a)
    return status(Path(a.out), Path(a.root))


if __name__ == "__main__":
    sys.exit(main())
