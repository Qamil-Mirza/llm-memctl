#!/usr/bin/env python
"""§46 pod helpers (scripts/exp46_pod.sh). Nothing here reads or prints an outcome.

    python scripts/exp46_pod_run.py make-stub --out D        # tiny random Qwen2 (36 layers, width 32), the real tokenizer
    python scripts/exp46_pod_run.py population [--expect-cases-sha S] [--expect-cases46-sha S]
    python scripts/exp46_pod_run.py project --capture-timing T --smoke S --elapsed 900 [--rate 0.59 --limit 1.80]
    python scripts/exp46_pod_run.py stage-time --file T --stage S --status ok --rc 0 --elapsed 12 --t-offset 40 ...
    python scripts/exp46_pod_run.py status --out D --capture C --patch P

`project` (rev 1 ruling 4): the D-22 smoke's measured seconds per prompt (capture) and per triple (each patch stage),
scaled to the full run (576 prompts; 144 triples, all entered: an upper bound), plus the seconds already spent; exit 7
if that projects above the limit in dollars at the rate (the chain stops and the Executor asks).
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
N_CAPTURE, N_TRIPLES = 576, 144
CASES = Path("runs/exp45_maps/cases.jsonl")
CASES46 = Path("runs/exp46_probe/cases46.jsonl")
STAGE1 = Path("runs/exp45_maps/j56-jrefit-logit/stage1")
EXIT_INPUT, EXIT_PROJECTION = 3, 7


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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
    torch.manual_seed(46)
    transformers.Qwen2ForCausalLM(cfg).eval().save_pretrained(target)
    tok.save_pretrained(target)
    print(f"stub qwen3b: {STUB_LAYERS} layers, width 32, vocab {len(tok)} -> {target}", flush=True)
    return 0


def population(expect_cases: str | None, expect_cases46: str | None) -> int:
    problems = []
    cases = [json.loads(l) for l in CASES.read_text().splitlines() if l.strip()]
    rc_tu = [c for c in cases if c["task"] in ("RC", "TU")]
    if len(rc_tu) != N_CAPTURE:
        problems.append(f"{len(rc_tu)} RC/TU cases, want {N_CAPTURE}")
    missing = [c["case_id"] for c in rc_tu if not (STAGE1 / f"{c['case_id']}.json").exists()]
    if missing:
        problems.append(f"{len(missing)} §45 stage-1 records missing, e.g. {missing[:3]}")
    rows = [json.loads(l) for l in CASES46.read_text().splitlines() if l.strip()]
    triples = {r["triple_id"] for r in rows}
    if len(triples) != N_TRIPLES or len(rows) != 3 * N_TRIPLES:
        problems.append(f"cases46: {len(triples)} triples / {len(rows)} rows, want {N_TRIPLES} / {3 * N_TRIPLES}")
    if expect_cases and sha(CASES) != expect_cases:
        problems.append("cases.jsonl sha differs from bundle_info")
    if expect_cases46 and sha(CASES46) != expect_cases46:
        problems.append("cases46.jsonl sha differs from bundle_info")
    print(json.dumps({"rc_tu_cases": len(rc_tu), "stage1_records": len(rc_tu) - len(missing), "triples": len(triples),
                      "problems": problems}), flush=True)
    return EXIT_INPUT if problems else 0


def project(capture_timing: Path, smoke: Path, elapsed: float, rate: float, limit: float) -> int:
    cap = json.loads(Path(capture_timing).read_text())
    sm = json.loads(Path(smoke).read_text())
    cap_s = (sum(x["s"] for x in cap) / max(1, len(cap))) * N_CAPTURE * 1.15       # 15 % for load / IO
    patch_s = sm["projection_total_s"] * 1.10
    total = elapsed + cap_s + patch_s
    dollars = total / 3600 * rate
    payload = {"elapsed_s": elapsed, "capture_projection_s": cap_s, "patch_projection_s": patch_s,
               "total_s": total, "dollars": round(dollars, 3), "limit": limit, "rate": rate,
               "stop_and_ask": dollars > limit}
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


def status(out: Path, capture: Path, patch: Path) -> int:
    s = {"capture": len([p for p in capture.glob("*.npz")]),
         **{stage: len(list((patch / stage).glob("*.json"))) for stage in ("baselines", "s1", "controls", "s2")},
         "readout": (json.loads((patch / "readout.json").read_text()).get("rule")
                     if (patch / "readout.json").exists() else None)}
    (out / "status.json").write_text(json.dumps(s, indent=1))
    print(json.dumps(s), flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make-stub"); m.add_argument("--out", required=True)
    m = sub.add_parser("population"); m.add_argument("--expect-cases-sha", default=None)
    m.add_argument("--expect-cases46-sha", default=None)
    m = sub.add_parser("project"); m.add_argument("--capture-timing", required=True); m.add_argument("--smoke", required=True)
    m.add_argument("--elapsed", type=float, required=True); m.add_argument("--rate", type=float, default=0.59)
    m.add_argument("--limit", type=float, default=1.80)
    m = sub.add_parser("stage-time")
    for k in ("--file", "--stage", "--status", "--note", "--engine"):
        m.add_argument(k, default="")
    for k in ("--rc", "--elapsed", "--t-offset", "--hard-stop-s", "--self-stop-s"):
        m.add_argument(k, type=int, default=0)
    m = sub.add_parser("status"); m.add_argument("--out", required=True); m.add_argument("--capture", required=True)
    m.add_argument("--patch", required=True)
    a = ap.parse_args()
    if a.cmd == "make-stub":
        return make_stub(Path(a.out))
    if a.cmd == "population":
        return population(a.expect_cases_sha, a.expect_cases46_sha)
    if a.cmd == "project":
        return project(Path(a.capture_timing), Path(a.smoke), a.elapsed, a.rate, a.limit)
    if a.cmd == "stage-time":
        return stage_time(a)
    return status(Path(a.out), Path(a.capture), Path(a.patch))


if __name__ == "__main__":
    sys.exit(main())
