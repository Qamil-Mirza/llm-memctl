#!/usr/bin/env python
"""§49 pod helpers (scripts/exp49_pod.sh). Nothing here reads or prints an outcome.

    python scripts/exp49_pod_run.py make-stub --out D           # tiny random Qwen2 (36 layers, width 32), real tokenizer
    python scripts/exp49_pod_run.py population [--expect-cases-sha S]
    python scripts/exp49_pod_run.py project --smoke S --elapsed 900 [--rate 0.59 --limit 4.00]
    python scripts/exp49_pod_run.py wait-mark --root R --timeout 1800   # held guard (git-verified) or exit 10 / 11
    python scripts/exp49_pod_run.py stage-time --file T --stage S --status ok --rc 0 --elapsed 12 --t-offset 40 ...
    python scripts/exp49_pod_run.py status --out D --root R

`project`: the D-22 smoke's measured seconds per baseline row (generation + capture) x 1,584 rows, plus its seconds
per arm generation x 2,400 arm generations (rev 0b B7 counts, rounded up), x 1.30, plus the seconds already spent and
a 900 s allowance for the design commit; exit 7 above the limit in dollars.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TOKENIZER_ID = "Qwen/Qwen2.5-3B-Instruct"
STUB_LAYERS = 36
CASES = Path("runs/exp49/cases49.jsonl")
N_ROWS, N_ARMS, FACTOR, MARK_ALLOWANCE_S = 1584, 2400, 1.30, 900
EXIT_INPUT, EXIT_PROJECTION, EXIT_MARK_TIMEOUT, EXIT_GUARD = 3, 7, 10, 11


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
    torch.manual_seed(49)
    transformers.Qwen2ForCausalLM(cfg).eval().save_pretrained(target)
    tok.save_pretrained(target)
    print(f"stub qwen3b: {STUB_LAYERS} layers, width 32, vocab {len(tok)} -> {target}", flush=True)
    return 0


def population(expect: str | None) -> int:
    data = CASES.read_bytes()
    rows = [json.loads(l) for l in data.decode().splitlines() if l.strip()]
    c = Counter(r["split"] for r in rows)
    problems = []
    if c != Counter({"dev": 792, "held": 792}):
        problems.append(f"split counts {dict(c)}, want 792 / 792")
    if expect and hashlib.sha256(data).hexdigest() != expect:
        problems.append("cases49.jsonl sha differs from bundle_info")
    print(json.dumps({"rows": len(rows), "by_split": dict(c), "problems": problems}), flush=True)
    return EXIT_INPUT if problems else 0


def project(smoke: Path, elapsed: float, rate: float, limit: float) -> int:
    sm = json.loads(Path(smoke).read_text())
    run_s = (sm["per_row_baseline_s"] * N_ROWS + sm["per_arm_s"] * N_ARMS) * FACTOR
    total = elapsed + run_s + MARK_ALLOWANCE_S
    dollars = total / 3600 * rate
    payload = {"elapsed_s": elapsed, "per_row_baseline_s": sm["per_row_baseline_s"], "per_arm_s": sm["per_arm_s"],
               "run_projection_s": run_s, "total_s": total, "dollars": round(dollars, 3), "limit": limit, "rate": rate,
               "stop_and_ask": dollars > limit}
    Path(smoke).with_name("projection.json").write_text(json.dumps(payload, indent=1))
    print(json.dumps(payload), flush=True)
    return EXIT_PROJECTION if dollars > limit else 0


def wait_mark(root: Path, timeout: float, poll: float = 10.0) -> int:
    """Wait for the laptop's design_committed.json (and the shallow .git it ships), then run the git-verified guard."""
    from memctl.analysis import abst49 as A
    t0 = time.time()
    marker = Path(root) / A.DESIGN_MARKER
    while not marker.exists():
        if time.time() - t0 > timeout:
            print(f"wait-mark: no {marker} after {timeout:.0f} s", flush=True)
            return EXIT_MARK_TIMEOUT
        time.sleep(poll)
    time.sleep(2)
    try:
        A.require_design(root)
    except PermissionError as e:
        print(f"wait-mark: guard refused: {e}", flush=True)
        return EXIT_GUARD
    print(f"wait-mark: design verified against its commit after {time.time() - t0:.0f} s", flush=True)
    return 0


def stage_time(a) -> int:
    path = Path(a.file)
    t = json.loads(path.read_text()) if path.exists() else {"stages": [], "hard_stop_s": a.hard_stop_s,
                                                            "self_stop_s": a.self_stop_s, "engine": a.engine}
    t["stages"].append({"stage": a.stage, "status": a.status, "rc": a.rc, "elapsed_s": a.elapsed,
                        "t_offset_s": a.t_offset, "note": a.note})
    path.write_text(json.dumps(t, indent=1))
    return 0


def status(out: Path, root: Path) -> int:
    s = {"baselines": len(list((root / "baselines").glob("*.json"))),
         "arms_dev": len(list((root / "arms_dev").glob("*.json"))),
         "arms_held": len(list((root / "arms_held").glob("*.json"))),
         "design": (root / "design.json").exists(), "report": (root / "report.json").exists()}
    (out / "status.json").write_text(json.dumps(s, indent=1))
    print(json.dumps(s), flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make-stub"); m.add_argument("--out", required=True)
    m = sub.add_parser("population"); m.add_argument("--expect-cases-sha", default=None)
    m = sub.add_parser("project"); m.add_argument("--smoke", required=True)
    m.add_argument("--elapsed", type=float, required=True); m.add_argument("--rate", type=float, default=0.59)
    m.add_argument("--limit", type=float, default=4.00)
    m = sub.add_parser("wait-mark"); m.add_argument("--root", required=True)
    m.add_argument("--timeout", type=float, default=1800); m.add_argument("--poll", type=float, default=10)
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
        return population(a.expect_cases_sha)
    if a.cmd == "project":
        return project(Path(a.smoke), a.elapsed, a.rate, a.limit)
    if a.cmd == "wait-mark":
        return wait_mark(Path(a.root), a.timeout, a.poll)
    if a.cmd == "stage-time":
        return stage_time(a)
    return status(Path(a.out), Path(a.root))


if __name__ == "__main__":
    sys.exit(main())
