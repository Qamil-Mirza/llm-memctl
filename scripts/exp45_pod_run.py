#!/usr/bin/env python
"""§45 pod helpers (scripts/exp45_pod.sh): stub model, the population check, stage times, refit box, status.

    python scripts/exp45_pod_run.py make-stub --out D            # tiny random Qwen2 (4 layers), the real tokenizer
    python scripts/exp45_pod_run.py population [--expect-cases-sha S]
    python scripts/exp45_pod_run.py refit-box --left 9000 --max-min 75 --reserve 5400   # minutes for the refit
    python scripts/exp45_pod_run.py stage-time --file T --stage S --status ok --rc 0 --elapsed 12 --t-offset 40 ...
    python scripts/exp45_pod_run.py status --out D --maps-root R

Nothing here reads or prints an outcome.
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
STUB_LAYERS = 4
N_CASES, N_ROWS44 = 864, 140
LENS56 = Path("runs/exp34_jlens/lenses/qwen3b.pt")
LENS56_SHA = "ddbdb32de0eefd4ed0c2ca42d64620c322e931a24813df5fa25da95c482de21b"
CASES = Path("runs/exp45_maps/cases.jsonl")
ROWS44 = Path("runs/exp45_maps/rows44.jsonl")


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def make_stub(out: Path) -> int:
    import torch
    import transformers

    target = out / "qwen3b"
    if (target / "config.json").exists():
        return 0
    tok = transformers.AutoTokenizer.from_pretrained(TOKENIZER_ID)
    cfg = transformers.Qwen2Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=STUB_LAYERS,
                                   num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=8192,
                                   tie_word_embeddings=True)
    torch.manual_seed(45)
    transformers.Qwen2ForCausalLM(cfg).eval().save_pretrained(target)
    tok.save_pretrained(target)
    print(f"stub qwen3b: {STUB_LAYERS} layers, width 64, vocab {len(tok)} -> {target}", flush=True)
    return 0


def population(expect_cases_sha: str | None, check_lens: bool = True) -> int:
    import hashlib as _h

    problems = []
    cases = [json.loads(l) for l in CASES.read_text().splitlines() if l.strip()]
    counts = json.loads((CASES.parent / "cases_counts.json").read_text())
    cases_sha = sha_file(CASES)
    if len(cases) != N_CASES:
        problems.append(f"{len(cases)} cases, want {N_CASES}")
    if cases_sha != counts["cases_sha256"] or not counts["pass"]:
        problems.append("cases.jsonl sha differs from cases_counts.json, or its checks did not pass")
    if expect_cases_sha and cases_sha != expect_cases_sha:
        problems.append(f"cases sha {cases_sha[:16]} != bundle's {expect_cases_sha[:16]}")
    bad_prompt = [c["case_id"] for c in cases if _h.sha256(c["prompt"].encode()).hexdigest() != c["prompt_sha256"]]
    if bad_prompt:
        problems.append(f"{len(bad_prompt)} case prompts differ from their sha")
    rows = [json.loads(l) for l in ROWS44.read_text().splitlines() if l.strip()]
    if len(rows) != N_ROWS44:
        problems.append(f"{len(rows)} stage 3 prompts, want {N_ROWS44}")
    bad_rows = [r["key"] for r in rows if _h.sha256(r["prompt"].encode()).hexdigest() != r["expected_sha256"]]
    if bad_rows:
        problems.append(f"{len(bad_rows)} stage 3 prompts differ from the §44 gate's sha")
    lens_sha = sha_file(LENS56) if (check_lens and LENS56.exists()) else None
    if check_lens and lens_sha != LENS56_SHA:
        problems.append(f"{LENS56} sha {str(lens_sha)[:16]} is not the §34 56-prompt lens")
    info = {"n_cases": len(cases), "cases_sha256": cases_sha, "n_rows44": len(rows),
            "n_rows44_questions": len({r["question_id"] for r in rows}), "lens56_sha256": lens_sha, "problems": problems}
    print("population", json.dumps(info), flush=True)
    return 0 if not problems else 3


def refit_box(left: int, max_min: float, reserve: int, chunk_s: int) -> int:
    """Minutes for the refit time box: what is left after the maps reserve and one chunk, capped at max_min; 0 when
    the box would be under 5 minutes (the chain then skips the refit and maps with j56 + logit)."""
    budget = (left - reserve - chunk_s) / 60.0
    box = min(max_min, budget)
    return int(box) if box >= 5 else 0


def stage_time(a) -> int:
    path = Path(a.file)
    t = json.loads(path.read_text()) if path.exists() else {"stages": [], "hard_stop_s": a.hard_stop_s,
                                                            "self_stop_s": a.self_stop_s, "engine": a.engine}
    t["stages"].append({"stage": a.stage, "status": a.status, "rc": a.rc, "elapsed_s": a.elapsed,
                        "t_offset_s": a.t_offset, "note": a.note})
    path.write_text(json.dumps(t, indent=1))
    return 0


def status(out: Path, maps_root: Path, lenses: Path) -> int:
    def count(d: Path) -> int:
        return sum(1 for p in d.glob("*.json") if not p.name.startswith("_")) if d.exists() else 0

    s = {}
    for lens_set in ("j56-jrefit-logit", "j56-logit"):
        for stage in (1, 3):
            n = count(maps_root / lens_set / f"stage{stage}")
            if n:
                s[f"{lens_set}/stage{stage}"] = n
    side = lenses / "qwen3b.json"
    s["refit"] = ({k: v for k, v in json.loads(side.read_text()).items() if k in ("n_prompts", "below_minimum", "wall_s_total")}
                  if side.exists() else None)
    s["eval"] = (maps_root / "eval" / "qwen3b.json").exists()
    (out / "status.json").write_text(json.dumps(s, indent=1))
    print("status", json.dumps(s), flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make-stub"); m.add_argument("--out", required=True)
    m = sub.add_parser("population"); m.add_argument("--expect-cases-sha", default=None)
    m.add_argument("--no-lens", action="store_true", help="skip the lens sha (never on the pod)")
    m = sub.add_parser("refit-box"); m.add_argument("--left", type=int, required=True)
    m.add_argument("--max-min", type=float, default=75); m.add_argument("--reserve", type=int, required=True)
    m.add_argument("--chunk-s", type=int, default=140)
    m = sub.add_parser("stage-time")
    for name, kind in (("file", str), ("stage", str), ("status", str), ("rc", int), ("elapsed", int), ("t-offset", int),
                       ("hard-stop-s", int), ("self-stop-s", int), ("engine", str)):
        m.add_argument(f"--{name}", type=kind, required=True)
    m.add_argument("--note", default="")
    m = sub.add_parser("status"); m.add_argument("--out", required=True); m.add_argument("--maps-root", required=True)
    m.add_argument("--lenses", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "make-stub":
        return make_stub(Path(a.out))
    if a.cmd == "population":
        return population(a.expect_cases_sha, check_lens=not a.no_lens)
    if a.cmd == "refit-box":
        print(refit_box(a.left, a.max_min, a.reserve, a.chunk_s))
        return 0
    if a.cmd == "stage-time":
        return stage_time(a)
    return status(Path(a.out), Path(a.maps_root), Path(a.lenses))


if __name__ == "__main__":
    sys.exit(main())
