#!/usr/bin/env python
"""§34: fit a Jacobian lens for one reader on WikiText paragraphs, under a time box.

    .venv/bin/python scripts/jlens_fit.py --reader qwen05b --n-prompts 4 --min-prompts 2 --max-minutes 30 --device cpu --dtype float32
    python scripts/jlens_fit.py --reader qwen7b --n-prompts 128 --min-prompts 32 --max-minutes 90 --dim-batch 32 --device cuda

Writes <out>/<reader>.pt (fp16 Jacobians, one per source layer), <out>/<reader>.json (n_prompts, settings,
wall time, corpus and prompt hashes, jlens commit) and keeps <out>/<reader>.ckpt.pt for resume. Never
deletes anything. Exit 0 whenever a lens was saved; the sidecar's below_minimum says whether it is usable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MODEL_IDS = {"qwen05b": "Qwen/Qwen2.5-0.5B-Instruct", "qwen3b": "Qwen/Qwen2.5-3B-Instruct", "qwen7b": "Qwen/Qwen2.5-7B-Instruct"}
JLENS_DIR = Path(__file__).resolve().parents[1] / "external" / "jacobian-lens"


def load_corpus(parquet_path, min_words: int = 150) -> list[str]:
    import pyarrow.parquet as pq

    texts = pq.read_table(str(parquet_path)).column("text").to_pylist()
    return [t.strip() for t in texts if len(t.split()) >= min_words]


def select_prompts(paragraphs: list[str], n: int, seed: int = 0) -> list[str]:
    order = list(range(len(paragraphs)))
    random.Random(seed).shuffle(order)
    return [paragraphs[i] for i in order[:n]]


def jlens_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(JLENS_DIR), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def fit_under_time_box(model, prompts, *, max_minutes: float, chunk: int, checkpoint_path: str, **fit_kwargs):
    """Call jlens.fit on growing prefixes of ``prompts`` (resume processes only the new ones) until the
    time box is spent or the list is done. Returns (lens, n_prompts_processed)."""
    import jlens

    started = time.perf_counter()
    lens, done = None, 0
    for upto in range(chunk, len(prompts) + chunk, chunk):
        upto = min(upto, len(prompts))
        lens = jlens.fit(model, prompts[:upto], checkpoint_path=checkpoint_path, checkpoint_every=1, resume=True, **fit_kwargs)
        done = upto
        if upto >= len(prompts) or (time.perf_counter() - started) / 60.0 >= max_minutes:
            break
    return lens, done


def sidecar(reader, model_id, *, n_prompts, n_target, n_minimum, settings, wall_s, corpus_sha256, prompt_sha256s) -> dict:
    return {"reader": reader, "model_id": model_id, "n_prompts": n_prompts, "n_target": n_target, "n_minimum": n_minimum,
            "below_minimum": n_prompts < n_minimum, "settings": settings, "wall_s": wall_s, "jlens_commit": jlens_commit(),
            "corpus_sha256": corpus_sha256, "prompt_sha256s": prompt_sha256s}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reader", required=True, choices=sorted(MODEL_IDS))
    p.add_argument("--model-id", default=None)
    p.add_argument("--n-prompts", type=int, default=128)
    p.add_argument("--min-prompts", type=int, default=32)
    p.add_argument("--max-minutes", type=float, default=90.0)
    p.add_argument("--chunk", type=int, default=4)
    p.add_argument("--dim-batch", type=int, default=8)
    p.add_argument("--max-seq-len", type=int, default=128)
    p.add_argument("--skip-first", type=int, default=16)
    p.add_argument("--source-layers", default="all", help="'all' (every layer below the last) or 'a:b' (half-open range)")
    p.add_argument("--out", default="runs/exp34_jlens/lenses")
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--corpus", default="data/jlens_corpus/wikitext-103-raw-v1-test.parquet")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    import torch, transformers, jlens

    model_id = args.model_id or MODEL_IDS[args.reader]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    paragraphs = load_corpus(args.corpus)
    prompts = select_prompts(paragraphs, args.n_prompts, seed=args.seed)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    hf = transformers.AutoModelForCausalLM.from_pretrained(model_id, dtype=getattr(torch, args.dtype)).to(device)
    tok = transformers.AutoTokenizer.from_pretrained(model_id)
    model = jlens.from_hf(hf, tok)
    source_layers = None if args.source_layers == "all" else list(range(*map(int, args.source_layers.split(":"))))
    settings = {"dim_batch": args.dim_batch, "max_seq_len": args.max_seq_len, "skip_first": args.skip_first,
                "source_layers": args.source_layers, "seed": args.seed, "chunk": args.chunk, "max_minutes": args.max_minutes,
                "device": device, "dtype": args.dtype}
    print(f"{model} fitting on {len(prompts)} prompts (min {args.min_prompts}, box {args.max_minutes} min)", flush=True)
    started = time.perf_counter()
    lens, n_done = fit_under_time_box(model, prompts, max_minutes=args.max_minutes, chunk=args.chunk,
                                      checkpoint_path=str(out / f"{args.reader}.ckpt.pt"), source_layers=source_layers,
                                      dim_batch=args.dim_batch, max_seq_len=args.max_seq_len, skip_first=args.skip_first)
    wall = time.perf_counter() - started
    lens.save(str(out / f"{args.reader}.pt"))
    side = sidecar(args.reader, model_id, n_prompts=lens.n_prompts, n_target=args.n_prompts, n_minimum=args.min_prompts,
                   settings=settings, wall_s=wall, corpus_sha256=hashlib.sha256(Path(args.corpus).read_bytes()).hexdigest(),
                   prompt_sha256s=[hashlib.sha256(x.encode()).hexdigest() for x in prompts[:n_done]])
    (out / f"{args.reader}.json").write_text(json.dumps(side, indent=1))
    print(json.dumps({k: v for k, v in side.items() if k != "prompt_sha256s"}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
