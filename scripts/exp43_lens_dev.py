#!/usr/bin/env python
"""§43 laptop logit-lens dev reads: one family's 126 dev population rows (folds 0-1 ONLY), positions P1, P0, E, Q
(and P1_raw) exactly as §34's forced mode (`readout_for`, an appended `Answer:`), k = 10, the logit lens only.

No J-lens is fitted or read: the lens is the identity (logit lens), so only the `logit` variant is written; every
json says `lens: identity (logit lens; no J-lens fitted)` and `variants: ["logit"]`. Tracked ids at every position:
the gold's first token, the family's refusal ids (" unknown", " Unknown" after "Answer:", checked against the
tokenizer at load) and §34's bare "unknown".

    GUARD_VM_GB=22 MALLOC_ARENA_MAX=2 HF_ENABLE_PARALLEL_LOADING=false runs/_pipelines/guard.sh \\
        .venv/bin/python scripts/exp43_lens_dev.py llama3b

Outputs under runs/exp43_lens_dev/<family>/: <qid>.npz (ranks_logit [layers, positions, tracked], top_ids_logit /
top_logits_logit [layers, positions, k], layers, position_index), <qid>.json (§34 fields, measures under "logit"),
_timing.json, _failed.json, _dropped.json. Resumable: a readable <qid>.json with this script's READ_TAG is done.
Folds 2-4 are refused (exit 2) and never loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from memctl.analysis import families  # noqa: E402
from memctl.analysis.jlens_readout import (POSITION_NAMES, encode_chat_ids, question_measures, rank_table,  # noqa: E402
                                           readout_for, topk_table)
from memctl.jacobian_data import family_population, load_family_records  # noqa: E402
from scripts.jlens_read import MAX_SEQ_LEN, _flat_positions, _readable, append_timing, write_json  # noqa: E402

DEV_FOLDS = (0, 1)
READERS = ("llama3b", "gemma4b", "phi4mini")
TRACKED = ("gold", "unknown_sp", "Unknown_sp", "unknown")   # §34's three plus the family's " Unknown"
LENS_LABEL = "identity (logit lens; no J-lens fitted)"
READ_TAG = "exp43_lens_dev_v1"
REFUSED = 2


def load_hf_token() -> None:
    """HF_TOKEN from .env into the environment (never printed)."""
    env = ROOT / ".env"
    if "HF_TOKEN" in os.environ or not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip().removeprefix("export ")
        if line.startswith("HF_TOKEN="):
            os.environ["HF_TOKEN"] = line.split("=", 1)[1].strip().strip('"').strip("'")


def check_folds(folds) -> None:
    bad = sorted(set(folds) - set(DEV_FOLDS))
    if bad:
        raise SystemExit(f"REFUSED: folds {bad} are confirmation folds; this script reads folds 0-1 only")


def dev_population(cfg, folds=DEV_FOLDS):
    check_folds(folds)
    records, dropped = [], {}
    for fold in folds:
        rs, d = load_family_records(cfg, fold)
        records.extend(rs)
        dropped[str(fold)] = d
    pop = family_population(records)
    assert all(r.fold in DEV_FOLDS for r in pop), "a non-dev row reached the population"
    return pop, dropped


def is_done(out: Path, question_id: str) -> bool:
    try:
        payload = json.loads((out / f"{question_id}.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    return isinstance(payload, dict) and payload.get("read_tag") == READ_TAG


def read_logit(lens_model, tokenizer, cfg, record, k: int = 10):
    """§34 forced-mode read with the logit lens only (jlens_read.read_record without the transport)."""
    import torch
    from jlens.hooks import ActivationRecorder

    started = time.perf_counter()
    readout = readout_for(tokenizer, record.prompt, record.memory_lines, record.evidence_lines, record.gold,
                          chat_text_fn=lambda p: families.chat_text(tokenizer, cfg, p))
    chat_ids = encode_chat_ids(tokenizer, readout.chat_text, MAX_SEQ_LEN)
    if chat_ids != list(readout.input_ids[:readout.n_chat]) or len(readout.input_ids) > MAX_SEQ_LEN:
        raise ValueError(f"{record.question_id}: chat ids differ from the readout's or exceed {MAX_SEQ_LEN}")
    families.check_bos(tokenizer, cfg, list(readout.input_ids))
    tracked = dict(readout.tracked)
    if tracked["unknown_sp"] != cfg.refusal_ids[0]:
        raise ValueError(f"{record.question_id}: unknown_sp {tracked['unknown_sp']} != config {cfg.refusal_ids[0]}")
    tracked["Unknown_sp"] = cfg.refusal_ids[1]
    tracked = {name: tracked[name] for name in TRACKED}
    input_ids = torch.tensor([list(readout.input_ids)], dtype=torch.long, device=lens_model.input_device)
    final = lens_model.n_layers - 1
    layers = list(range(lens_model.n_layers))
    flat, index = _flat_positions(readout)
    with torch.no_grad(), ActivationRecorder(lens_model.layers, at=layers) as recorder:
        lens_model.forward(input_ids)
        residuals = {l: recorder.activations[l][0][flat].float() for l in layers}
    with torch.no_grad():
        ll = {l: lens_model.unembed(residuals[l]).float().cpu().numpy() for l in layers if l != final}
        model_logits = lens_model.unembed(residuals[final]).float().cpu().numpy()
    del residuals
    tracked_ids = [tracked[name] for name in TRACKED]
    ranks, layer_list = rank_table(ll, model_logits, lens_model.n_layers, tracked_ids)
    top_ids, top_logits, _ = topk_table(ll, model_logits, lens_model.n_layers, k)
    arrays = {"ranks_logit": ranks, "top_ids_logit": top_ids, "top_logits_logit": top_logits,
              "layers": np.array(layer_list, dtype=np.int32), "position_index": np.array(flat, dtype=np.int32)}
    measures = {"logit": question_measures(ranks, layer_list, lens_model.n_layers, TRACKED, index, k)}
    payload = {"question_id": record.question_id, "fold": record.fold, "reader": record.reader,
               "n_tokens": len(readout.input_ids), "n_leading_ws": readout.n_leading_ws,
               "prompt_sha256": hashlib.sha256(record.prompt.encode()).hexdigest(),
               "intact_prompt_sha256": hashlib.sha256(record.prompt.encode()).hexdigest(),
               "n_evidence_lines_cut": 0,
               "positions": {name: list(readout.positions[name]) for name in POSITION_NAMES},
               "tracked_names": list(TRACKED), "tracked": tracked, "gold_token_text": readout.gold_token_text,
               "position_mode": "forced", "answer_marker": readout.answer_marker,
               "lens": LENS_LABEL, "variants": ["logit"], "read_tag": READ_TAG,
               "measures": measures, "family": cfg.stamp(), "elapsed_s": time.perf_counter() - started}
    return payload, arrays


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("family", choices=READERS)
    p.add_argument("--folds", default="0,1")
    p.add_argument("--weights", default=None, help="default: the family's laptop fp32 copy (Gemma, Phi), else the hub id")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)
    folds = tuple(int(f) for f in args.folds.split(","))
    try:
        check_folds(folds)
    except SystemExit as e:
        print(e, flush=True)
        return REFUSED
    cfg = families.get(args.family)
    for key, value in cfg.laptop_env.items():
        if os.environ.get(key) != value:
            print(f"exp43_lens_dev: set {key}={value} (the family's laptop_env) before launching", file=sys.stderr)
            return REFUSED
    out = Path(args.out or f"runs/exp43_lens_dev/{cfg.key}")
    out.mkdir(parents=True, exist_ok=True)
    pop, dropped = dev_population(cfg, folds)
    if args.limit:
        pop = pop[: args.limit]
    write_json(out / "_dropped.json", dropped)
    pending = [r for r in pop if not is_done(out, r.question_id)]
    print(f"{cfg.key}: {len(pop)} dev rows, {len(pending)} to read", flush=True)
    if not pending:
        return 0
    load_hf_token()
    import torch
    from scripts.jlens_read import load_model

    torch.set_num_threads(args.threads)
    weights = args.weights or cfg.weights
    t_load = time.perf_counter()
    lens_model, tok = load_model(cfg.model_id, "cpu", "float32", family=cfg, weights=weights)
    load_s = time.perf_counter() - t_load
    families.check_refusal_ids(lambda t: families.encode(tok, t), cfg)
    failed_path = out / "_failed.json"
    failed = json.loads(failed_path.read_text()) if _readable(failed_path) else {}
    timing, started = {}, time.perf_counter()
    for number, record in enumerate(pending):
        try:
            payload, arrays = read_logit(lens_model, tok, cfg, record, k=args.k)
        except (ValueError, RuntimeError) as error:
            failed[record.question_id] = repr(error)
            write_json(failed_path, failed)
            print(f"[{number + 1}/{len(pending)}] {record.question_id} FAILED {error!r}", flush=True)
            continue
        payload.update({"model_id": cfg.model_id, "weights": weights})
        np.savez_compressed(out / f"{record.question_id}.npz", **arrays)        # npz first: json marks done
        write_json(out / f"{record.question_id}.json", payload)
        if failed.pop(record.question_id, None) is not None:
            write_json(failed_path, failed)
        timing[record.question_id] = payload["elapsed_s"]
        g = payload["measures"]["logit"]["P1"]["gold"]
        print(f"[{number + 1}/{len(pending)}] {record.question_id} {payload['elapsed_s']:.1f}s n_tokens={payload['n_tokens']} "
              f"P1 gold logit={g['outcome']} final_rank={g['final_rank']}", flush=True)
    summary = {"model_id": cfg.model_id, "weights": weights, "lens": LENS_LABEL, "read_tag": READ_TAG,
               "device": "cpu", "dtype": "float32", "threads": args.threads, "load_s": load_s,
               "questions": len(timing), "seconds_per_question": (sum(timing.values()) / len(timing)) if timing else None,
               "wall_s": time.perf_counter() - started,
               "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}
    append_timing(out, summary)
    print(json.dumps(summary), flush=True)
    return 8 if (not timing and failed) else 0


if __name__ == "__main__":
    sys.exit(main())
