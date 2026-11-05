#!/usr/bin/env python
"""§43 Part B pod stage, the Python half of scripts/exp43_lens_pod.sh (proposal 2026-10-26-exp43-cross-family.md,
amendment 1 Part B, rev 1 rulings B1-B3). One family per pod (--family llama | gemma | phi), both sizes.

Subcommands (the shell script calls them in this order; each is resumable):
    population   the §34 population of the family's two §24 readers (answerable, all evidence in view, >= 1
                 evidence line, both sizes, the same prompt sha256, a judge label on both), counts asserted
                 61/65/66/68/70; writes <out>/population.json (ids, folds, prompt hashes: no outcomes). Exit 3.
    read         logit-lens (--variant logit) or J-lens + logit (--variant jlens --lens P) reads, §34 forced mode
                 (an appended `Answer:`), positions P1/P0/E/Q/P1_raw, k = 10, tracked: the gold's first token and
                 the family's refusal ids (" unknown", " Unknown") and §34's bare "unknown". --smoke N reads the
                 first N development rows only and asserts the D-22-style checks (exit 5 on any failure).
                 --deadline (unix s): no row is started that would not end before it (exit 9: partial, logged).
    make-stub    tiny random models with the real tokenizers (ENGINE=stub, CPU; no weights downloaded).
    bundle-list  the §24 cache files of the population rows for both readers (scripts/exp43_lens_bundle.sh).
    estimate     seconds a stage is expected to take on the A40 (stage skip rule; time table in the report).
    stage-time   append one stage record to <out>/stage_times.json (copied back; later quotes are set from it).

Outputs (per reader dir, the §34/exp43_lens_dev format): <qid>.npz (ranks_<variant> [layers, positions, tracked],
top_ids_<variant>, top_logits_<variant>, layers, position_index), <qid>.json (positions, tracked, measures per
variant, family stamp, size, read_tag, lens_sha256), _timing.json, _failed.json, _deadline.json.
Confirmation folds 2-4 are read here (the pod reads all 330 rows) but their outcomes never reach a log line;
scripts/exp43_lens_report.py reads them only with --confirm.
"""

from __future__ import annotations

import argparse
import dataclasses
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
from memctl.analysis.jlens_readout import (POSITION_NAMES, encode_chat_ids, file_sha256, question_measures,  # noqa: E402
                                           rank_table, readout_for, topk_table)

DEV_FOLDS = (0, 1)
ALL_FOLDS = (0, 1, 2, 3, 4)
EXPECTED_COUNTS = {0: 61, 1: 65, 2: 66, 3: 68, 4: 70}       # stage 0 pop34.json, every family (and Qwen under §34)
TRACKED = ("gold", "unknown_sp", "Unknown_sp", "unknown")
READ_TAG = "exp43_lens_pod_v1"
LOGIT_LABEL = "identity (logit lens)"
MAX_SEQ_LEN = 4096
K = 10
SMOKE_N = 5
SMOKE_TOP1_MIN = 4              # of 5: final-layer top-1 at P1 equal to the laptop fp32 dev read (small model)
EXIT_INPUT, EXIT_SMOKE, EXIT_LENS, EXIT_ALL_FAILED, EXIT_DEADLINE = 3, 5, 7, 8, 9

# Expected A40 seconds per stage (model load included). Fit: stage 0's power law through §34's two fits (±50%);
# reads: §34's 0.51-0.55 s/question scaled by parameters above 7.6 B, plus the load; downloads at ~150 MB/s.
# `fit_min` is the time to reach the 32-prompt minimum. scripts/exp43_lens_report.py prints this table.
EST_S = {
    "llama": {"install": 60, "download": 170, "smoke": 200, "logit_small": 240, "logit_large": 300,
              "fit": 3780, "fit_min": 2160, "jread": 300, "eval": 150},
    "gemma": {"install": 60, "download": 240, "smoke": 240, "logit_small": 270, "logit_large": 420,
              "fit": 3060, "fit_min": 1750, "jread": 330, "eval": 160},
    "phi":   {"install": 60, "download": 270, "smoke": 280, "logit_small": 270, "logit_large": 540,
              "fit": 4380, "fit_min": 2500, "jread": 330, "eval": 160},
}
STUB_EST_S = {"install": 0, "download": 0, "smoke": 30, "logit_small": 120, "logit_large": 120,
              "fit": 60, "fit_min": 20, "jread": 120, "eval": 30}
WEIGHTS_GB = {"llama3b": 6.4, "llama8b": 16.1, "gemma4b": 8.6, "gemma12b": 24.4, "phi4mini": 7.7, "phi4": 29.3}


def write_json(path: Path, payload) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1))
    os.replace(tmp, path)


def read_json(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def load_hf_token() -> None:
    """HF_TOKEN from /workspace/.hf_token or .env into the environment (never printed)."""
    if os.environ.get("HF_TOKEN"):
        return
    pod = Path("/workspace/.hf_token")
    if pod.exists() and pod.read_text().strip():
        os.environ["HF_TOKEN"] = pod.read_text().strip()
        return
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip().removeprefix("export ")
            if line.startswith("HF_TOKEN="):
                os.environ["HF_TOKEN"] = line.split("=", 1)[1].strip().strip('"').strip("'")


# --- population ---------------------------------------------------------------------------------------------------

def pop34_pairs(small, large, folds=ALL_FOLDS, runs_dir: str = "runs"):
    """[(small record, large record)] of the §34 population, paired by question id and prompt sha256, sorted by
    (fold, question id). The rule of stage 0's pop34.py and exp34_report.population, plus the judge label on both."""
    from memctl.jacobian_data import load_family_records

    pairs = []
    for fold in folds:
        a = {r.question_id: r for r in load_family_records(small, fold, runs_dir)[0]}
        b = {r.question_id: r for r in load_family_records(large, fold, runs_dir)[0]}
        inview = lambda r: r.answerable and r.all_in_view and bool(r.evidence_lines)  # noqa: E731
        for q in sorted(set(a) & set(b)):
            x, y = a[q], b[q]
            if not (inview(x) and inview(y)):
                continue
            if hashlib.sha256(x.prompt.encode()).hexdigest() != hashlib.sha256(y.prompt.encode()).hexdigest():
                continue
            if x.correct is None or y.correct is None:
                continue
            pairs.append((x, y))
    pairs.sort(key=lambda p: (p[0].fold, p[0].question_id))
    return pairs


def fold_counts(pairs) -> dict[int, int]:
    out = {f: 0 for f in ALL_FOLDS}
    for x, _ in pairs:
        out[x.fold] += 1
    return out


def check_counts(counts: dict[int, int], folds=ALL_FOLDS) -> list[str]:
    return [f"fold {f}: {counts.get(f, 0)} rows, expected {EXPECTED_COUNTS[f]}" for f in folds
            if counts.get(f, 0) != EXPECTED_COUNTS[f]]


def fingerprint(pairs) -> str:
    """sha256 over every population row's id, fold, prompt and both outputs and judge labels: equal on the laptop and
    in the bundle (the cache subset rebuilds the same records). A hash, so no outcome is readable from it."""
    sha = lambda t: hashlib.sha256(t.encode()).hexdigest()  # noqa: E731
    rows = [[x.question_id, x.fold, sha(x.prompt), sha(x.output), sha(y.output), x.correct, x.unknown, y.correct,
             y.unknown, x.gold] for x, y in pairs]
    return sha(json.dumps(rows))


def population_payload(family: str, pairs) -> dict:
    small, large = families.pair(family)
    return {"family": family, "small": small.key, "large": large.key, "fingerprint": fingerprint(pairs),
            "counts": {str(k): v for k, v in fold_counts(pairs).items()},
            "rows": [{"question_id": x.question_id, "fold": x.fold, "prompt_sha256": hashlib.sha256(x.prompt.encode()).hexdigest()}
                     for x, _ in pairs]}


# --- one read -----------------------------------------------------------------------------------------------------

def read_row(lens_model, tokenizer, cfg, record, lens=None, k: int = K, break_bos: bool = False):
    """§34 forced-mode read of one record (exp43_lens_dev.read_logit, plus the J-lens transport when `lens`)."""
    import torch
    from jlens.hooks import ActivationRecorder

    from scripts.jlens_read import _flat_positions

    started = time.perf_counter()
    readout = readout_for(tokenizer, record.prompt, record.memory_lines, record.evidence_lines, record.gold,
                          chat_text_fn=lambda p: families.chat_text(tokenizer, cfg, p))
    chat_ids = encode_chat_ids(tokenizer, readout.chat_text, MAX_SEQ_LEN)
    if chat_ids != list(readout.input_ids[:readout.n_chat]) or len(readout.input_ids) > MAX_SEQ_LEN:
        raise ValueError(f"{record.question_id}: chat ids differ from the readout's or exceed {MAX_SEQ_LEN}")
    ids = list(readout.input_ids)
    if break_bos:                                   # test hook (STUB_BREAK=bos): the stage 0 double-BOS bug
        ids = [tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 0] + ids
    families.check_bos(tokenizer, cfg, ids)
    tracked = dict(readout.tracked)
    if tracked["unknown_sp"] != cfg.refusal_ids[0]:
        raise ValueError(f"{record.question_id}: unknown_sp {tracked['unknown_sp']} != config {cfg.refusal_ids[0]}")
    tracked["Unknown_sp"] = cfg.refusal_ids[1]
    tracked = {name: tracked[name] for name in TRACKED}
    input_ids = torch.tensor([ids], dtype=torch.long, device=lens_model.input_device)
    final = lens_model.n_layers - 1
    layers = list(range(lens_model.n_layers))
    flat, index = _flat_positions(readout)
    with torch.no_grad(), ActivationRecorder(lens_model.layers, at=layers) as recorder:
        lens_model.forward(input_ids)
        residuals = {l: recorder.activations[l][0][flat].float() for l in layers}
    variants = ("jlens", "logit") if lens is not None else ("logit",)
    logits = {v: {} for v in variants}
    with torch.no_grad():
        for l in layers:
            if l == final:
                continue
            logits["logit"][l] = lens_model.unembed(residuals[l]).float().cpu().numpy()
            if lens is not None:
                logits["jlens"][l] = lens_model.unembed(lens.transport(residuals[l], l)).float().cpu().numpy()
        model_logits = lens_model.unembed(residuals[final]).float().cpu().numpy()
    del residuals
    tracked_ids = [tracked[name] for name in TRACKED]
    arrays, measures = {}, {}
    for v in variants:
        ranks, layer_list = rank_table(logits[v], model_logits, lens_model.n_layers, tracked_ids)
        top_ids, top_logits, _ = topk_table(logits[v], model_logits, lens_model.n_layers, k)
        arrays.update({f"ranks_{v}": ranks, f"top_ids_{v}": top_ids, f"top_logits_{v}": top_logits})
        measures[v] = question_measures(ranks, layer_list, lens_model.n_layers, TRACKED, index, k)
    arrays["layers"] = np.array(layer_list, dtype=np.int32)
    arrays["position_index"] = np.array(flat, dtype=np.int32)
    payload = {"question_id": record.question_id, "fold": record.fold, "reader": cfg.key,
               "n_tokens": len(ids), "n_leading_ws": readout.n_leading_ws,
               "prompt_sha256": hashlib.sha256(record.prompt.encode()).hexdigest(),
               "intact_prompt_sha256": hashlib.sha256(record.prompt.encode()).hexdigest(),
               "n_evidence_lines_cut": 0, "n_bos": ids.count(tokenizer.bos_token_id) if tokenizer.bos_token_id is not None else 0,
               "positions": {name: list(readout.positions[name]) for name in POSITION_NAMES},
               "tracked_names": list(TRACKED), "tracked": tracked, "gold_token_text": readout.gold_token_text,
               "position_mode": "forced", "answer_marker": readout.answer_marker,
               "variants": list(variants), "read_tag": READ_TAG, "measures": measures, "family": cfg.stamp(),
               "elapsed_s": time.perf_counter() - started}
    return payload, arrays


def is_done(out: Path, question_id: str, lens_hash: str | None) -> bool:
    payload = read_json(Path(out) / f"{question_id}.json")
    return (isinstance(payload, dict) and payload.get("read_tag") == READ_TAG
            and payload.get("lens_sha256") == lens_hash and (Path(out) / f"{question_id}.npz").exists())


def smoke_problems(payloads, arrays_list, cfg, model_info: dict, reference_dir: str | None) -> list[str]:
    """D-22-style checks on the smoke rows: every row read, the model's depth and width are the config's, one BOS
    (or none) per the family, the tracked refusal ids are the config's, the final layer is the model's own logits
    (rank arrays cover layers 0..L-1, finite top logits), and (small model, when the laptop's fp32 dev reads are in
    the bundle) the final-layer top-1 at P1 equals the laptop's on >= SMOKE_TOP1_MIN of the rows."""
    problems = []
    if model_info["n_layers"] != cfg.n_layers or model_info["d_model"] != cfg.hidden:
        problems.append(f"model has {model_info['n_layers']} layers / width {model_info['d_model']}, config says "
                        f"{cfg.n_layers} / {cfg.hidden}")
    want_bos = 1 if cfg.bos_in_chat else 0
    agree, compared = 0, 0
    for payload, arrays in zip(payloads, arrays_list):
        q = payload["question_id"]
        if payload["n_bos"] != want_bos:
            problems.append(f"{q}: {payload['n_bos']} BOS, want {want_bos}")
        if [payload["tracked"]["unknown_sp"], payload["tracked"]["Unknown_sp"]] != list(cfg.refusal_ids):
            problems.append(f"{q}: tracked refusal ids {payload['tracked']} != config {cfg.refusal_ids}")
        if arrays["layers"].tolist() != list(range(cfg.n_layers)):
            problems.append(f"{q}: layers {arrays['layers'].tolist()[:3]}... are not 0..{cfg.n_layers - 1}")
        for key, value in arrays.items():
            if key.startswith("top_logits_") and not np.isfinite(value).all():
                problems.append(f"{q}: non-finite {key}")
        if reference_dir:
            ref = Path(reference_dir) / f"{q}.npz"
            if ref.exists():
                with np.load(ref) as z:
                    compared += 1
                    agree += int(z["top_ids_logit"][-1, 0, 0]) == int(arrays["top_ids_logit"][-1, 0, 0])
    if reference_dir and compared:
        if agree < min(SMOKE_TOP1_MIN, compared):
            problems.append(f"final-layer top-1 at P1 equals the laptop fp32 read on {agree} of {compared} rows "
                            f"(want >= {min(SMOKE_TOP1_MIN, compared)})")
    elif reference_dir:
        problems.append(f"no laptop reference rows found in {reference_dir}")
    return problems


# --- the read command ---------------------------------------------------------------------------------------------

def _stub_cfg(cfg, weights: str):
    """The family config with the stub model's depth and width (the stub keeps the template, BOS, refusal ids)."""
    conf = json.loads((Path(weights) / "config.json").read_text())
    return dataclasses.replace(cfg, n_layers=conf["num_hidden_layers"], hidden=conf["hidden_size"])


def cmd_read(args) -> int:
    small, large = families.pair(args.family)
    cfg = small if args.size == "small" else large
    if args.variant == "jlens" and args.size != "small":
        print("exp43_lens_pod_run: the J-lens is fitted and read for the small model only (rev 1 B2)", file=sys.stderr)
        return 2
    if args.engine == "stub":
        cfg = _stub_cfg(cfg, args.weights)
    folds = tuple(int(f) for f in args.folds.split(","))
    if args.smoke:
        folds = tuple(f for f in folds if f in DEV_FOLDS)
    pairs = pop34_pairs(small, large, folds)
    bad = check_counts(fold_counts(pairs), folds)
    if bad:
        print("population: " + "; ".join(bad), file=sys.stderr)
        return EXIT_INPUT
    records = [x if args.size == "small" else y for x, y in pairs]
    if args.dev_first:                                   # the J-lens gate needs folds 0-1 first
        records.sort(key=lambda r: (r.fold not in DEV_FOLDS, r.fold, r.question_id))
    if args.smoke:
        records = records[: args.smoke]
    elif args.limit_per_fold:
        per = {}
        records = [r for r in records if per.setdefault(r.fold, []).append(r) or len(per[r.fold]) <= args.limit_per_fold]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lens_hash = file_sha256(args.lens) if args.variant == "jlens" else None
    pending = [r for r in records if args.smoke or not is_done(out, r.question_id, lens_hash)]
    print(f"{cfg.key} {args.variant}{' smoke' if args.smoke else ''}: {len(records)} rows, {len(pending)} to read", flush=True)
    if not pending:
        return 0
    load_hf_token()
    import torch

    from scripts.jlens_read import lens_mismatch, lens_to, load_model

    if args.threads:
        torch.set_num_threads(args.threads)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    t_load = time.perf_counter()
    lens_model, tok = load_model(cfg.model_id, device, args.dtype, family=cfg, weights=args.weights)
    load_s = time.perf_counter() - t_load
    families.check_refusal_ids(lambda t: families.encode(tok, t), cfg)
    lens = None
    if args.variant == "jlens":
        import jlens
        lens = jlens.JacobianLens.load(args.lens)
        problem = lens_mismatch(lens, lens_model)
        if problem:
            print(f"lens {args.lens} does not fit {cfg.model_id}: {problem}", file=sys.stderr)
            return EXIT_LENS
        lens_to(lens, device)
    failed_path = out / "_failed.json"
    failed = read_json(failed_path, {}) or {}
    timing, payloads, arrays_list, started = {}, [], [], time.perf_counter()
    cut = None
    for number, record in enumerate(pending):
        per_row = (sum(timing.values()) / len(timing)) if timing else 0.0
        if args.deadline and time.time() + 2 * per_row + 1 > args.deadline:
            cut = {"stopped_at_row": number, "of": len(pending), "unread": [r.question_id for r in pending[number:]],
                   "deadline": args.deadline, "at": time.time()}
            write_json(out / "_deadline.json", cut)
            print(f"deadline: stopped before row {number + 1} of {len(pending)} ({len(pending) - number} unread, listed "
                  f"in {out / '_deadline.json'})", flush=True)
            break
        try:
            payload, arrays = read_row(lens_model, tok, cfg, record, lens=lens, k=args.k,
                                       break_bos=os.environ.get("STUB_BREAK") == "bos")
        except (ValueError, RuntimeError) as error:
            if args.smoke:
                print(f"SMOKE FAILED: {record.question_id}: {error!r}", file=sys.stderr, flush=True)
                return EXIT_SMOKE
            failed[record.question_id] = repr(error)
            write_json(failed_path, failed)
            print(f"[{number + 1}/{len(pending)}] {record.question_id} FAILED {error!r}", flush=True)
            continue
        payload.update({"model_id": cfg.model_id, "weights": args.weights, "size": args.size, "device": device,
                        "dtype": args.dtype, "lens": args.lens or LOGIT_LABEL, "lens_sha256": lens_hash})
        np.savez_compressed(out / f"{record.question_id}.npz", **arrays)        # npz first: json marks done
        write_json(out / f"{record.question_id}.json", payload)
        if failed.pop(record.question_id, None) is not None:
            write_json(failed_path, failed)
        timing[record.question_id] = payload["elapsed_s"]
        payloads.append(payload)
        arrays_list.append(arrays)
        line = f"[{number + 1}/{len(pending)}] {record.question_id} f{record.fold} {payload['elapsed_s']:.2f}s n_tokens={payload['n_tokens']}"
        if record.fold in DEV_FOLDS:                    # confirmation-fold outcomes never reach the log
            g = payload["measures"]["logit"]["P1"]["gold"]
            line += f" P1 gold logit={g['outcome']} final_rank={g['final_rank']}"
        print(line, flush=True)
    summary = {"reader": cfg.key, "model_id": cfg.model_id, "variant": args.variant, "smoke": bool(args.smoke),
               "device": device, "dtype": args.dtype, "load_s": load_s, "questions": len(timing),
               "seconds_per_question": (sum(timing.values()) / len(timing)) if timing else None,
               "wall_s": time.perf_counter() - started, "deadline_cut": cut is not None,
               "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
               "peak_gpu_mb": (torch.cuda.max_memory_allocated() / 2**20) if device == "cuda" else None,
               "n_layers": lens_model.n_layers, "d_model": lens_model.d_model}
    timing_path = out / "_timing.json"
    existing = read_json(timing_path, {}) or {}
    existing.setdefault("runs", []).append(summary)
    write_json(timing_path, existing)
    print(json.dumps(summary), flush=True)
    if args.smoke:
        problems = smoke_problems(payloads, arrays_list, cfg, {"n_layers": lens_model.n_layers, "d_model": lens_model.d_model},
                                  None if args.engine == "stub" else args.reference)
        write_json(out / "_smoke.json", {"reader": cfg.key, "rows": [p["question_id"] for p in payloads],
                                         "problems": problems, "pass": not problems})
        if problems or len(payloads) != len(pending):
            print("SMOKE FAILED: " + "; ".join(problems or ["not every smoke row was read"]), file=sys.stderr, flush=True)
            return EXIT_SMOKE
        print(f"smoke ok: {cfg.key}, {len(payloads)} rows", flush=True)
        return 0
    if cut is not None:
        return EXIT_DEADLINE
    if not timing and failed:
        return EXIT_ALL_FAILED
    return 0


# --- the small commands -------------------------------------------------------------------------------------------

def cmd_population(args) -> int:
    pairs = pop34_pairs(*families.pair(args.family))
    payload = population_payload(args.family, pairs)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "population.json", payload)
    bad = check_counts(fold_counts(pairs))
    if args.expect_fingerprint and payload["fingerprint"] != args.expect_fingerprint:
        bad.append(f"fingerprint {payload['fingerprint'][:16]} != the bundle's {args.expect_fingerprint[:16]}")
    print(f"population {args.family}: {payload['counts']} (expected {EXPECTED_COUNTS}); fingerprint "
          f"{payload['fingerprint'][:16]}", flush=True)
    if bad:
        print("population: " + "; ".join(bad), file=sys.stderr)
        return EXIT_INPUT
    return 0


def cache_files_for(cfg, prompts: set[str]) -> list[str]:
    """The cache json files of `cfg`'s §24 cache whose prompt is one of `prompts`."""
    out = []
    for path in sorted(Path(cfg.cache_dir).glob("*/*.json")):
        try:
            entry = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if entry.get("prompt") in prompts:
            out.append(str(path))
    return out


def cmd_bundle_list(args) -> int:
    small, large = families.pair(args.family)
    pairs = pop34_pairs(small, large)
    bad = check_counts(fold_counts(pairs))
    if bad:
        print("population: " + "; ".join(bad), file=sys.stderr)
        return EXIT_INPUT
    for cfg, idx in ((small, 0), (large, 1)):
        files = cache_files_for(cfg, {p[idx].prompt for p in pairs})
        if len(files) < len(pairs):
            print(f"bundle-list: {cfg.key}: {len(files)} cache files for {len(pairs)} rows", file=sys.stderr)
            return EXIT_INPUT
        print("\n".join(files))
    if args.with_reference:                  # the laptop fp32 dev reads of the smoke rows (cmd_read --smoke order)
        dev = [x for x, _ in pairs if x.fold in DEV_FOLDS][:SMOKE_N]
        ref = Path("runs/exp43_lens_dev") / small.key
        for x in dev:
            for ext in ("json", "npz"):
                path = ref / f"{x.question_id}.{ext}"
                if not path.exists():
                    print(f"bundle-list: smoke reference {path} missing", file=sys.stderr)
                    return EXIT_INPUT
                print(path)
    return 0


def cmd_make_stub(args) -> int:
    """Tiny random LlamaForCausalLM per size with the real tokenizer's vocabulary (cached, local_files_only). The
    family's template, BOS, refusal ids and stop ids come from the tokenizer as on the pod; only the network is tiny."""
    import torch
    import transformers

    out = Path(args.out)
    for size, cfg, n_layers, hidden in (("small", families.pair(args.family)[0], 4, 32),
                                        ("large", families.pair(args.family)[1], 6, 48)):
        target = out / cfg.key
        if (target / "config.json").exists():
            continue
        tok = transformers.AutoTokenizer.from_pretrained(cfg.model_id, local_files_only=True)
        conf = transformers.LlamaConfig(vocab_size=len(tok), hidden_size=hidden, intermediate_size=2 * hidden,
                                        num_hidden_layers=n_layers, num_attention_heads=2, num_key_value_heads=1,
                                        max_position_embeddings=8192, tie_word_embeddings=True)
        torch.manual_seed(43 + n_layers)
        transformers.LlamaForCausalLM(conf).eval().save_pretrained(target)
        print(f"stub {size}: {cfg.key} -> {target} ({n_layers} layers, width {hidden}, vocab {len(tok)})", flush=True)
    return 0


def estimate(family: str, stage: str, stub: bool = False) -> int:
    table = STUB_EST_S if stub else EST_S[family]
    return int(table[stage])


def cmd_estimate(args) -> int:
    print(estimate(args.family, args.stage, args.engine == "stub"))
    return 0


def cmd_stage_time(args) -> int:
    path = Path(args.file)
    payload = read_json(path, None) or {"family": args.family, "hard_stop_s": args.hard_stop_s,
                                        "self_stop_s": args.self_stop_s, "engine": args.engine, "stages": []}
    payload["stages"].append({"stage": args.stage, "status": args.status, "elapsed_s": args.elapsed,
                              "t_offset_s": args.t_offset, "rc": args.rc, "note": args.note,
                              "estimate_s": estimate(args.family, args.est_key, args.engine == "stub") if args.est_key else None})
    write_json(path, payload)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    fam = dict(choices=sorted(families.PAIRS), required=True)
    s = sub.add_parser("population"); s.add_argument("--family", **fam); s.add_argument("--out", required=True)
    s.add_argument("--expect-fingerprint", default=None)
    s = sub.add_parser("bundle-list"); s.add_argument("--family", **fam); s.add_argument("--with-reference", action="store_true")
    s = sub.add_parser("make-stub"); s.add_argument("--family", **fam); s.add_argument("--out", required=True)
    s = sub.add_parser("estimate"); s.add_argument("--family", **fam); s.add_argument("--stage", required=True)
    s.add_argument("--engine", default="hf", choices=("hf", "stub"))
    s = sub.add_parser("stage-time")
    s.add_argument("--family", **fam); s.add_argument("--file", required=True); s.add_argument("--stage", required=True)
    s.add_argument("--status", required=True); s.add_argument("--elapsed", type=int, required=True)
    s.add_argument("--t-offset", type=int, default=None); s.add_argument("--rc", type=int, default=None)
    s.add_argument("--note", default=""); s.add_argument("--est-key", default=None)
    s.add_argument("--hard-stop-s", type=int, default=None); s.add_argument("--self-stop-s", type=int, default=None)
    s.add_argument("--engine", default="hf", choices=("hf", "stub"))
    s = sub.add_parser("read")
    s.add_argument("--family", **fam); s.add_argument("--size", choices=("small", "large"), required=True)
    s.add_argument("--variant", choices=("logit", "jlens"), default="logit"); s.add_argument("--lens", default=None)
    s.add_argument("--out", required=True); s.add_argument("--folds", default="0,1,2,3,4")
    s.add_argument("--smoke", type=int, default=0); s.add_argument("--reference", default=None)
    s.add_argument("--deadline", type=float, default=None); s.add_argument("--dev-first", action="store_true")
    s.add_argument("--limit-per-fold", type=int, default=None); s.add_argument("--k", type=int, default=K)
    s.add_argument("--engine", choices=("hf", "stub"), default="hf"); s.add_argument("--weights", default=None)
    s.add_argument("--device", default="auto"); s.add_argument("--dtype", default="bfloat16")
    s.add_argument("--threads", type=int, default=None)
    args = p.parse_args(argv)
    if args.cmd == "read":
        if args.variant == "jlens" and not args.lens:
            p.error("--variant jlens needs --lens")
        if args.engine == "stub" and not args.weights:
            p.error("--engine stub needs --weights (scripts/exp43_lens_pod_run.py make-stub)")
        return cmd_read(args)
    return {"population": cmd_population, "bundle-list": cmd_bundle_list, "make-stub": cmd_make_stub,
            "estimate": cmd_estimate, "stage-time": cmd_stage_time}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
