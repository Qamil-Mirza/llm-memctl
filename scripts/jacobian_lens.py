#!/usr/bin/env python
"""§33 Jacobian lens: per-question input-embedding gradients of a reader on its cached §23 prompts.

    .venv/bin/python scripts/jacobian_lens.py --reader qwen05b --records-from qwen3b --folds 0 --limit 5 --device cpu
    python scripts/jacobian_lens.py --reader qwen3b --folds 0,1,2,3,4 --ablate --regen-check 20 --device cuda

Outputs one json (measures) and one npz (per-token gradient mass) per (question, target) under
runs/exp33_jacobian/<reader>/, plus _timing.json, _dropped.json and _regen_check.json. Resumable: existing,
readable outputs are skipped; when every question is done the model is not loaded ("nothing to do").
A question that raises ValueError or RuntimeError (CUDA out of memory included) is logged to _failed.json
and skipped; nothing is written for it, so the next run retries it automatically (and drops its
_failed.json entry once it succeeds). Never deletes anything.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memctl.analysis.jacobian import REGIONS, ablate_evidence, jacobian_pass, measures, target_value, wrap_chat  # noqa: E402
from memctl.jacobian_data import PromptRecord, load_records  # noqa: E402

MODEL_IDS = {
    "qwen05b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen3b": "Qwen/Qwen2.5-3B-Instruct",
    "qwen7b": "Qwen/Qwen2.5-7B-Instruct",
}
REGION_CODE = {name: code for code, name in enumerate(REGIONS)}


def prompt_sha256(prompt: str) -> str:
    """The identity of the prompt text a question was run on; the report checks both readers saw the same one."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def output_path(out: Path, question_id: str, target: str) -> Path:
    return Path(out) / f"{question_id}.{target}.json"


def write_json(path: Path, payload: dict) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, path)


def _readable(path: Path) -> bool:
    try:
        json.loads(Path(path).read_text())
        return True
    except (FileNotFoundError, json.JSONDecodeError):
        return False


def is_done(out: Path, question_id: str, targets: tuple[str, ...], ablate: bool = False) -> bool:
    """Done when every target's json is readable. The npz is written before the json, so a readable json
    means its npz exists; a crash between them leaves no json and the question is redone."""
    paths = [output_path(out, question_id, t) for t in targets]
    if ablate:
        paths.append(output_path(out, question_id, "ablation"))
    return all(_readable(p) for p in paths)


def _read_json(path: Path) -> dict | None:
    """The parsed file, {} when it does not exist, None when it is unreadable."""
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return {}
    except (json.JSONDecodeError, OSError):
        return None


def append_timing(out: Path, summary: dict) -> Path:
    """Append a run summary to _timing.json; a corrupt _timing.json is kept and the summary goes to
    _timing.<unix time>.json instead. Returns the path written."""
    path = Path(out) / "_timing.json"
    existing = _read_json(path)
    if existing is None:
        existing, path = {}, Path(out) / f"_timing.{int(time.time())}.json"
    existing.setdefault("runs", []).append(summary)
    write_json(path, existing)
    return path


def targets_for(record: PromptRecord, targets: tuple[str, ...]) -> tuple[tuple[str, ...], bool]:
    """The targets to compute for a record, and whether own was skipped. An empty cached output has no own
    continuation to differentiate (0/0 shares): skip that target and flag it in the other targets' json, so
    resume does not wait for an own file."""
    own_skipped = "own" in targets and not record.output.strip()
    return tuple(t for t in targets if not (own_skipped and t == "own")), own_skipped


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def run_records(model, tokenizer, records: list[PromptRecord], out: Path, targets: tuple[str, ...], ablate: bool,
                k: int, model_id: str) -> dict[str, float]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    timing: dict[str, float] = {}
    failed_path = out / "_failed.json"
    failed = _read_json(failed_path) or {}
    for number, record in enumerate(records):
        record_targets, own_skipped = targets_for(record, targets)
        if is_done(out, record.question_id, record_targets, ablate):
            continue
        started = time.perf_counter()
        try:
            _run_one(model, tokenizer, record, out, record_targets, own_skipped, ablate, k, model_id, started)
        except (ValueError, RuntimeError) as error:
            # Nothing partial stays: _run_one computes everything before it writes anything, so no json exists
            # for this question and resume (which keys on the json) retries it on the next run.
            failed[record.question_id] = repr(error)
            write_json(failed_path, failed)
            print(f"[{number + 1}/{len(records)}] {record.question_id} FAILED: {error!r}", flush=True)
            _empty_cuda_cache()
            continue
        if failed.pop(record.question_id, None) is not None:
            write_json(failed_path, failed)
        timing[record.question_id] = time.perf_counter() - started
        print(f"[{number + 1}/{len(records)}] {record.question_id} {timing[record.question_id]:.1f}s "
              f"rss {peak_rss_mb():.0f} MB", flush=True)
    return timing


def _empty_cuda_cache() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _run_one(model, tokenizer, record: PromptRecord, out: Path, record_targets: tuple[str, ...], own_skipped: bool,
             ablate: bool, k: int, model_id: str, started: float) -> None:
    """All outputs for one question: everything is computed first (gradients, then the ablation values), so an
    error leaves no json for the question; then per target an npz then its json, then the ablation json."""
    results = jacobian_pass(model, tokenizer, record.prompt, record.memory_lines, record.evidence_lines,
                            record.gold, record.output, targets=record_targets) if record_targets else {}
    gradient_s = time.perf_counter() - started
    prompt_hash = prompt_sha256(record.prompt)
    if ablate:
        full = target_value(model, tokenizer, record.prompt, record.gold, record.output, "gold")
        ablated = target_value(model, tokenizer, ablate_evidence(record.prompt, record.memory_lines, record.evidence_lines),
                               record.gold, record.output, "gold")
    for target, result in results.items():
        np.savez_compressed(output_path(out, record.question_id, target).with_suffix(".npz"),
                            g_l2=result.g_l2.astype(np.float32), g_gxi=result.g_gxi.astype(np.float32),
                            regions=np.array([REGION_CODE[r] for r in result.regions], dtype=np.int8))
        payload = {
            "question_id": record.question_id, "fold": record.fold, "reader": record.reader, "target": target,
            "model_id": model_id, "prompt_sha256": prompt_hash, "measures": measures(result, k=k), "elapsed_s": gradient_s,
        }
        payload.update(result.extra)
        if own_skipped:
            payload["own_skipped"] = True
        write_json(output_path(out, record.question_id, target), payload)
    if ablate:
        write_json(output_path(out, record.question_id, "ablation"), {
            "question_id": record.question_id, "prompt_sha256": prompt_hash, "gold_full": full, "gold_ablated": ablated,
            "delta": full - ablated, "n_evidence_lines": len(record.evidence_lines),
        })


def regen_check(model, tokenizer, records: list[PromptRecord], n: int, max_new_tokens: int = 256) -> dict:
    """Greedy regeneration of the first n prompts against the cached vLLM outputs."""
    import torch

    pairs = []
    device = model.get_input_embeddings().weight.device
    for record in records[:n]:
        ids = tokenizer(wrap_chat(tokenizer, record.prompt), return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        with torch.no_grad():
            generated = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new_tokens,
                                       do_sample=False)
        text = tokenizer.decode(generated[0][ids.shape[1]:], skip_special_tokens=True).strip()
        pairs.append({"question_id": record.question_id, "cached": record.output, "regenerated": text,
                      "match": text == record.output.strip()})
    return {"n": len(pairs), "matches": sum(p["match"] for p in pairs), "pairs": pairs}


def load_model(model_id: str, device: str, dtype: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=getattr(torch, dtype)).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, tokenizer, device


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reader", required=True, choices=sorted(MODEL_IDS))
    parser.add_argument("--records-from", default=None, help="reader whose §23 prompts to use (default: --reader)")
    parser.add_argument("--folds", default="0,1")
    parser.add_argument("--targets", default="gold,margin,own")
    parser.add_argument("--ablate", action="store_true")
    parser.add_argument("--out", default=None)
    parser.add_argument("--limit", type=int, default=None, help="first N records per fold (development)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--regen-check", type=int, default=0)
    parser.add_argument("--regen-min", type=int, default=15, help="abort if fewer than this many of the regen prompts match")
    parser.add_argument("--k", type=int, default=50)
    args = parser.parse_args(argv)

    model_id = args.model_id or MODEL_IDS[args.reader]
    out = Path(args.out or f"runs/exp33_jacobian/{args.reader}")
    out.mkdir(parents=True, exist_ok=True)
    targets = tuple(args.targets.split(","))
    records, dropped = [], {}
    for fold in (int(f) for f in args.folds.split(",")):
        fold_records, fold_dropped = load_records(args.records_from or args.reader, fold)
        records.extend(fold_records[: args.limit] if args.limit else fold_records)
        dropped[str(fold)] = fold_dropped
    write_json(out / "_dropped.json", dropped)
    print(f"{len(records)} records, dropped {dropped}; model {model_id}", flush=True)

    pending = [r for r in records if not is_done(out, r.question_id, targets_for(r, targets)[0], args.ablate)]
    if not pending:
        print("nothing to do: every question is done", flush=True)
        return 0
    print(f"{len(pending)} of {len(records)} questions to do", flush=True)
    model, tokenizer, device = load_model(model_id, args.device, args.dtype)
    if args.regen_check:
        check = regen_check(model, tokenizer, records, args.regen_check)
        write_json(out / "_regen_check.json", check)
        print(f"regen check: {check['matches']}/{check['n']} match", flush=True)
        if check["matches"] < min(args.regen_min, check["n"]):
            print("regen check FAILED; stopping", flush=True)
            return 2
    started = time.perf_counter()
    timing = run_records(model, tokenizer, records, out, targets, args.ablate, args.k, model_id)
    summary = {"model_id": model_id, "device": device, "dtype": args.dtype, "questions": len(timing),
               "seconds_per_question": (sum(timing.values()) / len(timing)) if timing else None,
               "wall_s": time.perf_counter() - started, "peak_rss_mb": peak_rss_mb(), "per_question": timing}
    append_timing(out, summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_question"}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
