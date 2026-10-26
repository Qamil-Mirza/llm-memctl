#!/usr/bin/env python
"""§34: read a fitted Jacobian lens (and the logit-lens baseline) at the answer, prompt-end, evidence and
question positions of every cached §23 prompt. One forward per question; resumable per question.

    .venv/bin/python scripts/jlens_read.py --reader qwen05b --records-from qwen3b --lens runs/exp34_jlens/lenses/qwen05b.pt --folds 0 --limit 10 --device cpu --dtype float32
    python scripts/jlens_read.py --reader qwen3b --lens runs/exp34_jlens/lenses/qwen3b.pt --folds 0,1,2,3,4 --device cuda

Outputs under runs/exp34_jlens/<reader>/: <qid>.npz (ranks and top-k for both variants), <qid>.json
(positions, tracked ids, measures, lens_sha256), _timing.json, _failed.json (retried next run), _dropped.json.
Exit 7: the lens does not fit the model (width or layers). Exit 8: this run read nothing and _failed.json is non-empty.
A question counts as done only if its json carries the current lens's sha256, so a lens refitted or resumed
since is re-read automatically (the older outputs are overwritten). Never deletes anything else.
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

from memctl.analysis.jacobian import ablate_evidence
from memctl.analysis import families  # noqa: E402
from memctl.analysis.jlens_readout import (POSITION_NAMES, encode_chat_ids, file_sha256,  # noqa: E402
                                           question_measures, rank_table, readout_for, topk_table)
from memctl.jacobian_data import PromptRecord, load_family_records, load_records  # noqa: E402

MODEL_IDS = {"qwen05b": "Qwen/Qwen2.5-0.5B-Instruct", "qwen3b": "Qwen/Qwen2.5-3B-Instruct", "qwen7b": "Qwen/Qwen2.5-7B-Instruct"}
# §43: other families' small readers (memctl/analysis/families.py); records from their §24 runs, template pinned
FAMILY_READERS = ("llama3b", "gemma4b", "phi4mini")
MAX_SEQ_LEN = 4096
TRACKED = ("gold", "unknown_sp", "unknown")
DEV_FOLDS = {0, 1}


def write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, path)


def _readable(path: Path) -> bool:
    try:
        json.loads(path.read_text())
        return True
    except (FileNotFoundError, json.JSONDecodeError):
        return False


def is_done(out: Path, question_id: str, lens_hash: str) -> bool:
    """Done = a readable <qid>.json read with this exact lens. An output from another lens (e.g. a fit that
    was resumed to more prompts since), or one without a hash, is not done and is overwritten."""
    try:
        payload = json.loads((out / f"{question_id}.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    return isinstance(payload, dict) and payload.get("lens_sha256") == lens_hash


def append_timing(out: Path, summary: dict) -> Path:
    """Append a run summary to _timing.json (as §33); a corrupt _timing.json is kept and the summary goes
    to _timing.<unix time>.json instead. Returns the path written."""
    path = Path(out) / "_timing.json"
    if path.exists() and not _readable(path):
        existing, path = {}, Path(out) / f"_timing.{int(time.time())}.json"
    else:
        existing = json.loads(path.read_text()) if path.exists() else {}
    existing.setdefault("runs", []).append(summary)
    write_json(path, existing)
    return path


def _flat_positions(readout) -> tuple[list[int], dict[str, list[int]]]:
    """Concatenate the positions in POSITION_NAMES order; return the flat list and each name's slots in it."""
    flat, index = [], {}
    for name in POSITION_NAMES:
        index[name] = list(range(len(flat), len(flat) + len(readout.positions[name])))
        flat.extend(readout.positions[name])
    return flat, index


def lens_to(lens, device) -> None:
    """Move the lens's Jacobians to `device` once, so `transport` does not copy them on every call."""
    lens.jacobians = {layer: J.to(device) for layer, J in lens.jacobians.items()}


def read_record(lens, lens_model, tokenizer, record: PromptRecord, k: int = 10,
                position_mode: str = "forced", family=None) -> tuple[dict, dict[str, np.ndarray]]:
    """`family` (a FamilyConfig, §43): the chat text is the family's template with its pinned arguments; None (every
    §34 read) is `wrap_chat`."""
    import torch
    from jlens.hooks import ActivationRecorder

    chat_fn = None if family is None else (lambda p: families.chat_text(tokenizer, family, p))

    started = time.perf_counter()
    if position_mode == "ablated":
        # §34c: the same prompt with every evidence line cut out (exact spans), forced position, no evidence positions
        prompt = ablate_evidence(record.prompt, record.memory_lines, record.evidence_lines)
        memory = tuple(l for l in record.memory_lines if l not in set(record.evidence_lines))
        readout = readout_for(tokenizer, prompt, memory, (), record.gold, chat_text_fn=chat_fn)
    else:
        prompt = record.prompt
        readout = readout_for(tokenizer, record.prompt, record.memory_lines, record.evidence_lines, record.gold,
                              own_output=record.output if position_mode == "own" else None, chat_text_fn=chat_fn)
    # truncation guard only; no special tokens (§43 B1: the chat text carries its own BOS, or none)
    chat_ids = encode_chat_ids(lens_model.tokenizer, readout.chat_text, MAX_SEQ_LEN)
    if chat_ids != list(readout.input_ids[:readout.n_chat]) or len(readout.input_ids) > MAX_SEQ_LEN:
        raise ValueError(f"{record.question_id}: lens encode of the chat text ({len(chat_ids)} ids) differs from the "
                         f"readout's chat ids ({readout.n_chat}) or the sequence exceeds {MAX_SEQ_LEN}; truncated or retokenised")
    input_ids = torch.tensor([list(readout.input_ids)], dtype=torch.long, device=lens_model.input_device)
    final = lens_model.n_layers - 1
    layers = sorted(set(lens.source_layers) | {final})
    flat, index = _flat_positions(readout)
    with torch.no_grad(), ActivationRecorder(lens_model.layers, at=layers) as recorder:
        lens_model.forward(input_ids)
        residuals = {l: recorder.activations[l][0][flat].float() for l in layers}
    tracked_ids = [readout.tracked[name] for name in TRACKED]
    jl, ll = {}, {}
    with torch.no_grad():
        for l in layers:
            if l == final:
                continue
            jl[l] = lens_model.unembed(lens.transport(residuals[l], l)).float().cpu().numpy()
            ll[l] = lens_model.unembed(residuals[l]).float().cpu().numpy()
        model_logits = lens_model.unembed(residuals[final]).float().cpu().numpy()   # final layer: both variants
    arrays = {}
    for variant, logits in (("jlens", jl), ("logit", ll)):
        ranks, layer_list = rank_table(logits, model_logits, lens_model.n_layers, tracked_ids)
        top_ids, top_logits, _ = topk_table(logits, model_logits, lens_model.n_layers, k)
        arrays[f"ranks_{variant}"], arrays[f"top_ids_{variant}"], arrays[f"top_logits_{variant}"] = ranks, top_ids, top_logits
    arrays["layers"] = np.array(layer_list, dtype=np.int32)
    arrays["position_index"] = np.array(flat, dtype=np.int32)
    measures = {variant: question_measures(arrays[f"ranks_{variant}"], layer_list, lens_model.n_layers, TRACKED, index, k)
                for variant in ("jlens", "logit")}
    payload = {"question_id": record.question_id, "fold": record.fold, "reader": record.reader,
               "n_tokens": len(readout.input_ids), "n_leading_ws": readout.n_leading_ws,
               "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
               "intact_prompt_sha256": hashlib.sha256(record.prompt.encode()).hexdigest(),
               "n_evidence_lines_cut": len(record.evidence_lines) if position_mode == "ablated" else 0,
               "positions": {name: list(readout.positions[name]) for name in POSITION_NAMES},
               "tracked": dict(readout.tracked), "gold_token_text": readout.gold_token_text,
               "position_mode": position_mode, "answer_marker": readout.answer_marker,
               "measures": measures, "elapsed_s": time.perf_counter() - started}
    if family is not None:
        payload["family"] = family.stamp()
    return payload, arrays


def run_records(lens, lens_model, tokenizer, records, out: Path, *, k: int, model_id: str, lens_path: str,
                lens_sha256: str, position_mode: str = "forced", family=None) -> dict[str, float]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    failed_path = out / "_failed.json"
    if failed_path.exists() and not _readable(failed_path):
        failed_path = out / f"_failed.{int(time.time())}.json"          # keep a corrupt _failed.json
    failed = json.loads(failed_path.read_text()) if failed_path.exists() else {}
    timing = {}
    for number, record in enumerate(records):
        if is_done(out, record.question_id, lens_sha256):
            continue
        try:
            payload, arrays = read_record(lens, lens_model, tokenizer, record, k=k, position_mode=position_mode,
                                          family=family)
        except (ValueError, RuntimeError) as error:
            failed[record.question_id] = repr(error)
            write_json(failed_path, failed)
            print(f"[{number + 1}/{len(records)}] {record.question_id} FAILED {error!r}", flush=True)
            continue
        payload.update({"model_id": model_id, "lens_path": lens_path, "lens_sha256": lens_sha256,
                        "lens_n_prompts": getattr(lens, "n_prompts", None)})
        np.savez_compressed(out / f"{record.question_id}.npz", **arrays)        # npz first: json marks done
        write_json(out / f"{record.question_id}.json", payload)
        if record.question_id in failed:
            failed.pop(record.question_id)
            write_json(failed_path, failed)
        timing[record.question_id] = payload["elapsed_s"]
        line = f"[{number + 1}/{len(records)}] {record.question_id} {payload['elapsed_s']:.1f}s n_tokens={payload['n_tokens']}"
        if record.fold in DEV_FOLDS:                    # confirmation-fold outcomes never reach the log
            m = payload["measures"]
            line += (f" P1 gold jlens={m['jlens']['P1']['gold']['outcome']} logit={m['logit']['P1']['gold']['outcome']} "
                     f"final_rank={m['jlens']['P1']['gold']['final_rank']}")
        print(line, flush=True)
    return timing


LENS_MISMATCH = 7
ALL_FAILED = 8


def load_model(model_id: str, device: str, dtype: str, family=None, weights: str | None = None):
    """(jlens LensModel, tokenizer) for a Hugging Face model id. With a family (§43): the weights from `weights`
    (e.g. the laptop's fp32 copy) or the model id, the tokenizer from the model id, jlens's fit-time BOS flag from
    the config (reads never use it: the chat ids are encoded without special tokens)."""
    import jlens
    import torch
    import transformers

    hf = transformers.AutoModelForCausalLM.from_pretrained(weights or model_id, dtype=getattr(torch, dtype)).to(device)
    hf.eval()
    tok = transformers.AutoTokenizer.from_pretrained(model_id)
    if family is None:
        return jlens.from_hf(hf, tok), tok
    return jlens.from_hf(hf, tok, force_bos=family.lens_force_bos), tok


def lens_mismatch(lens, lens_model) -> str | None:
    """Why the lens cannot be read on this model (width from a Jacobian's shape, layers in range), or None."""
    widths = {tuple(J.shape) for J in lens.jacobians.values()}
    if widths != {(lens_model.d_model, lens_model.d_model)}:
        return f"Jacobian shapes {sorted(widths)} do not match the model's d_model {lens_model.d_model}"
    if not lens.source_layers or max(lens.source_layers) >= lens_model.n_layers:
        return f"source layer {max(lens.source_layers, default=None)} is not below the model's n_layers {lens_model.n_layers}"
    return None


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reader", required=True, choices=sorted(MODEL_IDS) + list(FAMILY_READERS))
    p.add_argument("--weights", default=None, help="§43 families: a local copy of the weights (e.g. cache/exp43_fp32/...)")
    p.add_argument("--records-from", default=None, help="reader whose §23 prompts to read (default: --reader)")
    p.add_argument("--lens", required=True)
    p.add_argument("--folds", default="0,1,2,3,4")
    p.add_argument("--limit", type=int, default=None, help="first N records per fold")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--model-id", default=None)
    p.add_argument("--position-mode", default="forced", choices=("forced", "own", "ablated"),
                   help="forced: P1 right after an appended 'Answer:' (§34); own: P1 at the reader's own answer after its"
                        " cached reasoning (§34b); ablated: forced position on the prompt with the evidence lines cut (§34c);"
                        " default --out runs/exp34_jlens_<mode>/<reader> for own and ablated")
    args = p.parse_args(argv)
    family = families.get(args.reader) if args.reader in FAMILY_READERS else None
    if family is not None and args.records_from not in (None, args.reader):
        p.error("--records-from is a Qwen option; a family reads its own §24 records")
    base = "runs/exp43_jlens" if family is not None else "runs/exp34_jlens"
    out = Path(args.out or (f"{base}/{args.reader}" if args.position_mode == "forced" else f"{base}_{args.position_mode}/{args.reader}"))
    out.mkdir(parents=True, exist_ok=True)
    records, dropped = [], {}
    for fold in (int(f) for f in args.folds.split(",")):
        rs, d = (load_family_records(family, fold) if family is not None
                 else load_records(args.records_from or args.reader, fold))
        records.extend(rs[: args.limit] if args.limit else rs)
        dropped[str(fold)] = d
    write_json(out / "_dropped.json", dropped)
    lens_hash = file_sha256(args.lens)                          # outputs from another lens are redone
    pending = [r for r in records if not is_done(out, r.question_id, lens_hash)]
    if not pending:
        print("nothing to do", flush=True)
        return 0
    import jlens
    import torch

    model_id = args.model_id or (family.model_id if family is not None else MODEL_IDS[args.reader])
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    lens_model, tok = (load_model(model_id, device, args.dtype) if family is None
                       else load_model(model_id, device, args.dtype, family=family, weights=args.weights))
    lens = jlens.JacobianLens.load(args.lens)
    problem = lens_mismatch(lens, lens_model)
    if problem:
        print(f"jlens_read: {args.lens} does not fit {model_id}: {problem}", file=sys.stderr, flush=True)
        return LENS_MISMATCH
    lens_to(lens, device)
    started = time.perf_counter()
    timing = run_records(lens, lens_model, tok, records, out, k=args.k, model_id=model_id, lens_path=args.lens,
                         lens_sha256=lens_hash, position_mode=args.position_mode, family=family)
    summary = {"model_id": model_id, "lens": args.lens, "lens_sha256": lens_hash, "device": device, "dtype": args.dtype, "questions": len(timing),
               "seconds_per_question": (sum(timing.values()) / len(timing)) if timing else None,
               "wall_s": time.perf_counter() - started,
               "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}
    append_timing(out, summary)
    print(json.dumps(summary), flush=True)
    failed = json.loads((out / "_failed.json").read_text()) if _readable(out / "_failed.json") else {}
    if not timing and failed:
        print(f"jlens_read: this run read nothing and {len(failed)} failed (see {out / '_failed.json'})",
              file=sys.stderr, flush=True)
        return ALL_FAILED
    return 0


if __name__ == "__main__":
    sys.exit(main())
