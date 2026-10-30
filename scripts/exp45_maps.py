#!/usr/bin/env python
"""§45: J-lens maps (layers x positions x lenses) for the case list (stage 1) or the §44 rev 8 rows (stage 3).
One greedy generation (<= 16 new tokens) and one forward per prompt; resumable per prompt.

    python scripts/exp45_maps.py --stage 1 --lens56 runs/exp34_jlens/lenses/qwen3b.pt --lens-refit runs/exp45_maps/lenses/qwen3b.pt
    python scripts/exp45_maps.py --stage 3 --lens56 ... [--lens-refit ...]
    python scripts/exp45_maps.py --smoke 3 --lens56 ... --reference runs/exp45_maps/smoke_ref --out runs/exp45_pod/smoke
    .venv/bin/python scripts/exp45_maps.py --write-reference --device cpu --dtype float32   # laptop fp32, logit lens only

Spec: research/proposals/2026-10-30-exp45-jlens-maps.md §2, §4, §6 (rev 0, 78d5ad1). Lenses: j56 (the §34 lens,
sha ddbdb32d...de21b), jrefit (the refit lens, used only when its sidecar says below_minimum false) and the logit
lens. Outputs: <root>/<lens set>/stage<n>/<key>.npz (ranks, logp, top_ids, top_logp: [lens, layer, position, .];
layers; positions; input_ids) and <key>.json (prompt, tokens, generated text, outcome, pinned ids, lens hashes),
_timing.json, _failed.json. The lens set is "j56-jrefit-logit" or "j56-logit". Stage 1 runs the dev half first.
A prompt counts as done only if its json carries the same lens hashes, chat sha, max_new / max_new_note and
MAPS_VERSION. Generation (rev 1): F1 16 new tokens; F2 and stage 3 up to 160, stopping at the end of the "Answer:"
line; the outcome is graded on the Answer line when present, the whole output otherwise (marker recorded).
Exit 5: smoke failed. 7: a lens does not fit the model. 8: nothing read and failures recorded. 9: stopped at the
deadline (rows so far kept). Nothing here prints an outcome except --smoke / --write-reference (3 dev cases).
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

from memctl.analysis import maps as M  # noqa: E402
from memctl.analysis.jlens_readout import file_sha256  # noqa: E402

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
MAPS_VERSION = "exp45_maps_v2"      # rev 1: answer-line generation, answer span, counted positions
CASES = Path("runs/exp45_maps/cases.jsonl")
ROWS44 = Path("runs/exp45_maps/rows44.jsonl")
ROOT_OUT = Path("runs/exp45_maps")
LENS56_SHA = "ddbdb32de0eefd4ed0c2ca42d64620c322e931a24813df5fa25da95c482de21b"
SMOKE_CASES = ("DE00_v0_F1_H0", "RC00_v0_F1_H0", "TU00_v0_F1_H0")
EXIT_SMOKE, EXIT_LENS, EXIT_ALL_FAILED, EXIT_DEADLINE = 5, 7, 8, 9


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, path)


def read_json(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def ordered_cases(cases: list[dict]) -> list[dict]:
    """Dev half first, then held; within a half by task, base, variant, format, hardness."""
    return sorted(cases, key=lambda c: (c["split"] != "dev", c["task"], c["base_index"], c["variant"], c["format"],
                                        c["hardness"]))


def lens_set_name(names) -> str:
    return "-".join(names)


def refit_usable(lens_path: str | None) -> tuple[bool, str]:
    """(usable, why) for the refit lens: its .pt exists and its sidecar says below_minimum false."""
    if not lens_path:
        return False, "no refit lens given"
    pt = Path(lens_path)
    side = read_json(pt.with_suffix(".json"))
    if not pt.exists() or side is None:
        return False, f"{pt} or its sidecar is missing"
    if side.get("below_minimum", True):
        return False, f"refit closed at {side.get('n_prompts')} prompts, below its minimum {side.get('n_minimum')}"
    return True, f"refit lens {side.get('n_prompts')} prompts"


# ---------------------------------------------------------------------------------------------- inputs

def case_inputs(tokenizer, case: dict, max_new: int, max_new_note: int = M.MAX_NEW_NOTE) -> tuple[M.MapInputs, dict]:
    """F1: `max_new` tokens, every generated token in the answer span; F2: up to `max_new_note`, stopping at the end
    of the "Answer:" line, the answer span after the marker (rev 1 rulings 2-3)."""
    from memctl.analysis.jacobian import wrap_chat

    chat = wrap_chat(tokenizer, case["prompt"])
    ids = [p["id"] for p in case["pinned"]]
    meta = {"case_id": case["case_id"], "task": case["task"], "split": case["split"], "variant": case["variant"],
            "format": case["format"], "hardness": case["hardness"], "base_index": case["base_index"],
            "prompt": case["prompt"], "prompt_sha256": case["prompt_sha256"], "gold": case["gold"],
            "competitors": case["competitors"], "intermediate": case["intermediate"], "pinned": case["pinned"]}
    if case["format"] == "F2":
        return M.MapInputs(case["case_id"], chat, ids, None, max_new_note, "after_marker"), meta
    return M.MapInputs(case["case_id"], chat, ids, None, max_new, "all"), meta


def span_positions(tokenizer, chat: str, prompt: str, spans: list[tuple[int, int]]) -> list[int]:
    """Token indices of the chat text overlapping any of the prompt-relative char spans."""
    enc = tokenizer(chat, add_special_tokens=False, return_offsets_mapping=True)
    base = chat.index(prompt)
    keep = []
    for i, (s, e) in enumerate(enc["offset_mapping"]):
        for a, b in spans:
            if s < base + b and e > base + a:
                keep.append(i)
                break
    return keep


def row44_inputs(tokenizer, row: dict, max_new: int, max_new_note: int = M.MAX_NEW_NOTE) -> tuple[M.MapInputs, dict]:
    """Stage 3: the §23 Note prompts, generated as F2 (rev 1 ruling 5); `max_new` is unused."""
    from memctl.analysis.jacobian import wrap_chat

    chat = wrap_chat(tokenizer, row["prompt"])
    dropped: list = []
    fixed = {(row["gold"], "sp"): row["gold_read_id"], (row["substitute"], "sp"): row["substitute_read_id"]}
    pinned = M.pin_tokens(tokenizer, [("gold", row["gold"]), ("substitute", row["substitute"])], fixed=fixed,
                          strict=False, dropped=dropped)
    positions = span_positions(tokenizer, chat, row["prompt"], [tuple(row["evidence_span"]), tuple(row["question_span"])])
    meta = {k: row[k] for k in ("key", "question_id", "fold", "kind", "prompt_sha256", "gold", "substitute",
                                "label23", "question_type", "output23", "evidence_span", "question_span")}
    meta.update({"pinned": pinned, "pinned_dropped": dropped, "competitors": [row["substitute"]]})
    return M.MapInputs(row["key"], chat, [p["id"] for p in pinned], positions, max_new_note, "after_marker"), meta


# ---------------------------------------------------------------------------------------------- model

def load(device: str, dtype: str, weights: str | None):
    import jlens
    import torch
    import transformers

    started = time.perf_counter()
    hf = transformers.AutoModelForCausalLM.from_pretrained(weights or MODEL_ID, dtype=getattr(torch, dtype)).to(device)
    hf.eval()
    tok = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
    return hf, jlens.from_hf(hf, tok), tok, time.perf_counter() - started


def load_lenses(lens56: str | None, refit: str | None, lens_model, device) -> tuple[dict, dict]:
    """name -> lens (None = logit) in LENS_NAMES order, and name -> sha256 (the logit lens: "identity")."""
    import jlens

    lenses, shas = {}, {}
    for name, path in (("j56", lens56), ("jrefit", refit)):
        if not path:
            continue
        lens = jlens.JacobianLens.load(path)
        problem = M.lens_mismatch(lens, lens_model)
        if problem:
            print(f"exp45_maps: {path} does not fit the model: {problem}", file=sys.stderr, flush=True)
            raise SystemExit(EXIT_LENS)
        M.lens_to(lens, device)
        lenses[name], shas[name] = lens, file_sha256(path)
    lenses["logit"], shas["logit"] = None, "identity"
    return lenses, shas


def is_done(out: Path, key: str, stamp: dict) -> bool:
    payload = read_json(out / f"{key}.json")
    return isinstance(payload, dict) and payload.get("stamp") == stamp and (out / f"{key}.npz").exists()


def run_one(hf, lens_model, tok, x: M.MapInputs, meta: dict, lenses: dict, stamp: dict, out: Path, k: int,
            break_bos: bool = False) -> dict:
    payload, arrays = M.map_prompt(hf, lens_model, tok, x, lenses, k=k, break_bos=break_bos)
    graded = M.classify(payload["generated_text"], meta["gold"], meta.get("competitors", []))
    payload.update(meta)
    payload.update({"outcome": graded["outcome"], "ambiguous": graded["ambiguous"], "answer": graded["answer"],
                    "answer_marker": graded["answer_marker"], "hits": graded["hits"], "stamp": stamp,
                    "lens_sha256": {n: stamp["lenses"][n] for n in payload["lens_names"]},
                    "pinned_ids": x.pinned_ids})
    np.savez_compressed(out / f"{x.key}.npz", **arrays)               # npz first: the json marks done
    write_json(out / f"{x.key}.json", payload)
    return payload


def append_timing(out: Path, summary: dict) -> None:
    path = out / "_timing.json"
    t = read_json(path, {}) or {}
    t.setdefault("runs", []).append(summary)
    write_json(path, t)


# ---------------------------------------------------------------------------------------------- smoke

def smoke_problems(payloads: list[dict], arrays: list[dict], n_layers: int, reference: dict | None,
                   lens_names: list[str]) -> list[str]:
    problems = []
    for p, a in zip(payloads, arrays):
        key = p["key"]
        if p["n_bos"] != 0:
            problems.append(f"{key}: {p['n_bos']} BOS-like tokens in the prompt ids (Qwen has none)")
        if p["position_tokens"][0] != "<|im_start|>":
            problems.append(f"{key}: the first token is {p['position_tokens'][0]!r}, not <|im_start|>")
        if p["n_layers"] != n_layers or a["ranks"].shape[1] != n_layers:
            problems.append(f"{key}: {a['ranks'].shape[1]} layers read, model has {p['n_layers']}, want {n_layers}")
        if list(p["lens_names"]) != lens_names:
            problems.append(f"{key}: lenses {p['lens_names']}, want {lens_names}")
        if a["ranks"].shape[2] != p["n_tokens"]:
            problems.append(f"{key}: {a['ranks'].shape[2]} positions for {p['n_tokens']} tokens")
        if (a["ranks"] < 1).any() or not np.isfinite(a["logp"].astype(np.float32)).all():
            problems.append(f"{key}: ranks below 1 or non-finite log-probs")
        if not (a["ranks"][:, -1] == a["ranks"][0, -1]).all():
            problems.append(f"{key}: final layer differs across lenses")
        if reference is not None:
            ref = reference.get(key)
            top1 = M.final_top1_at_last_prompt(a, p)
            if ref is None:
                problems.append(f"{key}: no reference read")
            elif int(ref["final_top1_id"]) != top1:
                problems.append(f"{key}: final-layer top-1 at the last prompt token {top1} != laptop fp32 {ref['final_top1_id']}")
    return problems


def load_reference(path: str | None) -> dict | None:
    if not path:
        return None
    ref = read_json(Path(path) / "reference.json")
    if ref is None:
        raise SystemExit(f"exp45_maps: no reference.json under {path}")
    return ref["cases"]


# ---------------------------------------------------------------------------------------------- main

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", type=int, choices=(1, 3), default=1)
    p.add_argument("--lens56", default="runs/exp34_jlens/lenses/qwen3b.pt")
    p.add_argument("--lens-refit", default=None)
    p.add_argument("--root", default=str(ROOT_OUT), help="outputs under <root>/<lens set>/stage<n>/")
    p.add_argument("--out", default=None, help="explicit output directory (smoke, reference)")
    p.add_argument("--cases", default=str(CASES))
    p.add_argument("--rows44", default=str(ROWS44))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--smoke", type=int, default=0, help="read the first N smoke cases and check them (exit 5 on failure)")
    p.add_argument("--reference", default=None, help="smoke: the laptop fp32 reference directory")
    p.add_argument("--write-reference", action="store_true", help="laptop: the 3 smoke cases, logit lens, -> --out")
    p.add_argument("--max-new", type=int, default=M.MAX_NEW, help="F1 new tokens (16)")
    p.add_argument("--max-new-note", type=int, default=M.MAX_NEW_NOTE,
                   help="F2 and stage 3: at most this many new tokens, stopping at the end of the Answer line (160)")
    p.add_argument("--k", type=int, default=M.K)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="auto", help="auto: bfloat16 on CUDA, float32 on CPU")
    p.add_argument("--weights", default=None, help="stub: a local tiny model directory")
    p.add_argument("--deadline", type=float, default=None, help="unix time; stop between prompts before it (exit 9)")
    p.add_argument("--allow-lens-sha", action="store_true", help="stub: do not require the §34 lens sha")
    p.add_argument("--break-bos", action="store_true", help="D-30 negative test: the smoke must fail")
    args = p.parse_args(argv)

    import torch

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    dtype = ("bfloat16" if device == "cuda" else "float32") if args.dtype == "auto" else args.dtype
    if args.write_reference:
        lens56, refit = None, None
    else:
        lens56 = args.lens56
        if not args.allow_lens_sha and file_sha256(lens56) != LENS56_SHA:
            print(f"exp45_maps: {lens56} is not the §34 56-prompt lens (sha {LENS56_SHA[:8]}...)", file=sys.stderr)
            return EXIT_LENS
        ok, why = refit_usable(args.lens_refit)
        refit = args.lens_refit if ok else None
        print(f"exp45_maps: refit lens {'used' if ok else 'not used'}: {why}", flush=True)

    if args.stage == 3 and not (args.smoke or args.write_reference):
        items = load_jsonl(args.rows44)
    else:
        items = ordered_cases(load_jsonl(args.cases))
    if args.smoke or args.write_reference:
        by = {c["case_id"]: c for c in items}
        items = [by[c] for c in SMOKE_CASES[: (args.smoke or len(SMOKE_CASES))]]
    elif args.limit:
        items = items[: args.limit]

    hf, lens_model, tok, load_s = load(device, dtype, args.weights)
    lenses, shas = load_lenses(lens56, refit, lens_model, device)
    names = [n for n in M.LENS_NAMES if n in lenses]
    stamp = {"maps_version": MAPS_VERSION, "lenses": shas, "max_new": args.max_new, "max_new_note": args.max_new_note, "k": args.k,
             "model": args.weights or MODEL_ID, "dtype": dtype, "device": device}
    if args.out:
        out = Path(args.out)
    else:
        out = Path(args.root) / lens_set_name(names) / f"stage{args.stage}"
    out.mkdir(parents=True, exist_ok=True)
    build = case_inputs if (args.stage == 1 or args.smoke or args.write_reference) else row44_inputs
    failed = read_json(out / "_failed.json", {}) or {}
    started, seconds, payloads, arrays_list, stopped = time.perf_counter(), {}, [], [], False
    for number, item in enumerate(items):
        x, meta = build(tok, item, args.max_new, args.max_new_note)
        stamp_i = dict(stamp, chat_sha256=M.sha256_text(x.chat_text))
        if not (args.smoke or args.write_reference) and is_done(out, x.key, stamp_i):
            continue
        if args.deadline is not None and seconds:
            per = max(seconds.values())
            if time.time() + 2 * per > args.deadline:
                print(f"exp45_maps: deadline: stopping before {x.key} ({len(seconds)} read this run)", flush=True)
                stopped = True
                break
        try:
            payload = run_one(hf, lens_model, tok, x, meta, lenses, stamp_i, out, args.k, break_bos=args.break_bos)
        except (ValueError, RuntimeError) as error:
            failed[x.key] = repr(error)
            write_json(out / "_failed.json", failed)
            print(f"[{number + 1}/{len(items)}] {x.key} FAILED {error!r}", flush=True)
            continue
        failed.pop(x.key, None)
        if (out / "_failed.json").exists():
            write_json(out / "_failed.json", failed)
        seconds[x.key] = payload["elapsed_s"]
        print(f"[{number + 1}/{len(items)}] {x.key} {payload['elapsed_s']:.2f}s n_tokens={payload['n_tokens']} "
              f"positions={len(payload['position_tokens'])}", flush=True)
        if args.smoke or args.write_reference:
            payloads.append(payload)
            arrays_list.append(dict(np.load(out / f"{x.key}.npz")))
    summary = {"stage": args.stage, "lens_set": lens_set_name(names), "device": device, "dtype": dtype,
               "prompts": len(seconds), "seconds_per_prompt": (sum(seconds.values()) / len(seconds)) if seconds else None,
               "load_s": load_s, "wall_s": time.perf_counter() - started, "stopped_at_deadline": stopped,
               "n_failed": len(failed), "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}
    append_timing(out, summary)
    print(json.dumps(summary), flush=True)

    if args.write_reference:
        ref = {"device": device, "dtype": dtype, "model": args.weights or MODEL_ID,
               "cases": {p_["key"]: {"final_top1_id": M.final_top1_at_last_prompt(a, p_), "generated_ids": p_["generated_ids"],
                                     "generated_text": p_["generated_text"]} for p_, a in zip(payloads, arrays_list)}}
        write_json(out / "reference.json", ref)
        print("reference", json.dumps({k: v["final_top1_id"] for k, v in ref["cases"].items()}), flush=True)
        return 0
    if args.smoke:
        n_layers = lens_model.n_layers
        problems = smoke_problems(payloads, arrays_list, n_layers, load_reference(args.reference), names)
        if len(payloads) != len(items):
            problems.append(f"{len(payloads)} of {len(items)} smoke cases read")
        result = {"pass": not problems, "problems": problems, "n": len(payloads), "lens_set": lens_set_name(names),
                  "seconds_per_prompt": summary["seconds_per_prompt"], "load_s": load_s,
                  "top1": {p_["key"]: M.final_top1_at_last_prompt(a, p_) for p_, a in zip(payloads, arrays_list)},
                  "n_tokens": {p_["key"]: p_["n_tokens"] for p_ in payloads}}
        write_json(out / "_smoke.json", result)
        print(f"smoke: {len(payloads)} cases, {summary['seconds_per_prompt'] and round(summary['seconds_per_prompt'], 2)} s/prompt, "
              f"{'PASS' if not problems else 'FAIL ' + '; '.join(problems)}", flush=True)
        return 0 if not problems else EXIT_SMOKE
    if stopped:
        return EXIT_DEADLINE
    if not seconds and failed:
        return EXIT_ALL_FAILED
    return 0


if __name__ == "__main__":
    sys.exit(main())
