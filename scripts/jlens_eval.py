#!/usr/bin/env python
"""§34: lens quality on the jacobian-lens package's six evaluation sets (J-lens and logit lens, one forward).

    .venv/bin/python scripts/jlens_eval.py --reader qwen05b --lens runs/exp34_jlens/lenses/qwen05b.pt --sets multihop,association --limit 20 --device cpu --dtype float32

Conventions (external/jacobian-lens/data/evaluations/README.md, as fixed by the §34 plan):
- Readout is at one position across all layers. multihop/multilingual/association/typo/order-ops: the token
  immediately preceding `target` when `target` is present (the prompt is the text, position
  len(ids(prompt)) - 1), else the last prompt token. poetry: the last newline token (end of line 1).
- A chat-list prompt ([{"role","content"}]) is rendered with the tokenizer's chat template, no generation prompt.
- An intermediate's token id is the first content token of " " + intermediate (the gold rule of
  memctl.analysis.jlens_readout.tracked_token_ids). `target` is never scored.
- pass@k per item = fraction of intermediates whose min rank over the band of layers is <= k; a set's
  pass@k is the mean over items. The band is [n_layers//3, 2*n_layers//3) by default (--band a:b overrides),
  taken over the lens layers plus the final layer (the final layer is outside the default band).
  Deviation: the README's order-ops synonym expansion is not applied (the data holds only the keys).
- Items whose readout text exceeds MAX_SEQ_LEN tokens are skipped and counted.

Output: runs/exp34_jlens/eval/<reader>.json. Never deletes anything.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memctl.analysis.jlens_readout import _token_at_char, rank_table, tracked_token_ids  # noqa: E402

MODEL_IDS = {"qwen05b": "Qwen/Qwen2.5-0.5B-Instruct", "qwen3b": "Qwen/Qwen2.5-3B-Instruct", "qwen7b": "Qwen/Qwen2.5-7B-Instruct"}
SETS = ("association", "multihop", "multilingual", "order-ops", "poetry", "typo")
DATA = Path("external/jacobian-lens/data/evaluations")
MAX_SEQ_LEN = 4096


def _ids(tokenizer, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False).input_ids


def readout_position(tokenizer, item: dict, set_name: str) -> tuple[str, int]:
    prompt = item["prompt"]
    if isinstance(prompt, list):
        text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=False)
    else:
        text = prompt
    if set_name == "poetry":
        offsets = [tuple(o) for o in tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]]
        return text, _token_at_char(offsets, text.rindex("\n"))
    return text, len(_ids(tokenizer, text)) - 1          # the token before `target` (or the last token)


def pass_at_k(ranks_by_layer: np.ndarray, band_mask: np.ndarray, k: int) -> float:
    best = np.asarray(ranks_by_layer)[np.asarray(band_mask, bool)].min(axis=0)     # [n_intermediates]
    return float(np.mean(best <= k))


def lens_to(lens, device) -> None:
    lens.jacobians = {layer: J.to(device) for layer, J in lens.jacobians.items()}


def logits_at(lens, lens_model, text: str, position: int):
    """One forward; J-lens and logit-lens logits [vocab] per lens layer at `position`, plus the model logits."""
    import torch
    from jlens.hooks import ActivationRecorder

    input_ids = lens_model.encode(text, max_length=MAX_SEQ_LEN)
    if input_ids[0].tolist() != _ids(lens_model.tokenizer, text):
        raise ValueError("lens encode differs from the tokenizer ids; the readout position would be off")
    final = lens_model.n_layers - 1
    layers = sorted(set(lens.source_layers) | {final})
    with torch.no_grad(), ActivationRecorder(lens_model.layers, at=layers) as recorder:
        lens_model.forward(input_ids)
        residuals = {l: recorder.activations[l][0][position:position + 1].float() for l in layers}
    jl, ll = {}, {}
    with torch.no_grad():
        for l in layers:
            if l == final:
                continue
            jl[l] = lens_model.unembed(lens.transport(residuals[l], l)).float().cpu().numpy()[0]
            ll[l] = lens_model.unembed(residuals[l]).float().cpu().numpy()[0]
        model_logits = lens_model.unembed(residuals[final]).float().cpu().numpy()[0]
    return jl, ll, model_logits


def score_set(lens, lens_model, tokenizer, items: list[dict], set_name: str, band: tuple[int, int], k: int) -> dict:
    scores, skipped = {"jlens": [], "logit": []}, 0
    for item in items:
        text, position = readout_position(tokenizer, item, set_name)
        if len(_ids(tokenizer, text)) > MAX_SEQ_LEN:
            skipped += 1
            continue
        token_ids = [tracked_token_ids(tokenizer, w)[0]["gold"] for w in item["intermediates"]]
        jl, ll, model_logits = logits_at(lens, lens_model, text, position)
        for variant, logits in (("jlens", jl), ("logit", ll)):
            ranks, layers = rank_table({l: v[None] for l, v in logits.items()}, model_logits[None], lens_model.n_layers, token_ids)
            mask = np.array([band[0] <= l < band[1] for l in layers])
            scores[variant].append(pass_at_k(ranks[:, 0, :], mask, k))
    return {"n_items": len(scores["jlens"]), "n_skipped": skipped,
            "pass_at_k": {v: (float(np.mean(s)) if s else None) for v, s in scores.items()}}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reader", required=True, choices=sorted(MODEL_IDS))
    p.add_argument("--lens", required=True)
    p.add_argument("--sets", default="all")
    p.add_argument("--band", default="auto", help="auto = [n_layers//3, 2*n_layers//3), or a:b")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--out", default="runs/exp34_jlens/eval")
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--model-id", default=None)
    p.add_argument("--limit", type=int, default=None, help="first N items per set")
    args = p.parse_args(argv)
    names = list(SETS) if args.sets == "all" else args.sets.split(",")
    unknown = [n for n in names if n not in SETS]
    if unknown:
        p.error(f"unknown sets: {unknown}")
    import jlens
    import torch
    import transformers

    model_id = args.model_id or MODEL_IDS[args.reader]
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    hf = transformers.AutoModelForCausalLM.from_pretrained(model_id, dtype=getattr(torch, args.dtype)).to(device)
    hf.eval()
    tok = transformers.AutoTokenizer.from_pretrained(model_id)
    lens_model = jlens.from_hf(hf, tok)
    lens = jlens.JacobianLens.load(args.lens)
    lens_to(lens, device)
    n = lens_model.n_layers
    band = (n // 3, 2 * n // 3) if args.band == "auto" else tuple(int(x) for x in args.band.split(":"))
    started = time.perf_counter()
    sets = {}
    for name in names:
        items = json.loads((DATA / f"lens-eval-{name}.json").read_text())["items"]
        sets[name] = score_set(lens, lens_model, tok, items[: args.limit] if args.limit else items, name, band, args.k)
        print(name, json.dumps(sets[name]), flush=True)
    mean = {v: float(np.mean([s["pass_at_k"][v] for s in sets.values() if s["pass_at_k"][v] is not None]))
            for v in ("jlens", "logit")}
    payload = {"reader": args.reader, "lens": args.lens, "k": args.k, "band": list(band), "sets": sets,
               "mean_pass_at_k": mean, "model_id": model_id, "device": device, "dtype": args.dtype,
               "limit": args.limit, "wall_s": time.perf_counter() - started}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.reader}.json"
    if path.exists():                                            # keep an earlier result, never overwrite it
        os.replace(path, out / f"{args.reader}.{int(path.stat().st_mtime)}.json")
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=1))
    os.replace(tmp, path)
    print(json.dumps({"out": str(path), "band": list(band), "mean_pass_at_k": mean}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
