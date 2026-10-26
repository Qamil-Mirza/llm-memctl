#!/usr/bin/env python
"""§34: lens quality on the jacobian-lens package's six evaluation sets (J-lens and logit lens, one forward).

    .venv/bin/python scripts/jlens_eval.py --reader qwen05b --lens runs/exp34_jlens/lenses/qwen05b.pt --sets multihop,association --limit 20 --device cpu --dtype float32

Conventions (external/jacobian-lens/data/evaluations/README.md, as fixed by the §34 plan):
- Readout is at one position across all layers. multihop/multilingual/association/typo/order-ops: trailing
  whitespace is stripped from the prompt (from the last turn's content for a chat list), so the position
  len(ids(text)) - 1 is the last word token, the token immediately preceding `target` under natural
  tokenisation (the space joins the target word). poetry: the last newline token (end of line 1), unstripped.
- A chat-list prompt ([{"role","content"}]) is rendered with the tokenizer's chat template, no generation prompt.
- An intermediate's token id is the first content token of " " + intermediate (the gold rule of
  memctl.analysis.jlens_readout.tracked_token_ids). `target` is never scored.
- pass@k per item = fraction of intermediates whose min rank over the band of layers is <= k; a set's
  pass@k is the mean over items. The band is [n_layers//3, 2*n_layers//3) by default (--band a:b overrides),
  taken over the lens layers plus the final layer (the final layer is outside the default band).
  pass_at_k_all is the README's metric: min over every one of those layers, final included.
  Deviation: the README's order-ops synonym expansion is not applied (the data holds only the keys).
- Items whose readout text exceeds MAX_SEQ_LEN tokens are skipped and counted.

Output: runs/exp34_jlens/eval/<reader>.json, with the lens file's lens_sha256 and the --limit used (the report
flags an eval whose hash differs from the reads' or whose limit is not null). Never deletes anything.
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

from memctl.analysis import families  # noqa: E402
from memctl.analysis.jlens_readout import (_token_at_char, encode_chat_ids, file_sha256, rank_table,  # noqa: E402
                                           tracked_token_ids)

MODEL_IDS = {"qwen05b": "Qwen/Qwen2.5-0.5B-Instruct", "qwen3b": "Qwen/Qwen2.5-3B-Instruct", "qwen7b": "Qwen/Qwen2.5-7B-Instruct"}
FAMILY_READERS = ("llama3b", "gemma4b", "phi4mini")       # §43 (memctl/analysis/families.py)
SETS = ("association", "multihop", "multilingual", "order-ops", "poetry", "typo")
DATA = Path("external/jacobian-lens/data/evaluations")
MAX_SEQ_LEN = 4096


def _ids(tokenizer, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False).input_ids


def readout_position(tokenizer, item: dict, set_name: str) -> tuple[str, int]:
    prompt, strip = item["prompt"], set_name != "poetry"
    if isinstance(prompt, list):
        if strip:
            prompt = prompt[:-1] + [{**prompt[-1], "content": prompt[-1]["content"].rstrip()}]
        text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=False)
    else:
        text = prompt.rstrip() if strip else prompt
    if set_name == "poetry":
        offsets = [tuple(o) for o in tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]]
        return text, _token_at_char(offsets, text.rindex("\n"))
    return text, len(_ids(tokenizer, text)) - 1          # the last word token: the space joins `target` naturally


def pass_at_k(ranks_by_layer: np.ndarray, band_mask: np.ndarray, k: int) -> float:
    best = np.asarray(ranks_by_layer)[np.asarray(band_mask, bool)].min(axis=0)     # [n_intermediates]
    return float(np.mean(best <= k))


def lens_to(lens, device) -> None:
    lens.jacobians = {layer: J.to(device) for layer, J in lens.jacobians.items()}


def eval_ids(tokenizer, text: str, raw_bos: bool) -> tuple[list[int], int]:
    """(ids, offset) for an evaluation text (§43 B1): encoded without special tokens, so a chat-template text keeps
    exactly its own BOS (Llama, Gemma) or none (Qwen, Phi). A raw (non-chat) text gets the tokenizer's BOS prepended
    when `raw_bos` (the family's fit-time flag, so it is read the way the lens was fitted) and the tokenizer has
    one; `offset` (0 or 1) shifts the readout position. Qwen has no BOS token: ids == _ids(text), offset 0."""
    ids = encode_chat_ids(tokenizer, text, MAX_SEQ_LEN)
    bos = getattr(tokenizer, "bos_token_id", None)
    if raw_bos and bos is not None and not (ids and ids[0] == bos):
        return [bos] + ids, 1
    return ids, 0


def logits_at(lens, lens_model, text: str, position: int, raw_bos: bool = True):
    """One forward; J-lens and logit-lens logits [vocab] per lens layer at `position` (an index into _ids(text)),
    plus the model logits."""
    import torch
    from jlens.hooks import ActivationRecorder

    ids, offset = eval_ids(lens_model.tokenizer, text, raw_bos)
    if ids[offset:] != _ids(lens_model.tokenizer, text):
        raise ValueError("lens encode differs from the tokenizer ids; the readout position would be off")
    input_ids = torch.tensor([ids], dtype=torch.long, device=lens_model.input_device)
    position = position + offset
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


def score_set(lens, lens_model, tokenizer, items: list[dict], set_name: str, band: tuple[int, int], k: int,
              raw_bos: bool = True) -> dict:
    scores, skipped = {f"{v}{m}": [] for v in ("jlens", "logit") for m in ("", "_all")}, 0
    for item in items:
        text, position = readout_position(tokenizer, item, set_name)
        if len(_ids(tokenizer, text)) > MAX_SEQ_LEN:
            skipped += 1
            continue
        token_ids = [tracked_token_ids(tokenizer, w)[0]["gold"] for w in item["intermediates"]]
        jl, ll, model_logits = logits_at(lens, lens_model, text, position, raw_bos)
        for variant, logits in (("jlens", jl), ("logit", ll)):
            ranks, layers = rank_table({l: v[None] for l, v in logits.items()}, model_logits[None], lens_model.n_layers, token_ids)
            mask = np.array([band[0] <= l < band[1] for l in layers])
            scores[variant].append(pass_at_k(ranks[:, 0, :], mask, k))
            scores[f"{variant}_all"].append(pass_at_k(ranks[:, 0, :], np.ones(len(layers), bool), k))   # README metric
    mean = {key: (float(np.mean(s)) if s else None) for key, s in scores.items()}
    return {"n_items": len(scores["jlens"]), "n_skipped": skipped,
            "pass_at_k": {v: mean[v] for v in ("jlens", "logit")},
            "pass_at_k_all": {v: mean[f"{v}_all"] for v in ("jlens", "logit")}}


def load_model(model_id: str, device: str, dtype: str, family=None, weights: str | None = None):
    """(jlens LensModel, tokenizer) for a Hugging Face model id (a §43 family: weights may be a local copy)."""
    import jlens
    import torch
    import transformers

    hf = transformers.AutoModelForCausalLM.from_pretrained(weights or model_id, dtype=getattr(torch, dtype)).to(device)
    hf.eval()
    tok = transformers.AutoTokenizer.from_pretrained(model_id)
    if family is None:
        return jlens.from_hf(hf, tok), tok
    return jlens.from_hf(hf, tok, force_bos=family.lens_force_bos), tok


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reader", required=True, choices=sorted(MODEL_IDS) + list(FAMILY_READERS))
    p.add_argument("--weights", default=None, help="§43 families: a local copy of the weights")
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

    family = families.get(args.reader) if args.reader in FAMILY_READERS else None
    model_id = args.model_id or (family.model_id if family is not None else MODEL_IDS[args.reader])
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    lens_model, tok = (load_model(model_id, device, args.dtype) if family is None
                       else load_model(model_id, device, args.dtype, family=family, weights=args.weights))
    raw_bos = True if family is None else family.lens_force_bos
    lens = jlens.JacobianLens.load(args.lens)
    lens_to(lens, device)
    n = lens_model.n_layers
    band = (n // 3, 2 * n // 3) if args.band == "auto" else tuple(int(x) for x in args.band.split(":"))
    started = time.perf_counter()
    sets = {}
    for name in names:
        items = json.loads((DATA / f"lens-eval-{name}.json").read_text())["items"]
        sets[name] = score_set(lens, lens_model, tok, items[: args.limit] if args.limit else items, name, band, args.k,
                               raw_bos)
        print(name, json.dumps(sets[name]), flush=True)
    mean, mean_all = ({v: float(np.mean([s[key][v] for s in sets.values() if s[key][v] is not None]))
                       for v in ("jlens", "logit")} for key in ("pass_at_k", "pass_at_k_all"))
    payload = {"reader": args.reader, "lens": args.lens, "lens_sha256": file_sha256(args.lens), "k": args.k, "band": list(band), "sets": sets,
               "mean_pass_at_k": mean, "mean_pass_at_k_all": mean_all, "model_id": model_id, "device": device, "dtype": args.dtype,
               "limit": args.limit, "wall_s": time.perf_counter() - started}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.reader}.json"
    if path.exists():                                            # keep an earlier result, never overwrite it
        os.replace(path, out / f"{args.reader}.{int(path.stat().st_mtime)}.json")
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=1))
    os.replace(tmp, path)
    print(json.dumps({"out": str(path), "band": list(band), "mean_pass_at_k": mean, "mean_pass_at_k_all": mean_all}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
