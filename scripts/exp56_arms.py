#!/usr/bin/env python
"""§56 stage-2 engine (rev 0a, df5d028): one prefill per row, the KV copied per arm, the arms run as one batch with a
per-sequence neuron ablation at P_ans (step 0) only, and an optional per-sequence logit bias at step 0 only.

Exactness: the ablation and the bias act only at P_ans, so the KV over positions < P_ans is the same for every arm.
Batch rows share no padding (same prefix length) and attend only to their own cache, so a finished row can keep
being fed without touching the others; its output is cut at its own stop.

Arm spec: {"name": str, "ablate": {layer: (idx, values)} | None, "bias": {"ids": [...], "value": float} | None}.
Stop rule = the §51 prefix mode: EOS / <|im_end|>, or a newline after a non-empty answer, or MAX_NEW_RELEASE (16).

`reference` is the batch-1 path with a full prefill per arm (the check the batched route must pass: greedy outputs
identical on >= 99% of row-arms, |ΔD| <= 1e-3 at P_ans, every difference listed).
"""
from __future__ import annotations

import copy
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import exp51_common as CM  # noqa: E402
import exp51_engine as EN  # noqa: E402


def _stop(e, gen, text):
    return "\n" in text and bool(text.strip())


def _hooks(e, specs, p_ans):
    """Per-layer pre-hooks on down_proj: at the step whose offset is P_ans, row b gets its arm's ablation."""
    torch = e.torch
    per_layer: dict = {}
    for b, s in enumerate(specs):
        for l, (idx, vals) in (s.get("ablate") or {}).items():
            per_layer.setdefault(int(l), []).append((b, torch.as_tensor(np.asarray(idx), dtype=torch.long),
                                                     torch.as_tensor(np.asarray(vals, np.float32))))
    hs = []
    for l, items in per_layer.items():
        def fn(module, args, items=items):
            x = args[0]
            lo, T = e._offset, x.shape[1]
            if not (lo <= p_ans < lo + T):
                return None
            x = x.clone()
            for b, idx, vals in items:
                x[b, p_ans - lo, idx] = vals
            return (x,) + tuple(args[1:])
        hs.append(e.layers[l].mlp.down_proj.register_forward_pre_hook(fn))
    return hs


def _bias_step0(e, logits, specs):
    for b, s in enumerate(specs):
        if s.get("bias"):
            logits[b, list(s["bias"]["ids"])] += float(s["bias"]["value"])
    return logits


def run_arms(e, rec: dict, specs: list[dict], unk_id: int, t_star: int, cache=None, max_new=EN.MAX_NEW_RELEASE):
    """Batched route. Returns per arm: gen_ids, text (new tokens only), stop, D0 (step-0 logit[unk] - logit[t*],
    before any bias)."""
    torch = e.torch
    ids = CM.prefix_ids(rec)
    p_ans = len(ids) - 1
    assert p_ans == rec["p_ans"]
    if cache is None:
        cache = e._prefill(ids[:-1])
    B = len(specs)
    c = copy.deepcopy(cache)
    c.batch_repeat_interleave(B)
    hs = _hooks(e, specs, p_ans)
    gens, done, stops = [[] for _ in range(B)], [False] * B, ["max_new"] * B
    D0 = None
    try:
        tok = torch.full((B, 1), ids[-1], dtype=torch.long)
        pos = p_ans
        step = 0
        while not all(done):
            e._offset = pos
            with torch.no_grad():
                out = e.net(input_ids=tok, past_key_values=c, use_cache=True, logits_to_keep=1,
                            position_ids=torch.full((B, 1), pos, dtype=torch.long))
            lg = out.logits[:, -1].clone()
            if step == 0:
                D0 = (lg[:, unk_id] - lg[:, t_star]).tolist()
                lg = _bias_step0(e, lg, specs)
            nxt = lg.argmax(-1).tolist()
            for b in range(B):
                if done[b]:
                    continue
                if nxt[b] in e.stop_ids:
                    done[b], stops[b] = True, "eos"; continue
                gens[b].append(nxt[b])
                text = e.decode(gens[b])
                if _stop(e, gens[b], text):
                    done[b], stops[b] = True, "newline"
                elif len(gens[b]) >= max_new:
                    done[b] = True
            tok = torch.tensor(nxt, dtype=torch.long)[:, None]
            pos += 1; step += 1
    finally:
        for h in hs:
            h.remove()
    return [{"name": s["name"], "gen_ids": g, "text": e.decode(g), "stop": st, "D0": d}
            for s, g, st, d in zip(specs, gens, stops, D0)]


def reference(e, rec: dict, spec: dict, unk_id: int, t_star: int, max_new=EN.MAX_NEW_RELEASE):
    """Batch 1, full prefill for this arm (no KV reuse): the comparison path."""
    torch = e.torch
    ids = CM.prefix_ids(rec)
    p_ans = len(ids) - 1
    hs = _hooks(e, [spec], p_ans)
    gen, stop, D0 = [], "max_new", None
    try:
        cache = e._prefill(ids[:-1])
        tok, pos, step = ids[-1], p_ans, 0
        while True:
            lg = e._step(tok, cache, pos).clone()[None]
            if step == 0:
                D0 = float(lg[0, unk_id] - lg[0, t_star])
                lg = _bias_step0(e, lg, [spec])
            nxt = int(lg[0].argmax())
            if nxt in e.stop_ids:
                stop = "eos"; break
            gen.append(nxt)
            if _stop(e, gen, e.decode(gen)):
                stop = "newline"; break
            if len(gen) >= max_new:
                break
            tok, pos, step = nxt, pos + 1, step + 1
    finally:
        for h in hs:
            h.remove()
    return {"name": spec["name"], "gen_ids": gen, "text": e.decode(gen), "stop": stop, "D0": D0}


def compare(e, rec, specs, unk_id, t_star):
    t0 = time.time()
    bat = run_arms(e, rec, specs, unk_id, t_star)
    tb = time.time() - t0
    t1 = time.time()
    ref = [reference(e, rec, s, unk_id, t_star) for s in specs]
    tr = time.time() - t1
    rows = [{"arm": s["name"], "same": a["gen_ids"] == r["gen_ids"], "dD": abs(a["D0"] - r["D0"]),
             "batched": a["text"], "ref": r["text"]} for s, a, r in zip(specs, bat, ref)]
    return rows, tb, tr
