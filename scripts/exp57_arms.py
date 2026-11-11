#!/usr/bin/env python
"""§57 stage-3 engine: arms with neuron resets over region N (note + marker positions before P_ans) and/or region P
(P_ans only), plus an optional step-0 logit bias (the dial B1), generated greedily in the §51 prefix mode.

Batched route: the prompt (positions < n_prompt, identical for every arm) is prefilled once; its KV is copied to a
batch of B arms; the generated part through P_ans − 1 is re-run for the whole batch with each row's region-N reset;
step 0 (P_ans) applies each row's region-P reset and bias; then greedy steps follow. No padding (same length).
Finished rows keep being fed; each output is cut at its own stop.

`reference` is batch 1 with a full prefill per arm (no KV reuse): the comparison path for the stage-0 equality test
(proposal §5 stage 0 item 4: identical greedy outputs on >= 99% of row-arms, |ΔD| <= 1e-3 at P_ans).

Arm: {"name", "reset_N": [(layer, idx, vals), ...] | None, "reset_P": [...] | None, "bias": {"ids", "value"} | None}.
D0 = logit[unk] − logit[t*] at P_ans before the bias.
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
import exp57_common as X  # noqa: E402


def _hooks(e, state, arms, n0, p):
    """Per-layer down_proj pre-hooks: batch row b gets its N reset on [n0, p) and its P reset on [p, p+1)."""
    torch = e.torch
    per_l: dict = {}
    for b, s in enumerate(arms):
        for key, plo, phi in (("reset_N", n0, p), ("reset_P", p, p + 1)):
            for l, idx, vals in (s.get(key) or []):
                per_l.setdefault(int(l), []).append((b, plo, phi, torch.as_tensor(np.asarray(idx), dtype=torch.long),
                                                     torch.as_tensor(np.asarray(vals, np.float32))))
    hs = []
    for l, items in per_l.items():
        def fn(module, args, items=items):
            x = args[0]
            lo, T = state["lo"], x.shape[1]
            hit = [(b, max(plo, lo) - lo, min(phi, lo + T) - lo, idx, vals) for b, plo, phi, idx, vals in items
                   if max(plo, lo) < min(phi, lo + T)]
            if not hit:
                return None
            x = x.clone()
            for b, s, t, idx, vals in hit:
                x[b, s:t, idx] = vals[None, :].expand(t - s, -1)
            return (x,) + tuple(args[1:])
        hs.append(e.layers[l].mlp.down_proj.register_forward_pre_hook(fn))
    return hs


def _bias(lg, arms):
    for b, s in enumerate(arms):
        if s.get("bias"):
            lg[b, list(s["bias"]["ids"])] += float(s["bias"]["value"])
    return lg


def _done(e, gen):
    t = e.decode(gen)
    return "\n" in t and bool(t.strip())


def run_arms(e, rec, arms, unk_id, t_star, cache_prompt=None, max_new=EN.MAX_NEW_RELEASE):
    torch = e.torch
    ids = CM.prefix_ids(rec)
    n0, p, _ = X.regions(rec)
    B = len(arms)
    state = {"lo": 0}
    if cache_prompt is None:
        with torch.no_grad():
            cache_prompt = e.net(input_ids=torch.tensor([ids[:n0]]), use_cache=True, logits_to_keep=1).past_key_values
    c = copy.deepcopy(cache_prompt)
    c.batch_repeat_interleave(B)
    hs = _hooks(e, state, arms, n0, p)
    gens, done, stops = [[] for _ in range(B)], [False] * B, ["max_new"] * B
    try:
        with torch.no_grad():
            state["lo"] = n0
            out = e.net(input_ids=torch.tensor([ids[n0:p]] * B), past_key_values=c, use_cache=True, logits_to_keep=1)
            c = out.past_key_values
            tok, pos, step, D0 = torch.full((B, 1), ids[p], dtype=torch.long), p, 0, None
            while not all(done):
                state["lo"] = pos
                out = e.net(input_ids=tok, past_key_values=c, use_cache=True, logits_to_keep=1)
                c = out.past_key_values
                lg = out.logits[:, -1].clone()
                if step == 0:
                    D0 = (lg[:, unk_id] - lg[:, t_star]).tolist()
                    lg = _bias(lg, arms)
                nxt = lg.argmax(-1).tolist()
                for b in range(B):
                    if done[b]:
                        continue
                    if nxt[b] in e.stop_ids:
                        done[b], stops[b] = True, "eos"; continue
                    gens[b].append(nxt[b])
                    if _done(e, gens[b]):
                        done[b], stops[b] = True, "newline"
                    elif len(gens[b]) >= max_new:
                        done[b] = True
                tok = torch.tensor(nxt, dtype=torch.long)[:, None]
                pos += 1; step += 1
    finally:
        for h in hs:
            h.remove()
    return [{"name": s["name"], "gen_ids": g, "text": e.decode(g), "stop": st, "D0": d}
            for s, g, st, d in zip(arms, gens, stops, D0)]


def reference(e, rec, arm, unk_id, t_star, max_new=EN.MAX_NEW_RELEASE):
    torch = e.torch
    ids = CM.prefix_ids(rec)
    n0, p, _ = X.regions(rec)
    state = {"lo": 0}
    hs = _hooks(e, state, [arm], n0, p)
    gen, stop, D0 = [], "max_new", None
    try:
        with torch.no_grad():
            out = e.net(input_ids=torch.tensor([ids[:p]]), use_cache=True, logits_to_keep=1)
            c = out.past_key_values
            tok, pos, step = ids[p], p, 0
            while True:
                state["lo"] = pos
                out = e.net(input_ids=torch.tensor([[tok]]), past_key_values=c, use_cache=True, logits_to_keep=1)
                c = out.past_key_values
                lg = out.logits[:, -1].clone()
                if step == 0:
                    D0 = float(lg[0, unk_id] - lg[0, t_star])
                    lg = _bias(lg, [arm])
                nxt = int(lg[0].argmax())
                if nxt in e.stop_ids:
                    stop = "eos"; break
                gen.append(nxt)
                if _done(e, gen):
                    stop = "newline"; break
                if len(gen) >= max_new:
                    break
                tok, pos, step = nxt, pos + 1, step + 1
    finally:
        for h in hs:
            h.remove()
    return {"name": arm["name"], "gen_ids": gen, "text": e.decode(gen), "stop": stop, "D0": D0}


def compare(e, rec, arms, unk_id, t_star):
    t0 = time.time()
    bat = run_arms(e, rec, arms, unk_id, t_star)
    tb = time.time() - t0
    t1 = time.time()
    ref = [reference(e, rec, a, unk_id, t_star) for a in arms]
    tr = time.time() - t1
    rows = [{"arm": a["name"], "same": x["gen_ids"] == r["gen_ids"], "dD": abs(x["D0"] - r["D0"]),
             "batched": x["text"], "ref": r["text"]} for a, x, r in zip(arms, bat, ref)]
    return rows, tb, tr
