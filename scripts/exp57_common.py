#!/usr/bin/env python
"""§57 shared pieces (Qwen 2.5 3B, §51 engine, laptop fp32, $0): attribution patching of every SwiGLU neuron's reset
effect on D in two regions, and the exact (real) patch used to check it.

Spec: research/proposals/2026-11-10-exp57-differential-neurons.md rev 0 (9e51a1b). Rows, classes, t* and D are §56's
(freeze 55d3d87b, rev 0a): D = logit(" unknown") − logit(t*) at P_ans, t* = the best city first token of the row's pool.

Fixed choices of this build (stage-0 freeze):
- Forward ids = CM.prefix_ids(rec) = prompt ids + the row's own generated note and marker through the ':' (P_ans = the
  last position). The note is forced: it is the model's own stored P0 note.
- Regions. P = {P_ans}. N = every generated position before P_ans, i.e. positions n_prompt .. P_ans − 1: the note text
  and the "\\nAnswer" marker tokens (the ':' itself is P_ans).
- Neuron activation a_i(t) = the input of layer l's mlp.down_proj at position t (silu(gate)·up).
- Reset values: region P uses ā^C_P = the dev-A C-row mean at P_ans (§56 runs/exp56/c_mean_act.npy, 7d78ee0); region N
  uses ā^C_N = the dev-A C-row mean over all N positions of all dev-A C rows, token-weighted (`note_means`).
  One value per neuron and region; a reset sets a_i(t) to it at every position of the region.
- Predicted effect Δ̂_i = Σ_{t∈R} (ā^C_i − a_i(t)) · ∂D/∂a_i(t), from one no-cache forward and one backward on D,
  t* held at its clean value. Parameters carry no gradient; the graph starts at the embedding output, or, when
  `grad_from` > 0, at the residual entering layer `grad_from` (detached below it; Δ̂ below it is not computed).
- Real effect: ΔD = D(reset) − D(clean), both on the same cached path (prompt prefilled once; for region N the note
  is re-run with the reset in place, for region P only the P_ans step is re-run). t* held at its clean value.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import exp51_common as CM  # noqa: E402
import exp56_common as X56  # noqa: E402

OUT = ROOT / "runs/exp57"
C_MEAN_P = ROOT / "runs/exp56/c_mean_act.npy"
C_MEAN_N = OUT / "c_mean_note.npy"


def regions(rec: dict) -> tuple[int, int, int]:
    """(n_prompt, p_ans, n_ids): region N = [n_prompt, p_ans), region P = [p_ans]."""
    ids = CM.prefix_ids(rec)
    n0, p = len(rec["prompt_ids"]), len(ids) - 1
    assert p == rec["p_ans"] and n0 < p
    return n0, p, len(ids)


def attrib(a, g, reset, lo: int, hi: int):
    """Δ̂ over positions [lo, hi): Σ_t (reset − a[t]) · g[t]. a, g: [T, I] tensors; reset: [I]."""
    return ((reset[None, :] - a[lo:hi]) * g[lo:hi]).sum(0)


class Attributor:
    """One no-cache forward with the graph from the embedding (or from layer `grad_from`), one backward on D."""

    def __init__(self, e, reset_p: np.ndarray, reset_n: np.ndarray | None, grad_from: int = 0):
        self.e, self.torch = e, e.torch
        for p in e.net.parameters():
            p.requires_grad_(False)
        self.unk_id = e.encode(X56.UNK_TEXT)[0]
        self.grad_from = int(grad_from)
        T = self.torch
        self.reset_p = T.as_tensor(np.asarray(reset_p, np.float32)).reshape(e.n_layers, -1)
        self.reset_n = None if reset_n is None else T.as_tensor(np.asarray(reset_n, np.float32)).reshape(e.n_layers, -1)

    def run(self, rec: dict, cand_ids: list[int], keep_ag: bool = False) -> dict:
        torch, e = self.torch, self.e
        ids = CM.prefix_ids(rec)
        n0, p, _ = regions(rec)
        cap: dict = {}
        hs = []
        lo_l = self.grad_from

        def start_graph(module, args, kwargs):
            if "hidden_states" in kwargs:
                kwargs = dict(kwargs); kwargs["hidden_states"] = kwargs["hidden_states"].detach().requires_grad_(True)
                return args, kwargs
            h = args[0].detach().requires_grad_(True)
            return (h,) + tuple(args[1:]), kwargs

        def emb_grad(module, args, out):
            return out.requires_grad_(True)

        if lo_l == 0:
            hs.append(e.net.model.embed_tokens.register_forward_hook(emb_grad))
        else:
            hs.append(e.layers[lo_l].register_forward_pre_hook(start_graph, with_kwargs=True))

        def pre_down(l):
            def fn(module, args):
                cap[l] = args[0]
            return fn
        for l in range(lo_l, e.n_layers):
            hs.append(e.layers[l].mlp.down_proj.register_forward_pre_hook(pre_down(l)))
        try:
            with torch.enable_grad():
                out = e.net(input_ids=torch.tensor([ids]), use_cache=False, logits_to_keep=1)
                lg = out.logits[0, -1]
                cand = torch.as_tensor(list(cand_ids), dtype=torch.long)
                t_star = int(cand[int(lg.detach()[cand].argmax())])
                D = lg[self.unk_id] - lg[t_star]
                layers = list(range(lo_l, e.n_layers))
                grads = torch.autograd.grad(D, [cap[l] for l in layers])
        finally:
            for h in hs:
                h.remove()
        I = self.reset_p.shape[1]
        dP = np.zeros((e.n_layers, I), np.float32)
        dN = np.zeros((e.n_layers, I), np.float32)
        aP = np.zeros((e.n_layers, I), np.float32)
        ag = {}
        with torch.no_grad():
            for l, g in zip(layers, grads):
                a, g = cap[l][0].detach(), g[0]
                dP[l] = attrib(a, g, self.reset_p[l], p, p + 1).numpy()
                if self.reset_n is not None:
                    dN[l] = attrib(a, g, self.reset_n[l], n0, p).numpy()
                aP[l] = a[p].numpy()
                if keep_ag:
                    ag[l] = (a.numpy().copy(), g.numpy().copy())
        res = {"D": float(D.detach()), "t_star": t_star, "n_prompt": n0, "p_ans": p, "dhat_P": dP, "dhat_N": dN,
               "a_P": aP, "grad_from": lo_l}
        if keep_ag:
            res["ag"] = ag
        return res


# ---------------------------------------------------------------------------------------------- real patches

class Patcher:
    """Exact reset effects on D for one row, on a cached path: the prompt is prefilled once; region-N resets re-run
    the generated part [n_prompt, P_ans]; region-P resets re-run only the P_ans step from a cache through P_ans − 1.

    spec: list of (layer, idx int array, values float array). alpha (default 1) moves each a toward its reset value
    by that fraction (alpha = 1 is the reset; small alpha is the linear regime, used by the tests)."""

    def __init__(self, e, rec: dict, unk_id: int, t_star: int):
        self.e, self.torch = e, e.torch
        self.ids = CM.prefix_ids(rec)
        self.n0, self.p, _ = regions(rec)
        self.unk_id, self.t_star = unk_id, t_star
        with self.torch.no_grad():
            self.c_prompt = self._run(None, self.ids[:self.n0], 0)[1]
            self.c_note = self._run(copy.deepcopy(self.c_prompt), self.ids[self.n0:self.p], self.n0)[1]
        self.D_clean_N = self.D("N", [])
        self.D_clean_P = self.D("P", [])

    def _run(self, cache, toks, lo, hooks=()):
        torch = self.torch
        hs = [self.e.layers[l].mlp.down_proj.register_forward_pre_hook(fn) for l, fn in hooks]
        try:
            with torch.no_grad():
                out = self.e.net(input_ids=torch.tensor([toks]), past_key_values=cache, use_cache=True,
                                 logits_to_keep=1)
        finally:
            for h in hs:
                h.remove()
        return out.logits[0, -1], out.past_key_values

    def _hooks(self, spec, lo, plo, phi, alpha):
        """Hooks for a chunk starting at absolute position lo: reset positions [plo, phi)."""
        torch = self.torch
        by_l: dict = {}
        for l, idx, vals in spec:
            by_l.setdefault(int(l), []).append((torch.as_tensor(np.asarray(idx), dtype=torch.long),
                                                torch.as_tensor(np.asarray(vals, np.float32))))
        out = []
        for l, items in by_l.items():
            def fn(module, args, items=items):
                x = args[0].clone()
                s, t = plo - lo, phi - lo
                for idx, vals in items:
                    cur = x[0, s:t][:, idx]
                    x[0, s:t, idx] = cur + alpha * (vals[None, :] - cur)
                return (x,) + tuple(args[1:])
            out.append((l, fn))
        return out

    def D(self, region: str, spec, alpha: float = 1.0) -> float:
        if region == "N":
            hooks = self._hooks(spec, self.n0, self.n0, self.p, alpha) if spec else ()
            lg, _ = self._run(copy.deepcopy(self.c_prompt), self.ids[self.n0:], self.n0, hooks)
        else:
            hooks = self._hooks(spec, self.p, self.p, self.p + 1, alpha) if spec else ()
            lg, _ = self._run(copy.deepcopy(self.c_note), self.ids[self.p:], self.p, hooks)
        return float(lg[self.unk_id] - lg[self.t_star])

    def effect(self, region: str, spec, alpha: float = 1.0) -> float:
        base = self.D_clean_N if region == "N" else self.D_clean_P
        return self.D(region, spec, alpha) - base


# ---------------------------------------------------------------------------------------------- ā^C over the note

def note_sums(e, rec: dict):
    """Sum over region-N positions of every neuron's activation, and the count. No-cache forward, no grad."""
    torch = e.torch
    ids = CM.prefix_ids(rec)
    n0, p, _ = regions(rec)
    acc = {}

    def mk(l):
        def fn(module, args):
            acc[l] = args[0][0, n0:p].sum(0, dtype=torch.float64)
        return fn
    hs = [e.layers[l].mlp.down_proj.register_forward_pre_hook(mk(l)) for l in range(e.n_layers)]
    try:
        with torch.no_grad():
            e.net(input_ids=torch.tensor([ids]), use_cache=False, logits_to_keep=1)
    finally:
        for h in hs:
            h.remove()
    return np.stack([acc[l].numpy() for l in range(e.n_layers)]), p - n0
