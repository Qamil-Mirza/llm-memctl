"""§54 patching core: component captures, real patches, AtP and the direct effect at P_ans with a reused prefix cache.

Components (freeze definitions.components): head (l, h) = the 128-dim slice h of layer l's o_proj input at P_ans;
MLP l = layer l's down_proj output at P_ans. A patch is applied as a change to the module's OUTPUT at P_ans (head:
o_proj output += W_O[:, slice] @ (z_donor - z_recip); MLP: down_proj output set to the donor value), so all later
components recompute. Only P_ans is touched: the recipient's KV over positions < P_ans is computed once and cropped
back after every one-token step.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

HEAD_DIM = 128
N_HEADS = 16


def comp_name(c: tuple) -> str:
    return f"MLP{c[0]}" if c[1] == "mlp" else f"L{c[0]}h{c[1]}"


def parse(name: str) -> tuple:
    if name.startswith("MLP"):
        return (int(name[3:]), "mlp")
    l, h = name[1:].split("h")
    return (int(l), int(h))


def components(layers, n_heads: int = N_HEADS) -> list[tuple]:
    out = []
    for l in layers:
        out += [(l, h) for h in range(n_heads)] + [(l, "mlp")]
    return out


@dataclass
class Capture:
    z: dict          # layer -> o_proj input at P_ans [hidden]
    mlp: dict        # layer -> down_proj output at P_ans [hidden]
    final: np.ndarray  # final residual (input of model.norm) at P_ans
    D: float


class Site:
    """One recipient prefix (ids end with the P_ans token). Holds the prefix cache."""

    def __init__(self, e, ids: list[int], t_star: int, unk: int, layers):
        import torch
        self.e, self.torch = e, torch
        e.net.requires_grad_(False)
        cfg = e.net.config
        self.hd = getattr(cfg, 'head_dim', None) or cfg.hidden_size // cfg.num_attention_heads
        self.n_heads = cfg.num_attention_heads
        self.ids, self.pos, self.tok = list(ids), len(ids) - 1, ids[-1]
        self.pair_ids = [unk, t_star]
        self.layers = list(layers)
        self.cache = e._prefill(ids[:-1])
        self.n_cache = len(ids) - 1
        self.base = self.step()

    # ------------------------------------------------------------------ one patched step
    def step(self, patch: dict | None = None, grad: bool = False) -> Capture | tuple:
        """patch: {"z": {layer: {head: z_value[128]}}, "mlp": {layer: value}, "z_add": {layer: vec}, "mlp_add": ...}.
        Returns the capture at P_ans (after the patch). grad=True returns (Capture, grads) with dD/dz and dD/dmlp."""
        torch, e = self.torch, self.e
        patch = patch or {}
        cap_z, cap_m, cap_f, handles, keep = {}, {}, {}, [], {}
        W = {l: e.layers[l].self_attn.o_proj.weight for l in self.layers}

        for l in self.layers:
            attn, mlp = e.layers[l].self_attn, e.layers[l].mlp

            def pre(module, args, l=l):
                z = args[0]
                if grad:
                    z = z.detach().requires_grad_(True)
                    keep[("z", l)] = z
                cap_z[l] = z
                return (z,) + tuple(args[1:]) if grad else None

            def post_o(module, args, out, l=l):
                zp = patch.get("z", {}).get(l)
                add = patch.get("z_add", {}).get(l)
                if not zp and add is None:
                    return None
                out = out.clone()
                z0 = cap_z[l][0, -1]
                for h, v in (zp or {}).items():
                    sl = slice(h * self.hd, (h + 1) * self.hd)
                    dz = torch.as_tensor(np.asarray(v, np.float32)) - z0[sl]
                    out[0, -1] = out[0, -1] + W[l][:, sl] @ dz
                if add is not None:
                    out[0, -1] = out[0, -1] + torch.as_tensor(np.asarray(add, np.float32))
                return out

            def post_m(module, args, out, l=l):
                v = patch.get("mlp", {}).get(l)
                add = patch.get("mlp_add", {}).get(l)
                if v is not None or add is not None:
                    out = out.clone()
                    if v is not None:
                        out[0, -1] = torch.as_tensor(np.asarray(v, np.float32))
                    if add is not None:
                        out[0, -1] = out[0, -1] + torch.as_tensor(np.asarray(add, np.float32))
                if grad:
                    out = out if out.requires_grad else out.detach().requires_grad_(True)
                    out.retain_grad()
                    keep[("m", l)] = out
                cap_m[l] = out
                return out
            handles += [attn.o_proj.register_forward_pre_hook(pre), attn.o_proj.register_forward_hook(post_o),
                        mlp.down_proj.register_forward_hook(post_m)]

        def fin(module, args):
            cap_f["x"] = args[0]
        handles.append(e.net.model.norm.register_forward_pre_hook(fin))
        e._offset = self.pos
        try:
            ctx = torch.enable_grad() if grad else torch.no_grad()
            with ctx:
                out = e.net(input_ids=torch.tensor([[self.tok]]), past_key_values=self.cache, use_cache=True,
                            logits_to_keep=1)
                lg = out.logits[0, -1]
                D = lg[self.pair_ids[0]] - lg[self.pair_ids[1]]
                grads = None
                if grad:
                    D.backward()
                    grads = {"z": {l: keep[("z", l)].grad[0, -1].detach().numpy().copy() for l in self.layers},
                             "mlp": {l: keep[("m", l)].grad[0, -1].detach().numpy().copy() for l in self.layers}}
        finally:
            for h in handles:
                h.remove()
            self.cache.crop(-1)
        c = Capture({l: cap_z[l][0, -1].detach().numpy().copy() for l in self.layers},
                    {l: cap_m[l][0, -1].detach().numpy().copy() for l in self.layers},
                    cap_f["x"][0, -1].detach().numpy().copy(), float(D.detach()))
        return (c, grads) if grad else c

    # ------------------------------------------------------------------ patch specs
    def spec(self, comps, donor: Capture) -> dict:
        p = {"z": {}, "mlp": {}}
        for l, h in comps:
            if h == "mlp":
                p["mlp"][l] = donor.mlp[l]
            else:
                p["z"].setdefault(l, {})[h] = donor.z[l][h * self.hd:(h + 1) * self.hd]
        return p

    def write(self, c: tuple, cap: Capture) -> np.ndarray:
        """What component c writes to the residual at P_ans."""
        l, h = c
        if h == "mlp":
            return cap.mlp[l]
        W = self.e.layers[l].self_attn.o_proj.weight.detach().numpy()
        sl = slice(h * self.hd, (h + 1) * self.hd)
        return W[:, sl] @ cap.z[l][sl]

    def direct(self, comps, donor: Capture) -> float:
        """D with only the patched components' write change reaching the logits (final norm recomputed)."""
        torch = self.torch
        r = self.base.final.copy()
        for c in comps:
            r = r + self.write(c, donor) - self.write(c, self.base)
        with torch.no_grad():
            x = self.e.net.model.norm(torch.as_tensor(r[None, None, :]))
            w = self.e.net.lm_head.weight[self.pair_ids]
            lg = (x[0, 0] @ w.T).numpy()
        return float(lg[0] - lg[1])

    def gen_arm(self, comps, donor: Capture, add: dict | None = None):
        """An exp51 Arm with the same patch at P_ans (o_proj output add / down_proj set), for greedy generation."""
        import exp51_engine as EN
        ops = []
        by_l: dict = {}
        for l, h in comps:
            if h == "mlp":
                ops.append({"layer": l, "tensor": "down_proj", "op": "set", "values": {self.pos: donor.mlp[l]}})
            else:
                W = self.e.layers[l].self_attn.o_proj.weight.detach().numpy()
                sl = slice(h * self.hd, (h + 1) * self.hd)
                by_l[l] = by_l.get(l, 0) + W[:, sl] @ (donor.z[l][sl] - self.base.z[l][sl])
        for l, v in by_l.items():
            ops.append({"layer": l, "tensor": "o_proj", "op": "add", "values": {self.pos: v.astype(np.float32)}})
        for (l, t), v in (add or {}).items():
            ops.append({"layer": l, "tensor": t, "op": "add", "values": {self.pos: np.asarray(v, np.float32)}})
        return EN.Arm(comp=ops)
