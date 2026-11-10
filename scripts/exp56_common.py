#!/usr/bin/env python
"""§56 shared pieces (Qwen 2.5 3B, §51 engine, laptop fp32, $0): row lists and classes, the t* rule, the neuron
read at P_ans with direct logit attribution through the final-RMSNorm scale, and the P_ans-only neuron ablation.

Spec: research/proposals/2026-11-10-exp56-answer-neurons.md rev 0 (6e6b075), D-83.

Fixed choices of this build (stage-0 freeze):
- Forward input = the row's stored B0 prompt ids + gen ids through the designated marker's ':' (CM.prefix_ids), so
  the last position is P_ans (= rec["p_ans"]). One teacher-forced pass, no cache.
- "unknown" first token: UNK_ID = the first id of " unknown" (9788 in the Qwen 2.5 vocabulary; it is the token after
  ':' on every abstention row of §54 B0 and §55 P0).
- t* (rev 0a, df5d028, rule (c), every class): the highest-logit token at P_ans among the first tokens of the row's
  city_pool (" " + city; every city is one token, asserted). Reported beside it, descriptively: the rev-0 rule's t*
  (argmax over the vocabulary minus the frozen §51 B1 set, every id whose decoded, lower-cased, stripped text begins
  with "unknown").
- D = logit[UNK_ID] − logit[t*] at P_ans.
- Neuron i of layer l: a_i = the input of mlp.down_proj at P_ans (= silu(gate_i(x)) · up_i(x)); write vector w_i =
  down_proj.weight[:, i]. v = g ⊙ (u_unk − u_t*) / σ with g = model.norm.weight, σ = sqrt(mean(h²) + eps) of the
  final-norm input h at P_ans, u = lm_head rows. DLA_i = a_i · ⟨w_i, v⟩; MLP DLA of layer l = ⟨mlp_out_l, v⟩ with
  mlp_out_l captured from the down_proj output. Positive = pushes "unknown".
- Stored per row (rev 0a): a and DLA for all 36 layers (36 × 11,008 each, float32), plus for all layers the MLP DLA,
  the attention (o_proj) DLA and the embedding DLA, so the full decomposition of D is checkable.
- Ablation: a forward pre-hook on mlp.down_proj sets the chosen neurons to the given values at absolute position P_ans
  only (the §51 engine's prefix mode: KV over positions < P_ans, step 0 = P_ans); later steps are not touched.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import exp51_common as CM  # noqa: E402

OUT = ROOT / "runs/exp56"
CASES = ROOT / "runs/exp54/cases54.jsonl"
B054 = ROOT / "runs/exp54/run"                 # §54 B0: ANS rows, both halves (laptop, max_new 160)
P055 = ROOT / "runs/exp55/pod/run/P0"          # §55 P0: UNA rows, even half (pod, HF fp32 no-tf32, max_new 256)
DEVB_UNA = OUT / "run/devb_una"                # new P0 UNA generation, odd half (laptop, §55 settings)
LAYERS = tuple(range(36))                      # rev 0a: all layers (rev 0 default was 20-35)
UNK_TEXT = " unknown"
CLASSES = ("FA", "C", "TA", "UA")


def sha_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def cases() -> list[dict]:
    return [json.loads(l) for l in CASES.read_text().splitlines()]


def city_pool(row: dict) -> list[str]:
    from memctl.analysis import grade51 as GR
    if row["city_pool"] == "main":
        return list(GR.MAIN_CITIES)
    if row["city_pool"] == "T":
        return list(CM.t_cities())
    raise ValueError(row["city_pool"])


def city_ids(encode, row: dict) -> list[int]:
    """First (and only) token of " " + city for each city of the row's pool; single tokens asserted."""
    out = []
    for c in city_pool(row):
        t = encode(" " + c)
        assert len(t) == 1, (c, t)
        out.append(t[0])
    return out


def half_of(row: dict) -> str:
    return "A" if row["base_index"] % 2 == 0 else "B"


def klass(row: dict, rec: dict | None) -> str:
    """FA / C / TA / UA, or why the row is out: NM (no usable P_ans), W (ANS answered wrong), PENDING (no record)."""
    if rec is None:
        return "PENDING"
    if not rec.get("p_ans_ok"):
        return "NM"
    cls = rec["cls"]
    if row["set"] == "ANS":
        return {"abstention": "FA", "gold": "C"}.get(cls, "W")
    if row["set"] in ("UNA-rel", "UNA-swap"):
        return "TA" if cls == "abstention" else "UA"
    raise ValueError(row["set"])


def record_for(row: dict) -> tuple[dict | None, str]:
    """The P0 generation record a row's class and P_ans come from, and its source tag."""
    if row["set"] == "ANS":
        return CM.read_json(B054 / "b0/W" / f"{row['case_id']}.json"), "exp54_b0"
    if half_of(row) == "A":
        return CM.read_json(P055 / "b0/W" / f"{row['case_id']}.json"), "exp55_pod_p0"
    return CM.read_json(DEVB_UNA / "b0/W" / f"{row['case_id']}.json"), "exp56_devb_p0"


# ---------------------------------------------------------------------------------------------- the read at P_ans

class Reader:
    """One teacher-forced pass per row, capturing what the DLA needs at the last position (P_ans)."""

    def __init__(self, e, layers=LAYERS):
        self.e, self.torch = e, e.torch
        net = e.net
        self.layers = tuple(layers)
        self.unk_id = e.encode(UNK_TEXT)[0]
        self.unk_set = sorted(CM.b1_token_set(e.tok))
        assert self.unk_id in self.unk_set
        self.U = net.lm_head.weight.detach()                     # [vocab, hidden]
        self.g = net.model.norm.weight.detach()
        self.eps = float(getattr(net.model.norm, "variance_epsilon", getattr(net.model.norm, "eps", 1e-6)))
        self.Wd = {l: e.layers[l].mlp.down_proj.weight.detach() for l in range(e.n_layers)}   # [hidden, inter]
        mask = self.torch.zeros(self.U.shape[0], dtype=self.torch.bool)
        mask[self.unk_set] = True
        self.mask = mask

    def read(self, ids: list[int], cand_ids: list[int]) -> dict:
        """cand_ids: the row's city first tokens; t* = the highest-logit one (rev 0a)."""
        torch, e = self.torch, self.e
        n = e.n_layers
        cap: dict = {}
        hs = []

        def pre_down(l):
            def fn(module, args):
                cap[("a", l)] = args[0][0, -1].detach().clone()
            return fn

        def post(kind, l):
            def fn(module, args, output):
                h = output[0] if isinstance(output, tuple) else output
                cap[(kind, l)] = h[0, -1].detach().clone()
            return fn

        def pre_norm(module, args):
            cap["h"] = args[0][0, -1].detach().clone()
        for l in range(n):
            hs.append(e.layers[l].mlp.down_proj.register_forward_pre_hook(pre_down(l)))
            hs.append(e.layers[l].mlp.down_proj.register_forward_hook(post("mlp", l)))
            hs.append(e.layers[l].self_attn.o_proj.register_forward_hook(post("attn", l)))
        hs.append(e.net.model.embed_tokens.register_forward_hook(post("emb", 0)))
        hs.append(e.net.model.norm.register_forward_pre_hook(pre_norm))
        try:
            with torch.no_grad():
                out = e.net(input_ids=torch.tensor([ids]), use_cache=False, logits_to_keep=1)
        finally:
            for h in hs:
                h.remove()
        logits = out.logits[0, -1].float()
        masked = logits.masked_fill(self.mask, float("-inf"))
        t_frozen = int(masked.argmax())                                             # rev-0 rule, descriptive
        cand = self.torch.as_tensor(list(cand_ids), dtype=self.torch.long)
        t_star = int(cand[int(logits[cand].argmax())])
        h = cap["h"].float()
        sigma = float(torch.sqrt((h * h).mean() + self.eps))
        v = self.g * (self.U[self.unk_id] - self.U[t_star]) / sigma
        D = float(logits[self.unk_id] - logits[t_star])
        res = {"t_star": t_star, "t_star_rev0": t_frozen, "logit_t_rev0": float(logits[t_frozen]),
               "unk_id": self.unk_id, "D": D, "D_lin": float(h @ v), "sigma": sigma,
               "logit_unk": float(logits[self.unk_id]), "logit_tstar": float(logits[t_star]),
               "top1": int(logits.argmax()),
               "mlp_dla": np.array([float(cap[("mlp", l)] @ v) for l in range(n)], np.float64),
               "attn_dla": np.array([float(cap[("attn", l)] @ v) for l in range(n)], np.float64),
               "emb_dla": float(cap[("emb", 0)] @ v)}
        a = torch.stack([cap[("a", l)] for l in self.layers])                       # [L, inter]
        proj = torch.stack([self.Wd[l].T @ v for l in self.layers])                 # [L, inter]
        res["a"] = a.cpu().numpy().astype(np.float32)
        res["dla"] = (a * proj).cpu().numpy().astype(np.float32)
        res["layers"] = np.array(self.layers, np.int16)
        return res


def save_read(path: Path, res: dict, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, a=res["a"], dla=res["dla"], layers=res["layers"], mlp_dla=res["mlp_dla"],
             attn_dla=res["attn_dla"],
             meta=np.array(json.dumps({**meta, **{k: res[k] for k in ("t_star", "t_star_rev0", "logit_t_rev0", "unk_id", "D", "D_lin", "sigma",
                                                                         "logit_unk", "logit_tstar", "top1",
                                                                         "emb_dla")}})))
    tmp.replace(path)


# ---------------------------------------------------------------------------------------------- ablation at P_ans

def install_ablation(e, p_ans: int, spec: dict) -> list:
    """spec: {layer: (idx int array, values float array)}. Sets a[idx] = values at absolute position p_ans only."""
    torch = e.torch
    hs = []
    for l, (idx, vals) in spec.items():
        idx_t = torch.as_tensor(np.asarray(idx), dtype=torch.long)
        val_t = torch.as_tensor(np.asarray(vals, np.float32))

        def fn(module, args, idx_t=idx_t, val_t=val_t):
            x = args[0]
            lo, T = e._offset, x.shape[1]
            if not (lo <= p_ans < lo + T):
                return None
            x = x.clone()
            x[0, p_ans - lo, idx_t] = val_t
            return (x,) + tuple(args[1:])
        hs.append(e.layers[int(l)].mlp.down_proj.register_forward_pre_hook(fn))
    return hs


def ablated_generate(e, rec: dict, spec: dict):
    """§51 prefix mode (stored ids through P_ans, MAX_NEW_RELEASE) with the P_ans-only ablation."""
    ids = CM.prefix_ids(rec)
    assert len(ids) - 1 == rec["p_ans"]
    hs = install_ablation(e, rec["p_ans"], spec)
    try:
        return e.generate(ids, mode="prefix")
    finally:
        for h in hs:
            h.remove()
