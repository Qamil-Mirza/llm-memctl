"""§51 weight-edit layer (proposal rev 1a §5.4, §10, §13.5): the edit family, saved edits with a hashed manifest,
lazy fp32 copies of edited tensors, bit-for-bit apply and per-component revert on a loaded Qwen2 model.

An edit is a list of component deltas, stored in one .npz plus a manifest JSON (edit_id = the first 16 hex of the
manifest's sha256). Component = (layer, tensor) with tensor in EDITABLE. Kinds:
- "dense":  ΔW = U @ V.T, U [out, k], V [in, k], k <= 4 (matrices only);
- "rows":   ΔW[idx, :] = D, D [len(idx), in] (matrices);
- "cols":   ΔW[:, idx] = D, D [out, len(idx)] (matrices);
- "coords": Δw[idx] = d (RMSNorm vectors only, §13.5);
- comparator kinds (B5 / B6, `family="comparator"`, exempt from the candidate limits): "dense" with any number of
  components, and "random_matched" (B6: a seeded Gaussian on the support of a candidate component, each column scaled
  to the candidate's column norm on that support; RMSNorm: L2 on the same coordinates).

Candidate family (refused otherwise): <= 4 components; one kind per component; sparse rows + cols + coords <= 512 in
total; no q/k/v, gate, up, embedding, lm_head or final norm (not in EDITABLE, so no "unknown" embedding row either);
no whole-tensor scaling (ΔW proportional to W, |cos| > 0.999, is refused); no new parameters (the edit only changes
existing tensors). Arithmetic: ΔW is formed in float64 (numpy), rounded once to float32, added to the float32
original in torch (elementwise, deterministic); the result's sha256 is in the manifest and checked on every apply.

Fixed choices of this build: deltas are additive; a dose d scales every component's delta (U by d, D by d) before
rounding, so dose 1 is the saved edit; the original of a tensor is checked bit-for-bit against the bf16 value on disk
(cast to float32) the first time it is edited.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

EDITABLE = {"o_proj": "model.layers.{l}.self_attn.o_proj.weight",
            "down_proj": "model.layers.{l}.mlp.down_proj.weight",
            "input_layernorm": "model.layers.{l}.input_layernorm.weight",
            "post_attention_layernorm": "model.layers.{l}.post_attention_layernorm.weight"}
VECTORS = ("input_layernorm", "post_attention_layernorm")
MAX_COMPONENTS, MAX_RANK, MAX_SPARSE = 4, 4, 512
SCALE_COS = 0.999
MANIFEST_VERSION = "exp51_edit_v1"


class EditRefused(ValueError):
    pass


def sha_array(a: np.ndarray) -> str:
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def comp_name(c: dict) -> str:
    return EDITABLE[c["tensor"]].format(l=c["layer"])


# ---------------------------------------------------------------------------------------------- validation

def validate(components: list[dict], shapes: dict[str, tuple], family: str = "candidate",
             originals=None) -> dict:
    """Check an edit against the family. components: dicts with layer, tensor, kind and arrays. shapes: param name ->
    shape. originals(name) -> float32 np array (for the scaling check; None skips it). Returns counts."""
    if family not in ("candidate", "comparator"):
        raise EditRefused(f"unknown family {family!r}")
    seen, sparse = set(), 0
    for c in components:
        if c.get("tensor") not in EDITABLE:
            raise EditRefused(f"tensor {c.get('tensor')!r} is not editable (allowed: {sorted(EDITABLE)})")
        name = comp_name(c)
        if name not in shapes:
            raise EditRefused(f"{name} is not in the model")
        if name in seen:
            raise EditRefused(f"{name}: one edit type per component")
        seen.add(name)
        shape, kind, vec = shapes[name], c["kind"], c["tensor"] in VECTORS
        if kind == "dense":
            if vec:
                raise EditRefused(f"{name}: RMSNorm vectors take coordinate edits only (§13.5)")
            U, V = np.asarray(c["U"]), np.asarray(c["V"])
            if U.ndim != 2 or V.ndim != 2 or U.shape[0] != shape[0] or V.shape[0] != shape[1] or U.shape[1] != V.shape[1]:
                raise EditRefused(f"{name}: U {U.shape} / V {V.shape} do not fit {shape}")
            if family == "candidate" and U.shape[1] > MAX_RANK:
                raise EditRefused(f"{name}: rank {U.shape[1]} > {MAX_RANK}")
        elif kind in ("rows", "cols", "coords"):
            idx = np.asarray(c["idx"])
            if idx.ndim != 1 or len(np.unique(idx)) != len(idx) or len(idx) == 0:
                raise EditRefused(f"{name}: indices must be distinct and non-empty")
            if kind == "coords" and not vec:
                raise EditRefused(f"{name}: coordinate edits are for RMSNorm vectors")
            if kind in ("rows", "cols") and vec:
                raise EditRefused(f"{name}: a vector takes coordinate edits")
            D = np.asarray(c["D"])
            axis_len = shape[0] if kind in ("rows", "coords") else shape[1]
            if idx.min() < 0 or idx.max() >= axis_len:
                raise EditRefused(f"{name}: index out of range")
            want = {"rows": (len(idx), shape[1]) if not vec else None, "cols": (shape[0], len(idx)) if not vec else None,
                    "coords": (len(idx),)}[kind]
            if D.shape != want:
                raise EditRefused(f"{name}: D {D.shape}, want {want}")
            sparse += len(idx)
        elif kind == "random_matched":
            if family != "comparator":
                raise EditRefused("random_matched is a comparator kind")
        else:
            raise EditRefused(f"{name}: unknown kind {kind!r}")
        for k in ("U", "V", "D"):
            if k in c and not np.all(np.isfinite(np.asarray(c[k]))):
                raise EditRefused(f"{name}: non-finite values")
        if originals is not None and kind != "random_matched":
            d = delta64(c, shape)
            w = originals(name).astype(np.float64)
            nd, nw = np.linalg.norm(d), np.linalg.norm(w)
            if nd > 0 and nw > 0 and abs(float((d * w).sum()) / (nd * nw)) > SCALE_COS:
                raise EditRefused(f"{name}: the delta is a scaling of the whole tensor")
    if family == "candidate":
        if len(components) > MAX_COMPONENTS:
            raise EditRefused(f"{len(components)} components > {MAX_COMPONENTS}")
        if sparse > MAX_SPARSE:
            raise EditRefused(f"{sparse} sparse rows / columns / coordinates > {MAX_SPARSE}")
        if not components:
            raise EditRefused("an edit needs at least one component")
    return {"n_components": len(components), "n_sparse": sparse}


def delta64(c: dict, shape: tuple, dose: float = 1.0) -> np.ndarray:
    """The component's full delta in float64 (dose-scaled)."""
    kind = c["kind"]
    if kind == "dense":
        return (np.asarray(c["U"], np.float64) * dose) @ np.asarray(c["V"], np.float64).T
    if kind == "random_matched":
        return random_matched(c, shape) * dose
    out = np.zeros(shape, np.float64)
    idx, D = np.asarray(c["idx"]), np.asarray(c["D"], np.float64) * dose
    if kind in ("rows", "coords"):
        out[idx] = D
    else:
        out[:, idx] = D
    return out


def n_changed(c: dict, shape: tuple) -> int:
    return int(np.count_nonzero(delta64(c, shape)))


# ---------------------------------------------------------------------------------------------- B6 / B5

def support_of(c: dict, shape: tuple) -> dict:
    """The entries a candidate component edits: all (dense), rows, cols or coords."""
    if c["kind"] == "dense":
        return {"axis": "all"}
    return {"axis": c["kind"], "idx": np.asarray(c["idx"]).tolist()}


def random_spec(c: dict, shape: tuple, seed: int) -> dict:
    """B6 for one candidate component: the same support, matched per column (Frobenius on the support) for a matrix,
    L2 on the same coordinates for a vector."""
    d = delta64(c, shape)
    sup = support_of(c, shape)
    if len(shape) == 1:
        idx = np.asarray(sup["idx"]) if sup["axis"] != "all" else np.arange(shape[0])
        norms = np.array([np.linalg.norm(d[idx])])
    else:
        rows = np.asarray(sup["idx"]) if sup["axis"] == "rows" else slice(None)
        cols = np.asarray(sup["idx"]) if sup["axis"] == "cols" else slice(None)
        sub = d[rows][:, cols] if sup["axis"] == "rows" else d[:, cols]
        norms = np.linalg.norm(sub, axis=0)
    return {"layer": c["layer"], "tensor": c["tensor"], "kind": "random_matched", "seed": int(seed),
            "support": sup, "norms": norms}


def random_matched(c: dict, shape: tuple) -> np.ndarray:
    rng = np.random.default_rng([5106, int(c["seed"]), int(c["layer"]), list(EDITABLE).index(c["tensor"])])
    sup, norms = c["support"], np.asarray(c["norms"], np.float64)
    out = np.zeros(shape, np.float64)
    if len(shape) == 1:
        idx = np.asarray(sup["idx"]) if sup["axis"] != "all" else np.arange(shape[0])
        g = rng.standard_normal(len(idx))
        out[idx] = g / np.linalg.norm(g) * norms[0]
        return out
    if sup["axis"] == "all":
        g = rng.standard_normal(shape)
        out = g / np.linalg.norm(g, axis=0, keepdims=True) * norms[None, :]
    elif sup["axis"] == "rows":
        idx = np.asarray(sup["idx"])
        g = rng.standard_normal((len(idx), shape[1]))
        out[idx] = g / np.linalg.norm(g, axis=0, keepdims=True) * norms[None, :]
    else:
        idx = np.asarray(sup["idx"])
        g = rng.standard_normal((shape[0], len(idx)))
        out[:, idx] = g / np.linalg.norm(g, axis=0, keepdims=True) * norms[None, :]
    return out


def orthogonalize_spec(direction: np.ndarray, weights: dict[str, np.ndarray], layers, beta: float) -> list[dict]:
    """B5 (Arditi-style, adapted): W <- W − β d dᵀ W for o_proj and down_proj of each layer (rank-1 dense, U = −β d,
    V = Wᵀ d); `weights[name]` = the original float32 matrix; `direction` unit-normalised here."""
    d = np.asarray(direction, np.float64)
    d = d / np.linalg.norm(d)
    out = []
    for l in layers:
        for t in ("o_proj", "down_proj"):
            W = weights[EDITABLE[t].format(l=l)].astype(np.float64)
            out.append({"layer": int(l), "tensor": t, "kind": "dense", "U": (-beta * d)[:, None], "V": (W.T @ d)[:, None]})
    return out


# ---------------------------------------------------------------------------------------------- storage

def _arrays_of(components: list[dict]) -> dict[str, np.ndarray]:
    arrs = {}
    for i, c in enumerate(components):
        for k in ("U", "V", "D", "idx", "norms"):
            if k in c:
                arrs[f"c{i}_{k}"] = np.asarray(c[k], np.float64 if k != "idx" else np.int64)
    return arrs


def save_edit(directory: Path, components: list[dict], shapes: dict[str, tuple], family: str = "candidate",
              originals=None, meta: dict | None = None, model_fingerprint: dict | None = None) -> dict:
    """Validate, then write <edit_id>.npz and <edit_id>.json; returns the manifest (with edit_id). Re-saving the same
    edit returns the same id."""
    counts = validate(components, shapes, family, originals)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    arrs = _arrays_of(components)
    comp_meta = []
    for i, c in enumerate(components):
        shape = shapes[comp_name(c)]
        m = {"i": i, "layer": int(c["layer"]), "tensor": c["tensor"], "kind": c["kind"], "name": comp_name(c),
             "shape": list(shape)}
        if c["kind"] == "dense":
            m["rank"] = int(np.asarray(c["U"]).shape[1])
        if c["kind"] in ("rows", "cols", "coords"):
            m["n_idx"] = int(len(c["idx"]))
        if c["kind"] == "random_matched":
            m["seed"], m["support_axis"] = int(c["seed"]), c["support"]["axis"]
            if c["support"]["axis"] != "all":
                arrs[f"c{i}_idx"] = np.asarray(c["support"]["idx"], np.int64)
        m["array_sha256"] = {k.split("_", 1)[1]: sha_array(v) for k, v in arrs.items() if k.startswith(f"c{i}_")}
        comp_meta.append(m)
    manifest = {"version": MANIFEST_VERSION, "family": family, "components": comp_meta, **counts,
                "meta": meta or {}, "model": model_fingerprint or {}}
    edit_id = hashlib.sha256(canonical(manifest).encode()).hexdigest()[:16]
    manifest["edit_id"] = edit_id
    np.savez(directory / f"{edit_id}.npz", **arrs)
    manifest["npz_sha256"] = hashlib.sha256((directory / f"{edit_id}.npz").read_bytes()).hexdigest()
    (directory / f"{edit_id}.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    return manifest


def load_edit(directory: Path, edit_id: str) -> tuple[dict, list[dict]]:
    """The manifest and the components (arrays checked against their recorded hashes)."""
    directory = Path(directory)
    man = json.loads((directory / f"{edit_id}.json").read_text())
    if hashlib.sha256((directory / f"{edit_id}.npz").read_bytes()).hexdigest() != man["npz_sha256"]:
        raise EditRefused(f"{edit_id}: npz hash mismatch")
    z = np.load(directory / f"{edit_id}.npz")
    comps = []
    for m in man["components"]:
        i = m["i"]
        c = {"layer": m["layer"], "tensor": m["tensor"], "kind": m["kind"]}
        for k, sha in m["array_sha256"].items():
            a = z[f"c{i}_{k}"]
            if sha_array(a) != sha:
                raise EditRefused(f"{edit_id} c{i}_{k}: array hash mismatch")
            c[k] = a
        if m["kind"] == "random_matched":
            c["seed"] = m["seed"]
            c["support"] = {"axis": m["support_axis"], **({"idx": c.pop("idx").tolist()} if "idx" in c else {})}
        comps.append(c)
    return man, comps


def manifest_sha(man: dict) -> str:
    return hashlib.sha256(canonical(man).encode()).hexdigest()


# ---------------------------------------------------------------------------------------------- model side

def file_sha256(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def snapshot_fingerprint(snapshot: Path, hash_files: bool = True) -> dict:
    """The snapshot directory's revision (its name) and the sha256 of every weight / tokenizer / config file."""
    snapshot = Path(snapshot)
    files = sorted(p for p in snapshot.iterdir() if p.is_file() or p.is_symlink())
    return {"snapshot": str(snapshot), "revision": snapshot.name,
            "files": {p.name: (file_sha256(p) if hash_files else None) for p in files}}


class EditLayer:
    """Applies saved edits to a loaded model's parameters in place. Keeps a lazy float32 copy of each tensor the first
    time it is edited (checked against the bf16 value on disk); `revert` restores those copies bit-for-bit."""

    def __init__(self, net, snapshot: Path | None, edit_dir: Path):
        import torch
        self.torch, self.net, self.snapshot, self.edit_dir = torch, net, snapshot, Path(edit_dir)
        self.params = dict(net.named_parameters())
        self.orig: dict = {}
        self.applied: dict[str, dict] = {}          # param name -> {"edit_id", "i", "dose", "sha"}
        self._index = None

    def shapes(self) -> dict[str, tuple]:
        return {EDITABLE[t].format(l=l): tuple(self.params[EDITABLE[t].format(l=l)].shape)
                for t in EDITABLE for l in range(self.n_layers()) if EDITABLE[t].format(l=l) in self.params}

    def n_layers(self) -> int:
        return max(int(m.group(1)) for k in self.params if (m := re.match(r"model\.layers\.(\d+)\.", k))) + 1

    def _disk(self, name: str):
        if self.snapshot is None:
            return None
        from safetensors import safe_open
        if self._index is None:
            idx = Path(self.snapshot) / "model.safetensors.index.json"
            if idx.exists():
                self._index = json.loads(idx.read_text())["weight_map"]
            else:
                self._index = {k: "model.safetensors" for k in self.params}
        with safe_open(str(Path(self.snapshot) / self._index[name]), "pt") as f:
            return f.get_tensor(name)

    def original(self, name: str):
        """The float32 original (lazily copied; verified against disk on the first copy)."""
        if name not in self.orig:
            p = self.params[name].detach()
            d = self._disk(name)
            if d is not None and not self.torch.equal(d.to(p.device, self.torch.float32), p):
                raise RuntimeError(f"{name}: the loaded parameter differs from the safetensors value on disk")
            self.orig[name] = p.clone()
        return self.orig[name]

    def original_np(self, name: str) -> np.ndarray:
        return self.original(name).cpu().numpy()

    def tensor_sha(self, name: str) -> str:
        return sha_array(self.params[name].detach().cpu().numpy())

    def apply(self, edit_id: str, dose: float = 1.0, only: set | None = None) -> dict:
        """Apply (replacing any edit on the same tensors) and return {param name: sha256 of the result}. `only`:
        the component indices to apply (per-component revert = apply all but one)."""
        man, comps = load_edit(self.edit_dir, edit_id)
        out = {}
        with self.torch.no_grad():
            for m, c in zip(man["components"], comps):
                if only is not None and m["i"] not in only:
                    continue
                name = m["name"]
                orig = self.original(name)
                d32 = self.torch.from_numpy(delta64(c, tuple(m["shape"]), dose).astype(np.float32)).to(orig.device)
                self.params[name].copy_(orig + d32)
                sha = self.tensor_sha(name)
                rec = man.get("result_sha256", {}).get(f"{name}@{dose:g}")
                if rec is not None and rec != sha:
                    raise RuntimeError(f"{edit_id} {name} dose {dose:g}: result {sha[:12]} != manifest {rec[:12]}")
                self.applied[name] = {"edit_id": edit_id, "i": m["i"], "dose": dose, "sha": sha}
                out[name] = sha
        return out

    def record_results(self, edit_id: str, doses=(1.0,)) -> dict:
        """Apply at each dose, store the resulting tensor hashes in the manifest (result_sha256), revert."""
        path = self.edit_dir / f"{edit_id}.json"
        man = json.loads(path.read_text())
        res = man.get("result_sha256", {})
        for d in doses:
            for name, sha in self.apply(edit_id, d).items():
                res[f"{name}@{d:g}"] = sha
            self.revert()
        man["result_sha256"] = res
        path.write_text(json.dumps(man, indent=1, sort_keys=True))
        return res

    def revert(self, names: set | None = None) -> None:
        """Restore the originals of `names` (default: every edited tensor) bit-for-bit."""
        with self.torch.no_grad():
            for name in list(self.applied if names is None else names):
                if name in self.orig:
                    self.params[name].copy_(self.orig[name])
                self.applied.pop(name, None)

    def revert_component(self, layer: int, tensor: str) -> None:
        self.revert({EDITABLE[tensor].format(l=layer)})

    def state(self) -> dict:
        return {k: dict(v) for k, v in sorted(self.applied.items())}

    def assert_clean(self) -> None:
        """Every tensor ever copied equals its original (no edit left applied)."""
        for name, o in self.orig.items():
            if not self.torch.equal(self.params[name].detach(), o):
                raise RuntimeError(f"{name} is not at its original value")
