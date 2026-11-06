#!/usr/bin/env python
"""§51 pod S D-22 smoke on the dummy model (4 layers, hidden 64). The real submissions, B5 edits and §38 B4 vector are
2048-wide and cannot be applied to the dummy. So this builds stand-ins of the same shape kind on the dummy, then runs
exp51_stage4.main unchanged with that argv:
- one random small edit per real component, same tensor and kind, layer = real layer mod n_layers;
- the submission fields kept, with remapped layers;
- four B5 stand-ins;
- B4 = a constant vector at STEER_LAYER 1.
The smoke root is separate; nothing here touches the real root, the real edits or any result. Mirrors
tests/test_exp51.py::test_stage4_end_to_end_dummy.
    python scripts/exp51_smoke_s.py --stage s-arms --root SM --subs runs/exp51/gpu/subs --snapshot DUMMY ...
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp51_common as CM  # noqa: E402
import exp51_engine as EN  # noqa: E402
import exp51_stage4 as S4  # noqa: E402
from memctl.analysis import edit51 as ED  # noqa: E402


def standin_comp(shapes, layer, tensor, kind, seed, n=2, scale=1e-2):
    shape = shapes[ED.EDITABLE[tensor].format(l=layer)]
    rng = np.random.default_rng(seed)
    c = {"layer": layer, "tensor": tensor, "kind": kind}
    if kind == "dense":
        return {**c, "U": rng.standard_normal((shape[0], 1)) * scale, "V": rng.standard_normal((shape[1], 1)) * scale}
    if kind == "rows":
        return {**c, "idx": np.arange(n), "D": rng.standard_normal((n, shape[1])) * scale}
    if kind == "cols":
        return {**c, "idx": np.arange(n), "D": rng.standard_normal((shape[0], n)) * scale}
    return {**c, "idx": np.arange(n), "D": rng.standard_normal(n) * scale}


def build(root: Path, real_subs: Path, snapshot: str, device: str) -> Path:
    out = root / "subs_standin"
    if (out / "done").exists():
        return out
    out.mkdir(parents=True, exist_ok=True)
    e = EN.load("hf", 8, edit_dir=root / "edits", device=device, snapshot=snapshot)
    sh, L, hid = e.edits.shapes(), e.n_layers, e.hidden
    real_edits = real_subs.parent / "edits"
    for p in sorted(real_subs.glob("[0-9].json"), key=lambda p: int(p.stem)):
        s = json.loads(p.read_text())
        man, _ = ED.load_edit(real_edits, s["edit_id"])
        comps = [standin_comp(sh, m["layer"] % L, m["tensor"], m["kind"], seed=10 * int(p.stem) + i)
                 for i, m in enumerate(man["components"])]
        s["edit_id"] = ED.save_edit(root / "edits", comps, sh, meta={"smoke_standin_for": s["edit_id"]})["edit_id"]
        for k in ("most_important_component", "f_site"):
            if isinstance(s.get(k), dict) and "layer" in s[k]:
                s[k] = {**s[k], "layer": s[k]["layer"] % L}
        CM.write_json(out / p.name, s)
        v = np.ones(hid, np.float32)
        np.save(out / f"{p.stem}_f_direction.npy", v / np.linalg.norm(v))
    b5 = {str(b): ED.save_edit(root / "edits", [standin_comp(sh, L - 1, "o_proj", "dense", seed=9)], sh, "comparator",
                               meta={"b": b, "smoke": True})["edit_id"] for b in CM.B5_DOSES}
    CM.write_json(root / "b5_fit.json", {"edit_ids": b5, "smoke_standin": True})
    (out / "done").write_text("ok\n")
    del e
    return out


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    get = lambda k, d=None: argv[argv.index(k) + 1] if k in argv else d   # noqa: E731
    root, real_subs = Path(get("--root")), Path(get("--subs"))
    sub_dir = build(root, real_subs, get("--snapshot"), get("--device", "cpu"))
    argv[argv.index("--subs") + 1] = str(sub_dir)
    hid = EN.load("hf", 1, snapshot=get("--snapshot"), device=get("--device", "cpu")).hidden
    EN.STEER_LAYER = 1
    CM.b4_vector = lambda a, path=None: np.full(hid, 0.01 * a, np.float32)
    return S4.main(argv)


if __name__ == "__main__":
    sys.exit(main())
