#!/usr/bin/env python
"""§51 mechanism kit, model side (controlled prefix, B0's stored ids): M1 component revert and B6 seeds, M2
transplant and clamp with hooks only at shared-prefix positions <= P_ans (§13.3), M3's f on the unedited model, the
f-intervention and the B1-matched comparator. Each function returns graded records like exp51_run.run_arm_row.

Spec: research/proposals/2026-11-02-exp51-agent-weight-editing.md rev 1a §6, §13.3, §13.4 (D-72).

Fixed choices of this build:
- Transplant / clamp values come from one teacher-forced read of the stored prefix ids under E* and one under the
  unedited model (component outputs at the hook positions); the arm then recomputes the KV over the prefix with the
  hook and generates <= 16 tokens with no hook after the prefix (the hook acts only where position <= P_ans).
- A site is (layer, tensor) with tensor in EN.COMPONENT_MODULE ("resid" = the decoder layer's output); f = the
  projection of the site's output at P_ans on the account's frozen unit direction; the f-intervention adds
  alpha x direction at that site at P_ans only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import exp51_common as CM  # noqa: E402
import exp51_engine as EN  # noqa: E402


def _grade(e, row, rec, ids, g, tc):
    text = e.decode(ids[rec["n_prompt"]:]) + g.text
    return {"case_id": row["case_id"], "gen_ids": g.gen_ids, "text": text, "gen_s": round(g.wall_s, 3),
            **CM.grade(row, text, tc)}


def check_positions(positions, rec) -> list[int]:
    pos = sorted(set(int(p) for p in positions))
    if not pos or pos[-1] > rec["p_ans"]:
        raise ValueError(f"M2 hooks act only at shared-prefix positions <= P_ans ({rec['p_ans']}); got {pos}")
    return pos


def revert_arm(e, row, rec, edit_id: str, dose: float, primary_i: int, n_components: int, tc) -> dict:
    """M1a: E* with its primary component reverted (every other component applied)."""
    ids = CM.prefix_ids(rec)
    for i in range(n_components):
        if i != primary_i:
            e.edits.apply(edit_id, dose, only={i})
    try:
        g = e.generate(ids, mode="prefix")
    finally:
        e.edits.revert()
    return _grade(e, row, rec, ids, g, tc)


def component_outputs(e, rec, layer: int, tensor: str, positions, edit=None) -> dict[int, np.ndarray]:
    ids = CM.prefix_ids(rec)
    rd = e.read(ids, positions, comps=[(layer, tensor)], edit=edit)
    return rd["comp"][(layer, tensor)]


def transplant_arm(e, row, rec, edit_id, dose, layer, tensor, positions, tc) -> dict:
    """M2a: unedited model + Δh (E* output − unedited output) at the positions."""
    pos = check_positions(positions, rec)
    hE = component_outputs(e, rec, layer, tensor, pos, (edit_id, dose))
    h0 = component_outputs(e, rec, layer, tensor, pos)
    arm = EN.Arm(comp=[{"layer": layer, "tensor": tensor, "op": "add", "values": {p: hE[p] - h0[p] for p in pos}}])
    ids = CM.prefix_ids(rec)
    g = e.generate(ids, arm, mode="prefix")
    return {**_grade(e, row, rec, ids, g, tc), "delta_norm": float(np.linalg.norm([hE[p] - h0[p] for p in pos]))}


def clamp_arm(e, row, rec, edit_id, dose, layer, tensor, positions, tc) -> dict:
    """M2b: E* with the component's output clamped to its unedited output at the positions."""
    pos = check_positions(positions, rec)
    h0 = component_outputs(e, rec, layer, tensor, pos)
    arm = EN.Arm(comp=[{"layer": layer, "tensor": tensor, "op": "set", "values": {p: h0[p] for p in pos}}])
    ids = CM.prefix_ids(rec)
    g = e.generate(ids, arm, mode="prefix", edit=(edit_id, dose))
    return _grade(e, row, rec, ids, g, tc)


def f_value(e, rec, site: tuple, direction: np.ndarray) -> float:
    """M3: f on the unedited model at P_ans."""
    layer, tensor = site
    h = component_outputs(e, rec, layer, tensor, [rec["p_ans"]])[rec["p_ans"]]
    u = np.asarray(direction, np.float64)
    return float(h.astype(np.float64) @ (u / np.linalg.norm(u)))


def f_intervention_arm(e, row, rec, site: tuple, direction: np.ndarray, alpha: float, tc) -> dict:
    """M3b: set f in the unedited model by adding alpha x unit direction at the site at P_ans only."""
    layer, tensor = site
    u = np.asarray(direction, np.float64)
    v = (alpha * u / np.linalg.norm(u)).astype(np.float32)
    arm = EN.Arm(comp=[{"layer": layer, "tensor": tensor, "op": "add", "values": {rec["p_ans"]: v}}])
    ids = CM.prefix_ids(rec)
    g = e.generate(ids, arm, mode="prefix")
    return _grade(e, row, rec, ids, g, tc)


def b1_prefix_arm(e, row, rec, value: float, b1_ids, tc) -> dict:
    """The M3b comparator: B1 at a signed-grid value, controlled prefix (on from P_ans)."""
    ids = CM.prefix_ids(rec)
    g = e.generate(ids, EN.Arm(bias={"ids": b1_ids, "value": value, "mode": "prefix"}), mode="prefix")
    return _grade(e, row, rec, ids, g, tc)


def lexical_unknown_logodds(e, rec, b1_ids, edit=None) -> float:
    """Descriptive: log-odds of the 'unknown' token set at P_ans (log p(set) − log(1 − p(set)))."""
    ids = CM.prefix_ids(rec)
    rd = e.read(ids, [rec["p_ans"]], logit_positions=[rec["p_ans"]], edit=edit)
    z = rd["logits"][0].astype(np.float64)
    z = z - z.max()
    p = np.exp(z) / np.exp(z).sum()
    ps = float(p[list(b1_ids)].sum())
    return float(np.log(ps) - np.log1p(-ps))


def m_city(e, row, rec, tc, edit=None) -> float:
    """M_city at P_ans (oracle covariate): logit(gold first token) − max over other pool cities' first tokens."""
    ids = CM.prefix_ids(rec)
    rd = e.read(ids, [rec["p_ans"]], logit_positions=[rec["p_ans"]], edit=edit)
    z = rd["logits"][0]
    pool = CM.t_cities() if row["split"] == "Tcity" else list(CM.GR.MAIN_CITIES)
    tid = {c: e.encode(" " + c)[0] for c in pool}
    g = row["gold"] or row["base_gold"]
    return float(z[tid[g]] - max(z[t] for c, t in tid.items() if c != g))
