#!/usr/bin/env python
"""§53 rev 1: are the two §51 edits specific, and do they carry over to real questions? Laptop build stages and the
guarded pod stages. Spec: research/proposals/2026-11-05-exp53-head-controls.md rev 1 (D-75, D-76).

    python scripts/exp53_run.py --stage edits      # laptop, no model: A-single, A-rand, A-amp, A-late, D-randwrite
    python scripts/exp53_run.py --stage calibrate  # laptop, model + N-dev B0: D-randread scales, C5 margins
    python scripts/exp53_run.py --stage manifest   # laptop: runs/exp53/manifest.json (committed before the pod)
    python scripts/exp53_run.py --stage pod --root R [--deadline-s S] [--limit-rows K]   # guarded: everything held
    python scripts/exp53_run.py --stage report --root R

Fixed choices of this build:
- Edits are §51 edit-layer edits (memctl/analysis/edit51.py), applied as weight edits under the total effect (every
  position), generated with §51's batched engine (LAYOUT 64, rows sorted by case id, strict fp32) and graded with
  §51's grader; B0 labels (FA, C, UNA) from §51's population rule on the N B0 records.
- Heads: Qwen2.5-3B o_proj input columns [16 heads x 128]; zeroing head (l, h) = cols h*128..h*128+127 of layer l's
  o_proj set to −W (exactly as submission 1); amplifying = +W (exactly as the §51 diagnostic).
- A-rand: 20 pairs drawn without replacement from the 15 x 15 (L32 head != 2, L33 head != 10) pairs, seed 5301.
  A-late: 10 distinct unordered pairs of heads in L30-L35, targets excluded, seed 5302 (two heads of one layer = one
  component of 256 columns). A-amp: the §51 diagnostic (target x 2) and each A-rand pair x 2.
- D-randread: U = submission 2's U (write, |U| 20), V = s x r (r a seeded unit vector, seed 5303, 20 draws), s > 0 set
  on N-dev so that the RMS over FA rows (with a P_ans) of V.x equals that of submission 2's V.x (rev 1a: RMS, not the
  median), x = the L35 o_proj input at P_ans in the unedited model; run at dose 2 like the target. D-randwrite: V = submission 2's V, U = |U| x r
  (seed 5304, 10 draws), dose 2.
- C5: f = submission 2's frozen f (L35 o_proj output at P_ans on its f_direction), the four frozen activity
  predictions, §51's pairs, Δf and tests; margins = 2 x SD of the per-base mean Δf on N-dev. IUT over the four.
- Sample (seed 5300, from the N-held B0 labels, before any edited output): per class FA / C / UNA, 150 rows; quota
  split across tasks (UNA: across task x set) in proportion to availability by largest remainder; within a stratum
  a seeded choice without replacement from the rows sorted by case id. Fewer than the quota: all, reported.
- Claims: rank p = (1 + #draws >= target) / (1 + n_draws), with the claim's extra condition; C1: Rel (base-weighted,
  FA sample) and target − median(A-rand) >= 0.30; C2: C -> abstention (C sample) under amplification; C4: U(λ=1)
  on the sample (§51 base weights) and Uns(target) < median Uns(D-randread). Multiplicity (rev 1a ruling 1): each of
  C1, C2, C4 at its own bar (rank 1 of 21 plus its co-requirement), C5 at the §51 IUT at 0.05; no family-wise
  claim; the parametric z of the target against the draws is descriptive only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import exp51_common as CM  # noqa: E402
import exp51_engine as EN  # noqa: E402
import exp51_stage4 as S4  # noqa: E402
from memctl.analysis import edit51 as ED  # noqa: E402
from memctl.analysis import mech51 as MK  # noqa: E402
from memctl.analysis import stats51 as S  # noqa: E402

OUT = ROOT / "runs/exp53"
CASES = OUT / "cases53.jsonl"
EDITS = OUT / "edits"
VECS = OUT / "vectors.npz"
CALIB = OUT / "calibration.json"
MANIFEST = OUT / "manifest.json"
DEV_ROOT = OUT / "dev"
SUBS51 = ROOT / "runs/exp51/gpu/subs"
EDITS51 = ROOT / "runs/exp51/gpu/edits"
SUB1, SUB2, DIAG = "c64ea090781450cf", "8364bf135ac59a83", "87224655b7a5361e"
TARGETS = ((32, 2), (33, 10))
N_HEADS, HEAD_DIM = 16, 128
SEEDS = {"sample": 5300, "A_rand": 5301, "A_late": 5302, "D_randread": 5303, "D_randwrite": 5304}
N_DRAWS = {"A_rand": 20, "A_late": 10, "D_randread": 20, "D_randwrite": 10}
QUOTA = {"FA": 150, "C": 150, "UNA": 150}
D_DOSE = 2.0
B1_REF = -8.0
SPLIT = "N"
MULTIPLICITY = ("rev 1a ruling 1: C1, C2, C4 each at rank 1 of 21 plus its co-requirement; C5 IUT at 0.05; no "
                "family-wise claim (under a global null P(at least one of four passes) <= ~0.19); z descriptive")


# ---------------------------------------------------------------------------------------------- data

def all_rows() -> list[dict]:
    return [json.loads(l) for l in CASES.read_text().splitlines() if l.strip()]


def rows_of(nsplit: str) -> list[dict]:
    return sorted([r for r in all_rows() if r["nsplit"] == nsplit], key=lambda r: r["case_id"])


def sha_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------------------------- stage: edits

def o_proj_weights(layers) -> dict[int, np.ndarray]:
    from safetensors import safe_open
    want = {f"model.layers.{l}.self_attn.o_proj.weight": l for l in layers}
    out = {}
    for f in sorted(Path(EN.SNAPSHOT).glob("*.safetensors")):
        with safe_open(str(f), "pt") as s:
            for k in s.keys():
                if k in want:
                    out[want[k]] = s.get_tensor(k).float().numpy()
    return out


def head_cols(h: int) -> np.ndarray:
    return np.arange(h * HEAD_DIM, (h + 1) * HEAD_DIM)


def head_edit(W: dict, heads, sign: float) -> list[dict]:
    """Zero (sign −1) or double (sign +1) the o_proj input columns of the given heads (one component per layer)."""
    by = defaultdict(list)
    for l, h in heads:
        by[l].append(h)
    comps = []
    for l in sorted(by):
        idx = np.concatenate([head_cols(h) for h in sorted(by[l])])
        comps.append({"layer": l, "tensor": "o_proj", "kind": "cols", "idx": idx,
                      "D": sign * W[l][:, idx].astype(np.float64)})
    return comps


def shapes_all() -> dict:
    return {ED.EDITABLE["o_proj"].format(l=l): (2048, 2048) for l in range(36)}


def save(comps, meta) -> str:
    return ED.save_edit(EDITS, comps, shapes_all(), "comparator", meta=meta)["edit_id"]


def copy_51_edits() -> dict:
    """Byte-identical copies of the §51 edits used here (sha checked)."""
    EDITS.mkdir(parents=True, exist_ok=True)
    out = {}
    for eid in (SUB1, SUB2, DIAG):
        for ext in ("json", "npz"):
            src, dst = EDITS51 / f"{eid}.{ext}", EDITS / f"{eid}.{ext}"
            dst.write_bytes(src.read_bytes())
            assert sha_file(src) == sha_file(dst)
        ED.load_edit(EDITS, eid)
        out[eid] = sha_file(EDITS / f"{eid}.npz")
    return out


def draw_pairs() -> dict:
    rng = np.random.default_rng(SEEDS["A_rand"])
    grid = [(a, b) for a in range(N_HEADS) if a != 2 for b in range(N_HEADS) if b != 10]
    pick = rng.choice(len(grid), N_DRAWS["A_rand"], replace=False)
    a_rand = [((32, grid[i][0]), (33, grid[i][1])) for i in pick]
    rng = np.random.default_rng(SEEDS["A_late"])
    pool = [(l, h) for l in range(30, 36) for h in range(N_HEADS) if (l, h) not in TARGETS]
    a_late, seen = [], set()
    while len(a_late) < N_DRAWS["A_late"]:
        i, j = rng.choice(len(pool), 2, replace=False)
        key = tuple(sorted((pool[i], pool[j])))
        if key in seen:
            continue
        seen.add(key)
        a_late.append(key)
    return {"A_rand": [[list(x) for x in p] for p in a_rand], "A_late": [[list(x) for x in p] for p in a_late]}


def stage_edits() -> dict:
    shas = copy_51_edits()
    W = o_proj_weights(range(30, 36))
    man1, comps1 = ED.load_edit(EDITS, SUB1)
    for c in comps1:                                         # submission 1 = −W on the target heads, exactly
        assert np.array_equal(np.asarray(c["D"]), -W[c["layer"]][:, c["idx"]].astype(np.float64))
    pairs = draw_pairs()
    ed = {"A_single": {}, "A_rand": [], "A_amp_rand": [], "A_late": []}
    for c in comps1:
        ed["A_single"][f"L{c['layer']}"] = save([c], {"arm": "A-single", "of": SUB1, "layer": int(c["layer"])})
    for p in pairs["A_rand"]:
        hs = [tuple(x) for x in p]
        ed["A_rand"].append(save(head_edit(W, hs, -1.0), {"arm": "A-rand", "heads": p, "seed": SEEDS["A_rand"]}))
        ed["A_amp_rand"].append(save(head_edit(W, hs, +1.0), {"arm": "A-amp", "heads": p, "seed": SEEDS["A_rand"]}))
    for p in pairs["A_late"]:
        ed["A_late"].append(save(head_edit(W, [tuple(x) for x in p], -1.0),
                                 {"arm": "A-late", "heads": p, "seed": SEEDS["A_late"]}))
    _, (c2,) = ED.load_edit(EDITS, SUB2)
    U, V = np.asarray(c2["U"], np.float64), np.asarray(c2["V"], np.float64)
    rr = np.random.default_rng(SEEDS["D_randread"]).standard_normal((N_DRAWS["D_randread"], 2048))
    rr /= np.linalg.norm(rr, axis=1, keepdims=True)
    rw = np.random.default_rng(SEEDS["D_randwrite"]).standard_normal((N_DRAWS["D_randwrite"], 2048))
    rw /= np.linalg.norm(rw, axis=1, keepdims=True)
    np.savez(VECS, randread_unit=rr, randwrite_unit=rw)
    ed["D_randwrite"] = [save([{"layer": 35, "tensor": "o_proj", "kind": "dense",
                                "U": (r * np.linalg.norm(U))[:, None], "V": V}],
                              {"arm": "D-randwrite", "draw": i, "seed": SEEDS["D_randwrite"]}) for i, r in enumerate(rw)]
    out = {"copied_51": shas, "pairs": pairs, "edits": ed, "vectors_sha256": sha_file(VECS)}
    CM.write_json(OUT / "edits_built.json", out)
    return out


# ---------------------------------------------------------------------------------------------- stage: calibrate

def oproj_input(e, rec, layer: int) -> np.ndarray:
    """The o_proj input (attention heads' concatenated output) at P_ans, unedited, teacher-forced."""
    torch = e.torch
    saved = {}
    mod = e.layers[layer].self_attn.o_proj
    p = rec["p_ans"]

    def pre(module, args):
        saved["x"] = args[0][0, p].detach().float().cpu().numpy().copy()
    h = mod.register_forward_pre_hook(pre)
    try:
        ids = list(rec["prompt_ids"]) + list(rec["gen_ids"])[: p + 1 - len(rec["prompt_ids"])]
        with torch.no_grad():
            e.net(input_ids=torch.tensor([ids], device=e.net.device), use_cache=False, logits_to_keep=1)
    finally:
        h.remove()
    return saved["x"]


def stage_calibrate(e) -> dict:
    import exp51_mech as MC  # noqa: F401
    built = CM.read_json(OUT / "edits_built.json")
    rows = rows_of("dev")
    recs = {r["case_id"]: CM.read_json(S4.b0_path(DEV_ROOT, SPLIT, r["case_id"])) for r in rows}
    missing = [c for c, v in recs.items() if v is None]
    if missing:
        raise SystemExit(f"N-dev B0 incomplete: {len(missing)} rows missing")
    fa = [r for r in rows if "FA" in recs[r["case_id"]]["populations"] and recs[r["case_id"]].get("p_ans_ok")]
    _, (c2,) = ED.load_edit(EDITS, SUB2)
    V = np.asarray(c2["V"], np.float64)[:, 0]
    U = np.asarray(c2["U"], np.float64)
    X = np.stack([oproj_input(e, recs[r["case_id"]], 35).astype(np.float64) for r in fa])
    rr = np.load(VECS)["randread_unit"]
    gate_t = X @ V
    rms = lambda g: float(np.sqrt(np.mean(np.square(g))))  # noqa: E731
    rms_t = rms(gate_t)
    scales, edits = [], []
    for i, r in enumerate(rr):
        g = X @ r
        s = rms_t / rms(g)
        scales.append({"draw": i, "rms_unit_gate": rms(g), "median_unit_gate": float(np.median(g)), "scale": s,
                       "median_scaled_gate": float(np.median(g * s)),
                       "iqr_scaled_gate": [float(np.percentile(g * s, 25)), float(np.percentile(g * s, 75))]})
        edits.append(save([{"layer": 35, "tensor": "o_proj", "kind": "dense", "U": U, "V": (s * r)[:, None]}],
                          {"arm": "D-randread", "draw": i, "seed": SEEDS["D_randread"], "scale": s}))
    # C5 margins on N-dev: Δf SD per manipulation (2 x SD of per-base means)
    sub2 = json.loads((SUBS51 / "2.json").read_text())
    fdir = np.load(SUBS51 / "2_f_direction.npy")
    fvals = {r["case_id"]: S4.f_of(e, recs[r["case_id"]], sub2["f_site"], fdir) for r in rows}
    pr = S4.pairs_for(rows)
    acts = []
    for a in sub2["activity"]:
        df = S4.delta_f(fvals, pr[a["manipulation"]])
        sd = MK.sd_of_base_means(df)
        acts.append({**a, "sd_dev": sd, "margin": 2 * sd, "mean_dev": S.bw_mean(df, list(df)), "n_bases_dev": len(df)})
    pops = Counter(p for v in recs.values() for p in v["populations"])
    out = {"n_dev_rows": len(rows), "dev_populations": dict(pops), "n_fa_calib": len(fa),
           "target_gate": {"rms": rms_t, "median": float(np.median(gate_t)), "iqr": [float(np.percentile(gate_t, 25)), float(np.percentile(gate_t, 75))]},
           "D_randread": {"edits": edits, "scales": scales}, "C5": {"activity": acts, "f_site": sub2["f_site"],
                                                                     "f_direction_sha256": sub2["f_direction_sha256"]},
           "engine": e.fingerprint()}
    CM.write_json(CALIB, out)
    CM.write_json(DEV_ROOT / "f_dev.json", fvals)
    return out


# ---------------------------------------------------------------------------------------------- stage: manifest

CODE = ("scripts/exp53_cases.py", "scripts/exp53_run.py", "scripts/exp51_stage4.py", "scripts/exp51_engine.py",
        "scripts/exp51_common.py", "scripts/exp51_mech.py", "memctl/analysis/edit51.py", "memctl/analysis/mech51.py",
        "memctl/analysis/stats51.py")


def arm_list(built: dict, calib: dict) -> list[dict]:
    """Every generation arm, in run order (the Advisor's priority: C4, C1, then C, C5, C2, B)."""
    e = built["edits"]
    A = []
    A.append({"name": "D-target", "edit_id": SUB2, "dose": D_DOSE, "pops": ["FA", "C", "UNA"], "claim": "C4"})
    for i, eid in enumerate(calib["D_randread"]["edits"]):
        A.append({"name": f"D-randread{i:02d}", "edit_id": eid, "dose": D_DOSE, "pops": ["FA", "C", "UNA"], "claim": "C4"})
    A.append({"name": "A-target", "edit_id": SUB1, "dose": 1.0, "pops": ["FA", "C", "UNA"], "claim": "C1"})
    for i, eid in enumerate(e["A_rand"]):
        A.append({"name": f"A-rand{i:02d}", "edit_id": eid, "dose": 1.0, "pops": ["FA", "UNA"], "claim": "C1"})
    A.append({"name": "A-amp-target", "edit_id": DIAG, "dose": 1.0, "pops": ["C"], "claim": "C2"})
    for i, eid in enumerate(e["A_amp_rand"]):
        A.append({"name": f"A-amp-rand{i:02d}", "edit_id": eid, "dose": 1.0, "pops": ["C"], "claim": "C2"})
    for k, eid in e["A_single"].items():
        A.append({"name": f"A-single-{k}", "edit_id": eid, "dose": 1.0, "pops": ["FA", "C", "UNA"], "claim": None})
    for i, eid in enumerate(e["A_late"]):
        A.append({"name": f"A-late{i:02d}", "edit_id": eid, "dose": 1.0, "pops": ["FA", "UNA"], "claim": None})
    for i, eid in enumerate(e["D_randwrite"]):
        A.append({"name": f"D-randwrite{i:02d}", "edit_id": eid, "dose": D_DOSE, "pops": ["FA", "UNA"], "claim": None})
    A.append({"name": "B1-ref", "b1": B1_REF, "pops": ["FA", "C", "UNA"], "claim": None})
    return A


def stage_manifest() -> dict:
    built = CM.read_json(OUT / "edits_built.json")
    calib = CM.read_json(CALIB)
    edit_files = sorted(p.name for p in EDITS.iterdir())
    m = {"experiment": "§53 rev 1", "proposal_sha256": sha_file(ROOT / "research/proposals/2026-11-05-exp53-head-controls.md"),
         "cases53_sha256": sha_file(CASES), "cases_counts_sha256": sha_file(OUT / "cases53_counts.json"),
         "seeds": SEEDS, "n_draws": N_DRAWS, "quota": QUOTA, "d_dose": D_DOSE, "b1_ref": B1_REF,
         "pairs": built["pairs"], "edits": built["edits"], "copied_51": built["copied_51"],
         "D_randread": calib["D_randread"], "C5": calib["C5"], "target_gate": calib["target_gate"],
         "vectors_sha256": sha_file(VECS), "edit_files_sha256": {f: sha_file(EDITS / f) for f in edit_files},
         "arms": arm_list(built, calib), "multiplicity": MULTIPLICITY,
         "sample_rule": __doc__.split("- Sample")[1].split("- Claims")[0].strip(),
         "code_sha256": {f: sha_file(ROOT / f) for f in CODE},
         "lme": CM.read_json(OUT / "lme_rows_meta.json"), "built_at_head": subprocess.run(
             ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()}
    CM.write_json(MANIFEST, m)
    return m


# ---------------------------------------------------------------------------------------------- guard

def require_manifest(root: Path) -> dict:
    """The pod refuses unless runs/exp53/manifest.json is byte-equal to its blob in the commit named by
    runs/exp53/manifest_committed.json, that commit exists, and every listed code / edit / data file hashes as
    recorded."""
    mk = CM.read_json(OUT / "manifest_committed.json")
    if not mk or not mk.get("commit"):
        raise PermissionError("manifest_committed.json missing or names no commit")
    c = mk["commit"]
    if subprocess.run(["git", "cat-file", "-e", f"{c}^{{commit}}"], cwd=ROOT).returncode != 0:
        raise PermissionError(f"commit {c} does not exist")
    blob = subprocess.run(["git", "show", f"{c}:runs/exp53/manifest.json"], cwd=ROOT, capture_output=True).stdout
    if not blob or blob != MANIFEST.read_bytes():
        raise PermissionError("manifest.json differs from its committed blob")
    if hashlib.sha256(blob).hexdigest() != mk.get("manifest_sha256"):
        raise PermissionError("committed manifest does not hash to the marker")
    m = json.loads(blob)
    bad = [f for f, h in m["code_sha256"].items() if sha_file(ROOT / f) != h]
    bad += [f for f, h in m["edit_files_sha256"].items() if sha_file(EDITS / f) != h]
    if sha_file(CASES) != m["cases53_sha256"] or sha_file(VECS) != m["vectors_sha256"]:
        bad.append("cases53 / vectors")
    if bad:
        raise PermissionError(f"hash mismatch: {bad}")
    return m


# ---------------------------------------------------------------------------------------------- pod stages

class Deadline(Exception):
    pass


def check(deadline):
    if deadline and time.time() > deadline:
        raise Deadline()


def draw_sample(root: Path, rows: list[dict]) -> dict:
    p = root / "sample.json"
    if p.exists():
        return CM.read_json(p)
    recs = {r["case_id"]: CM.read_json(S4.b0_path(root, SPLIT, r["case_id"])) for r in rows}
    rng = np.random.default_rng(SEEDS["sample"])
    out, avail_rep = {}, {}
    for cls, q in QUOTA.items():
        strata = defaultdict(list)
        for r in rows:
            if cls in recs[r["case_id"]]["populations"]:
                strata[(r["task"], r["set"]) if cls == "UNA" else (r["task"],)].append(r["case_id"])
        keys = sorted(strata)
        n = {k: len(strata[k]) for k in keys}
        tot = sum(n.values())
        take = min(q, tot)
        raw = {k: take * n[k] / tot for k in keys}
        alloc = {k: int(np.floor(raw[k])) for k in keys}
        for k in sorted(keys, key=lambda k: (-(raw[k] - alloc[k]), k))[: take - sum(alloc.values())]:
            alloc[k] += 1
        ids = []
        for k in keys:
            pool = sorted(strata[k])
            ids += [pool[i] for i in sorted(rng.choice(len(pool), alloc[k], replace=False))]
        out[cls] = sorted(ids)
        avail_rep[cls] = {"|".join(k): n[k] for k in keys} | {"_taken": take}
    res = {"ids": out, "available": avail_rep, "seed": SEEDS["sample"]}
    CM.write_json(p, res)
    return res


def run_edit_arm(e, root, rows, arm, tc, deadline):
    spec = ({"arm": "B1", "dose": float(arm["b1"]), "variant": "online", "b1_ids": CM.b1_token_set(e.tok), "tag": "ref"}
            if "b1" in arm else {"arm": "EDIT", "edit_id": arm["edit_id"], "dose": float(arm["dose"]), "tag": arm["name"]})
    check(deadline)
    S4.run_arm(e, rows, root, SPLIT, spec, "total", tc, deadline and deadline)
    return spec


def graded_for(root, rows, arm, b1_ids=None):
    spec = ({"arm": "B1", "dose": float(arm["b1"]), "variant": "online", "tag": "ref"} if "b1" in arm else
            {"arm": "EDIT", "edit_id": arm["edit_id"], "dose": float(arm["dose"]), "tag": arm["name"]})
    return S4.load_graded(root, SPLIT, spec, "total", rows)


def f_reads(e, root, rows, deadline):
    """C5: f on every N-held row (unedited, each row's own B0 P_ans)."""
    sub2 = json.loads((SUBS51 / "2.json").read_text())
    fdir = np.load(SUBS51 / "2_f_direction.npy")
    path = root / "f_held.json"
    fv = CM.read_json(path, {})
    for r in rows:
        if r["case_id"] in fv:
            continue
        if deadline and time.time() > deadline:
            CM.write_json(path, fv)
            raise Deadline()
        fv[r["case_id"]] = S4.f_of(e, CM.read_json(S4.b0_path(root, SPLIT, r["case_id"])), sub2["f_site"], fdir)
    CM.write_json(path, fv)
    return fv


def dla(e, root, rows, deadline):
    """Arm B-DLA (descriptive): each target head's direct logit attribution at P_ans onto ' unknown' − the gold's first
    token (final-norm scale from the final residual at P_ans), unedited; plus submission 2's L35 write at dose 2 and
    the L35 heads 12 and 13 (the editor-claimed carriers)."""
    path = root / "dla.json"
    out = CM.read_json(path, {})
    W = {l: e.layers[l].self_attn.o_proj.weight.detach().float().cpu().numpy() for l in (32, 33, 35)}
    WU = e.net.lm_head.weight.detach().float().cpu().numpy()
    gnorm = e.net.model.norm.weight.detach().float().cpu().numpy()
    eps = float(e.net.config.rms_norm_eps)
    unk = e.encode(" unknown")[0]
    _, (c2,) = ED.load_edit(EDITS, SUB2)
    U2, V2 = np.asarray(c2["U"], np.float64)[:, 0], np.asarray(c2["V"], np.float64)[:, 0]
    for r in rows:
        cid = r["case_id"]
        if cid in out:
            continue
        check(deadline)
        rec = CM.read_json(S4.b0_path(root, SPLIT, cid))
        if not rec.get("p_ans_ok") or not (r.get("gold") or r.get("base_gold")):
            out[cid] = None
            continue
        gold = e.encode(" " + (r.get("gold") or r["base_gold"]))[0]
        xs = {l: oproj_input(e, rec, l).astype(np.float64) for l in (32, 33, 35)}
        res = e.read(list(rec["prompt_ids"]) + list(rec["gen_ids"])[: rec["p_ans"] + 1 - len(rec["prompt_ids"])],
                     [rec["p_ans"]], layers=[35])["resid"][35][0].astype(np.float64)
        scale = 1.0 / np.sqrt(np.mean(res ** 2) + eps)
        d = (WU[unk] - WU[gold]).astype(np.float64) * gnorm * scale
        o = {}
        for l, h in TARGETS + ((35, 12), (35, 13)):
            cols = head_cols(h)
            o[f"L{l}h{h}"] = float((W[l][:, cols].astype(np.float64) @ xs[l][cols]) @ d)
        o["D_write_dose2"] = float(D_DOSE * (V2 @ xs[35]) * (U2 @ d))
        out[cid] = o
    CM.write_json(path, out)
    return out


def bopt(e, root, fa_rows, tc, deadline):
    """Arm B-opt: B0 under the option-removed prompt (§51 B3 wording) on the FA sample; rows that still abstain are
    run under A-target and D-target with the same prompt."""
    path = root / "bopt.json"
    out = CM.read_json(path, {"B0": {}, "A-target": {}, "D-target": {}})
    for name, edit in (("B0", None), ("A-target", (SUB1, 1.0)), ("D-target", (SUB2, D_DOSE))):
        todo = [r for r in fa_rows if r["case_id"] not in out[name] and
                (name == "B0" or (out["B0"].get(r["case_id"]) or {}).get("cls") == "abstention")]
        for i in range(0, len(todo), S4.LAYOUT):
            check(deadline)
            ch = todo[i:i + S4.LAYOUT]
            gs = e.generate_batched([e.chat_ids(r["prompt_b3"]) for r in ch], mode="total", edit=edit)
            for r, g in zip(ch, gs):
                out[name][r["case_id"]] = {"text": g.text, **CM.grade(r, g.text, tc)}
            CM.write_json(path, out)
    return out


def lme(e, root, deadline, limit=None):
    """Arm C: LongMemEval folds 0-1 (126 population rows + the 12 _abs rows), B0 / A-target / D-target, fresh on this
    engine; frozen reader prompt (precomputed on the laptop), greedy, 256 new tokens."""
    rows = [json.loads(l) for l in (OUT / "lme_rows.jsonl").read_text().splitlines() if l.strip()][:limit]
    path = root / "lme.json"
    out = CM.read_json(path, {"B0": {}, "A-target": {}, "D-target": {}})
    for name, edit in (("B0", None), ("D-target", (SUB2, D_DOSE)), ("A-target", (SUB1, 1.0))):
        todo = sorted([r for r in rows if r["question_id"] not in out[name]], key=lambda r: len(r["chat_ids"]))
        for i in range(0, len(todo), 16):
            check(deadline)
            ch = todo[i:i + 16]
            gs = e.generate_batched([r["chat_ids"] for r in ch], mode="total", edit=edit, max_new=256)
            for r, g in zip(ch, gs):
                out[name][r["question_id"]] = {"text": g.text, "stop": g.stop}
            CM.write_json(path, out)
    return out


def stage_pod(e, root: Path, deadline, limit_rows=None, nsplit="held", smoke=False, case_ids=None, lme_limit=4) -> None:
    """Order (the Advisor's priority; a deadline stop cuts only what comes last): B0 on the N rows -> the sample ->
    the C4 arms (D-target, D-randread) -> the C1 arms (A-target, A-rand) -> arm C (LongMemEval) -> C5 f reads -> the
    C2 arms (A-amp) -> arm B (B-opt, B-DLA) -> the descriptive arms (A-single, A-late, D-randwrite, B1-ref).
    smoke: N-dev rows only, one arm of each kind, 4 LongMemEval rows (never N-held)."""
    tc = CM.t_cities()
    rows_n = rows_of(nsplit)
    if case_ids:
        if not smoke:
            raise SystemExit("--case-ids is for the smoke only")
        want = set(case_ids)
        rows_n = [r for r in rows_n if r["case_id"] in want]
        if len(rows_n) != len(want):
            raise SystemExit("--case-ids: some ids are not N-dev rows")
    if limit_rows:
        rows_n = rows_n[:limit_rows]
    log = lambda s: print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)  # noqa: E731
    t0 = time.time()
    S4.run_b0(e, rows_n, root, SPLIT, tc, deadline)
    log(f"B0 on {len(rows_n)} {nsplit} rows done at +{time.time() - t0:.0f} s")
    smp = draw_sample(root, rows_n)
    by = {r["case_id"]: r for r in rows_n}
    pop_rows = {k: [by[c] for c in v] for k, v in smp["ids"].items()}
    log(f"sample: {[(k, len(v)) for k, v in pop_rows.items()]}")
    m = json.loads(MANIFEST.read_text()) if MANIFEST.exists() and not smoke else {"arms": arm_list(
        CM.read_json(OUT / "edits_built.json"), CM.read_json(CALIB))}
    arms = m["arms"]
    if smoke:
        keep = ("D-target", "D-randread00", "A-target", "A-rand00", "A-amp-target", "A-amp-rand00", "A-single-L32",
                "A-late00", "D-randwrite00", "B1-ref")
        arms = [a for a in arms if a["name"] in keep]
    timing = CM.read_json(root / "timing.json", {})

    def run(arm):
        rows = sorted([r for p in arm["pops"] for r in pop_rows[p]], key=lambda r: r["case_id"])
        t = time.time()
        run_edit_arm(e, root, rows, arm, tc, deadline)
        timing[arm["name"]] = {"n": len(rows), "s": time.time() - t}
        CM.write_json(root / "timing.json", timing)
        log(f"{arm['name']}: {len(rows)} rows, {time.time() - t:.0f} s")
    for claim in ("C4", "C1"):
        for arm in [a for a in arms if a["claim"] == claim]:
            run(arm)
    t = time.time()
    lme(e, root, deadline, limit=lme_limit if smoke else None)
    log(f"arm C done ({time.time() - t:.0f} s)")
    f_reads(e, root, rows_n, deadline)
    log("C5 f reads done")
    for arm in [a for a in arms if a["claim"] == "C2"]:
        run(arm)
    bopt(e, root, pop_rows["FA"], tc, deadline)
    log("B-opt done")
    dla(e, root, sorted(pop_rows["FA"] + pop_rows["C"] + pop_rows["UNA"], key=lambda r: r["case_id"]), deadline)
    log("B-DLA done")
    for arm in [a for a in arms if a["claim"] is None]:
        run(arm)
    log(f"all stages done at +{time.time() - t0:.0f} s")


# ---------------------------------------------------------------------------------------------- report

def z_desc(target: float, draws: list[float]) -> float | None:
    sd = float(np.std(draws, ddof=1)) if len(draws) > 1 else 0.0
    return (target - float(np.mean(draws))) / sd if sd > 0 else None


def rank_p(target: float, draws: list[float], higher: bool = True) -> float:
    ge = sum((d >= target) if higher else (d <= target) for d in draws)
    return (1 + ge) / (1 + len(draws))


def lme_summary(root: Path) -> dict | None:
    """Arm C (descriptive): per arm, on the 126 population rows: refusals, gold-containing answers (d1 on the
    answer), in-view refusals of B0 released (and how many contain the gold), damage to B0-correct rows; on the 12
    _abs rows: answers given. Reproduction: the cached §23 output vs the fresh B0 (refusal flag; first token after
    'Answer:' by the Qwen tokenizer is approximated by the first answer word)."""
    from memctl.analysis import answer_selection as AS
    path = root / "lme.json"
    if not path.exists():
        return None
    out = CM.read_json(path)
    rows = {r["question_id"]: r for r in (json.loads(l) for l in (OUT / "lme_rows.jsonl").read_text().splitlines() if l)}

    def judge(text, r):
        ans, _ = AS.agent_answer(text)
        refusal = AS.is_refusal_answer(ans)
        return {"refusal": refusal, "gold": (not refusal) and bool(AS.d1(ans, r["gold"])) if r["answerable"] else None,
                "marked": AS.split_note(text)[0] is not None, "first": (ans.split() or [""])[0].lower()}
    J = {a: {q: judge(v["text"], rows[q]) for q, v in d.items()} for a, d in out.items()}
    b0 = J.get("B0", {})
    res = {"n": {a: len(d) for a, d in J.items()}}
    pop = [q for q, r in rows.items() if r["population"] and q in b0]
    absq = [q for q, r in rows.items() if not r["answerable"] and q in b0]
    inview_ref = [q for q in pop if b0[q]["marked"] and b0[q]["refusal"]]
    correct = [q for q in pop if b0[q]["gold"]]
    for a, d in J.items():
        have = [q for q in pop if q in d]
        res[a] = {"population_rows": len(have), "refusals": sum(d[q]["refusal"] for q in have),
                  "gold": sum(bool(d[q]["gold"]) for q in have),
                  "inview_refusals_of_B0": len(inview_ref),
                  "released": sum(not d[q]["refusal"] for q in inview_ref if q in d),
                  "released_with_gold": sum(bool(d[q]["gold"]) for q in inview_ref if q in d),
                  "B0_correct": len(correct), "damaged": sum(not d[q]["gold"] for q in correct if q in d),
                  "abs_rows": len([q for q in absq if q in d]), "abs_answered": sum(not d[q]["refusal"] for q in absq if q in d)}
    cached = {q: judge(rows[q]["cached_output"], rows[q]) for q in b0}
    res["reproduction"] = {"n": len(b0), "refusal_agree": sum(cached[q]["refusal"] == b0[q]["refusal"] for q in b0),
                           "first_word_agree": sum(cached[q]["first"] == b0[q]["first"] for q in b0),
                           "marked_agree": sum(cached[q]["marked"] == b0[q]["marked"] for q in b0)}
    return res


def stage_report(root: Path, nsplit: str = "held", smoke: bool = False) -> dict:
    m = json.loads(MANIFEST.read_text()) if (MANIFEST.exists() and not smoke) else {
        "arms": arm_list(CM.read_json(OUT / "edits_built.json"), CM.read_json(CALIB)),
        "C5": CM.read_json(CALIB)["C5"], "multiplicity": MULTIPLICITY}
    rows_n = rows_of(nsplit)
    by = {r["case_id"]: r for r in rows_n}
    smp = CM.read_json(root / "sample.json")
    rows_n = [r for r in rows_n if S4.b0_path(root, SPLIT, r["case_id"]).exists()]
    labels, _ = S4.labels_meta(root, SPLIT, rows_n)
    rows_p = {k: [by[c] for c in v] for k, v in smp["ids"].items()}
    allr = sorted(sum(rows_p.values(), []), key=lambda r: r["case_id"])
    meta_s = {r["case_id"]: r for r in allr}
    b0 = {r["case_id"]: CM.read_json(S4.b0_path(root, SPLIT, r["case_id"])) for r in allr}
    res = {"arms": {}, "sample": smp, "smoke": smoke}
    bw = lambda d: S.bw_mean(d, list(d)) if d else None  # noqa: E731

    def rates_of(gr):
        rows = {c: g for c, g in gr.items() if g is not None and "cls" in g}
        rt = MK.rates(rows, labels, meta_s)
        return {"n": len(rows), "Rel": bw(rt["Rel"]), "Uns": bw(rt["Uns"]), "Dam": bw(rt["Dam"]), "Deg": bw(rt["Deg"]),
                "U1": bw(MK.mixture(rows, meta_s, 1.0)), "C_abst": bw(_c_abst(rows, labels, meta_s))}
    res["arms"]["B0"] = rates_of(b0)
    for arm in m["arms"]:
        rows = sorted([r for p in arm["pops"] for r in rows_p[p]], key=lambda r: r["case_id"])
        gr = graded_for(root, rows, arm)
        if any(v is not None for v in gr.values()):
            res["arms"][arm["name"]] = rates_of(gr)
    A = res["arms"]
    rnd = lambda pre, k: [v[k] for n, v in A.items() if n.startswith(pre) and v[k] is not None]  # noqa: E731
    bar = 1 / (1 + N_DRAWS["A_rand"])
    claims = {"multiplicity": m["multiplicity"], "bar": bar,
              "s_disjoint_note": "every N row is S-disjoint by construction (cases53 excluded E and S keys)"}
    if "A-target" in A and rnd("A-rand", "Rel"):
        d = rnd("A-rand", "Rel")
        c1 = {"target": A["A-target"]["Rel"], "draws": d, "n_draws": len(d), "median": float(np.median(d)),
              "rank_p": rank_p(A["A-target"]["Rel"], d), "z_desc": z_desc(A["A-target"]["Rel"], d)}
        c1["diff_ok"] = bool(c1["target"] - c1["median"] >= 0.30)
        c1["shown"] = bool(len(d) == N_DRAWS["A_rand"] and c1["rank_p"] <= bar + 1e-12 and c1["diff_ok"])
        du = rnd("A-rand", "Uns")
        claims["C1"] = c1
        claims["C3_desc"] = {"target_Uns": A["A-target"]["Uns"], "draws_Uns": du,
                             "rank_from_top": 1 + sum(x > A["A-target"]["Uns"] for x in du)}
    if "A-amp-target" in A and rnd("A-amp-rand", "C_abst"):
        d = rnd("A-amp-rand", "C_abst")
        c2 = {"target": A["A-amp-target"]["C_abst"], "draws": d, "n_draws": len(d),
              "rank_p": rank_p(A["A-amp-target"]["C_abst"], d), "z_desc": z_desc(A["A-amp-target"]["C_abst"], d)}
        c2["shown"] = bool(len(d) == N_DRAWS["A_rand"] and c2["rank_p"] <= bar + 1e-12)
        claims["C2"] = c2
    if "D-target" in A and rnd("D-randread", "U1"):
        d, du = rnd("D-randread", "U1"), rnd("D-randread", "Uns")
        c4 = {"target_U1": A["D-target"]["U1"], "draws_U1": d, "n_draws": len(d), "rank_p": rank_p(A["D-target"]["U1"], d),
              "z_desc": z_desc(A["D-target"]["U1"], d), "target_Uns": A["D-target"]["Uns"], "draws_Uns": du,
              "median_Uns": float(np.median(du))}
        c4["uns_ok"] = bool(c4["target_Uns"] < c4["median_Uns"])
        c4["shown"] = bool(len(d) == N_DRAWS["D_randread"] and c4["rank_p"] <= bar + 1e-12 and c4["uns_ok"])
        claims["C4"] = c4
    fv = CM.read_json(root / "f_held.json", {})
    if fv:
        pr = S4.pairs_for(rows_n)
        bases = MK.bases_of({r["case_id"]: r for r in rows_n})
        acts = []
        for a in m["C5"]["activity"]:
            df = S4.delta_f(fv, pr[a["manipulation"]])
            if not df:
                acts.append({**a, "p": None, "why": "no pairs"})
                continue
            acts.append({**a, **(MK.activity_directional(bases, df, int(a["sign"]) or 1) if a["type"] == "directional"
                                 else MK.activity_equivalence(bases, df, a["sd_dev"]))})
        c5 = {"activity": acts, "iut_p": S.iut([x["p"] for x in acts])}
        c5["shown"] = bool(c5["iut_p"] is not None and c5["iut_p"] <= 0.05)
        claims["C5"] = c5
    res["claims"] = claims
    res["armC"] = lme_summary(root)
    bo = CM.read_json(root / "bopt.json")
    if bo:
        res["B_opt"] = {k: {"n": len(v), "abstain": sum(x["cls"] == "abstention" for x in v.values()),
                            "gold": sum(x["cls"] == "gold" for x in v.values())} for k, v in bo.items()}
    dl = CM.read_json(root / "dla.json")
    if dl:
        res["B_DLA"] = {}
        for pop in ("FA", "C", "UNA"):
            ids = [c for c in smp["ids"][pop] if dl.get(c)]
            res["B_DLA"][pop] = {k: {"n": len(ids), "mean": float(np.mean([dl[c][k] for c in ids])) if ids else None}
                                 for k in (list(dl[ids[0]]) if ids else [])}
    res["timing"] = CM.read_json(root / "timing.json", {})
    CM.write_json(root / "report.json", res)
    return res


def _c_abst(rows, labels, meta):
    """C -> abstention per base on the C sample."""
    by = defaultdict(list)
    for c, g in rows.items():
        if "C" in labels.get(c, []):
            by[meta[c]["base_key"]].append(1.0 if g["cls"] == "abstention" else 0.0)
    return {b: float(np.mean(v)) for b, v in by.items()}


# ---------------------------------------------------------------------------------------------- main

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", required=True, choices=("edits", "calibrate", "manifest", "pod", "report"))
    p.add_argument("--root", default=str(OUT / "pod"))
    p.add_argument("--device", default="cpu")
    p.add_argument("--snapshot", default=None)
    p.add_argument("--backend", default="hf")
    p.add_argument("--deadline-s", type=float, default=None)
    p.add_argument("--limit-rows", type=int, default=None)
    p.add_argument("--case-ids", default=None, help="smoke only: comma-separated N-dev case ids")
    p.add_argument("--lme-limit", type=int, default=4, help="smoke only: LongMemEval rows")
    p.add_argument("--smoke", action="store_true", help="N-dev rows only, a few arms; no guard (never reads N-held)")
    a = p.parse_args(argv)
    if a.stage == "edits":
        print(json.dumps({k: (v if k != "edits" else {kk: len(vv) for kk, vv in v.items()})
                          for k, v in stage_edits().items() if k != "pairs"}))
        return 0
    if a.stage == "manifest":
        m = stage_manifest()
        print(json.dumps({"manifest_sha256": sha_file(MANIFEST), "n_arms": len(m["arms"])}))
        return 0
    if a.stage == "report":
        r = stage_report(Path(a.root), "dev" if a.smoke else "held", a.smoke)
        print(json.dumps({"written": str(Path(a.root) / "report.json")}))
        return 0
    deadline = time.time() + a.deadline_s if a.deadline_s else None
    if a.stage == "pod" and not a.smoke:
        require_manifest(Path(a.root))
        print("manifest guard ok", flush=True)
    e = EN.load(a.backend, 16, edit_dir=EDITS, device=a.device, snapshot=Path(a.snapshot) if a.snapshot else None)
    if a.stage == "calibrate":
        c = stage_calibrate(e)
        print(json.dumps({"n_fa": c["n_fa_calib"], "target_gate": c["target_gate"],
                          "scales": [round(x["scale"], 3) for x in c["D_randread"]["scales"]],
                          "C5": [(x["manipulation"], round(x["margin"], 3)) for x in c["C5"]["activity"]]}))
        return 0
    try:
        stage_pod(e, Path(a.root), deadline, a.limit_rows, "dev" if a.smoke else "held", a.smoke,
                  a.case_ids.split(",") if a.case_ids else None, a.lme_limit)
    except Deadline:
        print("deadline: stopped between rows (resumable)", flush=True)
        return 9
    return 0


if __name__ == "__main__":
    sys.exit(main())
