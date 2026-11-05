#!/usr/bin/env python
"""§46 measurement A: per-layer contrastive probes on the captured residuals (CPU, scikit-learn).

    .venv/bin/python scripts/exp46_probe.py --capture runs/exp46_probe/cuda-bfloat16          # dev: fits, CV, design.json
    .venv/bin/python scripts/exp46_probe.py --capture ... --held                               # held read, once

Spec: research/proposals/2026-10-30-exp46-instrument-check.md §3 with rev 1 ruling 3 (6ff90c8); helpers in
memctl/analysis/probe46.py. Reads (task, target, position): RC supervisor at P_last (the §5 verdict read) and at P_q;
TU current and superseded city at P_last. Dev CV folds: the (format x hardness) cell (rev 3, 10505f3), identical for
the probe and c1; c2 on the H1/H2 cells. The dev run fits every probe and control on dev, writes probe_dev.json (CV
numbers and the descriptive dev-CV clauses, nothing from held) and design.json (the design whose sha the held read must match). The held run
refuses (exit 4) unless <out>/design_committed.json exists with a commit and the same design sha; it then refits on
dev with the committed C per (read, layer) and reads held once into probe_held.json (accuracy, Wilson, chance,
accuracy - 1/12, held c1; decodable = held accuracy >= 0.70 with lower bound >= 0.55, rev 5).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from memctl.analysis import probe46 as P  # noqa: E402

EXIT_DESIGN = 4
NAMES = ("Alice", "Ben", "David", "Emma", "Henry", "James", "Kate", "Laura", "Paul", "Rachel", "Sarah", "Peter")
CITIES = ("York", "Manchester", "Liverpool", "Leeds", "Cardiff", "Cambridge", "Edinburgh", "Brighton", "Norwich",
          "Durham", "Derby", "Chester")      # = exp45_cases.NAMES / CITIES (test_exp46 pins the equality)
LAYERS = tuple(range(P.N_LAYERS))


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1))
    os.replace(tmp, path)


def load_capture(capture: Path, cases_path: Path) -> list[dict]:
    """One record per captured prompt: meta json + residuals + the case row (for prompt-level labels)."""
    cases = {json.loads(l)["case_id"]: json.loads(l) for l in cases_path.read_text().splitlines() if l.strip()}
    rows = []
    for npz in sorted(capture.glob("*.npz")):         # one per captured prompt (probe outputs live beside them)
        meta = json.loads(npz.with_suffix(".json").read_text())
        z = np.load(npz)
        rows.append({"meta": meta, "case": cases[meta["case_id"]], "resid": z["resid"],
                     "sites": [str(s) for s in z["sites"]]})
    return rows


def design_config(capture_stamp: dict) -> dict:
    return {"method": P.METHOD_VERSION, "cv_scheme": P.CV_SCHEME, "solver": P.SOLVER, "reads": [list(r) for r in P.READS], "verdict_read": list(P.VERDICT_READ),
            "c_grid": list(P.C_GRID), "folds": P.CV_FOLDS, "seed": P.SEED, "layers": list(LAYERS),
            "criterion": {"rev": P.CRITERION_REV, "acc_min": P.ACC_MIN, "lower_min": P.LOWER_MIN,
                          "decodable": "held accuracy >= acc_min and Wilson lower bound >= lower_min (rev 5, 45bb089)",
                          "descriptive_only": "dev-CV selectivity (probe - c1) and c2 presence, reported as dev_ok",
                          "dev_cv_groups": 6, "c2_groups": 4},
            "capture": capture_stamp}


def read_matrix(rows: list[dict], task: str, target: str, position: str, layer: int, split: str):
    sel = [r for r in rows if r["meta"]["task"] == task and r["meta"]["split"] == split]
    X = np.stack([r["resid"][r["sites"].index(position), layer] for r in sel]) if sel else np.zeros((0, 1))
    return sel, X


N_JOBS = int(os.environ.get("EXP46_PROBE_JOBS", "8"))


def _one(Xd, Xh, layer, ctx):
    from threadpoolctl import threadpool_limits
    with threadpool_limits(1):
        return _layer(Xd, Xh, layer, ctx)


def _parallel(Xd_all: np.ndarray, Xh_all, ctx: dict) -> list[dict]:
    """One worker per layer (BLAS threads 1 each), each sent only its own layer's slice; results in layer order."""
    from joblib import Parallel, delayed
    return Parallel(n_jobs=N_JOBS)(delayed(_one)(np.ascontiguousarray(Xd_all[:, l]),
                                                 None if Xh_all is None else np.ascontiguousarray(Xh_all[:, l]), l, ctx)
                                   for l in LAYERS)


def _layer(Xd: np.ndarray, Xh, layer: int, ctx: dict) -> dict:
    if ctx["held"]:
        c_best = ctx["committed"]["C"][ctx["key"]][str(layer)]
        fit = P.fit_probe(Xd, ctx["y_dev"], c_grid=(c_best,))
        probe = P.accuracy(fit["model"], Xh, ctx["y_held"]) if Xh is not None else P.accuracy(fit["model"], Xd[:0], [])
        cfit = P.fit_probe(Xd, ctx["c_dev"], c_grid=(c_best,))
        c1 = P.accuracy(cfit["model"], Xh, ctx["c_held"]) if Xh is not None else P.accuracy(cfit["model"], Xd[:0], [])
        return {"C": c_best, "probe": probe, "c1_held": c1, "chance_c3": ctx["chance"],
                "acc_minus_1_12": (probe["acc"] - 1 / 12) if probe["acc"] is not None else None}
    g, pres = ctx["g_dev"], ctx["pres"]
    fit = P.fit_probe(Xd, ctx["y_dev"], groups=g)
    probe_cv = P.cv_accuracy(Xd, ctx["y_dev"], fit["C"], groups=g)
    c1_cv = P.cv_accuracy(Xd, ctx["c_dev"], fit["C"], groups=g)
    pres_cv = P.cv_accuracy(Xd[pres], ctx["y_pres"], fit["C"], groups=ctx["g_pres"]) if pres else None
    return {"C": fit["C"], "cv_scores": fit["cv_scores"], "probe_cv": probe_cv, "c1_cv": c1_cv, "c2_cv": pres_cv,
            "selectivity_cv": (probe_cv["acc"] - c1_cv["acc"])
            if probe_cv["acc"] is not None and c1_cv["acc"] is not None else None,
            "dev_ok": P.dev_conditions(probe_cv, c1_cv, pres_cv)}


def run(rows: list[dict], held: bool, committed: dict | None) -> dict:
    out = {}
    for task, target, position in P.READS:
        pool = NAMES if task == "RC" else CITIES
        dev, _ = read_matrix(rows, task, target, position, 0, "dev")
        hel, _ = read_matrix(rows, task, target, position, 0, "held")
        y_dev = [P.target_label(r["case"], target) for r in dev]
        y_held = [P.target_label(r["case"], target) for r in hel]
        g_dev = [P.cv_group(r["case"]) for r in dev]
        ctrl = P.control_labels([r["meta"]["base_index"] for r in rows if r["meta"]["task"] == task], pool)
        c_dev = [ctrl[r["meta"]["base_index"]] for r in dev]
        c_held = [ctrl[r["meta"]["base_index"]] for r in hel]
        pres = [i for i, r in enumerate(dev) if r["meta"]["hardness"] != "H0"] if task == "RC" else []
        y_pres = [P.first_distractor(dev[i]["case"]) for i in pres]
        g_pres = [g_dev[i] for i in pres]
        chance = float(np.mean([1.0 / max(1, P.names_present(r["case"], pool)) for r in (hel if held else dev)]))
        key = f"{task}:{target}:{position}"
        per_layer = {}
        if not dev:
            out[key] = {"task": task, "target": target, "position": position, "n_dev": 0, "n_held": len(hel),
                        "chance_c3": None, "layers": {}, "first_decodable": None}
            continue
        si = dev[0]["sites"].index(position)
        Xd_all = np.stack([r["resid"][si] for r in dev])            # [n, layer, d]
        Xh_all = np.stack([r["resid"][si] for r in hel]) if (held and hel) else None
        ctx = {"held": held, "key": key, "committed": committed, "y_dev": y_dev, "y_held": y_held, "g_dev": g_dev,
               "c_dev": c_dev, "c_held": c_held, "pres": pres, "y_pres": y_pres, "g_pres": g_pres, "chance": chance}
        recs = _parallel(Xd_all, Xh_all, ctx)
        per_layer = {str(layer): rec for layer, rec in zip(LAYERS, recs)}
        out[key] = {"task": task, "target": target, "position": position, "n_dev": len(dev),
                    "n_held": len(hel) if held else None, "chance_c3": chance, "layers": per_layer}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", type=Path, required=True)
    ap.add_argument("--cases", type=Path, default=Path("runs/exp45_maps/cases.jsonl"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--held", action="store_true")
    args = ap.parse_args()
    out = args.out or args.capture
    rows = load_capture(args.capture, args.cases)
    if not rows:
        print("no captured prompts", file=sys.stderr)
        return 2
    m0 = rows[0]["meta"]
    stamp = {k: m0[k] for k in ("capture_version", "device", "dtype", "model")}
    stamp["n_prompts"] = len(rows)
    config = design_config(stamp)
    if not args.held:
        dev_rows = [r for r in rows if r["meta"]["split"] == "dev"]
        result = run(dev_rows, held=False, committed=None)
        write_json(out / "probe_dev.json", {"design_sha256": P.design_stamp(config), "config": config,
                                            "reads": result})
        chosen = {k: {l: v["C"] for l, v in r["layers"].items()} for k, r in result.items()}
        write_json(out / "design.json", {"design_sha256": P.design_stamp(config), "config": config, "C": chosen})
        return 0
    try:
        committed = P.require_design(out, config)
    except PermissionError as e:
        print(f"held read refused: {e}", file=sys.stderr)
        return EXIT_DESIGN
    design = json.loads((out / "design.json").read_text())
    if design["design_sha256"] != committed["design_sha256"]:
        print("held read refused: design.json differs from the committed design", file=sys.stderr)
        return EXIT_DESIGN
    committed = dict(committed, C=design["C"])
    result = run(rows, held=True, committed=committed)
    dev = json.loads((out / "probe_dev.json").read_text())["reads"]
    for key, r in result.items():
        for layer, rec in r["layers"].items():
            d = dev[key]["layers"].get(layer, {"dev_ok": False})
            rec["dev_ok_descriptive"] = d["dev_ok"]
            rec["decodable"] = P.decodable(rec["probe"])
        r["first_decodable"] = P.first_decodable({int(l): v["decodable"] for l, v in r["layers"].items()})
    for key, r in result.items():
        r.setdefault("first_decodable", None)
    write_json(out / "probe_held.json", {"design_sha256": committed["design_sha256"], "commit": committed["commit"],
                                         "reads": result})
    return 0


if __name__ == "__main__":
    sys.exit(main())
