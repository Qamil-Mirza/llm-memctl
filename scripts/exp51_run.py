#!/usr/bin/env python
"""§51 evaluation runner: B0 on E / S, the arms (B0-B6 and candidates) under the total-effect, controlled-prefix and
hybrid estimands, the no-op gate, per-pass timing with peak RSS, the workload inventory and the B5 fit on E. Resumable
per row. H and T are refused unless the F2 freeze is committed (guard; nothing here reads them otherwise).

    HF_HUB_OFFLINE=1 .venv/bin/python scripts/exp51_run.py --stage timing
    HF_HUB_OFFLINE=1 .venv/bin/python scripts/exp51_run.py --stage b0 --split E [--sets ANS]
    HF_HUB_OFFLINE=1 .venv/bin/python scripts/exp51_run.py --stage noop --split E --n 24
    HF_HUB_OFFLINE=1 .venv/bin/python scripts/exp51_run.py --stage b5-fit
    .venv/bin/python scripts/exp51_run.py --stage inventory
    HF_HUB_OFFLINE=1 .venv/bin/python scripts/exp51_run.py --stage arm --split S --arm B1 --dose -4 --estimand total

Spec: research/proposals/2026-11-02-exp51-agent-weight-editing.md rev 1a (D-72). Engine: scripts/exp51_engine.py;
grader memctl/analysis/grade51.py; edits memctl/analysis/edit51.py; shared helpers scripts/exp51_common.py.

Fixed choices of this build:
- B0 records store the prompt ids, all generated ids, the designated marker (last), P_ans, the class and the residual
  of every decoder layer at P_ans (captured at that stepped position during generation; npy beside the record).
- B5 (fit on E only): μ_abst − μ_ans at P_ans over E's ANS rows with B0 class abstention vs gold (both with a P_ans),
  per layer; the direction layer ℓ* = the layer whose unit mean difference gives the highest in-sample AUROC of the
  projection (ties: lower layer); orthogonalized out of o_proj and down_proj of every layer (Arditi's layer set);
  doses β in B5_DOSES (W <- W − β d dᵀ W). The 36-layer orthogonalization is a comparator (exempt from the
  candidate limits).
- Candidate doses: the saved delta x CAND_DOSES (dose 1 = as submitted).
- B6: per candidate component, edit51.random_spec, seeds 1-5, at the candidate's selected dose.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import exp51_common as CM  # noqa: E402
import exp51_engine as EN  # noqa: E402
from memctl.analysis import edit51 as ED  # noqa: E402

EXIT_GUARD, EXIT_DEADLINE, EXIT_STOP = 4, 9, 7
ESTIMANDS = ("total", "prefix", "hybrid")


# ---------------------------------------------------------------------------------------------- B0

def run_b0_row(e, row: dict, run: Path, tc: list[str]) -> dict:
    ids = e.chat_ids(row["prompt"])
    g = e.generate(ids, mode="total", capture_resid=True)
    d = CM.designated(e.decode, g.gen_ids)
    rec = {"case_id": row["case_id"], "split": row["split"], "prompt_ids": ids, "gen_ids": g.gen_ids,
           "text": g.text, "n_prompt": len(ids), "stop": g.stop, "first_trigger_step": g.trigger_step,
           "j": d["j"], "p_ans_ok": d["ok"], "p_ans_why": d["why"],
           "p_ans": (len(ids) + d["j"] - 1) if d["ok"] else None, "gen_s": round(g.wall_s, 3),
           "engine": e.fingerprint(), **CM.grade(row, g.text, tc)}
    rec["populations"] = CM.population(row, rec, tc)
    CM.b0_path(row["split"], row["case_id"], run).parent.mkdir(parents=True, exist_ok=True)
    if d["ok"]:
        np.save(CM.b0_path(row["split"], row["case_id"], run).with_suffix(".npy"),
                g.step_resid[rec["p_ans"]].astype(np.float32))
    return rec


def run_b0(e, rws: list[dict], run: Path, deadline: float | None, tc: list[str]) -> int:
    for r in rws:
        path = CM.b0_path(r["split"], r["case_id"], run)
        if path.exists():
            continue
        if deadline and time.time() > deadline:
            return EXIT_DEADLINE
        path.parent.mkdir(parents=True, exist_ok=True)
        rec = run_b0_row(e, r, run, tc)
        CM.write_json(path, rec)
    return 0


# ---------------------------------------------------------------------------------------------- arms

def arm_key(arm: str, dose=None, variant: str | None = None, seed=None, edit_id: str | None = None) -> str:
    parts = [arm]
    if edit_id:
        parts.append(edit_id)
    if dose is not None:
        parts.append(f"d{dose:g}")
    if seed is not None:
        parts.append(f"s{seed}")
    if variant:
        parts.append(variant)
    return "_".join(parts)


def run_arm_row(e, row: dict, rec: dict, spec: dict, estimand: str, tc: list[str], b1_ids=None) -> dict:
    """One arm on one row. spec: {"arm": B0|B1|B2|B3|B4|EDIT, "dose", "variant", "edit_id"}; EDIT covers B5, B6 and
    candidates (a saved edit at a dose). Returns the generation, its class and timing."""
    arm, dose = spec["arm"], spec.get("dose")
    a = EN.Arm()
    edit = (spec["edit_id"], float(dose if dose is not None else 1.0)) if arm == "EDIT" else None
    if estimand == "total":
        prompt = {"B2": "prompt_b2", "B3": "prompt_b3"}.get(arm, "prompt")
        ids = e.chat_ids(row[prompt])
        if arm == "B1":
            a.bias = {"ids": b1_ids, "value": float(dose), "mode": spec.get("variant") or "online"}
        if arm == "B4":
            a.steer = {"layer": EN.STEER_LAYER, "vector": CM.b4_vector(float(dose)), "mode": "online"}
        g = e.generate(ids, a, mode="total", edit=edit)
        text, base_text = g.text, ""
    else:
        if not rec.get("p_ans_ok"):
            return {"case_id": row["case_id"], "skipped": "NM"}
        if arm == "B2":
            ids = CM.b2_prefix(e, row, rec)
        elif arm == "B3":
            ids = CM.b3_prefix(e, row, rec)
        else:
            ids = CM.prefix_ids(rec)
        if arm == "B1":
            a.bias = {"ids": b1_ids, "value": float(dose), "mode": "prefix"}
        if arm == "B4":
            a.steer = {"layer": EN.STEER_LAYER, "vector": CM.b4_vector(float(dose)), "mode": "prefix"}
        a.hybrid = estimand == "hybrid"
        g = e.generate(ids, a, mode="prefix", edit=edit)
        n_head = len(e.chat_ids(row["prompt_b3"])) if arm == "B3" else rec["n_prompt"]
        base_text = e.decode(ids[n_head:])
        text = base_text + g.text
    out = {"case_id": row["case_id"], "estimand": estimand, "spec": spec, "gen_ids": g.gen_ids, "text": text,
           "n_input": len(ids), "trigger_step": g.trigger_step, "stop": g.stop, "gen_s": round(g.wall_s, 3),
           **CM.grade(row, text, tc)}
    return out


def arm_dir(run: Path, split: str, key: str, estimand: str) -> Path:
    return Path(run) / "arms" / split / key / estimand


def run_arm(e, rws: list[dict], run: Path, spec: dict, estimand: str, deadline, tc, b1_ids=None) -> int:
    key = arm_key(spec["arm"], spec.get("dose"), spec.get("variant"), spec.get("seed"), spec.get("edit_id"))
    for r in rws:
        path = arm_dir(run, r["split"], key, estimand) / f"{r['case_id']}.json"
        if path.exists():
            continue
        if deadline and time.time() > deadline:
            return EXIT_DEADLINE
        rec = CM.b0(r["split"], r["case_id"], run)
        if rec is None:
            raise RuntimeError(f"{r['case_id']}: no B0 record")
        CM.write_json(path, run_arm_row(e, r, rec, spec, estimand, tc, b1_ids))
    return 0


# ---------------------------------------------------------------------------------------------- no-op gate

def zero_edit(e, edit_dir: Path) -> str:
    """A zero rank-1 dense edit on down_proj of the last layer (a no-op that goes through the whole edit path)."""
    shapes = e.edits.shapes()
    L = e.n_layers - 1
    name = ED.EDITABLE["down_proj"].format(l=L)
    out, inn = shapes[name]
    comp = {"layer": L, "tensor": "down_proj", "kind": "dense", "U": np.zeros((out, 1)), "V": np.zeros((inn, 1))}
    man = ED.save_edit(edit_dir, [comp], shapes, "comparator", meta={"purpose": "no-op gate"},
                       model_fingerprint=e.fingerprint())
    return man["edit_id"]


def noop_gate(e, rws: list[dict], run: Path, tc: list[str]) -> dict:
    """Zero edit, total effect and controlled prefix, on rows with B0 records; every token mismatch listed. Also B0's
    own controlled-prefix replay (recomputed KV vs B0's incremental KV) as a separate mismatch class."""
    eid = zero_edit(e, e.edits.edit_dir)
    sha_before = {n: e.edits.tensor_sha(n) for n in [ED.EDITABLE["down_proj"].format(l=e.n_layers - 1)]}
    res = {"edit_id": eid, "rows": [], "total_mismatch": [], "prefix_mismatch": [], "prefix_b0_vs_total": []}
    for r in rws:
        rec = CM.b0(r["split"], r["case_id"], run)
        if rec is None:
            continue
        t = run_arm_row(e, r, rec, {"arm": "EDIT", "edit_id": eid, "dose": 1.0}, "total", tc)
        ok_t = t["gen_ids"] == rec["gen_ids"]
        item = {"case_id": r["case_id"], "total_equal": ok_t, "cls_b0": rec["cls"], "cls_noop": t["cls"]}
        if not ok_t:
            k = next((i for i, (a, b) in enumerate(zip(t["gen_ids"], rec["gen_ids"])) if a != b),
                     min(len(t["gen_ids"]), len(rec["gen_ids"])))
            item["first_diff_token"] = k
            res["total_mismatch"].append(r["case_id"])
        if rec.get("p_ans_ok"):
            p = run_arm_row(e, r, rec, {"arm": "EDIT", "edit_id": eid, "dose": 1.0}, "prefix", tc)
            b = run_arm_row(e, r, rec, {"arm": "B0"}, "prefix", tc)
            b0_answer = rec["gen_ids"][rec["j"]:]
            item.update({"prefix_noop_equals_prefix_b0": p["gen_ids"] == b["gen_ids"],
                         "prefix_b0_equals_b0_answer": b["gen_ids"] == b0_answer[:len(b["gen_ids"])],
                         "cls_prefix_b0": b["cls"]})
            if p["gen_ids"] != b["gen_ids"]:
                res["prefix_mismatch"].append(r["case_id"])
            if b["gen_ids"] != b0_answer[:len(b["gen_ids"])]:
                res["prefix_b0_vs_total"].append(r["case_id"])
        res["rows"].append(item)
    e.edits.assert_clean()
    sha_after = {n: e.edits.tensor_sha(n) for n in sha_before}
    res["weights_restored"] = sha_before == sha_after
    res["pass"] = not res["total_mismatch"] and not res["prefix_mismatch"] and res["weights_restored"]
    return res


# ---------------------------------------------------------------------------------------------- (c) and (d)

def shortcut_row(e, row: dict, rec: dict, arm: "EN.Arm", tc) -> dict:
    """The B1 / B4 online total effect as B0's ids through its first marker + a triggered continuation."""
    t = rec.get("first_trigger_step")
    if t is None:
        return {"gen_ids": list(rec["gen_ids"]), "text": rec["text"], "no_marker": True}   # never triggers: = B0
    ids = list(rec["prompt_ids"]) + list(rec["gen_ids"][:t])
    head = e.decode(rec["gen_ids"][:t])
    g = e.generate(ids, arm, mode="resume", max_new=EN.MAX_NEW_TOTAL - t, text_before=head)
    gen = list(rec["gen_ids"][:t]) + g.gen_ids
    return {"gen_ids": gen, "text": e.decode(gen), "gen_s": round(g.wall_s, 3)}


def mixed_pick(rws: list[dict], run: Path, n: int) -> list[dict]:
    """n rows with B0 records, round-robin over (task, ANS / CF / UNA) cells, then sorted by case id (fixed order)."""
    have = [r for r in rws if CM.b0("E", r["case_id"], run) is not None]
    groups = {}
    for r in have:
        groups.setdefault((r["task"], r["set"] if r["set"] in ("ANS", "CF") else "UNA"), []).append(r)
    pick, i = [], 0
    while len(pick) < n and any(len(v) > i for v in groups.values()):
        pick += [v[i] for _, v in sorted(groups.items()) if len(v) > i]
        i += 1
    return sorted(pick[:n], key=lambda r: r["case_id"])


def parity(e, rws: list[dict], run: Path, tc, n: int = 24) -> dict:
    """Build review 8c: B1 −4 and B4 α 2, full online vs the shortcut, on n mixed E rows; any token mismatch blocks."""
    b1_ids = CM.b1_token_set(e.tok)
    arms = {"B1_-4": EN.Arm(bias={"ids": b1_ids, "value": -4.0, "mode": "online"}),
            "B4_2": EN.Arm(steer={"layer": EN.STEER_LAYER, "vector": CM.b4_vector(2.0), "mode": "online"})}
    pick = mixed_pick(rws, run, n)
    out = {"rows": [], "mismatch": []}
    for r in pick:
        rec = CM.b0("E", r["case_id"], run)
        for name, arm in arms.items():
            t0 = time.perf_counter()
            full = e.generate(list(rec["prompt_ids"]), arm, mode="total")
            t_full = time.perf_counter() - t0
            t0 = time.perf_counter()
            sc = shortcut_row(e, r, rec, arm, tc)
            t_sc = time.perf_counter() - t0
            same = full.gen_ids == sc["gen_ids"]
            item = {"case_id": r["case_id"], "arm": name, "equal": same, "full_s": round(t_full, 2),
                    "shortcut_s": round(t_sc, 2), "cls_full": CM.grade(r, full.text, tc)["cls"],
                    "cls_shortcut": CM.grade(r, sc["text"], tc)["cls"], "no_marker": sc.get("no_marker", False),
                    "trigger_full": full.trigger_step, "trigger_b0": rec.get("first_trigger_step")}
            if not same:
                k = next((j for j, (a, b) in enumerate(zip(full.gen_ids, sc["gen_ids"])) if a != b),
                         min(len(full.gen_ids), len(sc["gen_ids"])))
                item["first_diff"] = k
                out["mismatch"].append(f"{r['case_id']}:{name}")
            out["rows"].append(item)
    out["n_rows"], out["pass"] = len(pick), not out["mismatch"]
    return out


def batching(e, rws: list[dict], run: Path, tc, sizes=(8, 16), n: int = 24) -> dict:
    """Build review 8d / the batched route, measurement only: B0 at fixed batch layouts (rows sorted by case id, cut
    into consecutive batches, left padding with an attention mask) vs the batch-1 B0 records: token and graded-class
    agreement, rows per hour."""
    have = mixed_pick(rws, run, n)
    recs = [CM.b0("E", r["case_id"], run) for r in have]
    b1_s = sum(x["gen_s"] for x in recs)
    out = {"n_rows": len(have), "batch1_rows_per_h": round(3600 * len(have) / b1_s, 1), "by_size": {}}
    for bs in sizes:
        t0 = time.perf_counter()
        gens = []
        for i in range(0, len(have), bs):
            gens += [g.gen_ids for g in e.generate_batched([list(x["prompt_ids"]) for x in recs[i:i + bs]])]
        wall = time.perf_counter() - t0
        tok_eq = sum(g == x["gen_ids"] for g, x in zip(gens, recs))
        cls_eq = sum(CM.grade(r, e.decode(g), tc)["cls"] == x["cls"] for r, g, x in zip(have, gens, recs))
        diffs = [{"case_id": r["case_id"], "first_diff": next((j for j, (a, b) in enumerate(zip(g, x["gen_ids"]))
                                                                if a != b), min(len(g), len(x["gen_ids"]))),
                  "cls_b1": x["cls"], "cls_batched": CM.grade(r, e.decode(g), tc)["cls"]}
                 for r, g, x in zip(have, gens, recs) if g != x["gen_ids"]]
        out["by_size"][str(bs)] = {"rows_per_h": round(3600 * len(have) / wall, 1), "wall_s": round(wall, 1),
                                   "token_equal": tok_eq, "class_equal": cls_eq, "diffs": diffs,
                                   "peak_rss_gb": round(EN.peak_rss_gb(), 2)}
    return out


# ---------------------------------------------------------------------------------------------- B6

def make_b6(e, candidate_id: str, dose: float, seeds=CM.B6_SEEDS) -> dict[int, str]:
    """B6 for a candidate at its selected dose: per component, a seeded Gaussian on the same support matched per
    column (matrices) or in L2 (RMSNorm), seeds 1-5; saved as comparator edits. Returns seed -> edit id."""
    man, comps = ED.load_edit(e.edits.edit_dir, candidate_id)
    shapes = e.edits.shapes()
    out = {}
    for seed in seeds:
        rs = []
        for m, c in zip(man["components"], comps):
            scaled = dict(c)
            for k in ("U", "D"):
                if k in scaled:
                    scaled[k] = np.asarray(scaled[k], np.float64) * dose
            rs.append(ED.random_spec(scaled, tuple(m["shape"]), seed))
        mb = ED.save_edit(e.edits.edit_dir, rs, shapes, "comparator",
                          meta={"arm": "B6", "candidate": candidate_id, "dose": dose, "seed": seed},
                          model_fingerprint=e.fingerprint())
        out[seed] = mb["edit_id"]
    return out


# ---------------------------------------------------------------------------------------------- B5

def b5_fit(e, rws: list[dict], run: Path) -> dict:
    """§5.2 B5 on E only (see module doc). Writes the direction and the four dose edits (comparator family)."""
    from memctl.analysis import stats51 as S
    X, y = [], []
    for r in rws:
        if r["set"] != "ANS":
            continue
        rec = CM.b0(r["split"], r["case_id"], run)
        if rec is None or not rec.get("p_ans_ok") or rec["cls"] not in ("abstention", "gold"):
            continue
        X.append(np.load(CM.b0_path(r["split"], r["case_id"], run).with_suffix(".npy")))
        y.append(1 if rec["cls"] == "abstention" else 0)
    X, y = np.stack(X).astype(np.float64), np.asarray(y)
    per_layer = []
    for l in range(X.shape[1]):
        d = X[y == 1, l].mean(0) - X[y == 0, l].mean(0)
        u = d / np.linalg.norm(d)
        per_layer.append({"layer": l, "auroc": S.auroc(X[:, l] @ u, y), "norm": float(np.linalg.norm(d))})
    best = max(per_layer, key=lambda x: (x["auroc"], -x["layer"]))
    L = best["layer"]
    d = X[y == 1, L].mean(0) - X[y == 0, L].mean(0)
    d = d / np.linalg.norm(d)
    shapes = e.edits.shapes()
    ids = {}
    for beta in CM.B5_DOSES:
        comps = []
        for l in range(e.n_layers):
            W = {ED.EDITABLE[t].format(l=l): e.edits.original_np(ED.EDITABLE[t].format(l=l)) for t in ("o_proj", "down_proj")}
            comps += ED.orthogonalize_spec(d, W, [l], beta)
        man = ED.save_edit(e.edits.edit_dir, comps, shapes, "comparator",
                           meta={"arm": "B5", "beta": beta, "layer": L}, model_fingerprint=e.fingerprint())
        ids[str(beta)] = man["edit_id"]
    e.edits.orig.clear()
    out = {"n_abst": int((y == 1).sum()), "n_gold": int((y == 0).sum()), "layer": L, "per_layer": per_layer,
           "direction_sha256": ED.sha_array(d.astype(np.float32)), "edit_ids": ids, "doses": list(CM.B5_DOSES),
           "layer_set": "all layers, o_proj and down_proj"}
    np.save(Path(run) / "b5_direction.npy", d.astype(np.float32))
    CM.write_json(Path(run) / "b5_fit.json", out)
    return out


# ---------------------------------------------------------------------------------------------- timing

def timing(e, rws: list[dict], run: Path, tc: list[str], n: int = 3) -> dict:
    """One real timed unit of each pass type (n rows of mixed hardness), peak RSS throughout. Writes timing.json."""
    out = {"engine": e.fingerprint(), "load_s": round(e.load_s, 2), "rss_after_load_gb": round(EN.rss_gb(), 2),
           "peak_after_load_gb": round(EN.peak_rss_gb(), 2), "units": []}
    pick = [r for r in rws if r["set"] == "ANS" and r["rep"] == "a"]
    pick = [next(r for r in pick if r["hardness"] == h and r["task"] == t)
            for h, t in (("H0", "DE"), ("H1", "RC"), ("H2", "TU"))][:n]
    shapes = e.edits.shapes()
    L = e.n_layers - 8
    name = ED.EDITABLE["down_proj"].format(l=L)
    rng = np.random.default_rng(0)
    comps = [{"layer": L, "tensor": "down_proj", "kind": "dense", "U": rng.standard_normal((shapes[name][0], 4)) * 1e-3,
              "V": rng.standard_normal((shapes[name][1], 4)) * 1e-3},
             {"layer": L, "tensor": "o_proj", "kind": "cols", "idx": np.arange(64),
              "D": rng.standard_normal((shapes[ED.EDITABLE["o_proj"].format(l=L)][0], 64)) * 1e-3},
             {"layer": L + 1, "tensor": "down_proj", "kind": "rows", "idx": np.arange(128),
              "D": rng.standard_normal((128, shapes[name][1])) * 1e-3},
             {"layer": L, "tensor": "input_layernorm", "kind": "coords", "idx": np.arange(256),
              "D": rng.standard_normal(256) * 1e-3}]
    t0 = time.perf_counter()
    man = ED.save_edit(e.edits.edit_dir, comps, shapes, "candidate", meta={"purpose": "timing"},
                       originals=e.edits.original_np, model_fingerprint=e.fingerprint())
    out["save_edit_s"] = round(time.perf_counter() - t0, 2)
    t0 = time.perf_counter()
    e.edits.apply(man["edit_id"])
    out["apply_s"] = round(time.perf_counter() - t0, 3)
    t0 = time.perf_counter()
    e.edits.revert()
    out["revert_s"] = round(time.perf_counter() - t0, 3)
    e.edits.assert_clean()
    for r in pick:
        t0 = time.perf_counter()
        rec = run_b0_row(e, r, run, tc) if CM.b0(r["split"], r["case_id"], run) is None else CM.b0(r["split"], r["case_id"], run)
        CM.write_json(CM.b0_path(r["split"], r["case_id"], run), rec)
        u = {"case_id": r["case_id"], "n_prompt": rec["n_prompt"], "n_gen": len(rec["gen_ids"]),
             "b0_total_s": rec["gen_s"], "cls": rec["cls"], "p_ans_ok": rec["p_ans_ok"]}
        t1 = time.perf_counter()
        x = run_arm_row(e, r, rec, {"arm": "EDIT", "edit_id": man["edit_id"], "dose": 1.0}, "total", tc)
        u.update({"edit_total_s": round(time.perf_counter() - t1, 2), "edit_total_n_gen": len(x["gen_ids"])})
        if rec["p_ans_ok"]:
            t1 = time.perf_counter()
            x = run_arm_row(e, r, rec, {"arm": "EDIT", "edit_id": man["edit_id"], "dose": 1.0}, "prefix", tc)
            u.update({"edit_prefix_s": round(time.perf_counter() - t1, 2), "prefix_n_gen": len(x["gen_ids"])})
            ids = CM.prefix_ids(rec)
            t1 = time.perf_counter()
            rd = e.read(ids, positions=[len(ids) - 1], layers=list(range(e.n_layers)),
                        comps=[(L, "down_proj"), (L, "o_proj")], logit_positions=[len(ids) - 1])
            u.update({"read_s": round(time.perf_counter() - t1, 2), "read_n_tokens": len(ids)})
        u["peak_rss_gb"] = round(EN.peak_rss_gb(), 2)
        out["units"].append(u)
    out["peak_rss_gb"] = round(EN.peak_rss_gb(), 2)
    CM.write_json(Path(run) / "timing.json", out)
    return out


# ---------------------------------------------------------------------------------------------- main

def require_f2(run: Path) -> dict:
    """Sealed splits need the F2 freeze: manifest_f2.json + its marker, verified against git (§49's guard)."""
    from memctl.analysis import abst49 as A
    d = Path(run) / "design.json"
    if not d.exists():
        raise PermissionError("no F2 design: H / T are sealed")
    return A.require_design(Path(run))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", required=True, choices=("timing", "b0", "noop", "b5-fit", "inventory", "arm", "parity",
                                                      "batching"))
    p.add_argument("--split", default="E")
    p.add_argument("--sets", default=None, help="comma list of sets (default all)")
    p.add_argument("--n", type=int, default=24)
    p.add_argument("--backend", default="hf", choices=("hf", "dummy"))
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--run", default=str(CM.RUN))
    p.add_argument("--hours", type=float, default=None)
    p.add_argument("--arm"), p.add_argument("--dose", type=float), p.add_argument("--variant")
    p.add_argument("--edit-id"), p.add_argument("--estimand", default="total", choices=ESTIMANDS)
    a = p.parse_args(argv)
    run = Path(a.run)
    run.mkdir(parents=True, exist_ok=True)
    if a.stage == "inventory":
        import exp51_inventory as INV
        out = INV.inventory(run)
        print(json.dumps(out["totals"]))
        return 0
    if a.split in CM.SEALED:
        require_f2(run)
    tc = CM.t_cities()
    rws = CM.rows(a.split, allow_sealed=a.split in CM.SEALED)
    if a.sets:
        keep = set(a.sets.split(","))
        rws = [r for r in rws if r["set"] in keep]
    deadline = time.time() + a.hours * 3600 if a.hours else None
    e = EN.load(a.backend, a.threads, edit_dir=run / "edits")
    if a.stage == "timing":
        print(json.dumps(timing(e, CM.rows("E"), run, tc), indent=1))
        return 0
    if a.stage == "b0":
        by_task = [[r for r in rws if r["task"] == t] for t in ("DE", "RC", "TU")]
        order = [r for grp in __import__("itertools").zip_longest(*by_task) for r in grp if r is not None]
        return run_b0(e, order, run, deadline, tc)            # tasks interleaved, so RC / TU exist early
    if a.stage == "noop":
        have = [r for r in rws if CM.b0(r["split"], r["case_id"], run) is not None]
        by_task = {t: [r for r in have if r["task"] == t] for t in ("DE", "RC", "TU")}
        sel = mixed_pick(rws, run, a.n)                                         # mixed tasks and sets
        res = noop_gate(e, sel, run, tc)
        CM.write_json(run / "noop_gate.json", res)
        print(json.dumps({k: v for k, v in res.items() if k != "rows"}))
        return 0 if res["pass"] else 3
    if a.stage == "parity":
        res = parity(e, rws, run, tc, a.n)
        CM.write_json(run / "parity_shortcut.json", res)
        print(json.dumps({k: v for k, v in res.items() if k != "rows"}))
        return 0 if res["pass"] else 3
    if a.stage == "batching":
        res = batching(e, rws, run, tc)
        CM.write_json(run / "batching.json", res)
        print(json.dumps(res))
        return 0
    if a.stage == "b5-fit":
        if a.split != "E":
            raise SystemExit("B5 is fit on E only")
        print(json.dumps({k: v for k, v in b5_fit(e, rws, run).items() if k != "per_layer"}))
        return 0
    if a.stage == "arm":
        b1 = CM.b1_token_set(e.tok) if a.arm == "B1" else None
        spec = {"arm": a.arm, "dose": a.dose, "variant": a.variant, "edit_id": a.edit_id}
        return run_arm(e, rws, run, spec, a.estimand, deadline, tc, b1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
