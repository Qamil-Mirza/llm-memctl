#!/usr/bin/env python
"""§51 stage 4 (pod S) and stage 5 (pod H + T-city): batched arms on the A40 (fixed layout: rows sorted by case id,
consecutive batches of LAYOUT), the in-pod S summary with rule-driven selection -> f2_candidate.json, and the guarded
confirmatory stage on H and T-city. Resumable per row. Spec: proposal rev 1a §5-§8, §13; D-72, D-73; the Advisor's
stage-4 pod structure ruling (two pods).

    python scripts/exp51_stage4.py --stage replay   --root R --subs SUBS       # §10 CPU<->pod check on 200 E rows
    python scripts/exp51_stage4.py --stage s-arms   --root R --subs SUBS       # B0 + every frozen arm on S
    python scripts/exp51_stage4.py --stage s-select --root R --subs SUBS       # selection -> f2_candidate.json
    python scripts/exp51_stage4.py --stage h-run    --root R --subs SUBS       # guarded (require_f2); H then T-city
    python scripts/exp51_stage4.py --stage h-report --root R

Pod S's bundle holds no H or T-city rows; pod H+T's chain runs require_f2 (git-verified design.json = the committed
F2) before any sealed row is read. `--deadline-s` stops between rows (exit 9, resumable).

Fixed choices of this build (each in f2_candidate.json "rules"):
- Every generation pass is batched at LAYOUT (64) except M2 (transplant / clamp: batch 1, hooks), the hybrid
  estimand (batch 1) and teacher-forced reads (batch 1). B1 / B4 online total effect = B0 prefix through the first
  marker + triggered continuation (approved shortcut (c); parity 48/48 on the laptop); a row with no marker equals B0.
- R arms run under the total effect on every row of the split; M arms under the controlled prefix on ANS + UNA rows
  with a P_ans.
- M dose candidates: the submission's own delta x CAND_DOSES; f-intervention doses: f_int_sign x {0.5, 1, 2, 4} x SD
  of f over the split's ANS rows with a P_ans (the SD is S's, frozen at F2).
- Primary component: among the M candidate's components, the one whose revert removes most of the oriented endpoint
  effect on S at the M dose (ties: the editor's most important component, then component index).
- Manipulations for the activity tests (per base, mean over the hardness levels present in both rows):
  evidence_removed = UNA-rel − ANS rep a; city_changed = CF − ANS rep a; distractors_added = ANS rep a H2 − H1;
  unrelated_vs_absent = UNA-swap − UNA-rel (H1). f is read at each row's own B0 P_ans. distractors_added: same base,
  rep and evidence text (every H1 line is kept in H2, the core facts are identical); the 7 added lines are distractors,
  and because F2 dates lines by position the dates move with them (on E: the question date in 144 / 144 pairs, the
  evidence line's date in 129 / 144).
- §13.4(b) population pair: (the declared endpoint population, UNA); if the declared population is UNA, (UNA, FA).
  The f-intervention grid and the B1 signed grid run on the union of the pair; the pair is written to F2
  (M_followups.pair) and h_report reads it from there.
- Pod-side replay check (§10, 200 E rows sorted by case id: B0 and submission 1 at dose 1): batched layout and batch 1
  vs the laptop's batch-1 records; the S arms run only if graded-class agreement >= 194 / 200 in all four comparisons
  (the 62 / 64 rule scaled; Advisor ruling (c)), else the chain stops (exit 8); every mismatch is listed either way,
  for the Advisor's investigation before S results are used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import exp51_common as CM  # noqa: E402
import exp51_engine as EN  # noqa: E402
from memctl.analysis import edit51 as ED  # noqa: E402
from memctl.analysis import mech51 as MK  # noqa: E402
from memctl.analysis import stats51 as S  # noqa: E402

LAYOUT = 64
EXIT_DEADLINE, EXIT_REPLAY, EXIT_GUARD = 9, 8, 4
REPLAY_MIN = 62 / 64
F_INT_MULT = (0.5, 1.0, 2.0, 4.0)
MANIPS = ("evidence_removed", "city_changed", "distractors_added", "unrelated_vs_absent")


class Deadline(Exception):
    pass


# ---------------------------------------------------------------------------------------------- stores

def b0_path(root: Path, split: str, cid: str) -> Path:
    return Path(root) / "b0" / split / f"{cid}.json"


def arm_key(spec: dict) -> str:
    parts = [spec["arm"]]
    for k in ("edit_id", "dose", "variant", "seed", "tag"):
        if spec.get(k) is not None:
            parts.append(f"{k[0]}{spec[k]:g}" if isinstance(spec[k], float) else f"{k[0]}{spec[k]}")
    return "_".join(str(p) for p in parts)


def out_path(root: Path, split: str, spec: dict, estimand: str, cid: str) -> Path:
    return Path(root) / "arms" / split / arm_key(spec) / estimand / f"{cid}.json"


def load_graded(root, split, spec, estimand, rows) -> dict:
    return {r["case_id"]: CM.read_json(out_path(root, split, spec, estimand, r["case_id"])) for r in rows}


def chunks(rows: list, n: int = LAYOUT):
    rows = sorted(rows, key=lambda r: r["case_id"])
    for i in range(0, len(rows), n):
        yield rows[i:i + n]


# ---------------------------------------------------------------------------------------------- B0 (batched)

def run_b0(e, rows, root, split, tc, deadline=None) -> None:
    todo = [r for r in rows if not b0_path(root, split, r["case_id"]).exists()]
    for ch in chunks(todo):
        if deadline and time.time() > deadline:
            raise Deadline()
        ids = [e.chat_ids(r["prompt"]) for r in ch]
        gs = e.generate_batched(ids, mode="total")
        for r, x, g in zip(ch, ids, gs):
            d = CM.designated(e.decode, g.gen_ids)
            rec = {"case_id": r["case_id"], "split": split, "prompt_ids": x, "gen_ids": g.gen_ids, "text": g.text,
                   "n_prompt": len(x), "stop": g.stop, "first_trigger_step": g.trigger_step, "j": d["j"],
                   "p_ans_ok": d["ok"], "p_ans_why": d["why"], "p_ans": (len(x) + d["j"] - 1) if d["ok"] else None,
                   "engine": e.fingerprint(), "layout": LAYOUT, **CM.grade(r, g.text, tc)}
            rec["populations"] = CM.population(r, rec, tc)
            CM.write_json(b0_path(root, split, r["case_id"]), rec)


# ---------------------------------------------------------------------------------------------- arms (batched)

def build(e, row, rec, spec, estimand):
    """(ids, mode, max_new, text_before, head_text, arm, copy_b0) for one row."""
    arm_name, dose = spec["arm"], spec.get("dose")
    a = EN.Arm()
    if arm_name == "B1":
        a.bias = {"ids": spec["b1_ids"], "value": float(dose),
                  "mode": "prefix" if estimand == "prefix" else (spec.get("variant") or "online")}
    if arm_name == "B4":
        a.steer = {"layer": EN.STEER_LAYER, "vector": CM.b4_vector(float(dose)),
                   "mode": "prefix" if estimand == "prefix" else "online"}
    if estimand == "total":
        online = arm_name in ("B1", "B4") and (spec.get("variant") or "online") == "online"
        if online:                                               # shortcut (c)
            t = rec.get("first_trigger_step")
            if t is None:
                return None, None, None, None, None, a, True
            if a.bias:
                a.bias = {**a.bias, "mode": "prefix"}
            if a.steer:
                a.steer = {**a.steer, "mode": "prefix"}
            ids = list(rec["prompt_ids"]) + list(rec["gen_ids"][:t])
            head = e.decode(rec["gen_ids"][:t])
            return ids, "resume", EN.MAX_NEW_TOTAL - t, head, head, a, False
        prompt = {"B2": "prompt_b2", "B3": "prompt_b3"}.get(arm_name, "prompt")
        ids = list(rec["prompt_ids"]) if prompt == "prompt" else e.chat_ids(row[prompt])
        return ids, "total", EN.MAX_NEW_TOTAL, "", "", a, False
    if not rec.get("p_ans_ok"):
        return None, None, None, None, None, a, None
    if arm_name == "B2":
        ids = CM.b2_prefix(e, row, rec)
        head = e.decode(ids[rec["n_prompt"]:])
    elif arm_name == "B3":
        ids = CM.b3_prefix(e, row, rec)
        head = e.decode(rec["gen_ids"][:rec["j"]])
    else:
        ids = CM.prefix_ids(rec)
        head = e.decode(rec["gen_ids"][:rec["j"]])
    return ids, "prefix", EN.MAX_NEW_RELEASE, "", head, a, False


def run_arm(e, rows, root, split, spec, estimand, tc, deadline=None) -> None:
    """One arm on the rows (batched at LAYOUT; the edit, if any, applies to the whole batch)."""
    edit = (spec["edit_id"], float(spec.get("dose", 1.0) if spec["arm"] == "EDIT" else 1.0)) \
        if spec.get("edit_id") else None
    todo = [r for r in rows if not out_path(root, split, spec, estimand, r["case_id"]).exists()]
    keyed = {}
    for r in todo:
        rec = CM.read_json(b0_path(root, split, r["case_id"]))
        built = build(e, r, rec, spec, estimand)
        ids, mode, mx, tb, head, a, copy = built
        if copy is None:
            CM.write_json(out_path(root, split, spec, estimand, r["case_id"]), {"case_id": r["case_id"], "skipped": "NM"})
            continue
        if copy:
            CM.write_json(out_path(root, split, spec, estimand, r["case_id"]),
                          {"case_id": r["case_id"], "gen_ids": rec["gen_ids"], "text": rec["text"],
                           "no_marker_equals_b0": True, **CM.grade(r, rec["text"], tc)})
            continue
        keyed.setdefault(mode, []).append((r, rec, ids, mx, tb, head, a))
    for mode, items in keyed.items():
        items.sort(key=lambda x: x[0]["case_id"])
        for i in range(0, len(items), LAYOUT):
            if deadline and time.time() > deadline:
                raise Deadline()
            ch = items[i:i + LAYOUT]
            gs = e.generate_batched([x[2] for x in ch], ch[0][6], mode=mode, max_new=[x[3] for x in ch],
                                    edit=edit, text_before=[x[4] for x in ch])
            for (r, rec, ids, mx, tb, head, a), g in zip(ch, gs):
                text = (head + g.text) if mode in ("prefix", "resume") else g.text
                gen = (list(rec["gen_ids"][:rec["first_trigger_step"]]) + g.gen_ids) if mode == "resume" else g.gen_ids
                CM.write_json(out_path(root, split, spec, estimand, r["case_id"]),
                              {"case_id": r["case_id"], "gen_ids": gen, "text": text, "stop": g.stop,
                               "spec": {k: v for k, v in spec.items() if k != "b1_ids"}, "estimand": estimand,
                               **CM.grade(r, text, tc)})


# ---------------------------------------------------------------------------------------------- submissions

def load_subs(subs: Path) -> list[dict]:
    out = []
    for p in sorted(Path(subs).glob("[0-9].json"), key=lambda p: int(p.stem)):
        s = json.loads(p.read_text())
        s["index"] = int(p.stem)
        s["f_direction_vec"] = np.load(Path(subs) / f"{p.stem}_f_direction.npy")
        out.append(s)
    return out


def sub_components(root, s) -> tuple[dict, list]:
    return ED.load_edit(Path(root) / "edits", s["edit_id"])


def n_params(root, edit_id: str) -> int:
    man, comps = ED.load_edit(Path(root) / "edits", edit_id)
    return sum(ED.n_changed(c, tuple(m["shape"])) for m, c in zip(man["components"], comps))


def r_specs(e, subs, b5_ids: dict, b1_ids) -> list[tuple[dict, str]]:
    """Every R arm x dose under the total effect (the R grid frozen at F0)."""
    sp = [({"arm": "B1", "dose": d, "variant": "online", "b1_ids": b1_ids}, "total") for d in CM.B1_GRID]
    sp += [({"arm": "B2"}, "total"), ({"arm": "B3"}, "total")]
    sp += [({"arm": "B4", "dose": a}, "total") for a in CM.B4_GRID]
    sp += [({"arm": "EDIT", "edit_id": b5_ids[str(b)], "dose": 1.0, "tag": f"B5b{b:g}"}, "total") for b in CM.B5_DOSES]
    for s in subs:
        sp += [({"arm": "EDIT", "edit_id": s["edit_id"], "dose": d, "tag": f"sub{s['index']}"}, "total")
               for d in CM.CAND_DOSES]
    return sp


def m_specs(subs) -> list[tuple[dict, str]]:
    sp = [({"arm": "B0"}, "prefix")]
    for s in subs:
        sp += [({"arm": "EDIT", "edit_id": s["edit_id"], "dose": d, "tag": f"sub{s['index']}"}, "prefix")
               for d in CM.CAND_DOSES]
    return sp


# ---------------------------------------------------------------------------------------------- analysis helpers

def labels_meta(root, split, rows):
    labels, meta = {}, {}
    for r in rows:
        rec = CM.read_json(b0_path(root, split, r["case_id"]))
        labels[r["case_id"]] = rec["populations"] if rec else []
        meta[r["case_id"]] = r
    return labels, meta


def total_graded(root, split, spec, rows):
    """The total-effect graded records of an arm; B0 = the split's own B0 records."""
    if spec["arm"] == "B0":
        return {r["case_id"]: CM.read_json(b0_path(root, split, r["case_id"])) for r in rows}
    return load_graded(root, split, spec, "total", rows)


def fa_bases_by_task(labels, meta) -> dict:
    out = defaultdict(set)
    for c, l in labels.items():
        if "FA" in l:
            out[meta[c]["task"]].add(meta[c]["base_key"])
    return {k: len(v) for k, v in out.items()}


def endpoint_values(graded, labels, meta, ep) -> dict:
    return MK.per_base(graded, labels, meta, ep["population"], ep["outcome"])


def interaction_pair(ep) -> list[str]:
    """§13.4(b): the declared population and UNA (UNA declared -> FA)."""
    return [ep["population"], "FA" if ep["population"] == "UNA" else "UNA"]


def sign_of(ep) -> int:
    return 1 if ep["direction"] == "increase" else -1


def closeness(spec) -> float:
    d = spec.get("dose")
    return abs(d) if isinstance(d, (int, float)) and np.isfinite(d) else 1e9


# ---------------------------------------------------------------------------------------------- f and manipulations

def f_of(e, rec, site, vec) -> float | None:
    import exp51_mech as MC
    if not rec or not rec.get("p_ans_ok"):
        return None
    return MC.f_value(e, rec, (int(site["layer"]), site["tensor"]), vec)


def pairs_for(rows) -> dict[str, list[tuple[str, str, str]]]:
    """manipulation -> [(base_key, case_a, case_b)] with Δ = f(b) − f(a)."""
    by = {r["case_id"]: r for r in rows}
    out = {m: [] for m in MANIPS}
    for r in rows:
        if r["set"] != "ANS" or r["rep"] != "a":
            continue
        cid, b = r["case_id"], r["base_key"]
        h = r["hardness"]
        una = cid.replace("_ANSa_", "_UNArel_")
        cf = cid.replace("_ANSa_", "_CF_")
        if una in by:
            out["evidence_removed"].append((b, cid, una))
        if cf in by:
            out["city_changed"].append((b, cid, cf))
        if h == "H1" and cid.replace("_H1", "_H2") in by:
            out["distractors_added"].append((b, cid, cid.replace("_H1", "_H2")))
        if h == "H1" and una in by and cid.replace("_ANSa_H1", "_UNAswap_H1") in by:
            out["unrelated_vs_absent"].append((b, una, cid.replace("_ANSa_H1", "_UNAswap_H1")))
    return out


def delta_f(fvals: dict, pairs) -> dict:
    by = defaultdict(list)
    for b, a, c in pairs:
        if fvals.get(a) is not None and fvals.get(c) is not None:
            by[b].append(fvals[c] - fvals[a])
    return {b: float(np.mean(v)) for b, v in by.items()}


# ---------------------------------------------------------------------------------------------- S selection

def s_select(e, root, rows_S, subs, b5_ids, tc, B=S.B_DEFAULT) -> dict:
    """§5.3 / §8 / §13 selection on S -> the F2 candidate (no H anywhere)."""
    labels, meta = labels_meta(root, "S", rows_S)
    bases = MK.bases_of(meta)
    b0_tot = total_graded(root, "S", {"arm": "B0"}, rows_S)
    b0_rates = MK.rates(b0_tot, labels, meta)
    fab = fa_bases_by_task(labels, meta)
    b1_ids = CM.b1_token_set(e.tok)
    out = {"fa_bases_by_task": fab, "R": {}, "M": {}, "rules": __doc__.split("Fixed choices of this build")[1]}
    # R doses for comparators
    groups = defaultdict(list)
    for spec, est in r_specs(e, subs, b5_ids, b1_ids):
        tag = spec.get("tag") or spec["arm"]
        groups["B5" if tag.startswith("B5") else tag].append(spec)
    chosen = {}
    for g, specs in groups.items():
        sc = {}
        for sp in specs:
            u = MK.mixture(total_graded(root, "S", sp, rows_S), meta)
            sc[arm_key(sp)] = float(np.mean(list(u.values()))) if u else None
        best = MK.pick_dose({k: (v if v is not None else -1e9) for k, v in sc.items()},
                            {arm_key(sp): closeness(sp) if not (sp.get("tag") or "").startswith("B5") else
                             float(sp["tag"][3:]) for sp in specs})
        chosen[g] = {"spec": next({k: v for k, v in sp.items() if k != "b1_ids"} for sp in specs
                                  if arm_key(sp) == best), "U1_by_dose": sc}
    out["R"]["comparator_doses"] = {g: c for g, c in chosen.items() if not g.startswith("sub")}
    # R entry and candidate selection
    entries = []
    for s in subs:
        c = chosen[f"sub{s['index']}"]
        spec = {**c["spec"]}
        cr = MK.rates(total_graded(root, "S", spec, rows_S), labels, meta)
        ent = MK.r_entry(cr, b0_rates, bases, fab)
        entries.append({"index": s["index"], "edit_id": s["edit_id"], "dose": spec["dose"], **ent,
                        "U1": S.bw_mean(cr["U1"], list(cr["U1"])), "n_params": n_params(root, s["edit_id"])})
    ok = [x for x in entries if x["entry"]]
    out["R"]["entries"] = entries
    out["R"]["candidate"] = (sorted(ok, key=lambda x: (-x["U1"], x["n_params"], x["index"]))[0] if ok else None)
    # M: per submission, dose by the declared signed endpoint (controlled prefix)
    b0_pre = load_graded(root, "S", {"arm": "B0"}, "prefix", rows_S)
    m_rows = []
    for s in subs:
        ep, sg = s["signed_endpoint"], sign_of(s["signed_endpoint"])
        base = endpoint_values(b0_pre, labels, meta, ep)
        effs, degs, res_by = {}, {}, {}
        for d in CM.CAND_DOSES:
            spec = {"arm": "EDIT", "edit_id": s["edit_id"], "dose": d, "tag": f"sub{s['index']}"}
            gr = load_graded(root, "S", spec, "prefix", rows_S)
            v = endpoint_values(gr, labels, meta, ep)
            res = MK.paired(bases, v, base, B)
            effs[d], res_by[d] = res["est"], res
            degs[d] = S.bw_mean(MK.per_base(gr, labels, meta, "ALL", "Deg"), list(meta[c]["base_key"] for c in meta))
        dose = MK.m_dose(effs, degs, sg, {d: d for d in effs})
        if dose is None:
            m_rows.append({"index": s["index"], "eligible": False, "why": "no dose with Deg <= 0.10"})
            continue
        res = res_by[dose]
        se = float(np.std(res["draws"], ddof=1)) if res["B_valid"] > 1 else 0.0
        lo, hi = S.bounds(res)["lo"], S.bounds(res)["hi"]
        support = len([b for b in base])
        man, comps = sub_components(root, s)
        m_rows.append({"index": s["index"], "edit_id": s["edit_id"], "dose": dose, "effect": res["est"], "se": se,
                       "sign": sg, "bound_ok": bool((lo > 0) if sg > 0 else (hi < 0)), "support": support,
                       "n_components": len(comps), "n_params": n_params(root, s["edit_id"]),
                       "endpoint": ep})
    ranked = S.m_order([x for x in m_rows if x.get("effect") is not None])
    out["M"]["entries"], out["M"]["candidate"] = m_rows, (ranked[0] if ranked else None)
    return out


def m_followups(e, root, rows_S, subs, sel, tc, B=S.B_DEFAULT, deadline=None) -> dict:
    """Second S phase for the selected M candidate: per-component reverts (primary pick), B6 seeds, f values and the
    manipulations' Δf (SD -> margins), the f-intervention grid and the B1 signed grid (matched value), the predictor
    fit. Writes nothing on H."""
    import exp51_mech as MC
    import exp51_run as RUN
    m = sel["M"]["candidate"]
    if m is None:
        return {"none": True}
    s = next(x for x in subs if x["index"] == m["index"])
    labels, meta = labels_meta(root, "S", rows_S)
    bases = MK.bases_of(meta)
    ep, sg = s["signed_endpoint"], sign_of(s["signed_endpoint"])
    man, comps = sub_components(root, s)
    au = [r for r in rows_S if r["set"] in ("ANS", "UNA-rel", "UNA-swap")]
    star = endpoint_values(load_graded(root, "S", {"arm": "EDIT", "edit_id": s["edit_id"], "dose": m["dose"],
                                                   "tag": f"sub{s['index']}"}, "prefix", rows_S), labels, meta, ep)
    revert = {}
    for i in range(len(comps)):
        spec = {"arm": "REVERT", "edit_id": s["edit_id"], "dose": m["dose"], "tag": f"sub{s['index']}c{i}"}
        for r in sorted(au, key=lambda r: r["case_id"]):
            p = out_path(root, "S", spec, "prefix", r["case_id"])
            if p.exists():
                continue
            if deadline and time.time() > deadline:
                raise Deadline()
            rec = CM.read_json(b0_path(root, "S", r["case_id"]))
            if not rec.get("p_ans_ok"):
                CM.write_json(p, {"case_id": r["case_id"], "skipped": "NM"})
                continue
            CM.write_json(p, MC.revert_arm(e, r, rec, s["edit_id"], m["dose"], i, len(comps), tc))
        rv = endpoint_values(load_graded(root, "S", spec, "prefix", rows_S), labels, meta, ep)
        revert[i] = S.bw_mean({k: sg * (star[k] - rv[k]) for k in star if k in rv}, list(star))
    named = (s["most_important_component"]["layer"], s["most_important_component"]["tensor"])
    order = sorted(revert, key=lambda i: (-(revert[i] or -1e9), (man["components"][i]["layer"],
                                                                 man["components"][i]["tensor"]) != named, i))
    primary = order[0]
    b6 = RUN.make_b6(e, s["edit_id"], m["dose"])
    for seed, eid in b6.items():
        run_arm(e, au, root, "S", {"arm": "EDIT", "edit_id": eid, "dose": 1.0, "tag": f"B6s{seed}"}, "prefix", tc,
                deadline)
    # f values on S (unedited model, each row's own P_ans)
    site, vec = s["f_site"], s["f_direction_vec"]
    fpath = Path(root) / "f" / "S.json"
    fvals = CM.read_json(fpath, {})
    for r in sorted(rows_S, key=lambda r: r["case_id"]):
        if r["case_id"] in fvals:
            continue
        if deadline and time.time() > deadline:
            CM.write_json(fpath, fvals)
            raise Deadline()
        fvals[r["case_id"]] = f_of(e, CM.read_json(b0_path(root, "S", r["case_id"])), site, vec)
    CM.write_json(fpath, fvals)
    pr = pairs_for(rows_S)
    acts = []
    for a in s["activity"]:
        df = delta_f(fvals, pr[a["manipulation"]])
        acts.append({**a, "sd_S": MK.sd_of_base_means(df), "margin": 2 * MK.sd_of_base_means(df),
                     "mean_S": S.bw_mean(df, list(df)), "n_bases_S": len(df)})
    ans_f = [v for c, v in fvals.items() if v is not None and meta[c]["set"] == "ANS"]
    sd_f = float(np.std(ans_f, ddof=1))
    fint_doses = [s["f_int_sign"] * k * sd_f for k in F_INT_MULT]
    pair = interaction_pair(ep)
    fa_una = [r for r in rows_S if any(p in labels[r["case_id"]] for p in pair)]
    b1_ids = CM.b1_token_set(e.tok)
    fint_eff, fint_deg = {}, {}
    b0_pre = load_graded(root, "S", {"arm": "B0"}, "prefix", rows_S)
    base_ep = endpoint_values(b0_pre, labels, meta, ep)
    for alpha in fint_doses:
        spec = {"arm": "FINT", "dose": float(alpha), "tag": f"sub{s['index']}"}
        for r in sorted(fa_una, key=lambda r: r["case_id"]):
            p = out_path(root, "S", spec, "prefix", r["case_id"])
            if p.exists():
                continue
            if deadline and time.time() > deadline:
                raise Deadline()
            rec = CM.read_json(b0_path(root, "S", r["case_id"]))
            CM.write_json(p, MC.f_intervention_arm(e, r, rec, (int(site["layer"]), site["tensor"]), vec, alpha, tc)
                          if rec.get("p_ans_ok") else {"case_id": r["case_id"], "skipped": "NM"})
        gr = load_graded(root, "S", spec, "prefix", fa_una)
        v = endpoint_values(gr, labels, meta, ep)
        fint_eff[alpha] = S.bw_mean({k: v[k] - base_ep[k] for k in v if k in base_ep}, list(v))
        fint_deg[alpha] = S.bw_mean(MK.per_base(gr, labels, meta, "ALL", "Deg"), [meta[c]["base_key"] for c in gr])
    fint_dose = MK.m_dose(fint_eff, fint_deg, sg, {a: abs(a) for a in fint_eff})
    b1_eff = {}
    for val in CM.B1_SIGNED_GRID:
        spec = {"arm": "B1", "dose": float(val), "b1_ids": b1_ids, "tag": "signed"}
        run_arm(e, fa_una, root, "S", spec, "prefix", tc, deadline)
        v = endpoint_values(load_graded(root, "S", spec, "prefix", fa_una), labels, meta, ep)
        b1_eff[val] = S.bw_mean({k: v[k] - base_ep[k] for k in v if k in base_ep}, list(v))
    matched = MK.closest_b1(fint_eff[fint_dose], {k: v for k, v in b1_eff.items() if v is not None}) \
        if fint_dose is not None else None
    # predictor (M3c), fitted on S
    preds = []
    ppath = Path(root) / "f" / "S_pred.json"
    feats = CM.read_json(ppath, {})
    for r in rows_S:
        if r["set"] != "ANS":
            continue
        lab = labels[r["case_id"]]
        rec = CM.read_json(b0_path(root, "S", r["case_id"]))
        if "NM" in lab or not ({"FA", "C"} & set(lab)) or fvals.get(r["case_id"]) is None:
            continue
        if r["case_id"] not in feats:
            feats[r["case_id"]] = {"m_city": MC.m_city(e, r, rec, tc), "unk": MC.lexical_unknown_logodds(e, rec, b1_ids)}
        preds.append({"task": r["task"], "hardness": r["hardness"], "city": r["base_gold"],
                      "label": 1 if "FA" in lab else 0, "m_city": feats[r["case_id"]]["m_city"],
                      "unk_logodds": feats[r["case_id"]]["unk"], "f": fvals[r["case_id"]],
                      "base_key": r["base_key"]})
    CM.write_json(ppath, feats)
    enc = MK.cell_encoding(preds) if preds else {}
    coef = {}
    if len({x["label"] for x in preds}) < 2:                 # §13.4c: both classes must be present overall
        coef = {"unestimable": "one class (or no rows) in the S prediction population"}
    for name, wf in (() if "unestimable" in coef else (("baseline", False), ("full", True))):
        tr = MK.predictor_rows(preds, enc, wf)
        X, cols = MK.design_matrix(tr)
        fit = MK.fit_logistic(X, np.array([x["label"] for x in tr]))
        coef[name] = {"cols": cols, "coef": fit.coef_.tolist(), "intercept": fit.intercept_.tolist()}
    return {"submission": s["index"], "edit_id": s["edit_id"], "dose": m["dose"], "endpoint": ep, "sign": sg,
            "revert_oriented_effect": revert, "primary_component": {"i": primary, **{k: man["components"][primary][k]
                                                                                  for k in ("layer", "tensor")}},
            "b6_edit_ids": b6, "f": {"site": site, "direction_sha256": ED.sha_array(np.asarray(vec, np.float32)),
                                     "sd_ans_S": sd_f}, "activity": acts, "f_int_doses": fint_doses,
            "f_int_effects": fint_eff, "f_int_dose": fint_dose, "b1_signed_effects": b1_eff, "b1_matched": matched,
            "hook_positions": "P_ans only (§13.3)", "pair": pair, "predictor": {"encoding": enc, "coef": coef, "n_train": len(preds)},
            "audit_seed": 5101}


# ---------------------------------------------------------------------------------------------- replay check

def replay(e, root, subs, laptop_run: Path, tc, n=200) -> dict:
    """§10 on the pod: 200 E rows (sorted) x B0 and submission 1 at dose 1, batched and batch 1, vs the laptop."""
    rows = sorted(CM.rows("E"), key=lambda r: r["case_id"])[:n]
    s1 = subs[0]
    out = {"n": len(rows), "classes": {}}
    for label, edit in (("B0", None), ("sub1", (s1["edit_id"], 1.0))):
        lap = []
        for r in rows:
            if edit is None:
                lap.append(CM.b0("E", r["case_id"], laptop_run))
            else:
                lap.append(CM.read_json(Path(laptop_run) / "arms" / "E" / f"EDIT_{s1['edit_id']}_d1" / "total" /
                                        f"{r['case_id']}.json"))
        ids = [list(x["prompt_ids"]) if x else e.chat_ids(r["prompt"]) for r, x in zip(rows, lap)]
        gb = []
        for i in range(0, len(ids), LAYOUT):
            gb += e.generate_batched(ids[i:i + LAYOUT], mode="total", edit=edit)
        g1 = [e.generate(x, mode="total", edit=edit) for x in ids]
        for name, gs in (("batched", gb), ("batch1", g1)):
            cls = [CM.grade(r, g.text, tc)["cls"] for r, g in zip(rows, gs)]
            agree_c = sum(c == (x or {}).get("cls") for c, x in zip(cls, lap))
            agree_t = sum(g.gen_ids == (x or {}).get("gen_ids") for g, x in zip(gs, lap))
            out["classes"][f"{label}_{name}"] = {"class_agree": agree_c, "token_agree": agree_t,
                                                 "mismatch": [r["case_id"] for r, c, x in zip(rows, cls, lap)
                                                              if c != (x or {}).get("cls")]}
    out["pass"] = all(v["class_agree"] >= REPLAY_MIN * len(rows) for v in out["classes"].values())
    return out


# ---------------------------------------------------------------------------------------------- confirmatory (H)

def frozen(root) -> dict:
    from memctl.analysis import abst49 as A
    return A.require_design(Path(root))              # git-verified design.json = the committed F2


def h_run(e, root, subs, tc, deadline=None) -> None:
    f2 = frozen(root)
    rows_H = CM.rows("H", allow_sealed=True)
    b1_ids = CM.b1_token_set(e.tok)
    run_b0(e, rows_H, root, "H", tc, deadline)
    for g, c in f2["R"]["comparator_doses"].items():
        spec = {**c["spec"], **({"b1_ids": b1_ids} if c["spec"]["arm"] == "B1" else {})}
        run_arm(e, rows_H, root, "H", spec, "total", tc, deadline)
        if spec["arm"] == "B1":
            run_arm(e, rows_H, root, "H", {**spec, "variant": "whole"}, "total", tc, deadline)
    if f2["R"]["candidate"]:
        rc = f2["R"]["candidate"]
        run_arm(e, rows_H, root, "H", {"arm": "EDIT", "edit_id": rc["edit_id"], "dose": rc["dose"],
                                       "tag": f"sub{rc['index']}"}, "total", tc, deadline)
    mf = f2.get("M_followups") or {}
    if not mf or mf.get("none"):
        return
    import exp51_mech as MC
    s = next(x for x in subs if x["index"] == mf["submission"])
    au = [r for r in rows_H if r["set"] in ("ANS", "UNA-rel", "UNA-swap")]
    run_arm(e, au, root, "H", {"arm": "B0"}, "prefix", tc, deadline)
    run_arm(e, au, root, "H", {"arm": "EDIT", "edit_id": mf["edit_id"], "dose": mf["dose"],
                               "tag": f"sub{mf['submission']}"}, "prefix", tc, deadline)
    for seed, eid in mf["b6_edit_ids"].items():
        run_arm(e, au, root, "H", {"arm": "EDIT", "edit_id": eid, "dose": 1.0, "tag": f"B6s{seed}"}, "prefix", tc,
                deadline)
    pc = mf["primary_component"]
    man, comps = sub_components(root, s)
    site, vec = s["f_site"], s["f_direction_vec"]
    for r in sorted(au, key=lambda r: r["case_id"]):
        rec = CM.read_json(b0_path(root, "H", r["case_id"]))
        for name, fn in (("REVERT", lambda: MC.revert_arm(e, r, rec, mf["edit_id"], mf["dose"], pc["i"], len(comps), tc)),
                         ("TRANSPLANT", lambda: MC.transplant_arm(e, r, rec, mf["edit_id"], mf["dose"], pc["layer"],
                                                                  pc["tensor"], [rec["p_ans"]], tc)),
                         ("CLAMP", lambda: MC.clamp_arm(e, r, rec, mf["edit_id"], mf["dose"], pc["layer"],
                                                        pc["tensor"], [rec["p_ans"]], tc)),
                         ("FINT", lambda: MC.f_intervention_arm(e, r, rec, (int(site["layer"]), site["tensor"]), vec,
                                                                mf["f_int_dose"], tc))):
            spec = {"arm": name, "tag": f"sub{mf['submission']}"}
            p = out_path(root, "H", spec, "prefix", r["case_id"])
            if p.exists():
                continue
            if deadline and time.time() > deadline:
                raise Deadline()
            CM.write_json(p, fn() if rec.get("p_ans_ok") else {"case_id": r["case_id"], "skipped": "NM"})
    labels_H, _ = labels_meta(root, "H", rows_H)
    pair_rows = [r for r in rows_H if any(p in labels_H[r["case_id"]] for p in mf["pair"])]
    for r in sorted(pair_rows, key=lambda r: r["case_id"]):
        if r in au:
            continue
        rec = CM.read_json(b0_path(root, "H", r["case_id"]))
        p = out_path(root, "H", {"arm": "FINT", "tag": f"sub{mf['submission']}"}, "prefix", r["case_id"])
        if not p.exists():
            CM.write_json(p, MC.f_intervention_arm(e, r, rec, (int(site["layer"]), site["tensor"]), vec,
                                                   mf["f_int_dose"], tc) if rec.get("p_ans_ok") else
                          {"case_id": r["case_id"], "skipped": "NM"})
    if mf.get("b1_matched") is not None:
        run_arm(e, pair_rows, root, "H", {"arm": "B1", "dose": float(mf["b1_matched"]), "b1_ids": b1_ids,
                                          "tag": "matched"}, "prefix", tc, deadline)
    fpath = Path(root) / "f" / "H.json"
    fvals = CM.read_json(fpath, {})
    for r in sorted(rows_H, key=lambda r: r["case_id"]):
        if r["case_id"] not in fvals:
            if deadline and time.time() > deadline:
                CM.write_json(fpath, fvals)
                raise Deadline()
            fvals[r["case_id"]] = f_of(e, CM.read_json(b0_path(root, "H", r["case_id"])), site, vec)
    CM.write_json(fpath, fvals)
    ppath = Path(root) / "f" / "H_pred.json"
    feats = CM.read_json(ppath, {})
    for r in rows_H:
        rec = CM.read_json(b0_path(root, "H", r["case_id"]))
        if r["set"] == "ANS" and rec.get("p_ans_ok") and r["case_id"] not in feats:
            feats[r["case_id"]] = {"m_city": MC.m_city(e, r, rec, tc), "unk": MC.lexical_unknown_logodds(e, rec, b1_ids)}
    CM.write_json(ppath, feats)
    # T-city (secondary; R arms only)
    rows_T = CM.rows("Tcity", allow_sealed=True)
    run_b0(e, rows_T, root, "Tcity", tc, deadline)
    for g, c in f2["R"]["comparator_doses"].items():
        spec = {**c["spec"], **({"b1_ids": b1_ids} if c["spec"]["arm"] == "B1" else {})}
        run_arm(e, rows_T, root, "Tcity", spec, "total", tc, deadline)
    if f2["R"]["candidate"]:
        rc = f2["R"]["candidate"]
        run_arm(e, rows_T, root, "Tcity", {"arm": "EDIT", "edit_id": rc["edit_id"], "dose": rc["dose"],
                                           "tag": f"sub{rc['index']}"}, "total", tc, deadline)


# ---------------------------------------------------------------------------------------------- H report

def h_report(root, subs, rows_S, rows_H, rows_T, B=S.B_DEFAULT) -> dict:
    """The confirmatory read of H (and the separate T-city report) from the frozen F2: family R (repair, O1 / O2
    per comparator, Holm over 10, the B1 equivalence) and family M (M1, M2, M3 as intersection-unions, Holm over the
    three). Read once; no rescue (D-45)."""
    f2 = frozen(root)
    labels, meta = labels_meta(root, "H", rows_H)
    bases = MK.bases_of(meta)
    b0 = total_graded(root, "H", {"arm": "B0"}, rows_H)
    out = {"R": None, "M": None, "Tcity": None}
    comps = {}
    for g, c in f2["R"]["comparator_doses"].items():
        comps[g] = MK.rates(total_graded(root, "H", c["spec"], rows_H), labels, meta)
    rc = f2["R"]["candidate"]
    if rc:
        cand = MK.rates(total_graded(root, "H", {"arm": "EDIT", "edit_id": rc["edit_id"], "dose": rc["dose"],
                                                 "tag": f"sub{rc['index']}"}, rows_H), labels, meta)
        out["R"] = MK.family_r(cand, comps, MK.rates(b0, labels, meta), bases, B)
        out["R"]["rates_summary"] = {k: {r: S.bw_mean(v, list(v)) for r, v in x.items()}
                                     for k, x in {"candidate": cand, "B0": MK.rates(b0, labels, meta), **comps}.items()}
        lt, mt = labels_meta(root, "Tcity", rows_T)
        tb0 = MK.rates(total_graded(root, "Tcity", {"arm": "B0"}, rows_T), lt, mt)
        tc_ = MK.rates(total_graded(root, "Tcity", {"arm": "EDIT", "edit_id": rc["edit_id"], "dose": rc["dose"],
                                                    "tag": f"sub{rc['index']}"}, rows_T), lt, mt)
        out["Tcity"] = {"descriptive": True, "candidate": {k: S.bw_mean(v, list(v)) for k, v in tc_.items()},
                        "B0": {k: S.bw_mean(v, list(v)) for k, v in tb0.items()}}
    mf = f2.get("M_followups") or {}
    if mf and not mf.get("none"):
        s = next(x for x in subs if x["index"] == mf["submission"])
        ep, sg = mf["endpoint"], mf["sign"]
        tag = f"sub{mf['submission']}"

        def ev(spec, rows=rows_H):
            return endpoint_values(load_graded(root, "H", spec, "prefix", rows), labels, meta, ep)
        star = ev({"arm": "EDIT", "edit_id": mf["edit_id"], "dose": mf["dose"], "tag": tag})
        b0p = ev({"arm": "B0"})
        rev = ev({"arm": "REVERT", "tag": tag})
        b6 = MK.seed_mean([ev({"arm": "EDIT", "edit_id": eid, "dose": 1.0, "tag": f"B6s{sd}"})
                           for sd, eid in mf["b6_edit_ids"].items()])
        m1 = MK.m1(bases, star, rev, b6, sg, B)
        m2 = MK.m2(bases, ev({"arm": "TRANSPLANT", "tag": tag}), b0p, star, ev({"arm": "CLAMP", "tag": tag}), sg, B)
        fH = CM.read_json(Path(root) / "f" / "H.json", {})
        pr = pairs_for(rows_H)
        acts = []
        for a in mf["activity"]:
            df = delta_f(fH, pr[a["manipulation"]])
            if a["type"] == "directional":
                acts.append({**a, **MK.activity_directional(bases, df, int(a["sign"]) or 1, B)})
            else:
                acts.append({**a, **MK.activity_equivalence(bases, df, a["sd_S"], B)})
        # (b) the Commit interaction: (f-int − matched B1) on FA minus the same on UNA
        pair = mf["pair"]
        commit_ep_fa = {"population": pair[0], "outcome": "Commit"}
        commit_ep_una = {"population": pair[1], "outcome": "Commit"}
        gf = load_graded(root, "H", {"arm": "FINT", "tag": tag}, "prefix", rows_H)
        gb = load_graded(root, "H", {"arm": "B1", "dose": float(mf["b1_matched"]), "tag": "matched"}, "prefix",
                         rows_H) if mf.get("b1_matched") is not None else {}
        inter = MK.interaction(bases, endpoint_values(gf, labels, meta, commit_ep_fa),
                               endpoint_values(gb, labels, meta, commit_ep_fa),
                               endpoint_values(gf, labels, meta, commit_ep_una),
                               endpoint_values(gb, labels, meta, commit_ep_una), sg, B) if gb else {"p": None}
        # (c) predictor: fitted on S (the frozen features), scored on H
        def pred_rows(split, rows, fvals, feats):
            lab, _ = labels_meta(root, split, rows)
            outp = []
            for r in rows:
                l = lab[r["case_id"]]
                if r["set"] != "ANS" or "NM" in l or not ({"FA", "C"} & set(l)) or fvals.get(r["case_id"]) is None \
                        or r["case_id"] not in feats:
                    continue
                outp.append({"task": r["task"], "hardness": r["hardness"], "city": r["base_gold"],
                             "label": 1 if "FA" in l else 0, "m_city": feats[r["case_id"]]["m_city"],
                             "unk_logodds": feats[r["case_id"]]["unk"], "f": fvals[r["case_id"]],
                             "base_key": r["base_key"]})
            return outp
        tr = pred_rows("S", rows_S, CM.read_json(Path(root) / "f" / "S.json", {}),
                       CM.read_json(Path(root) / "f" / "S_pred.json", {}))
        te = pred_rows("H", rows_H, fH, CM.read_json(Path(root) / "f" / "H_pred.json", {}))
        m3c = MK.m3c(tr, te, bases, B) if tr and te else {"p": 1.0, "unestimable": "no rows"}
        m3_ps = [x["p"] for x in acts] + [inter.get("p"), m3c.get("p")]
        m3 = {"activity": acts, "interaction": inter, "predictor": {k: v for k, v in m3c.items() if k != "coef"},
              "p": S.iut(m3_ps)}
        holm = S.holm({"M1": m1["p"], "M2": m2["p"], "M3": m3["p"]})
        out["M"] = {"M1": m1, "M2": m2, "M3": m3, "holm": holm,
                    "pathway_located": holm["M1"]["reject"], "causal_path": holm["M2"]["reject"],
                    "account_supported": all(holm[k]["reject"] for k in ("M1", "M2", "M3"))}
    return out


# ---------------------------------------------------------------------------------------------- main

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", required=True, choices=("replay", "s-arms", "s-select", "h-run", "h-report"))
    p.add_argument("--root", required=True)
    p.add_argument("--subs", required=True)
    p.add_argument("--laptop-run", default=str(CM.RUN))
    p.add_argument("--backend", default="hf", choices=("hf", "dummy"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--snapshot", default=None)
    p.add_argument("--deadline-s", type=float, default=None)
    p.add_argument("--boot", type=int, default=S.B_DEFAULT)
    p.add_argument("--limit-bases", type=int, default=None, help="D-22 smoke: the first k bases per task only")
    a = p.parse_args(argv)
    if a.limit_bases:
        real_rows = CM.rows

        def limited(split, path=CM.CASES, allow_sealed=False):
            rs = real_rows(split, path, allow_sealed)
            keep = {f"{split}-{t}{k:02d}" for t in ("DE", "RC", "TU") for k in range(a.limit_bases)}
            return [r for r in rs if r["base_key"] in keep]
        CM.rows = limited
    root = Path(a.root)
    root.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + a.deadline_s if a.deadline_s else None
    tc = CM.t_cities()
    subs = load_subs(Path(a.subs))
    e = EN.load(a.backend, 8, edit_dir=root / "edits", device=a.device, snapshot=a.snapshot)
    try:
        if a.stage == "replay":
            res = replay(e, root, subs, Path(a.laptop_run), tc)
            CM.write_json(root / "replay_check.json", res)
            print(json.dumps({"pass": res["pass"], **{k: {kk: vv for kk, vv in v.items() if kk != "mismatch"}
                                                      for k, v in res["classes"].items()}}))
            return 0 if res["pass"] else EXIT_REPLAY
        if a.stage == "s-arms":
            rows_S = CM.rows("S")
            b1_ids = CM.b1_token_set(e.tok)
            b5 = CM.read_json(root / "b5_fit.json")["edit_ids"]
            run_b0(e, rows_S, root, "S", tc, deadline)
            for spec, est in r_specs(e, subs, b5, b1_ids):
                run_arm(e, rows_S, root, "S", spec, est, tc, deadline)
            au = [r for r in rows_S if r["set"] in ("ANS", "UNA-rel", "UNA-swap")]
            for spec, est in m_specs(subs):
                run_arm(e, au, root, "S", spec, est, tc, deadline)
            return 0
        if a.stage == "s-select":
            rows_S = CM.rows("S")
            b5 = CM.read_json(root / "b5_fit.json")["edit_ids"]
            sel = s_select(e, root, rows_S, subs, b5, tc, a.boot)
            CM.write_json(root / "s_select.json", sel)
            if sel["R"]["candidate"] or sel["M"]["candidate"]:
                g = sel["R"]["comparator_doses"].get("B1")
                if g:                                           # the whole-generation variant at B1's R dose
                    run_arm(e, rows_S, root, "S", {**g["spec"], "variant": "whole", "b1_ids": CM.b1_token_set(e.tok)},
                            "total", tc, deadline)
                sel["M_followups"] = m_followups(e, root, rows_S, subs, sel, tc, a.boot, deadline)
            else:
                sel["stop"] = "§8.4.3: neither entry rule holds on S: no H read and no pod"
            text = json.dumps(sel, indent=1, sort_keys=True, default=float)
            (root / "f2_candidate.json").write_text(text)
            print(json.dumps({"f2_candidate_sha256": hashlib.sha256(text.encode()).hexdigest(),
                              "R_candidate": bool(sel["R"]["candidate"]), "M_candidate": bool(sel["M"]["candidate"])}))
            return 0
        if a.stage == "h-report":
            res = h_report(root, subs, CM.rows("S"), CM.rows("H", allow_sealed=True),
                           CM.rows("Tcity", allow_sealed=True), a.boot)
            CM.write_json(root / "h_report.json", json.loads(json.dumps(res, default=float)))
            print("h_report.json written")              # no outcome printed
            return 0
        if a.stage == "h-run":
            try:
                frozen(root)
            except PermissionError as ex:
                print(f"GUARD: {ex}")
                return EXIT_GUARD
            h_run(e, root, subs, tc, deadline)
            return 0
    except Deadline:
        print("deadline: stopped between rows (resumable)")
        return EXIT_DEADLINE
    return 0


if __name__ == "__main__":
    sys.exit(main())
