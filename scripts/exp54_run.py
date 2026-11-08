#!/usr/bin/env python
"""§54 runner: twin pairs and component patching for false "unknown" (Qwen2.5-3B, laptop, $0).

    HF_HUB_OFFLINE=1 .venv/bin/python scripts/exp54_run.py --stage b0        # B0 on the 900 W ANS rows
    HF_HUB_OFFLINE=1 .venv/bin/python scripts/exp54_run.py --stage pairs     # flip pairs, forced note, D, gate, split

Spec: research/proposals/2026-11-07-exp54-native-switch.md rev 0; every rule comes from the stage-0 freeze
runs/exp54/freeze.json (66fcafe). Stages 2-4 are added after the stage-1 gate.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import exp51_common as CM  # noqa: E402
import exp51_stage4 as S4  # noqa: E402

OUT = ROOT / "runs/exp54"
RUN = OUT / "run"
FREEZE = json.loads((OUT / "freeze.json").read_text())
SPLIT = FREEZE["data"]["split_name"]
UNK = FREEZE["definitions"]["unknown_token"]["id"]
CASES = OUT / "cases54.jsonl"
PAIRS = OUT / "pairs.json"


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def all_rows() -> list[dict]:
    rows = [json.loads(l) for l in CASES.read_text().splitlines() if l.strip()]
    assert all(r["split"] == SPLIT and r["seed"] == FREEZE["seeds"]["dev"] for r in rows)
    return rows


def b0(cid: str) -> dict | None:
    return CM.read_json(S4.b0_path(RUN, SPLIT, cid))


# ---------------------------------------------------------------------------------------------- stage 1

def stage_b0(e, limit: int | None = None) -> None:
    tc = CM.t_cities()
    rows = sorted([r for r in all_rows() if r["set"] == "ANS"], key=lambda r: r["case_id"])
    if limit:
        rows = rows[:limit]
    t0 = time.time()
    S4.run_b0(e, rows, RUN, SPLIT, tc)
    log(f"B0 on {len(rows)} ANS rows done in {time.time() - t0:.0f} s; peak RSS {S4.EN.peak_rss_gb():.1f} GB")


def d_at(e, ids: list[int], t_star: int) -> float:
    p = len(ids) - 1
    lg = e.read(ids, positions=[p], logit_positions=[p], logit_ids=[UNK, t_star])["logits"][0]
    return float(lg[0] - lg[1])


def flip_pairs(rows: list[dict]) -> tuple[list[dict], Counter]:
    by = {r["case_id"]: r for r in rows}
    out, why, stable = [], Counter(), []
    for r in rows:
        if r["set"] != "ANS" or r["rep"] != "a":
            continue
        a_id, b_id = r["case_id"], r["case_id"].replace("_ANSa_", "_ANSb_")
        ra, rb = b0(a_id), b0(b_id)
        if ra is None or rb is None:
            why["missing_b0"] += 1
            continue
        cls = {ra["cls"], rb["cls"]}
        if cls == {"gold"}:
            stable.append({"base_key": r["base_key"], "task": r["task"], "hardness": r["hardness"], "a": a_id,
                           "b": b_id})
            why["both_gold"] += 1
            continue
        if cls != {"gold", "abstention"}:
            why["no_flip"] += 1
            continue
        A, F = (a_id, b_id) if ra["cls"] == "gold" else (b_id, a_id)
        out.append({"base_key": r["base_key"], "task": r["task"], "hardness": r["hardness"], "A": A, "F": F,
                    "A_rep": A.split("_ANS")[1][0]})
    return out, why, stable


def stable_pairs(e, cands: list[dict], by: dict, tc) -> list[dict]:
    """freeze definitions.stable_pair: both gold under own B0, P_ans ok, equal prompt length; b gets rep a's note ids
    and must still grade gold (a's controlled prefix is its own B0 prefix)."""
    res = []
    for k, p in enumerate(cands):
        ra, rb = b0(p["a"]), b0(p["b"])
        q = dict(p)
        if not (ra["p_ans_ok"] and rb["p_ans_ok"]) or len(ra["prompt_ids"]) != len(rb["prompt_ids"]):
            q["stable"], q["why"] = False, "no P_ans or prompt lengths differ"
            res.append(q)
            continue
        b_ids = list(rb["prompt_ids"]) + list(ra["gen_ids"][:ra["j"]])
        g = e.generate(b_ids, mode="prefix")
        q["forced_b_text"], q["forced_b_cls"] = g.text, CM.grade(by[p["b"]], g.text, tc)["cls"]
        q["stable"] = q["forced_b_cls"] == "gold"
        res.append(q)
        if (k + 1) % 20 == 0:
            log(f"stable candidate {k + 1}/{len(cands)}")
    return res


def stage_pairs(e) -> dict:
    tc = CM.t_cities()
    rows = all_rows()
    by = {r["case_id"]: r for r in rows}
    flips, why, stable_c = flip_pairs(rows)
    log(f"{len(flips)} flip pairs; {dict(why)}")
    res = []
    for k, p in enumerate(flips):
        rA, rF = b0(p["A"]), b0(p["F"])
        q = dict(p)
        if not (rA["p_ans_ok"] and rF["p_ans_ok"]):
            q["kind"], q["why"] = "other", "no P_ans"
            res.append(q)
            continue
        if len(rA["prompt_ids"]) != len(rF["prompt_ids"]):
            q["kind"], q["why"] = "other", "prompt lengths differ"
            res.append(q)
            continue
        t_star = rA["gen_ids"][rA["j"]] if len(rA["gen_ids"]) > rA["j"] else None
        q["t_star"] = t_star
        q["t_star_text"] = e.decode([t_star]) if t_star is not None else None
        a_ids = CM.prefix_ids(rA)
        f_ids = list(rF["prompt_ids"]) + list(rA["gen_ids"][:rA["j"]])
        g = e.generate(f_ids, mode="prefix")
        gr = CM.grade(by[p["F"]], g.text, tc)
        q["forced_F_text"], q["forced_F_cls"] = g.text, gr["cls"]
        q["n_prefix"] = len(f_ids)
        q["D_A"], q["D_F"] = d_at(e, a_ids, t_star), d_at(e, f_ids, t_star)
        q["D_F_own"] = d_at(e, CM.prefix_ids(rF), t_star)
        q["secondary_ok"] = q["D_A"] < 0 < q["D_F_own"]
        if gr["cls"] == "gold":
            q["kind"] = "note_driven"
        elif gr["cls"] == "abstention" and q["D_A"] < 0 < q["D_F"]:
            q["kind"] = "primary"
        else:
            q["kind"] = "other"
            q["why"] = f"forced F {gr['cls']}; D_A {q['D_A']:.2f}, D_F {q['D_F']:.2f}"
        res.append(q)
        log(f"pair {k + 1}/{len(flips)} {p['A']} {q['kind']}  D_A {q['D_A']:.2f}  D_F {q['D_F']:.2f}")
    prim = [q for q in res if q["kind"] == "primary"]
    split = assign_halves(res, rows)
    for q in res:
        q["half"] = split[q["base_key"]]
    stab = stable_pairs(e, stable_c, by, tc)
    for q in stab:
        q["half"] = split[q["base_key"]]
    gate = {"n_primary": len(prim), "n_bases": len({q["base_key"] for q in prim}),
            "n_tasks": len({q["task"] for q in prim}),
            "primary_by_task": dict(Counter(q["task"] for q in prim)),
            "primary_by_hardness": dict(Counter(q["hardness"] for q in prim)),
            "primary_by_half": dict(Counter(q["half"] for q in prim)),
            "bases_by_half": dict(Counter(split[b] for b in {q["base_key"] for q in prim}))}
    gate["pass"] = gate["n_primary"] >= 40 and gate["n_bases"] >= 20 and gate["n_tasks"] >= 2
    n_fa = sum(1 for r in rows if r["set"] == "ANS" and (b0(r["case_id"]) or {}).get("cls") == "abstention")
    summ = {"n_twin_candidates": sum(1 for r in rows if r["set"] == "ANS" and r["rep"] == "a"),
            "no_flip": dict(why), "n_flip": len(flips), "kinds": dict(Counter(q["kind"] for q in res)),
            "n_secondary": sum(1 for q in res if q.get("secondary_ok")),
            "n_fa_rows": n_fa, "share_fa_in_flip_pairs": len(flips) / n_fa if n_fa else None,
            "n_stable_by_half": dict(Counter(q["half"] for q in stab if q["stable"])),
            "n_stable_candidates": len(stab),
            "kinds_by_half": {h: dict(Counter(q["kind"] for q in res if q["half"] == h)) for h in ("dev-A", "dev-B")},
            "secondary_by_half": dict(Counter(q["half"] for q in res if q.get("secondary_ok"))),
            "gate": gate, "pairs": res, "stable": stab, "split": split}
    CM.write_json(PAIRS, summ)
    log(f"gate {gate}")
    return summ


def assign_halves(pairs: list[dict], rows: list[dict]) -> dict[str, str]:
    """freeze split_rule: primary-pair bases, per task sorted and shuffled, concatenated DE, RC, TU, alternating
    dev-A / dev-B; the other bases by the same procedure separately."""
    prim_bases = {q["base_key"] for q in pairs if q["kind"] == "primary"}
    all_bases = {(r["task"], r["base_key"]) for r in rows}
    out = {}
    for group in (prim_bases, {b for _, b in all_bases} - prim_bases):
        seq = []
        for task in ("DE", "RC", "TU"):
            bs = sorted(b for t, b in all_bases if t == task and b in group)
            random.Random(f"54-split-{task}").shuffle(bs)
            seq += bs
        for i, b in enumerate(seq):
            out[b] = "dev-A" if i % 2 == 0 else "dev-B"
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("b0", "pairs"))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args(argv)
    import exp51_engine as EN
    e = EN.load("hf", threads=a.threads)
    if a.stage == "b0":
        stage_b0(e, a.limit)
    else:
        stage_pairs(e)
    return 0


if __name__ == "__main__":
    sys.exit(main())
