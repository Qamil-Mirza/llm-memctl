#!/usr/bin/env python
"""§45 stage reports from the map outputs: stage 1 readability on solved prompts (Wilson intervals, depth quartiles,
fit stability between the two J-lenses), stage 2 matched failures, stage 3 the §44 rev 8 rows (descriptive).

    .venv/bin/python scripts/exp45_report.py --root runs/exp45_maps --half dev        # the dev half first (§3)
    .venv/bin/python scripts/exp45_report.py --root runs/exp45_maps --half all [--swap-dir runs/exp44_swap/qwen3b_rev8/S]

Criteria (proposal §4-§5, rev 1 / review 2d7122f): readable = the entity's " X" id in the lens top-10 at a counted
position (the last prompt token and the answer span: F1 every generated token, F2 and stage 3 the tokens after
"Answer:"), or its bare "X" id at the last prompt token (which predicts the first generated token), at any layer; (a)
uses layers before the final layer only. Beside: "last prompt token only", "any generated token", "either form
anywhere". (a) gold on every task type; (b) the
intermediate on RC (v0, v1, v3; in v2 the supervisor is the gold); (c) the competitor on TU. A criterion "recovers"
when the share is >= 0.70 with the Wilson lower bound >= 0.55. Lens stability: the two J-lenses disagree on >= 0.20 of
the (prompt, pinned entity) calls (before the final layer) -> "the 56-prompt lens is not stable". Stage 2: the
pre-registered sentence per lens; fewer than 10 failures at H2 -> "too few failures on these tasks". The HELD half
decides every criterion (rev 1 ruling 4); dev is shown first (presentation only), pooled beside.
Writes <root>/<lens set>/report_<half>.md and .json. Never writes into --swap-dir.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from memctl.analysis import maps as M  # noqa: E402

CRITERIA = ("a_gold", "b_intermediate", "c_competitor")
H2_MIN_FAILURES = 10


def load_dir(d: Path, half: str = "all") -> list[tuple[dict, dict]]:
    out = []
    if not d.exists():
        return out
    for path in sorted(d.glob("*.json")):
        if path.name.startswith("_") or not path.with_suffix(".npz").exists():
            continue
        payload = json.loads(path.read_text())
        if half != "all" and payload.get("split") not in (None, half):
            continue
        out.append((payload, dict(np.load(path.with_suffix(".npz")))))
    return out


VARIANTS = ("fixed", "last prompt token only", "any generated token", "either form anywhere")


def _cols(pinned: list[dict], surfaces) -> tuple[list[int], list[int]]:
    sp, bare = [], []
    for surf in surfaces:
        a, b = M.form_columns(pinned, surf)
        sp += a
        bare += b
    return sp, bare


def entity_call(r: np.ndarray, pinned: list[dict], surfaces, layers, payload: dict, variant: str = "fixed",
                exclude_final: bool = False) -> dict:
    """One readability call under the rev 1 rule. fixed: space form at the counted positions (last prompt token + the
    answer span), bare form at the last prompt token (which predicts the first generated token); the beside variants
    change only the positions (or, for "either form anywhere", use both forms everywhere from the last prompt token)."""
    lp, P = payload["last_prompt_index"], r.shape[1]
    sp, bare = _cols(pinned, surfaces)
    if variant == "either form anywhere":
        return M.readability(r, sp + bare, layers, lp, exclude_final=exclude_final)
    counted = {"fixed": payload.get("counted_positions", list(range(lp, P))), "last prompt token only": [lp],
               "any generated token": list(range(lp, P))}[variant]
    return M.readability_rule(r, sp, bare, layers, counted, [lp], exclude_final=exclude_final)


def calls(payload: dict, arrays: dict) -> dict:
    """Per lens and variant: the three criterion calls (None when the criterion does not apply); per entity the fixed
    before-final call (for lens agreement)."""
    pinned, layers = payload["pinned"], [int(l) for l in arrays["layers"]]
    inter = [payload["intermediate"]] if payload.get("intermediate") else []
    comps = list(payload.get("competitors", []))
    surfaces = list(dict.fromkeys(p["surface"] for p in pinned))
    out = {}
    for li, name in enumerate(payload["lens_names"]):
        r = arrays["ranks"][li]
        out[name] = {"entities": {s: entity_call(r, pinned, [s], layers, payload, exclude_final=True)["readable"]
                                  for s in surfaces}}
        for v in VARIANTS:
            suffix = "" if v == "fixed" else f" | {v}"
            out[name]["a_gold" + suffix] = entity_call(r, pinned, [payload["gold"]], layers, payload, v, exclude_final=True)
            out[name]["b_intermediate" + suffix] = (entity_call(r, pinned, inter, layers, payload, v)
                                                    if (payload.get("task") == "RC" and inter) else None)
            out[name]["c_competitor" + suffix] = (entity_call(r, pinned, comps, layers, payload, v)
                                                  if (payload.get("task") == "TU" and comps) else None)
    return out


def share_row(flags: list[bool], depths: list) -> dict:
    n, s = len(flags), sum(bool(f) for f in flags)
    p, lo, hi = M.wilson(s, n)
    return {"n": n, "readable": s, "share": p, "lo": lo, "hi": hi, "verdict": M.criterion_verdict(s, n),
            "first_depth_quartiles": M.quartiles(depths)}


def criterion_table(items: list[tuple[dict, dict, dict]], lens_names: list[str]) -> dict:
    """{task: {lens: {criterion: share_row}}} over the given (payload, arrays, calls) items."""
    table = defaultdict(lambda: defaultdict(dict))
    for task in ("DE", "RC", "TU"):
        sub = [(p, c) for p, _, c in items if p.get("task") == task]
        for lens in lens_names:
            for crit in CRITERIA:
                for v in VARIANTS:
                    key = crit if v == "fixed" else f"{crit} | {v}"
                    vals = [c[lens][key] for _, c in sub if c[lens][key] is not None]
                    if vals:
                        table[task][lens][key] = share_row([x["readable"] for x in vals], [x["first_depth"] for x in vals])
    return json.loads(json.dumps(table))


def lens_agreement(items: list[tuple[dict, dict, dict]]) -> dict | None:
    """Readability-call agreement j56 vs jrefit per (prompt, entity), and the Spearman correlation of the two lenses'
    trajectories (log rank of the entity's best form, layers before the final, the counted positions), per (prompt,
    entity). The calls are the fixed rule's."""
    agree, total, rhos = 0, 0, []
    for p, a, c in items:
        if "j56" not in c or "jrefit" not in c:
            continue
        i56, ir = p["lens_names"].index("j56"), p["lens_names"].index("jrefit")
        counted = p.get("counted_positions", list(range(p["last_prompt_index"], a["ranks"].shape[2])))
        for s in c["j56"]["entities"]:
            x, y = c["j56"]["entities"][s], c["jrefit"]["entities"][s]
            total += 1
            agree += int(x == y)
            cols = M.entity_columns(p["pinned"], s)
            ra = np.log(a["ranks"][i56, :-1, counted, :][:, :, cols].min(-1))
            rb = np.log(a["ranks"][ir, :-1, counted, :][:, :, cols].min(-1))
            rho = M.spearman(ra, rb)
            if rho is not None:
                rhos.append(rho)
    if not total:
        return None
    disagree = 1 - agree / total
    return {"n_calls": total, "agreement": agree / total, "disagreement": disagree,
            "stable": disagree < 0.20, "verdict": "stable" if disagree < 0.20 else "the 56-prompt lens is not stable",
            "spearman_median": float(np.median(rhos)) if rhos else None,
            "spearman_quartiles": M.quartiles(rhos), "n_trajectories": len(rhos)}


def fixed_trajectory(r: np.ndarray, payload: dict) -> np.ndarray:
    """Per layer, the gold's best rank under the fixed rule (space form at the counted positions, bare form at the
    last prompt token)."""
    lp = payload["last_prompt_index"]
    sp, bare = _cols(payload["pinned"], [payload["gold"]])
    counted = payload.get("counted_positions", list(range(lp, r.shape[1])))
    parts = []
    if sp:
        parts.append(r[:, counted, :][:, :, sp].min(axis=(1, 2)))
    if bare:
        parts.append(r[:, [lp], :][:, :, bare].min(axis=(1, 2)))
    return np.minimum.reduce(parts)


def stage2(all_items: list[tuple[dict, dict, dict]], lens_names: list[str]) -> dict:
    solved = {}
    for p, a, c in all_items:
        if p["outcome"] == "gold":
            key = (p["task"], p["base_index"], p["variant"], p["format"])
            if key not in solved or p["hardness"] < solved[key][0]["hardness"]:
                solved[key] = (p, a, c)
    failures = [(p, a, c) for p, a, c in all_items if p["outcome"] != "gold"]
    n_h2 = sum(1 for p, _, _ in failures if p["hardness"] == "H2")
    pairs = []
    for p, a, c in failures:
        pair = solved.get((p["task"], p["base_index"], p["variant"], p["format"]))
        row = {"key": p["key"], "task": p["task"], "hardness": p["hardness"], "outcome": p["outcome"],
               "ambiguous": p.get("ambiguous"), "answer_marker": p.get("answer_marker"),
               "generated": p["generated_text"][:80], "pair": pair[0]["key"] if pair else None, "departure_layer": {}}
        if pair:
            layers = [int(l) for l in a["layers"]]
            for lens in lens_names:
                fi, si = p["lens_names"].index(lens), pair[0]["lens_names"].index(lens)
                f_t = fixed_trajectory(a["ranks"][fi], p)
                s_t = fixed_trajectory(pair[1]["ranks"][si], pair[0])
                row["departure_layer"][lens] = M.departure_layer(f_t, s_t, layers)
                row.setdefault("gold_traj", {})[lens] = {"failure": f_t.tolist(), "pair": s_t.tolist()}
        pairs.append(row)
    sentences = {}
    for lens in lens_names:
        flags = [c[lens]["a_gold"]["readable"] for _, _, c in failures]
        p_, lo, hi = M.wilson(sum(flags), len(flags))
        sentences[lens] = {"n": len(flags), "share": p_, "lo": lo, "hi": hi,
                           "sentence": None if not flags else
                           f"on the matched failures the gold is {'' if p_ >= 0.70 else 'not '}readable before the final "
                           f"layer on >= 0.70 of them ({lens})"}
    return {"n_failures": len(failures), "n_failures_h2": n_h2,
            "too_few": n_h2 < H2_MIN_FAILURES,
            "headline": "too few failures on these tasks" if n_h2 < H2_MIN_FAILURES else None,
            "by_task_hardness_outcome": {"|".join(k): v for k, v in sorted(Counter(
                (p["task"], p["hardness"], p["outcome"]) for p, _, _ in failures).items())},
            "n_paired": sum(1 for r in pairs if r["pair"]), "sentences": sentences,
            "table": criterion_table(failures, lens_names), "pairs": pairs}


def stage3(items: list[tuple[dict, dict]], swap_dir: Path | None) -> list[dict]:
    rows = []
    for p, a in items:
        s_row = None
        if swap_dir is not None and (swap_dir / f"{p['question_id']}.json").exists():
            s_row = json.loads((swap_dir / f"{p['question_id']}.json").read_text())
        layers, fp = [int(l) for l in a["layers"]], p["last_prompt_index"]
        entry = {"key": p["key"], "kind": p["kind"], "label23": p["label23"], "outcome": p["outcome"],
                 "answer_marker": p.get("answer_marker"), "answer_span_found": p.get("answer_span_found"),
                 "generated": p["generated_text"][:60], "cls_1nat": (s_row or {}).get("cls", {}).get("1"),
                 "d_T_fwd": (s_row or {}).get("d_T_fwd"), "lenses": {}}
        for li, lens in enumerate(p["lens_names"]):
            r = a["ranks"][li]
            entry["lenses"][lens] = {}
            for role in ("gold", "substitute", "unknown"):
                surf = M.role_surfaces(p["pinned"], role)
                if not surf:
                    continue
                lpt = entity_call(r, p["pinned"], surf, layers, p, "last prompt token only")
                gen = entity_call(r, p["pinned"], surf, layers, p, "fixed")
                entry["lenses"][lens][role] = {"best_rank_last_prompt": lpt["best_rank"],
                                               "first_top10_layer_last_prompt": lpt["first_layer"],
                                               "readable_fixed": gen["readable"], "first_layer_fixed": gen["first_layer"]}
        rows.append(entry)
    return rows


def _f(x, nd=2):
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def markdown(rep: dict) -> str:
    L = [f"# §45 report ({rep['lens_set']}, half: {rep['half']})", "",
         "Descriptive. Readable (fixed, rev 1) = the entity's space-form id in the lens top-10 at a counted position "
         "(the last prompt token and the answer span: F1 every generated token, F2 / stage 3 the tokens after "
         "'Answer:'), or its bare-form id at the last prompt token; criterion (a) before the final layer. Rows marked "
         "'| ...' are descriptive beside. A miss is 'not recovered by this lens', never 'not computed'. **The held half "
         "decides**; dev is shown first for presentation, pooled beside.", "",
         "Caveat: in F2 the memory lines carry dates that increase with line order, so in TU the timestamp order is "
         "a second update cue beside the wording (a format effect, which F2 is there to show).", ""]
    s1 = rep["stage1"]
    L += [f"## Stage 1: solved prompts ({s1['n_solved']} of {s1['n_prompts']})", "",
          "Outcomes: " + json.dumps(s1["outcomes"]),
          "Answer marker present (format|present): " + json.dumps(s1["answer_marker_by_format"]), ""]
    label = {"dev": "dev (presentation only)", "held": "held (decides)", "all": "pooled (beside)"}
    for scope in ("dev", "held", "all"):
        if scope not in s1["tables"]:
            continue
        L += [f"### {label[scope]}", "", "| task | lens | criterion | n | readable | share | Wilson 95% | verdict | first depth q1/q2/q3 |",
              "|---|---|---|---|---|---|---|---|---|"]
        for task, by_lens in s1["tables"][scope].items():
            for lens, by_c in by_lens.items():
                for crit, row in by_c.items():
                    q = row["first_depth_quartiles"]
                    L.append(f"| {task} | {lens} | {crit} | {row['n']} | {row['readable']} | {_f(row['share'])} | "
                             f"[{_f(row['lo'])}, {_f(row['hi'])}] | {row['verdict']} | {'/'.join(_f(v) for v in q) if q else '-'} |")
        L.append("")
    L += ["### Fit stability (j56 vs jrefit, before the final layer)", ""]
    for label, ag in (("solved prompts", s1.get("lens_agreement")), ("all prompts (beside)", s1.get("lens_agreement_all_prompts"))):
        L.append(f"- {label}: " + ("no prompts read with both J-lenses" if not ag else
                 f"agreement {_f(ag['agreement'], 3)} over {ag['n_calls']} calls -> **{ag['verdict']}**; Spearman median "
                 f"{_f(ag['spearman_median'], 3)} (quartiles {ag['spearman_quartiles']}, n={ag['n_trajectories']})"))
    L.append("")
    s2 = rep["stage2"]
    L += [f"## Stage 2: matched failures ({s2['n_failures']}, {s2['n_failures_h2']} at H2, {s2['n_paired']} paired)", ""]
    if s2["headline"]:
        L += [f"**{s2['headline']}** (fewer than {H2_MIN_FAILURES} failures at H2; nothing is added to induce more).", ""]
    for lens, s in s2["sentences"].items():
        L.append(f"- {lens}: {s['sentence'] or 'no failures'} (share {_f(s['share'])}, Wilson [{_f(s['lo'])}, {_f(s['hi'])}], n={s['n']})")
    L += ["", "Failures by task, hardness and outcome: " + json.dumps(s2["by_task_hardness_outcome"]), "",
          "| failure | outcome | ambiguous | marker | pair | departure layer per lens | generated |", "|---|---|---|---|---|---|---|"]
    for r in s2["pairs"]:
        L.append(f"| {r['key']} | {r['outcome']} | {r['ambiguous']} | {r['answer_marker']} | {r['pair'] or '-'} | "
                 f"{json.dumps(r['departure_layer'])} | {r['generated'].replace('|', '/').replace(chr(10), ' ')} |")
    L += ["", f"## Stage 3: the §44 rev 8 rows ({len(rep['stage3'])} prompts)", "",
          "| key | §23 | §44 class (1 nat) | d_T fwd | outcome | marker | lens: gold / substitute / unknown best rank at the last prompt token | generated |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rep["stage3"]:
        lens_txt = "; ".join(f"{lens} " + "/".join(_f(v.get(role, {}).get("best_rank_last_prompt")) for role in ("gold", "substitute", "unknown"))
                             for lens, v in r["lenses"].items())
        L.append(f"| {r['key']} | {r['label23']} | {r['cls_1nat'] or '-'} | {_f(r['d_T_fwd'])} | {r['outcome']} | {r['answer_marker']} | {lens_txt} | "
                 f"{r['generated'].replace('|', '/').replace(chr(10), ' ')} |")
    return "\n".join(L) + "\n"


def find_lens_set(root: Path, name: str | None) -> Path:
    if name:
        return root / name
    for cand in ("j56-jrefit-logit", "j56-logit"):
        if (root / cand / "stage1").exists() or (root / cand / "stage3").exists():
            return root / cand
    raise SystemExit(f"exp45_report: no lens-set directory under {root}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="runs/exp45_maps")
    p.add_argument("--lens-set", default=None)
    p.add_argument("--half", choices=("dev", "held", "all"), default="dev")
    p.add_argument("--swap-dir", default="runs/exp44_swap/qwen3b_rev8/S", help="read-only: the §44 rev 8 S rows")
    args = p.parse_args(argv)
    d = find_lens_set(Path(args.root), args.lens_set)
    items1 = [(pl, a, calls(pl, a)) for pl, a in load_dir(d / "stage1", args.half)]
    lens_names = items1[0][0]["lens_names"] if items1 else []
    solved = [x for x in items1 if x[0]["outcome"] == "gold"]
    tables = {}
    for scope in (("dev", "held", "all") if args.half == "all" else (args.half,)):
        sub = solved if scope == "all" else [x for x in solved if x[0].get("split") == scope]
        tables[scope] = criterion_table(sub, lens_names)
    swap_dir = Path(args.swap_dir) if args.swap_dir and Path(args.swap_dir).exists() else None
    items3 = load_dir(d / "stage3") if args.half == "all" else []
    rep = {"lens_set": d.name, "half": args.half,
           "stage1": {"n_prompts": len(items1), "n_solved": len(solved),
                      "outcomes": dict(Counter(x[0]["outcome"] for x in items1)),
                      "outcomes_by_task_format": {"|".join(k): v for k, v in sorted(Counter(
                          (x[0]["task"], x[0]["format"], x[0]["outcome"]) for x in items1).items())},
                      "answer_marker_by_format": {"|".join(map(str, k)): v for k, v in sorted(Counter(
                          (x[0]["format"], bool(x[0].get("answer_marker"))) for x in items1).items())},
                      "tables": tables, "lens_agreement": lens_agreement(solved),
                      "lens_agreement_all_prompts": lens_agreement(items1)},
           "stage2": stage2(items1, lens_names), "stage3": stage3(items3, swap_dir),
           "swap_dir": str(swap_dir) if swap_dir else None}
    (d / f"report_{args.half}.json").write_text(json.dumps(rep, indent=1))
    (d / f"report_{args.half}.md").write_text(markdown(rep))
    print(f"report: {len(items1)} stage 1 prompts, {len(items3)} stage 3 -> {d / f'report_{args.half}.md'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
