"""§39 report (development only): Steer-0 against the reused §38 Steer-all, the dose curve, cut shares, Damage-0.

    .venv/bin/python scripts/exp39_report.py --run-dir runs/exp39_dev/qwen3b [--out runs/exp39_report_dev.md]

Pre-registration: research/proposals/2026-10-23-exp39-one-position.md (rev 1, 25d257f; amendment 1, 7db734d).
Everything is recomputed from the rows of <run-dir> ({repro,arms,steer_all}.jsonl; the Steer-all rows are the §38
dev rows copied by the run) and the §38 dev repro rows (<source>/repro.jsonl, from manifest.json). The report
refuses rows of folds other than 0-1, more than one engine (rows + manifest) and more than one wording.

Reading (§3, arithmetic exact on counts, amendment 1.3), on the reproduced R rows where Steer-0, Steer-all and
Random-0 all exist: a0 = ans(Steer-0), aA = ans(Steer-all), g0, gA = D1 gold (PROVISIONAL), r0 = ans(Random-0);
M2 if a0 < 0.5 aA; M1 if a0 >= 0.8 aA and g0 >= 0.8 gA and r0 <= 0.2 a0 ("the pushed ':' state carries the answer
path"; the ':' K/V above layer 29 stay pushed for every later token, amendment 1.1); split if a0 >= 0.8 aA and
g0 < 0.5 gA; else "undetermined on n rows". Every verdict line carries the counts and the count thresholds.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from memctl.analysis import steering as st  # noqa: E402
from scripts.exp35_generate import read_jsonl  # noqa: E402
from scripts.exp39_dev import (ALPHA, BYTE_GATE, DEV, RANDOM_NOTE, UNIT, Arms,  # noqa: E402
                               compare_repro)

STAGES = ("repro", "arms", "steer_all")
M1_SHARE, M2_SHARE, SPLIT_SHARE = (4, 5), (1, 2), (1, 2)     # 0.8, 0.5, 0.5 as exact fractions
M1_WORDING = ("the pushed ':' state carries the answer path (the ':' K/V above layer 29 stay pushed for every later "
              "token)")
M2_WORDING = "the refusal re-asserts at the generated tokens; the push has to be maintained"
SPLIT_WORDING = "the first token is released at ':', the content needs the push during generation; read with the dose curve"


class Refused(SystemExit):
    pass


# --- the §3 reading -------------------------------------------------------------------------------------------

def thresholds(aA: int, gA: int, a0: int) -> dict:
    """Count thresholds on the same n: M1 needs a0 >= ceil(0.8 aA), g0 >= ceil(0.8 gA), r0 <= floor(0.2 a0);
    M2 needs a0 <= ceil(0.5 aA) - 1; the split needs g0 <= ceil(0.5 gA) - 1."""
    return {"m1_a0_min": math.ceil(4 * aA / 5), "m1_g0_min": math.ceil(4 * gA / 5), "m1_r0_max": a0 // 5,
            "m2_a0_max": math.ceil(aA / 2) - 1, "split_g0_max": math.ceil(gA / 2) - 1}


def reading(n: int, a0: int, aA: int, g0: int, gA: int, r0: int) -> dict:
    """§3 on counts over the same n rows (exact integer arithmetic; no float boundary error)."""
    a_m1 = 5 * a0 >= 4 * aA
    g_m1 = 5 * g0 >= 4 * gA
    r_ok = 5 * r0 <= a0
    m2 = 2 * a0 < aA
    g_split = 2 * g0 < gA
    if m2:
        verdict, words = "M2 supported", M2_WORDING
    elif a_m1 and g_m1 and r_ok:
        verdict, words = "M1 supported", M1_WORDING
    elif a_m1 and g_split:
        verdict, words = "split", SPLIT_WORDING
    else:
        verdict, words = f"undetermined on {n} rows", None
    why = []
    if not m2 and verdict.startswith("undetermined"):
        if not a_m1:
            why.append("0.5 aA <= a0 < 0.8 aA")
        else:
            if not g_m1 and not g_split:
                why.append("0.5 gA <= g0 < 0.8 gA")
            if g_m1 and not r_ok:
                why.append("ans(Random-0) > 0.2 a0")
    return {"n": n, "a0": a0, "aA": aA, "g0": g0, "gA": gA, "r0": r0, "a0_ge_08aA": a_m1, "g0_ge_08gA": g_m1,
            "r0_le_02a0": r_ok, "a0_lt_05aA": m2, "g0_lt_05gA": g_split, "verdict": verdict, "meaning": words,
            "why_undetermined": why, "thresholds": thresholds(aA, gA, a0)}


def reading_line(r: dict) -> str:
    t = r["thresholds"]
    n = r["n"]
    return (f"{r['verdict']} (development only, gold PROVISIONAL): a0 = {r['a0']}/{n}, aA = {r['aA']}/{n}, "
            f"g0 = {r['g0']}/{n}, gA = {r['gA']}/{n}, ans(Random-0) = {r['r0']}/{n}. "
            f"M1 needs a0 >= {t['m1_a0_min']}/{n}, g0 >= {t['m1_g0_min']}/{n}, Random-0 <= {t['m1_r0_max']}/{n}; "
            f"M2 needs a0 <= {t['m2_a0_max']}/{n}; split needs a0 >= {t['m1_a0_min']}/{n} and g0 <= {t['split_g0_max']}/{n}."
            + (f" Meaning: {r['meaning']}." if r["meaning"] else "")
            + (f" Between: {'; '.join(r['why_undetermined'])}." if r["why_undetermined"] else ""))


# --- rows -----------------------------------------------------------------------------------------------------

def read_rows(run_dir: Path) -> dict:
    rows = {}
    for stage in STAGES:
        for r in read_jsonl(run_dir / f"{stage}.jsonl"):
            rows[(stage,) + st.row_key(r)] = r
    return rows


def check_rows(rows: dict, run_dir: Path) -> dict:
    mpath = run_dir / "manifest.json"
    if not mpath.exists():
        raise Refused(f"REFUSED: {mpath} missing")
    manifest = json.loads(mpath.read_text())
    folds = {int(r["fold"]) for r in rows.values()} | set(manifest.get("folds", []))
    if not folds <= set(DEV):
        raise Refused(f"REFUSED: rows of folds {sorted(folds - set(DEV))}; §39 is development only (folds 0-1)")
    engines = sorted({r["engine"] for r in rows.values()} | {manifest["engine"]})
    if len(engines) > 1:
        raise Refused(f"REFUSED: the run directory mixes engines {engines} (one engine per run directory)")
    shas = sorted({r["wording_sha256"] for r in rows.values()} | {manifest["wording_sha256"]})
    if len(shas) > 1:
        raise Refused(f"REFUSED: the run directory mixes wordings {shas}")
    return manifest


def pick(rows: dict, stage: str, pop: str, kind: str, key: str, qs) -> dict[str, dict]:
    return {q: rows[(stage, pop, kind, key, q)] for q in qs if (stage, pop, kind, key, q) in rows}


def block(sel_rows: dict[str, dict], qs) -> dict:
    rs = [sel_rows[q] for q in qs if q in sel_rows]
    n = len(rs)
    k = lambda f: sum(1 for r in rs if f(r))  # noqa: E731
    out = {"n": n, "ans_k": k(lambda r: r["cls"] == "answer"), "degenerate_k": k(lambda r: r["cls"] == "degenerate"),
           "refusal_k": k(lambda r: r["cls"] == "refusal"), "gold_k": k(lambda r: bool(r["gold_d1_provisional"]))}
    for name in ("ans", "degenerate", "refusal", "gold"):
        out[name] = out[f"{name}_k"] / n if n else None
    return out


def share(num: int, den: int) -> float | None:
    return num / den if den else None


def analyse(run_dir: Path, d, pop, source: Path | None = None) -> dict:
    rows = read_rows(run_dir)
    manifest = check_rows(rows, run_dir)
    source = Path(source or manifest["source"])
    src = {st.row_key(r): r for r in read_jsonl(source / "repro.jsonl")}
    repro_rows = {k[1:]: r for k, r in rows.items() if k[0] == "repro"}
    rep = compare_repro(repro_rows, src, d, pop, source)
    R = rep["R_ok"]
    pairs = [q for q in pop.pairs if q in R]
    unseen = [q for q in pop.unseen if q in R]
    arms = Arms(manifest["layer"], manifest["ks"])
    key_all = arms.steer_all.key
    table = {
        "A (unsteered)": pick(rows, "repro", "R", "A", "none", R),
        "B (unsteered)": pick(rows, "repro", "R", "B", "none", R),
        "Steer-all (§38, reused)": pick(rows, "steer_all", "R", "A", key_all, R),
        "Steer-0": pick(rows, "arms", "R", "A", arms.steer0.key, R),
        **{f"Steer-0..{k}": pick(rows, "arms", "R", "A", s.key, R) for k, s in arms.steer_k.items()},
        "Random-0": pick(rows, "arms", "R", "A", arms.random0.key, R),
        "A-cut (unsteered)": pick(rows, "arms", "R", "cut", "none", R),
        "Steer-0-cut": pick(rows, "arms", "R", "cut", arms.steer0.key, R),
        "Steer-all-cut": pick(rows, "arms", "R", "cut", key_all, R),
    }
    n: dict = {"run_dir": str(run_dir), "source": str(source), "engine": manifest["engine"],
               "layer": manifest["layer"], "alpha": ALPHA, "unit": UNIT, "ks": manifest["ks"],
               "npz_sha256": manifest.get("npz_sha256"), "v_sha256": manifest.get("v_sha256"),
               "vector_sha256": manifest.get("vector_sha256"), "random_note": RANDOM_NOTE,
               "random_29_sha256": manifest.get("random_29_sha256"),
               "random_added_sha256": manifest.get("random_added_sha256"),
               "n_R": len(R), "n_pairs": len(pairs), "n_unseen": len(unseen), "repro": rep}
    n["arms"] = {name: {"R": block(v, R), "pairs": block(v, pairs), "unseen": block(v, unseen),
                        "missing": len(R) - len(v)} for name, v in table.items()}
    # the reading, on rows where all three arms exist
    s0, sa, r0 = table["Steer-0"], table["Steer-all (§38, reused)"], table["Random-0"]
    common = [q for q in R if q in s0 and q in sa and q in r0]
    b0, bA, bR = block(s0, common), block(sa, common), block(r0, common)
    n["reading"] = reading(len(common), b0["ans_k"], bA["ans_k"], b0["gold_k"], bA["gold_k"], bR["ans_k"])
    # dose curve
    dose = [("steps 0", table["Steer-0"])] + [(f"steps 0-{k}", table[f"Steer-0..{k}"]) for k in arms.ks] + \
           [("all steps", table["Steer-all (§38, reused)"])]
    n["dose"] = [{"steps": label, **block(v, R)} for label, v in dose]
    # cut shares (plausibility share = gold(cut arm) / gold(intact arm), counts on the same rows)
    cuts = {}
    for intact, cut in (("Steer-0", "Steer-0-cut"), ("Steer-all (§38, reused)", "Steer-all-cut"),
                        ("A (unsteered)", "A-cut (unsteered)")):
        qs = [q for q in R if q in table[intact] and q in table[cut]]
        bi, bc = block(table[intact], qs), block(table[cut], qs)
        cuts[cut] = {"n": len(qs), "gold_intact_k": bi["gold_k"], "gold_cut_k": bc["gold_k"],
                     "ans_intact_k": bi["ans_k"], "ans_cut_k": bc["ans_k"],
                     "plausibility_share": share(bc["gold_k"], bi["gold_k"]), "ans_share": share(bc["ans_k"], bi["ans_k"])}
    n["cut"] = cuts
    n["cut_no_marker"] = sum(r["cut_no_marker"] for r in table["Steer-0-cut"].values())
    # Damage-0
    dmg_a = pick(rows, "repro", "C", "A", "none", pop.damage)
    dmg_0 = pick(rows, "arms", "C", "A", arms.steer0.key, pop.damage)
    n["damage"] = {sub: {"A": block(dmg_a, qs), "Damage-0": block(dmg_0, qs), "qids": qs}
                   for sub, qs in (("reproduced", rep["damage_ok"]), ("all", list(pop.damage)))}
    # descriptive (amendment 1.6): Steer-0 against Steer-all per row
    per_row = {}
    for q in common:
        per_row[q] = {"first_token_same": s0[q]["first_token"] == sa[q]["first_token"],
                      "text_same": s0[q]["generated"] == sa[q]["generated"],
                      "steer0": s0[q]["phrase"], "steer_all": sa[q]["phrase"],
                      "steer0_cls": s0[q]["cls"], "steer_all_cls": sa[q]["cls"]}
    n["identity"] = {"n": len(common), "first_token_same_k": sum(v["first_token_same"] for v in per_row.values()),
                     "text_same_k": sum(v["text_same"] for v in per_row.values()), "rows": per_row}
    # same-run repeats (descriptive): arms 'none' A vs repro A; arms 'none' cut vs §38's controls cut 'none'
    src_ctl = {st.row_key(r): r for r in read_jsonl(source / "controls.jsonl")}
    same = lambda x, y: x["generated"] == y["generated"] and x["gen_ids"] == y["gen_ids"]  # noqa: E731
    rep_a = [same(table_r, repro_rows[("R", "A", "none", q)]) for q, table_r in
             pick(rows, "arms", "R", "A", "none", R).items()]
    cut38 = [same(r, src_ctl[("R", "cut", "none", q)]) for q, r in table["A-cut (unsteered)"].items()
             if ("R", "cut", "none", q) in src_ctl]
    n["repeats"] = {"A_none_vs_repro": {"k": sum(rep_a), "n": len(rep_a)},
                    "cut_none_vs_38_controls": {"k": sum(cut38), "n": len(cut38)}}
    return n


# --- rendering ------------------------------------------------------------------------------------------------

def _c(b: dict, name: str) -> str:
    return "-" if not b["n"] else f"{b[name + '_k']}/{b['n']} ({b[name]:.3f})"


def _s(x) -> str:
    return "-" if x is None else f"{x:.3f}"


def render(n: dict) -> str:
    rep = n["repro"]
    L = ["# §39 report: the §38 layer-29 direction at the ':' position only (DEVELOPMENT ONLY, folds 0-1)", "",
         f"Run {n['run_dir']} (engine {n['engine']}); source {n['source']}. Layer {n['layer']}, a{n['alpha']:g} "
         f"{n['unit']}, Steer-k k in {n['ks']}. Gold: D1 string rule, PROVISIONAL; ans is the primary quantity.",
         f"directions.npz sha256 {n['npz_sha256']}; v_29 sha256 {n['v_sha256']}; vector added sha256 {n['vector_sha256']}.",
         f"Random-0: {n['random_note']}; sha256 {n['random_29_sha256']} (vector added in this run: "
         f"{n['random_added_sha256']}).", "",
         "## Reproduction",
         f"Gate (flag rule against the §38 dev rows, bound {st.REPRO_GATE}): {rep['rate']:.3f} -> "
         f"{'pass' if rep['pass'] else 'FAIL'} (R {rep['rate_R']:.3f}; dropped {rep['dropped']}). vs §35 cache "
         f"{_s(rep['rate_vs35'])}.",
         f"Byte-identity with the §38 dev rows: {rep['byte_k']}/{rep['byte_n']} ({_s(rep['byte_rate'])}; stop below "
         f"{BYTE_GATE}) -> {'pass' if rep['byte_pass'] else 'FAIL: engine is not the one the Steer-all rows came from'}.",
         f"Flags match, bytes differ (kept, marked): {rep['flags_ok_bytes_differ'] or 'none'}.",
         "Rows whose bytes differ: " + (", ".join(k for k, v in rep["byte_identity"].items() if not v) or "none") + ".",
         f"Same-run repeats (descriptive): unsteered A in the arms stage = repro A on "
         f"{n['repeats']['A_none_vs_repro']['k']}/{n['repeats']['A_none_vs_repro']['n']}; unsteered cut = §38 A-cut on "
         f"{n['repeats']['cut_none_vs_38_controls']['k']}/{n['repeats']['cut_none_vs_38_controls']['n']}.", ""]
    if not (rep["pass"] and rep["byte_pass"]):
        L += ["**Reproduction or byte-identity failed: no steered output is read.**", ""]
        return "\n".join(L)
    L += ["## Reading (§3, pre-registered)", "", reading_line(n["reading"]), "",
          f"## Arms on the {n['n_R']} reproduced R rows (pairs {n['n_pairs']}, unseen {n['n_unseen']})", "",
          "| arm | n | ans | degenerate | refusal | D1 gold (PROVISIONAL) | ans pairs | gold pairs | ans unseen | gold unseen |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for name, v in n["arms"].items():
        a, p, u = v["R"], v["pairs"], v["unseen"]
        L.append(f"| {name} | {a['n']} | {_c(a, 'ans')} | {_c(a, 'degenerate')} | {_c(a, 'refusal')} | {_c(a, 'gold')} | "
                 f"{_c(p, 'ans')} | {_c(p, 'gold')} | {_c(u, 'ans')} | {_c(u, 'gold')} |")
    L += ["", "## Dose curve (descriptive)", "", "| steps steered | ans | D1 gold (PROVISIONAL) | refusal | degenerate |",
          "|---|---|---|---|---|"]
    for e in n["dose"]:
        L.append(f"| {e['steps']} | {_c(e, 'ans')} | {_c(e, 'gold')} | {_c(e, 'refusal')} | {_c(e, 'degenerate')} |")
    L += ["", "## Cut shares (descriptive; plausibility share = gold(cut) / gold(intact), same rows)", ""]
    for name, c in n["cut"].items():
        L.append(f"- {name}: gold {c['gold_cut_k']}/{c['n']} vs intact {c['gold_intact_k']}/{c['n']} -> plausibility "
                 f"share {_s(c['plausibility_share'])}; ans {c['ans_cut_k']} vs {c['ans_intact_k']} (share "
                 f"{_s(c['ans_share'])})")
    L.append(f"- cut prompts without a marker (no note): {n['cut_no_marker']}")
    L += ["", "## Damage-0 (descriptive)", ""]
    for sub, v in n["damage"].items():
        L.append(f"- {sub} damage rows ({len(v['qids'])}): A ans {_c(v['A'], 'ans')}, gold {_c(v['A'], 'gold')}; "
                 f"Damage-0 ans {_c(v['Damage-0'], 'ans')}, gold {_c(v['Damage-0'], 'gold')}, refusal "
                 f"{_c(v['Damage-0'], 'refusal')}")
    idn = n["identity"]
    L += ["", "## Steer-0 against Steer-all per row (descriptive, amendment 1.6)", "",
          f"First token the same on {idn['first_token_same_k']}/{idn['n']}; full text identical on "
          f"{idn['text_same_k']}/{idn['n']}.", "", "| question | first token same | text same | Steer-0 | Steer-all |",
          "|---|---|---|---|---|"]
    for q, v in idn["rows"].items():
        L.append(f"| {q} | {'yes' if v['first_token_same'] else 'no'} | {'yes' if v['text_same'] else 'no'} | "
                 f"{v['steer0']!r} ({v['steer0_cls']}) | {v['steer_all']!r} ({v['steer_all_cls']}) |")
    L += ["", "Development only: no claim on folds 2-4."]
    return "\n".join(L)


def build(run_dir: Path, exp35: Path = Path("runs/exp35"), runs: str = "runs",
          own_root: Path = Path("runs/exp34_jlens_own"), source: Path | None = None) -> tuple[str, dict]:
    rows = read_rows(run_dir)
    check_rows(rows, run_dir)
    from scripts.exp38_dev import load_source
    d, pop = load_source(list(DEV), exp35, runs, own_root)
    numbers = analyse(run_dir, d, pop, source)
    return render(numbers), numbers


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="runs/exp39_dev/qwen3b")
    parser.add_argument("--source", default=None, help="the §38 dev run directory (default: from manifest.json)")
    parser.add_argument("--exp35", default="runs/exp35")
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--own-root", default="runs/exp34_jlens_own")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    try:
        text, numbers = build(Path(args.run_dir), Path(args.exp35), args.runs, Path(args.own_root),
                              Path(args.source) if args.source else None)
    except Refused as exc:
        print(exc.code if isinstance(exc.code, str) else exc)
        return 2
    print(text)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.with_suffix(".md").write_text(text)
        out.with_suffix(".json").write_text(json.dumps(numbers, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
