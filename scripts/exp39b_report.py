"""§39b report (development only): the first-token-only clean-cache control against §39's Steer-0.

    .venv/bin/python scripts/exp39b_report.py --run-dir runs/exp39b_dev/qwen3b [--out runs/exp39b_report_dev.md]

Pre-registration: research/proposals/2026-10-23-exp39-one-position.md, amendment 2 (3e7c349). Everything is
recomputed from the rows of <run-dir> ({repro,unknown_first,step0,forced,steer0}.jsonl; steer0 = the §39 Steer-0 and
Steer-0-cut rows copied verbatim by the run) and the §38 dev repro rows (<source>/repro.jsonl, from manifest.json).
Refuses rows of folds other than 0-1, more than one engine (rows + manifest) and more than one wording.

Reading (amendment 2, exact counts): a0, g0 = ans and D1 gold (PROVISIONAL) of §39 Steer-0 on its 28 rows, read from
the rows and asserted equal to the pre-registered 26 and 7 (on the stub the assertion is skipped and said so);
aF, gF = the same for Steer-first. Checked in this order: state needed if 2 aF < a0; token carries it if
5 aF >= 4 a0 and 5 gF >= 4 g0; split if 5 aF >= 4 a0 and 2 gF < g0; else undetermined on n rows. The reading uses
counts on the rows where both arms exist; when that is all 28, they are a0 and g0 themselves.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from memctl.analysis import steering as st  # noqa: E402
from scripts.exp35_generate import read_jsonl  # noqa: E402
from scripts.exp39_dev import BYTE_GATE, DEV, Arms, compare_repro  # noqa: E402
from scripts.exp39b_dev import ARM_GOLD, ARM_STEER, ARM_STEER_CUT, ARM_UNKNOWN  # noqa: E402

STAGES = ("repro", "unknown_first", "step0", "forced", "steer0")
EXPECTED_A0, EXPECTED_G0, EXPECTED_N = 26, 7, 28       # amendment 2, from §39 (asserted against the rows)
SOFT_REFUSAL_PATTERN = st.SOFT_REFUSAL_PATTERN   # frozen in §41 rev 1 (877f86d), the §39 review's pattern
SOFT_REFUSAL_RE = re.compile(SOFT_REFUSAL_PATTERN, re.IGNORECASE)
WORDS = {"token carries it": "the first generated token alone keeps the model answering and right; the pushed ':' "
                             "state is not needed",
         "state needed": "without the pushed ':' state the forced token does not keep the model answering",
         "split": "the token keeps the model answering, the pushed state keeps it right"}


class Refused(SystemExit):
    pass


# --- the reading ----------------------------------------------------------------------------------------------

def thresholds(a0: int, g0: int) -> dict:
    """Count thresholds: token carries needs aF >= ceil(4 a0 / 5), gF >= ceil(4 g0 / 5); state needed needs
    aF <= ceil(a0 / 2) - 1; split needs aF >= ceil(4 a0 / 5) and gF <= ceil(g0 / 2) - 1."""
    return {"carries_aF_min": math.ceil(4 * a0 / 5), "carries_gF_min": math.ceil(4 * g0 / 5),
            "state_aF_max": math.ceil(a0 / 2) - 1, "split_gF_max": math.ceil(g0 / 2) - 1}


def reading(n: int, aF: int, gF: int, a0: int, g0: int) -> dict:
    """Amendment 2 on counts over the same n rows; order: state needed, token carries it, split, undetermined."""
    a_ok = 5 * aF >= 4 * a0
    g_ok = 5 * gF >= 4 * g0
    state = 2 * aF < a0
    g_split = 2 * gF < g0
    if state:
        verdict = "state needed"
    elif a_ok and g_ok:
        verdict = "token carries it"
    elif a_ok and g_split:
        verdict = "split"
    else:
        verdict = f"undetermined on {n} rows"
    why = []
    if verdict.startswith("undetermined"):
        why.append("0.5 a0 <= aF < 0.8 a0" if not a_ok else "0.5 g0 <= gF < 0.8 g0")
    return {"n": n, "aF": aF, "gF": gF, "a0": a0, "g0": g0, "aF_ge_08a0": a_ok, "gF_ge_08g0": g_ok,
            "aF_lt_05a0": state, "gF_lt_05g0": g_split, "verdict": verdict, "meaning": WORDS.get(verdict),
            "why_undetermined": why, "thresholds": thresholds(a0, g0)}


def reading_line(r: dict) -> str:
    t, n = r["thresholds"], r["n"]
    return (f"{r['verdict']} (development only, gold PROVISIONAL): aF = {r['aF']}/{n}, gF = {r['gF']}/{n} against "
            f"a0 = {r['a0']}/{n}, g0 = {r['g0']}/{n}. State needed needs aF <= {t['state_aF_max']}/{n} (2 aF < a0); "
            f"token carries it needs aF >= {t['carries_aF_min']}/{n} and gF >= {t['carries_gF_min']}/{n}; split needs "
            f"aF >= {t['carries_aF_min']}/{n} and gF <= {t['split_gF_max']}/{n}."
            + (f" Meaning: {r['meaning']}." if r["meaning"] else "")
            + (f" Between: {'; '.join(r['why_undetermined'])}." if r["why_undetermined"] else ""))


def soft_refusal(row: dict) -> bool:
    """An answer-class row whose phrase hedges (the §39 review's pattern, frozen in §41 rev 1: `SOFT_REFUSAL_PATTERN`)."""
    return row["cls"] == "answer" and bool(SOFT_REFUSAL_RE.search(row["phrase"] or ""))


def divergence(a: list[int], b: list[int]) -> int | None:
    """First generated-token index where the two id lists differ (a prefix differs at the shorter length); None if
    they are equal."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


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
        raise Refused(f"REFUSED: rows of folds {sorted(folds - set(DEV))}; §39b is development only (folds 0-1)")
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
           "refusal_k": k(lambda r: r["cls"] == "refusal"), "gold_k": k(lambda r: bool(r["gold_d1_provisional"])),
           "soft_k": k(soft_refusal)}
    for name in ("ans", "degenerate", "refusal", "gold"):
        out[name] = out[f"{name}_k"] / n if n else None
    return out


def same_bytes(x: dict, y: dict) -> bool:
    return x["generated"] == y["generated"] and x["gen_ids"] == y["gen_ids"]


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
    s0key = Arms(manifest["layer"]).steer0.key
    steer0_all = pick(rows, "steer0", "R", "A", s0key, pop.R)
    table = {
        "A (unsteered)": pick(rows, "repro", "R", "A", "none", R),
        "Steer-0 (§39)": pick(rows, "steer0", "R", "A", s0key, R),
        "Steer-first": pick(rows, "forced", "R", "A", ARM_STEER.key, R),
        "Steer-0-cut (§39)": pick(rows, "steer0", "R", "cut", s0key, R),
        "Steer-first-cut": pick(rows, "forced", "R", "cut", ARM_STEER_CUT.key, R),
        "Gold-first": pick(rows, "forced", "R", "A", ARM_GOLD.key, R),
        "Unknown-first": pick(rows, "unknown_first", "R", "A", ARM_UNKNOWN.key, R),
    }
    n: dict = {"run_dir": str(run_dir), "source": str(source), "source39": manifest.get("source39"),
               "source39_arms_sha256": manifest.get("source39_arms_sha256"), "engine": manifest["engine"],
               "layer": manifest["layer"], "vector_sha256": manifest.get("vector_sha256"),
               "forcing": manifest.get("forcing"), "tokenisation": manifest.get("tokenisation"),
               "n_R": len(R), "n_pairs": len(pairs), "n_unseen": len(unseen), "repro": rep}
    # checks (ii) and (iii)
    unk = table["Unknown-first"]
    a_rows = table["A (unsteered)"]
    n["unknown_check"] = {"n": len(unk), "k": sum(same_bytes(unk[q], a_rows[q]) for q in unk if q in a_rows),
                          "natural_first_is_forced_k": sum(r["natural_first_token"] == r["forced_token"]
                                                           for r in unk.values())}
    s0chk = pick(rows, "step0", "R", "A", s0key, R)
    n["step0_check"] = {"n": len(s0chk), "k": sum(s0chk[q]["first_token"] == steer0_all[q]["gen_ids"][0]
                                                  for q in s0chk if q in steer0_all)}
    n["arms"] = {name: {"R": block(v, R), "pairs": block(v, pairs), "unseen": block(v, unseen),
                        "missing": len(R) - len(v)} for name, v in table.items()}
    # a0, g0 from the §39 rows (all of §39's R rows), asserted against the pre-registration
    b39 = block(steer0_all, pop.R)
    n["a0_g0_from_39"] = {"n": b39["n"], "a0": b39["ans_k"], "g0": b39["gold_k"]}
    if manifest["engine"] == "stub":
        n["a0_g0_from_39"]["assertion"] = "skipped on the stub engine"
    else:
        assert (b39["n"], b39["ans_k"], b39["gold_k"]) == (EXPECTED_N, EXPECTED_A0, EXPECTED_G0), \
            f"§39 Steer-0 rows give n {b39['n']}, a0 {b39['ans_k']}, g0 {b39['gold_k']}; amendment 2 says 28, 26, 7"
        n["a0_g0_from_39"]["assertion"] = f"equal to the pre-registered a0 = {EXPECTED_A0}, g0 = {EXPECTED_G0}"
    sf, s0 = table["Steer-first"], table["Steer-0 (§39)"]
    common = [q for q in R if q in sf and q in s0]
    bF, b0 = block(sf, common), block(s0, common)
    n["reading"] = reading(len(common), bF["ans_k"], bF["gold_k"], b0["ans_k"], b0["gold_k"])
    # per row (descriptive)
    gold_first, cut_first, cut0 = table["Gold-first"], table["Steer-first-cut"], table["Steer-0-cut (§39)"]
    per_row = {}
    for q in common:
        f, z = sf[q], s0[q]
        g = gold_first.get(q)
        per_row[q] = {"forced": f["forced_text"], "forced_id": f["forced_token"], "natural_first": f["natural_first_token"],
                      "steer_first": f["phrase"], "steer_first_cls": f["cls"], "steer_first_gold": f["gold_d1_provisional"],
                      "steer0": z["phrase"], "steer0_cls": z["cls"], "steer0_gold": z["gold_d1_provisional"],
                      "same_text": same_bytes(f, z), "divergence": divergence(f["gen_ids"], z["gen_ids"]),
                      "gold_first_forced": g["forced_text"] if g else None, "gold_first": g["phrase"] if g else None,
                      "gold_first_cls": g["cls"] if g else None, "gold_first_gold": g["gold_d1_provisional"] if g else None,
                      "gold_first_tok_agree": (g["forced_source"].get("agree") if g else None),
                      "unknown_first_same_as_A": same_bytes(unk[q], a_rows[q]) if q in unk and q in a_rows else None}
    divs = [v["divergence"] for v in per_row.values() if v["divergence"] is not None]
    n["identity"] = {"n": len(common), "same_text_k": sum(v["same_text"] for v in per_row.values()),
                     "divergence_counts": {str(i): divs.count(i) for i in sorted(set(divs))}, "rows": per_row}
    cq = [q for q in R if q in cut_first and q in cut0]
    n["cut_identity"] = {"n": len(cq), "same_text_k": sum(same_bytes(cut_first[q], cut0[q]) for q in cq)}
    n["gold_tokenisation_disagree"] = sorted(q for q, r in gold_first.items() if not r["forced_source"].get("agree"))
    n["soft_refusals"] = {name: {"k": n["arms"][name]["R"]["soft_k"], "of_ans": n["arms"][name]["R"]["ans_k"],
                                 "n": n["arms"][name]["R"]["n"]}
                          for name in ("Steer-first", "Steer-0 (§39)", "Steer-first-cut", "Gold-first")}
    n["soft_refusal_pattern"] = SOFT_REFUSAL_PATTERN
    return n


# --- rendering ------------------------------------------------------------------------------------------------

def _c(b: dict, name: str) -> str:
    return "-" if not b["n"] else f"{b[name + '_k']}/{b['n']} ({b[name]:.3f})"


def _s(x) -> str:
    return "-" if x is None else f"{x:.3f}"


def _yn(x) -> str:
    return "-" if x is None else ("yes" if x else "no")


def render(n: dict) -> str:
    rep = n["repro"]
    L = ["# §39b report: first-token-only clean-cache control (DEVELOPMENT ONLY, folds 0-1)", "",
         f"Run {n['run_dir']} (engine {n['engine']}); §38 source {n['source']}; §39 rows {n['source39']} (arms.jsonl "
         f"sha256 {n['source39_arms_sha256']}). Gold: D1 string rule, PROVISIONAL; ans is the primary quantity.",
         f"Forcing: {n['forcing']}.", f"Tokenisation of a forced text after 'Answer:': {n['tokenisation']}.", "",
         "## Reproduction",
         f"Gate (flag rule against the §38 dev rows, bound {st.REPRO_GATE}): {rep['rate']:.3f} -> "
         f"{'pass' if rep['pass'] else 'FAIL'} (R {rep['rate_R']:.3f}; dropped {rep['dropped']}).",
         f"Byte-identity with the §38 dev rows: {rep['byte_k']}/{rep['byte_n']} ({_s(rep['byte_rate'])}; stop below "
         f"{BYTE_GATE}) -> {'pass' if rep['byte_pass'] else 'FAIL'}.",
         f"Flags match, bytes differ (kept, marked): {rep['flags_ok_bytes_differ'] or 'none'}.", ""]
    if not (rep["pass"] and rep["byte_pass"]):
        return "\n".join(L + ["**Reproduction or byte-identity failed: no forced output is read.**"])
    u, s0 = n["unknown_check"], n["step0_check"]
    L += ["## Checks",
          f"- Unknown-first equals the A row byte-exact on {u['k']}/{u['n']} (must be all; the unpushed ':' argmax "
          f"equals the forced ' unknown' on {u['natural_first_is_forced_k']}/{u['n']}).",
          f"- Step-0 first token (Steer-0, one new token) equals the §39 Steer-0 first token on {s0['k']}/{s0['n']} "
          "(must be all).", ""]
    if u["k"] != u["n"] or not u["n"] or s0["k"] != s0["n"] or not s0["n"]:
        return "\n".join(L + ["**A check failed: no forced output is read.**"])
    a = n["a0_g0_from_39"]
    L += ["## Reading (amendment 2, pre-registered)", "",
          f"a0 = {a['a0']}/{a['n']}, g0 = {a['g0']}/{a['n']} read from the §39 Steer-0 rows ({a['assertion']}).", "",
          reading_line(n["reading"]), "",
          f"## Arms on the {n['n_R']} reproduced R rows (pairs {n['n_pairs']}, unseen {n['n_unseen']})", "",
          "| arm | n | ans | degenerate | refusal | D1 gold (PROVISIONAL) | ans pairs | gold pairs | ans unseen | gold unseen |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for name, v in n["arms"].items():
        b, p, q = v["R"], v["pairs"], v["unseen"]
        L.append(f"| {name} | {b['n']} | {_c(b, 'ans')} | {_c(b, 'degenerate')} | {_c(b, 'refusal')} | {_c(b, 'gold')} | "
                 f"{_c(p, 'ans')} | {_c(p, 'gold')} | {_c(q, 'ans')} | {_c(q, 'gold')} |")
    L += ["", "## Soft refusals (descriptive; answer-class phrases matching "
          + f"/{n['soft_refusal_pattern']}/, case-insensitive; §41 rev 1)", ""]
    for name, v in n["soft_refusals"].items():
        L.append(f"- {name}: {v['k']} of {v['of_ans']} answers ({v['n']} rows)")
    idn = n["identity"]
    L += ["", "## Steer-first against Steer-0 per row (descriptive)", "",
          f"Full text identical on {idn['same_text_k']}/{idn['n']}; divergence index (first differing generated token) "
          f"counts: {idn['divergence_counts'] or 'none'}. Steer-first-cut = Steer-0-cut text on "
          f"{n['cut_identity']['same_text_k']}/{n['cut_identity']['n']}. Gold-first in-context and alone first ids "
          f"disagree on: {n['gold_tokenisation_disagree'] or 'no rows'}.", "",
          "| question | forced | Steer-first | Steer-0 | same text | divergence | Gold-first | Unknown-first = A |",
          "|---|---|---|---|---|---|---|---|"]
    for q, v in idn["rows"].items():
        gtxt = "-" if v["gold_first"] is None else (f"{v['gold_first_forced']!r} -> {v['gold_first']!r} "
                                                    f"({v['gold_first_cls']}, gold {_yn(v['gold_first_gold'])})")
        L.append(f"| {q} | {v['forced']!r} | {v['steer_first']!r} ({v['steer_first_cls']}, gold "
                 f"{_yn(v['steer_first_gold'])}) | {v['steer0']!r} ({v['steer0_cls']}, gold {_yn(v['steer0_gold'])}) | "
                 f"{_yn(v['same_text'])} | {'-' if v['divergence'] is None else v['divergence']} | {gtxt} | "
                 f"{_yn(v['unknown_first_same_as_A'])} |")
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
    parser.add_argument("--run-dir", default="runs/exp39b_dev/qwen3b")
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
