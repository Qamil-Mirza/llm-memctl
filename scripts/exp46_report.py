#!/usr/bin/env python
"""§46 report: probe tables, the S1 / S2 transfer profiles with k1 / k2 / k3, and the §5 verdict, computed.

    .venv/bin/python scripts/exp46_report.py --probe runs/exp46_probe/cuda-bfloat16 --patch runs/exp46_patch/cuda-bfloat16
    .venv/bin/python scripts/exp46_report.py ... --held          # refuses without design_committed.json in both dirs

Spec: research/proposals/2026-10-30-exp46-instrument-check.md §3-§5 with revs 1-3. Dev mode reads only dev
triples and probe_dev.json; --held adds probe_held.json and the held triples and computes the §5 verdict from the
first decodable layer of the RC supervisor probe at P_last and L* at S1 on held pairs (under 30 held pairs: the
verdict row is marked under-powered, rev 1 ruling 1). Writes <patch>/report_{dev,held}.json and .md.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from memctl.analysis import patch46 as T  # noqa: E402
from memctl.analysis import probe46 as P  # noqa: E402

EXIT_DESIGN = 4


def read_json(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def load_stage(patch: Path, stage: str, split: str) -> list[dict]:
    out = []
    for f in sorted((patch / stage).glob("*.json")):
        r = read_json(f)
        if r and r.get("entered") and r.get("split") == split:
            out.append(r)
    return out


def profiles(records: list[dict], kinds=("patch", "k1", "k2"), flags: dict | None = None) -> dict:
    by = defaultdict(lambda: defaultdict(list))
    copy = defaultdict(lambda: defaultdict(list))
    for rec in records:
        for x in rec["results"]:
            if x["kind"] in kinds:
                by[x["kind"]][x["layer"]].append(x["outcome"])
                copy[x["kind"]][x["layer"]].append(bool(x["names_before_city"]))
    out = {k: {str(l): v for l, v in T.transfer_profile(dict(by[k])).items()} for k in by}
    out["name_copy_share"] = {k: {str(l): sum(v) / len(v) for l, v in sorted(copy[k].items())} for k in copy}
    return out


def k3_share(records: list[dict]) -> dict:
    xs = [x["reproduces_Pp"] for r in records for x in r["results"] if x["kind"] == "k3"]
    k = sum(bool(x) for x in xs)
    p, lo, hi = T.wilson(k, len(xs)) if xs else (None, None, None)
    return {"n": len(xs), "k": k, "share": p, "lower": lo, "pass": (p is not None and p >= T.K3_MIN)}


def k2_by_flag(records: list[dict], cases: dict) -> dict:
    split = defaultdict(list)
    for r in records:
        split[bool(cases[r["triple_id"]]["b2_reuses_distractor"])].append(r)
    return {("b2_reuses_distractor" if f else "b2_unused"): profiles(v, kinds=("k2",)).get("k2", {})
            for f, v in split.items()}


def lstar(prof: dict) -> int | None:
    def ints(d):
        return {int(l): v for l, v in d.items()}
    return T.carried_through(ints(prof.get("patch", {})), ints(prof.get("k1", {})), ints(prof.get("k2", {})))


def build(probe_dir: Path, patch: Path, split: str, cases: dict) -> dict:
    s1 = load_stage(patch, "s1", split)
    ctl = load_stage(patch, "controls", split)
    s2 = load_stage(patch, "s2", split)
    entered_k2 = [r for r in ctl if any(x["kind"] == "k2" for x in r["results"])]
    prof = profiles(s1 + ctl)
    rep = {"split": split, "pairs": len(s1), "pairs_with_k2": len(entered_k2), "readout": read_json(patch / "readout.json"),
           "s1": prof, "k3": k3_share(s1), "k2_by_flag": k2_by_flag(ctl, cases), "s2": profiles(s2, kinds=("patch", "k1")),
           "l_star_s1": lstar(prof)}
    probe_file = probe_dir / ("probe_held.json" if split == "held" else "probe_dev.json")
    rep["probe"] = read_json(probe_file)
    if split == "held" and rep["probe"]:
        fd = rep["probe"]["reads"][":".join(P.VERDICT_READ)]["first_decodable"]
        rep["verdict"] = T.verdict(fd, rep["l_star_s1"], rep["pairs"])
    return rep


def markdown(rep: dict) -> str:
    lines = [f"# §46 report ({rep['split']})", "", f"Pairs entered: {rep['pairs']} (with k2: {rep['pairs_with_k2']}). "
             f"Readout rule: {(rep['readout'] or {}).get('rule')} (agreement {(rep['readout'] or {}).get('share')}).",
             f"k3 token-substitution floor: {rep['k3']['k']}/{rep['k3']['n']} (pass {rep['k3']['pass']}). "
             f"L* at S1: {rep['l_star_s1']}.", "", "## S1 transfer by layer", "",
             "| layer | Transfer | lower | k1 | k2 | name-copy (patch) |", "|---|---|---|---|---|---|"]
    s1 = rep["s1"]
    for l in range(P.N_LAYERS):
        t = s1.get("patch", {}).get(str(l), {})
        a = s1.get("k1", {}).get(str(l), {})
        b = s1.get("k2", {}).get(str(l), {})
        nc = s1.get("name_copy_share", {}).get("patch", {}).get(str(l))
        f = lambda v: "—" if v is None else f"{v:.3f}"
        lines.append(f"| {l} | {f(t.get('share'))} | {f(t.get('lower'))} | {f(a.get('share'))} | {f(b.get('share'))} | {f(nc)} |")
    if rep.get("probe"):
        lines += ["", "## Probe", ""]
        for key, r in rep["probe"]["reads"].items():
            if rep["split"] == "held":
                lines.append(f"- {key}: first decodable layer {r.get('first_decodable')}")
            else:
                ok = [l for l, v in r["layers"].items() if v.get("dev_ok")]
                lines.append(f"- {key}: dev-CV clauses hold at layers {', '.join(ok) or 'none'}")
    if rep.get("verdict"):
        v = rep["verdict"]
        lines += ["", "## §5 verdict (computed)", "", f"**{v['verdict']}**" + (" (under-powered: fewer than 30 held pairs)"
                  if v["under_powered"] else ""), "", "```", json.dumps(v, indent=1), "```"]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", type=Path, required=True)
    ap.add_argument("--patch", type=Path, required=True)
    ap.add_argument("--cases46", type=Path, default=Path("runs/exp46_probe/cases46.jsonl"))
    ap.add_argument("--held", action="store_true")
    args = ap.parse_args()
    if args.held:
        for d in (args.probe, args.patch):
            marker = read_json(d / P.DESIGN_MARKER)
            if not marker or not marker.get("commit"):
                print(f"held report refused: {d / P.DESIGN_MARKER} missing or names no commit", file=sys.stderr)
                return EXIT_DESIGN
    cases = {}
    for line in args.cases46.read_text().splitlines():
        r = json.loads(line)
        if r["role"] == "P":
            cases[r["triple_id"]] = r
    split = "held" if args.held else "dev"
    rep = build(args.probe, args.patch, split, cases)
    (args.patch / f"report_{split}.json").write_text(json.dumps(rep, indent=1))
    (args.patch / f"report_{split}.md").write_text(markdown(rep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
