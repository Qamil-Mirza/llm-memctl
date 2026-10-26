"""§43 stage 1 report, Llama 3.2 3B (development folds 0-1 only): the rev 1 A2 gate arithmetic per (L, alpha).

    .venv/bin/python scripts/exp43_llama_report.py --run-dir runs/exp43_llama_dev/llama3b

Recomputes every count from the row files (sweep / cut / random .jsonl), not from gate.json, and checks it against
gate.json's choice. Refuses rows from more than one engine or wording, and any row outside folds 0-1. Writes
<run-dir>/report.json and report.md and prints the table. Gold is the D1 string rule on the phrase: PROVISIONAL.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import exp43_llama_dev as dev  # noqa: E402
from scripts.exp35_generate import read_jsonl  # noqa: E402
from memctl.analysis import steering as st  # noqa: E402


def load(run_dir: Path) -> dict:
    files = {name: read_jsonl(run_dir / f"{name}.jsonl") for name in ("repro", "sweep", "cut", "random", "ncut", "sanity")}
    rows = [r for v in files.values() for r in v]
    engines = {r["engine"] for r in rows}
    wordings = {r["wording_sha256"] for r in rows}
    if len(engines) > 1 or len(wordings) > 1:
        raise SystemExit(f"REFUSED: rows from {len(engines)} engines / {len(wordings)} wordings in {run_dir}")
    bad = sorted({r["fold"] for r in rows} - set(dev.DEV))
    if bad:
        raise SystemExit(f"REFUSED: rows on folds {bad}; the design report reads folds 0-1 only")
    return files


def table(files: dict, G: list[str], settings: list[st.Setting]) -> dict:
    idx = {name: {st.row_key(r): r for r in rs} for name, rs in files.items()}

    def arm(name, kind, key):
        got = [idx[name][("G", kind, key, q)] for q in G if ("G", kind, key, q) in idx[name]]
        return dev.counts(got) if G and len(got) == len(G) else None
    out = {}
    for s in settings:
        steer = arm("sweep", "A", s.key)
        if steer is None:
            continue
        cut = arm("cut", "cut", s.key)
        rnd = arm("random", "A", st.Setting("random", s.layer, s.alpha, s.unit).key)
        out[s.key] = {"layer": s.layer, "alpha": s.alpha, **dev.setting_gate(steer, cut, rnd)}
    return out


def fmt(x) -> str:
    return "—" if x is None else ("yes" if x is True else "no" if x is False else str(x))


def markdown(t: dict, order: list[str], n: int, chosen: str | None, alt: str | None, a_cut: dict | None) -> str:
    lines = [f"Gate on {n} reproduced in-view dev refusals (rev 1 A2): Steer answers >= {dev.ANS_MIN}; "
             f"gold(Steer) - gold(Steer-cut) >= {dev.GOLD_DIFF_MIN}; Random answers <= {dev.RANDOM_SHARE} x Steer's; "
             f"degenerate <= {dev.DEGEN_MAX}. Rows in rule order (deeper layer, then smaller alpha). Gold: D1, provisional.",
             "",
             "| setting | L | alpha | Steer ans | ok | degen | ok | gold Steer | gold cut | diff | ok | Random ans | bound | ok | gate |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k in order:
        if k not in t:
            continue
        e = t[k]
        s, c, r = e["steer"], e["cut"], e["random"]
        lines.append(f"| {k} | {e['layer']} | {e['alpha']:g} | {s['ans']}/{s['n']} | {fmt(e['ans_ok'])} | {s['degenerate']} | "
                     f"{fmt(e['degen_ok'])} | {s['gold']} | {fmt(None if c is None else c['gold'])} | {fmt(e['gold_diff'])} | "
                     f"{fmt(e['gold_ok'])} | {fmt(None if r is None else r['ans'])} | {e['random_bound']:.1f} | "
                     f"{fmt(e['random_ok'])} | {fmt(e['pass'])} |")
    lines += ["", f"A-cut (unsteered, cut prompt): {a_cut}", f"Chosen (rule order): {chosen}",
              f"Highest-Steer-answers alternative (descriptive, §38-style): {alt}"]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="runs/exp43_llama_dev/llama3b")
    args = ap.parse_args(argv)
    run_dir = Path(args.run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    repro = json.loads((run_dir / "repro.json").read_text())
    files = load(run_dir)
    G = repro["G_ok"]
    settings = st.sweep_settings(manifest["layers"], manifest["alphas"], manifest["unit"])
    t = table(files, G, settings)
    order = [s.key for s in dev.gate_order(settings)]
    chosen = next((k for k in order if k in t and t[k]["pass"] is True), None)
    alt = next((s.key for s in dev.alt_order(settings, {k: v["steer"]["ans"] for k, v in t.items()})
                if s.key in t and t[s.key]["pass"] is True), None)
    cut_none = [r for r in files["cut"] if r["setting"]["key"] == "none" and r["question_id"] in G]
    a_cut = dev.counts(cut_none) if len(cut_none) == len(G) else None
    gate_path = run_dir / "gate.json"
    agrees = None
    if gate_path.exists():
        agrees = json.loads(gate_path.read_text()).get("chosen") == chosen
    directions = json.loads((run_dir / "directions.json").read_text()) if (run_dir / "directions.json").exists() else None
    ncheck = json.loads((run_dir / "ncheck.json").read_text()) if (run_dir / "ncheck.json").exists() else None
    out = {"rule": dev.GATE_RULE, "n_G": len(G), "G_dropped": sorted(set(manifest["G"]) - set(G)),
           "n_pairs": len(repro["pairs"]), "n_P0": len(manifest["P0"]), "repro_rate": repro["rate"],
           "repro_first_match": f"{repro['first_match']}/{repro['n']}",
           "ncheck": None if ncheck is None else {k: ncheck[k] for k in ("n", "first_match", "marker", "exact", "pass")},
           "directions": None if directions is None else directions["layers"],
           "table": t, "order": order, "chosen": chosen, "alt_rule_choice": alt, "a_cut": a_cut,
           "gate_json_agrees": agrees, "gold": "D1 string rule, provisional"}
    md = markdown(t, order, len(G), chosen, alt, a_cut)
    (run_dir / "report.json").write_text(json.dumps(out, indent=1, default=float))
    (run_dir / "report.md").write_text(md + "\n")
    print(f"pairs {out['n_pairs']} of {out['n_P0']} candidates; gate rows {len(G)} (dropped {out['G_dropped']}); "
          f"reproduction {repro['rate']:.3f}; N check {out['ncheck']}")
    print(md)
    if agrees is False:
        print("WARNING: gate.json's choice differs from the recomputed one")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
