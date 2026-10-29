#!/usr/bin/env python
"""§44 rev 1 amendment §C: claims S0-S3 and U1 from runs/exp44_swap/<model>/ (conditions S and U).

    .venv/bin/python scripts/exp44_swap_report.py --model qwen3b                 # folds 0-1, descriptive
    .venv/bin/python scripts/exp44_swap_report.py --model qwen3b --root runs/exp44_swap/smoke --tag smoke
    .venv/bin/python scripts/exp44_swap_report.py --model qwen3b --pooled        # all 330 (refused until authorised)

Spec: research/proposals/2026-10-29-exp44-patching.md rev 1 amendment §C, rev 2 (5146f5b): the claims are read on all
330 rows (the POOL verdict), with the fold scope a parameter. A scope with a fold outside 0-1 needs --pooled and
POOL_AUTHORISED (exit 2 otherwise); a dev-only scope is descriptive and decides nothing. Rows of other folds are never
opened. Llama 3B: descriptive (rev 2): S0 and the class shares are reported, S1-S2 are not read.

- S0: following rate (1 nat) among successes >= 0.80 with CI lower >= 0.70, else "swap not valid" and S1-S3 not read.
- S1: among in-view failures (refusal + wrong): class shares at 0.5 / 1 / 2 nats, the answer-production share and CI;
  established iff (a) indeterminate <= 0.20, (b) |AP(0.5) - AP(2)| < 0.15, (c) CI half-width <= 0.15; then >= 0.50
  "answer-production majority", <= 0.25 "evidence-use majority", else mixed.
- S2: refusal rows of the swap subset: p(A_steer gold | AP) - p(A_steer gold | EU), CI lower > 0, read only with >= 8
  rows per class.
- S3: median forward d_T with CI, reverse beside; mismatched-donor mean shift within +-0.5 nat at every layer, else not
  read. Beside: rev 0's d_c (runs/exp44_patch) and the §34 lens depths (Qwen; runs/exp34_jlens), where on disk.
- U1: refusals that become gold under U (full population in scope; the swap subset beside): >= 0.50 with lower > 0.40
  "mostly a prompt artefact", <= 0.25 with upper < 0.35 "not the format", else partial; Steer-under-U - U beside.
Percentile bootstrap, 10,000 draws (--samples). Writes <root>/<model>/report_<tag>.{md,json}.
Exit 2: scope refused. Exit 6: a following row is not patched (or lacks its donor control).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memctl.analysis import swap as sw  # noqa: E402

DEV_FOLDS = (0, 1)
POOL_AUTHORISED = True    # rev 4 (f9952b2): POOL on all 330 rows; the swap set is fixed from here
READ_CLAIMS = {"qwen3b": True, "llama3b": False}       # rev 2: Llama descriptive (S0 and shares only)
REFUSED, UNPATCHED = 2, 6


def check_scope(folds, pooled: bool) -> None:
    bad = sorted(set(folds) - set(DEV_FOLDS))
    if bad and not (pooled and POOL_AUTHORISED):
        print(f"exp44_swap_report: folds {bad} outside 0-1 need --pooled and POOL_AUTHORISED", file=sys.stderr)
        raise SystemExit(REFUSED)


def load_dir(d: Path, folds) -> list[dict]:
    out = []
    for p in sorted(d.glob("*.json")) if d.exists() else []:
        row = json.loads(p.read_text())
        if row["fold"] in folds:
            out.append(row)
    return out


def beside(model: str, rows: list[dict], patch_root: Path, exp34: Path) -> dict:
    """rev 0 d_c and §34 lens depths for the patched rows, where on disk (descriptive)."""
    dc, lens = [], {"logit": [], "jlens": []}
    for r in rows:
        p = patch_root / model / f"{r['question_id']}.json"
        if p.exists():
            v = json.loads(p.read_text()).get("d_c")
            if v is not None:
                dc.append(v)
        q = exp34 / model / f"{r['question_id']}.json"
        if q.exists():
            m = json.loads(q.read_text())["measures"]
            for k in lens:
                v = m[k]["P1"]["gold"]["emergence_depth"]
                if v is not None:
                    lens[k].append(v)
    med = lambda xs: {"n": len(xs), "median": median(xs) if xs else None}  # noqa: E731
    return {"rev0_d_c": med(dc), "lens_logit": med(lens["logit"]), "lens_jlens": med(lens["jlens"])}


def build(root: Path, model: str, folds, samples: int = sw.BOOT_SAMPLES, patch_root: Path = Path("runs/exp44_patch"),
          exp34: Path = Path("runs/exp34_jlens")) -> tuple[str, dict]:
    S = load_dir(root / model / "S", folds)
    U = load_dir(root / model / "U", folds)
    for r in S:
        if r["follows"]["1"] and (not r.get("patched") or (r.get("donor") and not r.get("control_mismatch"))):
            print(f"exp44_swap_report: {r['question_id']} follows but is not fully patched", file=sys.stderr)
            raise SystemExit(UNPATCHED)
    succ = [r for r in S if r["label"] == "success"]
    fail = [r for r in S if r["label"] in sw.FAILURE_LABELS]
    patched = [r for r in S if r.get("patched")]
    u_by_q = {r["question_id"]: r for r in U}
    claims = {"S0": sw.s0(succ, samples)}
    valid = claims["S0"]["pass"]
    claims["S1"] = sw.s1(fail, samples)
    ref = []
    for r in S:
        u = u_by_q.get(r["question_id"])
        if r["label"] == "refusal" and u and not u.get("no_marker") and "A_steer" in u:
            ref.append({"cls": r["cls"], "steer_gold": u["A_steer"]["gold"], "a_gold": u["A"]["gold"],
                        "question_id": r["question_id"]})
    claims["S2"] = sw.s2(ref, samples)
    claims["S3"] = sw.s3(patched, samples)
    claims["S3"]["beside"] = beside(model, patched, patch_root, exp34)
    claims["U1"] = sw.u1(U, samples)
    claims["U1_swap_subset"] = sw.u1([u for u in U if u.get("in_swap_subset")], samples)
    read = READ_CLAIMS[model] and bool(set(folds) - set(DEV_FOLDS))
    status = {"S0": "read" if read or not READ_CLAIMS[model] else "descriptive (dev scope)",
              "S1": "read" if read and valid else ("not read: S0 failed" if read else "descriptive"),
              "S2": "read" if read and valid and claims["S2"]["readable"] else "descriptive",
              "S3": "read" if read and valid and claims["S3"]["read"] else "descriptive / not read",
              "U1": "read" if read else "descriptive"}
    if not READ_CLAIMS[model]:
        status = {k: "descriptive (Llama, rev 2)" for k in status}
    labels = {lab: sum(r["label"] == lab for r in S) for lab in ("success", "refusal", "wrong")}
    cls_by_label = {lab: {c: sum(r["label"] == lab and r["cls"]["1"] == c for r in S) for c in sw.CLASSES}
                    for lab in ("success", "refusal", "wrong")}
    summary = {"model": model, "folds": list(folds), "n_swap_rows": len(S), "labels": labels,
               "classes_by_label_1nat": cls_by_label, "n_patched": len(patched), "n_u_rows": len(U),
               "status": status, "claims": claims}
    return markdown(summary), summary


def _f(x, d=3):
    return "—" if x is None else (f"{x:.{d}f}" if isinstance(x, float) else str(x))


def _ci(s):
    return f"{_f(s.get('value'))} [{_f(s.get('ci_low'))}, {_f(s.get('ci_high'))}]"


def markdown(s: dict) -> str:
    c = s["claims"]
    L = [f"# §44 swap conditions S / U: {s['model']}, folds {s['folds']}", "",
         "Spec: research/proposals/2026-10-29-exp44-patching.md rev 1 amendment §C, rev 2. Status per claim: "
         + json.dumps(s["status"]), "",
         f"Swap rows {s['n_swap_rows']} (labels {s['labels']}); patched {s['n_patched']}; U rows {s['n_u_rows']}.",
         f"Classes at 1 nat by §23 label: {s['classes_by_label_1nat']}", "",
         f"**S0** following rate among successes: {_ci(c['S0'])} (n={c['S0']['n']}, k={c['S0'].get('k')}); by margin "
         f"{c['S0'].get('by_margin')}; need >= 0.80 and lower >= 0.70 -> {c['S0']['reading']}", "",
         f"**S1** failures n={c['S1']['n']}; shares by margin {json.dumps(c['S1']['shares'])}; answer-production "
         f"(1 nat) {_ci(c['S1']['ap'])}, half-width {_f(c['S1']['ap'].get('half_width'))}; (a) {c['S1']['a']} "
         f"(b) {c['S1']['b']} (move {_f(c['S1'].get('ap_margin_move'))}) (c) {c['S1']['c']} -> {c['S1']['reading']}",
         "",
         f"**S2** refusal rows: AP n={c['S2']['n_a']} p={_f(c['S2']['p_ap'])}, EU n={c['S2']['n_b']} "
         f"p={_f(c['S2']['p_eu'])}; difference {_ci(c['S2'])} -> {c['S2']['reading']}", "",
         f"**S3** d_T forward median {_ci(c['S3']['d_T_fwd'])} (n={c['S3']['d_T_fwd']['n']}), reverse "
         f"{_ci(c['S3']['d_T_rev'])}; asymmetry median {_f(c['S3']['asymmetry_median'])}; mismatched-donor max |mean "
         f"shift| {_f(c['S3']['mismatch_max_abs_mean_shift'])} (bound 0.5) -> within {c['S3']['mismatch_within_bound']}"
         f"; beside {json.dumps(c['S3']['beside'])}", "",
         f"Mismatched-donor mean shift per layer: {[round(v, 3) for v in c['S3']['mismatch_mean_shift_per_layer']]}",
         f"Random-overwrite mean shift per layer: {[round(v, 3) for v in c['S3']['random_mean_shift_per_layer']]}", "",
         f"**U1** (full population refusals in scope) gold under U {_ci(c['U1'])} (n={c['U1']['n']}, no marker "
         f"{c['U1']['n_no_marker']}); U_steer share {_f(c['U1']['u_steer_share'])}; Steer-under-U - U "
         f"{_ci(c['U1']['steer_minus_u'])}; U still refuses {_f(c['U1']['u_refusal_share'])} -> {c['U1']['reading']}",
         f"U1 on the swap subset: {_ci(c['U1_swap_subset'])} (n={c['U1_swap_subset']['n']}) -> "
         f"{c['U1_swap_subset']['reading']}", ""]
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(READ_CLAIMS))
    ap.add_argument("--root", default="runs/exp44_swap")
    ap.add_argument("--folds", default="0,1")
    ap.add_argument("--pooled", action="store_true")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--samples", type=int, default=sw.BOOT_SAMPLES)
    args = ap.parse_args(argv)
    folds = tuple(range(5)) if args.pooled else tuple(int(f) for f in args.folds.split(","))
    check_scope(folds, args.pooled)
    text, summary = build(Path(args.root), args.model, folds, args.samples)
    tag = args.tag or ("pooled" if args.pooled else "dev")
    out = Path(args.root) / args.model
    out.mkdir(parents=True, exist_ok=True)
    (out / f"report_{tag}.md").write_text(text)
    (out / f"report_{tag}.json").write_text(json.dumps(summary, indent=1))
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
