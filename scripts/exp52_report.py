"""§52 stage 0 report: per-reader measures and the frozen gate from runs/exp52/gen_{key}.jsonl.
Writes runs/exp52/report.json and prints a markdown table."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from memctl.analysis import answer_selection as sel
from memctl.envs.utilmem import is_refusal as phrase_refusal

OUT = Path("runs/exp52")
fz = json.loads((OUT / "freeze.json").read_text())
G = fz["gate"]
rows = {x["question_id"]: x for x in map(json.loads, (OUT / "rows52.jsonl").read_text().splitlines())}
frac = lambda k, n: {"k": k, "n": n, "rate": round(k / n, 4) if n else None}
rep = {}
for key in fz["reader_order"]:
    p = OUT / f"gen_{key}.jsonl"
    if not p.exists():
        continue
    gens = {g["question_id"]: g for g in map(json.loads, p.read_text().splitlines())}
    prim = [q for q, x in rows.items() if x["set"] == "primary"]
    if not all(q in gens for q in prim):
        rep[key] = {"status": f"incomplete primary ({sum(q in gens for q in prim)}/126)"}
        continue
    def parse(q):
        t = gens[q]["text"]
        a, m = sel.agent_answer(t)
        return t, a, m, sel.split_note(t)[0]
    P = {q: parse(q) for q in prim}
    marked = [q for q in prim if P[q][2]]
    abst = [q for q in marked if sel.is_refusal_answer(P[q][1])]
    floor = [q for q in prim if sel.d1(P[q][1], rows[q]["gold"])]
    floor_m = [q for q in marked if sel.d1(P[q][1], rows[q]["gold"])]
    ntr = [q for q in abst if sel.d1(P[q][3], rows[q]["gold"])]
    phr = [q for q in prim if phrase_refusal(P[q][1]) and not sel.is_refusal_answer(P[q][1])]
    r = {"marker_rate": frac(len(marked), len(prim)), "in_view_abstention": frac(len(abst), len(marked)),
         "reading_floor": frac(len(floor), len(prim)), "reading_floor_marked_only": frac(len(floor_m), len(prim)),
         "note_then_refuse": frac(len(ntr), len(abst)), "refusal_phrase_not_unknown": len(phr),
         "post_hoc_any_output_unknown": frac(sum(sel.is_refusal_answer(P[q][1]) for q in prim), len(prim)),
         "post_hoc_unmarked_unknown": frac(sum(sel.is_refusal_answer(P[q][1]) for q in prim if not P[q][2]), len(prim) - len(marked)),
         "hit_max_tokens": sum(gens[q]["hit_max"] for q in prim),
         "mean_row_s": round(sum(gens[q]["row_s"] for q in prim) / len(prim), 1)}
    sec = [q for q, x in rows.items() if x["set"] == "secondary" and q in gens]
    if sec:
        Sx = {q: parse(q) for q in sec}
        niv = [q for q in sec if rows[q]["answerable"] and not rows[q]["all_in_view"] and Sx[q][2]]
        una = [q for q in sec if not rows[q]["answerable"] and Sx[q][2]]
        r["unknown_not_in_view_answerable"] = frac(sum(sel.is_refusal_answer(Sx[q][1]) for q in niv), len(niv))
        nivall = [q for q in sec if rows[q]["answerable"] and not rows[q]["all_in_view"]]
        r["post_hoc_any_output_unknown_not_in_view"] = frac(sum(sel.is_refusal_answer(Sx[q][1]) for q in nivall), len(nivall))
        r["unknown_unanswerable"] = frac(sum(sel.is_refusal_answer(Sx[q][1]) for q in una), len(una))
        r["secondary_done"] = f"{len(sec)}/74"
    fails = []
    if r["marker_rate"]["rate"] < G["marker_rate_min"]: fails.append("marker")
    if r["reading_floor"]["rate"] < G["reading_floor_min"]: fails.append("floor")
    ia = r["in_view_abstention"]
    if ia["rate"] is None or ia["rate"] < G["in_view_abstention_min"] or ia["k"] < G["in_view_abstention_k_min"]: fails.append("abstention")
    r["gate"] = "PASS" if not fails else "FAIL (" + ", ".join(fails) + ")"
    rep[key] = r
(OUT / "report.json").write_text(json.dumps(rep, indent=1))
print("| reader | marker | in-view abstention | floor | note-then-refuse | §24 phrases | unknown, not in view | post hoc: any-output unknown | gate |")
print("|---|---|---|---|---|---|---|---|---|")
f = lambda d: f"{d['rate']:.3f} ({d['k']}/{d['n']})" if d and d.get("rate") is not None else "–"
for k, r in rep.items():
    if "gate" not in r:
        print(f"| {k} | {r['status']} |"); continue
    print(f"| {k} | {f(r['marker_rate'])} | {f(r['in_view_abstention'])} | {f(r['reading_floor'])} | {f(r['note_then_refuse'])} "
          f"| {r['refusal_phrase_not_unknown']} | {f(r.get('unknown_not_in_view_answerable'))} | {f(r['post_hoc_any_output_unknown'])} | {r['gate']} |")
