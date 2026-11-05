"""§43 stage 0, data side (no model): §24 fixed8 records for folds 0-1 only, rates and overlap with the Qwen §38 dev rows.
Adapted copy of memctl/jacobian_data.load_records pointed at the exp24 runs/caches/verdicts."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from memctl import jacobian_data as jd
from memctl.analysis import answer_selection as sel

FOLDS = (0, 1)
FAMILIES = ("llama3b", "gemma4b", "phi4mini")


def load_records24(reader, fold, runs_dir="runs", data_path="data/longmemeval/longmemeval_s_cleaned.json",
                   arm="fixed8__fraction0.05", allow_any_fold=False):
    assert allow_any_fold or fold in FOLDS  # folds 2-4 only for label-presence counts (pop34.py)
    run = Path(runs_dir) / f"exp24_{reader}_f{fold}" / arm
    cache = jd.cached_prompts(f"cache/generations_exp24_{reader}")
    items = {(it["episode_id"], it["id"]): it for it in jd._read_jsonl(run / "items.jsonl.gz")}
    scored = {s["episode_id"]: s for s in jd._read_jsonl(run / "steps.jsonl.gz") if s.get("scored")}
    episodes = {e["episode_id"]: e for e in jd._read_jsonl(run / "episodes.jsonl")}
    verdicts = {v["question_id"]: v for v in json.loads((Path(runs_dir) / "exp24_verdicts.json").read_text())
                if v["reader"] == reader and v["arm"] == "fixed8" and v["fold"] == fold}
    dropped = {"no_cached_prompt": 0, "evidence_not_found": 0}
    recs = []
    for eid, step in scored.items():
        q = str(episodes[eid]["env_stats"]["episode"])
        active = [items[(eid, i)] for i in step["memory"]["active_ids"] if i != step["observation_id"]]
        active.sort(key=lambda it: it.get("created_at", 0))
        lines = tuple(jd.memory_line(it) for it in active)
        hit = cache.get("\n".join(lines))
        if hit is None:
            dropped["no_cached_prompt"] += 1; continue
        prompt, output = hit
        gold, qt, answerable = jd.question_meta(data_path, q)
        complete = (episodes[eid].get("evidence_complete_rate") or 0.0) >= 1.0
        req = jd.evidence_item_ids(data_path, q) if answerable else ()
        needed = set().union(*req) if req else set()
        ev = tuple(l for it, l in zip(active, lines) if it["id"] in needed)
        if complete and answerable and not ev:
            dropped["evidence_not_found"] += 1; continue
        v = verdicts.get(q, {})
        recs.append(jd.PromptRecord(q, fold, reader, prompt, output, gold, qt, answerable, lines, ev, complete,
                                    v.get("correct"), v.get("unknown")))
    recs.sort(key=lambda r: r.question_id)
    return recs, dropped, len(verdicts)


def rates(recs):
    n = len(recs)
    marked = [r for r in recs if sel.split_note(r.output)[0] is not None]
    def frac(k, d): return {"k": k, "n": d, "rate": (k / d if d else None)}
    ref_cached = [r for r in marked if r.correct is False and r.unknown is True]
    ref_rule = [r for r in marked if sel.is_refusal_answer(sel.agent_answer(r.output)[0])]
    d1s = [sel.gold_in_note(r) for r in marked]
    d1_def = [x for x in d1s if x is not None]
    return {
        "n": n,
        "marker": frac(len(marked), n),
        "refusal_cached_among_marked": frac(len(ref_cached), len(marked)),
        "refusal_rule_among_marked": frac(len(ref_rule), len(marked)),
        "refusal_rule_all_rows": frac(sum(sel.is_refusal_answer(sel.agent_answer(r.output)[0]) for r in recs), n),
        "judged_correct_all": frac(sum(r.correct is True for r in recs), sum(r.correct is not None for r in recs)),
        "judged_correct_marked": frac(sum(r.correct is True for r in marked), sum(r.correct is not None for r in marked)),
        "d1_marked": frac(sum(d1_def), len(d1_def)),
        "d1_among_rule_refusals": frac(sum(bool(sel.gold_in_note(r)) for r in ref_rule), len(ref_rule)),
    }


def population(recs):  # the §35 rule for one reader: in view, answerable, >=1 evidence line, judge label present
    return [r for r in recs if r.answerable and r.all_in_view and r.evidence_lines and r.correct is not None]


def qwen_dev():
    from scripts.exp35_report import load
    data, _, _ = load({0, 1}, Path("runs/exp35"), "runs", Path("runs/exp34_jlens_own"))
    d = data["qwen3b"]
    return d, sorted(d.frozen)


if __name__ == "__main__":
    d, qdev = qwen_dev()
    qrec = {q: d.by_q[q] for q in qdev}
    out = {"folds": list(FOLDS), "qwen_dev": {"n": len(qdev), "qids": qdev,
           "rates": rates([qrec[q] for q in qdev])}, "families": {}}
    print("qwen dev rows", len(qdev))
    for fam in FAMILIES:
        recs, dropped, nverd = [], {}, 0
        for f in FOLDS:
            r, dr, nv = load_records24(fam, f)
            recs += r; nverd += nv
            for k, v in dr.items(): dropped[k] = dropped.get(k, 0) + v
        by = {r.question_id: r for r in recs}
        pop = population(recs)
        inview = [r for r in recs if r.all_in_view]
        common = [q for q in qdev if q in by]
        common_pop = [q for q in qdev if q in {r.question_id for r in pop}]
        same_mem = sum(by[q].memory_lines == qrec[q].memory_lines for q in common)
        same_prompt = sum(by[q].prompt == qrec[q].prompt for q in common)
        cm = [by[q] for q in common]
        fam_out = {
            "n_records": len(recs), "n_verdicts": nverd, "dropped": dropped,
            "p_all_in_view": len(inview) / len(recs),
            "all_rows": rates(recs), "all_in_view": rates(inview),
            "population_35": rates(pop),
            "qwen_dev_overlap": {"present": len(common), "in_family_population": len(common_pop),
                                 "same_memory_lines": same_mem, "same_prompt_string": same_prompt,
                                 "marked_in_family": sum(sel.split_note(r.output)[0] is not None for r in cm)},
            "common_rows": rates(cm),
            "common_marked_qids": [r.question_id for r in cm if sel.split_note(r.output)[0] is not None],
        }
        out["families"][fam] = fam_out
        print(fam, json.dumps({k: v for k, v in fam_out.items() if k not in ("common_marked_qids",)}, indent=0)[:3000])
    Path("runs/exp43_stage0/data.json").write_text(json.dumps(out, indent=1))
