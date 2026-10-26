"""§43 amendment 1 item 2: §34-style population counts per family from the §24 caches (both sizes), by fold.
Counts only on folds 2-4 (no outcome is read there: only label presence)."""
import hashlib, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import data as D

PAIRS = {"llama": ("llama3b", "llama8b"), "gemma": ("gemma4b", "gemma12b"), "phi": ("phi4mini", "phi4")}


def load_any(reader, fold):
    D.FOLDS_ALL = True
    return D.load_records24(reader, fold, allow_any_fold=True)


out = {}
for fam, (small, large) in PAIRS.items():
    out[fam] = {}
    for fold in range(5):
        rs = {}
        for r in (small, large):
            recs, dropped, nv = D.load_records24(r, fold, allow_any_fold=True)
            rs[r] = {x.question_id: x for x in recs}
        def inview(x): return x.answerable and x.all_in_view and bool(x.evidence_lines)
        both = sorted(set(q for q, x in rs[small].items() if inview(x)) & set(q for q, x in rs[large].items() if inview(x)))
        same = [q for q in both if hashlib.sha256(rs[small][q].prompt.encode()).hexdigest() == hashlib.sha256(rs[large][q].prompt.encode()).hexdigest()]
        labelled = [q for q in same if rs[small][q].correct is not None and rs[large][q].correct is not None]
        out[fam][fold] = {"records_small": len(rs[small]), "records_large": len(rs[large]),
                          "inview_answerable_small": sum(inview(x) for x in rs[small].values()),
                          "inview_answerable_large": sum(inview(x) for x in rs[large].values()),
                          "both_sizes": len(both), "same_prompt_hash": len(same), "with_judge_label_both": len(labelled)}
    tot = {k: sum(v[k] for v in out[fam].values()) for k in out[fam][0]}
    out[fam]["total"] = tot
    out[fam]["dev_0_1"] = {k: out[fam][0][k] + out[fam][1][k] for k in tot}
    print(fam, json.dumps(out[fam]))
# Qwen §34 reference (same rule, exp23 caches)
from memctl.jacobian_data import load_records
q = {}
for fold in range(5):
    a = {x.question_id: x for x in load_records("qwen3b", fold)[0]}; b = {x.question_id: x for x in load_records("qwen7b", fold)[0]}
    iv = lambda x: x.answerable and x.all_in_view and bool(x.evidence_lines)
    both = [k for k in a if k in b and iv(a[k]) and iv(b[k])]
    q[fold] = {"both_sizes": len(both), "same_prompt": sum(a[k].prompt == b[k].prompt for k in both)}
out["qwen_reference"] = q
print("qwen", q)
Path("runs/exp43_stage0/pop34.json").write_text(json.dumps(out, indent=1))
