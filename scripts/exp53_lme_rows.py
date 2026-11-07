#!/usr/bin/env python
"""§53 arm C rows (laptop, no model): LongMemEval folds 0-1 for Qwen 3B fixed8 (the §23 traces), the 126 §35/§43
population rows plus the 12 _abs rows of the same folds, with the frozen reader prompt's chat ids (Qwen chat
template, as the §51 engine builds them), gold, flags and the cached §23 output (for the reproduction check).
    HF_HUB_OFFLINE=1 .venv/bin/python scripts/exp53_lme_rows.py   # -> runs/exp53/lme_rows.jsonl + lme_rows_meta.json
"""
import hashlib, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import transformers  # noqa: E402
import exp51_engine as EN  # noqa: E402
from memctl import jacobian_data as JD  # noqa: E402
from memctl.analysis import answer_selection as AS  # noqa: E402
from memctl.analysis.jacobian import wrap_chat  # noqa: E402

tok = transformers.AutoTokenizer.from_pretrained(str(EN.SNAPSHOT))
out, meta = [], {"folds": [0, 1], "dropped": {}}
for f in (0, 1):
    recs, dropped = JD.load_records("qwen3b", f)
    meta["dropped"][f] = dropped
    pop = {r.question_id for r in JD.family_population(recs)}
    for r in recs:
        if r.question_id not in pop and r.answerable:
            continue
        note, _ = AS.split_note(r.output)
        ans = AS.agent_answer(r.output)[0]
        out.append({"question_id": r.question_id, "fold": f, "answerable": r.answerable, "population": r.question_id in pop,
                    "gold": r.gold, "question_type": r.question_type, "all_in_view": r.all_in_view,
                    "cached_output": r.output, "cached_marked": note is not None,
                    "cached_refusal": AS.is_refusal_answer(ans), "cached_correct": r.correct,
                    "chat_ids": list(tok(wrap_chat(tok, r.prompt), add_special_tokens=False).input_ids)})
out.sort(key=lambda r: r["question_id"])
p = ROOT / "runs/exp53/lme_rows.jsonl"
p.write_text("".join(json.dumps(r) + "\n" for r in out))
meta.update({"n": len(out), "n_population": sum(r["population"] for r in out), "n_abs": sum(not r["answerable"] for r in out),
             "n_inview_refusal_marked": sum(r["population"] and r["cached_marked"] and r["cached_refusal"] for r in out),
             "max_len": max(len(r["chat_ids"]) for r in out), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
(ROOT / "runs/exp53/lme_rows_meta.json").write_text(json.dumps(meta, indent=1))
print(json.dumps(meta))
