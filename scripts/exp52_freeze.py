"""§52 stage 0 freeze (before any model output): readers at pinned revisions, the row list (folds 0-1 only), prompt
hashes, decoding, measures and the gate as frozen in research/proposals/2026-11-04-exp52-small-readers.md rev 0
(094cc5e, D-74). Writes runs/exp52/rows52.jsonl and runs/exp52/freeze.json."""
import hashlib, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import transformers, torch
from memctl import jacobian_data as jd
from memctl.analysis import families as F

OUT = Path("runs/exp52")
sha = lambda s: hashlib.sha256(s.encode()).hexdigest()
probe = json.loads((OUT / "probe.json").read_text())
recs = {r.question_id: r for f in (0, 1) for r in jd.load_family_records(F.LLAMA3B, f)[0]}
# the prompt bytes are the same in every §24 family (fixed8 does not depend on the reader); checked here again
for k in ("gemma4b", "phi4mini"):
    o = {r.question_id: r for f in (0, 1) for r in jd.load_family_records(F.get(k), f)[0]}
    assert set(o) == set(recs) and all(o[q].prompt == recs[q].prompt for q in recs), k
pop = {r.question_id for r in jd.family_population(list(recs.values()))}
assert len(recs) == 200 and len(pop) == 126
rows = []
for q in sorted(recs, key=lambda q: (q not in pop, recs[q].fold, q)):
    r = recs[q]
    rows.append({"question_id": q, "fold": r.fold, "set": "primary" if q in pop else "secondary",
                 "answerable": r.answerable, "all_in_view": r.all_in_view, "n_evidence_lines": len(r.evidence_lines),
                 "question_type": r.question_type, "gold": r.gold, "prompt_sha256": sha(r.prompt)})
(OUT / "rows52.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows))
prim = [x for x in rows if x["set"] == "primary"]
sec = [x for x in rows if x["set"] == "secondary"]
ORDER = ("gemma270m", "qwen05b", "llama1b", "gemma1b", "qwen15b", "smol17b")
freeze = {
    "experiment": "§52 stage 0: screen of readers below 3B (laptop, CPU fp32, $0)",
    "spec": "research/proposals/2026-11-04-exp52-small-readers.md rev 0 (094cc5e), D-74; scope relayed by the Advisor 2026-11-08",
    "frozen_before_any_model_output": True,
    "readers": {k: {"model_id": probe[k]["model_id"], "revision": probe[k]["revision"], "config": F.get52(k).stamp(),
                    "n_layers": probe[k]["n_layers"], "hidden": probe[k]["hidden"],
                    "chat_template_sha256": probe[k]["chat_template_sha256"]} for k in ORDER},
    "reader_order": list(ORDER),
    "gemma270m_role": "optional floor (proposal); screened and gated like the others",
    "rows": {"file": "runs/exp52/rows52.jsonl", "sha256": hashlib.sha256((OUT / "rows52.jsonl").read_bytes()).hexdigest(),
             "folds": [0, 1], "folds_2_4": "not read",
             "primary": {"n": len(prim), "rule": "jacobian_data.family_population on the §24 fixed8 records (answerable, all evidence in view, >=1 evidence line, §24 judge label present); the §43 stage-0 126 rows, identical in Llama/Gemma/Phi"},
             "secondary": {"n": len(sec), "answerable_not_all_in_view": sum(x["answerable"] and not x["all_in_view"] for x in sec),
                           "unanswerable": sum(not x["answerable"] for x in sec), "use": "descriptive only (the 'unknown' rate when the evidence is not all in view); run after every reader's primary set"},
             "prompt_source": "cache/generations_exp24_llama3b (byte-identical in gemma4b, phi4mini)",
             "prompts_sha256": sha("".join(x["prompt_sha256"] for x in rows))},
    "prompt": {"frozen_reader_prompt": "7b5fc30 (D-05), fixed8 arm, §24 contexts", "tuning": "none for any model",
               "chat": "each model's own template, one user turn, no system message added (families.chat_text); Llama date pinned to 09 Oct 2026; Qwen and SmolLM2 templates insert their own default system text"},
    "decoding": {"greedy": True, "max_new_tokens": 256, "stop": "families.stop_ids (config stop pieces + tokenizer eos)",
                 "text": "decode(generated, skip_special_tokens=True).strip()", "dtype": "float32", "device": "cpu",
                 "threads": 8, "transformers": transformers.__version__, "torch": torch.__version__,
                 "encode": "tokenizer(chat_text, add_special_tokens=False)"},
    "measures": {
        "answer": "answer_selection.agent_answer(text): the phrase after the last 'answer:' (empty -> 'unknown'), the whole output when there is no marker (the §23 grading)",
        "marker_rate": "share of the 126 primary rows with a marker (agent_answer(text)[1])",
        "in_view_abstention": "among marked primary rows (all evidence in view by construction), share with is_refusal_answer(answer) (starts with 'unknown'); k reported",
        "reading_floor": "share of the 126 primary rows with answer_selection.d1(answer, gold) True (normalised gold string or every gold token in the answer; None counts False); the marked-only version is descriptive",
        "descriptive": {
            "unknown_not_in_view": "is_refusal_answer among marked secondary rows that are answerable and not all in view (and, separately, the unanswerable rows)",
            "note_then_refuse": "share of in-view abstentions with d1(note, gold) True (the §49 signature)",
            "refusal_phrase_count": "primary rows with utilmem.is_refusal(answer) True but is_refusal_answer False (§24 phrases)"}},
    "gate": {"rule": "a reader is queued only if all three hold", "marker_rate_min": 0.70, "reading_floor_min": 0.15,
             "in_view_abstention_min": 0.15, "in_view_abstention_k_min": 10,
             "fail_labels": {"marker": "does not follow the format; not measurable here",
                             "floor": "barely reads; an 'unknown' can't be read as an evidence-use failure",
                             "abstention": "habit absent (or below the line) at this size"},
             "queue_cap": "at most two readers to Q-A..Q-D; order: highest in-view abstention, then the family not yet covered (none approved here)"},
    "stopping": "D-45: a failed gate is documented, not rescued; no prompt tuning, no re-decoding",
    "not_approved": ["Q-A..Q-D", "any pod or API spend", "folds 2-4", "prompt tuning"],
    "timing_rule": "one full row per reader timed before any duration is quoted (measure-before-ETA)",
}
(OUT / "freeze.json").write_text(json.dumps(freeze, indent=1, ensure_ascii=False))
print(json.dumps(freeze["rows"], indent=1))
