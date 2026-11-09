"""§55 freeze (before any model output): prompt arms and shas, row lists, engine, grader version, measures, readings.
Spec: research/proposals/2026-11-08-exp55-clean-prompt.md rev 0 (9ddb074) + rev 0a (45eaee6), D-81."""
import hashlib, json, sys
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import exp55_prompts as P
from memctl import jacobian_data as jd
from memctl.analysis import families as F

OUT = ROOT / "runs/exp55"
sha = lambda s: hashlib.sha256(s.encode()).hexdigest()
fsha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
cases = [json.loads(l) for l in (ROOT / "runs/exp54/cases54.jsonl").read_text().splitlines() if l.strip()]
W = sorted([r for r in cases if r["base_index"] % 2 == 0 and r["set"] in ("ANS", "UNA-rel", "UNA-swap")],
           key=lambda r: r["case_id"])
assert Counter(r["set"] for r in W) == {"ANS": 450, "UNA-rel": 225, "UNA-swap": 75}
assert len({r["base_key"] for r in W}) == 75
rows52 = [json.loads(l) for l in (ROOT / "runs/exp52/rows52.jsonl").read_text().splitlines()]
rec = {r.question_id: r for f in (0, 1) for r in jd.load_family_records(F.LLAMA3B, f)[0]}
q3 = {r.question_id: r for f in (0, 1) for r in jd.load_records("qwen3b", f)[0]}
wl, ll = [], []
for r in W:
    ps = {a: P.build(r["prompt"], a) for a in P.ARMS}
    assert ps["P0"] == r["prompt"] and ps["P2"] == r["prompt_b3"]
    wl.append({"case_id": r["case_id"], "set": r["set"], "base_key": r["base_key"], "task": r["task"],
               **{f"sha_{a}": sha(p) for a, p in ps.items()}})
for x in rows52:
    q = x["question_id"]
    p0 = rec[q].prompt
    assert q3[q].prompt == p0 and sha(p0) == x["prompt_sha256"]
    ps = {a: P.build(p0, a) for a in P.ARMS}
    ll.append({"question_id": q, "set52": x["set"], "answerable": x["answerable"], "all_in_view": x["all_in_view"],
               "gold": x["gold"], **{f"sha_{a}": sha(p) for a, p in ps.items()}})
(OUT / "rows_W.jsonl").write_text("".join(json.dumps(x) + "\n" for x in wl))
(OUT / "rows_LME.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in ll))
cat = lambda xs, a: sha("".join(x[f"sha_{a}"] for x in xs))
fz = {
 "experiment": "§55 clean-prompt check: P0 frozen vs P1 consistent vs P2 no-option, Qwen 2.5 3B, laptop fp32, $0",
 "spec": "research/proposals/2026-11-08-exp55-clean-prompt.md rev 0 (9ddb074) + rev 0a (45eaee6); D-81",
 "frozen_before_any_model_output": True,
 "prompts": {"builder": "scripts/exp55_prompts.py", "builder_sha256": fsha(ROOT / "scripts/exp55_prompts.py"),
             "P0": {"task": P.TASK0, "note": P.NOTE0, "source": "7b5fc30, byte-identical"},
             "P1": {"task": P.TASK1, "note": P.NOTE1},
             "P2": {"task": "(removed)", "note": "(removed)", "equals": "§51 prompt_b3 (asserted on all 750 W rows)"},
             "check": "exact replacement; the only edits are the two sentences (asserted per row)",
             "concat_sha256": {"W": {a: cat(wl, a) for a in P.ARMS}, "LME": {a: cat(ll, a) for a in P.ARMS}}},
 "data": {
   "W": {"file": "runs/exp55/rows_W.jsonl", "sha256": fsha(OUT / "rows_W.jsonl"), "source": "runs/exp54/cases54.jsonl (sha256 " + fsha(ROOT / "runs/exp54/cases54.jsonl")[:16] + "…), seed 55",
         "rule": "base_index even (75 bases: 25 per task), sets ANS 450 / UNA-rel 225 / UNA-swap 75, no CF"},
   "LME": {"file": "runs/exp55/rows_LME.jsonl", "sha256": fsha(OUT / "rows_LME.jsonl"),
           "rows": "the 200 §52 rows (folds 0-1): 126 answerable all-in-view, 62 answerable not all in view, 12 unanswerable",
           "prompt_check": "Qwen 3B exp23 record prompt == §52 prompt on all 200 rows (asserted)"},
   "sealed": "seed 56 not generated; no held-out split read"},
 "engine": {"name": "scripts/exp51_engine.py EN.load('hf'), Qwen/Qwen2.5-3B-Instruct snapshot aa8e7253…, fp32 CPU, 16 threads, batched at 64",
            "free_generation": "mode total, greedy, max_new 256 (rev 0a ruling 1)",
            "W_P0_ANS": "reuse §54 B0 (runs/exp54/run/b0/W, max_new 160, 0 of 900 rows hit the limit, max 100 new tokens); a 20-row rerun at 256 must be byte-identical (text and gen_ids), else stop and report before the reuse",
            "W_P0_UNA": "new, mode total, 256",
            "LME": "P0, P1, P2 all rerun on this engine, 256 (rev 0a ruling 2); §23 vLLM bf16 cache agreement is descriptive",
            "controlled_prefix": "W only, P1 and P2: chat_ids(prompt_Pk) + B0 gen_ids[:j] (P0 note through the designated marker), mode prefix, 16 new tokens (MAX_NEW_RELEASE), graded head + tail; P2 = CM.b3_prefix; P0 prefix arm = B0 itself; rows without P_ans are NM"},
 "grader": {"W": "memctl/analysis/grade51.py classify via exp51_common.grade", "grade51_sha256": fsha(ROOT / "memctl/analysis/grade51.py"),
            "exp51_common_sha256": fsha(ROOT / "scripts/exp51_common.py"),
            "LME": "rev 0a ruling 3: answer = answer_selection.agent_answer(text) (whole output when no marker); FA = starts with 'unknown'; Gold proxy = d1(answer, gold); Uns = not 'unknown' on the 12 unanswerable rows; marker rate per arm; descriptive only, no U(λ)"},
 "measures_W": {
   "FA": "ANS rows with cls 'abstention' (base-weighted: mean over bases of the per-base share; raw counts beside)",
   "Gold": "ANS rows with cls 'gold'",
   "Uns": "UNA-rel + UNA-swap rows with cls in grade51.UNSUPPORTED (wrong, conflicting, other, mixed)",
   "Dam": "controlled prefix only: ANS rows with B0 cls 'gold' and a P_ans whose Pk prefix cls is not 'gold'",
   "U": "grade51.base_mixture, λ = 1 (0.5 ANS utility + 0.5 UNA utility per base)",
   "note_then_refuse": "P1 abstentions whose note (text before the last marker) contains the gold: answer_selection.d1(note, gold); descriptive",
   "marker_rate": "per arm, descriptive",
   "inference": "paired base-cluster bootstrap stratified by task, stats51.bootstrap, B 10,000, seed 5102, 95% two-sided intervals"},
 "measures_LME": {"inference": "paired bootstrap over questions, B 10,000, seed 5102, 95% intervals; descriptive only"},
 "readings": {
   "R1": {"compare": "FA(P1) vs FA(P0), W free generation",
          "mostly": "FA(P1) <= 1/3 FA(P0): 'the false abstention on controlled tasks is mostly created by the inconsistent wording'",
          "not_main": "FA(P1) >= 2/3 FA(P0): 'the inconsistency is not the main cause'", "else": "'partly'"},
   "R2": {"compare": "P2 vs P1, W free generation; FA, Gold, Uns and U(λ=1) reported together",
          "price": "FA(P2) <= 1/3 FA(P1) and the 95% lower bound of Uns(P2) − Uns(P1) > 0: 'the false abstention is largely the price of having an abstention option'",
          "rules": "no reading uses FA alone; removing the option is never called a fix unless Uns(P2) stays within 0.05 of Uns(P1)"},
   "LME": "the same quantities, descriptive only (12 unanswerable rows are too few for a claim)",
   "consequences": "as decided in the proposal ('What each outcome changes')"},
 "timing": "one W row and one LME row timed (on the engine, batch 1) before any duration is quoted; timing outputs discarded",
 "budget": "if the measured total exceeds 24 h: drop LME P2 first, then W UNA-swap; tell the Advisor before dropping",
 "order": ["W P0 20-row rerun check", "W free: P0 UNA, P1, P2", "W controlled prefix: P1, P2", "LME: P0, P1, P2"],
 "stopping": "D-45: a failed check is documented, not rescued; no prompt tuned after outputs",
 "hard_rules": ["laptop only, $0, no pod, no API", "dev data only", "seed 56 sealed"],
}
(OUT / "freeze.json").write_text(json.dumps(fz, indent=1, ensure_ascii=False))
print(json.dumps(fz["prompts"]["concat_sha256"], indent=1))
