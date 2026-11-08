"""§52 pipeline check (Advisor request, not a rescue; no gate changes): the Qwen 0.5B and 1.5B prompt ids are equal on
all 126 primary rows, and equal the Qwen 2.5 3B prompt (§23 exp23 cache prompt string, encoded with the 3B tokenizer
and the QWEN3B config) for the same qids. Tokenizers only. Writes runs/exp52/pipecheck.json."""
import hashlib, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from transformers import AutoTokenizer
from memctl import jacobian_data as jd
from memctl.analysis import families as F

OUT = Path("runs/exp52")
fz = json.loads((OUT / "freeze.json").read_text())
rows = [json.loads(l) for l in (OUT / "rows52.jsonl").read_text().splitlines()]
prim = [x["question_id"] for x in rows if x["set"] == "primary"]
recs = {r.question_id: r for f in (0, 1) for r in jd.load_family_records(F.LLAMA3B, f)[0]}
q3 = {r.question_id: r for f in (0, 1) for r in jd.load_records("qwen3b", f)[0]}
toks = {k: AutoTokenizer.from_pretrained(F.get52(k).model_id, revision=fz["readers"][k]["revision"]) for k in ("qwen05b", "qwen15b")}
t3 = AutoTokenizer.from_pretrained(F.QWEN3B.model_id)
gens = {k: {g["question_id"]: g for g in map(json.loads, (OUT / f"gen_{k}.jsonl").read_text().splitlines())} for k in toks}
res = {"n": len(prim), "ids_05_eq_15": 0, "chat_05_eq_15": 0, "stored_len_eq": 0, "qwen3b_present": 0,
       "prompt_eq_qwen3b_record": 0, "chat_eq_qwen3b": 0, "ids_eq_qwen3b": 0,
       "vocab_equal_05_15": toks["qwen05b"].get_vocab() == toks["qwen15b"].get_vocab(),
       "vocab_equal_05_3b": toks["qwen05b"].get_vocab() == t3.get_vocab(),
       "template_equal_05_15_3b": len({toks["qwen05b"].chat_template, toks["qwen15b"].chat_template, t3.chat_template}) == 1,
       "configs_stamp_05_15": [F.QWEN05B.stamp(), F.QWEN15B.stamp()], "mismatches": []}
for q in prim:
    p = recs[q].prompt
    c = {k: F.chat_text(t, F.get52(k), p) for k, t in toks.items()}
    i = {k: F.encode(t, c[k]) for k, t in toks.items()}
    res["chat_05_eq_15"] += c["qwen05b"] == c["qwen15b"]
    res["ids_05_eq_15"] += i["qwen05b"] == i["qwen15b"]
    res["stored_len_eq"] += all(gens[k][q]["prompt_tokens"] == len(i[k]) for k in toks)
    if q in q3:
        res["qwen3b_present"] += 1
        res["prompt_eq_qwen3b_record"] += q3[q].prompt == p
        c3 = F.chat_text(t3, F.QWEN3B, q3[q].prompt)
        res["chat_eq_qwen3b"] += c3 == c["qwen05b"]
        res["ids_eq_qwen3b"] += F.encode(t3, c3) == i["qwen05b"]
    if i["qwen05b"] != i["qwen15b"]:
        res["mismatches"].append(q)
(OUT / "pipecheck.json").write_text(json.dumps(res, indent=1))
print(json.dumps({k: v for k, v in res.items() if k != "configs_stamp_05_15"}, indent=1))
