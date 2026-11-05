"""§43 rev 1 B3: Qwen-unchanged fingerprints, computed BEFORE the FamilyConfig change and again after (same script).
Hashes: the Qwen 3B chat text and ids of a fixed prompt (tokenizer only), HFEngine-equivalent stop ids, and a §38 stub
design run on the real §35 cache (every jsonl row minus its timing field). Usage: qwen_baseline.py OUT.json"""
import hashlib, json, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import pytest
from transformers import AutoTokenizer
from memctl.analysis import steering as st
from memctl.analysis.jacobian import wrap_chat

PROMPT = "## Memory\nUser, 2023/05/01: my cat is Luna\n\n## Current input\nQuestion: what is my cat's name?\n\nNote:"
h = lambda x: hashlib.sha256(json.dumps(x, sort_keys=True).encode()).hexdigest()
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
chat = wrap_chat(tok, PROMPT)
ids = list(tok(chat + "Note text.\nAnswer:", add_special_tokens=False).input_ids)
out = {"qwen_chat_sha": h(chat), "qwen_ids_sha": h(ids), "qwen_n_ids": len(ids),
       "qwen_stop_ids": sorted({tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id})}
from scripts import exp38_dev
d = Path("runs/exp43_build/_stub_tmp")
d.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory(dir=d) as tmp:
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(st, "ANS_GATE", 0.0)
        assert exp38_dev.main(["--run", "--backend", "stub", "--out", tmp]) == 0
    files = {}
    for p in sorted((Path(tmp) / "qwen3b").glob("*.jsonl")):
        rows = [json.loads(l) for l in p.read_text().splitlines()]
        for r in rows: r.pop("timing", None)
        files[p.name] = {"n": len(rows), "sha": h(rows)}
    for name in ("directions.json", "gate.json", "sanity.json"):
        files[name] = h(json.loads((Path(tmp) / "qwen3b" / name).read_text()))
    out["stub38"] = files
d.rmdir()
Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
