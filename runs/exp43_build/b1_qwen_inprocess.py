"""§43 rev 1 B1 evidence, in one process: one Qwen 3B dev row (fold 0, the first §34-population record) read with the
§34 Qwen 3B lens through read_record as committed at 9490d07 (before the fix; a verbatim copy of that file,
_jlens_read_9490d07.py) and through the fixed scripts/jlens_read.py, twice each, alternating, on one loaded model
(CPU fp32). Separate processes are not bitwise reproducible on this CPU (max |dlogit| ~1.6e-3 between two runs of the
same code), so the before/after test is in-process, with the run-to-run check (old vs old) beside it."""
import hashlib, importlib.util, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch, transformers, jlens
from scripts import jlens_read as new
from memctl.jacobian_data import load_records
spec = importlib.util.spec_from_file_location("old_jlens_read", "runs/exp43_build/_jlens_read_9490d07.py")
old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
torch.set_num_threads(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
recs, _ = load_records("qwen3b", 0)
rec = next(r for r in recs if r.answerable and r.all_in_view and r.evidence_lines and r.correct is not None)
hf = transformers.AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct", dtype=torch.float32).eval()
tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
lm = jlens.from_hf(hf, tok)
lens = jlens.JacobianLens.load("runs/exp34_jlens/lenses/qwen3b.pt")
seen = []
orig = lm.forward
lm.forward = lambda ids: seen.append(hashlib.sha256(str(ids[0].tolist()).encode()).hexdigest()) or orig(ids)
runs = {}
for name, mod in (("old1", old), ("new1", new), ("old2", old), ("new2", new)):
    runs[name] = mod.read_record(lens, lm, tok, rec, k=10)
def same(a, b):
    pa, aa = runs[a]; pb, ab = runs[b]
    out = {k: bool(np.array_equal(aa[k], ab[k])) for k in aa}
    out["measures"] = pa["measures"] == pb["measures"]
    return out
res = {"question_id": rec.question_id, "fold": rec.fold, "n_tokens": runs["new1"][0]["n_tokens"],
       "threads": torch.get_num_threads(), "forwarded_ids_sha256": seen, "same_ids_every_read": len(set(seen)) == 1,
       "old1_vs_old2": same("old1", "old2"), "new1_vs_new2": same("new1", "new2"),
       "old1_vs_new1": same("old1", "new1"), "old2_vs_new2": same("old2", "new2"),
       "n_layers_read": int(len(runs["new1"][1]["layers"])),
       "P1_gold_logit": runs["new1"][0]["measures"]["logit"]["P1"]["gold"],
       "P1_gold_jlens": runs["new1"][0]["measures"]["jlens"]["P1"]["gold"],
       "old_code": "scripts/jlens_read.py at 9490d07 (copy: runs/exp43_build/_jlens_read_9490d07.py)"}
res["before_after_identical"] = all(res["old1_vs_new1"].values()) and all(res["old2_vs_new2"].values())
res["run_to_run_identical"] = all(res["old1_vs_old2"].values()) and all(res["new1_vs_new2"].values())
Path("runs/exp43_build/b1_qwen_inprocess.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
