"""§43 rev 1 B1 evidence: one Qwen 3B dev row (fold 0) read through scripts/jlens_read.read_record with the §34 Qwen 3B
lens (runs/exp34_jlens/lenses/qwen3b.pt) on the laptop (CPU fp32), before and after the BOS fix (run once per code
state). Saves every array; compare.py-style check is in the 'compare' mode. Usage:
    b1_qwen_read.py read TAG      # writes runs/exp43_build/b1_qwen_<TAG>.npz/.json
    b1_qwen_read.py compare A B   # identical ranks / top ids / top logits at every layer and position?"""
import hashlib, json, subprocess, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
OUT = Path("runs/exp43_build")
QID_FOLD = 0

if sys.argv[1] == "compare":
    a, b = (np.load(OUT / f"b1_qwen_{t}.npz") for t in sys.argv[2:4])
    ja, jb = (json.loads((OUT / f"b1_qwen_{t}.json").read_text()) for t in sys.argv[2:4])
    res = {"question_id": ja["question_id"], "keys": sorted(a.files)}
    for k in sorted(a.files):
        res[k] = {"shape": list(a[k].shape), "identical": bool(a[k].shape == b[k].shape and np.array_equal(a[k], b[k]))}
    res["measures_identical"] = ja["measures"] == jb["measures"]
    res["positions_identical"] = ja["positions"] == jb["positions"]
    res["code_before"], res["code_after"] = ja["git_head"], jb["git_head"]
    res["guard_before"], res["guard_after"] = ja["guard_src_sha"], jb["guard_src_sha"]
    res["all_identical"] = all(v["identical"] for k, v in res.items() if isinstance(v, dict)) and res["measures_identical"]
    (OUT / "b1_qwen_compare.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    sys.exit(0 if res["all_identical"] else 1)

tag = sys.argv[2]
import inspect, torch, transformers, jlens
from scripts import jlens_read as jr
from memctl.jacobian_data import load_records
torch.set_num_threads(8)
recs, _ = load_records("qwen3b", QID_FOLD)
rec = next(r for r in recs if r.answerable and r.all_in_view and r.evidence_lines and r.correct is not None)
hf = transformers.AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct", dtype=torch.float32).eval()
tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
lm = jlens.from_hf(hf, tok)
lens = jlens.JacobianLens.load("runs/exp34_jlens/lenses/qwen3b.pt")
t0 = time.perf_counter()
payload, arrays = jr.read_record(lens, lm, tok, rec, k=10)
payload["read_s"] = time.perf_counter() - t0
payload["git_head"] = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
payload["git_dirty_files"] = subprocess.run(["git", "status", "--short", "scripts/jlens_read.py", "memctl/analysis/jlens_readout.py"],
                                            capture_output=True, text=True).stdout
payload["guard_src_sha"] = hashlib.sha256(inspect.getsource(jr.read_record).encode()).hexdigest()
payload["lens_sha256"] = jr.file_sha256("runs/exp34_jlens/lenses/qwen3b.pt")
np.savez_compressed(OUT / f"b1_qwen_{tag}.npz", **arrays)
(OUT / f"b1_qwen_{tag}.json").write_text(json.dumps(payload, indent=1))
print(rec.question_id, payload["n_tokens"], payload["read_s"], payload["measures"]["logit"]["P1"]["gold"], payload["measures"]["jlens"]["P1"]["gold"])
