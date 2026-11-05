"""§43 amendment 1 item 5: laptop logit-lens read on 5 dev rows (folds 0-1) through scripts/jlens_read.read_record,
unchanged, with an identity 'lens' (transport = identity, so the jlens rows equal the logit rows; no J-lens is
fitted). Plumbing only, no claims. Usage: lensplumb.py FAMILY [--force-bos default|off]"""
import argparse, json, os, sys, time, resource
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
for line in Path(".env").read_text().splitlines():
    line = line.strip().removeprefix("export ")
    if line.startswith("HF_TOKEN="):
        os.environ["HF_TOKEN"] = line.split("=", 1)[1].strip().strip('"').strip("'")
import torch, transformers, jlens
from scripts.jlens_read import read_record
from memctl.analysis.jlens_readout import readout_for
import data as D
from tok import MODELS

ap = argparse.ArgumentParser(); ap.add_argument("family"); ap.add_argument("--date-string", default=None); ap.add_argument("--weights", default=None)
args = ap.parse_args()
torch.set_num_threads(8)
fam = args.family
recs = [r for f in D.FOLDS for r in D.load_records24(fam, f)[0]]
pop = [r for r in D.population(recs)][:5]
hf = transformers.AutoModelForCausalLM.from_pretrained(args.weights or MODELS[fam], dtype=torch.float32).eval()
def make_tok():
    tok = transformers.AutoTokenizer.from_pretrained(MODELS[fam])
    if args.date_string:   # Llama: pin the template date as vLLM rendered it in §24
        orig = tok.apply_chat_template
        tok.apply_chat_template = lambda *a, **k: orig(*a, **{"date_string": args.date_string, **k})
    return tok
res = {"family": fam, "rows": [], "date_string": args.date_string}
attempts = {}
for force_bos in (True, False, "patched_encode"):
    tok = make_tok()
    lm = jlens.from_hf(hf, tok, force_bos=force_bos is True)
    if force_bos == "patched_encode":   # the fix a per-family config needs: the guard encodes like the readout
        def _enc(text, *, max_length=512, _lm=lm, _tok=tok):
            e = _tok(text, return_tensors="pt", truncation=True, max_length=max_length, add_special_tokens=False)
            return e.input_ids.to(_lm.input_device)
        lm.encode = _enc
    res.setdefault("layout", lm.layout.path); res["n_layers"], res["d_model"] = lm.n_layers, lm.d_model
    res["softcap"] = lm._logit_softcap
    class IdentityLens:
        source_layers = list(range(lm.n_layers - 1)); n_prompts = 0
        def transport(self, h, l): return h
    ok = []
    for r in pop:
        t0 = time.perf_counter()
        try:
            payload, arrays = read_record(IdentityLens(), lm, tok, r, k=10)
            m = payload["measures"]["logit"]["P1"]
            ro = readout_for(tok, r.prompt, r.memory_lines, r.evidence_lines, r.gold)
            row = {"question_id": r.question_id, "ok": True, "n_tokens": payload["n_tokens"], "n_leading_ws": payload["n_leading_ws"],
                   "gold_token_text": payload["gold_token_text"], "tracked": payload["tracked"],
                   "P1_minus_P0": payload["positions"]["P1"][0] - payload["positions"]["P0"][0],
                   "P1_piece": tok.convert_ids_to_tokens([ro.input_ids[ro.positions["P1"][0]]])[0],
                   "P1_raw_piece": tok.convert_ids_to_tokens([ro.input_ids[ro.positions["P1_raw"][0]]])[0],
                   "logit_P1_gold_outcome": m["gold"]["outcome"], "logit_P1_gold_emergence": m["gold"]["emergence_depth"],
                   "logit_P1_unknown_sp_final_rank": m["unknown_sp"]["final_rank"],
                   "n_layers_read": len(arrays["layers"]), "s": time.perf_counter() - t0}
        except Exception as e:
            row = {"question_id": r.question_id, "ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}"}
        ok.append(row); print(force_bos, row, flush=True)
    attempts[f"force_bos={force_bos}"] = ok
res["attempts"] = attempts
res["peak_rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
Path(f"runs/exp43_stage0/{fam}").mkdir(parents=True, exist_ok=True)
Path(f"runs/exp43_stage0/{fam}/lensplumb.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
