"""§43 amendment 1 item 1: HF access for the large partners (config fetch only; no weights)."""
import json, os
from pathlib import Path
for line in Path(".env").read_text().splitlines():
    line = line.strip().removeprefix("export ")
    if line.startswith("HF_TOKEN="):
        os.environ["HF_TOKEN"] = line.split("=", 1)[1].strip().strip('"').strip("'")
from huggingface_hub import model_info, hf_hub_download
out = {}
for m in ["meta-llama/Llama-3.1-8B-Instruct", "google/gemma-3-12b-it", "microsoft/phi-4"]:
    r = {}
    try:
        i = model_info(m, files_metadata=True); r["gated"] = str(getattr(i, "gated", None))
        r["weights_gb"] = round(sum((s.size or 0) for s in i.siblings if s.rfilename.endswith(".safetensors")) / 1e9, 2)
        c = json.loads(Path(hf_hub_download(m, "config.json")).read_text()); t = c.get("text_config", c)
        r.update(ok=True, arch=c.get("architectures"), n_layers=t.get("num_hidden_layers"), hidden=t.get("hidden_size"),
                 vocab=t.get("vocab_size"), tie=t.get("tie_word_embeddings", c.get("tie_word_embeddings")))
    except Exception as e:
        r.update(ok=False, error_class=type(e).__name__, error=str(e).splitlines()[0][:200])
    out[m] = r; print(m, r)
Path("runs/exp43_stage0/access_large.json").write_text(json.dumps(out, indent=1))
