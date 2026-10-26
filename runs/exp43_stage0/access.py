"""§43 stage 0: HF access check (token loaded from .env, never printed)."""
import json, os, sys
from pathlib import Path
for line in Path(".env").read_text().splitlines():
    line = line.strip()
    if line.startswith("export "): line = line[7:]
    if line.startswith("HF_TOKEN="):
        os.environ["HF_TOKEN"] = line.split("=", 1)[1].strip().strip('"').strip("'")
from huggingface_hub import model_info, hf_hub_download
out = {}
for m in ["meta-llama/Llama-3.2-3B-Instruct", "google/gemma-3-4b-it", "microsoft/Phi-4-mini-instruct"]:
    r = {}
    try:
        i = model_info(m); r["model_info"] = True; r["gated"] = str(getattr(i, "gated", None))
        p = hf_hub_download(m, "config.json"); r["config"] = True
        r["config_json"] = json.loads(Path(p).read_text())
    except Exception as e:
        r["ok"] = False; r["error_class"] = type(e).__name__; r["error"] = str(e).splitlines()[0][:200]
    r.setdefault("ok", r.get("config", False))
    out[m] = r
    print(m, {k: v for k, v in r.items() if k != "config_json"})
Path("runs/exp43_stage0/access.json").write_text(json.dumps(out, indent=1))
