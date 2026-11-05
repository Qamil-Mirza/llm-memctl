import os, sys, time
from pathlib import Path
for line in Path(".env").read_text().splitlines():
    line = line.strip().removeprefix("export ")
    if line.startswith("HF_TOKEN="):
        os.environ["HF_TOKEN"] = line.split("=", 1)[1].strip().strip('"').strip("'")
from huggingface_hub import snapshot_download
for m in sys.argv[1:]:
    t = time.time()
    p = snapshot_download(m, allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja", "*.py"])
    print(m, "downloaded in %.0f s" % (time.time() - t), flush=True)
