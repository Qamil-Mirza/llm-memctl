"""Stream a bf16 HF checkpoint to an fp32 copy (same tensors, upcast exactly) in ~1 GB shards, so a fp32 load is a
zero-copy mmap and fits the laptop's 22 GB address-space guard. Usage: to_fp32.py MODEL_ID OUT_DIR"""
import json, shutil, sys
from pathlib import Path
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file
import torch
src = Path(snapshot_download(sys.argv[1], allow_patterns=["*.json", "*.safetensors"]))
out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
for f in src.glob("*.json"):
    if f.name != "model.safetensors.index.json":
        shutil.copy(f, out / f.name)
weight_map, buf, size, n = {}, {}, 0, 0
def flush():
    global buf, size, n
    if buf:
        name = f"model-fp32-{n:05d}.safetensors"; save_file(buf, str(out / name), metadata={"format": "pt"})
        for k in buf: weight_map[k] = name
        n += 1; buf, size = {}, 0
for shard in sorted(src.glob("*.safetensors")):
    with safe_open(str(shard), framework="pt") as fh:
        for k in fh.keys():
            t = fh.get_tensor(k)
            buf[k] = t.float().contiguous() if t.is_floating_point() else t.clone()
            size += buf[k].numel() * buf[k].element_size()
            if size > 1_000_000_000: flush()
flush()
total = sum((out / v).stat().st_size for v in set(weight_map.values()))
(out / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}, indent=1))
cfg = json.loads((out / "config.json").read_text()); cfg["torch_dtype"] = "float32"
if "text_config" in cfg: cfg["text_config"]["torch_dtype"] = "float32"
(out / "config.json").write_text(json.dumps(cfg, indent=1))
print("wrote", n, "shards", total / 1e9, "GB to", out)
