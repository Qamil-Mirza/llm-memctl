"""Collect every stage-0 output into runs/exp43_stage0/stage0.json, plus the pod-time scaling estimates."""
import json, math
from pathlib import Path
R = Path("runs/exp43_stage0")
J = lambda p: json.loads((R / p).read_text())
out = {"spec": "research/proposals/2026-10-26-exp43-cross-family.md rev 0 + amendment 1 (21e4e35)", "folds_read": [0, 1],
       "folds_2_4": "label-presence counts only (pop34.json), no outcome",
       "access_small": {m: {k: v for k, v in r.items() if k != "config_json"} for m, r in J("access.json").items()},
       "access_large": J("access_large.json"), "data": J("data.json"), "tokenizer": {}, "laptop": {},
       "pop34": J("pop34.json"), "refusal_overlap_qwen_dev": J("refusal_overlap.json"), "lens_plumbing": {}}
for f, t in J("tok.json").items():
    out["tokenizer"][f] = {k: v for k, v in t.items() if k not in ("chat_user_only",)}
for f in ("llama3b", "gemma4b", "phi4mini"):
    g = J(f"{f}/gen_date09Oct2026.json" if f == "llama3b" else f"{f}/gen.json")
    out["laptop"][f] = g
    if f == "llama3b":
        out["laptop"]["llama3b_default_date_A_only"] = J("llama3b/gen.json")["summary"]
for f in ("llama3b", "gemma4b"):
    out["lens_plumbing"][f] = J(f"{f}/lensplumb.json")
# pod-time estimates: J-lens fit (56 prompts, max_seq_len 128, dim_batch 32, A40 bf16) scaled from §34's two fits by
# a power law in d_model^2 * n_layers fitted through Qwen 3B (1,931 s) and 7B (5,419 s); reads from §34's s/question.
ref = {"qwen3b": (2048, 36, 1930.9), "qwen7b": (3584, 28, 5418.5)}
x = {k: d * d * L for k, (d, L, _) in ref.items()}
a = math.log(ref["qwen7b"][2] / ref["qwen3b"][2]) / math.log(x["qwen7b"] / x["qwen3b"])
models = {"llama3b": (3072, 28, 3.21, 6.4), "llama8b": (4096, 32, 8.03, 16.06), "gemma4b": (2560, 34, 4.30, 8.6),
          "gemma12b": (3840, 48, 12.2, 24.37), "phi4mini": (3072, 32, 3.84, 7.7), "phi4": (5120, 40, 14.7, 29.32)}
est = {"exponent": a, "basis": "fit_s = 1931 s * (d^2 L / (2048^2*36))^a, a fitted on the Qwen 3B/7B pair; reads 330 prompts at "
       "0.51 s/q (3B) .. 0.55 s/q (7B) on the A40, scaled by params/7.6B for the larger models; download at ~150 MB/s",
       "models": {}}
for m, (d, L, p, gb) in models.items():
    fit = ref["qwen3b"][2] * (d * d * L / x["qwen3b"]) ** a
    spq = 0.51 if p < 5 else 0.55 * max(1.0, p / 7.6)
    est["models"][m] = {"d_model": d, "n_layers": L, "params_b": p, "weights_gb": gb, "fit56_min": fit / 60,
                        "read330_min": 330 * spq / 60, "download_min": gb * 1000 / 150 / 60}
for fam, (s, l) in {"llama": ("llama3b", "llama8b"), "gemma": ("gemma4b", "gemma12b"), "phi": ("phi4mini", "phi4")}.items():
    e = est["models"]
    est[fam + "_pair"] = {"reads_plus_downloads_min": sum(e[k]["read330_min"] + e[k]["download_min"] for k in (s, l)),
                          "both_fits_min": e[s]["fit56_min"] + e[l]["fit56_min"]}
out["pod_estimates"] = est
(R / "stage0.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=str))
print(json.dumps(est, indent=1))
