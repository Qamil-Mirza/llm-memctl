"""§52 stage 0, tokenizer/config side (no weights, no model output): pinned HF revisions, chat-template fit on one
user turn, BOS, stop pieces, ' unknown' / ' Unknown' first ids after 'Answer:', layer count and width.
HF_TOKEN read from .env, never printed. Writes runs/exp52/probe.json."""
import json, os, sys, hashlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
for line in Path(".env").read_text().splitlines():
    line = line.strip().removeprefix("export ")
    if line.startswith("HF_TOKEN="):
        os.environ["HF_TOKEN"] = line.split("=", 1)[1].strip().strip('"').strip("'")
from huggingface_hub import HfApi
from transformers import AutoTokenizer, AutoConfig
from memctl.analysis import families as F

MODELS = {"qwen05b": "Qwen/Qwen2.5-0.5B-Instruct", "qwen15b": "Qwen/Qwen2.5-1.5B-Instruct",
          "llama1b": "meta-llama/Llama-3.2-1B-Instruct", "gemma1b": "google/gemma-3-1b-it",
          "gemma270m": "google/gemma-3-270m-it", "smol17b": "HuggingFaceTB/SmolLM2-1.7B-Instruct"}
P = "<<PROMPT>>"
api = HfApi()
out = {}
for key, mid in MODELS.items():
    info = api.model_info(mid)
    rev = info.sha
    tok = AutoTokenizer.from_pretrained(mid, revision=rev)
    cfg = AutoConfig.from_pretrained(mid, revision=rev)
    tc = getattr(cfg, "text_config", cfg)
    kw = {"date_string": F.LLAMA_DATE} if key.startswith("llama") else {}
    chat = tok.apply_chat_template([{"role": "user", "content": P}], tokenize=False, add_generation_prompt=True, **kw)
    enc = lambda t: list(tok(t, add_special_tokens=False).input_ids)
    probe = F.REFUSAL_PROBE
    ref = []
    for t in F.REFUSAL_TEXTS:
        a, b = enc(probe), enc(probe + t)
        ref.append(b[len(a)] if b[:len(a)] == a else None)
    cands = ["<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<|end_of_text|>", "<|eom_id|>", "<end_of_turn>", "<eos>"]
    vocab = tok.get_vocab()
    out[key] = {"model_id": mid, "revision": rev, "gated": info.gated, "arch": cfg.architectures,
                "model_type": cfg.model_type, "n_layers": tc.num_hidden_layers, "hidden": tc.hidden_size,
                "n_heads": tc.num_attention_heads, "head_dim": getattr(tc, "head_dim", None),
                "max_pos": getattr(tc, "max_position_embeddings", None), "tie_word_embeddings": getattr(cfg, "tie_word_embeddings", None),
                "torch_dtype": str(getattr(cfg, "torch_dtype", None) or getattr(cfg, "dtype", None)),
                "chat_head": chat.split(P)[0], "chat_tail": chat.split(P)[1], "template_kwargs": kw,
                "chat_template_sha256": hashlib.sha256((tok.chat_template or "").encode()).hexdigest(),
                "bos": tok.bos_token, "eos": tok.eos_token, "eos_id": tok.eos_token_id,
                "chat_starts_with_bos": bool(tok.bos_token) and chat.startswith(tok.bos_token),
                "stop_pieces_in_vocab": [c for c in cands if c in vocab],
                "refusal_ids": ref, "refusal_pieces": tok.convert_ids_to_tokens([i for i in ref if i is not None]),
                "colon_last": tok.convert_ids_to_tokens(enc("Some note.\nAnswer:")[-1:])}
    try:
        from huggingface_hub import hf_hub_download
        g = json.loads(Path(hf_hub_download(mid, "generation_config.json", revision=rev)).read_text())
        out[key]["generation_eos"] = g.get("eos_token_id")
    except Exception as e:
        out[key]["generation_eos"] = f"none ({type(e).__name__})"
    print(key, json.dumps({k: v for k, v in out[key].items() if k not in ("chat_head",)}), flush=True)
    print("  head:", repr(out[key]["chat_head"][:300]), flush=True)
Path("runs/exp52/probe.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
