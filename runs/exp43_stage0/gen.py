"""§43 stage 0, one family on the laptop (CPU fp32, transformers): load, layer path, timing at ~1,100 tokens, and the
5-row regeneration check (A arm: cached note + 'Answer:', 16 tokens, stop at newline/EOS; N arm: the §23 prompt, 256
tokens) against the §24 cache (vLLM bf16). Folds 0-1 only. Usage: gen.py FAMILY [--date-string "09 Oct 2026"]"""
import argparse, json, os, resource, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
for line in Path(".env").read_text().splitlines():
    line = line.strip().removeprefix("export ")
    if line.startswith("HF_TOKEN="):
        os.environ["HF_TOKEN"] = line.split("=", 1)[1].strip().strip('"').strip("'")
import torch, transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from memctl.analysis import answer_selection as sel
import data as D
from tok import MODELS

ap = argparse.ArgumentParser()
ap.add_argument("family")
ap.add_argument("--date-string", default=None, help="Llama template date (vLLM rendered the pod's date in §24)")
ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--n-rows", type=int, default=5)
ap.add_argument("--skip-n", action="store_true")
ap.add_argument("--weights", default=None, help="local fp32 copy of the checkpoint (to_fp32.py)")
ap.add_argument("--upcast", action="store_true", help="load bf16 (the checkpoint dtype), then upcast each parameter to fp32 in place: same fp32 weights, lower peak address space")
args = ap.parse_args()
torch.set_num_threads(args.threads)
fam = args.family
OUT = Path("runs/exp43_stage0") / fam
OUT.mkdir(parents=True, exist_ok=True)
rss = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024  # GB
def vmpeak():
    for l in open("/proc/self/status"):
        if l.startswith("VmPeak"): return int(l.split()[1]) / 1024 / 1024
res_env = {k: os.environ.get(k) for k in ("MALLOC_ARENA_MAX", "HF_ENABLE_PARALLEL_LOADING", "OMP_NUM_THREADS")}
res = {"family": fam, "model": MODELS[fam], "transformers": transformers.__version__, "torch": torch.__version__,
       "threads": args.threads, "date_string": args.date_string}

tok = AutoTokenizer.from_pretrained(MODELS[fam])
kw = {"date_string": args.date_string} if args.date_string else {}
def chat(prompt):
    return tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, **kw)
def ids(t): return list(tok(t, add_special_tokens=False).input_ids)

# rows: Qwen dev rows marked in the family; family refusals first (population, folds 0-1), then answers
d, qdev = D.qwen_dev()
recs = {r.question_id: r for f in D.FOLDS for r in D.load_records24(fam, f)[0]}
cand = [recs[q] for q in qdev if q in recs and sel.split_note(recs[q].output)[0] is not None]
isref = lambda r: sel.is_refusal_answer(sel.agent_answer(r.output)[0])
lens = {r.question_id: len(ids(chat(r.prompt))) + len(ids(sel.frozen_continuation("A", sel.split_note(r.output)[0]))) for r in cand}
refs = sorted([r for r in cand if isref(r)], key=lambda r: abs(lens[r.question_id] - 1100))[:3]
answers = sorted([r for r in cand if not isref(r)], key=lambda r: abs(lens[r.question_id] - 1100))[:args.n_rows - len(refs)]
rows = sorted(refs + answers, key=lambda r: r.question_id)
res["rows"] = [{"question_id": r.question_id, "fold": r.fold, "cached_refusal": isref(r), "a_tokens": lens[r.question_id]} for r in rows]
print("rows", res["rows"], flush=True)

t0 = time.time()
if args.upcast:
    net = AutoModelForCausalLM.from_pretrained(MODELS[fam], dtype=torch.bfloat16).eval()
    res["vmpeak_gb_after_bf16_load"] = vmpeak()
    with torch.no_grad():
        for mod in net.modules():
            for name, prm in list(mod.named_parameters(recurse=False)):
                prm.data = prm.data.float()
            for name, buf in list(mod.named_buffers(recurse=False)):
                if buf.is_floating_point():
                    setattr(mod, name, buf.float()) if name in mod._buffers else None
    net.config.dtype = torch.float32
else:
    net = AutoModelForCausalLM.from_pretrained(args.weights or MODELS[fam], dtype=torch.float32).eval()
res["weights"] = args.weights
res["load_mode"] = "bf16 load + per-parameter fp32 upcast" if args.upcast else "fp32 load"
res["load_s"] = time.time() - t0
res["peak_rss_gb_after_load"] = rss(); res["vmpeak_gb_after_load"] = vmpeak(); res["env"] = res_env
res["model_class"] = type(net).__name__
layer_paths = [n for n, m in net.named_modules() if n.endswith("layers") and isinstance(m, torch.nn.ModuleList)]
res["layer_paths"] = layer_paths
lp = [p for p in layer_paths if "vision" not in p and "siglip" not in p][0]
layers = net.get_submodule(lp)
res["layer_path"], res["n_layers"] = lp, len(layers)
cfg = getattr(net.config, "text_config", net.config)
res["hidden"] = cfg.hidden_size
res["n_params_total"] = sum(p.numel() for p in net.parameters())
res["param_dtypes"] = sorted({str(p.dtype) for p in net.parameters()})
gen_eos = net.generation_config.eos_token_id
gen_eos = gen_eos if isinstance(gen_eos, list) else [gen_eos]
stop = set(x for x in gen_eos + [tok.eos_token_id] if x is not None)
for t in ("<|im_end|>", "<end_of_turn>", "<|eot_id|>", "<|end|>"):
    i = tok.convert_tokens_to_ids(t)
    if isinstance(i, int) and i != tok.unk_token_id and i is not None:
        stop.add(i)
res["stop_ids"] = sorted(stop)
res["stop_pieces"] = tok.convert_ids_to_tokens(sorted(stop))
print({k: res[k] for k in ("load_s", "peak_rss_gb_after_load", "model_class", "layer_paths", "n_layers", "hidden", "stop_pieces")}, flush=True)

# decoder-layer output format
fmt = {}
def _fmt_hook(m, i, o):
    fmt.setdefault("t", type(o).__name__ + (f" len {len(o)}" if isinstance(o, tuple) else ""))
    return None
h = layers[len(layers) // 2].register_forward_hook(_fmt_hook)
with torch.no_grad():
    net(input_ids=torch.tensor([ids(chat("hi"))]), use_cache=False, logits_to_keep=1)
h.remove()
res["decoder_layer_output"] = fmt.get("t")


def greedy(all_ids, max_new, stop_newline):
    with torch.no_grad():
        t0 = time.perf_counter()
        out = net(input_ids=torch.tensor([all_ids]), use_cache=True, logits_to_keep=1)
        tp = time.perf_counter() - t0
        cache, logits, gen, steps = out.past_key_values, out.logits[0, -1], [], []
        while True:
            nxt = int(logits.argmax())
            if nxt in stop: break
            gen.append(nxt)
            if (stop_newline and "\n" in tok.decode(gen)) or len(gen) >= max_new: break
            t1 = time.perf_counter()
            o = net(input_ids=torch.tensor([[nxt]]), past_key_values=cache, use_cache=True, logits_to_keep=1)
            steps.append(time.perf_counter() - t1)
            logits = o.logits[0, -1]
    return gen, tp, steps


# timing: the row nearest 1,100 A-arm tokens, prefill + 16 forced decode steps (no stop)
tr = min(rows, key=lambda r: abs(lens[r.question_id] - 1100))
note = sel.split_note(tr.output)[0]
tids = ids(chat(tr.prompt)) + ids(sel.frozen_continuation("A", note))
saved = set(stop); stop.clear()
greedy(ids(chat("warm up")), 2, False)
g, tp, steps = greedy(tids, 16, False)
stop |= saved
res["timing"] = {"question_id": tr.question_id, "n_prompt_tokens": len(tids), "prefill_s": tp,
                 "prefill_ms_per_token": 1000 * tp / len(tids), "decode_steps": len(steps),
                 "decode_s_per_token_mean": sum(steps) / len(steps), "decode_s_per_token_max": max(steps)}
res["peak_rss_gb_after_timing"] = rss()
print("timing", res["timing"], "rss", res["peak_rss_gb_after_timing"], flush=True)


def first_after(cont, text):
    a, b = ids(cont), ids(cont + text)
    return b[len(a)] if b[:len(a)] == a and len(b) > len(a) else None


checks = []
for r in rows:
    note, cached_ans = sel.split_note(r.output)
    cont = sel.frozen_continuation("A", note)
    chat_ids = ids(chat(r.prompt))
    a_ids = chat_ids + ids(cont)
    g, tp, _ = greedy(a_ids, 16, True)
    gtext = tok.decode(g, skip_special_tokens=True)
    a_ans, _ = sel.agent_answer(cont + gtext)
    c_first = first_after(cont, cached_ans)
    row = {"question_id": r.question_id, "fold": r.fold, "last_prompt_piece": tok.convert_ids_to_tokens([a_ids[-1]])[0],
           "A": {"generated": gtext, "first_id": g[0] if g else None, "first_piece": tok.convert_ids_to_tokens(g[:1]),
                 "cached_first_id": c_first, "cached_first_piece": tok.convert_ids_to_tokens([c_first]) if c_first is not None else None,
                 "first_match": bool(g) and g[0] == c_first,
                 "refusal": sel.is_refusal_answer(a_ans), "cached_refusal": isref(r),
                 "refusal_match": sel.is_refusal_answer(a_ans) == isref(r),
                 "phrase_match": a_ans.strip() == sel.agent_answer(r.output)[0].strip(),
                 "cached_answer": sel.agent_answer(r.output)[0]}}
    print(r.question_id, "A", row["A"], flush=True)
    if not args.skip_n:
        g, tp, steps = greedy(chat_ids, 256, False)
        ntext = tok.decode(g, skip_special_tokens=True).strip()
        cached_first = ids(chat(r.prompt) + r.output)[len(chat_ids)]
        # first id of the regenerated text after strip, as the cache stored it
        n_first = ids(chat(r.prompt) + ntext)[len(chat_ids)] if ntext else None
        n_ans, n_marker = sel.agent_answer(ntext)
        row["N"] = {"generated": ntext, "n_new": len(g), "first_id": n_first, "cached_first_id": cached_first,
                    "first_match": n_first == cached_first, "marker": n_marker, "cached_marker": True,
                    "marker_match": n_marker is True, "refusal": sel.is_refusal_answer(n_ans),
                    "refusal_match": sel.is_refusal_answer(n_ans) == isref(r), "exact_match": ntext == r.output.strip(),
                    "answer": n_ans, "cached_answer": sel.agent_answer(r.output)[0],
                    "decode_s_per_token": (sum(steps) / len(steps)) if steps else None}
        print(r.question_id, "N", {k: v for k, v in row["N"].items() if k != "generated"}, flush=True)
    checks.append(row)
res["checks"] = checks
def agg(arm, key): 
    xs = [c[arm][key] for c in checks if arm in c]
    return {"k": sum(xs), "n": len(xs)}
res["summary"] = {"A_first_match": agg("A", "first_match"), "A_refusal_match": agg("A", "refusal_match"),
                  "A_phrase_match": agg("A", "phrase_match"), "last_token_colon": sum(c["last_prompt_piece"].endswith(":") for c in checks)}
if not args.skip_n:
    res["summary"].update({"N_first_match": agg("N", "first_match"), "N_marker_match": agg("N", "marker_match"),
                           "N_refusal_match": agg("N", "refusal_match"), "N_exact_match": agg("N", "exact_match")})
s = res["summary"]
res["d22_pass_A"] = s["A_first_match"]["k"] >= 4 and s["A_refusal_match"]["k"] >= 4
if not args.skip_n:
    res["d22_pass_N"] = s["N_first_match"]["k"] >= 4 and s["N_marker_match"]["k"] >= 4
res["peak_rss_gb"] = rss(); res["vmpeak_gb"] = vmpeak()
res["wall_s"] = time.time() - t0
print("SUMMARY", s, "pass A", res["d22_pass_A"], "pass N", res.get("d22_pass_N"), "peak", res["peak_rss_gb"], flush=True)
tag = f"_date{args.date_string.replace(' ', '')}" if args.date_string else ""
(OUT / f"gen{tag}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
