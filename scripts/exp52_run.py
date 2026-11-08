"""§52 stage 0 generation (laptop, CPU fp32, greedy, 256 new tokens), one reader at a time, resumable.
Usage: exp52_run.py KEY --set primary|secondary   |   exp52_run.py KEY --timing (one full row, timed, not cached)
Outputs: runs/exp52/gen_{KEY}.jsonl (the cache later stages reuse), runs/exp52/timing_{KEY}.json."""
import argparse, hashlib, json, os, resource, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
for line in Path(".env").read_text().splitlines():
    line = line.strip().removeprefix("export ")
    if line.startswith("HF_TOKEN="):
        os.environ["HF_TOKEN"] = line.split("=", 1)[1].strip().strip('"').strip("'")
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from memctl import jacobian_data as jd
from memctl.analysis import families as F

ap = argparse.ArgumentParser()
ap.add_argument("key")
ap.add_argument("--set", default="primary", choices=("primary", "secondary"))
ap.add_argument("--timing", action="store_true")
ap.add_argument("--threads", type=int, default=8)
args = ap.parse_args()
OUT = Path("runs/exp52")
freeze = json.loads((OUT / "freeze.json").read_text())
fr = freeze["readers"][args.key]
cfg = F.get52(args.key)
torch.set_num_threads(args.threads)
sha = lambda s: hashlib.sha256(s.encode()).hexdigest()
rss = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024

rows = [json.loads(l) for l in (OUT / "rows52.jsonl").read_text().splitlines()]
assert hashlib.sha256((OUT / "rows52.jsonl").read_bytes()).hexdigest() == freeze["rows"]["sha256"]
recs = {r.question_id: r for f in (0, 1) for r in jd.load_family_records(F.LLAMA3B, f)[0]}
for x in rows:
    assert sha(recs[x["question_id"]].prompt) == x["prompt_sha256"], x["question_id"]

tok = AutoTokenizer.from_pretrained(cfg.model_id, revision=fr["revision"])
net = AutoModelForCausalLM.from_pretrained(cfg.model_id, revision=fr["revision"], dtype=torch.float32).eval()
gen_eos = net.generation_config.eos_token_id
stop = F.stop_ids(tok, cfg, gen_eos)
assert F.check_refusal_ids(lambda t: F.encode(tok, t), cfg) == cfg.refusal_ids


def enc_prompt(prompt):
    ids = F.encode(tok, F.chat_text(tok, cfg, prompt))
    if cfg.bos_in_chat:
        F.check_bos(tok, cfg, ids)
    return ids


def greedy(ids, max_new=256):
    with torch.no_grad():
        t0 = time.perf_counter()
        out = net(input_ids=torch.tensor([ids]), use_cache=True, logits_to_keep=1)
        tp = time.perf_counter() - t0
        cache, logits, gen, steps = out.past_key_values, out.logits[0, -1], [], []
        while True:
            nxt = int(logits.argmax())
            if nxt in stop:
                break
            gen.append(nxt)
            if len(gen) >= max_new:
                break
            t1 = time.perf_counter()
            o = net(input_ids=torch.tensor([[nxt]]), past_key_values=cache, use_cache=True, logits_to_keep=1)
            steps.append(time.perf_counter() - t1)
            logits = o.logits[0, -1]
    return gen, tp, steps


if args.timing:
    prim = [x for x in rows if x["set"] == "primary"]
    lens = {x["question_id"]: len(enc_prompt(recs[x["question_id"]].prompt)) for x in prim}
    med = sorted(lens.values())[len(lens) // 2]
    q = min(lens, key=lambda k: (abs(lens[k] - med), k))
    greedy(enc_prompt("warm up"), 2)
    t0 = time.time()
    g, tp, steps = greedy(enc_prompt(recs[q].prompt))
    res = {"key": args.key, "question_id": q, "prompt_tokens": lens[q], "median_prompt_tokens": med,
           "max_prompt_tokens": max(lens.values()), "new_tokens": len(g), "row_s": time.time() - t0,
           "prefill_s": tp, "decode_s_per_token": sum(steps) / max(1, len(steps)), "peak_rss_gb": rss(),
           "note": "timing only; this output is not the cache and is not read"}
    (OUT / f"timing_{args.key}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res), flush=True)
    sys.exit(0)

path = OUT / f"gen_{args.key}.jsonl"
done = {json.loads(l)["question_id"] for l in path.read_text().splitlines()} if path.exists() else set()
todo = [x for x in rows if x["set"] == args.set and x["question_id"] not in done]
print(f"{args.key} {args.set}: {len(todo)} to do, {len(done)} cached", flush=True)
with path.open("a") as fh:
    for i, x in enumerate(todo):
        ids = enc_prompt(recs[x["question_id"]].prompt)
        t0 = time.time()
        g, tp, steps = greedy(ids)
        text = tok.decode(g, skip_special_tokens=True).strip()
        fh.write(json.dumps({"question_id": x["question_id"], "set": x["set"], "reader": args.key,
                             "revision": fr["revision"], "prompt_tokens": len(ids), "gen_ids": g,
                             "hit_max": len(g) >= 256, "text": text, "row_s": round(time.time() - t0, 2)},
                            ensure_ascii=False) + "\n")
        fh.flush()
        print(f"{i + 1}/{len(todo)} {x['question_id']} {len(ids)}+{len(g)} tok {time.time() - t0:.1f}s rss {rss():.1f}", flush=True)
print("DONE", args.key, args.set, flush=True)
