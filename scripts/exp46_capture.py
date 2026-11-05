#!/usr/bin/env python
"""§46 measurement A input: residuals at P_last and the task's sites, every layer, one forward per prompt (no
generation; outcomes are reused from §45 stage 1). Resumable per prompt.

    HF_HUB_OFFLINE=1 .venv/bin/python scripts/exp46_capture.py --device cpu --dtype float32      # laptop default
    python scripts/exp46_capture.py --device cuda --dtype bfloat16 --deadline-s 3000              # pod
    .venv/bin/python scripts/exp46_capture.py --limit 3 --out runs/exp46_probe/smoke              # smoke

Spec: research/proposals/2026-10-30-exp46-instrument-check.md §2-§3 (rev 0, e70d03f). Cases: RC and TU from
runs/exp45_maps/cases.jsonl (576 prompts), dev half first. The chat text must hash to the §45 stage-1 chat_sha256 of
the same case, else the case is refused (exit 6). Output: <out>/<case>.npz (resid [site, layer, d_model] float32,
sites, positions, input_ids) and <case>.json (case meta, §45 outcome, stamps); _timing.json. Exit 6: chat sha
mismatch. 9: stopped at the deadline (prompts so far kept).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from memctl.analysis import maps as M  # noqa: E402
from memctl.analysis import probe46 as P  # noqa: E402

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
CAPTURE_VERSION = "exp46_capture_v0"
CASES = Path("runs/exp45_maps/cases.jsonl")
STAGE1 = Path("runs/exp45_maps/j56-jrefit-logit/stage1")
SITES = {"RC": ("P_last", "P_B", "P_q"), "TU": ("P_last", "P_cur", "P_old")}
EXIT_SHA, EXIT_DEADLINE = 6, 9


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, path)


def load_cases(path: Path = CASES) -> list[dict]:
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r["task"] in P.TASKS]
    return sorted(rows, key=lambda c: (c["split"] != "dev", c["task"], c["base_index"], c["variant"], c["format"],
                                       c["hardness"]))


def stage1_record(case_id: str, stage1: Path = STAGE1) -> dict:
    return json.loads((stage1 / f"{case_id}.json").read_text())


def done(out: Path, case_id: str, stamp: dict) -> bool:
    try:
        meta = json.loads((out / f"{case_id}.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    return (out / f"{case_id}.npz").exists() and all(meta.get(k) == v for k, v in stamp.items())


def capture(hf, tok, chat: str, pos: dict[str, int], sites: tuple[str, ...], device,
            n_layers: int = P.N_LAYERS) -> tuple[np.ndarray, list[int]]:
    import torch

    ids = tok(chat, add_special_tokens=False).input_ids
    with torch.no_grad():
        out = hf(torch.tensor([ids], device=device), output_hidden_states=True, use_cache=False)
    hs = out.hidden_states[1:]                     # output of decoder layer l = hidden_states[l + 1]
    if len(hs) != n_layers:
        raise RuntimeError(f"{len(hs)} layers, expected {n_layers}")
    idx = [pos[s] for s in sites]
    resid = torch.stack([h[0, idx, :] for h in hs], dim=1)     # [site, layer, d]
    return resid.float().cpu().numpy(), ids


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=Path, default=CASES)
    ap.add_argument("--stage1", type=Path, default=STAGE1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--every", type=int, default=None, help="every n-th case (the stub chain's spread sample)")
    ap.add_argument("--split", choices=("dev", "held", "all"), default="all")
    ap.add_argument("--deadline-s", type=float, default=None)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--weights", default=None, help="model weights dir (the pod stub); tokenizer stays the 3B's")
    ap.add_argument("--n-layers", type=int, default=P.N_LAYERS)
    args = ap.parse_args()

    import torch
    import transformers
    from memctl.analysis.jacobian import wrap_chat

    if args.threads:
        torch.set_num_threads(args.threads)
    out = args.out or Path("runs/exp46_probe") / f"{args.device}-{args.dtype}"
    out.mkdir(parents=True, exist_ok=True)
    cases = load_cases(args.cases)
    if args.split != "all":
        cases = [c for c in cases if c["split"] == args.split]
    if args.every:
        cases = cases[:: args.every]
    if args.limit:
        cases = cases[: args.limit]

    tok = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
    # Refuse before loading the model if any chat text differs from §45's.
    chats = {}
    for c in cases:
        chat = wrap_chat(tok, c["prompt"])
        s1 = stage1_record(c["case_id"], args.stage1)
        if M.sha256_text(chat) != s1["chat_sha256"]:
            print(f"chat sha mismatch on {c['case_id']}", file=sys.stderr)
            return EXIT_SHA
        chats[c["case_id"]] = (chat, s1)

    hf = transformers.AutoModelForCausalLM.from_pretrained(args.weights or MODEL_ID,
                                                           dtype=getattr(torch, args.dtype)).to(args.device)
    hf.eval()
    stamp = {"capture_version": CAPTURE_VERSION, "probe_method": P.METHOD_VERSION, "device": args.device,
             "dtype": args.dtype, "model": args.weights or MODEL_ID}
    t0 = time.time()
    timing = []
    for c in cases:
        if done(out, c["case_id"], stamp):
            continue
        if args.deadline_s and time.time() - t0 > args.deadline_s:
            write_json(out / "_timing.json", timing)
            return EXIT_DEADLINE
        chat, s1 = chats[c["case_id"]]
        pos = P.positions(tok, chat, c)
        if pos["P_last"] != s1["last_prompt_index"]:
            raise RuntimeError(f"{c['case_id']}: P_last {pos['P_last']} vs §45 {s1['last_prompt_index']}")
        sites = SITES[c["task"]]
        t = time.time()
        resid, ids = capture(hf, tok, chat, pos, sites, args.device, args.n_layers)
        dt = time.time() - t
        np.savez(out / f"{c['case_id']}.npz", resid=resid, sites=np.array(sites), input_ids=np.array(ids),
                 positions=np.array([pos[s] for s in sites]))
        meta = {k: c[k] for k in ("case_id", "task", "split", "variant", "format", "hardness", "base_index", "gold",
                                  "competitors", "intermediate", "entities")}
        meta.update(stamp, positions=pos, n_prompt_tokens=len(ids), chat_sha256=s1["chat_sha256"],
                    outcome45=s1["outcome"], first_distractor=P.first_distractor(c), capture_s=round(dt, 3))
        write_json(out / f"{c['case_id']}.json", meta)
        timing.append({"case_id": c["case_id"], "n_tokens": len(ids), "s": round(dt, 3)})
    write_json(out / "_timing.json", timing)
    return 0


if __name__ == "__main__":
    sys.exit(main())
