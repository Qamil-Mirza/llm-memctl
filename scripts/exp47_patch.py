#!/usr/bin/env python
"""§47 the bridge: span interchange on the 36 aligned §44 rev 8 rows, forced-P1 Δ and generated outcome from one
patched suffix (fork), k1, k3, S2. Resumable per row.

    python scripts/exp47_patch.py --device cuda --dtype float32                 # the pod: every row (36)
    python scripts/exp47_patch.py --smoke --out runs/exp47_pod/smoke            # D-22: the 3 smoke rows, timing
    .venv/bin/python scripts/exp47_patch.py --measure 5809eb10 --batch 12      # laptop pre-launch: memory / time only

Spec: research/proposals/2026-10-31-exp47-bridge.md rev 0 (be2b9e5) + rev 1 (a9b70ff); helpers in
memctl/analysis/bridge47.py; rows from scripts/exp47_rows.py (runs/exp47_bridge/rows47.jsonl, sha checked).

Per row: (1) baselines: clean and swap, one forward of the forced sequence (Δ at P1 and the residuals at the site
positions and P1, all layers) and one greedy generation of the chat (to the end of the Answer line, max 160). Asserted:
|Δ - Δ_§44| < 0.05 for clean and swap (rev 1 ruling 3). (2) S1: 36 layer patches (the site residuals of the swap run)
+ k3 (the swap tokens' input embeddings at the sites) in one batch; then 36 k1 in one batch. Each batch: the clean
chat's prefix before the first site cached once and repeated, the suffix recomputed with the hooks on, logits kept for
the last position only (rev 1 ruling 5), the cache copied: (i) the forced tail fed on one copy for Δ at P1, (ii)
greedy generation on the other. (3) S2 at P1 of the forced sequence, layers 0, 3, ..., 33, patch and k1, readout (i)
only. Sanity per row: |T(35)| < 0.05 under the patch and |1 - T(k3)| < 0.05; more than 5 % of rows failing a sanity
or the Δ assert exits 8. No outcome is printed except --smoke (timing and counts only). Exit 3: rows file mismatch.
8: sanity / Δ assert above 5 %. 9: deadline.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from memctl.analysis import bridge47 as B  # noqa: E402
from memctl.analysis import maps as M  # noqa: E402

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
ROWS = Path("runs/exp47_bridge/rows47.jsonl")
SMOKE = ("ec81a493", "7401057b", "5809eb10")
EXIT_ROWS, EXIT_SANITY, EXIT_DEADLINE = 3, 8, 9


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, path)


def read_json(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def load_rows(path: Path = ROWS, expect_sha: str | None = None) -> list[dict]:
    data = Path(path).read_bytes()
    if expect_sha and hashlib.sha256(data).hexdigest() != expect_sha:
        raise SystemExit(EXIT_ROWS)
    return [json.loads(l) for l in data.decode().splitlines() if l.strip()]


class Runner:
    """Batched patched suffix with position-indexed overwrites at several positions, a cache fork for the forced
    readout and the greedy generation."""

    def __init__(self, hf, tok, device: str, batch: int):
        import torch
        self.torch, self.hf, self.tok, self.device, self.batch = torch, hf.eval(), tok, device, batch
        self.layers = hf.model.layers
        self.embed = hf.model.embed_tokens
        eos = hf.generation_config.eos_token_id
        self.eos = set(eos if isinstance(eos, list) else [eos])
        self.n_layers = len(self.layers)

    @classmethod
    def load(cls, device: str, dtype: str, batch: int, weights: str | None = None) -> "Runner":
        import torch
        import transformers
        tok = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
        hf = transformers.AutoModelForCausalLM.from_pretrained(weights or MODEL_ID, dtype=getattr(torch, dtype)).to(device)
        return cls(hf, tok, device, batch)

    # -- unpatched
    def capture(self, ids: list[int], positions: list[int]) -> tuple[np.ndarray, np.ndarray]:
        """(logits at the last position, resid [n_pos, layer, d]) from one forward."""
        torch = self.torch
        with torch.no_grad():
            o = self.hf(torch.tensor([ids], device=self.device), output_hidden_states=True, use_cache=False,
                        logits_to_keep=1)
        hs = o.hidden_states[1:]
        resid = torch.stack([h[0, positions, :] for h in hs], dim=1).float().cpu().numpy()
        return o.logits[0, -1].float().cpu().numpy(), resid

    def embeddings(self, token_ids: list[int]) -> np.ndarray:
        with self.torch.no_grad():
            return self.embed(self.torch.tensor(token_ids, device=self.device)).float().cpu().numpy()

    # -- patched
    def run(self, chat: list[int], sites: list[int], variants: list[tuple], forced_tail: list[int] | None,
            generate: bool, max_new: int = B.MAX_NEW) -> list[dict]:
        """variants: ("none", None, None) | ("resid", layer, vecs [n_sites, d]) | ("embed", None, vecs). Sites are
        positions in `chat`; the prefix before the first site is cached once. Returns per variant {"logits_p1"}
        (forced tail fed after the chat) and/or {"ids", "text"} (greedy generation)."""
        start = min(sites)
        prefix = self._prefix(chat[:start]) if start > 0 else None
        out = []
        for i in range(0, len(variants), self.batch):
            out.extend(self._chunk(chat, sites, start, variants[i:i + self.batch], forced_tail, generate, max_new,
                                   prefix))
        return out

    def _prefix(self, ids):
        with self.torch.no_grad():
            return self.hf(self.torch.tensor([ids], device=self.device), use_cache=True, logits_to_keep=1).past_key_values

    def _hooks(self, variants, rel: list[int], state: dict):
        torch = self.torch

        def make(rows_vecs):
            def fn(module, inputs, output):
                if not state["on"]:
                    return None
                h = output[0] if isinstance(output, tuple) else output
                h = h.clone()
                for row, vecs in rows_vecs:
                    h[row, rel, :] = torch.as_tensor(vecs, dtype=h.dtype, device=h.device)
                return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h
            return fn

        by_layer, emb = {}, []
        for row, (kind, layer, vecs) in enumerate(variants):
            if kind == "resid":
                by_layer.setdefault(layer, []).append((row, vecs))
            elif kind == "embed":
                emb.append((row, vecs))
        handles = [self.layers[l].register_forward_hook(make(rv)) for l, rv in by_layer.items()]
        if emb:
            handles.append(self.embed.register_forward_hook(make(emb)))
        return handles

    def _chunk(self, chat, sites, start, variants, forced_tail, generate, max_new, prefix):
        torch = self.torch
        n = len(variants)
        cache = None
        if prefix is not None:
            cache = copy.deepcopy(prefix)
            if n > 1:
                cache.batch_repeat_interleave(n)
        state = {"on": True}
        rel = [p - start for p in sites]
        handles = self._hooks(variants, rel, state)
        try:
            with torch.no_grad():
                o = self.hf(torch.tensor([chat[start:]] * n, device=self.device), past_key_values=cache,
                            use_cache=True, logits_to_keep=1)
        finally:
            state["on"] = False
            for h in handles:
                h.remove()
        cache = o.past_key_values
        res = [{} for _ in range(n)]
        if forced_tail:
            fork = copy.deepcopy(cache) if generate else cache
            with torch.no_grad():
                f = self.hf(torch.tensor([forced_tail] * n, device=self.device), past_key_values=fork, use_cache=True,
                            logits_to_keep=1)
            for r in range(n):
                res[r]["logits_p1"] = f.logits[r, -1].float().cpu().numpy()
            del fork, f
        if generate:
            gen = [[] for _ in range(n)]
            alive = [True] * n
            nxt = o.logits[:, -1].argmax(-1)
            for step in range(max_new):
                for r in range(n):
                    if alive[r]:
                        t = int(nxt[r])
                        gen[r].append(t)
                        if t in self.eos or M.answer_line_done(self.tok.decode(gen[r], skip_special_tokens=True)):
                            alive[r] = False
                if not any(alive) or step == max_new - 1:
                    break
                with torch.no_grad():
                    o2 = self.hf(nxt.view(n, 1), past_key_values=cache, use_cache=True, logits_to_keep=1)
                cache = o2.past_key_values
                nxt = o2.logits[:, -1].argmax(-1)
            for r in range(n):
                res[r]["ids"] = gen[r]
                res[r]["text"] = self.tok.decode(gen[r], skip_special_tokens=True)
        return res

    def s2(self, forced: list[int], variants: list[tuple]) -> list[np.ndarray]:
        """Readout (i) only, the site = P1 (the last forced id): prefix cached, the last token run under the hook."""
        return [r["logits_p1"] for r in self._s2(forced, variants)]

    def _s2(self, forced, variants):
        torch = self.torch
        prefix = self._prefix(forced[:-1])
        out = []
        for i in range(0, len(variants), self.batch):
            chunk = variants[i:i + self.batch]
            n = len(chunk)
            cache = copy.deepcopy(prefix)
            if n > 1:
                cache.batch_repeat_interleave(n)
            state = {"on": True}
            handles = self._hooks([(k, l, v[None, :]) for k, l, v in chunk], [0], state)
            try:
                with torch.no_grad():
                    o = self.hf(torch.tensor([forced[-1:]] * n, device=self.device), past_key_values=cache,
                                use_cache=True, logits_to_keep=1)
            finally:
                for h in handles:
                    h.remove()
            out += [{"logits_p1": o.logits[r, -1].float().cpu().numpy()} for r in range(n)]
        return out


def forced_ids(row: dict, kind: str) -> list[int]:
    return list(row[f"chat_{kind}"]) + list(row["forced_tail"])


def baseline(run: Runner, row: dict, kind: str) -> tuple[dict, np.ndarray]:
    forced = forced_ids(row, kind)
    sites = list(row["diff_positions"]) + [row["p1"]]
    logits, resid = run.capture(forced, sites)
    d = B.delta(logits, row["g_id"], row["s_id"])
    g = run.run(list(row[f"chat_{kind}"]), list(row["diff_positions"]), [("none", None, None)], None, True)[0]
    rec = {"delta": d, "delta44": row[f"delta44_{kind}"], "delta_ok": abs(d - row[f"delta44_{kind}"]) < B.DELTA_TOL,
           "top1": int(np.argmax(logits)), "text": g["text"], "n_generated": len(g["ids"]),
           "outcome": B.generated_outcome(g["text"], row["gold"], row["substitute"], g["ids"])}
    return rec, resid


def do_row(run: Runner, row: dict, assert44: bool) -> dict:
    t0 = time.time()
    q = row["question_id"]
    sites = list(row["diff_positions"])
    ns = len(sites)
    bc, hc = baseline(run, row, "clean")
    bs, hs = baseline(run, row, "swap")
    dc, ds = bc["delta"], bs["delta"]
    L = run.n_layers
    chat = list(row["chat_clean"])
    tail = list(row["forced_tail"])
    swap_tok = [row["chat_swap"][p] for p in sites]
    t1 = time.time()
    v1 = [("resid", l, hs[:ns, l]) for l in range(L)] + [("embed", None, run.embeddings(swap_tok))]
    r1 = run.run(chat, sites, v1, tail, True)
    t2 = time.time()
    v2 = [("resid", l, B.k1_vectors(hc[:ns, l], hs[:ns, l], q, l, sites)) for l in range(L)]
    r2 = run.run(chat, sites, v2, tail, True)
    t3 = time.time()
    s2_layers = [l for l in B.S2_LAYERS if l < L]
    v3 = []
    for l in s2_layers:
        v3.append(("resid", l, hs[ns, l]))
        v3.append(("resid", l, B.k1_vectors(hc[ns:, l], hs[ns:, l], q, l, [row["p1"]], site="S2")[0]))
    r3 = run.s2(forced_ids(row, "clean"), v3)
    t4 = time.time()

    def pack(r, kind, layer):
        d = B.delta(r["logits_p1"], row["g_id"], row["s_id"]) if "logits_p1" in r else None
        out = {"kind": kind, "layer": layer, "delta": d, "T": None if d is None else B.transfer(d, dc, ds)}
        if "text" in r:
            out.update({"text": r["text"], "n_generated": len(r["ids"]), "top1": int(np.argmax(r["logits_p1"])),
                        "outcome": B.generated_outcome(r["text"], row["gold"], row["substitute"], r["ids"])})
        return out

    patch = [pack(r1[l], "patch", l) for l in range(L)]
    k3 = pack(r1[L], "k3", None)
    k3["reproduces_swap"] = bool(k3["top1"] == bs["top1"] and k3["outcome"] == bs["outcome"])
    k1 = [pack(r2[l], "k1", l) for l in range(L)]
    s2 = []
    for j, l in enumerate(s2_layers):
        for kind, logits in (("patch", r3[2 * j]), ("k1", r3[2 * j + 1])):
            d = B.delta(logits, row["g_id"], row["s_id"])
            s2.append({"kind": kind, "layer": l, "delta": d, "T": B.transfer(d, dc, ds)})
    t_last = patch[-1]["T"]
    sanity = {"t_last": t_last, "t_last_ok": t_last is not None and abs(t_last) < B.SANITY_TOL,
              "t_k3": k3["T"], "t_k3_ok": k3["T"] is not None and abs(1 - k3["T"]) < B.SANITY_TOL,
              "delta_ok": bc["delta_ok"] and bs["delta_ok"], "delta_asserted": assert44}
    return {"question_id": q, "label": row["label"], "cls1": row["cls1"], "entered": row["entered"],
            "multi_occurrence": row["multi_occurrence"], "n_sites": ns, "method": B.METHOD_VERSION,
            "T_fwd44": row.get("T_fwd44"),
            "baseline": {"clean": bc, "swap": bs}, "patch": patch, "k3": k3, "k1": k1, "s2": s2, "sanity": sanity,
            "timing_s": {"baselines": t1 - t0, "s1_patch_k3": t2 - t1, "k1": t3 - t2, "s2": t4 - t3, "total": t4 - t0},
            "n_passes": {"s1": L + 1, "k1": L, "s2": len(v3), "baselines": 4}}


def sanity_gate(root: Path, assert44: bool) -> tuple[int, int]:
    recs = [read_json(p) for p in sorted((root / "rows").glob("*.json"))]
    bad = 0
    for r in recs:
        s = r["sanity"]
        if not (s["t_last_ok"] and s["t_k3_ok"] and (s["delta_ok"] or not assert44)):
            bad += 1
    return bad, len(recs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=Path, default=ROWS)
    ap.add_argument("--expect-sha", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--batch", type=int, default=19, help="variants per chunk (37 = one S1 batch; 19 = two chunks, "
                    "about half the A40 peak in float32 on the longest row)")
    ap.add_argument("--weights", default=None, help="model weights dir (the pod stub); tokenizer stays the 3B's")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--measure", default=None, help="one row id: one S1 batch, peak RSS and seconds; no outcome kept")
    ap.add_argument("--no-assert44", action="store_true", help="the stub only (its Δ cannot match §44)")
    ap.add_argument("--deadline-s", type=float, default=None)
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args()
    if args.threads:
        import torch
        torch.set_num_threads(args.threads)
    rows = load_rows(args.rows, args.expect_sha)
    root = args.out or Path("runs/exp47_bridge") / f"{args.device}-{args.dtype}"
    run = Runner.load(args.device, args.dtype, args.batch, args.weights)

    if args.measure:
        row = next(r for r in rows if r["question_id"] == args.measure)
        sites = list(row["diff_positions"])
        d = run.hf.config.hidden_size
        vecs = [("resid", l, np.zeros((len(sites), d), np.float32)) for l in range(run.batch)]
        t = time.time()
        run.run(list(row["chat_clean"]), sites, vecs, list(row["forced_tail"]), True)
        dt = time.time() - t
        payload = {"row": args.measure, "n_chat": row["n_chat"], "suffix": row["n_chat"] - min(sites),
                   "batch": run.batch, "seconds": dt, "s_per_pass": dt / run.batch,
                   "peak_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
                   "device": args.device, "dtype": args.dtype}
        if args.device == "cuda":
            payload["peak_cuda_gb"] = run.torch.cuda.max_memory_allocated() / 2**30
        write_json(root / f"measure_{args.measure}.json", payload)
        print(json.dumps(payload))
        return 0

    todo = [r for r in rows if (r["question_id"] in SMOKE if args.smoke else True)]
    if args.only:
        todo = [r for r in todo if r["question_id"] in args.only]
    t0 = time.time()
    timing = []
    for row in todo:
        path = root / "rows" / f"{row['question_id']}.json"
        if read_json(path, {}).get("method") == B.METHOD_VERSION:
            continue
        if args.deadline_s and time.time() - t0 > args.deadline_s:
            return EXIT_DEADLINE
        rec = do_row(run, row, not args.no_assert44)
        write_json(path, rec)
        timing.append({"question_id": row["question_id"], "n_chat": row["n_chat"], **rec["timing_s"]})
    write_json(root / ("_timing_smoke.json" if args.smoke else "_timing.json"), timing)
    bad, n = sanity_gate(root, not args.no_assert44)
    if args.smoke:
        per_row = sum(x["total"] for x in timing) / max(1, len(timing))
        k3 = [read_json(root / "rows" / f"{q}.json")["k3"]["reproduces_swap"] for q in SMOKE
              if (root / "rows" / f"{q}.json").exists()]
        peak = run.torch.cuda.max_memory_allocated() / 2**30 if args.device == "cuda" else None
        payload = {"rows": [x["question_id"] for x in timing], "per_row_s": per_row, "timing": timing,
                   "peak_cuda_gb": peak, "batch": run.batch,
                   "projection_total_s": per_row * len(rows), "n_rows_all": len(rows), "sanity_bad": bad,
                   "k3_reproduces": k3, "pass": bad == 0}
        write_json(root / "smoke.json", payload)
        print(json.dumps({k: payload[k] for k in ("per_row_s", "projection_total_s", "sanity_bad", "pass")}))
        return 0 if bad == 0 else EXIT_SANITY
    if n and bad / n > B.SANITY_MAX_FAIL:
        print(f"sanity / Δ assert failed on {bad} of {n} rows", file=sys.stderr)
        return EXIT_SANITY
    return 0


if __name__ == "__main__":
    sys.exit(main())
