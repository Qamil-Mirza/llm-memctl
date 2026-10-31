#!/usr/bin/env python
"""§46 measurement B: interchange intervention on aligned P / P' / P'' triples (RC). Resumable per triple and stage.

    .venv/bin/python scripts/exp46_patch.py --stage build-cases                     # -> runs/exp46_probe/cases46.jsonl
    python scripts/exp46_patch.py --stage baselines --device cuda --dtype bfloat16  # (i) unpatched generations + resid
    python scripts/exp46_patch.py --stage validate                                  # first-city vs §45 rule, dev only
    python scripts/exp46_patch.py --stage s1 [--split dev|held]                     # (ii) S1 patch, 36 layers + k3
    python scripts/exp46_patch.py --stage controls                                  # (iii) k1, k2 at S1, 36 layers
    python scripts/exp46_patch.py --stage s2                                        # (iv) S2 at layers 0,3..33, + k1
    python scripts/exp46_patch.py --smoke 3 --out runs/exp46_pod/smoke              # D-22: all stages on 3 dev triples

Spec: research/proposals/2026-10-30-exp46-instrument-check.md §4 with rev 1 rulings 1, 2, 4 (6ff90c8); helpers in
memctl/analysis/patch46.py. Sites: S1 = P_B (the supervisor's token in sentence 1), S2 = P_last. The residual is
the output of decoder layer l (HF hidden_states[l + 1]); a patch overwrites it at the site for one batch row, the
prefix before the site is computed once per triple and repeated over the batch, the 36 layer variants of a pair run
as one batch (--batch splits it). k3 (token-substitution floor) overwrites the input embedding at P_B with P''s:
it must reproduce P''s baseline outcome on >= 0.95 of pairs. The readout rule comes from validate (first-city if
agreement >= 0.95 on dev baselines, else the §45 rule with max_new 48 / 160). Nothing here prints an outcome except
--smoke. Exit 3: validate not run. 6: triple alignment broken at run time. 9: stopped at the deadline.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from memctl.analysis import maps as M  # noqa: E402
from memctl.analysis import patch46 as T  # noqa: E402
from memctl.analysis import probe46 as P  # noqa: E402

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
CASES = Path("runs/exp45_maps/cases.jsonl")
CASES46 = Path("runs/exp46_probe/cases46.jsonl")
STAGES = ("baselines", "s1", "controls", "s2")
EXIT_NO_VALIDATE, EXIT_ALIGN, EXIT_DEADLINE = 3, 6, 9


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


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def pools():
    import exp45_cases as C
    return C.NAMES, C.CITIES


# ---------------------------------------------------------------------------------------------- cases46

def build_cases(tokenizer, h2_policy: str, out: Path = CASES46) -> dict:
    import exp45_cases as C
    from memctl.analysis.jacobian import wrap_chat

    src = [r for r in load_jsonl(CASES) if r["task"] == "RC" and r["variant"] in ("v0", "v3")]
    rows, drops = [], []
    for c in src:
        try:
            tr = T.build_triple_rows(c, C.NAMES, C.CITIES, C.render_prompt, h2_policy=h2_policy)
            ids, site = {}, {}
            for r in tr:
                chat = wrap_chat(tokenizer, r["prompt"])
                ids[r["role"]] = tokenizer(chat, add_special_tokens=False).input_ids
                site[r["role"]] = P.positions(tokenizer, chat, r)["P_B"]
            pb = T.check_triple_alignment(ids, site, c["case_id"])
        except T.TripleAlignmentError as e:
            drops.append({"case_id": c["case_id"], "why": str(e)})
            continue
        for r in tr:
            r["P_B"] = pb
            r["n_prompt_tokens"] = len(ids[r["role"]])
        rows.extend(tr)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    triples = [r for r in rows if r["role"] == "P"]
    summary = {"addition": T.ADDITION, "seed": T.SEED, "h2_policy": h2_policy, "source_rows": len(src),
               "triples": len(triples), "alignment_drops": drops,
               "by_split": dict(Counter(r["split"] for r in triples)),
               "by_split_hardness": dict(Counter(f"{r['split']}|{r['hardness']}" for r in triples)),
               "b2_reuses_distractor": sum(r["b2_reuses_distractor"] for r in triples),
               "b2_has_rival_city": sum(r["b2_has_rival_city"] for r in triples),
               "cases46_sha256": M.sha256_text(out.read_text())}
    write_json(out.with_name("cases46_counts.json"), summary)
    return summary


def triples_of(rows: list[dict]) -> list[dict[str, dict]]:
    by = {}
    for r in rows:
        by.setdefault(r["triple_id"], {})[r["role"]] = r
    order = sorted(by.values(), key=lambda t: (t["P"]["split"] != "dev", t["P"]["base_index"], t["P"]["variant"],
                                              t["P"]["format"], t["P"]["hardness"]))
    return order


# ---------------------------------------------------------------------------------------------- model

class Runner:
    """Batched greedy generation with position-indexed overwrites (resid at a layer, or the input embedding)."""

    def __init__(self, hf, tok, device: str, batch: int):
        import torch

        self.torch = torch
        self.tok, self.hf = tok, hf.eval()
        self.device, self.batch = device, batch
        self.layers = self.hf.model.layers
        self.embed = self.hf.model.embed_tokens
        eos = self.hf.generation_config.eos_token_id
        self.eos = set(eos if isinstance(eos, list) else [eos])
        self.n_layers = len(self.layers)

    @classmethod
    def load(cls, device: str, dtype: str, batch: int, weights: str | None = None) -> "Runner":
        import torch
        import transformers

        tok = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
        hf = transformers.AutoModelForCausalLM.from_pretrained(weights or MODEL_ID, dtype=getattr(torch, dtype)).to(device)
        return cls(hf, tok, device, batch)

    def ids(self, prompt: str) -> list[int]:
        from memctl.analysis.jacobian import wrap_chat
        return self.tok(wrap_chat(self.tok, prompt), add_special_tokens=False).input_ids

    def residuals(self, ids: list[int], sites: list[int]) -> np.ndarray:
        """[site, layer, d] float32, output of decoder layer l."""
        torch = self.torch
        with torch.no_grad():
            out = self.hf(torch.tensor([ids], device=self.device), output_hidden_states=True, use_cache=False)
        hs = out.hidden_states[1:]
        return torch.stack([h[0, sites, :] for h in hs], dim=1).float().cpu().numpy()

    def embedding(self, token_id: int) -> np.ndarray:
        with self.torch.no_grad():
            return self.embed(self.torch.tensor([token_id], device=self.device))[0].float().cpu().numpy()

    def generate(self, ids: list[int], site: int, variants: list[tuple[str, int | None, np.ndarray | None]],
                 max_new: int, stop) -> list[dict]:
        """One greedy continuation per variant. variant = ("none", None, None) | ("resid", layer, vec) |
        ("embed", None, vec): overwrite at token `site` of the prompt. `stop(text)` ends a row early."""
        out = []
        prefix_cache = self._prefix(ids[:site]) if site > 0 else None
        for i in range(0, len(variants), self.batch):
            out.extend(self._chunk(ids, site, variants[i:i + self.batch], max_new, stop, prefix_cache))
        return out

    def _prefix(self, prefix: list[int]):
        torch = self.torch
        with torch.no_grad():
            return self.hf(torch.tensor([prefix], device=self.device), use_cache=True).past_key_values

    def _chunk(self, ids, site, variants, max_new, stop, prefix_cache):
        torch = self.torch
        n = len(variants)
        cache = None
        if prefix_cache is not None:
            cache = copy.deepcopy(prefix_cache)
            if n > 1:
                cache.batch_repeat_interleave(n)
        state = {"on": True}
        handles = []

        def resid_hook(rows_vecs):
            def fn(module, inputs, output):
                if not state["on"]:
                    return None
                h = output[0] if isinstance(output, tuple) else output
                h = h.clone()
                for row, vec in rows_vecs:
                    h[row, 0, :] = torch.as_tensor(vec, dtype=h.dtype, device=h.device)
                return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h
            return fn

        by_layer, embed_rows = {}, []
        for row, (kind, layer, vec) in enumerate(variants):
            if kind == "resid":
                by_layer.setdefault(layer, []).append((row, vec))
            elif kind == "embed":
                embed_rows.append((row, vec))
        try:
            for layer, rv in by_layer.items():
                handles.append(self.layers[layer].register_forward_hook(resid_hook(rv)))
            if embed_rows:
                handles.append(self.embed.register_forward_hook(resid_hook(embed_rows)))
            suffix = torch.tensor([ids[site:]] * n, device=self.device)
            with torch.no_grad():
                o = self.hf(suffix, past_key_values=cache, use_cache=True)
            state["on"] = False
            gen = [[] for _ in range(n)]
            alive = [True] * n
            nxt = o.logits[:, -1].argmax(-1)
            cache = o.past_key_values
            for step in range(max_new):
                for r in range(n):
                    if alive[r]:
                        t = int(nxt[r])
                        gen[r].append(t)
                        if t in self.eos or stop(self.tok.decode(gen[r], skip_special_tokens=True)):
                            alive[r] = False
                if not any(alive) or step == max_new - 1:
                    break
                with torch.no_grad():
                    o = self.hf(nxt.view(n, 1), past_key_values=cache, use_cache=True)
                cache = o.past_key_values
                nxt = o.logits[:, -1].argmax(-1)
        finally:
            for h in handles:
                h.remove()
        return [{"ids": g, "text": self.tok.decode(g, skip_special_tokens=True)} for g in gen]


# ---------------------------------------------------------------------------------------------- readout

def readout_rule(root: Path) -> dict | None:
    return read_json(root / "readout.json")


def stop_fn(rule: str, fmt: str, cities):
    if rule == "first_city":
        return lambda text: T.stop_now(text, cities)
    if fmt == "F2":
        return M.answer_line_done
    return lambda text: False


def max_new(rule: str, fmt: str) -> int:
    return (T.MAX_NEW_FIRST_CITY if rule == "first_city" else T.MAX_NEW_45)[fmt]


def outcome(rule: str, text: str, c1: str, c1p: str, cities) -> str:
    if rule == "first_city":
        return T.classify_first(text, c1, c1p, cities)
    o = M.classify(text, c1, [c1p])["outcome"]
    return {"gold": "C1", "competitor": "C1p"}.get(o, o)


SOLVED_AS = {"P": "C1", "Pp": "C1p", "Ppp": "C1"}


# ---------------------------------------------------------------------------------------------- stages

def stage_baselines(run: Runner, triple: dict, root: Path, names, cities) -> dict:
    """(i) per role: full §45-length generation (F1 48, F2 160 to the end of the Answer line), the first-city
    readout on its prefix, the §45 outcome, and the residuals at P_B and P_last (all layers)."""
    rec = {"triple_id": triple["P"]["triple_id"], "roles": {}}
    p = triple["P"]
    for role in T.ROLES:
        r = triple[role]
        ids = run.ids(r["prompt"])
        if len(ids) != r["n_prompt_tokens"]:
            raise T.TripleAlignmentError(f"{r['case_id']}: {len(ids)} tokens, cases46 says {r['n_prompt_tokens']}")
        t = time.time()
        g = run.generate(ids, len(ids) - 1, [("none", None, None)], T.MAX_NEW_45[r["format"]],
                         M.answer_line_done if r["format"] == "F2" else (lambda text: False))[0]
        gen_s = time.time() - t
        first_text = run.tok.decode(g["ids"][:T.MAX_NEW_FIRST_CITY[r["format"]]], skip_special_tokens=True)
        resid = run.residuals(ids, [r["P_B"], len(ids) - 1])
        np.save(root / "resid" / f"{r['case_id']}.npy", resid)
        rec["roles"][role] = {"case_id": r["case_id"], "text": g["text"], "n_generated": len(g["ids"]),
                              "first_text": first_text,
                              "first_city": T.classify_first(first_text, p["C1"], p["C1p"], cities),
                              "rule45": outcome("rule45", g["text"], p["C1"], p["C1p"], cities),
                              "names_before_city": T.names_before_city(first_text, [p["B"], p["Bp"], p["Bpp"]], cities),
                              "token_at_PB": int(ids[r["P_B"]]), "gen_s": round(gen_s, 3)}
    return rec


def validate(root: Path, rows: list[dict]) -> dict:
    """Ruling 2: first-city vs the §45 rule on every dev baseline generation; >= 0.95 -> first_city."""
    agree = n = 0
    disagreements = []
    for t in triples_of(rows):
        if t["P"]["split"] != "dev":
            continue
        b = read_json(root / "baselines" / f"{t['P']['triple_id']}.json")
        if b is None:
            continue
        for role, r in b["roles"].items():
            n += 1
            if r["first_city"] == r["rule45"]:
                agree += 1
            else:
                disagreements.append({"case_id": r["case_id"], "first_city": r["first_city"], "rule45": r["rule45"]})
    share = agree / n if n else None
    payload = {"n": n, "agree": agree, "share": share, "min": T.AGREE_MIN,
               "rule": "first_city" if (share is not None and share >= T.AGREE_MIN) else "rule45",
               "disagreements": disagreements}
    write_json(root / "readout.json", payload)
    return payload


def baseline_outcome(rule: str, role_rec: dict, p: dict, cities) -> str:
    """A role's baseline outcome under the readout rule in force."""
    return role_rec["first_city"] if rule == "first_city" else role_rec["rule45"]


def solved(b: dict, rule: str) -> dict[str, bool]:
    key = "first_city" if rule == "first_city" else "rule45"
    return {role: b["roles"][role][key] == SOLVED_AS[role] for role in T.ROLES}


def resid_of(root: Path, case_id: str) -> np.ndarray:
    return np.load(root / "resid" / f"{case_id}.npy")      # [2 (P_B, P_last), layer, d]


def stage_patch(run: Runner, triple: dict, root: Path, rule: str, stage: str, cities, force: bool = False) -> dict:
    """One stage for one triple. `force` (the smoke only): run the passes whether or not the pair is solved."""
    p, pp, ppp = triple["P"], triple["Pp"], triple["Ppp"]
    b = read_json(root / "baselines" / f"{p['triple_id']}.json")
    ok = solved(b, rule)
    if force:
        ok = {k: True for k in ok}
    if not (ok["P"] and ok["Pp"]):
        return {"triple_id": p["triple_id"], "entered": False, "solved": ok}
    ids = run.ids(p["prompt"])
    hP, hPp = resid_of(root, p["case_id"]), resid_of(root, pp["case_id"])
    L = run.n_layers
    variants, tags = [], []
    if stage == "s1":
        for l in range(L):
            variants.append(("resid", l, hPp[0, l])); tags.append(("patch", l))
        variants.append(("embed", None, run.embedding(b["roles"]["Pp"]["token_at_PB"]))); tags.append(("k3", None))
        site = p["P_B"]
    elif stage == "controls":
        for l in range(L):
            rng = np.random.default_rng(T.k1_seed(p["triple_id"], "S1", l))
            variants.append(("resid", l, T.k1_vector(hP[0, l], hPp[0, l], rng))); tags.append(("k1", l))
        if ok["Ppp"]:
            hPpp = resid_of(root, ppp["case_id"])
            for l in range(L):
                variants.append(("resid", l, hPpp[0, l])); tags.append(("k2", l))
        site = p["P_B"]
    elif stage == "s2":
        for l in [x for x in T.S2_LAYERS if x < L]:
            variants.append(("resid", l, hPp[1, l])); tags.append(("patch", l))
            rng = np.random.default_rng(T.k1_seed(p["triple_id"], "S2", l))
            variants.append(("resid", l, T.k1_vector(hP[1, l], hPp[1, l], rng))); tags.append(("k1", l))
        site = len(ids) - 1
    else:
        raise ValueError(stage)
    t = time.time()
    gens = run.generate(ids, site, variants, max_new(rule, p["format"]), stop_fn(rule, p["format"], cities))
    dt = time.time() - t
    results = [{"kind": k, "layer": l, "outcome": outcome(rule, g["text"], p["C1"], p["C1p"], cities),
                "names_before_city": T.names_before_city(g["text"], [p["B"], p["Bp"], p["Bpp"]], cities),
                "n_generated": len(g["ids"]), "text": g["text"]} for (k, l), g in zip(tags, gens)]
    if stage == "s1":
        k3 = results[-1]
        k3["reproduces_Pp"] = k3["outcome"] == SOLVED_AS["Pp"]
        k3["matches_Pp_baseline"] = k3["outcome"] == baseline_outcome(rule, b["roles"]["Pp"], p, cities)
    return {"triple_id": p["triple_id"], "split": p["split"], "entered": True, "solved": ok, "rule": rule,
            "stage": stage, "n_passes": len(variants), "s": round(dt, 3), "results": results}


def run_stage(run, stage, rows, root, rule, names, cities, split, deadline, limit=None, fmt=None, force=False) -> int:
    t0 = time.time()
    todo = [t for t in triples_of(rows) if split in ("all", t["P"]["split"]) and fmt in (None, t["P"]["format"])]
    if limit:
        todo = todo[:limit]
    for t in todo:
        path = root / stage / f"{t['P']['triple_id']}.json"
        if path.exists():
            continue
        if deadline and time.time() - t0 > deadline:
            return EXIT_DEADLINE
        if stage != "baselines" and not (root / "baselines" / f"{t['P']['triple_id']}.json").exists():
            continue                                   # no baseline yet (cut by a deadline): nothing to patch
        if stage == "baselines":
            rec = stage_baselines(run, t, root, names, cities)
        else:
            rec = stage_patch(run, t, root, rule, stage, cities, force=force)
        write_json(path, rec)
    return 0


def smoke(run: Runner, rows: list[dict], root: Path, n: int, names, cities) -> int:
    """D-22: every stage on the first n dev triples, the passes forced (solved or not) so the rate is measured; the
    readout rule from these baselines only (provisional, smoke dir only). Pass: every k3 row reproduces P''s baseline
    outcome. Writes smoke.json with seconds per triple per stage and the projection for the full run (all 144
    triples entered: an upper bound). Exit 5 on a failed check."""
    dev = [r for r in rows if r["split"] == "dev"]
    first = triples_of(dev)[:n]
    sub = [r for t in first for r in t.values()]
    timing = {}
    t = time.time()
    run_stage(run, "baselines", sub, root, None, names, cities, "dev", None)
    timing["baselines"] = time.time() - t
    rule = validate(root, sub)["rule"]
    for stage in STAGES[1:]:
        t = time.time()
        run_stage(run, stage, sub, root, rule, names, cities, "dev", None, force=True)
        timing[stage] = time.time() - t
    recs = {s: [read_json(root / s / f"{x['P']['triple_id']}.json") for x in first] for s in STAGES[1:]}
    k3 = [x for r in recs["s1"] for x in r["results"] if x["kind"] == "k3"]
    passes = {s: sum(r["n_passes"] for r in recs[s]) for s in STAGES[1:]}
    per_triple = {s: v / max(1, len(first)) for s, v in timing.items()}
    n_all = len(triples_of(rows))
    projection = {s: per_triple[s] * n_all for s in per_triple}
    ok = bool(k3) and all(x["matches_Pp_baseline"] for x in k3)
    payload = {"n_triples": len(first), "timing_s": timing, "per_triple_s": per_triple, "passes": passes,
               "s_per_pass": {s: timing[s] / max(1, passes[s]) for s in passes}, "projection_s": projection,
               "projection_total_s": sum(projection.values()), "n_triples_all": n_all, "rule": rule,
               "k3_matches_Pp_baseline": [x["matches_Pp_baseline"] for x in k3], "pass": ok}
    write_json(root / "smoke.json", payload)
    print(json.dumps({k: payload[k] for k in ("per_triple_s", "projection_total_s", "k3_matches_Pp_baseline", "pass")}))
    return 0 if ok else 5


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("build-cases", "baselines", "validate", "s1", "controls", "s2"))
    ap.add_argument("--smoke", type=int, default=None)
    ap.add_argument("--h2-policy", choices=T.H2_POLICIES, default="reuse")
    ap.add_argument("--cases46", type=Path, default=CASES46)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--batch", type=int, default=37)
    ap.add_argument("--split", choices=("dev", "held", "all"), default="all")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--deadline-s", type=float, default=None)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--format", choices=("F1", "F2"), default=None, help="development runs only")
    ap.add_argument("--weights", default=None, help="model weights dir (the pod stub); tokenizer stays the 3B's")
    ap.add_argument("--limit-triples", type=int, default=None, help="the stub chain only")
    args = ap.parse_args()
    names, cities = pools()

    if args.stage == "build-cases":
        import transformers
        tok = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
        s = build_cases(tok, args.h2_policy, args.cases46)
        print(json.dumps({k: v for k, v in s.items() if k != "alignment_drops"} | {"drops": len(s["alignment_drops"])}))
        return 0

    root = args.out or Path("runs/exp46_patch") / f"{args.device}-{args.dtype}"
    (root / "resid").mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(args.cases46)
    if args.stage == "validate":
        v = validate(root, rows)
        print(json.dumps({k: v[k] for k in ("n", "agree", "share", "rule")}))
        return 0

    if args.threads:
        import torch
        torch.set_num_threads(args.threads)
    run = Runner.load(args.device, args.dtype, args.batch, args.weights)
    if args.smoke:
        return smoke(run, rows, root, args.smoke, names, cities)

    if args.stage == "baselines":
        return run_stage(run, "baselines", rows, root, None, names, cities, args.split, args.deadline_s, fmt=args.format,
                         limit=args.limit_triples)
    rule = (readout_rule(root) or {}).get("rule")
    if rule is None:
        print("run --stage validate first", file=sys.stderr)
        return EXIT_NO_VALIDATE
    return run_stage(run, args.stage, rows, root, rule, names, cities, args.split, args.deadline_s,
                     limit=args.limit_triples)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except T.TripleAlignmentError as e:
        print(f"alignment: {e}", file=sys.stderr)
        sys.exit(EXIT_ALIGN)
