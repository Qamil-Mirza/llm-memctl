#!/usr/bin/env python
"""Stage 2 pilot (PROPOSAL decision 8; Advisor scope 2026-11-09): Qwen3-1.7B, GSM8K DEV only, U0 rows.
Tests 1 and 2 of PROPOSAL Stage 2, on dev stale failures vs dev successes. Readings and bars are written in and frozen
by the Advisor before any held-out run; this file only measures.

    # 1. laptop, $0: pick the rows from the dev store (pod 1 step 0) and build every prompt string
    .venv/bin/python selfdoubt/stage2_pilot.py select --store runs/selfdoubt/s0c_pod1/dev/full/Qwen3-1.7B/gen.jsonl \
        --out runs/selfdoubt/s2_pilot/rows.jsonl
    # 2. pod (or laptop on Qwen3-0.6B for the build test): teacher-forced forward passes, resumable
    python selfdoubt/stage2_pilot.py run --rows runs/selfdoubt/s2_pilot/rows.jsonl --out runs/selfdoubt/s2_pilot/res.jsonl
    # 3. descriptive tables (no verdicts)
    python selfdoubt/stage2_pilot.py report --res runs/selfdoubt/s2_pilot/res.jsonl

Rows. A U0 continuation at a cut (0.3 / 0.5 / 0.9) scored by the frozen Stage 0 scorer (stage0_report.score_model) as
"stale" (failure: the same prefix's N answer, not the gold) or "new" (success: the gold). "new" = original gold, "old"
= that prefix's N answer (never the gold: such item-cuts are dropped by the scorer). Only items in split["dev"]["gsm8k"]
are accepted; anything else aborts. The base prompt is rebuilt exactly (common.continuation_prompt) and checked
against the record's prompt_sha256.

Readout. At offset k tokens of the row's own U0 continuation after "</update>\\n\\n" (k in OFFSETS; 0 = right after the
update), the forced text READOUT is appended, then the tokens the new and old answers share; the margin is
logit(new token) - logit(old token) at the first token where "READOUT + new + }" and "READOUT + old + }" differ
(Qwen3 writes one digit per token). READOUT and the continuation are tokenised as their own segments.

Test 2 (lens). margin_l = the same difference read through the final norm + unembedding from the residual stream after
layer l (l = 0 is the embedding output; l = n_layers is the last layer, before the final norm). On the base prompt
(lens_base), on the donor (lens_donor), and on N (the same prefix, no update, READOUT appended: lens_n, offset-free).

Test 1 (interchange at the update's number tokens). Donor = the base prompt with every digit of the update replaced by a
different digit (seeded per row; same token count, so positions align; only those tokens differ). m_donor - m_base is
the readout's response to the swap (E1 predicts ~0 on failures; E2 predicts it as large as on successes). For each l,
the residual stream at the digit positions after layer l is
  swap : replaced by the donor's                 -> patch_swap[l]  (l = 0 reproduces m_donor; l = n_layers gives m_base)
  rand : base + a random vector with the norm of (donor - base) at that position and layer      -> patch_rand[l]
  x0   : replaced by the same-update donor: the item's U0 prompt at X = 0 (same update text, so the same answer,
         different context), digit tokens aligned by order                                       -> patch_x0[l]
then everything after is recomputed and READOUT is teacher-forced (patch-then-teacher-force, the primary regime). All
margins are in logits. Positions before the first digit are shared by every condition and run once (KV cache).

Advisor review of 220292c (2026-11-09), applied here:
- every margin is logged with logit(new), logit(old) and their ranks (base and donor), and the lens per layer for new and
  old separately, so a floor at k = 0 is visible (a multi-step answer can rarely be computed right after </update>);
- offset "end" = just before the row's own final \\boxed: base, donor and lens only, descriptive (class-confounded);
- second target "fact": FACT_READOUT + the changed quantity itself, the update's number (from the original problem) vs
  the revised problem's number it replaces (the first that differs, in problem order; rows without one are skipped and
  counted), at the same offsets with the same lens and patches. Tests E1 without computing the answer;
- donor token counts are asserted per row; mismatches are skipped and counted; patch_x0 is a secondary control;
- select's cap is split over cuts in proportion to k_c, so both classes keep the same cut mix.
Caveat recorded for the Advisor: at k > 0 the continuation is the base row's own text, which may restate the update's
numbers; the donor keeps that text, so k = 0 is the clean point and k > 0 is secondary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from selfdoubt import common as C  # noqa: E402

MODEL = "Qwen/Qwen3-1.7B"
READOUT = "\n\nSo the answer is \\boxed{"
FACT_READOUT = "\n\nSo the updated value is "     # second target (Advisor 2026-11-09): the changed quantity itself
OFFSETS = (0, 32, 128, "end")    # "end": just before the row's own final \\boxed (descriptive only: no patch sweeps)
CLASSES = ("stale", "new")
GATE = 40
SEED = 56
NUM_RE = re.compile(r"^-?\d+(\.\d+)?$")


# ---------------------------------------------------------------- strings and tokens (no model needed)

def update_span(prompt: str, head: str, arm: str, update: str) -> tuple[int, int]:
    """Character span of the update text inside prompt = head + injected_text(arm, update)."""
    w = C.load_wrapper(arm)
    s = len(head) + len(w.get("reasoning_prefix", "")) + len(w.get("update_prefix", ""))
    assert prompt[s:s + len(update)] == update, "update not where the wrapper puts it"
    return s, s + len(update)


def encode_with_offsets(tok, text: str):
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    return list(enc["input_ids"]), [tuple(x) for x in enc["offset_mapping"]]


def digit_positions(tok, ids, offs, span) -> list[int]:
    """Token indices inside the char span whose token is a single digit."""
    s, e = span
    return [i for i, (a, b) in enumerate(offs) if a >= s and b <= e and b > a and tok.decode([ids[i]]).isdigit()]


def swap_digits(text: str, span: tuple[int, int], seed_key: str) -> str:
    """Replace every digit inside span by a different digit; a run's first digit stays non-zero if the run is longer
    than one digit. Deterministic in seed_key."""
    rng = random.Random(hashlib.sha256(seed_key.encode()).hexdigest())
    s, e = span
    out = list(text)
    i = s
    while i < e:
        if text[i].isdigit():
            j = i
            while j < e and text[j].isdigit():
                j += 1
            for k in range(i, j):
                pool = [d for d in "0123456789" if d != text[k] and not (k == i and j - i > 1 and d == "0")]
                out[k] = rng.choice(pool)
            i = j
        else:
            i += 1
    return "".join(out)


def readout_tokens(tok, new: str, old: str, readout: str = READOUT, close: str = "}") -> tuple[list[int], int, int]:
    """(forced ids, new token, old token): readout plus the shared prefix of the two values, then the first token where
    readout + value + close differ."""
    a = tok(readout + new + close, add_special_tokens=False)["input_ids"]
    b = tok(readout + old + close, add_special_tokens=False)["input_ids"]
    j = 0
    while j < min(len(a), len(b)) and a[j] == b[j]:
        j += 1
    assert j < min(len(a), len(b)), f"answers {new!r} and {old!r} give the same tokens"
    return a[:j], a[j], b[j]


def numbers(text: str) -> list[str]:
    """Numbers in order of appearance (thousands commas removed)."""
    return re.findall(r"\d+(?:\.\d+)?", re.sub(r"(?<=\d),(?=\d{3}\b)", "", text))


def fact_pair(original: str, revised: str, update: str) -> tuple[tuple[str, str] | None, str | None]:
    """The changed quantity: the first number (in problem order) where the original problem (what the update restores)
    differs from the revised problem (what the model solved). Returns ((new, old), None) or (None, skip reason).
    new must also appear among the update's numbers."""
    a, b = numbers(original), numbers(revised)
    if len(a) != len(b):
        return None, "number counts differ between original and revised problem"
    for x, y in zip(a, b):
        if x != y:
            return ((x, y), None) if x in numbers(update) else (None, "first changed number not in the update")
    return None, "no changed number"


def end_offset(tok, continuation: str) -> int | None:
    """Tokens of the continuation before its last \\boxed (None if there is none)."""
    i = continuation.rfind("\\boxed")
    if i < 0:
        return None
    ids, offs = encode_with_offsets(tok, continuation)
    return sum(1 for a, b in offs if b <= i)


def answer_str(p) -> str | None:
    if not isinstance(p, str):
        return None
    p = p.strip().replace(",", "")
    if p.endswith(".0"):
        p = p[:-2]
    return p if NUM_RE.match(p) else None


# ---------------------------------------------------------------- select (laptop, dev only)

def select(store: Path, out: Path, cap: int | None, match: str, tok=None) -> dict:
    from selfdoubt import scoring as S
    from selfdoubt import stage0_report as R
    from selfdoubt.engines import load_tokenizer
    items_all = C.load_items()
    split, sha = C.ensure_split(items_all)
    dev = set(split["dev"]["gsm8k"])
    items = {int(r["id"]): r for r in items_all}
    recs = list(C.Store(store).recs.values())
    dev_all = {i for src in C.SOURCES for i in split["dev"][src]}
    bad = sorted({r["item_id"] for r in recs if r["item_id"] not in dev_all})
    if bad:
        raise SystemExit(f"store holds {len(bad)} non-dev items (e.g. {bad[:3]}): dev only")
    recs = [r for r in recs if r["source"] == "gsm8k"]
    assert all(r["item_id"] in dev for r in recs)
    models = {r["model"] for r in recs}
    assert models == {MODEL}, models
    tok = tok or load_tokenizer(MODEL)
    rows = R.score_model(recs, items)
    byk = {(r["item_id"], r["arm"], r["cut"]): r for r in recs}
    full = {r["item_id"]: r for r in recs if r["arm"] == "FULL"}
    cand = {c: defaultdict(list) for c in CLASSES}
    skipped = Counter()
    for r in rows:
        if r["arm"] != "U0" or r["cut"] not in C.CUTS or r["category"] not in CLASSES:
            continue
        new, old = answer_str(r["gold"]), answer_str(r["ref_pred"])
        if new is None or old is None or new == old:
            skipped["non-numeric or equal answers"] += 1
            continue
        cand[r["category"]][r["cut"]].append((r, new, old))
    n_avail = {c: {str(x): len(v) for x, v in sorted(cand[c].items())} for c in CLASSES}
    rng = random.Random(SEED)
    pick = {c: [] for c in CLASSES}
    srt = lambda v: sorted(v, key=lambda t: t[0]["item_id"])  # noqa: E731
    if match == "cut":
        # per cut the same number of rows in both classes: k_c = min(stale_c, new_c); the cap is split over cuts in
        # proportion to k_c (largest remainder), so capping keeps the two classes' cut mix identical
        k = {x: min(len(cand["stale"][x]), len(cand["new"][x])) for x in C.CUTS}
        tot = sum(k.values())
        q = dict(k)
        if cap and tot > cap:
            raw = {x: cap * k[x] / tot for x in C.CUTS}
            q = {x: int(raw[x]) for x in C.CUTS}
            for x in sorted(C.CUTS, key=lambda x: -(raw[x] - q[x]))[:cap - sum(q.values())]:
                q[x] += 1
        for x in C.CUTS:
            for c in CLASSES:
                pick[c] += rng.sample(srt(cand[c][x]), q[x])
    else:
        for c in CLASSES:
            v = [t for x in C.CUTS for t in srt(cand[c][x])]
            pick[c] = rng.sample(v, cap) if cap and len(v) > cap else v
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    lines = []
    # classes interleaved (stale, new, stale, ...) so a run stopped by its deadline is still balanced
    order = {c: sorted(pick[c], key=lambda t: (t[0]["cut"], t[0]["item_id"])) for c in CLASSES}
    seq = [(c, order[c][i]) for i in range(max(map(len, order.values()))) for c in CLASSES if i < len(order[c])]
    for c, (r, new, old) in seq:
        it = items[r["item_id"]]
        rec = byk[(r["item_id"], "U0", r["cut"])]
        fp = C.full_prompt(tok, MODEL, it["revised_problem"])
        prefix, _, _ = C.cut_trace(tok, MODEL, full[r["item_id"]]["text"], r["cut"])
        base = C.continuation_prompt(fp, prefix, "U0", it["update"])
        if C.sha256_text(base) != rec["prompt_sha256"]:
            skipped["prompt sha mismatch"] += 1
            continue
        x0pre, _, _ = C.cut_trace(tok, MODEL, full[r["item_id"]]["text"], C.X0)
        x0 = C.continuation_prompt(fp, x0pre, "U0", it["update"])
        fp_, fskip = fact_pair(it["original_problem"], it["revised_problem"], it["update"])
        lines.append({"row_id": f"{r['item_id']}|{r['cut']:.2f}", "item_id": r["item_id"], "cut": r["cut"],
                      "cls": c, "new": new, "old": old, "update": it["update"],
                      "fact_new": fp_[0] if fp_ else None, "fact_old": fp_[1] if fp_ else None, "fact_skip": fskip,
                      "base_prompt": base, "base_span": update_span(base, fp + prefix, "U0", it["update"]),
                      "x0_prompt": x0, "x0_span": update_span(x0, fp + x0pre, "U0", it["update"]),
                      "n_prompt": fp + prefix, "continuation": rec["text"],
                      "prompt_sha256": rec["prompt_sha256"]})
    out.write_text("".join(json.dumps(x) + "\n" for x in lines))
    n = Counter(x["cls"] for x in lines)
    summ = {"store": str(store), "split_sha256": sha, "match": match, "cap": cap, "available": n_avail,
            "n_rows": dict(n), "cuts": {c: dict(Counter(f"{x['cut']:.1f}" for x in lines if x["cls"] == c))
                                        for c in CLASSES},
            "n_items": len({x["item_id"] for x in lines}), "skipped": dict(skipped),
            "fact": {c: dict(Counter(x["fact_skip"] or "ok" for x in lines if x["cls"] == c)) for c in CLASSES},
            "gate": {c: n[c] >= GATE for c in CLASSES}, "gate_pass": all(n[c] >= GATE for c in CLASSES),
            "rows_sha256": hashlib.sha256(out.read_bytes()).hexdigest()}
    out.with_suffix(".summary.json").write_text(json.dumps(summ, indent=1) + "\n")
    return summ


# ---------------------------------------------------------------- run (model)

class Runner:
    """Teacher-forced forwards with residual-stream capture and patching at chosen positions and layers."""

    def __init__(self, model_name: str, path: str | None, device: str, dtype: str, model=None, tok=None):
        import torch
        self.torch = torch
        if model is None:
            from transformers import AutoModelForCausalLM
            from selfdoubt.engines import load_tokenizer
            tok = load_tokenizer(model_name, path)
            dt = {"bfloat16": torch.bfloat16, "float32": torch.float32, "float16": torch.float16}[dtype]
            kw = {} if path else {"revision": C.MODELS[model_name]}
            kw["dtype" if _tf_major() >= 5 else "torch_dtype"] = dt
            model = AutoModelForCausalLM.from_pretrained(path or model_name, **kw)
        self.tok, self.model = tok, model
        self.model.to(device).eval()
        self.device = device
        inner = self.model.model
        self.embed, self.layers, self.norm = inner.embed_tokens, inner.layers, inner.norm
        self.head = self.model.lm_head
        self.n_layers = len(self.layers)
        self._capture = None      # {"pos": [...], "store": {l: tensor}} or None
        self._patch = None        # (l, pos list, tensor [len(pos), d]) or None
        self._hooks = [self.embed.register_forward_hook(self._hook(0))]
        self._hooks += [m.register_forward_hook(self._hook(i + 1)) for i, m in enumerate(self.layers)]

    def _hook(self, l):
        def f(_m, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if self._patch is not None and self._patch[0] == l:
                _, pos, val = self._patch
                h = h.clone()
                h[0, pos] = val.to(h.dtype)
            if self._capture is not None:
                self._capture["store"][l] = h[0].detach().float().clone()
            if isinstance(out, tuple):
                return (h,) + tuple(out[1:])
            return h
        return f

    def prefix_cache(self, ids):
        from transformers import DynamicCache
        cache = DynamicCache()
        if ids:
            with self.torch.no_grad():
                self.model(input_ids=self.torch.tensor([ids], device=self.device), past_key_values=cache, use_cache=True,
                           logits_to_keep=1)
        return cache

    def forward(self, suffix, cache=None, plen=0, capture=False, patch=None):
        """Logits at the last position, and (capture) the residual stream at every suffix position for every l."""
        torch = self.torch
        self._capture = {"store": {}} if capture else None
        self._patch = patch
        try:
            with torch.no_grad():
                kw = {"past_key_values": cache, "use_cache": True} if cache is not None else {"use_cache": False}
                pos = torch.arange(plen, plen + len(suffix), device=self.device)[None]
                out = self.model(input_ids=torch.tensor([suffix], device=self.device), position_ids=pos,
                                 logits_to_keep=1, **kw)       # only the last position's logits are used
            logits = out.logits[0, -1].float()
            hs = self._capture["store"] if capture else None
        finally:
            self._capture, self._patch = None, None
            if cache is not None:
                cache.crop(plen)
        return logits, hs

    def lens(self, h_last):
        """Logit lens: final norm + unembedding of residual vectors [n, d] -> logits [n, vocab]."""
        torch = self.torch
        with torch.no_grad():
            x = self.norm(h_last.to(self.head.weight.dtype))
            return self.head(x).float()


def _tf_major() -> int:
    import transformers
    return int(transformers.__version__.split(".")[0])


def measure_row(rn: Runner, row: dict, offsets=OFFSETS, layers=None) -> list[dict]:
    """One record per (target, offset) that ran, or one {"skip": reason} record for the row (or the target/offset).
    Targets: "ans" (READOUT, new vs old answer) and "fact" (FACT_READOUT, the update's changed number vs the number it
    replaces; only when select found one). Offsets: ints = tokens of the row's continuation; "end" = before its last
    \\boxed, base and donor only (descriptive)."""
    torch = rn.torch
    tok = rn.tok
    L = list(range(rn.n_layers + 1)) if layers is None else list(layers)
    base_ids, offs = encode_with_offsets(tok, row["base_prompt"])
    assert base_ids == tok(row["base_prompt"], add_special_tokens=False)["input_ids"]
    D = digit_positions(tok, base_ids, offs, tuple(row["base_span"]))
    head = {"row_id": row["row_id"], "item_id": row["item_id"], "cut": row["cut"], "cls": row["cls"]}
    if not D:
        return [dict(head, skip="no digit tokens in the update")]
    donor_str = swap_digits(row["base_prompt"], tuple(row["base_span"]), f"stage2|{row['row_id']}")
    donor_ids = tok(donor_str, add_special_tokens=False)["input_ids"]
    if len(donor_ids) != len(base_ids):
        return [dict(head, skip="donor token count differs")]
    if [i for i, (a, b) in enumerate(zip(base_ids, donor_ids)) if a != b] != D:
        return [dict(head, skip="donor differs outside the digit positions")]
    x0_ids, x0_offs = encode_with_offsets(tok, row["x0_prompt"])
    D0 = digit_positions(tok, x0_ids, x0_offs, tuple(row["x0_span"]))
    x0_ok = len(D0) == len(D) and [x0_ids[i] for i in D0] == [base_ids[i] for i in D]
    cont = tok(row["continuation"], add_special_tokens=False)["input_ids"]
    k_end = end_offset(tok, row["continuation"])
    targets = [("ans", READOUT, "}", row["new"], row["old"])]
    if row.get("fact_new"):
        targets.append(("fact", FACT_READOUT, ".", row["fact_new"], row["fact_old"]))
    out = []
    if not row.get("fact_new"):
        out.append(dict(head, target="fact", skip=f"fact: {row.get('fact_skip')}"))
    if not x0_ok:
        out.append(dict(head, skip_x0="x0 digit tokens do not align (patch_x0 left empty)"))

    p0 = D[0]
    Drel = [d - p0 for d in D]
    cache = rn.prefix_cache(base_ids[:p0])
    Hx0 = None
    if x0_ok:
        _, hx0 = rn.forward(x0_ids, capture=True)
        Hx0 = {l: hx0[l][D0] for l in L}
    n_ids = tok(row["n_prompt"], add_special_tokens=False)["input_ids"]

    def rank(lg, t):
        return int((lg > lg[t]).sum())

    for tname, ro, close, new, old in targets:
        try:
            force, t_new, t_old = readout_tokens(tok, new, old, ro, close)
        except AssertionError:
            out.append(dict(head, target=tname, skip="values give the same tokens"))
            continue

        def margin(lg):
            return float(lg[t_new] - lg[t_old])

        def lens(hs, idx=-1):
            lg = rn.lens(torch.stack([hs[l][idx] for l in range(rn.n_layers + 1)]))
            return ([round(float(x), 4) for x in lg[:, t_new] - lg[:, t_old]],
                    [round(float(x), 4) for x in lg[:, t_new]], [round(float(x), 4) for x in lg[:, t_old]])

        _, hn = rn.forward(n_ids + force, capture=True)
        lens_n = lens(hn)[0]
        for k in offsets:
            kk = k_end if k == "end" else k
            if kk is None or kk > len(cont):
                out.append(dict(head, target=tname, offset=k, skip=f"offset {k} not available"))
                continue
            tail = cont[:kk] + force
            sb, sd = base_ids[p0:] + tail, donor_ids[p0:] + tail
            t0 = time.time()
            lb, hb = rn.forward(sb, cache, p0, capture=True)
            ld, hd = rn.forward(sd, cache, p0, capture=True)
            lbase, ldonor = lens(hb), lens(hd)
            res = dict(head, target=tname, offset=k, k_tokens=kk, n_digits=len(D), seq_len=p0 + len(sb),
                       tok_new=tok.decode([t_new]), tok_old=tok.decode([t_old]),
                       m_base=round(margin(lb), 4), m_donor=round(margin(ld), 4),
                       base_logit_new=round(float(lb[t_new]), 4), base_logit_old=round(float(lb[t_old]), 4),
                       base_rank_new=rank(lb, t_new), base_rank_old=rank(lb, t_old),
                       donor_logit_new=round(float(ld[t_new]), 4), donor_logit_old=round(float(ld[t_old]), 4),
                       donor_rank_new=rank(ld, t_new), donor_rank_old=rank(ld, t_old),
                       lens_base=lbase[0], lens_base_new=lbase[1], lens_base_old=lbase[2],
                       lens_donor=ldonor[0], lens_n=lens_n, layers=L, patch_swap=[], patch_rand=[], patch_x0=[])
            if k != "end":
                for l in L:
                    hb_l, hd_l = hb[l][Drel], hd[l][Drel]
                    res["patch_swap"].append(round(margin(rn.forward(sb, cache, p0, patch=(l, Drel, hd_l))[0]), 4))
                    g = torch.Generator().manual_seed(int(hashlib.sha256(
                        f"rand|{row['row_id']}|{tname}|{k}|{l}".encode()).hexdigest()[:8], 16))
                    r = torch.randn(hd_l.shape, generator=g)
                    r = r / r.norm(dim=-1, keepdim=True) * (hd_l - hb_l).norm(dim=-1, keepdim=True).cpu()
                    res["patch_rand"].append(
                        round(margin(rn.forward(sb, cache, p0, patch=(l, Drel, hb_l + r.to(hb_l.device)))[0]), 4))
                    if Hx0 is not None:
                        res["patch_x0"].append(round(margin(rn.forward(sb, cache, p0, patch=(l, Drel, Hx0[l]))[0]), 4))
            res["secs"] = round(time.time() - t0, 2)
            out.append(res)
    return out


def run(rows_path: Path, out: Path, model: str, path: str | None, device: str, dtype: str, offsets, deadline_s,
        limit: int | None, layers=None) -> dict:
    rows = [json.loads(x) for x in rows_path.read_text().splitlines() if x.strip()]
    if limit:
        rows = rows[:limit]
    done = set()
    if out.exists():
        for x in out.read_text().splitlines():
            try:
                done.add(json.loads(x)["row_id"])
            except (json.JSONDecodeError, KeyError):
                pass
    rn = Runner(model, path, device, dtype)
    t_end = time.time() + deadline_s if deadline_s else None
    n = 0
    t0 = time.time()
    for row in rows:
        if row["row_id"] in done:
            continue
        if t_end and time.time() > t_end:
            print(f"deadline: {len([r for r in rows if r['row_id'] not in done]) - n} rows left (resumable)")
            break
        recs = measure_row(rn, row, offsets, layers)
        with out.open("a") as f:      # one row (all offsets) per write; a torn line is ignored on resume
            for r in recs:
                f.write(json.dumps(r) + "\n")
            f.flush()
            os.fsync(f.fileno())
        n += 1
        print(f"row {row['row_id']} {row['cls']}: {time.time() - t0:.0f} s, {n} done", flush=True)
    return {"rows_done_now": n, "secs": round(time.time() - t0, 1)}


# ---------------------------------------------------------------- report (descriptive only; no verdicts)

def report(res_path: Path, n_boot: int = 2000) -> dict:
    """Descriptive tables per (target, offset, class); no verdicts (the Advisor freezes readings). Skips are counted by
    reason and class. patch_x0 is a secondary control (positions and context differ, not a clean null)."""
    import numpy as np
    recs = []
    for x in res_path.read_text().splitlines():
        try:
            recs.append(json.loads(x))
        except json.JSONDecodeError:
            pass
    ok = [r for r in recs if "m_base" in r]
    out = {"n_rows": len({r["row_id"] for r in recs}),
           "skips": dict(Counter(f"{r.get('target', 'row')}|{r['skip']}|{r['cls']}" for r in recs if "skip" in r)),
           "skips_x0": dict(Counter(r["cls"] for r in recs if "skip_x0" in r)), "tables": {}}

    def mean_lists(rs, key, sub=None):
        v = [np.array(r[key], dtype=float) - (r[sub] if sub else 0) for r in rs if r.get(key)]
        return np.mean(v, 0).round(3).tolist() if v else None

    for t in sorted({r["target"] for r in ok}):
        for k in sorted({r["offset"] for r in ok if r["target"] == t}, key=str):
            rs = [r for r in ok if r["target"] == t and r["offset"] == k]
            d = {}
            for c in CLASSES:
                cr = [r for r in rs if r["cls"] == c]
                if not cr:
                    continue
                f = lambda key: float(np.mean([r[key] for r in cr]))  # noqa: E731
                d[c] = {"n_rows": len(cr), "n_items": len({r["item_id"] for r in cr}),
                        "m_base": f("m_base"), "swap_effect": f("m_donor") - f("m_base"),
                        "base_logit_new": f("base_logit_new"), "base_logit_old": f("base_logit_old"),
                        "base_rank_new_median": float(np.median([r["base_rank_new"] for r in cr])),
                        "base_rank_old_median": float(np.median([r["base_rank_old"] for r in cr])),
                        "lens_base": mean_lists(cr, "lens_base"), "lens_base_new": mean_lists(cr, "lens_base_new"),
                        "lens_base_old": mean_lists(cr, "lens_base_old"), "lens_n": mean_lists(cr, "lens_n"),
                        "patch_swap_minus_base": mean_lists(cr, "patch_swap", "m_base"),
                        "patch_rand_minus_base": mean_lists(cr, "patch_rand", "m_base"),
                        "patch_x0_minus_base_secondary": mean_lists(cr, "patch_x0", "m_base")}
            if all(c in d for c in CLASSES):
                by_item = defaultdict(lambda: {c: [] for c in CLASSES})
                for r in rs:
                    by_item[r["item_id"]][r["cls"]].append(r["m_donor"] - r["m_base"])
                ids = sorted(by_item)
                rng = np.random.default_rng(SEED)
                stats = []
                for _ in range(n_boot):
                    pick = rng.choice(ids, len(ids))
                    a = [v for i in pick for v in by_item[i]["stale"]]
                    b = [v for i in pick for v in by_item[i]["new"]]
                    if a and b:
                        stats.append(np.mean(a) - np.mean(b))
                d["swap_effect_stale_minus_new"] = {
                    "point": d["stale"]["swap_effect"] - d["new"]["swap_effect"],
                    "ci95": [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))] if stats else None}
            out["tables"][f"{t}|{k}"] = d
    secs = [r["secs"] for r in ok if r["offset"] != "end"]
    out["secs_per_patched_record"] = {"mean": float(np.mean(secs)) if secs else None,
                                      "max": max(secs) if secs else None}
    return out


def parse_offsets(s: str):
    return [x if x == "end" else int(x) for x in s.split(",") if x]


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--store", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--cap", type=int, default=80, help="max rows per class (seeded sample)")
    s.add_argument("--match", choices=["cut", "none"], default="cut",
                   help="cut: per cut, as many successes as failures (same cut mix in both classes)")
    r = sub.add_parser("run")
    r.add_argument("--rows", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--model", default=MODEL)
    r.add_argument("--snapshot", default=None)
    r.add_argument("--device", default="cuda")
    r.add_argument("--dtype", default="bfloat16")
    r.add_argument("--offsets", default=",".join(map(str, OFFSETS)))
    r.add_argument("--layers", default=None, help="comma list (test only); default all")
    r.add_argument("--deadline-s", type=float, default=None)
    r.add_argument("--limit", type=int, default=None)
    p = sub.add_parser("report")
    p.add_argument("--res", required=True)
    p.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args(argv)
    if a.cmd == "select":
        print(json.dumps(select(Path(a.store), Path(a.out), a.cap, a.match), indent=1))
    elif a.cmd == "run":
        lay = [int(x) for x in a.layers.split(",")] if a.layers else None
        print(json.dumps(run(Path(a.rows), Path(a.out), a.model, a.snapshot, a.device, a.dtype,
                             parse_offsets(a.offsets), a.deadline_s, a.limit, lay)))
    else:
        rep = report(Path(a.res), a.n_boot)
        Path(a.res).with_suffix(".report.json").write_text(json.dumps(rep, indent=1) + "\n")
        print(json.dumps({k: v for k, v in rep.items() if k != "tables"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
