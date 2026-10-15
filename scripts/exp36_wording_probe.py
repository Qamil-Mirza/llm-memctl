"""§36b wording probe (CPU, development folds 0-1 only): does the 3B reply with a label under candidate wordings?

    GUARD_VM_GB=22 runs/_pipelines/guard.sh .venv/bin/python scripts/exp36_wording_probe.py [--out runs/exp36_probe]

Design: research/proposals/2026-10-15-exp36b-verifier-wording.md (commit 30415c3), with the coordinator's instruction
to run every candidate and report the full table. Candidates (memctl/analysis/verification.py PROBE_WORDINGS, texts
committed before this probe ran) in the fixed order a, b, c, d, f (W1), g (W2), h (W3), e, on the 3B's 84 dev
candidate rows (§35 arm B, not a refusal), item V (intact prompt) and item CutB (evidence-cut prompt). Chat template
with the generation prompt, plus the wording's completion prefix (d: "Label: "). transformers, fp32 on CPU (no bf16
units on this CPU), one prefill per prompt, top-20 next-token logprobs decoded by the tokenizer and scored with the
v2 scorer (primary spellings 'A' and ' A'; label_mass_tab beside, never gating). c, d and e share one prefill: d
continues c's KV cache with "Label: ", e decodes greedily from it for up to 8 tokens and takes the first standalone
A/B/C (its per-row "mass" is 1 if it has one, else 0; argmax only).

Gate (label-free, fixed in the proposal): label mass >= 0.5 on >= 0.9 of rows, on V and on CutB. The first candidate
in the order that passes both is the pick; all are run and tabulated. Selection is on format compliance only: no
AUROC, risk or other correctness-linked number is computed, and no judge verdict is read.

Folds other than 0 and 1 are refused. Rows are appended per question to <out>/rows.jsonl (resumable by question,
item, wording); <out>/summary.md and summary.json are rewritten at the end. HF_TOKEN is read from .env, never printed.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from memctl.analysis import answer_selection as sel  # noqa: E402
from memctl.analysis import verification as ver  # noqa: E402
from memctl.jacobian_data import load_records  # noqa: E402
from scripts.exp35_generate import append_jsonl, read_jsonl  # noqa: E402

MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEV = (0, 1)
ITEMS = ("V", "CutB")


def hf_token() -> str | None:
    path = Path(".env")
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        if line.startswith("HF_TOKEN="):
            return line.split("=", 1)[1].strip().strip('"').strip("'") or None
    return None


def dev_rows(folds: list[int], exp35: Path, runs: str) -> list[tuple]:
    """(question id, fold, record, candidate) for the 3B's dev candidate rows."""
    records = {r: [x for f in folds for x in load_records(r, f, runs_dir=runs)[0]] for r in ("qwen3b", "qwen7b")}
    qids, _ = sel.population(records)
    by_q = {x.question_id: x for x in records["qwen3b"]}
    b_rows = {r["question_id"]: r for f in folds for r in read_jsonl(exp35 / "qwen3b" / "B" / f"fold{f}.jsonl")}
    out = []
    for q in qids:
        cand = ver.candidate_of_row(b_rows.get(q))
        if cand is not None:
            out.append((q, by_q[q].fold, by_q[q], cand))
    return out


def is_label(token: str | None) -> bool:
    return token is not None and token.strip() in ver.LABELS


def summarise(rows: list[dict]) -> dict:
    out: dict = {"wordings": {}, "gate": {}, "pick": None}
    for key in ver.PROBE_WORDINGS:
        masses = {}
        for item in ITEMS:
            got = [r for r in rows if r["wording"] == key and r["item"] == item]
            if not got:
                continue
            mass = [r["label_mass"] for r in got]
            masses[item] = mass
            fails: dict[str, int] = {}
            for r in got:
                ok = r.get("label") is not None if key == "e" else is_label(r["top1"])
                if not ok:
                    fails[repr(r["top1"])] = fails.get(repr(r["top1"]), 0) + 1
            out["wordings"][f"{key}/{item}"] = {
                "n": len(got), "mass_ge_0.5": sum(m >= ver.PROBE_GATE_MASS for m in mass) / len(got),
                "median_mass": statistics.median(mass),
                "top1_is_label": (sum(r.get("label") is not None for r in got) if key == "e"
                                  else sum(is_label(r["top1"]) for r in got)) / len(got),
                "mass_tab_ge_0.5": (sum(r.get("label_mass_tab", 0) >= ver.PROBE_GATE_MASS for r in got) / len(got)
                                    if key != "e" else None),
                "failure_top1": sorted(fails.items(), key=lambda kv: (-kv[1], kv[0]))[:10],
                "seconds_per_prompt": statistics.mean(r["seconds"] for r in got),
                "tokens_per_prompt": statistics.mean(r["n_input_tokens"] for r in got),
                "wording_sha256": ver.probe_wording_sha256(key)}
        if len(masses) == len(ITEMS):
            out["gate"][key] = ver.probe_gate(masses["V"], masses["CutB"])
            if out["gate"][key] and out["pick"] is None:
                out["pick"] = key
    return out


def render(summary: dict, n_rows: int) -> str:
    lines = ["# §36b wording probe (3B, CPU fp32, development folds 0-1; format compliance only)", "",
             f"{n_rows} rows. Gate: label mass ≥ {ver.PROBE_GATE_MASS} on ≥ {ver.PROBE_GATE_SHARE} of rows, on V and on "
             "CutB (primary spellings only). Run-1 frozen wording on vLLM, rescored: V 0.107, CutB 0.083. Order a, b, c, "
             "d, f, g, h, e. No AUROC, risk or correctness-linked number is computed here.", "",
             f"**First candidate to pass: {summary['pick'] or 'none'}.**", "",
             "| wording | item | n | share mass ≥ 0.5 | median mass | share top-1 is a label | (tab-inclusive mass ≥ 0.5) | gate (V and CutB) | top-1 on failures (count) | s/prompt | tokens/prompt |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for key, s in summary["wordings"].items():
        wording, item = key.split("/")
        fails = ", ".join(f"{t} ({c})" for t, c in s["failure_top1"]) or "-"
        gate = summary["gate"].get(wording)
        lines.append(f"| {wording} | {item} | {s['n']} | {s['mass_ge_0.5']:.3f} | {s['median_mass']:.3f} | "
                     f"{s['top1_is_label']:.3f} | {'-' if s['mass_tab_ge_0.5'] is None else format(s['mass_tab_ge_0.5'], '.3f')} | "
                     f"{'-' if gate is None else 'PASS' if gate else 'fail'} | {fails} | {s['seconds_per_prompt']:.1f} | "
                     f"{s['tokens_per_prompt']:.0f} |")
    lines += ["", "e: the 'mass' is 1 when the greedy 8-token reply contains a standalone A/B/C, else 0; its seconds "
              "include c's prefill. d's seconds include c's prefill (d continues c's cache).", "",
              "Wording sha256: " + ", ".join(f"{k} {ver.probe_wording_sha256(k)[:12]}" for k in ver.PROBE_WORDINGS)]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", default="0,1")
    parser.add_argument("--out", default="runs/exp36_probe")
    parser.add_argument("--exp35", default="runs/exp35")
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--limit", type=int, default=0, help="first N questions only (timing check)")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args(argv)
    folds = [int(f) for f in args.folds.split(",")]
    if not set(folds) <= set(DEV):
        print(f"REFUSED: folds {sorted(set(folds) - set(DEV))} are not development folds; this probe runs on folds 0-1 only")
        return 2
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows_path = out / "rows.jsonl"
    items = dev_rows(folds, Path(args.exp35), args.runs)
    if args.limit:
        items = items[: args.limit]
    print(f"{len(items)} dev candidate rows; wordings {ver.PROBE_WORDINGS}; items {ITEMS}")

    import copy

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(args.threads)
    token = hf_token()
    tok = AutoTokenizer.from_pretrained(MODEL, token=token)
    net = AutoModelForCausalLM.from_pretrained(MODEL, token=token, dtype=torch.float32).eval()
    stop_ids = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}
    done = {(r["question_id"], r["item"], r["wording"]) for r in read_jsonl(rows_path)}

    def top20(logits) -> dict[str, float]:
        logp = torch.log_softmax(logits.float(), -1)
        values, idx = logp.topk(ver.TOP_LOGPROBS)
        top: dict[str, float] = {}
        for v, i in zip(values.tolist(), idx.tolist()):
            text_tok = tok.decode([i])
            top[text_tok] = v if text_tok not in top else max(top[text_tok], v) + math.log1p(math.exp(-abs(top[text_tok] - v)))
        return top

    def chat_ids(messages) -> tuple[str, list[int]]:
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return text, tok(text, add_special_tokens=False).input_ids

    def row(q, fold, rec, cand, item, key, text, n_tokens, seconds, top=None, generated=None):
        r = {"question_id": q, "fold": fold, "item": item, "wording": key, "wording_sha256": ver.probe_wording_sha256(key),
             "model": MODEL, "engine": "hf-cpu-fp32", "prompt_sha256": ver.sha256(text),
             "intact_prompt_sha256": ver.sha256(rec.prompt), "candidate": cand, "n_input_tokens": n_tokens,
             "seconds": seconds, "scorer_version": ver.SCORER_VERSION}
        if top is not None:
            scored = ver.label_probs(top)
            r.update({"top1": max(top, key=top.get), "label_mass": scored["label_mass"],
                      "label_mass_tab": scored["label_mass_tab"], "p_A": scored["p_A"], "p_B": scored["p_B"],
                      "p_C": scored["p_C"], "argmax": scored["argmax"],
                      "top_logprobs": sorted(([t, round(v, 4)] for t, v in top.items()), key=lambda x: -x[1])})
        else:
            label = ver.first_label(generated)
            r.update({"generated": generated, "label": label, "argmax": label, "top1": generated,
                      "label_mass": 1.0 if label else 0.0})
        return r

    began = time.perf_counter()
    for n, (q, fold, rec, cand) in enumerate(items):
        prompts = {"V": rec.prompt, "CutB": ver.cut_prompt(rec.prompt, rec.memory_lines, rec.evidence_lines)}
        new_rows = []
        for item in ITEMS:
            todo = [k for k in ver.PROBE_WORDINGS if (q, item, k) not in done]
            for key in [k for k in todo if k in ("a", "b", "f", "g", "h")]:
                messages, _ = ver.probe_messages(prompts[item], cand, key)
                text, ids = chat_ids(messages)
                t0 = time.perf_counter()
                with torch.no_grad():
                    logits = net(input_ids=torch.tensor([ids]), logits_to_keep=1).logits[0, -1]
                new_rows.append(row(q, fold, rec, cand, item, key, text, len(ids), time.perf_counter() - t0, top20(logits)))
            if any(k in todo for k in ("c", "d", "e")):
                messages, _ = ver.probe_messages(prompts[item], cand, "c")
                text, ids = chat_ids(messages)
                t0 = time.perf_counter()
                with torch.no_grad():
                    out = net(input_ids=torch.tensor([ids]), use_cache=True, logits_to_keep=1)
                    t_c = time.perf_counter() - t0
                    cache, logits_c = out.past_key_values, out.logits[0, -1]
                    if "c" in todo:
                        new_rows.append(row(q, fold, rec, cand, item, "c", text, len(ids), t_c, top20(logits_c)))
                    if "d" in todo:
                        t1 = time.perf_counter()
                        prefix = tok(ver.PROBE_LABEL_PREFIX, add_special_tokens=False).input_ids
                        out_d = net(input_ids=torch.tensor([prefix]), past_key_values=copy.deepcopy(cache), use_cache=True,
                                    logits_to_keep=1)
                        new_rows.append(row(q, fold, rec, cand, item, "d", text + ver.PROBE_LABEL_PREFIX, len(ids) + len(prefix),
                                            t_c + time.perf_counter() - t1, top20(out_d.logits[0, -1])))
                    if "e" in todo:
                        t1 = time.perf_counter()
                        gen, nxt, cache_e = [], int(logits_c.argmax()), cache
                        for _ in range(ver.PROBE_E_MAX_NEW_TOKENS):
                            if nxt in stop_ids:
                                break
                            gen.append(nxt)
                            step = net(input_ids=torch.tensor([[nxt]]), past_key_values=cache_e, use_cache=True)
                            cache_e, nxt = step.past_key_values, int(step.logits[0, -1].argmax())
                        new_rows.append(row(q, fold, rec, cand, item, "e", text, len(ids), t_c + time.perf_counter() - t1,
                                            generated=tok.decode(gen)))
        append_jsonl(rows_path, new_rows)
        print(f"{n + 1}/{len(items)} questions, {time.perf_counter() - began:.0f} s", flush=True)
    rows = read_jsonl(rows_path)
    summary = summarise(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    (out / "summary.md").write_text(render(summary, len(rows)))
    print(render(summary, len(rows)))
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    raise SystemExit(main())
