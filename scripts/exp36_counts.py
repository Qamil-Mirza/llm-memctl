"""§36 volume counts for the pod estimate: per reader and item, the number of verifier prompts and their prefill tokens
on every fold (counts of inputs, no model output), and the counterfactual qualifying count on the development folds.

    .venv/bin/python scripts/exp36_counts.py [--out runs/exp36_stub/counts.json] [--handcheck runs/exp36_stub/cf_handcheck_dev.md]

Tokens: the locally cached Qwen2.5 tokenizer (same vocabulary and chat template as the 3B and 7B); raw input =
ids(chat text) + ids(continuation) [+ ids(suffix) per C4 pass, both passes summed]. --handcheck writes the first 10
development-fold counterfactual substitutions with their context, for the hand check the proposal asks for.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from memctl.analysis import verification as ver  # noqa: E402
from scripts.exp36_verify import ALL_FOLDS, ITEMS_BY_READER, Sources, StubEngine, build_inputs  # noqa: E402

DEV = (0, 1)


def tokens(engine, x) -> int:
    chat = engine.encode(engine.chat_text(x.prompt))
    base = len(chat) + len(engine.encode(x.continuation))
    if not x.suffixes:
        return base
    return sum(base + len(engine.encode(s)) for s in x.suffixes)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None)
    parser.add_argument("--handcheck", default=None)
    args = parser.parse_args(argv)
    src = Sources()
    engine = StubEngine("count")
    if engine.tokenizer is None:
        raise SystemExit("no cached Qwen2.5 tokenizer; token counts need one")
    table: dict = {}
    cache: dict = {}
    for reader, items in ITEMS_BY_READER.items():
        for item in items:
            key_reader = reader if item in ("V_own", "C4") else "qwen3b"
            per_fold = {}
            for fold in ALL_FOLDS:
                key = (key_reader, item, fold)
                if key not in cache:
                    inputs, errors, skipped = build_inputs(item, reader, fold, src)
                    lengths = [tokens(engine, x) for x in inputs]
                    cache[key] = {"prompts": len(inputs), "passes": len(inputs) * (2 if item == "C4" else 1),
                                  "tokens": sum(lengths), "max_tokens": max(lengths, default=0), "errors": len(errors),
                                  **skipped}
                per_fold[fold] = cache[key]
            table.setdefault(reader, {})[item] = {
                "by_fold": per_fold,
                "prompts": sum(v["prompts"] for v in per_fold.values()),
                "passes": sum(v["passes"] for v in per_fold.values()),
                "tokens": sum(v["tokens"] for v in per_fold.values()),
                "max_tokens": max(v["max_tokens"] for v in per_fold.values()),
                "errors": sum(v["errors"] for v in per_fold.values())}
    cf_dev = sum(cache[("qwen3b", "CF", f)]["prompts"] for f in DEV) // 4
    print("| reader | item | prompts (passes) | prefill tokens | max tokens | errors |")
    print("|---|---|---|---|---|---|")
    for reader, items in table.items():
        for item, v in items.items():
            print(f"| {reader} | {item} | {v['prompts']} ({v['passes']}) | {v['tokens']:,} | {v['max_tokens']} | {v['errors']} |")
        total_p = sum(v["passes"] for v in items.values())
        total_t = sum(v["tokens"] for v in items.values())
        print(f"| {reader} | **total** | {total_p} passes | {total_t:,} | | |")
    print(f"\ncounterfactual: {cf_dev} qualifying questions on development folds 0-1 "
          f"(fold 0 {cache[('qwen3b', 'CF', 0)]['prompts'] // 4}, fold 1 {cache[('qwen3b', 'CF', 1)]['prompts'] // 4})")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"table": table, "cf_dev_qualifying": cf_dev}, indent=1))
    if args.handcheck:
        lines = ["# §36 counterfactual hand check (development folds 0-1, first 10 qualifying questions)", ""]
        shown = 0
        for fold in DEV:
            inputs, _, _ = build_inputs("CF", "qwen3b", fold, src)
            for x in inputs:
                if x.variant != "orig@cf" or shown >= 10:
                    continue
                e, rec = x.extra, src.rec["qwen3b"][x.question_id]
                cf_text = ver.counterfactual(rec.prompt, rec.memory_lines, rec.evidence_lines, rec.gold, src.name_pool,
                                             x.question_id)["prompt"]
                at = e["start"]
                lines += [f"## {x.question_id} (fold {fold}, {e['kind']}): {e['original']!r} -> {e['substitute']!r}; "
                          f"gold {rec.gold!r}; other occurrences in memory {e['outside']}", "",
                          "intact: `" + rec.prompt[max(0, e['start'] - 120):e['end'] + 80].replace("\n", " / ") + "`", "",
                          "counterfactual: `" + cf_text[max(0, at - 120):at + len(e['substitute']) + 80].replace("\n", " / ")
                          + "`", ""]
                shown += 1
        Path(args.handcheck).write_text("\n".join(lines) + "\n")
        print(f"hand-check file: {args.handcheck} ({shown} substitutions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
