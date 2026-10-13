"""Classify the §34 tracked gold token per question and write question-id lists for exploratory subset reports.

    .venv/bin/python scripts/exp34_gold_kinds.py [--root runs/exp34_jlens] [--reader qwen3b]

Rule (fixed before any subset number was looked at, 2026-10-13): the tracked token is the gold's first token after
leading whitespace (`gold_token_text` in each read file). STRICT = a digit or currency sign, a number word, Yes/No, or
a capitalised word that is not a sentence opener. FUNCTION = everything else (sentence openers like "The"/"I"/"You",
prepositions, articles, fragments). Writes <root>/qids_strict.txt and <root>/qids_function.txt plus a count table.
"""
import argparse
import collections
import json
import re
from pathlib import Path

OPENERS = {"the", "a", "an", "i", "you", "he", "she", "it", "they", "we", "my", "your", "his", "her", "their", "our",
           "in", "on", "at", "to", "for", "with", "of", "by", "from", "about", "around", "over", "there", "this",
           "that", "when", "after", "before", "during", "it's", "its", "as", "if", "one", "some", "both", "all"}
NUMBER_WORDS = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
                "twelve", "thirteen", "fourteen", "fifteen", "twenty", "thirty", "forty", "fifty", "hundred",
                "first", "second", "third", "half", "twice"}


def kind(token: str) -> str:
    s = token.strip()
    low = s.lower()
    if not s:
        return "function"
    if re.fullmatch(r"[\d$€£%.,:/-]+", s):
        return "strict"
    if low in NUMBER_WORDS or low in {"yes", "no"}:
        return "strict"
    if s[0].isupper() and low not in OPENERS and len(s) > 1:
        return "strict"
    return "function"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="runs/exp34_jlens")
    parser.add_argument("--reader", default="qwen3b")
    args = parser.parse_args()
    root = Path(args.root)
    by_kind: dict[str, list[str]] = collections.defaultdict(list)
    examples: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for path in sorted((root / args.reader).glob("*.json")):
        if path.name.startswith("_"):
            continue
        token = json.loads(path.read_text()).get("gold_token_text")
        if token is None:
            continue
        k = kind(token)
        by_kind[k].append(path.stem)
        examples[k][token] += 1
    for k in ("strict", "function"):
        (root / f"qids_{k}.txt").write_text("\n".join(by_kind[k]) + "\n")
        print(f"{k}: {len(by_kind[k])} questions; most common tokens: {examples[k].most_common(12)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
