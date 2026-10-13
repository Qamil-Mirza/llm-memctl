"""§35 free check (development folds by default): does the reader's own note, written before its final
answer, already contain the gold, and what does it then answer?

    .venv/bin/python scripts/exp35_note_check.py --folds 0,1

The frozen §23 prompt (memctl/agents/llm.py REASONING_INSTRUCTIONS) makes every reader write a note and then
one line 'Answer: <phrase>'. Criterion D1 (string): the normalised gold string, or every gold token, occurs in
the note (the text before the last 'Answer:'). D1 is an upper bound on "derived the answer": it also counts a
note that copied the evidence line without doing the comparison or count the question asks for.

Folds 2, 3, 4 are the confirmation folds of §34/§35: do not run them before the §35 pre-registration is accepted.
"""
import argparse
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from memctl.jacobian_data import load_records  # noqa: E402

READERS = ("qwen3b", "qwen7b")


def _tokens(text: str) -> list[str]:
    return [t for t in re.sub(r"[^a-z0-9 ]", " ", text.lower()).split() if t]


def split_note(output: str) -> tuple[str | None, str]:
    """(note, answer) around the last 'Answer:'; note is None when the output has no marker."""
    i = output.rfind("Answer:")
    return (output[:i], output[i + len("Answer:"):]) if i >= 0 else (None, output)


def gold_in_note(record) -> bool | None:
    """D1; None when the output has no marker or the gold has no alphanumeric token."""
    note, _ = split_note(record.output)
    gold = _tokens(record.gold)
    if note is None or not gold:
        return None
    words = _tokens(note)
    return (" ".join(gold) in " ".join(words)) or all(t in words for t in gold)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", default="0,1")
    parser.add_argument("--samples", type=int, default=0, help="print this many refusal notes per bucket")
    args = parser.parse_args(argv)
    folds = [int(f) for f in args.folds.split(",")]
    print(f"# §35 note check, folds {folds}\n")
    print("| reader | class | n | with Answer: marker | D1: gold in own note | share of marked |")
    print("|---|---|---|---|---|---|")
    for reader in READERS:
        rows = []
        for fold in folds:
            recs, _ = load_records(reader, fold)
            rows += [r for r in recs if r.answerable and r.all_in_view and r.correct is not None]
        classes = (("refusal", [r for r in rows if r.unknown is True]),
                   ("answered wrong", [r for r in rows if r.correct is False and r.unknown is not True]),
                   ("success", [r for r in rows if r.correct is True]))
        for name, group in classes:
            marked = [r for r in group if gold_in_note(r) is not None]
            hits = [r for r in marked if gold_in_note(r)]
            share = f"{len(hits) / len(marked):.2f}" if marked else "-"
            print(f"| {reader} | {name} | {len(group)} | {len(marked)} | {len(hits)} | {share} |")
            if name == "refusal":
                by_type = Counter(r.question_type for r in hits)
                print(f"|  | refusals with D1, by type | | | {dict(by_type)} | |")
        if args.samples and reader == READERS[0]:
            refusals = classes[0][1]
            for label, pick in (("D1 true", True), ("D1 false", False)):
                print(f"\n## {reader} refusal notes, {label}\n")
                for r in [r for r in refusals if gold_in_note(r) is pick][: args.samples]:
                    note, answer = split_note(r.output)
                    print(f"- [{r.question_id}] {r.question_type}; gold {r.gold!r}\n  note: {note.strip()[-240:]!r}\n  answer: {answer.strip()[:60]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
