"""§34 alignment audit (reviewer questions 1 and 5), from the read files on disk.

    .venv/bin/python scripts/exp34_alignment_audit.py --folds 2,3,4

(1) At the reader's OWN answer position (runs/exp34_jlens_own), on judged-correct outputs that carry an `Answer:`
    marker: does the cached answer text start with the tracked gold token, and what is the gold's final-layer rank
    (model logits)? If the token and position are right, "starts with the gold token" must give rank 1 (or 2 where our
    greedy pass and the cached sample differ by one place), and every rank > 10 must be a paraphrase.
(2) Among the small reader's refusals (judge `unknown`), the large reader's outcome on the same questions: answerable by
    a model one size up, or hard for both.
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.exp34_report import _paired_outputs, gold  # noqa: E402

READERS = ("qwen3b", "qwen7b")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", default="2,3,4")
    parser.add_argument("--forced-root", default="runs/exp34_jlens")
    parser.add_argument("--own-root", default="runs/exp34_jlens_own")
    args = parser.parse_args(argv)
    folds = {int(f) for f in args.folds.split(",")}
    rf, _, _, of, bf = _paired_outputs(READERS, folds, Path(args.forced_root), None)
    ro, _, _, oo, bo = _paired_outputs(READERS, folds, Path(args.own_root), None)
    both = [q for q in bf if q in set(bo)]
    recs = {r: {x.question_id: x for x in rows} for r, rows in rf.items()}
    print(f"# alignment audit, folds {sorted(folds)}, {len(both)} paired questions\n")
    print("## (1) own position, marked, judged-correct: cached answer starts with the gold token vs final-layer rank\n")
    print("| reader | n | starts & rank 1 | starts & rank 2 | starts & rank 3-10 | starts & rank > 10 | paraphrase & rank 1 | paraphrase & rank 2-10 | paraphrase & rank > 10 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for reader in READERS:
        c = Counter()
        odd = []
        for q in both:
            e, rec = oo[reader][q], recs[reader][q]
            if not e.get("answer_marker") or rec.correct is not True:
                continue
            ans = rec.output[rec.output.rindex("Answer:") + len("Answer:"):]
            tok = e["gold_token_text"].strip()
            starts = bool(tok) and ans.lstrip().startswith(tok)
            fr = gold(e, "logit")["final_rank"]
            bucket = "1" if fr == 1 else "2" if fr == 2 else "3-10" if fr <= 10 else ">10"
            c[(starts, bucket)] += 1
            if starts and fr > 2:
                odd.append((q, tok, fr, ans[:50]))
        n = sum(c.values())
        print(f"| {reader} | {n} | {c[(True, '1')]} | {c[(True, '2')]} | {c[(True, '3-10')]} | {c[(True, '>10')]} | {c[(False, '1')]} | "
              f"{c[(False, '2')] + c[(False, '3-10')]} | {c[(False, '>10')]} |")
        for q, tok, fr, ans in odd:
            print(f"  unexplained: {q} token {tok!r} final rank {fr} answer {ans!r}")
    print("\n## (2) the small reader's refusals: the large reader's outcome on the same questions\n")
    small, large = READERS
    ref = [q for q in both if recs[small][q].unknown is True]
    lg = recs[large]
    print(f"{small} refusals: {len(ref)}; {large} correct on them: {sum(1 for q in ref if lg[q].correct is True)}; "
          f"{large} also refused: {sum(1 for q in ref if lg[q].unknown is True)}; "
          f"{large} wrong, not a refusal: {sum(1 for q in ref if lg[q].correct is False and lg[q].unknown is not True)}")
    print("by question type (all refusals):", dict(Counter(recs[small][q].question_type for q in ref).most_common()))
    missed = [q for q in ref if lg[q].correct is not True]
    print(f"by question type ({len(missed)} the large reader also missed):", dict(Counter(recs[small][q].question_type for q in missed).most_common()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
