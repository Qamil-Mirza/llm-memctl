"""§30, development side only (20 bundles, seed 30): where each question sits relative to its evidence, and the
sizes that set the cost. Labels and text only; no reader, no outcome.
    python -I configs/sweeps/exp30/exp30_structure.py <worktree> <data file> <out.json>"""
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
from memctl.envs.utilmem import DOMAINS, _bundle, parse_question, question_pool  # noqa: E402
from memctl.memory.items import count_tokens, label_prefix  # noqa: E402

path, out = sys.argv[2], Path(sys.argv[3])
rows = []
for q in question_pool(path, "dev"):
    bundle = _bundle(path, q["bundle"])
    episode, groups = parse_question(bundle, q["domain"], q["number"])
    size = [count_tokens(label_prefix(t.metadata, t.source_type) + t.content) for t in episode.turns]
    total = sum(size)
    after = {}  # tokens that arrive after each turn (its distance to the question)
    running = 0
    for turn, n in zip(reversed(episode.turns), reversed(size)):
        after[turn.id] = running
        running += n
    position = {t.id: k for k, t in enumerate(episode.turns)}
    newest = [min(after[i] for i in ids) for ids in groups]  # distance of each session's last turn
    oldest = [max(after[i] + size[position[i]] for i in ids) for ids in groups]
    evidence_tokens = sum(size[position[i]] for ids in groups for i in ids)
    oracle, _ = parse_question(bundle, q["domain"], q["number"], oracle=True)
    rows.append({"domain": q["domain"], "sessions": len(groups), "history": total, "evidence": evidence_tokens,
                 "oracle_prompt": sum(count_tokens(label_prefix(t.metadata, t.source_type) + t.content) for t in oracle.turns),
                 "nearest": min(newest), "farthest": max(oldest),
                 "within_3k": sum(1 for d in newest if d < 3000), "within_6k": sum(1 for d in newest if d < 6000),
                 "evidence_turns": sum(len(ids) for ids in groups), "turns": len(episode.turns),
                 "max_turn": max(size), "question": count_tokens(episode.questions[0].question)})
out.write_text(json.dumps(rows))


def show(name, values):
    values = sorted(values)
    return f"{name} median {st.median(values):,.0f} mean {st.mean(values):,.0f} min {values[0]:,} max {values[-1]:,}"


print("dev questions", len(rows))
for domain in (*DOMAINS, None):
    part = [r for r in rows if domain in (None, r["domain"])]
    print(f"== {domain or 'all'} (n={len(part)})")
    for key in ("history", "evidence", "oracle_prompt", "nearest", "farthest", "sessions", "max_turn", "question"):
        print("  ", show(key, [r[key] for r in part]))
    print("   share of questions with any evidence session ending in the last 3k tokens:",
          round(sum(r["within_3k"] > 0 for r in part) / len(part), 3),
          "| all sessions within 6k:", round(sum(r["within_6k"] == r["sessions"] for r in part) / len(part), 3))
