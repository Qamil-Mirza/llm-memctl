"""§30, development side only: the turn-size distribution, and the oracle (evidence-only) prompt under a turn cap.
    python -I configs/sweeps/exp30/exp30_turns.py <worktree> <data file>"""
import statistics as st
import sys

sys.path.insert(0, sys.argv[1])
from memctl.envs.utilmem import DOMAINS, _bundle, parse_question, question_pool  # noqa: E402
from memctl.memory.items import count_tokens, label_prefix, truncate_tokens  # noqa: E402

path = sys.argv[2]
pool = question_pool(path, "dev")
sizes, by_domain, evid_sizes = [], {}, []
seen = set()
for q in pool:
    bundle = _bundle(path, q["bundle"])
    if q["bundle"] not in seen:
        seen.add(q["bundle"])
        for meta, session in zip(bundle["haystack_session_meta"], bundle["haystack_sessions"]):
            for turn in session:
                n = count_tokens(turn["content"])
                sizes.append(n)
                by_domain.setdefault((meta["domain"], meta["role"], turn["role"]), []).append(n)
sizes.sort()


def pct(values, p):
    return values[min(len(values) - 1, int(p * len(values)))]


print("dev turns", len(sizes), "median", pct(sizes, .5), "p90", pct(sizes, .9), "p95", pct(sizes, .95),
      "p99", pct(sizes, .99), "max", sizes[-1], "mean", round(st.mean(sizes)))
for cap in (500, 1000, 2000):
    print(f"turns over {cap}: {sum(s > cap for s in sizes) / len(sizes):.3f}; tokens kept under a {cap} cap:",
          round(sum(min(s, cap) for s in sizes) / sum(sizes), 3))
for key in sorted(by_domain):
    v = sorted(by_domain[key])
    print(key, "n", len(v), "median", pct(v, .5), "p90", pct(v, .9), "max", v[-1])
for cap in (None, 1000, 2000):
    rows = {}
    for q in pool:
        episode, _ = parse_question(_bundle(path, q["bundle"]), q["domain"], q["number"], oracle=True, max_turn_tokens=cap)
        rows.setdefault(q["domain"], []).append(sum(count_tokens(label_prefix(t.metadata, t.source_type) + t.content)
                                                    for t in episode.turns))
    print("oracle prompt, cap", cap, {d: (int(st.median(v)), max(v)) for d, v in rows.items()},
          "over 12k:", sum(x > 12000 for v in rows.values() for x in v), "of", len(pool))
for cap in (None, 1000, 2000):
    hist = []
    for q in pool[::5]:
        episode, _ = parse_question(_bundle(path, q["bundle"]), q["domain"], q["number"], max_turn_tokens=cap)
        hist.append(sum(count_tokens(label_prefix(t.metadata, t.source_type) + t.content) for t in episode.turns))
    print("history, cap", cap, "median", int(st.median(hist)), "5% budget median", int(0.05 * st.median(hist)))
