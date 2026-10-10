"""§30 in-view check (labels only, stub reader): what each arm puts in the reader's view at the question.
Per question and arm: session recall (share of the question's evidence sessions with at least one turn in view),
all in view (every evidence session touched), any in view, the share of evidence turns in view, the share of
in-view turns that are same-domain distractors ("strong noise"), and the reader prompt tokens. 95% bootstrap
intervals by question (paired for differences), as §25.
    python -I configs/sweeps/exp30/exp30_inview.py <worktree> <data file> <runs root> <prefix> <out.json>"""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, sys.argv[1])
from memctl.envs.utilmem import _bundle, _index  # noqa: E402

data, root, prefix, save = sys.argv[2], Path(sys.argv[3]), sys.argv[4], Path(sys.argv[5])
ARMS = ["fifo_top5_t3000", "fixed8", "fixed16"]
bundle_of = {b["sample_id"]: b["bundle"] for b in _index(data)["questions"]}


def kinds(sample_id: str, domain: str) -> dict:
    """neutral session label (s000, ...) -> 'gt' | 'strong' (same-domain distractor) | 'weak' (other domain)."""
    bundle = _bundle(data, bundle_of[sample_id])
    order = sorted(range(len(bundle["haystack_sessions"])), key=lambda n: bundle["haystack_dates"][n])
    evidence = set(bundle["answer_session_ids"][domain])
    found = {}
    for rank, position in enumerate(order):
        meta = bundle["haystack_session_meta"][position]
        sid = bundle["haystack_session_ids"][position]
        found[f"s{rank:03d}"] = "gt" if sid in evidence else ("strong" if meta["domain"] == domain else "weak")
    return found


rows = defaultdict(dict)  # arm -> question_id -> measures
for cell in sorted(root.glob(f"{prefix}_s*/*/")):
    arm = cell.name.split("__")[0]
    if arm not in ARMS:
        continue
    question_of = {}
    for line in open(cell / "episodes.jsonl"):
        episode = json.loads(line)
        question_of[episode["episode_id"]] = episode["env_stats"]["answers"][0]["question_id"]
    evidence = {}
    for line in open(cell / "failures.jsonl"):
        row = json.loads(line)
        evidence[row["episode_id"]] = row["evidence"]
    for line in open(cell / "steps.jsonl"):
        step = json.loads(line)
        if not step["requires_response"]:
            continue
        qid = question_of[step["episode_id"]]
        sample_id, domain, _ = qid.split("/")
        label = kinds(sample_id, domain)
        reqs = evidence[step["episode_id"]]
        touched = [bool(r["active_carriers"]) for r in reqs]
        active = [i for i in step["memory"]["active_ids"] if not i.startswith("q")]
        kind = [label.get(i.split(":")[0], "other") for i in active]
        rows[arm][qid] = {
            "domain": domain, "sessions": len(reqs), "recall": sum(touched) / len(touched), "all": int(all(touched)),
            "any": int(any(touched)),
            "turn_share": sum(len(r["active_carriers"]) for r in reqs) / sum(len(r["source_ids"]) for r in reqs),
            "strong_share": kind.count("strong") / len(kind) if kind else 0.0,
            "gt_share": kind.count("gt") / len(kind) if kind else 0.0,
            "prompt": step["agent_prompt_tokens"], "active_items": len(active),
        }
questions = sorted(rows["fifo_top5_t3000"])
for arm in ARMS:
    assert sorted(rows[arm]) == questions, (arm, len(rows[arm]), len(questions))
print("questions:", len(questions))
rng = random.Random(0)
draws = [[rng.randrange(len(questions)) for _ in questions] for _ in range(2000)]


def interval(values):
    means = sorted(sum(values[i] for i in d) / len(values) for d in draws)
    return round(sum(values) / len(values), 4), round(means[49], 4), round(means[1949], 4)


summary = {}
for measure in ("recall", "all", "any", "turn_share", "gt_share", "strong_share", "prompt", "active_items"):
    for arm in ARMS:
        summary[f"{arm}/{measure}"] = interval([rows[arm][q][measure] for q in questions])
    for arm in ("fixed8", "fixed16"):
        summary[f"{arm}-fifo/{measure}"] = interval([rows[arm][q][measure] - rows["fifo_top5_t3000"][q][measure]
                                                     for q in questions])
for name, (m, lo, hi) in summary.items():
    print(f"{name:30s} {m:+.3f} ({lo:+.3f}, {hi:+.3f})")
by_domain = {}
for arm in ARMS:
    for domain in sorted({rows[arm][q]["domain"] for q in questions}):
        part = [rows[arm][q] for q in questions if rows[arm][q]["domain"] == domain]
        by_domain[f"{arm}/{domain}"] = {k: round(sum(r[k] for r in part) / len(part), 3) for k in ("recall", "all", "any")}
        print(f"{arm:16s} {domain:14s} n={len(part):3d}", by_domain[f"{arm}/{domain}"])
save.write_text(json.dumps({"questions": questions, "summary": summary, "by_domain": by_domain, "rows": rows}))
