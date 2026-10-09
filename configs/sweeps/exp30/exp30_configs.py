"""§30: write the sweep configs. Evaluation side (pod; committed to configs/sweeps/exp30/) or development side (the
free in-view check; written under runs/).
    python -I configs/sweeps/exp30/exp30_configs.py <worktree> <data file> <out dir> eval|dev [--stub]"""
import sys
from pathlib import Path

import yaml

sys.path.insert(0, sys.argv[1])
from memctl.envs.utilmem import question_pool  # noqa: E402

data, out, part = sys.argv[2], Path(sys.argv[3]), sys.argv[4]
stub = "--stub" in sys.argv
SHARDS = 5
SAMPLE = {"seed": 30, "per_domain": 100} if part == "eval" else None
HEAD = "/home/qamil-mirza/Code/llm-memctl/runs/lme_n4_head_f0_a/checkpoints/policy_best.pt"
ARMS = [
    {"name": "fifo", "label": "fifo_top5_t3000", "removal": ["MOVE_TO_ARCHIVE"],
     "retrieve": {"top_k": 5, "fit": True, "method": "lexical"}, "target_tokens": 3000},
    {"name": "rl", "label": "fixed8", "checkpoint": HEAD, "retrieval_method": "lexical", "retrieve_floor": 8,
     "retrieve_candidates": 32, "floor_head": True, "keep_none": True},
    {"name": "rl", "label": "fixed16", "checkpoint": HEAD, "retrieval_method": "lexical", "retrieve_floor": 16,
     "retrieve_candidates": 32, "floor_head": True, "keep_none": True},
]
pool = question_pool(data, part, SAMPLE)
out.mkdir(parents=True, exist_ok=True)


def model(cache):
    if stub:
        return {"backend": "stub", "name": "qwen2.5-7b-instruct", "cache_dir": cache}
    return {"backend": "openai", "name": "qwen2.5-7b-instruct", "base_url": "POD_URL/v1", "timeout_s": 600,
            "cache_dir": cache}


for kind in ("arms", "oracle"):
    for index in range(SHARDS):
        episodes = len(pool[index::SHARDS])
        env = {"name": "utilmem", "path": data, "part": part, "shard": {"k": SHARDS, "index": index},
               "max_turn_tokens": 1000}
        if SAMPLE:
            env["sample"] = dict(SAMPLE)
        if kind == "oracle":
            env["oracle"] = True
        name = f"exp30_{part}_{'oracle' if kind == 'oracle' else 'qwen7b'}_s{index}"
        base = {
            "schema_version": 1, "seed": 0, "episodes": episodes, "env": env,
            "agent": {"name": "llm", "reasoning": False, "labels": "compact", "max_new_tokens": 512,
                      "model": model(f"cache/generations_exp30_{part}_qwen7b")},
            "memory": {"allowed_operations": ["KEEP", "MOVE_TO_ARCHIVE", "RETRIEVE_FROM_ARCHIVE", "NO_OP"],
                       "embedder": "hashing", "count_labels": True},
            "hindsight": {"enabled": False},
            "logging": {"detail_episodes": 100000},
        }
        grid = ({"controller": [{"name": "full_context", "label": "oracle"}], "memory.budget.fraction": [1.0]}
                if kind == "oracle" else {"controller": ARMS, "memory.budget.fraction": [0.05]})
        header = (f"# Experiment 30 (§30, pre-registered, NOT APPROVED for spend), UtilMem {part} side, shard {index} of "
                  f"{SHARDS}, {episodes} questions"
                  + (", sample: 100 per domain from the 80 evaluation bundles, seed 30" if SAMPLE else "") + ".\n"
                  + ("# The oracle reference: the question domain's evidence sessions only, unlimited budget (the paper's"
                     " reference answer).\n" if kind == "oracle" else
                     "# Arms: FIFO + floor, fixed8, fixed16 (the §19 head, trained on LongMemEval only, zero-shot).\n"))
        (out / f"{name}.yaml").write_text(header + yaml.safe_dump({"name": name, "base": base, "grid": grid},
                                                                  sort_keys=False))
        print(name, episodes)
print("questions", len(pool))
