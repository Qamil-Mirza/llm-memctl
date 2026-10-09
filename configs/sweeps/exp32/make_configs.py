"""Writes the §32 sweep configs (run once; the YAML files are committed).
    python configs/sweeps/exp32/make_configs.py

- exp32_free_f{k}.yaml: the reader-free pass. Stub reader (the real 7b5fc30 prompt is built and counted), stub judge,
  the five test folds, all arms. It also writes the cross-encoder score caches the reader pass replays.
- exp32_latency.yaml: selection latency, one process, one thread, fold 0's test part, fresh caches.
- exp32_qwen7b_f{k}.yaml: the reader pass (NOT RUN; needs spend and the user's approval). The §27 Qwen 7B reader
  config with the same arms; the cross-encoder arms replay the free pass's scores (cache_only).
"""
import copy
from pathlib import Path

import yaml

HERE = Path(__file__).parent
HEAD = "/home/qamil-mirza/Code/llm-memctl/runs/lme_n4_head_f{k}_a/checkpoints/policy_best.pt"
SLOT = {"retrieval_method": "lexical", "retrieve_floor": 8, "retrieve_candidates": 32, "floor_head": True,
        "keep_none": True}
CE = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def arms(k: int, cache_only: bool, fresh: bool = False) -> list[dict]:
    ce_cache = (lambda name: None) if fresh else (lambda name: f"cache/exp32_ce/{name}_f{k}.jsonl")
    ce = lambda label, model, name: {"name": "rerank", "label": label, **SLOT, "scorer": {
        "kind": "cross_encoder", "model": model, "cache": ce_cache(name), "cache_only": cache_only and not fresh}}
    return [
        {"name": "rl", "label": "fixed8", "checkpoint": HEAD.format(k=k), **SLOT},
        {"name": "rerank", "label": "bm25_top8", **SLOT, "scorer": {"kind": "bm25"}},
        {"name": "rerank", "label": "rrf_top8", **SLOT, "scorer": {"kind": "rrf", "model": "BAAI/bge-small-en-v1.5"}},
        {"name": "rerank", "label": "lr_pointwise", **SLOT,
         "scorer": {"kind": "pointwise", "model": f"configs/sweeps/exp32/models/lr_f{k}.json"}},
        {"name": "rerank", "label": "gbdt_pointwise", **SLOT,
         "scorer": {"kind": "pointwise", "model": f"configs/sweeps/exp32/models/gbdt_f{k}.pkl"}},
        ce("cross_encoder_zero", CE, "zero"),
        # cross_encoder_tuned was dropped before any test-fold run: CPU fine-tuning is not feasible in minutes here
        # (EXPERIMENTS.md §32, "Cross-encoder fine-tuning dropped").
    ]


def base(k: int, part: str = "test", reader: bool = False) -> dict:
    model = ({"backend": "openai", "name": "qwen2.5-7b-instruct", "base_url": "POD_URL/v1", "timeout_s": 300,
              "cache_dir": "cache/generations_exp32_qwen7b"} if reader else {"backend": "stub"})
    return {
        "schema_version": 1, "seed": 0, "episodes": 100,
        "env": {"name": "longmemeval", "path": "data/longmemeval/longmemeval_s_cleaned.json",
                "folds": {"k": 5, "fold": k, "part": part}, "judge": {"backend": "stub", "style": "longmemeval"}},
        "agent": {"name": "llm", "reasoning": True, "labels": "compact", "max_new_tokens": 256, "model": model},
        "memory": {"allowed_operations": ["KEEP", "MOVE_TO_ARCHIVE", "RETRIEVE_FROM_ARCHIVE", "NO_OP"],
                   "embedder": "hashing", "count_labels": True},
        "logging": {"detail_episodes": 100000},
    }


HEADER = {
    "free": "# Experiment 32 (§32) reader-free pass, LongMemEval fold {k}, test part. No reader: a stub model answers so\n"
            "# the real prompt is built and counted. Every arm ranks the same 32 BM25 candidates (keep_none) and shows 8.\n",
    "reader": "# Experiment 32 (§32) reader pass, NOT APPROVED (needs spend), LongMemEval fold {k}: the §27 Qwen 7B reader\n"
              "# config (frozen 7b5fc30 prompt), all §32 arms. Answers only (stub judge); one blind Qwen 7B judge pass\n"
              "# follows (configs/sweeps/exp32/judge.py). The cross-encoder arms replay the free pass's scores.\n",
}

for k in range(5):
    free = {"name": f"exp32_free_f{k}", "base": base(k), "grid": {"controller": arms(k, cache_only=False),
                                                                  "memory.budget.fraction": [0.05]}}
    (HERE / f"exp32_free_f{k}.yaml").write_text(HEADER["free"].format(k=k) + yaml.safe_dump(free, sort_keys=False))
    reader = {"name": f"exp32_qwen7b_f{k}", "base": base(k, reader=True),
              "grid": {"controller": arms(k, cache_only=True), "memory.budget.fraction": [0.05]}}
    (HERE / f"exp32_qwen7b_f{k}.yaml").write_text(HEADER["reader"].format(k=k) + yaml.safe_dump(reader, sort_keys=False))
timed = arms(0, cache_only=False, fresh=True)
# fixed8 timed through the rerank controller with the head's own logit (same features, same policy, same picks).
timed[0] = {"name": "rerank", "label": "fixed8", "checkpoint": HEAD.format(k=0), **SLOT, "scorer": {"kind": "head"}}
latency = {"name": "exp32_latency", "base": base(0), "grid": {"controller": timed, "memory.budget.fraction": [0.05]}}
(HERE / "exp32_latency.yaml").write_text(
    "# Experiment 32 (§32): selection latency per question. Fold 0's test part, one worker, one thread, no score\n"
    "# cache (the cross-encoder computes every pair). Run alone on an idle laptop.\n"
    + yaml.safe_dump(latency, sort_keys=False))
print("written")
