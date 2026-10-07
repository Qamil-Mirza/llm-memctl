"""Writes the Experiment 13 sweep files (run from the repo root: python configs/sweeps/exp13/make.py).

Experiment 13 re-runs LongMemEval and LoCoMo question answering under the protocol of the peer review
(EXPERIMENTS.md 11e, 12): stratified LongMemEval folds over all 500 questions, LoCoMo categories 1-4,
the reasoning reader (a note, then "Answer: ..."), LongMemEval's official judge prompts, one shared
search per benchmark, a retrieval floor for the learned controller, no EVICT, and matched token targets.

Stages (each a separate sweep, run in this order so a bad prompt costs minutes):
  check     the oracle on every LongMemEval fold at 2%: multi-session must move off zero and the
            truncation rate must be low before anything else runs
  ceiling   the evidence-only reader (longmemeval_oracle.json, full context of the evidence sessions)
  lme_f{k}  per fold: oracle, FIFO and the learned controller at 1/2/5%, then token targets at 2/5%
  lme_keep  keep-last-n x top-k (these ignore the budget, so one budget)
  locomo    oracle, FIFO (fusion and plain BM25), salience, the learned controller (transfer), keep-last
"""

from pathlib import Path

import yaml

HERE = Path(__file__).parent
URL = "POD_URL"  # replaced with the pod's proxy URL at launch (runs/_pipelines/exp13.sh)
CACHE = "cache/generations_qwen7b_vllm_v2"  # new reader prompt, new cache: 11b-11e stay reproducible
MODEL = {"backend": "openai", "name": "qwen2.5-7b-instruct", "base_url": f"{URL}/v1", "timeout_s": 300,
         "cache_dir": CACHE}
AGENT = {"name": "llm", "reasoning": True, "labels": "compact", "max_new_tokens": 256, "model": MODEL}
MEMORY = {"allowed_operations": ["KEEP", "MOVE_TO_ARCHIVE", "RETRIEVE_FROM_ARCHIVE", "NO_OP"], "embedder": "hashing"}
LME = "data/longmemeval/longmemeval_s_cleaned.json"
LME_SEARCH = {"method": "lexical"}  # Experiment 12: labels lower BM25 recall on LongMemEval
# The synthetic-trained regret imitation policy (Experiment 10, seed 0): with the floor and EVICT masked out it
# is a learned eviction policy whose features were never fitted to real text (peer review D3).
SYNTHETIC = "runs/rl_bc_archive_regret_s0/checkpoints/policy.pt"
LOCOMO_SEARCH = {"method": "fusion", "model": "BAAI/bge-small-en-v1.5"}  # labelled BM25 + bge-small RRF


def base(env: dict, episodes: int) -> dict:
    return {"schema_version": 1, "seed": 0, "episodes": episodes, "env": env, "agent": AGENT, "memory": MEMORY,
            "logging": {"detail_episodes": 100_000}}


def fifo(search: dict, label: str, target: int | None = None) -> dict:
    controller = {"name": "fifo", "label": label, "removal": ["MOVE_TO_ARCHIVE"],
                  "retrieve": {"top_k": 5, "fit": True, **search}}
    if target:
        controller["target_tokens"] = target
    return controller


def learned(checkpoint: str, search: dict, label: str, target: int | None = None) -> dict:
    controller = {"name": "rl", "label": label, "checkpoint": checkpoint, "retrieve_floor": 5,
                  "retrieve_candidates": 16, "retrieval_method": search["method"]}
    if search.get("model"):
        controller["retrieval_model"] = search["model"]
    if target:
        controller["target_tokens"] = target
    return controller


def keep_last(search: dict, n: int, k: int) -> dict:
    return {"name": "archive_everything", "label": f"keep_last{n}_top{k}", "keep_last": n,
            "retrieve": {"top_k": k, **search}}


def write(name: str, comment: str, sweep: dict) -> None:
    sweep = {"name": name, **sweep}
    (HERE / f"{name}.yaml").write_text(f"# {comment}\n" + yaml.safe_dump(sweep, sort_keys=False, width=120))


def lme_env(fold: int, path: str = LME, judge_style: str = "longmemeval") -> dict:
    return {"name": "longmemeval", "path": path, "folds": {"k": 5, "fold": fold, "part": "test"},
            "judge": {**MODEL, "style": judge_style}}


def main() -> None:
    folds = list(range(5))
    write("exp13_check", "Stage 1: the oracle with the reasoning reader on every LongMemEval fold at 2%.", {
        "base": base(lme_env(0), 100),
        "grid": {"env.folds.fold": folds, "controller": [{"name": "oracle", "method": "approx"}],
                 "memory.budget.fraction": [0.02]},
    })
    write("exp13_ceiling", "Evidence-only reader: full context of longmemeval_oracle.json (evidence sessions only).", {
        "base": base(lme_env(0, "data/longmemeval/longmemeval_oracle.json"), 100),
        "grid": {"env.folds.fold": folds, "controller": [{"name": "full_context"}], "memory.budget.fraction": [1.0]},
    })
    for fold in folds:
        checkpoint = f"runs/lme_floor_f{fold}_s0/checkpoints/policy_best.pt"
        write(f"exp13_lme_f{fold}", f"LongMemEval fold {fold} (test part), fill to budget.", {
            "base": base(lme_env(fold), 100),
            "grid": {"controller": [{"name": "oracle", "method": "approx"}, fifo(LME_SEARCH, "fifo_top5"),
                                    learned(checkpoint, LME_SEARCH, "learned_floor5"),
                                    learned(SYNTHETIC, LME_SEARCH, "synthetic_floor5")],
                     "memory.budget.fraction": [0.01, 0.02, 0.05]},
        })
        write(f"exp13_lme_f{fold}_target", f"LongMemEval fold {fold}: matched token targets (2% and 5% only).", {
            "base": base(lme_env(fold), 100),
            "grid": {"controller": [c for t in (2000, 3000, 4000) for c in (
                fifo(LME_SEARCH, f"fifo_top5_t{t}", t), learned(checkpoint, LME_SEARCH, f"learned_floor5_t{t}", t))],
                "memory.budget.fraction": [0.02, 0.05]},
        })
    write("exp13_lme_keep", "LongMemEval keep-last-n x top-k (budget-free), every fold.", {
        "base": base(lme_env(0), 100),
        "grid": {"env.folds.fold": folds,
                 "controller": [keep_last(LME_SEARCH, n, k) for n in (0, 4, 8, 16) for k in (3, 5, 8)],
                 "memory.budget.fraction": [0.05]},
    })
    locomo_env = {"name": "locomo", "include_adversarial": False, "judge": MODEL}
    transfer = "runs/lme_floor_f0_s0/checkpoints/policy_best.pt"
    write("exp13_locomo", "LoCoMo categories 1-4: shared fusion search; plain-BM25 FIFO links to 11e.", {
        "base": base(locomo_env, 10),
        "grid": {"controller": [{"name": "oracle", "method": "approx"}, fifo(LOCOMO_SEARCH, "fifo_top5_fusion"),
                                fifo({"method": "lexical"}, "fifo_top5_bm25"),
                                {**fifo(LOCOMO_SEARCH, "salience_top5_fusion"), "name": "salience"},
                                learned(transfer, LOCOMO_SEARCH, "learned_floor5_lme_f0"),
                                learned(SYNTHETIC, LOCOMO_SEARCH, "synthetic_floor5")],
                 "memory.budget.fraction": [0.05, 0.1, 0.25]},
    })
    write("exp13_locomo_keep", "LoCoMo keep-last-n with the fusion top 5 (budget-free).", {
        "base": base(locomo_env, 10),
        "grid": {"controller": [keep_last(LOCOMO_SEARCH, n, 5) for n in (0, 4, 16)], "memory.budget.fraction": [0.1]},
    })


if __name__ == "__main__":
    main()
