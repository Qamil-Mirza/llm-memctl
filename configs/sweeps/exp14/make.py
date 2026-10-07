"""Writes the Experiment 14 (N1) evidence-and-tokens gate sweeps (python configs/sweeps/exp14/make.py).

No reader (agent null): evidence in view and prompt size, LongMemEval test folds, labels counted in the budget
(memory.count_labels), the same plain-BM25 retrieval floor as Experiment 13. Candidates: the composed-episode
policy with the token price at the pre-specified 3e-5 per token (and 1e-4, 3e-4 as a declared sensitivity
check) against FIFO + floor and keep-last-0 + top 5, the rule frontier of Experiment 13.
"""

from pathlib import Path

import yaml

HERE = Path(__file__).parent
LME = "data/longmemeval/longmemeval_s_cleaned.json"
MEMORY = {"allowed_operations": ["KEEP", "MOVE_TO_ARCHIVE", "RETRIEVE_FROM_ARCHIVE", "NO_OP"], "embedder": "hashing",
          "count_labels": True}


def learned(fold: int, label: str, price: float = 0.0) -> dict:
    controller = {"name": "rl", "label": label, "checkpoint": f"runs/lme_compose4_f{fold}_s0/checkpoints/policy_best.pt",
                  "retrieve_floor": 5, "retrieve_candidates": 16, "retrieval_method": "lexical"}
    if price:
        controller.update(needed_model=f"runs/lme_needed_f{fold}/needed.pt", need_value=0.44, token_price=price)
    return controller


def main() -> None:
    for fold in range(5):
        controllers = [
            {"name": "fifo", "label": "fifo_top5", "removal": ["MOVE_TO_ARCHIVE"],
             "retrieve": {"top_k": 5, "fit": True, "method": "lexical"}},
            {"name": "archive_everything", "label": "keep_last0_top5", "keep_last": 0, "retrieve": {"top_k": 5, "method": "lexical"}},
            learned(fold, "compose_floor5"),
            learned(fold, "compose_price3e-5", 3e-5),
            learned(fold, "compose_price1e-4", 1e-4),
            learned(fold, "compose_price3e-4", 3e-4),
        ]
        sweep = {"name": f"exp14_gate_f{fold}",
                 "base": {"schema_version": 1, "seed": 0, "episodes": 100, "agent": {"name": "null"}, "memory": MEMORY,
                          "env": {"name": "longmemeval", "path": LME, "folds": {"k": 5, "fold": fold, "part": "test"}},
                          "logging": {"detail_episodes": 100_000}},
                 "grid": {"controller": controllers, "memory.budget.fraction": [0.02, 0.05]}}
        (HERE / f"exp14_gate_f{fold}.yaml").write_text(
            f"# Experiment 14 gate, LongMemEval fold {fold}: no reader, labels counted.\n" + yaml.safe_dump(sweep, sort_keys=False, width=120))


if __name__ == "__main__":
    main()
