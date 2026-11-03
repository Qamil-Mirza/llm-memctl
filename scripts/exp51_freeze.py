#!/usr/bin/env python
"""§51 freeze manifests (§9, §13.5). Writes a DRAFT manifest for the Advisor's review; nothing is frozen until the
Advisor commits the file (the commit is the freeze record; its sha256 is the manifest hash).

    HF_HUB_OFFLINE=1 .venv/bin/python scripts/exp51_freeze.py --point F0     # -> runs/exp51/freeze/F0_draft.json
    .venv/bin/python scripts/exp51_freeze.py --point F1 --r <reduction>      # -> F1_draft.json

F0 holds: checkpoint and tokenizer revisions with file hashes; the editor's model and API settings; the instruction
and system text; prompt formats and max_new_tokens; generator rules, seeds, sizes and the case-list hash; the
partition key; the grader (source hash, cue lists, alias list, fixtures and their results); comparator
specifications and grids (the B1 token set printed); the edit family; the search-bound maxima with the reduction rule
r = min(1, 48 h / projected time for 30 passes); the allowed-analysis rule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import exp51_cases as G  # noqa: E402
import exp51_common as CM  # noqa: E402
import exp51_engine as EN  # noqa: E402
import exp51_harness as H  # noqa: E402
import exp51_service as SV  # noqa: E402
from memctl.analysis import edit51 as ED  # noqa: E402
from memctl.analysis import grade51 as GR  # noqa: E402

OUT = ROOT / "runs/exp51/freeze"
SOURCES = ("scripts/exp51_cases.py", "scripts/exp51_common.py", "scripts/exp51_engine.py", "scripts/exp51_run.py",
           "scripts/exp51_service.py", "scripts/exp51_harness.py", "scripts/exp51_mech.py",
           "scripts/exp51_inventory.py", "memctl/analysis/grade51.py", "memctl/analysis/stats51.py",
           "memctl/analysis/edit51.py", "memctl/analysis/mech51.py")


def sha_text(t: str) -> str:
    return hashlib.sha256(t.encode()).hexdigest()


def f0() -> dict:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(EN.SNAPSHOT))
    b1 = CM.b1_token_set(tok)
    counts = json.loads((ROOT / "runs/exp51/cases51_counts.json").read_text())
    return {
        "point": "F0", "status": "DRAFT (not frozen until committed by the Advisor)",
        "checkpoint": ED.snapshot_fingerprint(EN.SNAPSHOT), "model_id": EN.MODEL_ID,
        "inference": {"dtype": "float32", "device": "cpu", "threads": 8, "decoding": "greedy, batch 1",
                      "max_new_total": EN.MAX_NEW_TOTAL, "max_new_release": EN.MAX_NEW_RELEASE, "marker": GR.MARKER,
                      "chat_template": "tokenizer.apply_chat_template, one user message, generation prompt"},
        "editor": {**{k: v for k, v in H.API_CONFIG.items()}, "prices_usd_per_mtok": H.PRICES,
                   "harness_stop_usd": H.STOP_USD, "console_workspace_cap_usd": 50,
                   "system_sha256": sha_text(H.SYSTEM), "instruction_sha256": sha_text(H.INSTRUCTION),
                   "instruction": H.INSTRUCTION, "system": H.SYSTEM, "nudge": H.NUDGE,
                   "sandbox_packages": list(H.SANDBOX_PKGS)},
        "prompts": {"f2_instruction": G.C.F23_INSTRUCTION, "reasoning_instructions": G.REASONING_INSTRUCTIONS,
                    "t_format_instruction": G.T_FORMAT_INSTRUCTION, "b2_total_suffix": G.B2_SUFFIX,
                    "b2_prefix_tail": CM.B2_TAIL, "b3_removed": [G.G.R4_SENTENCE, G.G.R4_SENTENCE_NOTE]},
        "generator": {"seeds": G.SEEDS, "requested_sizes": G.SIZES, "feasibility": counts["feasibility"],
                      "sizes": counts["sizes"], "cases51_sha256": counts["cases51_sha256"], "n_rows": counts["n_rows"],
                      "t_cities": counts["design"]["t_cities"], "max_attempts": G.MAX_ATTEMPTS,
                      "partition_key": counts["design"]["effective_key"], "canonical_key": "exp51_cases.key()"},
        "grader": {"version": GR.GRADER_VERSION, "abstain": GR.ABSTAIN, "doubt": GR.DOUBT, "neg": GR.NEG,
                   "suggest": GR.SUGGEST, "aliases": GR.ALIASES, "main_cities": GR.MAIN_CITIES,
                   "fixtures": [list(f[:2]) + [f[2], f[5], f[6]] for f in GR.FIXTURES],
                   "fixture_results": GR.run_fixtures()},
        "comparators": {"B1": {"grid": CM.B1_GRID, "signed_grid_m3b": CM.B1_SIGNED_GRID, "n_tokens": len(b1),
                               "token_ids": b1, "tokens": [tok.decode([i]) for i in b1]},
                        "B4": {"layer": EN.STEER_LAYER, "alpha_grid": CM.B4_GRID, "file": str(CM.B4_FILE),
                               "file_sha256": CM.b4_sha(), "unit": "vnorm (alpha x v_29)"},
                        "B5": {"doses": CM.B5_DOSES, "rule": "see exp51_run.py b5_fit"},
                        "B6": {"seeds": CM.B6_SEEDS, "rule": "edit51.random_spec / random_matched"},
                        "candidate_doses": CM.CAND_DOSES},
        "edit_family": {"editable": ED.EDITABLE, "max_components": ED.MAX_COMPONENTS, "max_rank": ED.MAX_RANK,
                        "max_sparse": ED.MAX_SPARSE, "scale_cos": ED.SCALE_COS},
        "bounds_max": SV.BOUNDS_F0,
        "reduction_rule": "every bound x r, r = min(1, 48 h / projected time for 30 passes) (§13.5)",
        "allowed_analysis": "means, covariances, projections, PCA/SVD, closed-form solves computed once; no autograd, "
                            "optimisers or iterative loss-based fitting (sandbox: numpy and tokenizers only)",
        "sources_sha256": {s: hashlib.sha256((ROOT / s).read_bytes()).hexdigest() for s in SOURCES},
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--point", choices=("F0",), required=True)
    a = p.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    m = f0()
    text = json.dumps(m, indent=1, sort_keys=True, default=lambda x: None if x != x else x)
    path = OUT / f"{a.point}_draft.json"
    path.write_text(text)
    print(json.dumps({"path": str(path), "sha256": hashlib.sha256(text.encode()).hexdigest(),
                      "b1_tokens": m["comparators"]["B1"]["n_tokens"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
