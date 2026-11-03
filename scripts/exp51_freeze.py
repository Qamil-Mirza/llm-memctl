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
           "memctl/analysis/edit51.py", "memctl/analysis/mech51.py", "scripts/exp51_stage4.py",
           "scripts/exp51_gpu_measure.py", "scripts/exp51_quote.py", "scripts/exp51_freeze.py",
           "scripts/exp51_bundle.sh", "scripts/exp51_pod_measure.sh", "scripts/exp51_d30.sh",
           "scripts/exp51_s_bundle.sh", "scripts/exp51_pod_s.sh", "scripts/exp51_ht_bundle.sh", "scripts/exp51_pod_ht.sh",
           "tests/test_exp51.py")


def sha_text(t: str) -> str:
    return hashlib.sha256(t.encode()).hexdigest()


MEASURE = ROOT / "runs/exp51/gpu_measure/pod_copy/measure/measure.json"


def _backend_basis() -> dict:
    """The measured basis for the backend choice (timing pod 1t1acrr39w08aw, 2026-11-03): unit rates and the route
    gate fixed before the measurement."""
    if not MEASURE.exists():
        return {"status": "no measurement"}
    m = json.loads(MEASURE.read_text())
    return {"pod": "1t1acrr39w08aw (A40 secure, EU-SE-1, 2026-11-03 16:03:58-16:08:22 UTC, 264 s)",
            "measure_sha256": hashlib.sha256(MEASURE.read_bytes()).hexdigest(), "engine": m["engine"]["engine"],
            "gpu": m.get("gpu"), "unit_s_per_row": {k: v.get("s_per_row") for k, v in m["steps"].items()},
            "route_gate": m.get("route_gate"),
            "laptop_cpu_reference": "runs/exp51/run/timing.json + batching.json (batch 8 / 16 = 24/24 vs batch 1)",
            "rule": "one backend and one fixed batch layout per split for every confirmatory arm, B0 included"}


def _e_b0() -> dict:
    """B0 on E (analyst work before F0, §13.5): counts by set x class, NM, the engine, and one hash over the records."""
    from collections import Counter
    rows = CM.rows("E")
    h = hashlib.sha256()
    cnt, eng, nm, mig = Counter(), set(), 0, 0
    for r in sorted(rows, key=lambda r: r["case_id"]):
        p = CM.b0_path("E", r["case_id"])
        if not p.exists():
            return {"status": f"incomplete ({r['case_id']} missing)"}
        h.update(p.read_bytes())
        rec = json.loads(p.read_text())
        cnt[f"{r['set']}|{rec['cls']}"] += 1
        eng.add(rec["engine"]["engine"])
        nm += not rec["p_ans_ok"]
        mig += "migrated_from" in rec
    fit = CM.read_json(CM.RUN / "b5_fit.json", {})
    return {"n_rows": len(rows), "by_set_class": dict(sorted(cnt.items())), "nm": nm, "engines": sorted(eng),
            "migrated_by_prompt_sha": mig, "records_sha256": h.hexdigest(),
            "b5": {k: fit.get(k) for k in ("layer", "direction_sha256", "edit_ids", "n_abst", "n_gold")}}


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
                   "sandbox_packages": list(H.SANDBOX_PKGS),
                   "tools_verbatim": H.TOOLS,
                   "tools_sha256": sha_text(json.dumps(H.TOOLS, sort_keys=True, separators=(",", ":")))},
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
        "backend_basis": _backend_basis(),
        "e_b0": _e_b0(),
        "compute_plan": {
            "decision": "D-73 (the user's 'Do A'): cap $30 -> $36; batched A40 secure route; Advisor stage-4 pod "
                        "structure ruling (2026-11-03): two pods, each quoted and approved on its own",
            "E": "laptop CPU fp32, batch 1 (B0, editor service, the §8.4.2 submission replay on all of E)",
            "pod_S": {"work": "§10 CPU<->pod replay check (200 E rows x B0 + one edit, batch 1 and batch 64) before "
                              "any H work; B0 and every frozen arm on S; reads; in-pod summary and rule-driven "
                              "selection -> f2_candidate.json + sha", "quote_usd": 5.81, "stop_usd": 6.68,
                      "bundle": "no H or T-city rows"},
            "F2": "the Advisor's S review and git commit on the laptop between the pods",
            "pod_HT": {"work": "require_f2 (git-verified) before any sealed read; H once, then T-city",
                       "quote_usd": 3.92, "stop_usd": 4.51},
            "backend_rule": "S and H (and T-city) each run on the A40 at batch layout 64, rows sorted by case id, "
                            "strict fp32 (TF32 off); batch 1 only for M2 hooks, the hybrid estimand and reads",
            "replay_gate": ">= 194 / 200 graded-class agreement in each of B0 batched, B0 batch 1, submission 1 "
                           "batched, submission 1 batch 1 vs the laptop; every mismatch listed for the Advisor's "
                           "investigation before S results are used, even on a pass (stage-4 build review (c))",
            "creation": "live price re-read at create; above $0.59/h stop and ask; retries only for the approved card "
                        "and price; a create refuses if any pod exists"},
        "amendments_pre_F0": [
            {"id": "A1", "what": "T-format dropped; E 48 / S 32 / H 48 / T-city 24 bases per task",
             "why": "the rev 1a sizes did not fit the DE / TU key pool (152 bases > 144 keys; uniform r would be "
                    "0.958); §10 names the transfer sets as secondary and first to drop",
             "authority": "Advisor build review ruling 2; the user's 'yes drop T-format' (Advisor pane, 2026-11-02)",
             "superseded_case_list": "runs/exp51/superseded/cases51_r0958.jsonl",
             "b0_reuse": "E B0 records whose prompt is byte-identical in the new list are carried over"}],
        "deviations": [
            {"id": "D1", "what": "B5 direction from B0-abstained vs B0 gold-answered ANS rows (§5.2 says "
                                 "'B0-answered')", "why": "wrong answers would mix error with commitment",
             "authority": "Advisor build review ruling 7"}],
        "disclosures": {
            "b1_subword_gap": "the B1 rule (decoded, lower-cased, stripped text begins 'unknown') gives 6 ids; "
                              "sub-word spellings such as 'unk' + 'nown' are outside it",
            "rates": "base-weighted: each base with >= 1 row in a population has weight 1, equal within it",
            "distractors_added": "more distractor lines, with the date shifts that follow from position (E: question "
                                 "date moves in 144 / 144 pairs, the evidence line's date in 129 / 144); kept as built, "
                                 "no generator change (Advisor ruling (b))",
            "strict_marker_nm_exp50_dev": "runs/exp51/build_notes/exp50_dev_nm_rows.json (14 rows)",
            "e_fa_support_bases": "E bases with >= 1 FA row (B0, of 48): DE 44, RC 35, TU 15; TU is below the 20-base "
                                  "minimum on E, as §3 anticipated (declared, not a deviation)",
            "inline_answer_rate_exp49_50": "3 of 1,440 F2 outputs had an inline ' Answer:' (NM under the strict "
                                           "marker)"},
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
