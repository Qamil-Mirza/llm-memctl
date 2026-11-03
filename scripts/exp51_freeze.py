#!/usr/bin/env python
"""§51 freeze manifests (§9, §13.5). Writes a DRAFT manifest for the Advisor's review; nothing is frozen until the
Advisor commits the file (the commit is the freeze record; its sha256 is the manifest hash).

    HF_HUB_OFFLINE=1 .venv/bin/python scripts/exp51_freeze.py --point F0     # -> runs/exp51/freeze/F0_draft.json
    .venv/bin/python scripts/exp51_freeze.py --point F1                      # -> F1_draft.json (r computed)

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
F1_EXTRA_SOURCES = ("scripts/exp51_stop.sh", "scripts/exp51_hf_smoke.py")
F0_COMMIT, F0_SHA = "d29263a", "3ed50681f3001527d06cc2e1183187250991d3f39886975ef88946089ed3a9e7"
DRY_RUNS = (ROOT / "runs/exp51/dryrun_api", ROOT / "runs/exp51/dryrun_api2")


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
                   "dry_run": {"stop_usd": H.DRY_STOP_USD, "max_turns": H.DRY_MAX_TURNS,
                               "model": "the dummy tiny Qwen2", "rows": "DRY- synthetic rows (seed 99), no E content"},
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
            "cf_follow_b0_E": "B0 CF-follow on E = 282 / 432 = 0.65, so the §8.1 CF-follow >= 0.90 repair criterion "
                              "is a high bar (pre-registered, unchanged)",
            "distractors_added": "more distractor lines, with the date shifts that follow from position (E: question "
                                 "date moves in 144 / 144 pairs, the evidence line's date in 129 / 144); kept as built, "
                                 "no generator change (Advisor ruling (b))",
            "strict_marker_nm_exp50_dev": "runs/exp51/build_notes/exp50_dev_nm_rows.json (14 rows)",
            "e_fa_support_bases": "E bases with >= 1 FA row (B0, of 48): DE 44, RC 35, TU 15; TU is below the 20-base "
                                  "minimum on E, as §3 anticipated (declared, not a deviation)",
            "inline_answer_rate_exp49_50": "3 of 1,440 F2 outputs had an inline ' Answer:' (NM under the strict "
                                           "marker)"},
    }


def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _head() -> str:
    """The F1 build commit. The manifest's source shas only mean something if the sources are committed."""
    import subprocess
    git = lambda *a: subprocess.run(("git", *a), cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    dirty = git("status", "--porcelain", "--", *SOURCES, *F1_EXTRA_SOURCES)
    if dirty:
        raise SystemExit(f"F1 is built from committed sources; uncommitted:\n{dirty}")
    return git("rev-parse", "--short", "HEAD")


def _projection() -> dict:
    """r = min(1, 48 h / projected time for 30 passes) from the laptop inventory (runs/exp51/run/inventory.json):
    the slowest pass type at its worst case. A sensitivity line takes every edited paired sequence to 160 tokens."""
    inv = json.loads((CM.RUN / "inventory.json").read_text())
    worst = {i["what"]: i["cpu_h"] for i in inv["items"] if "30 " in i["what"] and "worst case" in i["what"]}
    t = json.loads((CM.RUN / "timing.json").read_text())
    dec = [(u["b0_total_s"] - u["read_s"]) / u["n_gen"] for u in t["units"]]
    s_row_160 = max(u["read_s"] for u in t["units"]) + max(dec) * EN.MAX_NEW_TOTAL
    paired_160_h = 30 * SV.BOUNDS_F0["paired_pairs"] * s_row_160 / 3600
    proj = max(worst.values())
    return {"inventory_sha256": _sha(CM.RUN / "inventory.json"), "timing_sha256": _sha(CM.RUN / "timing.json"),
            "worst_case_30_passes_h": worst, "projected_h": proj, "r": min(1.0, 48.0 / proj),
            "sensitivity": {"decode_s_per_token_max": round(max(dec), 3), "s_per_row_at_160_tokens": round(s_row_160, 1),
                            "paired_30_passes_all_160_tokens_h": round(paired_160_h, 1),
                            "r_if_used": round(min(1.0, 48.0 / paired_160_h), 3),
                            "note": "not the rule's basis (B0 mean length is the inventory unit); the 48 h wall bound "
                                    "stops the run in that case"}}


def _dry_runs() -> dict:
    out = {}
    for d in DRY_RUNS:
        calls = json.loads((d / "api_usage.json").read_text())
        real = [c for c in calls if not c.get("placeholder")]
        out[d.name] = {"calls": len(real), "usd": round(sum(c["usd"] for c in real), 4),
                       "usd_per_call": [round(c["usd"], 4) for c in real],
                       "new_tokens_per_call": round(sum(c["usage"].get("cache_creation_input_tokens", 0)
                                                        for c in real) / len(real)),
                       "output_tokens_per_call": round(sum(c["usage"]["output_tokens"] for c in real) / len(real)),
                       "api_usage_sha256": _sha(d / "api_usage.json"), "transcript_sha256": _sha(d / "transcript.jsonl")}
    return out


def cost_model(turns: int, new: int, out: int, expiries: int = 0, c0: int = 2900) -> dict:
    """Per call: context x cache read + new x 1h write + out x output price. `new` is the cache write of a call, which
    already holds the previous call's output (measured: context growth = the write), so the context grows by `new`
    only. An expiry re-writes the whole context once (write price instead of read). The harness stops when measured
    usd + the next call's worst case passes $45, so the run ends before $45; turn_at_stop applies that reserve."""
    P = H.PRICES
    ctx, usd, at_stop, at_ctx = c0, 0.0, None, None
    every = turns // expiries if expiries else None
    for k in range(1, turns + 1):
        reserve = (ctx * 2.9 / 2.5) * P["cache_write_1h"] / 1e6 + H.API_CONFIG["max_tokens"] * P["output"] / 1e6
        if at_stop is None and usd + reserve > H.STOP_USD:
            at_stop = {"turn": k, "usd_measured": round(usd, 2), "reserve_usd": round(reserve, 2)}
        if at_ctx is None and ctx > H.CONTEXT_STOP_TOKENS:
            at_ctx = k
        usd += (ctx * P["cache_read"] + new * P["cache_write_1h"] + out * P["output"]) / 1e6
        if every and k % every == 0 and k // every <= expiries:
            usd += ctx * (P["cache_write_1h"] - P["cache_read"]) / 1e6
        ctx += new
    return {"usd_if_never_stopped": round(usd, 2), "context_tokens_at_end": ctx, "harness_stop": at_stop,
            "turn_at_context_stop": at_ctx}


def _backtest() -> dict:
    """The model at each dry run's own means, against its measured cost."""
    out = {}
    for name, v in _dry_runs().items():
        m = cost_model(v["calls"], v["new_tokens_per_call"], v["output_tokens_per_call"])
        out[name] = {"model_usd": m["usd_if_never_stopped"], "measured_usd": v["usd"]}
    return out


def cost_estimate(turns: int) -> dict:
    d = _dry_runs()
    new = max(v["new_tokens_per_call"] for v in d.values())
    out = max(v["output_tokens_per_call"] for v in d.values())
    sc = {"dry_like": (new, out, 0), "expected": (4000, 1000, 3), "high_no_expiries": (7500, 2000, 0),
          "high": (7500, 2000, 10)}
    return {"turns": turns, "assumptions": {k: {"new_tokens_per_call": a, "output_tokens_per_call": b,
                                                "cache_expiries": e} for k, (a, b, e) in sc.items()},
            "scenarios": {k: cost_model(turns, a, b, e) for k, (a, b, e) in sc.items()},
            "backtest": _backtest(),
            "usd_per_expiry": {f"{c // 1000}k_context": round(c * (H.PRICES["cache_write_1h"] - H.PRICES["cache_read"])
                                                             / 1e6, 2) for c in (100_000, 300_000, 500_000, 900_000)},
            "harness_stop_usd": H.STOP_USD, "console_cap_usd": 50,
            "basis": "Measured: the per-call means and the back-test (dry runs 1-2). Judgement, not measurement: the "
                     "real-run scenarios. The dry runs used a dummy model, so real thinking and tool output are "
                     "scaled up: 'expected' about 1.7x the dry write and 1.8x the dry output, 'high' about 3.1x and "
                     "3.7x (7,500 per call reaches the 900k context stop at turn 120). Forward results stay small in "
                     "the real run (dry: about 400 chars; arrays go to npz files), so growth comes from run_python "
                     "stdout (<= 20k chars), the editor's code and its thinking. Expiries are the largest uncertainty: "
                     "the 1h cache lapses when a tool call plus generation passes 1 h start to start. At B0 mean "
                     "length a largest paired pass (both arms edited) takes about 52 min and a largest read pass about "
                     "54 min, so only near-largest passes or lengthened outputs lapse; a crash resume can add one."}


def f1() -> dict:
    proj = _projection()
    bounds = {k: (v if k in ("submissions",) else type(v)(v * proj["r"])) for k, v in SV.BOUNDS_F0.items()}
    counts = json.loads((ROOT / "runs/exp51/cases51_counts.json").read_text())
    d2 = {"scripts/exp51_harness.py": ("c115a9fb78dbed95e62f39cb743ab09c0d17c9e3a9f5d27d65dac2d35ed54c69",
                                       "ef298273497e94e775659969b0bfa378b57353e3194dd94484b4acf90021f659"),
          "tests/test_exp51.py": ("f10c82f1488fac0172b3077998101b17c8a7a59fdb7b4e66d86b1ac1d24c9b6d",
                                  "37d923bd0724b258d62e2c79e9485c0dcf40e38cfbadc70f5f70bb2d77f92819")}
    d3_old = {"scripts/exp51_harness.py": d2["scripts/exp51_harness.py"][1],
              "tests/test_exp51.py": d2["tests/test_exp51.py"][1],
              "scripts/exp51_service.py": "b0979e2d834b4807f9c81091cb22d59c55919a202af8fb980d35cfadd860c67e",
              "scripts/exp51_engine.py": "59d5ca562794a728761f139cbbef6d2df1a281fb12cdb8ba4599f0e47a91c5a1",
              "scripts/exp51_freeze.py": "e4891791541cdc6d5b2478745095ba3b6458601938b1ac998e0a5857c55e6baf",
              "scripts/exp51_stop.sh": None, "scripts/exp51_hf_smoke.py": None}
    return {
        "point": "F1", "status": "DRAFT (not frozen until committed by the Advisor)",
        "f0": {"commit": F0_COMMIT, "manifest": "runs/exp51/freeze/F0_draft.json", "sha256": F0_SHA,
               "manifest_sha256_now": _sha(OUT / "F0_draft.json")},
        "split_sizes_bases_per_task": counts["sizes"], "n_rows": counts["n_rows"],
        "cases51_sha256": counts["cases51_sha256"],
        "reduction": proj,
        "bounds_final": bounds, "bounds_within_f0_maxima": all(bounds[k] <= v for k, v in SV.BOUNDS_F0.items()),
        "resource_caps": {"harness_stop_usd": H.STOP_USD, "console_workspace_cap_usd": 50,
                          "context_stop_tokens": H.CONTEXT_STOP_TOKENS, "context_window_tokens": 1_000_000,
                          "sandbox_rlimit_as_gib": 6, "python_run_timeout_s": H.RUN_TIMEOUT_S,
                          "laptop_peak_rss_gb": json.loads((CM.RUN / "timing.json").read_text())["peak_rss_gb"]},
        "workload_inventory": {"file": "runs/exp51/run/inventory.json", "sha256": proj["inventory_sha256"]},
        "meter_validation": {"dry_runs": _dry_runs(), "meter_usd": 0.6965, "console_usd": 0.71,
                             "gap": "about $0.01: the one call in flight at dry run 2's kill (billed, never answered)",
                             "verdict": "PASS (Advisor, 2026-11-03); prices equal the published Opus 5.5 table "
                                        "($4 / $20, 1h write $8, 5m write $5, cache read $0.20; 1M window at "
                                        "standard price)"},
        "editor_cost_estimate": cost_estimate(bounds["turns"]),
        "deviations": [
            {"id": "D2", "what": "ApiClient imported urllib.error inside __call__ (UnboundLocalError before any "
                                 "request; dry run attempt 1 sent nothing, $0); module-level imports + a mocked-"
                                 "network test", "commit": "d2f1cee", "authority": "Advisor (ACCEPT)",
             "sha256_old_new": d2, "frozen_content_changed": False},
            {"id": "D3", "what": ["thinking {type adaptive, display summarized} so the editor's between-tool notes "
                                  "are recorded by transcript logging",
                                  "loop --resume (messages rebuilt verbatim from transcript.jsonl; cut-off tool "
                                  "calls answered INTERRUPTED; transcript.pre_resume.<n>.jsonl copy; ledger kind "
                                  "resume)",
                                  "lost-call placeholder on resume (worst case, never trued down)",
                                  "harness_pid.json + exp51_stop.sh (launch and stop by the harness's own process "
                                  "group)",
                                  "context stop at 900k prompt tokens (clean stop before the 1M window's 400)",
                                  "worst-case token estimate chars / 2.5 (was / 3; measured 2.9)",
                                  "dummy36 backend; refusal wording names valid layers; save_edit accepts /work/ "
                                  "spec paths (containment check kept)", "DRY_MAX_TURNS 10 -> 15",
                                  "$0 scripted end-to-end and resume tests",
                                  "stop_details logged on a refusal (transcript only)",
                                  "refusal handling: one fixed continuation, second refusal ends (rules.refusal)",
                                  "one budget note at $32 and budget_used_fraction in the ledger (rules.budget_note)",
                                  "ledger_copy omits the usd of a lost_call_placeholder (kept in the ledger)",
                                  "exp51_hf_smoke.py: $0 scripted loop on the real model, DRY- rows (setup_dry takes "
                                  "an engine)",
                                  "exp51_freeze.py: the F1 builder and cost model; refuses to rebuild F0"],
             "commits": ["475be08", "f0a97b9", "645c6ad", _head()],
             "authority": "Advisor (ACCEPT 475be08; required items of the dry-run reviews)",
             "sha256_old_new": {f: (o, _sha(ROOT / f)) for f, o in d3_old.items()},
             "frozen_content_changed": True,
             "frozen_content_changed_items": ["API request: thinking display summarized (what is returned changes; "
                                              "what the editor is told does not)", "dry_run.max_turns 10 -> 15",
                                              "new fixed text the editor can receive: rules.budget_note.text, and "
                                              "rules.refusal.text after a refusal; ledger_copy gains "
                                              "budget_used_fraction"],
             "frozen_content_unchanged_check": {"tools_sha256": sha_text(json.dumps(H.TOOLS, sort_keys=True,
                                                                          separators=(",", ":"))),
                                      "instruction_sha256": sha_text(H.INSTRUCTION),
                                      "system_sha256": sha_text(H.SYSTEM), "stop_usd": H.STOP_USD}}],
        "rules": {
            "resume": "A resume after an infrastructure or harness crash is the same run, not a second editor run "
                      "(D-45 is not engaged). Messages are rebuilt verbatim from transcript.jsonl (append-only); the "
                      "ledger and api_usage.json carry the bounds and the meter.",
            "lost_call": "On resume, if the transcript ends on a user record, one call is booked at its worst case "
                         "(context at the 1h write price + max_tokens at the output price) as a meter placeholder and "
                         "ledger kind lost_call_placeholder; never trued down.",
            "refusal": {"rule": "An isolated refusal: the refused response is logged and never joins the messages; the "
                                "fixed text below is appended once as a text block to the user turn that drew it "
                                "(transcript record user_amend) and the request is sent. Two consecutive refusals (the "
                                "continuation itself refused) or a third refusal overall: the run ends, documented "
                                "under D-45; submissions already made stand and go to replay. No fallback model. stop_details is in the transcript only (the ledger is "
                                "mirrored into the editor's sandbox).",
                        "text": H.REFUSAL_CONTINUE, "authority": "Advisor F1 ruling 3"},
            "budget_note": {"rule": "Appended once to the user turn being sent when measured spend first passes usd "
                                    "(tool_result content; the nudge text on a turn without tools); ledger kind "
                                    "budget_note. Every turn's ledger entry carries budget_used_fraction = spend / the "
                                    "harness stop, visible in ledger_copy.jsonl.",
                            "usd": H.BUDGET_NOTE_USD, "text": H.BUDGET_NOTE, "authority": "Advisor F1 ruling 2"},
            "launch": ["cd ~/Code/llm-memctl; the Console workspace limit is $50 and the key is in .env",
                       ".venv/bin/python scripts/exp51_harness.py setup --run runs/exp51/editor_run",
                       ".venv/bin/python scripts/exp51_harness.py check --run runs/exp51/editor_run (must pass)",
                       "setsid .venv/bin/python scripts/exp51_harness.py loop --run runs/exp51/editor_run "
                       "--client api --backend hf --i-have-the-go > runs/exp51/editor_run/loop.log 2>&1 &",
                       "the harness writes runs/exp51/editor_run/harness_pid.json; check own_group is true",
                       "stop only with scripts/exp51_stop.sh runs/exp51/editor_run (kills the recorded process "
                       "group; never $! or pgrep)",
                       "after a crash: the same command with --resume (no other change)"]},
        "disclosures": [
            "The continuation request shape (a text block after tool_results) has never been sent to the live API; a "
            "refusal cannot be triggered on demand. If the API rejected it, that is a non-transient 400: the harness "
            "raises and a --resume hits the same 400.",
            "The budget note can go unseen in one corner: at very large context the stop reserve is near $8.6, so one "
            "call jumping from below $32 past about $36 (for example a cache expiry near 900k context) sends the note "
            "and stops before another call."],
        "hf_smoke": {"file": "runs/exp51/hf_smoke/smoke_result.json", "sha256": _sha(ROOT / "runs/exp51/hf_smoke/smoke_result.json"),
                     **json.loads((ROOT / "runs/exp51/hf_smoke/smoke_result.json").read_text())},
        "sources_sha256": {s: _sha(ROOT / s) for s in SOURCES + F1_EXTRA_SOURCES},
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--point", choices=("F0", "F1"), required=True)
    a = p.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    if a.point == "F0" and (OUT / "F0_draft.json").exists():
        raise SystemExit("F0 is frozen (d29263a); it is not rebuilt")
    m = f0() if a.point == "F0" else f1()
    text = json.dumps(m, indent=1, sort_keys=True, default=lambda x: None if x != x else x)
    path = OUT / f"{a.point}_draft.json"
    path.write_text(text)
    print(json.dumps({"path": str(path), "sha256": hashlib.sha256(text.encode()).hexdigest()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
