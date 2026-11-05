"""§43 Part B lens pod stage (scripts/exp43_lens_pod.sh, exp43_lens_pod_run.py, exp43_lens_report.py), stub only:
population counts, the smoke abort, the deadline skip, the report's claim arithmetic on synthetic rows, the fold
guards, and the partner configs (template, stop ids, refusal ids, BOS, depth, width) from the cached tokenizer and
config files. No weights are downloaded; reads run on tiny random models with the real tokenizers."""
import dataclasses
import json
import math
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from memctl.analysis import families as F
from memctl.jacobian_data import PromptRecord

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("jlens")

from scripts import exp43_lens_pod_run as run  # noqa: E402
from scripts import exp43_lens_report as rep  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
needs_exp24 = pytest.mark.skipif(not (ROOT / "runs/exp24_llama8b_f0").exists(), reason="§24 runs not on disk")

PROMPT = ("Use only the memory.\n\n## Memory\nuser, d1: turn a\nuser, d2: I was born in Leeds.\n\n## Current input\n"
          "Question: (asked on 2023/05/30) Where was I born?\n\nNote:")
LINES = ("user, d1: turn a", "user, d2: I was born in Leeds.")


def tokenizer(model_id):
    try:
        return transformers.AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    except Exception as error:      # not cached on this machine
        pytest.skip(f"tokenizer for {model_id} unavailable: {error}")


# --- partner configs from the tokenizer and config files -----------------------------------------------------------

HEAD_TAIL = {
    "llama8b": ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\n"
                "Today Date: 26 Jul 2024\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n",
                "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"),
    "gemma12b": ("<bos><start_of_turn>user\n", "<end_of_turn>\n<start_of_turn>model\n"),
    "phi4": ("<|im_start|>user<|im_sep|>", "<|im_end|><|im_start|>assistant<|im_sep|>"),
}
STOP_IDS = {"llama8b": {128001, 128008, 128009}, "gemma12b": {1, 106}, "phi4": {100257, 100265}}
BOS_ID = {"llama8b": 128000, "gemma12b": 2, "phi4": None}


def test_pairs_and_partner_fields():
    assert F.PAIRS == {"llama": ("llama3b", "llama8b"), "gemma": ("gemma4b", "gemma12b"), "phi": ("phi4mini", "phi4")}
    for fam, (s, l) in F.PAIRS.items():
        small, large = F.pair(fam)
        assert (small.key, large.key) == (s, l)
        assert small.partner_key == l and large.partner_key == s
        assert small.partner_model_id == large.model_id and large.partner_model_id == small.model_id
        assert large.weights is None and large.exp_tag == "exp24" and large.verdict_arm == "fixed8"
    assert F.LLAMA8B.template_kwargs == {"date_string": "26 Jul 2024"}
    assert F.GEMMA12B.layer_path == "model.language_model.layers" and F.GEMMA12B.system_role is False
    assert F.PHI4.bos_in_chat is False and F.PHI4.lens_force_bos is False
    with pytest.raises(KeyError):
        F.pair("mistral")


@pytest.mark.parametrize("key", ["llama8b", "gemma12b", "phi4"])
def test_partner_template_stop_refusal_bos_depth(key):
    cfg = F.get(key)
    tok = tokenizer(cfg.model_id)
    head, tail = HEAD_TAIL[key]
    text = F.chat_text(tok, cfg, PROMPT)
    assert text == head + PROMPT + tail
    assert F.stop_ids(tok, cfg) == STOP_IDS[key]
    assert F.check_refusal_ids(lambda t: F.encode(tok, t), cfg) == cfg.refusal_ids
    ids = F.encode(tok, text + "Answer:")
    F.check_bos(tok, cfg, ids)
    assert (ids[0] == BOS_ID[key]) if BOS_ID[key] is not None else (tok.bos_token_id not in ids)
    assert tok.decode(ids[-1:]) == ":"
    conf = transformers.AutoConfig.from_pretrained(cfg.model_id, local_files_only=True)
    text_conf = getattr(conf, "text_config", conf)
    assert (text_conf.num_hidden_layers, text_conf.hidden_size) == (cfg.n_layers, cfg.hidden)


def test_llama31_pinned_date_is_the_template_default():
    tok = tokenizer(F.LLAMA8B.model_id)
    msgs = [{"role": "user", "content": PROMPT}]
    assert F.chat_text(tok, F.LLAMA8B, PROMPT) == tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def test_partner_tracked_ids_match_the_refusal_ids():
    """read_row asserts unknown_sp (jlens_readout's " unknown") equals the config's first refusal id: true per family."""
    from memctl.analysis.jlens_readout import readout_for
    for key in ("llama8b", "gemma12b", "phi4", "llama3b", "gemma4b", "phi4mini"):
        cfg = F.get(key)
        tok = tokenizer(cfg.model_id)
        r = readout_for(tok, PROMPT, LINES, (LINES[1],), "Leeds", chat_text_fn=lambda p: F.chat_text(tok, cfg, p))
        assert r.tracked["unknown_sp"] == cfg.refusal_ids[0], key


# --- population -----------------------------------------------------------------------------------------------------

def test_check_counts():
    assert run.check_counts(dict(run.EXPECTED_COUNTS)) == []
    bad = run.check_counts({0: 61, 1: 64, 2: 66, 3: 68, 4: 70})
    assert bad == ["fold 1: 64 rows, expected 65"]
    assert run.check_counts({0: 61, 1: 65}, folds=(0, 1)) == []


@needs_exp24
def test_population_counts_llama(tmp_path):
    """Counts and pairing only (no outcome is read)."""
    small, large = F.pair("llama")
    pairs = run.pop34_pairs(small, large)
    assert run.fold_counts(pairs) == run.EXPECTED_COUNTS
    assert all(x.question_id == y.question_id and x.prompt == y.prompt for x, y in pairs)
    assert {x.reader for x, _ in pairs} == {"llama3b"} and {y.reader for _, y in pairs} == {"llama8b"}
    assert run.main(["population", "--family", "llama", "--out", str(tmp_path)]) == 0
    payload = json.loads((tmp_path / "population.json").read_text())
    assert len(payload["rows"]) == 330 and payload["fingerprint"] == run.fingerprint(pairs)
    assert "correct" not in json.dumps(payload["rows"])
    assert run.main(["population", "--family", "llama", "--out", str(tmp_path), "--expect-fingerprint", "0" * 64]) == run.EXIT_INPUT


# --- reads on a tiny stub ---------------------------------------------------------------------------------------------

def _rec(qid, fold, reader, correct=True, unknown=False, gold="Leeds"):
    return PromptRecord(qid, fold, reader, PROMPT, "Answer: x", gold, "single-session-user", True, LINES, (LINES[1],),
                        True, correct, unknown)


@pytest.fixture
def stub(tmp_path, monkeypatch):
    tokenizer(F.LLAMA3B.model_id)
    tokenizer(F.LLAMA8B.model_id)
    assert run.main(["make-stub", "--family", "llama", "--out", str(tmp_path / "stub")]) == 0
    pairs = [(_rec(f"q{f}{i}", f, "llama3b"), _rec(f"q{f}{i}", f, "llama8b")) for f in (0, 1, 2) for i in range(3)]
    monkeypatch.setattr(run, "pop34_pairs", lambda small, large, folds=run.ALL_FOLDS, runs_dir="runs":
                        [p for p in pairs if p[0].fold in folds])
    monkeypatch.setattr(run, "check_counts", lambda counts, folds=run.ALL_FOLDS: [])
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    return tmp_path


def _read(stub, *extra, size="small"):
    key = "llama3b" if size == "small" else "llama8b"
    return run.main(["read", "--family", "llama", "--size", size, "--engine", "stub", "--weights", str(stub / "stub" / key),
                     "--device", "cpu", "--dtype", "float32", "--threads", "2", *extra])


def test_smoke_passes_and_reads_dev_rows_only(stub, capsys):
    out = stub / "smoke"
    assert _read(stub, "--smoke", "5", "--out", str(out)) == 0
    smoke = json.loads((out / "_smoke.json").read_text())
    assert smoke["pass"] and len(smoke["rows"]) == 5
    assert all(json.loads((out / f"{q}.json").read_text())["fold"] in (0, 1) for q in smoke["rows"])
    row = json.loads((out / f"{smoke['rows'][0]}.json").read_text())
    assert row["n_bos"] == 1 and row["tracked_names"] == list(run.TRACKED) and row["variants"] == ["logit"]


def test_smoke_aborts_on_a_doubled_bos(stub, monkeypatch, capsys):
    monkeypatch.setenv("STUB_BREAK", "bos")
    out = stub / "smoke_bad"
    assert _read(stub, "--smoke", "5", "--out", str(out)) == run.EXIT_SMOKE
    assert "SMOKE FAILED" in capsys.readouterr().err
    assert not list(out.glob("q*.json"))


def test_smoke_problems_catch_depth_bos_and_reference(tmp_path):
    cfg = dataclasses.replace(F.LLAMA3B, n_layers=2, hidden=8)
    payload = {"question_id": "q1", "n_bos": 2, "tracked": {"unknown_sp": 9987, "Unknown_sp": 22435}}
    arrays = {"layers": np.arange(2), "top_logits_logit": np.zeros((2, 5, 3), np.float32),
              "top_ids_logit": np.full((2, 5, 3), 7, np.int32)}
    ref = tmp_path / "ref"
    ref.mkdir()
    np.savez(ref / "q1.npz", top_ids_logit=np.full((2, 5, 3), 8, np.int32))
    problems = run.smoke_problems([payload], [arrays], cfg, {"n_layers": 3, "d_model": 8}, str(ref))
    text = " | ".join(problems)
    assert "3 layers" in text and "2 BOS" in text and "top-1" in text
    good = {**payload, "n_bos": 1}
    np.savez(ref / "q1.npz", top_ids_logit=arrays["top_ids_logit"])
    assert run.smoke_problems([good], [arrays], cfg, {"n_layers": 2, "d_model": 8}, str(ref)) == []


def test_deadline_stops_before_any_row(stub, capsys):
    import time
    out = stub / "late"
    assert _read(stub, "--out", str(out), "--deadline", str(time.time() - 1)) == run.EXIT_DEADLINE
    cut = json.loads((out / "_deadline.json").read_text())
    assert cut["stopped_at_row"] == 0 and len(cut["unread"]) == 9
    assert not list(out.glob("q*.json"))


def test_full_read_resumes_and_hides_confirmation_outcomes(stub, capsys):
    out = stub / "full"
    assert _read(stub, "--out", str(out)) == 0
    lines = [l for l in capsys.readouterr().out.splitlines() if l.startswith("[")]
    assert len(lines) == 9
    assert all("P1 gold" not in l for l in lines if " f2 " in l) and any("P1 gold" in l for l in lines if " f0 " in l)
    assert _read(stub, "--out", str(out)) == 0
    assert "0 to read" in capsys.readouterr().out


def test_jlens_is_small_only(stub, tmp_path):
    lens = tmp_path / "lens.pt"
    lens.write_bytes(b"x")
    assert _read(stub, "--variant", "jlens", "--lens", str(lens), "--out", str(tmp_path / "j"), size="large") == 2


def test_estimates_fit_the_hard_stop():
    for fam, est in run.EST_S.items():
        total = sum(v * (2 if k == "smoke" else 1) for k, v in est.items() if k != "fit_min")
        assert total < 9900, (fam, total)
        assert est["fit_min"] < est["fit"]


# --- the report, on synthetic rows ---------------------------------------------------------------------------------

def _write_row(d: Path, qid, fold, best, final, variants=("logit",), gold_text=" Leeds"):
    """A read with gold ranks per layer: rank `best` at layer 1, `final` at the last of 4 layers."""
    d.mkdir(parents=True, exist_ok=True)
    ranks = np.full((4, 5, 4), 500, np.int32)
    ranks[1, 0, 0], ranks[3, 0, 0] = best, final
    best_rank = int(ranks[:, 0, 0].min())
    out = "C" if final <= 10 else ("B" if best_rank <= 10 else "A")
    first = next((l for l in range(4) if ranks[l, 0, 0] <= 10), None)
    m = {"P1": {"gold": {"best_rank": best_rank, "final_rank": int(final), "outcome": out,
                         "emergence_depth": None if first is None else first / 4}}}
    arrays = {"layers": np.arange(4), "position_index": np.arange(5)}
    for v in variants:
        arrays[f"ranks_{v}"] = ranks
    np.savez(d / f"{qid}.npz", **arrays)
    sha = __import__("hashlib").sha256(PROMPT.encode()).hexdigest()
    (d / f"{qid}.json").write_text(json.dumps({"question_id": qid, "fold": fold, "prompt_sha256": sha,
                                               "gold_token_text": gold_text, "measures": {v: m for v in variants}}))


def _report_fixture(tmp_path):
    root = tmp_path / "pod"
    rows = {  # qid: fold, small (best, final), large (best, final), small correct, small unknown
        "a0": (0, (100, 100), (10, 1), False, True), "a1": (0, (5, 5), (1, 1), True, False),
        "a2": (1, (1000, 1000), (1000, 2000), False, False), "a3": (1, (3, 3), (2, 2), True, False),
        "c0": (2, (1000, 1000), (1, 1), False, True), "c1": (2, (10, 50), (1, 1), False, False),
        "c2": (3, (1, 1), (1, 1), True, False), "c3": (4, (2, 2), (100, 100), True, False),
    }
    pairs = []
    for q, (fold, s, l, corr, unk) in rows.items():
        _write_row(root / "logit" / "llama3b", q, fold, *s)
        _write_row(root / "logit" / "llama8b", q, fold, *l)
        pairs.append((_rec(q, fold, "llama3b", corr, unk), _rec(q, fold, "llama8b", True, False)))
    return root, rows, pairs


def test_report_claims_arithmetic_on_stub_rows(tmp_path):
    root, rows, pairs = _report_fixture(tmp_path)
    md, n = rep.build("llama", str(root), confirm=True, samples=300, pairs=pairs)
    conf = [q for q, r in rows.items() if r[0] >= 2]
    lr = lambda br: math.log10(br)  # noqa: E731
    want = np.mean([lr(min(500, *rows[q][1])) - lr(min(500, *rows[q][2])) for q in conf])
    assert n["B1"]["best_logrank"]["n"] == 4 and abs(n["B1"]["best_logrank"]["mean"] - want) < 1e-9
    a_small = [min(rows[q][1]) > 10 for q in conf]
    a_large = [min(rows[q][2]) > 10 for q in conf]
    assert abs(n["B1"]["outcome_A"]["mean"] - (np.mean(a_small) - np.mean(a_large))) < 1e-9
    # B2: wrong rows c0 (log 3), c1 (log 1); right rows c2 (0), c3 (log 2): every wrong ranks worse -> AUC 1
    assert (n["B2"]["n_pos"], n["B2"]["n_neg"], n["B2"]["auc"]) == (2, 2, 1.0)
    fails = n["abc"]["small"]["all failures"]
    assert fails["n"] == 2 and fails["A"]["p"] == 0.5 and fails["B"]["p"] == 0.5     # c0 A, c1 B (10 then 50)
    assert n["abc"]["small"]["refusal"]["n"] == 1
    assert n["gate"]["verdict"] == "NOT RUN" and "Claim B1" in md and "SHOWN" in md


def test_report_dev_only_without_confirm(tmp_path):
    root, rows, pairs = _report_fixture(tmp_path)
    md, n = rep.build("llama", str(root), confirm=False, samples=300, pairs=pairs)
    assert n["B1"]["best_logrank"]["n"] == 4 and n["n"]["confirm"] == 0
    assert n["B1"]["best_logrank"]["verdict_role"] == "development, descriptive"
    assert "descriptive (dev)" in md and "confirmation folds 2-4**" not in md


def test_report_fold_guards(tmp_path):
    with pytest.raises(SystemExit) as e:
        rep.check_folds((0, 1, 2), confirm=False)
    assert e.value.code == rep.REFUSED
    rep.check_folds((0, 1, 2), confirm=True)
    root, rows, pairs = _report_fixture(tmp_path)
    with pytest.raises(ValueError):                    # a read json whose fold is outside the loaded folds
        rep.load_dir(root / "logit" / "llama3b", ["c0"], (0, 1))


def test_report_stops_on_a_prompt_hash_mismatch(tmp_path):
    root, rows, pairs = _report_fixture(tmp_path)
    p = root / "logit" / "llama8b" / "a1.json"
    d = json.loads(p.read_text())
    d["prompt_sha256"] = "0" * 64
    p.write_text(json.dumps(d))
    with pytest.raises(SystemExit) as e:
        rep.build("llama", str(root), confirm=False, samples=100, pairs=pairs)
    assert e.value.code == rep.MISMATCH


# --- the shell chain (stub; all stages skipped under a 1 s hard stop) ----------------------------------------------

@needs_exp24
def test_pod_chain_skips_stages_past_the_deadline(tmp_path):
    out = tmp_path / "chain"
    env = {**os.environ, "FAMILY": "llama", "ENGINE": "stub", "HARD_STOP_S": "1", "OUT": str(out),
           "PYTHON": str(ROOT / ".venv/bin/python")}
    r = subprocess.run(["bash", "scripts/exp43_lens_pod.sh"], cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
    log = (out / "pod.log").read_text()
    assert "HARD_STOP_S=1 s" in log and "SELF_STOP_S=10500 s" in log
    assert "SKIP smoke_small" in log
    assert r.returncode == 5                              # nothing was smoke-tested, so no full read may start
    assert (out / "pod.done").read_text().startswith("exit=5")
    stages = json.loads((out / "stage_times.json").read_text())["stages"]
    assert [s["stage"] for s in stages][:3] == ["population", "stub_models", "smoke_small"]
    assert stages[2]["status"] == "skipped"
    assert not (out / "logit").exists()


def test_bad_family_is_refused():
    r = subprocess.run(["bash", "scripts/exp43_lens_pod.sh", "--family", "mistral"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 2 and "llama, gemma or phi" in r.stderr
