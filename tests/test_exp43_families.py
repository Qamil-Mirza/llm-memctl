"""§43 rev 1 B1 + B3: the per-family config (memctl/analysis/families.py), the lens BOS fix (no special tokens on a
chat text), the §24 record loader, and the Qwen paths unchanged. Tokenizers only (cached locally, loaded with
local_files_only); the HFEngine and read_record family paths run on tiny random models built from the real tokenizers
and configs. No weights are downloaded."""
import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from memctl.analysis import answer_selection as sel
from memctl.analysis import families as F

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

PROMPT = ("Use only the memory.\n\n## Memory\nuser, d1: turn a\nuser, d2: I was born in Leeds.\n\n## Current input\n"
          "Question: (asked on 2023/05/30) Where was I born?\n\nNote:")
LINES = ("user, d1: turn a", "user, d2: I was born in Leeds.")
NOTE = "The memory says I was born in Leeds.\n"

HEAD_TAIL = {
    "llama3b": ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\n"
                "Today Date: 09 Oct 2026\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n",
                "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"),
    "gemma4b": ("<bos><start_of_turn>user\n", "<end_of_turn>\n<start_of_turn>model\n"),
    "phi4mini": ("<|user|>", "<|end|><|assistant|>"),
}
STOP_IDS = {"llama3b": {128001, 128008, 128009}, "gemma4b": {1, 106}, "phi4mini": {199999, 200020}}
BOS_ID = {"llama3b": 128000, "gemma4b": 2, "phi4mini": None}
NEW = ("llama3b", "gemma4b", "phi4mini")


def tokenizer(key):
    try:
        return transformers.AutoTokenizer.from_pretrained(F.get(key).model_id, local_files_only=True)
    except Exception as error:      # not cached on this machine
        pytest.skip(f"tokenizer for {key} unavailable: {error}")


# --- the config itself (pure) ------------------------------------------------------------------------------------

def test_config_fields_and_sweep_layers():
    assert F.LLAMA3B.sweep_layers == (17, 20, 22, 25)
    assert F.LLAMA3B.template_kwargs == {"date_string": "09 Oct 2026"}
    assert F.GEMMA4B.system_role is False and F.GEMMA4B.layer_path == "model.language_model.layers"
    assert F.PHI4MINI.bos_in_chat is False and F.PHI4MINI.lens_force_bos is False
    assert F.LLAMA3B.run_dir(1) == "runs/exp24_llama3b_f1" and F.LLAMA3B.cache_dir == "cache/generations_exp24_llama3b"
    assert {c.refusal_ids for c in F.FAMILIES.values()} == {(9987, 22435), (11908, 60355), (17927, 41855), (9788, 21693)}
    for c in F.FAMILIES.values():
        assert c.laptop_env in ({}, F.LAPTOP_ENV)
    with pytest.raises(KeyError):
        F.get("mistral7b")


def test_gemma_folds_a_system_text_into_the_user_turn():
    cfg = dataclasses.replace(F.GEMMA4B, system_text="SYS")
    assert F.messages(cfg, "hi") == [{"role": "user", "content": "SYS\n\nhi"}]
    assert F.messages(dataclasses.replace(F.LLAMA3B, system_text="SYS"), "hi")[0] == {"role": "system", "content": "SYS"}
    assert F.messages(F.GEMMA4B, "hi") == [{"role": "user", "content": "hi"}]      # §23/§24: no system message


# --- per family: template, stop ids, ':' last token, refusal ids, BOS (tokenizer only) --------------------------

@pytest.mark.parametrize("key", NEW)
def test_family_template_stop_colon_refusal_bos(key):
    cfg, tok = F.get(key), tokenizer(key)
    chat = F.chat_text(tok, cfg, PROMPT)
    head, tail = HEAD_TAIL[key]
    assert chat == head + PROMPT + tail
    assert F.stop_ids(tok, cfg) == STOP_IDS[key]
    ids = F.encode(tok, chat) + F.encode(tok, sel.frozen_continuation("A", NOTE))
    assert tok.decode([ids[-1]]).endswith(":")                        # the ':' of Answer: is the last prompt token
    assert F.check_refusal_ids(lambda t: F.encode(tok, t), cfg) == cfg.refusal_ids
    F.check_bos(tok, cfg, F.encode(tok, chat))
    with pytest.raises(ValueError):
        F.check_refusal_ids(lambda t: F.encode(tok, t), dataclasses.replace(cfg, refusal_ids=(1, 2)))


def test_llama_date_is_pinned_not_today():
    tok = tokenizer("llama3b")
    chat = F.chat_text(tok, F.LLAMA3B, PROMPT)
    assert "Today Date: 09 Oct 2026\n" in chat and chat.count("Today Date:") == 1
    unpinned = tok.apply_chat_template([{"role": "user", "content": PROMPT}], tokenize=False, add_generation_prompt=True)
    assert unpinned != chat or "09 Oct 2026" in unpinned      # (equal only if run on 09 Oct 2026)


def test_gemma_template_has_no_system_role():
    tok = tokenizer("gemma4b")
    chat = F.chat_text(tok, F.GEMMA4B, PROMPT)
    assert "system" not in chat.split(PROMPT)[0]
    folded = F.chat_text(tok, dataclasses.replace(F.GEMMA4B, system_text="Be brief."), PROMPT)
    assert folded == HEAD_TAIL["gemma4b"][0] + "Be brief.\n\n" + PROMPT + HEAD_TAIL["gemma4b"][1]


def test_stop_ids_refuse_a_missing_token_and_a_generation_eos_outside_them():
    tok = tokenizer("llama3b")
    with pytest.raises(ValueError):
        F.stop_ids(tok, dataclasses.replace(F.LLAMA3B, stop_tokens=("<end_of_turn>",)))
    with pytest.raises(ValueError):
        F.stop_ids(tok, dataclasses.replace(F.LLAMA3B, stop_tokens=("<|eot_id|>",)), [128001, 128008, 128009])
    assert F.stop_ids(tok, F.LLAMA3B, [128001, 128008, 128009]) == STOP_IDS["llama3b"]


# --- B1: exactly one BOS (Llama, Gemma), none (Phi) in the lens encode of a chat prompt ----------------------------

@pytest.mark.parametrize("key", NEW)
def test_lens_encode_of_a_chat_prompt_has_the_right_bos(key):
    from memctl.analysis.jlens_readout import encode_chat_ids, readout_for
    cfg, tok = F.get(key), tokenizer(key)
    chat = F.chat_text(tok, cfg, PROMPT)
    ids = encode_chat_ids(tok, chat, 4096)
    if BOS_ID[key] is None:
        assert tok.bos_token_id not in ids                            # Phi: its '<|endoftext|>' never appears
    else:
        assert ids.count(BOS_ID[key]) == 1 and ids[0] == BOS_ID[key]
    # the readout's ids agree (the guard in jlens_read compares the two)
    ro = readout_for(tok, PROMPT, LINES, (LINES[1],), "Leeds", chat_text_fn=lambda p: F.chat_text(tok, cfg, p))
    assert list(ro.input_ids[:ro.n_chat]) == ids
    # the bug the fix removes: jlens's force_bos route (add_bos_token on, special tokens added) puts a second BOS for
    # Llama and Gemma and an <|endoftext|> for Phi in front of the chat text
    t2 = transformers.AutoTokenizer.from_pretrained(cfg.model_id, local_files_only=True)
    if hasattr(t2, "add_bos_token"):
        t2.add_bos_token = True
    buggy = list(t2(chat).input_ids)
    assert buggy[:1] == [t2.bos_token_id] and buggy[1:] == ids


def test_qwen_lens_encode_is_unchanged_by_the_fix():
    from memctl.analysis.jacobian import wrap_chat
    from memctl.analysis.jlens_readout import encode_chat_ids
    tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct", local_files_only=True)
    chat = wrap_chat(tok, PROMPT)
    assert tok.bos_token_id is None
    assert encode_chat_ids(tok, chat, 4096) == list(tok(chat, truncation=True, max_length=4096).input_ids)


def test_eval_ids_prepend_a_bos_only_to_raw_text_of_a_bos_family():
    from scripts.jlens_eval import _ids, eval_ids
    for key, raw_bos in (("llama3b", True), ("phi4mini", False)):
        tok = tokenizer(key)
        ids, off = eval_ids(tok, "The capital of France is", raw_bos)
        assert ids[off:] == _ids(tok, "The capital of France is")
        assert off == (1 if key == "llama3b" else 0)
        chat = F.chat_text(tok, F.get(key), "hi")
        ids, off = eval_ids(tok, chat, raw_bos)
        assert off == 0 and ids == _ids(tok, chat)                     # a chat text keeps its own (single) BOS
    tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct", local_files_only=True)
    assert eval_ids(tok, "abc def", True) == (_ids(tok, "abc def"), 0)


# --- the family paths on tiny random models (real tokenizer, family layout) ---------------------------------------

def _tiny_llama(tmp_path):
    tok = tokenizer("llama3b")
    cfg = transformers.LlamaConfig(vocab_size=len(tok), hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                                   num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=4096,
                                   bos_token_id=128000, eos_token_id=[128001, 128008, 128009])
    torch.manual_seed(0)
    net = transformers.LlamaForCausalLM(cfg).eval()
    net.generation_config.eos_token_id = [128001, 128008, 128009]
    net.save_pretrained(tmp_path / "tiny-llama")
    return tok, net, dataclasses.replace(F.LLAMA3B, n_layers=2, hidden=16)


def test_hf_engine_with_a_family_config(tmp_path):
    from memctl.analysis import steering as st
    tok, _, fam = _tiny_llama(tmp_path)
    eng = st.HFEngine(F.LLAMA3B.model_id, threads=2, family=fam, weights=str(tmp_path / "tiny-llama"))
    assert eng.n_layers == 2 and eng.hidden == 16 and eng.stop_ids == STOP_IDS["llama3b"]
    assert eng.chat_text(PROMPT) == F.chat_text(tok, F.LLAMA3B, PROMPT)
    it = st.SteerInput("A", "q", 0, PROMPT, NOTE + "Answer:", "x", "y")
    prepared = eng.prepare(it)
    assert eng.last_token_text(prepared) == ":" and prepared.ids.count(128000) == 1
    g = eng.run(prepared, [st.add_spec(1, np.zeros(16, np.float32))], 3, capture=(0, 1))
    assert set(g.captured) == {0, 1}
    with pytest.raises(ValueError):          # the config's depth must match the model
        st.HFEngine(F.LLAMA3B.model_id, threads=2, family=F.LLAMA3B, weights=str(tmp_path / "tiny-llama"))


def test_read_record_runs_on_a_bos_family(tmp_path):
    jlens = pytest.importorskip("jlens")
    from scripts import jlens_read as jr
    from memctl.jacobian_data import PromptRecord
    tok, net, fam = _tiny_llama(tmp_path)
    lm = jlens.from_hf(net, tok, force_bos=True)      # the stage 0 failure: encode() adds a second BOS

    class IdentityLens:
        source_layers = [0]
        n_prompts = 0

        def transport(self, h, layer):
            return h
    rec = PromptRecord("qL", 0, "llama3b", PROMPT, NOTE + "Answer: Leeds", "Leeds", "single-session-user", True, LINES,
                       (LINES[1],), True, True, False)
    chat = F.chat_text(tok, fam, PROMPT)
    assert lm.encode(chat)[0].tolist().count(128000) == 2                  # what the old guard compared against
    payload, arrays = jr.read_record(IdentityLens(), lm, tok, rec, k=5, family=fam)
    assert payload["family"]["template_kwargs"] == {"date_string": "09 Oct 2026"}
    assert arrays["layers"].tolist() == [0, 1]


# --- the §24 loader and the stage 0 counts --------------------------------------------------------------------------

needs_exp24 = pytest.mark.skipif(not Path("runs/exp24_llama3b_f0").exists(), reason="§24 runs not on disk")


@needs_exp24
def test_family_records_and_the_stage0_population():
    from memctl.jacobian_data import family_population, load_family_records
    recs = [r for f in (0, 1) for r in load_family_records(F.LLAMA3B, f)[0]]
    assert len(recs) == 200 and sum(r.correct is not None for r in recs) == 188
    pop = family_population(recs)
    marked = [r for r in pop if sel.split_note(r.output)[0] is not None]
    refusals = [r for r in marked if sel.is_refusal_answer(sel.agent_answer(r.output)[0])]
    assert (len(pop), len(marked), len(refusals)) == (126, 125, 13)           # stage 0: 13/125 = 0.104
    assert {r.reader for r in recs} == {"llama3b"} and {r.fold for r in recs} == {0, 1}


# --- Qwen unchanged -------------------------------------------------------------------------------------------------

BASELINE = Path("runs/exp43_build/qwen_baseline_before.json")    # computed before any §43 B3 code existed


def test_qwen_prompt_ids_and_stop_ids_match_the_pre_change_baseline():
    from memctl.analysis.jacobian import wrap_chat
    base = json.loads(BASELINE.read_text()) if BASELINE.exists() else pytest.skip("baseline not on disk")
    tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct", local_files_only=True)
    prompt = ("## Memory\nUser, 2023/05/01: my cat is Luna\n\n## Current input\nQuestion: what is my cat's name?\n\nNote:")
    h = lambda x: hashlib.sha256(json.dumps(x, sort_keys=True).encode()).hexdigest()  # noqa: E731
    chat = wrap_chat(tok, prompt)
    assert h(chat) == base["qwen_chat_sha"]
    assert h(list(tok(chat + "Note text.\nAnswer:", add_special_tokens=False).input_ids)) == base["qwen_ids_sha"]
    assert sorted(F.stop_ids(tok, F.QWEN3B)) == base["qwen_stop_ids"]      # the Qwen config gives HFEngine's default
    assert F.chat_text(tok, F.QWEN3B, prompt) == chat


@pytest.mark.skipif(not Path("runs/exp35/qwen3b/A/fold0.jsonl").exists(), reason="§35 cache not on disk")
def test_exp38_stub_run_is_byte_identical_to_the_pre_change_baseline(tmp_path):
    from memctl.analysis import steering as st
    from scripts import exp38_dev
    base = json.loads(BASELINE.read_text()) if BASELINE.exists() else pytest.skip("baseline not on disk")
    h = lambda x: hashlib.sha256(json.dumps(x, sort_keys=True).encode()).hexdigest()  # noqa: E731
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(st, "ANS_GATE", 0.0)
        assert exp38_dev.main(["--run", "--backend", "stub", "--out", str(tmp_path)]) == 0
    got = {}
    for p in sorted((tmp_path / "qwen3b").glob("*.jsonl")):
        rows = [json.loads(line) for line in p.read_text().splitlines()]
        for r in rows:
            r.pop("timing", None)
        got[p.name] = {"n": len(rows), "sha": h(rows)}
    for name in ("directions.json", "gate.json", "sanity.json"):
        got[name] = h(json.loads((tmp_path / "qwen3b" / name).read_text()))
    assert got == base["stub38"]
