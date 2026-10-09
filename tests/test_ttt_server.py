"""§31: query-only test-time training in front of the reader (memctl/ttt_server.py), on a tiny random Qwen2."""

import copy
import json
import threading

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from memctl.ttt_server import QTTT_SUFFIX, QueryTTT, TTTSettings, _from, make_handler  # noqa: E402


def tiny_model(dtype=torch.float32, seed=0):
    config = transformers.Qwen2Config(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512)
    torch.manual_seed(seed)
    return _from(transformers.AutoModelForCausalLM.from_config, config, dtype)


def engine(steps=4, span=16, lr=1e-3, dtype=torch.float32):
    settings = TTTSettings(steps=steps, span=span, lr=lr, repetition_penalty=1.0)
    return QueryTTT(tiny_model(dtype), settings=settings, eos_ids=None, pad_id=0)


def prompt(n=60, seed=1):
    return torch.randint(1, 128, (n,), generator=torch.Generator().manual_seed(seed)).tolist()


def snapshot(model):
    return {name: p.detach().clone() for name, p in model.state_dict().items()}


def test_only_query_weights_are_trainable_and_all_are_restored():
    e = engine()
    before = snapshot(e.model)
    trainable = [name for name, p in e.model.named_parameters() if p.requires_grad]
    assert trainable and all(name.endswith("self_attn.q_proj.weight") for name in trainable)
    seen = {}
    original_step = torch.optim.AdamW.step

    def spy(self, *args, **kwargs):  # record the weights mid-adaptation: they must move
        result = original_step(self, *args, **kwargs)
        seen["moved"] = any(not torch.equal(q.weight, q.original) for q in e.queries)
        return result

    torch.optim.AdamW.step = spy
    try:
        _, info = e.answer_ids(prompt(), 4, steps=4, seed=0)
    finally:
        torch.optim.AdamW.step = original_step
    assert seen["moved"] and info["steps"] == 4
    after = snapshot(e.model)
    assert before.keys() == after.keys()
    assert all(torch.equal(before[name], after[name]) for name in before)  # nothing kept across questions


def test_no_steps_is_the_plain_greedy_reader():
    e = engine()
    reference = copy.deepcopy(tiny_model())
    ids = prompt()
    new, info = e.answer_ids(ids, 8, steps=0, seed=0)
    stock = reference.generate(input_ids=torch.tensor([ids]), attention_mask=torch.ones(1, len(ids), dtype=torch.long),
                               max_new_tokens=8, do_sample=False, pad_token_id=0)
    assert new == stock[0, len(ids):].tolist()
    assert info["loss_first"] is None


def test_cached_span_loss_equals_a_full_forward():
    e = engine()
    ids = torch.tensor([prompt()])
    with torch.no_grad():
        cache = e.model.model(input_ids=ids[:, :-1], use_cache=True).past_key_values
        full = e.model(input_ids=ids[:, :40]).logits[0, 20:39]
        expected = torch.nn.functional.cross_entropy(full.float(), ids[0, 21:40])
        got = e._span_loss(ids, cache, start=20, length=19)
    assert torch.allclose(got, expected, atol=1e-5)
    assert cache.get_seq_length() == ids.shape[1] - 1  # the span forward did not grow the shared cache


def test_answers_are_deterministic_and_order_free():
    e = engine()
    a, b = prompt(seed=1), prompt(seed=2)
    first, _ = e.answer_ids(a, 6, steps=4, seed=11)
    e.answer_ids(b, 6, steps=4, seed=22)
    again, _ = e.answer_ids(a, 6, steps=4, seed=11)
    assert first == again


def test_bfloat16_model_still_adapts_at_the_paper_learning_rate():
    """At lr 3e-6 a bfloat16 weight would not move (a weight of 0.02 rounds in steps of about 1e-4)."""
    e = engine(steps=2, lr=3e-6, dtype=torch.bfloat16)
    assert all(q.weight.dtype == torch.float32 for q in e.queries)
    moved = {}
    original_step = torch.optim.AdamW.step

    def spy(self, *args, **kwargs):
        result = original_step(self, *args, **kwargs)
        moved["any"] = any(not torch.equal(q.weight, q.original) for q in e.queries)
        return result

    torch.optim.AdamW.step = spy
    try:
        e.answer_ids(prompt(), 2, steps=2, seed=0)
    finally:
        torch.optim.AdamW.step = original_step
    assert moved["any"]


class _FakePool:
    def __init__(self):
        self.calls = []

    def submit(self, text, max_new_tokens, steps):
        self.calls.append((text, max_new_tokens, steps))
        return f"answer:{steps}", {"prompt_tokens": 3, "output_tokens": 1, "steps": steps}


def test_http_endpoint_speaks_to_memctl_openai_backend(tmp_path):
    from http.server import ThreadingHTTPServer

    from memctl.llm import OpenAICompatibleLLM

    pool = _FakePool()
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(pool, "reader", 32, str(tmp_path / "log.jsonl")))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/v1"
    try:
        assert OpenAICompatibleLLM("reader", url).generate("hello", 7) == "answer:0"
        assert OpenAICompatibleLLM("reader" + QTTT_SUFFIX, url).generate("hello", 7) == "answer:32"
        with pytest.raises(Exception):  # a name not served is refused, never answered under the wrong label
            OpenAICompatibleLLM("other", url).generate("hello", 7)
        import urllib.request

        stats = json.load(urllib.request.urlopen(url.replace("/v1", "/stats")))
        assert stats["reader" + QTTT_SUFFIX]["n"] == 1 and stats["reader"]["prompt_tokens"] == 3
    finally:
        server.shutdown()
    assert pool.calls == [("hello", 7, 0), ("hello", 7, 32)]
    assert len((tmp_path / "log.jsonl").read_text().splitlines()) == 2
