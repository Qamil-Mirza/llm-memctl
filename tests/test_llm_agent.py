"""A language model as the task model, swapped in without touching the harness."""

import pytest

from memctl.agents.llm import LLMAgent
from memctl.config import resolve
from memctl.harness.runner import Experiment
from memctl.llm import ScriptedLLM, TrackedLLM, build_llm
from memctl.memory.actions import MemoryAction, Operation
from memctl.memory.compress import LLMCompressor, LLMConsolidator
from memctl.memory.engine import MemoryEngine
from memctl.memory.items import SourceType
from memctl.task import Observation, TaskState
from tests.helpers import make_state

FACT = "As of step 1, the access code of vault-317 is K93Q."


def test_the_prompt_holds_active_memory_only_and_the_answer_is_traced_to_its_item():
    state = make_state([FACT, "As of step 2, the serial of node-12 is B77Z.", "Question: what is the access code of vault-317?"])
    MemoryEngine().apply(state, [MemoryAction(Operation.MOVE_TO_ARCHIVE, ("o1",))])
    prompts = []
    agent = LLMAgent({}, llm=ScriptedLLM(lambda prompt: prompts.append(prompt) or "K93Q"))
    question = Observation("o2", state.get("o2").content, SourceType.USER, requires_response=True)
    step = agent.act(state.view(), question, TaskState(3, "Answer with the value.", question))
    assert step.action == "K93Q" and step.used_item_ids == ("o0",)
    assert "K93Q" in prompts[0] and "B77Z" not in prompts[0]  # the archived item is not shown
    assert prompts[0].startswith("Answer with the value.") and prompts[0].rstrip().endswith("Answer:")
    assert step.info["model_calls"] == 1 and step.info["prompt_tokens"] > 20


def test_a_scripted_model_that_reads_its_prompt_matches_the_scripted_reader_on_a_whole_episode():
    from memctl.envs.grammar import find_facts, parse_query

    def reader(prompt: str) -> str:
        memory, _, current = prompt.partition("## Current input")
        attribute, via, entity = parse_query(current)
        facts = find_facts(memory)
        def latest(a, e):
            found = [(stamp, value) for stamp, fa, fe, value in facts if fa == a and fe == e]
            return max(found)[1] if found else None
        if via:
            entity = latest(via, entity)
        answer = latest(attribute, entity) if entity else None
        return f"It is {answer}." if answer else "unknown"

    base = {"env": {"name": "synthetic_recall", "horizon": 120}, "controller": {"name": "lru"},
            "memory": {"budget": {"fraction": 0.25}}}
    scripted = Experiment(resolve(base)).run_episode(seed=0).episode
    with_llm = Experiment(resolve({**base, "agent": {"name": "llm", "model": {"backend": "stub"}}}))
    with_llm.agent = LLMAgent({}, llm=ScriptedLLM(reader, "scripted-llm"))
    episode = with_llm.run_episode(seed=0).episode
    assert episode["task_success"] == scripted["task_success"]
    assert episode["agent_model"] == "scripted-llm" and episode["task_model_calls"] == episode["queries"]
    assert episode["tokens_processed"] > 0


def test_usage_is_counted_and_generations_are_cached_on_disk(tmp_path):
    calls = []
    llm = TrackedLLM(ScriptedLLM(lambda prompt: calls.append(prompt) or "four words in reply"), str(tmp_path), {"name": "x"})
    assert llm.generate("one two three") == "four words in reply"
    assert llm.generate("one two three") == "four words in reply"
    assert len(calls) == 1 and llm.cache_hits == 1
    assert llm.usage.calls == 2 and llm.usage.input_tokens == 6 and llm.usage.output_tokens == 8
    assert build_llm({"backend": "stub"}).name == "stub"


def test_language_model_compressor_and_consolidator_respect_the_token_limit():
    wordy = ScriptedLLM(lambda prompt: "vault-317 code K93Q and a great many more words than were asked for here")
    assert LLMCompressor(wordy).compress("anything", 4) == "vault-317 code"  # vault, -, 317, code
    merged = LLMConsolidator(wordy).consolidate(["a b c", "d e f"], max_tokens=5)
    assert len(merged.split()) <= 5 and "vault" in merged


def test_openai_backend_retries_transient_failures(monkeypatch):
    import io
    import json as json_module
    import urllib.error

    import memctl.llm as llm_module

    replies = [urllib.error.HTTPError("u", 502, "bad gateway", {}, None), io.BytesIO(b""),
               io.BytesIO(json_module.dumps({"choices": [{"message": {"content": " K93Q "}}]}).encode())]

    class Response:
        def __init__(self, body):
            self.body = body

        def __enter__(self):
            return self.body

        def __exit__(self, *exc):
            return False

    def urlopen(request, timeout):
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return Response(reply)

    monkeypatch.setattr(llm_module.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(llm_module.time, "sleep", lambda seconds: None)
    backend = llm_module.OpenAICompatibleLLM("m", "http://example.invalid/v1")
    assert backend.generate("q") == "K93Q"
    assert not replies


def test_openai_backend_does_not_retry_client_errors(monkeypatch):
    import urllib.error

    import pytest

    import memctl.llm as llm_module

    calls = []

    def urlopen(request, timeout):
        calls.append(1)
        raise urllib.error.HTTPError("u", 403, "forbidden", {}, None)

    monkeypatch.setattr(llm_module.urllib.request, "urlopen", urlopen)
    with pytest.raises(urllib.error.HTTPError):
        llm_module.OpenAICompatibleLLM("m", "http://example.invalid/v1").generate("q")
    assert len(calls) == 1


def test_the_generation_cache_ignores_a_half_written_entry(tmp_path):
    from memctl.llm import TrackedLLM, build_llm

    llm = build_llm({"backend": "stub", "cache_dir": str(tmp_path)})
    first = llm.generate("## Memory\\nthe code is K93Q\\n## Current input\\nwhat is the code?", 8)
    entry = next(tmp_path.rglob("*.json"))
    entry.write_text('{"prompt": "trunc')  # another process caught mid-write
    again = build_llm({"backend": "stub", "cache_dir": str(tmp_path)})
    assert again.generate("## Memory\\nthe code is K93Q\\n## Current input\\nwhat is the code?", 8) == first
    assert again.cache_hits == 0 and not list(tmp_path.rglob("*.tmp"))
    assert isinstance(again, TrackedLLM)


def test_a_reasoning_reader_writes_a_note_and_its_final_answer_is_parsed_out():
    state = make_state([FACT, "As of step 2, the serial of node-12 is B77Z.", "Question: what is the access code of vault-317?"])
    prompts = []
    output = "The line [o0] gives the code of vault-317.\nAnswer: K93Q"
    agent = LLMAgent({"reasoning": True}, llm=ScriptedLLM(lambda prompt: prompts.append(prompt) or output))
    question = Observation("o2", state.get("o2").content, SourceType.USER, requires_response=True)
    step = agent.act(state.view(), question, TaskState(3, "Answer with the value.", question))
    assert step.action == "K93Q" and step.used_item_ids == ("o0",)
    assert "absolute dates" in prompts[0] and prompts[0].rstrip().endswith("Note:")
    silent = LLMAgent({"reasoning": True}, llm=ScriptedLLM(lambda prompt: "Answer:"))
    assert silent.act(state.view(), question, TaskState(3, "", question)).action == "unknown"
    # A note cut off before its answer is graded whole and flagged, never turned into a refusal.
    cut = LLMAgent({"reasoning": True}, llm=ScriptedLLM(lambda prompt: "The line [o0] says the code is K93Q and"))
    step = cut.act(state.view(), question, TaskState(3, "", question))
    assert step.action == "The line [o0] says the code is K93Q and" and step.info["truncated"]
    assert not LLMAgent({"reasoning": True}, llm=ScriptedLLM(lambda p: output)).act(
        state.view(), question, TaskState(3, "", question)).info["truncated"]


def test_memory_order_and_line_labels_are_separate_options():
    state = make_state([FACT, "As of step 2, the serial of node-12 is B77Z.", "Question: what is the access code of vault-317?"])
    question = Observation("o2", state.get("o2").content, SourceType.USER, requires_response=True)
    prompts = []
    agent = LLMAgent({"labels": "compact", "order": "arrival"}, llm=ScriptedLLM(lambda p: prompts.append(p) or "K93Q"))
    agent.act(state.view(), question, TaskState(3, "", question))
    assert "[o0]" not in prompts[0] and FACT in prompts[0] and prompts[0].rstrip().endswith("Answer:")
    assert LLMAgent({"reasoning": True}).order == "arrival" and LLMAgent({}).order == "memory"
    for bad in ({"order": "random"}, {"labels": "none"}):
        with pytest.raises(ValueError):
            LLMAgent(bad)


def test_with_counted_labels_an_items_tokens_are_the_line_the_reader_sees():
    from memctl.agents.llm import compact_line
    from memctl.memory.items import count_tokens
    from memctl.memory.state import MemoryState

    for counted in (False, True):
        state = MemoryState(100, count_labels=counted)
        item = state.ingest("t", "I went hiking.", SourceType.USER, metadata={"speaker": "Caroline", "date": "8 May 2023"})
        expected = count_tokens(compact_line(item)) if counted else count_tokens("I went hiking.")
        assert item.token_count == expected and state.active_tokens == expected
