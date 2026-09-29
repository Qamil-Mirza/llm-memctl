"""The agent: prompt contents, store retrieval and the recall(id) tool."""

from memctl.agent import Agent, StubLLM, build_prompt
from memctl.controllers.base import Placement
from memctl.embed import HashingEmbedder
from memctl.items import Place, make_item
from memctl.memory import MemoryState


class ScriptedLLM:
    """Replies with fixed outputs, one per call."""

    name = "scripted"

    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.prompts: list[str] = []

    def generate(self, prompt: str, max_new_tokens: int) -> str:
        self.prompts.append(prompt)
        return self.outputs.pop(0)


def memory_with_one_item_in_each_place() -> MemoryState:
    state = MemoryState(budget=500, embedder=HashingEmbedder())
    texts = {
        "ctx": "The meeting is on Tuesday morning.",
        "sto": "The garden gate code is 4417 for visitors.",
        "arc": "The long report says the bridge inspection found rust on the northern cable anchor.",
        "gone": "This sentence was deleted.",
    }
    for step, (item_id, text) in enumerate(texts.items(), start=1):
        state.offer(make_item(item_id, text, step, source={"speaker": "Ann"}))
    state.apply(
        [
            Placement("sto", Place.STORE, "test"),
            Placement("arc", Place.ARCHIVE, "test"),
            Placement("gone", Place.DROPPED, "test"),
        ]
    )
    return state


def test_prompt_shows_context_items_and_only_an_index_line_for_archived_ones():
    state = memory_with_one_item_in_each_place()
    prompt = build_prompt(state, "When is the meeting?", retrieved=[], recalled=[])
    assert "The meeting is on Tuesday morning." in prompt
    assert "[arc] Ann: The long report says the bridge …" in prompt
    assert "northern cable anchor" not in prompt
    assert "garden gate" not in prompt and "deleted" not in prompt


def test_store_items_are_retrieved_for_the_question():
    state = memory_with_one_item_in_each_place()
    result = Agent(StubLLM()).answer(state, "What is the garden gate code?")
    assert result.retrieved_ids == ["sto"]
    assert "4417" in result.text


def test_agent_recalls_an_archived_item_when_the_model_asks():
    state = memory_with_one_item_in_each_place()
    llm = ScriptedLLM(["recall(arc)", "rust on the northern cable anchor"])
    result = Agent(llm).answer(state, "What did the bridge inspection find?")
    assert result.recalled_ids == ["arc"] and state.recalls == 1
    assert "northern cable anchor" in llm.prompts[1] and "northern cable anchor" not in llm.prompts[0]
    assert result.text == "rust on the northern cable anchor"
    assert state.place["arc"] == Place.ARCHIVE  # recall is for this turn only


def test_stub_model_uses_recall_end_to_end():
    state = memory_with_one_item_in_each_place()
    result = Agent(StubLLM()).answer(state, "What did the long report say about the bridge?")
    assert result.recalled_ids == ["arc"]
    assert "rust" in result.text


def test_recalling_a_dropped_item_fails_and_is_recorded():
    state = memory_with_one_item_in_each_place()
    result = Agent(ScriptedLLM(["recall(gone)"])).answer(state, "What was deleted?")
    assert result.failed_recalls == ["gone"] and result.recalled_ids == []
    assert state.recalls == 0


def test_recall_rounds_are_limited():
    state = memory_with_one_item_in_each_place()
    state.apply([Placement("ctx", Place.ARCHIVE, "test"), Placement("sto", Place.ARCHIVE, "test")])
    llm = ScriptedLLM(["recall(arc)", "recall(ctx)", "recall(sto)"])
    Agent(llm, max_recall_rounds=2).answer(state, "anything")
    assert len(llm.prompts) == 3 and state.recalls == 2
