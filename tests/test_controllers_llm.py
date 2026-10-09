"""The prompted-LLM and JEV controller adapters."""

import json

import pytest

from memctl.config import resolve
from memctl.controllers import build_controller
from memctl.controllers.jev import (
    ENDPOINT, FakeJevClient, JevAnswer, JevResponse, JEVController, JevUnavailableError, TypeSafeJevClient,
    choice_confidence, server_model_seconds,
)
from tests.fake_jev_server import FakeJevServer
from memctl.controllers.prompted import PromptedLLMController
from memctl.harness.runner import run_one
from memctl.llm import ScriptedLLM
from memctl.memory.actions import Operation
from memctl.memory.engine import MemoryEngine
from memctl.memory.items import Tier
from tests.helpers import DELETE_ONLY, decide, episode_info, make_state, task_for

FOUR = ["a1 a2 a3", "b1 b2 b3", "c1 c2 c3", "d1 d2 d3"]
LONG = " ".join(["The system ran a routine check and nothing changed."] * 5)
ARCHIVE_OPS = ["KEEP", "EVICT", "MOVE_TO_ARCHIVE", "RETRIEVE_FROM_ARCHIVE", "NO_OP"]


# ---- prompted LLM -----------------------------------------------------------------


def prompted(reply, **config):
    prompts = []

    def model(prompt):
        prompts.append(prompt)
        return reply(prompt) if callable(reply) else reply

    return PromptedLLMController(config, llm=ScriptedLLM(model)), prompts


def test_prompted_controller_turns_a_json_reply_into_actions():
    controller, prompts = prompted('Sure: [{"operation": "EVICT", "target_ids": ["o0", "o1"]}]')
    actions = decide(controller, make_state(FOUR, budget=7), DELETE_ONLY)
    assert [(a.operation, a.target_ids) for a in actions] == [(Operation.EVICT, ("o0", "o1"))]
    info = controller.decision_info()
    assert info["model_calls"] == 1 and info["input_tokens"] > 0 and info["parse_ok"]


def test_prompted_controller_shows_the_budget_the_items_and_only_the_allowed_operations():
    controller, prompts = prompted("[]")
    decide(controller, make_state(FOUR, budget=7), DELETE_ONLY)
    prompt = prompts[0]
    assert "budget: 7 tokens" in prompt and "free at least 5 tokens" in prompt
    assert "o2 | 3 |" in prompt and "- EVICT:" in prompt
    assert "MOVE_TO_ARCHIVE" not in prompt and "COMPACT" not in prompt


def test_prompted_controller_is_not_called_while_memory_fits():
    controller, prompts = prompted("[]")
    assert decide(controller, make_state(FOUR, budget=100), DELETE_ONLY) == []
    assert prompts == [] and controller.decision_info() == {}


def test_a_reply_that_is_not_json_yields_no_action_and_is_reported():
    controller, _ = prompted("I think we should delete the first one.")
    assert decide(controller, make_state(FOUR, budget=7), DELETE_ONLY) == []
    assert controller.decision_info()["parse_ok"] is False


def test_the_models_mistakes_reach_the_engine_and_are_rejected_there():
    reply = '[{"operation": "MOVE_TO_ARCHIVE", "target_ids": ["o0"]}, {"operation": "EVICT", "target_ids": ["nope"]}, {"operation": "FLY"}]'
    controller, _ = prompted(reply)
    state = make_state(FOUR, budget=7)
    actions = decide(controller, state, DELETE_ONLY)
    assert len(actions) == 2 and controller.decision_info()["unparsed_entries"] == 1
    results = MemoryEngine(DELETE_ONLY).apply(state, actions)
    assert [r.applied for r in results] == [False, False]


def test_prompted_controller_runs_a_whole_episode_and_its_calls_are_counted():
    def evict_oldest(prompt):
        rows = [line for line in prompt.splitlines() if line.startswith("o") and " | " in line]
        return json.dumps([{"operation": "EVICT", "target_ids": [row.split(" | ")[0] for row in rows[:3]]}])

    controller = PromptedLLMController({}, llm=ScriptedLLM(evict_oldest, "scripted-evictor"))
    config = resolve({"env": {"name": "synthetic_recall", "horizon": 80}, "memory": {"budget": {"fraction": 0.2}}})
    episode = run_one(config, seed=0, controller=controller).episode
    assert episode["controller_model_calls"] > 0 and episode["tokens_processed"] > 0
    assert episode["controller_model"] == "scripted-evictor"


def test_prompted_controller_offers_archived_items_when_a_query_arrives():
    controller, prompts = prompted(lambda prompt: '[{"operation": "RETRIEVE_FROM_ARCHIVE", "target_ids": ["o0"]}]')
    state = make_state(["the code of vault-317 is K93Q", "filler words here"], budget=100)
    controller.reset(episode_info(state))
    state.move("o0", Tier.ARCHIVE, "MOVE_TO_ARCHIVE", "controller")
    state.step = 3
    from memctl.memory.items import SourceType

    state.ingest("q", "what is the code of vault-317?", SourceType.USER)
    actions = controller.decide(state.view(), task_for(state, "q", query=True))
    assert "Archived items that may be relevant" in prompts[0] and "o0 | " in prompts[0]
    assert actions[0].operation is Operation.RETRIEVE_FROM_ARCHIVE


# ---- JEV ----------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def no_jev_endpoint_override(monkeypatch):
    """A JEV_BASE_URL or JEV_MODEL in the developer's shell must not change what these tests see."""
    for name in ("JEV_BASE_URL", "JEV_MODEL", "OPENJEV_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def test_jev_refuses_to_start_without_credentials(monkeypatch):
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    with pytest.raises(JevUnavailableError, match="JEV_API_KEY"):
        build_controller({"name": "jev"})


def test_the_fake_client_must_be_asked_for_and_is_labelled(monkeypatch):
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    controller = build_controller({"name": "jev", "allow_fake": True})
    assert controller.display_name == "jev-FAKE" and controller.model_id == "FAKE-jev"


def test_with_a_key_the_real_client_is_built_and_the_key_is_not_exposed(monkeypatch):
    monkeypatch.setenv("JEV_API_KEY", "secret-key-123")
    controller = build_controller({"name": "jev"})
    assert isinstance(controller.client, TypeSafeJevClient) and controller.display_name == "jev"
    request = controller.client.build_request({"memory_items": {}}, {"item_0": {"type": "choice"}})
    assert set(request) == {"model", "state", "questions"} and "secret-key-123" not in json.dumps(request)
    assert "secret-key-123" not in repr(controller.client.__dict__.get("model"))


def test_the_real_client_parses_the_documented_response_shape():
    client = TypeSafeJevClient("k")
    payload = {
        "model": "jev-1.13.0",
        "answers": {"item_0": {"choice": "EVICT", "probabilities": {"KEEP": 0.2, "EVICT": 0.8}, "confidence": 0.6}},
        "usage": {"input_tokens": 1000},
    }
    response = client.parse_response(payload, latency_s=0.25)
    assert response.answers["item_0"] == JevAnswer("EVICT", {"KEEP": 0.2, "EVICT": 0.8}, 0.6)
    assert response.model == "jev-1.13.0" and response.cost_usd == pytest.approx(1000 * 0.042 / 1e6)
    assert choice_confidence({"KEEP": 0.2, "EVICT": 0.8}) == pytest.approx(0.6)


def test_without_an_override_the_typesafe_endpoint_model_and_price_are_unchanged(monkeypatch):
    monkeypatch.setenv("JEV_API_KEY", "secret-key-123")
    client = build_controller({"name": "jev"}).client
    assert client.endpoint == "https://api.typesafe.ai/v1/systemone" == ENDPOINT
    assert client.model == "jev-latest" and client.usd_per_million == 0.042 and client.retries == 0


def test_the_endpoint_and_model_come_from_env_and_config_wins(monkeypatch):
    monkeypatch.setenv("JEV_API_KEY", "secret-key-123")
    monkeypatch.setenv("JEV_BASE_URL", "http://env-host:8080/")
    monkeypatch.setenv("JEV_MODEL", "openjev-latest")
    client = build_controller({"name": "jev"}).client
    assert (client.endpoint, client.model) == ("http://env-host:8080/v1/systemone", "openjev-latest")
    client = build_controller({"name": "jev", "base_url": "http://cfg-host:9000", "model": "openjev-0.1"}).client
    assert (client.endpoint, client.model) == ("http://cfg-host:9000/v1/systemone", "openjev-0.1")
    assert build_controller({"name": "jev", "endpoint": "http://x/y"}).client.endpoint == "http://x/y"
    # A self-hosted server is free per token and never gets the TypeSafe key.
    assert client.usd_per_million == 0.0 and client._api_key is None


def test_the_override_reaches_a_local_server_over_http_and_the_typesafe_key_stays_home(monkeypatch):
    server = FakeJevServer().start()
    try:
        monkeypatch.setenv("JEV_API_KEY", "secret-key-123")
        monkeypatch.setenv("JEV_BASE_URL", server.base_url)
        controller = build_controller({"name": "jev", "model": "openjev-latest"})
        actions = decide(controller, make_state(FOUR, budget=7))
        assert server.requests and all(
            r == {"path": "/v1/systemone", "model": "openjev-latest", "authorization": False, "questions": 4}
            for r in server.requests
        )
        assert actions and controller.decision_info()["cost_usd"] == 0.0
        assert controller.decision_info()["server_model_s"] > 0 and controller.model_id == "openjev-0.1-FAKE"
    finally:
        server.shutdown()


def test_a_cached_answer_replays_without_any_server_and_a_miss_is_an_error(tmp_path):
    server = FakeJevServer().start()
    try:
        config = {"name": "jev", "base_url": server.base_url, "cache_dir": str(tmp_path / "jev")}
        first = build_controller(config)
        first_actions = decide(first, make_state(FOUR, budget=7))
        calls = len(server.requests)
        assert calls and first.decision_info()["cached_calls"] == 0
    finally:
        server.shutdown()
    # The server is gone and the endpoint changed: only the cache can answer.
    replay = build_controller({**config, "base_url": "http://127.0.0.1:9", "cache_only": True})
    assert decide(replay, make_state(FOUR, budget=7)) == first_actions
    info = replay.decision_info()
    assert info["cached_calls"] == calls and info["server_model_s"] == first.decision_info()["server_model_s"]
    with pytest.raises(JevUnavailableError, match="cache_only"):
        decide(replay, make_state(FOUR + ["e1 e2 e3"], budget=7))


def test_server_timing_is_read_in_seconds():
    assert server_model_seconds("model;dur=41.2, server;dur=2.8, total;dur=44.0") == pytest.approx(0.0412)
    assert server_model_seconds(None) is None and server_model_seconds("total;dur=3") is None


class ChoosingClient:
    """A test double: answers every question with a fixed choice per item text."""

    is_fake = True

    def __init__(self, choices):
        self.choices = choices
        self.requests = []

    def ask(self, state, questions):
        self.requests.append((state, questions))
        answers = {}
        for key, question in questions.items():
            options = list(question["criteria"])
            wanted = self.choices.get(state["memory_items"][key]["text"], "KEEP")
            choice = wanted if wanted in options else "KEEP"
            probabilities = {o: (0.7 if o == choice else 0.3 / (len(options) - 1)) for o in options}
            answers[key] = JevAnswer(choice, probabilities, choice_confidence(probabilities))
        return JevResponse(answers, "double", 100, 0, 0.01, 0.0, fake=True)


def test_jev_answers_become_structured_actions_with_confidence():
    client = ChoosingClient({"a1 a2 a3": "EVICT", "b1 b2 b3": "MOVE_TO_ARCHIVE"})
    controller = JEVController({}, client=client)
    actions = decide(controller, make_state(FOUR, budget=7))
    assert {(a.operation, a.target_ids) for a in actions} == {
        (Operation.EVICT, ("o0",)), (Operation.MOVE_TO_ARCHIVE, ("o1",))
    }
    assert all(a.confidence is not None for a in actions)
    info = controller.decision_info()
    assert set(info["probabilities"]) == {"o0", "o1", "o2", "o3"}
    assert info["model_calls"] == 1 and info["input_tokens"] == 100 and "latency_s" in info and info["repaired_items"] == 0


def test_jev_is_offered_only_the_operations_the_experiment_allows():
    client = ChoosingClient({})
    controller = JEVController({"repair": False}, client=client)
    decide(controller, make_state(FOUR + [LONG], budget=7), DELETE_ONLY)
    _, questions = client.requests[0]
    assert all(set(q["criteria"]) == {"KEEP", "EVICT"} for q in questions.values())
    all_ops = ChoosingClient({})
    decide(JEVController({}, client=all_ops), make_state(FOUR + [LONG], budget=7))
    options = [set(q["criteria"]) for q in all_ops.requests[0][1].values()]
    assert {"COMPACT", "COMPACT_AND_ARCHIVE"} <= options[-1] and "COMPACT" not in options[0]  # only long items


def test_jev_choices_that_do_not_fit_are_repaired_by_least_confident_keep():
    client = ChoosingClient({})  # keeps everything
    state = make_state(FOUR, budget=7)
    actions = decide(JEVController({}, client=client), state, DELETE_ONLY)
    assert actions and all(a.parameters.get("repaired") for a in actions)
    MemoryEngine(DELETE_ONLY).apply(state, actions)
    assert state.active_tokens <= 7
    unrepaired = decide(JEVController({"repair": False}, client=ChoosingClient({})), make_state(FOUR, budget=7), DELETE_ONLY)
    assert unrepaired == []


def test_jev_is_not_called_while_memory_fits_and_batches_large_decisions():
    client = ChoosingClient({})
    assert decide(JEVController({}, client=client), make_state(FOUR, budget=100)) == []
    assert client.requests == []
    big = ChoosingClient({})
    decide(JEVController({"max_items_per_request": 3}, client=big), make_state(FOUR * 2, budget=7))
    assert [len(q) for _, q in big.requests] == [3, 3, 2]


def test_the_fake_jev_runs_a_whole_episode_and_reports_its_cost_fields(monkeypatch):
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    config = resolve(
        {
            "env": {"name": "synthetic_recall", "horizon": 80},
            "controller": {"name": "jev", "allow_fake": True},
            "memory": {"budget": {"fraction": 0.2}, "allowed_operations": ARCHIVE_OPS, "embedder": "hashing"},
        }
    )
    result = run_one(config, seed=0)
    assert result.episode["controller"] == "jev-FAKE" and result.episode["controller_model"] == "FAKE-jev"
    assert result.episode["controller_model_calls"] > 0 and result.episode["forced_evictions"] == 0
    decisions = [row["controller_info"] for row in result.steps if row["controller_info"]]
    assert decisions and {"probabilities", "confidence", "latency_s", "input_tokens", "output_tokens"} <= set(decisions[0])
    assert isinstance(FakeJevClient(), FakeJevClient)
