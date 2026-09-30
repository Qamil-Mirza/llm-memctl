"""The sequential workflow environment: memory failures change the rest of the episode."""

from memctl.config import resolve
from memctl.harness.runner import run_one

ARCHIVE_OPS = ["KEEP", "EVICT", "MOVE_TO_ARCHIVE", "RETRIEVE_FROM_ARCHIVE", "NO_OP"]


def config(controller, fraction=0.2, operations=None, **env):
    controller = {"name": controller} if isinstance(controller, str) else controller
    memory = {"budget": {"fraction": fraction}}
    if operations:
        memory["allowed_operations"] = operations
    return resolve(
        {
            "env": {"name": "workflow", **env},
            "agent": {"name": "scripted_tool_agent"},
            "controller": controller,
            "memory": memory,
        }
    )


def test_with_full_memory_every_job_completes_and_nothing_restarts():
    episode = run_one(config("full_context"), seed=0).episode
    assert episode["task_success"] == 1.0
    assert episode["env_stats"]["restarts"] == 0 and episode["env_stats"]["completed_jobs"] == 16
    assert episode["steps"] < 400  # finished before the step budget ran out


def test_forgetting_a_token_forces_restarts_and_costs_completed_jobs():
    full = run_one(config("full_context"), seed=0).episode
    tight = run_one(config("fifo", fraction=0.05), seed=0).episode
    assert tight["env_stats"]["restarts"] > 0
    assert tight["task_success"] < full["task_success"]
    assert tight["steps"] > full["steps"]  # the episode itself is different, not only its score
    assert tight["failures"].get("evicted", 0) == tight["env_stats"]["restarts"]


def test_the_observation_stream_depends_on_the_memory_policy_and_hindsight_says_so():
    full = run_one(config("full_context"), seed=1)
    tight = run_one(config("fifo", fraction=0.05), seed=1)
    assert [i["content"] for i in full.items][:400] != [i["content"] for i in tight.items][:400]
    assert tight.episode["hindsight_exact"] is False


def test_the_oracle_never_forgets_a_needed_token_so_its_hindsight_stays_valid():
    for seed in range(3):
        oracle = run_one(config("oracle", fraction=0.1), seed=seed).episode
        assert oracle["task_success"] == 1.0 and oracle["env_stats"]["restarts"] == 0


def test_policies_differ_on_the_sequential_task():
    seeds = range(4)
    fifo = sum(run_one(config("fifo", fraction=0.2), s).episode["task_success"] for s in seeds)
    random_ = sum(run_one(config("random", fraction=0.2), s).episode["task_success"] for s in seeds)
    oracle = sum(run_one(config("oracle", fraction=0.2), s).episode["task_success"] for s in seeds)
    assert random_ < fifo < oracle


def test_when_live_tokens_exceed_the_budget_only_an_archive_saves_the_oracle():
    deleting = run_one(config("oracle", fraction=0.05), seed=3).episode
    archiving = run_one(config("oracle", fraction=0.05, operations=ARCHIVE_OPS), seed=3).episode
    assert deleting["task_success"] < 1.0 and deleting["env_stats"]["restarts"] > 0
    assert archiving["task_success"] == 1.0 and archiving["retrieved_items"] > 0


def test_a_retriever_that_returns_the_wrong_items_is_blamed_as_retrieval_not_as_placement():
    controller = {"name": "fifo", "removal": ["MOVE_TO_ARCHIVE"], "retrieve": {"top_k": 2}}
    episode = run_one(config(controller, fraction=0.05, operations=ARCHIVE_OPS), seed=0).episode
    assert episode["retrieved_items"] > 0 and episode["failures"]
    assert set(episode["failures"]) == {"archived_not_retrieved"}


def test_agent_noise_is_attributed_to_the_task_model():
    settings = config("full_context")
    settings["agent"]["noise"] = 0.3
    result = run_one(settings, seed=0)
    assert result.failures and {f["label"] for f in result.failures} == {"task_model_reasoning"}


def test_hindsight_metrics_are_withdrawn_once_the_episode_diverges_from_the_reference():
    diverged = run_one(config("fifo", fraction=0.05), seed=1).episode
    assert diverged["env_stats"]["restarts"] > 0
    assert diverged["hindsight_diverged_at"] is not None
    assert diverged["requirements_destroyed"] is None and diverged["unnecessary_token_share"] is None
    faithful = run_one(config("full_context"), seed=1).episode
    assert faithful["hindsight_diverged_at"] is None and faithful["requirements_destroyed"] == 0
