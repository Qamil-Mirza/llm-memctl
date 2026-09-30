"""The synthetic recall environment and the scripted reader that goes with it."""

from memctl.agents.scripted import ScriptedReader
from memctl.envs import build_env
from memctl.memory.items import count_tokens
from memctl.memory.state import MemoryState
from memctl.task import TaskState

CONFIG = {"name": "synthetic_recall", "horizon": 300}


def play(env_config, seed, noise=0.0, budget=10**9):
    """Run an episode with everything kept in memory. Returns (env, observations, outcomes)."""
    env = build_env(env_config)
    reader = ScriptedReader({"noise": noise})
    reader.reset(seed)
    state = MemoryState(budget=budget)
    observation = env.reset(seed)
    observations, outcomes = [], []
    while observation is not None:
        state.step += 1
        state.ingest(observation.id, observation.content, observation.source_type)
        observations.append(observation)
        action = None
        if observation.requires_response:
            task = TaskState(step=state.step, goal=env.goal, observation=observation)
            action = reader.act(state.view(), observation, task).action
        result = env.step(action)
        if observation.requires_response:
            outcomes.append(result.info["correct"])
        observation = result.observation
    return env, observations, outcomes


def test_the_same_seed_gives_the_same_episode_and_different_seeds_differ():
    _, first, _ = play(CONFIG, seed=3)
    _, again, _ = play(CONFIG, seed=3)
    _, other, _ = play(CONFIG, seed=4)
    assert [o.content for o in first] == [o.content for o in again]
    assert [o.content for o in first] != [o.content for o in other]


def test_the_episode_has_the_configured_length_and_unique_ids():
    env, observations, _ = play(CONFIG, seed=0)
    assert len(observations) == 300 and env.is_done()
    assert len({o.id for o in observations}) == 300


def test_a_perfect_reader_with_full_memory_answers_every_query():
    for seed in range(5):
        env, _, outcomes = play(CONFIG, seed)
        assert len(outcomes) >= 10
        assert all(outcomes)
        assert env.task_success() == 1.0


def test_every_dependency_points_at_earlier_items_that_contain_the_needle():
    env, observations, _ = play(CONFIG, seed=1)
    by_id = {o.id: (number + 1, o) for number, o in enumerate(observations)}
    dependencies = env.get_ground_truth_dependencies()
    queries = [o for o in observations if o.requires_response]
    assert {d.query_id for d in dependencies} == {q.id for q in queries}
    for dependency in dependencies:
        assert dependency.requirements
        for requirement in dependency.requirements:
            for item_id in requirement.item_ids:
                step, observation = by_id[item_id]
                assert step < dependency.step
                assert requirement.needle in observation.content


def test_dependency_gaps_respect_the_configured_minimum():
    config = {**CONFIG, "gap": {"min": 40, "max": 200}, "two_hop_prob": 0.0, "update_prob": 0.0}
    env, _, _ = play(config, seed=2)
    arrival = {o: int(o[1:]) for d in env.get_ground_truth_dependencies() for r in d.requirements for o in r.item_ids}
    for dependency in env.get_ground_truth_dependencies():
        newest = max(arrival[i] for r in dependency.requirements for i in r.item_ids[:1])
        assert dependency.step - newest >= 40


def test_the_generator_produces_updates_restatements_and_two_hop_queries():
    env, observations, _ = play({**CONFIG, "horizon": 800}, seed=5)
    dependencies = env.get_ground_truth_dependencies()
    assert any(len(d.requirements) == 2 for d in dependencies)  # two-hop
    assert any(len(r.item_ids) > 1 for d in dependencies for r in d.requirements)  # restated
    assert env.episode_stats()["updates"] > 0


def test_queries_come_from_more_than_one_source_type_and_distractors_vary_in_size():
    _, observations, _ = play(CONFIG, seed=6)
    sizes = [count_tokens(o.content) for o in observations if not o.requires_response]
    assert max(sizes) > 4 * min(sizes)
    assert len({o.source_type for o in observations}) >= 3


def test_reader_noise_turns_answers_wrong_at_the_configured_rate():
    _, _, outcomes = play(CONFIG, seed=0, noise=1.0)
    assert not any(outcomes)
    rates = []
    for seed in range(20):
        _, _, outcomes = play(CONFIG, seed, noise=0.25)
        rates += outcomes
    wrong = 1 - sum(rates) / len(rates)
    assert 0.15 < wrong < 0.35


def test_a_reader_without_the_evidence_answers_unknown():
    env = build_env(CONFIG)
    observation = env.reset(0)
    reader = ScriptedReader({})
    reader.reset(0)
    while not observation.requires_response:
        observation = env.step(None).observation
    empty = MemoryState(budget=10)
    step = reader.act(empty.view(), observation, TaskState(step=1, goal=env.goal, observation=observation))
    assert step.action == "unknown" and step.used_item_ids == ()
    assert env.step(step.action).info["correct"] is False


def test_snapshot_and_restore_rewind_the_environment():
    env = build_env(CONFIG)
    env.reset(0)
    for _ in range(10):
        env.step(None)
    saved = env.snapshot()
    expected = env.get_observation().content
    for _ in range(10):
        env.step(None)
    env.restore(saved)
    assert env.get_observation().content == expected


def test_an_answer_wrapped_in_a_sentence_still_counts():
    env = build_env(CONFIG)
    observation = env.reset(0)
    while not observation.requires_response:
        observation = env.step(None).observation
    gold = env.get_ground_truth_dependencies()[-1].gold
    assert env.step(f"The value is {gold}.").info["correct"] is True
