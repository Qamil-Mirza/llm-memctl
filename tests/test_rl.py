"""The RL controller, its expert labels and its training algorithms."""

import json
import random

import numpy as np
import pytest
import torch

from memctl.config import resolve
from memctl.controllers.rl import Decision, RLController, evaluate
from memctl.features import GLOBAL_DIM, ITEM_DIM, ITEM_FEATURES, Featurizer
from memctl.harness.runner import Experiment, run_experiment, run_one
from memctl.rl.algorithms import ALGORITHMS, imitation_loss
from memctl.rl.expert import ARCHIVE, EVICT, make_expert
from memctl.rl.policy import N_COLUMNS, ItemPolicy
from memctl.rl.train import fill_returns, train
from memctl.runlog import read_jsonl
from tests.helpers import DELETE_ONLY, episode_info, make_state, task_for

ARCHIVE_OPS = ["KEEP", "EVICT", "MOVE_TO_ARCHIVE", "RETRIEVE_FROM_ARCHIVE", "NO_OP"]


def config(fraction=0.1, horizon=120, operations=None, **controller):
    memory = {"budget": {"fraction": fraction}}
    if operations:
        memory["allowed_operations"] = operations
    return resolve(
        {"env": {"name": "synthetic_recall", "horizon": horizon}, "controller": {"name": "rl", **controller}, "memory": memory}
    )


def test_features_have_the_declared_shape_and_are_finite():
    state = make_state(["alpha beta gamma", "The code of vault-317 is K93Q.", "delta epsilon"], budget=5)
    rows = Featurizer().items(state.active(), state.view(), task_for(state))
    assert rows.shape == (3, ITEM_DIM) and len(ITEM_FEATURES) == ITEM_DIM and np.isfinite(rows).all()
    assert rows[1, ITEM_FEATURES.index("specific_tokens")] > rows[0, ITEM_FEATURES.index("specific_tokens")]
    assert rows[2, ITEM_FEATURES.index("is_current_observation")] == 1.0
    assert Featurizer.globals(state.view(), task_for(state)).shape == (GLOBAL_DIM,)


def test_the_policy_scores_any_number_of_items():
    policy = ItemPolicy(ITEM_DIM, GLOBAL_DIM)
    for n in (1, 7, 60):
        logits, value = policy(torch.zeros(n, ITEM_DIM), torch.zeros(GLOBAL_DIM))
        assert logits.shape == (n, N_COLUMNS) and value.shape == ()
    with pytest.raises(ValueError):
        ItemPolicy(ITEM_DIM, GLOBAL_DIM, architecture="transformer-xl")


def test_an_untrained_policy_always_fits_the_budget_with_valid_actions():
    for operations in (None, ARCHIVE_OPS):
        episode = run_one(config(operations=operations, greedy=False), seed=0).episode
        assert episode["forced_evictions"] == 0 and episode["invalid_actions"] == 0
        assert sum(episode["action_counts"]["controller"].values()) > 0
    archive_counts = episode["action_counts"]["controller"]
    assert "MOVE_TO_ARCHIVE" in archive_counts or "EVICT" in archive_counts


def test_the_recorded_probability_of_a_decision_can_be_recomputed():
    experiment = Experiment(config(operations=ARCHIVE_OPS, greedy=False))
    controller = experiment.controller
    controller.record = True
    experiment.run_episode(seed=1, detail=False)
    assert controller.recorded
    for decision in controller.recorded[:25]:
        log_prob, value, _ = evaluate(controller.policy, decision)
        assert log_prob.item() == pytest.approx(decision.log_prob, abs=1e-4)
        assert value.item() == pytest.approx(decision.value, abs=1e-5)


def test_returns_are_discounted_sums_of_later_rewards():
    decisions = [Decision(step, np.zeros((1, 1), np.float32), np.zeros(1, np.float32), ["a"], [("a",)], 1,
                          np.ones((1, 4), np.int64), np.ones((1, 4), bool), 1) for step in (1, 3)]
    fill_returns(decisions, [(1, 0.0), (2, 1.0), (3, 0.0), (4, 2.0)], gamma=0.5)
    assert decisions[0].return_to_go == pytest.approx(0.5 * 1.0 + 0.125 * 2.0)
    assert decisions[1].return_to_go == pytest.approx(0.5 * 2.0)
    fill_returns(decisions, [(1, 0.0), (2, 1.0), (3, 0.0), (4, 2.0)], gamma=0.0)
    assert [d.return_to_go for d in decisions] == [0.0, 0.0]


def test_the_expert_removes_never_needed_items_first_then_the_furthest_need():
    experiment = Experiment(config(fraction=0.05, operations=ARCHIVE_OPS))
    controller = experiment.controller
    hindsight = experiment.hindsight(0)
    controller.record, controller.expert, controller.follow_expert = True, make_expert(hindsight), True
    experiment.run_episode(seed=0, detail=False)
    checked = 0
    for decision in controller.recorded:
        rank = decision.expert_rank
        for i in range(decision.n_active):
            need = min(hindsight.next_need(root, decision.step) for root in decision.roots[i])
            if need > 10**8:
                assert rank[i, EVICT] == 0 and rank[i, ARCHIVE] == -1
            else:
                assert rank[i, ARCHIVE] >= 1 and rank[i, EVICT] == -1
                checked += 1
    assert checked > 0


def test_the_regret_expert_accepts_archive_and_deletion_alike_for_never_needed_items():
    experiment = Experiment(config(fraction=0.05, operations=ARCHIVE_OPS))
    controller = experiment.controller
    hindsight = experiment.hindsight(0)
    controller.record, controller.follow_expert = True, True
    controller.expert = make_expert(hindsight, kind="regret")
    episode = experiment.run_episode(seed=0, detail=False).episode
    never = needed = 0
    for decision in controller.recorded:
        rank = decision.expert_rank
        for i in range(decision.n_active):
            need = min(hindsight.next_need(root, decision.step) for root in decision.roots[i])
            if need > 10**8:
                assert rank[i, EVICT] == rank[i, ARCHIVE] == 0
                never += 1
            else:
                assert 0 <= rank[i, ARCHIVE] < rank[i, EVICT]
                needed += 1
    assert never > 0 and needed > 0 and episode["task_success"] == 1.0
    with pytest.raises(ValueError):
        make_expert(hindsight, kind="psychic")


def test_a_risky_archive_makes_the_regret_expert_prefer_deleting_what_is_never_needed():
    experiment = Experiment(config(fraction=0.05, operations=ARCHIVE_OPS))
    controller = experiment.controller
    hindsight = experiment.hindsight(0)
    controller.record, controller.follow_expert = True, True
    controller.expert = make_expert(hindsight, kind="regret", archive_cost=0.01)
    experiment.run_episode(seed=0, detail=False)
    for decision in controller.recorded:
        for i in range(decision.n_active):
            if min(hindsight.next_need(root, decision.step) for root in decision.roots[i]) > 10**8:
                assert decision.expert_rank[i, EVICT] < decision.expert_rank[i, ARCHIVE]


def test_following_the_expert_reproduces_the_approximate_oracle():
    for seed in (0, 1, 2):
        experiment = Experiment(config(fraction=0.03))
        controller = experiment.controller
        controller.expert, controller.follow_expert = make_expert(experiment.hindsight(seed)), True
        followed = experiment.run_episode(seed=seed, detail=False).episode["task_success"]
        oracle = resolve({**config(fraction=0.03), "controller": {"name": "oracle", "eager": False}})
        assert followed == run_one(oracle, seed).episode["task_success"]


def test_behaviour_cloning_learns_to_agree_with_the_expert():
    torch.manual_seed(0)
    experiment = Experiment(config())
    controller = experiment.controller
    controller.record, controller.follow_expert = True, True
    data = []
    for seed in range(1000, 1008):
        controller.expert = make_expert(experiment.hindsight(seed))
        experiment.run_episode(seed=seed, detail=False)
        data += controller.recorded
    before = sum(imitation_loss(controller.policy, d)[1] for d in data) / sum(imitation_loss(controller.policy, d)[2] for d in data)
    optimizer = torch.optim.Adam(controller.policy.parameters(), lr=3e-3)
    stats = {}
    for _ in range(3):
        stats = ALGORITHMS["bc"]({"epochs": 2}).update(controller.policy, optimizer, data, random.Random(0))
    assert stats["expert_agreement"] > 0.95 and stats["expert_agreement"] > before + 0.1

    controller.record, controller.expert, controller.follow_expert, controller.greedy = False, None, False, True
    learned = [experiment.run_episode(seed=s, detail=False).episode["task_success"] for s in range(8)]
    fifo = [run_one(resolve({**config(), "controller": {"name": "fifo"}}), s).episode["task_success"] for s in range(8)]
    assert sum(learned) > sum(fifo)


def test_a_policy_gradient_update_moves_the_parameters():
    torch.manual_seed(0)
    experiment = Experiment(config(greedy=False))
    controller = experiment.controller
    controller.record = True
    experiment.run_episode(seed=2000, detail=False)
    fill_returns(controller.recorded, controller.rewards, gamma=0.99)
    assert any(d.return_to_go != 0 for d in controller.recorded)
    before = [p.clone() for p in controller.policy.parameters()]
    for name in ("ppo", "reinforce"):
        optimizer = torch.optim.Adam(controller.policy.parameters(), lr=1e-2)
        stats = ALGORITHMS[name]({"epochs": 2}).update(controller.policy, optimizer, controller.recorded, random.Random(0))
        assert stats["decisions"] > 0 and np.isfinite(stats["loss"])
    assert any(not torch.equal(a, b) for a, b in zip(before, controller.policy.parameters()))


def test_a_checkpoint_restores_the_same_decisions(tmp_path):
    first = RLController({"hidden": 32}, seed=0)
    first.save(tmp_path / "policy.pt")
    second = RLController({"checkpoint": str(tmp_path / "policy.pt")}, seed=0)
    a = run_one(config(), seed=3, controller=first)
    b = run_one(config(), seed=3, controller=second)
    assert [x["target_ids"] for x in a.actions] == [x["target_ids"] for x in b.actions]
    assert second.policy.config["hidden"] == 32


def test_training_writes_a_checkpoint_that_an_ordinary_run_can_evaluate(tmp_path):
    settings = config(horizon=80)
    settings["name"] = "tiny_rl"
    settings["training"] = {
        "phases": [
            {"algorithm": "bc", "iterations": 2, "episodes": 3, "epochs": 1},
            {"algorithm": "ppo", "iterations": 2, "episodes": 3, "epochs": 1, "gamma": 0.99},
        ],
        "eval": {"every": 2, "episodes": 2},
        "budget_fractions": [0.1, 0.25],
    }
    folder = train(settings, tmp_path / "train")
    log = read_jsonl(folder / "train_log.jsonl")
    assert [row["algorithm"] for row in log] == ["bc", "bc", "ppo", "ppo"]
    assert "success@0.1" in log[1] and "expert_agreement" in log[0] and "mean_return" in log[2]
    assert json.loads((folder / "metadata.json").read_text())["status"] == "completed"
    checkpoint = folder / "checkpoints" / "policy.pt"
    assert checkpoint.exists() and (folder / "checkpoints" / "policy_phase0_bc.pt").exists()
    summary = json.loads((folder / "summary.json").read_text())
    assert (folder / "checkpoints" / "policy_best.pt").exists()
    assert summary["best_eval"]["validation"] >= sum(summary["final_eval"].values()) / len(summary["final_eval"])
    assert settings["training"]["eval"].get("seed_offset", 50_000) != 0  # validation never uses the test seeds

    evaluation = resolve({**config(horizon=80), "episodes": 2, "controller": {"name": "rl", "checkpoint": str(checkpoint)}})
    run = run_experiment(evaluation, tmp_path / "eval")
    episodes = read_jsonl(run / "episodes.jsonl")
    assert len(episodes) == 2 and episodes[0]["controller"] == "rl" and episodes[0]["forced_evictions"] == 0


def test_unused_helpers_are_importable():
    assert episode_info and DELETE_ONLY


def test_one_policy_can_be_trained_on_several_tasks(tmp_path):
    settings = config(horizon=80)
    settings["name"] = "tiny_multitask"
    settings["training"] = {
        "phases": [{"algorithm": "cost", "iterations": 2, "episodes": 2, "epochs": 1}],
        "eval": {"every": 2, "episodes": 2},
        "expert": {"kind": "regret"},
        "budget_fractions": [0.1],
        "tasks": [
            {"label": "recall", "env": {"name": "synthetic_recall", "horizon": 80}},
            {"label": "workflow", "env": {"name": "workflow", "horizon": 80, "jobs": 4},
             "agent": {"name": "scripted_tool_agent"}, "budget_fractions": [0.2]},
        ],
    }
    folder = train(settings, tmp_path / "train")
    log = read_jsonl(folder / "train_log.jsonl")
    assert {"recall/success@0.1", "workflow/success@0.2"} <= set(log[-1])
    assert log[0]["decisions"] > 0

def _ambiguous_decisions(needed_share: float, archive_cost: float, count: int = 200) -> list[Decision]:
    """One item to remove, which looks the same whether or not it is needed later."""
    rng = np.random.default_rng(0)
    decisions = []
    for _ in range(count):
        needed = rng.random() < needed_share
        mask = np.zeros((1, 4), bool)
        mask[0, [EVICT, ARCHIVE]] = True
        decision = Decision(5, np.ones((1, ITEM_DIM), np.float32), np.zeros(GLOBAL_DIM, np.float32), ["a"], [("a",)], 1,
                            np.full((1, 4), 10, np.int64), mask, 1)
        cost = np.full((1, 4), np.nan, np.float32)
        cost[0, EVICT], cost[0, ARCHIVE] = float(needed), archive_cost
        rank = np.full((1, 4), -1, np.int64)
        rank[0, EVICT], rank[0, ARCHIVE] = (1, 0) if needed else (0, 1)
        decision.expert_rank, decision.expert_retrieved, decision.expert_cost = rank, [], cost
        decisions.append(decision)
    return decisions


@pytest.mark.parametrize("algorithm, expected", [("bc", EVICT), ("cost", ARCHIVE)])
def test_with_a_priced_archive_only_cost_sensitive_imitation_archives_what_might_be_needed(algorithm, expected):
    """30% of these items are needed later and archiving costs 0.1: archiving has the lower
    expected regret (0.1 < 0.3), but deletion is least-regret 70% of the time."""
    torch.manual_seed(0)
    policy = ItemPolicy(ITEM_DIM, GLOBAL_DIM)
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)
    learner = ALGORITHMS[algorithm]({"epochs": 30, "batch_size": 50})
    learner.update(policy, optimizer, _ambiguous_decisions(0.3, 0.1), random.Random(0))
    logits, _ = policy(torch.ones(1, ITEM_DIM), torch.zeros(GLOBAL_DIM))
    assert int(torch.argmax(logits[0, [EVICT, ARCHIVE]])) == [EVICT, ARCHIVE].index(expected)


def test_the_plug_in_rule_archives_or_deletes_by_price(tmp_path):
    from memctl.rl.policy import NeededModel

    model = NeededModel(ITEM_DIM + GLOBAL_DIM)
    torch.save({"state_dict": model.state_dict(), "config": model.config}, tmp_path / "needed.pt")
    for price, kept in ((-1.0, "MOVE_TO_ARCHIVE"), (float("inf"), "EVICT")):
        settings = config(fraction=0.05, operations=ARCHIVE_OPS, needed_model=str(tmp_path / "needed.pt"), archive_price=price)
        counts = run_one(settings, seed=0).episode["action_counts"]["controller"]
        other = "EVICT" if kept == "MOVE_TO_ARCHIVE" else "MOVE_TO_ARCHIVE"
        assert counts.get(kept, 0) > 0 and counts.get(other, 0) == 0


def test_group_advantages_are_standardised_within_each_group():
    from memctl.rl.train import set_group_advantages

    make = lambda: Decision(1, np.zeros((1, 1), np.float32), np.zeros(1, np.float32), ["a"], [("a",)], 1,
                            np.ones((1, 4), np.int64), np.ones((1, 4), bool), 1)
    rows = [(0, 1.0, [make()]), (0, 3.0, [make()]), (1, 5.0, [make()]), (1, 5.0, [make()])]
    set_group_advantages(rows)
    assert rows[0][2][0].advantage == pytest.approx(-1.0, abs=1e-4)
    assert rows[1][2][0].advantage == pytest.approx(1.0, abs=1e-4)
    assert rows[2][2][0].advantage == 0.0 and rows[3][2][0].advantage == 0.0  # no spread, no signal


def test_group_advantage_modes():
    from memctl.rl.train import group_spread, set_group_advantages

    make = lambda: Decision(1, np.zeros((1, 1), np.float32), np.zeros(1, np.float32), ["a"], [("a",)], 1,
                            np.ones((1, 4), np.int64), np.ones((1, 4), bool), 1)

    def advantages(mode, values=((0, 1.0), (0, 3.0), (1, 5.0), (1, 5.1))):
        rows = [(group, value, [make()]) for group, value in values]
        set_group_advantages(rows, mode)
        return [row[2][0].advantage for row in rows]

    assert advantages("std")[2:] == pytest.approx([-1.0, 1.0], abs=1e-3)  # a 0.1 gap becomes a full +-1
    assert advantages("mean") == pytest.approx([-1.0, 1.0, -0.05, 0.05])  # Dr. GRPO keeps it small
    batch = advantages("batch")
    assert batch[1] / batch[3] == pytest.approx(20.0)  # one scale for the batch
    assert advantages("best") == [0.0, 1.0, 0.0, 1.0]
    assert advantages("best", ((0, 2.0), (0, 2.0))) == [0.0, 0.0]  # a shared best teaches nothing
    with pytest.raises(ValueError):
        advantages("nope")
    assert group_spread({"returns": [3.0, 3.0], "success": [0.5, 0.5]})["source"] == "tied"
    assert group_spread({"returns": [3.0, 2.9], "success": [0.5, 0.5]})["source"] == "fallbacks_only"
    assert group_spread({"returns": [3.0, 2.0], "success": [0.5, 0.4]})["source"] == "answers"


def test_samples_of_one_episode_differ_only_by_sample_offset():
    experiment = Experiment(config(operations=ARCHIVE_OPS, greedy=False))
    controller = experiment.controller
    controller.record = True

    def picks(offset):
        controller.sample_offset = offset
        experiment.run_episode(seed=3, detail=False)
        return [d.picks for d in controller.recorded]

    assert picks(1) == picks(1) and picks(1) != picks(2)


def test_grpo_training_runs_and_logs(tmp_path):
    settings = config(horizon=80, operations=ARCHIVE_OPS)
    settings["name"] = "tiny_grpo"
    settings["training"] = {
        "phases": [{"algorithm": "grpo", "iterations": 2, "episodes": 4, "group": 2, "epochs": 1}],
        "eval": {"every": 2, "episodes": 2},
        "budget_fractions": [0.05],
    }
    log = read_jsonl(train(settings, tmp_path / "train") / "train_log.jsonl")
    assert [row["algorithm"] for row in log] == ["grpo", "grpo"] and "success@0.05" in log[1]


def test_the_bridge_search_follows_the_first_fact_to_the_second():
    from memctl.retrieval import LexicalRetriever, bridge_search

    facts = [f"As of step {n}, the priority of node-{n} is X{n}." for n in range(10, 30)]
    facts += ["As of step 40, the courier of sensor-171 is clerk-172.",
              "As of step 42, the priority of clerk-172 is M37M.",
              "Question: what is the route of order-147?", "Question: what is the priority of the courier of sensor-171?"]
    items = make_state(facts, budget=10_000).active()
    hop1, hop2, question = items[20], items[21], items[-1]
    archive = [item for item in items if item.id not in (hop1.id, question.id)]
    retriever = LexicalRetriever()
    first = [item.id for item, _ in retriever.search(question.content, archive, 8)]
    assert hop2.id not in first  # the question's words do not reach it
    found = bridge_search(retriever, question.content, [hop1, question], archive, 4, exclude=(question.id,))
    assert found[0][0].id == hop2.id


def test_bridge_search_policies_use_feature_version_2_and_old_checkpoints_still_load(tmp_path):
    old = RLController({"hidden": 16}, seed=0)
    old.save(tmp_path / "v1.pt")
    new = RLController({"hidden": 16, "bridge_search": True}, seed=0)
    assert new.featurizer.version == 2 and new.featurizer.item_dim == old.featurizer.item_dim + 1
    new.save(tmp_path / "v2.pt")
    assert RLController({"checkpoint": str(tmp_path / "v1.pt")}).featurizer.version == 1
    assert RLController({"checkpoint": str(tmp_path / "v2.pt"), "bridge_search": True}).featurizer.version == 2
    with pytest.raises(ValueError):
        RLController({"checkpoint": str(tmp_path / "v2.pt")})
    result = run_one(config(fraction=0.05, operations=ARCHIVE_OPS), seed=1, controller=new)
    assert result.episode["action_counts"]["controller"].get("RETRIEVE_FROM_ARCHIVE", 0) > 0
