"""One episode: the only place where environment, agent, controller and memory meet.

Order of one step (design decision E3):

    1. the observation is added to ACTIVE (memory may now be over budget)
    2. the controller decides; the engine applies each action or rejects it
    3. the harness evicts oldest-first if memory is still over budget (logged as forced)
    4. the agent reads ACTIVE and responds, if the observation needs a response
    5. the environment scores the response and gives the next observation

There is no retrieval, compaction or any other memory change in this file
except the forced eviction in step 3. Everything else is a controller action.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field

from memctl.agents.base import Agent
from memctl.attribution import failure_label
from memctl.controllers.base import EpisodeInfo, Feedback, MemoryController, StepEvent
from memctl.envs.base import TaskEnvironment
from memctl.hindsight.collect import Hindsight
from memctl.hindsight.evidence import REMOVALS, EvidenceTracker
from memctl.memory.actions import ActionSource, ActionStatus, MemoryAction, Operation
from memctl.memory.engine import MemoryEngine
from memctl.memory.items import SourceType
from memctl.memory.state import MemoryState
from memctl.rewards import RewardFunction, StepContext
from memctl.task import Dependency, TaskState

UNLIMITED = 10**9


@dataclass
class EpisodeSettings:
    episode_id: str
    seed: int
    budget: int
    archive_budget: int | None = None
    store_agent_actions: bool = False
    detail: bool = True  # keep per-step rows; off for bulk episodes and reference passes
    max_steps: int | None = None
    experiment_id: str = ""
    budget_fraction: float | None = None
    history_tokens: int | None = None
    pricing: dict = field(default_factory=dict)
    interventions: dict = field(default_factory=dict)  # step -> [MemoryAction], applied by the harness


@dataclass
class EpisodeResult:
    episode: dict
    steps: list[dict] = field(default_factory=list)
    actions: list[dict] = field(default_factory=list)
    rewards: list[dict] = field(default_factory=list)
    retrievals: list[dict] = field(default_factory=list)
    failures: list[dict] = field(default_factory=list)
    regrets: list[dict] = field(default_factory=list)
    items: list[dict] = field(default_factory=list)
    state: MemoryState | None = None
    dependencies: list[Dependency] = field(default_factory=list)


def _removal_targets(actions: list[MemoryAction]) -> set[str]:
    return {i for action in actions if action.operation in REMOVALS for i in action.target_ids}


def run_episode(
    env: TaskEnvironment,
    agent: Agent,
    controller: MemoryController,
    engine: MemoryEngine,
    settings: EpisodeSettings,
    reward_fn: RewardFunction | None = None,
    hindsight: Hindsight | None = None,
    embedder=None,
    shadows: list[MemoryController] | None = None,
) -> EpisodeResult:
    reward_fn = reward_fn or RewardFunction()
    shadows = shadows or []
    state = MemoryState(settings.budget, settings.archive_budget, embedder)
    observation = env.reset(settings.seed)
    agent.reset(settings.seed)
    goal_embedding = embedder.embed(env.goal) if embedder and env.goal else None
    horizon = getattr(env, "horizon", None)
    info = EpisodeInfo(
        settings.episode_id, settings.seed, settings.budget, settings.archive_budget,
        engine.allowed, env.goal, horizon, embedder,
    )
    for each in [controller, *shadows]:
        each.reset(info)
        if each.uses_hindsight:
            if hindsight is None:
                raise ValueError(f"controller '{each.display_name}' needs hindsight, which is switched off")
            each.receive_hindsight(hindsight)
    tracker = EvidenceTracker(state, hindsight.dependencies if hindsight else None)
    dependency_cache: dict[str, Dependency] = {}

    result = EpisodeResult(episode={})
    diverged_at: int | None = None
    base = {"experiment_id": settings.experiment_id, "episode_id": settings.episode_id, "seed": settings.seed}
    totals = Counter()
    term_totals: Counter = Counter()
    action_counts: dict[str, Counter] = {"controller": Counter(), "harness": Counter(), "rejected": Counter()}
    failure_counts: Counter = Counter()
    shadow_overlap = {shadow.display_name: [0, 0] for shadow in shadows}
    cumulative, peak_active, peak_archive, sum_active, sum_useless = 0.0, 0, 0, 0, 0
    sum_active_with_hindsight = 0
    compact_before, compact_after = 0, 0
    controller_cost = 0.0

    while observation is not None and (settings.max_steps is None or state.step < settings.max_steps):
        state.step += 1
        step = state.step
        item = state.ingest(
            observation.id, observation.content, observation.source_type,
            metadata=observation.metadata, parent_ids=observation.parent_ids, pinned=observation.pinned,
        )
        if hindsight is not None and diverged_at is None and not hindsight.matches(item.id, observation.content):
            diverged_at = step  # the stream no longer follows the reference pass: hindsight is blind from here
        task = TaskState(step, env.goal, observation, horizon, goal_embedding)
        controller.observe(StepEvent(step, observation, item.id))
        view = state.view()

        started = time.perf_counter()
        actions = list(controller.decide(view, task))
        controller_latency = time.perf_counter() - started
        decision = controller.decision_info() or {}

        for shadow in shadows:  # asked the same question, never applied
            mine, theirs = _removal_targets(actions), _removal_targets(list(shadow.decide(view, task)))
            shadow_overlap[shadow.display_name][0] += len(mine & theirs)
            shadow_overlap[shadow.display_name][1] += len(mine | theirs)

        results = engine.apply(state, actions)
        intervened = []
        if step in settings.interventions:
            intervened = engine.apply(state, settings.interventions[step], ActionSource.INTERVENTION, enforce_allowed=False)
        forced = engine.enforce_budget(state)
        regret_rows = tracker.after_actions(step, results + intervened + forced) if diverged_at is None else []

        retrieved_ids, retrieved_tokens = [], 0
        for outcome in results:  # counted per item, so one action with many targets is not one removal
            if outcome.applied:
                action_counts["controller"][outcome.action.operation.value] += max(1, len(outcome.action.target_ids))
                if outcome.action.operation is Operation.RETRIEVE_FROM_ARCHIVE:
                    retrieved_ids += list(outcome.action.target_ids)
                    retrieved_tokens -= outcome.tokens_freed
                if outcome.created_ids:  # a rewrite: tokens after over tokens before
                    compact_before += sum(state.get(i).token_count for i in outcome.action.target_ids)
                    compact_after += sum(state.get(i).token_count for i in outcome.created_ids)
            else:
                action_counts["rejected"][outcome.action.operation.value] += max(1, len(outcome.action.target_ids))
        for outcome in forced:
            action_counts["harness"][outcome.action.operation.value] += 1
        invalid = sum(1 for outcome in results if outcome.status is ActionStatus.REJECTED)

        # ---- the agent reads memory ------------------------------------------
        agent_step, agent_latency, statuses, dependency = None, 0.0, None, None
        active_at_read = state.active_tokens
        if observation.requires_response:
            read_view = state.view()
            started = time.perf_counter()
            agent_step = agent.act(read_view, observation, task)
            agent_latency = time.perf_counter() - started
        outcome = env.step(agent_step.action if agent_step else None)
        scored = bool(outcome.info.get("scored"))
        correct = outcome.info.get("correct") if scored else None
        if observation.requires_response:
            if observation.id not in dependency_cache:
                dependency_cache = {d.query_id: d for d in env.get_ground_truth_dependencies()}
            dependency = dependency_cache.get(observation.id)
            statuses = tracker.dependency_status(dependency) if dependency else None
            state.record_access(agent_step.used_item_ids)

        # ---- evidence bookkeeping for metrics ------------------------------------
        requirements = len(statuses) if statuses else 0
        in_active = sum(1 for status in statuses if status.satisfied) if statuses else 0
        carriers_now = {i for status in statuses or [] for i in status.active}
        if scored and statuses:
            totals["queries_with_labels"] += 1
            totals["queries_with_all_evidence"] += int(in_active == requirements)
        hits = sum(1 for i in retrieved_ids if i in carriers_now)
        if statuses:
            for status in statuses:
                was_archived_only = status.satisfied and all(i in retrieved_ids for i in status.active)
                if was_archived_only or (not status.satisfied and status.archived):
                    totals["retrieval_needed"] += 1
                    totals["retrieval_got"] += int(was_archived_only)
        totals.update(
            requirements=requirements, requirements_in_active=in_active, retrieved_items=len(retrieved_ids),
            retrieved_hits=hits, retrieve_actions=sum(1 for o in results if o.applied and o.action.operation is Operation.RETRIEVE_FROM_ARCHIVE),
            forced_evictions=len(forced), invalid_actions=invalid, queries=int(scored), correct=int(bool(correct)),
            steps_with_forced=int(bool(forced)), regret_requirements=len(regret_rows),
        )

        if scored and not correct:
            label, cause = failure_label(statuses, state, retrieved_ids)
            failure_counts[label] += 1
            result.failures.append(
                {
                    **base, "step": step, "query_id": observation.id, "label": label, "cause": cause,
                    "gold": outcome.info.get("gold"), "agent_action": agent_step.action if agent_step else None,
                    "category": dependency.category if dependency else "",
                    "evidence": [status.log_row() for status in statuses or []],
                }
            )

        # ---- reward ------------------------------------------------------------------
        context = StepContext(
            step=step, env_reward=outcome.reward, active_tokens=active_at_read, budget=state.budget,
            retrieved_tokens=retrieved_tokens, controller_latency_s=controller_latency,
            controller_model_calls=int(decision.get("model_calls", 0)), invalid_actions=invalid,
            forced_evictions=len(forced), scored=scored, requirements=requirements,
            requirements_in_active=in_active, retrieved_items=len(retrieved_ids), retrieved_hits=hits,
            requirements_destroyed=len(regret_rows),
        )
        reward, terms = reward_fn(context)
        cumulative += reward
        term_totals.update(terms)

        # ---- totals --------------------------------------------------------------------
        agent_info = agent_step.info if agent_step else {}
        totals.update(
            agent_prompt_tokens=int(agent_info.get("prompt_tokens", 0)),
            agent_output_tokens=int(agent_info.get("output_tokens", 0)),
            task_model_calls=int(agent_info.get("model_calls", 0)),
            controller_model_calls=int(decision.get("model_calls", 0)),
            controller_input_tokens=int(decision.get("input_tokens", 0)),
            controller_output_tokens=int(decision.get("output_tokens", 0)),
        )
        controller_cost += float(decision.get("cost_usd", 0.0))
        totals["controller_latency_us"] += int(controller_latency * 1e6)
        totals["agent_latency_us"] += int(agent_latency * 1e6)
        peak_active = max(peak_active, active_at_read)
        peak_archive = max(peak_archive, state.archive_tokens)
        sum_active += active_at_read
        if hindsight is not None and diverged_at is None:
            sum_useless += tracker.useless_active_tokens(step, exclude=observation.id)
            sum_active_with_hindsight += active_at_read

        if settings.store_agent_actions and agent_step and agent_step.action:
            state.ingest(f"a{step:05d}", agent_step.action, SourceType.ACTION, parent_ids=(observation.id,))

        done = outcome.observation is None
        controller.update(
            Feedback(
                step, tuple(results), tuple(forced), reward, terms, agent_step,
                scored=scored, correct=correct, done=done,
            )
        )

        # ---- logs ------------------------------------------------------------------------
        for row in regret_rows:
            result.regrets.append({**base, **row})
        if settings.detail:
            for outcome_row in results + intervened + forced:
                result.actions.append({**base, **outcome_row.log_row()})
            if retrieved_ids:
                for record in state.retrieval_history:
                    if record["step"] == step:
                        result.retrievals.append(
                            {**base, **record, "active_carrier_ids": sorted(carriers_now), "hits": hits}
                        )
            result.rewards.append(
                {**base, "step": step, "reward": reward, "cumulative_reward": cumulative, "terms": terms}
            )
            result.steps.append(
                {
                    **base,
                    "step": step,
                    "observation_id": observation.id,
                    "requires_response": observation.requires_response,
                    "memory": state.summary(),
                    "active_tokens_at_read": active_at_read,
                    "archive_tokens": state.archive_tokens,
                    "controller": controller.display_name,
                    "controller_info": decision,
                    "controller_actions": [o.log_row() for o in results],
                    "forced_actions": [o.log_row() for o in forced],
                    "retrieved_ids": retrieved_ids,
                    "agent_action": agent_step.action if agent_step else None,
                    "agent_used_item_ids": list(agent_step.used_item_ids) if agent_step else [],
                    "scored": scored,
                    "correct": correct,
                    "reward": reward,
                    "cumulative_reward": cumulative,
                    "controller_latency_s": controller_latency,
                    "agent_latency_s": agent_latency,
                }
            )
        observation = outcome.observation

    steps = state.step
    removals = sum(
        count for source in ("controller", "harness") for op, count in action_counts[source].items()
        if Operation(op) in REMOVALS
    )
    pricing = settings.pricing or {}
    cost = (
        totals["agent_prompt_tokens"] * float(pricing.get("task_model_usd_per_mtok", 0.0)) / 1e6
        + totals["controller_input_tokens"] * float(pricing.get("controller_usd_per_mtok", 0.0)) / 1e6
        + controller_cost
    )
    final_info = controller.decision_info() or {}
    result.episode = {
        **base,
        "controller": controller.display_name,
        "controller_model": controller.model_id,
        "env": env.name,
        "agent": agent.name,
        "agent_model": agent.model_id,
        "budget": settings.budget,
        "budget_fraction": settings.budget_fraction,
        "history_tokens": settings.history_tokens,
        "steps": steps,
        "task_success": env.task_success(),
        "queries": totals["queries"],
        "correct": totals["correct"],
        "reward_total": cumulative,
        "reward_terms": dict(term_totals),
        "active_tokens_mean": sum_active / steps if steps else 0.0,
        "active_tokens_peak": peak_active,
        "archive_tokens_final": state.archive_tokens,
        "archive_tokens_peak": peak_archive,
        "retrievals": totals["retrieve_actions"],
        "retrieved_items": totals["retrieved_items"],
        "retrieval_precision": _ratio(totals["retrieved_hits"], totals["retrieved_items"]),
        "retrieval_recall": _ratio(totals["retrieval_got"], totals["retrieval_needed"]),
        "needed_hit_rate": _ratio(totals["requirements_in_active"], totals["requirements"]),
        "evidence_complete_rate": _ratio(totals["queries_with_all_evidence"], totals["queries_with_labels"]),
        "compression_ratio": _ratio(compact_after, compact_before),
        "controller_latency_s": totals["controller_latency_us"] / 1e6,
        "agent_latency_s": totals["agent_latency_us"] / 1e6,
        "tokens_processed": totals["agent_prompt_tokens"] + totals["controller_input_tokens"],
        "task_model_calls": totals["task_model_calls"],
        "controller_model_calls": totals["controller_model_calls"],
        "estimated_cost_usd": cost,
        "catastrophic_forgetting_events": sum(
            failure_counts[label] for label in ("evicted", "compression_lost_detail", "consolidation_incorrect", "invalid_action")
        ),
        # Hindsight-derived: None once the episode has diverged from the reference pass.
        "requirements_destroyed": totals["regret_requirements"] if hindsight is not None and diverged_at is None else None,
        "unnecessary_token_share": _ratio(sum_useless, sum_active_with_hindsight) if hindsight is not None and diverged_at is None else None,
        "hindsight_diverged_at": diverged_at,
        "forced_evictions": totals["forced_evictions"],
        "forced_fallback_rate": totals["steps_with_forced"] / steps if steps else 0.0,
        "forced_share_of_removals": _ratio(totals["forced_evictions"], removals),
        "invalid_actions": totals["invalid_actions"],
        "action_counts": {source: dict(counts) for source, counts in action_counts.items()},
        "failures": dict(failure_counts),
        "shadow_agreement": {name: _ratio(both, either) for name, (both, either) in shadow_overlap.items()},
        "hindsight_exact": hindsight.exact if hindsight is not None else None,
        "env_stats": env.episode_stats(),
        **{key: value for key, value in final_info.items() if key.startswith("oracle_")},
    }
    if settings.detail:
        result.items = [{**base, **item.log_row()} for item in state.items.values()]
    result.state = state
    result.dependencies = env.get_ground_truth_dependencies()
    return result


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None
