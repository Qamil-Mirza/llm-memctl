"""Reward terms. Each term is one number per step; the reward is a weighted sum.

Every term is computed and logged every step whatever its weight, so a run can
be re-scored under a different reward without being re-run. To add a term,
write a function of `StepContext` and register it with `@term("name")`.

    reward:
      weights: {task_reward: 1.0, forced_fallback: -0.1, active_memory_cost: -0.01}

Terms marked (hindsight) read ground truth about the future. They are for reward
shaping, diagnostics and imitation, and are zero when hindsight is off.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

DEFAULT_WEIGHTS = {"task_reward": 1.0, "forced_fallback": -0.1, "invalid_action": -0.1}


@dataclass
class StepContext:
    step: int
    env_reward: float
    active_tokens: int
    budget: int
    retrieved_tokens: int = 0
    controller_latency_s: float = 0.0
    controller_model_calls: int = 0
    invalid_actions: int = 0
    forced_evictions: int = 0
    scored: bool = False
    requirements: int = 0  # evidence requirements of the query answered at this step
    requirements_in_active: int = 0
    retrieved_items: int = 0
    retrieved_hits: int = 0  # retrieved items that the current query needed
    requirements_destroyed: int = 0  # (hindsight) future requirements made unrecoverable now
    extra: dict = field(default_factory=dict)


TERMS: dict[str, Callable[[StepContext], float]] = {}


def term(name: str):
    def register(function: Callable[[StepContext], float]):
        TERMS[name] = function
        return function

    return register


@term("task_reward")
def _task_reward(ctx: StepContext) -> float:
    return ctx.env_reward


@term("active_memory_cost")
def _active_memory_cost(ctx: StepContext) -> float:
    """Share of the budget in use."""
    return ctx.active_tokens / ctx.budget if ctx.budget else 0.0


@term("retrieval_cost")
def _retrieval_cost(ctx: StepContext) -> float:
    """Retrieved tokens as a share of the budget."""
    return ctx.retrieved_tokens / ctx.budget if ctx.budget else 0.0


@term("controller_compute_cost")
def _controller_compute_cost(ctx: StepContext) -> float:
    """Seconds the controller took to decide."""
    return ctx.controller_latency_s


@term("invalid_action")
def _invalid_action(ctx: StepContext) -> float:
    return float(ctx.invalid_actions)


@term("forced_fallback")
def _forced_fallback(ctx: StepContext) -> float:
    """Items the harness had to evict because the controller left memory over budget."""
    return float(ctx.forced_evictions)


@term("memory_hit_rate")
def _memory_hit_rate(ctx: StepContext) -> float:
    """On a query step: the share of needed evidence that was in ACTIVE."""
    return ctx.requirements_in_active / ctx.requirements if ctx.requirements else 0.0


@term("retrieval_success")
def _retrieval_success(ctx: StepContext) -> float:
    """Retrieved items that the current query needed."""
    return float(ctx.retrieved_hits)


@term("hindsight_eviction_regret")
def _hindsight_eviction_regret(ctx: StepContext) -> float:
    """(hindsight) Future evidence requirements made unrecoverable at this step."""
    return float(ctx.requirements_destroyed)


class RewardFunction:
    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self.weights = dict(DEFAULT_WEIGHTS if weights is None else weights)
        unknown = [name for name in self.weights if name not in TERMS]
        if unknown:
            raise KeyError(f"unknown reward terms {unknown}. Known: {sorted(TERMS)}")

    def __call__(self, ctx: StepContext) -> tuple[float, dict[str, float]]:
        """(weighted total, every term's raw value)."""
        values = {name: function(ctx) for name, function in TERMS.items()}
        total = sum(weight * values[name] for name, weight in self.weights.items())
        return total, values
