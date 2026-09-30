"""Aggregate the episode rows of one run into summary.json."""

from __future__ import annotations

import random
from collections import Counter

MEAN_FIELDS = [
    "task_success", "reward_total", "active_tokens_mean", "active_tokens_peak", "archive_tokens_final",
    "archive_tokens_peak", "retrievals", "retrieved_items", "retrieval_precision", "retrieval_recall",
    "needed_hit_rate", "evidence_complete_rate", "compression_ratio", "controller_latency_s", "agent_latency_s", "tokens_processed",
    "task_model_calls", "controller_model_calls", "estimated_cost_usd", "catastrophic_forgetting_events",
    "requirements_destroyed", "unnecessary_token_share", "forced_evictions", "forced_fallback_rate",
    "forced_share_of_removals", "invalid_actions", "queries", "correct", "steps", "budget",
]


def mean(values: list[float]) -> float | None:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def bootstrap_ci(values: list[float], samples: int = 2000, seed: int = 0, level: float = 0.95) -> tuple[float, float] | None:
    """A percentile bootstrap interval for the mean, resampling episodes."""
    values = [v for v in values if v is not None]
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(samples))
    low = means[int((1 - level) / 2 * samples)]
    high = means[min(samples - 1, int((1 + level) / 2 * samples))]
    return low, high


def summarize_episodes(episodes: list[dict]) -> dict:
    if not episodes:
        return {"episodes": 0}
    summary: dict = {"episodes": len(episodes)}
    for name in MEAN_FIELDS:
        summary[name] = mean([episode.get(name) for episode in episodes])
    summary["task_success_ci95"] = bootstrap_ci([episode["task_success"] for episode in episodes])
    summary["controller_latency_per_step_s"] = mean(
        [episode["controller_latency_s"] / episode["steps"] for episode in episodes if episode.get("steps")]
    )
    for key in ("controller", "env", "agent", "budget_fraction", "controller_model", "agent_model"):
        summary[key] = episodes[0].get(key)
    queries = sum(episode["queries"] for episode in episodes)
    summary["query_accuracy"] = sum(episode["correct"] for episode in episodes) / queries if queries else None

    failures: Counter = Counter()
    actions: dict[str, Counter] = {}
    terms: Counter = Counter()
    shadows: dict[str, list[float]] = {}
    for episode in episodes:
        failures.update(episode.get("failures", {}))
        terms.update(episode.get("reward_terms", {}))
        for source, counts in episode.get("action_counts", {}).items():
            actions.setdefault(source, Counter()).update(counts)
        for name, value in (episode.get("shadow_agreement") or {}).items():
            shadows.setdefault(name, []).append(value)
    summary["failures"] = dict(failures)
    failed = sum(failures.values())
    summary["failure_shares"] = {label: count / failed for label, count in failures.items()} if failed else {}
    summary["action_counts"] = {source: dict(counts) for source, counts in actions.items()}
    summary["reward_terms_mean"] = {name: total / len(episodes) for name, total in terms.items()}
    summary["shadow_agreement"] = {name: mean(values) for name, values in shadows.items()}
    summary["hindsight_exact"] = all(episode.get("hindsight_exact") for episode in episodes)
    oracle_methods = Counter(episode["oracle_method"] for episode in episodes if "oracle_method" in episode)
    if oracle_methods:
        summary["oracle_methods"] = dict(oracle_methods)
    return summary
