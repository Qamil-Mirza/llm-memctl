"""Statistics for comparing controllers that ran the same seeded episodes.

The episode is the unit, not the query: queries inside one episode share a
memory state and are correlated, so they are not independent samples.
"""

from __future__ import annotations

import math
import random
from statistics import NormalDist, mean, stdev


def paired_differences(a: dict[int, float], b: dict[int, float]) -> list[float]:
    """a - b for every seed both runs have."""
    return [a[seed] - b[seed] for seed in sorted(set(a) & set(b))]


def paired_bootstrap(differences: list[float], samples: int = 5000, seed: int = 0) -> dict:
    """Mean paired difference with a 95% percentile bootstrap interval and a two-sided p-value."""
    n = len(differences)
    if n < 2:
        return {"n": n, "mean": mean(differences) if differences else None, "ci_low": None, "ci_high": None, "p": None}
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(differences, k=n)) / n for _ in range(samples))
    observed = mean(differences)
    below = sum(1 for value in means if value <= 0) / samples
    above = sum(1 for value in means if value >= 0) / samples
    return {
        "n": n,
        "mean": observed,
        "ci_low": means[int(0.025 * samples)],
        "ci_high": means[min(samples - 1, int(0.975 * samples))],
        "p": min(1.0, 2 * min(below, above)),
    }


def minimum_detectable_effect(differences: list[float], alpha: float = 0.05, power: float = 0.8) -> float | None:
    """The smallest true mean difference a paired test on this many episodes would
    detect with the given power, from the spread of the observed paired differences."""
    if len(differences) < 2:
        return None
    z = NormalDist().inv_cdf(1 - alpha / 2) + NormalDist().inv_cdf(power)
    return z * stdev(differences) / math.sqrt(len(differences))


def oracle_gap_closed(method: float, baseline: float, oracle: float) -> float | None:
    """(method - baseline) / (oracle - baseline). None when the oracle leaves no gap."""
    gap = oracle - baseline
    return (method - baseline) / gap if abs(gap) > 1e-9 else None
