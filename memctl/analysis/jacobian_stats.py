"""Statistics for the §33 claims: paired region-share differences, an AUC for within-reader
separation, and a rank correlation for the ablation check. Bootstraps resample questions."""

from __future__ import annotations

import random
import warnings

from scipy.stats import ConstantInputWarning, spearmanr

from memctl.analysis.stats import paired_bootstrap


def auc(positives: list[float], negatives: list[float]) -> float:
    if not positives or not negatives:
        return float("nan")
    wins = 0.0
    for p in positives:
        for n in negatives:
            wins += 1.0 if p > n else 0.5 if p == n else 0.0
    return wins / (len(positives) * len(negatives))


def _interval(values: list[float]) -> tuple[float, float]:
    values = sorted(values)
    return values[int(0.025 * len(values))], values[min(len(values) - 1, int(0.975 * len(values)))]


def bootstrap_auc(positives: list[float], negatives: list[float], samples: int = 10000, seed: int = 0) -> dict:
    rng = random.Random(seed)
    draws = [auc(rng.choices(positives, k=len(positives)), rng.choices(negatives, k=len(negatives))) for _ in range(samples)]
    low, high = _interval(draws)
    return {"n_pos": len(positives), "n_neg": len(negatives), "auc": auc(positives, negatives), "ci_low": low, "ci_high": high}


def bootstrap_spearman(x: list[float], y: list[float], samples: int = 10000, seed: int = 0) -> dict:
    rng = random.Random(seed)
    pairs = list(zip(x, y))
    draws = []
    with warnings.catch_warnings():
        # A resample can draw one value repeatedly; its rho is NaN and is dropped below.
        warnings.simplefilter("ignore", ConstantInputWarning)
        for _ in range(samples):
            sample = rng.choices(pairs, k=len(pairs))
            draws.append(float(spearmanr([a for a, _ in sample], [b for _, b in sample]).statistic))
    low, high = _interval([d for d in draws if d == d])
    return {"n": len(pairs), "rho": float(spearmanr(x, y).statistic), "ci_low": low, "ci_high": high}


def paired_claim(a: dict[str, float], b: dict[str, float], samples: int = 10000, seed: int = 0) -> dict:
    keys = sorted(set(a) & set(b))
    result = paired_bootstrap([a[k] - b[k] for k in keys], samples=samples, seed=seed)
    result["verdict"] = "SHOWN" if result["ci_low"] is not None and result["ci_low"] > 0 else "NOT SHOWN"
    return result
