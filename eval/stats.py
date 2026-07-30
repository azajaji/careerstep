"""Bootstrap intervals and significance tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import stats


@dataclass
class TestResult:
    statistic: float
    p_value: float
    method: str
    significant: bool

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "statistic": float(self.statistic),
            "p_value": float(self.p_value),
            "significant": bool(self.significant),
        }


def paired_t_test(a: Sequence[float], b: Sequence[float], alpha: float = 0.05) -> TestResult:
    a_arr, b_arr = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    res = stats.ttest_rel(a_arr, b_arr)
    return TestResult(
        statistic=float(res.statistic),
        p_value=float(res.pvalue),
        method="paired_t_test",
        significant=bool(res.pvalue < alpha),
    )


def wilcoxon_signed_rank(a: Sequence[float], b: Sequence[float], alpha: float = 0.05) -> TestResult:
    a_arr, b_arr = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    # zero_method="zsplit" matches the behavior reported in most NLP papers
    diff = a_arr - b_arr
    if np.allclose(diff, 0):
        return TestResult(statistic=0.0, p_value=1.0, method="wilcoxon", significant=False)
    res = stats.wilcoxon(a_arr, b_arr, zero_method="zsplit")
    return TestResult(
        statistic=float(res.statistic),
        p_value=float(res.pvalue),
        method="wilcoxon_signed_rank",
        significant=bool(res.pvalue < alpha),
    )


def clopper_pearson_ci(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Exact (Clopper-Pearson) binomial confidence interval for a proportion.

    This is the correct interval for a 0/1 (Bernoulli) metric and, unlike a
    bootstrap, remains well-defined at the boundaries ``p=0`` and ``p=1`` where
    the resample variance is zero (the case that produced ``[nan, nan]`` from
    the BCa bootstrap on a saturated top-k hit rate).
    """
    if n <= 0:
        return float("nan"), float("nan")
    alpha = 1.0 - confidence
    lo = 0.0 if successes == 0 else float(stats.beta.ppf(alpha / 2.0, successes, n - successes + 1))
    hi = 1.0 if successes == n else float(stats.beta.ppf(1.0 - alpha / 2.0, successes + 1, n - successes))
    return lo, hi


def bootstrap_ci(
    values: Sequence[float],
    *,
    statistic=np.mean,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 20260101,
) -> tuple[float, float, float]:
    """Returns ``(point_estimate, ci_low, ci_high)`` via BCa bootstrap.

    Degenerate inputs are handled explicitly so the routine never emits
    ``nan`` for a defined statistic: an empty array returns ``nan`` bounds,
    and a zero-variance array (every resample identical, e.g. a saturated
    proportion) falls back to the exact binomial interval when the data are
    0/1 and to the point estimate otherwise.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(statistic(arr))
    if arr.size == 1 or np.ptp(arr) == 0:
        # No observed variability: BCa is undefined. For a 0/1 metric report
        # the exact binomial interval; otherwise the degenerate point interval.
        if statistic is np.mean and np.all((arr == 0) | (arr == 1)):
            lo, hi = clopper_pearson_ci(int(arr.sum()), int(arr.size), confidence)
            return point, lo, hi
        return point, point, point
    res = stats.bootstrap(
        (arr,),
        statistic=statistic,
        n_resamples=n_resamples,
        confidence_level=confidence,
        method="BCa",
        random_state=np.random.default_rng(seed),
    )
    return point, float(res.confidence_interval.low), float(res.confidence_interval.high)


def clustered_bootstrap_ci(
    values: Sequence[float],
    cluster_ids: Sequence,
    *,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 20260523,
) -> tuple[float, float, float]:
    """Cluster bootstrap CI for the mean of ``values``.

    ``cluster_ids[i]`` is the cluster label of observation ``values[i]``
    (e.g. its CV id, JD id, or role). Each resample draws clusters *with
    replacement* and pools every observation from the drawn clusters, then
    recomputes the mean; the percentile interval over ``n_resamples`` draws
    is returned. This accounts for within-cluster correlation (the same CVs
    and JDs recur across many ``(c, j)`` pairs), which the pair-level
    bootstrap ignores.

    Returns ``(point_estimate, ci_low, ci_high)``.
    """
    arr = np.asarray(values, dtype=float)
    ids = list(cluster_ids)
    if arr.size == 0 or len(ids) != arr.size:
        return float("nan"), float("nan"), float("nan")
    point = float(arr.mean())
    # Group observation indices by cluster (cluster ids may be tuples).
    groups: dict = {}
    for i, cid in enumerate(ids):
        groups.setdefault(cid, []).append(i)
    cluster_labels = list(groups.keys())
    cluster_idx = [np.asarray(groups[c], dtype=int) for c in cluster_labels]
    n_clusters = len(cluster_labels)
    if n_clusters < 2:
        return point, point, point
    rng = np.random.default_rng(seed)
    boots = np.empty(n_resamples, dtype=float)
    for b in range(n_resamples):
        chosen = rng.integers(0, n_clusters, size=n_clusters)
        pooled = np.concatenate([cluster_idx[c] for c in chosen])
        boots[b] = arr[pooled].mean()
    alpha = 1.0 - confidence
    return (point,
            float(np.quantile(boots, alpha / 2.0)),
            float(np.quantile(boots, 1.0 - alpha / 2.0)))


def cronbach_alpha(matrix) -> float:
    """Cronbach alpha for inter-item reliability (rows = participants, cols = items)."""
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return float("nan")
    item_vars = arr.var(axis=0, ddof=1)
    total_var = arr.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return float("nan")
    k = arr.shape[1]
    return float(k / (k - 1) * (1 - item_vars.sum() / total_var))


def summarize(values: Sequence[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"n": 0, "mean": None, "sd": None, "min": None, "max": None}
    mean, lo, hi = bootstrap_ci(arr)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "sd": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "ci95_low": lo,
        "ci95_high": hi,
    }
