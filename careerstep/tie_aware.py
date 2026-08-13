"""Tie-aware ranking.

Several rankers in this study produce heavy score ties. Feasibility takes a
handful of distinct values, and holding a criterion at its mean during a
leave-one-criterion-out step makes ties the normal case rather than the
exception. Any single sort then resolves those ties by whatever order the
catalog happens to be in, which is not a property of the scorer.

Two devices are provided:

``expected_metric``      averages a ranking metric over random tie-breaks, so
                         the reported value is an expectation rather than one
                         arbitrary draw, with a percentile interval.

``max_set``              the set of alternatives sharing the maximum score,
                         which supports statements that do not depend on a
                         tie-break at all: whether the original leader is still
                         among the maxima after an ablation.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Sequence

import numpy as np

DEFAULT_DRAWS = 200


def rank_random_tiebreak(scores: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Descending order of ``scores`` with ties broken uniformly at random."""
    jitter = rng.random(scores.shape[0])
    return np.lexsort((jitter, -scores))


def max_set(scores: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """Indices of every alternative within ``tol`` of the maximum score."""
    return np.flatnonzero(scores >= scores.max() - tol)


def expected_metric(scores_per_case: Sequence[np.ndarray],
                    metric: Callable[[np.ndarray, int], float],
                    *, draws: int = DEFAULT_DRAWS,
                    seed: int = 0) -> Dict[str, float]:
    """Average ``metric`` over random tie-breaks.

    ``metric(order, case_index)`` receives the ranked index array for one case
    and returns that case's score. The return value is the mean over cases,
    averaged over ``draws`` independent tie-break draws, with a 95 per cent
    percentile interval over the draws.
    """
    rng = np.random.default_rng(seed)
    per_draw: List[float] = []
    for _ in range(draws):
        vals = [metric(rank_random_tiebreak(s, rng), i)
                for i, s in enumerate(scores_per_case)]
        per_draw.append(float(np.mean(vals)))
    arr = np.asarray(per_draw, dtype=float)
    return {
        "mean": float(arr.mean()),
        "sd": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "ci95_low": float(np.percentile(arr, 2.5)),
        "ci95_high": float(np.percentile(arr, 97.5)),
        "draws": int(draws),
    }


def tie_profile(scores_per_case: Sequence[np.ndarray],
                tol: float = 1e-12) -> Dict[str, float]:
    """How tied a ranker is: mean size of the maximum-score set, and the share
    of cases whose maximum is not unique."""
    sizes = [len(max_set(np.asarray(s, dtype=float), tol)) for s in scores_per_case]
    a = np.asarray(sizes, dtype=float)
    return {
        "mean_max_set_size": float(a.mean()),
        "share_cases_with_tied_max": float((a > 1).mean()),
        "max_max_set_size": int(a.max()),
    }
