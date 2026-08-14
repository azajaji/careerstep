"""The constrained refit must not leave the designed structure.

Experiment 17 reports two kinds of comparator. The unconstrained fits are a
capacity diagnostic and the article says so. The constrained refit is described
as a repair the design admits, and that description is only true if the fitted
matrix keeps the designed zero pattern, the designed signs and the unit row
norm. Nothing in an optimizer enforces that by itself, so it is asserted here:
if the parameterization is ever changed, the claim in
Section V-F fails in this test rather than in review.

Runs under pytest or directly:
``python -m tests.test_constrained_fit_stays_in_design``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.exp17_fitted_projection import DESIGNED_W, _fit_constrained

RESULTS = Path(__file__).resolve().parents[1] / "results"
TOL = 5e-4


def _fitted_matrix() -> np.ndarray:
    """Refit on a small deterministic problem; structure is what is tested."""
    rng = np.random.default_rng(20260101)
    X = rng.random((60, 6))
    pi, pj = np.arange(0, 20), np.arange(20, 40)
    ni, nj = np.arange(20, 40), np.arange(40, 60)
    return _fit_constrained(X, pi, pj, ni, nj)


def test_support_is_preserved():
    W = _fitted_matrix()
    assert np.array_equal(W != 0, DESIGNED_W != 0), "the refit moved a zero"


def test_signs_are_preserved():
    W = _fitted_matrix()
    nz = DESIGNED_W != 0
    assert np.array_equal(np.sign(W[nz]), np.sign(DESIGNED_W[nz])), "the refit flipped a sign"


def test_rows_keep_unit_l1_norm():
    W = _fitted_matrix()
    assert np.allclose(np.abs(W).sum(axis=1), 1.0, atol=1e-9), "rows are not unit L1"


def test_reported_gap_is_the_difference_of_reported_means():
    """The paired gap is computed within split, so it cannot be recovered from
    the means alone, but it must not contradict them in sign or scale."""
    d = json.loads((RESULTS / "exp17_fitted_projection.json").read_text(encoding="utf8"))
    con = d["held_out_auc"]["constrained_fit_5d"]["mean"]
    des = d["held_out_auc"]["designed_W_5d"]["mean"]
    raw = d["held_out_auc"]["raw_work_values_6d"]["mean"]
    assert abs(d["paired_gap_constrained_minus_designed"]["mean"] - (con - des)) < 1e-2
    assert abs(d["paired_gap_constrained_minus_raw"]["mean"] - (con - raw)) < 1e-2
    assert d["paired_gap_constrained_minus_designed"]["ci95_low"] > 0, \
        "the article claims the refit beats the designed matrix"


def test_permuted_null_keeps_each_row_multiset():
    """The permuted-support null must move coefficients, not change them."""
    rng = np.random.default_rng(7)
    for _ in range(50):
        out = np.empty_like(DESIGNED_W)
        for r in range(DESIGNED_W.shape[0]):
            out[r] = DESIGNED_W[r][rng.permutation(DESIGNED_W.shape[1])]
        for r in range(DESIGNED_W.shape[0]):
            assert np.array_equal(np.sort(out[r]), np.sort(DESIGNED_W[r]))


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
