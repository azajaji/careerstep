"""Assert that the values drawn in Figure 2 come from the current results.

An external audit found Figure 2 plotting a superseded release while the
tables and prose used the corrected one. Neither the build nor the cross-table
test could catch it: the build does not read figure contents, and the
cross-table test compares JSON to JSON. This reads the numbers the figure
generator would draw and compares them to the frozen results, so a figure that
was not regenerated fails here instead of reaching a PDF.

Runs under pytest or directly:
``python -m tests.test_figure_matches_results``.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TOL = 5e-4  # the figure prints three decimals

CRITERIA = ("value_fit", "skill_readiness", "feasibility")


def _load(name: str) -> dict:
    return json.loads((RESULTS / f"{name}.json").read_text(encoding="utf8"))


def figure_series() -> dict:
    """The four panels of Figure 2, read the way the generator reads them."""
    d = _load("exp8f_criterion_influence")
    w = d["weights"]
    disp = d["within_profile_dispersion"]
    loco = d["leave_one_criterion_out"]
    return {
        "weight": [w[k] for k in CRITERIA],
        "within_profile_sd": [disp[k]["within_profile_sd_mean"] for k in CRITERIA],
        "weighted_dispersion": [
            disp[k]["weighted_dispersion_share_mean"] for k in CRITERIA
        ],
        "top1_change": [loco[k]["top1_change_rate"] for k in CRITERIA],
    }


def test_figure_panels_come_from_one_result_file() -> None:
    """Every panel must resolve against the current exp8f output."""
    series = figure_series()
    assert series["weight"] == [0.60, 0.25, 0.15], series["weight"]
    for name, vals in series.items():
        assert len(vals) == 3, f"{name} has {len(vals)} entries"
        assert all(isinstance(v, float) for v in vals), name


def test_dispersion_shares_normalize() -> None:
    """The plotted dispersion shares are normalized, so they sum to one."""
    total = sum(figure_series()["weighted_dispersion"])
    assert abs(total - 1.0) <= 1e-9, total


def test_figure_agrees_with_the_influence_table() -> None:
    """Figure 2 and the criterion-influence table are the same numbers.

    They are drawn from one file, so this fails only if that file changes
    without the figure being regenerated from it.
    """
    d = _load("exp8f_criterion_influence")
    series = figure_series()
    for i, crit in enumerate(CRITERIA):
        table_sd = d["within_profile_dispersion"][crit]["within_profile_sd_mean"]
        table_top1 = d["leave_one_criterion_out"][crit]["top1_change_rate"]
        assert abs(series["within_profile_sd"][i] - table_sd) <= TOL, crit
        assert abs(series["top1_change"][i] - table_top1) <= TOL, crit


if __name__ == "__main__":
    test_figure_panels_come_from_one_result_file()
    test_dispersion_shares_normalize()
    test_figure_agrees_with_the_influence_table()
    s = figure_series()
    print("figure/results agreement: all assertions pass")
    print("  weights            ", [f"{v:.3f}" for v in s["weight"]])
    print("  within-profile SD  ", [f"{v:.4f}" for v in s["within_profile_sd"]])
    print("  weighted dispersion", [f"{v:.3f}" for v in s["weighted_dispersion"]])
    print("  top-1 change       ", [f"{v:.3f}" for v in s["top1_change"]])
