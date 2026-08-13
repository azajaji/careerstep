"""Cross-table consistency assertions for the manuscript's repeated baselines.

Several tables report the same arm under different names. A reader comparing
them row to row is entitled to identical numbers, and an external audit found
that they were not identical: the weight-sweep panel ranked with an unstable
sort, so its default and value-only rows disagreed with the headline benchmark
wherever scores tied.

Runs under pytest or directly: ``python -m tests.test_cross_table_consistency``.
"""

from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"
TOL = 5e-4  # tables print three decimals


def _load(name: str) -> dict:
    return json.loads((RESULTS / f"{name}.json").read_text(encoding="utf8"))


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= TOL


def _headline(ranker: str) -> dict:
    per = _load("exp8_career_positioning")["simulated_ranking_benchmark"]["per_ranker"]
    r = per[ranker]
    return {
        "hit@3": r["recall@3"]["mean"],
        "hit@5": r["recall@5"]["mean"],
        "mrr": r["mrr"]["mean"],
    }


def _sweep_row(label: str) -> dict:
    return _load("exp8_suitability_weight_sweep")["weight_panel"][label]


def test_weight_sweep_default_matches_headline_benchmark() -> None:
    """The 0.60/0.25/0.15 row is the composite of the main ranking table."""
    head = _headline("csmq_khutwa_lexical")
    default = _sweep_row("0.60/0.25/0.15 (default)")

    for metric in ("hit@3", "hit@5", "mrr"):
        assert _close(head[metric], default[metric]), (
            f"composite {metric}: headline {head[metric]} vs sweep {default[metric]}"
        )


def test_weight_sweep_value_only_matches_headline_value_fit() -> None:
    """The 1.00/0.00/0.00 row is the value-fit-only ranker."""
    head = _headline("csmq_only")
    value_only = _sweep_row("1.00/0.00/0.00 (value-only)")

    for metric in ("hit@3", "hit@5", "mrr"):
        assert _close(head[metric], value_only[metric]), (
            f"value-only {metric}: headline {head[metric]} vs sweep {value_only[metric]}"
        )


def test_calibration_uncalibrated_arm_matches_influence_table() -> None:
    """exp16's uncalibrated arm is exp8f's base measurement on the same cohort.

    Kendall is deliberately excluded: exp8f reports tau-b on the score vectors
    and exp16 reports tau on rank positions. Both are printed in the manuscript
    with their domain named, and the manuscript states the two values.
    """
    influence = _load("exp8f_criterion_influence")
    calib = _load("exp16_scale_calibration")

    loco = influence["leave_one_criterion_out"]
    unc = calib["influence_by_calibration"]["none"]

    for crit in ("value_fit", "skill_readiness", "feasibility"):
        assert _close(loco[crit]["top1_change_rate"], unc[crit]["top1_change_rate"]), (
            f"{crit} top-1: {loco[crit]['top1_change_rate']} vs "
            f"{unc[crit]['top1_change_rate']}"
        )
        assert _close(
            loco[crit]["kendall_tau_mean"], unc[crit]["kendall_tau_mean"]
        ), f"{crit} tau(positions) disagrees between exp8f and exp16"


if __name__ == "__main__":
    test_weight_sweep_default_matches_headline_benchmark()
    test_weight_sweep_value_only_matches_headline_value_fit()
    test_calibration_uncalibrated_arm_matches_influence_table()
    print("cross-table consistency: all assertions pass")
