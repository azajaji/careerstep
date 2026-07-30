"""Run every experiment and write results/*.json."""

from __future__ import annotations

import runpy
import sys

MODULES = [
    "experiments.exp8_career_positioning",
    "experiments.exp8b_projection_sensitivity",
    "experiments.exp8c_suitability_weight_sweep",
    "experiments.exp2_resume_job_alignment",
    "experiments.exp4_roadmap_quality",
    "experiments.exp5_interview_generation",
    "experiments.exp6_feedback_evaluation",
]


def main() -> None:
    for name in MODULES:
        print(f"\n=== {name} ===")
        try:
            runpy.run_module(name, run_name="__main__")
        except Exception as exc:  # noqa: BLE001
            print(f"[failed] {name}: {exc.__class__.__name__}: {exc}", file=sys.stderr)
            raise


if __name__ == "__main__":
    main()
