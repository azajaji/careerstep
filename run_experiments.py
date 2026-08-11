"""Run every experiment and write results/*.json."""

from __future__ import annotations

import runpy
import sys

MODULES = [
    "experiments.exp7_questionnaire_scoring",
    "experiments.exp8_career_positioning",
    "experiments.exp8b_projection_sensitivity",
    "experiments.exp8c_suitability_weight_sweep",
    "experiments.exp8d_shortlist_stability",
    "experiments.exp8e_component_scale",
    "experiments.exp8f_criterion_influence",
    "experiments.exp2_resume_job_alignment",
    "experiments.exp4_roadmap_quality",
    "experiments.exp5_interview_generation",
    "experiments.exp6_feedback_evaluation",
    "experiments.exp12_melo_external",  # downloads MELO on first run
    "experiments.exp13_no_profile_ablation",
    "experiments.exp15_external_structure",  # downloads O*NET tables on first run
    "experiments.exp16_scale_calibration",
    "experiments.exp17_fitted_projection",
    "experiments.exp18_external_index_audit",  # downloads UNDP HDR data on first run
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
