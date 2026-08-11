"""No-profile ablation: what the persistent profile is worth downstream.

The persistent-profile design principle (DP1) says each module should reuse
context produced by the previous one instead of asking for it again. With the
profile, the roadmap generator receives the exact missing-skill set the
skill-gap detector produced, and the interview generator receives the selected
role's exact required skills. Without it, the student re-supplies that context
by hand.

Re-entry is modelled as a lossy channel with fidelity ``r``: each item of the
true context survives with probability ``r``, and each lost item is replaced by
a distractor drawn from another role, which models both forgetting and
misremembering. Downstream outputs are then scored against the *true* context,
which is what the profile would have carried.

The point of sweeping ``r`` rather than fixing it is that the profile's value
is a function of how much context a student loses, and that quantity is not
known. At ``r = 1`` the two conditions coincide by construction, which anchors
the sweep. Locating real students on the curve needs a user study; this
experiment measures the curve, not the position on it.
"""

from __future__ import annotations

import random
from typing import Dict, List, Sequence

from careerstep.backends import EmbeddingBackend
from careerstep.interview import InterviewQuestionGenerator
from careerstep.roadmap import LearningItem, RoadmapGenerator
from careerstep.seeding import set_global_seeds
from data.loaders import load_learning_resources, load_occupations
from eval.metrics_generation import coverage as coverage_metric
from eval.stats import summarize, summarize_clustered
from experiments._io import print_header, save_report

FIDELITIES = (0.0, 0.25, 0.50, 0.75, 1.0)
TRIALS_PER_ROLE = 20
MAX_ITEMS = 8


def _bank_from_df(df) -> List[LearningItem]:
    return [
        LearningItem(
            skill=row["skill"],
            resource=row["resource"],
            provider=row.get("provider", ""),
            estimated_hours=int(row.get("estimated_hours", 0)),
            level=row.get("level", "intermediate"),
            certification=bool(row.get("certification", False)),
        )
        for _, row in df.iterrows()
    ]


def _reentered(
    true_context: Sequence[str],
    distractor_pool: Sequence[str],
    fidelity: float,
    rng: random.Random,
) -> List[str]:
    """One student's hand re-entry of ``true_context`` at the given fidelity."""
    kept, lost = [], 0
    for item in true_context:
        if rng.random() < fidelity:
            kept.append(item)
        else:
            lost += 1
    pool = [s for s in distractor_pool if s not in set(true_context)]
    rng.shuffle(pool)
    kept.extend(pool[:lost])
    return kept


def _interview_skill_coverage(questions, true_skills: Sequence[str]) -> float:
    if not true_skills:
        return 0.0
    text = " ".join(q.text for q in questions).lower()
    return sum(1 for s in true_skills if s.lower() in text) / len(true_skills)


def run() -> dict:
    seeds = set_global_seeds()
    rng = random.Random(seeds["python_random_seed"])
    occ = load_occupations()
    bank = _bank_from_df(load_learning_resources())
    backend = EmbeddingBackend()
    roadmap_gen = RoadmapGenerator(bank, backend=backend)
    interview_gen = InterviewQuestionGenerator(backend=backend)

    roles: List[Dict] = []
    all_skills: List[str] = []
    for _, row in occ.iterrows():
        skills = list(row["skills"])
        if len(skills) < 2:
            continue
        all_skills.extend(skills)
        shuffled = list(skills)
        rng.shuffle(shuffled)
        cut = max(1, len(shuffled) // 2)
        roles.append({
            "role": row["role"],
            "required": skills,
            "true_missing": shuffled[:cut],
        })

    # With-profile condition: exact context, no re-entry, so it is one value
    # per role rather than a distribution.
    with_profile = {"roadmap_gap_coverage": [], "interview_skill_coverage": []}
    for r in roles:
        rm = roadmap_gen.generate(
            role=r["role"], missing_skills=r["true_missing"], max_items=MAX_ITEMS
        )
        with_profile["roadmap_gap_coverage"].append(
            coverage_metric(rm.as_text_list(), r["true_missing"])
        )
        qs = interview_gen.generate(role=r["role"], skills=r["required"])
        with_profile["interview_skill_coverage"].append(
            _interview_skill_coverage(qs, r["required"])
        )

    # No-profile condition: swept over re-entry fidelity.
    curve = []
    for fidelity in FIDELITIES:
        road_vals, intv_vals, owners = [], [], []
        for r in roles:
            for _ in range(TRIALS_PER_ROLE):
                owners.append(r["role"])
                re_missing = _reentered(r["true_missing"], all_skills, fidelity, rng)
                rm = roadmap_gen.generate(
                    role=r["role"], missing_skills=re_missing, max_items=MAX_ITEMS
                )
                road_vals.append(coverage_metric(rm.as_text_list(), r["true_missing"]))

                re_required = _reentered(r["required"], all_skills, fidelity, rng)
                qs = interview_gen.generate(role=r["role"], skills=re_required)
                intv_vals.append(_interview_skill_coverage(qs, r["required"]))
        # Trials are nested within the ten roles, so the role is the
        # resampling unit for the interval.
        curve.append({
            "fidelity": fidelity,
            "trials": len(road_vals),
            "resampling_unit": "role",
            "roadmap_gap_coverage": summarize_clustered(road_vals, owners),
            "interview_skill_coverage": summarize_clustered(intv_vals, owners),
        })

    payload = {
        "n_roles": len(roles),
        "trials_per_role": TRIALS_PER_ROLE,
        "fidelities": list(FIDELITIES),
        "with_profile": {
            k: summarize(v) for k, v in with_profile.items()
        },
        "no_profile_curve": curve,
        "notes": {
            "reentry_model": ("each true context item survives with probability "
                              "fidelity; each lost item is replaced by a "
                              "distractor skill drawn from another role"),
            "scoring": "downstream outputs are scored against the true context",
            "anchor": ("at fidelity 1.0 the no-profile condition reproduces the "
                       "with-profile condition by construction"),
        },
    }

    print_header("Experiment 13 - No-profile ablation")
    for k, v in payload["with_profile"].items():
        print(f"  with profile   {k:<26} {v['mean']:.3f}")
    for row in curve:
        print(f"  r={row['fidelity']:.2f}  roadmap={row['roadmap_gap_coverage']['mean']:.3f}"
              f"  interview={row['interview_skill_coverage']['mean']:.3f}")
    save_report("exp13_no_profile_ablation", payload)
    return payload


if __name__ == "__main__":
    run()
