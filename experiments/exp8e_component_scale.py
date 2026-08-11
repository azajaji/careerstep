"""Effective influence of each RoleSuitability component.

Equation (1) weights value-fit at 0.60, skill-readiness at 0.25, and
feasibility at 0.15. Those nominal weights say nothing about influence on the
ranking unless the three components vary over comparable ranges. A term that
is nearly constant across candidate roles cannot reorder them however large
its coefficient.

This measures the empirical spread of each component over every
profile-by-role pair in the benchmark, and reports the weighted spread
``w * sd`` as the share of rank-moving variation each term supplies.
"""

from __future__ import annotations

import random
from typing import Dict, List

import numpy as np

from careerstep.career_positioning import RoleRecommender, _cosine
from careerstep.career_positioning_benchmark import (
    FEASIBILITY, W_FEAS, W_SKILL, W_VALUE,
    lexical_skill_readiness, role_to_required_skills, simulate_profiles,
)
from careerstep.seeding import load_seeds, set_global_seeds
from data.loaders import load_onet_work_values, load_saudi_cyber_roles
from experiments._io import print_header, save_report


def run() -> dict:
    set_global_seeds()
    seeds = load_seeds()
    roles_df = load_saudi_cyber_roles()
    wv_df = load_onet_work_values()
    rec = RoleRecommender(roles_df, wv_df)
    required = role_to_required_skills(roles_df)

    profiles = simulate_profiles(
        rec, roles_df, n_profiles=120,
        rng=random.Random(seeds["python_random_seed"]),
        np_rng=np.random.default_rng(seeds["numpy_seed"]),
        skill_coverage=0.35, distractor_skills_per_profile=3,
    )

    metas = roles_df.set_index("role_id")
    role_ids = list(rec.centroid_df.index)
    vf: List[float] = []
    sk: List[float] = []
    fe: List[float] = []
    for p in profiles:
        for i, rid in enumerate(role_ids):
            vf.append(_cosine(p.csmq_vector, rec.centroids[i]))
            sk.append(lexical_skill_readiness(p.known_skills, required[rid]))
            seniority = str(metas.loc[rid, "seniority"]).lower()
            fe.append(FEASIBILITY.get((p.level, seniority), 0.5))

    weights = {"value_fit": W_VALUE, "skill_readiness": W_SKILL, "feasibility": W_FEAS}
    arrays = {"value_fit": vf, "skill_readiness": sk, "feasibility": fe}

    components: Dict[str, dict] = {}
    for name, vals in arrays.items():
        a = np.asarray(vals, dtype=float)
        components[name] = {
            "nominal_weight": weights[name],
            "mean": float(a.mean()),
            "sd": float(a.std(ddof=1)),
            "min": float(a.min()),
            "max": float(a.max()),
            "range": float(a.max() - a.min()),
            "weighted_sd": float(weights[name] * a.std(ddof=1)),
        }
    total = sum(c["weighted_sd"] for c in components.values())
    for c in components.values():
        c["share_of_weighted_spread"] = float(c["weighted_sd"] / total) if total else 0.0

    # Why value-fit is compressed. Non-negativity alone does not explain it:
    # two non-negative vectors can be orthogonal. Measure the actual geometry.
    C = np.asarray(rec.centroids, dtype=float)
    P = np.asarray([p.csmq_vector for p in profiles], dtype=float)
    uniq = np.unique(np.round(C, 9), axis=0)
    pairwise = [
        _cosine(uniq[i], uniq[j])
        for i in range(uniq.shape[0]) for j in range(i + 1, uniq.shape[0])
    ]
    Cc = C - C.mean(axis=1, keepdims=True)
    Pc = P - P.mean(axis=1, keepdims=True)
    centred = [_cosine(pc, cc) for pc in Pc for cc in Cc]
    raw = np.asarray(vf, dtype=float)
    cen = np.asarray(centred, dtype=float)

    geometry = {
        "centroid_entry_min": float(C.min()),
        "centroid_entry_max": float(C.max()),
        "within_centroid_min_over_max_mean": float(
            (C.min(axis=1) / np.maximum(C.max(axis=1), 1e-12)).mean()),
        "within_profile_min_over_max_mean": float(
            (P.min(axis=1) / np.maximum(P.max(axis=1), 1e-12)).mean()),
        "pairwise_centroid_cosine_min": float(np.min(pairwise)),
        "pairwise_centroid_cosine_mean": float(np.mean(pairwise)),
        "centred_cosine_mean": float(cen.mean()),
        "centred_cosine_sd": float(cen.std(ddof=1)),
        "centred_cosine_min": float(cen.min()),
        "centred_cosine_max": float(cen.max()),
        "sd_ratio_centred_over_raw": float(cen.std(ddof=1) / raw.std(ddof=1)),
    }

    payload = {
        "n_profiles": len(profiles),
        "n_roles": len(role_ids),
        "n_pairs": len(vf),
        "components": components,
        "value_fit_geometry": geometry,
        "notes": {
            "reading": ("nominal_weight is the Equation (1) coefficient; "
                        "share_of_weighted_spread is w*sd normalised across the "
                        "three terms, which approximates how much each term can "
                        "reorder roles"),
            "cosine": ("non-negativity alone does not compress the cosine, since "
                       "non-negative vectors can be orthogonal. Every centroid "
                       "and profile here has all five entries well away from "
                       "zero, so the catalogue occupies a narrow cone and the "
                       "cosines are close to one. Centring each vector before "
                       "the cosine expands the spread by the recorded ratio."),
        },
    }

    print_header("Experiment 8e - RoleSuitability component scale")
    for name, c in components.items():
        print(f"  {name:<18} w={c['nominal_weight']:.2f} "
              f"mean={c['mean']:.3f} sd={c['sd']:.3f} "
              f"range=[{c['min']:.3f}, {c['max']:.3f}] "
              f"share={100*c['share_of_weighted_spread']:.1f}%")
    g = payload["value_fit_geometry"]
    print(f"  centroid entries span [{g['centroid_entry_min']:.3f}, "
          f"{g['centroid_entry_max']:.3f}]; pairwise centroid cosine >= "
          f"{g['pairwise_centroid_cosine_min']:.3f}")
    print(f"  centring before the cosine widens the spread "
          f"{g['sd_ratio_centred_over_raw']:.1f}x "
          f"(sd {g['centred_cosine_sd']:.3f})")
    save_report("exp8e_component_scale", payload)
    return payload


if __name__ == "__main__":
    run()
