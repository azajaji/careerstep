"""Does the weight-influence inversion survive outside one operating point?

Experiment 19 varies the seed and holds the generator fixed, so it establishes
Monte Carlo stability and nothing about the assumptions that produce the
result. The reported cohort is one point in a parameter space: orientation
noise 0.08, 35% of the latent role's skills retained, three distractor skills,
a 70/30 student-to-graduate mix, and a 25-role candidate set.

This experiment sweeps that space. Profiles are regenerated for every
combination of the four generator parameters, and each profile cohort is then
audited under four candidate-set sizes and two ways of forming the candidate
set:

  random       k roles drawn uniformly from the catalog;
  prefiltered  the k roles with the highest value-fit for that student, which
               is the shortlist a values-first pre-filter would produce and the
               setting most favourable to value-fit.

For every configuration it reports the two statements the manuscript makes at
the reported operating point:

  dispersion inversion   value-fit has a smaller normalized weighted
                         dispersion share than skill-readiness, although it
                         carries the larger coefficient;
  full reversal          the dispersion ordering is the exact reverse of the
                         weight ordering.

The output is the fraction of the space in which each holds, and the fraction
by factor level, so a reader can see where the finding does and does not hold
rather than taking one cohort on trust.
"""

from __future__ import annotations

import itertools
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

CRITERIA = ("value_fit", "skill_readiness", "feasibility")
WEIGHTS = np.array([W_VALUE, W_SKILL, W_FEAS], dtype=float)

N_PROFILES = 240
SIGMAS = (0.04, 0.08, 0.12, 0.16)
COVERAGES = (0.20, 0.35, 0.50, 0.65)
DISTRACTORS = (1, 3, 5)
STUDENT_SHARES = (0.5, 0.7, 0.9)
SET_SIZES = (5, 10, 15, 25)
SET_MODES = ("random", "prefiltered")


def _criterion_matrices(rec, roles_df, required, metas, role_ids, profiles):
    """Full catalog criterion matrix for every profile: roles x criteria."""
    out = []
    for p in profiles:
        rows = []
        for i, rid in enumerate(role_ids):
            vf = _cosine(p.csmq_vector, rec.centroids[i])
            sk = lexical_skill_readiness(p.known_skills, required[rid])
            fe = FEASIBILITY.get((p.level, str(metas.loc[rid, "seniority"]).lower()), 0.5)
            rows.append([vf, sk, fe])
        out.append(np.asarray(rows, dtype=float))
    return out


def _audit(mats: List[np.ndarray], k: int, mode: str, rng: np.random.Generator) -> dict:
    """Dispersion shares and leave-one-criterion-out damage on k-role sets."""
    shares, top1 = [], []
    n_roles = mats[0].shape[0]
    for X in mats:
        if k >= n_roles:
            S = X
        elif mode == "random":
            S = X[rng.choice(n_roles, size=k, replace=False)]
        else:                                    # highest value-fit for this student
            S = X[np.argsort(-X[:, 0], kind="stable")[:k]]

        sd = S.std(axis=0, ddof=1)
        ws = WEIGHTS * sd
        shares.append(ws / ws.sum() if ws.sum() > 0 else np.zeros(3))

        full = np.argsort(-(S @ WEIGHTS), kind="stable")
        hits = []
        for j in range(3):
            Y = S.copy()
            Y[:, j] = Y[:, j].mean()
            hits.append(0.0 if full[0] == np.argsort(-(Y @ WEIGHTS), kind="stable")[0] else 1.0)
        top1.append(hits)

    shares = np.asarray(shares).mean(axis=0)
    top1 = np.asarray(top1).mean(axis=0)
    return {
        "dispersion_share": {c: float(shares[j]) for j, c in enumerate(CRITERIA)},
        "top1_change_rate": {c: float(top1[j]) for j, c in enumerate(CRITERIA)},
        "dispersion_inversion": bool(shares[0] < shares[1]),
        "full_reversal": bool(shares[0] < shares[1] < shares[2]),
        "top1_inversion": bool(top1[0] < top1[1]),
    }


def run() -> dict:
    set_global_seeds()
    seeds = load_seeds()
    roles_df = load_saudi_cyber_roles()
    rec = RoleRecommender(roles_df, load_onet_work_values())
    required = role_to_required_skills(roles_df)
    metas = roles_df.set_index("role_id")
    role_ids = list(rec.centroid_df.index)

    configs: List[dict] = []
    gen_grid = list(itertools.product(SIGMAS, COVERAGES, DISTRACTORS, STUDENT_SHARES))
    for gi, (sigma, coverage, distract, share) in enumerate(gen_grid):
        # One cohort per generator setting; every candidate-set variant is
        # measured on that same cohort, so the two factor families are not
        # confounded by resampling.
        py_rng = random.Random(seeds["python_random_seed"] + 9000 + gi)
        np_rng = np.random.default_rng(seeds["numpy_seed"] + 9000 + gi)
        profiles = simulate_profiles(
            rec, roles_df, n_profiles=N_PROFILES, noise_sigma=sigma,
            skill_coverage=coverage, distractor_skills_per_profile=distract,
            n_neighbours_acceptable=2, student_share=share,
            rng=py_rng, np_rng=np_rng,
        )
        mats = _criterion_matrices(rec, roles_df, required, metas, role_ids, profiles)
        for k, mode in itertools.product(SET_SIZES, SET_MODES):
            if k >= len(role_ids) and mode == "prefiltered":
                continue                          # identical to random at full size
            sub_rng = np.random.default_rng(seeds["numpy_seed"] + 41000 + gi)
            row = {"noise_sigma": sigma, "skill_coverage": coverage,
                   "distractors": distract, "student_share": share,
                   "candidate_set_size": k, "candidate_set_mode": mode}
            row.update(_audit(mats, k, mode, sub_rng))
            configs.append(row)

    def frac(rows, key):
        return float(np.mean([r[key] for r in rows])) if rows else float("nan")

    by_factor: Dict[str, dict] = {}
    for factor in ("noise_sigma", "skill_coverage", "distractors", "student_share",
                   "candidate_set_size", "candidate_set_mode"):
        by_factor[factor] = {}
        for level in sorted({r[factor] for r in configs}, key=str):
            rows = [r for r in configs if r[factor] == level]
            by_factor[factor][str(level)] = {
                "n_configs": len(rows),
                "dispersion_inversion": frac(rows, "dispersion_inversion"),
                "full_reversal": frac(rows, "full_reversal"),
                "top1_inversion": frac(rows, "top1_inversion"),
                "value_fit_share_mean": float(np.mean(
                    [r["dispersion_share"]["value_fit"] for r in rows])),
            }

    vf = np.array([r["dispersion_share"]["value_fit"] for r in configs])
    sk = np.array([r["dispersion_share"]["skill_readiness"] for r in configs])
    fe = np.array([r["dispersion_share"]["feasibility"] for r in configs])

    payload = {
        "n_configurations": len(configs),
        "n_profiles_per_configuration": N_PROFILES,
        "grid": {"noise_sigma": list(SIGMAS), "skill_coverage": list(COVERAGES),
                 "distractors": list(DISTRACTORS), "student_share": list(STUDENT_SHARES),
                 "candidate_set_size": list(SET_SIZES), "candidate_set_mode": list(SET_MODES)},
        "weights": {"value_fit": W_VALUE, "skill_readiness": W_SKILL, "feasibility": W_FEAS},
        "overall": {
            "dispersion_inversion": frac(configs, "dispersion_inversion"),
            "full_reversal": frac(configs, "full_reversal"),
            "top1_inversion": frac(configs, "top1_inversion"),
        },
        "share_ranges": {
            "value_fit": [float(vf.min()), float(vf.max())],
            "skill_readiness": [float(sk.min()), float(sk.max())],
            "feasibility": [float(fe.min()), float(fe.max())],
        },
        "by_factor": by_factor,
        "configurations": configs,
        "notes": {
            "scope": ("the catalog, the criterion producers and the weights are "
                      "held fixed; what varies is the profile generator and the "
                      "candidate set the ranking is taken over"),
            "prefiltered": ("the k highest value-fit roles for that student, "
                            "which is the setting most favourable to value-fit"),
            "statement": ("dispersion_inversion is the claim that the "
                          "largest-weight criterion has the smaller dispersion "
                          "share; full_reversal is the stronger claim that the "
                          "ordering is exactly reversed"),
        },
    }

    print_header("Experiment 20 - Operating region of the inversion")
    o = payload["overall"]
    print(f"  {len(configs)} configurations, {N_PROFILES} profiles each")
    print(f"  dispersion inversion holds in {o['dispersion_inversion']:.3f}")
    print(f"  full reversal holds in       {o['full_reversal']:.3f}")
    print(f"  top-1 inversion holds in     {o['top1_inversion']:.3f}")
    for factor, levels in by_factor.items():
        print(f"  {factor}:")
        for level, s in levels.items():
            print(f"    {level:<12} inv {s['dispersion_inversion']:.3f}  "
                  f"rev {s['full_reversal']:.3f}  "
                  f"top1 {s['top1_inversion']:.3f}  "
                  f"vf share {s['value_fit_share_mean']:.3f}")
    save_report("exp20_operating_region", payload)
    return payload


if __name__ == "__main__":
    run()
