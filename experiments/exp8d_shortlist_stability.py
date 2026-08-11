"""Shortlist stability under perturbation, for both rankers.

MRR measures exact rank recovery; the interface shows a shortlist, so this
reports what survives perturbation: top-k Jaccard overlap, first-rank churn,
retention counts, and rank correlation.

Two arms, because they behave very differently. The value-fit arm perturbs
role centroids and re-ranks by the CSMQ cosine alone. The composite arm
perturbs synthetic student profiles and re-ranks with the full
RoleSuitability blend, which is the ranker the paper reports. Skill and
feasibility evidence does not move with questionnaire noise, so the two
arms must not be conflated.

Uses its own generator so it cannot disturb the RNG stream of exp8.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from scipy import stats

from careerstep.career_positioning import ORIENTATIONS, OrientationProfile, RoleRecommender
from careerstep.seeding import load_seeds, set_global_seeds
from data.loaders import load_onet_work_values, load_saudi_cyber_roles
from eval.stats import summarize, summarize_clustered
from experiments._io import print_header, save_report

SIGMAS = [0.05, 0.10, 0.20, 0.30]
TRIALS_PER_ROLE = 20


def _ranked_ids(recommender: RoleRecommender, vec: np.ndarray, n: int) -> List[str]:
    profile = OrientationProfile(
        scores={o: float(vec[i]) for i, o in enumerate(ORIENTATIONS)}
    )
    return [r.role_id for r in recommender.recommend(profile, top_k=n)]


def _jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if (sa or sb) else 0.0


def _rank_vector(order: List[str], universe: List[str]) -> List[int]:
    pos = {rid: i for i, rid in enumerate(order)}
    return [pos[rid] for rid in universe]


def run() -> dict:
    set_global_seeds()
    seeds = load_seeds()
    rng = np.random.default_rng(seeds["numpy_seed"])

    roles = load_saudi_cyber_roles()
    wv = load_onet_work_values()
    recommender = RoleRecommender(roles_df=roles, wv_df=wv)

    role_ids = list(recommender.centroid_df.index)
    n = len(role_ids)
    universe = sorted(role_ids)

    # Several catalogue roles share an O*NET-SOC code and therefore have an
    # identical centroid. Order within such a tie is arbitrary, so shortlist
    # stability is also reported over centroid groups.
    group_of: Dict[str, int] = {}
    seen: Dict[tuple, int] = {}
    for i, rid in enumerate(role_ids):
        key = tuple(np.round(recommender.centroids[i], 10))
        group_of[rid] = seen.setdefault(key, len(seen))
    n_groups = len(seen)

    def groups(order: List[str]) -> List[int]:
        out: List[int] = []
        for rid in order:
            g = group_of[rid]
            if g not in out:
                out.append(g)
        return out

    clean: Dict[str, List[str]] = {}
    for rid in role_ids:
        centroid = recommender.centroid_df.loc[rid].to_numpy(dtype=float)
        clean[rid] = _ranked_ids(recommender, centroid, n)

    curve = []
    for sigma in SIGMAS:
        j3, j5, churn = [], [], []
        keep2of3, keep3of5 = [], []
        in3, out3, in5, out5 = [], [], [], []
        spear, kend = [], []
        g_churn, g_j3 = [], []
        owners: List[str] = []

        for rid in role_ids:
            centroid = recommender.centroid_df.loc[rid].to_numpy(dtype=float)
            base = clean[rid]
            b3, b5 = base[:3], base[:5]
            for _ in range(TRIALS_PER_ROLE):
                noisy = np.clip(centroid + rng.normal(0.0, sigma, size=centroid.shape), 0.0, 1.0)
                got = _ranked_ids(recommender, noisy, n)
                g3, g5 = got[:3], got[:5]

                j3.append(_jaccard(b3, g3))
                j5.append(_jaccard(b5, g5))
                churn.append(0.0 if got[0] == base[0] else 1.0)
                kept3 = len(set(b3) & set(g3))
                kept5 = len(set(b5) & set(g5))
                keep2of3.append(1.0 if kept3 >= 2 else 0.0)
                keep3of5.append(1.0 if kept5 >= 3 else 0.0)
                in3.append(float(len(set(g3) - set(b3))))
                out3.append(float(len(set(b3) - set(g3))))
                in5.append(float(len(set(g5) - set(b5))))
                out5.append(float(len(set(b5) - set(g5))))

                # tie-aware: compare centroid groups, since order inside a
                # group of identical centroids is arbitrary
                gb, gg = groups(base), groups(got)
                g_churn.append(0.0 if gg[0] == gb[0] else 1.0)
                g_j3.append(_jaccard([str(x) for x in gb[:3]],
                                     [str(x) for x in gg[:3]]))

                rb = _rank_vector(base, universe)
                rg = _rank_vector(got, universe)
                spear.append(float(stats.spearmanr(rb, rg).statistic))
                kend.append(float(stats.kendalltau(rb, rg).statistic))
                owners.append(rid)

        curve.append({
            "sigma": sigma,
            "trials": len(j3),
            "resampling_unit": "role",
            "jaccard_top3_clustered": summarize_clustered(j3, owners),
            "first_rank_changed_clustered": summarize_clustered(churn, owners),
            "jaccard_top3": summarize(j3),
            "jaccard_top5": summarize(j5),
            "first_rank_changed": summarize(churn),
            "retain_2_of_top3": summarize(keep2of3),
            "retain_3_of_top5": summarize(keep3of5),
            "entering_top3": summarize(in3),
            "leaving_top3": summarize(out3),
            "entering_top5": summarize(in5),
            "leaving_top5": summarize(out5),
            "spearman": summarize(spear),
            "kendall": summarize(kend),
            "group_first_rank_changed": summarize(g_churn),
            "group_jaccard_top3": summarize(g_j3),
        })

        c = curve[-1]
        print(f"  sigma={sigma:<5} J@3={c['jaccard_top3']['mean']:.3f} "
              f"J@5={c['jaccard_top5']['mean']:.3f} "
              f"top1-churn={c['first_rank_changed']['mean']:.3f} "
              f"(group {c['group_first_rank_changed']['mean']:.3f}) "
              f"rho={c['spearman']['mean']:.3f}")

    composite = _composite_arm(recommender, roles, rng)

    return {
        "n_roles": n,
        "n_distinct_centroid_groups": n_groups,
        "trials_per_role": TRIALS_PER_ROLE,
        "sigmas": SIGMAS,
        "stability_curve": curve,
        "composite_curve": composite,
    }


def _composite_arm(recommender, roles_df, rng) -> List[Dict]:
    """Same perturbation applied to student profiles, ranked by the full blend."""
    import random as _py_random
    from dataclasses import replace

    from careerstep.career_positioning_benchmark import (
        rank_khutwa, role_to_required_skills, simulate_profiles,
    )
    from careerstep.seeding import load_seeds

    seeds = load_seeds()
    required = role_to_required_skills(roles_df)

    def lexical(known, req):
        k = {s.lower() for s in known}
        r = [s.lower() for s in req]
        return sum(1 for s in r if s in k) / max(1, len(r))

    profiles = simulate_profiles(
        recommender, roles_df, n_profiles=120,
        rng=_py_random.Random(seeds["python_random_seed"]),
        np_rng=np.random.default_rng(seeds["numpy_seed"]),
        skill_coverage=0.35, noise_sigma=0.08, n_neighbours_acceptable=2,
    )

    out = []
    for sigma in SIGMAS:
        churn, j3, vf_churn, vf_j3 = [], [], [], []
        owners = []
        for p in profiles:
            vec = np.asarray(p.csmq_vector, dtype=float)
            comp_clean = rank_khutwa(p, recommender, roles_df, required, lexical)
            vf_clean = _ranked_ids(recommender, vec, len(recommender.centroid_df.index))
            for _ in range(TRIALS_PER_ROLE):
                noisy = np.clip(vec + rng.normal(0.0, sigma, size=vec.shape), 0.0, 1.0)
                comp_got = rank_khutwa(replace(p, csmq_vector=noisy), recommender,
                                       roles_df, required, lexical)
                vf_got = _ranked_ids(recommender, noisy, len(recommender.centroid_df.index))
                churn.append(0.0 if comp_got[0] == comp_clean[0] else 1.0)
                j3.append(_jaccard(comp_clean[:3], comp_got[:3]))
                vf_churn.append(0.0 if vf_got[0] == vf_clean[0] else 1.0)
                vf_j3.append(_jaccard(vf_clean[:3], vf_got[:3]))
                owners.append(p.profile_id)
        out.append({
            "sigma": sigma,
            "trials": len(churn),
            "resampling_unit": "profile",
            "composite_first_rank_changed_clustered": summarize_clustered(churn, owners),
            "valuefit_first_rank_changed_clustered": summarize_clustered(vf_churn, owners),
            "composite_first_rank_changed": summarize(churn),
            "composite_jaccard_top3": summarize(j3),
            "valuefit_first_rank_changed": summarize(vf_churn),
            "valuefit_jaccard_top3": summarize(vf_j3),
        })
        c = out[-1]
        print(f"  [student profiles] sigma={sigma:<5} "
              f"composite churn={c['composite_first_rank_changed']['mean']:.3f} "
              f"J@3={c['composite_jaccard_top3']['mean']:.3f}   |   "
              f"value-fit churn={c['valuefit_first_rank_changed']['mean']:.3f} "
              f"J@3={c['valuefit_jaccard_top3']['mean']:.3f}")
    return out


if __name__ == "__main__":
    print_header("Experiment 8d - Shortlist stability under profile perturbation")
    payload = run()
    path = save_report("exp8_shortlist_stability", payload)
    print(f"\nSaved {path}")
