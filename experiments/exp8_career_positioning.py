"""Career-positioning self-consistency, noise sweep, and ranking benchmark."""

from __future__ import annotations

import random
from typing import Dict, List

import numpy as np
import pandas as pd

from data.loaders import (
    load_csmq_questionnaire,
    load_onet_work_values,
    load_saudi_cyber_roles,
)
from eval.stats import summarize, summarize_clustered
from experiments._io import print_header, save_report
from careerstep.career_positioning import (
    ORIENTATIONS,
    OrientationProfile,
    RoleRecommender,
    project_work_values_to_csmq,
)
from careerstep.career_positioning_benchmark import (
    lexical_skill_readiness,
    mrr as ranking_mrr,
    ndcg_at_k,
    rank_csmq_only,
    rank_khutwa,
    rank_level_only,
    rank_random,
    rank_skills_only,
    recall_at_k,
    role_to_required_skills,
    semantic_skill_readiness,
    simulate_profiles,
)
from careerstep.seeding import set_global_seeds
from careerstep.benchmark_cohort import build_cohort, cohort_fingerprint


def _reciprocal_rank(target_role_id: str, ranked_role_ids: List[str]) -> float:
    for i, rid in enumerate(ranked_role_ids, start=1):
        if rid == target_role_id:
            return 1.0 / i
    return 0.0


def _topk_hit(target_role_id: str, ranked_role_ids: List[str], k: int) -> int:
    return int(target_role_id in ranked_role_ids[:k])


def _self_consistency(recommender: RoleRecommender) -> Dict[str, object]:
    role_ids = list(recommender.centroid_df.index)
    rrs: List[float] = []
    top1: List[int] = []
    top3: List[int] = []
    top5: List[int] = []
    for rid in role_ids:
        centroid = recommender.centroid_df.loc[rid].to_dict()
        profile = OrientationProfile(scores={o: float(centroid[o]) for o in ORIENTATIONS})
        ranked = recommender.recommend(profile, top_k=len(role_ids))
        ranked_ids = [r.role_id for r in ranked]
        rrs.append(_reciprocal_rank(rid, ranked_ids))
        top1.append(_topk_hit(rid, ranked_ids, 1))
        top3.append(_topk_hit(rid, ranked_ids, 3))
        top5.append(_topk_hit(rid, ranked_ids, 5))
    return {
        "n_roles": len(role_ids),
        "mrr": summarize(rrs),
        "top1": summarize(top1),
        "top3": summarize(top3),
        "top5": summarize(top5),
    }


def _noise_curve(
    recommender: RoleRecommender,
    *,
    sigmas: List[float],
    trials_per_role: int,
    rng: np.random.Generator,
) -> List[Dict[str, object]]:
    role_ids = list(recommender.centroid_df.index)
    curve = []
    for sigma in sigmas:
        rrs: List[float] = []
        owners: List[str] = []
        for rid in role_ids:
            centroid = recommender.centroid_df.loc[rid].to_numpy(dtype=float)
            for _ in range(trials_per_role):
                noisy = centroid + rng.normal(0.0, sigma, size=centroid.shape)
                noisy = np.clip(noisy, 0.0, 1.0)
                profile = OrientationProfile(
                    scores={o: float(noisy[i]) for i, o in enumerate(ORIENTATIONS)}
                )
                ranked = recommender.recommend(profile, top_k=len(role_ids))
                rrs.append(_reciprocal_rank(rid, [r.role_id for r in ranked]))
                owners.append(rid)
        # Trials repeat the same role, so the base role is the resampling unit.
        curve.append({
            "sigma": sigma,
            "mrr": summarize_clustered(rrs, owners),
            "mrr_unclustered": summarize(rrs),
            "trials": len(rrs),
            "resampling_unit": "role",
        })
    return curve


def _personas() -> Dict[str, Dict[str, float]]:
    """Five archetype CSMQ profiles, one per orientation."""
    base = {o: 0.30 for o in ORIENTATIONS}
    personas: Dict[str, Dict[str, float]] = {}
    for strong in ORIENTATIONS:
        profile = dict(base)
        profile[strong] = 0.95
        personas[strong] = profile
    return personas


def _persona_walkthrough(recommender: RoleRecommender) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for name, scores in _personas().items():
        profile = OrientationProfile(scores=scores)
        top = recommender.recommend(profile, top_k=3)
        out[name] = [
            {
                "rank": i + 1,
                "role_id": r.role_id,
                "title_en": r.title_en,
                "title_ar": r.title_ar,
                "specialty": r.specialty,
                "seniority": r.seniority,
                "sector_tag": r.sector_tag,
                "vision_2030_anchor": r.vision_2030_anchor,
                "score": r.score,
                "dominant_orientation": r.dominant_orientation,
                "rationale": r.rationale,
            }
            for i, r in enumerate(top)
        ]
    return out


def _simulated_ranking_benchmark(
    recommender: RoleRecommender,
    roles_df: pd.DataFrame,
    *,
    n_profiles: int,
    rng: random.Random,
    np_rng: np.random.Generator,
) -> Dict[str, object]:
    """Run the simulated-ranking benchmark over all rankers.

    Compares random, level-only, skills-only, csmq-only, and the composite
    under both a semantic and a lexical skill scorer."""
    profiles = build_cohort(recommender, roles_df)
    required = role_to_required_skills(roles_df)

    # Lazy-load embedding backend only if needed.
    from careerstep.backends import EmbeddingBackend
    embedding = EmbeddingBackend()
    encode_cache: Dict[str, np.ndarray] = {}

    def _encode(items: List[str]) -> np.ndarray:
        missing = [s for s in items if s not in encode_cache]
        if missing:
            vecs = embedding.encode(missing)
            for s, v in zip(missing, vecs):
                encode_cache[s] = v
        return np.stack([encode_cache[s] for s in items])

    sem_threshold = 0.70
    def _sem(user, req):
        return semantic_skill_readiness(user, req, encode=_encode, threshold=sem_threshold)
    def _lex(user, req): return lexical_skill_readiness(user, req)

    rankers = {
        "random":        lambda p: rank_random(p, list(recommender.centroid_df.index), rng),
        "level_only":    lambda p: rank_level_only(p, recommender, roles_df),
        "skills_only":   lambda p: rank_skills_only(p, recommender, required, _sem),
        "csmq_only":     lambda p: rank_csmq_only(p, recommender),
        "csmq_khutwa_lexical": lambda p: rank_khutwa(p, recommender, roles_df, required, _lex),
        "csmq_khutwa":   lambda p: rank_khutwa(p, recommender, roles_df, required, _sem),
    }

    per_ranker: Dict[str, Dict[str, List[float]]] = {
        name: {"recall@3": [], "recall@5": [], "mrr": [], "ndcg@5": []}
        for name in rankers
    }

    # Roles whose Work-Value centroid is identical to another role's. Ordering
    # inside such a group is arbitrary, so benchmark difficulty is not constant
    # across profiles and the strata are reported separately below.
    centroids = recommender.centroid_df
    by_vector: Dict[tuple, List[str]] = {}
    for role_id, row in centroids.iterrows():
        by_vector.setdefault(tuple(np.round(row.to_numpy(dtype=float), 6)), []).append(role_id)
    tied_roles = {r for group in by_vector.values() if len(group) > 1 for r in group}

    strata: Dict[str, List[bool]] = {"tied": [], "untied": []}

    for prof in profiles:
        is_tied = prof.latent_role_id in tied_roles
        strata["tied" if is_tied else "untied"].append(True)
        for name, fn in rankers.items():
            ranked = fn(prof)
            per_ranker[name]["recall@3"].append(recall_at_k(ranked, prof.acceptable_role_ids, 3))
            per_ranker[name]["recall@5"].append(recall_at_k(ranked, prof.acceptable_role_ids, 5))
            per_ranker[name]["mrr"].append(ranking_mrr(ranked, prof.acceptable_role_ids))
            per_ranker[name]["ndcg@5"].append(ndcg_at_k(ranked, prof.acceptable_role_ids, 5))

    summary: Dict[str, Dict[str, Dict]] = {}
    for name, metrics in per_ranker.items():
        summary[name] = {m: summarize(vals) for m, vals in metrics.items()}

    # Stratified view. Uses the values already collected above, so it draws no
    # further random numbers and leaves every aggregate metric unchanged.
    tied_mask = [prof.latent_role_id in tied_roles for prof in profiles]
    stratified: Dict[str, Dict[str, Dict]] = {}
    for stratum, want in (("tied_anchor", True), ("untied_anchor", False)):
        idx = [i for i, t in enumerate(tied_mask) if t == want]
        stratified[stratum] = {
            "n_profiles": len(idx),
            "per_ranker": {
                name: {
                    m: summarize([vals[i] for i in idx])
                    for m, vals in metrics.items()
                }
                for name, metrics in per_ranker.items()
            },
        }

    return {
        "n_profiles": int(n_profiles),
        "n_roles": int(len(recommender.centroid_df)),
        "n_distinct_centroids": len(by_vector),
        "n_roles_with_tied_centroid": len(tied_roles),
        "n_neighbours_acceptable": 2,
        "skill_scorer_semantic_threshold": sem_threshold,
        "weights": {"value_fit": 0.60, "skill_readiness": 0.25, "feasibility": 0.15},
        "per_ranker": summary,
        "stratified_by_anchor": stratified,
    }


def _wv_coverage(wv_df: pd.DataFrame, roles_df: pd.DataFrame) -> Dict[str, object]:
    role_socs = set(roles_df["onet_soc"].unique())
    wv_socs = set(wv_df["onet_soc"].unique())
    return {
        "n_roles": int(len(roles_df)),
        "n_unique_socs_in_roles": int(len(role_socs)),
        "n_socs_with_wv": int(len(role_socs & wv_socs)),
        "missing_socs": sorted(role_socs - wv_socs),
    }


def run() -> dict:
    seeds = set_global_seeds()
    rng = np.random.default_rng(seeds["numpy_seed"])

    questionnaire = load_csmq_questionnaire()
    wv = load_onet_work_values()
    roles = load_saudi_cyber_roles()

    recommender = RoleRecommender(roles_df=roles, wv_df=wv)

    coverage = _wv_coverage(wv, roles)
    self_consistency = _self_consistency(recommender)
    noise_curve = _noise_curve(
        recommender,
        sigmas=[0.00, 0.05, 0.10, 0.20, 0.30],
        trials_per_role=20,
        rng=rng,
    )
    personas = _persona_walkthrough(recommender)
    py_rng = random.Random(seeds["python_random_seed"])
    simulated_ranking = _simulated_ranking_benchmark(
        recommender, roles,
        n_profiles=120, rng=py_rng, np_rng=rng,
    )

    # Sanity: project a single role's WV directly so the report shows the
    # public-dataset anchor end-to-end.
    sample_soc = wv["onet_soc"].iloc[0]
    sample_wv = wv.set_index("onet_soc").loc[sample_soc].to_dict()
    projected = project_work_values_to_csmq(
        {k: sample_wv[k] for k in (
            "achievement", "independence", "recognition",
            "relationships", "support", "working_conditions",
        )}
    )

    return {
        "instrument": {
            "name": "Career Success Map Questionnaire (CSMQ)",
            "items": len(questionnaire.items),
            "orientations": list(ORIENTATIONS),
            "scale": [questionnaire.scale_min, questionnaire.scale_max],
        },
        "onet_anchor": {
            "dataset": "O*NET 28.x Work Values (Extent scale, 0-7)",
            "url": "https://www.onetcenter.org/database.html",
            "coverage": coverage,
        },
        "projection_demo": {
            "sample_onet_soc": sample_soc,
            "sample_wv": {k: float(sample_wv[k]) for k in (
                "achievement", "independence", "recognition",
                "relationships", "support", "working_conditions",
            )},
            "projected_csmq_centroid": projected.scores,
        },
        "self_consistency": self_consistency,
        "noise_robustness": noise_curve,
        "persona_walkthrough": personas,
        "simulated_ranking_benchmark": simulated_ranking,
    }


if __name__ == "__main__":
    print_header("Experiment 8 - career positioning (CSMQ -> Saudi cyber roles)")
    payload = run()
    path = save_report("exp8_career_positioning", payload)
    print(f"\nSaved {path}")
