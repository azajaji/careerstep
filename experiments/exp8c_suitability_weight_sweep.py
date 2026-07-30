"""Sensitivity of the benchmark to the RoleSuitability weight blend."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from data.loaders import (
    load_csmq_questionnaire,
    load_onet_work_values,
    load_saudi_cyber_roles,
)
from careerstep.career_positioning import RoleRecommender
from careerstep.career_positioning_benchmark import (
    FEASIBILITY,
    lexical_skill_readiness,
    mrr as ranking_mrr,
    recall_at_k,
    role_to_required_skills,
    simulate_profiles,
)
from careerstep.seeding import set_global_seeds


def _cosine(u: np.ndarray, v: np.ndarray) -> float:
    nu = float(np.linalg.norm(u)); nv = float(np.linalg.norm(v))
    return float(np.dot(u, v) / (nu * nv)) if nu and nv else 0.0


def _advance_like_exp8_noise_curve(np_rng: np.random.Generator, n_roles: int) -> None:
    for sigma in (0.00, 0.05, 0.10, 0.20, 0.30):
        for _ in range(n_roles):
            for _ in range(20):
                np_rng.normal(0.0, sigma, size=(5,))


def _rank_weighted(profile, rec, roles_df, required, w_value, w_skill, w_feas) -> List[str]:
    role_ids = list(rec.centroid_df.index)
    metas = roles_df.set_index("role_id")
    scores = []
    for i, rid in enumerate(role_ids):
        value_fit = _cosine(profile.csmq_vector, rec.centroids[i])
        skill = lexical_skill_readiness(profile.known_skills, required[rid])
        seniority = str(metas.loc[rid, "seniority"]).lower()
        feas = FEASIBILITY.get((profile.level, seniority), 0.5)
        scores.append(w_value * value_fit + w_skill * skill + w_feas * feas)
    order = np.argsort(-np.asarray(scores))
    return [role_ids[int(i)] for i in order]


def run() -> Dict:
    seeds = set_global_seeds()
    roles = load_saudi_cyber_roles()
    wv = load_onet_work_values()
    _ = load_csmq_questionnaire()

    rec = RoleRecommender(roles_df=roles, wv_df=wv)
    py_rng = random.Random(seeds["python_random_seed"])
    np_rng = np.random.default_rng(seeds["numpy_seed"])
    _advance_like_exp8_noise_curve(np_rng, n_roles=len(rec.centroid_df))
    profiles = simulate_profiles(
        rec, roles, n_profiles=120, rng=py_rng, np_rng=np_rng,
        skill_coverage=0.35, distractor_skills_per_profile=3,
    )
    required = role_to_required_skills(roles)

    # Panel of weight triples (must each sum to 1).
    panel = {
        "0.60/0.25/0.15 (default)": (0.60, 0.25, 0.15),
        "0.50/0.30/0.20":          (0.50, 0.30, 0.20),
        "0.70/0.20/0.10":          (0.70, 0.20, 0.10),
        "0.40/0.40/0.20":          (0.40, 0.40, 0.20),
        "0.33/0.34/0.33 (equal)":  (0.33, 0.34, 0.33),
        "1.00/0.00/0.00 (value-only)": (1.00, 0.00, 0.00),
    }

    out: Dict[str, Dict[str, float]] = {}
    for label, (wv_, ws_, wf_) in panel.items():
        r3: List[float] = []; r5: List[float] = []; rr: List[float] = []
        for prof in profiles:
            ranked = _rank_weighted(prof, rec, roles, required, wv_, ws_, wf_)
            r3.append(recall_at_k(ranked, prof.acceptable_role_ids, 3))
            r5.append(recall_at_k(ranked, prof.acceptable_role_ids, 5))
            rr.append(ranking_mrr(ranked, prof.acceptable_role_ids))
        out[label] = {
            "hit@3": float(np.mean(r3)),
            "hit@5": float(np.mean(r5)),
            "mrr": float(np.mean(rr)),
        }
    return {"n_profiles": len(profiles), "weight_panel": out}


if __name__ == "__main__":
    payload = run()
    p = Path("results/exp8_suitability_weight_sweep.json")
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("=== RoleSuitability weight sweep (120 profiles, lexical scorer) ===")
    print(f"{'weights (value/skill/feas)':28s} {'Hit@3':>7s} {'Hit@5':>7s} {'MRR':>7s}")
    for label, m in payload["weight_panel"].items():
        print(f"{label:28s} {m['hit@3']:7.3f} {m['hit@5']:7.3f} {m['mrr']:7.3f}")
    print(f"\nSaved {p}")
