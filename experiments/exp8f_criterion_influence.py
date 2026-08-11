"""How much does each criterion actually move the ranking?

The pooled statistic reported earlier, w_j sigma_j over all profile-by-role
pairs, is a dispersion heuristic and not a share of variance. It ignores
covariance between criteria, and it pools across profiles even though ranking
happens separately within each profile. A criterion that varies a lot between
students but little between the roles offered to one student cannot reorder
that student's shortlist.

This experiment replaces it with measurements taken at the level where the
decision is made:

1. Within-profile dispersion. For each profile, the standard deviation of each
   criterion across the candidate roles, weighted by its coefficient and
   normalized across criteria, then averaged over profiles.

2. Leave-one-criterion-out. Hold a criterion at its within-profile mean so it
   cannot discriminate, re-rank, and measure the damage with Kendall's tau,
   top-1 change rate, and top-5 overlap against the full ranking.

3. Pairwise score margins. For every ordered pair of roles inside a profile,
   the contribution w_j [x_j(a) - x_j(b)] each criterion makes to the margin
   that separates them, reported as a mean absolute contribution and as the
   share of pairs whose ordering that criterion alone would decide.
"""

from __future__ import annotations

import random
from typing import Dict, List

import numpy as np
from scipy import stats

from careerstep.career_positioning import RoleRecommender, _cosine
from careerstep.career_positioning_benchmark import (
    FEASIBILITY, W_FEAS, W_SKILL, W_VALUE,
    lexical_skill_readiness, role_to_required_skills, simulate_profiles,
)
from careerstep.seeding import load_seeds, set_global_seeds
from data.loaders import load_onet_work_values, load_saudi_cyber_roles
from experiments._io import print_header, save_report

CRITERIA = ("value_fit", "skill_readiness", "feasibility")
WEIGHTS = {"value_fit": W_VALUE, "skill_readiness": W_SKILL, "feasibility": W_FEAS}


def run() -> dict:
    set_global_seeds()
    seeds = load_seeds()
    roles_df = load_saudi_cyber_roles()
    rec = RoleRecommender(roles_df, load_onet_work_values())
    required = role_to_required_skills(roles_df)
    metas = roles_df.set_index("role_id")
    role_ids = list(rec.centroid_df.index)

    profiles = simulate_profiles(
        rec, roles_df, n_profiles=120,
        rng=random.Random(seeds["python_random_seed"]),
        np_rng=np.random.default_rng(seeds["numpy_seed"]),
        skill_coverage=0.35, distractor_skills_per_profile=3,
    )

    # Criterion matrix per profile: rows are candidate roles, columns criteria.
    per_profile: List[np.ndarray] = []
    for p in profiles:
        rows = []
        for i, rid in enumerate(role_ids):
            vf = _cosine(p.csmq_vector, rec.centroids[i])
            sk = lexical_skill_readiness(p.known_skills, required[rid])
            fe = FEASIBILITY.get((p.level, str(metas.loc[rid, "seniority"]).lower()), 0.5)
            rows.append([vf, sk, fe])
        per_profile.append(np.asarray(rows, dtype=float))

    w = np.array([WEIGHTS[c] for c in CRITERIA], dtype=float)

    # 1. Within-profile weighted dispersion, normalized per profile.
    shares = []
    raw_sd = []
    for X in per_profile:
        sd = X.std(axis=0, ddof=1)
        raw_sd.append(sd)
        ws = w * sd
        shares.append(ws / ws.sum() if ws.sum() > 0 else np.zeros_like(ws))
    shares = np.asarray(shares)
    raw_sd = np.asarray(raw_sd)
    within = {
        c: {
            "within_profile_sd_mean": float(raw_sd[:, j].mean()),
            "weighted_dispersion_share_mean": float(shares[:, j].mean()),
            "weighted_dispersion_share_sd": float(shares[:, j].std(ddof=1)),
        }
        for j, c in enumerate(CRITERIA)
    }

    # 2. Leave-one-criterion-out.
    def rank(X: np.ndarray) -> np.ndarray:
        return np.argsort(-(X @ w), kind="stable")

    loco: Dict[str, dict] = {}
    for j, c in enumerate(CRITERIA):
        taus, top1, ov5 = [], [], []
        for X in per_profile:
            full = rank(X)
            Y = X.copy()
            Y[:, j] = Y[:, j].mean()           # criterion can no longer discriminate
            got = rank(Y)
            pos_full = np.empty(len(full), dtype=int)
            pos_got = np.empty(len(got), dtype=int)
            pos_full[full] = np.arange(len(full))
            pos_got[got] = np.arange(len(got))
            taus.append(stats.kendalltau(pos_full, pos_got).statistic)
            top1.append(0.0 if full[0] == got[0] else 1.0)
            a, b = set(full[:5].tolist()), set(got[:5].tolist())
            ov5.append(len(a & b) / len(a | b))
        loco[c] = {
            "kendall_tau_mean": float(np.mean(taus)),
            "top1_change_rate": float(np.mean(top1)),
            "top5_jaccard_mean": float(np.mean(ov5)),
        }

    # 3. Pairwise margin contributions inside each profile.
    contrib_abs = np.zeros(len(CRITERIA))
    decisive = np.zeros(len(CRITERIA))
    n_pairs = 0
    for X in per_profile:
        d = X[:, None, :] - X[None, :, :]           # role a minus role b
        iu = np.triu_indices(X.shape[0], k=1)
        diffs = d[iu] * w                            # weighted contributions
        total = diffs.sum(axis=1)
        contrib_abs += np.abs(diffs).sum(axis=0)
        # A criterion is decisive for a pair when it alone matches the sign of
        # the total margin and exceeds the combined magnitude of the others.
        for j in range(len(CRITERIA)):
            others = np.delete(diffs, j, axis=1).sum(axis=1)
            decisive[j] += np.sum((np.sign(diffs[:, j]) == np.sign(total))
                                  & (np.abs(diffs[:, j]) > np.abs(others)))
        n_pairs += len(total)
    margins = {
        c: {
            "mean_abs_contribution": float(contrib_abs[j] / n_pairs),
            "share_of_pairs_decided": float(decisive[j] / n_pairs),
        }
        for j, c in enumerate(CRITERIA)
    }

    payload = {
        "n_profiles": len(profiles),
        "n_roles": len(role_ids),
        "n_within_profile_pairs": int(n_pairs),
        "weights": {c: WEIGHTS[c] for c in CRITERIA},
        "within_profile_dispersion": within,
        "leave_one_criterion_out": loco,
        "pairwise_margins": margins,
        "notes": {
            "measure": ("normalized weighted dispersion, not a share of "
                        "variance; it ignores covariance and is reported "
                        "alongside rank-based measures for that reason"),
            "unit": ("all statistics are computed within a profile, because "
                     "ranking happens among the roles offered to one student"),
            "loco": ("a criterion is held at its within-profile mean so that it "
                     "cannot discriminate; damage is measured against the full "
                     "ranking"),
        },
    }

    print_header("Experiment 8f - Criterion influence on the ranking")
    print(f"  {'criterion':<18}{'w':>6}{'within-sd':>11}{'disp share':>12}"
          f"{'tau (LOCO)':>12}{'top1 chg':>10}{'decides':>9}")
    for c in CRITERIA:
        print(f"  {c:<18}{WEIGHTS[c]:>6.2f}"
              f"{within[c]['within_profile_sd_mean']:>11.4f}"
              f"{within[c]['weighted_dispersion_share_mean']:>12.3f}"
              f"{loco[c]['kendall_tau_mean']:>12.3f}"
              f"{loco[c]['top1_change_rate']:>10.3f}"
              f"{margins[c]['share_of_pairs_decided']:>9.3f}")
    save_report("exp8f_criterion_influence", payload)
    return payload


if __name__ == "__main__":
    run()
